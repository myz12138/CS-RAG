"""Build a versioned, two-layer knowledge base from project sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from backend.app.algorithm_adapter.input_export import ReferenceKGExporter
from backend.app.application.build_settings import BuildSettings
from backend.app.indexing.text_index import TextIndexBuildError, TextIndexBuilder
from backend.app.indexing.vector_store import (
    QdrantVectorStore,
    VectorStore,
    VectorStoreError,
    make_collection_name,
)
from backend.app.ingestion.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingConfigurationError,
    SentenceTransformerEmbeddingProvider,
    create_chunker,
)
from backend.app.ingestion.document_loaders import (
    DocumentLoadError,
    DefaultDocumentLoader,
    load_document,
)
from backend.app.model_gateway.base import ModelGateway, ModelGatewayError
from backend.app.storage.file_store import FileStoreError, ProjectFileStore
from backend.app.storage.sqlite_store import SQLiteStore


ProgressCallback = Callable[[float, str], None]


class BuildError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BuildService:
    def __init__(
        self,
        store: SQLiteStore,
        files: ProjectFileStore,
        model_gateway: ModelGateway,
        *,
        models_root: Path | None = None,
        chunking_config: ChunkingConfig | None = None,
        vector_store_backend: str = "local_file",
        vector_store_root: Path | None = None,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        qdrant_collection_prefix: str = "csrag",
        qdrant_on_disk: bool = True,
        text_index_batch_size: int = 32,
        build_settings: BuildSettings | None = None,
    ) -> None:
        self.store = store
        self.files = files
        self.model_gateway = model_gateway
        self.models_root = (models_root or Path(__file__).resolve().parents[3] / "models").resolve()
        self.chunking_config = chunking_config
        self.build_settings = (
            build_settings or BuildSettings(text_index_batch_size=text_index_batch_size)
        ).validate()
        self.text_index_builder = TextIndexBuilder(
            batch_size=self.build_settings.text_index_batch_size
        )
        self.vector_store_backend = (vector_store_backend or "local_file").strip().lower()
        self.vector_store_root = (
            vector_store_root
            or (self.files.projects_root.parent / "vector_store")
        ).resolve()
        self.qdrant_url = qdrant_url.strip()
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_collection_prefix = qdrant_collection_prefix.strip() or "csrag"
        self.qdrant_on_disk = qdrant_on_disk
        self._vector_store: VectorStore | None = None

    def build(
        self,
        project_id: str,
        *,
        mode: str = "auto",
        allow_graph_only: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        sources = self.store.list_sources(project_id)
        text_sources = [source for source in sources if source.source_kind == "text"]
        kg_sources = [source for source in sources if source.source_kind == "kg"]
        effective_mode = self._resolve_mode(mode, bool(kg_sources))

        if not sources:
            raise BuildError(
                "NO_INPUT",
                "没有新的文本输入；如需直接使用已有知识库，请在查询页面选择已发布版本",
            )
        if kg_sources:
            raise BuildError(
                "KG_INPUT_DISABLED",
                "当前只允许从文本构建知识库，KG 文件不再作为新建输入",
            )
        if not text_sources:
            raise BuildError("TEXT_REQUIRED", "知识库构建必须至少包含一个文本来源")
        if allow_graph_only:
            raise BuildError("GRAPH_ONLY_DISABLED", "当前版本不支持仅构建图谱，请从文本开始构建")
        # With a text-only input there is no import-only branch: the KG layer
        # is always extracted from the text by the configured real LLM.
        needs_llm = True
        if needs_llm and not self.model_gateway.capabilities().get("configured_base_url"):
            raise BuildError("MODEL_NOT_CONFIGURED", "当前构建需要真实 LLM 抽取，但模型服务尚未配置")

        version = self.store.create_version(
            project_id,
            effective_mode,
            [source.source_id for source in sources],
        )
        workspace: Path | None = None
        parse_issues: list[dict[str, object]] = []
        duplicate_count = 0
        total_conflicts = 0
        text_entry_triple_ids: dict[str, list[str]] = {}
        parent_records: list[dict[str, object]] = []
        text_index_summary: dict[str, object] = {"status": "not_provided"}
        active_vector_store: VectorStore | None = None
        vector_collection_name: str | None = None
        active_build_settings = self.build_settings
        active_text_index_builder = TextIndexBuilder(
            batch_size=active_build_settings.text_index_batch_size
        )
        try:
            workspace = self.files.create_version_workspace(project_id, version.version_id)
            self._report(progress, 0.05, "已创建版本工作目录")
            text_entries = []
            stored_chunks: list[dict[str, object]] = []
            chunking_sources: list[dict[str, object]] = []
            chunking_summary: dict[str, object] = self._empty_chunking_summary()
            active_chunking_config: ChunkingConfig | None = None
            embedding_function: SentenceTransformerEmbeddingProvider | None = None
            if text_sources:
                try:
                    active_chunking_config = self._effective_chunking_config(active_build_settings)
                    if not active_chunking_config.embedding_model:
                        raise ChunkingConfigurationError("文本索引需要配置本地 embedding 模型")
                    embedding_function = SentenceTransformerEmbeddingProvider(
                        active_chunking_config.embedding_model,
                        device=active_chunking_config.embedding_device,
                        local_files_only=active_chunking_config.embedding_local_files_only,
                    )
                    chunker = create_chunker(
                        active_chunking_config,
                        embedding_function=embedding_function,
                    )
                except ChunkingConfigurationError as exc:
                    raise BuildError("CHUNKING_CONFIG_FAILED", str(exc)) from exc

                loader = DefaultDocumentLoader()
                retrieval_chunk_count = 0
                parent_chunk_count = 0
                for index, source in enumerate(text_sources):
                    raw = self.files.read_source(project_id, source.stored_path)
                    try:
                        document = load_document(raw, source.original_name, loader=loader)
                        chunking_result = chunker.chunk(document)
                    except DocumentLoadError as exc:
                        raise BuildError("TEXT_PARSE_FAILED", str(exc)) from exc
                    except ChunkingConfigurationError as exc:
                        raise BuildError("CHUNKING_FAILED", str(exc)) from exc

                    retrieval_chunks = [
                        chunk for chunk in chunking_result.chunks if chunk.level != "parent"
                    ]
                    if not retrieval_chunks:
                        raise BuildError("TEXT_EMPTY", f"{source.original_name} 没有生成可检索的文本条目")

                    entry_by_chunk_id = {}
                    for ordinal, chunk in enumerate(retrieval_chunks):
                        title = self._chunk_title(source.original_name, chunk)
                        source_location = self._chunk_source_location(chunk)
                        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                        entry = self.store.create_text_entry(
                            version.version_id,
                            source.source_id,
                            ordinal,
                            title,
                            chunk.text,
                            content_hash,
                            source_location,
                            parent_id=chunk.parent_id,
                        )
                        text_entries.append(entry)
                        entry_by_chunk_id[chunk.chunk_id] = entry

                    for chunk in chunking_result.chunks:
                        entry = entry_by_chunk_id.get(chunk.chunk_id)
                        if chunk.level == "parent":
                            parent_records.append(
                                {
                                    "parent_id": chunk.chunk_id,
                                    "content": chunk.text,
                                    "source_location": self._chunk_source_location(chunk),
                                }
                            )
                        stored_chunks.append(
                            self._chunk_record(
                                source,
                                chunk,
                                entry_id=entry.entry_id if entry is not None else None,
                                title=self._chunk_title(source.original_name, chunk),
                            )
                        )

                    parent_count = sum(1 for chunk in chunking_result.chunks if chunk.level == "parent")
                    parent_chunk_count += parent_count
                    retrieval_chunk_count += len(retrieval_chunks)
                    chunking_sources.append(
                        {
                            "source_id": source.source_id,
                            "source_name": source.original_name,
                            "document_id": chunking_result.document.document_id,
                            "strategy": chunking_result.strategy,
                            "chunk_count": len(retrieval_chunks),
                            "parent_count": parent_count,
                            "diagnostics": chunking_result.diagnostics,
                        }
                    )
                    self._report(
                        progress,
                        0.10 + 0.25 * ((index + 1) / max(len(text_sources), 1)),
                        f"文本切分已处理 {index + 1}/{len(text_sources)} 个来源，生成 {len(retrieval_chunks)} 个条目",
                    )

                chunking_summary = {
                    "strategy": active_chunking_config.normalized_strategy(),
                    "config": self._chunking_config_summary(
                        active_chunking_config,
                        text_index_batch_size=active_text_index_builder.batch_size,
                    ),
                    "source_count": len(text_sources),
                    "retrieval_chunk_count": retrieval_chunk_count,
                    "parent_chunk_count": parent_chunk_count,
                    "sources": chunking_sources,
                }

            self.store.update_version_status(
                version.version_id,
                status="building",
                text_status="ready" if text_sources else "not_provided",
                kg_status="extracting",
            )

            if needs_llm:
                if not text_entries:
                    raise BuildError("TEXT_EMPTY", "文本层没有可供抽取的条目")
                for index, entry in enumerate(text_entries):
                    try:
                        extracted = self.model_gateway.extract_triples(entry.content, entry.title)
                    except ModelGatewayError as exc:
                        raise BuildError("MODEL_CALL_FAILED", str(exc)) from exc
                    for triple in extracted:
                        stored_triple, duplicate, conflicts = self.store.insert_kg_triple(
                            version.version_id,
                            entry.source_id,
                            entry.entry_id,
                            None,
                            triple["head"],
                            triple["relation"],
                            triple["tail"],
                            "text_extracted",
                            triple.get("evidence") or entry.content,
                        )
                        duplicate_count += int(duplicate)
                        total_conflicts += len(conflicts)
                        text_entry_triple_ids.setdefault(entry.entry_id, []).append(
                            stored_triple.triple_id
                        )
                    self._report(
                        progress,
                        0.65 + 0.20 * ((index + 1) / max(len(text_entries), 1)),
                        f"文本抽取已处理 {index + 1}/{len(text_entries)} 个条目",
                    )

            for entry_id, triple_ids in text_entry_triple_ids.items():
                self.store.update_text_entry_kg_triple_ids(entry_id, triple_ids)

            if text_sources:
                if active_chunking_config is None or embedding_function is None:
                    raise BuildError("TEXT_INDEX_CONFIG_FAILED", "文本索引缺少 embedding 配置")
                try:
                    active_vector_store = self._configured_vector_store()
                    if active_vector_store is not None:
                        vector_collection_name = make_collection_name(
                            self.qdrant_collection_prefix,
                            project_id,
                            version.version_id,
                        )
                    text_index_summary = active_text_index_builder.build(
                        workspace / "indexes" / "text",
                        self.store.list_text_entries(version.version_id),
                        parent_records,
                        embedding_function=embedding_function,
                        embedding_model=active_chunking_config.embedding_model,
                        progress=lambda value, message: self._report(
                            progress,
                            0.85 + 0.10 * value,
                            message,
                        ),
                        vector_store=active_vector_store,
                        collection_name=vector_collection_name,
                        project_id=project_id,
                        version_id=version.version_id,
                    )
                except (TextIndexBuildError, VectorStoreError) as exc:
                    raise BuildError("TEXT_INDEX_FAILED", str(exc)) from exc

            linked_triples = {
                entry_id: list(triple_ids)
                for entry_id, triple_ids in text_entry_triple_ids.items()
            }
            for row in stored_chunks:
                entry_id = row.get("entry_id")
                row["kg_triple_ids"] = linked_triples.get(str(entry_id), []) if entry_id else []

            counts = self.store.version_counts(version.version_id)
            if counts["kg_triples"] == 0:
                raise BuildError("KG_EMPTY", "没有生成有效 KG 三元组，版本不能标记为完整可用")

            self._write_artifacts(
                workspace,
                version.version_id,
                version.project_id,
                effective_mode,
                text_entries=self.store.list_text_entries(version.version_id),
                triples=self.store.list_kg_triples(version.version_id),
                conflicts=self.store.list_kg_conflicts(version.version_id),
                counts=counts,
                parse_issues=parse_issues,
                duplicate_count=duplicate_count,
                chunks=stored_chunks,
                chunking=chunking_summary,
                text_index=text_index_summary,
            )
            if text_sources and text_index_summary.get("status") == "ready":
                self.store.upsert_vector_index(
                    project_id,
                    version.version_id,
                    "text",
                    text_index_summary,
                    status="ready",
                )
            reference_export = ReferenceKGExporter(self.store).export(
                version.version_id,
                workspace / "algorithm" / "reference_kg",
            )
            self.files.publish_version_workspace(project_id, version.version_id)
            final_status = "ready" if text_sources else "graph_only"
            final_version = self.store.publish_version(
                version.version_id,
                status=final_status,
                text_status="ready" if text_sources else "not_provided",
                kg_status="ready",
            )
            for source in sources:
                self.store.update_source_parse_status(source.source_id, "parsed")
            self._report(progress, 1.0, "知识库版本已发布")
            return {
                "version": final_version.to_dict(),
                "counts": counts,
                "duplicate_triples_skipped": duplicate_count,
                "conflicts_detected": total_conflicts,
                "parse_issues": parse_issues,
                "reference_export": reference_export,
                "chunking": chunking_summary,
                "text_index": text_index_summary,
            }
        except BuildError as exc:
            if active_vector_store is not None and vector_collection_name:
                try:
                    active_vector_store.delete(vector_collection_name)
                except VectorStoreError:
                    pass
            self.store.update_version_status(
                version.version_id,
                status="failed",
                text_status="failed" if text_sources else "not_provided",
                kg_status="failed" if needs_llm else "not_provided",
                error_code=exc.code,
                error_message=exc.message,
            )
            if workspace is not None:
                self.files.discard_version_workspace(project_id, version.version_id)
            raise
        except (FileStoreError, OSError) as exc:
            self.store.update_version_status(
                version.version_id,
                status="failed",
                text_status="failed" if text_sources else "not_provided",
                kg_status="failed" if needs_llm else "not_provided",
                error_code="STORAGE_FAILED",
                error_message=str(exc),
            )
            if workspace is not None:
                self.files.discard_version_workspace(project_id, version.version_id)
            raise BuildError("STORAGE_FAILED", str(exc)) from exc

    def vector_store_summary(self) -> dict[str, object]:
        """Expose storage selection without instantiating an optional client."""

        if self.vector_store_backend == "local_file":
            return {
                "backend": "local_file",
                "status": "ready",
                "description": "版本目录内的可检查本地向量索引",
                "query_mode": "exact_scan",
            }
        if self.vector_store_backend == "qdrant":
            return {
                "backend": "qdrant",
                "status": "configured",
                "location": self.qdrant_url or str(self.vector_store_root / "qdrant"),
                "collection_strategy": "one_collection_per_version",
                "payload_policy": "only_ids_and_filters",
                "on_disk": self.qdrant_on_disk,
            }
        return {
            "backend": self.vector_store_backend,
            "status": "unsupported",
        }

    def _configured_vector_store(self) -> VectorStore | None:
        if self.vector_store_backend == "local_file":
            return None
        if self.vector_store_backend != "qdrant":
            raise VectorStoreError(
                f"不支持的向量存储后端：{self.vector_store_backend}，可选 local_file 或 qdrant"
            )
        if self._vector_store is None:
            local_path = None if self.qdrant_url else self.vector_store_root / "qdrant"
            self._vector_store = QdrantVectorStore(
                url=self.qdrant_url,
                path=local_path,
                api_key=self.qdrant_api_key,
                on_disk=self.qdrant_on_disk,
            )
        return self._vector_store

    @staticmethod
    def _resolve_mode(mode: str, has_kg: bool) -> str:
        del has_kg
        if mode in {"auto", "supplement_extract", "import_only"}:
            return "supplement_extract"
        raise BuildError("INVALID_MODE", "当前构建模式必须是 auto 或 supplement_extract")

    @staticmethod
    def _report(progress: ProgressCallback | None, value: float, message: str) -> None:
        if progress is not None:
            progress(round(value, 4), message)

    def _effective_chunking_config(
        self,
        build_settings: BuildSettings | None = None,
    ) -> ChunkingConfig:
        if self.chunking_config is not None:
            return self.chunking_config

        active_settings = build_settings or self.build_settings
        configured_model = (
            self.model_gateway.settings.embedding_model or "BAAI/bge-small-zh-v1.5"
        ).strip()
        return active_settings.to_chunking_config(
            self._resolve_local_embedding_model(configured_model)
        )

    def chunking_plan(self) -> dict[str, object]:
        """Expose the active build contract without loading the embedding model."""

        config = self._effective_chunking_config()
        model_path = Path(config.embedding_model) if config.embedding_model else None
        embedding_ready = bool(
            model_path
            and model_path.is_dir()
            and (model_path / "config.json").is_file()
        )
        return {
            "strategy": config.normalized_strategy(),
            "retrieval_unit": "child_chunk",
            "token_limit": config.max_tokens,
            "min_tokens": config.min_tokens,
            "parent_token_limit": config.parent_max_tokens,
            "semantic_breakpoint_percentile": config.semantic_breakpoint_percentile,
            "include_heading": config.include_heading,
            "overlap_tokens": 0,
            "hierarchy_rule": "仅在识别到标题且标题下存在正文时启用父子层级",
            "embedding_model": model_path.name if model_path else config.embedding_model,
            "embedding_device": config.embedding_device,
            "embedding_ready": embedding_ready,
            "embedding_local_only": config.embedding_local_files_only,
            "tokenizer_model": config.tokenizer_model,
            "tokenizer_local_only": config.tokenizer_local_files_only,
            "index_batch_size": self.text_index_builder.batch_size,
            "llm_call_unit": "每个检索子块",
        }

    def configure_build_settings(self, settings: BuildSettings) -> None:
        """Apply validated settings to subsequent builds in this process."""

        self.build_settings = settings.validate()
        self.text_index_builder.batch_size = self.build_settings.text_index_batch_size

    def _resolve_local_embedding_model(self, configured_model: str) -> str:
        """Prefer the project's downloaded model without changing global caches."""

        if not configured_model:
            return configured_model
        raw = Path(configured_model).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    self.models_root / "embedding" / raw.name,
                    self.models_root / raw,
                ]
            )
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return str(candidate.resolve())
        return configured_model

    @staticmethod
    def _empty_chunking_summary() -> dict[str, object]:
        return {
            "strategy": None,
            "config": None,
            "source_count": 0,
            "retrieval_chunk_count": 0,
            "parent_chunk_count": 0,
            "sources": [],
        }

    @staticmethod
    def _chunking_config_summary(
        config: ChunkingConfig,
        *,
        text_index_batch_size: int | None = None,
    ) -> dict[str, object]:
        return {
            "strategy": config.normalized_strategy(),
            "max_tokens": config.max_tokens,
            "min_tokens": config.min_tokens,
            "parent_max_tokens": config.parent_max_tokens,
            "child_max_tokens": config.child_max_tokens,
            "include_heading": config.include_heading,
            "semantic_breakpoint_percentile": config.semantic_breakpoint_percentile,
            "tokenizer_model": config.tokenizer_model,
            "tokenizer_local_files_only": config.tokenizer_local_files_only,
            "embedding_model": config.embedding_model,
            "embedding_device": config.embedding_device,
            "embedding_local_files_only": config.embedding_local_files_only,
            "text_index_batch_size": text_index_batch_size,
            "overlap_tokens": 0,
            "hierarchy_rule": "仅在识别到标题且标题下存在正文时启用父子层级",
        }

    @staticmethod
    def _chunk_title(source_name: str, chunk: Chunk) -> str:
        if not chunk.title_path:
            return source_name
        return f"{source_name} / {' / '.join(chunk.title_path)}"

    @staticmethod
    def _chunk_source_location(chunk: Chunk) -> dict[str, object]:
        location = dict(chunk.source_location)
        location.update(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_strategy": chunk.strategy,
                "chunk_level": chunk.level,
                "parent_id": chunk.parent_id,
                "title_path": list(chunk.title_path),
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "block_ids": list(chunk.block_ids),
                "token_count": chunk.token_count,
                "chunk_metadata": chunk.metadata,
            }
        )
        return location

    @classmethod
    def _chunk_record(
        cls,
        source: Any,
        chunk: Chunk,
        *,
        entry_id: str | None,
        title: str,
    ) -> dict[str, object]:
        return {
            "chunk_id": chunk.chunk_id,
            "entry_id": entry_id,
            "source_id": source.source_id,
            "source_name": source.original_name,
            "document_id": chunk.document_id,
            "title": title,
            "title_path": list(chunk.title_path),
            "text": chunk.text,
            "strategy": chunk.strategy,
            "level": chunk.level,
            "token_count": chunk.token_count,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "block_ids": list(chunk.block_ids),
            "parent_id": chunk.parent_id,
            "source_location": cls._chunk_source_location(chunk),
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _write_artifacts(
        workspace: Path,
        version_id: str,
        project_id: str,
        mode: str,
        *,
        text_entries: list,
        triples: list,
        conflicts: list[dict[str, object]],
        counts: dict[str, int],
        parse_issues: list[dict[str, object]],
        duplicate_count: int,
        chunks: list[dict[str, object]],
        chunking: dict[str, object],
        text_index: dict[str, object],
    ) -> None:
        BuildService._write_jsonl(workspace / "documents" / "text_entries.jsonl", [entry.to_dict() for entry in text_entries])
        BuildService._write_jsonl(workspace / "documents" / "chunks.jsonl", chunks)
        BuildService._write_jsonl(workspace / "graph" / "triples.jsonl", [triple.to_dict() for triple in triples])
        BuildService._write_jsonl(workspace / "graph" / "conflicts.jsonl", conflicts)
        manifest = {
            "version_id": version_id,
            "project_id": project_id,
            "mode": mode,
            "counts": counts,
            "duplicate_triples_skipped": duplicate_count,
            "parse_issues": parse_issues,
            "chunking": chunking,
            "text_index": text_index,
        }
        (workspace / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
