"""Versioned text records, parent lookup and local vector indexing.

The text index deliberately keeps three concerns separate:

* SQLite remains the canonical store for text entries and KG links;
* ``records.jsonl`` is the compact, inspectable retrieval-record export;
* the vector file is a rebuildable document-side index keyed by physical vector
  position, with one-to-many logical ``entry_id`` mappings.

Parent chunks are written to a lookup file only.  They never receive an
embedding and therefore cannot accidentally become a second retrieval unit.
The vector implementation uses NumPy when available and falls back to a
portable JSONL representation.  This keeps the base API environment intact;
an ANN backend can later implement the same ``TextVectorIndex`` boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from backend.app.domain.models import TextEntry
from backend.app.indexing.vector_store import (
    VectorPoint,
    VectorStore,
    VectorStoreError,
    qdrant_point_payload,
)


ProgressCallback = Callable[[float, str], None]


class TextIndexBuildError(RuntimeError):
    """A text index cannot be built or read safely."""


class EmbeddingFunction(Protocol):
    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class TextIndexManifest:
    """Index-level metadata; it is intentionally not duplicated per chunk."""

    schema_version: int
    index_type: str
    retrieval_unit: str
    embedding_model: str
    embedding_input_rule: str
    dimension: int
    metric: str
    normalized: bool
    vector_count: int
    logical_entry_count: int
    duplicate_entry_count: int
    parent_count: int
    storage_format: str
    content_fingerprint: str
    backend: str = "local_file"
    storage_uri: str = ""
    collection_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ready",
            "schema_version": self.schema_version,
            "index_type": self.index_type,
            "retrieval_unit": self.retrieval_unit,
            "embedding_model": self.embedding_model,
            "embedding_input_rule": self.embedding_input_rule,
            "dimension": self.dimension,
            "metric": self.metric,
            "normalized": self.normalized,
            "vector_count": self.vector_count,
            "logical_entry_count": self.logical_entry_count,
            "duplicate_entry_count": self.duplicate_entry_count,
            "parent_count": self.parent_count,
            "storage_format": self.storage_format,
            "content_fingerprint": self.content_fingerprint,
            "backend": self.backend,
            "storage_uri": self.storage_uri,
            "collection_name": self.collection_name,
        }


@dataclass(frozen=True)
class TextVectorHit:
    """A vector result before loading the canonical text entry."""

    chunk_id: str
    score: float
    rank: int
    chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        chunk_ids = self.chunk_ids or (self.chunk_id,)
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "chunk_ids": list(chunk_ids),
        }


@dataclass
class _ContentGroup:
    """Logical entries sharing one exact content hash and content string."""

    content_hash: str
    content: str
    entries: list[TextEntry]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextIndexBuildError(f"文本索引清单无法读取：{path.name}") from exc
    if not isinstance(value, dict):
        raise TextIndexBuildError("文本索引清单格式无效")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TextIndexBuildError(f"文本索引文件无法读取：{path.name}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TextIndexBuildError(f"文本索引第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(value, dict):
            raise TextIndexBuildError(f"文本索引第 {line_number} 行不是对象")
        rows.append(value)
    return rows


def _normalise_vector(value: Any, *, index: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TextIndexBuildError(f"第 {index + 1} 个文本向量不是数组")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TextIndexBuildError(f"第 {index + 1} 个文本向量包含非数字内容") from exc
    if not vector:
        raise TextIndexBuildError(f"第 {index + 1} 个文本向量为空")
    if not all(math.isfinite(item) for item in vector):
        raise TextIndexBuildError(f"第 {index + 1} 个文本向量包含 NaN 或无穷值")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0:
        raise TextIndexBuildError(f"第 {index + 1} 个文本向量为零向量")
    return [item / norm for item in vector]


def _model_label(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.is_dir():
        return candidate.name
    return model_name


def _minimal_location(entry: TextEntry) -> dict[str, object]:
    """Keep only source-position information in the retrieval record.

    The chunking layer may attach diagnostics to ``source_location``.  Those
    details remain available in ``documents/chunks.jsonl``; the runtime text
    index keeps only the location needed for evidence display and fallback
    context lookup.
    """

    source_location = entry.source_location or {}
    allowed = (
        "title_path",
        "line_start",
        "line_end",
        "paragraph_index",
        "table_index",
        "block_ids",
        "char_start",
        "char_end",
    )
    location = {
        key: source_location[key]
        for key in allowed
        if key in source_location and source_location[key] not in (None, [], "")
    }
    return location


def _record_for_entry(entry: TextEntry) -> dict[str, object]:
    return {
        "chunk_id": entry.entry_id,
        "content": entry.content,
        "location": _minimal_location(entry),
        "parent_id": entry.parent_id,
        "content_hash": entry.content_hash,
        "kg_triple_ids": list(entry.kg_triple_ids),
    }


def _parent_record(value: dict[str, Any]) -> dict[str, object]:
    parent_id = str(value.get("parent_id") or value.get("chunk_id") or "").strip()
    content = str(value.get("content") or value.get("text") or "")
    if not parent_id or not content.strip():
        raise TextIndexBuildError("父块索引记录缺少 parent_id 或 content")
    location = value.get("location")
    if not isinstance(location, dict):
        source_location = value.get("source_location")
        location = source_location if isinstance(source_location, dict) else {}
    return {
        "parent_id": parent_id,
        "content": content,
        "location": _minimal_parent_location(location),
    }


def _minimal_parent_location(source_location: dict[str, Any]) -> dict[str, object]:
    allowed = (
        "title_path",
        "line_start",
        "line_end",
        "paragraph_index",
        "table_index",
        "block_ids",
        "char_start",
        "char_end",
    )
    return {
        key: source_location[key]
        for key in allowed
        if key in source_location and source_location[key] not in (None, [], "")
    }


def _content_fingerprint(entries: Sequence[TextEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            json.dumps(
                {
                    "chunk_id": entry.entry_id,
                    "content_hash": entry.content_hash,
                    "location": _minimal_location(entry),
                    "parent_id": entry.parent_id,
                    "kg_triple_ids": list(entry.kg_triple_ids),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _group_entries_by_content(entries: Sequence[TextEntry]) -> list[_ContentGroup]:
    """Group only exact content matches with the same declared hash.

    The logical entries are retained in ``entries`` and only the physical
    vector is shared.  Checking the content in addition to the hash prevents a
    theoretical hash collision from merging two different passages.
    """

    groups: list[_ContentGroup] = []
    groups_by_key: dict[tuple[str, str], _ContentGroup] = {}
    for entry in entries:
        key = (entry.content_hash, entry.content)
        group = groups_by_key.get(key)
        if group is None:
            group = _ContentGroup(
                content_hash=entry.content_hash,
                content=entry.content,
                entries=[],
            )
            groups_by_key[key] = group
            groups.append(group)
        group.entries.append(entry)
    return groups


class _LocalVectorWriter:
    """Write local vectors incrementally without retaining the full matrix."""

    def __init__(self, output_dir: Path, vector_count: int) -> None:
        self.output_dir = output_dir
        self.vector_count = vector_count
        self._numpy = None
        self._matrix = None
        self._json_stream = None
        self._dimension: int | None = None
        self._numpy_part = output_dir / ".vectors.npy.part"
        self._json_part = output_dir / ".vectors.jsonl.part"
        try:
            import numpy as np
        except ImportError:
            self._json_stream = self._json_part.open("w", encoding="utf-8")
        else:
            self._numpy = np

    def write(self, start: int, vectors: Sequence[Sequence[float]]) -> None:
        if not vectors:
            return
        dimension = len(vectors[0])
        if self._dimension is None:
            self._dimension = dimension
            if self._numpy is not None:
                self._matrix = self._numpy.lib.format.open_memmap(
                    self._numpy_part,
                    mode="w+",
                    dtype=self._numpy.float32,
                    shape=(self.vector_count, dimension),
                )
        if dimension != self._dimension:
            raise TextIndexBuildError("文本向量维度不一致，不能建立同一索引")

        if self._matrix is not None:
            self._matrix[start : start + len(vectors)] = self._numpy.asarray(
                vectors,
                dtype=self._numpy.float32,
            )
            self._matrix.flush()
            return

        if self._json_stream is None:  # pragma: no cover - defensive branch
            raise TextIndexBuildError("本地向量写入器未初始化")
        for offset, vector in enumerate(vectors, start=start):
            self._json_stream.write(
                json.dumps(
                    {"vector_index": offset, "vector": list(vector)},
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._json_stream.flush()

    def finish(self) -> str:
        if self._matrix is not None:
            self._matrix.flush()
            self._matrix = None
            self._numpy_part.replace(self.output_dir / "vectors.npy")
            return "numpy_float32"
        if self.vector_count == 0 and self._numpy is not None:
            temporary = self.output_dir / ".vectors.npy.part"
            with temporary.open("wb") as stream:
                self._numpy.save(
                    stream,
                    self._numpy.empty((0, 0), dtype=self._numpy.float32),
                    allow_pickle=False,
                )
            temporary.replace(self.output_dir / "vectors.npy")
            return "numpy_float32"
        if self._json_stream is None:  # pragma: no cover - defensive branch
            raise TextIndexBuildError("本地向量写入器未初始化")
        self._json_stream.flush()
        self._json_stream.close()
        self._json_stream = None
        self._json_part.replace(self.output_dir / "vectors.jsonl")
        return "jsonl_float64"

    def abort(self) -> None:
        if self._json_stream is not None:
            self._json_stream.close()
            self._json_stream = None
        self._matrix = None
        for path in (self._numpy_part, self._json_part):
            if path.is_file():
                path.unlink()


class TextIndexBuilder:
    """Build the version-local text record, parent and vector indexes."""

    def __init__(self, *, batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.batch_size = batch_size

    def build(
        self,
        output_dir: Path,
        entries: Sequence[TextEntry],
        parents: Iterable[dict[str, Any]],
        *,
        embedding_function: EmbeddingFunction,
        embedding_model: str,
        progress: ProgressCallback | None = None,
        vector_store: VectorStore | None = None,
        collection_name: str | None = None,
        project_id: str | None = None,
        version_id: str | None = None,
    ) -> dict[str, object]:
        """Build all text-side artifacts below one version workspace.

        ``entries`` are the only vectorized units.  Parent records are stored
        as a direct lookup table and are never passed to the embedding model.
        """

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_entries(entries)
        parent_rows = self._deduplicate_parents(parents)
        parent_ids = {str(row["parent_id"]) for row in parent_rows}
        missing_parents = sorted(
            {
                entry.parent_id
                for entry in entries
                if entry.parent_id and entry.parent_id not in parent_ids
            }
        )
        if missing_parents:
            raise TextIndexBuildError(
                f"文本条目引用了不存在的父块：{', '.join(missing_parents)}"
            )

        _write_jsonl(output_dir / "records.jsonl", (_record_for_entry(entry) for entry in entries))
        _write_jsonl(output_dir / "parents.jsonl", parent_rows)
        self._report(progress, 0.05, f"文本索引记录已写入 {len(entries)} 个子块")

        content_groups = _group_entries_by_content(entries)
        logical_entry_count = len(entries)
        vector_count = len(content_groups)
        duplicate_entry_count = logical_entry_count - vector_count
        self._report(
            progress,
            0.06,
            f"内容哈希去重完成：{logical_entry_count} 个逻辑条目，{vector_count} 个物理向量，去重 {duplicate_entry_count} 个",
        )

        backend = "local_file"
        storage_uri = str(output_dir)
        stored_collection_name: str | None = None
        streaming_backend = False
        pending_points: list[VectorPoint] = []
        if vector_store is not None:
            if not collection_name or not project_id or not version_id:
                raise TextIndexBuildError(
                    "使用外部向量库时必须提供 project_id、version_id 和 collection_name"
                )
            backend = vector_store.backend_name
            storage_uri = str(getattr(vector_store, "location", ""))
            stored_collection_name = collection_name
            streaming_backend = all(
                callable(getattr(vector_store, method, None))
                for method in ("begin", "upsert_batch", "finish")
            )

        local_writer = (
            _LocalVectorWriter(output_dir, vector_count)
            if vector_store is None
            else None
        )
        dimension = 0
        processed_vectors = 0
        backend_started = False
        try:
            for start in range(0, len(content_groups), self.batch_size):
                batch_groups = content_groups[start : start + self.batch_size]
                batch_texts = [group.content for group in batch_groups]
                try:
                    raw_vectors = list(embedding_function(batch_texts))
                except Exception as exc:  # pragma: no cover - provider-specific
                    raise TextIndexBuildError(f"文本向量生成失败：{exc}") from exc
                if len(raw_vectors) != len(batch_groups):
                    raise TextIndexBuildError(
                        f"文本向量数量不匹配：期望 {len(batch_groups)}，实际 {len(raw_vectors)}"
                    )

                batch_vectors: list[list[float]] = []
                for offset, raw_vector in enumerate(raw_vectors):
                    vector = _normalise_vector(
                        raw_vector,
                        index=start + offset,
                    )
                    if dimension and len(vector) != dimension:
                        raise TextIndexBuildError("文本向量维度不一致，不能建立同一索引")
                    dimension = len(vector)
                    batch_vectors.append(vector)

                if vector_store is None:
                    if local_writer is None:  # pragma: no cover - defensive branch
                        raise TextIndexBuildError("本地向量写入器未初始化")
                    local_writer.write(start, batch_vectors)
                else:
                    batch_points = [
                        VectorPoint(
                            chunk_id=group.entries[0].entry_id,
                            vector=vector,
                            payload=qdrant_point_payload(
                                project_id=project_id or "",
                                version_id=version_id or "",
                                source_id=group.entries[0].source_id,
                                parent_id=group.entries[0].parent_id,
                                content_hash=group.content_hash,
                                chunk_ids=[entry.entry_id for entry in group.entries],
                                source_ids=list(
                                    dict.fromkeys(
                                        entry.source_id for entry in group.entries
                                    )
                                ),
                                parent_ids=list(
                                    dict.fromkeys(
                                        entry.parent_id
                                        for entry in group.entries
                                        if entry.parent_id
                                    )
                                ),
                            ),
                        )
                        for group, vector in zip(
                            batch_groups,
                            batch_vectors,
                            strict=True,
                        )
                    ]
                    if streaming_backend:
                        try:
                            if not backend_started:
                                vector_store.begin(
                                    collection_name=collection_name or "",
                                    dimension=dimension,
                                )
                                backend_started = True
                            vector_store.upsert_batch(
                                collection_name=collection_name or "",
                                points=batch_points,
                                dimension=dimension,
                            )
                        except VectorStoreError as exc:
                            raise TextIndexBuildError(str(exc)) from exc
                        except Exception as exc:  # pragma: no cover - backend-specific
                            raise TextIndexBuildError(f"外部向量库批次写入失败：{exc}") from exc
                    else:
                        # Compatibility path for older/custom backends.  Qdrant
                        # uses the streaming branch above, so it never retains
                        # the complete point list in the builder.
                        pending_points.extend(batch_points)

                processed_vectors += len(batch_vectors)
                self._report(
                    progress,
                    0.06 + 0.72 * (processed_vectors / max(vector_count, 1)),
                    f"文本向量化与批量写入已处理 {processed_vectors}/{vector_count} 个物理向量（逻辑条目 {logical_entry_count}）",
                )

            if vector_store is None:
                if local_writer is None:  # pragma: no cover - defensive branch
                    raise TextIndexBuildError("本地向量写入器未初始化")
                storage_format = local_writer.finish()
            elif streaming_backend:
                if not backend_started:
                    raise TextIndexBuildError("没有生成有效向量，无法初始化外部向量库")
                try:
                    vector_store.finish(
                        collection_name=collection_name or "",
                        dimension=dimension,
                        vector_count=processed_vectors,
                    )
                except VectorStoreError as exc:
                    raise TextIndexBuildError(str(exc)) from exc
                except Exception as exc:  # pragma: no cover - backend-specific
                    raise TextIndexBuildError(f"外部向量库收尾失败：{exc}") from exc
                storage_format = f"{backend}_collection"
            else:
                try:
                    vector_store.build(
                        collection_name=collection_name or "",
                        points=pending_points,
                        dimension=dimension,
                        progress=lambda value, message: self._report(
                            progress,
                            0.78 + 0.10 * value,
                            message,
                        ),
                    )
                except VectorStoreError as exc:
                    raise TextIndexBuildError(str(exc)) from exc
                except Exception as exc:  # pragma: no cover - backend-specific
                    raise TextIndexBuildError(f"外部向量库写入失败：{exc}") from exc
                storage_format = f"{backend}_collection"
        except Exception:
            if local_writer is not None:
                local_writer.abort()
            raise

        _write_jsonl(
            output_dir / "mapping.jsonl",
            (
                {
                    "vector_index": index,
                    "chunk_id": group.entries[0].entry_id,
                    "chunk_ids": [entry.entry_id for entry in group.entries],
                    "content_hash": group.content_hash,
                }
                for index, group in enumerate(content_groups)
            ),
        )
        self._report(progress, 0.88, "文本向量索引映射已建立（重复内容共享物理向量）")

        manifest = TextIndexManifest(
            schema_version=1,
            index_type="text",
            retrieval_unit="child_chunk",
            embedding_model=_model_label(embedding_model),
            embedding_input_rule="chunk_content_unique_by_content_hash",
            dimension=dimension,
            metric="cosine",
            normalized=True,
            vector_count=vector_count,
            logical_entry_count=logical_entry_count,
            duplicate_entry_count=duplicate_entry_count,
            parent_count=len(parent_rows),
            storage_format=storage_format,
            content_fingerprint=_content_fingerprint(entries),
            backend=backend,
            storage_uri=storage_uri,
            collection_name=stored_collection_name,
        )
        _write_json(output_dir / "manifest.json", manifest.to_dict())
        self._report(progress, 1.0, "文本索引构建完成")
        return {
            "status": "ready",
            "directory": "indexes/text",
            **manifest.to_dict(),
        }

    @staticmethod
    def is_current(
        index_dir: Path,
        entries: Sequence[TextEntry],
        *,
        embedding_model: str,
    ) -> bool:
        """Return whether an existing index matches the current text inputs."""

        manifest = read_text_index_manifest(index_dir)
        if manifest is None:
            return False
        required_files = (
            index_dir / "records.jsonl",
            index_dir / "parents.jsonl",
            index_dir / "mapping.jsonl",
        )
        if not all(path.is_file() for path in required_files):
            return False
        storage_format = manifest.get("storage_format")
        if manifest.get("backend", "local_file") != "local_file":
            if not manifest.get("collection_name"):
                return False
        else:
            vector_file = (
                index_dir / "vectors.npy"
                if storage_format == "numpy_float32"
                else index_dir / "vectors.jsonl"
            )
            if not vector_file.is_file():
                return False
        expected_vector_count = len(_group_entries_by_content(entries))
        try:
            stored_logical_count = int(
                manifest.get("logical_entry_count", manifest.get("vector_count", -1))
            )
        except (TypeError, ValueError):
            return False
        return (
            manifest.get("schema_version") == 1
            and manifest.get("index_type") == "text"
            and manifest.get("retrieval_unit") == "child_chunk"
            and manifest.get("embedding_model") == _model_label(embedding_model)
            and manifest.get("embedding_input_rule")
            in {"chunk_content", "chunk_content_unique_by_content_hash"}
            and manifest.get("vector_count") == expected_vector_count
            and stored_logical_count == len(entries)
            and manifest.get("content_fingerprint") == _content_fingerprint(entries)
        )

    @staticmethod
    def _validate_entries(entries: Sequence[TextEntry]) -> None:
        seen: set[str] = set()
        for entry in entries:
            if not entry.entry_id.strip():
                raise TextIndexBuildError("文本条目缺少唯一 ID")
            if entry.entry_id in seen:
                raise TextIndexBuildError(f"文本条目 ID 重复：{entry.entry_id}")
            if not entry.content.strip():
                raise TextIndexBuildError(f"文本条目内容为空：{entry.entry_id}")
            if not entry.content_hash:
                raise TextIndexBuildError(f"文本条目缺少哈希：{entry.entry_id}")
            seen.add(entry.entry_id)

    @staticmethod
    def _deduplicate_parents(parents: Iterable[dict[str, Any]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for parent in parents:
            row = _parent_record(parent)
            if row["parent_id"] in seen:
                raise TextIndexBuildError(f"父块 ID 重复：{row['parent_id']}")
            seen.add(str(row["parent_id"]))
            rows.append(row)
        return rows

    @staticmethod
    def _report(progress: ProgressCallback | None, value: float, message: str) -> None:
        if progress is not None:
            progress(round(value, 4), message)


def read_text_index_manifest(index_dir: Path) -> dict[str, Any] | None:
    """Read an index manifest, returning ``None`` for an unbuilt index."""

    path = index_dir / "manifest.json"
    if not path.is_file():
        return None
    return _read_json(path)


class TextVectorIndex:
    """Read-only text vector index with local-file or external backends."""

    def __init__(self, index_dir: Path, vector_store: VectorStore | None = None) -> None:
        self.index_dir = index_dir.resolve()
        manifest = read_text_index_manifest(self.index_dir)
        if manifest is None:
            raise TextIndexBuildError("文本向量索引尚未构建")
        self.manifest = manifest
        self._vector_store = vector_store
        self._collection_name = str(manifest.get("collection_name") or "").strip()
        self._uses_external_backend = manifest.get("backend", "local_file") != "local_file"
        mapping_rows = _read_jsonl(self.index_dir / "mapping.jsonl")
        self._chunk_ids = self._load_mapping(mapping_rows)
        if self._uses_external_backend:
            if vector_store is None:
                raise TextIndexBuildError("该文本索引使用外部向量库，请传入已连接的向量存储后端")
            if not self._collection_name:
                raise TextIndexBuildError("Qdrant 文本索引缺少 collection_name")
            self._vectors: list[list[float]] = []
        else:
            self._vectors = self._load_vectors()
        self._parents = {
            str(row.get("parent_id")): row
            for row in _read_jsonl(self.index_dir / "parents.jsonl")
            if str(row.get("parent_id") or "").strip()
        }
        if not self._uses_external_backend and len(self._chunk_ids) != len(self._vectors):
            raise TextIndexBuildError("文本向量索引的映射数量与向量数量不一致")

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[TextVectorHit]:
        if top_k <= 0:
            return []
        query = _normalise_vector(query_vector, index=0)
        if len(query) != int(self.manifest.get("dimension", 0)):
            raise TextIndexBuildError("查询向量维度与文本索引不一致")
        if self._uses_external_backend:
            if self._vector_store is None:
                raise TextIndexBuildError("外部向量索引缺少向量存储后端")
            try:
                hits = self._vector_store.search(
                    collection_name=self._collection_name,
                    query_vector=query,
                    top_k=top_k,
                    filters=filters,
                )
            except VectorStoreError as exc:
                raise TextIndexBuildError(str(exc)) from exc
            return [
                TextVectorHit(
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    rank=hit.rank,
                    chunk_ids=hit.chunk_ids or (hit.chunk_id,),
                )
                for hit in hits
            ]
        if not self._vectors:
            return []
        scores = [sum(value * query[index] for index, value in enumerate(vector)) for vector in self._vectors]
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            TextVectorHit(
                chunk_id=self._chunk_ids[index][0],
                score=float(score),
                rank=rank + 1,
                chunk_ids=self._chunk_ids[index],
            )
            for rank, (index, score) in enumerate(ranked)
        ]

    def get_parent(self, parent_id: str) -> dict[str, Any] | None:
        return self._parents.get(parent_id)

    @staticmethod
    def _load_mapping(rows: Sequence[dict[str, Any]]) -> list[tuple[str, ...]]:
        ordered: list[tuple[str, ...]] = []
        for expected_index, row in enumerate(rows):
            if int(row.get("vector_index", -1)) != expected_index:
                raise TextIndexBuildError("文本索引映射的向量序号不连续")
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                raise TextIndexBuildError("文本索引映射缺少 chunk_id")
            raw_chunk_ids = row.get("chunk_ids")
            if isinstance(raw_chunk_ids, (list, tuple)):
                aliases = tuple(
                    str(value).strip()
                    for value in raw_chunk_ids
                    if str(value).strip()
                )
            else:
                aliases = ()
            if not aliases:
                aliases = (chunk_id,)
            elif chunk_id not in aliases:
                aliases = (chunk_id, *aliases)
            ordered.append(aliases)
        return ordered

    def _load_vectors(self) -> list[list[float]]:
        storage_format = self.manifest.get("storage_format")
        if storage_format == "numpy_float32" and (self.index_dir / "vectors.npy").is_file():
            try:
                import numpy as np

                matrix = np.load(self.index_dir / "vectors.npy", allow_pickle=False)
                if getattr(matrix, "ndim", None) != 2:
                    raise TextIndexBuildError("文本向量文件不是二维矩阵")
                return [
                    _normalise_vector(row, index=index)
                    for index, row in enumerate(matrix.tolist())
                ]
            except ImportError as exc:
                raise TextIndexBuildError("读取 NumPy 文本索引需要 numpy") from exc
        rows = _read_jsonl(self.index_dir / "vectors.jsonl")
        vectors: list[list[float]] = []
        for expected_index, row in enumerate(rows):
            if int(row.get("vector_index", -1)) != expected_index:
                raise TextIndexBuildError("文本向量文件的向量序号不连续")
            vectors.append(_normalise_vector(row.get("vector"), index=expected_index))
        return vectors
