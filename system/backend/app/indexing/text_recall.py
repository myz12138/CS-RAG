"""Read-only bridge from the CS-RAG reference process to the text index.

The reference Phase2 process is intentionally kept in its own subprocess.
This module gives that process the same read boundary as the application: it
can query the version-pinned text index without knowing how vectors are stored.
The bridge does not load or generate any LLM output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from backend.app.indexing.text_index import (
    TextIndexBuildError,
    TextVectorHit,
    TextVectorIndex,
    read_text_index_manifest,
)
from backend.app.indexing.vector_store import QdrantVectorStore, VectorStore


class TextIndexRecall:
    """Read a published text index through its configured vector backend."""

    def __init__(
        self,
        index_dir: Path,
        *,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.index_dir = index_dir.resolve()
        manifest = read_text_index_manifest(self.index_dir)
        if manifest is None:
            raise TextIndexBuildError("文本向量索引尚未构建")
        self.manifest = manifest
        self.index = TextVectorIndex(self.index_dir, vector_store=vector_store)

    @classmethod
    def from_environment(cls, index_dir: Path) -> "TextIndexRecall":
        """Construct the reader from query-process-only environment values.

        The manifest remains authoritative for the backend and collection. The
        environment supplies only the connection details, so a query cannot
        accidentally point at a different collection name.
        """

        resolved_dir = index_dir.resolve()
        manifest = read_text_index_manifest(resolved_dir)
        if manifest is None:
            raise TextIndexBuildError("文本向量索引尚未构建")

        backend = str(manifest.get("backend") or "local_file").strip().lower()
        if backend == "local_file":
            return cls(resolved_dir)
        if backend != "qdrant":
            raise TextIndexBuildError(f"文本索引使用了不支持的向量后端：{backend}")

        url = os.getenv("TEXT_INDEX_QDRANT_URL", "").strip()
        api_key = os.getenv("TEXT_INDEX_QDRANT_API_KEY", "")
        raw_path = os.getenv("TEXT_INDEX_QDRANT_PATH", "").strip()
        if raw_path:
            qdrant_path = Path(raw_path).expanduser().resolve()
        else:
            raw_root = os.getenv("TEXT_INDEX_VECTOR_STORE_ROOT", "").strip()
            qdrant_path = (
                Path(raw_root).expanduser().resolve() / "qdrant"
                if raw_root
                else None
            )
        on_disk = os.getenv("TEXT_INDEX_QDRANT_ON_DISK", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        vector_store = QdrantVectorStore(
            url=url,
            path=None if url else qdrant_path,
            api_key=api_key,
            on_disk=on_disk,
        )
        return cls(resolved_dir, vector_store=vector_store)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
    ) -> list[TextVectorHit]:
        return self.index.search(query_vector, top_k=top_k)


__all__ = ["TextIndexRecall"]
