"""Pluggable vector-storage backends for CS-RAG.

The application keeps SQLite as the source of truth.  A vector backend only
stores vectors plus the identifiers needed to filter and join a hit back to
SQLite.  This module intentionally imports the Qdrant client lazily so the
default local-file backend remains usable without changing the Python
environment.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5


ProgressCallback = Callable[[float, str], None]


class VectorStoreError(RuntimeError):
    """A vector backend cannot safely complete an operation."""


class VectorStoreUnavailableError(VectorStoreError):
    """A configured backend is not available in the current environment."""


@dataclass(frozen=True)
class VectorPoint:
    """One vector and its small, filterable payload."""

    chunk_id: str
    vector: Sequence[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchHit:
    """A backend result before loading the canonical text entry."""

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


class VectorStore(Protocol):
    """Common boundary used by the text-index builder and query layer."""

    backend_name: str

    def begin(self, *, collection_name: str, dimension: int) -> None: ...

    def upsert_batch(
        self,
        *,
        collection_name: str,
        points: Sequence[VectorPoint],
        dimension: int,
    ) -> int: ...

    def finish(
        self,
        *,
        collection_name: str,
        dimension: int,
        vector_count: int,
    ) -> dict[str, object]: ...

    def build(
        self,
        *,
        collection_name: str,
        points: Sequence[VectorPoint],
        dimension: int,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]: ...

    def search(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[VectorSearchHit]: ...

    def delete(self, collection_name: str) -> None: ...

    def health(self) -> dict[str, object]: ...


def normalise_vector(value: Sequence[float]) -> list[float]:
    """Validate and L2-normalise a vector for cosine similarity."""

    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise VectorStoreError("向量包含非数字内容") from exc
    if not vector:
        raise VectorStoreError("向量不能为空")
    if not all(math.isfinite(item) for item in vector):
        raise VectorStoreError("向量包含 NaN 或无穷值")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0:
        raise VectorStoreError("向量不能是零向量")
    return [item / norm for item in vector]


def make_collection_name(prefix: str, project_id: str, version_id: str) -> str:
    """Create a stable, short collection name without exposing user text."""

    clean_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", prefix).strip("-_") or "csrag"
    digest = hashlib.sha256(f"{project_id}:{version_id}:text".encode("utf-8")).hexdigest()[:20]
    return f"{clean_prefix}_text_{digest}"


def make_point_id(collection_name: str, chunk_id: str) -> str:
    """Qdrant accepts UUIDs; derive one so rebuilding is idempotent."""

    return str(uuid5(NAMESPACE_URL, f"cs-rag:{collection_name}:{chunk_id}"))


class QdrantVectorStore:
    """Qdrant adapter supporting both local persistence and a remote service.

    The import is lazy because Qdrant is an optional product dependency.  The
    adapter stores no full text: payloads contain IDs and filter fields only.
    """

    backend_name = "qdrant"

    def __init__(
        self,
        *,
        url: str = "",
        path: Path | None = None,
        api_key: str = "",
        on_disk: bool = True,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreUnavailableError(
                "Qdrant 后端未安装，请在项目环境中安装可选依赖 qdrant-client"
            ) from exc

        if url.strip():
            self.client = QdrantClient(url=url.strip(), api_key=api_key or None)
            self.location = url.strip()
        elif path is not None:
            path = path.resolve()
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.location = str(path)
        else:
            raise VectorStoreError("Qdrant 后端需要配置本地 path 或远程 url")
        self.on_disk = on_disk

    def build(
        self,
        *,
        collection_name: str,
        points: Sequence[VectorPoint],
        dimension: int,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        if dimension <= 0:
            raise VectorStoreError("向量维度必须大于 0")
        self.begin(collection_name=collection_name, dimension=dimension)
        batch_size = 128
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.upsert_batch(
                collection_name=collection_name,
                points=batch,
                dimension=dimension,
            )
            if progress is not None:
                processed = start + len(batch)
                progress(
                    round(processed / max(len(points), 1), 4),
                    f"Qdrant 已写入 {processed}/{len(points)} 个向量",
                )

        return self.finish(
            collection_name=collection_name,
            dimension=dimension,
            vector_count=len(points),
        )

    def begin(self, *, collection_name: str, dimension: int) -> None:
        if dimension <= 0:
            raise VectorStoreError("向量维度必须大于 0")
        self._create_collection(collection_name, dimension)

    def upsert_batch(
        self,
        *,
        collection_name: str,
        points: Sequence[VectorPoint],
        dimension: int,
    ) -> int:
        qdrant_points = [
            self._point_struct(collection_name, point, dimension)
            for point in points
        ]
        if not qdrant_points:
            return 0
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=True,
            )
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreError(f"Qdrant 向量批次写入失败：{exc}") from exc
        return len(qdrant_points)

    def finish(
        self,
        *,
        collection_name: str,
        dimension: int,
        vector_count: int,
    ) -> dict[str, object]:
        return {
            "status": "ready",
            "backend": self.backend_name,
            "collection_name": collection_name,
            "storage_uri": self.location,
            "dimension": dimension,
            "metric": "cosine",
            "vector_count": vector_count,
            "payload_policy": "仅保存 chunk_id、项目/版本/来源和过滤字段，不复制全文",
        }

    def search(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[VectorSearchHit]:
        if top_k <= 0:
            return []
        vector = normalise_vector(query_vector)
        query_filter = self._make_filter(filters or {})
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
                scored_points = list(getattr(response, "points", response))
            else:  # Compatibility with older qdrant-client releases.
                scored_points = self.client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreError(f"Qdrant 检索失败：{exc}") from exc

        hits: list[VectorSearchHit] = []
        for rank, point in enumerate(scored_points, start=1):
            payload = getattr(point, "payload", None) or {}
            chunk_id = str(payload.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            raw_chunk_ids = payload.get("chunk_ids")
            if isinstance(raw_chunk_ids, list):
                chunk_ids = tuple(
                    str(value).strip()
                    for value in raw_chunk_ids
                    if str(value).strip()
                )
            else:
                chunk_ids = ()
            if not chunk_ids:
                chunk_ids = (chunk_id,)
            hits.append(
                VectorSearchHit(
                    chunk_id=chunk_ids[0],
                    score=float(getattr(point, "score", 0.0)),
                    rank=rank,
                    chunk_ids=chunk_ids,
                )
            )
        return hits

    def delete(self, collection_name: str) -> None:
        try:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreError(f"Qdrant 集合删除失败：{exc}") from exc

    def health(self) -> dict[str, object]:
        try:
            collections = self.client.get_collections()
            names = [str(item.name) for item in getattr(collections, "collections", [])]
        except Exception as exc:  # pragma: no cover - depends on optional env
            return {
                "status": "unavailable",
                "backend": self.backend_name,
                "location": self.location,
                "message": str(exc),
            }
        return {
            "status": "ready",
            "backend": self.backend_name,
            "location": self.location,
            "collection_count": len(names),
        }

    def _create_collection(self, collection_name: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        try:
            exists = self.client.collection_exists(collection_name)
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreError(f"无法检查 Qdrant 集合：{exc}") from exc
        if exists:
            return

        params: dict[str, Any] = {"size": dimension, "distance": Distance.COSINE}
        if self.on_disk:
            params["on_disk"] = True
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(**params),
            )
        except TypeError:  # Compatibility with clients lacking VectorParams.on_disk.
            params.pop("on_disk", None)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(**params),
            )
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise VectorStoreError(f"Qdrant 集合创建失败：{exc}") from exc

    @staticmethod
    def _point_struct(
        collection_name: str,
        point: VectorPoint,
        dimension: int,
    ) -> Any:
        from qdrant_client.models import PointStruct

        vector = normalise_vector(point.vector)
        if len(vector) != dimension:
            raise VectorStoreError(f"向量维度不一致：{point.chunk_id}")
        payload = {
            "chunk_id": point.chunk_id,
            **point.payload,
        }
        return PointStruct(
            id=make_point_id(collection_name, point.chunk_id),
            vector=vector,
            payload=payload,
        )

    @staticmethod
    def _make_filter(filters: dict[str, str]) -> Any:
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(
                    key={"source_id": "source_ids", "parent_id": "parent_ids"}.get(
                        key,
                        key,
                    ),
                    match=MatchValue(value=value),
                )
                for key, value in filters.items()
                if str(key).strip() and str(value).strip()
            ]
        )


def qdrant_point_payload(
    *,
    project_id: str,
    version_id: str,
    source_id: str,
    parent_id: str | None,
    content_hash: str,
    chunk_ids: Sequence[str] | None = None,
    source_ids: Sequence[str] | None = None,
    parent_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the deliberately small payload stored beside a text vector."""

    return {
        "project_id": project_id,
        "version_id": version_id,
        "source_id": source_id,
        "source_ids": list(source_ids or [source_id]),
        "parent_id": parent_id,
        "parent_ids": list(
            parent_ids
            or ([parent_id] if parent_id else [])
        ),
        "content_hash": content_hash,
        "chunk_ids": list(chunk_ids or []),
    }


__all__ = [
    "ProgressCallback",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorSearchHit",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreUnavailableError",
    "make_collection_name",
    "make_point_id",
    "normalise_vector",
    "qdrant_point_payload",
]
