"""Runtime configuration for local and remote CS-RAG deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, base: Path = REPO_ROOT) -> Path:
    value = Path(raw).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str = "CS-RAG API"
    app_version: str = "0.1.0"
    deployment_mode: str = "local"
    host: str = "127.0.0.1"
    port: int = 8000
    data_root: Path = field(default_factory=lambda: REPO_ROOT / "data")
    reference_root: Path = field(default_factory=lambda: REPO_ROOT.parent)
    models_root: Path = field(default_factory=lambda: REPO_ROOT / "models")
    reference_commit: str = ""
    model_base_url: str = ""
    model_api_key: str = ""
    model_timeout_seconds: float = 90.0
    model_name: str = "gpt-4o-mini"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    vector_store_backend: str = "local_file"
    vector_store_root: Path = field(default_factory=lambda: REPO_ROOT / "data" / "vector_store")
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "csrag"
    qdrant_on_disk: bool = True
    text_index_batch_size: int = 32
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 80
    chunk_parent_max_tokens: int = 1536
    chunk_include_heading: bool = True
    chunk_semantic_breakpoint_percentile: float = 85.0
    chunk_tokenizer_model: str = ""
    chunk_embedding_device: str = "cpu"
    max_source_size_bytes: int = 50 * 1024 * 1024
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.getenv("CS_RAG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        return cls(
            app_name=os.getenv("CS_RAG_APP_NAME", "CS-RAG API"),
            app_version=os.getenv("CS_RAG_APP_VERSION", "0.1.0"),
            deployment_mode=os.getenv("CS_RAG_DEPLOYMENT_MODE", "local"),
            host=os.getenv("CS_RAG_HOST", "127.0.0.1"),
            port=int(os.getenv("CS_RAG_PORT", "8000")),
            data_root=_resolve_path(os.getenv("CS_RAG_DATA_ROOT", "data")),
            reference_root=_resolve_path(os.getenv("CS_RAG_REFERENCE_ROOT", "..")),
            models_root=_resolve_path(os.getenv("CS_RAG_MODELS_ROOT", "models")),
            reference_commit=os.getenv(
                "CS_RAG_REFERENCE_COMMIT",
                "",
            ),
            model_base_url=os.getenv("CS_RAG_MODEL_BASE_URL", ""),
            model_api_key=os.getenv("CS_RAG_MODEL_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            model_timeout_seconds=float(os.getenv("CS_RAG_MODEL_TIMEOUT_SECONDS", "90")),
            model_name=os.getenv("CS_RAG_MODEL_NAME", "gpt-4o-mini"),
            embedding_base_url=os.getenv("CS_RAG_EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.getenv("CS_RAG_EMBEDDING_API_KEY", ""),
            embedding_model=os.getenv("CS_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            reranker_model=os.getenv("CS_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            vector_store_backend=os.getenv("CS_RAG_VECTOR_STORE_BACKEND", "local_file"),
            vector_store_root=_resolve_path(
                os.getenv("CS_RAG_VECTOR_STORE_ROOT", "data/vector_store")
            ),
            qdrant_url=os.getenv("CS_RAG_QDRANT_URL", ""),
            qdrant_api_key=os.getenv("CS_RAG_QDRANT_API_KEY", ""),
            qdrant_collection_prefix=os.getenv("CS_RAG_QDRANT_COLLECTION_PREFIX", "csrag"),
            qdrant_on_disk=os.getenv("CS_RAG_QDRANT_ON_DISK", "true").lower()
            in {"1", "true", "yes", "on"},
            text_index_batch_size=int(os.getenv("CS_RAG_TEXT_INDEX_BATCH_SIZE", "32")),
            chunk_max_tokens=int(os.getenv("CS_RAG_CHUNK_MAX_TOKENS", "512")),
            chunk_min_tokens=int(os.getenv("CS_RAG_CHUNK_MIN_TOKENS", "80")),
            chunk_parent_max_tokens=int(os.getenv("CS_RAG_CHUNK_PARENT_MAX_TOKENS", "1536")),
            chunk_include_heading=os.getenv("CS_RAG_CHUNK_INCLUDE_HEADING", "true").lower()
            in {"1", "true", "yes", "on"},
            chunk_semantic_breakpoint_percentile=float(
                os.getenv("CS_RAG_CHUNK_SEMANTIC_BREAKPOINT_PERCENTILE", "85")
            ),
            chunk_tokenizer_model=os.getenv("CS_RAG_CHUNK_TOKENIZER_MODEL", ""),
            chunk_embedding_device=os.getenv("CS_RAG_CHUNK_EMBEDDING_DEVICE", "cpu"),
            max_source_size_bytes=int(os.getenv("CS_RAG_MAX_SOURCE_SIZE_BYTES", str(50 * 1024 * 1024))),
            cors_origins=tuple(x.strip() for x in origins.split(",") if x.strip()),
        )

    @property
    def metadata_db(self) -> Path:
        return self.data_root / "metadata" / "csrag.sqlite3"

    @property
    def projects_root(self) -> Path:
        return self.data_root / "projects"

    @property
    def model_config_path(self) -> Path:
        return self.data_root / "config" / "model-config.json"

    @property
    def build_config_path(self) -> Path:
        return self.data_root / "config" / "build-config.json"

    def ensure_runtime_dirs(self) -> None:
        self.metadata_db.parent.mkdir(parents=True, exist_ok=True)
        self.projects_root.mkdir(parents=True, exist_ok=True)
