"""Application state assembled once per backend process."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from backend.app.application.build_service import BuildService
from backend.app.application.build_settings import BuildSettings
from backend.app.application.query_service import QueryService
from backend.app.application.query_settings import QuerySettings
from backend.app.algorithm_adapter.reference import ReferenceAlgorithmAdapter
from backend.app.algorithm_adapter.reference_runner import ReferenceQueryRunner
from backend.app.config import Settings
from backend.app.jobs.manager import TaskManager
from backend.app.jobs.persistence import SQLiteTaskPersistence
from backend.app.model_gateway.base import ModelGateway, ModelSettings
from backend.app.storage.file_store import ProjectFileStore
from backend.app.storage.sqlite_store import SQLiteStore


class AppState:
    def __init__(self, settings: Settings) -> None:
        settings.ensure_runtime_dirs()
        self.settings = settings
        self.store = SQLiteStore(settings.metadata_db)
        self.files = ProjectFileStore(settings.projects_root)
        self.tasks = TaskManager(SQLiteTaskPersistence(self.store))
        self.model_gateway = ModelGateway(self._load_model_settings(settings))
        self.build_settings = self._load_build_settings(settings)
        self.query_settings = self._load_query_settings(settings)
        self.algorithm = ReferenceAlgorithmAdapter(
            settings.reference_root,
            settings.reference_commit,
        )
        self.algorithm_runner = ReferenceQueryRunner(
            settings.reference_root,
            self.store,
            self.files,
            self.model_gateway,
            models_root=settings.models_root,
            reference_commit=settings.reference_commit,
            timeout_seconds=max(600.0, settings.model_timeout_seconds * 20),
            vector_store_backend=settings.vector_store_backend,
            vector_store_root=settings.vector_store_root,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            qdrant_on_disk=settings.qdrant_on_disk,
            query_settings=self.query_settings,
        )
        self.algorithm.attach_runner(self.algorithm_runner)
        self.build_service = BuildService(
            self.store,
            self.files,
            self.model_gateway,
            models_root=settings.models_root,
            vector_store_backend=settings.vector_store_backend,
            vector_store_root=settings.vector_store_root,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            qdrant_collection_prefix=settings.qdrant_collection_prefix,
            qdrant_on_disk=settings.qdrant_on_disk,
            build_settings=self.build_settings,
        )
        self.query_service = QueryService(self.store, self.algorithm)

    @staticmethod
    def _load_model_settings(settings: Settings) -> ModelSettings:
        current = ModelSettings(
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
            timeout_seconds=settings.model_timeout_seconds,
            model_name=settings.model_name,
            embedding_base_url=settings.embedding_base_url,
            embedding_api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )
        path = settings.model_config_path
        if not path.is_file():
            return current
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return current
        if not isinstance(stored, dict):
            return current
        values: dict[str, Any] = {}
        for key in (
            "base_url",
            "api_key",
            "model_name",
            "embedding_base_url",
            "embedding_api_key",
            "embedding_model",
            "reranker_model",
        ):
            value = stored.get(key)
            if isinstance(value, str):
                values[key] = value
        timeout_value = stored.get("timeout_seconds")
        if isinstance(timeout_value, (int, float)) and not isinstance(timeout_value, bool):
            values["timeout_seconds"] = float(timeout_value)
        return replace(current, **values)

    @staticmethod
    def _load_build_settings(settings: Settings) -> BuildSettings:
        current = BuildSettings(
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
            parent_max_tokens=settings.chunk_parent_max_tokens,
            include_heading=settings.chunk_include_heading,
            semantic_breakpoint_percentile=settings.chunk_semantic_breakpoint_percentile,
            tokenizer_model=settings.chunk_tokenizer_model,
            embedding_device=settings.chunk_embedding_device,
            text_index_batch_size=settings.text_index_batch_size,
        ).validate()
        path = settings.build_config_path
        if not path.is_file():
            return current
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return current
        if not isinstance(stored, dict):
            return current
        try:
            return BuildSettings.from_mapping(stored, current)
        except ValueError:
            return current

    @staticmethod
    def _load_query_settings(settings: Settings) -> QuerySettings:
        current = QuerySettings().validate()
        path = settings.data_root / "config" / "query-config.json"
        if not path.is_file():
            return current
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return current
        if not isinstance(stored, dict):
            return current
        try:
            return QuerySettings.from_mapping(stored, current)
        except (TypeError, ValueError):
            return current

    def model_config(self) -> dict[str, object]:
        return self.model_gateway.config_summary()

    def build_config(self) -> dict[str, object]:
        config = self.build_settings.to_dict()
        config["embedding_model"] = self.model_gateway.settings.embedding_model
        return config

    def query_config(self) -> dict[str, object]:
        return self.query_settings.to_dict()

    def update_build_config(self, values: dict[str, object]) -> dict[str, object]:
        updated = BuildSettings.from_mapping(values, self.build_settings)
        path = self.settings.build_config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(updated.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.build_settings = updated
        self.build_service.configure_build_settings(updated)
        return self.build_config()

    def update_query_config(self, values: dict[str, object]) -> dict[str, object]:
        updated = QuerySettings.from_mapping(values, self.query_settings)
        path = self.settings.data_root / "config" / "query-config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(updated.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.query_settings = updated
        self.algorithm_runner.configure_query_settings(updated)
        return self.query_config()

    def update_model_config(self, values: dict[str, object]) -> dict[str, object]:
        current = self.model_gateway.settings
        timeout_value = values.get("timeout_seconds")
        timeout_seconds = (
            current.timeout_seconds
            if timeout_value is None
            else float(timeout_value)
        )
        if timeout_seconds <= 0:
            raise ValueError("模型请求超时必须大于 0 秒")
        updated = replace(
            current,
            base_url=str(values.get("base_url") or "").strip(),
            model_name=str(values.get("model_name") or current.model_name).strip(),
            embedding_base_url=str(values.get("embedding_base_url") or "").strip(),
            embedding_model=str(values.get("embedding_model") or current.embedding_model).strip(),
            reranker_model=str(values.get("reranker_model") or current.reranker_model).strip(),
            timeout_seconds=timeout_seconds,
            api_key=current.api_key if values.get("api_key") is None else str(values["api_key"]).strip(),
            embedding_api_key=(
                current.embedding_api_key
                if values.get("embedding_api_key") is None
                else str(values["embedding_api_key"]).strip()
            ),
        )
        if not updated.model_name:
            raise ValueError("模型名称不能为空")
        self.model_gateway.settings = updated
        path = self.settings.model_config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "base_url": updated.base_url,
                    "api_key": updated.api_key,
                    "model_name": updated.model_name,
                    "timeout_seconds": updated.timeout_seconds,
                    "embedding_base_url": updated.embedding_base_url,
                    "embedding_api_key": updated.embedding_api_key,
                    "embedding_model": updated.embedding_model,
                    "reranker_model": updated.reranker_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.model_config()
