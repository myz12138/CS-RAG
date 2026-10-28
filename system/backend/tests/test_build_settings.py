from pathlib import Path

import pytest

from backend.app.application.build_service import BuildService
from backend.app.application.build_settings import BuildSettings
from backend.app.model_gateway.base import ModelGateway, ModelSettings
from backend.app.storage.file_store import ProjectFileStore
from backend.app.storage.sqlite_store import SQLiteStore


def test_build_settings_exposes_the_active_build_contract() -> None:
    settings = BuildSettings(
        max_tokens=768,
        min_tokens=96,
        parent_max_tokens=2048,
        include_heading=False,
        semantic_breakpoint_percentile=72.5,
        tokenizer_model="  tokenizer-local  ",
        embedding_device=" CUDA:0 ",
        text_index_batch_size=16,
    ).validate()

    config = settings.to_chunking_config("embedding-local")

    assert config.strategy == "unified"
    assert config.max_tokens == 768
    assert config.min_tokens == 96
    assert config.parent_max_tokens == 2048
    assert config.include_heading is False
    assert config.semantic_breakpoint_percentile == 72.5
    assert config.tokenizer_model == "tokenizer-local"
    assert config.embedding_model == "embedding-local"
    assert config.embedding_device == "cuda:0"
    assert config.tokenizer_local_files_only is True
    assert config.embedding_local_files_only is True
    assert settings.to_dict()["overlap_tokens"] == 0


def test_build_settings_rejects_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="最小 Token"):
        BuildSettings(max_tokens=100, min_tokens=101).validate()

    with pytest.raises(ValueError, match="运行设备"):
        BuildSettings(embedding_device="tpu").validate()

    with pytest.raises(ValueError, match="批量写入"):
        BuildSettings(text_index_batch_size=0).validate()


def test_build_service_chunking_plan_reads_configurable_settings(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    files = ProjectFileStore(tmp_path / "projects")
    gateway = ModelGateway(ModelSettings("", "", "BAAI/bge-small-zh-v1.5", ""))
    service = BuildService(
        store,
        files,
        gateway,
        models_root=tmp_path / "models",
        build_settings=BuildSettings(
            max_tokens=768,
            min_tokens=96,
            parent_max_tokens=2048,
            semantic_breakpoint_percentile=72.5,
            embedding_device="cpu",
            text_index_batch_size=16,
        ),
    )

    plan = service.chunking_plan()

    assert plan["strategy"] == "unified"
    assert plan["token_limit"] == 768
    assert plan["min_tokens"] == 96
    assert plan["parent_token_limit"] == 2048
    assert plan["semantic_breakpoint_percentile"] == 72.5
    assert plan["embedding_device"] == "cpu"
    assert plan["index_batch_size"] == 16
    assert plan["overlap_tokens"] == 0
