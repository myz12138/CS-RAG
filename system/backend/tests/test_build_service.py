import hashlib

import pytest

from backend.app.application.build_service import BuildError, BuildService
from backend.app.model_gateway.base import ModelGateway, ModelNotConfiguredError, ModelSettings
from backend.app.storage.file_store import ProjectFileStore
from backend.app.storage.sqlite_store import SQLiteStore


def _add_source(store, files, project_id, name, kind, content):
    source_format = name.rsplit(".", 1)[1]
    checksum = hashlib.sha256(content).hexdigest()
    source, duplicate = store.create_source(
        project_id=project_id,
        original_name=name,
        stored_path=f"sources/pending.{source_format}",
        source_kind=kind,
        source_format=source_format,
        mime_type="text/plain",
        checksum_sha256=checksum,
        size_bytes=len(content),
    )
    assert duplicate is False
    stored_path = files.save_source(project_id, source.source_id, source_format, content)
    store.update_source_path(source.source_id, stored_path)
    return store.get_source(source.source_id)


def test_build_rejects_kg_source_for_new_version(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    files = ProjectFileStore(tmp_path / "projects")
    gateway = ModelGateway(ModelSettings("", "", "", ""))
    service = BuildService(store, files, gateway)
    project = store.create_project("构建测试")

    _add_source(store, files, project.project_id, "说明.md", "text", "实体 A 的说明。".encode("utf-8"))
    _add_source(
        store,
        files,
        project.project_id,
        "关系.jsonl",
        "kg",
        (
            '{"head":"A","relation":"R","tail":"B"}\n'
            '{"head":"A","relation":"R","tail":"B"}\n'
            '{"head":"A","relation":"R","tail":"C"}\n'
            '{"head":"D","relation":"R","tail":"C"}\n'
        ).encode("utf-8"),
    )

    with pytest.raises(BuildError, match="KG 文件"):
        service.build(project.project_id)

    assert store.list_versions(project.project_id) == []


def test_text_entry_persists_parent_and_kg_links(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project = store.create_project("文本索引字段")
    source = _add_source(
        store,
        ProjectFileStore(tmp_path / "projects"),
        project.project_id,
        "说明.md",
        "text",
        "正文".encode("utf-8"),
    )
    version = store.create_version(project.project_id, "import_only", [source.source_id])
    entry = store.create_text_entry(
        version.version_id,
        source.source_id,
        0,
        "说明.md",
        "正文",
        hashlib.sha256("正文".encode("utf-8")).hexdigest(),
        {"title_path": [], "paragraph_index": 0},
        parent_id="parent-1",
    )
    triple, duplicate, _ = store.insert_kg_triple(
        version.version_id,
        source.source_id,
        entry.entry_id,
        None,
        "甲",
        "关联",
        "乙",
        "text_extracted",
        "正文",
    )
    assert duplicate is False
    updated = store.update_text_entry_kg_triple_ids(entry.entry_id, [triple.triple_id, triple.triple_id])

    assert updated.parent_id == "parent-1"
    assert updated.kg_triple_ids == (triple.triple_id,)
    assert store.get_text_entry(entry.entry_id).to_dict()["kg_triple_ids"] == [triple.triple_id]


def test_text_only_build_requires_real_model_configuration(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    files = ProjectFileStore(tmp_path / "projects")
    service = BuildService(store, files, ModelGateway(ModelSettings("", "model", "", "")))
    project = store.create_project("需要模型")
    _add_source(store, files, project.project_id, "说明.md", "text", "A 认识 B。".encode("utf-8"))

    with pytest.raises(BuildError, match="LLM"):
        service.build(project.project_id)

    assert store.list_versions(project.project_id) == []


def test_model_gateway_does_not_fallback_when_endpoint_is_missing():
    gateway = ModelGateway(ModelSettings("", "model", "", ""))

    with pytest.raises(ModelNotConfiguredError):
        gateway.extract_triples("A 认识 B。", "说明.md")
