import pytest

from backend.app.ingestion.policy import (
    UnsupportedSourceFormatError,
    classify_source,
)
from backend.app.ingestion.parsers import parse_kg, parse_text
from backend.app.storage.sqlite_store import SQLiteStore


def test_classify_text_and_kg_without_parsing():
    assert classify_source("知识说明.md").source_kind == "text"
    assert classify_source("项目材料.docx").source_kind == "text"
    with pytest.raises(UnsupportedSourceFormatError, match="KG 文件"):
        classify_source("relations.json")
    with pytest.raises(UnsupportedSourceFormatError, match="只允许上传文本"):
        classify_source("relations.json", "kg")


def test_classification_rejects_mismatched_explicit_kind():
    try:
        classify_source("relations.json", "text")
    except UnsupportedSourceFormatError as exc:
        assert "文本格式" in str(exc)
    else:
        raise AssertionError("expected an explicit kind mismatch to be rejected")


def test_source_checksum_is_deduplicated(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project = store.create_project("输入测试")

    first, duplicate = store.create_source(
        project_id=project.project_id,
        original_name="a.md",
        stored_path="sources/a.md",
        source_kind="text",
        source_format="md",
        mime_type="text/markdown",
        checksum_sha256="abc123",
        size_bytes=3,
    )
    second, duplicate_again = store.create_source(
        project_id=project.project_id,
        original_name="renamed.md",
        stored_path="sources/renamed.md",
        source_kind="text",
        source_format="md",
        mime_type="text/markdown",
        checksum_sha256="abc123",
        size_bytes=3,
    )

    assert duplicate is False
    assert duplicate_again is True
    assert second.source_id == first.source_id
    assert len(store.list_sources(project.project_id)) == 1


def test_text_and_kg_parsers_keep_source_shape():
    entries = parse_text("第一段。\n\n第二段。".encode("utf-8"), "说明.md")
    parsed_kg = parse_kg(
        '{"head":"A","relation":"关联","tail":"B","evidence":"原文"}\n'.encode("utf-8"),
        "jsonl",
    )

    assert [entry.ordinal for entry in entries] == [0, 1]
    assert parsed_kg.issues == ()
    assert parsed_kg.triples[0].head == "A"
    assert parsed_kg.triples[0].evidence == "原文"
