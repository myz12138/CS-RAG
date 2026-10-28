from __future__ import annotations

import json
from dataclasses import replace

from backend.app.domain.models import TextEntry
from backend.app.algorithm_adapter.contracts import QueryRequest
from backend.app.algorithm_adapter.reference_runner import ReferenceQueryRunner
from backend.app.indexing.text_index import TextIndexBuilder, TextVectorIndex
from backend.app.indexing.text_recall import TextIndexRecall
from backend.app.indexing.vector_store import VectorSearchHit


def _entry(
    entry_id: str,
    content: str,
    *,
    content_hash: str | None = None,
    parent_id: str | None = None,
    kg_triple_ids: tuple[str, ...] = (),
) -> TextEntry:
    return TextEntry(
        entry_id=entry_id,
        version_id="version-1",
        source_id="source-1",
        ordinal=0,
        title="资料 / 章节",
        content=content,
        content_hash=content_hash or f"hash-{entry_id}",
        source_location={
            "title_path": ["章节"],
            "paragraph_index": 2,
            "char_start": 10,
            "char_end": 20,
            "chunk_metadata": {"should_not_be_in_runtime_record": True},
        },
        parent_id=parent_id,
        kg_triple_ids=kg_triple_ids,
    )


def test_text_index_keeps_minimal_records_and_does_not_embed_parents(tmp_path) -> None:
    entries = [
        _entry("entry-1", "甲主题内容。", parent_id="parent-1", kg_triple_ids=("triple-1",)),
        _entry("entry-2", "乙主题内容。"),
    ]
    embedded_texts: list[str] = []

    def embeddings(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[1.0, 0.0] if "甲" in text else [0.0, 2.0] for text in texts]

    index_dir = tmp_path / "indexes" / "text"
    summary = TextIndexBuilder(batch_size=1).build(
        index_dir,
        entries,
        [
            {
                "parent_id": "parent-1",
                "content": "父级上下文，不应被向量化。",
                "source_location": {"title_path": ["章节"], "paragraph_index": 1},
            }
        ],
        embedding_function=embeddings,
        embedding_model="local-test-model",
    )

    assert summary["status"] == "ready"
    assert summary["vector_count"] == 2
    assert summary["parent_count"] == 1
    assert embedded_texts == ["甲主题内容。", "乙主题内容。"]

    records = [
        json.loads(line)
        for line in (index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(records[0]) == {
        "chunk_id",
        "content",
        "location",
        "parent_id",
        "content_hash",
        "kg_triple_ids",
    }
    assert records[0]["parent_id"] == "parent-1"
    assert records[0]["kg_triple_ids"] == ["triple-1"]
    assert "chunk_metadata" not in records[0]["location"]

    index = TextVectorIndex(index_dir)
    hits = index.search([1.0, 0.0], top_k=2)
    assert [hit.chunk_id for hit in hits] == ["entry-1", "entry-2"]
    assert index.get_parent("parent-1")["content"] == "父级上下文，不应被向量化。"
    assert TextIndexBuilder.is_current(
        index_dir,
        entries,
        embedding_model="local-test-model",
    )
    changed = replace(entries[0], content_hash="changed-hash")
    assert not TextIndexBuilder.is_current(
        index_dir,
        [changed, entries[1]],
        embedding_model="local-test-model",
    )


def test_text_index_deduplicates_physical_vectors_but_keeps_logical_entries(tmp_path) -> None:
    entries = [
        _entry("entry-1", "同一段内容。", content_hash="same-hash"),
        _entry("entry-2", "同一段内容。", content_hash="same-hash"),
        _entry("entry-3", "另一段内容。", content_hash="other-hash"),
    ]
    calls: list[list[str]] = []

    def embeddings(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0, 0.0] if "同一" in text else [0.0, 2.0] for text in texts]

    index_dir = tmp_path / "indexes" / "text"
    summary = TextIndexBuilder(batch_size=2).build(
        index_dir,
        entries,
        [],
        embedding_function=embeddings,
        embedding_model="local-test-model",
    )

    assert summary["logical_entry_count"] == 3
    assert summary["vector_count"] == 2
    assert summary["duplicate_entry_count"] == 1
    assert calls == [["同一段内容。", "另一段内容。"]]

    mappings = [
        json.loads(line)
        for line in (index_dir / "mapping.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert mappings[0]["chunk_id"] == "entry-1"
    assert mappings[0]["chunk_ids"] == ["entry-1", "entry-2"]
    assert len(mappings) == 2

    hits = TextVectorIndex(index_dir).search([1.0, 0.0], top_k=2)
    assert hits[0].chunk_id == "entry-1"
    assert hits[0].chunk_ids == ("entry-1", "entry-2")
    assert TextIndexBuilder.is_current(
        index_dir,
        entries,
        embedding_model="local-test-model",
    )


def test_text_index_can_use_a_database_backend_without_writing_vector_file(tmp_path) -> None:
    class FakeVectorStore:
        backend_name = "qdrant"
        location = "local://qdrant"

        def __init__(self) -> None:
            self.collection_name = None
            self.points = []
            self.batch_sizes = []
            self.dimension = None

        def begin(self, *, collection_name, dimension):
            self.collection_name = collection_name
            self.dimension = dimension

        def upsert_batch(self, *, collection_name, points, dimension):
            assert collection_name == self.collection_name
            assert dimension == self.dimension == 2
            self.points.extend(points)
            self.batch_sizes.append(len(points))
            return len(points)

        def finish(self, *, collection_name, dimension, vector_count):
            assert collection_name == self.collection_name
            assert dimension == self.dimension == 2
            assert vector_count == len(self.points)
            return {"status": "ready"}

        def build(self, *, collection_name, points, dimension, progress=None):
            self.collection_name = collection_name
            self.points = list(points)
            assert dimension == 2
            if progress is not None:
                progress(1.0, "fake backend ready")
            return {"status": "ready"}

        def search(self, *, collection_name, query_vector, top_k=5, filters=None):
            assert collection_name == self.collection_name
            return [VectorSearchHit("entry-1", 0.99, 1)][:top_k]

        def delete(self, collection_name):
            self.collection_name = None

        def health(self):
            return {"status": "ready"}

    entries = [_entry("entry-1", "数据库后端内容。")]
    backend = FakeVectorStore()
    index_dir = tmp_path / "indexes" / "text"
    summary = TextIndexBuilder().build(
        index_dir,
        entries,
        [],
        embedding_function=lambda texts: [[3.0, 4.0] for _ in texts],
        embedding_model="local-test-model",
        vector_store=backend,
        collection_name="csrag_text_test",
        project_id="project-1",
        version_id="version-1",
    )

    assert summary["backend"] == "qdrant"
    assert summary["collection_name"] == "csrag_text_test"
    assert backend.batch_sizes == [1]
    assert backend.points[0].payload["chunk_ids"] == ["entry-1"]
    assert not (index_dir / "vectors.npy").exists()
    index = TextVectorIndex(index_dir, backend)
    assert index.search([3.0, 4.0])[0].chunk_id == "entry-1"


def test_text_index_recall_bridge_reads_local_index_and_dataset_sidecar(tmp_path) -> None:
    entries = [
        _entry("entry-1", "第一句。第二句。"),
        replace(_entry("entry-2", "另一篇内容。"), title="资料 / 另一章节"),
    ]
    index_dir = tmp_path / "indexes" / "text"
    TextIndexBuilder().build(
        index_dir,
        entries,
        [],
        embedding_function=lambda texts: [
            [1.0, 0.0] if "第一" in text else [0.0, 1.0]
            for text in texts
        ],
        embedding_model="local-test-model",
    )

    reader = TextIndexRecall.from_environment(index_dir)
    assert reader.search([1.0, 0.0], top_k=1)[0].chunk_id == "entry-1"

    class Store:
        def list_text_entries(self, version_id):
            assert version_id == "version-1"
            return entries

    class Files:
        projects_root = tmp_path / "projects"

    runner = ReferenceQueryRunner(
        tmp_path / "vendor" / "CS-RAG-reference",
        Store(),
        Files(),
        None,
    )
    dataset_path = tmp_path / "query" / "dataset.json"
    dataset_path.parent.mkdir(parents=True)
    sidecar_path = runner._write_dataset(
        dataset_path,
        "version-1",
        QueryRequest(
            question="问题",
            project_id="project-1",
            version_id="version-1",
            query_id="query-1",
        ),
    )
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload[0]["context"] == [
        ["资料 / 章节", ["第一句。第二句。"]],
        ["资料 / 另一章节", ["另一篇内容。"]],
    ]
    assert sidecar["version_id"] == "version-1"
    assert [unit["entry_id"] for unit in sidecar["units"]] == ["entry-1", "entry-2"]
