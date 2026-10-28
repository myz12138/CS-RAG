from dataclasses import dataclass
from backend.app.application.version_reader import browse_version_items
from backend.app.application.version_reader import attach_graph_entities
from backend.app.storage.sqlite_store import SQLiteStore


@dataclass
class Item:
    source_id: str
    ordinal: int
    content: str

    def to_dict(self):
        return {"source_id": self.source_id, "ordinal": self.ordinal, "content": self.content, "kg_triple_ids": ["kept"]}


def test_pagination_preserves_source_order_and_fields_without_mutating_inputs():
    first = Item("old", 0, "旧来源")
    second = Item("new", 1, "第二条")
    third = Item("new", 0, "第一条")
    original = [first, second, third]
    page = browse_version_items(original, ["new", "old"], kind="text", offset=1, limit=1)
    assert page["total"] == 3
    assert page["items"] == [second.to_dict()]
    assert original == [first, second, third]
    assert page["items"][0]["kg_triple_ids"] == ["kept"]


def test_search_applies_to_all_records_before_pagination():
    items = [Item("s", i, "needle" if i in (32, 45) else "other") for i in range(60)]
    page = browse_version_items(items, ["s"], kind="text", search="NEEDLE", offset=1, limit=1)
    assert page["total"] == 2
    assert page["version_total"] == 60
    assert page["items"][0]["ordinal"] == 45


def test_empty_or_past_end_page_has_no_fabricated_records():
    assert browse_version_items([], [], kind="kg")["items"] == []
    page = browse_version_items([Item("s", 0, "x")], ["s"], kind="text", offset=100)
    assert page["total"] == 1
    assert page["items"] == []


def test_graph_projection_uses_existing_entity_identity_without_rewriting_facts(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project = store.create_project("graph")
    source, _ = store.create_source(project.project_id, "a.txt", "sources/a.txt", "text", "txt", "text/plain", "hash", 1)
    version = store.create_version(project.project_id, "supplement_extract", [source.source_id])
    first, _, _ = store.insert_kg_triple(version.version_id, source.source_id, None, None, "Alpha", "rel", "乙", "text_extracted", "a")
    second, _, _ = store.insert_kg_triple(version.version_id, source.source_id, None, None, "ALPHA", "other", "丙", "text_extracted", "b")
    records = [first.to_dict(), second.to_dict()]
    before = [dict(record) for record in records]
    projected = attach_graph_entities(records, store.resolve_graph_entities(version.version_id, ["Alpha", "ALPHA", "乙", "丙"]))
    assert projected[0]["head_entity"]["id"] == projected[1]["head_entity"]["id"]
    assert projected[1]["head"] == "ALPHA"
    assert records == before
    assert store.version_counts(version.version_id)["kg_triples"] == 2
    other = store.create_version(project.project_id, "supplement_extract", [])
    assert store.resolve_graph_entities(other.version_id, ["Alpha"]) == {}
