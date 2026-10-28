"""Convert versioned project records to the reference KG JSONL contract."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.storage.sqlite_store import SQLiteStore


class ReferenceKGExporter:
    """Export without changing the stored domain records."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def export(self, version_id: str, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        triples = self.store.list_kg_triples(version_id)
        entries = {entry.entry_id: entry for entry in self.store.list_text_entries(version_id)}
        sources = {}
        for triple in triples:
            if triple.source_id not in sources:
                sources[triple.source_id] = self.store.get_source(triple.source_id)

        entity_ids: dict[str, int] = {}
        entities: list[dict[str, object]] = []
        exported_triples: list[dict[str, object]] = []
        title2entity_ids: dict[str, set[int]] = {}
        title2triple_ids: dict[str, list[int]] = {}

        def entity_id(name: str) -> int:
            key = " ".join(name.strip().casefold().split())
            if key not in entity_ids:
                entity_ids[key] = len(entities)
                entities.append({"id": entity_ids[key], "name": name, "type": "UNKNOWN"})
            return entity_ids[key]

        for triple_index, triple in enumerate(triples):
            title = entries[triple.source_entry_id].title if triple.source_entry_id in entries else sources[triple.source_id].original_name
            head_id = entity_id(triple.head)
            tail_id = entity_id(triple.tail)
            title2entity_ids.setdefault(title, set()).update({head_id, tail_id})
            title2triple_ids.setdefault(title, []).append(triple_index)
            exported_triples.append(
                {
                    "h": head_id,
                    "t": tail_id,
                    "r": triple.relation,
                    "evidence": triple.evidence or "",
                    "title": title,
                }
            )

        self._write_jsonl(output_dir / "entities.jsonl", entities)
        self._write_jsonl(output_dir / "triples.jsonl", exported_triples)
        self._write_jsonl(
            output_dir / "title2entities.jsonl",
            [
                {"title": title, "entity_ids": sorted(ids)}
                for title, ids in title2entity_ids.items()
            ],
        )
        self._write_jsonl(
            output_dir / "title2triples.jsonl",
            [
                {"title": title, "triple_idxs": indexes}
                for title, indexes in title2triple_ids.items()
            ],
        )
        summary = {
            "output_dir": str(output_dir),
            "entities": len(entities),
            "triples": len(exported_triples),
            "titles": len(title2triple_ids),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
