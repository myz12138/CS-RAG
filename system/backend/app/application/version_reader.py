"""Read-only presentation queries over an immutable knowledge-base version."""
from __future__ import annotations
from typing import Any


def attach_graph_entities(records: list[dict[str, Any]], entities: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Add display-only node identities while preserving original triple fields."""
    return [
        {
            **record,
            **{
                f"{side}_entity": entities[str(record[side])]
                for side in ("head", "tail")
                if str(record.get(side, "")) in entities
            },
        }
        for record in records
    ]


def browse_version_items(items: list[Any], source_snapshot: tuple[str, ...] | list[str], *, kind: str, offset: int = 0, limit: int = 30, search: str = "") -> dict[str, Any]:
    source_order = {source_id: index for index, source_id in enumerate(source_snapshot)}
    ordered = sorted(items, key=lambda item: (
        source_order.get(item.source_id, len(source_order)),
        getattr(item, "ordinal", None) if getattr(item, "ordinal", None) is not None else (getattr(item, "source_line", None) or 0),
    ))
    records = [item.to_dict() for item in ordered]
    query = search.strip().casefold()
    keys = ("content", "title", "source_location") if kind == "text" else ("head", "relation", "tail", "evidence")
    matched = [record for record in records if not query or query in " ".join(str(record.get(key, "")) for key in keys).casefold()]
    return {"total": len(matched), "version_total": len(records), "offset": offset, "limit": limit, "items": matched[offset:offset + limit]}
