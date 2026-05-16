"""Phase-2 IO helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    _load_json,
    iter_context_units_2wiki_like,
    iter_paragraphs_musique,
    load_id2ex,
    load_kg,
    load_list_json,
)

__all__ = [
    "_load_json",
    "iter_context_units_2wiki_like",
    "iter_paragraphs_musique",
    "load_id2ex",
    "load_kg",
    "load_list_json",
]

