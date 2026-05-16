"""Phase-1 IO helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    _load_json,
    load_raw_dataset_id2qa,
    normalize_query_items,
)

__all__ = ["_load_json", "load_raw_dataset_id2qa", "normalize_query_items"]

