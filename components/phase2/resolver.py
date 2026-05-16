"""Phase-2 resolved/unresolved helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    dedup_keep_order,
    get_candidate_names,
    get_known_entity_candidate_names,
    get_var_candidate_names,
    is_resolved_triple,
    is_var,
)

__all__ = [
    "dedup_keep_order",
    "get_candidate_names",
    "get_known_entity_candidate_names",
    "get_var_candidate_names",
    "is_resolved_triple",
    "is_var",
]

