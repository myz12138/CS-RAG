"""Phase-1 relation scoring helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    _clean_relation_variants,
    relation_score,
    top_by_relation,
)

__all__ = ["_clean_relation_variants", "relation_score", "top_by_relation"]

