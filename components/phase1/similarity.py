"""Phase-1 similarity helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    has_token_overlap,
    str_sim,
    text_sim,
)

__all__ = ["has_token_overlap", "str_sim", "text_sim"]

