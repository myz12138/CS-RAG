"""Phase-1 reranker helpers (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    _rerank_score_pairs,
    build_reranker,
    rerank_select_topn,
)

__all__ = ["_rerank_score_pairs", "build_reranker", "rerank_select_topn"]

