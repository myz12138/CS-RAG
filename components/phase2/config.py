"""Phase-2 config surface (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    DATASET,
    DATA_JSON,
    EMB_DEVICE,
    EMB_MODEL,
    FOCUS_MAX_PER_VAR,
    KG_PATH,
    MAX_SAMPLES,
    OUTPUT_JSON,
    PHASE1_JSON,
    RERANK_BATCH,
    RERANK_FP16,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    TOPK_UNRESOLVED,
    unres_top_similiar,
)

__all__ = [
    "DATASET",
    "DATA_JSON",
    "EMB_DEVICE",
    "EMB_MODEL",
    "FOCUS_MAX_PER_VAR",
    "KG_PATH",
    "MAX_SAMPLES",
    "OUTPUT_JSON",
    "PHASE1_JSON",
    "RERANK_BATCH",
    "RERANK_FP16",
    "RERANK_MAX_LENGTH",
    "RERANK_MODEL",
    "TOPK_UNRESOLVED",
    "unres_top_similiar",
]

