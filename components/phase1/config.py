"""Phase-1 config surface (delegated from legacy implementation)."""

from .legacy_impl import (  # noqa: F401
    DATASET,
    EMB_DEVICE,
    EMB_MODEL,
    INCLUDE_QUESTION_IN_RERANK,
    KG_PATH,
    MAX_SAMPLES,
    NEFF_THRESHOLD,
    OUTPUT_JSON,
    QUERY_JSON,
    RAW_DATA_JSON,
    RELATION_K,
    RERANK_BATCH,
    RERANK_FP16,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    TOP_K_ENTITY,
    TOP_N_LLM,
)

__all__ = [
    "DATASET",
    "EMB_DEVICE",
    "EMB_MODEL",
    "INCLUDE_QUESTION_IN_RERANK",
    "KG_PATH",
    "MAX_SAMPLES",
    "NEFF_THRESHOLD",
    "OUTPUT_JSON",
    "QUERY_JSON",
    "RAW_DATA_JSON",
    "RELATION_K",
    "RERANK_BATCH",
    "RERANK_FP16",
    "RERANK_MAX_LENGTH",
    "RERANK_MODEL",
    "TOP_K_ENTITY",
    "TOP_N_LLM",
]

