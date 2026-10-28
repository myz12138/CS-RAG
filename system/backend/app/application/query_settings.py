"""User-adjustable settings for the CS-RAG query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class QuerySettings:
    """Parameters that control retrieval, fallback, and answer context size.

    Defaults mirror the values used by the bundled CS-RAG reference scripts.
    The settings are applied to later queries and are also written into each
    query result as an execution snapshot for reproducibility.
    """

    planner_max_retries: int = 3
    top_n_llm: int = 3
    top_k_entity: int = 3
    relation_k: int = 8
    neff_threshold: float = 1.5
    phase1_rerank_max_length: int = 128
    phase2_rerank_max_length: int = 256
    rerank_batch: int = 32
    rerank_fp16: bool = False
    include_question_in_rerank: bool = True
    unresolved_top_similar: int = 20
    top_k_unresolved: int = 3
    focus_max_per_variable: int = 5
    text_index_recall_multiplier: int = 4
    prompt_max_triples: int = 20
    prompt_max_text_per_triple: int = 8

    def validate(self) -> "QuerySettings":
        normalized = replace(
            self,
            planner_max_retries=int(self.planner_max_retries),
            top_n_llm=int(self.top_n_llm),
            top_k_entity=int(self.top_k_entity),
            relation_k=int(self.relation_k),
            neff_threshold=float(self.neff_threshold),
            phase1_rerank_max_length=int(self.phase1_rerank_max_length),
            phase2_rerank_max_length=int(self.phase2_rerank_max_length),
            rerank_batch=int(self.rerank_batch),
            unresolved_top_similar=int(self.unresolved_top_similar),
            top_k_unresolved=int(self.top_k_unresolved),
            focus_max_per_variable=int(self.focus_max_per_variable),
            text_index_recall_multiplier=int(self.text_index_recall_multiplier),
            prompt_max_triples=int(self.prompt_max_triples),
            prompt_max_text_per_triple=int(self.prompt_max_text_per_triple),
        )
        positive_limits = {
            "查询规划最大重试次数": normalized.planner_max_retries,
            "每条约束保留三元组数": normalized.top_n_llm,
            "实体候选数": normalized.top_k_entity,
            "关系粗排候选数": normalized.relation_k,
            "Phase1 重排最大长度": normalized.phase1_rerank_max_length,
            "Phase2 重排最大长度": normalized.phase2_rerank_max_length,
            "重排批量大小": normalized.rerank_batch,
            "未解析文本粗召回数": normalized.unresolved_top_similar,
            "未解析约束文本保留数": normalized.top_k_unresolved,
            "变量聚焦实体数": normalized.focus_max_per_variable,
            "文本索引召回倍数": normalized.text_index_recall_multiplier,
            "答案最大约束数": normalized.prompt_max_triples,
            "每条约束最大证据数": normalized.prompt_max_text_per_triple,
        }
        for label, value in positive_limits.items():
            if value <= 0:
                raise ValueError(f"{label}必须大于 0")
        if normalized.neff_threshold <= 0:
            raise ValueError("Neff 门控阈值必须大于 0")
        if normalized.top_n_llm > 100:
            raise ValueError("每条约束保留三元组数不能超过 100")
        if normalized.relation_k > 1000:
            raise ValueError("关系粗排候选数不能超过 1000")
        return normalized

    def to_dict(self) -> dict[str, object]:
        return {
            "planner_max_retries": self.planner_max_retries,
            "top_n_llm": self.top_n_llm,
            "top_k_entity": self.top_k_entity,
            "relation_k": self.relation_k,
            "neff_threshold": self.neff_threshold,
            "phase1_rerank_max_length": self.phase1_rerank_max_length,
            "phase2_rerank_max_length": self.phase2_rerank_max_length,
            "rerank_batch": self.rerank_batch,
            "rerank_fp16": self.rerank_fp16,
            "include_question_in_rerank": self.include_question_in_rerank,
            "unresolved_top_similar": self.unresolved_top_similar,
            "top_k_unresolved": self.top_k_unresolved,
            "focus_max_per_variable": self.focus_max_per_variable,
            "text_index_recall_multiplier": self.text_index_recall_multiplier,
            "prompt_max_triples": self.prompt_max_triples,
            "prompt_max_text_per_triple": self.prompt_max_text_per_triple,
        }

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        defaults: "QuerySettings" | None = None,
    ) -> "QuerySettings":
        base = defaults or cls()
        result = replace(
            base,
            planner_max_retries=_as_int(raw, "planner_max_retries", base.planner_max_retries),
            top_n_llm=_as_int(raw, "top_n_llm", base.top_n_llm),
            top_k_entity=_as_int(raw, "top_k_entity", base.top_k_entity),
            relation_k=_as_int(raw, "relation_k", base.relation_k),
            neff_threshold=_as_float(raw, "neff_threshold", base.neff_threshold),
            phase1_rerank_max_length=_as_int(
                raw,
                "phase1_rerank_max_length",
                base.phase1_rerank_max_length,
            ),
            phase2_rerank_max_length=_as_int(
                raw,
                "phase2_rerank_max_length",
                base.phase2_rerank_max_length,
            ),
            rerank_batch=_as_int(raw, "rerank_batch", base.rerank_batch),
            rerank_fp16=_as_bool(raw, "rerank_fp16", base.rerank_fp16),
            include_question_in_rerank=_as_bool(
                raw,
                "include_question_in_rerank",
                base.include_question_in_rerank,
            ),
            unresolved_top_similar=_as_int(
                raw,
                "unresolved_top_similar",
                base.unresolved_top_similar,
            ),
            top_k_unresolved=_as_int(raw, "top_k_unresolved", base.top_k_unresolved),
            focus_max_per_variable=_as_int(
                raw,
                "focus_max_per_variable",
                base.focus_max_per_variable,
            ),
            text_index_recall_multiplier=_as_int(
                raw,
                "text_index_recall_multiplier",
                base.text_index_recall_multiplier,
            ),
            prompt_max_triples=_as_int(
                raw,
                "prompt_max_triples",
                base.prompt_max_triples,
            ),
            prompt_max_text_per_triple=_as_int(
                raw,
                "prompt_max_text_per_triple",
                base.prompt_max_text_per_triple,
            ),
        )
        return result.validate()


def _as_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} 必须是整数")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是整数") from exc


def _as_float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} 必须是数字")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字") from exc


def _as_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} 必须是布尔值")
