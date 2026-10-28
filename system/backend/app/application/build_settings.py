"""User-adjustable settings for the text-to-index build pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from backend.app.ingestion.chunking import ChunkingConfig, ChunkingConfigurationError


@dataclass(frozen=True)
class BuildSettings:
    """Settings that change chunking or index construction for new versions.

    The product deliberately keeps the unified strategy and non-overlap rule
    fixed.  Only the parameters that change the resulting retrieval units or
    build resource usage are configurable here.
    """

    max_tokens: int = 512
    min_tokens: int = 80
    parent_max_tokens: int = 1536
    include_heading: bool = True
    semantic_breakpoint_percentile: float = 85.0
    tokenizer_model: str = ""
    embedding_device: str = "cpu"
    text_index_batch_size: int = 32

    def validate(self) -> "BuildSettings":
        normalized = replace(
            self,
            tokenizer_model=self.tokenizer_model.strip(),
            embedding_device=self.embedding_device.strip().lower(),
        )
        if normalized.min_tokens > normalized.max_tokens:
            raise ValueError("语义最小 Token 数不能大于 Chunk 最大 Token 数")
        if normalized.text_index_batch_size <= 0:
            raise ValueError("索引批量写入数量必须大于 0")
        device = normalized.embedding_device
        if device not in {"cpu", "cuda", "mps"} and not (
            device.startswith("cuda:") and device[5:].isdigit()
        ):
            raise ValueError("Embedding 运行设备仅支持 cpu、cuda、cuda:n 或 mps")
        try:
            ChunkingConfig(
                strategy="unified",
                max_tokens=normalized.max_tokens,
                min_tokens=normalized.min_tokens,
                parent_max_tokens=normalized.parent_max_tokens,
                include_heading=normalized.include_heading,
                semantic_breakpoint_percentile=normalized.semantic_breakpoint_percentile,
                embedding_model="configured-by-model-gateway",
                embedding_device=device,
            ).validate()
        except ChunkingConfigurationError as exc:
            raise ValueError(str(exc)) from exc
        return normalized

    def to_dict(self) -> dict[str, object]:
        """Return the persisted/API representation, including fixed rules."""

        return {
            "strategy": "unified",
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "parent_max_tokens": self.parent_max_tokens,
            "include_heading": self.include_heading,
            "semantic_breakpoint_percentile": self.semantic_breakpoint_percentile,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_local_files_only": True,
            "embedding_device": self.embedding_device,
            "embedding_local_files_only": True,
            "text_index_batch_size": self.text_index_batch_size,
            "overlap_tokens": 0,
        }

    def to_chunking_config(self, embedding_model: str) -> ChunkingConfig:
        self.validate()
        return ChunkingConfig(
            strategy="unified",
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            parent_max_tokens=self.parent_max_tokens,
            include_heading=self.include_heading,
            semantic_breakpoint_percentile=self.semantic_breakpoint_percentile,
            tokenizer_model=self.tokenizer_model.strip() or None,
            tokenizer_local_files_only=True,
            embedding_model=embedding_model,
            embedding_device=self.embedding_device.strip().lower(),
            embedding_local_files_only=True,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        defaults: "BuildSettings" | None = None,
    ) -> "BuildSettings":
        base = defaults or cls()
        result = replace(
            base,
            max_tokens=_as_int(raw, "max_tokens", base.max_tokens),
            min_tokens=_as_int(raw, "min_tokens", base.min_tokens),
            parent_max_tokens=_as_int(raw, "parent_max_tokens", base.parent_max_tokens),
            include_heading=_as_bool(raw, "include_heading", base.include_heading),
            semantic_breakpoint_percentile=_as_float(
                raw,
                "semantic_breakpoint_percentile",
                base.semantic_breakpoint_percentile,
            ),
            tokenizer_model=str(raw.get("tokenizer_model", base.tokenizer_model) or "").strip(),
            embedding_device=str(raw.get("embedding_device", base.embedding_device) or "cpu").strip(),
            text_index_batch_size=_as_int(
                raw,
                "text_index_batch_size",
                base.text_index_batch_size,
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
