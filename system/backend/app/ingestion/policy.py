"""Stable, intentionally small rules for the first input boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# The product input boundary is deliberately text-first.  KG parsers remain
# in the codebase for reading historical artifacts, but new uploads must start
# from source documents so the system can build both the text and KG layers.
TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "rst", "docx"})
KG_EXTENSIONS = frozenset({"json", "jsonl", "csv", "tsv"})
SOURCE_KINDS = frozenset({"auto", "text"})


class UnsupportedSourceFormatError(ValueError):
    """Raised when an input cannot be classified by the current boundary."""


@dataclass(frozen=True)
class ClassifiedSource:
    source_kind: str
    source_format: str


def classify_source(filename: str, requested_kind: str = "auto") -> ClassifiedSource:
    """Classify a raw input without parsing or calling an LLM."""

    normalized_kind = (requested_kind or "auto").strip().lower()
    if normalized_kind not in SOURCE_KINDS:
        raise UnsupportedSourceFormatError("当前只允许上传文本，source_kind 必须是 auto 或 text")

    suffix = Path(filename).suffix.lower().lstrip(".")
    if not suffix:
        raise UnsupportedSourceFormatError("文件缺少扩展名，暂时无法判断输入类型")

    if normalized_kind == "auto":
        if suffix in TEXT_EXTENSIONS:
            normalized_kind = "text"
        else:
            raise UnsupportedSourceFormatError(
                "当前只支持文本文件：.txt、.md、.markdown、.rst、.docx；KG 文件不作为新建知识库输入"
            )

    allowed = TEXT_EXTENSIONS
    if suffix not in allowed:
        raise UnsupportedSourceFormatError(f"{filename} 不是当前支持的文本格式")

    return ClassifiedSource(source_kind=normalized_kind, source_format=suffix)
