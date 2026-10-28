"""Format-specific parsers that produce internal, source-aware records."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import Any


class InputParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedTextEntry:
    ordinal: int
    title: str
    content: str
    source_location: dict[str, Any]


@dataclass(frozen=True)
class ParsedTriple:
    head: str
    relation: str
    tail: str
    source_line: int | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class ParseIssue:
    line: int | None
    message: str


@dataclass(frozen=True)
class ParsedKG:
    triples: tuple[ParsedTriple, ...]
    issues: tuple[ParseIssue, ...]


def parse_text(content: bytes, filename: str, *, max_chars: int = 1200) -> list[ParsedTextEntry]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"txt", "md", "markdown", "rst"}:
        raise InputParseError(f"当前构建器暂不支持解析 .{suffix or 'unknown'} 文本，请先使用 txt 或 md")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputParseError(f"{filename} 不是有效的 UTF-8 文本") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise InputParseError(f"{filename} 内容为空")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    entries: list[ParsedTextEntry] = []
    ordinal = 0
    for paragraph_index, paragraph in enumerate(paragraphs):
        for start in range(0, len(paragraph), max_chars):
            chunk = paragraph[start : start + max_chars].strip()
            if not chunk:
                continue
            entries.append(
                ParsedTextEntry(
                    ordinal=ordinal,
                    title=filename,
                    content=chunk,
                    source_location={
                        "paragraph": paragraph_index,
                        "char_start": start,
                        "char_end": start + len(chunk),
                    },
                )
            )
            ordinal += 1
    return entries


_HEAD_KEYS = ("head", "subject", "source", "from")
_RELATION_KEYS = ("relation", "predicate", "rel")
_TAIL_KEYS = ("tail", "object", "target", "to")
_EVIDENCE_KEYS = ("evidence", "context", "sentence", "title")


def _normal_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _value(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    normalized = {_normal_key(key): value for key, value in record.items()}
    for key in keys:
        value = normalized.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _records_from_json(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("triples", "relations", "edges", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return []


def _triple_from_record(record: Any, line: int | None) -> ParsedTriple:
    if not isinstance(record, dict):
        raise InputParseError("记录必须是对象，不能直接是字符串或数组")
    head = _value(record, _HEAD_KEYS)
    relation = _value(record, _RELATION_KEYS)
    tail = _value(record, _TAIL_KEYS)
    if not head or not relation or not tail:
        raise InputParseError("缺少主体、关系或客体字段（支持 head/relation/tail 或 subject/predicate/object）")
    return ParsedTriple(
        head=head,
        relation=relation,
        tail=tail,
        source_line=line,
        evidence=_value(record, _EVIDENCE_KEYS),
    )


def parse_kg(content: bytes, source_format: str) -> ParsedKG:
    fmt = source_format.lower().lstrip(".")
    if fmt in {"json", "jsonl"}:
        return _parse_json_kg(content, fmt)
    if fmt in {"csv", "tsv"}:
        return _parse_delimited_kg(content, fmt)
    raise InputParseError(f"当前构建器暂不支持解析 .{fmt} KG 文件")


def _parse_json_kg(content: bytes, fmt: str) -> ParsedKG:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputParseError("KG 文件不是有效的 UTF-8 文本") from exc
    triples: list[ParsedTriple] = []
    issues: list[ParseIssue] = []
    if fmt == "json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputParseError(f"JSON 解析失败：第 {exc.lineno} 行") from exc
        records = _records_from_json(value)
        for index, record in enumerate(records, start=1):
            try:
                triples.append(_triple_from_record(record, index))
            except InputParseError as exc:
                issues.append(ParseIssue(index, str(exc)))
        return ParsedKG(tuple(triples), tuple(issues))

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            triples.append(_triple_from_record(record, line_number))
        except (json.JSONDecodeError, InputParseError) as exc:
            issues.append(ParseIssue(line_number, str(exc)))
    return ParsedKG(tuple(triples), tuple(issues))


def _parse_delimited_kg(content: bytes, fmt: str) -> ParsedKG:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputParseError("KG 文件不是有效的 UTF-8 文本") from exc
    delimiter = "\t" if fmt == "tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise InputParseError("CSV/TSV 缺少表头")
    triples: list[ParsedTriple] = []
    issues: list[ParseIssue] = []
    for line_number, record in enumerate(reader, start=2):
        try:
            triples.append(_triple_from_record(record, line_number))
        except InputParseError as exc:
            issues.append(ParseIssue(line_number, str(exc)))
    return ParsedKG(tuple(triples), tuple(issues))
