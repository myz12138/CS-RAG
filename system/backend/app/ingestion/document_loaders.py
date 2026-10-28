"""Format-neutral document loading for the pluggable chunking layer.

The existing build path deliberately remains unchanged for now.  This module
provides a richer, source-aware representation that can later be selected by a
build configuration without coupling chunking to a particular file format.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree as ET


TEXT_DOCUMENT_EXTENSIONS = frozenset({"txt", "md", "markdown", "rst", "docx"})

_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = "{" + _WORD_NS + "}"
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_UNDERLINE_RE = re.compile(r"^\s*(=+|-+|\^+|~+|\++|" + re.escape('"') + r"+)\s*$")
_CHAPTER_HEADING_RE = re.compile(
    r"^第\s*[一二三四五六七八九十百千万\d]+\s*[章节部分篇](?:\s+.*)?$"
)


class DocumentLoadError(ValueError):
    """A document cannot be converted into structured text blocks."""


class UnsupportedDocumentFormatError(DocumentLoadError):
    """The input format has no registered document loader."""


@dataclass(frozen=True)
class DocumentBlock:
    """A structural unit with a stable location in normalized document text."""

    block_id: str
    kind: str
    text: str
    title_path: tuple[str, ...]
    heading_level: int | None
    char_start: int
    char_end: int
    source_location: dict[str, Any]


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized text plus structure shared by every chunking strategy."""

    document_id: str
    source_name: str
    source_format: str
    text: str
    blocks: tuple[DocumentBlock, ...]
    metadata: dict[str, Any]


class DocumentLoader(Protocol):
    extensions: frozenset[str]

    def load(self, content: bytes, filename: str) -> ParsedDocument: ...


def _document_id(source_name: str, text: str) -> str:
    digest = hashlib.sha256((source_name + "\x00" + text).encode("utf-8")).hexdigest()
    return digest[:20]


def _normalize_text(content: bytes, filename: str) -> str:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"{filename} 不是有效的 UTF-8 文本") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    if not text:
        raise DocumentLoadError(f"{filename} 内容为空")
    return text


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    return offsets


def _heading_title(line: str, suffix: str) -> tuple[str, int] | None:
    match = _HEADING_RE.match(line)
    if match:
        return match.group(2).strip(), len(match.group(1))
    # Plain text has no reliable heading styling.  Only the explicit Chinese
    # chapter form is strong enough to infer a heading; ordinary numbered
    # lines such as ``1. item`` and ``一、item`` remain list/body content.
    if suffix in {"txt", "rst"} and _CHAPTER_HEADING_RE.match(line.strip()):
        value = line.strip()
        return value, 1
    return None


def _is_setext_heading(line: str, next_line: str, suffix: str) -> tuple[str, int] | None:
    if not line.strip() or not next_line.strip() or not _UNDERLINE_RE.match(next_line):
        return None
    if suffix not in {"md", "markdown", "rst", "txt"}:
        return None
    if line.rstrip().endswith(("。", "！", "？", ".", "!", "?", ";", "；")):
        return None
    marker = next_line.strip()[0]
    return line.strip(), 1 if marker == "=" else 2


def _update_title_path(path: list[str], title: str, level: int) -> tuple[str, ...]:
    level = max(1, min(level, 9))
    del path[level - 1 :]
    path.append(title)
    return tuple(path)


def _classify_plain_block(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+|[一二三四五六七八九十]+、)", line) for line in lines):
        return "list"
    if len(lines) > 1 and all("|" in line for line in lines):
        return "table"
    if text.lstrip().startswith("```"):
        return "code"
    return "paragraph"


def _load_plain_text(content: bytes, filename: str) -> ParsedDocument:
    source_name = Path(filename).name
    suffix = Path(source_name).suffix.lower().lstrip(".")
    text = _normalize_text(content, source_name)
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    blocks: list[DocumentBlock] = []
    title_path: list[str] = []
    paragraph_lines: list[str] = []
    paragraph_start: int | None = None
    block_index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, paragraph_start, block_index
        if not paragraph_lines or paragraph_start is None:
            paragraph_lines = []
            paragraph_start = None
            return
        raw = "".join(paragraph_lines)
        value = raw.strip()
        if value:
            leading = len(raw) - len(raw.lstrip())
            start = paragraph_start + leading
            blocks.append(
                DocumentBlock(
                    block_id=f"b{block_index:06d}",
                    kind=_classify_plain_block(value),
                    text=value,
                    title_path=tuple(title_path),
                    heading_level=None,
                    char_start=start,
                    char_end=start + len(value),
                    source_location={
                        "line_start": text.count("\n", 0, start) + 1,
                        "line_end": text.count("\n", 0, start + len(value)) + 1,
                    },
                )
            )
            block_index += 1
        paragraph_lines = []
        paragraph_start = None

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        heading = _heading_title(line, suffix)
        consumed = 1
        setext = None
        if i + 1 < len(lines):
            setext = _is_setext_heading(line, lines[i + 1].rstrip("\n"), suffix)
        if heading is not None or setext is not None:
            flush_paragraph()
            heading_title, heading_level = heading or setext  # type: ignore[misc]
            path = _update_title_path(title_path, heading_title, heading_level)
            end_line = i + (2 if setext is not None else 1)
            block_end = offsets[end_line - 1] + len(lines[end_line - 1].rstrip("\n"))
            blocks.append(
                DocumentBlock(
                    block_id=f"b{block_index:06d}",
                    kind="heading",
                    text=heading_title,
                    title_path=path,
                    heading_level=heading_level,
                    char_start=offsets[i] + (len(line) - len(line.lstrip())),
                    char_end=block_end,
                    source_location={
                        "line_start": i + 1,
                        "line_end": end_line,
                        "marker": "setext" if setext is not None else "atx_or_heuristic",
                    },
                )
            )
            block_index += 1
            i = end_line
            continue

        if paragraph_start is None:
            paragraph_start = offsets[i]
        paragraph_lines.append(raw_line)
        i += consumed

    flush_paragraph()
    if not blocks:
        raise DocumentLoadError(f"{source_name} 没有可识别的文本块")
    return ParsedDocument(
        document_id=_document_id(source_name, text),
        source_name=source_name,
        source_format=suffix,
        text=text,
        blocks=tuple(blocks),
        metadata={"loader": "plain_text", "line_count": len(lines)},
    )


def _word_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == _W + "t":
            parts.append(node.text or "")
        elif node.tag == _W + "tab":
            parts.append("\t")
        elif node.tag == _W + "br":
            parts.append("\n")
    return "".join(parts).strip()


def _word_style(paragraph: ET.Element) -> tuple[str, int | None, bool]:
    properties = paragraph.find(_W + "pPr")
    if properties is None:
        return "", None, False
    style_node = properties.find(_W + "pStyle")
    style = str(style_node.get(_W + "val", "")) if style_node is not None else ""
    outline_node = properties.find(_W + "outlineLvl")
    outline = None
    if outline_node is not None:
        try:
            outline = int(outline_node.get(_W + "val", "0")) + 1
        except ValueError:
            outline = None
    is_list = properties.find(_W + "numPr") is not None or "list" in style.lower()
    return style, outline, is_list


def _word_heading(style: str, outline: int | None) -> int | None:
    lowered = style.lower()
    match = re.search(r"(?:heading|标题|title)[ _-]*([1-9])", lowered)
    if match:
        return int(match.group(1))
    if lowered.strip() in {"title", "标题"}:
        return 1
    if outline is not None:
        return max(1, min(outline, 9))
    if any(token in lowered for token in ("chapter", "section", "章节", "小节")):
        return 1
    return None


def _word_table_text(table: ET.Element) -> str:
    rows: list[str] = []
    for row in table.findall(_W + "tr"):
        cells: list[str] = []
        for cell in row.findall(_W + "tc"):
            paragraphs = [_word_text(p) for p in cell.findall(".//" + _W + "p")]
            value = " ".join(item for item in paragraphs if item)
            cells.append(value)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def _load_docx(content: bytes, filename: str) -> ParsedDocument:
    source_name = Path(filename).name
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise DocumentLoadError(f"{source_name} 不是有效的 DOCX 文件") from exc

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise DocumentLoadError(f"{source_name} 的 Word 文档结构无法解析") from exc

    body = root.find(".//" + _W + "body")
    if body is None:
        raise DocumentLoadError(f"{source_name} 不包含 Word 正文")

    raw_blocks: list[dict[str, Any]] = []
    title_path: list[str] = []
    paragraph_index = 0
    table_index = 0
    block_index = 0
    for child in list(body):
        if child.tag == _W + "p":
            value = _word_text(child)
            paragraph_index += 1
            if not value:
                continue
            style, outline, is_list = _word_style(child)
            level = _word_heading(style, outline)
            if level is not None:
                path = _update_title_path(title_path, value, level)
                kind = "heading"
                block_path = path
            else:
                kind = "list" if is_list else "paragraph"
                block_path = tuple(title_path)
            raw_blocks.append(
                {
                    "block_id": f"b{block_index:06d}",
                    "kind": kind,
                    "text": value,
                    "title_path": block_path,
                    "heading_level": level,
                    "source_location": {
                        "paragraph_index": paragraph_index,
                        "style": style,
                        "outline_level": outline,
                    },
                }
            )
            block_index += 1
        elif child.tag == _W + "tbl":
            value = _word_table_text(child)
            table_index += 1
            if not value:
                continue
            raw_blocks.append(
                {
                    "block_id": f"b{block_index:06d}",
                    "kind": "table",
                    "text": value,
                    "title_path": tuple(title_path),
                    "heading_level": None,
                    "source_location": {"table_index": table_index},
                }
            )
            block_index += 1

    if not raw_blocks:
        raise DocumentLoadError(f"{source_name} 不包含可提取的正文、表格或标题")

    pieces: list[str] = []
    blocks: list[DocumentBlock] = []
    cursor = 0
    for index, raw in enumerate(raw_blocks):
        value = str(raw["text"]).strip()
        if index:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(value)
        cursor += len(value)
        blocks.append(
            DocumentBlock(
                block_id=raw["block_id"],
                kind=raw["kind"],
                text=value,
                title_path=tuple(raw["title_path"]),
                heading_level=raw["heading_level"],
                char_start=start,
                char_end=cursor,
                source_location=raw["source_location"],
            )
        )
    text = "".join(pieces)
    return ParsedDocument(
        document_id=_document_id(source_name, text),
        source_name=source_name,
        source_format="docx",
        text=text,
        blocks=tuple(blocks),
        metadata={
            "loader": "docx_xml",
            "paragraph_count": paragraph_index,
            "table_count": table_index,
        },
    )


class DefaultDocumentLoader:
    """Small registry that can be extended with PDF/HTML/custom loaders."""

    def __init__(self) -> None:
        self._loaders: dict[str, DocumentLoader] = {}
        for extension in {"txt", "md", "markdown", "rst"}:
            self.register(extension, _PlainTextLoader())
        self.register("docx", _DocxLoader())

    def register(self, extension: str, loader: DocumentLoader) -> None:
        self._loaders[extension.lower().lstrip(".")] = loader

    def load(self, content: bytes, filename: str) -> ParsedDocument:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix == "doc":
            raise UnsupportedDocumentFormatError("暂不支持旧版 .doc；请另存为 .docx 后再导入")
        loader = self._loaders.get(suffix)
        if loader is None:
            supported = ", ".join(sorted(self._loaders))
            raise UnsupportedDocumentFormatError(f"暂不支持 .{suffix or 'unknown'}；当前文档加载器支持：{supported}")
        return loader.load(content, filename)


class _PlainTextLoader:
    extensions = frozenset({"txt", "md", "markdown", "rst"})

    def load(self, content: bytes, filename: str) -> ParsedDocument:
        return _load_plain_text(content, filename)


class _DocxLoader:
    extensions = frozenset({"docx"})

    def load(self, content: bytes, filename: str) -> ParsedDocument:
        return _load_docx(content, filename)


_DEFAULT_LOADER = DefaultDocumentLoader()


def load_document(content: bytes, filename: str, *, loader: DefaultDocumentLoader | None = None) -> ParsedDocument:
    """Load TXT/Markdown/RST/DOCX into the common structural document model."""

    return (loader or _DEFAULT_LOADER).load(content, filename)


__all__ = [
    "DocumentBlock",
    "DocumentLoadError",
    "DefaultDocumentLoader",
    "ParsedDocument",
    "TEXT_DOCUMENT_EXTENSIONS",
    "UnsupportedDocumentFormatError",
    "load_document",
]
