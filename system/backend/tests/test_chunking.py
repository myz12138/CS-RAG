"""In-memory tests for the format-neutral chunking layer."""

from __future__ import annotations

import io
import zipfile

import pytest

from backend.app.ingestion.chunking import (
    ChunkingConfig,
    ChunkingConfigurationError,
    chunk_document,
    create_chunker,
)
from backend.app.ingestion.document_loaders import load_document


MARKDOWN = """# 总则

这是第一句。这里继续说明背景。

## 细节

这是第二部分内容，包含更多说明。
第二行也属于同一段落。

| 字段 | 说明 |
| --- | --- |
| A | 示例 |
""".encode("utf-8")


def _docx_bytes() -> bytes:
    xml = """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>项目说明</w:t></w:r></w:p>
        <w:p><w:r><w:t>正文内容。</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      </w:body>
    </w:document>"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_document_loader_keeps_heading_path_and_block_kinds() -> None:
    document = load_document(MARKDOWN, "demo.md")

    assert document.source_format == "md"
    assert document.blocks[0].kind == "heading"
    assert document.blocks[1].title_path == ("总则",)
    assert any(block.kind == "table" for block in document.blocks)
    assert all(block.char_start < block.char_end for block in document.blocks)


def test_docx_loader_reads_heading_paragraph_and_table_without_extra_dependency() -> None:
    document = load_document(_docx_bytes(), "demo.docx")

    assert [block.kind for block in document.blocks] == ["heading", "paragraph", "table"]
    assert document.blocks[1].title_path == ("项目说明",)
    assert document.blocks[2].text == "A | B"


def test_all_non_semantic_strategies_return_traceable_chunks() -> None:
    configs = (
        ChunkingConfig(strategy="structure", max_tokens=12, min_tokens=1),
        ChunkingConfig(strategy="fixed", max_tokens=12),
        ChunkingConfig(
            strategy="parent_child",
            parent_max_tokens=20,
            child_max_tokens=8,
        ),
    )

    for config in configs:
        result = chunk_document(MARKDOWN, "demo.md", config)
        assert result.chunks
        assert result.diagnostics["chunk_count"] == len(result.chunks)
        assert all(chunk.document_id == result.document.document_id for chunk in result.chunks)
        assert all(chunk.block_ids for chunk in result.chunks)


def test_fixed_strategy_has_non_overlapping_source_ranges() -> None:
    result = chunk_document(
        ("这是很长的一段内容。" * 30).encode("utf-8"),
        "long.txt",
        ChunkingConfig(strategy="fixed", max_tokens=20, include_heading=False),
    )

    assert len(result.chunks) > 1
    assert all(left.char_end <= right.char_start for left, right in zip(result.chunks, result.chunks[1:]))


def test_token_limit_keeps_only_complete_sentences() -> None:
    text = "甲乙丙丁。" * 4 + "戊己庚辛壬癸子丑。"
    result = chunk_document(
        text.encode("utf-8"),
        "sentence-boundary.txt",
        ChunkingConfig(strategy="fixed", max_tokens=24, include_heading=False),
    )

    assert result.chunks[0].text == "甲乙丙丁。" * 4
    assert result.chunks[1].text == "戊己庚辛壬癸子丑。"
    assert all(chunk.text.endswith(("。", "！", "？", ".", "!", "?", "；", ";")) for chunk in result.chunks)


def test_oversized_single_sentence_is_not_cut() -> None:
    text = "这是一个本身就超过 Token 上限但必须保持完整的长句子。"
    result = chunk_document(
        text.encode("utf-8"),
        "oversized-sentence.txt",
        ChunkingConfig(strategy="fixed", max_tokens=5, include_heading=False),
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].text == text
    assert result.chunks[0].token_count > 5


def test_semantic_strategy_accepts_injected_embeddings() -> None:
    def embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "第一" in text else [0.0, 1.0] for text in texts]

    result = chunk_document(
        "# 章节\n\n第一主题的内容。第一主题的补充。第二主题的内容。".encode("utf-8"),
        "semantic.md",
        ChunkingConfig(strategy="semantic", max_tokens=80, min_tokens=1, semantic_breakpoint_percentile=50),
        embedding_function=embeddings,
    )

    assert result.chunks
    assert result.diagnostics["semantic_sections"] == 1
    assert result.diagnostics["semantic_breakpoints"] >= 1


def test_parent_child_strategy_links_children_to_parents() -> None:
    result = chunk_document(
        ("# 章节\n\n" + "内容。" * 120).encode("utf-8"),
        "hierarchy.md",
        ChunkingConfig(
            strategy="parent_child",
            parent_max_tokens=80,
            child_max_tokens=20,
        ),
    )

    parents = {chunk.chunk_id for chunk in result.chunks if chunk.level == "parent"}
    children = [chunk for chunk in result.chunks if chunk.level == "child"]
    assert parents
    assert children
    assert all(child.parent_id in parents for child in children)


def test_parent_child_sibling_spans_do_not_overlap() -> None:
    result = chunk_document(
        ("# 章节\n\n" + "连续内容。" * 180).encode("utf-8"),
        "hierarchy-non-overlap.md",
        ChunkingConfig(
            strategy="parent_child",
            parent_max_tokens=160,
            child_max_tokens=20,
        ),
    )

    children = [chunk for chunk in result.chunks if chunk.level == "child"]
    assert len(children) > 1
    by_parent: dict[str, list] = {}
    for child in children:
        assert child.parent_id is not None
        by_parent.setdefault(child.parent_id, []).append(child)
    for siblings in by_parent.values():
        siblings.sort(key=lambda chunk: int(chunk.metadata["relative_char_start"]))
        assert all(
            int(left.metadata["relative_char_end"])
            <= int(right.metadata["relative_char_start"])
            for left, right in zip(siblings, siblings[1:])
        )


def test_headingless_documents_use_paragraph_boundaries_without_virtual_titles() -> None:
    result = chunk_document(
        "第一段包含两个句子。它们属于同一段落。\n\n第二段是另一主题。".encode("utf-8"),
        "plain.txt",
        ChunkingConfig(strategy="structure", max_tokens=80),
    )

    assert len(result.chunks) == 2
    assert all(chunk.title_path == () for chunk in result.chunks)
    assert all("章节路径：" not in chunk.text for chunk in result.chunks)


def test_plain_numbered_lines_are_not_inferred_as_headings() -> None:
    document = load_document(
        "1. 第一条内容\n2. 第二条内容".encode("utf-8"),
        "numbered.txt",
    )

    assert all(block.kind != "heading" for block in document.blocks)
    assert all(block.title_path == () for block in document.blocks)


def test_unified_strategy_requires_real_embedding_and_preserves_paragraphs() -> None:
    with pytest.raises(ChunkingConfigurationError, match="embedding"):
        create_chunker(ChunkingConfig(strategy="unified"))

    def embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "第一" in text else [0.0, 1.0] for text in texts]

    result = chunk_document(
        "第一段包含两个句子。它们属于第一主题。\n\n第二段属于第二主题。".encode("utf-8"),
        "plain.txt",
        ChunkingConfig(
            strategy="unified",
            max_tokens=30,
            parent_max_tokens=60,
            min_tokens=1,
        ),
        embedding_function=embeddings,
    )

    assert result.diagnostics["headings_present"] is False
    assert result.diagnostics["hierarchy_available"] is False
    assert result.diagnostics["hierarchy_applied"] is False
    assert result.diagnostics["hierarchy_reason"] == "未识别到标题/章节，按自然段处理"
    assert result.diagnostics["parent_count"] == 0
    assert result.diagnostics["child_count"] == 0
    assert all(chunk.title_path == () for chunk in result.chunks)
    assert all(chunk.level == "document" for chunk in result.chunks)
    assert all(chunk.parent_id is None for chunk in result.chunks)
    assert all(chunk.metadata["heading_injected"] is False for chunk in result.chunks)
    assert all("章节路径：" not in chunk.text for chunk in result.chunks)


def test_parent_child_strategy_does_not_invent_hierarchy_without_headings() -> None:
    result = chunk_document(
        ("第一段内容。" * 20 + "\n\n" + "第二段内容。" * 20).encode("utf-8"),
        "flat.txt",
        ChunkingConfig(
            strategy="parent_child",
            parent_max_tokens=80,
            child_max_tokens=20,
        ),
    )

    assert result.diagnostics["hierarchy_available"] is False
    assert result.diagnostics["hierarchy_applied"] is False
    assert result.diagnostics["parent_count"] == 0
    assert result.diagnostics["child_count"] == 0
    assert all(chunk.level == "document" for chunk in result.chunks)
    assert all(chunk.parent_id is None for chunk in result.chunks)
    assert all(chunk.title_path == () for chunk in result.chunks)


def test_heading_only_document_does_not_invent_parent_body_relation() -> None:
    def embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    result = chunk_document(
        "# 目录标题\n\n## 子标题".encode("utf-8"),
        "heading-only.md",
        ChunkingConfig(
            strategy="unified",
            max_tokens=30,
            parent_max_tokens=60,
            min_tokens=1,
        ),
        embedding_function=embeddings,
    )

    assert result.diagnostics["headings_present"] is True
    assert result.diagnostics["hierarchy_available"] is False
    assert result.diagnostics["hierarchy_applied"] is False
    assert result.diagnostics["parent_count"] == 0
    assert result.diagnostics["child_count"] == 0
    assert all(chunk.level == "document" for chunk in result.chunks)
    assert all(chunk.parent_id is None for chunk in result.chunks)


def test_unscoped_preamble_stays_flat_when_later_sections_have_hierarchy() -> None:
    def embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    result = chunk_document(
        "开头没有章节归属。\n\n# 正式章节\n\n章节正文内容。".encode("utf-8"),
        "preamble.md",
        ChunkingConfig(
            strategy="unified",
            max_tokens=30,
            parent_max_tokens=60,
            min_tokens=1,
        ),
        embedding_function=embeddings,
    )

    flat = [chunk for chunk in result.chunks if chunk.metadata.get("unscoped_section")]
    parents = [chunk for chunk in result.chunks if chunk.level == "parent"]
    children = [chunk for chunk in result.chunks if chunk.level == "child"]
    assert result.diagnostics["hierarchy_available"] is True
    assert result.diagnostics["unscoped_section_count"] == 1
    assert flat and all(chunk.level == "document" and chunk.parent_id is None for chunk in flat)
    assert parents and children
