"""Pluggable, source-aware chunking strategies for CS-RAG.

The module sits between document loading and the build service.  It does not
call an LLM and it does not write to the knowledge base; the build service
owns persistence and invokes the real LLM only after retrieval chunks exist.

Supported strategies:

``structure``
    Keep headings/sections and pack natural document blocks up to a token
    budget.
``fixed``
    Deterministic non-overlapping token windows.
``semantic``
    Sentence units are grouped using adjacent embedding similarity.  The
    embedding provider is injectable; the default provider is local-only.
``parent_child``
    Produce larger structural parent chunks and smaller non-overlapping child
    chunks for hierarchical retrieval.
``unified``
    The product strategy: structure boundaries, embedding-guided semantic
    grouping, token guardrails, and parent/child context.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

from .document_loaders import DocumentBlock, DefaultDocumentLoader, ParsedDocument, load_document


ChunkStrategy = Literal["unified", "structure", "fixed", "semantic", "parent_child"]

_STRATEGY_ALIASES: dict[str, ChunkStrategy] = {
    "unified": "unified",
    "joint": "unified",
    "hybrid": "unified",
    "联合": "unified",
    "联合切分": "unified",
    "structure": "structure",
    "structure_first": "structure",
    "结构": "structure",
    "结构优先": "structure",
    "fixed": "fixed",
    "fixed_token": "fixed",
    "token": "fixed",
    "固定窗口": "fixed",
    "semantic": "semantic",
    "语义": "semantic",
    "semantic_sentence": "semantic",
    "parent_child": "parent_child",
    "parent-child": "parent_child",
    "hierarchical": "parent_child",
    "父子块": "parent_child",
}


class ChunkingConfigurationError(ValueError):
    """Raised when a strategy cannot be constructed from its configuration."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Parameters shared by the four strategies.

    ``max_tokens`` is the normal retrieval chunk budget.  For
    ``parent_child``, ``parent_max_tokens`` and ``child_max_tokens`` control
    the two levels independently.  Token counting is approximate by default
    so the module works without changing the environment; a local HuggingFace
    tokenizer can be selected with ``tokenizer_model`` when exact counting is
    needed.
    """

    strategy: str = "unified"
    max_tokens: int = 512
    min_tokens: int = 80
    parent_max_tokens: int = 1536
    child_max_tokens: int = 384
    include_heading: bool = True
    semantic_breakpoint_percentile: float = 85.0
    tokenizer_model: str | None = None
    tokenizer_local_files_only: bool = True
    embedding_model: str | None = None
    embedding_device: str = "cpu"
    embedding_local_files_only: bool = True

    def normalized_strategy(self) -> ChunkStrategy:
        normalized = _STRATEGY_ALIASES.get(self.strategy.strip().lower())
        if normalized is None:
            supported = ", ".join(sorted(set(_STRATEGY_ALIASES.values())))
            raise ChunkingConfigurationError(f"不支持的 chunk 策略：{self.strategy}；可选：{supported}")
        return normalized

    def validate(self) -> ChunkStrategy:
        strategy = self.normalized_strategy()
        positive = {
            "max_tokens": self.max_tokens,
            "parent_max_tokens": self.parent_max_tokens,
            "child_max_tokens": self.child_max_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ChunkingConfigurationError(f"{name} 必须大于 0")
        if self.min_tokens < 0:
            raise ChunkingConfigurationError("min_tokens 不能小于 0")
        if strategy == "parent_child" and self.parent_max_tokens < self.child_max_tokens:
            raise ChunkingConfigurationError("parent_max_tokens 不应小于 child_max_tokens")
        if strategy == "unified" and self.parent_max_tokens < self.max_tokens:
            raise ChunkingConfigurationError("联合切分中 parent_max_tokens 不应小于 max_tokens")
        if not 0 <= self.semantic_breakpoint_percentile <= 100:
            raise ChunkingConfigurationError("semantic_breakpoint_percentile 必须在 0 到 100 之间")
        return strategy


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit with source traceability and optional hierarchy."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    title_path: tuple[str, ...]
    strategy: ChunkStrategy
    level: Literal["document", "parent", "child"]
    token_count: int
    char_start: int
    char_end: int
    block_ids: tuple[str, ...]
    parent_id: str | None
    source_location: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkingResult:
    """Chunks plus diagnostics suitable for a preview panel or build log."""

    document: ParsedDocument
    strategy: ChunkStrategy
    chunks: tuple[Chunk, ...]
    diagnostics: dict[str, Any]


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class EmbeddingFunction(Protocol):
    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]",
    re.UNICODE,
)


class ApproximateTokenCounter:
    """Deterministic token estimate that needs no tokenizer or network access."""

    name = "approximate-cjk-aware"

    def count(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))


class HuggingFaceTokenCounter:
    """Lazy local HuggingFace tokenizer adapter for exact model-aware counts."""

    def __init__(self, model_name_or_path: str, *, local_files_only: bool = True) -> None:
        self.model_name_or_path = model_name_or_path
        self.local_files_only = local_files_only
        self.name = f"huggingface:{model_name_or_path}"
        self._tokenizer: Any | None = None

    def _load(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on optional runtime
                raise ChunkingConfigurationError(
                    "精确 token 计数需要 transformers；当前环境未安装"
                ) from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    local_files_only=self.local_files_only,
                    use_fast=True,
                )
            except Exception as exc:  # pragma: no cover - model/cache dependent
                mode = "本地缓存" if self.local_files_only else "模型源"
                raise ChunkingConfigurationError(
                    f"无法从{mode}加载 tokenizer：{self.model_name_or_path}"
                ) from exc
        return self._tokenizer

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._load()
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        return len(encoded.get("input_ids", []))


class SentenceTransformerEmbeddingProvider:
    """Lazy local sentence-transformers provider used by semantic chunking."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.local_files_only = local_files_only
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise ChunkingConfigurationError(
                    "语义切分需要 sentence-transformers；当前环境未安装"
                ) from exc
            kwargs: dict[str, Any] = {"device": self.device}
            if self.local_files_only:
                # sentence-transformers forwards this flag to the underlying
                # transformers loader in current releases.  Keeping it local
                # by default prevents an accidental download during preview.
                kwargs["local_files_only"] = True
            try:
                self._model = SentenceTransformer(self.model_name_or_path, **kwargs)
            except TypeError:
                # Older sentence-transformers versions do not accept the flag.
                # The caller can explicitly opt out of local-only mode; when
                # local-only is requested, fail rather than silently go online.
                if self.local_files_only:
                    raise ChunkingConfigurationError(
                        "当前 sentence-transformers 版本不支持 local_files_only，请升级后使用本地模型"
                    )
                self._model = SentenceTransformer(self.model_name_or_path, device=self.device)
            except Exception as exc:  # pragma: no cover - model/cache dependent
                mode = "本地模型" if self.local_files_only else "模型源"
                raise ChunkingConfigurationError(
                    f"无法从{mode}加载 embedding 模型：{self.model_name_or_path}"
                ) from exc
        return self._model

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors


@dataclass(frozen=True)
class _ChunkSpec:
    body: str
    title_path: tuple[str, ...]
    char_start: int
    char_end: int
    block_ids: tuple[str, ...]
    source_location: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _heading_prefix(title_path: tuple[str, ...], include_heading: bool) -> str:
    if not include_heading or not title_path:
        return ""
    return f"章节路径：{' / '.join(title_path)}\n\n"


def _render_body(body: str, title_path: tuple[str, ...], include_heading: bool) -> tuple[str, int]:
    prefix = _heading_prefix(title_path, include_heading)
    return prefix + body.strip(), len(prefix)


def _chunk_id(document_id: str, strategy: ChunkStrategy, level: str, ordinal: int, start: int, end: int) -> str:
    digest = hashlib.sha1(f"{document_id}:{strategy}:{level}:{ordinal}:{start}:{end}".encode()).hexdigest()[:10]
    return f"{strategy}-{level}-{ordinal:05d}-{digest}"


def _block_ids_for_span(document: ParsedDocument, start: int, end: int) -> tuple[str, ...]:
    selected = [block.block_id for block in document.blocks if block.char_end > start and block.char_start < end]
    if selected:
        return tuple(selected)
    if document.blocks:
        nearest = min(document.blocks, key=lambda block: abs(block.char_start - start))
        return (nearest.block_id,)
    return ()


def _source_location_for_blocks(document: ParsedDocument, block_ids: Sequence[str]) -> dict[str, Any]:
    locations = [block.source_location for block in document.blocks if block.block_id in block_ids]
    result: dict[str, Any] = {"block_ids": list(block_ids)}
    if locations:
        for key in ("line_start", "line_end", "paragraph_index", "table_index"):
            values = [value for location in locations if (value := location.get(key)) is not None]
            if values:
                result[key] = min(values) if key.endswith("start") else max(values)
    return result


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    value = text[start:end]
    left = len(value) - len(value.lstrip())
    right = len(value.rstrip())
    trimmed_start = start + left
    trimmed_end = start + right
    if trimmed_start < trimmed_end:
        return trimmed_start, trimmed_end
    return start, end


def _split_text_spans(
    text: str,
    max_tokens: int,
    counter: TokenCounter,
    *,
    base_offset: int = 0,
) -> list[tuple[str, int, int]]:
    """Pack complete sentences into non-overlapping token-bounded spans.

    The token budget is applied at sentence boundaries.  If one sentence is
    itself longer than the budget, it remains intact instead of being cut in
    the middle; preserving the source sentence is more important than
    enforcing an impossible hard limit for that atomic unit.
    """

    if not text:
        return []

    sentence_spans = _sentence_spans(text, 0)
    if not sentence_spans:
        return []

    spans: list[tuple[str, int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    def emit() -> None:
        nonlocal current_start, current_end
        if current_start is None or current_end is None:
            return
        clean_start, clean_end = _trimmed_span(text, current_start, current_end)
        if clean_start < clean_end:
            spans.append(
                (
                    text[clean_start:clean_end],
                    base_offset + clean_start,
                    base_offset + clean_end,
                )
            )
        current_start = None
        current_end = None

    for _, sentence_start, sentence_end in sentence_spans:
        if current_start is None or current_end is None:
            current_start = sentence_start
            current_end = sentence_end
            continue

        candidate_end = sentence_end
        candidate = text[current_start:candidate_end]
        if counter.count(candidate) <= max_tokens:
            current_end = candidate_end
        else:
            emit()
            current_start = sentence_start
            current_end = sentence_end

    emit()
    return spans


def _section_groups(document: ParsedDocument) -> list[list[DocumentBlock]]:
    """Group body blocks by contiguous heading context.

    Heading blocks act as hard boundaries and are represented in each output
    chunk through ``title_path``.  This prevents a chunk from accidentally
    mixing two sections while avoiding repeated raw heading text.
    """

    has_heading = any(block.kind == "heading" for block in document.blocks)
    if not has_heading:
        # A heading-less document has no reliable section hierarchy.  Keep
        # each natural paragraph/block independent instead of inventing a
        # virtual title or merging unrelated paragraphs.
        body_blocks = [block for block in document.blocks if block.kind != "heading"]
        if body_blocks:
            return [[block] for block in body_blocks]

    groups: list[list[DocumentBlock]] = []
    current: list[DocumentBlock] = []
    current_path: tuple[str, ...] | None = None
    for block in document.blocks:
        if block.kind == "heading":
            if current:
                groups.append(current)
                current = []
            current_path = None
            continue
        if current and block.title_path != current_path:
            groups.append(current)
            current = []
        current_path = block.title_path
        current.append(block)
    if current:
        groups.append(current)
    if groups:
        return groups
    # A heading-only document still needs a retrievable representation.
    return [[block] for block in document.blocks]


def _structure_specs(
    document: ParsedDocument,
    counter: TokenCounter,
    *,
    max_tokens: int,
    include_heading: bool,
) -> list[_ChunkSpec]:
    specs: list[_ChunkSpec] = []
    for group in _section_groups(document):
        if not group:
            continue
        path = group[0].title_path
        prefix_tokens = counter.count(_heading_prefix(path, include_heading))
        body_budget = max(1, max_tokens - prefix_tokens)
        current: list[DocumentBlock] = []

        def emit_current() -> None:
            nonlocal current
            if not current:
                return
            body = "\n\n".join(block.text for block in current).strip()
            if not body:
                current = []
                return
            start = min(block.char_start for block in current)
            end = max(block.char_end for block in current)
            ids = tuple(block.block_id for block in current)
            specs.append(
                _ChunkSpec(
                    body=body,
                    title_path=path,
                    char_start=start,
                    char_end=end,
                    block_ids=ids,
                    source_location=_source_location_for_blocks(document, ids),
                    metadata={"block_kinds": [block.kind for block in current]},
                )
            )
            current = []

        for block in group:
            if counter.count(block.text) > body_budget:
                emit_current()
                for value, start, end in _split_text_spans(
                    block.text,
                    body_budget,
                    counter,
                    base_offset=block.char_start,
                ):
                    specs.append(
                        _ChunkSpec(
                            body=value,
                            title_path=path,
                            char_start=start,
                            char_end=end,
                            block_ids=(block.block_id,),
                            source_location=dict(block.source_location),
                            metadata={"split_from_block": True, "block_kinds": [block.kind]},
                        )
                    )
                continue
            if block.kind in {"list", "table", "code"}:
                # Keep structured/special blocks intact and do not blend them
                # into an adjacent prose block.  They are split by tokens only
                # when the block itself exceeds the hard budget.
                emit_current()
                current.append(block)
                emit_current()
                continue
            if current and any(item.kind in {"list", "table", "code"} for item in current):
                emit_current()
            candidate = "\n\n".join([*(item.text for item in current), block.text]).strip()
            candidate_count = counter.count(_render_body(candidate, path, include_heading)[0])
            if current and candidate_count > max_tokens:
                emit_current()
            current.append(block)
        emit_current()
    return specs


def _make_chunk(
    document: ParsedDocument,
    spec: _ChunkSpec,
    *,
    strategy: ChunkStrategy,
    level: Literal["document", "parent", "child"],
    ordinal: int,
    counter: TokenCounter,
    include_heading: bool,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    rendered, prefix_chars = _render_body(spec.body, spec.title_path, include_heading)
    details = dict(spec.metadata)
    details.update(metadata or {})
    details.update(
        {
            "heading_injected": bool(prefix_chars),
            "synthetic_prefix_chars": prefix_chars,
            "source_block_count": len(spec.block_ids),
        }
    )
    return Chunk(
        chunk_id=_chunk_id(document.document_id, strategy, level, ordinal, spec.char_start, spec.char_end),
        document_id=document.document_id,
        source_name=document.source_name,
        text=rendered,
        title_path=spec.title_path,
        strategy=strategy,
        level=level,
        token_count=counter.count(rendered),
        char_start=spec.char_start,
        char_end=spec.char_end,
        block_ids=spec.block_ids,
        parent_id=parent_id,
        source_location=dict(spec.source_location),
        metadata=details,
    )


class Chunker(Protocol):
    strategy: ChunkStrategy

    def chunk(self, document: ParsedDocument) -> ChunkingResult: ...


class _BaseChunker:
    strategy: ChunkStrategy

    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self.config = config
        self.counter = counter

    def _result(self, document: ParsedDocument, chunks: Sequence[Chunk], **diagnostics: Any) -> ChunkingResult:
        heading_count = sum(1 for block in document.blocks if block.kind == "heading")
        values = {
            "strategy": self.strategy,
            "token_counter": self.counter.name,
            "document_id": document.document_id,
            "source_name": document.source_name,
            "block_count": len(document.blocks),
            "heading_count": heading_count,
            "headings_present": heading_count > 0,
            "heading_context_applied": heading_count > 0,
            "hierarchy_available": _has_structural_hierarchy(document),
            "hierarchy_applied": False,
            "chunk_count": len(chunks),
        }
        values.update(diagnostics)
        return ChunkingResult(document, self.strategy, tuple(chunks), values)


class StructureFirstChunker(_BaseChunker):
    """Prefer headings and natural blocks, then pack to a token budget."""

    strategy: ChunkStrategy = "structure"

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        specs = _structure_specs(
            document,
            self.counter,
            max_tokens=self.config.max_tokens,
            include_heading=self.config.include_heading,
        )
        chunks = [
            _make_chunk(
                document,
                spec,
                strategy=self.strategy,
                level="document",
                ordinal=index,
                counter=self.counter,
                include_heading=self.config.include_heading,
            )
            for index, spec in enumerate(specs)
        ]
        return self._result(document, chunks, boundary="heading/block")


class FixedTokenChunker(_BaseChunker):
    """Deterministic whole-document token windows without overlap."""

    strategy: ChunkStrategy = "fixed"

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        spans = _split_text_spans(
            document.text,
            self.config.max_tokens,
            self.counter,
        )
        expanded_spans: list[tuple[str, int, int]] = []
        for body, start, end in spans:
            block_ids = _block_ids_for_span(document, start, end)
            first_block = next((block for block in document.blocks if block.block_id == block_ids[0]), None)
            inject_heading = self.config.include_heading and bool(
                first_block and first_block.title_path and first_block.kind != "heading"
            )
            if inject_heading and first_block is not None:
                prefix_tokens = self.counter.count(_heading_prefix(first_block.title_path, True))
                if self.counter.count(body) + prefix_tokens > self.config.max_tokens:
                    expanded_spans.extend(
                        _split_text_spans(
                            body,
                            max(1, self.config.max_tokens - prefix_tokens),
                            self.counter,
                            base_offset=start,
                        )
                    )
                    continue
            expanded_spans.append((body, start, end))
        spans = expanded_spans
        chunks: list[Chunk] = []
        for index, (body, start, end) in enumerate(spans):
            block_ids = _block_ids_for_span(document, start, end)
            first_block = next((block for block in document.blocks if block.block_id == block_ids[0]), None)
            # Headings are already present in the fixed window.  Only inject a
            # title path when a later window starts in body text and therefore
            # would otherwise lose its section context.
            inject_heading = self.config.include_heading and bool(
                first_block and first_block.title_path and first_block.kind != "heading"
            )
            spec = _ChunkSpec(
                body=body,
                title_path=first_block.title_path if first_block else (),
                char_start=start,
                char_end=end,
                block_ids=block_ids,
                source_location=_source_location_for_blocks(document, block_ids),
            )
            chunks.append(
                _make_chunk(
                    document,
                    spec,
                    strategy=self.strategy,
                    level="document",
                    ordinal=index,
                    counter=self.counter,
                    include_heading=inject_heading,
                )
            )
        return self._result(
            document,
            chunks,
            boundary="token_window",
        )


def _sentence_spans(text: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Small bilingual sentence splitter with offsets and conservative rules."""

    spans: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    for index, char in enumerate(text):
        boundary = char in "。！？!?；;"
        if char in "." and index + 1 < length and text[index + 1].isspace():
            boundary = True
        if char == "\n":
            boundary = True
        if not boundary:
            continue
        end = index + (0 if char == "\n" else 1)
        left, right = _trimmed_span(text, start, end)
        if left < right:
            spans.append((text[left:right], base_offset + left, base_offset + right))
        start = index + 1
    left, right = _trimmed_span(text, start, length)
    if left < right:
        spans.append((text[left:right], base_offset + left, base_offset + right))
    return spans or ([(text.strip(), base_offset, base_offset + len(text))] if text.strip() else [])


def _semantic_units(
    document: ParsedDocument,
    counter: TokenCounter,
    max_tokens: int,
    include_heading: bool,
) -> list[list[_ChunkSpec]]:
    sections: list[list[_ChunkSpec]] = []
    for group in _section_groups(document):
        units: list[_ChunkSpec] = []
        path = group[0].title_path if group else ()
        prefix_tokens = counter.count(_heading_prefix(path, include_heading))
        body_budget = max(1, max_tokens - prefix_tokens)
        for block in group:
            raw_units = (
                [(block.text, block.char_start, block.char_end)]
                if block.kind in {"list", "table", "code"}
                else _sentence_spans(block.text, block.char_start)
            )
            for value, start, end in raw_units:
                if counter.count(value) <= body_budget:
                    units.append(
                        _ChunkSpec(
                            body=value,
                            title_path=block.title_path,
                            char_start=start,
                            char_end=end,
                            block_ids=(block.block_id,),
                            source_location=dict(block.source_location),
                        )
                    )
                    continue
                for split_value, split_start, split_end in _split_text_spans(
                    value,
                    body_budget,
                    counter,
                    base_offset=start,
                ):
                    units.append(
                        _ChunkSpec(
                            body=split_value,
                            title_path=block.title_path,
                            char_start=split_start,
                            char_end=split_end,
                            block_ids=(block.block_id,),
                            source_location=dict(block.source_location),
                            metadata={"split_from_sentence": True},
                        )
                    )
        if units:
            sections.append(units)
    return sections


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class SemanticChunker(_BaseChunker):
    """Group sentence units at embedding-similarity breakpoints."""

    strategy: ChunkStrategy = "semantic"

    def __init__(
        self,
        config: ChunkingConfig,
        counter: TokenCounter,
        embedding_function: EmbeddingFunction,
    ) -> None:
        super().__init__(config, counter)
        self.embedding_function = embedding_function

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        sections = _semantic_units(
            document,
            self.counter,
            self.config.max_tokens,
            self.config.include_heading,
        )
        chunks: list[Chunk] = []
        breakpoints = 0
        section_count = 0
        for units in sections:
            section_count += 1
            vectors = self.embedding_function([unit.body for unit in units])
            if len(vectors) != len(units):
                raise ChunkingConfigurationError("embedding_function 返回的向量数量与句子数量不一致")
            dissimilarities = [
                1.0 - _cosine(vectors[index - 1], vectors[index])
                for index in range(1, len(vectors))
            ]
            threshold = _percentile(
                dissimilarities,
                self.config.semantic_breakpoint_percentile,
            )
            current: list[_ChunkSpec] = []

            def emit() -> None:
                nonlocal current
                if not current:
                    return
                body = "\n\n".join(unit.body for unit in current).strip()
                ids = tuple(dict.fromkeys(block_id for unit in current for block_id in unit.block_ids))
                specs = _ChunkSpec(
                    body=body,
                    title_path=current[0].title_path,
                    char_start=min(unit.char_start for unit in current),
                    char_end=max(unit.char_end for unit in current),
                    block_ids=ids,
                    source_location=_source_location_for_blocks(document, ids),
                    metadata={"semantic_threshold": threshold},
                )
                chunks.append(
                    _make_chunk(
                        document,
                        specs,
                        strategy=self.strategy,
                        level="document",
                        ordinal=len(chunks),
                        counter=self.counter,
                        include_heading=self.config.include_heading,
                    )
                )
                current = []

            for index, unit in enumerate(units):
                candidate = "\n\n".join([*(item.body for item in current), unit.body]).strip()
                candidate_tokens = self.counter.count(
                    _render_body(candidate, unit.title_path, self.config.include_heading)[0]
                )
                if current and candidate_tokens > self.config.max_tokens:
                    emit()
                if current and index > 0 and dissimilarities[index - 1] > threshold:
                    current_tokens = self.counter.count(
                        _render_body(
                            "\n\n".join(item.body for item in current),
                            current[0].title_path,
                            self.config.include_heading,
                        )[0]
                    )
                    if current_tokens >= self.config.min_tokens and dissimilarities[index - 1] > threshold:
                        emit()
                        breakpoints += 1
                current.append(unit)
            emit()
        return self._result(
            document,
            chunks,
            boundary="sentence_embedding_breakpoint",
            semantic_breakpoint_percentile=self.config.semantic_breakpoint_percentile,
            semantic_breakpoints=breakpoints,
            semantic_sections=section_count,
        )


def _units_from_parent_spec(
    spec: _ChunkSpec,
    counter: TokenCounter,
    *,
    max_tokens: int,
    include_heading: bool,
) -> list[_ChunkSpec]:
    """Create sentence-level units inside one structural parent.

    Parent specs can contain several prose blocks, or one special block such
    as a table/list/code block.  Special blocks stay atomic until the hard
    token limit requires a fixed split.
    """

    path = spec.title_path
    prefix_tokens = counter.count(_heading_prefix(path, include_heading))
    body_budget = max(1, max_tokens - prefix_tokens)
    kinds = spec.metadata.get("block_kinds", [])
    keep_atomic = bool(kinds) and all(kind in {"list", "table", "code"} for kind in kinds)
    raw_units = [(spec.body, 0, len(spec.body))] if keep_atomic else _sentence_spans(spec.body, 0)
    units: list[_ChunkSpec] = []

    for value, relative_start, relative_end in raw_units:
        values = (
            [(value, relative_start, relative_end)]
            if counter.count(value) <= body_budget
            else [
                (part, start, end)
                for part, start, end in _split_text_spans(
                    value,
                    body_budget,
                    counter,
                )
            ]
        )
        for part, part_start, part_end in values:
            if len(spec.block_ids) == 1:
                char_start = spec.char_start + part_start
                char_end = spec.char_start + part_end
            else:
                # A structural parent assembled from multiple source blocks
                # has synthetic separators.  Keep its broad source range and
                # retain exact relative positions in metadata.
                char_start = spec.char_start
                char_end = spec.char_end
            metadata = dict(spec.metadata)
            metadata.update(
                {
                    "relative_char_start": part_start,
                    "relative_char_end": part_end,
                    "parent_char_start": spec.char_start,
                    "parent_char_end": spec.char_end,
                }
            )
            units.append(
                _ChunkSpec(
                    body=part,
                    title_path=path,
                    char_start=char_start,
                    char_end=char_end,
                    block_ids=spec.block_ids,
                    source_location=dict(spec.source_location),
                    metadata=metadata,
                )
            )
    return units


def _merge_semantic_units(
    document: ParsedDocument,
    units: Sequence[_ChunkSpec],
    counter: TokenCounter,
    config: ChunkingConfig,
    embedding_function: EmbeddingFunction,
) -> tuple[list[_ChunkSpec], int, float]:
    """Merge units by adjacent embedding similarity under hard Token limits."""

    if not units:
        return [], 0, 1.0
    vectors = embedding_function([unit.body for unit in units])
    if len(vectors) != len(units):
        raise ChunkingConfigurationError("embedding_function 返回的向量数量与原子单元数量不一致")
    dissimilarities = [
        1.0 - _cosine(vectors[index - 1], vectors[index])
        for index in range(1, len(vectors))
    ]
    threshold = _percentile(dissimilarities, config.semantic_breakpoint_percentile)
    merged: list[_ChunkSpec] = []
    current: list[_ChunkSpec] = []
    semantic_breakpoints = 0

    def emit() -> None:
        nonlocal current
        if not current:
            return
        body = "\n\n".join(item.body for item in current).strip()
        block_ids = tuple(dict.fromkeys(block_id for item in current for block_id in item.block_ids))
        metadata: dict[str, Any] = {
            "semantic_threshold": threshold,
            "semantic_unit_count": len(current),
        }
        if len(current) == 1:
            metadata.update(current[0].metadata)
        merged.append(
            _ChunkSpec(
                body=body,
                title_path=current[0].title_path,
                char_start=min(item.char_start for item in current),
                char_end=max(item.char_end for item in current),
                block_ids=block_ids,
                source_location=_source_location_for_blocks(document, block_ids),
                metadata=metadata,
            )
        )
        current = []

    for index, unit in enumerate(units):
        candidate = "\n\n".join([*(item.body for item in current), unit.body]).strip()
        candidate_tokens = counter.count(
            _render_body(candidate, unit.title_path, config.include_heading)[0]
        )
        if current and candidate_tokens > config.max_tokens:
            emit()
        if current and index > 0 and dissimilarities[index - 1] > threshold:
            current_tokens = counter.count(
                _render_body(
                    "\n\n".join(item.body for item in current),
                    current[0].title_path,
                    config.include_heading,
                )[0]
            )
            if current_tokens >= config.min_tokens and dissimilarities[index - 1] > threshold:
                emit()
                semantic_breakpoints += 1
        current.append(unit)
    emit()
    return merged, semantic_breakpoints, threshold


def _has_structural_hierarchy(document: ParsedDocument) -> bool:
    """Return true only when a heading actually scopes a body block."""

    has_heading = any(block.kind == "heading" for block in document.blocks)
    has_scoped_body = any(
        block.kind != "heading" and bool(block.title_path)
        for block in document.blocks
    )
    return has_heading and has_scoped_body


def _hierarchy_reason(document: ParsedDocument) -> str:
    if not any(block.kind == "heading" for block in document.blocks):
        return "未识别到标题/章节，按自然段处理"
    return "识别到标题，但没有标题下的正文层级"


def _flat_semantic_chunks(
    document: ParsedDocument,
    counter: TokenCounter,
    config: ChunkingConfig,
    embedding_function: EmbeddingFunction,
    strategy: ChunkStrategy,
) -> tuple[list[Chunk], int]:
    """Create document-level chunks when no real parent hierarchy is available."""

    sections = _semantic_units(
        document,
        counter,
        config.max_tokens,
        config.include_heading,
    )
    chunks: list[Chunk] = []
    semantic_breakpoints = 0
    for units in sections:
        leaf_specs, breakpoints, _ = _merge_semantic_units(
            document,
            units,
            counter,
            config,
            embedding_function,
        )
        semantic_breakpoints += breakpoints
        for leaf_spec in leaf_specs:
            chunks.append(
                _make_chunk(
                    document,
                    leaf_spec,
                    strategy=strategy,
                    level="document",
                    ordinal=len(chunks),
                    counter=counter,
                    include_heading=config.include_heading,
                    metadata={
                        "role": "flat_retrieval",
                        "hierarchy_applied": False,
                    },
                )
            )
    return chunks, semantic_breakpoints


def _flat_semantic_specs_from_spec(
    document: ParsedDocument,
    spec: _ChunkSpec,
    counter: TokenCounter,
    config: ChunkingConfig,
    embedding_function: EmbeddingFunction,
) -> tuple[list[_ChunkSpec], int]:
    """Flatten one unscoped section without inventing a parent relationship."""

    units = _units_from_parent_spec(
        spec,
        counter,
        max_tokens=config.max_tokens,
        include_heading=config.include_heading,
    )
    leaf_specs, semantic_breakpoints, _ = _merge_semantic_units(
        document,
        units,
        counter,
        config,
        embedding_function,
    )
    return leaf_specs, semantic_breakpoints


class UnifiedChunker(_BaseChunker):
    """The product's fixed pipeline combining all useful chunking ideas."""

    strategy: ChunkStrategy = "unified"

    def __init__(
        self,
        config: ChunkingConfig,
        counter: TokenCounter,
        embedding_function: EmbeddingFunction,
    ) -> None:
        super().__init__(config, counter)
        self.embedding_function = embedding_function

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        headings_present = any(block.kind == "heading" for block in document.blocks)
        if not _has_structural_hierarchy(document):
            chunks, semantic_breakpoints = _flat_semantic_chunks(
                document,
                self.counter,
                self.config,
                self.embedding_function,
                self.strategy,
            )
            return self._result(
                document,
                chunks,
                boundary="paragraph_then_semantic_then_token",
                parent_count=0,
                child_count=0,
                semantic_breakpoints=semantic_breakpoints,
                embedding_provider=type(self.embedding_function).__name__,
                headings_present=headings_present,
                hierarchy_applied=False,
                hierarchy_reason=_hierarchy_reason(document),
            )

        parent_specs = _structure_specs(
            document,
            self.counter,
            max_tokens=self.config.parent_max_tokens,
            include_heading=self.config.include_heading,
        )
        chunks: list[Chunk] = []
        semantic_breakpoints = 0
        unscoped_section_count = 0
        hierarchy_applied = False
        for parent_index, parent_spec in enumerate(parent_specs):
            if not parent_spec.title_path:
                flat_specs, breakpoints = _flat_semantic_specs_from_spec(
                    document,
                    parent_spec,
                    self.counter,
                    self.config,
                    self.embedding_function,
                )
                for flat_spec in flat_specs:
                    chunks.append(
                        _make_chunk(
                            document,
                            flat_spec,
                            strategy=self.strategy,
                            level="document",
                            ordinal=len(chunks),
                            counter=self.counter,
                            include_heading=self.config.include_heading,
                            metadata={
                                "role": "flat_retrieval",
                                "hierarchy_applied": False,
                                "unscoped_section": True,
                            },
                        )
                    )
                semantic_breakpoints += breakpoints
                unscoped_section_count += 1
                continue

            hierarchy_applied = True
            parent = _make_chunk(
                document,
                parent_spec,
                strategy=self.strategy,
                level="parent",
                ordinal=parent_index,
                counter=self.counter,
                include_heading=self.config.include_heading,
                metadata={"role": "context_parent"},
            )
            chunks.append(parent)
            units = _units_from_parent_spec(
                parent_spec,
                self.counter,
                max_tokens=self.config.max_tokens,
                include_heading=self.config.include_heading,
            )
            leaf_specs, breakpoints, _ = _merge_semantic_units(
                document,
                units,
                self.counter,
                self.config,
                self.embedding_function,
            )
            semantic_breakpoints += breakpoints
            for leaf_spec in leaf_specs:
                child = _make_chunk(
                    document,
                    leaf_spec,
                    strategy=self.strategy,
                    level="child",
                    ordinal=len(chunks),
                    counter=self.counter,
                    include_heading=self.config.include_heading,
                    parent_id=parent.chunk_id,
                    metadata={"role": "retrieval_child"},
                )
                chunks.append(child)
        return self._result(
            document,
            chunks,
            boundary="structure_then_semantic_then_token",
            parent_count=sum(1 for chunk in chunks if chunk.level == "parent"),
            child_count=sum(1 for chunk in chunks if chunk.level == "child"),
            semantic_breakpoints=semantic_breakpoints,
            embedding_provider=type(self.embedding_function).__name__,
            headings_present=headings_present,
            hierarchy_applied=hierarchy_applied,
            hierarchy_reason=(
                "已识别标题/章节并生成父子层级"
                if unscoped_section_count == 0
                else "已识别标题/章节；无章节归属的正文按平铺处理"
            ),
            unscoped_section_count=unscoped_section_count,
        )


class ParentChildChunker(_BaseChunker):
    """Build structural parents and non-overlapping child retrieval chunks."""

    strategy: ChunkStrategy = "parent_child"

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        headings_present = any(block.kind == "heading" for block in document.blocks)
        if not _has_structural_hierarchy(document):
            specs = _structure_specs(
                document,
                self.counter,
                max_tokens=self.config.child_max_tokens,
                include_heading=self.config.include_heading,
            )
            chunks = [
                _make_chunk(
                    document,
                    spec,
                    strategy=self.strategy,
                    level="document",
                    ordinal=index,
                    counter=self.counter,
                    include_heading=self.config.include_heading,
                    metadata={
                        "role": "flat_retrieval",
                        "hierarchy_applied": False,
                    },
                )
                for index, spec in enumerate(specs)
            ]
            return self._result(
                document,
                chunks,
                boundary="paragraph_then_child_token_without_hierarchy",
                parent_count=0,
                child_count=0,
                headings_present=headings_present,
                hierarchy_applied=False,
                hierarchy_reason=_hierarchy_reason(document),
            )

        parent_specs = _structure_specs(
            document,
            self.counter,
            max_tokens=self.config.parent_max_tokens,
            include_heading=self.config.include_heading,
        )
        chunks: list[Chunk] = []
        parent_count = 0
        child_count = 0
        unscoped_section_count = 0
        hierarchy_applied = False
        for parent_index, parent_spec in enumerate(parent_specs):
            if not parent_spec.title_path:
                flat_specs = _units_from_parent_spec(
                    parent_spec,
                    self.counter,
                    max_tokens=self.config.child_max_tokens,
                    include_heading=self.config.include_heading,
                )
                for flat_index, flat_spec in enumerate(flat_specs):
                    chunks.append(
                        _make_chunk(
                            document,
                            flat_spec,
                            strategy=self.strategy,
                            level="document",
                            ordinal=len(chunks),
                            counter=self.counter,
                            include_heading=self.config.include_heading,
                            metadata={
                                "role": "flat_retrieval",
                                "hierarchy_applied": False,
                                "unscoped_section": True,
                            },
                        )
                    )
                unscoped_section_count += 1
                continue

            hierarchy_applied = True
            parent = _make_chunk(
                document,
                parent_spec,
                strategy=self.strategy,
                level="parent",
                ordinal=parent_index,
                counter=self.counter,
                include_heading=self.config.include_heading,
                metadata={"parent_max_tokens": self.config.parent_max_tokens},
            )
            chunks.append(parent)
            parent_count += 1
            child_spans = _split_text_spans(
                parent_spec.body,
                max(
                    1,
                    self.config.child_max_tokens
                    - self.counter.count(_heading_prefix(parent_spec.title_path, self.config.include_heading)),
                ),
                self.counter,
            )
            for child_index, (body, relative_start, relative_end) in enumerate(child_spans):
                child_spec = _ChunkSpec(
                    body=body,
                    title_path=parent_spec.title_path,
                    # A parent may contain several source blocks separated by
                    # synthetic blank lines.  Keep the broad source range and
                    # preserve exact child offsets in metadata.
                    char_start=parent_spec.char_start,
                    char_end=parent_spec.char_end,
                    block_ids=parent_spec.block_ids,
                    source_location=dict(parent_spec.source_location),
                    metadata={
                        "relative_char_start": relative_start,
                        "relative_char_end": relative_end,
                        "child_index": child_index,
                    },
                )
                chunks.append(
                    _make_chunk(
                        document,
                        child_spec,
                        strategy=self.strategy,
                        level="child",
                        ordinal=len(chunks),
                        counter=self.counter,
                        include_heading=self.config.include_heading,
                        parent_id=parent.chunk_id,
                        metadata={
                            "child_max_tokens": self.config.child_max_tokens,
                        },
                    )
                )
                child_count += 1
        return self._result(
            document,
            chunks,
            boundary="parent_structure_then_child_window",
            parent_count=parent_count,
            child_count=child_count,
            headings_present=headings_present,
            hierarchy_applied=hierarchy_applied,
            hierarchy_reason=(
                "已识别标题/章节并生成父子层级"
                if unscoped_section_count == 0
                else "已识别标题/章节；无章节归属的正文按平铺处理"
            ),
            unscoped_section_count=unscoped_section_count,
        )


def create_token_counter(config: ChunkingConfig) -> TokenCounter:
    if config.tokenizer_model:
        return HuggingFaceTokenCounter(
            config.tokenizer_model,
            local_files_only=config.tokenizer_local_files_only,
        )
    return ApproximateTokenCounter()


def create_chunker(
    config: ChunkingConfig,
    *,
    token_counter: TokenCounter | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> Chunker:
    """Create one strategy for the active build pipeline."""

    strategy = config.validate()
    counter = token_counter or create_token_counter(config)
    if strategy == "unified":
        provider = embedding_function
        if provider is None and config.embedding_model:
            provider = SentenceTransformerEmbeddingProvider(
                config.embedding_model,
                device=config.embedding_device,
                local_files_only=config.embedding_local_files_only,
            )
        if provider is None:
            raise ChunkingConfigurationError(
                "联合切分必须配置 embedding_model 或注入真实 embedding_function"
            )
        return UnifiedChunker(config, counter, provider)
    if strategy == "structure":
        return StructureFirstChunker(config, counter)
    if strategy == "fixed":
        return FixedTokenChunker(config, counter)
    if strategy == "semantic":
        provider = embedding_function
        if provider is None and config.embedding_model:
            provider = SentenceTransformerEmbeddingProvider(
                config.embedding_model,
                device=config.embedding_device,
                local_files_only=config.embedding_local_files_only,
            )
        if provider is None:
            raise ChunkingConfigurationError(
                "语义切分需要 embedding_model 或注入 embedding_function；不会自动降级为固定切分"
            )
        return SemanticChunker(config, counter, provider)
    return ParentChildChunker(config, counter)


def chunk_document(
    content: bytes,
    filename: str,
    config: ChunkingConfig,
    *,
    token_counter: TokenCounter | None = None,
    embedding_function: EmbeddingFunction | None = None,
    loader: DefaultDocumentLoader | None = None,
) -> ChunkingResult:
    """Load TXT/Markdown/RST/DOCX and chunk it with the selected strategy."""

    document = load_document(content, filename, loader=loader)
    return create_chunker(
        config,
        token_counter=token_counter,
        embedding_function=embedding_function,
    ).chunk(document)


__all__ = [
    "ApproximateTokenCounter",
    "Chunk",
    "ChunkStrategy",
    "Chunker",
    "ChunkingConfig",
    "ChunkingConfigurationError",
    "ChunkingResult",
    "EmbeddingFunction",
    "FixedTokenChunker",
    "HuggingFaceTokenCounter",
    "ParentChildChunker",
    "SemanticChunker",
    "SentenceTransformerEmbeddingProvider",
    "StructureFirstChunker",
    "TokenCounter",
    "UnifiedChunker",
    "chunk_document",
    "create_chunker",
    "create_token_counter",
]
