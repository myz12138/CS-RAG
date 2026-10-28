"""Small domain models shared by the API and storage layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    description: str
    status: str
    created_at: datetime
    updated_at: datetime
    current_version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_version_id": self.current_version_id,
        }


@dataclass(frozen=True)
class Source:
    """One immutable uploaded input file in a project."""

    source_id: str
    project_id: str
    original_name: str
    stored_path: str
    source_kind: str
    source_format: str
    mime_type: str | None
    checksum_sha256: str
    size_bytes: int
    parse_status: str
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "project_id": self.project_id,
            "original_name": self.original_name,
            "stored_path": self.stored_path,
            "source_kind": self.source_kind,
            "source_format": self.source_format,
            "mime_type": self.mime_type,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
            "parse_status": self.parse_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class KnowledgeBaseVersion:
    version_id: str
    project_id: str
    version_no: int
    mode: str
    status: str
    text_status: str
    kg_status: str
    source_snapshot: tuple[str, ...]
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "project_id": self.project_id,
            "version_no": self.version_no,
            "mode": self.mode,
            "status": self.status,
            "text_status": self.text_status,
            "kg_status": self.kg_status,
            "source_snapshot": list(self.source_snapshot),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class TextEntry:
    entry_id: str
    version_id: str
    source_id: str
    ordinal: int
    title: str
    content: str
    content_hash: str
    source_location: dict[str, Any]
    parent_id: str | None = None
    kg_triple_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "version_id": self.version_id,
            "source_id": self.source_id,
            "ordinal": self.ordinal,
            "title": self.title,
            "content": self.content,
            "content_hash": self.content_hash,
            "source_location": self.source_location,
            "parent_id": self.parent_id,
            "kg_triple_ids": list(self.kg_triple_ids),
        }


@dataclass(frozen=True)
class KGTriple:
    triple_id: str
    version_id: str
    source_id: str
    source_entry_id: str | None
    source_line: int | None
    head: str
    relation: str
    tail: str
    origin: str
    evidence: str | None
    triple_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "triple_id": self.triple_id,
            "version_id": self.version_id,
            "source_id": self.source_id,
            "source_entry_id": self.source_entry_id,
            "source_line": self.source_line,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "origin": self.origin,
            "evidence": self.evidence,
            "triple_hash": self.triple_hash,
        }


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    project_id: str
    version_id: str
    question: str
    trace_level: str
    status: str
    current_step: str | None
    answer: str | None
    warnings: tuple[str, ...]
    metadata: dict[str, Any]
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "project_id": self.project_id,
            "version_id": self.version_id,
            "question": self.question,
            "trace_level": self.trace_level,
            "status": self.status,
            "current_step": self.current_step,
            "answer": self.answer,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class QueryEvent:
    query_id: str
    seq: int
    step: str
    status: str
    message: str
    constraint_id: str | None
    variable_id: str | None
    pass_index: int | None
    branch_id: str | None
    input_ref: str | None
    output_ref: str | None
    counts: dict[str, Any]
    metrics: dict[str, Any]
    reason_code: str | None
    warnings: tuple[str, ...]
    started_at: datetime
    finished_at: datetime | None
    elapsed_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "seq": self.seq,
            "step": self.step,
            "status": self.status,
            "message": self.message,
            "constraint_id": self.constraint_id,
            "variable_id": self.variable_id,
            "pass_index": self.pass_index,
            "branch_id": self.branch_id,
            "input_ref": self.input_ref,
            "output_ref": self.output_ref,
            "counts": self.counts,
            "metrics": self.metrics,
            "reason_code": self.reason_code,
            "warnings": list(self.warnings),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "elapsed_ms": self.elapsed_ms,
        }
