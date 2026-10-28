"""Contracts used by the application layer to call CS-RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class QueryRequest:
    question: str
    project_id: str
    version_id: str | None = None
    trace_level: str = "summary"
    query_id: str | None = None


@dataclass
class QueryResult:
    answer: str | None = None
    query_plan: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlgorithmAdapter(Protocol):
    def manifest(self) -> dict[str, Any]: ...

    def query(self, request: QueryRequest) -> QueryResult: ...
