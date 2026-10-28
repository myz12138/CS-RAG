"""Application orchestration for version-pinned CS-RAG queries."""

from __future__ import annotations

from typing import Any, Callable

from backend.app.algorithm_adapter.contracts import QueryRequest, QueryResult
from backend.app.algorithm_adapter.reference import ReferenceAlgorithmAdapter
from backend.app.storage.sqlite_store import SQLiteStore


ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]


class QueryExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class QueryService:
    def __init__(self, store: SQLiteStore, algorithm: ReferenceAlgorithmAdapter) -> None:
        self.store = store
        self.algorithm = algorithm

    def run(
        self,
        query_id: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        query = self.store.get_query(query_id)
        self.store.update_query(query_id, status="running", current_step="query_plan", error_code=None, error_message=None)
        self.store.append_query_event(
            query_id,
            step="query_plan",
            status="running",
            message="查询任务已启动",
        )

        streamed_trace: list[dict[str, Any]] = []

        def append_trace(item: dict[str, Any]) -> None:
            self.store.append_query_event(
                query_id, step=str(item.get("step", "query")),
                status=str(item.get("status", "completed")), message=str(item.get("message", "")),
                **{key: item.get(key) for key in (
                    "constraint_id", "variable_id", "pass_index", "branch_id", "input_ref", "output_ref", "reason_code")},
                counts=item.get("counts") or {}, metrics=item.get("metrics") or {}, warnings=item.get("warnings") or [],
            )

        def report(value: float, message: str, details: dict[str, Any] | None = None) -> None:
            details = details or {}
            step = str(details.get("step") or self._step_from_message(message))
            self.store.update_query(query_id, status="running", current_step=step)
            metadata_updates = details.get("metadata")
            if isinstance(metadata_updates, dict) and metadata_updates:
                current = self.store.get_query(query_id)
                metadata = dict(current.metadata)
                metadata.update(metadata_updates)
                self.store.update_query(
                    query_id,
                    status="running",
                    current_step=step,
                    metadata=metadata,
                )
            self.store.append_query_event(
                query_id,
                step=step,
                status=str(details.get("status", "running")),
                message=message,
                input_ref=details.get("input_ref"),
                output_ref=details.get("output_ref"),
                counts=details.get("counts") if isinstance(details.get("counts"), dict) else {},
                metrics={"progress": round(value, 4), **(
                    details.get("metrics") if isinstance(details.get("metrics"), dict) else {}
                )},
                reason_code=details.get("reason_code"),
                warnings=details.get("warnings") if isinstance(details.get("warnings"), list) else [],
            )
            for item in details.get("trace_events") or []:
                append_trace(item)
                streamed_trace.append(item)
            if progress is not None:
                progress(value, message, details)

        request = QueryRequest(
            question=query.question,
            project_id=query.project_id,
            version_id=query.version_id,
            trace_level=query.trace_level,
            query_id=query.query_id,
        )
        try:
            result = self.algorithm.execute(request, progress=report)
        except Exception as exc:
            code = getattr(exc, "code", "QUERY_FAILED")
            message = getattr(exc, "message", str(exc))
            stage = getattr(exc, "stage", None) or self.store.get_query(query_id).current_step or "query_plan"
            stage = {"phase1": "phase1_graph_retrieval", "phase2": "phase2_text_recovery", "evaluation": "answer_generation"}.get(stage, stage)
            self.store.update_query(
                query_id,
                status="failed",
                current_step=stage,
                error_code=code,
                error_message=message,
            )
            self.store.append_query_event(
                query_id,
                step=stage,
                status="failed",
                message=message,
                reason_code=code,
            )
            raise QueryExecutionError(code, message, stage=stage) from exc

        for trace_item in result.trace:
            if trace_item in streamed_trace:
                continue
            self.store.append_query_event(
                query_id,
                step=str(trace_item.get("step", "query")),
                status=str(trace_item.get("status", "completed")),
                message=str(trace_item.get("message", "")),
                counts=trace_item.get("counts") or {},
                metrics=trace_item.get("metrics") or {},
                constraint_id=trace_item.get("constraint_id"),
                variable_id=trace_item.get("variable_id"),
                pass_index=trace_item.get("pass_index"),
                branch_id=trace_item.get("branch_id"),
                output_ref=trace_item.get("output_ref"),
                input_ref=trace_item.get("input_ref"),
                reason_code=trace_item.get("reason_code"),
                warnings=trace_item.get("warnings") or [],
            )

        metadata: dict[str, Any] = dict(result.metadata)
        metadata.update(
            {
                "query_plan": result.query_plan,
                "evidence": result.evidence,
                "trace": result.trace,
            }
        )
        self.store.update_query(
            query_id,
            status="completed",
            current_step="answer_generation",
            answer=result.answer,
            warnings=result.warnings,
            metadata=metadata,
            error_code=None,
            error_message=None,
        )
        return {
            "query_id": query_id,
            "version_id": query.version_id,
            "answer": result.answer,
            "warnings": result.warnings,
            "query_plan": result.query_plan,
            "evidence": result.evidence,
            "trace": result.trace,
            "metadata": result.metadata,
        }

    @staticmethod
    def _step_from_message(message: str) -> str:
        lowered = message.lower()
        if "query_plan" in lowered or "查询计划" in message:
            return "query_plan"
        if "phase1" in lowered:
            return "phase1_graph_retrieval"
        if "phase2" in lowered:
            return "phase2_text_recovery"
        if "answer" in lowered or "答案" in message:
            return "answer_generation"
        return "query"
