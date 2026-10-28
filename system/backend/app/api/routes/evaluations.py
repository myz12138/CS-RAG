"""Read-only evaluation summaries derived from real query runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Query

from backend.app.api.responses import response
from backend.app.storage.sqlite_store import ProjectNotFoundError


router = APIRouter()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


@router.get("/projects/{project_id}/evaluations")
def evaluation_summary(
    project_id: str,
    request: Request,
    version_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    state = request.app.state.csrag
    try:
        project = state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    selected_version_id = version_id or project.current_version_id
    records = state.store.list_queries(project_id, version_id=selected_version_id, limit=limit)
    completed = [item for item in records if item.status == "completed"]
    metric_values: dict[str, list[float]] = {}
    recent: list[dict[str, Any]] = []
    for item in records:
        metrics = item.metadata.get("evaluation_metrics") or {}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                numeric = _number(value)
                if numeric is not None:
                    metric_values.setdefault(str(key), []).append(numeric)
        recent.append(
            {
                "query_id": item.query_id,
                "version_id": item.version_id,
                "question": item.question,
                "status": item.status,
                "answer": item.answer,
                "metrics": metrics if isinstance(metrics, dict) else {},
                "error_code": item.error_code,
                "updated_at": item.updated_at.isoformat(),
            }
        )
    aggregate = {
        key: round(sum(values) / len(values), 4)
        for key, values in metric_values.items()
        if values
    }
    return response(
        {
            "project_id": project_id,
            "version_id": selected_version_id,
            "sample_count": len(records),
            "completed_count": len(completed),
            "failed_count": sum(1 for item in records if item.status == "failed"),
            "aggregate": aggregate,
            "metric_notes": {
                "resolved_triple_ratio": "已恢复约束占比",
                "answer_supported_by_evidence": "答案是否被检索证据支持（1/0）",
                "unsupported_answer_flag": "算法对未被证据支持答案的告警（1/0）",
                "em": "只有输入标准答案时才有意义；当前查询不自动推算",
                "f1": "只有输入标准答案时才有意义；当前查询不自动推算",
            },
            "recent": recent,
        },
        request,
    )
