"""Query submission and trace APIs."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from backend.app.api.responses import response
from backend.app.storage.sqlite_store import ProjectNotFoundError, VersionNotFoundError


router = APIRouter()


class QueryCreateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    version_id: str | None = None
    trace_level: Literal["summary", "detailed"] = "summary"


def _query_payload(record):
    payload = record.to_dict()
    payload["query_plan"] = record.metadata.get("query_plan", [])
    payload["evidence"] = record.metadata.get("evidence", [])
    payload["trace"] = record.metadata.get("trace", [])
    payload["artifacts"] = record.metadata.get("artifacts", {})
    payload["phase1_debug"] = record.metadata.get("phase1_debug")
    return payload


@router.post("/projects/{project_id}/queries", status_code=status.HTTP_202_ACCEPTED)
def create_query(project_id: str, payload: QueryCreateRequest, request: Request):
    state = request.app.state.csrag
    try:
        project = state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    version_id = payload.version_id or project.current_version_id
    if not version_id:
        raise HTTPException(status_code=409, detail="项目还没有可查询的知识库版本")
    try:
        version = state.store.get_version(version_id)
    except VersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="知识库版本不存在") from exc
    if version.project_id != project_id:
        raise HTTPException(status_code=404, detail="知识库版本不存在")
    if version.status != "ready":
        raise HTTPException(status_code=409, detail="当前版本不是完整可查询版本")

    query_record = state.store.create_query(
        project_id,
        version_id,
        payload.question.strip(),
        payload.trace_level,
    )
    task = state.tasks.create(
        "query",
        {
            "query_id": query_record.query_id,
            "project_id": project_id,
            "version_id": version_id,
        },
    )

    def run_query():
        return state.query_service.run(
            query_record.query_id,
            progress=lambda progress_value, message, _details=None: state.tasks.update(
                task.task_id,
                progress=progress_value,
                message=message,
            ),
        )

    state.tasks.start(task.task_id, run_query)
    return response(
        {
            "query_id": query_record.query_id,
            "task_id": task.task_id,
            "version_id": version_id,
            "status": query_record.status,
        },
        request,
    )


@router.get("/queries/{query_id}")
def get_query(query_id: str, request: Request):
    try:
        record = request.app.state.csrag.store.get_query(query_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="查询不存在") from exc
    return response(_query_payload(record), request)


@router.get("/queries/{query_id}/trace")
def get_query_trace(
    query_id: str,
    request: Request,
    after_seq: int = Query(default=0, ge=0),
):
    try:
        request.app.state.csrag.store.get_query(query_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="查询不存在") from exc
    events = request.app.state.csrag.store.list_query_events(query_id, after_seq)
    return response(
        [event.to_dict() for event in events],
        request,
        meta={"after_seq": after_seq, "next_seq": events[-1].seq if events else after_seq},
    )


@router.get("/queries/{query_id}/constraints/{constraint_id}")
def get_query_constraint(query_id: str, constraint_id: str, request: Request):
    try:
        record = request.app.state.csrag.store.get_query(query_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="查询不存在") from exc
    plan = record.metadata.get("query_plan", [])
    evidence = record.metadata.get("evidence", [])
    phase1_debug = record.metadata.get("phase1_debug") or {}
    try:
        index = int(constraint_id.removeprefix("T"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="约束 ID 格式应为 T0、T1…") from exc
    selected_plan = plan[index] if 0 <= index < len(plan) else None
    selected_debug = [
        item
        for item in phase1_debug.get("triples_debug", [])
        if int(item.get("query_triple_index", -1)) == index
    ]
    selected_evidence = [
        item for item in evidence if int(item.get("query_triple_index", -1)) == index
    ]
    if selected_plan is None and not selected_evidence and not selected_debug:
        raise HTTPException(status_code=404, detail="约束不存在")
    return response(
        {
            "constraint_id": f"T{index}",
            "query_triple": selected_plan,
            "phase1_debug": selected_debug,
            "evidence": selected_evidence,
        },
        request,
    )


@router.get("/queries/{query_id}/evidence/{evidence_id}")
def get_query_evidence(query_id: str, evidence_id: str, request: Request):
    try:
        record = request.app.state.csrag.store.get_query(query_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="查询不存在") from exc
    evidence: list[dict] = []
    for constraint in record.metadata.get("evidence", []):
        for index, item in enumerate(constraint.get("candidate_evidences", []) or []):
            current_id = f"T{constraint.get('query_triple_index', 0)}-E{index}"
            if current_id == evidence_id:
                evidence.append(
                    {
                        "evidence_id": current_id,
                        "constraint": constraint,
                        "item": item,
                    }
                )
    if not evidence:
        raise HTTPException(status_code=404, detail="证据不存在")
    return response(evidence[0], request)
