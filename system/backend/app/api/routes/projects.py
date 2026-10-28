"""Project metadata endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from backend.app.api.responses import response
from backend.app.application.version_reader import attach_graph_entities, browse_version_items
from backend.app.storage.sqlite_store import ProjectNotFoundError, VersionNotFoundError


router = APIRouter()


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


@router.get("")
def list_projects(request: Request):
    state = request.app.state.csrag
    projects = [project.to_dict() for project in state.store.list_projects()]
    return response(projects, request, meta={"count": len(projects)})


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreateRequest, request: Request):
    state = request.app.state.csrag
    project = state.store.create_project(payload.name.strip(), payload.description.strip())
    return response(project.to_dict(), request)


@router.get("/{project_id}")
def get_project(project_id: str, request: Request):
    state = request.app.state.csrag
    try:
        project = state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    return response(project.to_dict(), request)


@router.get("/{project_id}/versions/{version_id}/contents")
def get_version_contents(
    project_id: str,
    version_id: str,
    request: Request,
    kind: Literal["text", "kg"] = Query(default="text"),
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default="", max_length=500),
):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
        version = state.store.get_version(version_id)
    except (ProjectNotFoundError, VersionNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="项目或知识库版本不存在") from exc
    if version.project_id != project_id:
        raise HTTPException(status_code=404, detail="知识库版本不存在")
    if kind == "text":
        all_items = state.store.list_text_entries(version_id)
    else:
        all_items = state.store.list_kg_triples(version_id)
    page = browse_version_items(all_items, version.source_snapshot, kind=kind, offset=offset, limit=limit, search=search)
    if kind == "kg":
        names = [str(record[side]) for record in page["items"] for side in ("head", "tail")]
        page["items"] = attach_graph_entities(page["items"], state.store.resolve_graph_entities(version_id, names))
    return response(
        {
            "project_id": project_id,
            "version_id": version_id,
            "kind": kind,
            **page,
        },
        request,
    )
