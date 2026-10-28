"""Project input/source endpoints."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from backend.app.api.responses import response
from backend.app.ingestion.policy import UnsupportedSourceFormatError, classify_source
from backend.app.storage.sqlite_store import ProjectNotFoundError


router = APIRouter()


@router.get("/projects/{project_id}/sources")
def list_sources(project_id: str, request: Request):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    sources = [source.to_dict() for source in state.store.list_sources(project_id)]
    return response(sources, request, meta={"count": len(sources)})


@router.post("/projects/{project_id}/sources", status_code=status.HTTP_201_CREATED)
async def upload_source(
    project_id: str,
    request: Request,
    file: UploadFile = File(...),
    source_kind: str = Form(default="auto"),
    display_name: str | None = Form(default=None),
):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc

    original_name = (display_name or file.filename or "").strip()
    if not original_name:
        raise HTTPException(status_code=400, detail="必须提供文件名")
    try:
        classified = classify_source(original_name, source_kind)
    except UnsupportedSourceFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    content = await file.read()
    if len(content) > state.settings.max_source_size_bytes:
        limit_mb = state.settings.max_source_size_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件超过当前限制（{limit_mb} MB）")
    checksum = hashlib.sha256(content).hexdigest()
    existing = state.store.find_source_by_checksum(project_id, checksum)
    if existing is not None:
        return response(
            {
                "accepted": False,
                "duplicate": True,
                "source": existing.to_dict(),
                "message": "内容摘要一致，未重复录入",
            },
            request,
        )

    source, duplicate = state.store.create_source(
        project_id=project_id,
        original_name=original_name,
        stored_path=f"sources/pending.{classified.source_format}",
        source_kind=classified.source_kind,
        source_format=classified.source_format,
        mime_type=file.content_type,
        checksum_sha256=checksum,
        size_bytes=len(content),
    )
    if duplicate:
        return response(
            {
                "accepted": False,
                "duplicate": True,
                "source": source.to_dict(),
                "message": "内容摘要一致，未重复录入",
            },
            request,
        )

    try:
        stored_path = state.files.save_source(
            project_id, source.source_id, classified.source_format, content
        )
        if stored_path != source.stored_path:
            state.store.update_source_path(source.source_id, stored_path)
            source = state.store.get_source(source.source_id)
    except Exception as exc:
        state.store.delete_source(source.source_id)
        raise HTTPException(status_code=500, detail="保存源文件失败") from exc

    return response(
        {
            "accepted": True,
            "duplicate": False,
            "source": source.to_dict(),
            "message": "源文件已保存，等待解析",
        },
        request,
    )
