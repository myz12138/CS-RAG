"""Knowledge-base build planning endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from backend.app.api.responses import response
from backend.app.indexing.text_index import read_text_index_manifest
from backend.app.storage.file_store import FileStoreError
from backend.app.storage.sqlite_store import ProjectNotFoundError


router = APIRouter()


class BuildPreviewRequest(BaseModel):
    mode: Literal["auto", "supplement_extract"] = "auto"
    allow_graph_only: bool = False


class BuildRequest(BuildPreviewRequest):
    pass


@router.post("/projects/{project_id}/builds/preview")
def preview_build(project_id: str, payload: BuildPreviewRequest, request: Request):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc

    sources = state.store.list_sources(project_id)
    text_sources = [source for source in sources if source.source_kind == "text"]
    kg_sources = [source for source in sources if source.source_kind == "kg"]
    chunking_plan = state.build_service.chunking_plan()
    # New knowledge bases always start from text.  ``auto`` is kept as the
    # friendly UI value, but it has one deterministic meaning now.
    effective_mode = "supplement_extract"

    blockers: list[dict[str, str]] = []
    actions: list[str] = []
    if payload.allow_graph_only:
        overall_status = "blocked_graph_only_disabled"
        blockers.append(
            {
                "code": "GRAPH_ONLY_DISABLED",
                "message": "当前版本只支持从文本构建，不能创建仅图谱知识库",
            }
        )
        actions.append("如需使用已有图谱，请在多跳问答页选择已有知识库版本")
    elif not sources:
        overall_status = "blocked_no_input"
        blockers.append(
            {
                "code": "NO_INPUT",
                "message": "没有新的文本输入；如需直接使用已有知识库，请到多跳问答选择已发布版本",
            }
        )
        actions.append("上传 txt、md、rst 或 docx 文本后构建新版本")
    elif kg_sources:
        overall_status = "blocked_kg_input"
        blockers.append(
            {
                "code": "KG_INPUT_DISABLED",
                "message": "当前只允许从文本构建；检测到 KG 文件来源，请移除该来源或新建项目",
            }
        )
        actions.append("已有知识库请在多跳问答页选择版本，不要作为新输入重新构建")
    elif not text_sources:
        overall_status = "blocked_no_input"
        blockers.append({"code": "TEXT_REQUIRED", "message": "知识库构建必须至少包含一个文本来源"})
    else:
        needs_llm = True
        model_ready = bool(state.model_gateway.capabilities().get("configured_base_url"))
        if not chunking_plan["embedding_ready"]:
            overall_status = "blocked_embedding_not_available"
            blockers.append(
                {
                    "code": "EMBEDDING_NOT_AVAILABLE",
                    "message": "联合切分需要本地 embedding 模型；当前项目模型目录中未找到可用权重",
                }
            )
            actions.append("将 embedding 模型放入 models/embedding 后重试；系统不会降级为无语义切分")
        elif needs_llm and not model_ready:
            overall_status = "blocked_model_not_configured"
            blockers.append(
                {
                    "code": "MODEL_NOT_CONFIGURED",
                    "message": "当前构建需要真实 LLM 抽取，但模型服务地址尚未配置",
                }
            )
            actions.append("在设置中配置模型服务地址后重试；系统不会使用伪造抽取结果")
        else:
            overall_status = "ready_for_build"
            actions.append("文本将保留为文本条目，并调用真实 LLM 抽取 KG")

    return response(
        {
            "project_id": project_id,
            "overall_status": overall_status,
            "mode": effective_mode,
            "allow_graph_only": payload.allow_graph_only,
            "text_layer": {
                "status": "provided" if text_sources else "not_provided",
                "source_count": len(text_sources),
                "source_ids": [source.source_id for source in text_sources],
            },
            "kg_layer": {
                "status": "derived" if text_sources and not kg_sources else "not_provided",
                "source_count": 0,
                "source_ids": [],
            },
            "expected_llm_calls": None
            if effective_mode == "supplement_extract" and text_sources
            else 0,
            "expected_llm_calls_basis": "按联合切分后生成的检索子块数量确定"
            if effective_mode == "supplement_extract" and text_sources
            else "当前模式不调用 LLM 抽取",
            "chunking": chunking_plan,
            "source_snapshot": [source.source_id for source in sources],
            "blockers": blockers,
            "actions": actions,
        },
        request,
    )


@router.post("/projects/{project_id}/builds", status_code=status.HTTP_202_ACCEPTED)
def create_build(project_id: str, payload: BuildRequest, request: Request):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc

    task = state.tasks.create(
        "knowledge_base_build",
        {
            "project_id": project_id,
            "mode": payload.mode,
            "allow_graph_only": payload.allow_graph_only,
        },
    )

    def run_build() -> dict[str, object]:
        return state.build_service.build(
            project_id,
            mode=payload.mode,
            allow_graph_only=payload.allow_graph_only,
            progress=lambda progress_value, message: state.tasks.update(
                task.task_id,
                progress=progress_value,
                message=message,
            ),
        )

    state.tasks.start(task.task_id, run_build)
    return response(
        {
            "task_id": task.task_id,
            "status": task.status,
            "project_id": project_id,
        },
        request,
    )


@router.get("/projects/{project_id}/versions")
def list_versions(project_id: str, request: Request):
    state = request.app.state.csrag
    try:
        state.store.get_project(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail="项目不存在") from exc
    versions = [version.to_dict() for version in state.store.list_versions(project_id)]
    return response(versions, request, meta={"count": len(versions)})


@router.get("/projects/{project_id}/versions/{version_id}")
def get_version(project_id: str, version_id: str, request: Request):
    state = request.app.state.csrag
    try:
        version = state.store.get_version(version_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="知识库版本不存在") from exc
    if version.project_id != project_id:
        raise HTTPException(status_code=404, detail="知识库版本不存在")
    try:
        text_index = read_text_index_manifest(
            state.files.get_version_dir(project_id, version_id) / "indexes" / "text"
        )
    except FileStoreError:
        text_index = None

    return response(
        {
            "version": version.to_dict(),
            "counts": state.store.version_counts(version_id),
            "conflicts": state.store.list_kg_conflicts(version_id),
            "text_index": text_index or {"status": "not_built"},
            "vector_index": state.store.get_vector_index(version_id) or {"status": "not_registered"},
        },
        request,
    )
