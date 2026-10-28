"""Health and capability endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.api.responses import response
from backend.app.ingestion.policy import TEXT_EXTENSIONS
from backend.app.model_gateway.base import ModelGatewayError, ModelNotConfiguredError


router = APIRouter()


class ModelConfigUpdate(BaseModel):
    base_url: str = Field(default="", max_length=500)
    model_name: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    timeout_seconds: float = Field(default=90.0, ge=1.0, le=3600.0)
    embedding_base_url: str = Field(default="", max_length=500)
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1, max_length=200)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=2000)
    embedding_api_key: str | None = Field(default=None, max_length=2000)


class BuildConfigUpdate(BaseModel):
    max_tokens: int = Field(default=512, ge=1, le=65536)
    min_tokens: int = Field(default=80, ge=0, le=65536)
    parent_max_tokens: int = Field(default=1536, ge=1, le=131072)
    include_heading: bool = True
    semantic_breakpoint_percentile: float = Field(default=85.0, ge=0.0, le=100.0)
    tokenizer_model: str = Field(default="", max_length=500)
    embedding_device: str = Field(default="cpu", min_length=1, max_length=32)
    text_index_batch_size: int = Field(default=32, ge=1, le=4096)


class QueryConfigUpdate(BaseModel):
    planner_max_retries: int = Field(default=3, ge=1, le=10)
    top_n_llm: int = Field(default=3, ge=1, le=100)
    top_k_entity: int = Field(default=3, ge=1, le=100)
    relation_k: int = Field(default=8, ge=1, le=1000)
    neff_threshold: float = Field(default=1.5, gt=0.0, le=1000.0)
    phase1_rerank_max_length: int = Field(default=128, ge=32, le=4096)
    phase2_rerank_max_length: int = Field(default=256, ge=32, le=4096)
    rerank_batch: int = Field(default=32, ge=1, le=4096)
    rerank_fp16: bool = False
    include_question_in_rerank: bool = True
    unresolved_top_similar: int = Field(default=20, ge=1, le=1000)
    top_k_unresolved: int = Field(default=3, ge=1, le=100)
    focus_max_per_variable: int = Field(default=5, ge=1, le=100)
    text_index_recall_multiplier: int = Field(default=4, ge=1, le=100)
    prompt_max_triples: int = Field(default=20, ge=1, le=1000)
    prompt_max_text_per_triple: int = Field(default=8, ge=1, le=100)


@router.get("/health")
def health(request: Request):
    state = request.app.state.csrag
    return response(
        {
            "status": "ok",
            "service": state.settings.app_name,
            "version": state.settings.app_version,
            "deployment_mode": state.settings.deployment_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        request,
    )


@router.get("/capabilities")
def capabilities(request: Request):
    state = request.app.state.csrag
    return response(
        {
            "api_version": "v1",
            "deployment_mode": state.settings.deployment_mode,
            "input_formats": {
                "text": [f".{extension}" for extension in sorted(TEXT_EXTENSIONS)],
                "knowledge_graph": [],
            },
            "query_trace_levels": ["summary", "detailed"],
            "query_config": state.query_config(),
            "algorithm": state.algorithm.manifest(),
            "model_gateway": state.model_gateway.capabilities(),
            "vector_store": state.build_service.vector_store_summary(),
            "build_config": state.build_config(),
        },
        request,
    )


@router.get("/model-config")
def model_config(request: Request):
    return response(request.app.state.csrag.model_config(), request)


@router.put("/model-config")
def update_model_config(payload: ModelConfigUpdate, request: Request):
    state = request.app.state.csrag
    try:
        config = state.update_model_config(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return response(config, request)


@router.get("/build-config")
def build_config(request: Request):
    return response(request.app.state.csrag.build_config(), request)


@router.get("/query-config")
def query_config(request: Request):
    return response(request.app.state.csrag.query_config(), request)


@router.put("/query-config")
def update_query_config(payload: QueryConfigUpdate, request: Request):
    state = request.app.state.csrag
    try:
        config = state.update_query_config(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return response(config, request)


@router.put("/build-config")
def update_build_config(payload: BuildConfigUpdate, request: Request):
    state = request.app.state.csrag
    try:
        config = state.update_build_config(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return response(config, request)


@router.post("/model-check")
def model_check(request: Request):
    """Call the configured real LLM endpoint; never return a simulated success."""

    state = request.app.state.csrag
    try:
        result = state.model_gateway.check_llm_connection()
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return response(result, request)
