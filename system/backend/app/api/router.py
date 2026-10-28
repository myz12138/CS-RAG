"""Versioned API router."""

from fastapi import APIRouter

from backend.app.api.routes import builds, evaluations, projects, queries, sources, system, tasks


api_router = APIRouter()
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(sources.router, tags=["sources"])
api_router.include_router(builds.router, tags=["builds"])
api_router.include_router(tasks.router, tags=["tasks"])
api_router.include_router(queries.router, tags=["queries"])
api_router.include_router(evaluations.router, tags=["evaluations"])
