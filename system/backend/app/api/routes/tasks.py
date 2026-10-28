"""Long-task status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.api.responses import response


router = APIRouter()


@router.get("/tasks")
def list_tasks(request: Request, status: str | None = Query(default=None)):
    tasks = request.app.state.csrag.tasks.list(status)
    return response([task.to_dict() for task in tasks], request, meta={"count": len(tasks)})


@router.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    task = request.app.state.csrag.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return response(task.to_dict(), request)
