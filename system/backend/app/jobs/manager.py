"""Small task registry with an optional durable persistence boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Protocol
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    task_id: str
    task_type: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "payload": self.payload,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskPersistence(Protocol):
    def load(self) -> list[TaskRecord]: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def save(self, record: TaskRecord) -> None: ...


class TaskManager:
    def __init__(self, persistence: TaskPersistence | None = None) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = Lock()
        self._persistence = persistence
        if persistence is not None:
            for record in persistence.load():
                if record.status in {"pending", "running"}:
                    record.status = "failed"
                    record.message = "服务重启，任务未完成"
                    record.error_code = "SERVICE_RESTARTED"
                    record.error_message = "服务在任务完成前重启，未自动重放任务"
                    record.updated_at = _now()
                    persistence.save(record)
                self._tasks[record.task_id] = record

    def create(self, task_type: str, payload: dict[str, Any] | None = None) -> TaskRecord:
        record = TaskRecord(task_id=str(uuid4()), task_type=task_type, payload=payload or {})
        with self._lock:
            self._tasks[record.task_id] = record
            if self._persistence is not None:
                self._persistence.save(record)
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None and self._persistence is not None:
                record = self._persistence.get(task_id)
                if record is not None:
                    self._tasks[task_id] = record
            return record

    def list(self, status: str | None = None) -> list[TaskRecord]:
        with self._lock:
            records = list(self._tasks.values())
            if status:
                records = [record for record in records if record.status == status]
            return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def update(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> TaskRecord:
        with self._lock:
            record = self._tasks[task_id]
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = progress
            if message is not None:
                record.message = message
            if result is not None:
                record.result = result
            if error_code is not None:
                record.error_code = error_code
            if error_message is not None:
                record.error_message = error_message
            record.updated_at = _now()
            if self._persistence is not None:
                self._persistence.save(record)
            return record

    def start(self, task_id: str, operation: Callable[[], dict[str, Any]]) -> None:
        from threading import Thread

        def runner() -> None:
            self.update(task_id, status="running", message="任务开始执行")
            try:
                result = operation()
            except Exception as exc:  # The API exposes the structured task error.
                error_code = getattr(exc, "code", "TASK_FAILED")
                error_message = getattr(exc, "message", str(exc))
                self.update(
                    task_id,
                    status="failed",
                    message="任务失败",
                    error_code=error_code,
                    error_message=error_message,
                )
            else:
                self.update(
                    task_id,
                    status="completed",
                    progress=1.0,
                    message="任务完成",
                    result=result,
                )

        Thread(target=runner, name=f"csrag-task-{task_id[:8]}", daemon=True).start()
