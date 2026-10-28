"""Durable task persistence without coupling the task manager to SQLite."""

from __future__ import annotations

from backend.app.jobs.manager import TaskRecord
from backend.app.storage.sqlite_store import SQLiteStore


class SQLiteTaskPersistence:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def load(self) -> list[TaskRecord]:
        return [self._from_dict(item) for item in self.store.list_task_records()]

    def get(self, task_id: str) -> TaskRecord | None:
        record = self.store.get_task_record(task_id)
        return self._from_dict(record) if record is not None else None

    def save(self, record: TaskRecord) -> None:
        self.store.upsert_task(record.to_dict())

    @staticmethod
    def _from_dict(value: dict[str, object]) -> TaskRecord:
        return TaskRecord(
            task_id=str(value["task_id"]),
            task_type=str(value["task_type"]),
            status=str(value["status"]),
            progress=float(value["progress"]),
            message=str(value["message"]),
            payload=dict(value.get("payload") or {}),
            result=dict(value["result"]) if isinstance(value.get("result"), dict) else None,
            error_code=str(value["error_code"]) if value.get("error_code") is not None else None,
            error_message=str(value["error_message"]) if value.get("error_message") is not None else None,
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
        )
