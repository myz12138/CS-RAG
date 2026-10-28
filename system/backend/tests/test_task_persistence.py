from backend.app.jobs.manager import TaskManager
from backend.app.jobs.persistence import SQLiteTaskPersistence
from backend.app.storage.sqlite_store import SQLiteStore


def test_task_history_survives_manager_recreation(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    persistence = SQLiteTaskPersistence(store)

    first_manager = TaskManager(persistence)
    completed = first_manager.create("demo", {"name": "完成任务"})
    first_manager.update(
        completed.task_id,
        status="completed",
        progress=1.0,
        message="任务完成",
        result={"ok": True},
    )

    second_manager = TaskManager(persistence)
    restored = second_manager.get(completed.task_id)

    assert restored is not None
    assert restored.status == "completed"
    assert restored.result == {"ok": True}


def test_running_task_is_marked_interrupted_after_restart(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    persistence = SQLiteTaskPersistence(store)

    first_manager = TaskManager(persistence)
    running = first_manager.create("demo")
    first_manager.update(running.task_id, status="running", message="处理中")

    second_manager = TaskManager(persistence)
    restored = second_manager.get(running.task_id)

    assert restored is not None
    assert restored.status == "failed"
    assert restored.error_code == "SERVICE_RESTARTED"
