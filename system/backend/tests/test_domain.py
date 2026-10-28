from datetime import datetime, timezone

from backend.app.domain.models import Project


def test_project_serialization_is_stable():
    project = Project(
        project_id="p-1",
        name="测试项目",
        description="",
        status="empty",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    payload = project.to_dict()

    assert payload["project_id"] == "p-1"
    assert payload["name"] == "测试项目"
    assert payload["current_version_id"] is None
