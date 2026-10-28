"""Project-scoped raw file storage."""

from __future__ import annotations

import shutil
from pathlib import Path


class FileStoreError(ValueError):
    pass


class ProjectFileStore:
    """Store uploaded files below the configured project root only."""

    def __init__(self, projects_root: Path) -> None:
        self.projects_root = projects_root.resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_project_id(project_id: str) -> None:
        candidate = Path(project_id)
        if not project_id or project_id in {".", ".."} or candidate.name != project_id:
            raise FileStoreError("非法项目标识")

    def _source_dir(self, project_id: str) -> Path:
        self._validate_project_id(project_id)
        directory = (self.projects_root / project_id / "sources").resolve()
        if self.projects_root not in directory.parents:
            raise FileStoreError("项目文件路径越界")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _project_dir(self, project_id: str) -> Path:
        self._validate_project_id(project_id)
        directory = (self.projects_root / project_id).resolve()
        if self.projects_root not in directory.parents:
            raise FileStoreError("项目文件路径越界")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_source(self, project_id: str, source_id: str, source_format: str, content: bytes) -> str:
        source_dir = self._source_dir(project_id)
        safe_format = source_format.strip().lower().lstrip(".")
        if not safe_format.isalnum():
            raise FileStoreError("非法文件格式")
        target = source_dir / f"{source_id}.{safe_format}"
        temporary = source_dir / f".{source_id}.{safe_format}.part"
        temporary.write_bytes(content)
        temporary.replace(target)
        return f"sources/{target.name}"

    def remove_source(self, project_id: str, stored_path: str) -> None:
        source_dir = self._source_dir(project_id)
        relative = Path(stored_path)
        target = (source_dir.parent / relative).resolve()
        if source_dir not in target.parents:
            raise FileStoreError("源文件路径越界")
        if target.exists():
            target.unlink()

    def read_source(self, project_id: str, stored_path: str) -> bytes:
        project_dir = self._project_dir(project_id)
        target = (project_dir / Path(stored_path)).resolve()
        if project_dir not in target.parents:
            raise FileStoreError("源文件路径越界")
        if not target.is_file():
            raise FileStoreError("源文件不存在")
        return target.read_bytes()

    def create_version_workspace(self, project_id: str, version_id: str) -> Path:
        project_dir = self._project_dir(project_id)
        versions_dir = project_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        workspace = (versions_dir / f".{version_id}.building").resolve()
        if versions_dir not in workspace.parents:
            raise FileStoreError("版本工作目录越界")
        if workspace.exists():
            raise FileStoreError("版本工作目录已存在")
        workspace.mkdir(parents=True)
        for child in ("documents", "graph", "indexes", "reports"):
            (workspace / child).mkdir()
        return workspace

    def publish_version_workspace(self, project_id: str, version_id: str) -> Path:
        project_dir = self._project_dir(project_id)
        versions_dir = project_dir / "versions"
        workspace = (versions_dir / f".{version_id}.building").resolve()
        target = (versions_dir / version_id).resolve()
        if versions_dir not in workspace.parents or versions_dir not in target.parents:
            raise FileStoreError("版本目录越界")
        if not workspace.is_dir():
            raise FileStoreError("版本工作目录不存在")
        if target.exists():
            raise FileStoreError("版本目标目录已存在")
        workspace.replace(target)
        return target

    def discard_version_workspace(self, project_id: str, version_id: str) -> None:
        project_dir = self._project_dir(project_id)
        workspace = (project_dir / "versions" / f".{version_id}.building").resolve()
        versions_dir = (project_dir / "versions").resolve()
        if versions_dir not in workspace.parents:
            raise FileStoreError("版本目录越界")
        if workspace.exists():
            shutil.rmtree(workspace)

    def get_version_dir(self, project_id: str, version_id: str) -> Path:
        project_dir = self._project_dir(project_id)
        versions_dir = (project_dir / "versions").resolve()
        target = (versions_dir / version_id).resolve()
        if versions_dir not in target.parents or target.name != version_id:
            raise FileStoreError("版本路径越界")
        if not target.is_dir():
            raise FileStoreError("知识库版本目录不存在")
        return target

    def create_query_workspace(self, project_id: str, version_id: str, query_id: str) -> Path:
        version_dir = self.get_version_dir(project_id, version_id)
        queries_dir = (version_dir / "queries").resolve()
        workspace = (queries_dir / query_id).resolve()
        if queries_dir not in workspace.parents or workspace.name != query_id:
            raise FileStoreError("查询工作目录越界")
        queries_dir.mkdir(parents=True, exist_ok=True)
        if workspace.exists():
            raise FileStoreError("查询工作目录已存在")
        workspace.mkdir()
        return workspace
