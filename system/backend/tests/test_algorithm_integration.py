import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from backend.app.config import REPO_ROOT, Settings
from backend.app.algorithm_adapter.integration import STAGE_SCRIPTS, missing_entrypoints
from backend.app.algorithm_adapter.reference_runner import ReferenceQueryRunner
from backend.app.model_gateway.base import ModelGateway, ModelSettings


def test_parent_algorithm_default_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("CS_RAG_REFERENCE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings().reference_root == REPO_ROOT.parent
    assert Settings.from_env().reference_root == REPO_ROOT.parent
    assert missing_entrypoints(REPO_ROOT.parent) == []


def test_external_algorithm_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CS_RAG_REFERENCE_ROOT", str(tmp_path))
    assert Settings.from_env().reference_root == tmp_path
    assert missing_entrypoints(tmp_path) == list(STAGE_SCRIPTS.values())
    monkeypatch.setenv("CS_RAG_REFERENCE_ROOT", "../another-algorithm")
    assert Settings.from_env().reference_root == (REPO_ROOT.parent / "another-algorithm").resolve()


def test_subprocess_paths_do_not_depend_on_algorithm_nesting(tmp_path):
    gateway = ModelGateway(ModelSettings(
        base_url="", model_name="unconfigured",
        embedding_model="BAAI/bge-small-zh-v1.5",
        reranker_model="BAAI/bge-reranker-v2-m3",
    ))
    runner = ReferenceQueryRunner(
        tmp_path / "external", None,
        SimpleNamespace(projects_root=tmp_path / "projects"), gateway,
    )
    assert runner.models_root == REPO_ROOT / "models"
    environment = runner._build_environment()
    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(REPO_ROOT), str((tmp_path / "external").resolve()),
    ]
    assert Path(environment["HF_HOME"]).is_relative_to(REPO_ROOT)
    assert runner._reference_commit() == "external-working-tree"
    # Only imports engineering configuration; never executes an algorithm/model.
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "from backend.app.config import REPO_ROOT; print(REPO_ROOT)"],
        cwd=tmp_path, env=environment, capture_output=True,
        text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == REPO_ROOT


def test_system_has_no_algorithm_copy_and_readme_images_resolve():
    assert not (REPO_ROOT / "vendor").exists()
    assert not (REPO_ROOT / "components").exists()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert images == [
        "../docs/system/knowledge-base.png",
        "../docs/system/question-answering.png",
    ]
    assert all((REPO_ROOT / path).is_file() for path in images)
    assert (REPO_ROOT / "frontend/dist/index.html").is_file()
