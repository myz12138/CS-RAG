"""Reference algorithm adapter and runtime capability inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from importlib.util import find_spec
from collections.abc import Callable

from backend.app.algorithm_adapter.contracts import QueryRequest, QueryResult


class ReferenceAlgorithmAdapter:
    def __init__(self, reference_root: Path, reference_commit: str) -> None:
        self.reference_root = reference_root
        self.reference_commit = reference_commit
        self._runner: Callable[..., QueryResult] | None = None

    def manifest(self) -> dict[str, Any]:
        runtime = self.runtime_capabilities()
        return {
            "name": "CS-RAG reference algorithm",
            "reference_root": str(self.reference_root),
            "reference_commit": self.reference_commit,
            "query_stages": [
                "query_plan",
                "phase1_graph_retrieval",
                "neff_gating",
                "variable_propagation",
                "phase2_text_recovery",
                "answer_generation",
            ],
            "adapter_status": "subprocess_runner_ready_dependency_gated",
            "runtime": runtime,
        }

    @staticmethod
    def runtime_capabilities() -> dict[str, Any]:
        required = {
            "torch": "torch",
            "transformers": "transformers",
            "sentence_transformers": "sentence_transformers",
            "numpy": "numpy",
            "requests": "requests",
            "tqdm": "tqdm",
            "openai": "openai",
        }
        available = {label: find_spec(module) is not None for label, module in required.items()}
        return {
            "available": all(available.values()),
            "dependencies": available,
            "missing": [label for label, is_available in available.items() if not is_available],
            "query_bridge": "official_scripts_via_isolated_query_workspace",
        }

    def attach_runner(self, runner: Callable[..., QueryResult]) -> None:
        self._runner = runner

    def execute(
        self,
        request: QueryRequest,
        *,
        progress: Callable[[float, str, dict[str, Any] | None], None] | None = None,
    ) -> QueryResult:
        if self._runner is None:
            raise RuntimeError("参考算法运行器尚未接入")
        return self._runner(request, progress=progress)

    def query(self, request: QueryRequest) -> QueryResult:
        return self.execute(request)
