"""Algorithm entrypoints only; no CS-RAG algorithm implementation lives here."""

from pathlib import Path


STAGE_SCRIPTS = {
    "query_plan": "components/query_plan_builder.py",
    "phase1": "components/phase1_run.py",
    "phase2": "components/phase2_run.py",
    "answer_generation": "components/eval_run.py",
}


def missing_entrypoints(root: Path) -> list[str]:
    return [path for path in STAGE_SCRIPTS.values() if not (root / path).is_file()]
