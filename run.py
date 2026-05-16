#!/usr/bin/env python3
"""
Simple sequential runner:
phase1 -> phase2 -> evaluation

No stdout/stderr interception. Child scripts print their own tqdm bars.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_config(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping.")
        return cfg
    except Exception:
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("Config root must be a mapping.")
        return cfg


def _as_env_map(d: Optional[Dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[str(k)] = "1" if v else "0"
        elif isinstance(v, (list, tuple)):
            out[str(k)] = ",".join(str(x) for x in v)
        else:
            out[str(k)] = str(v)
    return out


def _resolve_output(path_value: str, result_dir: Path) -> str:
    p = Path(str(path_value))
    if p.is_absolute():
        return str(p)
    return str(result_dir / p)


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return base_dir / p


def _run(stage: str, script: str, python_exec: str, env: Dict[str, str], result_dir: Path, cwd: Path) -> None:
    cmd = [python_exec, script]
    print(f"[Pipeline] {stage}: {' '.join(cmd)}")
    ret = subprocess.run(cmd, env=env, cwd=str(cwd)).returncode

    log_path = result_dir / f"{stage}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.write(f"exit_code={ret}\n")

    if ret != 0:
        raise RuntimeError(f"{stage} failed with exit code {ret}. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple sequential runner for phase1->phase2->evaluation.")
    parser.add_argument("--config", default="configs/robustness_runner.yaml", help="Path to YAML/JSON config.")
    args = parser.parse_args()

    cfg_path = _resolve_path(str(args.config), PROJECT_ROOT)
    cfg = _load_config(cfg_path)

    python_exec = str(cfg.get("python_exec", "python"))
    run_cfg = cfg.get("run", {}) or {}
    scripts = cfg.get("scripts", {}) or {}
    paths = cfg.get("paths", {}) or {}

    required_scripts = ["phase1", "phase2", "evaluation"]
    for key in required_scripts:
        if key not in scripts:
            raise ValueError(f"Missing scripts.{key} in config.")

    required_paths = ["query_json", "raw_data_json", "data_json", "kg_path", "result_dir"]
    for key in required_paths:
        if key not in paths:
            raise ValueError(f"Missing paths.{key} in config.")

    dataset = str(run_cfg.get("dataset", "2wiki"))
    max_samples = int(run_cfg.get("max_samples", 1000))
    run_phase1 = bool(run_cfg.get("run_phase1", False))
    run_phase2 = bool(run_cfg.get("run_phase2", True))
    run_eval = bool(run_cfg.get("run_evaluation", True))

    result_dir = _resolve_path(str(paths["result_dir"]), PROJECT_ROOT)
    result_dir.mkdir(parents=True, exist_ok=True)

    resolved_scripts: Dict[str, Path] = {}
    for key in required_scripts:
        sp = _resolve_path(str(scripts[key]), PROJECT_ROOT)
        if not sp.exists():
            raise FileNotFoundError(f"Script not found: {sp}")
        resolved_scripts[key] = sp

    must_exist_inputs = ["query_json", "raw_data_json", "data_json", "kg_path"]
    for key in must_exist_inputs:
        pp = _resolve_path(str(paths[key]), PROJECT_ROOT)
        if not pp.exists():
            raise FileNotFoundError(f"Path not found for paths.{key}: {pp}")

    phase1_output = _resolve_output(str(paths.get("phase1_output", "phase1_evidence.json")), result_dir)
    phase2_output = _resolve_output(str(paths.get("phase2_output", "phase2_evidence.json")), result_dir)
    eval_output = _resolve_output(str(paths.get("eval_output", "qa_results_with_recall.json")), result_dir)

    base_env = os.environ.copy()
    base_env.update(_as_env_map(cfg.get("global_env", {})))
    existing_pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    base_env.update(
        {
            "DATASET": dataset,
            "MAX_SAMPLES": str(max_samples),
        }
    )

    if run_phase1:
        env1 = dict(base_env)
        env1.update(
            {
                "QUERY_JSON": str(paths["query_json"]),
                "RAW_DATA_JSON": str(paths["raw_data_json"]),
                "KG_PATH": str(paths["kg_path"]),
                "OUTPUT_JSON": phase1_output,
            }
        )
        env1.update(_as_env_map((cfg.get("phase1", {}) or {}).get("env", {})))
        _run("phase1", str(resolved_scripts["phase1"]), python_exec, env1, result_dir, PROJECT_ROOT)
    elif not Path(phase1_output).exists():
        raise FileNotFoundError(f"run_phase1=false but phase1 output not found: {phase1_output}")

    if run_phase2:
        env2 = dict(base_env)
        env2.update(
            {
                "PHASE1_JSON": phase1_output,
                "DATA_JSON": str(paths["data_json"]),
                "KG_PATH": str(paths["kg_path"]),
                "OUTPUT_JSON": phase2_output,
            }
        )
        env2.update(_as_env_map((cfg.get("phase2", {}) or {}).get("env", {})))
        _run("phase2", str(resolved_scripts["phase2"]), python_exec, env2, result_dir, PROJECT_ROOT)
    elif not Path(phase2_output).exists():
        raise FileNotFoundError(f"run_phase2=false but phase2 output not found: {phase2_output}")

    if run_eval:
        env3 = dict(base_env)
        env3.update(
            {
                "PHASE1_JSON": phase1_output,
                "PHASE2_JSON": phase2_output,
                "DATA_JSON": str(paths["data_json"]),
                "OUTPUT_JSON": eval_output,
            }
        )
        env3.update(_as_env_map((cfg.get("evaluation", {}) or {}).get("env", {})))
        _run("evaluation", str(resolved_scripts["evaluation"]), python_exec, env3, result_dir, PROJECT_ROOT)

    print("[Pipeline] done.")
    print(f"[Pipeline] phase1_output={phase1_output}")
    print(f"[Pipeline] phase2_output={phase2_output}")
    print(f"[Pipeline] eval_output={eval_output}")


if __name__ == "__main__":
    main()
