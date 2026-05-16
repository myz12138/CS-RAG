# CS-RAG

This repository contains the reference implementation of the paper:

**Toward Robust GraphRAG: Mitigating Retrieval Drift and Hallucination from Imperfect Knowledge Graphs** (NeurIPS 2026 submission).

## 1. Paper Overview

In multi-hop QA, practical Knowledge Graphs are often imperfect and induce two major failure modes:

- **Retrieval Drift**: noisy KG edges gradually divert the reasoning trajectory.
- **Retrieval Hallucination**: key KG evidence is missing, so the system can keep traversing structure without sufficient support.

The core contribution of this work is to make multi-hop reasoning robust under these conditions:

1. Convert complex questions into traceable query constraints (`query_plan`) and perform constraint-level structured retrieval to reduce drift.
2. Apply sufficiency checking to discriminate between sufficiently resolved and unresolved constraints, then use binding propagation for resolved parts and text recovery for unresolved parts.

The end-to-end logic is organized as:

`Query Plan -> Phase1 (Structured Constraint Retrieval) -> Phase2 (Text Recovery for Unresolved Constraints) -> Evaluation`

---

## 2. CS-RAG Workflow System (Desktop)

In addition to the research code, this project also includes a workflow-driven CS-RAG system design for practical usage and analysis.

### 2.1 System Positioning

The CS-RAG workflow system is designed for local or private deployment of robust GraphRAG-style multi-hop QA, with an emphasis on:

- end-to-end operability,
- transparent intermediate reasoning states,
- and reproducible robustness experiments.

### 2.2 Core Capabilities

- **Zero-friction runtime**: packaged deployment can run without manual Python environment setup.
- **Local-first privacy**: compatible with OpenAI-format local serving backends (e.g., vLLM/Ollama adapters).
- **Anti-hallucination retrieval**: Neff-based sufficiency checking and automatic text fallback when graph evidence is insufficient.
- **Robust ingestion**: sentence-level deduplication and source archiving for traceability.
- **Glass-box observability**: explicit exposure of decomposition, retrieval, reranking, and evidence decisions.

### 2.3 Workflow Steps

1. **Initialization**
   - Start the packaged CS-RAG engine (desktop release) and open the local Web UI.
   - On first launch, runtime folders are initialized (workspace/data cache/model cache/config file).

2. **Configuration**
   - Configure LLM endpoint (`Base URL`, `API Key`, `Model Name`).
   - Configure retrieval controls (Top-N, entity/fallback recall depth, Neff threshold).

3. **Knowledge Base Construction**
   - Create/select a project.
   - Choose **fresh build** or **incremental merge**.
   - Build graph + dense index from text input and inspect graph preview.

4. **Multi-hop QA and Reasoning Inspection**
   - Submit a complex question.
   - Inspect grounded output with source evidence chain (`[KG]` vs `[Text]`).
   - Inspect activated reasoning subgraph and pipeline diagnostics.

### 2.4 Operational Notes

- Keep the backend terminal alive during active service.
- First run may download required NLP/model dependencies.
- Models remain in memory for low-latency inference.
- If the browser is closed accidentally, reconnect to the local UI endpoint without restarting the backend.

---

## 3. Repository Execution Flow (Code)

Main entry:

```bash
python run.py --config configs/robustness_runner.yaml
```

`run.py` sequentially invokes:

1. `components/phase1_run.py`
2. `components/phase2_run.py`
3. `components/eval_run.py`

Responsibilities:

- **Phase1**: read `query plan + KG + dataset`, produce structural evidence and variable candidates (`phase1_evidence.json`).
- **Phase2**: read phase1 outputs and recover unresolved constraints with text evidence (`phase2_evidence.json`).
- **Evaluation**: generate final answers from phase2 evidence and compute EM/F1 (`qa_results_with_recall.json`).

Primary configuration file:

- `configs/robustness_runner.yaml`

Commonly edited fields:

- `run.dataset`, `run.max_samples`
- `run_phase1`, `run_phase2`, `run_evaluation`
- `paths.query_json`, `paths.data_json`, `paths.raw_data_json`, `paths.kg_path`, `paths.result_dir`
- `global_env.OPENAI_API_KEY`, `global_env.OPENAI_BASE_URL`, `global_env.OPENAI_MODEL`

---

## 4. Example Assets: `KGs` and `planned_queries`

### 4.1 `KGs/` as runnable examples

Available example KGs:

- `KGs/KG_2wiki`
- `KGs/KG_hotpotqa`
- `KGs/KG_musique`

Each KG directory includes:

- `entities.jsonl`
- `triples.jsonl`
- `title2entities.jsonl`
- `title2triples.jsonl`

These are project-provided runnable examples, aligned with the default model setting in `configs/robustness_runner.yaml`:

- `OPENAI_MODEL = gpt-4o-mini`

### 4.2 `planned_queries/` as runnable examples

Available query-plan examples:

- `planned_queries/2wiki_data`
- `planned_queries/hotpotqa_data`
- `planned_queries/musique_data`

Files such as `query_graph_v8_*.json` can be used directly as Phase1 input.

---

## 5. Offline KG Perturbation Utility

Script:

- `tools/perturb_kg_offline.py`

Purpose:

Generate controlled noisy/incomplete KG variants from base KGs for robustness benchmarking.

Supported capabilities:

- issue-type-driven perturbation (e.g., `semantic_flip`, `mis_bound_relation`, `missing_bridge_edge`)
- ratio sweep (`ratios`)
- scope control (`global` or `query_subgraph`)
- scenario composition (`combine_mode` + `issue_sets`)

### 5.1 Command usage

List perturbation scenarios under current config:

```bash
python tools/perturb_kg_offline.py --config configs/kg_perturb.yaml --list
```

Generate all configured scenarios:

```bash
python tools/perturb_kg_offline.py --config configs/kg_perturb.yaml
```

Override dataset from CLI:

```bash
python tools/perturb_kg_offline.py --config configs/kg_perturb.yaml --dataset 2wiki
```

Run selected scenario(s) (`--scenario` can be repeated):

```bash
python tools/perturb_kg_offline.py --config configs/kg_perturb.yaml --scenario semantic_flip
```

### 5.2 Key parameters (`configs/kg_perturb.yaml`)

- `seed`: random seed
- `output_root`: output root for perturbed KGs
- `run.dataset`: default dataset key
- `run.combine_mode`: scenario expansion mode (`cartesian` / `aligned`)
- `run.ratios`: perturbation ratios
- `run.scopes`: perturbation scopes (`global` / `query_subgraph`)
- `run.issue_sets`: scenario definitions
- `run.query_subgraph.*`: query-subgraph construction controls
- `datasets.*.kg_dir`: base KG path per dataset
- `datasets.*.query_json`: query-plan path per dataset
- `issue_params.*`: issue-specific perturbation rules

### 5.3 Output layout

Default output structure:

`output_root/<scenario_name>/<KG_dir_name>/`

Typical files:

- `entities.jsonl`
- `triples.jsonl`
- `title2entities.jsonl`
- `title2triples.jsonl`
- `perturb_meta.json` (configuration and perturbation statistics)

---

## 6. Regenerating `planned_queries`

The query-plan builder script has been renamed:

- old: `components/query_graph_builder_v8_universal_v2.py`
- new: `components/query_plan_builder.py`

Purpose:

Use an LLM (default: `gpt-4o-mini`) to decompose raw questions into `query_plan`, then write to `planned_queries/.../query_graph_v8_*.json`.

### 6.1 Run command

```bash
python components/query_plan_builder.py
```

### 6.2 Common environment variables

- `DATASET`: `2wiki | hotpotqa | musique`
- `DATA_PATH`: input dataset path
- `OUTPUT_FILE`: output query-plan path
- `NUM_SAMPLES`: number of examples
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`

PowerShell example:

```powershell
$env:DATASET="2wiki"
$env:NUM_SAMPLES="1000"
$env:OUTPUT_FILE="planned_queries/2wiki_data/query_graph_v8_2wiki.json"
python components/query_plan_builder.py
```

---

## 7. Recommended Reproduction Procedure

1. Prepare/verify `planned_queries` (reuse examples or regenerate with `query_plan_builder.py`).
2. Prepare base KG or generate perturbed KG with `perturb_kg_offline.py`.
3. Configure paths, model endpoint, and output directory in `configs/robustness_runner.yaml`.
4. Execute:

```bash
python run.py --config configs/robustness_runner.yaml
```

5. Check outputs under `paths.result_dir`:
   - `phase1_evidence.json`
   - `phase2_evidence.json`
   - `qa_results_with_recall.json`
   - `phase1.log`, `phase2.log`, `evaluation.log`
