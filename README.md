# CS-RAG

![Windows](https://img.shields.io/badge/Platform-Windows-blue?logo=windows)
![Local](https://img.shields.io/badge/Deployment-100%25_Local-brightgreen)
![Status](https://img.shields.io/badge/Status-Stable_v1.1-orange)

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

## 2. CS-RAG Workflow System Guide (Desktop)

This section integrates the complete workflow-oriented system guide and aligns all terminology to **CS-RAG**. Thanks for our co-worker. ![Jinchuan Xu](https://github.com/203824552)

### 2.1 Core Features

- **Zero-Configuration Runtime**: Built-in deep learning runtime (PyTorch/Transformers). In desktop release mode, launching the executable starts both backend terminal and Web UI.
- **100% Data Privacy (Local-First)**: Compatible with local LLM serving via OpenAI-style API format (e.g., vLLM/Ollama adapters). Dense vector indexing remains local.
- **Anti-Hallucination Retrieval**: A robust multi-stage pipeline with Neff-based sufficiency checking and automatic `Text Fallback` retrieval when graph evidence is insufficient.
- **Robust Data Management**:
  - Sentence-level MD5 deduplication to block redundant ingestion.
  - Timestamped source archiving for traceability.
- **Glass-Box Observability**: Intermediate decomposition, retrieval, reranking, and evidence decisions are explicitly exposed for analysis.

### 2.2 Download

The packaged desktop system (including runtime dependencies and offline model assets) is approximately **2.44 GB**.

- **[Download CS-RAG Workflow v1.1 (Windows)](https://huggingface.co/203824552xjc/GraphRAG-Studio-Release/resolve/main/GraphRAG_Studio_v1.1_Windows.zip?download=true)**

If download speed is limited in your region, using a download manager is recommended.

### 2.3 Quick Start

#### Step 1: Initialization

Extract the package to an English-only path (for example: `D:\CS-RAG_Workflow`) and launch the executable. The backend terminal opens first, then the browser UI is available at `http://127.0.0.1:8866`.

On first launch, the system auto-generates runtime directories next to the executable:

- `workspace/`: local cache for KG JSON files, entity dictionaries, and dense vector indexes.
- `data_input/`: immutable archive for uploaded `.txt` sources (with timestamping).
- `models/`: offline cache for embedding/reranker models.
- `config.json`: runtime configuration; auto-regenerated if accidentally removed.

#### Step 2: System Configuration

![CS-RAG Configuration](https://github.com/user-attachments/assets/97f88902-f28f-43b4-8931-ddb7fabe41a2)

In the **Configuration** tab:

1. **LLM Settings**
   - Configure `Base URL`, `Model Name`, and `API Key`.
   - Supports both cloud APIs and local OpenAI-format serving endpoints.

2. **Retrieval Parameters**
   - **Top-N**: maximum evidence snippets sent to the final answer model.
   - **Entity / Fallback K**: recall depth for graph traversal and text fallback.
   - **Neff Threshold**: topology entropy threshold for evidence sufficiency control.

After clicking **Save**, parameter updates are applied in-memory immediately (no service restart required).

#### Step 3: Knowledge Base Construction

![CS-RAG KG Construction](https://github.com/user-attachments/assets/7e95027a-0f15-4c28-80c1-6602965b31fd)

In the **KG Construction** tab:

1. **Project Management**
   - Create a new project or select an existing project.

2. **Build Modes**
   - **Fresh Build**: reset graph/vector indexes for the selected project.
   - **Incremental Merging**: append new knowledge to an existing project.
   - Sentence-level MD5 deduplication is applied to prevent duplicate ingestion.

3. **Automated Pipeline**
   - Upload `.txt` files or paste text.
   - Click **Start Building** for chunking, graph extraction, and vector indexing.
   - After completion, the right panel renders a graph preview (top core nodes for browser stability).

4. **Vector Refresh**
   - If embedding model settings are changed, vector index can be rebuilt from cached text without rerunning extraction.

#### Step 4: Multi-hop QA and Reasoning

![CS-RAG Multi-hop QA](https://github.com/user-attachments/assets/f43bec92-3253-47f1-8b3b-0e8aa64eee8d)

In the **Multi-hop QA** tab:

1. **Transparent Execution**
   - Submit a multi-hop question and track pipeline progress in backend logs.

2. **Grounded Answer + Evidence Chain**
   - The system returns concise answers and explicit evidence provenance (`[KG]` vs `[Text]`).

3. **Activated Reasoning Subgraph**
   - The panel highlights the activated reasoning nodes/paths for explainability.

### 2.4 Glass-Box Analysis Dashboard

![CS-RAG Glass-Box Dashboard](https://github.com/user-attachments/assets/f936e1a0-67a8-4a40-998b-8c112827fd28)

The dashboard provides synchronized introspection of the internal pipeline:

- **Phase 1 (Logical Skeleton Decomposition)**: decomposes complex query into atomic triples.
- **Phase 2 (Graph Coarse Ranking & Neff Entropy)**: evaluates graph evidence sufficiency; unresolved/high-entropy cases trigger text fallback.
- **Phase 3 (Evidence Re-ranking Top-K)**: displays recalled evidence with similarity scores and detailed context view.

### 2.5 System Mechanics and Operational Details

- **Backend Terminal**: the terminal is the active backend runtime; closing it terminates the service.
- **First-Run Dependency Download**: initial execution may fetch tokenizer/model resources.
- **Memory Residency**: models remain loaded for low-latency QA.
- **Graceful Shutdown**: use the UI stop operation (if provided by desktop build) to safely release memory/resources.
- **Session Recovery**: if browser closes accidentally, reconnect to `http://127.0.0.1:8866` without restarting backend.

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
