"""Run externally supplied CS-RAG stages in an isolated query workspace."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from backend.app.algorithm_adapter.contracts import QueryRequest, QueryResult
from backend.app.algorithm_adapter.integration import STAGE_SCRIPTS, missing_entrypoints
from backend.app.config import REPO_ROOT
from backend.app.algorithm_adapter.trace_projection import phase1_trace, phase2_trace
from backend.app.application.query_settings import QuerySettings
from backend.app.algorithm_adapter.reference import ReferenceAlgorithmAdapter
from backend.app.model_gateway.base import ModelGateway
from backend.app.storage.file_store import ProjectFileStore
from backend.app.storage.sqlite_store import SQLiteStore


ProgressCallback = Callable[[float, str, dict[str, Any] | None], None]


class ReferenceAlgorithmError(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


class ReferenceQueryRunner:
    """Use the original scripts and file contracts without changing their logic."""

    def __init__(
        self,
        reference_root: Path,
        store: SQLiteStore,
        files: ProjectFileStore,
        model_gateway: ModelGateway,
        *,
        models_root: Path | None = None,
        reference_commit: str = "",
        timeout_seconds: float = 1800.0,
        vector_store_backend: str = "local_file",
        vector_store_root: Path | None = None,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        qdrant_on_disk: bool = True,
        text_index_recall_multiplier: int = 4,
        query_settings: QuerySettings | None = None,
    ) -> None:
        self.reference_root = reference_root.resolve()
        self.store = store
        self.files = files
        self.model_gateway = model_gateway
        self.models_root = (models_root or REPO_ROOT / "models").resolve()
        self.reference_commit = reference_commit
        self.timeout_seconds = max(120.0, timeout_seconds)
        self.vector_store_backend = (vector_store_backend or "local_file").strip().lower()
        self.vector_store_root = (
            vector_store_root or (self.files.projects_root.parent / "vector_store")
        ).resolve()
        self.qdrant_url = qdrant_url.strip()
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_on_disk = qdrant_on_disk
        self.text_index_recall_multiplier = max(1, int(text_index_recall_multiplier))
        self.query_settings = (query_settings or QuerySettings()).validate()

    def configure_query_settings(self, settings: QuerySettings) -> None:
        self.query_settings = settings.validate()

    def __call__(
        self,
        request: QueryRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> QueryResult:
        return self.run(request, progress=progress)

    def run(
        self,
        request: QueryRequest,
        *,
        progress: ProgressCallback | None = None,
    ) -> QueryResult:
        if not request.query_id:
            raise ReferenceAlgorithmError("QUERY_ID_REQUIRED", "运行参考算法必须有 query_id")
        if not self.reference_root.is_dir():
            raise ReferenceAlgorithmError("REFERENCE_ROOT_MISSING", "参考算法目录不存在")
        missing = missing_entrypoints(self.reference_root)
        if missing:
            raise ReferenceAlgorithmError(
                "ALGORITHM_SCRIPT_MISSING",
                "算法接入不完整，缺少入口：" + "、".join(missing),
            )

        runtime = ReferenceAlgorithmAdapter.runtime_capabilities()
        if not runtime["available"]:
            missing = "、".join(runtime["missing"])
            raise ReferenceAlgorithmError(
                "ALGORITHM_RUNTIME_UNAVAILABLE",
                f"参考算法运行依赖未安装：{missing}。请安装目标工程的算法可选依赖后重试。",
            )

        version = self.store.get_version(request.version_id or "")
        if version.project_id != request.project_id or version.status != "ready":
            raise ReferenceAlgorithmError(
                "VERSION_NOT_QUERYABLE",
                "查询必须绑定同一项目下状态为 ready 的完整知识库版本",
            )
        if not self.store.list_text_entries(version.version_id):
            raise ReferenceAlgorithmError(
                "TEXT_LAYER_REQUIRED",
                "原 CS-RAG Phase2 需要文本条目，图谱专用库不能启动混合问答",
            )
        if not self._effective_model_base_url():
            raise ReferenceAlgorithmError(
                "MODEL_NOT_CONFIGURED",
                "查询计划和答案生成需要真实 LLM 服务，请先配置模型服务地址或 API Key",
            )

        version_dir = self.files.get_version_dir(request.project_id, version.version_id)
        workspace = self.files.create_query_workspace(
            request.project_id,
            version.version_id,
            request.query_id,
        )
        kg_dir = version_dir / "algorithm" / "reference_kg"
        if not kg_dir.is_dir():
            raise ReferenceAlgorithmError("REFERENCE_KG_MISSING", "当前版本缺少原算法 KG 适配目录")

        dataset_path = workspace / "dataset.json"
        query_plan_path = workspace / "query_plan.json"
        phase1_path = workspace / "phase1_evidence.json"
        phase2_path = workspace / "phase2_evidence.json"
        evaluation_path = workspace / "evaluation.json"
        text_index_candidates_path = self._write_dataset(
            dataset_path,
            version.version_id,
            request,
        )
        env = self._build_environment()
        env.update(
            {
                "DATASET": "hotpotqa",
                "NUM_SAMPLES": "1",
                "MAX_SAMPLES": "1",
                "SLEEP_SECS": "0",
            }
        )

        self._run_stage(
            "query_plan",
            STAGE_SCRIPTS["query_plan"],
            env | {"DATA_PATH": str(dataset_path), "OUTPUT_FILE": str(query_plan_path)},
            workspace,
            progress,
            0.08,
            0.20,
        )
        plan_items = self._load_list(query_plan_path, "query_plan")
        if not plan_items or not plan_items[0].get("query_plan"):
            raise ReferenceAlgorithmError("QUERY_PLAN_EMPTY", "查询计划为空，无法进入 Phase1", stage="query_plan")
        query_plan = plan_items[0].get("query_plan", [])
        self._report(
            progress,
            0.21,
            f"查询计划已生成：{len(query_plan)} 条原子约束",
            details={
                "step": "query_plan", "status": "completed",
                "metadata": {"query_plan": query_plan},
                "counts": {"constraints": len(query_plan)},
                "output_ref": str(query_plan_path.relative_to(version_dir)).replace("\\", "/"),
            },
        )

        phase_env = env | {
            "QUERY_JSON": str(query_plan_path),
            "RAW_DATA_JSON": str(dataset_path),
            "KG_PATH": str(kg_dir),
            "OUTPUT_JSON": str(phase1_path),
        }
        self._run_stage(
            "phase1",
            STAGE_SCRIPTS["phase1"],
            phase_env,
            workspace,
            progress,
            0.25,
            0.47,
        )
        phase1_items = self._load_list(phase1_path, "phase1")
        if not phase1_items:
            raise ReferenceAlgorithmError("PHASE1_EMPTY", "Phase1 没有返回结果", stage="phase1")
        phase1_item = phase1_items[0]
        phase1_evidence = phase1_item.get("evidence_triples", []) or []
        self._report(
            progress,
            0.49,
            "Phase1 图检索结果已生成",
            details={
                "step": "phase1_graph_retrieval", "status": "completed",
                "trace_events": self._build_phase1_detail_trace(
                    phase1_item.get("debug") or {},
                    str(phase1_path.relative_to(version_dir)).replace("\\", "/"),
                ) if request.trace_level == "detailed" else [],
                "metadata": {
                    "phase1_debug": phase1_item.get("debug") or {},
                    "phase1_evidence": phase1_evidence,
                    "phase1_support": phase1_item.get("support", []) or [],
                },
                "counts": {
                    "evidence_triples": len(phase1_evidence),
                    "support_groups": len(phase1_item.get("support", []) or []),
                },
                "output_ref": str(phase1_path.relative_to(version_dir)).replace("\\", "/"),
            },
        )

        phase2_env = env | {
            "PHASE1_JSON": str(phase1_path),
            "DATA_JSON": str(dataset_path),
            "KG_PATH": str(kg_dir),
            "OUTPUT_JSON": str(phase2_path),
            # Retained for index-adapter compatibility. The current algorithm
            # scans passages directly; these inputs do not prove index usage.
            "TEXT_INDEX_DIR": str(version_dir / "indexes" / "text"),
            "TEXT_INDEX_CANDIDATES": str(text_index_candidates_path),
            "TEXT_INDEX_RECALL_MULTIPLIER": str(self.query_settings.text_index_recall_multiplier),
            "TEXT_INDEX_RECALL_STATS": str(workspace / "text_index_recall_stats.json"),
        }
        self._run_stage(
            "phase2",
            STAGE_SCRIPTS["phase2"],
            phase2_env,
            workspace,
            progress,
            0.52,
            0.72,
        )
        phase2_items = self._load_list(phase2_path, "phase2")
        if not phase2_items:
            raise ReferenceAlgorithmError("PHASE2_EMPTY", "Phase2 没有返回结果", stage="phase2")
        phase2_item = phase2_items[0]
        phase2_evidence = phase2_item.get("triples_evidence", []) or []
        resolved_count_live = sum(
            1
            for item in phase2_evidence
            if str(item.get("triple_status", "")).lower() == "resolved"
        )
        self._report(
            progress,
            0.74,
            "Phase2 证据收集结果已生成",
            details={
                "step": "phase2_text_recovery", "status": "completed",
                "trace_events": self._build_phase2_detail_trace(
                    phase2_evidence,
                    str(phase2_path.relative_to(version_dir)).replace("\\", "/"),
                ) if request.trace_level == "detailed" else [],
                "metadata": {"evidence": phase2_evidence},
                "counts": {
                    "constraints": len(phase2_evidence),
                    "resolved_constraints": resolved_count_live,
                    "candidate_evidences": sum(
                        len(item.get("candidate_evidences", []) or [])
                        for item in phase2_evidence
                    ),
                },
                "output_ref": str(phase2_path.relative_to(version_dir)).replace("\\", "/"),
            },
        )
        text_index_recall = self._load_optional_json(
            workspace / "text_index_recall_stats.json"
        )

        evaluation_env = env | {
            "PHASE1_JSON": str(phase1_path),
            "PHASE2_JSON": str(phase2_path),
            "DATA_JSON": str(dataset_path),
            "OUTPUT_JSON": str(evaluation_path),
        }
        self._run_stage(
            "answer_generation",
            STAGE_SCRIPTS["answer_generation"],
            evaluation_env,
            workspace,
            progress,
            0.76,
            0.94,
        )
        evaluation_items = self._load_list(evaluation_path, "evaluation")
        if not evaluation_items:
            raise ReferenceAlgorithmError("EVALUATION_EMPTY", "答案生成没有返回结果", stage="answer_generation")
        evaluation_item = evaluation_items[0]
        answer_preview = str(evaluation_item.get("model_answer", "") or "").strip()
        self._report(
            progress,
            0.95,
            f"答案已生成：{len(answer_preview)} 个字符",
            details={
                "step": "answer_generation", "status": "completed",
                "metadata": {"answer_preview": answer_preview},
                "counts": {"answer_length": len(answer_preview)},
                "output_ref": str(evaluation_path.relative_to(version_dir)).replace("\\", "/"),
            },
        )

        phase1 = phase1_items[0]
        phase2 = phase2_items[0]
        evaluation = evaluation_item
        triples_evidence = phase2.get("triples_evidence", []) or []
        resolved_count = sum(
            1 for item in triples_evidence if str(item.get("triple_status", "")).lower() == "resolved"
        )
        unresolved_count = max(0, len(triples_evidence) - resolved_count)
        evidence_count = sum(
            len(item.get("candidate_evidences", []) or []) for item in triples_evidence
        )
        relative = lambda path: str(path.relative_to(version_dir)).replace("\\", "/")
        trace = [
            {
                "step": "query_plan",
                "status": "completed",
                "message": "查询计划已生成",
                "counts": {"constraints": len(query_plan)},
                "output_ref": relative(query_plan_path),
            },
            {
                "step": "phase1_graph_retrieval",
                "status": "completed",
                "message": "Phase1 图检索已完成",
                "counts": {
                    "evidence_triples": len(phase1.get("evidence_triples", []) or []),
                    "support_groups": len(phase1.get("support", []) or []),
                },
                "output_ref": relative(phase1_path),
            },
            {
                "step": "phase2_text_recovery",
                "status": "completed",
                "message": "Phase2 已按约束状态完成 KG evidence 收集或文本回退检索",
                "counts": {
                    "constraints": len(triples_evidence),
                    "resolved_constraints": resolved_count,
                    "candidate_evidences": evidence_count,
                },
                "output_ref": relative(phase2_path),
            },
            {
                "step": "answer_generation",
                "status": "completed",
                "message": "答案生成已完成",
                "counts": {"answer_length": len(str(evaluation.get("model_answer", "")))},
                "output_ref": relative(evaluation_path),
            },
        ]
        if request.trace_level == "detailed":
            phase1_detail = self._build_phase1_detail_trace(
                phase1.get("debug") or {},
                relative(phase1_path),
            )
            phase2_detail = self._build_phase2_detail_trace(
                triples_evidence,
                relative(phase2_path),
            )
            trace = [trace[0], *phase1_detail, trace[1], *phase2_detail, trace[2], trace[3]]
        warnings: list[str] = []
        if unresolved_count and (text_index_recall or {}).get("error"):
            warnings.append(
                "文本索引组件记录了错误（不据此推断召回分支）："
                + str(text_index_recall["error"])
            )
        conflict_count = self.store.version_counts(version.version_id)["kg_conflicts"]
        if conflict_count:
            warnings.append(f"当前知识库存在 {conflict_count} 条结构冲突记录，相关事实均按原算法参与检索")
        self._report(progress, 1.0, "问答流程已完成")
        return QueryResult(
            answer=str(evaluation.get("model_answer", "")).strip() or None,
            query_plan=query_plan,
            evidence=triples_evidence,
            trace=trace,
            warnings=warnings,
            metadata={
                "version_id": version.version_id,
                "reference_commit": self._reference_commit(),
                "artifacts": {
                    "dataset": relative(dataset_path),
                    "query_plan": relative(query_plan_path),
                    "phase1": relative(phase1_path),
                    "phase2": relative(phase2_path),
                    "evaluation": relative(evaluation_path),
                },
                "model_input": evaluation.get("model_input"),
                "phase1_debug": phase1.get("debug"),
                "phase1_evidence": phase1.get("evidence_triples", []) or [],
                "phase1_support": phase1.get("support", []) or [],
                "query_config": self.query_settings.to_dict(),
                "evaluation_metrics": {
                    key: evaluation.get(key)
                    for key in ("em", "f1", "resolved_triple_ratio", "answer_supported_by_evidence", "unsupported_answer_flag")
                    if key in evaluation
                },
                "text_index_recall": text_index_recall,
            },
        )

    def _build_environment(self) -> dict[str, str]:
        settings = self.model_gateway.settings
        environment = os.environ.copy()
        # Keep any first-run HuggingFace downloads inside this project. The
        # normal path is the checked-in-by-configuration local snapshot under
        # models/, while this cache is the safe fallback when a snapshot is
        # missing or a user selects another model.
        model_cache_root = REPO_ROOT / "runtime" / "cs-rag-studio-model-cache"
        environment["HF_HOME"] = str(model_cache_root)
        environment["HF_HUB_CACHE"] = str(model_cache_root / "hub")
        environment["TRANSFORMERS_CACHE"] = str(model_cache_root / "transformers")
        environment["SENTENCE_TRANSFORMERS_HOME"] = str(model_cache_root / "sentence-transformers")
        current_pythonpath = environment.get("PYTHONPATH", "")
        pythonpath_parts = [str(REPO_ROOT), str(self.reference_root)]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        environment["OPENAI_MODEL"] = settings.model_name
        environment["OPENAI_API_KEY"] = settings.api_key or ("local" if settings.base_url else "")
        base_url = self._effective_model_base_url()
        if base_url:
            environment["OPENAI_BASE_URL"] = base_url
        environment.pop("EMB_ENDPOINT", None)
        environment.pop("EMB_API_KEY", None)
        embedding_base_url = settings.embedding_base_url
        if embedding_base_url:
            embedding_endpoint = embedding_base_url.rstrip("/")
            if embedding_endpoint.endswith("/v1"):
                embedding_endpoint = embedding_endpoint[:-3]
            environment["EMB_ENDPOINT"] = embedding_endpoint
            environment["EMB_API_KEY"] = settings.embedding_api_key or settings.api_key or "local"
        # A blank embedding endpoint means the reference algorithm should use
        # a local SentenceTransformer. Resolve the configured HuggingFace id
        # to a project-local snapshot when one has already been downloaded.
        environment["EMB_MODEL"] = (
            settings.embedding_model
            if embedding_base_url
            else self._resolve_local_model("embedding", settings.embedding_model)
        )
        environment["EMB_DEVICE"] = "cpu"
        environment["RERANK_MODEL"] = self._resolve_local_model("reranker", settings.reranker_model)
        query_settings = self.query_settings
        environment.update(
            {
                "QUERY_PLAN_MAX_RETRIES": str(query_settings.planner_max_retries),
                "TOP_N_LLM": str(query_settings.top_n_llm),
                "TOP_K_ENTITY": str(query_settings.top_k_entity),
                "RELATION_K": str(query_settings.relation_k),
                "NEFF_THRESHOLD": str(query_settings.neff_threshold),
                "PHASE1_RERANK_MAX_LENGTH": str(query_settings.phase1_rerank_max_length),
                "PHASE2_RERANK_MAX_LENGTH": str(query_settings.phase2_rerank_max_length),
                "RERANK_BATCH": str(query_settings.rerank_batch),
                "RERANK_FP16": "1" if query_settings.rerank_fp16 else "0",
                "INCLUDE_QUESTION_IN_RERANK": "1" if query_settings.include_question_in_rerank else "0",
                "UNRES_TOP_SIMILAR": str(query_settings.unresolved_top_similar),
                "TOPK_UNRESOLVED": str(query_settings.top_k_unresolved),
                "FOCUS_MAX_PER_VAR": str(query_settings.focus_max_per_variable),
                "TEXT_INDEX_RECALL_MULTIPLIER": str(query_settings.text_index_recall_multiplier),
                "PROMPT_MAX_TRIPLES": str(query_settings.prompt_max_triples),
                "PROMPT_MAX_TEXT_PER_TRIPLE": str(query_settings.prompt_max_text_per_triple),
            }
        )
        environment["TEXT_INDEX_BACKEND"] = self.vector_store_backend
        environment["TEXT_INDEX_VECTOR_STORE_ROOT"] = str(self.vector_store_root)
        environment["TEXT_INDEX_QDRANT_URL"] = self.qdrant_url
        environment["TEXT_INDEX_QDRANT_API_KEY"] = self.qdrant_api_key
        environment["TEXT_INDEX_QDRANT_PATH"] = str(self.vector_store_root / "qdrant")
        environment["TEXT_INDEX_QDRANT_ON_DISK"] = "1" if self.qdrant_on_disk else "0"
        return environment

    def _resolve_local_model(self, kind: str, configured: str) -> str:
        """Return a project-local model directory when it is available.

        Keeping the original identifier as the fallback preserves the
        reference algorithm's first-run HuggingFace download behaviour and
        still allows an explicit absolute/local model path in the settings.
        """
        value = str(configured or "").strip()
        if not value:
            return value

        configured_path = Path(value).expanduser()
        candidates: list[Path] = []
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            project_root = REPO_ROOT
            candidates.extend(
                [
                    project_root / configured_path,
                    self.models_root / kind / configured_path,
                    self.models_root / kind / configured_path.name,
                ]
            )

        for candidate in candidates:
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return str(candidate.resolve())
        return value

    def _effective_model_base_url(self) -> str:
        settings = self.model_gateway.settings
        if settings.base_url:
            return settings.base_url
        if settings.api_key:
            return "https://api.openai.com/v1"
        return ""

    def _reference_commit(self) -> str:
        return self.reference_commit or "external-working-tree"

    @staticmethod
    def _build_phase1_detail_trace(debug: dict[str, Any], output_ref: str) -> list[dict[str, Any]]:
        return phase1_trace(debug, output_ref)

    @staticmethod
    def _build_phase2_detail_trace(
        triples_evidence: list[dict[str, Any]],
        output_ref: str,
    ) -> list[dict[str, Any]]:
        return phase2_trace(triples_evidence, output_ref)

    def _run_stage(
        self,
        stage: str,
        script: str,
        environment: dict[str, str],
        workspace: Path,
        progress: ProgressCallback | None,
        start_progress: float,
        end_progress: float,
    ) -> None:
        script_path = self.reference_root / script
        if not script_path.is_file():
            raise ReferenceAlgorithmError("ALGORITHM_SCRIPT_MISSING", f"缺少参考算法脚本：{script}", stage=stage)
        stage_messages = {
            "query_plan": ("查询计划开始：调用 LLM 生成原子约束", "查询计划完成：已校验结构化输出"),
            "phase1": ("Phase1 图检索执行中：实体映射、候选重排、充分性与变量一致性处理；内部结果待阶段结束后展示", "Phase1 脚本执行结束，正在读取图检索结果"),
            "phase2": ("Phase2 证据收集执行中：已解决约束收集 KG evidence，未解决约束执行文本初筛与重排", "Phase2 脚本执行结束，正在读取证据结果"),
            "answer_generation": ("答案生成开始：将各查询元组及入选文本证据送入 LLM", "答案脚本执行结束，正在读取 LLM 回答"),
        }
        start_message, completed_message = stage_messages.get(
            stage,
            (f"{stage}开始", f"{stage}完成"),
        )
        self._report(progress, start_progress, start_message)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.reference_root),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            self._write_log(workspace, stage, "进程超时", str(exc))
            raise ReferenceAlgorithmError("ALGORITHM_STAGE_TIMEOUT", f"{stage} 执行超时", stage=stage) from exc
        except OSError as exc:
            raise ReferenceAlgorithmError("ALGORITHM_PROCESS_FAILED", f"无法启动 {stage}：{exc}", stage=stage) from exc
        self._write_log(workspace, stage, result.stdout, result.stderr)
        if result.returncode != 0:
            raise ReferenceAlgorithmError(
                "ALGORITHM_STAGE_FAILED",
                f"{stage} 执行失败（退出码 {result.returncode}），请查看查询过程日志",
                stage=stage,
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        self._report(progress, end_progress, f"{completed_message}（{elapsed} ms）")

    @staticmethod
    def _write_log(workspace: Path, stage: str, stdout: str, stderr: str = "") -> None:
        log_dir = workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{stage}.log").write_text(
            f"[stdout]\n{stdout}\n[stderr]\n{stderr}\n",
            encoding="utf-8",
        )

    def _write_dataset(self, path: Path, version_id: str, request: QueryRequest) -> Path:
        entries = self.store.list_text_entries(version_id)
        grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        for entry in entries:
            sentences = [
                part.strip()
                for part in re.split(r"(?<=[。！？.!?])\s+|\n+", entry.content)
                if part.strip()
            ]
            units = [
                {"text": sentence, "entry_id": entry.entry_id}
                for sentence in (sentences or [entry.content])
            ]
            grouped.setdefault(entry.title or f"entry-{entry.entry_id}", []).extend(units)
        context = [
            [title, [unit["text"] for unit in units]]
            for title, units in grouped.items()
        ]
        payload = [
            {
                "id": request.query_id,
                "question": request.question,
                "answer": "",
                "ground_truth_answer": "",
                "context": context,
            }
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        candidates = {
            "schema_version": 1,
            "version_id": version_id,
            "units": [
                {
                    "kind": "context_sentence",
                    "doc_idx": doc_idx,
                    "sent_idx": sent_idx,
                    "title": title,
                    "text": unit["text"],
                    "entry_id": unit["entry_id"],
                }
                for doc_idx, (title, units) in enumerate(grouped.items())
                for sent_idx, unit in enumerate(units)
            ],
        }
        candidates_path = path.with_name("text_index_candidates.json")
        temporary = candidates_path.with_name(f".{candidates_path.name}.part")
        temporary.write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(candidates_path)
        return candidates_path

    @staticmethod
    def _load_optional_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _load_list(path: Path, label: str) -> list[dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReferenceAlgorithmError("ALGORITHM_OUTPUT_INVALID", f"{label} 输出无法读取", stage=label) from exc
        if not isinstance(value, list):
            raise ReferenceAlgorithmError("ALGORITHM_OUTPUT_INVALID", f"{label} 输出不是列表", stage=label)
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _report(
        progress: ProgressCallback | None,
        value: float,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if progress is not None:
            progress(round(value, 4), message, details)
