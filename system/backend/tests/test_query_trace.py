import pytest

from backend.app.application.query_service import QueryExecutionError, QueryService
from backend.app.algorithm_adapter.reference import ReferenceAlgorithmAdapter
from backend.app.algorithm_adapter.reference_runner import ReferenceQueryRunner
from backend.app.storage.sqlite_store import SQLiteStore


def _ready_version(store: SQLiteStore):
    project = store.create_project("查询追踪测试")
    version = store.create_version(project.project_id, "import_only", [])
    version = store.publish_version(
        version.version_id,
        status="ready",
        text_status="ready",
        kg_status="ready",
    )
    return project, version


def test_query_and_event_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project, version = _ready_version(store)

    query = store.create_query(project.project_id, version.version_id, "谁与谁有关？", "detailed")
    event = store.append_query_event(
        query.query_id,
        step="phase1_graph_retrieval",
        status="completed",
        message="图检索完成",
        constraint_id="T0",
        counts={"triples": 2},
        metrics={"score": 0.8},
        warnings=["存在冲突事实"],
    )
    updated = store.update_query(
        query.query_id,
        status="completed",
        current_step="answer_generation",
        answer="测试答案",
        warnings=["存在冲突事实"],
        metadata={"query_plan": [{"head": "谁"}]},
    )

    assert event.seq == 1
    assert store.list_query_events(query.query_id)[0].to_dict()["constraint_id"] == "T0"
    assert updated.status == "completed"
    assert updated.answer == "测试答案"
    assert updated.metadata["query_plan"][0]["head"] == "谁"


def test_query_service_persists_structured_failure(tmp_path):
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project, version = _ready_version(store)
    query = store.create_query(project.project_id, version.version_id, "测试问题")

    adapter = ReferenceAlgorithmAdapter(tmp_path / "missing-reference", "test")
    service = QueryService(store, adapter)

    with pytest.raises(QueryExecutionError) as error:
        service.run(query.query_id)

    assert error.value.code == "QUERY_FAILED"
    failed = store.get_query(query.query_id)
    assert failed.status == "failed"
    assert failed.error_code == "QUERY_FAILED"
    assert store.list_query_events(query.query_id)[-1].status == "failed"


def test_detailed_trace_exposes_only_observed_algorithm_fields():
    phase1_events = ReferenceQueryRunner._build_phase1_detail_trace(
        {
            "known_entities": ["甲"],
            "entity_map": [{"query_entity": "甲"}],
            "triples_debug": [
                {
                    "query_triple_index": 0,
                    "stage": "second_pass",
                    "anchors": [1],
                    "candidate_ids_topk": [2, 3],
                    "llm_selected_triple_ids": [2],
                    "checked": True,
                    "neff": 1.25,
                    "status": "resolved",
                }
            ],
            "variable_candidates": {
                "?城市": {
                    "union_ids": [4, 5],
                    "intersection_ids": [4],
                    "primary_ranked": [{"entity_id": 4, "score": 0.9}],
                }
            },
        },
        "phase1.json",
    )
    phase2_events = ReferenceQueryRunner._build_phase2_detail_trace(
        [
            {
                "query_triple_index": 0,
                "triple_status": "unresolved",
                "kg_triples": [],
                "candidate_evidences": [{"title": "说明", "context": "原文"}],
            }
        ],
        "phase2.json",
    )

    assert phase1_events[1]["step"] == "variable_propagation"
    assert phase1_events[2]["step"] == "neff_gating"
    assert phase1_events[2]["metrics"]["neff"] == 1.25
    assert "一致性处理后状态：已解决" in phase1_events[2]["message"]
    assert phase1_events[-1]["step"] == "variable_propagation"
    assert "pass_index" not in phase1_events[1]
    assert phase2_events[0]["reason_code"] == "UNRESOLVED_CONSTRAINT"
    assert phase2_events[0]["branch_id"] == "text_fallback"


def test_neff_projection_distinguishes_unchecked_missing_and_empty_pool():
    debug = {"triples_debug": [
        {"query_triple_index": 0, "checked": False, "status": "unresolved"},
        {"query_triple_index": 1, "checked": True, "neff": None, "status": "unresolved"},
        {"query_triple_index": 2, "checked": True, "neff": 1.1, "status": "unresolved"},
        {"query_triple_index": 3},
    ], "variable_candidates": {"?x": {"union_ids": [1], "intersection_ids": None, "primary_ranked": [{}]}}}
    events = ReferenceQueryRunner._build_phase1_detail_trace(debug, "phase1.json")
    assert len(events) == 5  # no fabricated entity mapping/retrieval events
    assert "未执行" in events[0]["message"] and events[0]["metrics"] == {}
    assert events[1]["metrics"] == {"neff": None}
    assert "无有限值" in events[1]["message"]
    assert "未解决" in events[2]["message"]  # low Neff does not override consolidation
    assert "未记录" in events[3]["message"]
    assert events[-1]["counts"]["intersection_candidates"] is None


def test_resolved_constraint_does_not_claim_text_fallback_or_evidence_success():
    events = ReferenceQueryRunner._build_phase2_detail_trace([
        {"query_triple_index": 4, "triple_status": "resolved", "kg_triples": [{}], "candidate_evidences": []},
    ], "phase2.json")
    assert events[0]["branch_id"] == "kg_evidence"
    assert events[0]["warnings"]
    assert "重排" not in events[0]["message"]


def test_stage_details_survive_later_failure_and_failure_uses_ui_stage_id(tmp_path):
    # Only exercise event transport; no algorithm computation or model stand-in.
    from backend.app.algorithm_adapter.reference_runner import ReferenceAlgorithmError
    store = SQLiteStore(tmp_path / "metadata.sqlite3")
    project, version = _ready_version(store)
    query = store.create_query(project.project_id, version.version_id, "事件传输测试", "detailed")
    adapter = ReferenceAlgorithmAdapter(tmp_path, "test")
    trace = ReferenceQueryRunner._build_phase1_detail_trace({"triples_debug": [
        {"query_triple_index": 0, "checked": True, "neff": 1.2, "status": "resolved"}
    ]}, "phase1.json")

    def emit_events_then_fail(request, *, progress):
        progress(0.49, "阶段结果", {"step": "phase1_graph_retrieval", "status": "completed", "trace_events": trace})
        assert store.list_query_events(query.query_id)[-1].constraint_id == "T0"
        raise ReferenceAlgorithmError("ALGORITHM_STAGE_FAILED", "事件传输测试异常", stage="phase2")

    adapter.attach_runner(emit_events_then_fail)
    with pytest.raises(QueryExecutionError):
        QueryService(store, adapter).run(query.query_id)
    events = store.list_query_events(query.query_id)
    assert events[1].status == "completed"
    assert events[-2].metrics["neff"] == 1.2
    assert events[-1].step == "phase2_text_recovery"
