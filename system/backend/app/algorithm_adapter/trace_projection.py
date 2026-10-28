"""Project observed algorithm output into UI events; never rerun/infer decisions."""

from typing import Any


def phase1_trace(debug: dict[str, Any], output_ref: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if "entity_map" in debug:
        mappings = debug.get("entity_map") or []
        events.append({
            "step": "phase1_graph_retrieval", "status": "completed",
            "message": "实体映射结果已输出（阶段结束后的结果快照）",
            "counts": {"known_entities": len(debug.get("known_entities") or []),
                       "mapped_entities": sum(bool(row.get("kg_entity_ids")) for row in mappings)},
            "output_ref": output_ref,
        })
    for item in debug.get("triples_debug") or []:
        constraint = f"T{item['query_triple_index']}"
        stage = item.get("stage")
        if stage in {"first_pass", "second_pass"}:
            events.append({
                "step": "variable_propagation" if stage == "second_pass" else "phase1_graph_retrieval",
                "status": "completed", "constraint_id": constraint,
                "message": f"{constraint}：{'变量绑定后的图检索' if stage == 'second_pass' else '锚点图检索'}结果；入选数为一致性处理后的保留数",
                "counts": {"candidate_topk": len(item.get("candidate_ids_topk") or []),
                           "selected_triples": len(item.get("llm_selected_triple_ids") or []),
                           "anchors": len(item.get("anchors") or [])},
                "output_ref": output_ref,
            })
        checked = item.get("checked")
        state = {"resolved": "已解决", "unresolved": "未解决"}.get(item.get("status"), "未记录")
        neff = item.get("neff")
        if checked is True:
            value = (str(neff) if neff is not None else "无有限值（检查时无候选）") if "neff" in item else "未记录"
            message = f"{constraint}：已执行充分性检查，Neff={value}；一致性处理后状态：{state}"
        elif checked is False:
            message = f"{constraint}：未执行充分性检查；最终状态：{state}"
        else:
            message = f"{constraint}：算法未记录充分性检查状态；最终状态：{state}"
        events.append({
            "step": "neff_gating", "status": "completed", "constraint_id": constraint,
            "message": message, "counts": {"sufficiency_checked": checked, "constraint_status": item.get("status")},
            "metrics": {"neff": neff} if checked is True and "neff" in item else {},
            "output_ref": output_ref,
        })
    for variable, item in (debug.get("variable_candidates") or {}).items():
        intersection = item.get("intersection_ids")
        events.append({
            "step": "variable_propagation", "status": "completed", "variable_id": variable,
            "message": f"{variable}：最终变量候选已输出" + ("；单约束候选，未执行跨约束求交" if intersection is None else "；已记录跨约束交集"),
            "counts": {"union_candidates": len(item.get("union_ids") or []),
                       "intersection_candidates": len(intersection) if intersection is not None else None,
                       "primary_candidates": len(item.get("primary_ranked") or [])},
            "output_ref": output_ref,
        })
    return events


def phase2_trace(items: list[dict[str, Any]], output_ref: str) -> list[dict[str, Any]]:
    events = []
    for item in items:
        resolved = item.get("triple_status") == "resolved"
        candidates = item.get("candidate_evidences") or []
        constraint = f"T{item['query_triple_index']}"
        branch = "KG 原始 evidence 收集" if resolved else "文本相似度初筛 → reranker 重排 → Top-K 保留"
        events.append({
            "step": "phase2_text_recovery", "status": "completed", "constraint_id": constraint,
            "branch_id": "kg_evidence" if resolved else "text_fallback",
            "message": f"{constraint}：{branch}，输出 {len(candidates)} 条证据" + ("；未找到可用证据" if not candidates else ""),
            "counts": {"kg_triples": len(item.get("kg_triples") or []), "candidate_evidences": len(candidates)},
            "reason_code": None if resolved else "UNRESOLVED_CONSTRAINT", "output_ref": output_ref,
            "warnings": ["该约束没有输出文本证据"] if not candidates else [],
        })
    return events
