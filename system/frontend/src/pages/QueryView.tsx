import { errorMessage } from "../utils/messages";
import { useEffect, useRef, useState } from "react";
import type { Project, VersionSummary, QueryRecord, QueryEvent, QueryConstraintDetail, QueryEvidenceDetail, QueryConfig } from "../types";
import { stepLabel, stringValue, evidenceSummary, candidateText, constraintCheckSummary, formatTime } from "../utils/format";
export default function QueryView({
  projects,
  selectedProjectId,
  versions,
  selectedVersionId,
  question,
  running,
  record,
  events,
  taskProgress,
  taskMessage,
  selectedConstraintId,
  constraintDetail,
  constraintLoading,
  evidenceDetail,
  queryConfig,
  traceLevel,
  onProjectChange,
  onVersionChange,
  onQuestionChange,
  onTraceLevelChange,
  onSubmit,
  onConstraintSelect,
  onEvidenceSelect,
}: {
  projects: Project[];
  selectedProjectId: string;
  versions: VersionSummary[];
  selectedVersionId: string;
  question: string;
  running: boolean;
  record: QueryRecord | null;
  events: QueryEvent[];
  taskProgress: number;
  taskMessage: string;
  selectedConstraintId: string | null;
  constraintDetail: QueryConstraintDetail | null;
  constraintLoading: boolean;
  evidenceDetail: QueryEvidenceDetail | null;
  queryConfig: QueryConfig | null;
  traceLevel: "summary" | "detailed";
  onProjectChange: (projectId: string) => void;
  onVersionChange: (versionId: string) => void;
  onQuestionChange: (question: string) => void;
  onTraceLevelChange: (level: "summary" | "detailed") => void;
  onSubmit: () => void;
  onConstraintSelect: (constraintId: string) => void;
  onEvidenceSelect: (evidenceId: string) => void;
}) {
  const [view, setView] = useState("answer");
  const [eventStage, setEventStage] = useState("all");
  const [eventStatus, setEventStatus] = useState("all");
  const logRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => { setView(record && ["pending", "running"].includes(record.status) ? "process" : "answer"); setEventStage("all"); setEventStatus("all"); }, [record?.query_id]);
  const visibleEvents = events.filter(event => (eventStage === "all" || event.step === eventStage || (eventStage === "phase1_graph_retrieval" && ["neff_gating", "variable_propagation"].includes(event.step))) && (eventStatus === "all" || event.status === eventStatus));
  if (projects.length === 0) {
    return <div className="module-placeholder panel"><div className="empty-icon">⌁</div><h2>先创建一个项目</h2><p>问答会固定使用项目当前的知识库版本。</p></div>;
  }

  const project = projects.find((item) => item.project_id === selectedProjectId) ?? projects[0];
  const readyVersions = versions.filter((version) => version.status === "ready");
  const selectedVersion = readyVersions.find((version) => version.version_id === selectedVersionId);
  const canQuery = project.status === "ready" && Boolean(selectedVersion);
  const stages = [
    { key: "query_plan", label: "规划" },
    { key: "phase1_graph_retrieval", label: "图检索" },
    { key: "phase2_text_recovery", label: "证据收集" },
    { key: "answer_generation", label: "生成答案" },
  ];
  const currentStep = record?.current_step ?? (running ? "query_plan" : "");
  const answered = record?.status === "completed";
  const activeEvent = events[events.length - 1] ?? null;
  const activeStageLabel = answered ? "全部完成" : record?.status === "failed" ? "执行失败" : activeEvent ? stepLabel(activeEvent.step) : currentStep ? stepLabel(currentStep) : "等待启动";
  const activeConfig = record?.metadata.query_config && typeof record.metadata.query_config === "object" ? record.metadata.query_config as QueryConfig : queryConfig;
  const candidateEvidenceCount = record?.evidence.reduce((total, item) => total + (Array.isArray(item.candidate_evidences) ? item.candidate_evidences.length : 0), 0) ?? 0;
  // Keep an already-open inspector in sync with stage results arriving via polling.
  const selectedIndex = selectedConstraintId ? Number(selectedConstraintId.slice(1)) : -1;
  const debugSnapshot = record?.metadata.phase1_debug as { triples_debug?: Array<Record<string, unknown>> } | undefined;
  const liveConstraintDetail: QueryConstraintDetail | null = record && selectedConstraintId ? {
    constraint_id: selectedConstraintId,
    query_triple: record.query_plan[selectedIndex] ?? null,
    phase1_debug: (debugSnapshot?.triples_debug ?? []).filter(item => item.query_triple_index === selectedIndex),
    evidence: record.evidence.filter(item => item.query_triple_index === selectedIndex),
  } : constraintDetail;

  return (
    <div className="query-layout">
      <section className="panel query-compose">
        <div className="query-compose-head">
          <div>
            
            <h2>向知识库提问</h2>
          </div>
          <span className={"query-status " + (running ? "running" : answered ? "done" : "idle")}>{running ? "处理中" : answered ? "已完成" : "待提问"}</span>
        </div>
        <label className="field-label" htmlFor="query-project">知识库项目</label>
        <select id="query-project" className="project-select query-project-select" value={project.project_id} onChange={(event) => onProjectChange(event.target.value)} disabled={running}>
          {projects.map((item) => <option key={item.project_id} value={item.project_id}>{item.name}</option>)}
        </select>
        <label className="field-label" htmlFor="query-version">使用知识库版本</label>
        <select id="query-version" className="project-select query-project-select" value={selectedVersionId} onChange={(event) => onVersionChange(event.target.value)} disabled={running || readyVersions.length === 0}>
          {readyVersions.length === 0 ? <option value="">暂无可查询版本</option> : readyVersions.map((version) => <option key={version.version_id} value={version.version_id}>v{version.version_no} · {formatTime(version.updated_at)}</option>)}
        </select>
        <label className="field-label" htmlFor="query-trace-level">过程展示</label>
        <select id="query-trace-level" className="project-select query-project-select" value={traceLevel} onChange={(event) => onTraceLevelChange(event.target.value as "summary" | "detailed")} disabled={running}>
          <option value="detailed">详细：显示中间结果</option>
          <option value="summary">摘要：只显示阶段结果</option>
        </select>
        <textarea
          className="query-input"
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") onSubmit(); }}
          placeholder="输入一个需要多步推理的问题…"
          disabled={running || !canQuery}
        />
        {!canQuery && <div className="query-hint warning">请选择一个已发布的知识库版本；不上传新文本也可以直接使用已有版本</div>}
        <div className="query-compose-foot">
          <span className="query-hint">Ctrl / ⌘ + Enter 执行</span>
          <button className="primary-button" onClick={onSubmit} disabled={running || !canQuery || !question.trim()}>{running ? "检索中…" : "执行检索"}</button>
        </div>
      </section>

      {record && <div className="query-run-ribbon" role="status"><span className={"running-dot " + record.status}/><div><strong>{record.status === "completed" ? "本次问答已完成" : record.status === "failed" ? "本次问答未完成" : stepLabel(currentStep)}</strong><span>{taskMessage || activeEvent?.message || "等待服务反馈"}</span></div><span>{Math.round((answered ? 1 : taskProgress) * 100)}%</span><button className="link-button" onClick={() => setView("process")}>查看过程 →</button></div>}
      <div className="section-tabs query-details-tabs" role="tablist" aria-label="问答结果分类"><button role="tab" aria-selected={view === "answer"} className={view === "answer" ? "active" : ""} onClick={() => setView("answer")}>回答与告警</button><button role="tab" aria-selected={view === "evidence"} className={view === "evidence" ? "active" : ""} onClick={() => setView("evidence")}>约束与证据<small>{record?.query_plan.length ?? 0}</small></button><button role="tab" aria-selected={view === "process"} className={view === "process" ? "active" : ""} onClick={() => setView("process")}>运行过程<small>{events.length}</small></button></div>
      <section className="panel query-answer-panel" hidden={view !== "answer"}>
        <div className="panel-header compact-header"><div><h2>回答与告警</h2></div>{record?.version_id && <span className="version-chip">版本 {record.version_id.slice(0, 8)}</span>}</div>
        {!record ? <div className="query-empty"><span>⌁</span><p>提交问题后，这里会显示答案</p></div> : record.status === "failed" ? <div className="query-error"><strong>{record.error_code || "QUERY_FAILED"}</strong><p>{errorMessage(record.error_message, "查询执行失败")}</p></div> : <>
          <div className={"answer-box " + (answered ? "answered" : "pending")}><span className="answer-label">{answered ? "最终答案" : "当前状态"}</span><p>{record.answer || metadataString(record.metadata, "answer_preview") || taskMessage || "正在等待算法阶段返回…"}</p></div>
          {record.warnings.length > 0 && <div className="warning-list">{record.warnings.map((warning) => <span key={warning}>! {warning}</span>)}</div>}
        </>}
      </section>

      <section className="panel query-trace-panel" hidden={view !== "process"}>
        <div className="panel-header compact-header"><div><h2>算法过程</h2></div><span className="trace-progress">{Math.round((answered ? 1 : taskProgress) * 100)}%</span></div>
        <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(100, (answered ? 1 : taskProgress) * 100))}%` }} /></div>
        <div className="query-live-summary">
          <div className={"query-live-focus " + (running ? "active" : answered ? "done" : "idle")}>
            <span className="live-pulse" />
            <div><span>当前阶段 · {activeStageLabel}</span><strong>{activeEvent?.message || taskMessage || "等待算法反馈"}</strong><small>{activeEvent ? `事件 #${activeEvent.seq} · ${activeEvent.elapsed_ms ? `${activeEvent.elapsed_ms} ms` : answered ? "已完成" : record?.status === "failed" ? "已停止" : "进行中"}` : "提交后会在这里显示真实执行状态"}</small></div>
          </div>
          <div className="query-signal-list">
            <QuerySignal label="事件" value={String(events.length)} />
            <QuerySignal label="约束" value={String(record?.query_plan.length ?? 0)} />
            <QuerySignal label="候选" value={String(candidateEvidenceCount)} />
          </div>
        </div>
        {activeConfig && <details className="run-config-details"><summary>本次运行参数</summary><div className="query-config-strip"><span>规划重试 {activeConfig.planner_max_retries}</span><span>实体候选 {activeConfig.top_k_entity}</span><span>关系候选 {activeConfig.relation_k}</span><span>Neff {activeConfig.neff_threshold}</span><span>原文候选 {activeConfig.top_k_unresolved}</span><span>答案约束 {activeConfig.prompt_max_triples}</span><small>{record?.metadata.query_config ? "本次提交的系统配置" : "当前默认配置 · 后续查询生效"}</small></div></details>}
        <div className="stage-line">{stages.map((stage) => {
          const event = events.find((item) => item.step === stage.key && item.status === "completed");
          const active = currentStep === stage.key;
          return <div className={"stage-node " + (event ? "completed" : active ? "active" : "")} key={stage.key}><span className="stage-dot" /><span>{stage.label}</span></div>;
        })}</div>
        <QueryStageCards stages={stages} events={events} currentStep={currentStep} record={record} onInspect={stage => { setEventStage(stage); setEventStatus("all"); logRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }} />
        <div className="trace-toolbar" ref={logRef}><div><h3>执行日志</h3><span>{visibleEvents.length} / {events.length} 条事件</span></div><label>阶段<select aria-label="筛选日志阶段" value={eventStage} onChange={event => setEventStage(event.target.value)}><option value="all">全部阶段</option>{stages.map(stage => <option value={stage.key} key={stage.key}>{stage.label}</option>)}{[...new Set(events.map(event => event.step))].filter(step => !stages.some(stage => stage.key === step) && !["neff_gating", "variable_propagation"].includes(step)).map(step => <option value={step} key={step}>{stepLabel(step)}</option>)}</select></label><label>状态<select aria-label="筛选日志状态" value={eventStatus} onChange={event => setEventStatus(event.target.value)}><option value="all">全部状态</option><option value="running">执行中</option><option value="completed">已完成</option><option value="failed">失败</option></select></label></div>
        {visibleEvents.length === 0 ? <div className="trace-empty">{events.length ? "当前筛选下没有事件" : "执行后显示阶段日志与统计"}</div> : <div className="trace-list">{visibleEvents.map((event) => <div className="trace-row" key={`${event.query_id}-${event.seq}`}><span className={"trace-marker " + event.status} /><div><strong>{stepLabel(event.step)}</strong><p>{event.message}</p>{eventFacts(event).length > 0 && <div className="trace-facts">{eventFacts(event).map((fact) => <span key={fact}>{fact}</span>)}</div>}{event.warnings.length > 0 && <div className="trace-warning">! {event.warnings.join("；")}</div>}<details className="trace-event-data"><summary>完整事件数据</summary><pre>{JSON.stringify(event, null, 2)}</pre></details></div><span className="trace-seq">#{event.seq}</span></div>)}</div>}
        {record && <details className="execution-data"><summary>本次执行参数与中间数据</summary>{["query_config", "phase1_debug", "phase1_evidence", "phase1_support", "text_index_recall", "model_input", "artifacts"].map(key => <details key={key}><summary>{({ query_config: "实际运行参数", phase1_debug: "图检索与变量调试", phase1_evidence: "图证据", phase1_support: "约束支持组", text_index_recall: "文本索引召回", model_input: "实际生成输入", artifacts: "中间产物引用" } as Record<string, string>)[key]}</summary><pre>{typeof record.metadata[key] === "string" ? String(record.metadata[key]) : JSON.stringify(record.metadata[key] ?? "未记录", null, 2)}</pre></details>)}</details>}
      </section>

      <section className="panel query-evidence-panel" hidden={view !== "evidence"}>
        <div className="panel-header compact-header"><div><h2>查询计划与原文证据</h2></div><span className="count-badge">{record?.query_plan.length ?? 0}</span></div>
        {!record || record.query_plan.length === 0 ? <div className="trace-empty">查询计划生成后显示约束与证据</div> : <div className="constraint-list">{record.query_plan.map((item, index) => { const constraintId = `T${index}`; return <button className={"constraint-card " + (selectedConstraintId === constraintId ? "selected" : "")} key={`constraint-${index}`} onClick={() => onConstraintSelect(constraintId)}><span className="constraint-id">{constraintId}</span><div><strong>{stringValue(item, "head")} · {stringValue(item, "relation")} · {stringValue(item, "tail")}</strong><p>{evidenceSummary(record.evidence.find(evidence => evidence.query_triple_index === index))}</p></div><span className="constraint-arrow">›</span></button>; })}</div>}
        {selectedConstraintId && <ConstraintDetailPanel detail={liveConstraintDetail} loading={constraintLoading} evidenceDetail={evidenceDetail} onEvidenceSelect={onEvidenceSelect} />}
      </section>
    </div>
  );
}

function ConstraintDetailPanel({ detail, loading, evidenceDetail, onEvidenceSelect }: { detail: QueryConstraintDetail | null; loading: boolean; evidenceDetail: QueryEvidenceDetail | null; onEvidenceSelect: (evidenceId: string) => void }) {
  if (loading) return <div className="constraint-detail loading-detail">正在读取该约束的图检索与证据结果…</div>;
  if (!detail) return <div className="constraint-detail loading-detail">暂时没有可展示的约束详情</div>;
  const candidates = detail.evidence.flatMap((item) => Array.isArray(item.candidate_evidences) ? item.candidate_evidences : []);
  return <div className="constraint-detail">
    <div className="detail-head"><div><span className="eyebrow">{detail.constraint_id} 详情</span><h3>中间结果</h3></div><span className="detail-count">图检索 {detail.phase1_debug.length} 条记录 · 入选证据 {candidates.length} 条</span></div>
    {detail.query_triple && <div className="detail-triple"><span>{stringValue(detail.query_triple, "head")}</span><b>{stringValue(detail.query_triple, "relation")}</b><span>{stringValue(detail.query_triple, "tail")}</span></div>}
    {Array.isArray(detail.query_triple?.relation_variants) && <p className="constraint-check-summary">关系变体：{detail.query_triple.relation_variants.join(" · ")}</p>}
    {detail.phase1_debug.map((item, index) => <p className="constraint-check-summary" key={index}>{constraintCheckSummary(item)}</p>)}
    {detail.evidence.map((item, index) => <p className="constraint-check-summary" key={`branch-${index}`}>{evidenceSummary(item)}</p>)}
    {candidates.length > 0 ? <div className="candidate-list">{candidates.map((candidate, index) => { const evidenceId = `${detail.constraint_id}-E${index}`; return <button className="candidate-row" key={evidenceId} onClick={() => onEvidenceSelect(evidenceId)}><span>{evidenceId}</span><p>{candidateText(candidate)}</p><i>›</i></button>; })}</div> : <div className="detail-empty">该约束暂未输出可用文本证据</div>}
    <details><summary>图检索与证据原始字段</summary><pre>{JSON.stringify({ phase1: detail.phase1_debug, evidence: detail.evidence }, null, 2)}</pre></details>
    {evidenceDetail && <div className="evidence-focus"><span>{evidenceDetail.evidence_id}</span><p>{candidateText(evidenceDetail.item)}</p><details><summary>完整证据字段</summary><pre>{JSON.stringify(evidenceDetail.item, null, 2)}</pre></details></div>}
  </div>;
}

function QuerySignal({ label, value }: { label: string; value: string }) {
  return <div className="query-signal"><span>{label}</span><strong>{value}</strong></div>;
}

function QueryStageCards({ stages, events, currentStep, record, onInspect }: { stages: Array<{ key: string; label: string }>; events: QueryEvent[]; currentStep: string; record: QueryRecord | null; onInspect: (stage: string) => void }) {
  return <div className="query-stage-cards">{stages.map((stage) => {
    const stageEvents = events.filter((event) => event.step === stage.key);
    const latest = stageEvents[stageEvents.length - 1] ?? null;
    const summaryEvent = stageSummaryEvent(stage.key, stageEvents) ?? latest;
    const failed = latest?.status === "failed";
    const completed = (stage.key === "query_plan" && Boolean(record?.query_plan.length)) || (stage.key === "phase1_graph_retrieval" && Array.isArray(record?.metadata.phase1_evidence)) || (stage.key === "phase2_text_recovery" && Array.isArray(record?.metadata.evidence)) || stageEvents.some((event) => event.status === "completed") || (stage.key === "answer_generation" && record?.status === "completed");
    const active = !failed && !completed && currentStep === stage.key;
    const status = failed ? "failed" : completed ? "completed" : active ? "active" : "pending";
    const output = stageOutput(stage.key, record, summaryEvent);
    return <article className={`query-stage-card ${status}`} key={stage.key}>
      <div className="query-stage-card-head"><span className="stage-card-index">{stages.indexOf(stage) + 1}</span><div><strong>{stage.label}</strong><small>{stageStatusLabel(status)}</small></div><span className="stage-card-state" /></div>
      <div className="query-stage-io"><div><span>输入</span><p>{stageInput(stage.key)}</p></div><div><span>输出</span><p>{output}</p></div></div>
      <div className="query-stage-message">{stageDescription(stage.key)}</div>
      {latest?.message && <div className="query-stage-message">{latest.message}</div>}
      <button className="link-button stage-inspect-button" disabled={!stageEvents.length} onClick={() => onInspect(stage.key)}>查看阶段日志 →</button>
    </article>;
  })}</div>;
}

function stageSummaryEvent(stage: string, events: QueryEvent[]) {
  const requiredCounts: Record<string, string[]> = {
    phase1_graph_retrieval: ["evidence_triples", "support_groups"],
    phase2_text_recovery: ["resolved_constraints", "candidate_evidences"],
    answer_generation: ["answer_length"],
  };
  const keys = requiredCounts[stage];
  if (!keys) return events[events.length - 1] ?? null;
  return [...events].reverse().find((event) => keys.every((key) => Object.prototype.hasOwnProperty.call(event.counts, key))) ?? null;
}

function stageDescription(stage: string) {
  const descriptions: Record<string, string> = {
    query_plan: "调用规划模型，解析查询元组与关系变体。",
    phase1_graph_retrieval: "实体映射、关系筛选与重排，随后进行 Neff 检查和变量一致性处理。内部结果在阶段结束后展示，不表示每个约束都执行了所有步骤。",
    phase2_text_recovery: "已解决约束收集 KG evidence；未解决约束执行文本相似度初筛与重排。具体分支以约束结果为准。",
    answer_generation: "按查询元组组织入选文本证据，调用模型生成答案。",
  };
  return descriptions[stage] ?? "";
}

function stageInput(stage: string) {
  const inputs: Record<string, string> = {
    query_plan: "用户问题",
    phase1_graph_retrieval: "原子约束 + 知识图谱",
    phase2_text_recovery: "图检索结果 + 版本原文",
    answer_generation: "问题 + 筛选后的原文证据",
  };
  return inputs[stage] ?? "上一阶段结果";
}

function stageOutput(stage: string, record: QueryRecord | null, event: QueryEvent | null) {
  const counts = event?.counts ?? {};
  const count = (key: string) => typeof counts[key] === "number" ? String(counts[key]) : "—";
  if (stage === "query_plan") return record?.query_plan.length ? `${record.query_plan.length} 条原子约束` : "等待查询规划";
  if (stage === "phase1_graph_retrieval") return event ? `图证据 ${count("evidence_triples")} 条 · 支持组 ${count("support_groups")} 组` : "等待图检索";
  if (stage === "phase2_text_recovery") return event ? `图约束已解决 ${count("resolved_constraints")} 条 · 入选证据 ${count("candidate_evidences")} 条` : "等待证据收集";
  if (stage === "answer_generation") {
    const answer = record?.answer || metadataString(record?.metadata, "answer_preview");
    return answer ? `已生成 · ${answer.length} 字` : "等待答案生成";
  }
  return "等待执行";
}

function stageStatusLabel(status: string) {
  return status === "active" ? "进行中" : status === "completed" ? "已完成" : status === "failed" ? "失败" : "待执行";
}

function metadataString(metadata: Record<string, unknown> | undefined, key: string) {
  const value = metadata?.[key];
  return typeof value === "string" ? value : "";
}

function eventFacts(event: QueryEvent) {
  const facts: string[] = [];
  for (const [key, value] of Object.entries(event.counts)) {
    facts.push(`${eventFactLabel(key)} ${compactEventValue(value)}`);
  }
  for (const [key, value] of Object.entries(event.metrics)) {
    if (key === "progress") continue;
    facts.push(`${eventFactLabel(key)} ${compactEventValue(value)}`);
  }
  return facts.slice(0, 3);
}

function eventFactLabel(key: string) {
  const labels: Record<string, string> = {
    evidence_triples: "图证据",
    support_groups: "支持组",
    constraints: "约束",
    resolved_constraints: "图约束已解决",
    candidate_evidences: "入选证据",
    kg_triples: "KG 三元组",
    union_candidates: "并集候选",
    intersection_candidates: "交集候选",
    primary_candidates: "主候选",
    candidate_topk: "候选 TopK",
    selected_triples: "入选三元组",
    anchors: "锚点",
    known_entities: "已知实体",
    mapped_entities: "已映射实体",
    text_index_recall_status: "文本索引状态",
    answer_length: "答案字数",
    sufficiency_checked: "充分性检查",
    constraint_status: "最终约束状态",
    neff: "Neff",
  };
  return labels[key] ?? key;
}

function compactEventValue(value: unknown) {
  if (value === true) return "已执行";
  if (value === false) return "未执行";
  if (value === "resolved") return "已解决";
  if (value === "unresolved") return "未解决";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value.length > 18 ? `${value.slice(0, 18)}…` : value;
  return String(value ?? "—");
}


