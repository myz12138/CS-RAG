import { statusLabel } from "../utils/messages";
import type { Project, EvaluationSummary } from "../types";
import { formatMetric } from "../utils/format";
import { MetricCard } from "../components/Common";
export default function EvaluationView({ projects, selectedProjectId, summary, onProjectChange, onOpenQuery }: { projects: Project[]; selectedProjectId: string; summary: EvaluationSummary | null; onProjectChange: (projectId: string) => void; onOpenQuery: (queryId: string) => void }) {
  if (projects.length === 0) return <div className="module-placeholder panel"><div className="empty-icon">▥</div><h2>先创建一个项目</h2><p>完成真实查询后，这里会汇总算法运行指标。</p></div>;
  const resolved = summary?.aggregate.resolved_triple_ratio;
  const supported = summary?.aggregate.answer_supported_by_evidence;
  const unsupported = summary?.aggregate.unsupported_answer_flag;
  return <div className="evaluation-layout">
    <section className="panel evaluation-head"><div><h2>评测与可靠性</h2><p>只统计 CS-RAG 真实执行产生的结果；没有标准答案时不推算 EM/F1。</p></div><select className="project-select evaluation-select" value={selectedProjectId} onChange={(event) => onProjectChange(event.target.value)}>{projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}</select></section>
    <section className="evaluation-metrics"><MetricCard label="已完成查询" value={String(summary?.completed_count ?? "—")} hint={`样本 ${summary?.sample_count ?? "—"}`} tone="blue" /><MetricCard label="约束恢复率" value={formatMetric(resolved)} hint="来自 Phase2" tone="violet" /><MetricCard label="证据支持率" value={formatMetric(supported)} hint="启发式支持信号，非正确率" tone="green" /><MetricCard label="未支持告警" value={formatMetric(unsupported)} hint="越低越好" tone="orange" /></section>
    <section className="panel evaluation-table"><div className="panel-header compact-header"><div><h2>最近运行</h2><p>点击记录，回看本次问答与证据链</p></div><span className="count-badge">{summary?.failed_count ?? 0} 失败</span></div>{!summary || summary.recent.length === 0 ? <div className="trace-empty">完成一次真实查询后，这里会出现运行记录</div> : <div className="evaluation-list">{summary.recent.map((item) => <button type="button" className="evaluation-row evaluation-row-button" key={item.query_id} onClick={() => onOpenQuery(item.query_id)}><span className={"evaluation-status " + item.status}>{statusLabel(item.status)}</span><div><strong>{item.question}</strong><p>{item.answer || item.error_code || "暂无答案"}</p></div><span className="evaluation-time">{new Date(item.updated_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}</span></button>)}</div>}</section>
    <section className="panel evaluation-note"><span className="eyebrow">指标口径</span><p>{summary?.metric_notes.answer_supported_by_evidence || "答案是否被检索证据支持"}；EM/F1 需要带标准答案的评测集，当前查询页不会伪造这两个指标。</p></section>
  </div>;
}


