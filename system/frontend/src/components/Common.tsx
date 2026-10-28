import type { Project } from "../types";
export function MetricCard({ label, value, hint, tone }: { label: string; value: string; hint: string; tone: string }) {
  return (
    <div className="metric-card">
      <div className={"metric-icon " + tone} />
      <div className="metric-body"><span>{label}</span><strong>{value}</strong><small>{hint}</small></div>
    </div>
  );
}

export function ProjectCard({ project, onEnter }: { project: Project; onEnter: () => void }) {
  const ready = project.status === "ready";
  return (
    <article className="project-card">
      <div className="project-icon">▣</div>
      <div className="project-main">
        <div className="project-title-row"><h3>{project.name}</h3><span className={"status-pill " + (ready ? "ready" : "empty")}>{ready ? "健康" : "待构建"}</span></div>
        <p>{project.description || "尚未添加项目描述"}</p>
        <div className="project-meta"><span>文本层 · 来源可管理</span><span>KG 层 · 支持版本构建</span></div>
      </div>
      <button type="button" className="link-button" onClick={onEnter}>进入项目 <span>→</span></button>
    </article>
  );
}

export function StatusLine({ label, value, active }: { label: string; value: string; active: boolean }) {
  return <div className="status-line"><span className={"status-dot " + (active ? "connected" : "disconnected")} /><span className="status-label">{label}</span><span className="status-value">{value}</span></div>;
}


