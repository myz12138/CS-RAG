import type { Workspace } from "../hooks/useWorkspace";
import { Icon } from "../components/Icon";
import { KnowledgeGraphExplorer } from "../components/KnowledgeGraphExplorer";
import { formatTime, formatBytes } from "../utils/format";
import type { LibraryLanding } from "../types/workspace";
import "./overview.css";

export default function OverviewView({ workspace: w, onCreateProject, onOpenLibrary }: {
  workspace: Workspace; onCreateProject: () => void; onOpenLibrary: (landing: LibraryLanding) => void;
}) {
  const version = w.versions.find(item => item.version_id === w.selectedVersionId);
  const counts = w.versionDetail?.counts;
  const versionSources = version ? w.sources.filter(source => version.source_snapshot.includes(source.source_id)) : w.sources;
  const tasks = w.tasks.filter(task => task.payload.project_id === w.selectedProjectId).slice(0, 4);
  const canQuery = version?.status === "ready";
  return <div className="overview-workbench overview-organized">
    <div className="overview-commandbar">
      <div className="overview-version-state">
        <span className={"overview-version-state-label " + (version ? "ready" : "")}><i/>{version ? "v" + version.version_no + " · 已发布" : "尚无已发布版本"}</span>
        {!!counts?.kg_conflicts && <button onClick={() => onOpenLibrary("summary")}>{counts.kg_conflicts} 条多值提醒<Icon name="arrow" size={14}/></button>}
      </div>
      <div className="overview-primary-actions">
        <button className="secondary-button" onClick={() => w.projects.length ? onOpenLibrary("sources") : onCreateProject()}><Icon name="upload" size={17}/>{w.projects.length ? "管理资料" : "创建知识库"}</button>
        <button className="primary-button" onClick={() => w.setActiveNav("多跳问答")} disabled={!canQuery}><Icon name="search" size={17}/>向知识库提问</button>
      </div>
    </div>

    <section className="overview-asset-grid" aria-label="当前版本知识资产">
      <button onClick={() => onOpenLibrary("sources")} disabled={!w.selectedProjectId}><span><Icon name="file" size={18}/>来源资料</span><strong>{versionSources.length}<small>份</small></strong><Icon name="arrow" size={15}/></button>
      <button onClick={() => onOpenLibrary("text")} disabled={!version}><span><Icon name="layers" size={18}/>文本条目</span><strong>{counts?.text_entries ?? "—"}<small>条</small></strong><Icon name="arrow" size={15}/></button>
      <div><span><Icon name="graph" size={18}/>图谱实体</span><strong>{counts?.kg_entities ?? "—"}<small>个</small></strong></div>
      <button onClick={() => onOpenLibrary("kg")} disabled={!version}><span><Icon name="graph" size={18}/>图谱关系</span><strong>{counts?.kg_triples ?? "—"}<small>条</small></strong><Icon name="arrow" size={15}/></button>
    </section>

    <KnowledgeGraphExplorer key={w.selectedProjectId + ":" + w.selectedVersionId}
      projectId={w.selectedProjectId} versionId={w.selectedVersionId}
      contents={w.versionContents?.kg ?? null} sources={versionSources} loading={w.knowledgeVersionLoading}
      onOpenLibrary={kind => onOpenLibrary(version ? kind : "sources")}/>

    <section className="overview-support-grid">
      <div className="overview-support-panel">
        <header><h2>来源资料</h2><button className="link-button" onClick={() => onOpenLibrary("sources")} disabled={!w.selectedProjectId}>管理资料<Icon name="arrow" size={15}/></button></header>
        {versionSources.length ? <div className="overview-source-list">{versionSources.slice(0, 4).map(source => <div key={source.source_id}><Icon name="file" size={19}/><div><strong>{source.original_name}</strong><span>{source.source_format.toUpperCase()} · {formatBytes(source.size_bytes)}</span></div></div>)}</div> : <div className="overview-empty">导入资料后，在这里查看当前知识来源</div>}
        {versionSources.length > 4 && <button className="link-button" onClick={() => onOpenLibrary("sources")}>查看全部 {versionSources.length} 份资料</button>}
      </div>
      <div className="overview-support-panel">
        <header><h2>最近运行</h2><button className="link-button" onClick={() => w.setActiveNav("任务")}>任务中心<Icon name="arrow" size={15}/></button></header>
        <div className="overview-recent-list">{tasks.map(task => <button key={task.task_id} onClick={() => typeof task.payload.query_id === "string" ? void w.handleOpenQuery(task.payload.query_id) : w.setActiveNav("任务")}><span className={"overview-task-dot " + task.status}/><div><strong>{task.task_type === "knowledge_base_build" ? "知识库构建" : "检索问答"}</strong><span>{task.message || "等待执行"}</span></div><time>{formatTime(task.updated_at)}</time><Icon name="arrow" size={15}/></button>)}</div>
        {!tasks.length && <div className="overview-empty">当前项目暂无运行记录</div>}
      </div>
    </section>
  </div>;
}


