import type { ReactNode } from "react";
import type { Workspace } from "../hooks/useWorkspace";
import { Icon, type IconName } from "./Icon";
import "./workspaceShell.css";
const navigation: Array<{ id: string; title: string; description: string; icon: IconName }> = [
  { id: "总览", title: "知识总览", description: "探索知识关系与原文证据", icon: "overview" },
  { id: "知识库", title: "资料与知识库", description: "导入资料、构建索引与管理版本", icon: "library" },
  { id: "多跳问答", title: "检索与问答", description: "追踪约束、图检索与证据收集", icon: "search" },
  { id: "任务", title: "任务中心", description: "查看构建和查询的实际运行记录", icon: "tasks" },
  { id: "评测", title: "评测与分析", description: "查看真实问答结果与证据覆盖", icon: "chart" },
  { id: "设置", title: "系统配置", description: "模型连接、构建参数与检索参数", icon: "settings" },
];
export function WorkspaceShell({ workspace: w, children, onCreateProject }: { workspace: Workspace; children: ReactNode; onCreateProject: () => void }) {
  const page = navigation.find(item => item.id === w.activeNav) ?? navigation[0];
  const locked = w.building || w.queryRunning || Boolean(w.queryRecord && ["pending", "running"].includes(w.queryRecord.status));
  return <div className="app-shell">
    <a className="skip-link" href="#workspace-content">跳转到工作区</a>
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Icon name="layers" size={27} /></span><div className="brand-name">CS-RAG</div></div>
      <nav className="nav-list" aria-label="主导航">{navigation.map(item => <button className={"nav-item " + (w.activeNav === item.id ? "active" : "")} key={item.id} onClick={() => w.setActiveNav(item.id)} aria-current={w.activeNav === item.id ? "page" : undefined}><Icon name={item.icon} /><span>{item.title}</span>{item.id === "任务" && w.runningTasks > 0 ? <small className="nav-count">{w.runningTasks}</small> : w.activeNav === item.id ? <i className="nav-active-dot" /> : null}</button>)}</nav>
      <div className="sidebar-footer"><span className={"status-dot " + w.connection} /><span>{w.connection === "connected" ? "服务连接正常" : w.connection === "checking" ? "正在连接服务" : "服务未连接"}</span><small>{w.health?.deployment_mode === "remote" ? "远程部署" : "本地部署"}</small></div>
    </aside>
    <main className="main-area">
      <header className="topbar workspace-header"><div className="topbar-title"><h1>{page.title}</h1><p>{page.description}</p></div>
        <div className="workspace-context"><label>知识库<select aria-label="当前知识库项目" value={w.selectedProjectId} disabled={!w.projects.length || locked} onChange={e => w.setSelectedProjectId(e.target.value)}>{!w.projects.length && <option value="">尚未创建项目</option>}{w.projects.map(project => <option value={project.project_id} key={project.project_id}>{project.name}</option>)}</select></label><label>知识版本<select aria-label="当前知识库版本" value={w.selectedVersionId} disabled={!w.versions.some(v => v.status === "ready") || locked} onChange={e => w.setSelectedVersionId(e.target.value)}>{!w.versions.some(v => v.status === "ready") && <option value="">暂无已发布版本</option>}{w.versions.filter(v => v.status === "ready").map(version => <option key={version.version_id} value={version.version_id}>v{version.version_no} · {version.source_snapshot.length} 个来源</option>)}</select></label><button className="icon-button" aria-label="刷新系统状态" onClick={() => void w.refresh()} disabled={w.connection === "checking"}><Icon name="refresh" /></button><button className="secondary-button new-project-button" onClick={onCreateProject}><Icon name="plus" size={15}/>新建项目</button></div>
      </header>
      <div id="workspace-content" className={"content-wrap page-" + w.activeNav} tabIndex={-1}>
        {w.message && <div className="notice" role="status"><span>{w.message}</span><button aria-label="关闭提示" onClick={() => w.setMessage("")}><Icon name="close" size={15}/></button></div>}
        {children}
      </div>
    </main>
  </div>;
}
