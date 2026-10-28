import { errorMessage } from "../utils/messages";
import { useEffect, useRef, useState } from "react";
import type { BuildPreview, Project, Source, Task, VersionContents, VersionDetail, VersionSummary } from "../types";
import { formatBytes, formatTime } from "../utils/format";
import type { ProgressSnapshot } from "../types/workspace";
import { getVersionContents } from "../api/client";
import type { LibraryLanding } from "../types/workspace";
type KnowledgeDetailTab = "summary" | "text" | "kg";

export default function KnowledgeBaseView({
  initialLanding = "sources",
  projects,
  selectedProjectId,
  versions,
  selectedVersionId,
  sources,
  versionDetail,
  versionContents,
  buildPreview,
  uploading,
  building,
  buildTask,
  buildSnapshots,
  knowledgeVersionLoading,
  onProjectChange,
  onVersionChange,
  onUpload,
  onUseExisting,
  onPreview,
  onBuild,
  onOpenVersion,
}: {
  initialLanding?: LibraryLanding;
  projects: Project[];
  selectedProjectId: string;
  versions: VersionSummary[];
  selectedVersionId: string;
  sources: Source[];
  versionDetail: VersionDetail | null;
  versionContents: { text: VersionContents; kg: VersionContents } | null;
  buildPreview: BuildPreview | null;
  uploading: boolean;
  building: boolean;
  buildTask: Task | null;
  buildSnapshots: ProgressSnapshot[];
  knowledgeVersionLoading: boolean;
  onProjectChange: (projectId: string) => void;
  onVersionChange: (versionId: string) => void;
  onUpload: (files: FileList | null) => void;
  onUseExisting: () => void;
  onPreview: () => void;
  onBuild: () => void;
  onOpenVersion: (versionId: string) => void;
}) {
  const [workspaceTab, setWorkspaceTab] = useState<"sources" | "build" | "versions">("sources");
  const [inputMode, setInputMode] = useState<"text" | "existing">("text");
  const [page, setPage] = useState<"overview" | "detail">("overview");
  const [detailTab, setDetailTab] = useState<KnowledgeDetailTab>("summary");
  const [sourceSearch, setSourceSearch] = useState("");
  const [sourcePage, setSourcePage] = useState(0);
  useEffect(() => { setSourceSearch(""); setSourcePage(0); }, [selectedProjectId]);
  useEffect(() => { window.scrollTo({ top: 0, behavior: "auto" }); }, [workspaceTab, page]);

  useEffect(() => {
    const detail = ["summary", "text", "kg"].includes(initialLanding) && Boolean(selectedVersionId);
    setPage(detail ? "detail" : "overview");
    setWorkspaceTab(initialLanding === "build" ? "build" : initialLanding === "versions" ? "versions" : "sources");
    setDetailTab(detail ? initialLanding as KnowledgeDetailTab : "summary");
  }, [selectedProjectId]);

  if (projects.length === 0) {
    return <div className="module-placeholder panel"><div className="empty-icon">▤</div><h2>先创建一个项目</h2><p>项目是资料、图谱和版本的隔离边界。</p></div>;
  }

  const statusLabels: Record<string, string> = {
    blocked_no_input: "等待文本",
    blocked_kg_input: "不支持 KG 输入",
    blocked_graph_only_disabled: "请选择已有版本",
    blocked_model_not_configured: "等待模型配置",
    blocked_embedding_not_available: "等待 embedding 模型",
    ready_for_build: "可以构建",
  };
  const previewLabel = buildPreview ? statusLabels[buildPreview.overall_status] ?? buildPreview.overall_status : "预检中";
  const readyVersions = versions.filter((version) => version.status === "ready");
  const selectedVersion = readyVersions.find((version) => version.version_id === selectedVersionId);
  const detailVersionId = selectedVersion?.version_id ?? versionDetail?.version.version_id ?? readyVersions[0]?.version_id ?? "";
  const headerLabel = inputMode === "existing"
    ? selectedVersion ? `使用 v${selectedVersion.version_no}` : "选择已有版本"
    : previewLabel;
  const isBlocked = inputMode === "text"
    ? Boolean(buildPreview?.overall_status.startsWith("blocked"))
    : !selectedVersion;
  const matchedSources = sources.filter(source => source.original_name.toLowerCase().includes(sourceSearch.toLowerCase()));
  const sourcePageCount = Math.max(1, Math.ceil(matchedSources.length / 6));
  const currentSourcePage = Math.min(sourcePage, sourcePageCount - 1);
  const visibleSources = matchedSources.slice(currentSourcePage * 6, currentSourcePage * 6 + 6);
  const openDetails = (tab: KnowledgeDetailTab) => {
    if (!detailVersionId) return;
    setDetailTab(tab);
    setPage("detail");
    onOpenVersion(detailVersionId);
  };

  if (page === "detail") {
    const project = projects.find((item) => item.project_id === selectedProjectId);
    return (
      <KnowledgeBaseDetailView
        projectName={project?.name ?? "当前项目"}
        sources={sources}
        versions={readyVersions}
        selectedVersionId={selectedVersionId || detailVersionId}
        versionDetail={versionDetail}
        versionContents={versionContents}
        loading={knowledgeVersionLoading}
        tab={detailTab}
        onBack={() => setPage("overview")}
        onVersionOpen={onOpenVersion}
        onTabChange={setDetailTab}
      />
    );
  }

  return (
    <div className="kb-layout">
      <div className="workspace-intro"><div><h2>构建你的知识空间</h2></div><span className="scope-pill">{sources.length} 个来源 · {readyVersions.length} 个已发布版本</span></div>
      <div className="section-tabs" role="tablist" aria-label="知识库工作区"><button role="tab" aria-selected={workspaceTab === "sources"} className={workspaceTab === "sources" ? "active" : ""} onClick={() => setWorkspaceTab("sources")}>资料导入<small>{sources.length}</small></button><button role="tab" aria-selected={workspaceTab === "build"} className={workspaceTab === "build" ? "active" : ""} onClick={() => setWorkspaceTab("build")}>构建与进度{building && <i className="running-dot"/>}</button><button role="tab" aria-selected={workspaceTab === "versions"} className={workspaceTab === "versions" ? "active" : ""} onClick={() => setWorkspaceTab("versions")}>知识版本<small>{readyVersions.length}</small></button></div>
      <section className="panel kb-control-panel" hidden={workspaceTab !== "sources"}>
        <div className="kb-toolbar">
          <div>
            <span className="eyebrow">当前项目</span>
            <select className="project-select" value={selectedProjectId} onChange={(event) => onProjectChange(event.target.value)}>
              {projects.map((project) => <option key={project.project_id} value={project.project_id}>{project.name}</option>)}
            </select>
          </div>
          <span className={"build-state " + (isBlocked ? "blocked" : "ready")}>{headerLabel}</span>
        </div>
        <div className="input-mode-tabs" role="tablist" aria-label="知识库使用方式">
          <button className={inputMode === "text" ? "active" : ""} onClick={() => setInputMode("text")} role="tab" aria-selected={inputMode === "text"}>从文本构建</button>
          <button className={inputMode === "existing" ? "active" : ""} onClick={() => setInputMode("existing")} role="tab" aria-selected={inputMode === "existing"}>使用已有知识库</button>
        </div>
        {inputMode === "text" ? (
          <label className="upload-zone" htmlFor="source-upload">
            <span className="upload-orbit">＋</span>
            <span className="upload-title">导入文本，构建新版本</span>
            <span className="upload-hint">系统将从文本生成条目、KG 和文本索引</span>
            <input id="source-upload" type="file" multiple accept=".txt,.md,.markdown,.rst,.docx" onChange={(event) => { onUpload(event.target.files); event.currentTarget.value = ""; }} disabled={uploading} />
            <span className="secondary-button upload-button">{uploading ? "上传中…" : "选择文本"}</span>
          </label>
        ) : (
          <div className="existing-kb-choice">
            <div className="existing-kb-icon">▤</div>
            <div className="existing-kb-main">
              <strong>不上传新文本，直接使用已发布版本</strong>
              <span>选择版本后进入多跳问答，不会重新构建或修改知识库。</span>
              <select className="project-select existing-version-select" value={selectedVersionId} onChange={(event) => onVersionChange(event.target.value)} disabled={readyVersions.length === 0}>
                {readyVersions.length === 0 ? <option value="">暂无可查询版本</option> : readyVersions.map((version) => <option key={version.version_id} value={version.version_id}>v{version.version_no} · {formatTime(version.updated_at)}</option>)}
              </select>
              <button className="primary-button" onClick={onUseExisting} disabled={!selectedVersion}>进入多跳问答</button>
            </div>
          </div>
        )}
      </section>

      <section className="kb-layer-grid" hidden={workspaceTab !== "versions"}>
        <LayerCard title="当前版本 · 文本" icon="文" count={versionDetail?.version.version_id === detailVersionId ? versionDetail.counts.text_entries : null} unit="条文本" hint="查看正文与原文位置" tone="blue" disabled={!detailVersionId} onClick={() => openDetails("text")} />
        <LayerCard title="当前版本 · 知识图谱" icon="图" count={versionDetail?.version.version_id === detailVersionId ? versionDetail.counts.kg_triples : null} unit="条关系" hint="查看事实与关联证据" tone="violet" disabled={!detailVersionId} onClick={() => openDetails("kg")} />
      </section>

      <section className="panel source-panel" hidden={workspaceTab !== "sources"}>
        <div className="panel-header compact-header"><div><h2>已导入来源</h2><p>只记录原始输入；解析与版本构建在下一步进行</p></div><span className="count-badge">{sources.length}</span></div>
        {sources.length === 0 ? <div className="source-empty">暂无来源文件</div> : <>
          <label className="source-search"><span className="field-label">查找来源</span><input className="text-input" aria-label="按文件名查找来源" placeholder="输入文件名" value={sourceSearch} onChange={event => { setSourceSearch(event.target.value); setSourcePage(0); }}/></label>
          {!visibleSources.length && <div className="source-empty">没有匹配的来源文件</div>}
          <div className="source-list">{visibleSources.map((source) => <button type="button" className="source-row source-row-button" key={source.source_id} onClick={() => openDetails(source.source_kind === "kg" ? "kg" : "text")} disabled={!detailVersionId}><span className={"source-kind " + source.source_kind}>{source.source_kind === "kg" ? "历史图" : "文"}</span><div className="source-main"><strong>{source.original_name}</strong><span>{source.source_format.toUpperCase()} · {formatBytes(source.size_bytes)}{source.source_kind === "kg" ? " · 不参与新建" : ""}</span></div><span className={"parse-status " + (source.parse_status === "parsed" ? "parsed" : "")}>{source.parse_status === "parsed" ? "已解析" : "待解析"}</span></button>)}</div>
          {matchedSources.length > 6 && <div className="pagination"><span>{matchedSources.length} 个来源 · 第 {currentSourcePage + 1} / {sourcePageCount} 页</span><button className="secondary-button" disabled={currentSourcePage === 0} onClick={() => setSourcePage(currentSourcePage - 1)}>上一页</button><button className="secondary-button" disabled={currentSourcePage + 1 >= sourcePageCount} onClick={() => setSourcePage(currentSourcePage + 1)}>下一页</button></div>}
        </>}
      </section>

      <section className="panel kb-version-launcher" hidden={workspaceTab !== "versions"}>
        <div className="panel-header compact-header"><div><span className="eyebrow">知识库版本</span><h2>已发布版本</h2><p>进入详情页查看文本条目、三元组和冲突提醒</p></div><span className="count-badge">{readyVersions.length}</span></div>
        {readyVersions.length === 0 ? <div className="source-empty">暂无可查询版本，完成构建后会显示在这里</div> : <div className="kb-version-list">{readyVersions.map((version) => <button type="button" className={"kb-version-row " + (version.version_id === detailVersionId ? "active" : "")} key={version.version_id} onClick={() => { setDetailTab("summary"); setPage("detail"); onOpenVersion(version.version_id); }}><span className="kb-version-no">v{version.version_no}</span><div><strong>可查询版本</strong><small>{formatTime(version.updated_at)} · 来源 {version.source_snapshot.length}</small></div><span>查看详情 ›</span></button>)}</div>}
      </section>

      {workspaceTab === "build" && buildPreview && <section className={"panel preview-panel " + (isBlocked ? "preview-blocked" : "preview-ready")}><div className="preview-head"><div><span className="eyebrow">构建预检</span><h2>{previewLabel}</h2></div><span className="preview-count">{buildPreview.expected_llm_calls === null ? "LLM 按切分子块调用" : `预计模型调用 ${buildPreview.expected_llm_calls} 次`}</span></div>{buildPreview.blockers.map((blocker) => <div className="preview-message" key={blocker.code}>{blocker.message}</div>)}{buildPreview.actions.map((action) => <div className="preview-action" key={action}>{action}</div>)}<dl className="build-config-summary"><div><dt>文本与父级长度</dt><dd>子块 {buildPreview.chunking.token_limit} / 最小 {buildPreview.chunking.min_tokens} Token<br/>父级 {buildPreview.chunking.parent_token_limit} Token</dd></div><div><dt>分割与索引</dt><dd>联合切分 · 无重叠<br/>语义断点 {buildPreview.chunking.semantic_breakpoint_percentile}% · 每批 {buildPreview.chunking.index_batch_size} 条</dd></div><div><dt>嵌入模型 · {buildPreview.chunking.embedding_ready ? "已就绪" : "未就绪"}</dt><dd>{buildPreview.chunking.embedding_model || "未配置"}</dd></div></dl><div className="preview-footer">{buildPreview.overall_status === "ready_for_build" && <button className="primary-button" onClick={onBuild} disabled={building}>{building ? "构建中…" : "开始构建"}</button>}{buildPreview.overall_status === "blocked_no_input" && <button className="secondary-button" onClick={onPreview}>重新检查</button>}</div></section>}

      {workspaceTab === "build" && (buildTask || building) && <BuildProgressPanel task={buildTask} preview={buildPreview} snapshots={buildSnapshots} />}
      {workspaceTab === "sources" && <div className="workflow-next"><span>资料已就绪？预检将核对模型、分割参数与构建条件。</span><button className="primary-button" onClick={() => { setWorkspaceTab("build"); onPreview(); }}>下一步 · 构建知识库 →</button></div>}
      {workspaceTab === "build" && <div className="workflow-next"><button className="secondary-button" onClick={() => setWorkspaceTab("sources")}>← 返回资料</button><button className="secondary-button" onClick={() => { setWorkspaceTab("versions"); }}>查看已发布版本 →</button></div>}
    </div>
  );
}

function KnowledgeBaseDetailView({
  projectName,
  sources,
  versions,
  selectedVersionId,
  versionDetail,
  versionContents,
  loading,
  tab,
  onBack,
  onVersionOpen,
  onTabChange,
}: {
  projectName: string;
  sources: Source[];
  versions: VersionSummary[];
  selectedVersionId: string;
  versionDetail: VersionDetail | null;
  versionContents: { text: VersionContents; kg: VersionContents } | null;
  loading: boolean;
  tab: KnowledgeDetailTab;
  onBack: () => void;
  onVersionOpen: (versionId: string) => void;
  onTabChange: (tab: KnowledgeDetailTab) => void;
}) {
  const [selectedItem, setSelectedItem] = useState<Record<string, unknown> | null>(null);
  const [conflictPage, setConflictPage] = useState(0);
  const versionSourceNames = versionDetail
    ? versionDetail.version.source_snapshot
      .map((sourceId) => sources.find((source) => source.source_id === sourceId)?.original_name)
      .filter((name): name is string => Boolean(name))
    : [];

  useEffect(() => {
    setSelectedItem(null);
    setConflictPage(0);
  }, [tab, versionDetail?.version.version_id]);

  return (
    <div className="kb-detail-layout">
      <div className="kb-detail-toolbar">
        <button type="button" className="kb-back-button" onClick={onBack}>← 返回概览</button>
        <div><span className="eyebrow">{projectName}</span><h2>知识库详情</h2></div>
        <select className="project-select kb-detail-version-select" value={selectedVersionId} onChange={(event) => onVersionOpen(event.target.value)} disabled={versions.length === 0} aria-label="选择知识库版本">
          {versions.length === 0 ? <option value="">暂无版本</option> : versions.map((version) => <option key={version.version_id} value={version.version_id}>v{version.version_no} · {formatTime(version.updated_at)}</option>)}
        </select>
      </div>

      {loading ? <section className="panel kb-detail-loading"><div className="loading-orbit" /><span>正在加载版本内容…</span></section> : !versionDetail ? <section className="panel kb-detail-loading"><span>暂时没有可展示的知识库版本</span></section> : <>
        <section className="panel kb-detail-hero">
          <div className="panel-header compact-header"><div><span className="eyebrow">版本 v{versionDetail.version.version_no}</span><h2>知识库内容</h2><p>{versionDetail.version.status === "ready" ? "已发布，可用于多跳问答" : "当前版本尚未开放查询"}</p></div><span className={"status-pill " + (versionDetail.version.status === "ready" ? "ready" : "empty")}>{versionDetail.version.status === "ready" ? "可查询" : "图谱专用"}</span></div>
          <div className="version-counts"><div><span>文本条目</span><strong>{versionDetail.counts.text_entries}</strong></div><div><span>实体</span><strong>{versionDetail.counts.kg_entities}</strong></div><div><span>三元组</span><strong>{versionDetail.counts.kg_triples}</strong></div><div className={versionDetail.counts.kg_conflicts > 0 ? "has-conflict" : ""}><span>结构冲突</span><strong>{versionDetail.counts.kg_conflicts}</strong></div></div>
          <div className="kb-detail-tabs" role="tablist" aria-label="知识库详情分类"><button type="button" className={tab === "summary" ? "kb-detail-tab active" : "kb-detail-tab"} onClick={() => onTabChange("summary")} role="tab" aria-selected={tab === "summary"}>概览</button><button type="button" className={tab === "text" ? "kb-detail-tab active" : "kb-detail-tab"} onClick={() => onTabChange("text")} role="tab" aria-selected={tab === "text"}>文本条目</button><button type="button" className={tab === "kg" ? "kb-detail-tab active" : "kb-detail-tab"} onClick={() => onTabChange("kg")} role="tab" aria-selected={tab === "kg"}>图谱三元组</button></div>
        </section>

        {tab === "summary" ? <section className="panel version-panel detail-summary-panel"><div className="panel-header compact-header"><div><h2>选择要查看的内容</h2></div></div><div className="detail-summary-actions"><button type="button" className="secondary-button" onClick={() => onTabChange("text")}>查看文本条目</button><button type="button" className="secondary-button" onClick={() => onTabChange("kg")}>查看图谱三元组</button></div>{versionDetail.text_index && <div className="content-detail-meta"><div><span>向量条目</span><strong>{versionDetail.text_index.vector_count ?? versionDetail.counts.text_entries}</strong></div><div><span>向量维度</span><strong>{versionDetail.text_index.dimension ?? "—"}</strong></div><div><span>嵌入模型</span><strong>{versionDetail.text_index.embedding_model || "—"}</strong></div><div><span>存储格式</span><strong>{versionDetail.text_index.storage_format || "—"}</strong></div></div>}{versionDetail.conflicts.length > 0 && <div className="detail-conflict-list"><div className="panel-header compact-header"><div><h2>检测到 {versionDetail.conflicts.length} 个结构冲突</h2></div></div><p className="conflict-explanation">头实体＋关系相同，或尾实体＋关系相同而另一端不同，仅作多值提醒；所有事实仍保留。</p>{versionDetail.conflicts.slice(conflictPage * 6, conflictPage * 6 + 6).map((conflict) => <div className="conflict-row" key={String(conflict.conflict_id)}><span>{conflict.conflict_type === "same_tail_relation" ? "尾实体相同" : "头实体相同"}</span><div className="conflict-pair">{[conflict.triple, conflict.other_triple].map((raw, index) => { const triple = raw as Record<string, unknown> | undefined; return <p key={index}><strong>{String(triple?.head ?? "—")}</strong><b>{String(triple?.relation ?? conflict.relation ?? "—")}</b><strong>{String(triple?.tail ?? "—")}</strong></p>; })}</div></div>)}{versionDetail.conflicts.length > 6 && <div className="pagination"><span>第 {conflictPage + 1} / {Math.ceil(versionDetail.conflicts.length / 6)} 页</span><button className="secondary-button" disabled={conflictPage === 0} onClick={() => setConflictPage(value => value - 1)}>上一页</button><button className="secondary-button" disabled={(conflictPage + 1) * 6 >= versionDetail.conflicts.length} onClick={() => setConflictPage(value => value + 1)}>下一页</button></div>}</div>}</section> : <VersionContentsPanel contents={versionContents} kind={tab} sourceNames={versionSourceNames} onKindChange={onTabChange} selectedItem={selectedItem} onSelect={setSelectedItem} />}
      </>}
    </div>
  );
}

type BuildStageKey = "prepare" | "text" | "kg" | "extract" | "publish";

const buildStages: Array<{ key: BuildStageKey; label: string; detail: string }> = [
  { key: "prepare", label: "版本准备", detail: "创建隔离工作区" },
  { key: "text", label: "文本层", detail: "加载、联合切分并保存条目" },
  { key: "kg", label: "KG 层", detail: "合并文本抽取出的三元组" },
  { key: "extract", label: "事实抽取", detail: "调用 LLM 补充 KG" },
  { key: "publish", label: "发布版本", detail: "写入索引并开放查询" },
];

function BuildProgressPanel({ task, preview, snapshots }: { task: Task | null; preview: BuildPreview | null; snapshots: ProgressSnapshot[] }) {
  const progress = Math.max(0, Math.min(1, task?.progress ?? 0));
  const status = task?.status ?? "pending";
  const currentStage = inferBuildStage(task?.message ?? "", progress);
  const activeSnapshot = snapshots[snapshots.length - 1];
  const extractionSkipped = false;
  const processed = parseProcessedCount(task?.message ?? "");
  const statusLabel = status === "completed" ? "构建完成" : status === "failed" ? "构建失败" : status === "pending" ? "排队中" : "正在构建";

  return (
    <section className={"panel build-progress-panel " + (status === "failed" ? "failed" : status === "completed" ? "completed" : "running")}>
      <div className="build-progress-head">
        <div>
          
          <h2>{statusLabel}</h2>
          <p>{task?.message || "正在等待构建任务反馈…"}</p>
        </div>
        <div className="build-progress-number"><strong>{Math.round(progress * 100)}%</strong><span>{task?.updated_at ? formatTime(task.updated_at) : "刚刚"}</span></div>
      </div>
      <div className="progress-track build-progress-track"><span style={{ width: `${progress * 100}%` }} /></div>
      <div className="build-stage-line">
        {buildStages.map((stage, index) => {
          const skipped = stage.key === "extract" && extractionSkipped;
          const stageStatus = skipped ? "skipped" : buildStageStatus(stage.key, currentStage, status, progress, index);
          return <div className={"build-stage " + stageStatus} key={stage.key}><span className="build-stage-dot">{stageStatus === "completed" ? "✓" : stageStatus === "skipped" ? "–" : index + 1}</span><div><strong>{stage.label}</strong><small>{skipped ? "按导入模式跳过" : stage.detail}</small></div></div>;
        })}
      </div>
      <div className="build-insight-grid">
        <BuildInsight label="当前动作" value={task?.message || "等待反馈"} accent="blue" />
        <BuildInsight label="输入结构" value={`${preview?.text_layer.source_count ?? 0} 个文本源 · KG 由文本抽取`} accent="violet" />
        <BuildInsight label="处理进度" value={processed ? `${processed.done}/${processed.total} ${processed.unit}` : preview?.expected_llm_calls === null ? "按切分子块调用 LLM" : `预计模型调用 ${preview?.expected_llm_calls ?? 0} 次`} accent="green" />
      </div>
      {task?.error_message && <div className="build-failure-note"><strong>{task.error_code || "BUILD_FAILED"}</strong><span>{errorMessage(task.error_message)}</span></div>}
      <div className="build-activity">
        <div className="build-activity-head"><span>最近活动</span><small>{activeSnapshot ? `已记录 ${snapshots.length} 个状态点` : "等待首个状态点"}</small></div>
        {snapshots.length === 0 ? <div className="build-activity-empty">任务启动后，这里会连续记录每个阶段的实际反馈。</div> : <div className="build-activity-list">{snapshots.slice(-7).reverse().map((item, index) => <div className="build-activity-row" key={`${item.recordedAt}-${index}`}><span className="activity-dot" /><strong>{Math.round(item.progress * 100)}%</strong><p>{item.message}</p><time>{formatTime(item.recordedAt)}</time></div>)}</div>}
      </div>
    </section>
  );
}

function BuildInsight({ label, value, accent }: { label: string; value: string; accent: string }) {
  return <div className={"build-insight " + accent}><span>{label}</span><strong>{value}</strong></div>;
}

function inferBuildStage(message: string, progress: number): BuildStageKey {
  const value = message.toLowerCase();
  if (value.includes("工作目录") || progress <= 0.1) return "prepare";
  if (value.includes("文本层") || value.includes("文本切分") || value.includes("文本抽取")) return value.includes("文本抽取") ? "extract" : "text";
  if (value.includes("kg 层") || value.includes("kg层")) return "kg";
  if (value.includes("发布") || value.includes("完成") || progress >= 0.85) return "publish";
  if (progress < 0.4) return "text";
  if (progress < 0.65) return "kg";
  return "extract";
}

function buildStageStatus(stage: BuildStageKey, current: BuildStageKey, status: string, progress: number, index: number) {
  if (status === "failed" && stage === current) return "failed";
  if (status === "completed" || stageIndex(stage) < stageIndex(current) || (stage === "publish" && progress >= 1)) return "completed";
  if (stage === current && (status === "running" || status === "pending")) return "active";
  return index === 0 && progress === 0 ? "active" : "pending";
}

function stageIndex(stage: BuildStageKey) {
  return buildStages.findIndex((item) => item.key === stage);
}

function parseProcessedCount(message: string) {
  const match = message.match(/(\d+)\s*\/\s*(\d+)\s*个(来源|条目)/);
  return match ? { done: match[1], total: match[2], unit: match[3] } : null;
}

function LayerCard({ title, icon, count, unit, hint, tone, disabled = false, onClick }: { title: string; icon: string; count: number | null; unit: string; hint: string; tone: string; disabled?: boolean; onClick?: () => void }) {
  return <button type="button" className="layer-card layer-card-button" onClick={onClick} disabled={disabled}><span className={"layer-icon " + tone}>{icon}</span><div><span>{title}</span><strong>{count ?? "—"} {unit}</strong><small>{hint}</small></div></button>;
}

function VersionContentsPanel({ contents, kind, sourceNames, onKindChange, selectedItem, onSelect }: { contents: { text: VersionContents; kg: VersionContents } | null; kind: "text" | "kg"; sourceNames: string[]; onKindChange: (kind: "text" | "kg") => void; selectedItem: Record<string, unknown> | null; onSelect: (item: Record<string, unknown> | null) => void }) {
  const detailRef = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<VersionContents | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const currentIdentity = contents ? contents[kind].version_id + ":" + kind : "";
  useEffect(() => { setDraft(""); setSearch(""); setOffset(0); setPage(null); setError(""); }, [currentIdentity]);
  useEffect(() => {
    if (!contents) return;
    if (offset === 0 && !search) { setPage(contents[kind]); setLoading(false); return; }
    let cancelled = false;
    setLoading(true); setError("");
    const item = contents[kind];
    void getVersionContents(item.project_id, item.version_id, kind, { offset, search }).then(result => { if (!cancelled) setPage(result); }).catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : "内容获取失败"); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [contents, kind, offset, search]);
  const selectedKey = selectedItem ? contentItemKey(kind, selectedItem) : "";

  useEffect(() => {
    if (!selectedKey) return;
    const timer = window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
    return () => window.clearTimeout(timer);
  }, [selectedKey]);

  if (!contents) return null;
  const current = page?.version_id === contents[kind].version_id && page?.kind === kind ? page : contents[kind];
  const items = current.items;

  const renderTextItem = (item: Record<string, unknown>, index: number) => {
    const itemKey = contentItemKey("text", item);
    const selected = itemKey === selectedKey;
    return <div className="content-item-group" key={itemKey}><button type="button" className={"entry-row entry-row-button " + (selected ? "selected" : "")} onClick={() => onSelect(selected ? null : item)}><span>{offset + index + 1}</span><div><p>{previewTextContent(item.content)}</p></div><b className="row-open-mark">{selected ? "收起 ×" : "查看 ›"}</b></button>{selected && <div className="inline-content-detail" ref={detailRef}><ContentDetailPanel kind="text" item={item} onClose={() => onSelect(null)} /></div>}</div>;
  };

  const renderKgItem = (item: Record<string, unknown>) => {
    const itemKey = contentItemKey("kg", item);
    const selected = itemKey === selectedKey;
    return <div className="content-item-group" key={itemKey}><button type="button" className={"triple-row triple-row-button " + (selected ? "selected" : "")} onClick={() => onSelect(selected ? null : item)}><span>{String(item.head)}</span><b>{String(item.relation)}</b><span>{String(item.tail)}</span><em>{selected ? "收起 ×" : "查看 ›"}</em></button>{selected && <div className="inline-content-detail" ref={detailRef}><ContentDetailPanel kind="kg" item={item} onClose={() => onSelect(null)} /></div>}</div>;
  };

  return <section className="panel contents-panel"><div className="panel-header compact-header"><div><h2>{kind === "text" ? "文本" : "图谱三元组"}</h2><p className="contents-source-summary">整体来源：{sourceNames.length > 0 ? sourceNames.join("、") : "当前版本输入"}</p></div><div className="content-tabs"><button type="button" className={kind === "text" ? "active" : ""} onClick={() => onKindChange("text")}>文本</button><button type="button" className={kind === "kg" ? "active" : ""} onClick={() => onKindChange("kg")}>图谱</button></div></div>
    <form className="content-browser-toolbar" onSubmit={event => { event.preventDefault(); onSelect(null); setOffset(0); setSearch(draft.trim()); }}><input aria-label="搜索当前版本内容" className="text-input" placeholder={kind === "text" ? "在当前版本的全部正文中搜索" : "按实体、关系或证据搜索"} value={draft} onChange={event => setDraft(event.target.value)} maxLength={500}/><button className="secondary-button" type="submit" disabled={loading}>搜索</button>{search && <button type="button" className="link-button" onClick={() => { setDraft(""); setSearch(""); setOffset(0); onSelect(null); }}>清除筛选</button>}</form>
    {error ? <div className="query-error" role="alert"><p>{error}</p></div> : loading ? <div className="trace-empty" role="status">正在读取版本内容…</div> : items.length === 0 ? <div className="trace-empty">{search ? "没有匹配的内容" : "当前版本暂无内容"}</div> : kind === "text" ? <div className="entry-list">{items.map(renderTextItem)}</div> : <div className="triple-list">{items.map(renderKgItem)}</div>}
    <div className="pagination"><span>{current.total} 条{search ? "匹配内容" : "内容"} · 第 {Math.floor(offset / 30) + 1} / {Math.max(1, Math.ceil(current.total / 30))} 页</span><button className="secondary-button" disabled={loading || offset === 0} onClick={() => { onSelect(null); setOffset(value => Math.max(0, value - 30)); }}>上一页</button><button className="secondary-button" disabled={loading || offset + 30 >= current.total} onClick={() => { onSelect(null); setOffset(value => value + 30); }}>下一页</button></div>
    <div className="contents-footer">只读浏览 · {current.version_total ?? current.total} 条版本内容</div>
  </section>;
}

function contentItemKey(kind: "text" | "kg", item: Record<string, unknown>) {
  const value = kind === "text" ? item.entry_id ?? item.content_hash : item.triple_id ?? item.triple_hash;
  return String(value ?? "");
}

function previewTextContent(value: unknown) {
  const content = String(value ?? "");
  const withoutHeadingPath = content.replace(/^章节路径：[^\r\n]*(?:\r?\n)+/, "").trimStart();
  return (withoutHeadingPath || content).slice(0, 180);
}

function ContentDetailPanel({ kind, item, onClose }: { kind: "text" | "kg"; item: Record<string, unknown>; onClose: () => void }) {
  const value = (key: string, fallback = "—") => {
    const raw = item[key];
    if (raw === null || raw === undefined || raw === "") return fallback;
    if (Array.isArray(raw)) return raw.length > 0 ? raw.map((entry) => String(entry)).join("、") : fallback;
    if (typeof raw === "object") return JSON.stringify(raw);
    return String(raw);
  };
  const title = kind === "text" ? value("title", "未命名条目") : "图谱三元组详情";
  return <section className="panel content-detail-panel"><div className="content-detail-head"><div><h3>{title}</h3></div><button type="button" className="content-detail-close" onClick={onClose}>收起 ×</button></div>{kind === "text" ? <><div className="content-detail-body">{value("content", "暂无文本内容")}</div><div className="content-detail-meta"><div><span>来源</span><strong>{textSourceName(item)}</strong></div><div><span>原文位置</span><strong>{value("source_location")}</strong></div><div><span>父级上下文</span><strong>{value("parent_id")}</strong></div><div><span>关联三元组</span><strong>{value("kg_triple_ids")}</strong></div><div><span>内容哈希</span><strong>{value("content_hash")}</strong></div></div></> : <><div className="content-detail-triple"><span>{value("head")}</span><b>{value("relation")}</b><span>{value("tail")}</span></div><div className="content-detail-body">{value("evidence", "暂无证据文本")}</div><div className="content-detail-meta"><div><span>来源</span><strong>{value("origin")}</strong></div><div><span>来源条目</span><strong>{value("source_entry_id")}</strong></div><div><span>来源行</span><strong>{value("source_line")}</strong></div><div><span>三元组哈希</span><strong>{value("triple_hash")}</strong></div></div></>}</section>;
}

function textSourceName(item: Record<string, unknown>) {
  const title = String(item.title ?? "").trim();
  return title.split(" / ")[0] || "当前版本输入";
}


