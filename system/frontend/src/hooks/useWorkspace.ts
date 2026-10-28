import { errorMessage } from "../utils/messages";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createProject,
  checkModelConnection,
  getCapabilities,
  getBuildConfig,
  getQueryConfig,
  getEvaluationSummary,
  getHealth,
  getModelConfig,
  getProject,
  getQueryConstraint,
  getQueryEvidence,
  getQuery,
  getQueryTrace,
  getTask,
  listProjects,
  listSources,
  listTasks,
  previewBuild,
  startBuild,
  startQuery,
  uploadSource,
  updateModelConfig,
  updateBuildConfig,
  updateQueryConfig,
  getVersionDetail,
  getVersionContents,
  listVersions,
} from "../api/client";
import type {
  BuildConfig,
  BuildConfigUpdate,
  BuildPreview,
  CapabilitiesData,
  ConnectionStatus,
  EvaluationSummary,
  HealthData,
  ModelConfig,
  ModelConfigUpdate,
  QueryConfig,
  QueryConfigUpdate,
  Project,
  QueryConstraintDetail,
  QueryEvent,
  QueryEvidenceDetail,
  QueryRecord,
  Source,
  Task,
  VersionContents,
  VersionDetail,
  VersionSummary,
} from "../types";

import type { ProgressSnapshot } from "../types/workspace";

export function useWorkspace() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesData | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;
  const [connection, setConnection] = useState<ConnectionStatus>("checking");
  const [activeNav, setActiveNav] = useState(() => {
    const page = new URLSearchParams(window.location.search).get("page");
    return page && ["总览", "知识库", "多跳问答", "任务", "评测", "设置"].includes(page) ? page : "总览";
  });
  const [projectName, setProjectName] = useState("");
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [versions, setVersions] = useState<VersionSummary[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [versionDetail, setVersionDetail] = useState<VersionDetail | null>(null);
  const [versionContents, setVersionContents] = useState<{ text: VersionContents; kg: VersionContents } | null>(null);
  const [knowledgeVersionLoading, setKnowledgeVersionLoading] = useState(false);
  const [buildPreview, setBuildPreview] = useState<BuildPreview | null>(null);
  const [uploading, setUploading] = useState(false);
  const [building, setBuilding] = useState(false);
  const [buildTask, setBuildTask] = useState<Task | null>(null);
  const [buildSnapshots, setBuildSnapshots] = useState<ProgressSnapshot[]>([]);
  const [creating, setCreating] = useState(false);
  const [queryQuestion, setQueryQuestion] = useState("");
  const [queryRunning, setQueryRunning] = useState(false);
  const [queryRecord, setQueryRecord] = useState<QueryRecord | null>(null);
  const [queryEvents, setQueryEvents] = useState<QueryEvent[]>([]);
  const [queryTaskProgress, setQueryTaskProgress] = useState(0);
  const [queryTaskMessage, setQueryTaskMessage] = useState("");
  const [queryTraceLevel, setQueryTraceLevel] = useState<"summary" | "detailed">("detailed");
  const [selectedConstraintId, setSelectedConstraintId] = useState<string | null>(null);
  const [constraintDetail, setConstraintDetail] = useState<QueryConstraintDetail | null>(null);
  const [constraintLoading, setConstraintLoading] = useState(false);
  const [evidenceDetail, setEvidenceDetail] = useState<QueryEvidenceDetail | null>(null);
  const [evaluationSummary, setEvaluationSummary] = useState<EvaluationSummary | null>(null);
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [buildConfig, setBuildConfig] = useState<BuildConfig | null>(null);
  const [queryConfig, setQueryConfig] = useState<QueryConfig | null>(null);
  const [modelChecking, setModelChecking] = useState(false);
  const [modelCheckMessage, setModelCheckMessage] = useState("");
  const [modelSaving, setModelSaving] = useState(false);
  const [modelSaveMessage, setModelSaveMessage] = useState("");
  const [buildSaving, setBuildSaving] = useState(false);
  const [buildSaveMessage, setBuildSaveMessage] = useState("");
  const [querySaving, setQuerySaving] = useState(false);
  const [querySaveMessage, setQuerySaveMessage] = useState("");
  const [message, setMessage] = useState("");
  const projectRequest = useRef(0);
  const versionRequest = useRef(0);
  const selectionRef = useRef({ projectId: selectedProjectId, versionId: selectedVersionId });
  selectionRef.current = { projectId: selectedProjectId, versionId: selectedVersionId };

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
    const url = new URL(window.location.href);
    url.searchParams.set("page", activeNav);
    window.history.replaceState(null, "", url);
  }, [activeNav]);

  async function refresh() {
    setConnection("checking");
    try {
      const [healthData, capabilityData, projectData, taskData, modelConfigData, buildConfigData, queryConfigData] = await Promise.all([getHealth(), getCapabilities(), listProjects(), listTasks(), getModelConfig(), getBuildConfig(), getQueryConfig()]);
      setHealth(healthData);
      setCapabilities(capabilityData);
      setProjects(projectData);
      setTasks(taskData);
      setModelConfig(modelConfigData);
      setBuildConfig(buildConfigData);
      setQueryConfig(queryConfigData);
      setConnection("connected");
      setMessage("");
    } catch (error) {
      setConnection("disconnected");
      setMessage(errorMessage(error, "无法连接服务"));
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedProjectId && projects.length > 0) {
      setSelectedProjectId(projects[0].project_id);
    } else if (selectedProjectId && !projects.some((project) => project.project_id === selectedProjectId)) {
      setSelectedProjectId(projects[0]?.project_id ?? "");
    }
  }, [projects, selectedProjectId]);

  useEffect(() => {
    projectRequest.current += 1;
    versionRequest.current += 1;
    setVersions([]);
    setSelectedVersionId("");
    setSources([]);
    setBuildPreview(null);
    setVersionDetail(null);
    setVersionContents(null);
    setBuildTask(null);
    setBuildSnapshots([]);
  }, [selectedProjectId]);

  async function refreshKnowledgeBase(projectId: string) {
    const requestId = ++projectRequest.current;
    try {
      const [sourceData, previewData, projectData, versionData] = await Promise.all([
        listSources(projectId),
        previewBuild(projectId),
        getProject(projectId),
        listVersions(projectId),
      ]);
      if (requestId !== projectRequest.current || selectionRef.current.projectId !== projectId) return;
      setSources(sourceData);
      setBuildPreview(previewData);
      setVersions(versionData);
      setSelectedVersionId((current) => (
        versionData.some((version) => version.version_id === current && version.status === "ready")
          ? current
          : versionData.find((version) => version.version_id === projectData.current_version_id && version.status === "ready")?.version_id
            ?? versionData.find((version) => version.status === "ready")?.version_id
            ?? ""
      ));
      setProjects((current) => current.map((item) => item.project_id === projectData.project_id ? projectData : item));
      if (!projectData.current_version_id) {
        setVersionDetail(null);
        setVersionContents(null);
      }
    } catch (error) {
      setMessage(errorMessage(error, "知识库状态获取失败"));
    }
  }

  async function loadKnowledgeVersion(projectId: string, versionId: string) {
    const requestId = ++versionRequest.current;
    setKnowledgeVersionLoading(true);
    const [detail, text, kg] = await Promise.all([
      getVersionDetail(projectId, versionId),
      getVersionContents(projectId, versionId, "text"),
      getVersionContents(projectId, versionId, "kg"),
    ]);
    if (requestId !== versionRequest.current || selectionRef.current.projectId !== projectId || selectionRef.current.versionId !== versionId) return;
    setVersionDetail(detail);
    setVersionContents({ text, kg });
    setKnowledgeVersionLoading(false);
  }

  async function handleOpenKnowledgeVersion(versionId: string) {
    if (!selectedProjectId || !versionId) return;
    setSelectedVersionId(versionId);
    setMessage("");
  }

  useEffect(() => {
    if (connection === "connected" && selectedProjectId) {
      void refreshKnowledgeBase(selectedProjectId);
    }
  }, [connection, selectedProjectId]);

  useEffect(() => {
    setVersionDetail(null);
    setVersionContents(null);
    if (!selectedProjectId || !selectedVersionId) { setKnowledgeVersionLoading(false); return; }
    void loadKnowledgeVersion(selectedProjectId, selectedVersionId).catch(error => {
      if (selectionRef.current.projectId === selectedProjectId && selectionRef.current.versionId === selectedVersionId) {
        setKnowledgeVersionLoading(false);
        setMessage(errorMessage(error, "知识库版本获取失败"));
      }
    });
    return () => { versionRequest.current += 1; };
  }, [selectedProjectId, selectedVersionId]);

  useEffect(() => {
    if (queryRecord && !queryRunning && (
      queryRecord.project_id !== selectedProjectId ||
      (selectedVersionId && queryRecord.version_id !== selectedVersionId)
    )) {
      setQueryRecord(null);
      setQueryEvents([]);
      setSelectedConstraintId(null);
      setConstraintDetail(null);
      setEvidenceDetail(null);
      setQueryTaskProgress(0);
      setQueryTaskMessage("");
    }
  }, [selectedProjectId, selectedVersionId]);

  useEffect(() => {
    if (activeNav === "评测" && connection === "connected" && selectedProjectId) {
      void getEvaluationSummary(selectedProjectId)
        .then(setEvaluationSummary)
        .catch((error) => setMessage(errorMessage(error, "评测数据获取失败")));
    }
  }, [activeNav, connection, selectedProjectId]);

  useEffect(() => {
    const activeBuild = tasks.find(
      (task) => task.task_type === "knowledge_base_build"
        && task.payload.project_id === selectedProjectId
        && ["pending", "running", "completed", "failed"].includes(task.status),
    );
    if (!activeBuild) return;
    setBuildTask(activeBuild);
    setBuildSnapshots((current) => {
      const last = current[current.length - 1];
      if (last && last.progress === activeBuild.progress && last.message === activeBuild.message && last.status === activeBuild.status) {
        return current;
      }
      return [...current, {
        progress: activeBuild.progress,
        message: activeBuild.message || "任务已创建",
        status: activeBuild.status,
        recordedAt: activeBuild.updated_at,
      }].slice(-40);
    });
  }, [tasks, selectedProjectId]);

  const healthyProjects = useMemo(
    () => projects.filter((project) => project.status === "ready").length,
    [projects],
  );
  const runningTasks = useMemo(
    () => tasks.filter((task) => task.status === "pending" || task.status === "running").length,
    [tasks],
  );

  // Resume observing an existing query after opening it from persisted tasks.
  // This only reads records; it never resubmits the LLM operation.
  useEffect(() => {
    if (!queryRecord || queryRunning || !["pending", "running"].includes(queryRecord.status)) return;
    const queryId = queryRecord.query_id;
    let cancelled = false;
    let busy = false;
    const observe = async () => {
      if (busy) return;
      busy = true;
      try {
        const [record, events] = await Promise.all([getQuery(queryId), getQueryTrace(queryId)]);
        if (cancelled) return;
        setQueryRecord(record);
        setQueryEvents(events);
        const task = tasksRef.current.find(item => item.payload.query_id === queryId);
        if (task) {
          setQueryTaskProgress(task.progress);
          setQueryTaskMessage(task.message);
        }
      } catch {
        if (!cancelled) setQueryTaskMessage("连接中断，正在重新同步服务端状态");
      } finally { busy = false; }
    };
    void observe();
    const timer = window.setInterval(() => void observe(), 1300);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [queryRecord?.query_id, queryRecord?.status, queryRunning]);

  useEffect(() => {
    if (runningTasks === 0 || connection !== "connected") return undefined;
    const timer = window.setInterval(() => {
      void listTasks().then(setTasks).catch(() => undefined);
    }, 700);
    return () => window.clearInterval(timer);
  }, [connection, runningTasks]);

  async function handleCreateProject() {
    const name = projectName.trim();
    if (!name) {
      setMessage("请输入项目名称");
      return false;
    }
    setCreating(true);
    setMessage("");
    try {
      const created = await createProject(name);
      setProjects((current) => [created, ...current]);
      setSelectedProjectId(created.project_id);
      setProjectName("");
      setMessage("项目已创建");
      return true;
    } catch (error) {
      setMessage(errorMessage(error, "项目创建失败"));
      return false;
    } finally {
      setCreating(false);
    }
  }

  async function handleOpenQuery(queryId: string) {
    if (queryRunning) { setMessage("当前问答仍在执行，可先查看其运行过程"); return; }
    try {
      const [record, events] = await Promise.all([getQuery(queryId), getQueryTrace(queryId)]);
      if (record.project_id === selectedProjectId) setSelectedVersionId(record.version_id);
      setQueryRecord(record);
      setQueryQuestion(record.question);
      setQueryEvents(events);
      setSelectedConstraintId(null);
      setConstraintDetail(null);
      setEvidenceDetail(null);
      setQueryTaskProgress(record.status === "completed" ? 1 : 0);
      setQueryTaskMessage(record.error_message ? errorMessage(record.error_message) : "历史运行记录");
      setActiveNav("多跳问答");
    } catch (error) { setMessage(errorMessage(error, "查询记录获取失败")); }
  }

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0 || !selectedProjectId) return;
    setUploading(true);
    setMessage("");
    try {
      let uploaded = 0;
      for (const file of Array.from(files)) {
        await uploadSource(selectedProjectId, file);
        uploaded += 1;
      }
      await refreshKnowledgeBase(selectedProjectId);
      setMessage(`已处理 ${uploaded} 个文件`);
    } catch (error) {
      setMessage(errorMessage(error, "文件上传失败"));
    } finally {
      setUploading(false);
    }
  }

  async function handleBuild() {
    if (building || !selectedProjectId || !buildPreview) return;
    setBuilding(true);
    setBuildTask(null);
    setBuildSnapshots([]);
    setMessage("");
    try {
      const started = await startBuild(selectedProjectId, buildPreview.mode);
      let task = await getTask(started.task_id);
      setTasks((current) => [task, ...current.filter((item) => item.task_id !== task.task_id)]);
      setBuildTask(task);
      setBuildSnapshots([{
        progress: task.progress,
        message: task.message || "任务已提交，等待执行",
        status: task.status,
        recordedAt: task.updated_at,
      }]);
      while (!["completed", "failed"].includes(task.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        task = await getTask(started.task_id);
        setTasks((current) => [task, ...current.filter((item) => item.task_id !== task.task_id)]);
        setBuildTask(task);
        setBuildSnapshots((current) => {
          const last = current[current.length - 1];
          if (last && last.progress === task.progress && last.message === task.message && last.status === task.status) return current;
          return [...current, { progress: task.progress, message: task.message, status: task.status, recordedAt: task.updated_at }].slice(-40);
        });
      }
      if (task.status === "failed") {
        throw new Error(task.error_message || "知识库构建失败");
      }
      if (task.status !== "completed") {
        throw new Error("构建任务仍在运行，请稍后刷新状态");
      }
      await refresh();
      await refreshKnowledgeBase(selectedProjectId);
      setMessage("知识库版本已发布");
    } catch (error) {
      setMessage(errorMessage(error, "知识库构建失败"));
    } finally {
      setBuilding(false);
    }
  }

  async function handleQuery() {
    if (queryRunning) return;
    const question = queryQuestion.trim();
    if (!selectedProjectId) {
      setMessage("请先选择项目");
      return;
    }
    if (!question) {
      setMessage("请输入问题");
      return;
    }
    setQueryRunning(true);
    setQueryRecord(null);
    setQueryEvents([]);
    setQueryTaskProgress(0);
    setQueryTaskMessage("正在提交查询");
    setSelectedConstraintId(null);
    setConstraintDetail(null);
    setEvidenceDetail(null);
    setMessage("");
    try {
      const started = await startQuery(selectedProjectId, question, queryTraceLevel, selectedVersionId || undefined);
      let task = await getTask(started.task_id);
      setTasks(current => [task, ...current.filter(item => item.task_id !== task.task_id)]);
      let record = await getQuery(started.query_id);
      setQueryRecord(record);
      while (!["completed", "failed"].includes(task.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        task = await getTask(started.task_id);
        setTasks(current => [task, ...current.filter(item => item.task_id !== task.task_id)]);
        record = await getQuery(started.query_id);
        setQueryTaskProgress(task.progress);
        setQueryTaskMessage(task.message);
        setQueryRecord(record);
        const events = await getQueryTrace(started.query_id);
        setQueryEvents(events);
      }
      const finalRecord = await getQuery(started.query_id);
      setQueryRecord(finalRecord);
      setQueryEvents(await getQueryTrace(started.query_id));
      setQueryTaskProgress(task.progress);
      setQueryTaskMessage(task.message);
      if (task.status === "failed" || finalRecord.status === "failed") {
        setMessage(errorMessage(finalRecord.error_message || task.error_message, "查询失败"));
      }
    } catch (error) {
      setMessage(errorMessage(error, "查询提交失败"));
    } finally {
      setQueryRunning(false);
    }
  }

  async function handleConstraintSelect(constraintId: string) {
    if (!queryRecord) return;
    setSelectedConstraintId(constraintId);
    setConstraintLoading(true);
    setEvidenceDetail(null);
    try {
      setConstraintDetail(await getQueryConstraint(queryRecord.query_id, constraintId));
    } catch (error) {
      setConstraintDetail(null);
      setMessage(errorMessage(error, "约束详情获取失败"));
    } finally {
      setConstraintLoading(false);
    }
  }

  async function handleEvidenceSelect(evidenceId: string) {
    if (!queryRecord) return;
    try {
      setEvidenceDetail(await getQueryEvidence(queryRecord.query_id, evidenceId));
    } catch (error) {
      setMessage(errorMessage(error, "证据详情获取失败"));
    }
  }

  async function handleModelCheck() {
    setModelChecking(true);
    setModelCheckMessage("");
    try {
      const result = await checkModelConnection();
      setModelCheckMessage(`连接成功 · ${result.model} · ${result.elapsed_ms} ms`);
    } catch (error) {
      setModelCheckMessage(errorMessage(error, "模型连接失败"));
    } finally {
      setModelChecking(false);
    }
  }

  async function handleModelSave(update: ModelConfigUpdate) {
    setModelSaving(true);
    setModelSaveMessage("");
    try {
      const saved = await updateModelConfig(update);
      setModelConfig(saved);
      const capabilityData = await getCapabilities();
      setCapabilities(capabilityData);
      setBuildConfig(capabilityData.build_config);
      setModelSaveMessage("配置已保存并立即生效");
      if (selectedProjectId && activeNav === "知识库") {
        await refreshKnowledgeBase(selectedProjectId);
      }
    } catch (error) {
      setModelSaveMessage(errorMessage(error, "配置保存失败"));
    } finally {
      setModelSaving(false);
    }
  }

  async function handleBuildConfigSave(update: BuildConfigUpdate) {
    setBuildSaving(true);
    setBuildSaveMessage("");
    try {
      const saved = await updateBuildConfig(update);
      setBuildConfig(saved);
      setBuildSaveMessage("构建参数已保存，后续新版本构建生效");
      if (selectedProjectId && activeNav === "知识库") {
        await refreshKnowledgeBase(selectedProjectId);
      }
    } catch (error) {
      setBuildSaveMessage(errorMessage(error, "构建参数保存失败"));
    } finally {
      setBuildSaving(false);
    }
  }

  async function handleQueryConfigSave(update: QueryConfigUpdate) {
    setQuerySaving(true);
    setQuerySaveMessage("");
    try {
      const saved = await updateQueryConfig(update);
      setQueryConfig(saved);
      setQuerySaveMessage("检索参数已保存，后续查询生效");
    } catch (error) {
      setQuerySaveMessage(errorMessage(error, "检索参数保存失败"));
    } finally {
      setQuerySaving(false);
    }
  }

  return {
    health,
    capabilities,
    projects,
    tasks,
    connection,
    activeNav,
    setActiveNav,
    projectName,
    setProjectName,
    selectedProjectId,
    setSelectedProjectId,
    versions,
    selectedVersionId,
    setSelectedVersionId,
    sources,
    versionDetail,
    versionContents,
    knowledgeVersionLoading,
    buildPreview,
    uploading,
    building,
    buildTask,
    buildSnapshots,
    creating,
    queryQuestion,
    setQueryQuestion,
    queryRunning,
    queryRecord,
    queryEvents,
    queryTaskProgress,
    queryTaskMessage,
    queryTraceLevel,
    setQueryTraceLevel,
    selectedConstraintId,
    constraintDetail,
    constraintLoading,
    evidenceDetail,
    evaluationSummary,
    modelConfig,
    buildConfig,
    queryConfig,
    modelChecking,
    modelCheckMessage,
    modelSaving,
    modelSaveMessage,
    buildSaving,
    buildSaveMessage,
    querySaving,
    querySaveMessage,
    message,
    setMessage,
    healthyProjects,
    runningTasks,
    refresh,
    refreshKnowledgeBase,
    handleOpenKnowledgeVersion,
    handleCreateProject,
    handleUpload,
    handleBuild,
    handleQuery,
    handleOpenQuery,
    handleConstraintSelect,
    handleEvidenceSelect,
    handleModelCheck,
    handleModelSave,
    handleBuildConfigSave,
    handleQueryConfigSave
  };
}

export type Workspace = ReturnType<typeof useWorkspace>;

