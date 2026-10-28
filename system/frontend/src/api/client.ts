import { fetchData } from "./transport";
import type {
  BuildConfig,
  BuildConfigUpdate,
  BuildPreview,
  CapabilitiesData,
  EvaluationSummary,
  HealthData,
  ModelConfig,
  ModelConfigUpdate,
  ModelCheckResult,
  Project,
  QueryEvent,
  QueryConstraintDetail,
  QueryConfig,
  QueryConfigUpdate,
  QueryEvidenceDetail,
  QueryRecord,
  QueryStart,
  Source,
  Task,
  VersionContents,
  VersionDetail,
  VersionSummary,
} from "../types";

const baseUrl = String(import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(/\/$/, "");

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  return fetchData<T>(baseUrl + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
}

export function getHealth() {
  return request<HealthData>("/system/health");
}

export function getCapabilities() {
  return request<CapabilitiesData>("/system/capabilities");
}

export function checkModelConnection() {
  return request<ModelCheckResult>("/system/model-check", { method: "POST" });
}

export function getModelConfig() {
  return request<ModelConfig>("/system/model-config");
}

export function getBuildConfig() {
  return request<BuildConfig>("/system/build-config");
}

export function getQueryConfig() {
  return request<QueryConfig>("/system/query-config");
}

export function updateModelConfig(payload: ModelConfigUpdate) {
  return request<ModelConfig>("/system/model-config", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function updateBuildConfig(payload: BuildConfigUpdate) {
  return request<BuildConfig>("/system/build-config", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function updateQueryConfig(payload: QueryConfigUpdate) {
  return request<QueryConfig>("/system/query-config", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function listProjects() {
  return request<Project[]>("/projects");
}

export function createProject(name: string, description = "") {
  return request<Project>("/projects", {
    method: "POST",
    body: JSON.stringify({ name, description }),
  });
}

export function listSources(projectId: string) {
  return request<Source[]>(`/projects/${encodeURIComponent(projectId)}/sources`);
}

export function getProject(projectId: string) {
  return request<Project>(`/projects/${encodeURIComponent(projectId)}`);
}

export function getVersionDetail(projectId: string, versionId: string) {
  return request<VersionDetail>(`/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}`);
}

export function listVersions(projectId: string) {
  return request<VersionSummary[]>(`/projects/${encodeURIComponent(projectId)}/versions`);
}

export function getVersionContents(projectId: string, versionId: string, kind: "text" | "kg" = "text", options: { offset?: number; limit?: number; search?: string } = {}) {
  const params = new URLSearchParams({ kind, offset: String(options.offset ?? 0), limit: String(options.limit ?? 30), search: options.search ?? "" });
  return request<VersionContents>(`/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/contents?${params}`);
}

export async function uploadSource(projectId: string, file: File, sourceKind = "auto") {
  const form = new FormData();
  form.append("file", file);
  form.append("source_kind", sourceKind);
  form.append("display_name", file.name);
  return fetchData<{ accepted: boolean; duplicate: boolean; message: string; source: Source }>(
    `${baseUrl}/projects/${encodeURIComponent(projectId)}/sources`, { method: "POST", body: form },
  );
}

export function previewBuild(projectId: string) {
  return request<BuildPreview>(`/projects/${encodeURIComponent(projectId)}/builds/preview`, {
    method: "POST",
    body: JSON.stringify({ mode: "auto" }),
  });
}

export function startBuild(projectId: string, mode: string) {
  return request<{ task_id: string; status: string; project_id: string }>(`/projects/${encodeURIComponent(projectId)}/builds`, {
    method: "POST",
    body: JSON.stringify({ mode }),
  });
}

export function getTask(taskId: string) {
  return request<Task>(`/tasks/${encodeURIComponent(taskId)}`);
}

export function listTasks(status?: string) {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return request<Task[]>(`/tasks${query}`);
}

export function startQuery(projectId: string, question: string, traceLevel: "summary" | "detailed" = "detailed", versionId?: string) {
  return request<QueryStart>(`/projects/${encodeURIComponent(projectId)}/queries`, {
    method: "POST",
    body: JSON.stringify({ question, trace_level: traceLevel, version_id: versionId || undefined }),
  });
}

export function getQuery(queryId: string) {
  return request<QueryRecord>(`/queries/${encodeURIComponent(queryId)}`);
}

export function getQueryTrace(queryId: string, afterSeq = 0) {
  return request<QueryEvent[]>(`/queries/${encodeURIComponent(queryId)}/trace?after_seq=${afterSeq}`);
}

export function getQueryConstraint(queryId: string, constraintId: string) {
  return request<QueryConstraintDetail>(`/queries/${encodeURIComponent(queryId)}/constraints/${encodeURIComponent(constraintId)}`);
}

export function getQueryEvidence(queryId: string, evidenceId: string) {
  return request<QueryEvidenceDetail>(`/queries/${encodeURIComponent(queryId)}/evidence/${encodeURIComponent(evidenceId)}`);
}

export function getEvaluationSummary(projectId: string) {
  return request<EvaluationSummary>(`/projects/${encodeURIComponent(projectId)}/evaluations`);
}
