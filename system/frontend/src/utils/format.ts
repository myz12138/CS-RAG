export function stepLabel(step: string) {
  const labels: Record<string, string> = {
    query: "流程汇总",
    query_plan: "查询规划",
    phase1_graph_retrieval: "Phase1 图检索",
    neff_gating: "充分性门控",
    variable_propagation: "变量传播",
    phase2_text_recovery: "Phase2 证据收集",
    answer_generation: "答案生成",
  };
  return labels[step] ?? step;
}

export function stringValue(item: Record<string, unknown> | undefined, key: string) {
  const value = item?.[key];
  return typeof value === "string" && value ? value : "—";
}

export function evidenceSummary(item: Record<string, unknown> | undefined) {
  if (!item) return "等待证据收集结果";
  const candidates = item.candidate_evidences;
  const count = Array.isArray(candidates) ? candidates.length : 0;
  const branch = item.triple_status === "resolved" ? "KG evidence 分支" : item.triple_status === "unresolved" ? "文本初筛与重排分支" : "分支未记录";
  return `${branch} · ${count ? `${count} 条入选证据` : "没有可用证据"}`;
}

export function constraintCheckSummary(item: Record<string, unknown>) {
  const state = item.status === "resolved" ? "已解决" : item.status === "unresolved" ? "未解决" : "未记录";
  const check = item.checked === true ? `已检查 · Neff ${typeof item.neff === "number" ? item.neff : item.neff === null ? "无有限值（检查时无候选）" : "未记录"}` : item.checked === false ? "未执行充分性检查" : "检查状态未记录";
  return `${check} · 最终状态：${state}`;
}

export function candidateText(candidate: unknown) {
  if (typeof candidate === "string") return candidate;
  if (candidate && typeof candidate === "object") {
    const record = candidate as Record<string, unknown>;
    for (const key of ["context", "content", "text", "sentence", "evidence", "title"]) {
      if (typeof record[key] === "string" && record[key]) return record[key] as string;
    }
    return Object.entries(record).slice(0, 3).map(([key, value]) => `${key}: ${String(value)}`).join(" · ");
  }
  return String(candidate ?? "暂无内容");
}

export function formatMetric(value: number | undefined) {
  if (value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "刚刚";
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}


