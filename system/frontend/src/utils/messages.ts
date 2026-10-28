/** UI messages only. Never translate answers, evidence, identifiers or raw JSON. */
export function statusLabel(value: string) {
  const labels: Record<string, string> = {
    pending: "等待中", running: "执行中", completed: "已完成", failed: "失败",
    ready: "已就绪", building: "构建中", extracting: "抽取中", parsed: "已解析",
    uploaded: "已上传", empty: "暂无内容", skipped: "已跳过", cancelled: "已取消",
    not_provided: "未提供", not_built: "未构建", unknown: "未记录", unavailable: "不可用",
  };
  return labels[value] ?? "状态未识别";
}

export function httpErrorMessage(status: number) {
  const labels: Record<number, string> = {
    400: "请求内容有误，请检查输入", 401: "身份验证失败，请检查 API Key 或登录状态",
    403: "没有访问权限，请检查服务授权", 404: "请求的资源不存在，请刷新后重试",
    405: "接口不支持此操作", 408: "请求超时，请稍后重试",
    409: "当前状态不允许此操作，请刷新后重试", 413: "上传内容过大，请缩小文件后重试",
    422: "输入参数不符合要求，请检查填写内容", 429: "请求过于频繁或服务额度不足，请稍后重试",
    500: "服务内部异常，请查看后端日志", 502: "上游服务响应异常，请稍后重试",
    503: "服务暂时不可用，请稍后重试", 504: "上游服务响应超时，请稍后重试",
  };
  return `${labels[status] ?? "服务请求失败"}（HTTP ${status}）`;
}

export function errorMessage(error: unknown, fallback = "操作失败，请稍后重试或查看后端日志") {
  const raw = error instanceof Error ? error.message : typeof error === "string" ? error : "";
  if (/failed to fetch|fetch failed|networkerror|network request failed|load failed|connection refused|failed to establish.*connection/i.test(raw))
    return "无法连接服务，请确认后端已启动，并检查网络和服务地址";
  if ((error instanceof Error && error.name === "AbortError") || /aborted|aborterror/i.test(raw)) return "请求已取消或中断，请重试";
  if (/timeout|timed out/i.test(raw)) return "请求超时，请检查服务状态或稍后重试";
  if (/unexpected (token|end)|not valid json|jsondecodeerror/i.test(raw)) return "服务返回的数据格式异常，请检查服务地址和后端日志";
  const http = raw.match(/HTTP\s+(\d{3})/i);
  if (http) return httpErrorMessage(Number(http[1]));
  const exact: Record<string, string> = {
    "Not Found": "请求的资源不存在，请刷新后重试", "Internal Server Error": "服务内部异常，请查看后端日志",
    "Forbidden": "没有访问权限", "Unauthorized": "身份验证失败，请检查服务凭据",
    "Method Not Allowed": "接口不支持此操作", "Field required": "此项为必填项",
  };
  return exact[raw] ?? (/[\u3400-\u9fff]/.test(raw) ? raw : fallback);
}

export function validationMessage(value: { type?: string; msg?: string; ctx?: Record<string, unknown> }) {
  if (value.msg && /[\u3400-\u9fff]/.test(value.msg)) return value.msg;
  const ctx = value.ctx ?? {};
  const messages: Record<string, string> = {
    missing: "此项为必填项", string_type: "请输入文本", int_type: "请输入整数", int_parsing: "请输入有效整数",
    float_type: "请输入数值", float_parsing: "请输入有效数值", bool_parsing: "请选择有效的开关值",
    string_too_short: `文本长度至少为 ${ctx.min_length ?? "规定数量"} 个字符`,
    string_too_long: `文本长度最多为 ${ctx.max_length ?? "规定数量"} 个字符`,
    greater_than_equal: `数值不能小于 ${ctx.ge ?? "规定下限"}`, less_than_equal: `数值不能大于 ${ctx.le ?? "规定上限"}`,
    greater_than: `数值必须大于 ${ctx.gt ?? "规定下限"}`, less_than: `数值必须小于 ${ctx.lt ?? "规定上限"}`,
    literal_error: "请选择允许的选项", json_invalid: "请求 JSON 格式不正确", extra_forbidden: "包含不支持的参数",
  };
  return messages[value.type ?? ""] ?? errorMessage(value.msg, "参数格式不正确");
}
