import { errorMessage, httpErrorMessage, validationMessage } from "../utils/messages";

export class ServiceError extends Error {
  constructor(message: string, readonly originalDetail: unknown) {
    super(message);
    this.name = "ServiceError";
  }
}

export async function fetchData<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new ServiceError(errorMessage(error, "请求未能发送，请检查网络和服务地址"), error);
  }
  let body: unknown;
  try { body = await response.json(); } catch (error) {
    throw new ServiceError(response.ok ? "服务返回的数据格式异常，请检查服务地址和后端日志" : httpErrorMessage(response.status), error);
  }
  const payload = body && typeof body === "object" ? body as Record<string, unknown> : null;
  if (!response.ok) {
    const detail = payload?.detail;
    const message = Array.isArray(detail) ? detail.map(item => {
      if (!item || typeof item !== "object") return "参数格式不正确";
      const value = item as { loc?: Array<string | number>; msg?: string; type?: string; ctx?: Record<string, unknown> };
      const fields: Record<string, string> = { name: "名称", question: "问题", description: "描述", file: "文件", model_name: "模型名称", base_url: "服务地址", timeout_seconds: "请求超时", max_tokens: "Chunk 最大 Token", min_tokens: "Chunk 最小 Token" };
      const location = Array.isArray(value.loc) ? value.loc.filter(part => !["body", "query", "path"].includes(String(part))).map(part => fields[String(part)] ?? String(part)).join(".") : "";
      return [location, validationMessage(value)].filter(Boolean).join("：");
    }).join("；") : errorMessage(detail, httpErrorMessage(response.status));
    throw new ServiceError(message || httpErrorMessage(response.status), body);
  }
  if (!payload || !("data" in payload)) throw new ServiceError("服务响应缺少 data 字段，请确认前后端版本一致", body);
  return payload.data as T;
}
