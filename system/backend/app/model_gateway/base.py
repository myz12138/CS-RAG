"""Small OpenAI-compatible model gateway used by ingestion and query layers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSettings:
    base_url: str
    model_name: str
    embedding_model: str
    reranker_model: str
    api_key: str = ""
    timeout_seconds: float = 90.0
    embedding_base_url: str = ""
    embedding_api_key: str = ""


class ModelGatewayError(RuntimeError):
    pass


class ModelNotConfiguredError(ModelGatewayError):
    pass


class ModelGateway:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def capabilities(self) -> dict[str, object]:
        effective_base_url = self._effective_base_url()
        return {
            "llm": bool(effective_base_url and self.settings.model_name),
            "embedding": bool(self.settings.embedding_model),
            "reranker": bool(self.settings.reranker_model),
            "configured_base_url": bool(effective_base_url),
            "configured_api_key": bool(self.settings.api_key),
            "embedding_remote": bool(self.settings.embedding_base_url),
        }

    def config_summary(self) -> dict[str, object]:
        return {
            "base_url": self.settings.base_url,
            "model_name": self.settings.model_name,
            "timeout_seconds": self.settings.timeout_seconds,
            "embedding_base_url": self.settings.embedding_base_url,
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
            "api_key_configured": bool(self.settings.api_key),
            "embedding_api_key_configured": bool(self.settings.embedding_api_key),
        }

    def extract_triples(self, text: str, source_title: str) -> list[dict[str, str]]:
        """Call the configured real LLM and parse its JSON extraction result.

        There is intentionally no local heuristic fallback. If the model
        service is unavailable, the caller receives a typed error and the
        build remains incomplete instead of storing invented graph facts.
        """

        if not self._effective_base_url() or not self.settings.model_name:
            raise ModelNotConfiguredError("未配置可用的 LLM 服务地址或模型名")
        system_prompt = (
            "你是知识图谱抽取器。只抽取文本中明确表达的事实，不补全常识，不猜测。"
            "严格返回 JSON 对象：{\"triples\":[{\"head\":\"主体\","
            "\"relation\":\"关系\",\"tail\":\"客体\",\"evidence\":\"原文短句\"}]}。"
        )
        user_prompt = f"来源：{source_title}\n请从下面文本抽取事实：\n{text}"
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        result = self._chat_completions(payload)
        content = self._message_content(result)
        try:
            parsed = json.loads(self._strip_code_fence(content))
        except json.JSONDecodeError as exc:
            raise ModelGatewayError("LLM 返回的抽取结果不是合法 JSON") from exc
        records: Any = parsed.get("triples", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(records, list):
            raise ModelGatewayError("LLM 抽取结果缺少 triples 数组")
        triples: list[dict[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            head = str(record.get("head", record.get("subject", ""))).strip()
            relation = str(record.get("relation", record.get("predicate", ""))).strip()
            tail = str(record.get("tail", record.get("object", ""))).strip()
            if head and relation and tail:
                triples.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "evidence": str(record.get("evidence", "")).strip(),
                    }
                )
        return triples

    def check_llm_connection(self) -> dict[str, object]:
        """Probe the configured real LLM endpoint with a minimal request."""

        if not self._effective_base_url() or not self.settings.model_name:
            raise ModelNotConfiguredError("未配置可用的 LLM 服务地址或模型名")
        started = time.perf_counter()
        payload = {
            "model": self.settings.model_name,
            "messages": [
                {"role": "system", "content": "你是连接检查器。只回复 OK。"},
                {"role": "user", "content": "连接检查"},
            ],
            "temperature": 0,
            "max_tokens": 2,
        }
        result = self._chat_completions(payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        content = self._message_content(result).strip()
        return {
            "available": True,
            "model": self.settings.model_name,
            "response_preview": content[:40],
            "elapsed_ms": elapsed_ms,
        }

    def _chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = self._effective_base_url().rstrip("/")
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelGatewayError(f"模型服务返回 HTTP {exc.code}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelGatewayError(f"模型服务调用失败: {exc}") from exc
        if not isinstance(value, dict):
            raise ModelGatewayError("模型服务响应不是 JSON 对象")
        return value

    def _effective_base_url(self) -> str:
        if self.settings.base_url:
            return self.settings.base_url
        if self.settings.api_key:
            return "https://api.openai.com/v1"
        return ""

    @staticmethod
    def _message_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelGatewayError("模型服务响应缺少 choices.message.content") from exc
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        return str(content)

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        value = content.strip()
        if value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
            if value.lower().startswith("json"):
                value = value[4:].lstrip()
        return value
