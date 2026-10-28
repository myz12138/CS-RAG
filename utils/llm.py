import json
import os
import re


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None,
                 temperature=None, max_tokens=None):
        from openai import OpenAI

        self.backend = os.getenv("LLM_BACKEND", "api").strip().lower()
        if self.backend not in ("api", "local"):
            raise ValueError("LLM_BACKEND must be api or local.")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.default_temperature = float(
            temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0.2")
        )
        self.default_max_tokens = int(
            max_tokens if max_tokens is not None else os.getenv("LLM_MAX_TOKENS", "256")
        )
        if self.backend == "local":
            self.base_url = base_url or os.getenv("LLM_ENDPOINT", "")
            self.api_key = api_key or os.getenv("LLM_LOCAL_API_KEY") or "EMPTY"
            if not self.base_url or not (model or os.getenv("LLM_MODEL")):
                raise ValueError("Local inference requires LLM_ENDPOINT and LLM_MODEL.")
        else:
            self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            self.base_url = base_url or os.getenv("LLM_ENDPOINT") or os.getenv("OPENAI_BASE_URL", "")
            if not self.api_key:
                raise ValueError("Set LLM_API_KEY or OPENAI_API_KEY.")
        self.base_url = self.base_url.strip().rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[:-len("/chat/completions")]
        if self.base_url and not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)

    def ask_json(self, system_prompt, user_prompt):
        try:
            content = self.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            match = re.search(r"\{[\s\S]*\}$|^\[[\s\S]*\]$", content)
            return json.loads(match.group(0) if match else content)
        except Exception:
            return {}

    def chat(self, messages, temperature=None, max_tokens=None):
        kwargs = {}
        if self.backend == "local" and "qwen3" in self.model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.default_temperature if temperature is None else float(temperature),
            max_tokens=self.default_max_tokens if max_tokens is None else int(max_tokens),
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
