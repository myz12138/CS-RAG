import json
import os
import re


class LLMClient:
    """
    OpenAI-compatible LLM client.

    Environment-based defaults:
    - LLM_API_KEY or OPENAI_API_KEY
    - LLM_ENDPOINT or OPENAI_BASE_URL (falls back to official OpenAI endpoint)
    - LLM_MODEL (default: gpt-4o-mini)
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"openai package is required: pip install 'openai>=1.0.0' ({e})")

        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("LLM_ENDPOINT")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.default_temperature = float(
            temperature if temperature is not None else os.environ.get("LLM_TEMPERATURE", 0.2)
        )
        self.default_max_tokens = int(
            max_tokens if max_tokens is not None else os.environ.get("LLM_MAX_TOKENS", 256)
        )

        if not self.api_key:
            raise RuntimeError("Missing API key: set LLM_API_KEY or OPENAI_API_KEY.")

        b = str(self.base_url).strip().rstrip("/")
        if not b.endswith("/v1"):
            b = b + "/v1"
        self.base_url = b

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def ask_json(self, system_prompt: str, user_prompt: str):
        """
        Ask model and parse JSON from response.
        Returns {} on parse failure.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.default_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.default_max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}$|^\[[\s\S]*\]$", content)
            if m:
                content = m.group(0)
            return json.loads(content)
        except Exception:
            return {}

    def chat(self, messages, temperature=None, max_tokens=None):
        t = self.default_temperature if temperature is None else float(temperature)
        mt = self.default_max_tokens if max_tokens is None else int(max_tokens)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=t,
            messages=messages,
            max_tokens=mt,
        )
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = getattr(resp.choices[0], "text", "")
        return (content or "").strip()
