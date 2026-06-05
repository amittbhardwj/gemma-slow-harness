from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import HarnessConfig


Message = dict[str, str]


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]
    elapsed_sec: float


class LocalLLMClient:
    """Small OpenAI/Ollama-compatible HTTP client using only stdlib."""

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        provider = self.cfg.provider.lower()
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens
        if provider == "ollama":
            return self._chat_ollama(messages, temperature=temperature, max_tokens=max_tokens)
        if provider == "openai":
            return self._chat_openai_compatible(messages, temperature=temperature, max_tokens=max_tokens)
        raise LLMError(f"Unsupported provider: {self.cfg.provider}. Use 'ollama' or 'openai'.")

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach local model server at {url}: {exc}") from exc

    def _chat_ollama(self, messages: list[Message], *, temperature: float, max_tokens: int) -> LLMResponse:
        url = f"{self.cfg.base_url}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        started = time.perf_counter()
        raw = self._post_json(url, payload)
        elapsed = time.perf_counter() - started
        try:
            content = raw["message"]["content"]
        except KeyError as exc:
            raise LLMError(f"Unexpected Ollama response: {raw}") from exc
        return LLMResponse(content=content.strip(), raw=raw, elapsed_sec=elapsed)

    def _chat_openai_compatible(self, messages: list[Message], *, temperature: float, max_tokens: int) -> LLMResponse:
        base = self.cfg.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        started = time.perf_counter()
        raw = self._post_json(url, payload, headers=headers)
        elapsed = time.perf_counter() - started
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenAI-compatible response: {raw}") from exc
        return LLMResponse(content=content.strip(), raw=raw, elapsed_sec=elapsed)
