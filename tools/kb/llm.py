"""Provider-agnostic LLM client for the ingest subagents.

Supports any OpenAI-compatible chat API (OpenAI, xAI/Grok, or a local
server such as Ollama/LM Studio with an OpenAI-compatible endpoint).
Roles map to models per the plan: a mid-tier model for bulk work
(extractor, proofreader, structurer), a stronger model for verifier and
conflict detection. Configure with environment variables:

  KB_LLM_BASE_URL      e.g. https://api.x.ai/v1 or http://localhost:11434/v1
  KB_LLM_API_KEY
  KB_MODEL_BULK        default model for extractor/proofreader/structurer
  KB_MODEL_STRONG      model for verifier/conflict/hard OCR

Every call records a cost/usage line so runs can be audited.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class LLMError(RuntimeError):
    pass


@dataclass
class UsageLog:
    entries: list[dict] = field(default_factory=list)

    def record(self, role: str, model: str, usage: dict | None) -> None:
        self.entries.append(
            {
                "role": role,
                "model": model,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "usage": usage or {},
            }
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        bulk_model: str | None = None,
        strong_model: str | None = None,
        usage: UsageLog | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("KB_LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("KB_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.bulk_model = bulk_model or os.environ.get("KB_MODEL_BULK", "gpt-4o-mini")
        self.strong_model = strong_model or os.environ.get("KB_MODEL_STRONG", self.bulk_model)
        self.usage = usage or UsageLog()
        self.timeout = timeout
        self._client = None

    def model_for(self, role: str) -> str:
        return self.strong_model if role in {"verifier", "conflict", "ocr-hard"} else self.bulk_model

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("openai package is required for LLM-backed steps") from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    def chat(
        self,
        *,
        role: str,
        system: str,
        user: str,
        images: list[bytes] | None = None,
        json_mode: bool = False,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """One chat completion. Raises LLMError on failure; callers decide fallback."""
        client = self._get_client()
        chosen = model or self.model_for(role)
        content: list[dict] | str
        if images:
            import base64

            parts: list[dict] = [{"type": "text", "text": user}]
            for blob in images:
                b64 = base64.b64encode(blob).decode("ascii")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            content = parts
        else:
            content = user

        kwargs: dict = {
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # provider-specific exception types
            raise LLMError(f"LLM call failed for role={role} model={chosen}: {exc}") from exc

        usage_dict = None
        if getattr(response, "usage", None) is not None:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        self.usage.record(role, chosen, usage_dict)
        message = response.choices[0].message.content
        if message is None:
            raise LLMError(f"empty response for role={role} model={chosen}")
        return message
