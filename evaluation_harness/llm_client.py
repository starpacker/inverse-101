"""Thin OpenAI-compatible chat completions client using requests."""

import logging
import time
from typing import Any

import requests

from .config import LLMConfig

log = logging.getLogger(__name__)


class LLMClient:
    """Calls any OpenAI-compatible chat/completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Send a chat completion request.

        Returns
        -------
        (response_text, usage_dict)
            usage_dict has keys ``prompt_tokens`` and ``completion_tokens``.
        """
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if stop:
            body["stop"] = stop

        # Simple retry: one retry with 5 s backoff
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    log.warning("LLM request failed (%s), retrying in 5 s…", exc)
                    time.sleep(5)
                else:
                    raise RuntimeError(f"LLM request failed after retry: {exc}") from exc

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        usage_out = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        self._total_usage["prompt_tokens"] += usage_out["prompt_tokens"]
        self._total_usage["completion_tokens"] += usage_out["completion_tokens"]

        log.debug(
            "LLM call: +%d prompt, +%d completion tokens",
            usage_out["prompt_tokens"],
            usage_out["completion_tokens"],
        )
        return text, usage_out

    # ------------------------------------------------------------------
    @property
    def total_usage(self) -> dict[str, int]:
        """Accumulated token counts across all calls."""
        return dict(self._total_usage)
