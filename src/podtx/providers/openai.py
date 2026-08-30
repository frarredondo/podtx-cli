from __future__ import annotations

import json

import httpx

from podtx.providers.base import Provider, ProviderError, post_json


class OpenAICompatibleProvider:
    """Calls any OpenAI-compatible `/chat/completions` endpoint.

    Serves OpenAI, OpenRouter, OpenCode Zen Go, LM Studio, and arbitrary
    self-hosted (or proxied) servers via a custom ``base_url``.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float = 120.0,
        temperature: float = 0.3,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=timeout) as client:
            data = post_json(client, url, headers, body, name=self.name)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.name} response missing choices/message/content: {data!r}"
            ) from exc
        return content if isinstance(content, str) else json.dumps(content)
