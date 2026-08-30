from __future__ import annotations

import json

import httpx

from podtx.providers.base import Provider, ProviderError


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
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=body)
        except httpx.RequestError as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc
        if resp.status_code != 200:
            preview = resp.text[:500]
            raise ProviderError(
                f"{self.name} request failed ({resp.status_code}): {preview}"
            )
        try:
            data = resp.json()
        except Exception as exc:
            preview = resp.text[:500].strip()
            raise ProviderError(
                f"{self.name} returned invalid JSON response (HTTP {resp.status_code}): {preview!r}"
            ) from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.name} response missing choices/message/content: {data!r}"
            ) from exc
        return content if isinstance(content, str) else json.dumps(content)