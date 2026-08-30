from __future__ import annotations

import json

import httpx

from podtx.providers.base import Provider, ProviderError, post_json

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider:
    """Calls the Anthropic Messages API (`/v1/messages`)."""

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
        url = f"{self.base_url}/messages"
        if not self.api_key:
            raise ProviderError(f"{self.name} requires an API key")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") != "system"
        ]
        if not chat:
            raise ProviderError(f"{self.name} requires at least one user message")
        body: dict = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "messages": chat,
        }
        if system_parts:
            body["system"] = "\n\n".join(part for part in system_parts if part)
        with httpx.Client(timeout=timeout) as client:
            data = post_json(client, url, headers, body, name=self.name)
        try:
            blocks = data["content"]
            content = "\n".join(
                str(block.get("text", ""))
                for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"{self.name} response missing content: {data!r}") from exc
        if not content:
            raise ProviderError(f"{self.name} returned empty text content: {data!r}")
        return content
