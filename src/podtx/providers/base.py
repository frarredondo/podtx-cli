from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx


class ProviderError(Exception):
    """Raised for provider failures: missing config, network, or HTTP errors."""


def post_json(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict,
    *,
    name: str,
) -> dict:
    """POST JSON and parse the JSON response; raise ProviderError on any failure."""
    try:
        resp = client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        raise ProviderError(f"{name} request failed: {exc}") from exc
    if resp.status_code != 200:
        preview = resp.text[:500]
        raise ProviderError(f"{name} request failed ({resp.status_code}): {preview}")
    try:
        data = resp.json()
    except Exception as exc:
        preview = resp.text[:500].strip()
        raise ProviderError(
            f"{name} returned invalid JSON response (HTTP {resp.status_code}): {preview!r}"
        ) from exc
    return data


@runtime_checkable
class Provider(Protocol):
    """A chat-completion provider (OpenAI-compatible or Anthropic)."""

    name: str

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float = 120.0,
        temperature: float = 0.3,
    ) -> str:
        """Return the raw completion text for the given chat messages."""
        ...
