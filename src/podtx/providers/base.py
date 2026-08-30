from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Raised for provider failures: missing config, network, or HTTP errors."""


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