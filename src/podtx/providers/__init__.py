"""Pluggable LLM providers (shared infrastructure for `podtx nuggets`).

Mirrors the ASR engine pattern (`podtx/engines/`) and the shipped
summarize backends: a hosted provider is only used when explicitly
configured; `fake` (offline) does not go through this package.
"""

from podtx.providers.base import Provider, ProviderError
from podtx.providers.registry import (
    available_providers,
    build_provider,
    get_spec,
    normalize_backend,
    resolve_api_key,
)

__all__ = [
    "Provider",
    "ProviderError",
    "available_providers",
    "build_provider",
    "get_spec",
    "normalize_backend",
    "resolve_api_key",
]