from __future__ import annotations

import os
from dataclasses import dataclass

from podtx.config import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_OPENCODE_BASE_URL,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
)
from podtx.providers.anthropic import AnthropicProvider
from podtx.providers.base import Provider, ProviderError
from podtx.providers.openai import OpenAICompatibleProvider

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    provider_class: type[AnthropicProvider] | type[OpenAICompatibleProvider]
    base_url: str | None
    default_model: str | None
    env_var: str | None
    keychain_service: str | None
    requires_api_key: bool
    requires_model: bool = False


_REGISTRY: dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec(
        name="openrouter",
        provider_class=OpenAICompatibleProvider,
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        default_model=DEFAULT_OPENROUTER_MODEL,
        env_var="OPENROUTER_API_KEY",
        keychain_service="podtx-openrouter",
        requires_api_key=True,
    ),
    "opencode": ProviderSpec(
        name="opencode",
        provider_class=OpenAICompatibleProvider,
        base_url=DEFAULT_OPENCODE_BASE_URL,
        default_model=DEFAULT_OPENCODE_MODEL,
        env_var="OPENCODE_API_KEY",
        keychain_service="podtx-opencode",
        requires_api_key=True,
    ),
    "openai": ProviderSpec(
        name="openai",
        provider_class=OpenAICompatibleProvider,
        base_url=OPENAI_DEFAULT_BASE_URL,
        default_model=None,
        env_var="OPENAI_API_KEY",
        keychain_service="podtx-openai",
        requires_api_key=True,
        requires_model=True,
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        provider_class=AnthropicProvider,
        base_url=ANTHROPIC_DEFAULT_BASE_URL,
        default_model=None,
        env_var="ANTHROPIC_API_KEY",
        keychain_service="podtx-anthropic",
        requires_api_key=True,
        requires_model=True,
    ),
    "lmstudio": ProviderSpec(
        name="lmstudio",
        provider_class=OpenAICompatibleProvider,
        base_url=DEFAULT_LMSTUDIO_BASE_URL,
        default_model=None,
        env_var=None,
        keychain_service=None,
        requires_api_key=False,
        requires_model=True,
    ),
}

_ALIASES: dict[str, str] = {
    "local": "lmstudio",
    "openai-compatible": "openai",
}


def normalize_backend(name: str) -> str:
    b = name.strip().lower()
    return _ALIASES.get(b, b)


def available_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> ProviderSpec:
    key = normalize_backend(name)
    if key not in _REGISTRY:
        raise ValueError(f"Unknown provider {name!r}. Available: {', '.join(available_providers())}")
    return _REGISTRY[key]


def _keychain_get(service: str, account: str) -> str | None:
    from podtx.keychain import get_api_key

    return get_api_key(service, account)


def resolve_api_key(
    name: str,
    *,
    api_key: str | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> str | None:
    """Resolve a provider API key: flag > settings > env > Keychain."""
    if api_key:
        return api_key
    if settings_api_key:
        return settings_api_key
    spec = get_spec(name)
    if spec.env_var and (v := os.environ.get(spec.env_var)):
        return v
    if spec.keychain_service:
        svc = service or spec.keychain_service
        acct = account or "api-key"
        return _keychain_get(svc, acct)
    return None


def build_provider(
    name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> Provider:
    """Build a configured provider for ``name``, resolving defaults and keys.

    Raises ProviderError when the backend needs a model/API key/base URL that
    is not configured. Never falls through to a silent network call.
    """
    key = normalize_backend(name)
    if key not in _REGISTRY:
        known = ", ".join(available_providers())
        raise ProviderError(f"Unknown provider {name!r}. Available: {known}")
    spec = _REGISTRY[key]

    resolved_model = model or spec.default_model
    if spec.requires_model and not resolved_model:
        raise ProviderError(f"{key} provider requires --model (no default model)")

    resolved_base = base_url or spec.base_url
    if not resolved_base:
        raise ProviderError(f"{key} provider requires --base-url (no default endpoint)")

    resolved_key = resolve_api_key(
        key,
        api_key=api_key,
        settings_api_key=settings_api_key,
        service=service,
        account=account,
    )
    if spec.requires_api_key and not resolved_key:
        hint = "`podtx auth set ...`" if spec.keychain_service else "--api-key"
        raise ProviderError(
            f"{key} provider requires an API key (--api-key, env {spec.env_var}, or {hint})"
        )

    return spec.provider_class(key, resolved_base, resolved_model, resolved_key)
