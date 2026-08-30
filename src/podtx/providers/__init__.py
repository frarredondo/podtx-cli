"""Pluggable LLM providers (shared infrastructure for `podtx nuggets`).

Mirrors the ASR engine pattern (`podtx/engines/`) and the shipped
summarize backends: a hosted provider is only used when explicitly
configured; `fake` (offline) does not go through this package.
"""

from podtx.providers.base import Provider, ProviderError
from podtx.providers.catalog import (
    DEFAULT_DRY_OUTPUT_CHARS,
    CatalogError,
    CostEstimate,
    ModelInfo,
    catalog_providers,
    estimate_cost,
    estimate_tokens,
    fetch_catalog,
    get_model,
    list_models,
    load_catalog,
    parse_catalog,
)
from podtx.providers.registry import (
    available_providers,
    build_provider,
    get_spec,
    normalize_backend,
    resolve_api_key,
)

__all__ = [
    "DEFAULT_DRY_OUTPUT_CHARS",
    "Provider",
    "ProviderError",
    "CatalogError",
    "CostEstimate",
    "ModelInfo",
    "available_providers",
    "build_provider",
    "catalog_providers",
    "estimate_cost",
    "estimate_tokens",
    "fetch_catalog",
    "get_model",
    "get_spec",
    "list_models",
    "load_catalog",
    "normalize_backend",
    "parse_catalog",
    "resolve_api_key",
]