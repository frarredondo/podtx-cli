"""models.dev catalog: model metadata (context, pricing) for validation + cost estimates.

The [models.dev](https://models.dev) catalog is a metadata-only database, **not** an
inference API. ``api.json`` maps provider id -> provider entry with a ``models`` map of
model id -> ``{limit: {context, output}, cost: {input, output}}`` (USD per million tokens).

The catalog is fetched once and cached to ``<data_dir>/models-cache.json`` so offline use
works from cache; a stale cache is served when the network is unreachable. No silent
network calls happen on the happy path — ``load_catalog`` hits the network only when the
cache is missing, stale, or ``refresh=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from podtx import __version__

MODELS_API_URL = "https://models.dev/api.json"
MODELS_CACHE_TTL_SECONDS = 60 * 60 * 24
DEFAULT_FETCH_TIMEOUT = 30.0
DEFAULT_DRY_OUTPUT_CHARS = 2000


class CatalogError(Exception):
    """Raised when the models.dev catalog cannot be fetched or cached."""


@dataclass(frozen=True)
class ModelInfo:
    """Pricing + context facts for one model from the models.dev catalog."""

    id: str
    name: str
    context_length: int | None
    output_length: int | None
    cost_input_per_million: float | None
    cost_output_per_million: float | None


@dataclass(frozen=True)
class CostEstimate:
    """Token + USD estimate for a dry run."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float | None
    cost_known: bool
    model_known: bool


def _get(url: str, timeout: float) -> httpx.Response:
    return httpx.get(
        url,
        timeout=timeout,
        headers={"User-Agent": f"podtx/{__version__}"},
    )


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_info(key: str, value: Any) -> ModelInfo | None:
    if not isinstance(value, dict):
        return None
    limit = value.get("limit")
    cost = value.get("cost")
    model_id = value.get("id") or key
    return ModelInfo(
        id=model_id,
        name=value.get("name") or model_id,
        context_length=_to_int(limit.get("context")) if isinstance(limit, dict) else None,
        output_length=_to_int(limit.get("output")) if isinstance(limit, dict) else None,
        cost_input_per_million=_to_float(cost.get("input")) if isinstance(cost, dict) else None,
        cost_output_per_million=_to_float(cost.get("output")) if isinstance(cost, dict) else None,
    )


def parse_catalog(raw: dict) -> dict[str, list[ModelInfo]]:
    """Normalize a raw models.dev provider map into provider -> sorted ModelInfo list."""
    parsed: dict[str, list[ModelInfo]] = {}
    for provider, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if not isinstance(models, dict):
            parsed[provider] = []
            continue
        info: list[ModelInfo] = []
        for key, value in models.items():
            model = _model_info(key, value)
            if model is not None:
                info.append(model)
        info.sort(key=lambda m: m.id)
        parsed[provider] = info
    return parsed


def catalog_providers(raw: dict) -> list[str]:
    """Sorted provider ids present in a raw models.dev provider map."""
    return sorted(k for k, v in raw.items() if isinstance(v, dict))


def list_models(raw: dict, provider: str) -> list[ModelInfo]:
    """Models served by ``provider`` in the catalog, sorted by model id."""
    entry = raw.get(provider)
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, dict):
        return []
    info: list[ModelInfo] = []
    for key, value in models.items():
        model = _model_info(key, value)
        if model is not None:
            info.append(model)
    info.sort(key=lambda m: m.id)
    return info


def get_model(raw: dict, provider: str, model_id: str) -> ModelInfo | None:
    """Look up a model by its catalog key or reported id, or None."""
    entry = raw.get(provider)
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, dict):
        return None
    for key, value in models.items():
        if key == model_id:
            return _model_info(key, value)
        if isinstance(value, dict) and value.get("id") == model_id:
            return _model_info(key, value)
    return None


def estimate_tokens(chars: int) -> int:
    """Rough token estimate: ~4 chars per token."""
    return chars // 4


def estimate_cost(
    raw: dict,
    provider: str,
    model_id: str,
    *,
    input_chars: int,
    output_chars: int,
) -> CostEstimate:
    """Token + USD estimate for ``model_id`` under ``provider`` in the catalog."""
    input_tokens = estimate_tokens(input_chars)
    output_tokens = estimate_tokens(output_chars)
    total = input_tokens + output_tokens
    model = get_model(raw, provider, model_id)
    if model is None:
        return CostEstimate(input_tokens, output_tokens, total, None, False, False)
    if model.cost_input_per_million is None or model.cost_output_per_million is None:
        return CostEstimate(input_tokens, output_tokens, total, None, False, True)
    cost = (
        input_tokens / 1e6 * model.cost_input_per_million
        + output_tokens / 1e6 * model.cost_output_per_million
    )
    return CostEstimate(input_tokens, output_tokens, total, cost, True, True)


def fetch_catalog(*, timeout: float = DEFAULT_FETCH_TIMEOUT) -> dict:
    """Fetch and parse the raw models.dev provider map; raise CatalogError on failure."""
    try:
        resp = _get(MODELS_API_URL, timeout)
    except httpx.RequestError as exc:
        raise CatalogError(f"models.dev fetch failed: {exc}") from exc
    if resp.status_code != 200:
        raise CatalogError(
            f"models.dev fetch failed (HTTP {resp.status_code}): {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise CatalogError(f"models.dev returned invalid JSON: {resp.text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise CatalogError("models.dev returned unexpected data shape (expected a dict of providers)")
    return data


def catalog_cache_path(data_dir: Path) -> Path:
    return data_dir / "models-cache.json"


def _read_cache(data_dir: Path) -> dict | None:
    try:
        raw = json.loads(catalog_cache_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
        return None
    return raw


def _cache_fresh(cached: dict, ttl_seconds: float) -> bool:
    fetched = cached.get("fetched_at")
    if not isinstance(fetched, str):
        return False
    try:
        when = datetime.fromisoformat(fetched)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - when <= timedelta(seconds=ttl_seconds)


def _write_cache(data_dir: Path, data: dict) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    path = catalog_cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_catalog(
    data_dir: Path,
    *,
    refresh: bool = False,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
    ttl_seconds: float = MODELS_CACHE_TTL_SECONDS,
) -> dict:
    """Raw models.dev provider map, served from fresh cache, network, or stale cache.

    Ordering: fresh cache -> fetch (persisted) -> stale cache -> CatalogError.
    """
    cached = _read_cache(data_dir)
    if cached is not None and not refresh and _cache_fresh(cached, ttl_seconds):
        return cached["data"]
    try:
        data = fetch_catalog(timeout=timeout)
    except CatalogError:
        if cached is not None:
            return cached["data"]
        raise
    _write_cache(data_dir, data)
    return data