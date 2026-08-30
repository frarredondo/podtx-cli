from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from podtx.providers import catalog as catalog_mod
from podtx.providers.catalog import (
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

FAKE_API = {
    "lmstudio": {
        "id": "lmstudio",
        "name": "LM Studio",
        "models": {
            "gpt-oss-20b": {
                "id": "openai/gpt-oss-20b",
                "name": "GPT-OSS 20B",
                "limit": {"context": 131072, "output": 32768},
                "cost": {"input": 0.0, "output": 0.0},
            },
            "qwen/qwen2.5-14b": {
                "id": "qwen/qwen2.5-14b",
                "name": "Qwen 2.5 14B",
                "limit": {"context": 32768},
            },
            "not-a-dict": "skip-me",
        },
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter",
        "models": {
            "anthropic/claude-sonnet-4": {
                "id": "anthropic/claude-sonnet-4",
                "name": "Claude Sonnet 4",
                "limit": {"context": 200000, "output": 64000},
                "cost": {"input": 3.0, "output": 15.0},
            },
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "name": "GPT-4o mini",
                "limit": {"context": 128000},
                "cost": {"input": 0.15, "output": 0.6},
            },
            "mistral/unknown-cost": {
                "id": "mistral/unknown-cost",
                "name": "Unknown Cost",
                "limit": {"context": 32000},
                "cost": {"input": 0.2},
            },
        },
    },
    "openai": {"id": "openai", "name": "OpenAI", "models": {}},
    "weird": "not-a-provider",
}


def _with_fetch(monkeypatch, payload, raises=None):
    def fake_get(url, timeout):
        assert url == catalog_mod.MODELS_API_URL
        if raises is not None:
            raise raises
        return payload

    monkeypatch.setattr(catalog_mod, "_get", fake_get)


# --- parse / lookup ---------------------------------------------------------


def test_parse_catalog_shape() -> None:
    parsed = parse_catalog(FAKE_API)
    assert set(parsed) == {"lmstudio", "openrouter", "openai"}
    lm = parsed["lmstudio"]
    assert len(lm) == 2
    assert lm[0].id == "openai/gpt-oss-20b"
    assert lm[0].name == "GPT-OSS 20B"
    assert lm[0].context_length == 131072
    assert lm[0].output_length == 32768
    assert lm[0].cost_input_per_million == 0.0
    assert lm[0].cost_output_per_million == 0.0
    qwen = lm[1]
    assert qwen.id == "qwen/qwen2.5-14b"
    assert qwen.output_length is None
    assert qwen.cost_input_per_million is None
    assert parsed["openai"] == []


def test_catalog_providers_sorted() -> None:
    assert catalog_providers(FAKE_API) == ["lmstudio", "openai", "openrouter"]


def test_parse_catalog_non_dict_provider_skipped() -> None:
    assert "weird" not in parse_catalog(FAKE_API)


def test_list_models_sorted_by_id() -> None:
    ids = [m.id for m in list_models(FAKE_API, "lmstudio")]
    assert ids == ["openai/gpt-oss-20b", "qwen/qwen2.5-14b"]


def test_list_models_unknown_provider_empty() -> None:
    assert list_models(FAKE_API, "nope") == []


def test_list_models_non_dict_provider_empty() -> None:
    assert list_models(FAKE_API, "weird") == []
    assert list_models(FAKE_API, "openai") == []


def test_get_model_by_key() -> None:
    m = get_model(FAKE_API, "lmstudio", "gpt-oss-20b")
    assert m is not None
    assert m.id == "openai/gpt-oss-20b"


def test_get_model_by_reported_id() -> None:
    m = get_model(FAKE_API, "lmstudio", "openai/gpt-oss-20b")
    assert m is not None
    assert m.context_length == 131072


def test_get_model_missing() -> None:
    assert get_model(FAKE_API, "lmstudio", "nope") is None
    assert get_model(FAKE_API, "nope", "gpt-oss-20b") is None
    assert get_model(FAKE_API, "weird", "gpt-oss-20b") is None
    assert get_model(FAKE_API, "openai", "anything") is None
    assert get_model(FAKE_API, "lmstudio", "not-a-dict") is None


# --- token / cost estimates -------------------------------------------------


def test_estimate_tokens() -> None:
    assert estimate_tokens(4096) == 1024
    assert estimate_tokens(1) == 0
    assert estimate_tokens(0) == 0


def test_estimate_cost_known_pricing() -> None:
    est = estimate_cost(
        FAKE_API,
        "openrouter",
        "anthropic/claude-sonnet-4",
        input_chars=4000,
        output_chars=2000,
    )
    assert est.input_tokens == 1000
    assert est.output_tokens == 500
    assert est.total_tokens == 1500
    assert est.model_known is True
    assert est.cost_known is True
    assert est.cost_usd == pytest.approx(0.0105)


def test_estimate_cost_zero_pricing() -> None:
    est = estimate_cost(
        FAKE_API, "lmstudio", "gpt-oss-20b", input_chars=1000, output_chars=1000
    )
    assert est.cost_known is True
    assert est.cost_usd == 0.0


def test_estimate_cost_unknown_model() -> None:
    est = estimate_cost(
        FAKE_API, "openrouter", "nope/x", input_chars=1000, output_chars=500
    )
    assert est.input_tokens == 250
    assert est.output_tokens == 125
    assert est.total_tokens == 375
    assert est.model_known is False
    assert est.cost_usd is None


def test_estimate_cost_partial_pricing_treated_unknown() -> None:
    est = estimate_cost(
        FAKE_API, "openrouter", "mistral/unknown-cost", input_chars=1000, output_chars=500
    )
    assert est.model_known is True
    assert est.cost_known is False
    assert est.cost_usd is None


def test_estimate_cost_no_pricing() -> None:
    est = estimate_cost(
        FAKE_API, "lmstudio", "qwen/qwen2.5-14b", input_chars=1000, output_chars=500
    )
    assert est.model_known is True
    assert est.cost_known is False


# --- fetch ------------------------------------------------------------------


def test_fetch_catalog_ok(monkeypatch) -> None:
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert fetch_catalog() == FAKE_API


def test_fetch_catalog_http_error(monkeypatch) -> None:
    _with_fetch(monkeypatch, httpx.Response(500, text="boom"))
    with pytest.raises(CatalogError):
        fetch_catalog()


def test_fetch_catalog_bad_json(monkeypatch) -> None:
    _with_fetch(monkeypatch, httpx.Response(200, text="not-json{{"))
    with pytest.raises(CatalogError):
        fetch_catalog()


def test_fetch_catalog_not_dict(monkeypatch) -> None:
    _with_fetch(monkeypatch, httpx.Response(200, json=["not", "a", "dict"]))
    with pytest.raises(CatalogError):
        fetch_catalog()


def test_fetch_catalog_request_error(monkeypatch) -> None:
    _with_fetch(monkeypatch, None, raises=httpx.ConnectError("boom"))
    with pytest.raises(CatalogError):
        fetch_catalog()


# --- cache / load -----------------------------------------------------------


def _cache_path(tmp_path: Path) -> Path:
    return catalog_mod.catalog_cache_path(tmp_path)


def _write_cache(tmp_path: Path, data, age_hours=0.0) -> None:
    fetched = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    _cache_path(tmp_path).write_text(
        json.dumps({"fetched_at": fetched.isoformat(), "data": data}), encoding="utf-8"
    )


def test_load_no_cache_fetches_and_persists(tmp_path, monkeypatch) -> None:
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    raw = load_catalog(tmp_path, ttl_seconds=86400)
    assert raw == FAKE_API
    cached = json.loads(_cache_path(tmp_path).read_text(encoding="utf-8"))
    assert cached["data"] == FAKE_API
    assert "fetched_at" in cached


def test_load_fresh_cache_does_not_fetch(tmp_path, monkeypatch) -> None:
    _write_cache(tmp_path, FAKE_API, age_hours=1)

    def boom(url, timeout):
        raise AssertionError("should not fetch")

    monkeypatch.setattr(catalog_mod, "_get", boom)
    raw = load_catalog(tmp_path, ttl_seconds=86400)
    assert raw == FAKE_API


def test_load_stale_cache_serves_stale_when_unreachable(tmp_path, monkeypatch) -> None:
    _write_cache(tmp_path, FAKE_API, age_hours=25)
    _with_fetch(monkeypatch, None, raises=httpx.ConnectError("offline"))
    raw = load_catalog(tmp_path, ttl_seconds=86400)
    assert raw == FAKE_API


def test_load_no_cache_unreachable_raises(tmp_path, monkeypatch) -> None:
    _with_fetch(monkeypatch, None, raises=httpx.ConnectError("offline"))
    with pytest.raises(CatalogError):
        load_catalog(tmp_path, ttl_seconds=86400)


def test_load_stale_cache_refreshes_and_persists(tmp_path, monkeypatch) -> None:
    newer = {"openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {}}}
    _write_cache(tmp_path, FAKE_API, age_hours=25)
    _with_fetch(monkeypatch, httpx.Response(200, json=newer))
    raw = load_catalog(tmp_path, ttl_seconds=86400)
    assert raw == newer
    cached = json.loads(_cache_path(tmp_path).read_text(encoding="utf-8"))
    assert cached["data"] == newer


def test_load_refresh_forces_fetch(tmp_path, monkeypatch) -> None:
    newer = {"openai": {"id": "openai", "name": "OpenAI", "models": {}}}
    _write_cache(tmp_path, FAKE_API, age_hours=0)
    _with_fetch(monkeypatch, httpx.Response(200, json=newer))
    raw = load_catalog(tmp_path, ttl_seconds=86400, refresh=True)
    assert raw == newer


def test_load_malformed_cache_fetches(tmp_path, monkeypatch) -> None:
    _cache_path(tmp_path).write_text("not-json{{", encoding="utf-8")
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=86400) == FAKE_API


def test_load_cache_missing_keys_fetches(tmp_path, monkeypatch) -> None:
    _cache_path(tmp_path).write_text(json.dumps({"foo": 1}), encoding="utf-8")
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=86400) == FAKE_API


def test_load_cache_bad_fetched_at_fetches(tmp_path, monkeypatch) -> None:
    _cache_path(tmp_path).write_text(
        json.dumps({"fetched_at": "garbage", "data": FAKE_API}), encoding="utf-8"
    )
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=86400) == FAKE_API


def test_load_cache_data_not_dict_fetches(tmp_path, monkeypatch) -> None:
    _cache_path(tmp_path).write_text(
        json.dumps({"fetched_at": "garbage", "data": [1, 2]}), encoding="utf-8"
    )
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=86400) == FAKE_API


def test_load_stale_threshold(tmp_path, monkeypatch) -> None:
    _write_cache(tmp_path, FAKE_API, age_hours=59 / 60)
    _with_fetch(monkeypatch, None, raises=httpx.ConnectError("should not fetch"))
    assert load_catalog(tmp_path, ttl_seconds=3600) == FAKE_API
    _write_cache(tmp_path, FAKE_API, age_hours=61 / 60)
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=3600) == FAKE_API


def test_model_info_is_frozen() -> None:
    m = ModelInfo("a", "b", None, None, None, None)
    with pytest.raises(Exception):
        m.id = "x"  # type: ignore[misc]


def test_cost_estimate_is_frozen() -> None:
    c = CostEstimate(1, 2, 3, None, False, False)
    with pytest.raises(Exception):
        c.total_tokens = 99  # type: ignore[misc]

def test_load_cache_fetched_at_not_string_fetches(tmp_path, monkeypatch) -> None:
    _cache_path(tmp_path).write_text(
        json.dumps({"fetched_at": 12345, "data": FAKE_API}), encoding="utf-8"
    )
    _with_fetch(monkeypatch, httpx.Response(200, json=FAKE_API))
    assert load_catalog(tmp_path, ttl_seconds=86400) == FAKE_API


def test_parse_catalog_non_dict_models_empty() -> None:
    parsed = parse_catalog({"broken": {"id": "broken", "name": "Broken", "models": "oops"}})
    assert parsed["broken"] == []


def test_get_real_http_error(monkeypatch) -> None:
    with pytest.raises(httpx.RequestError):
        catalog_mod._get("http://127.0.0.1:1/", timeout=0.5)
