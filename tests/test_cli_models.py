from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from podtx.cli import app
from podtx.providers.catalog import CatalogError

runner = CliRunner()

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
            "vague/none": {
                "id": "vague/none",
                "name": "Vague None",
            },
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
        },
    },
    "openai": {"id": "openai", "name": "OpenAI", "models": {}},
}


def _load_fake(tmp_path: Path):
    return lambda data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400: FAKE_API


def test_models_lists_supported_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "lmstudio" in result.stdout and "2" in result.stdout
    assert "openrouter" in result.stdout and "OpenRouter" in result.stdout
    assert "anthropic" not in result.stdout


def test_models_provider_listing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--provider", "openrouter", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Claude Sonnet 4" in result.stdout
    assert "200,000" in result.stdout or "200000" in result.stdout
    assert "GPT-4o mini" in result.stdout
    assert "Qwen 2.5 14B" not in result.stdout


def test_models_provider_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["models", "--provider", "openrouter", "--limit", "1", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Claude Sonnet 4" in result.stdout
    assert "GPT-4o mini" not in result.stdout


def test_models_unknown_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--provider", "nope", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "nope" in result.stdout or "nope" in result.stderr


def test_models_model_found_in_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        ["models", "--provider", "lmstudio", "--model", "qwen/qwen2.5-14b", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Qwen 2.5 14B" in result.stdout
    assert "32,768" in result.stdout or "32768" in result.stdout


def test_models_model_search_across_providers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--model", "anthropic/claude-sonnet-4", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "openrouter" in result.stdout
    assert "Claude Sonnet 4" in result.stdout


def test_models_model_not_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--model", "nope/nope", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "nope/nope" in result.stdout or "nope/nope" in result.stderr


def test_models_model_not_found_in_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        ["models", "--provider", "openai", "--model", "gpt-oss-20b", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code != 0


def test_models_catalog_unavailable(tmp_path, monkeypatch) -> None:
    def boom(data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400):
        raise CatalogError("no cache and offline")

    monkeypatch.setattr("podtx.cli.load_catalog", boom)
    result = runner.invoke(app, ["models", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "no cache and offline" in result.stdout or "no cache and offline" in result.stderr


def test_models_refresh_forces_fetch(tmp_path, monkeypatch) -> None:
    calls = {}

    def record(data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400):
        calls["refresh"] = refresh
        return FAKE_API

    monkeypatch.setattr("podtx.cli.load_catalog", record)
    result = runner.invoke(app, ["models", "--refresh", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls["refresh"] is True

def test_models_provider_listing_shows_unknowns(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["models", "--provider", "lmstudio", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Vague None" in result.stdout
    assert "—" in result.stdout


def test_models_no_configured_providers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "podtx.cli.load_catalog",
        lambda data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400: {
            "some-other-vendor": {"id": "some-other-vendor", "name": "Other", "models": {}}
        },
    )
    result = runner.invoke(app, ["models", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "No configured providers found" in result.stdout
