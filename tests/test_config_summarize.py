from __future__ import annotations

from pathlib import Path

from podtx.config import load_settings


def test_config_defaults_summarize(tmp_path: Path) -> None:
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.summarize_backend == "fake"
    assert s.summarize_model is None
    assert s.summarize_base_url is None
    assert s.summarize_api_key is None
    assert s.summarize_timeout == 60.0
    assert s.summarize_temperature == 0.3
    assert s.summarize_max_input_chars is None


def test_config_toml_summarize(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        """
summarize_backend = "openrouter"
summarize_model = "test-model"
summarize_base_url = "https://example.com/v1"
summarize_api_key = "sk-test"
summarize_api_key_service = "svc"
summarize_api_key_account = "acct"
summarize_timeout = 10
summarize_temperature = 0.5
summarize_max_input_chars = 1234
""",
        encoding="utf-8",
    )
    s = load_settings(config_path=p)
    assert s.summarize_backend == "openrouter"
    assert s.summarize_model == "test-model"
    assert s.summarize_base_url == "https://example.com/v1"
    assert s.summarize_api_key == "sk-test"
    assert s.summarize_api_key_service == "svc"
    assert s.summarize_api_key_account == "acct"
    assert s.summarize_timeout == 10.0
    assert s.summarize_temperature == 0.5
    assert s.summarize_max_input_chars == 1234


def test_config_env_summarize(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_BACKEND", "lmstudio")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_MODEL", "env-model")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_API_KEY", "env-key")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_API_KEY_SERVICE", "env-svc")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_API_KEY_ACCOUNT", "env-acct")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_TIMEOUT", "5")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_TEMPERATURE", "0.9")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_MAX_INPUT_CHARS", "999")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_backend == "lmstudio"
    assert s.summarize_model == "env-model"
    assert s.summarize_base_url == "https://env.example/v1"
    assert s.summarize_api_key == "env-key"
    assert s.summarize_api_key_service == "env-svc"
    assert s.summarize_api_key_account == "env-acct"
    assert s.summarize_timeout == 5.0
    assert s.summarize_temperature == 0.9
    assert s.summarize_max_input_chars == 999


def test_config_env_provider_aliases(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_api_key == "or-key"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_api_key == "oc-key"
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "https://lm.example/v1")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_base_url == "https://lm.example/v1"
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://oc.example/v1")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_base_url == "https://oc.example/v1"
    monkeypatch.delenv("OPENCODE_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://or.example/v1")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.summarize_base_url == "https://or.example/v1"


def test_config_cli_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_SUMMARIZE_BACKEND", "fake")
    s = load_settings(summarize_backend="openrouter", summarize_model="cli-model", summarize_base_url="https://cli/v1", summarize_api_key="cli-key", summarize_api_key_service="cli-svc", summarize_api_key_account="cli-acct", summarize_timeout=77, summarize_temperature=0.1, summarize_max_input_chars=42)
    assert s.summarize_backend == "openrouter"
    assert s.summarize_model == "cli-model"
    assert s.summarize_base_url == "https://cli/v1"
    assert s.summarize_api_key == "cli-key"
    assert s.summarize_api_key_service == "cli-svc"
    assert s.summarize_api_key_account == "cli-acct"
    assert s.summarize_timeout == 77
    assert s.summarize_temperature == 0.1
    assert s.summarize_max_input_chars == 42


def test_config_toml_none_max_input(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('summarize_max_input_chars = 100\n', encoding="utf-8")
    s = load_settings(config_path=p)
    assert s.summarize_max_input_chars == 100
