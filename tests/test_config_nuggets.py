from __future__ import annotations

from pathlib import Path

from podtx.config import load_settings


def test_config_defaults_nuggets(tmp_path: Path) -> None:
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.nuggets_backend == "fake"
    assert s.nuggets_model is None
    assert s.nuggets_base_url is None
    assert s.nuggets_api_key is None
    assert s.nuggets_api_key_service is None
    assert s.nuggets_api_key_account is None
    assert s.nuggets_timeout == 120.0
    assert s.nuggets_temperature == 0.3
    assert s.nuggets_max_input_chars is None


def test_config_toml_nuggets(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        """
nuggets_backend = "openrouter"
nuggets_model = "test-model"
nuggets_base_url = "https://example.com/v1"
nuggets_api_key = "sk-test"
nuggets_api_key_service = "svc"
nuggets_api_key_account = "acct"
nuggets_timeout = 10
nuggets_temperature = 0.5
nuggets_max_input_chars = 1234
""",
        encoding="utf-8",
    )
    s = load_settings(config_path=p)
    assert s.nuggets_backend == "openrouter"
    assert s.nuggets_model == "test-model"
    assert s.nuggets_base_url == "https://example.com/v1"
    assert s.nuggets_api_key == "sk-test"
    assert s.nuggets_api_key_service == "svc"
    assert s.nuggets_api_key_account == "acct"
    assert s.nuggets_timeout == 10.0
    assert s.nuggets_temperature == 0.5
    assert s.nuggets_max_input_chars == 1234


def test_config_env_nuggets(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_BACKEND", "lmstudio")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_MODEL", "env-model")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_API_KEY", "env-key")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_API_KEY_SERVICE", "env-svc")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_API_KEY_ACCOUNT", "env-acct")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_TIMEOUT", "5")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_TEMPERATURE", "0.9")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_MAX_INPUT_CHARS", "999")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.nuggets_backend == "lmstudio"
    assert s.nuggets_model == "env-model"
    assert s.nuggets_base_url == "https://env.example/v1"
    assert s.nuggets_api_key == "env-key"
    assert s.nuggets_api_key_service == "env-svc"
    assert s.nuggets_api_key_account == "env-acct"
    assert s.nuggets_timeout == 5.0
    assert s.nuggets_temperature == 0.9
    assert s.nuggets_max_input_chars == 999


def test_config_env_provider_aliases(monkeypatch) -> None:
    for env_name, expected_key in [
        ("OPENROUTER_API_KEY", "or-key"),
        ("OPENCODE_API_KEY", "oc-key"),
        ("OPENAI_API_KEY", "oa-key"),
        ("ANTHROPIC_API_KEY", "an-key"),
    ]:
        monkeypatch.setenv(env_name, expected_key)
        s = load_settings(config_path=Path("/tmp/missing.toml"))
        assert s.nuggets_api_key == expected_key
        monkeypatch.delenv(env_name, raising=False)

    for env_name, expected_url in [
        ("OPENROUTER_BASE_URL", "https://or.example/v1"),
        ("OPENCODE_BASE_URL", "https://oc.example/v1"),
        ("OPENAI_BASE_URL", "https://oa.example/v1"),
        ("ANTHROPIC_BASE_URL", "https://an.example/v1"),
        ("LMSTUDIO_BASE_URL", "https://lm.example/v1"),
    ]:
        monkeypatch.setenv(env_name, expected_url)
        s = load_settings(config_path=Path("/tmp/missing.toml"))
        assert s.nuggets_base_url == expected_url
        monkeypatch.delenv(env_name, raising=False)


def test_config_cli_overrides_nuggets(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_NUGGETS_BACKEND", "fake")
    s = load_settings(
        nuggets_backend="openrouter",
        nuggets_model="cli-model",
        nuggets_base_url="https://cli/v1",
        nuggets_api_key="cli-key",
        nuggets_api_key_service="cli-svc",
        nuggets_api_key_account="cli-acct",
        nuggets_timeout=77,
        nuggets_temperature=0.1,
        nuggets_max_input_chars=42,
    )
    assert s.nuggets_backend == "openrouter"
    assert s.nuggets_model == "cli-model"
    assert s.nuggets_base_url == "https://cli/v1"
    assert s.nuggets_api_key == "cli-key"
    assert s.nuggets_api_key_service == "cli-svc"
    assert s.nuggets_api_key_account == "cli-acct"
    assert s.nuggets_timeout == 77
    assert s.nuggets_temperature == 0.1
    assert s.nuggets_max_input_chars == 42