from __future__ import annotations

from pathlib import Path

from podtx.config import load_settings


def test_config_defaults_diarize(tmp_path: Path) -> None:
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.diarize is False
    assert s.diarize_backend == "fake"
    assert s.diarize_model is None
    assert s.diarize_base_url is None
    assert s.diarize_api_key is None
    assert s.diarize_api_key_service is None
    assert s.diarize_api_key_account is None
    assert s.diarize_timeout == 120.0


def test_config_toml_diarize(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        """
diarize = true
diarize_backend = "pyannote"
diarize_model = "pyannote/speaker-diarization-3.1"
diarize_base_url = "https://example.com"
diarize_api_key = "hf-test"
diarize_api_key_service = "svc"
diarize_api_key_account = "acct"
diarize_timeout = 30
""",
        encoding="utf-8",
    )
    s = load_settings(config_path=p)
    assert s.diarize is True
    assert s.diarize_backend == "pyannote"
    assert s.diarize_model == "pyannote/speaker-diarization-3.1"
    assert s.diarize_base_url == "https://example.com"
    assert s.diarize_api_key == "hf-test"
    assert s.diarize_api_key_service == "svc"
    assert s.diarize_api_key_account == "acct"
    assert s.diarize_timeout == 30.0


def test_config_env_diarize(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE", "true")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_BACKEND", "hf")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_MODEL", "env-model")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_BASE_URL", "https://env.example")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_API_KEY", "env-key")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_API_KEY_SERVICE", "env-svc")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_API_KEY_ACCOUNT", "env-acct")
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_TIMEOUT", "9")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize is True
    assert s.diarize_backend == "hf"
    assert s.diarize_model == "env-model"
    assert s.diarize_base_url == "https://env.example"
    assert s.diarize_api_key == "env-key"
    assert s.diarize_api_key_service == "env-svc"
    assert s.diarize_api_key_account == "env-acct"
    assert s.diarize_timeout == 9.0


def test_config_env_diarize_provider_aliases(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_api_key == "hf-key"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hug-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_api_key == "hug-key"
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "asm-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_api_key == "asm-key"
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_api_key == "dg-key"
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("HF_BASE_URL", "https://hf.example")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_base_url == "https://hf.example"
    monkeypatch.delenv("HF_BASE_URL", raising=False)
    monkeypatch.setenv("ASSEMBLYAI_BASE_URL", "https://asm.example")
    s = load_settings(config_path=Path("/tmp/missing.toml"))
    assert s.diarize_base_url == "https://asm.example"


def test_config_cli_diarize(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DIARIZE_BACKEND", "fake")
    s = load_settings(
        diarize=True,
        diarize_backend="pyannote",
        diarize_model="cli-model",
        diarize_base_url="https://cli.example",
        diarize_api_key="cli-key",
        diarize_api_key_service="cli-svc",
        diarize_api_key_account="cli-acct",
        diarize_timeout=42,
    )
    assert s.diarize is True
    assert s.diarize_backend == "pyannote"
    assert s.diarize_model == "cli-model"
    assert s.diarize_base_url == "https://cli.example"
    assert s.diarize_api_key == "cli-key"
    assert s.diarize_api_key_service == "cli-svc"
    assert s.diarize_api_key_account == "cli-acct"
    assert s.diarize_timeout == 42.0