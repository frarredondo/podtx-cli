from __future__ import annotations

from pathlib import Path

import pytest

from podtx.config import Settings, _parse_formats, load_settings


def test_resolved_model_all_paths() -> None:
    s = Settings()
    assert s.resolved_model() == "mlx-community/parakeet-tdt-0.6b-v3"
    s2 = Settings(engine="whisper")
    assert s2.resolved_model() == "mlx-community/whisper-large-v3-turbo"
    s3 = Settings(model="explicit-model")
    assert s3.resolved_model() == "explicit-model"


def test_parse_formats_all_paths() -> None:
    assert _parse_formats("txt, json") == ("txt", "json")
    assert _parse_formats("   ") == ("txt", "json")
    assert _parse_formats([]) == ()
    assert _parse_formats(["TXT", "JSON"]) == ("txt", "json")


def _write_toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_toml_overrides_base(tmp_path: Path) -> None:
    p = _write_toml(
        tmp_path,
        """
        engine = "whisper"
        model = "toml-model"
        limit = 3
        formats = "txt,md"
        keep_audio = true
        data_dir = "~/toml-data"
        quiet = true
        language = "es"
        local_attention = false
        local_attention_context_size = 128
        readable = true
        cleanup = true
        """,
    )
    s = load_settings(config_path=p)
    assert s.engine == "whisper"
    assert s.model == "toml-model"
    assert s.limit == 3
    assert s.formats == ("txt", "md")
    assert s.keep_audio is True
    assert s.data_dir == Path("~/toml-data").expanduser()
    assert s.quiet is True
    assert s.language == "es"
    assert s.local_attention is False
    assert s.local_attention_context_size == 128
    assert s.readable is True
    assert s.cleanup is True


def test_env_overrides(monkeypatch, tmp_path: Path) -> None:
    import os

    env = {
        "PODCAST_TRANSCRIBER_ENGINE": "whisper",
        "PODCAST_TRANSCRIBER_MODEL": "env-model",
        "PODCAST_TRANSCRIBER_LIMIT": "5",
        "PODCAST_TRANSCRIBER_FORMATS": "json,md",
        "PODCAST_TRANSCRIBER_KEEP_AUDIO": "true",
        "PODCAST_TRANSCRIBER_DATA_DIR": "~/env-data",
        "PODCAST_TRANSCRIBER_QUIET": "yes",
        "PODCAST_TRANSCRIBER_LANGUAGE": "fr",
        "PODCAST_TRANSCRIBER_LOCAL_ATTENTION": "on",
        "PODCAST_TRANSCRIBER_LOCAL_ATTENTION_CONTEXT_SIZE": "512",
        "PODCAST_TRANSCRIBER_READABLE": "1",
        "PODCAST_TRANSCRIBER_CLEANUP": "true",
        "MODEL_API_KEY": "secret-model-key",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = load_settings(config_path=tmp_path / "missing.toml")
    assert s.engine == "whisper"
    assert s.model == "env-model"
    assert s.limit == 5
    assert s.formats == ("json", "md")
    assert s.keep_audio is True
    assert s.data_dir == Path("~/env-data").expanduser()
    assert s.quiet is True
    assert s.language == "fr"
    assert s.local_attention is True
    assert s.local_attention_context_size == 512
    assert s.readable is True
    assert s.cleanup is True
    assert s.summarize_api_key == "secret-model-key"


def test_cli_flags_override(tmp_path: Path) -> None:
    s = load_settings(
        engine="openai",
        model="cli-model",
        limit=9,
        formats=["txt"],
        keep_audio=False,
        data_dir=str(tmp_path / "cli-data"),
        quiet=False,
        language="de",
        readable=False,
        cleanup=False,
        config_path=tmp_path / "missing.toml",
    )
    assert s.engine == "openai"
    assert s.model == "cli-model"
    assert s.limit == 9
    assert s.formats == ("txt",)
    assert s.keep_audio is False
    assert s.data_dir == tmp_path / "cli-data"
    assert s.quiet is False
    assert s.language == "de"
    assert s.readable is False
    assert s.cleanup is False


def test_cli_booleans_true(tmp_path: Path) -> None:
    s = load_settings(
        keep_audio=True,
        quiet=True,
        readable=True,
        cleanup=True,
        config_path=tmp_path / "missing.toml",
    )
    assert s.keep_audio is True
    assert s.quiet is True
    assert s.readable is True
    assert s.cleanup is True


def test_trim_start_cli_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_settings(trim_start=-1.0, config_path=tmp_path / "missing.toml")
    s = load_settings(trim_start=2.5, config_path=tmp_path / "missing.toml")
    assert s.trim_start == 2.5
