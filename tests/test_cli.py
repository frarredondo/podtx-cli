from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.config import load_settings

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.stdout


def test_cli_add_feeds_show_remove(tmp_path: Path, monkeypatch) -> None:
    # Point data dir at temp via env
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DATA_DIR", str(tmp_path))
    xml_url = FIXTURE.resolve().as_uri()

    # feedparser can parse file:// URIs
    result = runner.invoke(app, ["add", xml_url, "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Added" in result.stdout

    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "demo-podcast" in result.stdout

    result = runner.invoke(app, ["show", "demo-podcast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0

    result = runner.invoke(app, ["remove", "demo-podcast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_load_settings_precedence(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'engine = "whisper"\nlimit = 2\nlocal_attention = false\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PODCAST_TRANSCRIBER_LIMIT", "9")
    settings = load_settings(engine="parakeet", config_path=cfg)
    assert settings.engine == "parakeet"  # CLI wins
    assert settings.limit == 9  # env wins over toml
    assert settings.local_attention is False  # from toml
    assert settings.local_attention_context_size == 256  # default

    settings2 = load_settings(
        local_attention=True,
        local_attention_context_size=128,
        config_path=cfg,
    )
    assert settings2.local_attention is True  # CLI wins
    assert settings2.local_attention_context_size == 128
