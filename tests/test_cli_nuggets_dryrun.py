from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.models import Episode, Segment, Transcript
from podtx.providers.catalog import CatalogError
from podtx.writers import write_outputs

runner = CliRunner()

FAKE_API = {
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
    "lmstudio": {
        "id": "lmstudio",
        "name": "LM Studio",
        "models": {
            "openai/gpt-oss-20b": {
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
        },
    },
}


def _episode() -> Episode:
    return Episode(
        guid="fake-guid-1",
        title="Fake Episode Title",
        enclosure_url="https://example.com/ep.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=42,
        show_title="Fake Show",
        link="https://example.com/ep",
    )


def _transcript() -> Transcript:
    return Transcript(
        text="First sentence is overview. Second sentence also overview. Third is a key point. Fourth more. Fifth extra.",
        segments=[
            Segment(0.0, 1.5, "First sentence is overview."),
            Segment(2.0, 3.5, "Second sentence also overview."),
            Segment(10.0, 12.0, "Third is a key point."),
            Segment(65.0, 70.0, "Fourth more."),
            Segment(120.0, 125.0, "Fifth extra."),
        ],
        language="en",
        model="fake-model",
        engine="fake",
    )


def _write_transcript(tmp_path: Path, basename: str = "ep") -> Path:
    write_outputs(
        out_dir=tmp_path,
        basename=basename,
        episode=_episode(),
        transcript=_transcript(),
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )
    return tmp_path / f"{basename}.json"


def _load_fake(_tmp_path: Path):
    def fake_load(data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400):
        return FAKE_API

    return fake_load


def test_dry_run_single_no_sidecar_no_call(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        [
            "nuggets",
            str(path),
            "--backend",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-4",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "dry run" in result.stdout.lower()
    assert "Claude Sonnet 4" in result.stdout
    assert not (tmp_path / "ep.nuggets.json").exists()
    assert not (tmp_path / "ep.nuggets.md").exists()


def test_dry_run_single_cost_estimate(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", str(path), "--backend", "openrouter", "--model", "anthropic/claude-sonnet-4", "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "tokens" in result.stdout
    assert "cost" in result.stdout


def test_dry_run_fake_backend(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["nuggets", str(path), "--backend", "fake", "--dry-run"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "no inference" in result.stdout.lower()


def test_dry_run_catalog_unavailable(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)

    def boom(data_dir, *, refresh=False, timeout=120.0, ttl_seconds=86400):
        raise CatalogError("no cache and offline")

    monkeypatch.setattr("podtx.cli.load_catalog", boom)
    result = runner.invoke(
        app, ["nuggets", str(path), "--backend", "openrouter", "--model", "anthropic/claude-sonnet-4", "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "catalog unavailable" in result.stdout.lower()
    assert "cost: unknown" in result.stdout.lower() or "cost: unknown" in result.stderr.lower()


def test_dry_run_unknown_model(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", str(path), "--backend", "openrouter", "--model", "nope/x", "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "not in catalog" in result.stdout.lower()


def test_dry_run_single_bad_json(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{{{{", encoding="utf-8")
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["nuggets", str(bad), "--backend", "fake", "--dry-run"])
    assert result.exit_code != 0
    assert "Could not read transcript JSON" in result.stdout + result.stderr


def test_dry_run_feed_sums(tmp_path, monkeypatch) -> None:
    root = tmp_path / "transcripts" / "myshow"
    root.mkdir(parents=True, exist_ok=True)
    _write_transcript(root, basename="ep0")
    _write_transcript(root, basename="ep1")
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        [
            "nuggets",
            "--feed",
            "myshow",
            "--data-dir",
            str(tmp_path),
            "--backend",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-4",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "TOTAL" in result.stdout
    assert "2 episodes" in result.stdout


def test_dry_run_feed_no_transcripts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", "--feed", "nope", "--data-dir", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code != 0

def test_dry_run_no_pricing_in_catalog(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", str(path), "--backend", "lmstudio", "--model", "qwen/qwen2.5-14b", "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "no pricing in catalog" in result.stdout


def test_dry_run_single_chunked(tmp_path, monkeypatch) -> None:
    path = _write_transcript(tmp_path)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        [
            "nuggets",
            str(path),
            "--backend",
            "openrouter",
            "--model",
            "anthropic/claude-sonnet-4",
            "--max-input-chars",
            "5",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "chunks" in result.stdout


def test_dry_run_feed_skips_broken(tmp_path, monkeypatch) -> None:
    root = tmp_path / "transcripts" / "myshow"
    root.mkdir(parents=True, exist_ok=True)
    _write_transcript(root, basename="ep0")
    _write_transcript(root, basename="ep1")
    (root / "ep2.json").write_text("{{{{", encoding="utf-8")
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", "--feed", "myshow", "--data-dir", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code != 0
    assert "Skipping" in result.stdout + result.stderr


def test_dry_run_feed_quiet(tmp_path, monkeypatch) -> None:
    root = tmp_path / "transcripts" / "myshow"
    root.mkdir(parents=True, exist_ok=True)
    _write_transcript(root, basename="ep0")
    _write_transcript(root, basename="ep1")
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app,
        ["nuggets", "--feed", "myshow", "--data-dir", str(tmp_path), "--backend", "fake", "--dry-run", "--quiet"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "TOTAL" not in result.stdout


def test_dry_run_single_file_not_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    missing = tmp_path / "missing.json"
    result = runner.invoke(app, ["nuggets", str(missing), "--dry-run"])
    assert result.exit_code != 0
    assert "File not found" in result.stdout + result.stderr


def test_dry_run_feed_empty_transcripts(tmp_path, monkeypatch) -> None:
    (tmp_path / "transcripts" / "myshow").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(app, ["nuggets", "--feed", "myshow", "--data-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code != 0
    assert "No transcript JSON files found" in result.stdout + result.stderr


def test_dry_run_feed_limit(tmp_path, monkeypatch) -> None:
    root = tmp_path / "transcripts" / "myshow"
    root.mkdir(parents=True, exist_ok=True)
    _write_transcript(root, basename="ep0")
    _write_transcript(root, basename="ep1")
    monkeypatch.setattr("podtx.cli.load_catalog", _load_fake(tmp_path))
    result = runner.invoke(
        app, ["nuggets", "--feed", "myshow", "--data-dir", str(tmp_path), "--limit", "1", "--dry-run"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "TOTAL" in result.stdout
    assert "1 episodes" in result.stdout
