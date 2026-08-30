from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

from podtx.cli import app
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kwargs):
        payload = json.dumps(
            {
                "nuggets": [
                    {
                        "insight": "Trust matters most in an incident.",
                        "context": "Fake Show — ep",
                        "why_it_matters": "so engineers prioritize trust",
                        "quote": "Second sentence also overview.",
                        "scores": {"T": 2, "S": 2, "E": 1, "A": 1},
                        "tag": "eng",
                    }
                ]
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": payload}}]})


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


def _build_library(tmp_path: Path, feed: str = "myshow", count: int = 2) -> Path:
    root = tmp_path / "transcripts" / feed
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        _write_transcript(root, basename=f"ep{i}")
    return tmp_path


def test_nuggets_single_fake(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Wrote" in result.stdout
    sidecar = tmp_path / "ep.nuggets.json"
    assert sidecar.is_file()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["backend"] == "fake"
    assert data["nuggets"]


def test_nuggets_single_quiet(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "-q"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.stdout == ""
    assert (tmp_path / "ep.nuggets.json").is_file()


def test_nuggets_single_provider_error(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "--backend", "openai", "--api-key", "k"])
    assert result.exit_code != 0
    assert "requires --model" in result.stdout or "requires --model" in result.stderr


def test_nuggets_single_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{{{{", encoding="utf-8")
    result = runner.invoke(app, ["nuggets", str(bad)])
    assert result.exit_code != 0
    assert "Could not read transcript JSON" in result.stdout or "Could not read transcript JSON" in result.stderr


def test_nuggets_rerun_skips_quiet(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    runner.invoke(app, ["nuggets", str(path)])
    result = runner.invoke(app, ["nuggets", str(path), "-q"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_nuggets_feed_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["nuggets", "--feed", "nope", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "Feed transcript folder not found" in combined


def test_nuggets_rerun_skips(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    runner.invoke(app, ["nuggets", str(path)])
    result = runner.invoke(app, ["nuggets", str(path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Skipping" in result.stdout


def test_nuggets_format_md_only(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "--format", "md"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "ep.nuggets.md").is_file()
    assert not (tmp_path / "ep.nuggets.json").exists()


def test_nuggets_both_formats(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "-f", "json", "-f", "md"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "ep.nuggets.json").is_file()
    assert (tmp_path / "ep.nuggets.md").is_file()


def test_nuggets_unsupported_format(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "--format", "vcf"])
    assert result.exit_code != 0
    assert "Unsupported format" in result.stdout or "Unsupported format" in result.stderr


def test_nuggets_no_target() -> None:
    result = runner.invoke(app, ["nuggets"])
    assert result.exit_code != 0
    assert "Specify exactly one" in result.stdout or "Specify exactly one" in result.stderr


def test_nuggets_file_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["nuggets", str(tmp_path / "missing.json")])
    assert result.exit_code != 0
    assert "File not found" in result.stdout or "File not found" in result.stderr


def test_nuggets_unknown_backend(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = runner.invoke(app, ["nuggets", str(path), "--backend", "bogus"])
    assert result.exit_code != 0
    assert "Unknown backend" in result.stdout or "Unknown backend" in result.stderr


def test_nuggets_provider_alias(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    import podtx.providers.openai as mod

    with patch.object(mod.httpx, "Client", _DummyClient):
        result = runner.invoke(
            app, ["nuggets", str(path), "--backend", "fake", "--provider", "openrouter", "--api-key", "k"]
        )
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads((tmp_path / "ep.nuggets.json").read_text(encoding="utf-8"))
    assert data["backend"] == "openrouter"
    assert data["nuggets"][0]["timestamp"] == "00:02"


def test_nuggets_mocked_llm(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    import podtx.providers.openai as mod

    with patch.object(mod.httpx, "Client", _DummyClient):
        result = runner.invoke(app, ["nuggets", str(path), "--backend", "openrouter", "--api-key", "k"])
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads((tmp_path / "ep.nuggets.json").read_text(encoding="utf-8"))
    assert data["backend"] == "openrouter"
    assert data["model"] == "meta/muse-spark-1.2-contributor"
    assert data["nuggets"] or data["nuggets"] == []


def test_nuggets_feed(tmp_path: Path) -> None:
    lib = _build_library(tmp_path, count=2)
    result = runner.invoke(
        app, ["nuggets", "--feed", "myshow", "--data-dir", str(lib), "-q"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (lib / "transcripts" / "myshow" / "ep0.nuggets.json").is_file()
    assert (lib / "transcripts" / "myshow" / "ep1.nuggets.json").is_file()

    result2 = runner.invoke(app, ["nuggets", "--feed", "myshow", "--data-dir", str(lib)])
    assert result2.exit_code == 0, result2.stdout + result2.stderr
    assert "0 ok, 2 skipped, 0 failed" in result2.stdout


def test_nuggets_feed_empty(tmp_path: Path) -> None:
    lib = _build_library(tmp_path, count=0)
    result = runner.invoke(app, ["nuggets", "--feed", "myshow", "--data-dir", str(lib)])
    assert result.exit_code != 0
    assert "No transcript JSON files found" in result.stdout or "No transcript JSON files found" in result.stderr


def test_nuggets_all_with_limit(tmp_path: Path) -> None:
    lib = _build_library(tmp_path, feed="a", count=2)
    _build_library(tmp_path, feed="b", count=1)
    result = runner.invoke(app, ["nuggets", "--all", "--data-dir", str(lib), "--limit", "2", "-q"])
    assert result.exit_code == 0, result.stdout + result.stderr
    sidecars = list(lib.rglob("*.nuggets.json"))
    assert len(sidecars) == 2


def test_nuggets_continues_on_bad_json(tmp_path: Path) -> None:
    root = tmp_path / "transcripts" / "feed"
    root.mkdir(parents=True, exist_ok=True)
    _write_transcript(root, basename="good")
    (root / "bad.json").write_text("{{{{", encoding="utf-8")
    result = runner.invoke(app, ["nuggets", "--feed", "feed", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "1 ok," in combined
    assert "1 failed" in combined
    assert "Failed" in combined


def test_auth_set_anthropic() -> None:
    with patch("podtx.keychain.save_api_key") as mock:
        result = runner.invoke(app, ["auth", "set", "anthropic", "--api-key", "sk-ant"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "Saved" in result.stdout
        assert mock.call_args[0][0] == "podtx-anthropic"


def test_auth_set_openai() -> None:
    with patch("podtx.keychain.save_api_key") as mock:
        result = runner.invoke(app, ["auth", "set", "openai", "--api-key", "sk-openai"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert mock.call_args[0][0] == "podtx-openai"


def test_auth_get_anthropic() -> None:
    with patch("podtx.keychain.get_api_key", return_value="secret"):
        result = runner.invoke(app, ["auth", "get", "anthropic"])
        assert result.exit_code == 0
        assert "Found" in result.stdout


def test_auth_delete_anthropic() -> None:
    with patch("podtx.keychain.delete_api_key", return_value=True):
        result = runner.invoke(app, ["auth", "delete", "anthropic"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout


def test_auth_unknown_backend_rejected() -> None:
    result = runner.invoke(app, ["auth", "set", "bogus", "--api-key", "k"])
    assert result.exit_code != 0
    assert "Unknown backend" in result.stdout or "Unknown backend" in result.stderr