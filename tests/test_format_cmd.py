from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.format_cmd import load_transcript_json, reformat_transcript
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()


def _sample_json(tmp_path: Path) -> Path:
    episode = Episode(
        guid="g1",
        title="Demo Episode",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=3,
        show_title="Demo Show",
        link="https://example.com/ep",
    )
    transcript = Transcript(
        text="So uh yeah. The the fox.",
        segments=[
            Segment(0.0, 1.0, "So uh yeah."),
            Segment(2.0, 3.0, "The the fox."),
        ],
        language="en",
        model="test-model",
        engine="fake",
    )
    write_outputs(
        out_dir=tmp_path,
        basename="ep",
        episode=episode,
        transcript=transcript,
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )
    return tmp_path / "ep.json"


def test_load_transcript_json(tmp_path: Path) -> None:
    path = _sample_json(tmp_path)
    episode, transcript = load_transcript_json(path)
    assert episode.title == "Demo Episode"
    assert episode.guid == "g1"
    assert transcript.engine == "fake"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "So uh yeah."


def test_reformat_applies_readable_and_cleanup(tmp_path: Path) -> None:
    path = _sample_json(tmp_path)
    out = tmp_path / "out"
    paths = reformat_transcript(
        path,
        out_dir=out,
        readable=True,
        cleanup=True,
        formats=("txt", "json"),
    )
    assert len(paths) == 2
    txt = (out / "ep.txt").read_text(encoding="utf-8")
    assert "So yeah." in txt
    assert "The fox." in txt
    assert "\n\n" in txt.split("\n\n", 1)[1]  # body has paragraph break
    payload = json.loads((out / "ep.json").read_text(encoding="utf-8"))
    assert payload["readable"] is True
    assert payload["cleanup"] is True
    assert payload["segments"][0]["text"] == "So uh yeah."  # raw segments


def test_cli_format_command(tmp_path: Path) -> None:
    path = _sample_json(tmp_path)
    out = tmp_path / "formatted"
    result = runner.invoke(
        app,
        ["format", str(path), "--readable", "--cleanup", "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (out / "ep.txt").is_file()
    body = (out / "ep.txt").read_text(encoding="utf-8")
    assert " uh " not in body
    assert "the the" not in body.lower()


def _library(tmp_path: Path) -> Path:
    """Build a mini transcripts library: feed-a/*.json, feed-b/*.json."""
    root = tmp_path / "transcripts"
    root.mkdir()
    for slug in ("feed-a", "feed-b"):
        feed_dir = root / slug
        feed_dir.mkdir()
        episode = Episode(
            guid=f"g-{slug}",
            title=f"Ep {slug}",
            enclosure_url="https://example.com/a.mp3",
            published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
            episode_num=1,
            show_title=slug,
        )
        transcript = Transcript(
            text="So uh yeah. The the fox.",
            segments=[
                Segment(0.0, 1.0, "So uh yeah."),
                Segment(2.0, 3.0, "The the fox."),
            ],
            language="en",
            model="test-model",
            engine="fake",
        )
        write_outputs(
            out_dir=feed_dir,
            basename=f"{slug}-ep",
            episode=episode,
            transcript=transcript,
            formats=("txt", "json"),
            readable=False,
            cleanup=False,
        )
    return root


def test_discover_transcript_jsons_for_feed(tmp_path: Path) -> None:
    from podtx.format_cmd import discover_transcript_jsons

    root = _library(tmp_path)
    found = discover_transcript_jsons(root, feed="feed-a")
    assert [p.name for p in found] == ["feed-a-ep.json"]


def test_discover_transcript_jsons_all_feeds(tmp_path: Path) -> None:
    from podtx.format_cmd import discover_transcript_jsons

    root = _library(tmp_path)
    found = discover_transcript_jsons(root, feed=None)
    assert [p.name for p in found] == ["feed-a-ep.json", "feed-b-ep.json"]


def test_discover_unknown_feed_raises(tmp_path: Path) -> None:
    from podtx.format_cmd import TranscriptJsonError, discover_transcript_jsons

    root = _library(tmp_path)
    try:
        discover_transcript_jsons(root, feed="missing")
        assert False, "expected TranscriptJsonError"
    except TranscriptJsonError as exc:
        assert "missing" in str(exc)


def test_reformat_many_reports_success_and_errors(tmp_path: Path) -> None:
    from podtx.format_cmd import reformat_many

    root = _library(tmp_path)
    paths = sorted(root.rglob("*.json"))
    bad = tmp_path / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    result = reformat_many(
        [*paths, bad],
        readable=True,
        cleanup=True,
        formats=("txt", "json"),
    )
    assert result.ok == 2
    assert result.failed == 1
    assert len(result.errors) == 1
    assert "broken.json" in result.errors[0][0].name
    # Successful rewrite applied cleanup in place
    body = (root / "feed-a" / "feed-a-ep.txt").read_text(encoding="utf-8")
    assert " uh " not in body


def test_cli_format_feed(tmp_path: Path) -> None:
    root = _library(tmp_path)
    data_dir = tmp_path
    # transcripts live at data_dir/transcripts
    assert (data_dir / "transcripts").is_dir()
    result = runner.invoke(
        app,
        [
            "format",
            "--feed",
            "feed-a",
            "--data-dir",
            str(data_dir),
            "--cleanup",
            "--readable",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "feed-a-ep" in result.stdout
    body = (root / "feed-a" / "feed-a-ep.txt").read_text(encoding="utf-8")
    assert "the the" not in body.lower()


def test_cli_format_all(tmp_path: Path) -> None:
    root = _library(tmp_path)
    result = runner.invoke(
        app,
        ["format", "--all", "--data-dir", str(tmp_path), "--cleanup"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (root / "feed-a" / "feed-a-ep.txt").is_file()
    assert (root / "feed-b" / "feed-b-ep.txt").is_file()
    assert "2" in result.stdout or "Wrote" in result.stdout


def test_cli_format_requires_target(tmp_path: Path) -> None:
    result = runner.invoke(app, ["format", "--cleanup"])
    assert result.exit_code != 0
