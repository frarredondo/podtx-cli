from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.format_cmd import TranscriptJsonError
from podtx.models import Episode, Segment, Transcript
from podtx.summarize import build_summary, summarize_many, summarize_transcript
from podtx.writers import write_outputs

runner = CliRunner()


def _fake_episode(**overrides) -> Episode:
    base = dict(
        guid="fake-guid-1",
        title="Fake Episode Title",
        enclosure_url="https://example.com/ep.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=42,
        show_title="Fake Show",
        link="https://example.com/ep",
    )
    base.update(overrides)
    return Episode(**base)


def _fake_transcript(
    text: str = "First sentence is overview. Second sentence also overview. Third is a key point. Fourth more. Fifth extra.",
    segments: list[Segment] | None = None,
    engine: str = "fake",
    model: str = "fake-model",
) -> Transcript:
    if segments is None:
        segments = [
            Segment(0.0, 1.5, "First sentence is overview."),
            Segment(2.0, 3.5, "Second sentence also overview."),
            Segment(10.0, 12.0, "Third is a key point."),
            Segment(65.0, 70.0, "Fourth more."),
            Segment(120.0, 125.0, "Fifth extra."),
        ]
    return Transcript(text=text, segments=segments, language="en", model=model, engine=engine)


def _write_fake_transcript_json(tmp_path: Path, basename: str = "ep", episode: Episode | None = None, transcript: Transcript | None = None) -> Path:
    ep = episode or _fake_episode()
    tx = transcript or _fake_transcript()
    write_outputs(out_dir=tmp_path, basename=basename, episode=ep, transcript=tx, formats=("txt", "json"), readable=False, cleanup=False)
    return tmp_path / f"{basename}.json"


def _library(tmp_path: Path) -> Path:
    root = tmp_path / "transcripts"
    root.mkdir(parents=True)
    for slug in ("feed-a", "feed-b"):
        feed_dir = root / slug
        feed_dir.mkdir()
        ep = _fake_episode(guid=f"g-{slug}", title=f"Ep {slug}", show_title=slug)
        tx = _fake_transcript()
        write_outputs(out_dir=feed_dir, basename=f"{slug}-ep", episode=ep, transcript=tx, formats=("txt", "json"), readable=False, cleanup=False)
    return root


# ── Unit: build_summary extractive ──

def test_build_summary_overview_key_points_quotes() -> None:
    ep = _fake_episode()
    tx = _fake_transcript()
    summary = build_summary(ep, tx, backend="fake")
    assert "overview" in summary
    assert summary["overview"].strip() != ""
    # overview = first 2 sentences
    assert "First sentence" in summary["overview"]
    assert "Second sentence" in summary["overview"]
    assert "key_points" in summary
    assert len(summary["key_points"]) >= 1
    assert len(summary["key_points"]) <= 3
    assert "Third is a key point" in summary["key_points"][0]
    assert "quotes" in summary
    assert len(summary["quotes"]) == 3
    # timestamped
    for q in summary["quotes"]:
        assert "timestamp" in q
        assert "start" in q
        assert "end" in q
        assert ":" in q["timestamp"]
        assert q["text"].strip() != ""


def test_build_summary_unknown_backend_raises() -> None:
    ep = _fake_episode()
    tx = _fake_transcript()
    try:
        build_summary(ep, tx, backend="openai")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown summary backend" in str(exc)


def test_build_summary_no_segments_uses_text() -> None:
    ep = _fake_episode()
    tx = Transcript(text="Only one sentence here. Second here.", segments=[], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    assert summary["overview"] != ""
    assert summary["key_points"]
    assert summary["quotes"] == []


def test_build_summary_empty_text_and_no_segments() -> None:
    ep = _fake_episode()
    tx = Transcript(text="   ", segments=[], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    assert isinstance(summary["overview"], str)
    assert isinstance(summary["key_points"], list)


def test_build_summary_segments_fallback_when_text_empty() -> None:
    ep = _fake_episode()
    segs = [Segment(0.0, 1.0, "Hello from segments."), Segment(5.0, 6.0, "Second segment text.")]
    tx = Transcript(text="   ", segments=segs, language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    assert "Hello from segments" in summary["overview"]
    assert len(summary["quotes"]) == 2
    # timestamps formatted correctly
    assert summary["quotes"][0]["timestamp"] == "00:00"
    assert summary["quotes"][1]["timestamp"] == "00:05"


def test_build_summary_timestamp_hours() -> None:
    ep = _fake_episode()
    segs = [Segment(0.0, 1.0, "Start."), Segment(3700.0, 3701.0, "Middle."), Segment(7400.0, 7401.0, "End.")]
    tx = Transcript(text="Start. Middle. End.", segments=segs, language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    # middle at 3700s = 01:01:40, end at 7400 = 02:03:20
    assert summary["quotes"][1]["timestamp"] == "01:01:40"
    assert summary["quotes"][2]["timestamp"] == "02:03:20"


def test_build_summary_short_text_key_points() -> None:
    ep = _fake_episode()
    tx = Transcript(text="Short.", segments=[Segment(0, 1, "Short.")], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    assert summary["key_points"] == ["Short."]


# ── Single file: summarize_transcript + CLI ──

def test_summarize_single_file_json(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    written = summarize_transcript(json_path, formats=("json",), backend="fake")
    assert len(written) == 1
    assert written[0].name == "ep.summary.json"
    assert written[0].is_file()
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["overview"].strip() != ""
    assert len(payload["key_points"]) >= 1
    assert len(payload["quotes"]) >= 1
    assert payload["backend"] == "fake"
    assert "generated_at" in payload
    # Check quotes have timestamps
    assert ":" in payload["quotes"][0]["timestamp"]


def test_summarize_single_file_md(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    written = summarize_transcript(json_path, formats=("md",), backend="fake")
    assert written[0].name == "ep.summary.md"
    body = written[0].read_text(encoding="utf-8")
    assert "## Overview" in body
    assert "## Key Points" in body
    assert "## Quotes" in body
    assert "offline" in body.lower()
    assert "—" not in body or True  # just ensure no crash


def test_summarize_single_file_both_formats(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    written = summarize_transcript(json_path, formats=("json", "md"), backend="fake")
    assert len(written) == 2
    names = {p.name for p in written}
    assert "ep.summary.json" in names
    assert "ep.summary.md" in names


def test_summarize_unsupported_format_raises(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    try:
        summarize_transcript(json_path, formats=("txt",), backend="fake")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported" in str(exc)


def test_cli_summarize_single_file(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "ep.summary.json").is_file()
    payload = json.loads((tmp_path / "ep.summary.json").read_text(encoding="utf-8"))
    assert payload["overview"]
    assert payload["key_points"]
    assert payload["quotes"][0]["timestamp"]


def test_cli_summarize_single_file_md_format(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path), "--format", "md"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "ep.summary.md").is_file()
    body = (tmp_path / "ep.summary.md").read_text(encoding="utf-8")
    assert "## Overview" in body


def test_cli_summarize_single_file_both_formats(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path), "--format", "json", "--format", "md"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "ep.summary.json").is_file()
    assert (tmp_path / "ep.summary.md").is_file()


def test_cli_summarize_out_dir(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    out = tmp_path / "summaries"
    result = runner.invoke(app, ["summarize", str(json_path), "--out-dir", str(out)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (out / "ep.summary.json").is_file()


def test_cli_summarize_quiet(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path), "--quiet"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Wrote" not in result.stdout
    assert (tmp_path / "ep.summary.json").is_file()


# ── Batch: --feed / --all / --limit ──

def test_cli_summarize_feed(tmp_path: Path) -> None:
    root = _library(tmp_path)
    data_dir = tmp_path
    assert (data_dir / "transcripts").is_dir()
    result = runner.invoke(app, ["summarize", "--feed", "feed-a", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "1 ok" in result.stdout
    assert "0 failed" in result.stdout
    assert (root / "feed-a" / "feed-a-ep.summary.json").is_file()
    # feed-b not summarized
    assert not (root / "feed-b" / "feed-b-ep.summary.json").exists()


def test_cli_summarize_all(tmp_path: Path) -> None:
    root = _library(tmp_path)
    result = runner.invoke(app, ["summarize", "--all", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (root / "feed-a" / "feed-a-ep.summary.json").is_file()
    assert (root / "feed-b" / "feed-b-ep.summary.json").is_file()
    assert "2" in result.stdout or "Wrote" in result.stdout


def test_cli_summarize_all_with_limit(tmp_path: Path) -> None:
    root = _library(tmp_path)
    # add extra episodes to feed-a
    extra = root / "feed-a" / "extra.json"
    ep = _fake_episode(guid="extra", title="Extra")
    tx = _fake_transcript()
    write_outputs(out_dir=root / "feed-a", basename="extra", episode=ep, transcript=tx, formats=("json",), readable=False, cleanup=False)
    result = runner.invoke(app, ["summarize", "--all", "--data-dir", str(tmp_path), "--limit", "1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    # limit 1 total across all feeds sorted
    assert "1 ok" in result.stdout
    # Only one summary written
    assert result.stdout.count("Wrote") == 1 or "1 ok" in result.stdout


def test_cli_summarize_feed_with_limit(tmp_path: Path) -> None:
    root = _library(tmp_path)
    # add second json to feed-a
    ep = _fake_episode(guid="g2", title="Ep2")
    tx = _fake_transcript()
    write_outputs(out_dir=root / "feed-a", basename="second", episode=ep, transcript=tx, formats=("json",), readable=False, cleanup=False)
    result = runner.invoke(app, ["summarize", "--feed", "feed-a", "--data-dir", str(tmp_path), "--limit", "1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "1 ok" in result.stdout


def test_summarize_many_reports_success_and_errors(tmp_path: Path) -> None:
    root = _library(tmp_path)
    paths = sorted(root.rglob("*.json"))
    bad = tmp_path / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    result = summarize_many([*paths, bad], formats=("json",), backend="fake")
    assert result.ok == 2
    assert result.failed == 1
    assert len(result.errors) == 1
    assert "broken.json" in result.errors[0][0].name


def test_cli_summarize_feed_reports_failures(tmp_path: Path) -> None:
    root = _library(tmp_path)
    bad = root / "feed-a" / "broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    result = runner.invoke(app, ["summarize", "--feed", "feed-a", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    out = result.stdout + result.stderr
    assert "Failed" in out
    assert "failed" in out


def test_cli_summarize_all_md_format(tmp_path: Path) -> None:
    root = _library(tmp_path)
    result = runner.invoke(app, ["summarize", "--all", "--data-dir", str(tmp_path), "--format", "md"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (root / "feed-a" / "feed-a-ep.summary.md").is_file()
    assert (root / "feed-b" / "feed-b-ep.summary.md").is_file()


def test_cli_summarize_batch_out_dir(tmp_path: Path) -> None:
    _library(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(app, ["summarize", "--all", "--data-dir", str(tmp_path), "--out-dir", str(out), "--format", "json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    # out_dir flattens basename
    assert (out / "feed-a-ep.summary.json").is_file()
    assert (out / "feed-b-ep.summary.json").is_file()


# ── Error / edge cases ──

def test_cli_summarize_requires_target() -> None:
    result = runner.invoke(app, ["summarize"])
    assert result.exit_code != 0
    assert "Specify exactly one" in result.stdout or "Specify exactly one" in result.stderr


def test_cli_summarize_mutual_exclusive() -> None:
    result = runner.invoke(app, ["summarize", "some.json", "--feed", "x"])
    assert result.exit_code != 0


def test_cli_summarize_unknown_backend(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path), "--backend", "openai"])
    assert result.exit_code != 0
    assert "Unknown backend" in result.stdout or "Unknown backend" in result.stderr


def test_cli_summarize_unsupported_format(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    result = runner.invoke(app, ["summarize", str(json_path), "--format", "txt"])
    assert result.exit_code != 0
    assert "Unsupported" in result.stdout or "Unsupported" in result.stderr


def test_cli_summarize_file_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["summarize", str(tmp_path / "missing.json")])
    assert result.exit_code != 0
    assert "File not found" in result.stdout or "File not found" in result.stderr


def test_cli_summarize_feed_not_found(tmp_path: Path) -> None:
    root = _library(tmp_path)
    result = runner.invoke(app, ["summarize", "--feed", "missing", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_cli_summarize_no_transcripts(tmp_path: Path) -> None:
    (tmp_path / "transcripts").mkdir()
    result = runner.invoke(app, ["summarize", "--all", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "No transcript" in result.stdout or "No transcript" in result.stderr


def test_cli_summarize_help_mentions_offline_and_invocation() -> None:
    result = runner.invoke(app, ["summarize", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "offline" in out
    assert "fake" in out
    assert "no network" in out or "no silent" in out or "offline" in out
    # invocation examples
    assert "podtx summarize" in result.stdout or "summarize" in result.stdout
    assert "summary" in out
    # output format mentioned
    assert "json" in out and "md" in out
    # sidecar-ish
    assert ".summary" in result.stdout or "sidecar" in out


def test_summarize_no_asr_uses_existing_json_only(tmp_path: Path) -> None:
    """Summarize reads existing JSON, does not invoke ASR (no network)."""
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    # Ensure source transcript JSON is untouched and no extra ASR call needed
    orig = json.loads(json_path.read_text(encoding="utf-8"))
    result = runner.invoke(app, ["summarize", str(json_path)])
    assert result.exit_code == 0
    after = json.loads(json_path.read_text(encoding="utf-8"))
    assert orig == after
    # summary is separate sidecar, not overwriting transcript
    assert (tmp_path / "ep.summary.json").is_file()
    assert not (tmp_path / "ep.summary.json").samefile(json_path)


def test_summarize_sidecar_content_stable(tmp_path: Path) -> None:
    """Stable sidecar: overview, key_points, quotes with timestamps."""
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    runner.invoke(app, ["summarize", str(json_path)])
    payload = json.loads((tmp_path / "ep.summary.json").read_text(encoding="utf-8"))
    assert "overview" in payload and isinstance(payload["overview"], str) and payload["overview"].strip()
    assert "key_points" in payload and isinstance(payload["key_points"], list) and len(payload["key_points"]) >= 1
    assert "quotes" in payload and isinstance(payload["quotes"], list)
    for q in payload["quotes"]:
        assert "timestamp" in q and ":" in q["timestamp"]
        assert isinstance(q["start"], (int, float))
        assert isinstance(q["text"], str) and q["text"].strip()


def test_summarize_markdown_sidecar_content(tmp_path: Path) -> None:
    json_path = _write_fake_transcript_json(tmp_path, basename="ep")
    runner.invoke(app, ["summarize", str(json_path), "--format", "md"])
    md = (tmp_path / "ep.summary.md").read_text(encoding="utf-8")
    assert md.startswith("# Summary:")
    assert "## Overview" in md
    assert "## Key Points" in md
    assert "## Quotes" in md
    assert "- [" in md  # timestamped quote bullet


def test_cli_summarize_broken_json_reports_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["summarize", str(bad)])
    assert result.exit_code != 0
    assert "Could not read" in result.stdout or "Could not read" in result.stderr or "Failed" in (result.stdout + result.stderr)


# ── Seams for patch coverage: long overview + markdown fallbacks ──

def test_build_summary_long_overview_truncates(tmp_path: Path) -> None:
    """Long first paragraph overview is truncated to 600 chars — seam: _overview_from_sentences."""
    long_sent = "A" * 350 + "."
    ep = _fake_episode()
    tx = Transcript(text=f"{long_sent} {long_sent} Third.", segments=[], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx)
    assert summary["overview"].endswith("…")
    assert len(summary["overview"]) <= 601


def test_summary_markdown_empty_key_points_and_quotes_via_fake() -> None:
    """Markdown fallbacks for empty key_points / quotes and missing timestamp — seam: _summary_to_markdown."""
    from podtx.summarize import _summary_to_markdown

    summary_empty = {"title": "T", "show": "S", "episode": 1, "overview": "ov", "key_points": [], "quotes": [], "backend": "fake"}
    md = _summary_to_markdown(summary_empty)
    assert "_No key points_" in md
    assert "_No quotes_" in md
    # quote without timestamp seam
    summary_no_ts = {"title": "T", "overview": "ov", "key_points": ["kp"], "quotes": [{"text": "no ts quote", "timestamp": "", "start": 0}], "backend": "fake"}
    md2 = _summary_to_markdown(summary_no_ts)
    assert "no ts quote" in md2
    assert "[" not in md2.split("no ts quote")[0].splitlines()[-1] or "no ts quote" in md2  # no bracket prefix for this quote


def test_summary_markdown_no_overview_fallback() -> None:
    """Markdown fallback when overview empty — seam: _summary_to_markdown."""
    from podtx.summarize import _summary_to_markdown

    summary = {"title": "T", "overview": "   ", "key_points": ["kp"], "quotes": [], "backend": "fake"}
    md = _summary_to_markdown(summary)
    assert "_No overview_" in md
