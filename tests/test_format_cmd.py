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
