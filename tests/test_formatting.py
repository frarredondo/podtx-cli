from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from podtx.formatting import body_text, round_ts, segments_to_paragraphs
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs


def test_round_ts() -> None:
    assert round_ts(2.8800000000000003) == 2.88
    assert round_ts(13.520000000000001) == 13.52


def test_segments_to_paragraphs_breaks_on_gap() -> None:
    segments = [
        Segment(0.0, 1.0, "Hello."),
        Segment(1.1, 2.0, "World."),
        Segment(3.5, 4.0, "New paragraph."),
    ]
    text = segments_to_paragraphs(segments, gap_seconds=0.8)
    assert text == "Hello. World.\n\nNew paragraph."


def test_body_text_raw_vs_readable() -> None:
    segments = [
        Segment(0.0, 1.0, "One."),
        Segment(2.0, 3.0, "Two."),
    ]
    raw = "One. Two. continuous"
    assert body_text(raw, segments, readable=False) == "One. Two. continuous"
    assert body_text(raw, segments, readable=True) == "One.\n\nTwo."


def test_writers_readable_and_rounded(tmp_path: Path) -> None:
    episode = Episode(
        guid="g1",
        title="Interview with Ada",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=3,
        show_title="Demo Podcast",
    )
    transcript = Transcript(
        text="Hello world. Next bit.",
        segments=[
            Segment(start=0.0, end=1.5000000000000002, text="Hello world."),
            Segment(start=3.0, end=4.0, text="Next bit."),
        ],
        language="en",
        model="test-model",
        engine="fake",
    )
    paths = write_outputs(
        out_dir=tmp_path,
        basename="ep",
        episode=episode,
        transcript=transcript,
        formats=("txt", "json"),
        readable=True,
    )
    assert len(paths) == 2
    txt = (tmp_path / "ep.txt").read_text(encoding="utf-8")
    assert "Hello world.\n\nNext bit." in txt

    payload = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert payload["readable"] is True
    assert payload["segments"][0]["end"] == 1.5
    assert "\n\n" in payload["text"]
