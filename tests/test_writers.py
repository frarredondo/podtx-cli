from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs


def test_writers(tmp_path: Path) -> None:
    episode = Episode(
        guid="g1",
        title="Interview with Ada",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=3,
        show_title="Demo Podcast",
    )
    transcript = Transcript(
        text="Hello world.",
        segments=[Segment(start=0.0, end=1.5, text="Hello world.")],
        language="en",
        model="test-model",
        engine="fake",
    )
    paths = write_outputs(
        out_dir=tmp_path,
        basename="2026-03-15_003_interview-with-ada",
        episode=episode,
        transcript=transcript,
        formats=("txt", "json", "srt", "vtt"),
    )
    assert len(paths) == 4
    txt = (tmp_path / "2026-03-15_003_interview-with-ada.txt").read_text(encoding="utf-8")
    assert "Title: Interview with Ada" in txt
    assert "Hello world." in txt
    json_body = (tmp_path / "2026-03-15_003_interview-with-ada.json").read_text(encoding="utf-8")
    assert '"engine": "fake"' in json_body
    assert '"start": 0.0' in json_body or '"start": 0' in json_body
    # timestamps should not retain float noise
    assert "00000000000000" not in json_body
    srt = (tmp_path / "2026-03-15_003_interview-with-ada.srt").read_text(encoding="utf-8")
    assert "-->" in srt
    vtt = (tmp_path / "2026-03-15_003_interview-with-ada.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
