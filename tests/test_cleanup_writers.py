from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from podtx.formatting import body_text
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs


def test_body_text_applies_cleanup() -> None:
    segments = [
        Segment(0.0, 1.0, "So uh yeah."),
        Segment(2.0, 3.0, "The the end."),
    ]
    text = body_text("ignored", segments, readable=True, cleanup=True)
    assert text == "So yeah.\n\nThe end."


def test_body_text_cleanup_without_readable() -> None:
    raw = "So uh the the fox."
    assert body_text(raw, [], readable=False, cleanup=True) == "So the fox."


def test_writers_cleanup_keeps_raw_segments(tmp_path: Path) -> None:
    episode = Episode(
        guid="g1",
        title="Demo",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        show_title="Show",
    )
    transcript = Transcript(
        text="So uh the the fox.",
        segments=[Segment(0.0, 1.5, "So uh the the fox.")],
        language="en",
        model="test",
        engine="fake",
    )
    write_outputs(
        out_dir=tmp_path,
        basename="ep",
        episode=episode,
        transcript=transcript,
        formats=("txt", "json"),
        readable=True,
        cleanup=True,
    )
    txt = (tmp_path / "ep.txt").read_text(encoding="utf-8")
    assert "So the fox." in txt
    assert " uh " not in txt

    payload = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert payload["cleanup"] is True
    assert payload["text"] == "So the fox."
    # Segments remain raw archive
    assert payload["segments"][0]["text"] == "So uh the the fox."
