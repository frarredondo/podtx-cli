from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from podtx.cleanup import clean_text
from podtx.formatting import body_text
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs


def test_clean_text_normalizes_spoken_time_words():
    assert clean_text("seven point zero AM") == "7:00 AM"


def test_clean_text_normalizes_scattered_digits():
    assert clean_text("6 07 a.m.") == "6:07 a.m."


def test_clean_text_conservative_false_positive():
    assert clean_text("I have seven apples") == "I have seven apples"


def test_body_text_cleanup_normalizes_time():
    assert body_text("seven point zero AM", [], readable=False, cleanup=True) == "7:00 AM"
    assert body_text("6 07 a.m.", [], readable=False, cleanup=True) == "6:07 a.m."


def test_writers_cleanup_normalizes_body_but_not_segments(tmp_path: Path):
    episode = Episode(
        guid="g1",
        title="Demo",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        show_title="Show",
    )
    transcript = Transcript(
        text="Meeting at seven point zero AM please.",
        segments=[Segment(0.0, 1.5, "Meeting at seven point zero AM please.")],
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
        readable=False,
        cleanup=True,
    )
    txt = (tmp_path / "ep.txt").read_text(encoding="utf-8")
    assert "7:00 AM" in txt
    payload = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert payload["text"] == "Meeting at 7:00 AM please."
    assert payload["segments"][0]["text"] == "Meeting at seven point zero AM please."


# ── Seams: use fakes to cover remaining branches of _parse_hour/_parse_minute/_normalize ──

def _fake_episode(title: str = "Fake") -> Episode:
    return Episode(guid="f", title=title, enclosure_url="https://x/f.mp3", show_title="Fake Show")


def test_parse_hour_invalid_cases():
    from podtx.cleanup import _parse_hour

    # Not a digit, not in WORD_NUM
    assert _parse_hour("foobar") is None
    # Word out of range (zero is not valid hour)
    assert _parse_hour("zero") is None
    assert _parse_hour("thirteen") is None
    # Digit out of range
    assert _parse_hour("0") is None
    assert _parse_hour("13") is None
    # Valid cases
    assert _parse_hour("7") == 7
    assert _parse_hour("seven") == 7
    assert _parse_hour("Twelve") == 12


def test_parse_minute_invalid_cases():
    from podtx.cleanup import _parse_minute

    # Invalid first word
    assert _parse_minute("foobar", None) is None
    assert _parse_minute("foobar", "five") is None
    # Out of range digit
    assert _parse_minute("60", None) is None
    assert _parse_minute("99", None) is None
    # Word minute with invalid second
    assert _parse_minute("seven", "foobar") is None
    assert _parse_minute("seven", "eleven") is None  # 11 not 0-9
    # Valid combos
    assert _parse_minute("zero", "five") == "05"
    assert _parse_minute("twenty", "five") == "25"
    assert _parse_minute("0", None) == "00"
    assert _parse_minute("5", None) == "05"
    assert _parse_minute("thirty", "zero") == "30"
    assert _parse_minute("one", "five") == "15"


def test_scattered_time_rejects_invalid_hour_minute():
    from podtx.cleanup import clean_text

    # Hour 0 or 13 should not normalize
    assert clean_text("13 07 a.m.") == "13 07 a.m."
    assert clean_text("0 07 p.m.") == "0 07 p.m."
    # Minute 60+ should not normalize (regex allows 2 digits, but _scattered_repl rejects)
    # Note: scattered regex matches \d{2} exactly, so 60+ would be tested via spoken
    assert clean_text("seven point sixty AM") == "seven point sixty AM"


def test_spoken_time_edge_variants():
    from podtx.cleanup import clean_text

    # Hour digit + point
    assert clean_text("7 point zero AM") == "7:00 AM"
    # Words with second word
    assert clean_text("seven point zero zero AM") == "7:00 AM"
    assert clean_text("seven point fifteen AM") == "7:15 AM"
    # Lowercase am/pm preserved
    assert clean_text("seven point zero pm") in ("7:00 pm", "7:00 pm")
    # Scattered with different ampm casing/dots preserved
    assert clean_text("6 07 PM") == "6:07 PM"
    assert clean_text("6 07 am") == "6:07 am"
