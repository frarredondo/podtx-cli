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


def test_segments_to_paragraphs_breaks_on_sentence_after_min_duration() -> None:
    """When gaps are ~0, still paragraph at sentence ends once long enough."""
    segments = [
        Segment(0.0, 8.0, "First clause continues for a while"),
        Segment(8.0, 16.0, "and then wraps up here."),
        Segment(16.0, 24.0, "Next idea starts after the break"),
        Segment(24.0, 32.0, "and finishes this second paragraph."),
    ]
    text = segments_to_paragraphs(
        segments,
        gap_seconds=0.8,
        min_paragraph_seconds=15.0,
        max_paragraph_seconds=60.0,
        max_paragraph_words=200,
    )
    paras = text.split("\n\n")
    assert len(paras) == 2
    assert paras[0].endswith("here.")
    assert "Next idea" in paras[1]


def test_segments_to_paragraphs_force_break_on_max_seconds() -> None:
    """Force a break when a paragraph runs too long without useful gaps."""
    # No sentence punctuation; still must split by max duration.
    segments = [
        Segment(0.0, 20.0, "talking for a long stretch without ending"),
        Segment(20.0, 40.0, "still going on and on about the topic"),
        Segment(40.0, 55.0, "and somehow never stops for breath"),
    ]
    text = segments_to_paragraphs(
        segments,
        gap_seconds=0.8,
        min_paragraph_seconds=10.0,
        max_paragraph_seconds=35.0,
        max_paragraph_words=500,
    )
    assert "\n\n" in text
    assert len(text.split("\n\n")) >= 2


def test_segments_to_paragraphs_force_break_on_max_words() -> None:
    words = " ".join(f"word{i}" for i in range(80))
    segments = [
        Segment(0.0, 5.0, words),
        Segment(5.0, 10.0, words),
        Segment(10.0, 15.0, "tail end of the thought."),
    ]
    text = segments_to_paragraphs(
        segments,
        gap_seconds=0.8,
        min_paragraph_seconds=100.0,
        max_paragraph_seconds=100.0,
        max_paragraph_words=100,
    )
    assert "\n\n" in text


def test_segments_to_paragraphs_does_not_break_before_min_duration() -> None:
    """Short early sentences should not create tiny paragraphs."""
    segments = [
        Segment(0.0, 1.0, "Hi."),
        Segment(1.0, 2.0, "Hey."),
        Segment(2.0, 3.5, "Welcome back."),
    ]
    text = segments_to_paragraphs(
        segments,
        gap_seconds=0.8,
        min_paragraph_seconds=20.0,
        max_paragraph_seconds=60.0,
        max_paragraph_words=200,
    )
    assert "\n\n" not in text
    assert text == "Hi. Hey. Welcome back."


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


def test_segments_to_paragraphs_skips_empty_segment() -> None:
    from podtx.formatting import segments_to_paragraphs
    from podtx.models import Segment

    segs = [
        Segment(0.0, 1.0, "   "),
        Segment(2.0, 3.0, "Actual words here."),
    ]
    out = segments_to_paragraphs(segs)
    assert "Actual words here." in out
    assert "   " not in out


def test_body_text_diarize_non_readable_uses_speaker_lines() -> None:
    from podtx.formatting import body_text
    from podtx.models import Segment

    segs = [
        Segment(0.0, 1.0, "Hello", speaker="person-a"),
        Segment(1.0, 2.0, "There", speaker="person-a"),
    ]
    out = body_text("IGNORED", segs, readable=False, diarize=True)
    assert out == "person-a: Hello\nperson-a: There"


def test_segments_to_paragraphs_with_speaker_change_and_gap() -> None:
    from podtx.formatting import segments_to_paragraphs_with_speaker
    from podtx.models import Segment

    segs = [
        Segment(0.0, 1.0, "Opening words.", speaker="alice"),
        Segment(2.0, 3.0, "Gap after opener.", speaker="bob"),
        Segment(4.0, 5.0, "Bob continues", speaker="bob"),
    ]
    out = segments_to_paragraphs_with_speaker(
        segs,
        gap_seconds=0.8,
        min_paragraph_seconds=100.0,
        max_paragraph_seconds=1000.0,
        max_paragraph_words=1000,
    )
    assert "alice: Opening words." in out
    assert "bob: Gap after opener." in out


def test_body_text_with_report_correct_names() -> None:
    from podtx.formatting import body_text_with_report
    from podtx.models import Episode

    ep = Episode(
        guid="g",
        title="Sabine Wojcieszak",
        enclosure_url="https://x/a.mp3",
    )
    text, subs = body_text_with_report(
        "Today we talk with Sabina Vosheshak.",
        [],
        readable=False,
        correct_names=True,
        episode=ep,
    )
    assert "Sabine Wojcieszak" in text
    assert len(subs) >= 0


def test_segments_to_text_with_speaker_skips_blank() -> None:
    from podtx.formatting import segments_to_text_with_speaker
    from podtx.models import Segment

    segs = [Segment(0.0, 1.0, "  "), Segment(2.0, 3.0, "Real", speaker="bob")]
    assert segments_to_text_with_speaker(segs) == "bob: Real"


def test_segments_to_paragraphs_with_speaker_flushes_empty() -> None:
    from podtx.formatting import segments_to_paragraphs_with_speaker

    assert segments_to_paragraphs_with_speaker([]) == ""


def test_body_text_with_report_no_correct_names() -> None:
    from podtx.formatting import body_text_with_report

    text, subs = body_text_with_report("  Some raw text.  ", [], readable=False, correct_names=False)
    assert text == "Some raw text."
    assert subs == []
