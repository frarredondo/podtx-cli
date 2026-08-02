from __future__ import annotations

from podtx.engines.whisper import (
    _drop_trailing_hallucinations,
    _is_suspicious_segment,
)
from podtx.models import Segment


def test_suspicious_stutter() -> None:
    assert _is_suspicious_segment("Moving tohohohohohohohohohohoho")
    assert not _is_suspicious_segment("Thank you so much for listening.")


def test_drop_trailing_hallucinations() -> None:
    segs = [
        Segment(0.0, 1.0, "Life is good."),
        Segment(1.0, 2.0, "Thank you for listening."),
        Segment(2.0, 30.0, "tohohohohohohohohohohohohohoho"),
    ]
    cleaned = _drop_trailing_hallucinations(segs)
    assert len(cleaned) == 2
    assert cleaned[-1].text == "Thank you for listening."
