from __future__ import annotations

from podtx.cleanup import clean_text
from podtx.models import Segment


def round_ts(seconds: float, *, ndigits: int = 3) -> float:
    """Round timestamps for stable, human-friendly serialization."""
    return round(float(seconds), ndigits)


def segments_to_paragraphs(
    segments: list[Segment],
    *,
    gap_seconds: float = 0.8,
) -> str:
    """Join segments into paragraphs, breaking on silence gaps."""
    paragraphs: list[str] = []
    current: list[str] = []
    prev_end: float | None = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if prev_end is not None and (seg.start - prev_end) >= gap_seconds and current:
            paragraphs.append(" ".join(current))
            current = [text]
        else:
            current.append(text)
        prev_end = seg.end

    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def body_text(
    transcript_text: str,
    segments: list[Segment],
    *,
    readable: bool,
    cleanup: bool = False,
) -> str:
    if readable and segments:
        text = segments_to_paragraphs(segments)
    else:
        text = transcript_text.strip()
    if cleanup:
        text = clean_text(text)
    return text
