from __future__ import annotations

import re

from podtx.cleanup import clean_text
from podtx.models import Segment

# Sentence end: . ! ? … with optional closing quotes/brackets.
_SENTENCE_END = re.compile(r'[.!?…]["\'”’)\]]*$')


def round_ts(seconds: float, *, ndigits: int = 3) -> float:
    """Round timestamps for stable, human-friendly serialization."""
    return round(float(seconds), ndigits)


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.rstrip()))


def _word_count(text: str) -> int:
    return len(text.split())


def segments_to_paragraphs(
    segments: list[Segment],
    *,
    gap_seconds: float = 0.8,
    min_paragraph_seconds: float = 20.0,
    max_paragraph_seconds: float = 45.0,
    max_paragraph_words: int = 120,
) -> str:
    """Join segments into paragraphs for human reading.

    Breaks when:
    - silence gap between segments is >= ``gap_seconds``, or
    - a sentence ends after the paragraph has reached ``min_paragraph_seconds``, or
    - paragraph duration/words hit ``max_paragraph_*`` (prefer a sentence end;
      otherwise force the break so walls of text still split).

    Timed segment data is not modified; this only shapes the text body.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    prev_end: float | None = None
    para_start: float | None = None
    para_words = 0

    def flush() -> None:
        nonlocal current, para_start, para_words
        if current:
            paragraphs.append(" ".join(current))
        current = []
        para_start = None
        para_words = 0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        if (
            prev_end is not None
            and (seg.start - prev_end) >= gap_seconds
            and current
        ):
            flush()

        if para_start is None:
            para_start = seg.start

        current.append(text)
        para_words += _word_count(text)
        prev_end = seg.end
        duration = seg.end - para_start

        sentence = _ends_sentence(text)
        soft_ready = duration >= min_paragraph_seconds
        over_budget = (
            duration >= max_paragraph_seconds or para_words >= max_paragraph_words
        )

        # Prefer sentence boundaries; force a split once over budget either way.
        if (sentence and soft_ready) or over_budget:
            flush()

    flush()
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
