from __future__ import annotations

import re

# Standalone filler tokens, including after punctuation / line starts.
_FILLER = re.compile(
    r"(?<!\w)(?:uh|um)(?!\w)[,.]?",
    flags=re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"[^\S\n]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")

# Word token: letters/digits with optional simple contraction (I'll, don't).
_WORD = r"[A-Za-z0-9]+(?:'[A-Za-z]+)?"

# Immediate phrase doubles, longest first (4 → 1) so "I would like to I would like to"
# is not partially eaten as overlapping shorter repeats.
_PHRASE_DOUBLES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(
        rf"\b((?:{_WORD}\s+){{{n - 1}}}{_WORD})\s+\1\b",
        flags=re.IGNORECASE,
    )
    for n in range(4, 0, -1)
)


def clean_text(text: str) -> str:
    """Light cleanup for human-readable transcripts.

    Removes common fillers (uh/um) and collapses immediate repeated words
    and short phrases (2–4 words). Preserves paragraph breaks. Does not
    alter timed segments.
    """
    if not text or not text.strip():
        return ""

    paragraphs = text.split("\n\n")
    cleaned_paras: list[str] = []
    for para in paragraphs:
        cleaned_paras.append(_clean_paragraph(para))
    return "\n\n".join(p for p in cleaned_paras if p)


def _clean_paragraph(para: str) -> str:
    text = _FILLER.sub(" ", para)
    # Collapse doubles repeatedly (the the the -> the; I think I think -> I think)
    prev = None
    while prev != text:
        prev = text
        for pattern in _PHRASE_DOUBLES:
            text = pattern.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()
