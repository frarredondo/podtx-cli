from __future__ import annotations

import re

# Standalone filler tokens, including after punctuation / line starts.
_FILLER = re.compile(
    r"(?<!\w)(?:uh|um)(?!\w)[,.]?",
    flags=re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"[^\S\n]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
# Match whole words/contractions: "the the", "I'll I'll"
_WORD_DOUBLE = re.compile(
    r"\b([A-Za-z0-9]+(?:'[A-Za-z]+)?)\s+\1\b",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Light cleanup for human-readable transcripts.

    Removes common fillers (uh/um) and collapses immediate repeated words.
    Preserves paragraph breaks. Does not alter timed segments.
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
    # Collapse doubles repeatedly (the the the -> the)
    prev = None
    while prev != text:
        prev = text
        text = _WORD_DOUBLE.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()
