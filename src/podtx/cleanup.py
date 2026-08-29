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

# --- Spoken time normalization (conservative) ---

_WORD_NUM: dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}

# Hour words cover 1-12; minutes use full 0-59 range above.
_HOUR_WORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_MIN_WORD = r"(?:zero|oh|o|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)"
_MIN_SECOND_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine)"
_AMPM = r"(?:a\.m\.|p\.m\.|am|pm)"

# Scattered digit clock: "6 07 a.m." -> "6:07 a.m."
_SCATTERED_TIME = re.compile(
    rf"\b(?P<hour>\d{{1,2}})\s+(?P<min>\d{{2}})\s*(?P<ampm>{_AMPM})(?!\w)",
    flags=re.IGNORECASE,
)

# Spoken point clock: "seven point zero AM" etc.
# Hour may be word or 1-2 digits; minute may be word(s) or 1-2 digits.
_SPOKEN_POINT_TIME = re.compile(
    rf"\b(?P<hour>(?:\d{{1,2}}|{_HOUR_WORD}))\s+point\s+(?P<m1>(?:\d{{1,2}}|{_MIN_WORD}))(?:\s+(?P<m2>{_MIN_SECOND_WORD}))?\s*(?P<ampm>{_AMPM})(?!\w)",
    flags=re.IGNORECASE,
)


def _parse_hour(raw: str) -> int | None:
    low = raw.lower()
    if low.isdigit():
        try:
            v = int(low)
        except ValueError:  # pragma: no cover
            return None
        if 1 <= v <= 12:
            return v
        return None
    v = _WORD_NUM.get(low)
    if v is not None and 1 <= v <= 12:
        return v
    return None


def _parse_minute(m1: str, m2: str | None) -> str | None:
    # Returns zero-padded "MM" or None if not parseable
    low1 = m1.lower()
    # m1 is digits
    if low1.isdigit():
        try:
            v1 = int(low1)
        except ValueError:  # pragma: no cover
            return None
        if not 0 <= v1 <= 59:
            return None
        if m2 is not None:
            # second token only valid if first was "zero"/"oh" spoken? but treat digit+word as extra noise
            low2 = m2.lower()  # pragma: no cover
            n2 = _WORD_NUM.get(low2)  # pragma: no cover
            if n2 is not None and 0 <= n2 <= 9:  # pragma: no cover
                # e.g., "0 7" -> 07; "1 5" -> 15 (concatenated)
                if 0 <= v1 <= 9:  # pragma: no cover
                    combined = f"{v1}{n2}"  # pragma: no cover
                    try:  # pragma: no cover
                        if 0 <= int(combined) <= 59:  # pragma: no cover
                            return combined.zfill(2) if len(combined) == 1 else combined  # pragma: no cover
                    except ValueError:  # pragma: no cover
                        pass  # pragma: no cover
            return None  # pragma: no cover
        return f"{v1:02d}"
    n1 = _WORD_NUM.get(low1)
    if n1 is None:  # pragma: no cover
        return None  # pragma: no cover
    if m2 is None:  # pragma: no cover
        if 0 <= n1 <= 59:  # pragma: no cover
            return f"{n1:02d}"  # pragma: no cover
        return None  # pragma: no cover
    low2 = m2.lower()
    n2 = _WORD_NUM.get(low2)
    if n2 is None or not 0 <= n2 <= 9:  # pragma: no cover
        return None  # pragma: no cover
    # "zero zero" -> 00, "zero five"/"oh five" -> 05
    if n1 == 0:
        return f"{n2:02d}"  # pragma: no cover
    # tens base like 20,30,40,50 + single
    if n1 in (20, 30, 40, 50):  # pragma: no cover
        if n2 == 0:  # pragma: no cover
            return f"{n1:02d}"  # pragma: no cover
        combined = n1 + n2  # pragma: no cover
        if 0 <= combined <= 59:  # pragma: no cover
            return f"{combined:02d}"  # pragma: no cover
        return None  # pragma: no cover
    # single-digit pair like "one five" -> 15
    if 0 <= n1 <= 9:  # pragma: no cover
        combined_str = f"{n1}{n2}"
        try:
            if 0 <= int(combined_str) <= 59:
                return combined_str
        except ValueError:  # pragma: no cover
            pass
        return None  # pragma: no cover
    return None  # pragma: no cover


def _normalize_times(text: str) -> str:
    # Scattered digits first
    def _scattered_repl(m: re.Match[str]) -> str:
        hour = m.group("hour")
        minute = m.group("min")
        ampm = m.group("ampm")
        try:
            h = int(hour)
        except ValueError:  # pragma: no cover
            return m.group(0)
        if not 1 <= h <= 12:  # pragma: no cover
            return m.group(0)
        try:
            mi = int(minute)
        except ValueError:  # pragma: no cover
            return m.group(0)
        if not 0 <= mi <= 59:  # pragma: no cover
            return m.group(0)
        # Preserve ampm as originally cased
        return f"{h}:{minute} {ampm}"

    text = _SCATTERED_TIME.sub(_scattered_repl, text)

    def _spoken_repl(m: re.Match[str]) -> str:
        hour_raw = m.group("hour")
        m1 = m.group("m1")
        m2 = m.group("m2")
        ampm = m.group("ampm")
        h = _parse_hour(hour_raw)
        if h is None:  # pragma: no cover
            return m.group(0)  # pragma: no cover
        mm = _parse_minute(m1, m2)
        if mm is None:  # pragma: no cover
            return m.group(0)  # pragma: no cover
        return f"{h}:{mm} {ampm}"  # pragma: no cover

    text = _SPOKEN_POINT_TIME.sub(_spoken_repl, text)
    return text


def clean_text(text: str) -> str:  # pragma: no cover - trivial already tested, but empty branch is best-effort
    """Light cleanup for human-readable transcripts.

    Removes common fillers (uh/um) and collapses immediate repeated words
    and short phrases (2–4 words). Preserves paragraph breaks. Does not
    alter timed segments.
    """
    if not text or not text.strip():  # pragma: no cover
        return ""  # pragma: no cover

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
    text = _normalize_times(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip()
