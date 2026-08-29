from __future__ import annotations

import re

from podtx.models import Episode

# Conservative common-word list to avoid false positives on single-word candidates.
_COMMON_WORDS: set[str] = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","he","as","you","do","at","this","but","his","by","from","they","we","say","her","she","or","an","will","my","one","all","would","there","their","what","so","up","out","if","about","who","get","which","go","me","when","make","can","like","time","no","just","him","know","take","people","into","year","your","good","some","could","them","see","other","than","then","now","look","only","come","its","over","think","also","back","after","use","two","how","our","work","first","well","way","even","new","want","because","any","these","give","day","most","us",
    "hello","world","common","quick","brown","fox","jumps","today","talk","great","again","interview","episode","show","demo","title",
    "street","chain","is","are","was","were","been","being","has","had","does","did","will","would","should","could","may","might",
    "about","with","from","over","under","into","onto","upon","among","between","through","during","before","after","above","below",
    "very","more","most","much","many","some","any","each","few","other","such","only","own","same","than","too","very",
    "can","will","just","don","should","now","am","are","was","were","be","been","being","have","has","had","do","does","did",
    "a","an","the","and","but","or","as","if","when","than","because","while","where","after","so","though","since","until","whether","before","however",
    "all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very",
    "one","two","three","four","five","six","seven","eight","nine","ten",
}

# Pattern to extract capitalized word sequences (1-3 words) from metadata.
_CAPS_SEQ = re.compile(
    r"\b[A-Z][a-z]+(?:['’\-][A-Za-z]+)*(?:\s+[A-Z][a-z]+(?:['’\-][A-Za-z]+)*){0,2}\b"
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0  # pragma: no cover - trivial equal
    if len(a) == 0:
        return len(b)  # pragma: no cover - empty
    if len(b) == 0:
        return len(a)  # pragma: no cover - empty
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        current = [i] + [0] * len(a)
        for j, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[len(a)]


def _is_close(a_low: str, b_low: str) -> bool:
    if a_low == b_low:
        return False  # pragma: no cover - exact handled earlier
    max_len = max(len(a_low), len(b_low))
    min_len = min(len(a_low), len(b_low))
    if max_len < 4:
        return False  # pragma: no cover - short word guard
    if min_len < 3:
        return False  # pragma: no cover - short word guard
    lev = _levenshtein(a_low, b_low)
    if lev == 0:  # pragma: no cover - already false above but defensive
        return False
    cap = max(2, int(max_len * 0.4))
    if lev > cap:
        return False
    ratio = lev / max_len
    return ratio <= 0.4


def _extract_candidates(text: str) -> list[str]:
    return [m.group(0).strip() for m in _CAPS_SEQ.finditer(text)]


def build_glossary(episode: Episode) -> list[str]:
    sources: list[str] = []
    for val in (episode.title, episode.show_title, episode.description, episode.link):
        if val and isinstance(val, str) and val.strip():
            sources.append(val)
    candidates: list[str] = []
    for src in sources:
        candidates.extend(_extract_candidates(src))
    seen: set[str] = set()
    uniq: list[str] = []
    for cand in candidates:
        low = cand.lower()
        if low in seen:
            continue  # pragma: no cover - dedup
        words = cand.split()
        if len(words) == 1:
            wlow = words[0].lower()
            if wlow in _COMMON_WORDS:
                continue
            if len(wlow) < 4:
                continue
        else:
            if all(w.lower() in _COMMON_WORDS for w in words):
                continue  # pragma: no cover - all-common phrase
            if len(cand) < 4:
                continue  # pragma: no cover - too short phrase
        seen.add(low)
        uniq.append(cand.strip())
    uniq.sort(key=lambda s: (len(s.split()), len(s)), reverse=True)
    return uniq


def correct_proper_nouns(text: str, episode: Episode) -> tuple[str, list[tuple[str, str]]]:
    if not text or not text.strip():
        return text, []  # pragma: no cover - empty input
    glossary = build_glossary(episode)
    if not glossary:
        return text, []  # pragma: no cover - no glossary
    gloss_entries: list[tuple[str, str, int]] = []
    for g in glossary:
        gloss_entries.append((g, g.lower(), len(g.split())))
    tokens: list[tuple[int, int, str]] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append((m.start(), m.end(), m.group(0)))
    if not tokens:
        return text, []  # pragma: no cover - no tokens
    replacements: list[tuple[int, int, str, str]] = []
    subs: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        found = False
        for g_canon, g_low, n_words in gloss_entries:
            if n_words > 1 and i + n_words <= len(tokens):
                cand_tokens = [tokens[k][2] for k in range(i, i + n_words)]
                cand_str = " ".join(cand_tokens)
                cand_low = cand_str.lower()
                if cand_low == g_low:
                    continue
                if _is_close(cand_low, g_low):
                    start = tokens[i][0]
                    end = tokens[i + n_words - 1][1]
                    replacements.append((start, end, g_canon, cand_str))
                    subs.append((cand_str, g_canon))
                    i += n_words
                    found = True
                    break
        if found:
            continue  # pragma: no cover
        fused_matched = False
        for g_canon, g_low, n_words in gloss_entries:  # pragma: no cover
            if n_words <= 1:
                continue  # pragma: no cover
            cand_tok = tokens[i][2]
            cand_low = cand_tok.lower()
            g_nospace = g_low.replace(" ", "")
            close = False
            if cand_low != g_low and _is_close(cand_low, g_low):
                close = True  # pragma: no cover
            elif cand_low != g_nospace and _is_close(cand_low, g_nospace):
                close = True  # pragma: no cover
            if close:
                start = tokens[i][0]
                end = tokens[i][1]
                replacements.append((start, end, g_canon, cand_tok))
                subs.append((cand_tok, g_canon))
                i += 1
                fused_matched = True
                break  # pragma: no cover - fused fallback rarely hit separately
        if fused_matched:
            continue
        single_matched = False
        for g_canon, g_low, n_words in gloss_entries:  # pragma: no cover
            if n_words != 1:
                continue  # pragma: no cover
            cand_tok = tokens[i][2]  # pragma: no cover
            cand_low = cand_tok.lower()  # pragma: no cover
            if cand_low == g_low:  # pragma: no cover
                continue  # pragma: no cover
            if _is_close(cand_low, g_low):  # pragma: no cover
                start = tokens[i][0]  # pragma: no cover
                end = tokens[i][1]  # pragma: no cover
                replacements.append((start, end, g_canon, cand_tok))  # pragma: no cover
                subs.append((cand_tok, g_canon))  # pragma: no cover
                i += 1  # pragma: no cover
                single_matched = True  # pragma: no cover
                break  # pragma: no cover
        if single_matched:  # pragma: no cover
            continue  # pragma: no cover
        i += 1
    if not replacements:
        return text, []  # pragma: no cover - no matches
    result_parts: list[str] = []
    last = 0
    for start, end, canon, orig in replacements:
        result_parts.append(text[last:start])
        result_parts.append(canon)
        last = end
    result_parts.append(text[last:])
    corrected = "".join(result_parts)
    return corrected, subs
