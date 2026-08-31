"""Durable insight ("nuggets") extraction with a scored rubric.

Uses the shared provider layer (`podtx/providers/`). Default backend `fake`
is offline extractive with no network, mirroring `podtx summarize`. Every
emitted quote is mechanically verified against the transcript and cited with
a timestamp resolved from the transcript's segments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from podtx.config import (
    DEFAULT_NUGGETS_MAX_INPUT_CHARS,
    DEFAULT_NUGGETS_TEMPERATURE,
    DEFAULT_NUGGETS_TIMEOUT,
)
from podtx.format_cmd import TranscriptJsonError, load_transcript_json
from podtx.models import Episode, Transcript
from podtx.providers import (
    DEFAULT_DRY_OUTPUT_CHARS,
    Provider,
    ProviderError,
    available_providers,
    build_provider,
    estimate_cost,
    estimate_tokens,
    get_model,
    normalize_backend,
    get_spec,
)

NUGGETS_PROMPT_VERSION = "nuggets-rubric-1"

_MAX_NUGGETS = 7
_RUBRIC_PASS = 5
_MAX_QUOTE_WORDS = 30
_OVERLAP_FRACTION = 0.1


class NuggetsError(Exception):
    """Raised for nugget extraction failures (backend, schema-invalid output)."""


def _valid_backend(backend: str) -> str:
    key = normalize_backend(backend)
    known = {"fake", *available_providers()}
    if key not in known:
        raise NuggetsError(
            f"Unknown backend {backend!r}. Choose from: {', '.join(sorted(known))}"
        )
    return key


def _checked_formats(formats: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and validate output formats."""
    out = tuple(f.lower().strip() for f in formats)
    for fmt in out:
        if fmt not in {"json", "md"}:
            raise NuggetsError(f"Unsupported format {fmt!r}. Choose from: json, md")
    return out


def _format_timestamp(seconds: float) -> str:
    total = int(float(seconds))
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _rubric_messages(episode: Episode, transcript_text: str, basename: str) -> list[dict[str, str]]:
    title = episode.title or "Untitled"
    show = episode.show_title or ""
    header = f"Podcast: {show} — Episode: {title}" if show else f"Episode: {title}"
    system = (
        "You are extracting durable, non-obvious wisdom from a podcast episode transcript for a "
        "software-engineering / technical-career audience.\n\n"
        "Extract 3-7 nuggets. A nugget is one of: a memorable quote, a counterintuitive insight, "
        "a hard-won lesson, a mental model, or an operating principle.\n\n"
        "Score each candidate 0-2 on every axis (silently; only output nuggets that clear the bar):\n"
        "- Timelessness: still true/relevant in 5+ years? 0 = tied to a tool/version/news, 2 = a durable principle\n"
        "- Surprise: inverts a common assumption or reveals something non-obvious? 0 = restates conventional wisdom, 2 = genuinely counterintuitive\n"
        "- Evidence: backed by a concrete example, quote, or number? 0 = vague assertion, 2 = a specific verbatim quote or data point\n"
        "- Actionability: could a listener apply or test this tomorrow? 0 = purely descriptive, 2 = a decision rule or practice\n\n"
        "Keep only nuggets scoring 5/8 or higher. Discard tool-version news, intro/outro chatter, "
        "sponsor reads, and guest self-promotion regardless of score.\n\n"
        "Primary lens: software engineering and technical careers — architecture, debugging, incentives, "
        "team dynamics, hiring, career growth, technical decision-making. Tag each nugget \"eng\" for that "
        "lens, or \"general\" if it is broader wisdom that still clears the bar.\n\n"
        "Per nugget output:\n"
        "- insight: 1-2 sentences, punchy, stated as a general principle (not \"the guest said...\")\n"
        "- context: guest name + episode\n"
        "- why_it_matters: 1 sentence, framed for a software engineer\n"
        "- quote: verbatim, under 30 words, copied character-for-character from the transcript — never "
        "paraphrase as a quote; use \"\" if none supports the insight\n"
        "- scores: object with keys T, S, E, A, each 0-2\n"
        "- tag: \"eng\" or \"general\"\n\n"
        "Rank the episode's nuggets best-first (highest total score first).\n"
        "Return strict JSON only: {\"nuggets\": [{\"insight\": string, \"context\": string, "
        "\"why_it_matters\": string, \"quote\": string, \"scores\": {\"T\": int, \"S\": int, \"E\": int, "
        "\"A\": int}, \"tag\": string}]}"
    )
    user = f"{header}\nEpisode file: {basename}\n\nTranscript:\n{transcript_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_json(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        raise NuggetsError(f"LLM did not return valid JSON: {raw[:500]!r}")
    return payload


def _clean_score(value: object) -> int:
    try:
        v = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(2, v))


def _validate_payload(payload: dict) -> list[dict]:
    nuggets = payload.get("nuggets")
    if not isinstance(nuggets, list) or not nuggets:
        raise NuggetsError("LLM payload missing 'nuggets' array")
    cleaned: list[dict] = []
    for raw in nuggets[: _MAX_NUGGETS]:
        if not isinstance(raw, dict):
            continue
        insight = str(raw.get("insight", "")).strip()
        if not insight:
            continue
        scores = raw.get("scores")
        scores = scores if isinstance(scores, dict) else {}
        t = _clean_score(scores.get("T")) if isinstance(scores, dict) else 0
        s = _clean_score(scores.get("S")) if isinstance(scores, dict) else 0
        e = _clean_score(scores.get("E")) if isinstance(scores, dict) else 0
        a = _clean_score(scores.get("A")) if isinstance(scores, dict) else 0
        total = t + s + e + a
        if total < _RUBRIC_PASS:
            continue
        quote = str(raw.get("quote", "")).strip()
        words = quote.split()
        if len(words) > _MAX_QUOTE_WORDS:
            quote = " ".join(words[: _MAX_QUOTE_WORDS])
        tag = str(raw.get("tag", "eng")).strip().lower()
        if tag not in {"eng", "general"}:
            tag = "eng"
        cleaned.append(
            {
                "insight": insight,
                "context": str(raw.get("context", "")).strip(),
                "why_it_matters": str(raw.get("why_it_matters", "")).strip(),
                "quote": quote,
                "scores": {"T": t, "S": s, "E": e, "A": a},
                "total": total,
                "tag": tag,
                "start": 0.0,
                "end": 0.0,
                "timestamp": "",
            }
        )
    if not cleaned:
        raise NuggetsError(
            "LLM payload 'nuggets' empty after validation (nothing scored >= 5/8)"
        )
    cleaned.sort(key=lambda n: (n["total"],), reverse=True)
    return cleaned


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _demote_quote(nugget: dict) -> dict:
    scores = dict(nugget["scores"])
    scores["E"] = min(scores["E"], 1)
    return {
        **nugget,
        "quote": "",
        "scores": scores,
        "total": scores["T"] + scores["S"] + scores["E"] + scores["A"],
        "start": 0.0,
        "end": 0.0,
        "timestamp": "",
    }


def _normalized_text(transcript: Transcript) -> str:
    """Full transcript text (normalized) aligned with segment boundaries."""
    segs = [s.text for s in transcript.segments if s.text.strip()]
    if segs:
        return _normalize(" ".join(segs))
    return _normalize(transcript.text)


def _locate_quote(nq: str, segs: list, full: str):
    """Return the segment covering ``nq``, or None.

    Prefers an exact single-segment containment match, then falls back to
    resolving the normalized character offset of the quote in ``full`` so
    quotes that cross a segment boundary still get a timestamp.
    """
    for seg in segs:
        ns = _normalize(seg.text)
        if ns and (nq in ns or ns in nq):
            return seg
    idx = full.find(nq)
    if idx == -1:
        return None
    pos = 0
    for seg in segs:
        length = len(_normalize(seg.text)) + 1
        if idx < pos + length:
            return seg
        pos += length
    return None


def _verify_quotes(nuggets: list[dict], transcript: Transcript) -> list[dict]:
    """Mechanical anti-hallucination check + segment timestamps."""
    segs = [s for s in transcript.segments if s.text.strip()]
    full = _normalized_text(transcript)
    verified: list[dict] = []
    for n in nuggets:
        quote = n.get("quote", "").strip()
        if not quote:
            verified.append(n)
            continue
        nq = _normalize(quote)
        if not nq or nq not in full:
            verified.append(_demote_quote(n))
            continue
        seg = _locate_quote(nq, segs, full)
        if seg is None:
            verified.append(_demote_quote(n))
            continue
        verified.append(
            {
                **n,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "timestamp": _format_timestamp(seg.start),
            }
        )
    return verified


def _chunk_segments(
    transcript: Transcript, *, budget_chars: int, overlap_chars: int
) -> list[list[int]]:
    segs = [s for s in transcript.segments if s.text.strip()]
    if not segs:
        return [[0]]
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i, seg in enumerate(segs):
        add = len(seg.text) + 1
        if cur and cur_len + add > budget_chars:
            chunks.append(cur)
            total = 0
            carry: list[int] = []
            for j in range(len(cur) - 1, -1, -1):
                inner = segs[cur[j]]
                total += len(inner.text) + 1
                if total > overlap_chars:
                    break
                carry.append(cur[j])
            carry.reverse()
            cur = carry
            cur_len = sum(len(segs[k].text) + 1 for k in cur)
        cur.append(i)
        cur_len += add
    chunks.append(cur)
    return chunks


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    """Split free text into word-wrapped pieces each within ``max_chars``.

    Used when a transcript has no segments, or a single segment exceeds the
    input budget, so the model is never given over-budget text.
    """
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for word in text.split():
        add = len(word) + 1
        if cur and cur_len + add > max_chars:
            pieces.append(" ".join(cur))
            tail: list[str] = []
            total = 0
            for prev in reversed(cur):
                total += len(prev) + 1
                if total > overlap_chars:
                    break
                tail.append(prev)
            tail.reverse()
            cur = tail
            cur_len = sum(len(w) + 1 for w in cur)
        cur.append(word)
        cur_len += add
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_chunks(transcript: Transcript, *, max_input_chars: int) -> list[str]:
    text = _transcript_text(transcript)
    if len(text) <= max_input_chars:
        return [text]
    overlap = max(1, int(max_input_chars * _OVERLAP_FRACTION))
    if not transcript.segments:
        return _split_text(text, max_chars=max_input_chars, overlap_chars=overlap)
    chunks = _chunk_segments(
        transcript, budget_chars=max_input_chars, overlap_chars=overlap
    )
    pieces: list[str] = []
    for idx in chunks:
        joined = " ".join(transcript.segments[i].text.strip() for i in idx)
        if len(joined) <= max_input_chars:
            pieces.append(joined)
        else:
            pieces.extend(
                _split_text(joined, max_chars=max_input_chars, overlap_chars=overlap)
            )
    return pieces


@dataclass(frozen=True)
class DryRunEstimate:
    """Token/cost estimate for `podtx nuggets --dry-run` (no inference)."""

    input_chars: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_length: int | None
    chunked: bool
    chunk_count: int
    fits: bool | None
    cost_usd: float | None
    cost_known: bool
    model_known: bool


def estimate_dry_run(
    episode: Episode,
    transcript: Transcript,
    *,
    backend: str,
    model: str | None,
    max_input_chars: int | None,
    providers: dict,
    output_chars: int = DEFAULT_DRY_OUTPUT_CHARS,
) -> DryRunEstimate:
    """Estimate token usage and cost for a nugget extraction without running it.

    ``providers`` is the raw models.dev provider map (an empty dict disables the
    catalog). Estimates are still produced for the offline `fake` backend, but no
    pricing applies. ``max_input_chars`` falls back to the nuggets default.
    """
    budget = (
        max_input_chars
        if max_input_chars is not None
        else DEFAULT_NUGGETS_MAX_INPUT_CHARS
    )
    input_chars = len(_transcript_text(transcript))
    input_tokens = estimate_tokens(input_chars)
    output_tokens = estimate_tokens(output_chars)
    total_tokens = input_tokens + output_tokens
    chunks = _split_chunks(transcript, max_input_chars=budget)
    chunked = len(chunks) > 1
    if backend == "fake":
        return DryRunEstimate(
            input_chars=input_chars,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            context_length=None,
            chunked=chunked,
            chunk_count=len(chunks),
            fits=None,
            cost_usd=None,
            cost_known=False,
            model_known=False,
        )
    info = get_model(providers, backend, model or "")
    model_known = info is not None
    fits = None
    if info is not None:
        if info.context_length is None:
            fits = None
        else:
            fits = input_tokens <= info.context_length
    est = estimate_cost(
        providers,
        backend,
        model or "",
        input_chars=input_chars,
        output_chars=output_chars,
    )
    return DryRunEstimate(
        input_chars=input_chars,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        context_length=info.context_length if info is not None else None,
        chunked=chunked,
        chunk_count=len(chunks),
        fits=fits,
        cost_usd=est.cost_usd,
        cost_known=est.cost_known,
        model_known=model_known,
    )


def _transcript_text(transcript: Transcript) -> str:
    t = transcript.text.strip()
    if not t and transcript.segments:
        t = " ".join(s.text.strip() for s in transcript.segments if s.text.strip())
    return t


def _extract_chunk_with_retry(
    provider: Provider,
    episode: Episode,
    chunk_text: str,
    basename: str,
    *,
    timeout: float,
    temperature: float,
) -> list[dict]:
    messages = _rubric_messages(episode, chunk_text, basename)
    raw = provider.complete(messages, timeout=timeout, temperature=temperature)
    try:
        return _validate_payload(_parse_json(raw))
    except NuggetsError:
        pass
    raw = provider.complete(messages, timeout=timeout, temperature=temperature)
    try:
        return _validate_payload(_parse_json(raw))
    except NuggetsError as exc:
        raise NuggetsError(
            f"{provider.name} returned schema-invalid output after 1 retry: {exc}"
        ) from exc


def _merge_nuggets(nuggets: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for n in nuggets:
        key = _normalize(n["insight"])
        if not key:
            continue
        if key not in seen or n["total"] > seen[key]["total"]:
            seen[key] = n
    merged = sorted(seen.values(), key=lambda n: n["total"], reverse=True)
    return merged[: _MAX_NUGGETS]


_JACCARD_THRESHOLD = 0.6
_MIN_QUOTE_CONTAINMENT_CHARS = 12
_MAX_BEST_OF_SHOW = 10
_CLUSTER_SYSTEM = (
    "You are clustering durable insights extracted from podcast episodes. "
    "Group nuggets that express the same underlying idea, even when worded "
    "differently or said by different guests. Every id must appear in exactly "
    "one group; leave unrelated nuggets ungrouped."
)


def _nugget_total(nugget: dict) -> int:
    total = nugget.get("total")
    if isinstance(total, int):
        return total
    values = nugget.get("scores") or {}
    tallied = [v for v in values.values() if isinstance(v, int)]
    return sum(tallied) if tallied else 0


def _fake_nuggets(episode: Episode, transcript: Transcript, basename: str) -> list[dict]:
    segs = [s for s in transcript.segments if s.text.strip()]
    nuggets: list[dict] = []
    if segs:
        n = min(5, len(segs))
        if n == 1:
            idxs = [0]
        elif n <= 3:
            idxs = list(range(n))
        else:
            idxs = [0, n // 2, n - 1]
        for i in idxs:
            seg = segs[i]
            text = seg.text.strip()
            quote = text
            words = quote.split()
            if len(words) > _MAX_QUOTE_WORDS:
                quote = " ".join(words[: _MAX_QUOTE_WORDS])
            insight = text if len(text) <= 140 else text[:137].rstrip() + "..."
            nuggets.append(
                {
                    "insight": insight,
                    "context": f"{episode.show_title or 'Show'} — {basename}".strip(" —"),
                    "why_it_matters": "Extractive placeholder — replace with LLM nuggets via --backend openrouter|opencode|openai|anthropic|lmstudio.",
                    "quote": quote,
                    "scores": {"T": 2, "S": 1, "E": 2, "A": 1},
                    "total": 6,
                    "tag": "general",
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "timestamp": _format_timestamp(seg.start),
                }
            )
    if not nuggets:
        text = _transcript_text(transcript)
        if text:
            insight = text if len(text) <= 140 else text[:137].rstrip() + "..."
            nuggets.append(
                {
                    "insight": insight,
                    "context": f"{episode.show_title or 'Show'} — {basename}".strip(" —"),
                    "why_it_matters": "No transcript content — nothing extractable.",
                    "quote": "",
                    "scores": {"T": 2, "S": 1, "E": 1, "A": 1},
                    "total": 5,
                    "tag": "eng",
                    "start": 0.0,
                    "end": 0.0,
                    "timestamp": "",
                }
            )
    return nuggets[: _MAX_NUGGETS]


def _normalize_tokens(text: str) -> set[str]:
    return set(_normalize(text or "").split())


def _insight_jaccard(a: dict, b: dict) -> float:
    ta = _normalize_tokens(a.get("insight"))
    tb = _normalize_tokens(b.get("insight"))
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return len(ta & tb) / union


def _quote_contains(a: dict, b: dict) -> bool:
    qa = _normalize(a.get("quote") or "")
    qb = _normalize(b.get("quote") or "")
    if len(qa) < _MIN_QUOTE_CONTAINMENT_CHARS or len(qb) < _MIN_QUOTE_CONTAINMENT_CHARS:
        return False
    return qa in qb or qb in qa


def _same_idea_offline(a: dict, b: dict) -> bool:
    """Offline overlap classifier: two nuggets express the same idea."""
    return _insight_jaccard(a, b) >= _JACCARD_THRESHOLD or _quote_contains(a, b)


def _cluster_offline(nuggets: list[dict]) -> list[list[int]]:
    used = [False] * len(nuggets)
    groups: list[list[int]] = []
    for i in range(len(nuggets)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(nuggets)):
            if used[j]:
                continue
            if any(_same_idea_offline(nuggets[k], nuggets[j]) for k in group):
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups


def _cluster_messages(nuggets: list[dict]) -> list[dict]:
    lines = [f"{i}: {n.get('insight', '')}" for i, n in enumerate(nuggets)]
    body = "\n".join(lines)
    return [
        {"role": "system", "content": _CLUSTER_SYSTEM},
        {
            "role": "user",
            "content": (
                "Cluster these nuggets by underlying idea:\n"
                f"{body}\n"
                'Reply JSON only: {"groups": [{"ids": [0, 2], "label": "short label"}]}. '
                "Put each id in at most one group; ids not listed stay ungrouped."
            ),
        },
    ]


def _parse_clusters(raw: str, count: int) -> list[list[int]] | None:
    try:
        data = _parse_json(raw)
    except NuggetsError:
        return None
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list) or not groups:
        return []
    seen: set[int] = set()
    result: list[list[int]] = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        ids = group.get("ids")
        if not isinstance(ids, list):
            return None
        clean: list[int] = []
        for i in ids:
            if (
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < count
                or i in seen
            ):
                return None
            seen.add(i)
            clean.append(i)
        if clean:
            result.append(clean)
    return result or None


def _verify_semantic_groups(groups: list[list[int]], nuggets: list[dict]) -> list[list[int]]:
    """Only keep members that also clear the offline overlap test vs the group seed."""
    verified: list[list[int]] = []
    for group in groups:
        seed = max(group, key=lambda i: _nugget_total(nuggets[i]))
        kept = [seed]
        for i in group:
            if i != seed and _same_idea_offline(nuggets[i], nuggets[seed]):
                kept.append(i)
        verified.append(kept)
    return verified


def _cluster_semantic(
    provider: Provider, nuggets: list[dict], *, timeout: float, temperature: float
) -> list[list[int]]:
    messages = _cluster_messages(nuggets)
    for _ in range(2):
        raw = provider.complete(messages, timeout=timeout, temperature=temperature)
        parsed = _parse_clusters(raw, len(nuggets))
        if parsed is not None:
            return parsed
    return []


def _load_sidecar(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("nuggets"), list):
        return None
    return data


def _sidecar_summary(data: dict) -> dict:
    return {
        "title": data.get("title", ""),
        "show": data.get("show", ""),
        "episode": data.get("episode"),
        "guid": data.get("guid", ""),
    }


def _nugget_source(sidecar: dict, nugget: dict) -> dict:
    return {
        "title": sidecar.get("title", ""),
        "show": sidecar.get("show", ""),
        "episode": sidecar.get("episode"),
        "guid": sidecar.get("guid", ""),
        "start": nugget.get("start"),
        "end": nugget.get("end"),
        "timestamp": nugget.get("timestamp", ""),
    }


def _corpus_entry(seed: dict, sources: list[dict]) -> dict:
    merged = len(sources) > 1
    return {
        "insight": seed.get("insight", ""),
        "context": (f"{len(sources)} episodes" if merged else seed.get("context", "")),
        "why_it_matters": seed.get("why_it_matters", ""),
        "quote": seed.get("quote", ""),
        "scores": seed.get("scores"),
        "total": _nugget_total(seed),
        "tag": seed.get("tag", "eng"),
        "merged": merged,
        "sources": sources,
    }


def merge_nugget_sidecars(
    sidecars: list[Path],
    *,
    in_scope: int,
    provider: Provider | None = None,
    timeout: float = DEFAULT_NUGGETS_TIMEOUT,
    temperature: float = DEFAULT_NUGGETS_TEMPERATURE,
) -> dict:
    """Merge same-idea nuggets across sidecars into a corpus report.

    Returns the corpus dict: clustering mode, episodes processed vs in scope,
    malformed sidecars skipped, and merged ``groups`` (best-first by score).
    ``provider`` enables semantic clustering; it falls back to offline
    clustering on unavailable or schema-invalid output.
    """
    sidecar_summaries: list[dict] = []
    nuggets: list[dict] = []
    origins: list[int] = []
    skipped: list[str] = []
    for path in sidecars:
        data = _load_sidecar(path)
        if data is None:
            skipped.append(str(path))
            continue
        index = len(sidecar_summaries)
        sidecar_summaries.append(_sidecar_summary(data))
        for nugget in data.get("nuggets", []):
            if isinstance(nugget, dict) and nugget.get("insight"):
                nuggets.append(nugget)
                origins.append(index)

    if not nuggets:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clustering": "offline",
            "episodes_in_scope": in_scope,
            "episodes_processed": len(sidecar_summaries),
            "sidecars_skipped": skipped,
            "groups": [],
        }

    if provider is None:
        clusters = _cluster_offline(nuggets)
        mode = "offline"
    else:
        clusters = _cluster_semantic(
            provider, nuggets, timeout=timeout, temperature=temperature
        )
        if clusters:
            clusters = _verify_semantic_groups(clusters, nuggets)
            mode = "semantic"
        else:
            mode = "offline"
            clusters = _cluster_offline(nuggets)

    grouped: set[int] = set()
    for group in clusters:
        grouped.update(group)
    for i in range(len(nuggets)):
        if i not in grouped:
            clusters.append([i])
            grouped.add(i)

    entries: list[dict] = []
    for group in clusters:
        ordered = sorted(group, key=lambda i: -_nugget_total(nuggets[i]))
        seed_index = ordered[0]
        sources = [
            _nugget_source(sidecar_summaries[origins[i]], nuggets[i]) for i in ordered
        ]
        entries.append(_corpus_entry(nuggets[seed_index], sources))
    entries.sort(key=lambda e: e["total"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clustering": mode,
        "episodes_in_scope": in_scope,
        "episodes_processed": len(sidecar_summaries),
        "sidecars_skipped": skipped,
        "groups": entries,
    }


def _merge_corpus_markdown(data: dict) -> str:
    lines: list[str] = []
    lines.append("# Nugget Corpus")
    lines.append("")
    lines.append(
        f"Episodes: {data.get('episodes_processed', 0)} processed of "
        f"{data.get('episodes_in_scope', 0)} in scope"
    )
    skipped = data.get("sidecars_skipped") or []
    if skipped:
        lines.append(f"Skipped: {len(skipped)} malformed sidecar(s)")
    lines.append(f"Clustering: {data.get('clustering', 'offline')}")
    generated = data.get("generated_at")
    if generated:
        lines.append(f"Generated: {generated}")
    lines.append("")
    groups = data.get("groups") or []
    if not groups:
        lines.append("_No nuggets to merge_")
        lines.append("")
        return "\n".join(lines)

    best = groups[: _MAX_BEST_OF_SHOW]
    lines.append("## Best of Show")
    lines.append("")
    for i, entry in enumerate(best, 1):
        lines.append(f"### {i}. {entry.get('insight', '').strip()} ({entry.get('total', 0)}/8)")
        lines.append("")
        quote = entry.get("quote", "")
        if quote:
            sources = entry.get("sources") or []
            ts = entry.get("timestamp") or (
                sources[0].get("timestamp") if sources else ""
            )
            lines.append(f'> "{quote}" — [{ts}]' if ts else f'> "{quote}"')
            lines.append("")
        if entry.get("why_it_matters"):
            lines.append(f"*Why:* {entry.get('why_it_matters')}")
            lines.append("")
        sources = entry.get("sources") or []
        labels = [
            _source_label(s.get("show"), s.get("episode"), s.get("timestamp")) for s in sources
        ]
        if labels:
            lines.append(f"*Sources:* {'; '.join(labels)}")
            lines.append("")
    if len(groups) > _MAX_BEST_OF_SHOW:
        lines.append("## All groups")
        lines.append("")
        for i, entry in enumerate(groups[_MAX_BEST_OF_SHOW:], _MAX_BEST_OF_SHOW + 1):
            refs = len(entry.get("sources") or [])
            lines.append(f"### {i}. {entry.get('insight', '').strip()} ({entry.get('total', 0)}/8)")
            lines.append("")
            if refs > 1:
                lines.append(f"*Sources:* {refs} episodes")
                lines.append("")
    return "\n".join(lines)


def _source_label(show: str, episode=None, timestamp: str = "") -> str:
    label = str(show or episode or "").strip()
    if isinstance(episode, int):
        label = f"{label} — {episode}".strip(" —")
    if timestamp:
        label = f"{label} [{timestamp}]"
    return label
    segs = [s for s in transcript.segments if s.text.strip()]
    nuggets: list[dict] = []
    if segs:
        n = min(5, len(segs))
        if n == 1:
            idxs = [0]
        elif n <= 3:
            idxs = list(range(n))
        else:
            idxs = [0, n // 2, n - 1]
        for i in idxs:
            seg = segs[i]
            text = seg.text.strip()
            quote = text
            words = quote.split()
            if len(words) > _MAX_QUOTE_WORDS:
                quote = " ".join(words[: _MAX_QUOTE_WORDS])
            insight = text if len(text) <= 140 else text[:137].rstrip() + "..."
            nuggets.append(
                {
                    "insight": insight,
                    "context": f"{episode.show_title or 'Show'} — {basename}".strip(" —"),
                    "why_it_matters": "Extractive placeholder — replace with LLM nuggets via --backend openrouter|opencode|openai|anthropic|lmstudio.",
                    "quote": quote,
                    "scores": {"T": 2, "S": 1, "E": 2, "A": 1},
                    "total": 6,
                    "tag": "general",
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "timestamp": _format_timestamp(seg.start),
                }
            )
    if not nuggets:
        text = _transcript_text(transcript)
        if text:
            insight = text if len(text) <= 140 else text[:137].rstrip() + "..."
            nuggets.append(
                {
                    "insight": insight,
                    "context": f"{episode.show_title or 'Show'} — {basename}".strip(" —"),
                    "why_it_matters": "No transcript content — nothing extractable.",
                    "quote": "",
                    "scores": {"T": 2, "S": 1, "E": 1, "A": 1},
                    "total": 5,
                    "tag": "eng",
                    "start": 0.0,
                    "end": 0.0,
                    "timestamp": "",
                }
            )
    return nuggets[: _MAX_NUGGETS]


def _nuggets_to_markdown(data: dict) -> str:
    lines: list[str] = []
    title = data.get("title") or "Nuggets"
    lines.append(f"# Nuggets: {title}")
    lines.append("")
    show = data.get("show")
    if show:
        lines.append(f"Show: {show}")
    ep = data.get("episode")
    if ep is not None:
        lines.append(f"Episode: {ep}")
    backend = data.get("backend", "fake")
    model = data.get("model")
    if model:
        lines.append(f"Backend: {backend} ({model})")
    elif backend == "fake":
        lines.append("Backend: fake (offline, no network)")
    else:
        lines.append(f"Backend: {backend}")
    version = data.get("prompt_version")
    if version:
        lines.append(f"Rubric: {version}")
    if data.get("chunked"):
        lines.append("Chunked: yes (episode exceeded input budget)")
    lines.append("")
    nuggets = data.get("nuggets") or []
    if not nuggets:
        lines.append("_No nuggets_")
        lines.append("")
        return "\n".join(lines)
    for i, n in enumerate(nuggets, 1):
        tag = f" [{n.get('tag', 'eng')}]"
        lines.append(f"### {i}. {n.get('insight', '').strip()}{tag} ({n.get('total', 0)}/8)")
        lines.append("")
        quote = n.get("quote", "")
        if quote:
            ts = n.get("timestamp", "")
            if ts:
                lines.append(f"> \"{quote}\" — [{ts}]")
            else:
                lines.append(f"> \"{quote}\"")
            lines.append("")
        if n.get("why_it_matters"):
            lines.append(f"*Why:* {n.get('why_it_matters')}")
            lines.append("")
        if n.get("context"):
            lines.append(f"*Context:* {n.get('context')}")
            lines.append("")
    return "\n".join(lines)


def _write_nugget_files(
    data: dict, *, out_dir: Path, basename: str, formats: tuple[str, ...]
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in _checked_formats(formats):
        if fmt == "json":
            path = out_dir / f"{basename}.nuggets.json"
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            written.append(path)
        else:
            path = out_dir / f"{basename}.nuggets.md"
            path.write_text(_nuggets_to_markdown(data), encoding="utf-8")
            written.append(path)
    return written


def _sidecar_fresh(
    sidecar: Path, *, backend: str, model: str | None, prompt_version: str
) -> bool:
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("prompt_version") == prompt_version
        and data.get("backend") == backend
        and data.get("model") == model
    )


@dataclass
class NuggetsRunResult:
    path: Path
    written: list[Path]
    skipped: bool


@dataclass
class BatchNuggetsResult:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    written: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def extract_nuggets_transcript(
    json_path: Path,
    *,
    out_dir: Path | None = None,
    formats: tuple[str, ...] = ("json",),
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    max_input_chars: int | None = None,
    force: bool = False,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> NuggetsRunResult:
    """Extract nuggets from a transcript JSON file, writing sidecars.

    Skips (returns skipped=True) when a fresh sidecar already exists for the
    current prompt version + backend + model unless ``force`` is set.
    """
    key = _valid_backend(backend)
    resolved_timeout = DEFAULT_NUGGETS_TIMEOUT if timeout is None else timeout
    resolved_temperature = (
        DEFAULT_NUGGETS_TEMPERATURE if temperature is None else temperature
    )
    budget = (
        max_input_chars
        if max_input_chars is not None
        else DEFAULT_NUGGETS_MAX_INPUT_CHARS
    )
    resolved_model = model if model is not None else (get_spec(key).default_model if key != "fake" else None)

    episode, transcript = load_transcript_json(json_path)
    basename = json_path.stem
    dest = out_dir or json_path.parent

    sidecar = dest / f"{basename}.nuggets.json"
    if (
        not force
        and sidecar.exists()
        and _sidecar_fresh(
            sidecar,
            backend=key,
            model=resolved_model,
            prompt_version=NUGGETS_PROMPT_VERSION,
        )
    ):
        return NuggetsRunResult(path=json_path, written=[], skipped=True)

    chunks = _split_chunks(transcript, max_input_chars=budget)
    if key == "fake":
        nuggets = _fake_nuggets(episode, transcript, basename)
        resolved_model = None
        chunked = False
    else:
        provider = build_provider(
            key,
            model=model,
            api_key=api_key,
            base_url=base_url,
            settings_api_key=settings_api_key,
            service=service,
            account=account,
        )
        all_nuggets: list[dict] = []
        for chunk in chunks:
            all_nuggets.extend(
                _extract_chunk_with_retry(
                    provider,
                    episode,
                    chunk,
                    basename,
                    timeout=resolved_timeout,
                    temperature=resolved_temperature,
                )
            )
        nuggets = _merge_nuggets(all_nuggets)
        chunked = len(chunks) > 1

    nuggets = _verify_quotes(nuggets, transcript)

    data = {
        "title": episode.title,
        "show": episode.show_title,
        "episode": episode.episode_num,
        "guid": episode.guid,
        "source": episode.enclosure_url,
        "backend": key,
        "model": resolved_model,
        "prompt_version": NUGGETS_PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunked": chunked,
        "nuggets": nuggets,
    }
    written = _write_nugget_files(data, out_dir=dest, basename=basename, formats=formats)
    return NuggetsRunResult(path=json_path, written=written, skipped=False)


def nuggets_many(
    json_paths: list[Path],
    *,
    out_dir: Path | None = None,
    formats: tuple[str, ...] = ("json",),
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    max_input_chars: int | None = None,
    force: bool = False,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> BatchNuggetsResult:
    """Extract nuggets for many transcript JSON files; continue on per-file errors."""
    result = BatchNuggetsResult()
    for path in json_paths:
        try:
            run = extract_nuggets_transcript(
                path,
                out_dir=out_dir,
                formats=formats,
                backend=backend,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                temperature=temperature,
                max_input_chars=max_input_chars,
                force=force,
                settings_api_key=settings_api_key,
                service=service,
                account=account,
            )
        except (TranscriptJsonError, OSError, ValueError, NuggetsError, ProviderError) as exc:
            result.failed += 1
            result.errors.append((path, str(exc)))
            continue
        if run.skipped:
            result.skipped += 1
        else:
            result.ok += 1
            result.written.extend(run.written)
    return result
