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
    Provider,
    ProviderError,
    build_provider,
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
    from podtx.providers import available_providers

    known = {"fake", *available_providers()}
    if key not in known:
        raise NuggetsError(
            f"Unknown backend {backend!r}. Choose from: {', '.join(sorted(known))}"
        )
    return key


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


def _verify_quotes(nuggets: list[dict], transcript: Transcript) -> list[dict]:
    """Mechanical anti-hallucination check + segment timestamps."""
    full = _normalize(
        transcript.text or " ".join(s.text for s in transcript.segments if s.text.strip())
    )
    segs = [s for s in transcript.segments if s.text.strip()]
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
        start = end = None
        for seg in segs:
            ns = _normalize(seg.text)
            if ns and (nq in ns or ns in nq):
                start, end = float(seg.start), float(seg.end)
                break
        if start is None and len(nq.split()) >= 4:
            chunk = " ".join(nq.split()[:4])
            for seg in segs:
                if chunk in _normalize(seg.text):
                    start, end = float(seg.start), float(seg.end)
                    break
        if start is not None:
            verified.append(
                {
                    **n,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "timestamp": _format_timestamp(start),
                }
            )
        else:
            verified.append(_demote_quote(n))
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


def _split_chunks(transcript: Transcript, *, max_input_chars: int) -> list[str]:
    text = _transcript_text(transcript)
    if len(text) <= max_input_chars or not transcript.segments:
        return [text]
    overlap = max(1, int(max_input_chars * _OVERLAP_FRACTION))
    chunks = _chunk_segments(
        transcript, budget_chars=max_input_chars, overlap_chars=overlap
    )
    return [
        " ".join(transcript.segments[i].text.strip() for i in idx)
        for idx in chunks
    ]


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
    for fmt in formats:
        key = fmt.lower().strip()
        if key not in {"json", "md"}:
            raise NuggetsError(f"Unsupported format {fmt!r}. Choose from: json, md")
        if key == "json":
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