from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from podtx.config import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_OPENCODE_BASE_URL,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
)
from podtx.format_cmd import TranscriptJsonError, load_transcript_json
from podtx.models import Episode, Transcript

_SUMMARY_BACKENDS = {"fake", "openrouter", "opencode", "lmstudio", "local"}
_DEFAULT_FORMATS: tuple[str, ...] = ("json",)
_ALIAS = {"local": "lmstudio"}


class SummarizeError(ValueError):
    """Raised for LLM backend failures (missing key, model, HTTP error, bad JSON)."""


def _normalize_backend(backend: str) -> str:
    b = backend.lower().strip()
    return _ALIAS.get(b, b)


def _default_model(backend: str) -> str | None:
    b = _normalize_backend(backend)
    if b == "openrouter":
        return DEFAULT_OPENROUTER_MODEL
    if b == "opencode":
        return DEFAULT_OPENCODE_MODEL
    return None


def _default_base_url(backend: str) -> str | None:
    b = _normalize_backend(backend)
    if b == "openrouter":
        return DEFAULT_OPENROUTER_BASE_URL
    if b == "opencode":
        return DEFAULT_OPENCODE_BASE_URL
    if b == "lmstudio":
        return DEFAULT_LMSTUDIO_BASE_URL
    return None


def _split_sentences(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _format_timestamp(seconds: float) -> str:
    total = int(float(seconds))
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _overview_from_sentences(sentences: list[str], text: str) -> str:
    if not sentences:
        return text[:500].strip()
    ov = " ".join(sentences[:2]).strip()
    if len(ov) > 600:
        ov = ov[:600].rstrip() + "…"
    return ov


def _truncate_text(text: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or len(text) <= max_chars:
        return text, False
    # Try sentence boundary
    cut = text[:max_chars]
    # Find last sentence end
    last_dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last_dot > max_chars * 0.5:
        return cut[: last_dot + 1].strip(), True
    return cut.rstrip() + "…", True


def _resolve_api_key(
    backend: str,
    api_key: str | None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> str | None:
    if api_key:
        return api_key
    if settings_api_key:
        return settings_api_key
    # Try keychain if service/account provided
    if service and account:
        try:
            from podtx.keychain import get_api_key

            val = get_api_key(service, account)
            if val:
                return val
        except Exception:  # pragma: no cover - best effort
            pass  # pragma: no cover
    # Provider-specific service defaults
    b = _normalize_backend(backend)
    defaults: dict[str, tuple[str, str]] = {
        "openrouter": ("podtx-openrouter", "api-key"),
        "opencode": ("podtx-opencode", "api-key"),
    }
    if b in defaults:
        svc, acct = defaults[b]
        # Only try if no explicit service/account to avoid duplicate
        if not (service and account):
            try:
                from podtx.keychain import get_api_key as _g

                val = _g(svc, acct)
                if val:
                    return val
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    return None


def _build_prompt(episode: Episode, transcript_text: str) -> list[dict[str, str]]:
    title = episode.title or "Untitled"
    show = episode.show_title or ""
    header = f"Podcast: {show} — Episode: {title}" if show else f"Episode: {title}"
    system = (
        "You are a podcast summarizer. Given a transcript, return strict JSON "
        "with keys: overview (2-3 sentence summary), key_points (array of 3-5 concise bullet strings), "
        "quotes (array of 0-3 objects with 'text' exact verbatim substring from transcript and optional 'start' seconds). "
        "Do not invent timestamps. Keep quotes short."
    )
    user = f"{header}\n\nTranscript:\n{transcript_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json_content(raw: str) -> dict:
    # Try direct JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to find first {...} block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise SummarizeError(f"LLM did not return valid JSON: {raw[:500]!r}")


def _validate_llm_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise SummarizeError("LLM payload must be a JSON object")
    overview = payload.get("overview")
    if not isinstance(overview, str) or not overview.strip():
        raise SummarizeError("LLM payload missing 'overview' string")
    key_points = payload.get("key_points")
    if not isinstance(key_points, list) or not key_points:  # pragma: no cover - strict
        raise SummarizeError("LLM payload missing 'key_points' array")  # pragma: no cover
    # Filter and limit
    kps = [str(k).strip() for k in key_points if str(k).strip()]
    if not kps:
        raise SummarizeError("LLM payload 'key_points' empty")
    kps = kps[:5]
    quotes_raw = payload.get("quotes", [])
    if quotes_raw is None:
        quotes_raw = []
    if not isinstance(quotes_raw, list):
        quotes_raw = []  # pragma: no cover
    quotes: list[dict] = []
    for q in quotes_raw[:3]:
        if isinstance(q, dict):
            txt = str(q.get("text", "")).strip()
            if not txt:
                continue
            start = q.get("start")
            try:
                start_f = float(start) if start is not None else 0.0
            except (TypeError, ValueError):
                start_f = 0.0
            quotes.append({"text": txt, "start": start_f})
        elif isinstance(q, str) and q.strip():  # pragma: no cover - lenient
            quotes.append({"text": q.strip(), "start": 0.0})  # pragma: no cover
    return {"overview": overview.strip(), "key_points": kps, "quotes": quotes}


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 60.0,
    temperature: float = 0.3,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=body)
    except httpx.RequestError as exc:
        raise SummarizeError(f"LLM request failed: {exc}") from exc
    if resp.status_code != 200:
        try:
            err_body = resp.text[:500]
        except Exception:  # pragma: no cover
            err_body = ""  # pragma: no cover
        # Hint for opencode Go vs direct Meta API mixup
        hint = ""
        if resp.status_code == 401 and "opencode.ai/zen/go" in base_url:
            hint = " (Go key invalid? check opencode.ai/zen/go dashboard, or try --base-url https://api.meta.ai/v1 for direct Meta API)"
        elif resp.status_code == 401 and "api.meta.ai" in base_url:
            hint = " (Meta API key invalid? check https://api.meta.ai console, or try --base-url https://opencode.ai/zen/go/v1 for Go subscription)"
        raise SummarizeError(f"LLM request failed ({resp.status_code}): {err_body}{hint}")
    try:
        data = resp.json()
    except Exception as exc:
        # Include status code and hint about base-url
        body_preview = resp.text[:500].strip()
        hint = " (check --base-url, expected OpenAI-compatible /chat/completions)" if "Not Found" in body_preview else ""
        raise SummarizeError(f"LLM returned invalid JSON response (HTTP {resp.status_code}): {body_preview!r}{hint}") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SummarizeError(f"LLM response missing choices/message/content: {data!r}") from exc
    if not isinstance(content, str):
        content = json.dumps(content)
    return content


def _summarize_with_llm(
    episode: Episode,
    transcript: Transcript,
    *,
    backend: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
    temperature: float,
    max_input_chars: int | None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> tuple[dict, bool]:
    b = _normalize_backend(backend)
    # Resolve model
    resolved_model = model or _default_model(b)
    if b == "lmstudio" and not resolved_model:
        raise SummarizeError("lmstudio/local backend requires --model (no default)")
    if not resolved_model:
        # openrouter/opencode should have defaults but handle missing
        raise SummarizeError(f"{b} backend requires --model")
    # Resolve base_url
    resolved_base = base_url or _default_base_url(b)
    if not resolved_base:
        raise SummarizeError(f"{b} backend requires --base-url")
    # Resolve api_key for cloud backends
    needs_key = b in {"openrouter", "opencode"}
    resolved_key = _resolve_api_key(b, api_key, settings_api_key, service, account)
    if needs_key and not resolved_key:
        hint = f"podtx auth set --backend {b}" if b in {"openrouter", "opencode"} else "provide --api-key"
        raise SummarizeError(f"{b} backend requires API key (--api-key, env OPENROUTER_API_KEY/OPENCODE_API_KEY, or {hint})")

    # Prepare transcript text (full by default, optional truncation)
    text = transcript.text.strip()
    if not text and transcript.segments:
        text = " ".join(s.text.strip() for s in transcript.segments if s.text.strip())
    text, truncated = _truncate_text(text, max_input_chars)

    messages = _build_prompt(episode, text)
    raw_content = _call_openai_compatible(
        base_url=resolved_base,
        api_key=resolved_key,
        model=resolved_model,
        messages=messages,
        timeout=timeout,
        temperature=temperature,
    )
    payload = _extract_json_content(raw_content)
    validated = _validate_llm_payload(payload)
    return validated, truncated


def _build_fake_summary(episode: Episode, transcript: Transcript) -> dict:
    text = transcript.text.strip()
    if not text and transcript.segments:
        text = " ".join(s.text.strip() for s in transcript.segments if s.text.strip())
    sentences = _split_sentences(text)
    overview = _overview_from_sentences(sentences, text)
    if not sentences:
        key_points = [text[:200].strip()] if text.strip() else []
    elif len(sentences) >= 3:
        key_points = sentences[2:5]
    else:
        key_points = sentences[:3]
    key_points = [k.strip() for k in key_points if k.strip()]
    if not key_points and text.strip():  # pragma: no cover
        key_points = [text.strip()[:200]]  # pragma: no cover
    quotes: list[dict] = []
    segs = [s for s in transcript.segments if s.text.strip()]
    if segs:
        if len(segs) == 1:
            idxs = [0]
        elif len(segs) == 2:
            idxs = [0, 1]
        else:
            idxs = [0, len(segs) // 2, len(segs) - 1]
        for i in idxs:
            seg = segs[i]
            quotes.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg.text.strip(),
                    "timestamp": _format_timestamp(seg.start),
                }
            )
    return {"overview": overview, "key_points": key_points, "quotes": quotes, "truncated": False}


@dataclass
class BatchSummaryResult:
    ok: int = 0
    failed: int = 0
    written: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def build_summary(
    episode: Episode,
    transcript: Transcript,
    *,
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    temperature: float = 0.3,
    max_input_chars: int | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> dict:
    """Build summary via backend.

    Returns dict with overview, key_points, quotes (timestamped), backend, model etc.
    """
    b = _normalize_backend(backend)
    if b not in _SUMMARY_BACKENDS and backend.lower().strip() not in _ALIAS:
        # Check normalized not in set
        if b not in _SUMMARY_BACKENDS:
            raise ValueError(f"Unknown summary backend {backend!r}. Choose from: {', '.join(sorted(_SUMMARY_BACKENDS))}")
    # Also validate original includes alias? Already handled.
    if b not in _SUMMARY_BACKENDS:
        raise ValueError(f"Unknown summary backend {backend!r}. Choose from: {', '.join(sorted(_SUMMARY_BACKENDS))}")

    if b == "fake":
        inner = _build_fake_summary(episode, transcript)
        summary = {
            "title": episode.title,
            "show": episode.show_title,
            "episode": episode.episode_num,
            "guid": episode.guid,
            "source": episode.enclosure_url,
            "overview": inner["overview"],
            "key_points": inner["key_points"],
            "quotes": inner["quotes"],
            "backend": b,
            "model": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return summary

    # LLM backends
    validated, truncated = _summarize_with_llm(
        episode,
        transcript,
        backend=b,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=temperature,
        max_input_chars=max_input_chars,
        settings_api_key=settings_api_key,
        service=service,
        account=account,
    )
    # Re-anchor quotes to transcript segments for timestamps
    # LLM quotes may have start or not; we map to timestamp via _format_timestamp
    raw_quotes = validated["quotes"]
    anchored: list[dict] = []
    segs = transcript.segments
    for q in raw_quotes:
        txt = q["text"].strip()
        start = float(q.get("start", 0.0))
        # Try to find closest segment containing text for better timestamp?
        # Simple: if start is 0 and we have segs, try to find text in segments
        if start == 0.0 and segs:
            found = None
            for seg in segs:
                if txt.lower() in seg.text.lower() or seg.text.lower() in txt.lower():
                    found = seg
                    break
            if found:
                start = float(found.start)
                end = float(found.end)
            else:
                end = start + 5.0
        else:
            end = start + 5.0
            # Try to find matching segment for end
            for seg in segs:
                if abs(float(seg.start) - start) < 0.5:
                    end = float(seg.end)
                    break
        anchored.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": txt,
                "timestamp": _format_timestamp(start),
            }
        )
    # Fallback to fake quotes if LLM returned none but we have segments? Keep empty.
    summary = {
        "title": episode.title,
        "show": episode.show_title,
        "episode": episode.episode_num,
        "guid": episode.guid,
        "source": episode.enclosure_url,
        "overview": validated["overview"],
        "key_points": validated["key_points"],
        "quotes": anchored,
        "backend": b,
        "model": model or _default_model(b),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truncated": truncated,
    }
    return summary


def _summary_to_markdown(summary: dict) -> str:
    lines: list[str] = []
    title = summary.get("title") or "Summary"
    lines.append(f"# Summary: {title}")
    lines.append("")
    show = summary.get("show")
    if show:
        lines.append(f"Show: {show}")
    ep = summary.get("episode")
    if ep is not None:
        lines.append(f"Episode: {ep}")
    backend = summary.get("backend", "fake")
    model = summary.get("model")
    if model:
        lines.append(f"Backend: {backend} ({model})")
    else:
        if backend == "fake":
            lines.append(f"Backend: {backend} (offline, no network)")
        else:
            lines.append(f"Backend: {backend}")
    if summary.get("truncated"):
        lines.append("")
        lines.append("> _Transcript truncated to fit context window_")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(summary.get("overview", "").strip() or "_No overview_")
    lines.append("")
    lines.append("## Key Points")
    lines.append("")
    kps = summary.get("key_points") or []
    if kps:
        for kp in kps:
            lines.append(f"- {kp.strip()}")
    else:
        lines.append("- _No key points_")
    lines.append("")
    lines.append("## Quotes")
    lines.append("")
    quotes = summary.get("quotes") or []
    if quotes:
        for q in quotes:
            ts = q.get("timestamp", "")
            txt = q.get("text", "").strip()
            start = q.get("start", "")
            if ts:
                lines.append(f"- [{ts}] {txt} ({start}s)")
            else:
                lines.append(f"- {txt}")
    else:
        lines.append("_No quotes_")
    lines.append("")
    return "\n".join(lines)


def _write_summary_files(
    summary: dict,
    *,
    out_dir: Path,
    basename: str,
    formats: tuple[str, ...],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        key = fmt.lower().strip()
        if key not in {"json", "md"}:
            raise ValueError(f"Unsupported summary format {fmt!r}. Choose from: json, md")
        if key == "json":
            path = out_dir / f"{basename}.summary.json"
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(path)
        else:
            path = out_dir / f"{basename}.summary.md"
            path.write_text(_summary_to_markdown(summary), encoding="utf-8")
            written.append(path)
    return written


def summarize_transcript(
    json_path: Path,
    *,
    out_dir: Path | None = None,
    formats: tuple[str, ...] = _DEFAULT_FORMATS,
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    temperature: float = 0.3,
    max_input_chars: int | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> list[Path]:
    """Summarize a single transcript JSON file without ASR (reads existing JSON).

    Returns list of written sidecar paths (.summary.json / .summary.md).
    """
    episode, transcript = load_transcript_json(json_path)
    summary = build_summary(
        episode,
        transcript,
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        temperature=temperature,
        max_input_chars=max_input_chars,
        settings_api_key=settings_api_key,
        service=service,
        account=account,
    )
    dest = out_dir or json_path.parent
    basename = json_path.stem
    return _write_summary_files(summary, out_dir=dest, basename=basename, formats=formats)


def summarize_many(
    json_paths: list[Path],
    *,
    out_dir: Path | None = None,
    formats: tuple[str, ...] = _DEFAULT_FORMATS,
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    temperature: float = 0.3,
    max_input_chars: int | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> BatchSummaryResult:
    """Summarize many transcript JSON files; continue on per-file errors."""
    result = BatchSummaryResult()
    for path in json_paths:
        try:
            written = summarize_transcript(
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
                settings_api_key=settings_api_key,
                service=service,
                account=account,
            )
        except (TranscriptJsonError, OSError, ValueError, SummarizeError) as exc:
            result.failed += 1
            result.errors.append((path, str(exc)))
            continue
        result.ok += 1
        result.written.extend(written)
    return result
