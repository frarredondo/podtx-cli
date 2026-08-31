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
    cut = text[:max_chars]
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
    if service and account:
        try:
            from podtx.keychain import get_api_key

            val = get_api_key(service, account)
            if val:
                return val
        except Exception:  # pragma: no cover - best effort
            pass  # pragma: no cover
    b = _normalize_backend(backend)
    defaults: dict[str, tuple[str, str]] = {
        "openrouter": ("podtx-openrouter", "api-key"),
        "opencode": ("podtx-opencode", "api-key"),
    }
    if b in defaults:
        svc, acct = defaults[b]
        if not (service and account):
            try:
                from podtx.keychain import get_api_key as _g

                val = _g(svc, acct)
                if val:
                    return val
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
    return None


def _duration_minutes(transcript: Transcript) -> float:
    if transcript.segments:
        try:
            return float(transcript.segments[-1].end) / 60.0
        except Exception:
            pass
    # fallback via word count ~150 wpm
    words = len(transcript.text.split())
    return words / 150.0


def _nugget_target_for_duration(minutes: float) -> str:
    if minutes < 20:
        return "1-3 nuggets (short ~10-15min episode)"
    if minutes > 60:
        return "3-7 nuggets (long 60+ min episode)"
    return "3-5 nuggets (standard episode)"


def _build_prompt(episode: Episode, transcript_text: str, basename: str, minutes: float) -> list[dict[str, str]]:
    title = episode.title or "Untitled"
    show = episode.show_title or ""
    target = _nugget_target_for_duration(minutes)
    header = f"Podcast: {show} — Episode: {title}" if show else f"Episode: {title}"
    # include guest hint from title/show, and filename for context field
    context_hint = f"{show} — {basename}" if show else basename
    system = (
        f"You are extracting durable insights for software engineers from podcast transcripts. Extract {target} using criteria: "
        "memorable quote, counterintuitive insight, hard-won lesson, mental model, or principle valuable to a software engineer / technical audience. "
        "Each nugget is JSON with: insight (1-2 sentences, punchy, self-contained), context (guest + episode filename, e.g. \""
        + context_hint
        + "\"), why_it_matters (1 sentence, why a software engineer should care), quote (short verbatim quote <30 words from transcript if present, else \"\" — exact substring, don't invent, <30 words). "
        "Focus on timeless wisdom, not ephemeral news or self-promo. Prioritize most surprising / actionable. "
        "Also return top5_best: array of up to 5 indices (0-based) of the best nuggets ranked most valuable first (\"Best of Show\"). "
        "Read the full transcript, don't hallucinate. Return strict JSON: {\"nuggets\": [ {insight, context, why_it_matters, quote} ], \"top5_best\": [indices] }"
    )
    user = f"{header}\nEpisode file: {basename}\nDuration: {minutes:.1f} min\n\nTranscript:\n{transcript_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_json_content(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
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
    # New nuggets format
    if "nuggets" in payload:
        nuggets = payload.get("nuggets")
        if not isinstance(nuggets, list) or not nuggets:
            raise SummarizeError("LLM payload missing 'nuggets' array")
        cleaned: list[dict] = []
        for n in nuggets[:7]:
            if not isinstance(n, dict):
                continue
            insight = str(n.get("insight", "")).strip()
            if not insight:
                continue
            context = str(n.get("context", "")).strip()
            why = str(n.get("why_it_matters", "")).strip() or str(n.get("why", "")).strip()
            quote = str(n.get("quote", "")).strip()
            # enforce <30 words for quote
            if quote and len(quote.split()) > 30:
                quote = " ".join(quote.split()[:30])
            cleaned.append({
                "insight": insight,
                "context": context,
                "why_it_matters": why,
                "quote": quote,
            })
        if not cleaned:
            raise SummarizeError("LLM payload 'nuggets' empty after validation")
        # top5_best optional
        top5 = payload.get("top5_best") or payload.get("top5") or []
        if not isinstance(top5, list):
            top5 = []
        # filter to valid indices
        top5_clean = []
        for idx in top5[:5]:
            try:
                i = int(idx)
                if 0 <= i < len(cleaned) and i not in top5_clean:
                    top5_clean.append(i)
            except Exception:
                continue
        if not top5_clean:
            # default ranking: first up to 5
            top5_clean = list(range(min(5, len(cleaned))))
        return {"nuggets": cleaned, "top5_best": top5_clean}
    # Legacy fallback: overview/key_points/quotes -> convert to nuggets for backwards compat
    overview = payload.get("overview")
    if isinstance(overview, str) and overview.strip():
        # Convert legacy to nuggets
        kps = payload.get("key_points") or []
        quotes = payload.get("quotes") or []
        nuggets = []
        for i, kp in enumerate(kps[:5]):
            if not str(kp).strip():
                continue
            q = ""
            if isinstance(quotes, list) and i < len(quotes):
                qi = quotes[i]
                if isinstance(qi, dict):
                    q = str(qi.get("text", "")).strip()
                elif isinstance(qi, str):
                    q = qi.strip()
            nuggets.append({
                "insight": str(kp).strip(),
                "context": "",
                "why_it_matters": "",
                "quote": " ".join(q.split()[:30]) if q else "",
            })
        if not nuggets:
            # fallback single nugget from overview
            nuggets = [{"insight": overview.strip(), "context": "", "why_it_matters": "", "quote": ""}]
        return {"nuggets": nuggets, "top5_best": list(range(min(5, len(nuggets))))}
    raise SummarizeError("LLM payload missing 'nuggets' (and no legacy overview)")


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
        hint = ""
        if resp.status_code == 401 and "opencode.ai/zen/go" in base_url:
            hint = " (Go key invalid? check opencode.ai/zen/go dashboard, or try --base-url https://api.meta.ai/v1 for direct Meta API)"
        elif resp.status_code == 401 and "api.meta.ai" in base_url:
            hint = " (Meta API key invalid? check https://api.meta.ai console, or try --base-url https://opencode.ai/zen/go/v1 for Go subscription)"
        elif resp.status_code == 500 and "opencode.ai/zen/go" in base_url:
            hint = " (Go 500 often = model not hosted on Go. Go hosts kimi-*/glm-*/deepseek-*, not Muse Spark. Try --backend openrouter --model meta/muse-spark-1.2-contributor or --base-url https://api.meta.ai/v1 --model muse-spark-1.2-contributor with a Meta key)"
        raise SummarizeError(f"LLM request failed ({resp.status_code}): {err_body}{hint}")
    try:
        data = resp.json()
    except Exception as exc:
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
    basename: str,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> tuple[dict, bool]:
    b = _normalize_backend(backend)
    resolved_model = model or _default_model(b)
    if b == "lmstudio" and not resolved_model:
        raise SummarizeError("lmstudio/local backend requires --model (no default)")
    if not resolved_model:
        raise SummarizeError(f"{b} backend requires --model")
    resolved_base = base_url or _default_base_url(b)
    if not resolved_base:
        raise SummarizeError(f"{b} backend requires --base-url")
    needs_key = b in {"openrouter", "opencode"}
    resolved_key = _resolve_api_key(b, api_key, settings_api_key, service, account)
    if needs_key and not resolved_key:
        hint = f"podtx auth set --backend {b}" if b in {"openrouter", "opencode"} else "provide --api-key"
        raise SummarizeError(f"{b} backend requires API key (--api-key, env OPENROUTER_API_KEY/OPENCODE_API_KEY, or {hint})")

    text = transcript.text.strip()
    if not text and transcript.segments:
        text = " ".join(s.text.strip() for s in transcript.segments if s.text.strip())
    text, truncated = _truncate_text(text, max_input_chars)

    minutes = _duration_minutes(transcript)
    messages = _build_prompt(episode, text, basename, minutes)
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


def _build_fake_summary(episode: Episode, transcript: Transcript, basename: str = "") -> dict:
    # Keep legacy extractive overview/key_points/quotes for test compat, plus nuggets
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
            quotes.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip(),
                "timestamp": _format_timestamp(seg.start),
            })
    # Derive nuggets from key_points for fake (extractive placeholder)
    nuggets: list[dict] = []
    for i, kp in enumerate(key_points[:3]):
        ctx = f"{episode.show_title or 'Show'} — {basename}" if basename else (episode.show_title or "")
        q = quotes[i]["text"] if i < len(quotes) else ""
        if q and len(q.split()) > 30:
            q = " ".join(q.split()[:30])
        nuggets.append({
            "insight": kp,
            "context": ctx,
            "why_it_matters": "Extractive placeholder — replace with LLM nuggets via --backend openrouter|opencode|lmstudio.",
            "quote": q,
            "start": quotes[i]["start"] if i < len(quotes) else 0.0,
            "end": quotes[i]["end"] if i < len(quotes) else 5.0,
            "timestamp": quotes[i]["timestamp"] if i < len(quotes) else "00:00",
        })
    if not nuggets:
        # fallback single nugget from overview/text
        insight = overview.strip() if overview.strip() else (text[:200].strip() or "No transcript")
        nuggets = [{
            "insight": insight,
            "context": basename,
            "why_it_matters": "No transcript content.",
            "quote": quotes[0]["text"] if quotes else "",
            "start": quotes[0]["start"] if quotes else 0.0,
            "end": quotes[0]["end"] if quotes else 5.0,
            "timestamp": quotes[0]["timestamp"] if quotes else "00:00",
        }]
    return {"overview": overview, "key_points": key_points, "quotes": quotes, "nuggets": nuggets, "top5_best": list(range(min(5, len(nuggets)))), "truncated": False}


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
    basename: str = "",
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> dict:
    """Build summary via backend. Returns dict with nuggets (and legacy overview/key_points/quotes for compat)."""
    b = _normalize_backend(backend)
    if b not in _SUMMARY_BACKENDS and backend.lower().strip() not in _ALIAS:
        if b not in _SUMMARY_BACKENDS:  # pragma: no cover - always true here (defensive)
            raise ValueError(f"Unknown summary backend {backend!r}. Choose from: {', '.join(sorted(_SUMMARY_BACKENDS))}")
    if b not in _SUMMARY_BACKENDS:  # pragma: no cover - aliases always expand into backends
        raise ValueError(f"Unknown summary backend {backend!r}. Choose from: {', '.join(sorted(_SUMMARY_BACKENDS))}")  # pragma: no cover

    if b == "fake":
        inner = _build_fake_summary(episode, transcript, basename=basename)
        summary = {
            "title": episode.title,
            "show": episode.show_title,
            "episode": episode.episode_num,
            "guid": episode.guid,
            "source": episode.enclosure_url,
            "overview": inner["overview"],
            "key_points": inner["key_points"],
            "quotes": inner["quotes"],
            "nuggets": inner["nuggets"],
            "top5_best": inner["top5_best"],
            "backend": b,
            "model": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "truncated": False,
        }
        return summary

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
        basename=basename,
        settings_api_key=settings_api_key,
        service=service,
        account=account,
    )
    # Anchor nugget quotes to transcript segments for timestamps
    nuggets = validated["nuggets"]
    top5 = validated["top5_best"]
    anchored: list[dict] = []
    segs = transcript.segments
    for n in nuggets:
        quote = n.get("quote", "").strip()
        if not quote:
            anchored.append({**n, "start": 0.0, "end": 0.0, "timestamp": ""})
            continue
        # Try to find verbatim in segments
        found_start = None
        found_end = None
        for seg in segs:
            if quote.lower() in seg.text.lower() or seg.text.lower() in quote.lower():
                found_start = float(seg.start)
                found_end = float(seg.end)
                break
        # Also check if LLM provided start (not in new schema, but handle)
        if found_start is None:
            # fallback: try to find any segment containing a chunk of quote
            words = quote.split()
            if len(words) >= 4:
                chunk = " ".join(words[:4]).lower()
                for seg in segs:
                    if chunk in seg.text.lower():
                        found_start = float(seg.start)
                        found_end = float(seg.end)
                        break
        start = found_start if found_start is not None else 0.0
        end = found_end if found_end is not None else (start + 5.0 if start else 0.0)
        # enforce <30 words already done in validation
        anchored.append({
            "insight": n["insight"],
            "context": n["context"] or f"{episode.show_title or ''} — {basename}".strip(" —"),
            "why_it_matters": n["why_it_matters"],
            "quote": quote,
            "start": round(start, 3),
            "end": round(end, 3),
            "timestamp": _format_timestamp(start) if start else "",
        })
    # Legacy compat
    overview = anchored[0]["insight"] if anchored else ""
    key_points = [n["insight"] for n in anchored]
    quotes = [{"start": n["start"], "end": n["end"], "text": n["quote"], "timestamp": n["timestamp"]} for n in anchored if n["quote"]]
    summary = {
        "title": episode.title,
        "show": episode.show_title,
        "episode": episode.episode_num,
        "guid": episode.guid,
        "source": episode.enclosure_url,
        "overview": overview,
        "key_points": key_points,
        "quotes": quotes,
        "nuggets": anchored,
        "top5_best": top5,
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
    # Best of Show if present
    top5 = summary.get("top5_best") or []
    nuggets = summary.get("nuggets") or []
    # Fallback to legacy
    if not nuggets and summary.get("overview"):
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

    # New nuggets markdown
    lines.append("")
    if top5 and nuggets:
        lines.append("## Best of Show (Top 5)")
        lines.append("")
        for rank, idx in enumerate(top5[:5], 1):
            if 0 <= idx < len(nuggets):
                n = nuggets[idx]
                lines.append(f"{rank}. **{n.get('insight','').strip()}**")
                if n.get("quote"):
                    ts = n.get("timestamp", "")
                    if ts:
                        lines.append(f"   — \"{n.get('quote','').strip()}\" [{ts}]")
                    else:
                        lines.append(f"   — \"{n.get('quote','').strip()}\"")
                if n.get("why_it_matters"):
                    lines.append(f"   *Why:* {n.get('why_it_matters','').strip()}")
                if n.get("context"):
                    lines.append(f"   *Context:* {n.get('context','').strip()}")
        lines.append("")
    lines.append("## Nuggets")
    lines.append("")
    if nuggets:
        for i, n in enumerate(nuggets):
            lines.append(f"### {i+1}. {n.get('insight','').strip()}")
            lines.append("")
            if n.get("context"):
                lines.append(f"*Context:* {n.get('context','').strip()}")
                lines.append("")
            if n.get("why_it_matters"):
                lines.append(f"*Why it matters:* {n.get('why_it_matters','').strip()}")
                lines.append("")
            if n.get("quote"):
                ts = n.get("timestamp", "")
                if ts:
                    lines.append(f"> \"{n.get('quote','').strip()}\" — [{ts}]")
                else:
                    lines.append(f"> \"{n.get('quote','').strip()}\"")
                lines.append("")
    else:
        lines.append("_No nuggets_")
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
        basename=json_path.stem,
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
