from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from podtx.format_cmd import TranscriptJsonError, load_transcript_json
from podtx.models import Episode, Transcript

_SUMMARY_BACKENDS = {"fake"}
_DEFAULT_FORMATS: tuple[str, ...] = ("json",)


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
    # First paragraph overview: first 2 sentences, stable extractive.
    ov = " ".join(sentences[:2]).strip()
    if len(ov) > 600:
        ov = ov[:600].rstrip() + "…"
    return ov


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
) -> dict:
    """Build extractive summary (offline fake backend, no network).

    Returns dict with overview, key_points, quotes (timestamped).
    Pluggable backend seam: currently only "fake" is supported.
    """
    if backend not in _SUMMARY_BACKENDS:
        raise ValueError(f"Unknown summary backend {backend!r}. Choose from: {', '.join(sorted(_SUMMARY_BACKENDS))}")

    text = transcript.text.strip()
    if not text and transcript.segments:
        text = " ".join(s.text.strip() for s in transcript.segments if s.text.strip())

    sentences = _split_sentences(text)
    overview = _overview_from_sentences(sentences, text)

    # Key points: next sentences after overview, up to 3; stable extractive
    if not sentences:
        key_points = [text[:200].strip()] if text.strip() else []
    elif len(sentences) >= 3:
        key_points = sentences[2:5]
    else:
        key_points = sentences[:3]

    # Ensure 1-3 key points, no empties
    key_points = [k.strip() for k in key_points if k.strip()]
    if not key_points and text.strip():  # pragma: no cover
        key_points = [text.strip()[:200]]  # pragma: no cover

    # Quotes: timestamped segments, extractive (first, middle, last)
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

    summary = {
        "title": episode.title,
        "show": episode.show_title,
        "episode": episode.episode_num,
        "guid": episode.guid,
        "source": episode.enclosure_url,
        "overview": overview,
        "key_points": key_points,
        "quotes": quotes,
        "backend": backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
    lines.append(f"Backend: {summary.get('backend', 'fake')} (offline, no network)")
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
) -> list[Path]:
    """Summarize a single transcript JSON file without ASR (reads existing JSON).

    Returns list of written sidecar paths (.summary.json / .summary.md).
    """
    episode, transcript = load_transcript_json(json_path)
    summary = build_summary(episode, transcript, backend=backend)
    dest = out_dir or json_path.parent
    basename = json_path.stem
    return _write_summary_files(summary, out_dir=dest, basename=basename, formats=formats)


def summarize_many(
    json_paths: list[Path],
    *,
    out_dir: Path | None = None,
    formats: tuple[str, ...] = _DEFAULT_FORMATS,
    backend: str = "fake",
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
            )
        except (TranscriptJsonError, OSError, ValueError) as exc:
            result.failed += 1
            result.errors.append((path, str(exc)))
            continue
        result.ok += 1
        result.written.extend(written)
    return result
