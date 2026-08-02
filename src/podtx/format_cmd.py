from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs


class TranscriptJsonError(ValueError):
    pass


def load_transcript_json(path: Path) -> tuple[Episode, Transcript]:
    """Load episode + transcript from a podtx JSON sidecar."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptJsonError(f"Could not read transcript JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise TranscriptJsonError(f"Invalid transcript JSON (expected object): {path}")

    published_at = None
    if payload.get("date"):
        try:
            published_at = datetime.fromisoformat(str(payload["date"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise TranscriptJsonError(f"Invalid date in {path}: {payload.get('date')}") from exc

    segments: list[Segment] = []
    for raw in payload.get("segments") or []:
        segments.append(
            Segment(
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")).strip(),
            )
        )

    # Prefer joining segment text for archive fidelity when present
    text = str(payload.get("text") or "").strip()
    if segments and not text:
        text = " ".join(s.text for s in segments if s.text)

    episode = Episode(
        guid=str(payload.get("guid") or path.stem),
        title=str(payload.get("title") or path.stem),
        enclosure_url=str(payload.get("source") or ""),
        published_at=published_at,
        episode_num=payload.get("episode"),
        link=payload.get("link"),
        show_title=payload.get("show"),
    )
    transcript = Transcript(
        text=text,
        segments=segments,
        language=str(payload.get("language") or "en"),
        model=str(payload.get("model") or "unknown"),
        engine=str(payload.get("engine") or "unknown"),
    )
    return episode, transcript


def reformat_transcript(
    json_path: Path,
    *,
    out_dir: Path | None = None,
    readable: bool = False,
    cleanup: bool = False,
    formats: tuple[str, ...] = ("txt", "json"),
) -> list[Path]:
    """Re-write outputs from an existing transcript JSON without re-running ASR."""
    episode, transcript = load_transcript_json(json_path)
    dest = out_dir or json_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    basename = json_path.stem
    return write_outputs(
        out_dir=dest,
        basename=basename,
        episode=episode,
        transcript=transcript,
        formats=formats,
        readable=readable,
        cleanup=cleanup,
    )
