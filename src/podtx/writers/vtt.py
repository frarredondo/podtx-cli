from __future__ import annotations

from pathlib import Path

from podtx.models import Episode, Transcript


def _ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_vtt(path: Path, episode: Episode, transcript: Transcript) -> Path:
    del episode
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
