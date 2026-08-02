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
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(path: Path, episode: Episode, transcript: Transcript) -> Path:
    del episode  # metadata not embedded in SRT body
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for i, seg in enumerate(transcript.segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        blocks.append(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{text}\n")
    path.write_text("\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return path
