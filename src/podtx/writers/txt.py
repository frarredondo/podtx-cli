from __future__ import annotations

from pathlib import Path

from podtx.formatting import body_text
from podtx.models import Episode, Transcript


def _header_lines(episode: Episode, transcript: Transcript) -> list[str]:
    lines = [
        f"Title: {episode.title}",
        f"Show: {episode.show_title or 'Unknown'}",
        f"Date: {episode.published_at.date().isoformat() if episode.published_at else 'unknown'}",
        f"Engine: {transcript.engine}",
        f"Model: {transcript.model}",
        f"Source: {episode.enclosure_url}",
    ]
    if episode.episode_num is not None:
        lines.insert(2, f"Episode: {episode.episode_num}")
    return lines


def write_txt(
    path: Path,
    episode: Episode,
    transcript: Transcript,
    *,
    readable: bool = False,
    cleanup: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(_header_lines(episode, transcript))
    body = body_text(
        transcript.text,
        transcript.segments,
        readable=readable,
        cleanup=cleanup,
    )
    content = f"{header}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return path
