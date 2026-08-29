from __future__ import annotations

from pathlib import Path

from podtx.formatting import body_text
from podtx.models import Episode, Transcript


def _yaml_quote(value: str) -> str:
    """Quote a scalar for YAML front matter using double quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return f'"{escaped}"'


def write_md(
    path: Path,
    episode: Episode,
    transcript: Transcript,
    *,
    readable: bool = False,
    cleanup: bool = False,
    correct_names: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = body_text(
        transcript.text,
        transcript.segments,
        readable=readable,
        cleanup=cleanup,
        correct_names=correct_names,
        episode=episode,
    )

    lines: list[str] = ["---"]
    lines.append(f"title: {_yaml_quote(episode.title)}")
    show = episode.show_title if episode.show_title else "Unknown"
    lines.append(f"show: {_yaml_quote(show)}")
    if episode.published_at is not None:
        lines.append(f"date: {episode.published_at.date().isoformat()}")
    if episode.episode_num is not None:
        lines.append(f"episode: {episode.episode_num}")
    lines.append(f"engine: {_yaml_quote(transcript.engine)}")
    lines.append(f"model: {_yaml_quote(transcript.model)}")
    if episode.enclosure_url:
        lines.append(f"source: {_yaml_quote(episode.enclosure_url)}")
    if episode.link:
        lines.append(f"link: {_yaml_quote(episode.link)}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path
