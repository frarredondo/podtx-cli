from __future__ import annotations

import hashlib
import re

from podtx.models import Episode

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# "Episode 860:", "Ep. 25", "Ep 9 — …"
_EPISODE_WORD = re.compile(
    r"(?i)^\s*(?:episode|ep\.?)\s*#?\s*(\d+)\b",
)
# "#860 …"
_HASH_NUM = re.compile(r"^\s*#\s*(\d+)\b")
# "860 - Title" / "860: Title" / "860 | Title" / "860 — Title"
# Reject section ids like "1.1 - …" / "5.6.3 …" via (?!\.\d).
_LEADING_NUM = re.compile(
    r"^\s*(\d+)(?!\.\d)\s*[-–—|:]\s+\S",
)


def slugify(text: str, *, max_length: int = 80) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not slug:
        slug = "episode"
    return slug[:max_length].rstrip("-")


def parse_episode_number_from_title(title: str) -> int | None:
    """Extract a leading episode number from a title, if clearly present.

    Recognizes patterns like ``860 - …``, ``#860 …``, ``Episode 860: …``.
    Rejects section-style ids such as ``1.1 - …`` / ``5.6.3 …``.
    """
    if not title or not title.strip():
        return None
    for pattern in (_EPISODE_WORD, _HASH_NUM, _LEADING_NUM):
        match = pattern.search(title)
        if match:
            return int(match.group(1))
    return None


def episode_number(episode: Episode) -> int:
    """Prefer RSS ``itunes:episode``; else parse a clear number from the title."""
    if episode.episode_num is not None and episode.episode_num >= 0:
        return episode.episode_num
    parsed = parse_episode_number_from_title(episode.title)
    if parsed is not None and parsed >= 0:
        return parsed
    return 0


def episode_date(episode: Episode) -> str:
    if episode.published_at is not None:
        return episode.published_at.strftime("%Y-%m-%d")
    return "unknown-date"


def transcript_basename(episode: Episode) -> str:
    """Return `{date}_{episode:03d}_{slug}` without extension."""
    date = episode_date(episode)
    num = f"{episode_number(episode):03d}"
    slug = slugify(episode.title)
    base = f"{date}_{num}_{slug}"
    return base


def unique_basename(episode: Episode, existing: set[str]) -> str:
    """Avoid collisions by appending a short guid hash when needed."""
    base = transcript_basename(episode)
    if base not in existing:
        return base
    digest = hashlib.sha1(episode.guid.encode()).hexdigest()[:8]
    candidate = f"{base}_{digest}"
    return candidate
