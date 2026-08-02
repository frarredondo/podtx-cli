from __future__ import annotations

import hashlib
import re

from podtx.models import Episode

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 80) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not slug:
        slug = "episode"
    return slug[:max_length].rstrip("-")


def episode_number(episode: Episode) -> int:
    if episode.episode_num is not None and episode.episode_num >= 0:
        return episode.episode_num
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
