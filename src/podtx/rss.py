from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import mktime, struct_time

import feedparser

from podtx.models import Episode
from podtx.naming import slugify


class FeedParseError(Exception):
    pass


def _to_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, struct_time):
        return datetime.fromtimestamp(mktime(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)  # pragma: no cover - parsedate returns aware in practice
            return dt
        except (TypeError, ValueError, IndexError):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return None
    return None


def _enclosure_url(entry: feedparser.FeedParserDict) -> str | None:
    if getattr(entry, "enclosures", None):
        for enc in entry.enclosures:
            href = enc.get("href") or enc.get("url")
            if href:
                return str(href)
    for link in getattr(entry, "links", []) or []:
        rel = link.get("rel")
        typ = (link.get("type") or "").lower()
        if rel == "enclosure" or typ.startswith("audio/") or typ.startswith("video/"):
            href = link.get("href")
            if href:
                return str(href)
    return None


def _episode_num(entry: feedparser.FeedParserDict) -> int | None:
    raw = entry.get("itunes_episode")
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _guid(entry: feedparser.FeedParserDict, enclosure_url: str) -> str:
    guid = entry.get("id") or entry.get("guid")
    if guid:
        return str(guid).strip()
    return enclosure_url


def parse_feed(url_or_content: str) -> tuple[str, str, list[Episode]]:
    """Parse an RSS feed URL or raw XML content.

    Returns (show_title, feed_slug_suggestion, episodes sorted newest-first).
    """
    parsed = feedparser.parse(url_or_content)
    if getattr(parsed, "bozo", False) and not parsed.entries and not parsed.feed:
        raise FeedParseError(f"Failed to parse feed: {getattr(parsed, 'bozo_exception', 'unknown error')}")

    show_title = str(parsed.feed.get("title") or "Unknown Podcast").strip()
    episodes: list[Episode] = []

    for entry in parsed.entries:
        enclosure = _enclosure_url(entry)
        if not enclosure:
            continue
        title = str(entry.get("title") or "Untitled Episode").strip()
        published = _to_datetime(entry.get("published_parsed") or entry.get("published"))
        if published is None:
            published = _to_datetime(entry.get("updated_parsed") or entry.get("updated"))
        episodes.append(
            Episode(
                guid=_guid(entry, enclosure),
                title=title,
                enclosure_url=enclosure,
                published_at=published,
                episode_num=_episode_num(entry),
                description=str(entry.get("summary") or "") or None,
                link=str(entry.get("link") or "") or None,
                show_title=show_title,
            )
        )

    episodes.sort(
        key=lambda e: e.published_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return show_title, slugify(show_title), episodes


def suggest_unique_slug(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"
