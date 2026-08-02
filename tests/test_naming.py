from __future__ import annotations

from datetime import datetime, timezone

from podtx.models import Episode
from podtx.naming import slugify, transcript_basename, unique_basename


def _ep(**kwargs: object) -> Episode:
    defaults = {
        "guid": "g1",
        "title": "Interview with Ada!",
        "enclosure_url": "https://example.com/a.mp3",
        "published_at": datetime(2026, 3, 15, tzinfo=timezone.utc),
        "episode_num": 3,
    }
    defaults.update(kwargs)
    return Episode(**defaults)  # type: ignore[arg-type]


def test_slugify() -> None:
    assert slugify("Interview with Ada!") == "interview-with-ada"


def test_transcript_basename_padded_episode() -> None:
    assert transcript_basename(_ep()) == "2026-03-15_003_interview-with-ada"


def test_transcript_basename_missing_episode() -> None:
    assert transcript_basename(_ep(episode_num=None)) == "2026-03-15_000_interview-with-ada"


def test_unique_basename_collision() -> None:
    base = transcript_basename(_ep())
    uniq = unique_basename(_ep(), existing={base})
    assert uniq.startswith(base + "_")
    assert len(uniq) > len(base)
