from __future__ import annotations

from datetime import datetime, timezone

from podtx.models import Episode
from podtx.naming import (
    episode_number,
    parse_episode_number_from_title,
    slugify,
    transcript_basename,
    unique_basename,
)


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


def test_transcript_basename_missing_episode_without_number_in_title() -> None:
    assert transcript_basename(_ep(episode_num=None)) == "2026-03-15_000_interview-with-ada"


def test_parse_episode_number_from_title_common_patterns() -> None:
    assert parse_episode_number_from_title("860 - Module Federation") == 860
    assert parse_episode_number_from_title("860: New APIs") == 860
    assert parse_episode_number_from_title("#702 Potluck") == 702
    assert parse_episode_number_from_title("Episode 122: The Bitter Lesson") == 122
    assert parse_episode_number_from_title("Ep. 25 Why ML Needs a New Language") == 25
    assert parse_episode_number_from_title("Ep 9 — Inside look") == 9


def test_parse_episode_number_from_title_rejects_section_style() -> None:
    # "1.1 - Introduction" / "5.6.3 and …" are section ids, not episode numbers.
    assert parse_episode_number_from_title("1.1 - Introduction to Software Engineering") is None
    assert parse_episode_number_from_title(
        "5.6.3 and 5.6.4 - Dependency Inversion"
    ) is None
    assert parse_episode_number_from_title("Hasty Treat - The Future of Testing") is None
    assert parse_episode_number_from_title("") is None


def test_episode_number_prefers_rss_over_title() -> None:
    ep = _ep(episode_num=3, title="860 - Module Federation")
    assert episode_number(ep) == 3


def test_episode_number_falls_back_to_title_when_rss_missing() -> None:
    ep = _ep(episode_num=None, title="860 - Module Federation")
    assert episode_number(ep) == 860
    # Slug still includes title text as-is (slug rules unchanged).
    assert transcript_basename(ep) == "2026-03-15_860_860-module-federation"


def test_unique_basename_collision() -> None:
    base = transcript_basename(_ep())
    uniq = unique_basename(_ep(), existing={base})
    assert uniq.startswith(base + "_")
    assert len(uniq) > len(base)
