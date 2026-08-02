from __future__ import annotations

from pathlib import Path

from podtx.rss import parse_feed, suggest_unique_slug

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def test_parse_feed_fixture() -> None:
    xml = FIXTURE.read_text(encoding="utf-8")
    title, slug, episodes = parse_feed(xml)
    assert title == "Demo Podcast"
    assert slug == "demo-podcast"
    assert len(episodes) == 3
    assert episodes[0].guid == "ep-3"
    assert episodes[0].episode_num == 3
    assert episodes[0].title == "Interview with Ada"
    assert episodes[2].episode_num is None  # pilot has no itunes:episode
    assert episodes[2].guid == "ep-1"


def test_suggest_unique_slug() -> None:
    assert suggest_unique_slug("demo", set()) == "demo"
    assert suggest_unique_slug("demo", {"demo"}) == "demo-2"
    assert suggest_unique_slug("demo", {"demo", "demo-2"}) == "demo-3"
