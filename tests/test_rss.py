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


def test_to_datetime_variants() -> None:
    from datetime import datetime, timezone
    from podtx.rss import _to_datetime

    assert _to_datetime(None) is None
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert _to_datetime(aware) == aware
    naive = _to_datetime(datetime(2026, 1, 1, 12, 0))
    assert naive.tzinfo == timezone.utc
    assert _to_datetime("2026-01-01T12:00:00") == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert _to_datetime("garbage-string") is None
    assert isinstance(_to_datetime(999999999), type(None))


def test_enclosure_from_link_rel_and_skip() -> None:
    from podtx.rss import _enclosure_url, _to_datetime
    import feedparser

    xml = """<rss version="2.0"><channel>
      <item>
        <title>t</title>
        <link>https://example.com/ep</link>
      </item>
      <item>
        <title>t2</title>
        <link rel="alternate" href="https://example.com/x"/>
      </item>
      <item>
        <title>t3</title>
        <link rel="enclosure" href="https://example.com/a.mp3"/>
      </item>
      <item>
        <title>t4</title>
        <link rel="alternate" type="audio/mpeg" href="https://example.com/b.mp3"/>
      </item>
    </channel></rss>"""
    parsed = feedparser.parse(xml)
    assert _enclosure_url(parsed.entries[0]) is None
    assert _enclosure_url(parsed.entries[1]) is None
    assert _enclosure_url(parsed.entries[2]) == "https://example.com/a.mp3"
    assert _enclosure_url(parsed.entries[3]) == "https://example.com/b.mp3"


def test_episode_num_and_guid_and_published_fallbacks() -> None:
    from podtx.rss import _episode_num, _guid
    import feedparser

    xml = """<rss version="2.0"><channel><item>
      <title>t</title>
      <itunes:episode>7</itunes:episode>
      <guid>my-guid</guid>
      <link>https://example.com/ep</link>
      <enclosure url="https://example.com/a.mp3" type="audio/mpeg"/>
    </item></channel></rss>"""
    parsed = feedparser.parse(xml)
    entry = parsed.entries[0]
    assert _episode_num(entry) == 7
    assert _guid(entry, "https://example.com/a.mp3") == "my-guid"


def test_parse_feed_skips_no_enclosure_and_guid_fallback() -> None:
    from podtx.rss import parse_feed

    xml = """<rss version="2.0"><channel><title>Show</title>
      <item><title>skipped</title><link>https://example.com/nope</link></item>
      <item><title>kept</title><link>https://example.com/ep</link>
        <enclosure url="https://example.com/a.mp3" type="audio/mpeg"/>
        <published>Tue, 03 Jun 2003 09:39:21 GMT</published>
      </item>
    </channel></rss>"""
    title, slug, eps = parse_feed(xml)
    assert len(eps) == 1
    assert eps[0].guid == "https://example.com/a.mp3"
    assert eps[0].published_at is not None


def test_to_datetime_iso_with_z_and_naive() -> None:
    from datetime import datetime, timezone
    from podtx.rss import _to_datetime

    assert _to_datetime("2026-05-04T05:06:07Z") == datetime(2026, 5, 4, 5, 6, 7, tzinfo=timezone.utc)
    assert _to_datetime("2026-05-04 05:06:07") == datetime(2026, 5, 4, 5, 6, 7, tzinfo=timezone.utc)


def test_episode_num_invalid_returns_none() -> None:
    from podtx.rss import _episode_num
    import feedparser

    parsed = feedparser.parse('<rss version="2.0"><channel><item><title>t</title><itunes:episode>abc</itunes:episode></item></channel></rss>')
    assert _episode_num(parsed.entries[0]) is None


def test_enclosure_enclosure_item_no_href_falls_to_links() -> None:
    from podtx.rss import _enclosure_url
    import feedparser

    xml = """<rss version="2.0"><channel><item><title>t</title>
      <enclosure url="" type="audio/mpeg"/>
      <link rel="enclosure" href="https://example.com/real.mp3"/>
    </item></channel></rss>"""
    parsed = feedparser.parse(xml)
    assert _enclosure_url(parsed.entries[0]) == "https://example.com/real.mp3"


def test_parse_feed_raises_on_bozo() -> None:
    from podtx.rss import FeedParseError, parse_feed

    try:
        parse_feed("this is not xml at all <<<")
        assert False
    except FeedParseError:
        pass


def test_parse_feed_published_falls_back_to_updated() -> None:
    from podtx.rss import parse_feed
    from datetime import datetime, timezone

    xml = """<rss version="2.0"><channel><title>Show</title><item><title>kept</title>
      <link>https://example.com/ep</link>
      <enclosure url="https://example.com/a.mp3" type="audio/mpeg"/>
      <updated>Tue, 03 Jun 2003 09:39:21 GMT</updated>
    </item></channel></rss>"""
    _, _, eps = parse_feed(xml)
    assert eps[0].published_at is not None
    assert eps[0].published_at.tzinfo is not None


def test_enclosure_edges_synthetic() -> None:
    from podtx.rss import _enclosure_url

    class E:
        pass

    e_enc_no_href = E()
    e_enc_no_href.enclosures = [{"href": "", "url": ""}]
    e_enc_no_href.links = []
    assert _enclosure_url(e_enc_no_href) is None

    e_link_no_href = E()
    e_link_no_href.enclosures = []
    e_link_no_href.links = [{"rel": "enclosure", "href": ""}]
    assert _enclosure_url(e_link_no_href) is None

    e_link_href = E()
    e_link_href.enclosures = []
    e_link_href.links = [{"rel": "enclosure", "href": "https://x/a.mp3", "type": "audio/mpeg"}]
    assert _enclosure_url(e_link_href) == "https://x/a.mp3"
