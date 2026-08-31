from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from podtx.db import Database
from podtx.models import Episode
from podtx.pipeline import select_episodes_for_sync


def test_db_feed_and_idempotency(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    feed = db.add_feed("https://example.com/feed.xml", "demo", "Demo")
    assert db.get_feed("demo") is not None
    assert db.get_feed("https://example.com/feed.xml") is not None

    db.upsert_episode(
        feed_id=feed.id,
        guid="ep-1",
        title="Pilot",
        published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        episode_num=1,
        enclosure_url="https://example.com/ep1.mp3",
    )
    assert not db.is_done(feed.id, "ep-1")
    db.mark_done(
        feed_id=feed.id,
        guid="ep-1",
        engine="parakeet",
        model="test",
        output_paths=[tmp_path / "out.txt"],
    )
    assert db.is_done(feed.id, "ep-1")
    assert "ep-1" in db.done_guids(feed.id)

    assert db.remove_feed("demo")
    assert db.get_feed("demo") is None
    db.close()


def test_select_episodes_limit() -> None:
    episodes = [
        Episode(guid=f"g{i}", title=f"E{i}", enclosure_url=f"https://x/{i}.mp3")
        for i in range(10)
    ]
    selected = select_episodes_for_sync(
        episodes, done_guids=set(), limit=5, process_all=False
    )
    assert len(selected) == 5
    assert [e.guid for e in selected] == [f"g{i}" for i in range(5)]

    selected2 = select_episodes_for_sync(
        episodes, done_guids={"g0", "g1"}, limit=5, process_all=False
    )
    assert [e.guid for e in selected2] == [f"g{i}" for i in range(2, 7)]

    all_pending = select_episodes_for_sync(
        episodes, done_guids={"g0"}, limit=1, process_all=True
    )
    assert len(all_pending) == 9


def test_db_counts(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.episode_count(1) == 0
    assert db.done_count(1) == 0
    feed = db.add_feed("https://example.com/feed.xml", "demo", "Demo")
    db.upsert_episode(
        feed_id=feed.id,
        guid="ep-1",
        title="Pilot",
        published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        episode_num=1,
        enclosure_url="https://example.com/ep1.mp3",
    )
    db.mark_done(
        feed_id=feed.id,
        guid="ep-1",
        engine="parakeet",
        model="test",
        output_paths=[tmp_path / "out.txt"],
    )
    assert db.episode_count(feed.id) == 1
    assert db.done_count(feed.id) == 1
    db.close()
