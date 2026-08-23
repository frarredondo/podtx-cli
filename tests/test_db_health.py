"""Tests for new Database health-check query methods (TDD Cycle: Phase 1).

Methods tested:
    - failed_guids(feed_id)
    - pending_guids(feed_id)
    - empty_feeds()
    - feed_health_summary()
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from podtx.db import Database


class TestFailedGuids:
    """failed_guids(feed_id) should return only 'error' status GUIDs."""

    def test_returns_only_error_guids(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")
        feed = db.add_feed("https://example.com", "test-feed", "Test Feed")

        # Insert mixed episodes
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed.id, guid="g2", title="Ep 2", published_at=None,
            episode_num=2, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed.id, guid="g3", title="Ep 3", published_at=None,
            episode_num=3, enclosure_url="http://x.mp3",
        )

        # Mark g1 as error, leave g2 pending, mark g3 done
        db.mark_error(feed_id=feed.id, guid="g1", message="network fail")
        db.mark_done(
            feed_id=feed.id, guid="g3", engine="parakeet", model="test",
            output_paths=[tmp_path / "out.txt"],
        )

        result = db.failed_guids(feed.id)
        assert result == {"g1"}

    def test_returns_empty_set_when_no_errors(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")
        feed = db.add_feed("https://example.com", "clean-feed", "Clean Feed")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )

        assert db.failed_guids(feed.id) == set()


class TestPendingGuids:
    """pending_guids(feed_id) should return only 'pending' status GUIDs."""

    def test_returns_only_pending_guids(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")
        feed = db.add_feed("https://example.com", "test-feed", "Test Feed")

        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed.id, guid="g2", title="Ep 2", published_at=None,
            episode_num=2, enclosure_url="http://x.mp3",
        )

        # g1: pending (default), g2: done
        db.mark_done(
            feed_id=feed.id, guid="g2", engine="parakeet", model="test",
            output_paths=[tmp_path / "out.txt"],
        )

        result = db.pending_guids(feed.id)
        assert result == {"g1"}

    def test_returns_empty_set_when_no_pending(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")
        feed = db.add_feed("https://example.com", "done-feed", "Done Feed")

        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_done(
            feed_id=feed.id, guid="g1", engine="parakeet", model="test",
            output_paths=[tmp_path / "out.txt"],
        )

        assert db.pending_guids(feed.id) == set()


class TestEmptyFeeds:
    """empty_feeds() should return feeds with zero episode records."""

    def test_returns_feeds_with_no_episodes(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        # Feed with episodes
        feed_a = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed_a.id, guid="ga1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )

        # Feed with NO episodes — empty
        feed_b = db.add_feed("https://b.com", "feed-b", "Feed B")

        result = db.empty_feeds()
        assert len(result) == 1
        assert result[0]["slug"] == "feed-b"

    def test_returns_empty_when_all_feeds_have_episodes(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        feed_a = db.add_feed("https://a.com", "feed-a", "Feed A")
        feed_b = db.add_feed("https://b.com", "feed-b", "Feed B")

        db.upsert_episode(
            feed_id=feed_a.id, guid="ga1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed_b.id, guid="gb1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )

        assert db.empty_feeds() == []


class TestFeedHealthSummary:
    """feed_health_summary() returns aggregated status counts per feed."""

    def test_returns_summaries_for_all_feeds(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        # Feed A: 2 done, 1 pending
        feed_a = db.add_feed("https://a.com", "feed-a", "Feed A")
        for i in range(2):
            db.upsert_episode(
                feed_id=feed_a.id, guid=f"a-{i}", title=f"Ep {i}", published_at=None,
                episode_num=i + 1, enclosure_url="http://x.mp3",
            )
        db.upsert_episode(
            feed_id=feed_a.id, guid="a-2", title="Ep 3", published_at=None,
            episode_num=3, enclosure_url="http://x.mp3",
        )
        for i in range(2):
            db.mark_done(
                feed_id=feed_a.id, guid=f"a-{i}", engine="parakeet", model="test",
                output_paths=[tmp_path / "out.txt"],
            )

        # Feed B: 1 done, 1 error
        feed_b = db.add_feed("https://b.com", "feed-b", "Feed B")
        db.upsert_episode(
            feed_id=feed_b.id, guid="b-0", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed_b.id, guid="b-1", title="Ep 2", published_at=None,
            episode_num=2, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed_b.id, guid="b-1", message="fail")
        db.mark_done(
            feed_id=feed_b.id, guid="b-0", engine="parakeet", model="test",
            output_paths=[tmp_path / "out.txt"],
        )

        summaries = db.feed_health_summary()

        # Two feeds in results
        assert len(summaries) == 2

        a_row = summaries[0]
        b_row = summaries[1]

        assert {a_row["feed_id"], b_row["feed_id"]} == {feed_a.id, feed_b.id}

        # Feed A: 2 done + 1 pending = 3 total, status=unhealthy
        if summaries[0]["feed_id"] == feed_a.id:
            assert summaries[0]["total_episodes"] == 3
            assert summaries[0]["done_count"] == 2
            assert summaries[0]["pending_count"] == 1
            assert summaries[0]["error_count"] == 0
            assert summaries[0]["health_status"] == "unhealthy"

        # Feed B: 1 done + 1 error = 2 total, status=unhealthy
        if summaries[0]["feed_id"] == feed_b.id:
            assert summaries[0]["total_episodes"] == 2
            assert summaries[0]["done_count"] == 1
            assert summaries[0]["pending_count"] == 0
            assert summaries[0]["error_count"] == 1
            assert summaries[0]["health_status"] == "unhealthy"

    def test_returns_healthy_when_all_done(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        feed = db.add_feed("https://a.com", "healthy-feed", "Healthy Feed")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_done(
            feed_id=feed.id, guid="g1", engine="parakeet", model="test",
            output_paths=[tmp_path / "out.txt"],
        )

        summaries = db.feed_health_summary()
        assert len(summaries) == 1
        assert summaries[0]["health_status"] == "healthy"

    def test_returns_unhealthy_when_errors_exist(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        feed = db.add_feed("https://a.com", "error-feed", "Error Feed")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed.id, guid="g1", message="network")

        summaries = db.feed_health_summary()
        assert len(summaries) == 1
        assert summaries[0]["health_status"] == "unhealthy"

    def test_returns_empty_when_no_feeds(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        assert db.feed_health_summary() == []


class TestFeedHealthSummaryEmptyFeeds:
    """Empty feeds (zero episodes) should still appear in health summary."""

    def test_empty_feed_appears_with_zero_counts(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "state.db")

        feed_a = db.add_feed("https://a.com", "feed-a", "Feed A")
        feed_b = db.add_feed("https://b.com", "feed-b", "Feed B")

        # Only feed-a has episodes
        db.upsert_episode(
            feed_id=feed_a.id, guid="g1", title="Ep 1", published_at=None,
            episode_num=1, enclosure_url="http://x.mp3",
        )

        summaries = db.feed_health_summary()
        assert len(summaries) == 2

        for s in summaries:
            if s["feed_id"] == feed_b.id:
                assert s["total_episodes"] == 0
                assert s["done_count"] == 0
                assert s["pending_count"] == 0
                assert s["error_count"] == 0
