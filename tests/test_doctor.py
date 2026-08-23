"""Tests for the `podtx doctor` library-health command (issue #13).

doctor must:
    - list failed (error) and stuck (pending) episodes that need attention
    - flag empty feeds (zero episode records)
    - sanity-check that recorded output files for done episodes still exist
    - flag empty/unhealthy feeds in a per-feed summary
    - always exit 0 (it is a reporting tool, not a pipeline step)
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app, collect_doctor_report
from podtx.db import Database

runner = CliRunner()


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "state.db")


def _done_with_existing_output(
    db: Database, tmp_path: Path, *, feed_id: int, guid: str, name: str = "ep.txt"
) -> Path:
    """Mark an episode done with an output file that exists on disk."""
    out = tmp_path / "transcripts" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("hello", encoding="utf-8")
    db.mark_done(
        feed_id=feed_id, guid=guid, engine="parakeet", model="test",
        output_paths=[out],
    )
    return out


class TestDoctorHelpAndEmptyLibrary:
    def test_listed_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "doctor" in result.stdout

    def test_no_feeds(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No feeds registered" in result.stdout


class TestDoctorHealthyLibrary:
    def test_all_healthy_needs_nothing(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        for i in range(2):
            db.upsert_episode(
                feed_id=feed.id, guid=f"g{i}", title=f"Ep {i}",
                published_at=None, episode_num=i + 1, enclosure_url="http://x.mp3",
            )
            _done_with_existing_output(
                db, tmp_path, feed_id=feed.id, guid=f"g{i}", name=f"ep{i}.txt"
            )
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "healthy" in result.stdout
        assert "nothing needs attention" in result.stdout
        assert "Needs attention" not in result.stdout

    def test_healthy_with_zero_feeds_is_exit_zero(self, tmp_path: Path) -> None:
        # Contract: doctor is a reporter; unhealthy findings must not fail the CLI.
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed.id, guid="g1", message="boom")
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0


class TestDoctorAttention:
    def test_failed_episode_listed_with_error_message(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Good Episode",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        _done_with_existing_output(
            db, tmp_path, feed_id=feed.id, guid="g1", name="good.txt"
        )
        db.upsert_episode(
            feed_id=feed.id, guid="g2", title="Bad Episode",
            published_at=None, episode_num=2, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed.id, guid="g2", message="network fail")
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Bad Episode" in result.stdout
        assert "network fail" in result.stdout
        assert "failed" in result.stdout
        assert "unhealthy" in result.stdout

    def test_pending_episode_listed(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Done Episode",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        _done_with_existing_output(
            db, tmp_path, feed_id=feed.id, guid="g1", name="done.txt"
        )
        db.upsert_episode(
            feed_id=feed.id, guid="g2", title="Stuck Episode",
            published_at=None, episode_num=2, enclosure_url="http://x.mp3",
        )
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Stuck Episode" in result.stdout
        assert "pending" in result.stdout

    def test_empty_feed_flagged(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        feed_a = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed_a.id, guid="g1", title="Ep 1",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        _done_with_existing_output(
            db, tmp_path, feed_id=feed_a.id, guid="g1", name="ep1.txt"
        )
        feed_b = db.add_feed("https://b.com", "feed-b", "Feed B")
        assert feed_b.id != feed_a.id
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "empty" in result.stdout
        assert "Feed B" in result.stdout

    def test_missing_output_files_flagged(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        db.upsert_episode(
            feed_id=feed.id, guid="g1", title="Ep 1",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        # Record an output path that does not exist on disk.
        db.mark_done(
            feed_id=feed.id, guid="g1", engine="parakeet", model="test",
            output_paths=[tmp_path / "transcripts" / "gone.txt"],
        )
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "missing" in result.stdout
        assert "gone.txt" in result.stdout
        assert "Ep 1" in result.stdout
        # Missing outputs escalate the feed's displayed status to unhealthy.
        assert "unhealthy" in result.stdout
        assert "1 of 1 feed(s) need attention" in result.stdout

    def test_summary_counts_needs_attention(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        healthy = db.add_feed("https://a.com", "feed-a", "Feed A")
        for i in range(2):
            db.upsert_episode(
                feed_id=healthy.id, guid=f"h{i}", title=f"H {i}",
                published_at=None, episode_num=i + 1, enclosure_url="http://x.mp3",
            )
            _done_with_existing_output(
                db, tmp_path, feed_id=healthy.id, guid=f"h{i}", name=f"h{i}.txt"
            )
        sick = db.add_feed("https://b.com", "feed-b", "Feed B")
        for i, guid in enumerate(("s0", "s1", "s2")):
            db.upsert_episode(
                feed_id=sick.id, guid=guid, title=f"S {i}",
                published_at=None, episode_num=i + 1, enclosure_url="http://x.mp3",
            )
        db.mark_error(feed_id=sick.id, guid="s0", message="dl fail")
        db.mark_done(
            feed_id=sick.id, guid="s1", engine="parakeet", model="test",
            output_paths=[tmp_path / "s1.txt"],  # missing on disk
        )
        # s2 left pending
        empty = db.add_feed("https://c.com", "feed-c", "Feed C")
        assert empty.id not in {healthy.id, sick.id}
        db.close()

        result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "2 of 3 feed(s) need attention" in result.stdout
        assert "3 episode(s)" in result.stdout


class TestCollectDoctorReport:
    """Unit tests for the row classifier used by the doctor command."""

    def _seed(self, tmp_path: Path) -> tuple[Database, int]:
        db = _db(tmp_path)
        feed = db.add_feed("https://a.com", "feed-a", "Feed A")
        return db, feed.id

    def test_classifies_failed_pending_and_missing(self, tmp_path: Path) -> None:
        db, feed_id = self._seed(tmp_path)
        db.upsert_episode(
            feed_id=feed_id, guid="e1", title="Err",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed_id, guid="e1", message="disk full")
        db.upsert_episode(
            feed_id=feed_id, guid="e2", title="Wait",
            published_at=None, episode_num=2, enclosure_url="http://x.mp3",
        )
        db.upsert_episode(
            feed_id=feed_id, guid="e3", title="Lost",
            published_at=None, episode_num=3, enclosure_url="http://x.mp3",
        )
        db.mark_done(
            feed_id=feed_id, guid="e3", engine="parakeet", model="test",
            output_paths=[tmp_path / "lost.txt"],
        )

        rows = collect_doctor_report(db)
        db.close()

        issues = {r["guid"]: r["issue"] for r in rows}
        assert issues == {"e1": "failed", "e2": "pending", "e3": "missing outputs"}
        assert [r for r in rows if r["guid"] == "e1"][0]["detail"] == "disk full"

    def test_done_with_all_outputs_present_is_not_flagged(self, tmp_path: Path) -> None:
        db, feed_id = self._seed(tmp_path)
        db.upsert_episode(
            feed_id=feed_id, guid="ok", title="Fine",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        out = tmp_path / "fine.txt"
        out.write_text("hi", encoding="utf-8")
        db.mark_done(
            feed_id=feed_id, guid="ok", engine="parakeet", model="test",
            output_paths=[out],
        )

        assert collect_doctor_report(db) == []
        db.close()

    def test_invalid_output_paths_json_is_ignored_not_fatal(self, tmp_path: Path) -> None:
        db, feed_id = self._seed(tmp_path)
        db.upsert_episode(
            feed_id=feed_id, guid="weird", title="Weird",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_done(
            feed_id=feed_id, guid="weird", engine="parakeet", model="test",
            output_paths=[tmp_path / "w.txt"],
        )
        db._conn.execute(
            "UPDATE episodes SET output_paths_json = 'not-json' WHERE guid = ?",
            ("weird",),
        )
        db._conn.commit()

        rows = collect_doctor_report(db)
        db.close()

        # Unparseable path records must not crash doctor; nothing else to report.
        assert [r["guid"] for r in rows] == []

    def test_error_without_recorded_message_has_fallback_detail(self, tmp_path: Path) -> None:
        db, feed_id = self._seed(tmp_path)
        db.upsert_episode(
            feed_id=feed_id, guid="err", title="Err",
            published_at=None, episode_num=1, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed_id, guid="err", message="")
        # Simulate legacy rows with no recorded message.
        db._conn.execute(
            "UPDATE episodes SET output_paths_json = ? WHERE guid = ?",
            (json.dumps({"error": None}), "err"),
        )
        db._conn.commit()

        rows = collect_doctor_report(db)
        db.close()

        assert len(rows) == 1
        assert rows[0]["issue"] == "failed"
        assert rows[0]["detail"]  # non-empty fallback
