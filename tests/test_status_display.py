"""Tests for richer feeds/show status output (issue #10).

Seams: `podtx feeds` and `podtx show <feed>` CLI output, exercised via
CliRunner against a temp data dir backed by a real state.db.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.db import Database

runner = CliRunner()

EP_TIME = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _seed_mixed_feed(
    db: Database,
    slug: str = "demo-podcast",
    done: int = 3,
    pending: int = 1,
    errors: int = 2,
) -> None:
    """Seed one feed with done/pending/error episodes (distinct counts)."""
    feed = db.add_feed("https://example.com/podcast.rss", slug, "Demo Podcast")
    i = 0
    for _ in range(done):
        db.upsert_episode(
            feed_id=feed.id, guid=f"done-{i}", title=f"Done {i}",
            published_at=EP_TIME, episode_num=i + 1, enclosure_url="http://x.mp3",
        )
        db.mark_done(
            feed_id=feed.id, guid=f"done-{i}", engine="parakeet",
            model="test-model", output_paths=[Path(f"/tmp/out-{i}.txt")],
        )
        i += 1
    for _ in range(pending):
        db.upsert_episode(
            feed_id=feed.id, guid=f"pending-{i}", title=f"Pending {i}",
            published_at=EP_TIME, episode_num=i + 1, enclosure_url="http://x.mp3",
        )
        i += 1
    for _ in range(errors):
        db.upsert_episode(
            feed_id=feed.id, guid=f"error-{i}", title=f"Error {i}",
            published_at=EP_TIME, episode_num=i + 1, enclosure_url="http://x.mp3",
        )
        db.mark_error(feed_id=feed.id, guid=f"error-{i}", message="boom")
        i += 1


def _row(stdout: str, needle: str) -> str:
    for line in stdout.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"no line containing {needle!r} in:\n{stdout}")


def test_feeds_shows_episode_counts_and_status(tmp_path: Path, monkeypatch) -> None:
    """`podtx feeds` lists done/pending/error counts and health per feed."""
    monkeypatch.setenv("COLUMNS", "200")
    db = Database(tmp_path / "state.db")
    _seed_mixed_feed(db)  # 3 done, 1 pending, 2 error
    db.close()

    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    header = _row(result.stdout, "Slug")
    for col in ("d/p/e", "Status"):
        assert col in header

    row = _row(result.stdout, "demo-podcast")
    assert "3/1/2" in row  # done/pending/error
    assert "unhealthy" in row


def test_feeds_shows_transcript_size_and_last_transcribed(
    tmp_path: Path, monkeypatch
) -> None:
    """`podtx feeds` shows on-disk transcript size and last transcribed date."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("podtx.db._utc_now", lambda: "2026-07-22T10:00:00+00:00")

    db = Database(tmp_path / "state.db")
    _seed_mixed_feed(db)
    db.close()

    tdir = tmp_path / "transcripts" / "demo-podcast"
    tdir.mkdir(parents=True)
    (tdir / "a.txt").write_bytes(b"x" * 1000)
    (tdir / "b.json").write_bytes(b"y" * 2500)

    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    header = _row(result.stdout, "Slug")
    assert "Size" in header
    assert "Last" in header

    row = _row(result.stdout, "demo-podcast")
    assert "3.4 KB" in row  # 1000 + 2500 bytes
    assert "2026-07-22" in row


def test_show_shows_summary_line(tmp_path: Path, monkeypatch) -> None:
    """`podtx show` prints counts, health, size, and last transcribed."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("podtx.db._utc_now", lambda: "2026-07-22T10:00:00+00:00")

    db = Database(tmp_path / "state.db")
    _seed_mixed_feed(db)  # 3 done, 1 pending, 2 error
    db.close()

    tdir = tmp_path / "transcripts" / "demo-podcast"
    tdir.mkdir(parents=True)
    (tdir / "a.txt").write_bytes(b"x" * 3500)

    result = runner.invoke(app, ["show", "demo-podcast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    out = result.stdout
    assert "3 done" in out
    assert "1 pending" in out
    assert "2 error" in out
    assert "unhealthy" in out
    assert "3.4 KB" in out
    assert "2026-07-22" in out
    # per-episode table still present
    assert "Pending 3" in out


def test_show_empty_feed_shows_empty_status(tmp_path: Path, monkeypatch) -> None:
    """`podtx show` on a feed with no episodes reports 'empty' and a hint."""
    monkeypatch.setenv("COLUMNS", "200")
    db = Database(tmp_path / "state.db")
    db.add_feed("https://example.com/r.rss", "empty-feed", "Empty Feed")
    db.close()

    result = runner.invoke(app, ["show", "empty-feed", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    assert "empty-feed" in result.stdout
    assert "empty" in result.stdout
    assert "No episodes recorded yet" in result.stdout
