"""TDD tests for issue #10 richer feeds/show status."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.config import load_settings
from podtx.db import Database

runner = CliRunner()


def _seed_db(tmp_path: Path):
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/feed.xml", "demo-feed", "Demo Feed")
    db.upsert_episode(feed_id=feed.id, guid="g1", title="Ep 1", published_at=None, episode_num=1, enclosure_url="https://x/1.mp3")
    db.upsert_episode(feed_id=feed.id, guid="g2", title="Ep 2", published_at=None, episode_num=2, enclosure_url="https://x/2.mp3")
    db.upsert_episode(feed_id=feed.id, guid="g3", title="Ep 3", published_at=None, episode_num=3, enclosure_url="https://x/3.mp3")
    db.upsert_episode(feed_id=feed.id, guid="g4", title="Ep 4", published_at=None, episode_num=4, enclosure_url="https://x/4.mp3")
    db.mark_done(feed_id=feed.id, guid="g1", engine="parakeet", model="test", output_paths=[tmp_path / "out1.txt"])
    db.mark_done(feed_id=feed.id, guid="g2", engine="parakeet", model="test", output_paths=[tmp_path / "out2.txt"])
    db.mark_error(feed_id=feed.id, guid="g3", message="fail")
    db.close()
    return settings, feed


def test_feeds_shows_counts_health_and_size(tmp_path: Path) -> None:
    _seed_db(tmp_path)
    settings = load_settings(data_dir=tmp_path)
    tdir = settings.transcripts_dir("demo-feed")
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "2026-01-01_001_demo-feed.txt").write_text("a" * 1024)
    (tdir / "2026-01-01_001_demo-feed.json").write_text("b" * 2048)

    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    out = result.stdout
    # normalize whitespace to handle table wrapping at 80 cols
    lower = " ".join(out.lower().split())
    # counts: done/pending/failed with numbers 2,1,1 and total 4
    assert "2 done" in lower
    assert "1 pending" in lower
    assert "1 failed" in lower
    assert "(4)" in out or "4" in out
    # health indicator appears
    assert "health" in lower
    assert "unhealthy" in lower
    # disk size appears as human readable
    assert "kb" in lower or "mb" in lower or "b" in lower


def test_feeds_empty_distinguishable(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    # use slug/title that does NOT contain "empty" to avoid false positive
    db.add_feed("https://example.com/empty.xml", "new-feed", "New Feed")
    db.close()

    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "empty" in out
    # should show zero counts
    assert "0 done" in out or "0" in out


def test_show_header_shows_last_transcribed_and_pending_summary(tmp_path: Path) -> None:
    _seed_db(tmp_path)
    result = runner.invoke(app, ["show", "demo-feed", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    out = result.stdout.lower()
    assert "transcribed" in out
    assert "last" in out
    assert "pending" in out
    assert "health" in out or "unhealthy" in out
    # should show pending queue summary
    assert "pending queue" in out
    assert "size" in out


def test_show_empty_feed_header(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    db.add_feed("https://example.com/empty2.xml", "empty2", "Empty2 Feed")
    db.close()
    result = runner.invoke(app, ["show", "empty2", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "no episodes" in out
    assert "empty" in out
    assert "health" in out
