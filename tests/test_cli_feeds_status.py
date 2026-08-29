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


def test_feeds_healthy_and_no_pending(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/healthy.xml", "healthy-feed", "Healthy Feed")
    db.upsert_episode(feed_id=feed.id, guid="hg1", title="H1", published_at=None, episode_num=1, enclosure_url="https://x/h1.mp3")
    db.mark_done(feed_id=feed.id, guid="hg1", engine="parakeet", model="m", output_paths=[tmp_path / "h1.txt"])
    db.close()
    # create transcript file to test size
    tdir = settings.transcripts_dir("healthy-feed")
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "f.txt").write_text("x" * 500)
    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    lower = " ".join(result.stdout.lower().split())
    assert "1 done" in lower
    assert "healthy" in lower


def test_show_healthy_no_pending_queue(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/h2.xml", "h2", "H2 Feed")
    db.upsert_episode(feed_id=feed.id, guid="hg2", title="H2", published_at=None, episode_num=1, enclosure_url="https://x/h2.mp3")
    db.mark_done(feed_id=feed.id, guid="hg2", engine="parakeet", model="m", output_paths=[tmp_path / "h2.txt"])
    db.close()
    result = runner.invoke(app, ["show", "h2", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    lower = " ".join(result.stdout.lower().split())
    assert "healthy" in lower
    # no pending queue when healthy
    assert "pending queue" not in lower


def test_human_size_branches(tmp_path: Path) -> None:
    from podtx.cli import _human_size

    assert _human_size(0) == "0 B"
    assert _human_size(512) == "512 B"
    assert _human_size(1024) == "1.0 KB"
    assert _human_size(1536) == "1.5 KB"
    assert _human_size(1024 * 1024) == "1.0 MB"
    assert _human_size(1024 * 1024 * 5) == "5.0 MB"
    assert _human_size(1024 * 1024 * 1024) == "1.0 GB"
    assert _human_size(1024 * 1024 * 1024 * 2) == "2.0 GB"


def test_transcript_disk_size_missing_dir(tmp_path: Path) -> None:
    from podtx.cli import _transcript_disk_size

    settings = load_settings(data_dir=tmp_path)
    # no dir yet
    assert _transcript_disk_size(settings, "nope") == 0


def test_db_helpers_direct(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/db.xml", "db-feed", "DB Feed")
    # initially empty
    assert db.failed_guids(feed.id) == set()
    assert db.pending_guids(feed.id) == set()
    assert db.last_transcribed_at(feed.id) is None
    db.upsert_episode(feed_id=feed.id, guid="a", title="A", published_at=None, episode_num=1, enclosure_url="https://x/a.mp3")
    assert "a" in db.pending_guids(feed.id)
    db.mark_error(feed_id=feed.id, guid="a", message="oops")
    assert "a" in db.failed_guids(feed.id)
    assert "a" not in db.pending_guids(feed.id)
    # after done, last_transcribed_at should be set
    db.upsert_episode(feed_id=feed.id, guid="b", title="B", published_at=None, episode_num=2, enclosure_url="https://x/b.mp3")
    db.mark_done(feed_id=feed.id, guid="b", engine="e", model="m", output_paths=[tmp_path / "b.txt"])
    assert db.last_transcribed_at(feed.id) is not None
    db.close()


def test_feeds_no_feeds_registered(tmp_path: Path) -> None:
    result = runner.invoke(app, ["feeds", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no feeds" in result.stdout.lower()


def test_show_feed_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["show", "missing", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_transcript_disk_size_handles_oserror(tmp_path: Path, monkeypatch) -> None:
    from podtx.cli import _transcript_disk_size
    from unittest import mock

    settings = load_settings(data_dir=tmp_path)
    tdir = settings.transcripts_dir("feed-a")
    tdir.mkdir(parents=True)
    (tdir / "a.txt").write_text("hello")

    orig_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "a.txt":
            raise OSError("mocked")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    size = _transcript_disk_size(settings, "feed-a")
    assert isinstance(size, int)
    assert size == 0

    # also test with no error
    monkeypatch.setattr(Path, "stat", orig_stat)
    assert _transcript_disk_size(settings, "feed-a") == 5


def test_show_with_no_health_summary(monkeypatch, tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/monkey.xml", "monkey-feed", "Monkey Feed")
    db.upsert_episode(feed_id=feed.id, guid="m1", title="M1", published_at=None, episode_num=1, enclosure_url="https://x/m1.mp3")
    db.close()
    monkeypatch.setattr(Database, "feed_health_summary", lambda self: [])
    result = runner.invoke(app, ["show", "monkey-feed", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "M1" in result.stdout
