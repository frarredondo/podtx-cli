from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from podtx.config import load_settings
from podtx.db import Database
from podtx.models import Episode, Segment, Transcript


def _ep(guid: str = "g1", title: str = "Ep") -> Episode:
    return Episode(
        guid=guid,
        title=title,
        enclosure_url=f"https://x/{guid}.mp3",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        episode_num=1,
        show_title="Feed",
    )


def _engine_mock() -> mock.Mock:
    return mock.Mock(
        name="parakeet",
        transcribe=lambda wav, model, language, local_attention, local_attention_context_size: Transcript(
            text="alpha beta words",
            segments=[Segment(0, 1, "alpha beta words")],
            language=language,
            model=model,
            engine="parakeet",
        ),
    )


def _settings(tmp_path: Path, quiet: bool = True) -> object:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    s = load_settings(data_dir=data_dir)
    return s.__class__(**{**s.__dict__, "quiet": quiet, "keep_audio": True})


def _setup(monkeypatch, tmp_path: Path, settings) -> Database:
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/feed.xml", "feed", "Feed")
    monkeypatch.setattr("podtx.pipeline.download_only", lambda ep, audio_dir, quiet=False: tmp_path / "fake.mp3")
    (tmp_path / "fake.mp3").write_bytes(b"fake")
    monkeypatch.setattr("podtx.pipeline.convert_to_wav", lambda src, dst, trim_start=0.0: dst.write_bytes(b"wav") or dst)
    monkeypatch.setattr("podtx.pipeline.get_engine", lambda name: _engine_mock())
    monkeypatch.setattr("podtx.pipeline.unique_basename", lambda ep, existing: "test_ep")
    monkeypatch.setattr("podtx.pipeline.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.download.require_ffmpeg", lambda: None)
    return db, feed.id


def test_download_only_quiet_path(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import download_only

    called: dict = {}
    monkeypatch.setattr("podtx.pipeline.require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "podtx.pipeline.download_episode_audio",
        lambda url, audio_dir, h, on_progress=None: (called.setdefault("control", True), Path("/tmp/x.mp3"))[1],
    )
    out = download_only(_ep(), tmp_path, quiet=True)
    assert str(out) == "/tmp/x.mp3"


def test_download_only_progress_path(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import download_only

    control: dict = {}
    monkeypatch.setattr("podtx.pipeline.require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "podtx.pipeline.download_episode_audio",
        lambda url, audio_dir, h, on_progress=None: (
            control.__setitem__("on_progress", on_progress),
            Path("/tmp/y.mp3"),
        )[1],
    )
    out = download_only(_ep(), tmp_path, quiet=False)
    assert control.get("on_progress") is not None
    # Exercise the progress callback branches
    control["on_progress"](10, 100)
    control["on_progress"](10, 0)
    assert str(out) == "/tmp/y.mp3"


def test_log_prints_when_not_quiet(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import _log

    printed: list = []
    monkeypatch.setattr("podtx.pipeline.console", mock.Mock(print=lambda m: printed.append(m)))
    s = _settings(tmp_path)
    _log(s.__class__(**{**s.__dict__, "quiet": False}), "hello")
    assert printed == ["hello"]
    _log(s, "quiet no print")
    assert printed == ["hello"]


def test_process_episodes_skips_done(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    db, feed_id = _setup(monkeypatch, tmp_path, s)
    ep = _ep()
    db.upsert_episode(feed_id=feed_id, guid=ep.guid, title=ep.title, published_at=ep.published_at, episode_num=ep.episode_num, enclosure_url=ep.enclosure_url)
    db.mark_done(feed_id=feed_id, guid=ep.guid, engine="parakeet", model="m", output_paths=["/tmp/a.txt"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = process_episodes([ep], settings=s, out_dir=out_dir, db=db, feed_id=feed_id)
    assert results == []
    db.close()


def test_process_episodes_download_failure(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    db, feed_id = _setup(monkeypatch, tmp_path, s)

    def boom(ep, audio_dir, quiet=False):
        raise RuntimeError("network down")

    monkeypatch.setattr("podtx.pipeline.download_only", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = process_episodes([_ep()], settings=s, out_dir=out_dir, db=db, feed_id=feed_id)
    assert results == []
    assert "g1" in db.failed_guids(feed_id=feed_id)
    db.close()


def test_process_episodes_empty(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    db, feed_id = _setup(monkeypatch, tmp_path, s)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert process_episodes([], settings=s, out_dir=out_dir, db=db, feed_id=feed_id) == []
    db.close()


def test_process_episodes_skip_done_prefetch_failure(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    db, feed_id = _setup(monkeypatch, tmp_path, s)
    ep1 = _ep("done1", "Done First")
    db.upsert_episode(feed_id=feed_id, guid=ep1.guid, title=ep1.title, published_at=ep1.published_at, episode_num=ep1.episode_num, enclosure_url=ep1.enclosure_url)
    db.mark_done(feed_id=feed_id, guid=ep1.guid, engine="parakeet", model="m", output_paths=["/tmp/a.txt"])

    ep2 = _ep("done2", "Second Ep")

    def boom(ep, audio_dir, quiet=False):
        raise RuntimeError("boom")

    monkeypatch.setattr("podtx.pipeline.download_only", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # prefetch is enqueued for ep1 before the loop; when skipping the done ep1,
    # prefetch.result() raises -> swallowed at 234-235; prefetch=None for last
    results = process_episodes([ep1, ep2], settings=s, out_dir=out_dir, db=db, feed_id=feed_id)
    assert results == []
    assert "done2" in db.failed_guids(feed_id=feed_id)
    db.close()


def test_process_episodes_index_feed_none(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    db, feed_id = _setup(monkeypatch, tmp_path, s)
    monkeypatch.setattr(db, "get_feed_by_id", lambda fid: None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = process_episodes([_ep("fnon", "Foo")], settings=s, out_dir=out_dir, db=db, feed_id=feed_id)
    assert results != []
    db.close()


def test_process_episodes_download_failure_no_db(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes

    s = _settings(tmp_path)
    _setup(monkeypatch, tmp_path, s)

    def boom(ep, audio_dir, quiet=False):
        raise RuntimeError("network down")

    monkeypatch.setattr("podtx.pipeline.download_only", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    results = process_episodes([_ep("ndb", "NoDb")], settings=s, out_dir=out_dir, db=None, feed_id=None)
    assert results == []
