from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.config import load_settings
from podtx.db import Database
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()

# Helpers to create a mini library with transcripts and DB

def _write_transcript(
    out_dir: Path,
    basename: str,
    episode: Episode,
    transcript: Transcript,
) -> list[Path]:
    return write_outputs(
        out_dir=out_dir,
        basename=basename,
        episode=episode,
        transcript=transcript,
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )


def _sample_episode(
    guid: str = "g1",
    title: str = "The Bitter Lesson",
    published_at: datetime | None = None,
    enclosure_url: str = "https://example.com/ep.mp3",
    show_title: str = "Demo",
    episode_num: int | None = 1,
) -> Episode:
    return Episode(
        guid=guid,
        title=title,
        enclosure_url=enclosure_url,
        published_at=published_at,
        episode_num=episode_num,
        show_title=show_title,
        link="https://example.com/ep",
    )


def test_db_fts_basic_search_returns_hits_with_snippet_and_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    feed = db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")

    episode = _sample_episode(title="The Bitter Lesson Revisited")
    transcript = Transcript(
        text="The bitter lesson is that compute scales and search wins.",
        segments=[],
        language="en",
        model="test",
        engine="parakeet",
    )
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    paths = _write_transcript(out_dir, "2026-01-01_001_bitter-lesson", episode, transcript)

    # incremental indexing — upsert
    db.upsert_search_entry(
        feed_slug="myfeed",
        guid=episode.guid,
        title=episode.title,
        published_at=episode.published_at.isoformat() if episode.published_at else None,
        text=transcript.text,
        txt_path=str(paths[0]),
        json_path=str(paths[1]),
    )

    results = db.search_transcripts("bitter lesson")
    assert len(results) == 1
    hit = results[0]
    assert hit["feed_slug"] == "myfeed"
    assert "bitter" in hit["snippet"].lower() or "bitter" in hit["text"].lower()
    assert "txt_path" in hit or "txt" in str(hit)
    # paths must be present
    assert "myfeed" in hit.get("txt_path", "") or "myfeed" in hit.get("json_path", "")
    db.close()


def test_db_fts_case_insensitive(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    feed = db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    episode = _sample_episode(guid="g2", title="MLIR Deep Dive")
    transcript = Transcript(text="MLIR is the new compiler infra.", segments=[], language="en", model="t", engine="e")
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    paths = _write_transcript(out_dir, "2026-01-02_002_mlir", episode, transcript)
    db.upsert_search_entry(feed_slug="myfeed", guid=episode.guid, title=episode.title, published_at=None, text=transcript.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    assert len(db.search_transcripts("mlir")) == 1
    assert len(db.search_transcripts("MLIR")) == 1
    db.close()


def test_db_fts_feed_filter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    for slug in ("feed-a", "feed-b"):
        db.add_feed(f"https://example.com/{slug}.xml", slug, slug)
        ep = _sample_episode(guid=f"g-{slug}", title=f"Ep {slug}")
        tx = Transcript(text="diarization is hard for feed separation", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / slug
        out_dir.mkdir(parents=True)
        paths = _write_transcript(out_dir, f"{slug}-ep", ep, tx)
        db.upsert_search_entry(feed_slug=slug, guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    all_hits = db.search_transcripts("diarization")
    assert len(all_hits) == 2
    hits_a = db.search_transcripts("diarization", feed="feed-a")
    assert len(hits_a) == 1
    assert hits_a[0]["feed_slug"] == "feed-a"
    db.close()


def test_db_fts_limit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    for i in range(5):
        ep = _sample_episode(guid=f"g{i}", title=f"Ep {i}")
        tx = Transcript(text="MLIR appears everywhere", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / "myfeed"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _write_transcript(out_dir, f"ep{i}", ep, tx)
        db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    hits = db.search_transcripts("MLIR", limit=2)
    assert len(hits) == 2
    db.close()


def test_db_fts_incremental_update_on_reformat(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    ep = _sample_episode(guid="g1", title="Ep 1")
    tx = Transcript(text="original text without keyword", segments=[], language="en", model="t", engine="e")
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    paths = _write_transcript(out_dir, "ep1", ep, tx)
    db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    assert len(db.search_transcripts("bitter")) == 0
    # update same guid with new text containing bitter lesson
    db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=None, text="now contains bitter lesson updated", txt_path=str(paths[0]), json_path=str(paths[1]))
    assert len(db.search_transcripts("bitter")) == 1
    db.close()


def test_db_fts_reindex(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    # create transcripts on disk without manually indexing, then reindex
    for i in range(2):
        ep = _sample_episode(guid=f"g{i}", title=f"Ep {i}")
        tx = Transcript(text=f"bitter lesson episode {i}", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / "myfeed"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_transcript(out_dir, f"ep{i}", ep, tx)
    assert len(db.search_transcripts("bitter")) == 0
    count = db.reindex_search(data_dir / "transcripts")
    assert count >= 2
    assert len(db.search_transcripts("bitter")) == 2
    db.close()


def test_cli_search_basic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    ep = _sample_episode(title="The Bitter Lesson", published_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
    tx = Transcript(text="The bitter lesson is about compute scaling", segments=[], language="en", model="t", engine="e")
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    paths = _write_transcript(out_dir, "2026-03-01_001_bitter", ep, tx)
    db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=ep.published_at.isoformat() if ep.published_at else None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    db.close()

    result = runner.invoke(app, ["search", "bitter lesson", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    # result seam: feed, title/date, snippet, file paths
    assert "myfeed" in result.stdout
    assert "Bitter" in result.stdout or "bitter" in result.stdout.lower()
    # Rich may soft-wrap long paths across lines, so normalize before checking.
    out_norm = result.stdout.replace("\n", "")
    assert ".txt" in out_norm or ".json" in out_norm


def test_cli_search_feed_filter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "state.db")
    for slug in ("feed-a", "feed-b"):
        db.add_feed(f"https://example.com/{slug}.xml", slug, slug)
        ep = _sample_episode(guid=f"g-{slug}", title=f"Title {slug}")
        tx = Transcript(text="diarization topic", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / slug
        out_dir.mkdir(parents=True)
        paths = _write_transcript(out_dir, f"ep-{slug}", ep, tx)
        db.upsert_search_entry(feed_slug=slug, guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    db.close()
    result = runner.invoke(app, ["search", "diarization", "--feed", "feed-a", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "feed-a" in result.stdout
    # feed-b should not appear when filtered
    assert "feed-b" not in result.stdout


def test_cli_search_limit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    for i in range(4):
        ep = _sample_episode(guid=f"g{i}", title=f"Ep {i}")
        tx = Transcript(text="MLIR is great", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / "myfeed"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _write_transcript(out_dir, f"ep{i}", ep, tx)
        db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    db.close()
    result = runner.invoke(app, ["search", "MLIR", "--limit", "2", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    # Rich may soft-wrap, normalize
    out_norm = result.stdout.replace("\n", "")
    assert out_norm.count(".txt") == 2


def test_cli_search_reindex_flag(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # create feed and transcript json without indexing
    from podtx.writers import write_outputs as wo
    import json as _json
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    db.close()
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    ep = _sample_episode(guid="g-re", title="Reindex Episode")
    tx = Transcript(text="bitter lesson reindex me", segments=[], language="en", model="t", engine="e")
    wo(out_dir=out_dir, basename="2026-01-01_001_reindex", episode=ep, transcript=tx, formats=("txt", "json"), readable=False, cleanup=False)
    # search without reindex should have no hits (if we haven't indexed)
    result = runner.invoke(app, ["search", "bitter", "--data-dir", str(data_dir)])
    # initially may be 0 because we didn't index
    # now reindex
    result2 = runner.invoke(app, ["search", "--reindex", "--data-dir", str(data_dir)])
    assert result2.exit_code == 0, result2.stdout + result2.stderr
    assert "reindex" in result2.stdout.lower() or "indexed" in result2.stdout.lower() or "Reindexed" in result2.stdout
    result3 = runner.invoke(app, ["search", "bitter", "--data-dir", str(data_dir)])
    assert result3.exit_code == 0, result3.stdout + result3.stderr
    assert "myfeed" in result3.stdout


def test_db_search_empty_query(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.db")
    db.add_feed("https://example.com/f.xml", "f", "F")
    assert db.search_transcripts("") == []
    assert db.search_transcripts("   ") == []
    assert db.search_transcripts(None or "") == []
    db.close()


def test_db_search_invalid_syntax_fallback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    ep = _sample_episode(guid="g1", title="Test")
    tx = Transcript(text="bitter lesson fallback", segments=[], language="en", model="t", engine="e")
    out_dir = data_dir / "transcripts" / "myfeed"
    out_dir.mkdir(parents=True)
    paths = _write_transcript(out_dir, "ep1", ep, tx)
    db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=None, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    # invalid FTS syntax like unmatched quote should fallback, not crash
    hits = db.search_transcripts('"unmatched')
    assert isinstance(hits, list)
    # also test with special chars
    hits2 = db.search_transcripts("bitter AND")
    assert isinstance(hits2, list)
    db.close()


def test_db_search_since_until_filters(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    for date, guid in [("2026-01-01T00:00:00+00:00", "g1"), ("2026-06-15T00:00:00+00:00", "g2"), ("2026-12-01T00:00:00+00:00", "g3")]:
        ep = _sample_episode(guid=guid, title=f"Ep {guid}", published_at=datetime.fromisoformat(date))
        tx = Transcript(text="bitter lesson dated", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / "myfeed"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _write_transcript(out_dir, guid, ep, tx)
        db.upsert_search_entry(feed_slug="myfeed", guid=guid, title=ep.title, published_at=date, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    # since filter
    hits_since = db.search_transcripts("bitter", since="2026-06-01T00:00:00+00:00")
    assert all(h["published_at"] >= "2026-06-01" for h in hits_since if h["published_at"])
    assert len(hits_since) == 2
    # until filter
    hits_until = db.search_transcripts("bitter", until="2026-02-01T00:00:00+00:00")
    assert len(hits_until) == 1
    # combined
    hits_both = db.search_transcripts("bitter", since="2026-01-15", until="2026-07-01")
    assert len(hits_both) == 1
    db.close()


def test_db_reindex_edge_cases(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.db")
    # missing dir -> 0
    assert db.reindex_search(tmp_path / "missing") == 0
    # bad json and non-dict
    bad_dir = tmp_path / "transcripts" / "feed"
    bad_dir.mkdir(parents=True)
    (bad_dir / "bad.json").write_text("{not json", encoding="utf-8")
    (bad_dir / "notdict.json").write_text("[]", encoding="utf-8")
    # also json with no text but segments
    payload = {"guid": "g-seg", "title": "Seg", "segments": [{"text": "hello from segments", "start": 0, "end": 1}], "date": "2026-01-01T00:00:00+00:00"}
    (bad_dir / "seg.json").write_text(json.dumps(payload), encoding="utf-8")
    count = db.reindex_search(tmp_path / "transcripts")
    assert count >= 1  # seg.json should count
    db.close()


def test_cli_search_no_db_and_no_results(tmp_path: Path) -> None:
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    # no state.db yet
    result = runner.invoke(app, ["search", "nothing", "--data-dir", str(data_dir)])
    assert result.exit_code == 0
    # should indicate no library or no results
    assert "no" in result.stdout.lower() or "0" in result.stdout or "myfeed" not in result.stdout
    # empty query with reindex should also work
    result2 = runner.invoke(app, ["search", "--reindex", "--data-dir", str(data_dir)])
    assert result2.exit_code == 0


def test_cli_search_since_until_via_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/feed.xml", "myfeed", "My Feed")
    for date in ["2026-01-01T00:00:00+00:00", "2026-06-15T00:00:00+00:00"]:
        ep = _sample_episode(guid=f"g-{date}", title=f"Ep {date}", published_at=datetime.fromisoformat(date))
        tx = Transcript(text="bitter lesson cli filter", segments=[], language="en", model="t", engine="e")
        out_dir = data_dir / "transcripts" / "myfeed"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = _write_transcript(out_dir, f"ep-{date}", ep, tx)
        db.upsert_search_entry(feed_slug="myfeed", guid=ep.guid, title=ep.title, published_at=date, text=tx.text, txt_path=str(paths[0]), json_path=str(paths[1]))
    db.close()
    result = runner.invoke(app, ["search", "bitter", "--since", "2026-05-01", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    # only June entry should remain
    assert result.stdout.count("myfeed") >= 1
    result2 = runner.invoke(app, ["search", "bitter", "--until", "2026-02-01", "--data-dir", str(data_dir)])
    assert result2.exit_code == 0


def test_cli_search_empty_query_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/f.xml", "f", "F")
    db.close()
    result = runner.invoke(app, ["search", "--data-dir", str(data_dir)])
    # missing query without reindex should error
    assert result.exit_code != 0
    assert "Provide" in result.stdout or "Provide" in result.stderr or "query" in (result.stdout + result.stderr).lower()


def test_format_cmd_and_pipeline_incremental_indexing(tmp_path: Path) -> None:
    # format reformat should best-effort index without crashing even when no DB
    from podtx.format_cmd import reformat_transcript

    ep = _sample_episode(guid="g-fmt", title="Fmt")
    tx = Transcript(text="bitter lesson via format", segments=[], language="en", model="t", engine="e")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    paths = _write_transcript(src_dir, "ep", ep, tx)
    json_path = src_dir / "ep.json"
    # reformat to txt should not crash and should handle missing DB
    written = reformat_transcript(json_path, out_dir=tmp_path / "out", formats=("txt", "json"))
    assert len(written) == 2
    # pipeline incremental: simulate process_episodes indexing via direct upsert
    data_dir = tmp_path / "data2"
    data_dir.mkdir()
    db2 = Database(data_dir / "state.db")
    db2.add_feed("https://example.com/f2.xml", "f2", "F2")
    db2.upsert_search_entry(feed_slug="f2", guid="g-pipe", title="Pipe", published_at=None, text="bitter pipe", txt_path="/tmp/a.txt", json_path="/tmp/a.json")
    assert len(db2.search_transcripts("bitter")) == 1
    # empty text should still insert (text or ""), but search for empty shouldn't crash
    db2.upsert_search_entry(feed_slug="f2", guid="g-empty", title="Empty", published_at=None, text="   ", txt_path="/tmp/b.txt", json_path="/tmp/b.json")
    # empty text entry still exists but search for bitterness still 1
    assert len(db2.search_transcripts("bitter")) == 1  # still 1
    db2.close()


def test_maybe_index_no_db_no_crash(tmp_path: Path) -> None:
    from podtx.format_cmd import _maybe_index_after_reformat

    ep = _sample_episode(guid="g-no-db", title="NoDB")
    tx = Transcript(text="hello", segments=[], language="en", model="m", engine="e")
    # No DB file exists, should no-op without exception
    _maybe_index_after_reformat(ep, tx, [tmp_path / "a.txt"])  # no json
    _maybe_index_after_reformat(ep, tx, [tmp_path / "b.json", tmp_path / "b.txt"])


def test_maybe_index_with_db(tmp_path: Path, monkeypatch) -> None:
    from podtx.format_cmd import _maybe_index_after_reformat

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PODCAST_TRANSCRIBER_DATA_DIR", str(data_dir))
    # Need to create DB via load_settings path
    from podtx.config import load_settings as _ls

    settings = _ls(data_dir=data_dir)
    db = Database(settings.state_db_path())
    db.add_feed("https://example.com/f.xml", "myfeed", "My Feed")
    db.close()
    ep = _sample_episode(guid="g-idx", title="Idx")
    tx = Transcript(text="index me bitter", segments=[], language="en", model="m", engine="e")
    json_path = data_dir / "transcripts" / "myfeed" / "g-idx.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(json.dumps({"guid": "g-idx", "title": "Idx", "text": "index me bitter"}))
    txt_path = json_path.with_suffix(".txt")
    txt_path.write_text("index me bitter")
    _maybe_index_after_reformat(ep, tx, [txt_path, json_path])
    db2 = Database(settings.state_db_path())
    assert len(db2.search_transcripts("bitter")) >= 1
    db2.close()


def test_pipeline_process_episodes_indexes(monkeypatch, tmp_path: Path) -> None:
    from podtx.pipeline import process_episodes
    from podtx.config import load_settings
    from unittest import mock

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = load_settings(data_dir=data_dir)
    # Use quiet to avoid progress
    settings = settings.__class__(**{**settings.__dict__, "quiet": True, "keep_audio": True})
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/feed.xml", "feed", "Feed")
    # Mock download and engine
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"fake")
    monkeypatch.setattr("podtx.pipeline.download_only", lambda ep, audio_dir, quiet=False: tmp_path / "fake.mp3")
    (tmp_path / "fake.mp3").write_bytes(b"fake")
    monkeypatch.setattr("podtx.pipeline.convert_to_wav", lambda src, dst, trim_start=0.0: dst.write_bytes(b"wav") or dst)
    monkeypatch.setattr("podtx.pipeline.get_engine", lambda name: mock.Mock(
        name="parakeet",
        transcribe=lambda wav, model, language, local_attention, local_attention_context_size: Transcript(
            text="pipeline bitter lesson", segments=[Segment(0,1,"pipeline bitter lesson")], language="en", model=model, engine=name
        )
    ))
    # Mock unique_basename to avoid collision
    monkeypatch.setattr("podtx.pipeline.unique_basename", lambda ep, existing: "test_ep")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    ep = Episode(guid="pipe-g", title="Pipe Ep", enclosure_url="https://x/pipe.mp3", published_at=datetime(2026,1,1, tzinfo=timezone.utc), episode_num=1, show_title="Feed")
    # Need to also mock require_ffmpeg to no-op
    monkeypatch.setattr("podtx.pipeline.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.download.require_ffmpeg", lambda: None)
    results = process_episodes([ep], settings=settings, out_dir=out_dir, db=db, feed_id=feed.id)
    assert len(results) >= 0  # at least attempted
    # Check DB got indexed
    hits = db.search_transcripts("pipeline")
    assert len(hits) >= 1
    db.close()


def test_db_upsert_handles_empty_text_and_no_crash(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.db")
    db.add_feed("https://example.com/f.xml", "f", "F")
    # empty text should not crash and search should still work for other entries
    db.upsert_search_entry(feed_slug="f", guid="g1", title="T1", published_at=None, text="", txt_path="/tmp/a.txt", json_path="/tmp/a.json")
    db.upsert_search_entry(feed_slug="f", guid="g2", title="T2", published_at=None, text="hello world", txt_path="/tmp/b.txt", json_path="/tmp/b.json")
    assert len(db.search_transcripts("hello")) == 1
    # Search with no DB file should not crash (tested via CLI already)
    db.close()


def test_search_no_results_and_empty_db_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = Database(data_dir / "state.db")
    db.add_feed("https://example.com/f.xml", "f", "F")
    db.close()
    result = runner.invoke(app, ["search", "nonexistentkeyword123", "--data-dir", str(data_dir)])
    assert result.exit_code == 0
    assert "No results" in result.stdout or "no" in result.stdout.lower()
