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
