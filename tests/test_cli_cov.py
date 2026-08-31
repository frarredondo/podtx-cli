"""Coverage-fill tests for the remaining uncovered lines/branches in podtx.cli."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx import __version__
from podtx.cli import (
    _error_detail,
    _looks_like_audio_url,
    _looks_like_url,
    _merge_formats,
    _parse_output_paths,
    _transcript_disk_size,
    app,
)
from podtx.config import load_settings
from podtx.db import Database
from podtx.download import FFmpegNotFoundError
from podtx.models import Episode, Segment, Transcript
from podtx.rss import FeedParseError
from podtx.writers import write_outputs

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def _transcript_file(dir_path: Path, basename: str = "ep") -> Path:
    """Write a transcript JSON sidecar and return its path."""
    episode = Episode(
        guid=f"g-{basename}",
        title=f"Ep {basename}",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=1,
        show_title="Demo",
    )
    transcript = Transcript(
        text="So uh yeah. The the fox.",
        segments=[
            Segment(0.0, 1.0, "So uh yeah."),
            Segment(2.0, 3.0, "The the fox."),
        ],
        language="en",
        model="test-model",
        engine="fake",
    )
    write_outputs(
        out_dir=dir_path,
        basename=basename,
        episode=episode,
        transcript=transcript,
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )
    return dir_path / f"{basename}.json"


def _seed_feed(tmp_path: Path, *, slug: str = "demo", title: str = "Demo") -> Database:
    settings = load_settings(data_dir=tmp_path)
    db = Database(settings.state_db_path())
    feed = db.add_feed("https://example.com/feed.xml", slug, title)
    db.close()
    return slug, feed.id


# ---- main_callback / version ----


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_no_args_prints_help_outcome() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.stdout


# ---- pure helpers ----


def test_looks_like_url_branches() -> None:
    assert _looks_like_url("https://example.com/a.mp3")
    assert _looks_like_url("http://example.com/x")
    assert not _looks_like_url("notaurl")
    assert not _looks_like_url("https://")  # scheme but no netloc
    assert not _looks_like_url("")


def test_looks_like_audio_url_branches() -> None:
    assert _looks_like_audio_url("https://example.com/a.mp3")
    assert _looks_like_audio_url("https://example.com/a.M4A")  # path lowered
    assert _looks_like_audio_url("https://example.com/a.mp4")
    assert not _looks_like_audio_url("https://example.com/a.txt")
    assert not _looks_like_audio_url("notaurl.mp3")  # not a URL at all


def test_merge_formats_branches() -> None:
    base = ("txt", "json")
    assert _merge_formats(base, None) == base
    assert _merge_formats(base, []) == base
    assert _merge_formats(base, ["srt"]) == ("txt", "json", "srt")
    assert _merge_formats(base, ["vtt", "srt"]) == ("txt", "json", "vtt", "srt")
    # explicit txt/json in the requested list overrides the defaults
    assert _merge_formats(base, ["JSON", "txt"]) == ("json", "txt")
    assert _merge_formats(base, ["md", "json"]) == ("md", "json")


def test_transcript_disk_size_skips_subdirs(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    tdir = settings.transcripts_dir("feed-x")
    tdir.mkdir(parents=True)
    (tdir / "sub").mkdir()
    (tdir / "a.txt").write_text("hello")  # directories are not files
    assert _transcript_disk_size(settings, "feed-x") == 5


def test_error_detail_branches() -> None:
    assert _error_detail(None) == "unknown error"
    assert _error_detail("") == "unknown error"
    assert _error_detail("not-json") == "unknown error"  # JSONDecodeError path
    assert _error_detail("[]") == "unknown error"  # payload not a dict
    assert _error_detail('{"error": ""}') == "unknown error"  # empty message
    assert _error_detail('{"error": "boom"}') == "boom"


def test_parse_output_paths_branches() -> None:
    assert _parse_output_paths(None) == []
    assert _parse_output_paths("") == []
    assert _parse_output_paths("not-json") == []  # JSONDecodeError path
    assert _parse_output_paths('["a", "b"]') == ["a", "b"]
    assert _parse_output_paths('{"a": 1}') == []  # not a list


# ---- add / remove ----


def test_add_feed_parse_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["add", "not-a-url", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error" in result.stdout + result.stderr


def test_add_feed_already_registered(tmp_path: Path) -> None:
    xml_url = FIXTURE.resolve().as_uri()
    first = runner.invoke(app, ["add", xml_url, "--data-dir", str(tmp_path)])
    assert first.exit_code == 0, first.stdout
    second = runner.invoke(app, ["add", xml_url, "--data-dir", str(tmp_path)])
    assert second.exit_code == 1
    assert "already registered" in (second.stdout + second.stderr).lower()


def test_remove_feed_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["remove", "nothing", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "not found" in (result.stdout + result.stderr).lower()


# ---- doctor ----


def test_doctor_db_exists_with_no_feeds(tmp_path: Path) -> None:
    settings = load_settings(data_dir=tmp_path)
    Database(settings.state_db_path()).close()
    result = runner.invoke(app, ["doctor", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No feeds registered" in result.stdout


# ---- sync ----


def test_sync_happy_path(tmp_path: Path, monkeypatch) -> None:
    _seed_feed(tmp_path)
    ep = Episode(
        guid="g1",
        title="Ep 1",
        enclosure_url="https://x/1.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=1,
        description="desc",
        link="https://l",
        show_title="Demo",
    )
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("Demo", "demo", [ep]))
    monkeypatch.setattr(
        "podtx.cli.select_episodes_for_sync",
        lambda episodes, done_guids, limit, process_all: list(episodes),
    )
    calls: list[tuple[list[Episode], Path, int]] = []
    monkeypatch.setattr(
        "podtx.cli.process_episodes",
        lambda selected, settings, out_dir, db, feed_id: calls.append((list(selected), out_dir, feed_id)),
    )
    result = runner.invoke(
        app,
        ["sync", "--data-dir", str(tmp_path), "--limit", "2", "--format", "txt"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert calls, "process_episodes was not called"
    selected, out_dir, feed_id = calls[0]
    assert selected[0].guid == "g1"
    assert selected[0].show_title == "Demo"  # show title reattached from feed
    assert "transcribing 1 episode(s) with parakeet" in result.stdout.lower()


def test_sync_happy_path_quiet(tmp_path: Path, monkeypatch) -> None:
    _seed_feed(tmp_path)
    ep = Episode(guid="g1", title="Ep 1", enclosure_url="https://x/1.mp3", show_title="Demo")
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("Demo", "demo", [ep]))
    monkeypatch.setattr("podtx.cli.select_episodes_for_sync", lambda episodes, done_guids, limit, process_all: list(episodes))
    monkeypatch.setattr("podtx.cli.process_episodes", lambda selected, settings, out_dir, db, feed_id: None)
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path), "-q"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "transcribing" not in result.stdout.lower()


def test_sync_nothing_new_quiet_and_loud(tmp_path: Path, monkeypatch) -> None:
    _seed_feed(tmp_path)
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("Demo", "demo", []))
    monkeypatch.setattr("podtx.cli.select_episodes_for_sync", lambda episodes, **kw: [])
    monkeypatch.setattr("podtx.cli.process_episodes", lambda selected, **kw: None)

    quiet = runner.invoke(app, ["sync", "--data-dir", str(tmp_path), "-q"])
    assert quiet.exit_code == 0, quiet.stdout + quiet.stderr
    loud = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert loud.exit_code == 0, loud.stdout + loud.stderr
    assert "nothing new" in loud.stdout.lower()


def test_sync_feed_arg_found_and_not_found(tmp_path: Path, monkeypatch) -> None:
    slug, feed_id = _seed_feed(tmp_path)
    ep = Episode(guid="g1", title="Ep 1", enclosure_url="https://x/1.mp3", show_title="Demo")
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("Demo", "demo", [ep]))
    monkeypatch.setattr("podtx.cli.select_episodes_for_sync", lambda episodes, done_guids, limit, process_all: list(episodes))
    seen: dict[int, int] = {}
    monkeypatch.setattr(
        "podtx.cli.process_episodes",
        lambda selected, settings, out_dir, db, feed_id: seen.__setitem__(feed_id, feed_id),
    )
    found = runner.invoke(app, ["sync", slug, "--data-dir", str(tmp_path)])
    assert found.exit_code == 0, found.stdout + found.stderr
    assert seen.get(feed_id) == feed_id

    missing = runner.invoke(app, ["sync", "bogus", "--data-dir", str(tmp_path)])
    assert missing.exit_code == 1
    assert "not found" in (missing.stdout + missing.stderr).lower()


def test_sync_no_feeds_registered(tmp_path: Path) -> None:
    Database(load_settings(data_dir=tmp_path).state_db_path()).close()
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No feeds registered" in (result.stdout + result.stderr)


def test_sync_unknown_engine(tmp_path: Path) -> None:
    _seed_feed(tmp_path)
    result = runner.invoke(
        app, ["sync", "--engine", "bogus-engine", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "Unknown engine" in (result.stdout + result.stderr)


def test_sync_ffmpeg_missing(tmp_path: Path, monkeypatch) -> None:
    _seed_feed(tmp_path)
    monkeypatch.setattr(
        "podtx.cli.require_ffmpeg", lambda: (_ for _ in ()).throw(FFmpegNotFoundError())
    )
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "ffmpeg" in (result.stdout + result.stderr).lower()


def test_sync_feed_parse_error_continues(tmp_path: Path, monkeypatch) -> None:
    _seed_feed(tmp_path)
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "podtx.cli.parse_feed", lambda url: (_ for _ in ()).throw(FeedParseError("bad xml"))
    )
    monkeypatch.setattr("podtx.cli.select_episodes_for_sync", lambda episodes, **kw: [])
    result = runner.invoke(app, ["sync", "--data-dir", str(tmp_path), "-q"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Failed to parse" in result.stderr


# ---- transcribe ----


def test_transcribe_local_file(tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "sample.mp3"
    local.write_bytes(b"fake")
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr(
        "podtx.cli.transcribe_local_file",
        lambda local, settings, out_dir: [out_dir / "sample.txt"],
    )
    result = runner.invoke(app, ["transcribe", str(local)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Wrote" in result.stdout


def test_transcribe_not_file_or_url(tmp_path: Path) -> None:
    result = runner.invoke(app, ["transcribe", "not-a-file-or-url"])
    assert result.exit_code == 1
    assert "Not a file or URL" in (result.stdout + result.stderr)


def test_transcribe_audio_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    calls: list[list[Episode]] = []
    monkeypatch.setattr(
        "podtx.cli.process_episodes",
        lambda episodes, settings, out_dir: calls.append(list(episodes)),
    )
    url = "https://example.com/audio/ep.mp3"
    result = runner.invoke(app, ["transcribe", url, "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    ep = calls[0][0]
    assert ep.guid == url
    assert ep.enclosure_url == url
    assert ep.title == "ep"


def test_transcribe_rss_default_limit_one(tmp_path: Path, monkeypatch) -> None:
    eps = [
        Episode(guid=f"g{i}", title=f"Ep {i}", enclosure_url=f"https://x/{i}.mp3")
        for i in range(3)
    ]
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("My Show", "my-show", eps))
    calls: list[list[Episode]] = []
    monkeypatch.setattr(
        "podtx.cli.process_episodes",
        lambda episodes, settings, out_dir: calls.append(list(episodes)),
    )
    result = runner.invoke(app, ["transcribe", "https://example.com/feed.xml"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(calls[0]) == 1
    assert calls[0][0].guid == "g0"
    assert "My Show" in result.stdout  # not quiet


def test_transcribe_rss_all_and_limit(tmp_path: Path, monkeypatch) -> None:
    eps = [Episode(guid=f"g{i}", title=f"Ep {i}", enclosure_url=f"https://x/{i}.mp3") for i in range(3)]
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("My Show", "my-show", eps))
    calls: list[list[Episode]] = []
    monkeypatch.setattr(
        "podtx.cli.process_episodes",
        lambda episodes, settings, out_dir: calls.append(list(episodes)),
    )
    all_res = runner.invoke(app, ["transcribe", "https://example.com/feed.xml", "--all"])
    assert all_res.exit_code == 0, all_res.stdout + all_res.stderr
    assert len(calls[-1]) == 3

    quiet_res = runner.invoke(app, ["transcribe", "https://example.com/feed.xml", "-q"])
    assert quiet_res.exit_code == 0, quiet_res.stdout + quiet_res.stderr
    assert "episode(s)" not in quiet_res.stdout

    limit_res = runner.invoke(app, ["transcribe", "https://example.com/feed.xml", "--limit", "2"])
    assert limit_res.exit_code == 0, limit_res.stdout + limit_res.stderr
    assert len(calls[-1]) == 2


def test_transcribe_rss_quiet_no_episodes_and_parse_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("podtx.cli.require_ffmpeg", lambda: None)
    monkeypatch.setattr("podtx.cli.parse_feed", lambda url: ("Show", "slug", []))
    monkeypatch.setattr("podtx.cli.process_episodes", lambda selected, **kw: None)
    quiet = runner.invoke(app, ["transcribe", "https://example.com/feed.xml", "-q"])
    assert quiet.exit_code == 1
    assert "No episodes with audio enclosures" in quiet.stderr

    monkeypatch.setattr(
        "podtx.cli.parse_feed", lambda url: (_ for _ in ()).throw(FeedParseError("bad xml"))
    )
    parse_err = runner.invoke(app, ["transcribe", "https://example.com/feed.xml"])
    assert parse_err.exit_code == 1
    assert "Error" in parse_err.stderr


def test_transcribe_ffmpeg_missing(tmp_path: Path) -> None:
    import podtx.cli as cli

    orig = cli.require_ffmpeg
    cli.require_ffmpeg = lambda: (_ for _ in ()).throw(FFmpegNotFoundError())
    try:
        result = runner.invoke(app, ["transcribe", "anything"])
    finally:
        cli.require_ffmpeg = orig
    assert result.exit_code == 1
    assert "ffmpeg" in (result.stdout + result.stderr).lower()


# ---- format ----


def test_format_json_path_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["format", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "File not found" in (result.stdout + result.stderr)


def test_format_single_file_quiet(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    json_path = _transcript_file(src)
    result = runner.invoke(
        app,
        ["format", str(json_path), "--correct-names", "-q", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr


def test_format_single_file_corrections_report_not_quiet(tmp_path: Path) -> None:
    src = tmp_path / "src2"
    src.mkdir()
    json_path = _transcript_file(src)
    result = runner.invoke(
        app,
        ["format", str(json_path), "--correct-names", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Wrote" in result.stdout


def test_format_single_file_transcript_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    result = runner.invoke(app, ["format", str(bad)])
    assert result.exit_code == 1
    assert "transcript json" in (result.stdout + result.stderr).lower()


def test_format_feed_missing_and_empty(tmp_path: Path) -> None:
    root = load_settings(data_dir=tmp_path).transcripts_dir()
    (root / "empty-feed").mkdir(parents=True)
    missing = runner.invoke(app, ["format", "--feed", "ghost", "--data-dir", str(tmp_path)])
    assert missing.exit_code == 1
    assert "not found" in (missing.stdout + missing.stderr).lower()

    empty = runner.invoke(app, ["format", "--feed", "empty-feed", "--data-dir", str(tmp_path)])
    assert empty.exit_code == 1
    assert "No transcript JSON" in (empty.stdout + empty.stderr)


def test_format_feed_quiet_success_and_failure(tmp_path: Path) -> None:
    root = load_settings(data_dir=tmp_path).transcripts_dir()
    good = root / "good-feed"
    good.mkdir(parents=True)
    _transcript_file(good)
    ok = runner.invoke(app, ["format", "--feed", "good-feed", "-q", "--data-dir", str(tmp_path)])
    assert ok.exit_code == 0, ok.stdout + ok.stderr

    bad = root / "bad-feed"
    bad.mkdir()
    (bad / "broken.json").write_text("{broken", encoding="utf-8")
    fail = runner.invoke(app, ["format", "--feed", "bad-feed", "-q", "--data-dir", str(tmp_path)])
    assert fail.exit_code == 1


# ---- summarize ----


def test_summarize_feed_quiet(tmp_path: Path) -> None:
    root = load_settings(data_dir=tmp_path).transcripts_dir()
    feed_dir = root / "sum-feed"
    feed_dir.mkdir(parents=True)
    _transcript_file(feed_dir, basename="ep1")
    result = runner.invoke(
        app,
        ["summarize", "--feed", "sum-feed", "-q", "--backend", "fake", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (feed_dir / "ep1.summary.json").is_file()


def test_summarize_missing_transcript_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["summarize", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "File not found" in (result.stdout + result.stderr)


# ---- auth sanitization ----


def test_auth_set_sanitizes_quoted_key(monkeypatch) -> None:
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        "podtx.keychain.save_api_key", lambda svc, acct, secret: saved.__setitem__("secret", secret)
    )
    result = runner.invoke(app, ["auth", "set", "openrouter", "--api-key", '"sk-quoted"'])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert saved["secret"] == "sk-quoted"
    assert "Sanitized pasted key" in result.stdout


def test_auth_set_sanitize_to_empty_errors(monkeypatch) -> None:
    monkeypatch.setattr("podtx.keychain.save_api_key", lambda *a, **k: None)
    result = runner.invoke(app, ["auth", "set", "openrouter", "--api-key", " "])
    assert result.exit_code == 1
    assert "No key after sanitizing" in result.stdout + result.stderr
    assert "Sanitized pasted key" in result.stdout


def test_auth_set_bracketed_paste_artifacts(monkeypatch) -> None:
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        "podtx.keychain.save_api_key", lambda svc, acct, secret: saved.__setitem__("secret", secret)
    )
    result = runner.invoke(
        app, ["auth", "set", "openrouter", "--api-key", "\x1b[200~sk-paste\x1b[201~"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert saved["secret"] == "sk-paste"


# ---- rss datetime string path (last gap module-wide) ----


def test_rss_to_datetime_parsedate_tz_aware_string() -> None:
    from podtx.rss import _to_datetime

    dt = _to_datetime("Sat, 15 Mar 2026 12:00:00 GMT")
    assert dt is not None
    assert dt.tzinfo is not None
    tz_naive = _to_datetime("2026-03-15T12:00:00")  # falls to fromisoformat
    assert tz_naive is not None
    assert tz_naive.tzinfo is not None  # normalized to UTC


# ---- search ----


def test_search_reindex_with_query(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "--reindex", "bitter", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr


def test_search_hit_without_snippet_or_paths(tmp_path: Path, monkeypatch) -> None:
    Database(load_settings(data_dir=tmp_path).state_db_path()).close()
    monkeypatch.setattr(
        Database,
        "search_transcripts",
        lambda self, *a, **k: [{
            "title": "Bare",
            "feed_slug": "f",
            "published_at": "",
            "snippet": "",
            "txt_path": "",
            "json_path": "",
        }],
    )
    result = runner.invoke(app, ["search", "bare", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Bare" in result.stdout