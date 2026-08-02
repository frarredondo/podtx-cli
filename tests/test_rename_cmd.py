from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.db import Database
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()


def _write_episode(
    out_dir: Path,
    *,
    basename: str,
    title: str,
    episode_num: int | None,
    guid: str = "g1",
    extensions: tuple[str, ...] = ("txt", "json"),
) -> Path:
    episode = Episode(
        guid=guid,
        title=title,
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=episode_num,
        show_title="Demo Show",
    )
    transcript = Transcript(
        text="Hello world.",
        segments=[Segment(0.0, 1.0, "Hello world.")],
        language="en",
        model="test-model",
        engine="fake",
    )
    write_outputs(
        out_dir=out_dir,
        basename=basename,
        episode=episode,
        transcript=transcript,
        formats=extensions,
        readable=False,
        cleanup=False,
    )
    return out_dir / f"{basename}.json"


def test_plan_rename_from_title_for_zero_padded_file(tmp_path: Path) -> None:
    from podtx.rename_cmd import plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_937-is-the-omarchy-hype-real",
        title="937: Is The Omarchy Hype Real?",
        episode_num=0,
        extensions=("txt", "json", "srt"),
    )
    plan = plan_rename_from_title(json_path)
    assert plan is not None
    assert plan.episode_num == 937
    assert plan.new_basename == "2026-03-15_937_937-is-the-omarchy-hype-real"
    srcs = {src.name for src, _ in plan.moves}
    dests = {dest.name for _, dest in plan.moves}
    assert srcs == {
        "2026-03-15_000_937-is-the-omarchy-hype-real.json",
        "2026-03-15_000_937-is-the-omarchy-hype-real.txt",
        "2026-03-15_000_937-is-the-omarchy-hype-real.srt",
    }
    assert dests == {
        "2026-03-15_937_937-is-the-omarchy-hype-real.json",
        "2026-03-15_937_937-is-the-omarchy-hype-real.txt",
        "2026-03-15_937_937-is-the-omarchy-hype-real.srt",
    }


def test_plan_rename_skips_section_style_titles(tmp_path: Path) -> None:
    from podtx.rename_cmd import plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_1-1-introduction",
        title="1.1 - Introduction",
        episode_num=None,
    )
    assert plan_rename_from_title(json_path) is None


def test_plan_rename_skips_when_already_numbered(tmp_path: Path) -> None:
    from podtx.rename_cmd import plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_042_episode-forty-two",
        title="Episode Forty Two",
        episode_num=42,
    )
    assert plan_rename_from_title(json_path) is None


def test_plan_rename_prefers_json_episode_over_title(tmp_path: Path) -> None:
    from podtx.rename_cmd import plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_wrong-title-number",
        title="999: Wrong Title Number",
        episode_num=42,
    )
    plan = plan_rename_from_title(json_path)
    assert plan is not None
    assert plan.episode_num == 42
    assert plan.new_basename.startswith("2026-03-15_042_")


def test_plan_rename_refuses_collision(tmp_path: Path) -> None:
    from podtx.rename_cmd import RenameError, plan_rename_from_title

    _write_episode(
        tmp_path,
        basename="2026-03-15_937_937-is-the-omarchy-hype-real",
        title="937: Already Exists",
        episode_num=937,
        guid="existing",
    )
    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_937-is-the-omarchy-hype-real",
        title="937: Is The Omarchy Hype Real?",
        episode_num=0,
        guid="new",
    )
    try:
        plan_rename_from_title(json_path)
        assert False, "expected RenameError"
    except RenameError as exc:
        assert "already exists" in str(exc).lower()


def test_apply_rename_updates_files_and_json_episode(tmp_path: Path) -> None:
    from podtx.rename_cmd import apply_rename, plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_ep-25-demo",
        title="Ep 25: Demo",
        episode_num=None,
        extensions=("txt", "json"),
    )
    plan = plan_rename_from_title(json_path)
    assert plan is not None
    new_json = apply_rename(plan, dry_run=False)
    assert new_json.name == "2026-03-15_025_ep-25-demo.json"
    assert new_json.is_file()
    assert not json_path.exists()
    assert (tmp_path / "2026-03-15_025_ep-25-demo.txt").is_file()
    assert not (tmp_path / "2026-03-15_000_ep-25-demo.txt").exists()
    payload = json.loads(new_json.read_text(encoding="utf-8"))
    assert payload["episode"] == 25


def test_apply_rename_dry_run_does_not_touch_files(tmp_path: Path) -> None:
    from podtx.rename_cmd import apply_rename, plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_ep-9-demo",
        title="Ep 9 — Demo",
        episode_num=0,
    )
    plan = plan_rename_from_title(json_path)
    assert plan is not None
    apply_rename(plan, dry_run=True)
    assert json_path.is_file()
    assert not (tmp_path / "2026-03-15_009_ep-9-demo.json").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["episode"] in (0, None)


def test_apply_rename_updates_db_when_tracked(tmp_path: Path) -> None:
    from podtx.rename_cmd import apply_rename, plan_rename_from_title

    feed_dir = tmp_path / "transcripts" / "syntax"
    feed_dir.mkdir(parents=True)
    json_path = _write_episode(
        feed_dir,
        basename="2026-03-15_000_episode-860-module-federation",
        title="Episode 860: Module Federation",
        episode_num=None,
        guid="syntax-860",
    )
    db = Database(tmp_path / "state.db")
    feed = db.add_feed("https://example.com/syntax.xml", "syntax", "Syntax")
    db.upsert_episode(
        feed_id=feed.id,
        guid="syntax-860",
        title="Episode 860: Module Federation",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=None,
        enclosure_url="https://example.com/a.mp3",
    )
    db.mark_done(
        feed_id=feed.id,
        guid="syntax-860",
        engine="fake",
        model="test",
        output_paths=[json_path, json_path.with_suffix(".txt")],
    )

    plan = plan_rename_from_title(json_path)
    assert plan is not None
    new_json = apply_rename(plan, dry_run=False, db=db)
    row = db.list_episodes(feed.id)[0]
    assert row["episode_num"] == 860
    paths = json.loads(row["output_paths_json"])
    assert str(new_json) in paths
    assert str(new_json.with_suffix(".txt")) in paths
    assert all("_000_" not in p for p in paths)
    db.close()


def test_rename_many_reports_ok_skipped_failed(tmp_path: Path) -> None:
    from podtx.rename_cmd import rename_many_from_title

    root = tmp_path / "transcripts" / "feed-a"
    root.mkdir(parents=True)
    fixable = _write_episode(
        root,
        basename="2026-03-15_000_937-fixable",
        title="937: Fixable",
        episode_num=0,
        guid="fixable",
    )
    _write_episode(
        root,
        basename="2026-03-15_000_1-1-section",
        title="1.1 - Section",
        episode_num=None,
        guid="section",
    )
    # Collision target for a third file
    _write_episode(
        root,
        basename="2026-03-15_100_100-already-there",
        title="100: Already There",
        episode_num=100,
        guid="exists",
    )
    colliding = _write_episode(
        root,
        basename="2026-03-15_000_100-already-there",
        title="100: Already There",
        episode_num=0,
        guid="collide",
    )

    result = rename_many_from_title(
        [fixable, root / "2026-03-15_000_1-1-section.json", colliding],
        dry_run=False,
    )
    assert result.ok == 1
    assert result.skipped == 1
    assert result.failed == 1
    assert (root / "2026-03-15_937_937-fixable.json").is_file()
    assert not fixable.exists()


def test_cli_rename_from_title_feed_dry_run(tmp_path: Path) -> None:
    feed_dir = tmp_path / "transcripts" / "syntax"
    feed_dir.mkdir(parents=True)
    _write_episode(
        feed_dir,
        basename="2026-03-15_000_937-is-the-omarchy-hype-real",
        title="937: Is The Omarchy Hype Real?",
        episode_num=0,
    )
    result = runner.invoke(
        app,
        [
            "rename",
            "--from-title",
            "--feed",
            "syntax",
            "--dry-run",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "dry-run" in result.stdout.lower() or "would rename" in result.stdout.lower()
    assert (feed_dir / "2026-03-15_000_937-is-the-omarchy-hype-real.json").is_file()
    assert not (feed_dir / "2026-03-15_937_937-is-the-omarchy-hype-real.json").exists()


def test_cli_rename_from_title_all(tmp_path: Path) -> None:
    for slug in ("feed-a", "feed-b"):
        feed_dir = tmp_path / "transcripts" / slug
        feed_dir.mkdir(parents=True)
        _write_episode(
            feed_dir,
            basename=f"2026-03-15_000_{slug}-ep",
            title=f"12: {slug}",
            episode_num=None,
            guid=slug,
        )
    result = runner.invoke(
        app,
        ["rename", "--from-title", "--all", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "2 ok" in result.stdout
    assert (tmp_path / "transcripts" / "feed-a" / "2026-03-15_012_12-feed-a.json").is_file()
    assert (tmp_path / "transcripts" / "feed-b" / "2026-03-15_012_12-feed-b.json").is_file()


def test_cli_rename_requires_from_title_and_scope(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rename", "--all"])
    assert result.exit_code != 0
    result2 = runner.invoke(app, ["rename", "--from-title"])
    assert result2.exit_code != 0


def test_plan_rename_returns_none_when_no_sibling_files(tmp_path: Path, monkeypatch) -> None:
    from podtx.rename_cmd import plan_rename_from_title

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_ep-7-lonely",
        title="Ep 7: Lonely",
        episode_num=None,
    )

    def never_file(self: Path) -> bool:
        return False

    monkeypatch.setattr(Path, "is_file", never_file)
    assert plan_rename_from_title(json_path) is None


def test_rename_many_records_apply_errors(tmp_path: Path, monkeypatch) -> None:
    from podtx import rename_cmd

    json_path = _write_episode(
        tmp_path,
        basename="2026-03-15_000_ep-8-boom",
        title="Ep 8: Boom",
        episode_num=None,
    )

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rename_cmd, "apply_rename", boom)
    result = rename_cmd.rename_many_from_title([json_path], dry_run=False)
    assert result.ok == 0
    assert result.failed == 1
    assert "disk full" in result.errors[0][1]


def test_cli_rename_unknown_feed_exits(tmp_path: Path) -> None:
    (tmp_path / "transcripts").mkdir()
    result = runner.invoke(
        app,
        ["rename", "--from-title", "--feed", "missing", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_cli_rename_no_targets_exits(tmp_path: Path) -> None:
    (tmp_path / "transcripts").mkdir()
    result = runner.invoke(
        app,
        ["rename", "--from-title", "--all", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "no transcript" in (result.stdout + result.stderr).lower()


def test_cli_rename_reports_skips_and_failures(tmp_path: Path) -> None:
    feed_dir = tmp_path / "transcripts" / "feed-a"
    feed_dir.mkdir(parents=True)
    _write_episode(
        feed_dir,
        basename="2026-03-15_000_12-fixable",
        title="12: Fixable",
        episode_num=0,
        guid="fixable",
    )
    _write_episode(
        feed_dir,
        basename="2026-03-15_000_no-number",
        title="No Number Here",
        episode_num=None,
        guid="skip",
    )
    _write_episode(
        feed_dir,
        basename="2026-03-15_100_100-already-there",
        title="100: Already There",
        episode_num=100,
        guid="exists",
    )
    _write_episode(
        feed_dir,
        basename="2026-03-15_000_100-already-there",
        title="100: Already There",
        episode_num=0,
        guid="collide",
    )
    result = runner.invoke(
        app,
        ["rename", "--from-title", "--feed", "feed-a", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    out = result.stdout + result.stderr
    assert "Skipped" in out
    assert "Failed" in out
    # Rich may soft-wrap the summary line in narrow consoles.
    assert "1 ok" in out
    assert "skipped" in out
    assert "failed" in out


def test_cli_rename_quiet_suppresses_progress(tmp_path: Path) -> None:
    feed_dir = tmp_path / "transcripts" / "feed-a"
    feed_dir.mkdir(parents=True)
    _write_episode(
        feed_dir,
        basename="2026-03-15_000_9-quiet",
        title="9: Quiet",
        episode_num=None,
    )
    result = runner.invoke(
        app,
        [
            "rename",
            "--from-title",
            "--feed",
            "feed-a",
            "--data-dir",
            str(tmp_path),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Renamed" not in result.stdout
    assert "Done" not in result.stdout
    assert (feed_dir / "2026-03-15_009_9-quiet.json").is_file()
