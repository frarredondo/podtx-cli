from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.nuggets import NUGGETS_PROMPT_VERSION
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()


def _episode() -> Episode:
    return Episode(
        guid="fake-guid-1",
        title="Fake Episode Title",
        enclosure_url="https://example.com/ep.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=42,
        show_title="Fake Show",
        link="https://example.com/ep",
    )


def _transcript() -> Transcript:
    return Transcript(
        text="First sentence is overview. Second sentence also overview. Third is a key point.",
        segments=[
            Segment(0.0, 1.5, "First sentence is overview."),
            Segment(2.0, 3.5, "Second sentence also overview."),
        ],
        language="en",
        model="fake-model",
        engine="fake",
    )


def _nugget(insight, quote, total=6, **over) -> dict:
    n = {
        "insight": insight,
        "context": "Fake Show — ep",
        "why_it_matters": "why it matters",
        "quote": quote,
        "scores": {"T": 2, "S": 2, "E": 1, "A": 1},
        "total": total,
        "tag": "eng",
    }
    n.update(over)
    return n


def _sidecar(title, basename, nuggets) -> dict:
    return {
        "title": title,
        "show": "Fake Show",
        "episode": 1,
        "guid": f"guid-{basename}",
        "backend": "fake",
        "model": None,
        "prompt_version": NUGGETS_PROMPT_VERSION,
        "generated_at": "2026-08-30T00:00:00+00:00",
        "nuggets": nuggets,
    }


def _write_transcript(feed_root: Path, basename: str) -> Path:
    write_outputs(
        out_dir=feed_root,
        basename=basename,
        episode=_episode(),
        transcript=_transcript(),
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )
    return feed_root / f"{basename}.json"


def _make_feed(tmp_path: Path, slug: str = "fakefeed", names=("ep1", "ep2")) -> Path:
    feed_root = tmp_path / "transcripts" / slug
    feed_root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        _write_transcript(feed_root, name)
        sidecar = feed_root / f"{name}.nuggets.json"
        nuggets = _nugget(
            insight=f"Ship small diffs so review is fast episode {i}",
            quote="Small diffs review fast",
            total=6 + i,
        )
        if i == 1:
            nuggets = _nugget(
                insight="Shipping small diffs makes review fast.",
                quote="Small diffs review fast.",
                total=8,
                scores={"T": 2, "S": 2, "E": 2, "A": 2},
            )
        sidecar.write_text(
            json.dumps(_sidecar(f"Ep {i + 1}", name, [nuggets])), encoding="utf-8"
        )
    return feed_root


def test_merge_feed_writes_corpus(tmp_path, monkeypatch) -> None:
    feed_root = _make_feed(tmp_path)
    result = runner.invoke(
        app, ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (feed_root / "corpus.nuggets.json").exists()
    assert (feed_root / "corpus.nuggets.md").exists()
    corpus = json.loads((feed_root / "corpus.nuggets.json").read_text(encoding="utf-8"))
    assert corpus["episodes_processed"] == 2
    assert corpus["clustering"] == "offline"
    assert "Merged" in result.stdout
    assert "offline" in result.stdout


def test_merge_all_writes_to_library_root(tmp_path) -> None:
    _make_feed(tmp_path, slug="a")
    _make_feed(tmp_path, slug="b")
    root = tmp_path / "transcripts"
    result = runner.invoke(app, ["nuggets", "--all", "--data-dir", str(tmp_path), "--merge"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (root / "corpus.nuggets.json").exists()


def test_merge_no_sidecars(tmp_path) -> None:
    feed_root = tmp_path / "transcripts" / "fakefeed"
    feed_root.mkdir(parents=True, exist_ok=True)
    _write_transcript(feed_root, "ep1")
    result = runner.invoke(
        app, ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge"]
    )
    assert result.exit_code == 0
    assert "No nugget sidecar files found" in result.stdout
    assert not (feed_root / "corpus.nuggets.json").exists()


def test_merge_rejects_json_path(tmp_path) -> None:
    path = _make_feed(tmp_path) / "ep1.json"
    result = runner.invoke(app, ["nuggets", str(path), "--merge"])
    assert result.exit_code != 0
    assert "exactly one of" in (result.stdout + result.stderr).lower()


def test_merge_rejects_dry_run(tmp_path) -> None:
    _make_feed(tmp_path)
    result = runner.invoke(
        app,
        ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in (result.stdout + result.stderr).lower()


def test_merge_semantic_cli(tmp_path, monkeypatch) -> None:
    feed_root = _make_feed(tmp_path)
    calls = {}

    class _Stub:
        name = "lmstudio"

        def complete(self, messages, *, timeout=120.0, temperature=0.3) -> str:
            calls["timeout"] = timeout
            calls["temperature"] = temperature
            return json.dumps({"groups": [{"ids": [0, 1], "label": "small diffs"}]})

    monkeypatch.setattr("podtx.cli.build_provider", lambda *a, **k: _Stub())
    result = runner.invoke(
        app,
        [
            "nuggets",
            "--feed",
            "fakefeed",
            "--data-dir",
            str(tmp_path),
            "--merge",
            "--backend",
            "lmstudio",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "semantic" in result.stdout
    corpus = json.loads((feed_root / "corpus.nuggets.json").read_text(encoding="utf-8"))
    assert corpus["clustering"] == "semantic"
    assert len(corpus["groups"]) == 1


def test_merge_skipped_sidecar_disclosed(tmp_path) -> None:
    feed_root = _make_feed(tmp_path, names=("ep1", "ep2"))
    bad = feed_root / "ep2.nuggets.json"
    bad.write_text("{broken", encoding="utf-8")
    result = runner.invoke(
        app, ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge"]
    )
    assert result.exit_code == 0
    assert "1" in result.stdout
    assert "skipped" in result.stdout.lower()
    corpus = json.loads((feed_root / "corpus.nuggets.json").read_text(encoding="utf-8"))
    assert corpus["sidecars_skipped"] == [str(bad)]


def test_merge_missing_feed_dir_errors(tmp_path) -> None:
    result = runner.invoke(
        app, ["nuggets", "--feed", "nope", "--data-dir", str(tmp_path), "--merge"]
    )
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_merge_out_dir_override(tmp_path) -> None:
    _make_feed(tmp_path)
    out = tmp_path / "reports"
    result = runner.invoke(
        app,
        ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge", "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert (out / "corpus.nuggets.json").exists()
    assert not (tmp_path / "transcripts" / "fakefeed" / "corpus.nuggets.json").exists()


def test_merge_quiet(tmp_path) -> None:
    feed_root = _make_feed(tmp_path)
    result = runner.invoke(
        app, ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge", "--quiet"]
    )
    assert result.exit_code == 0
    assert "Merged" not in result.stdout
    assert (feed_root / "corpus.nuggets.json").exists()


def test_merge_no_sidecars_quiet(tmp_path) -> None:
    feed_root = tmp_path / "transcripts" / "fakefeed"
    feed_root.mkdir(parents=True, exist_ok=True)
    _write_transcript(feed_root, "ep1")
    result = runner.invoke(
        app, ["nuggets", "--feed", "fakefeed", "--data-dir", str(tmp_path), "--merge", "--quiet"]
    )
    assert result.exit_code == 0
    assert result.stdout == ""
