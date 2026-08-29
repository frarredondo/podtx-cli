from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json

from typer.testing import CliRunner

from podtx.cli import app
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs
from podtx.writers.md import _yaml_quote, write_md
from podtx.writers import write_outputs as write_outputs_fn

runner = CliRunner()


def _episode(**overrides) -> Episode:
    base = dict(
        guid="g1",
        title="Interview with Ada",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=3,
        show_title="Demo Podcast",
        link="https://example.com/ep",
        description="desc",
    )
    base.update(overrides)
    return Episode(**base)


def _transcript(text="Hello world.", segments=None) -> Transcript:
    if segments is None:
        segments = [Segment(start=0.0, end=1.5, text=text)]
    return Transcript(
        text=text,
        segments=segments,
        language="en",
        model="test-model",
        engine="fake",
    )


def test_yaml_quote_escapes() -> None:
    assert _yaml_quote('a"b') == '"a\\"b"'
    assert _yaml_quote("a\\b") == '"a\\\\b"'
    assert _yaml_quote("a\nb") == '"a\\nb"'
    assert _yaml_quote("simple") == '"simple"'


def test_write_md_full_front_matter(tmp_path: Path) -> None:
    episode = _episode()
    transcript = _transcript()
    path = tmp_path / "out.md"
    out = write_md(path, episode, transcript)
    assert out == path
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert 'title: "Interview with Ada"' in content
    assert 'show: "Demo Podcast"' in content
    assert "date: 2026-03-15" in content
    assert "episode: 3" in content
    assert 'engine: "fake"' in content
    assert 'model: "test-model"' in content
    assert 'source: "https://example.com/a.mp3"' in content
    assert 'link: "https://example.com/ep"' in content
    assert "---\n\nHello world.\n" in content


def test_write_md_minimal_missing_optional_fields(tmp_path: Path) -> None:
    episode = Episode(
        guid="g1",
        title="No meta",
        enclosure_url="",
        show_title=None,
        published_at=None,
        episode_num=None,
        link=None,
    )
    transcript = _transcript(text="Body")
    path = tmp_path / "sub" / "nested.md"
    write_md(path, episode, transcript)
    content = path.read_text(encoding="utf-8")
    assert 'show: "Unknown"' in content
    assert "date:" not in content
    assert "episode:" not in content
    assert "source:" not in content
    assert "link:" not in content
    assert path.parent.is_dir()


def test_write_md_yaml_escaping_in_front_matter(tmp_path: Path) -> None:
    episode = _episode(title='He said "hi"\nand \\ backslash', show_title='Show "A"')
    transcript = _transcript()
    path = tmp_path / "esc.md"
    write_md(path, episode, transcript)
    content = path.read_text(encoding="utf-8")
    # title should be escaped
    assert '\\"hi\\"' in content
    assert "\\n" in content
    assert "\\\\" in content


def test_write_md_honors_readable_and_cleanup(tmp_path: Path) -> None:
    episode = _episode()
    # segments with gap and fillers to trigger readable+cleanup
    segs = [
        Segment(0.0, 1.0, "So uh yeah."),
        Segment(2.0, 3.0, "The the fox."),
    ]
    transcript = Transcript(
        text="So uh yeah. The the fox.",
        segments=segs,
        language="en",
        model="test-model",
        engine="fake",
    )
    # without flags -> raw text collapsed? body_text without readable returns transcript.text
    raw_path = tmp_path / "raw.md"
    write_md(raw_path, episode, transcript, readable=False, cleanup=False)
    raw = raw_path.read_text(encoding="utf-8")
    assert "So uh yeah." in raw

    cleaned = tmp_path / "clean.md"
    write_md(cleaned, episode, transcript, readable=True, cleanup=True)
    text = cleaned.read_text(encoding="utf-8")
    # cleanup should strip uh and dedupe the the
    body = text.split("---\n")[-1]
    assert " uh " not in body
    assert "the the" not in body.lower()
    # readable should add paragraph breaks when segments have gap
    assert "\n\n" in body


def test_write_outputs_with_md(tmp_path: Path) -> None:
    episode = _episode()
    transcript = _transcript()
    paths = write_outputs(
        out_dir=tmp_path,
        basename="2026-03-15_003_interview-with-ada",
        episode=episode,
        transcript=transcript,
        formats=("txt", "json", "md"),
    )
    assert len(paths) == 3
    md_path = tmp_path / "2026-03-15_003_interview-with-ada.md"
    assert md_path in paths
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "Hello world." in content


def test_write_outputs_md_with_readable_cleanup(tmp_path: Path) -> None:
    episode = _episode()
    segs = [Segment(0.0, 1.0, "Hello uh world."), Segment(2.0, 3.0, "Hello hello.")]
    transcript = Transcript(text="Hello uh world. Hello hello.", segments=segs, language="en", model="m", engine="e")
    paths = write_outputs_fn(
        out_dir=tmp_path,
        basename="ep",
        episode=episode,
        transcript=transcript,
        formats=("md",),
        readable=True,
        cleanup=True,
    )
    assert len(paths) == 1
    body = paths[0].read_text(encoding="utf-8")
    assert "uh" not in body.lower().split("---")[-1]


def test_cli_format_writes_md(tmp_path: Path) -> None:
    # Create a transcript JSON via write_outputs then format to md without ASR
    from podtx.writers import write_outputs as wout

    episode = _episode()
    transcript = _transcript()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    wout(out_dir=src_dir, basename="ep", episode=episode, transcript=transcript, formats=("json",))
    json_path = src_dir / "ep.json"
    assert json_path.is_file()
    # also need to test format via CLI
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["format", str(json_path), "--format", "md", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    md_file = out_dir / "ep.md"
    assert md_file.is_file()
    content = md_file.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert 'title:' in content
    assert "Hello world." in content


def test_cli_format_md_honors_readable_cleanup(tmp_path: Path) -> None:
    from podtx.writers import write_outputs as wout

    episode = _episode()
    segs = [Segment(0.0, 1.0, "So uh yeah."), Segment(2.0, 3.0, "The the fox.")]
    transcript = Transcript(text="So uh yeah. The the fox.", segments=segs, language="en", model="m", engine="e")
    src_dir = tmp_path / "src2"
    src_dir.mkdir()
    wout(out_dir=src_dir, basename="ep2", episode=episode, transcript=transcript, formats=("json",))
    json_path = src_dir / "ep2.json"
    out_dir = tmp_path / "out2"
    result = runner.invoke(app, ["format", str(json_path), "--format", "md", "--readable", "--cleanup", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout + result.stderr
    content = (out_dir / "ep2.md").read_text(encoding="utf-8")
    body = content.split("---\n")[-1]
    assert " uh " not in body
    assert "the the" not in body.lower()
