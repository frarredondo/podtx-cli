from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.config import load_settings
from podtx.formatting import body_text, segments_to_paragraphs_with_speaker, segments_to_text_with_speaker
from podtx.models import Episode, Segment, Transcript
from podtx.writers import write_outputs

runner = CliRunner()


def _episode() -> Episode:
    return Episode(
        guid="g1",
        title="Interview",
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=1,
        show_title="Demo",
        link="https://example.com/ep",
    )


def _segs_labeled() -> list[Segment]:
    return [
        Segment(0.0, 2.0, "Hello from Alice", speaker="SPEAKER_00"),
        Segment(2.5, 5.0, "Hi from Bob", speaker="SPEAKER_01"),
        Segment(5.5, 8.0, "Back to Alice", speaker="SPEAKER_00"),
        Segment(8.5, 10.0, "Bob again", speaker="SPEAKER_01"),
    ]


def _segs_raw() -> list[Segment]:
    return [
        Segment(0.0, 2.0, "Hello from Alice"),
        Segment(2.5, 5.0, "Hi from Bob"),
    ]


def test_segment_has_speaker_field():
    s = Segment(0.0, 1.0, "hi", speaker="SPEAKER_01")
    assert s.speaker == "SPEAKER_01"
    s2 = Segment(0.0, 1.0, "hi")
    assert s2.speaker is None


def test_body_text_without_diarize_is_single_speaker():
    segs = _segs_labeled()
    txt = body_text("Hello world", segs, readable=False, diarize=False)
    assert "SPEAKER" not in txt
    assert "Hello world" in txt


def test_body_text_with_diarize_reflects_turns():
    segs = _segs_labeled()
    txt = body_text("unused", segs, readable=False, diarize=True)
    assert "SPEAKER_00: Hello from Alice" in txt
    assert "SPEAKER_01: Hi from Bob" in txt
    assert txt.count("SPEAKER_00") == 2
    assert txt.count("SPEAKER_01") == 2


def test_body_text_diarize_readable_paragraphs_on_speaker_change():
    segs = _segs_labeled()
    txt = body_text("unused", segs, readable=True, diarize=True)
    assert "SPEAKER_00:" in txt
    assert "SPEAKER_01:" in txt
    assert "\n\n" in txt


def test_segments_to_text_with_speaker():
    segs = _segs_labeled()
    out = segments_to_text_with_speaker(segs)
    assert out.splitlines()[0] == "SPEAKER_00: Hello from Alice"
    assert out.splitlines()[1] == "SPEAKER_01: Hi from Bob"


def test_segments_to_paragraphs_with_speaker():
    segs = _segs_labeled()
    out = segments_to_paragraphs_with_speaker(segs)
    assert "SPEAKER_00: Hello from Alice" in out
    assert "\n\n" in out


def test_body_text_diarize_off_by_default_no_speaker_leak():
    segs = _segs_raw()
    txt = body_text("Hello world", segs, readable=False, diarize=False)
    assert "SPEAKER" not in txt
    txt2 = body_text("Hello", segs, readable=True, diarize=False)
    assert "SPEAKER" not in txt2


def test_write_outputs_with_diarize_includes_speaker(tmp_path: Path):
    ep = _episode()
    segs = _segs_labeled()
    tr = Transcript(text="Hello world", segments=segs, language="en", model="m", engine="fake")
    paths = write_outputs(out_dir=tmp_path, basename="ep", episode=ep, transcript=tr, formats=("txt", "json"), readable=False, diarize=True)
    assert len(paths) == 2
    txt = (tmp_path / "ep.txt").read_text(encoding="utf-8")
    assert "SPEAKER_00: Hello from Alice" in txt
    assert "SPEAKER_01: Hi from Bob" in txt
    payload = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert payload["diarize"] is True
    assert payload["text"].count("SPEAKER_00") >= 1
    assert payload["segments"][0].get("speaker") == "SPEAKER_00"
    assert payload["segments"][1].get("speaker") == "SPEAKER_01"


def test_write_outputs_without_diarize_no_speaker(tmp_path: Path):
    ep = _episode()
    segs = _segs_labeled()
    tr = Transcript(text="Hello", segments=segs, language="en", model="m", engine="fake")
    paths = write_outputs(out_dir=tmp_path, basename="ep2", episode=ep, transcript=tr, formats=("txt", "json"), readable=False, diarize=False)
    payload = json.loads((tmp_path / "ep2.json").read_text(encoding="utf-8"))
    assert "SPEAKER_00: Hello" not in payload["text"]
    assert payload["diarize"] is False


def test_write_outputs_md_with_diarize(tmp_path: Path):
    ep = _episode()
    segs = _segs_labeled()
    tr = Transcript(text="Hello", segments=segs, language="en", model="m", engine="fake")
    paths = write_outputs(out_dir=tmp_path, basename="epmd", episode=ep, transcript=tr, formats=("md",), readable=False, diarize=True)
    assert len(paths) == 1
    content = paths[0].read_text(encoding="utf-8")
    assert "SPEAKER_00" in content
    assert "SPEAKER_01" in content


def test_parakeet_engine_assigns_speaker_when_diarize(monkeypatch):
    from podtx.engines.parakeet import ParakeetEngine
    eng = ParakeetEngine()
    class FakeAsr:
        def transcribe(self, path):
            class S:
                text = "Hello"
                start = 0.0
                end = 1.0
            return type("R", (), {"text": "Hello", "segments": [S(), S()]})()
    monkeypatch.setattr(eng, "_load", lambda *a, **k: FakeAsr())
    tr = eng.transcribe(Path("/tmp/fake.wav"), diarize=True)
    assert tr.segments[0].speaker == "SPEAKER_00"
    assert tr.segments[1].speaker == "SPEAKER_01"


def test_parakeet_engine_no_speaker_when_not_diarize(monkeypatch):
    from podtx.engines.parakeet import ParakeetEngine
    eng = ParakeetEngine()
    class FakeAsr:
        def transcribe(self, path):
            class S:
                text = "Hi"
                start = 0.0
                end = 1.0
            return type("R", (), {"text": "Hi", "segments": [S()]})()
    monkeypatch.setattr(eng, "_load", lambda *a, **k: FakeAsr())
    tr = eng.transcribe(Path("/tmp/fake.wav"), diarize=False)
    assert tr.segments[0].speaker is None


def test_format_preserves_speaker(tmp_path: Path):
    from podtx.format_cmd import load_transcript_json, reformat_transcript
    ep = _episode()
    segs = _segs_labeled()
    tr = Transcript(text="Hello", segments=segs, language="en", model="m", engine="fake")
    write_outputs(out_dir=tmp_path, basename="ep", episode=ep, transcript=tr, formats=("json",), readable=False, diarize=True)
    payload = json.loads((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert payload["segments"][0].get("speaker") == "SPEAKER_00"
    ep2, tr2 = load_transcript_json(tmp_path / "ep.json")
    assert tr2.segments[0].speaker == "SPEAKER_00"
    out = tmp_path / "out"
    reformat_transcript(tmp_path / "ep.json", out_dir=out, diarize=True, formats=("json", "txt"))
    payload2 = json.loads((out / "ep.json").read_text(encoding="utf-8"))
    assert payload2["segments"][0].get("speaker") == "SPEAKER_00"


def test_cli_sync_and_transcribe_have_diarize_flag():
    for cmd in (["sync", "--help"], ["transcribe", "--help"], ["format", "--help"]):
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "--diarize" in result.stdout
        assert "Opt-in" in result.stdout or "diarization" in result.stdout.lower()


def test_cli_sync_diarize_is_opt_in(tmp_path: Path):
    s = load_settings(data_dir=tmp_path)
    assert s.diarize is False
    s2 = load_settings(data_dir=tmp_path, diarize=True)
    assert s2.diarize is True
