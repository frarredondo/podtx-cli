from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from podtx.cli import app
from podtx.config import load_settings
from podtx.models import Segment, Transcript

runner = CliRunner()


def _command_option_names(command_name: str) -> set[str]:
    """Option names of a registered CLI command, rendering-independent."""
    import typer.main

    click_cmd = typer.main.get_command(app)
    command = click_cmd.commands[command_name]
    return {param.name for param in command.params}


def test_cli_sync_has_trim_start_flag():
    names = _command_option_names("sync")
    assert "trim_start" in names


def test_cli_transcribe_has_trim_start_flag():
    names = _command_option_names("transcribe")
    assert "trim_start" in names


def test_settings_trim_start_default_zero(tmp_path: Path):
    s = load_settings(data_dir=tmp_path)
    assert hasattr(s, "trim_start")
    assert s.trim_start == 0 or s.trim_start == 0.0


def test_settings_trim_start_from_cli_flag(tmp_path: Path):
    s = load_settings(data_dir=tmp_path, trim_start=20)
    assert s.trim_start == 20


def test_settings_trim_start_from_toml(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("trim_start = 15\n", encoding="utf-8")
    s = load_settings(data_dir=tmp_path, config_path=cfg)
    assert s.trim_start == 15


def test_settings_trim_start_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PODCAST_TRANSCRIBER_TRIM_START", "25")
    s = load_settings(data_dir=tmp_path)
    assert s.trim_start == 25


def test_pipeline_trim_segments_filters_first_20s():
    from podtx.pipeline import trim_transcript

    transcript = Transcript(
        text="intro sponsor hello world substance",
        segments=[
            Segment(0.0, 10.0, "intro jingle garble"),
            Segment(10.0, 20.0, "sponsor read buy now"),
            Segment(20.0, 30.0, "hello world substance"),
            Segment(30.0, 40.0, "more substance here"),
        ],
        language="en",
        model="fake",
        engine="fake",
    )
    trimmed = trim_transcript(transcript, trim_start=20)
    # body text should lack first 20s
    assert "intro" not in trimmed.text.lower()
    assert "sponsor" not in trimmed.text.lower()
    assert "hello world" in trimmed.text.lower()
    # segments reflect trimmed view
    assert all(seg.start >= 20 or seg.end > 20 for seg in trimmed.segments)
    # first kept segment starts at or after 20
    assert trimmed.segments[0].start >= 20 or trimmed.segments[0].end > 20
    # exact: should have 2 segments remaning (20-30, 30-40)
    assert len(trimmed.segments) == 2


def test_pipeline_trim_zero_is_noop():
    from podtx.pipeline import trim_transcript

    transcript = Transcript(
        text="hello world",
        segments=[Segment(0.0, 5.0, "hello world")],
        language="en",
        model="fake",
        engine="fake",
    )
    trimmed = trim_transcript(transcript, trim_start=0)
    assert trimmed.text == transcript.text
    assert trimmed.segments == transcript.segments


def test_transcribe_local_file_respects_trim_start(tmp_path: Path, monkeypatch):
    """Integration: transcribe_local_file with trim_start=20 excludes first 20s text."""
    from podtx.config import Settings
    from podtx.models import Episode

    # fake audio file
    fake_audio = tmp_path / "episode.mp3"
    fake_audio.write_bytes(b"fake")

    # mock ffmpeg and engine
    import podtx.pipeline as pipeline_mod

    def fake_convert(src: Path, dest: Path | None = None, *args, **kwargs) -> Path:
        # also test that trim_start is passed to convert_to_wav if audio trimming path
        out = dest or src.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"wav")
        return out

    class FakeEngine:
        name = "fake"
        default_model = "fake-model"

        def transcribe(self, audio_path: Path, *, model=None, language="en", **kwargs):
            return Transcript(
                text="intro sponsor hello world substance",
                segments=[
                    Segment(0.0, 10.0, "intro jingle"),
                    Segment(10.0, 20.0, "sponsor read"),
                    Segment(20.0, 30.0, "hello world substance"),
                ],
                language=language,
                model=model or self.default_model,
                engine=self.name,
            )

    monkeypatch.setattr(pipeline_mod, "convert_to_wav", fake_convert)
    monkeypatch.setattr(pipeline_mod, "get_engine", lambda name: FakeEngine())
    # also need to mock require_ffmpeg
    monkeypatch.setattr(pipeline_mod, "require_ffmpeg", lambda: "/usr/bin/ffmpeg")
    # ensure ensure_data_dirs works with temp
    settings = Settings(data_dir=tmp_path / "data", trim_start=20, quiet=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    episode = Episode(guid="test-guid", title="Test Episode", enclosure_url=str(fake_audio), show_title="Show")

    from podtx.pipeline import transcribe_local_file
    paths = transcribe_local_file(fake_audio, settings=settings, episode=episode, out_dir=out_dir)

    # read txt body
    txt_paths = [p for p in paths if p.suffix == ".txt"]
    assert txt_paths, "should produce txt"
    body = txt_paths[0].read_text(encoding="utf-8")
    assert "intro" not in body.lower()
    assert "sponsor" not in body.lower()
    assert "hello world" in body.lower()

    # json also trimmed
    json_paths = [p for p in paths if p.suffix == ".json"]
    if json_paths:
        import json
        payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
        assert "intro" not in payload.get("text", "").lower()
        assert "hello world" in payload.get("text", "").lower()


def test_process_episodes_trims_and_writes(tmp_path: Path, monkeypatch):
    import podtx.pipeline as pipeline_mod
    from podtx.config import Settings
    from podtx.models import Episode

    def fake_convert(src: Path, dest: Path | None = None, *args, **kwargs) -> Path:
        out = dest or src.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"wav")
        return out

    class FakeEngine:
        name = "fake"
        default_model = "fake-model"
        def transcribe(self, audio_path: Path, *, model=None, language="en", **kwargs):
            return Transcript(
                text="intro sponsor real content here",
                segments=[
                    Segment(0.0, 5.0, "intro"),
                    Segment(5.0, 20.0, "sponsor"),
                    Segment(20.0, 30.0, "real content here"),
                ],
                language=language,
                model=model or self.default_model,
                engine=self.name,
            )

    def fake_download(episode: Episode, audio_dir: Path, quiet: bool = False, **kwargs) -> Path:
        # return a fake downloaded file path
        p = audio_dir / "fake_episode.mp3"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"audio")
        return p

    monkeypatch.setattr(pipeline_mod, "convert_to_wav", fake_convert)
    monkeypatch.setattr(pipeline_mod, "get_engine", lambda name: FakeEngine())
    monkeypatch.setattr(pipeline_mod, "require_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pipeline_mod, "download_only", fake_download)

    settings = Settings(data_dir=tmp_path / "data", trim_start=20, quiet=True)
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = Episode(guid="g1", title="Ep 1", enclosure_url="https://example.com/ep.mp3", show_title="Show")
    from podtx.pipeline import process_episodes
    results = process_episodes([ep], settings=settings, out_dir=out_dir)

    assert len(results) == 1
    txt_path = [p for p in results[0] if p.suffix == ".txt"][0]
    body = txt_path.read_text(encoding="utf-8")
    assert "intro" not in body.lower()
    assert "sponsor" not in body.lower()
    assert "real content" in body.lower()
