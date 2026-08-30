from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from podtx.cli import app
from podtx.writers import write_outputs
from podtx.models import Episode, Segment, Transcript
from datetime import datetime, timezone

runner = CliRunner()


def _helpers():
    def ep(**overrides):
        base = dict(guid="g1", title="T", enclosure_url="https://example.com/ep.mp3", published_at=datetime(2026, 3, 15, tzinfo=timezone.utc), episode_num=1, show_title="S")
        base.update(overrides)
        return Episode(**base)
    def tx(text="hello world. second.", segs=None):
        if segs is None:
            segs = [Segment(0, 1, "hello"), Segment(1, 2, "world")]
        return Transcript(text=text, segments=segs, language="en", model="m", engine="fake")
    return ep, tx


def _write_json(tmp_path: Path, basename="ep"):
    ep, txf = _helpers()
    e = ep()
    t = txf()
    write_outputs(out_dir=tmp_path, basename=basename, episode=e, transcript=t, formats=("json",), readable=False, cleanup=False)
    return tmp_path / f"{basename}.json"


def test_cli_summarize_openrouter_mocked(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "LLM OV", "key_points": ["a", "b"], "quotes": [{"text": "hello", "start": 0}]})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload):
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter", "--api-key", "sk-test"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert (tmp_path / "ep.summary.json").is_file()
        data = json.loads((tmp_path / "ep.summary.json").read_text())
        assert data["overview"] == "LLM OV"
        assert data["backend"] == "openrouter"


def test_cli_summarize_opencode_defaults(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OC", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload):
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "opencode", "--api-key", "k"])
        assert result.exit_code == 0
        data = json.loads((tmp_path / "ep.summary.json").read_text())
        assert data["model"] == "muse-spark-1.2-contributor"


def test_cli_summarize_lmstudio_requires_model(tmp_path: Path):
    jp = _write_json(tmp_path)
    result = runner.invoke(app, ["summarize", str(jp), "--backend", "lmstudio"])
    assert result.exit_code != 0
    assert "requires --model" in result.stdout or "requires --model" in result.stderr


def test_cli_summarize_local_alias(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload) as mock:
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "local", "--model", "m", "--base-url", "http://localhost:1234/v1"])
        assert result.exit_code == 0
        assert mock.called


def test_cli_summarize_custom_base_url(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload) as mock:
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter", "--api-key", "k", "--base-url", "https://custom/v1"])
        assert result.exit_code == 0
        assert mock.call_args[1]["base_url"] == "https://custom/v1"


def test_cli_summarize_max_input_chars(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload):
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter", "--api-key", "k", "--max-input-chars", "5"])
        assert result.exit_code == 0
        data = json.loads((tmp_path / "ep.summary.json").read_text())
        assert data["truncated"] is True


def test_cli_summarize_missing_key_error(tmp_path: Path):
    jp = _write_json(tmp_path)
    result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter"])
    assert result.exit_code != 0
    assert "api key" in (result.stdout + result.stderr).lower()


def test_cli_summarize_keychain_fallback(monkeypatch, tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload):
        with patch("podtx.keychain.get_api_key", return_value="kc-key"):
            result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter"])
            assert result.exit_code == 0


def test_cli_summarize_feed_with_llm(tmp_path: Path):
    # Setup library

    # Manually create feed
    root = tmp_path / "transcripts"
    feed_dir = root / "feed-a"
    feed_dir.mkdir(parents=True)
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com/ep.mp3", show_title="S")
    tx = Transcript(text="hello world", segments=[Segment(0,1,"hello")], language="en", model="m", engine="fake")
    write_outputs(out_dir=feed_dir, basename="ep", episode=ep, transcript=tx, formats=("json",), readable=False, cleanup=False)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload):
        result = runner.invoke(app, ["summarize", "--feed", "feed-a", "--data-dir", str(tmp_path), "--backend", "openrouter", "--api-key", "k"])
        assert result.exit_code == 0, result.stdout + result.stderr
        assert (feed_dir / "ep.summary.json").is_file()


def test_cli_summarize_api_error_reported(tmp_path: Path):
    jp = _write_json(tmp_path)
    with patch("podtx.summarize._call_openai_compatible", side_effect=Exception("LLM boom")):
        # Use SummarizeError? Our code catches ValueError and SummarizeError but generic Exception will propagate?
        # summarize_many catches SummarizeError/ValueError only, but build_summary via direct path raises ValueError? We'll use SummarizeError
        from podtx.summarize import SummarizeError
        with patch("podtx.summarize._call_openai_compatible", side_effect=SummarizeError("LLM boom")):
            result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter", "--api-key", "k"])
            assert result.exit_code != 0
            assert "LLM boom" in result.stdout or "LLM boom" in result.stderr


def test_cli_summarize_temperature_timeout(tmp_path: Path):
    jp = _write_json(tmp_path)
    fake_payload = json.dumps({"overview": "OV", "key_points": ["a"], "quotes": []})
    with patch("podtx.summarize._call_openai_compatible", return_value=fake_payload) as mock:
        result = runner.invoke(app, ["summarize", str(jp), "--backend", "openrouter", "--api-key", "k", "--temperature", "0.9", "--timeout", "5"])
        assert result.exit_code == 0
        assert mock.call_args[1]["temperature"] == 0.9
        assert mock.call_args[1]["timeout"] == 5


def test_cli_summarize_help_includes_new_backends():
    result = runner.invoke(app, ["summarize", "--help"])
    assert result.exit_code == 0
    out = result.stdout.lower()
    assert "openrouter" in out
    assert "opencode" in out
    assert "lmstudio" in out
