from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from podtx.models import Segment, Transcript

# This import will fail until implementation (RED phase)
from podtx.diarize import (
    DiarizeError,
    _DIARIZE_BACKENDS,
    _ALIAS,
    _default_base_url,
    _default_model,
    _resolve_api_key,
    align_segments,
    diarize_transcript,
)


def _seg(start, end, text, speaker=None):
    return Segment(start=float(start), end=float(end), text=text, speaker=speaker)


def _transcript(segments=None):
    if segments is None:
        segments = [_seg(0, 2, "Hello"), _seg(2, 4, "Hi"), _seg(5, 7, "Again")]
    text = " ".join(s.text for s in segments)
    return Transcript(text=text, segments=segments, language="en", model="m", engine="parakeet")


def test_backends_include_fake_and_real():
    assert "fake" in _DIARIZE_BACKENDS
    assert "pyannote" in _DIARIZE_BACKENDS
    # at least one cloud
    assert any(b in _DIARIZE_BACKENDS for b in ("hf", "assemblyai", "deepgram"))


def test_default_model_and_base():
    assert _default_model("fake") is None
    assert _default_model("pyannote") is not None
    assert _default_base_url("hf") is not None


def test_align_segments_max_overlap():
    segs = [_seg(0, 2, "a"), _seg(2, 4, "b"), _seg(4, 6, "c")]
    diar = [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")]
    aligned = align_segments(segs, diar)
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[1].speaker == "SPEAKER_00"  # 2-3 overlaps 00, 3-4 overlaps 01 but 1s vs 1s tie -> first
    assert aligned[2].speaker == "SPEAKER_01"


def test_align_segments_no_overlap_keeps_none():
    segs = [_seg(10, 12, "isolated")]
    diar = [(0, 2, "SPEAKER_00")]
    aligned = align_segments(segs, diar)
    assert aligned[0].speaker is None


def test_align_segments_empty_diarization():
    segs = [_seg(0, 2, "hi")]
    aligned = align_segments(segs, [])
    assert aligned[0].speaker is None


def test_diarize_fake_round_robin():
    tx = _transcript([_seg(0, 1, "a"), _seg(1, 2, "b"), _seg(2, 3, "c")])
    out = diarize_transcript(tx, audio_path=Path("/tmp/fake.wav"), backend="fake")
    assert out.segments[0].speaker == "SPEAKER_00"
    assert out.segments[1].speaker == "SPEAKER_01"
    assert out.segments[2].speaker == "SPEAKER_00"
    # preserves text/start/end
    assert out.segments[0].text == "a"
    assert out.segments[0].start == 0


def test_diarize_unknown_backend_raises():
    tx = _transcript()
    with pytest.raises(DiarizeError, match="Unknown backend"):
        diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="bogus")


def test_diarize_pyannote_missing_deps_raises():
    tx = _transcript()
    with patch.dict("sys.modules", {"pyannote.audio": None}):
        with patch("podtx.diarize._load_pyannote_pipeline", side_effect=ImportError("no pyannote")):
            with pytest.raises(DiarizeError, match="pyannote"):
                diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="pyannote")


def test_diarize_pyannote_mocked_pipeline():
    tx = _transcript([_seg(0, 2, "hello"), _seg(2, 4, "world"), _seg(4, 6, "again")])
    # mock pipeline returns diarization turns
    mock_turns = [(0, 2.5, "SPEAKER_00"), (2.5, 6, "SPEAKER_01")]
    with patch("podtx.diarize._call_pyannote", return_value=mock_turns) as mock_call:
        out = diarize_transcript(tx, audio_path=Path("/tmp/audio.wav"), backend="pyannote", model="pyannote/speaker-diarization-3.1")
        mock_call.assert_called_once()
        assert out.segments[0].speaker == "SPEAKER_00"
        assert out.segments[1].speaker == "SPEAKER_01"  # 2-2.5 overlap 0.5 vs 2.5-4 overlap 1.5 -> 01 wins
        assert out.segments[2].speaker == "SPEAKER_01"


def test_diarize_hf_requires_api_key():
    tx = _transcript()
    with pytest.raises(DiarizeError, match="API key"):
        diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="hf", api_key=None, base_url="https://api.example.com")


def test_diarize_hf_mocked_success():
    tx = _transcript([_seg(0, 2, "a"), _seg(2, 4, "b")])
    fake_resp = [{"start": 0, "end": 3, "speaker": "SPEAKER_01"}, {"start": 3, "end": 4, "speaker": "SPEAKER_00"}]
    # Mock httpx response for HF
    def fake_post(url, headers=None, content=None, timeout=None):
        # verify auth header present
        assert "Authorization" in headers
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_resp
        mock_resp.text = json.dumps(fake_resp)
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch("httpx.post", side_effect=fake_post):
        out = diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="hf", api_key="hf_test_123", base_url="https://api.example.com")
        assert out.segments[0].speaker == "SPEAKER_01"
        assert out.segments[1].speaker == "SPEAKER_01"


def test_diarize_hf_401_hint():
    tx = _transcript()
    def fake_post(url, headers=None, content=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.json.side_effect = ValueError("no json")
        # raise HTTPStatusError on raise_for_status
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)
        return mock_resp
    with patch("httpx.post", side_effect=fake_post):
        with pytest.raises(DiarizeError, match="Invalid API key"):
            diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="hf", api_key="bad", base_url="https://api.example.com")


def test_resolve_api_key_precedence():
    # CLI > env > keychain
    with patch("podtx.diarize.get_api_key", return_value="kc_key"):
        # CLI provided
        assert _resolve_api_key("hf", api_key="cli_key", settings_api_key=None, service=None, account=None) == "cli_key"
        # fallback to settings
        assert _resolve_api_key("hf", api_key=None, settings_api_key="settings_key", service=None, account=None) == "settings_key"
        # fallback to env
        with patch.dict("os.environ", {"HF_TOKEN": "env_key"}):
            assert _resolve_api_key("hf", api_key=None, settings_api_key=None, service=None, account=None) == "env_key"


def test_diarize_empty_transcript_returns_as_is():
    tx = Transcript(text="", segments=[], language="en", model="m", engine="e")
    out = diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="fake")
    assert out.segments == []
    assert out.text == ""


def test_diarize_preserves_language_model_engine():
    tx = Transcript(text="hi", segments=[_seg(0, 1, "hi")], language="es", model="my-model", engine="whisper")
    out = diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="fake")
    assert out.language == "es"
    assert out.model == "my-model"
    assert out.engine == "whisper"
