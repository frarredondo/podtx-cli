from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from podtx.diarize import (
    DiarizeError,
    _default_base_url,
    _default_model,
    _normalize_backend,
    _resolve_api_key,
    _call_hf,
    _call_pyannote,
    _load_pyannote_pipeline,
    align_segments,
    diarize_transcript,
)
from podtx.models import Segment, Transcript


def _seg(s, e, t, spk=None):
    return Segment(start=float(s), end=float(e), text=t, speaker=spk)


def _tx(segs=None):
    if segs is None:
        segs = [_seg(0, 1, "a"), _seg(1, 2, "b")]
    return Transcript(text=" ".join(s.text for s in segs), segments=segs, language="en", model="m", engine="e")


# _default_model / _default_base_url / _normalize
def test_default_model_variants():
    assert _default_model("pyannote") == "pyannote/speaker-diarization-3.1"
    assert _default_model("hf") == "pyannote/speaker-diarization-3.1"
    assert _default_model("assemblyai") is not None
    assert _default_model("deepgram") is not None
    assert _default_model("fake") is None
    assert _default_model("local") == "pyannote/speaker-diarization-3.1"  # alias
    assert _normalize_backend("LOCAL") == "pyannote"
    assert _normalize_backend("  HF  ") == "hf"


def test_default_base_variants():
    assert _default_base_url("hf") is not None
    assert _default_base_url("assemblyai") is not None
    assert _default_base_url("deepgram") is not None
    assert _default_base_url("fake") is None
    assert _default_base_url("pyannote") is None
    assert _default_base_url("local") is None


# _resolve_api_key branches
def test_resolve_api_key_all_envs():
    with patch.dict("os.environ", {"HF_TOKEN": "hf_tok"}, clear=False):
        assert _resolve_api_key("hf", None) == "hf_tok"
    with patch.dict("os.environ", {"HUGGINGFACE_API_KEY": "hf2"}, clear=False):
        # ensure HF_TOKEN not set for this check
        with patch.dict("os.environ", {}, clear=False):
            # need to clear HF_TOKEN first
            env = {"HUGGINGFACE_API_KEY": "hf2"}
            with patch.dict("os.environ", env, clear=True):
                assert _resolve_api_key("hf", None) == "hf2"
    with patch.dict("os.environ", {"ASSEMBLYAI_API_KEY": "asm"}, clear=False):
        assert _resolve_api_key("assemblyai", None) == "asm"
    with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "dg"}, clear=False):
        assert _resolve_api_key("deepgram", None) == "dg"
    with patch.dict("os.environ", {"DIARIZE_API_KEY": "gen"}, clear=False):
        assert _resolve_api_key("hf", None) == "gen" if not _resolve_api_key("hf", None) == "hf_tok" else True  # generic
    with patch.dict("os.environ", {"PODCAST_TRANSCRIBER_DIARIZE_API_KEY": "pod"}, clear=False):
        # Ensure no other env overrides; use a backend that doesn't have specific
        with patch.dict("os.environ", {"PODCAST_TRANSCRIBER_DIARIZE_API_KEY": "pod"}, clear=True):
            assert _resolve_api_key("deepgram", None) == "pod"
    # pyannote fallback to HF_TOKEN
    with patch.dict("os.environ", {"HF_TOKEN": "hf_for_pyan"}, clear=False):
        assert _resolve_api_key("pyannote", None) == "hf_for_pyan"


def test_resolve_api_key_keychain_and_service():
    with patch("podtx.diarize.get_api_key", return_value="kc_val"):
        assert _resolve_api_key("hf", None, service="svc", account="acct") == "kc_val"
        # default service fallback
        with patch.dict("os.environ", {}, clear=True):
            # ensure no env
            assert _resolve_api_key("hf", None, service="different", account="acct") == "kc_val"
    # keychain raises
    with patch("podtx.diarize.get_api_key", side_effect=Exception("boom")):
        assert _resolve_api_key("hf", None, service="svc", account="acct") is None
        # also default path exception
        assert _resolve_api_key("hf", None) is None
    # default service dedup branch (service == default) should skip second lookup
    with patch("podtx.diarize.get_api_key", return_value=None) as mock_get:
        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_api_key("hf", None, service="podtx-hf", account="api-key")
            assert result is None
            # should only be called once (first check with service/account), not twice
            assert mock_get.call_count == 1
    # CLI and settings precedence
    assert _resolve_api_key("hf", api_key="cli") == "cli"
    assert _resolve_api_key("hf", None, settings_api_key="settings") == "settings"


def test_resolve_api_key_default_keychain_success():
    # no explicit service/account -> default podtx-<backend>/api-key lookup returns value
    with patch("podtx.diarize.get_api_key", return_value="kc_val"):
        with patch.dict("os.environ", {}, clear=True):
            assert _resolve_api_key("hf", None) == "kc_val"


# align_segments branches
def test_align_zero_length_and_tie():
    # zero-length segment handling
    segs = [_seg(2, 2, "point")]
    diar = [(0, 3, "SPEAKER_00"), (3, 5, "SPEAKER_01")]
    aligned = align_segments(segs, diar)
    assert aligned[0].speaker == "SPEAKER_00"
    # no overlap
    segs2 = [_seg(10, 11, "far")]
    aligned2 = align_segments(segs2, [(0, 1, "SPEAKER_00")])
    assert aligned2[0].speaker is None
    # tie keeps first
    segs3 = [_seg(0, 2, "tie")]
    diar3 = [(0, 1, "SPEAKER_00"), (1, 2, "SPEAKER_01")]  # equal 1s each, tie -> first
    # Our logic picks first encountered with max, so 00 wins (first 1.0)
    aligned3 = align_segments(segs3, diar3)
    assert aligned3[0].speaker == "SPEAKER_00"
    # empty diarization already tested but ensure
    assert align_segments([_seg(0, 1, "hi")], [])[0].speaker is None
    # zero-length with no containing diarization
    segs4 = [_seg(10, 10, "point2")]
    aligned4 = align_segments(segs4, [(0, 3, "SPEAKER_00")])
    assert aligned4[0].speaker is None


# _load_pyannote_pipeline branches
def test_load_pyannote_success_and_failure():
    # Test ImportError path - mock import failure
    with patch.dict("sys.modules", {"pyannote.audio": None}):
        # Force the import inside _load to fail
        # _load will try from pyannote.audio import Pipeline and get None -> ImportError
        # We need to ensure it raises ImportError, not DiarizeError
        # Instead, directly test the ImportError raise
        import importlib
        # Simulate by patching the import
        with patch("builtins.__import__", side_effect=ImportError("no pyannote")):
            try:
                _load_pyannote_pipeline("model")
                assert False
            except ImportError as e:
                assert "pyannote" in str(e).lower()
    # Test from_pretrained failure - mock Pipeline to raise
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("boom")
    with patch.dict("sys.modules", {"pyannote.audio": MagicMock(Pipeline=mock_pipeline_cls)}):
        try:
            _load_pyannote_pipeline("bad-model")
            assert False
        except DiarizeError as e:
            assert "Failed to load" in str(e)
    # Test success path - mock successful load
    mock_pipe = MagicMock()
    mock_cls2 = MagicMock()
    mock_cls2.from_pretrained.return_value = mock_pipe
    with patch.dict("sys.modules", {"pyannote.audio": MagicMock(Pipeline=mock_cls2)}):
        pipe = _load_pyannote_pipeline("good-model")
        assert pipe is mock_pipe


# _call_pyannote branches
def test_call_pyannote_propagates_load_diarize_error():
    with patch("podtx.diarize._load_pyannote_pipeline", side_effect=DiarizeError("load failed")):
        with pytest.raises(DiarizeError, match="load failed"):
            _call_pyannote(Path("/tmp/x.wav"), "m", 120)


def test_call_pyannote_annotation_itertracks():
    mock_turn = MagicMock()
    mock_turn.start = 0
    mock_turn.end = 1
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_01"), (MagicMock(start=1, end=2), None, "SPEAKER_00")]
    mock_pipe = MagicMock(return_value=mock_annotation)
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe):
        turns = _call_pyannote(Path("/tmp/fake.wav"), "model", 120)
        # should normalize to SPEAKER_00/01 sorted
        assert len(turns) == 2
        assert turns[0][2] in ("SPEAKER_00", "SPEAKER_01")


def test_call_pyannote_list_dict_and_tuple_and_unknown():
    # list of dicts
    mock_pipe = MagicMock(return_value=[{"start": 0, "end": 1, "speaker": "SPEAKER_00"}])
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe):
        turns = _call_pyannote(Path("/tmp/x.wav"), "m", 120)
        assert turns == [(0.0, 1.0, "SPEAKER_00")]
    # list of tuples
    mock_pipe2 = MagicMock(return_value=[(0, 1, "a"), (1, 2, "b")])
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe2):
        turns2 = _call_pyannote(Path("/tmp/x.wav"), "m", 120)
        assert len(turns2) == 2
    # list with unknown item (should skip)
    mock_pipe3 = MagicMock(return_value=[{"start": 0, "end": 1, "speaker": "s"}, "bad", 123])
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe3):
        turns3 = _call_pyannote(Path("/tmp/x.wav"), "m", 120)
        assert len(turns3) == 1
    # unknown format (not list nor annotation)
    mock_pipe4 = MagicMock(return_value={"bad": "format"})
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe4):
        with pytest.raises(DiarizeError, match="Unexpected"):
            _call_pyannote(Path("/tmp/x.wav"), "m", 120)
    # pipeline raises
    mock_pipe5 = MagicMock(side_effect=RuntimeError("pipe fail"))
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe5):
        with pytest.raises(DiarizeError, match="pyannote diarization failed"):
            _call_pyannote(Path("/tmp/x.wav"), "m", 120)
    # parsing exception
    mock_ann = MagicMock()
    mock_ann.itertracks.side_effect = RuntimeError("parse fail")
    mock_pipe6 = MagicMock(return_value=mock_ann)
    mock_ann.__class__ = type("Ann", (), {})  # ensure has itertracks attr
    mock_ann.itertracks = MagicMock(side_effect=RuntimeError("parse fail"))
    # Need to make hasattr true but itertracks fails
    mock_pipe6 = MagicMock(return_value=mock_ann)
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe6):
        with pytest.raises(DiarizeError, match="Failed to parse"):
            _call_pyannote(Path("/tmp/x.wav"), "m", 120)


# _call_hf branches
def test_call_hf_missing_key_and_base():
    with pytest.raises(DiarizeError, match="API key"):
        _call_hf(Path("/tmp/x.wav"), "model", api_key="", base_url="https://example.com", timeout=10)
    # base_url None should fallback to default
    fake_resp = MagicMock(status_code=200, text="[]")
    fake_resp.json.return_value = []
    fake_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=fake_resp) as mp:
        turns = _call_hf(Path("/tmp/x.wav"), "model", api_key="key", base_url="", timeout=10)
        assert turns == []
        # ensure called with default base
        assert "huggingface.co" in mp.call_args[0][0]


def test_call_hf_data_fallback_and_httpx_exception():
    # audio_path not file -> should use fake-audio without exception
    with patch("httpx.post", side_effect=Exception("network fail")):
        with pytest.raises(DiarizeError, match="request failed"):
            _call_hf(Path("/nonexistent/path.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
    # read_bytes exception path (patch Path.read_bytes to raise on an existing file)
    f = Path("/tmp/podtx_diarize_readbytes_test.wav")
    f.write_bytes(b"\x00\x01")
    with patch.object(Path, "read_bytes", side_effect=OSError("read fail")):
        fake_resp = MagicMock(status_code=200, text="[]")
        fake_resp.json.return_value = []
        fake_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=fake_resp):
            turns = _call_hf(f, "model", api_key="k", base_url="https://example.com", timeout=5)
            assert turns == []
    f.unlink(missing_ok=True)


def test_call_hf_status_401_404_and_http_error():
    # 401
    mock401 = MagicMock(status_code=401, text="Unauthorized")
    mock401.json.side_effect = ValueError("no json")
    mock401.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mock401):
        with pytest.raises(DiarizeError, match="401"):
            _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
    # 404
    mock404 = MagicMock(status_code=404, text="Not found")
    mock404.json.return_value = {}
    mock404.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mock404):
        with pytest.raises(DiarizeError, match="404"):
            _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
    # HTTPStatusError via raise_for_status
    mock500 = MagicMock(status_code=500, text="Server error")
    mock500.json.return_value = []
    mock500.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=mock500)
    with patch("httpx.post", return_value=mock500):
        with pytest.raises(DiarizeError, match="500"):
            _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)


def test_call_hf_invalid_json_and_unexpected_format():
    mockBadJson = MagicMock(status_code=200, text="not json")
    mockBadJson.json.side_effect = ValueError("bad json")
    mockBadJson.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mockBadJson):
        with pytest.raises(DiarizeError, match="invalid JSON"):
            _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
    # unexpected format not list
    mockNotList = MagicMock(status_code=200, text="{}")
    mockNotList.json.return_value = {"foo": "bar"}
    mockNotList.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mockNotList):
        with pytest.raises(DiarizeError, match="unexpected response"):
            _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
    # dict with diarization key
    mockWithKey = MagicMock(status_code=200, text="{}")
    mockWithKey.json.return_value = {"diarization": [{"start": 0, "end": 1, "speaker": "s"}]}
    mockWithKey.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mockWithKey):
        turns = _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
        assert len(turns) == 1


def test_call_hf_speaker_variants_and_missing_fields():
    # label and speaker_label variants, and missing fields skipped
    payload = [
        {"start": 0, "end": 1, "label": "a"},
        {"start": 1, "end": 2, "speaker_label": "b"},
        {"start": 2, "end": 3, "speaker": "c"},
        {"start": 3, "end": 4},  # missing speaker -> skip
        {"start": None, "end": 1, "speaker": "x"},  # missing start -> skip
        "not a dict",  # skip
        {"start": 5, "end": 6, "speaker": "a"},  # duplicate a to test normalization
    ]
    mockResp = MagicMock(status_code=200, text=json.dumps(payload))
    mockResp.json.return_value = payload
    mockResp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mockResp):
        turns = _call_hf(Path("/tmp/x.wav"), "model", api_key="k", base_url="https://example.com", timeout=5)
        # should have 4 valid (a,b,c,a) -> normalized to SPEAKER_00/01/02 etc sorted
        assert len(turns) == 4
        # speakers normalized sorted unique
        uniq = sorted(set(t[2] for t in turns))
        assert uniq == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_call_assemblyai_and_deepgram():
    with pytest.raises(DiarizeError, match="API key"):
        from podtx.diarize import _call_assemblyai
        _call_assemblyai(Path("/tmp/x.wav"), api_key="", base_url="https://example.com", timeout=5)
    with pytest.raises(DiarizeError, match="not fully implemented"):
        from podtx.diarize import _call_assemblyai
        _call_assemblyai(Path("/tmp/x.wav"), api_key="k", base_url="https://example.com", timeout=5)
    with pytest.raises(DiarizeError, match="deepgram"):
        diarize_transcript(_tx(), audio_path=Path("/tmp/x.wav"), backend="deepgram", api_key="k", base_url="https://example.com")


def test_diarize_transcript_empty_and_unknown_and_missing_model():
    # empty transcript
    empty = Transcript(text="", segments=[], language="en", model="m", engine="e")
    out = diarize_transcript(empty, audio_path=Path("/tmp/x.wav"), backend="fake")
    assert out.segments == []
    # unknown backend
    with pytest.raises(DiarizeError, match="Unknown backend"):
        diarize_transcript(_tx(), audio_path=Path("/tmp/x.wav"), backend="unknown")
    # pyannote missing model (when default is None for fake? but for pyannote default exists, test by forcing None)
    with patch("podtx.diarize._default_model", return_value=None):
        with pytest.raises(DiarizeError, match="requires a model"):
            diarize_transcript(_tx(), audio_path=Path("/tmp/x.wav"), backend="pyannote", model=None)


def test_diarize_transcript_needs_key_missing():
    # hf without key
    with patch.dict("os.environ", {}, clear=True):
        with patch("podtx.diarize.get_api_key", return_value=None):
            with pytest.raises(DiarizeError, match="requires an API key"):
                diarize_transcript(_tx(), audio_path=Path("/tmp/x.wav"), backend="hf", api_key=None)


def test_diarize_transcript_assemblyai_and_hf_success():
    # hf success via mock
    tx = _tx([_seg(0, 1, "a"), _seg(1, 2, "b")])
    fake_hf_payload = [{"start": 0, "end": 2, "speaker": "s0"}]
    mockResp = MagicMock(status_code=200, text=json.dumps(fake_hf_payload))
    mockResp.json.return_value = fake_hf_payload
    mockResp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mockResp):
        out = diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="hf", api_key="k", base_url="https://example.com")
        assert out.segments[0].speaker == "SPEAKER_00"
    # assemblyai not implemented -> error
    with pytest.raises(DiarizeError, match="not fully implemented"):
        diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="assemblyai", api_key="k", base_url="https://example.com")


def test_diarize_transcript_deepgram_else_branch():
    # deepgram else branch already tested, also test unknown else (should not happen)
    with pytest.raises(DiarizeError, match="deepgram"):
        diarize_transcript(_tx(), audio_path=Path("/tmp/x.wav"), backend="deepgram", api_key="k", base_url="https://example.com")

def test_cover_remaining_unknown_format_and_data_read():
    # Cover unknown format else branch: pipeline returns int
    from podtx.diarize import _call_pyannote
    mock_pipe_int = MagicMock(return_value=123)  # int, not list nor annotation
    with patch("podtx.diarize._load_pyannote_pipeline", return_value=mock_pipe_int):
        with pytest.raises(DiarizeError, match="Unexpected"):
            _call_pyannote(Path("/tmp/x.wav"), "m", 120)
    # Cover data read branches: file exists vs not exists and read exception
    from podtx.diarize import _call_hf
    # Create a temp file that exists
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"fake audio data")
        tf_path = Path(tf.name)
    try:
        # Call with existing file - should read it (the if True branch)
        fake_resp = MagicMock(status_code=200, text="[]")
        fake_resp.json.return_value = []
        fake_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=fake_resp) as mp:
            turns = _call_hf(tf_path, "model", api_key="k", base_url="https://example.com", timeout=5)
            assert turns == []
            # Verify it was called with data from file
            assert mp.called
            # data should be file content, not b"fake-audio"
            assert mp.call_args[1]["content"] == b"fake audio data"
        # Call with non-existent file - should use else branch b"fake-audio"
        non_exist = Path("/tmp/nonexistent_12345.wav")
        assert not non_exist.exists()
        with patch("httpx.post", return_value=fake_resp):
            turns2 = _call_hf(non_exist, "model", api_key="k", base_url="https://example.com", timeout=5)
            assert turns2 == []
    finally:
        tf_path.unlink(missing_ok=True)


def test_diarize_unknown_backend_dispatch_else():
    # Top validation allows an unknown backend patched into _DIARIZE_BACKENDS,
    # so the dispatch falls through to the final else (defensive branch).
    tx = _tx()
    bogus = {"fake", "pyannote", "hf", "assemblyai", "deepgram", "bogus"}
    with patch("podtx.diarize._DIARIZE_BACKENDS", bogus):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(DiarizeError, match="Unknown backend"):
                diarize_transcript(tx, audio_path=Path("/tmp/x.wav"), backend="bogus")

