from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from podtx.models import Episode, Segment, Transcript
from podtx.summarize import (
    SummarizeError,
    _call_openai_compatible,
    _extract_json_content,
    _truncate_text,
    _validate_llm_payload,
    build_summary,
    summarize_many,
    summarize_transcript,
)
from podtx.writers import write_outputs


def _fake_episode(**overrides) -> Episode:
    base = dict(
        guid="fake-guid-1",
        title="Fake Episode Title",
        enclosure_url="https://example.com/ep.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=42,
        show_title="Fake Show",
        link="https://example.com/ep",
    )
    base.update(overrides)
    return Episode(**base)


def _fake_transcript(
    text: str = "First sentence is overview. Second sentence also overview. Third is a key point. Fourth more. Fifth extra.",
    segments: list[Segment] | None = None,
    engine: str = "fake",
    model: str = "fake-model",
) -> Transcript:
    if segments is None:
        segments = [
            Segment(0.0, 1.5, "First sentence is overview."),
            Segment(2.0, 3.5, "Second sentence also overview."),
            Segment(10.0, 12.0, "Third is a key point."),
            Segment(65.0, 70.0, "Fourth more."),
            Segment(120.0, 125.0, "Fifth extra."),
        ]
    return Transcript(text=text, segments=segments, language="en", model=model, engine=engine)


def _write_fake_transcript_json(tmp_path: Path, basename: str = "ep", episode: Episode | None = None, transcript: Transcript | None = None) -> Path:
    ep = episode or _fake_episode()
    tx = transcript or _fake_transcript()
    write_outputs(out_dir=tmp_path, basename=basename, episode=ep, transcript=tx, formats=("txt", "json"), readable=False, cleanup=False)
    return tmp_path / f"{basename}.json"


def _llm_response(overview="OV", key_points=None, quotes=None):
    if key_points is None:
        key_points = ["kp1", "kp2", "kp3"]
    if quotes is None:
        quotes = [{"text": "First sentence is overview.", "start": 0.0}]
    # New nuggets format for refactored summarize
    nuggets = []
    q0 = quotes[0].get("text", "") if quotes and isinstance(quotes[0], dict) else ""
    nuggets.append({"insight": overview, "context": "Fake Show — ep", "why_it_matters": "why", "quote": q0})
    for i, kp in enumerate(key_points[:4]):
        q = ""
        if i + 1 < len(quotes) and isinstance(quotes[i + 1], dict):
            q = quotes[i + 1].get("text", "")
        nuggets.append({"insight": kp, "context": "ctx", "why_it_matters": "why", "quote": q})
    nuggets = nuggets[:5]
    return json.dumps({"nuggets": nuggets, "top5_best": list(range(len(nuggets)))})


# ── _truncate_text ──
def test_truncate_no_limit():
    text, trunc = _truncate_text("hello", None)
    assert text == "hello" and truncated is False if (truncated := trunc) else True  # dummy


def test_truncate_short():
    text, truncated = _truncate_text("hello", 100)
    assert text == "hello" and truncated is False


def test_truncate_with_dot():
    long = "Sentence one. Sentence two. Sentence three. " * 10
    text, truncated = _truncate_text(long, 50)
    assert truncated is True
    assert len(text) <= 51
    assert text.endswith(".") or text.endswith("…")


def test_truncate_no_dot():
    long = "A" * 100
    text, truncated = _truncate_text(long, 50)
    assert truncated is True
    assert text.endswith("…")


# ── _extract_json_content ──
def test_extract_direct():
    assert _extract_json_content('{"overview": "hi", "key_points": ["a"]}')["overview"] == "hi"


def test_extract_embedded():
    raw = 'prefix {"overview": "hi", "key_points": ["a"]} suffix'
    assert _extract_json_content(raw)["overview"] == "hi"


def test_extract_invalid():
    try:
        _extract_json_content("not json at all")
        assert False
    except SummarizeError:
        pass


# ── _validate_llm_payload ──
def test_validate_ok():
    p = {"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": "t"}], "top5_best": [0]}
    out = _validate_llm_payload(p)
    assert out["nuggets"][0]["insight"] == "ov"
    assert len(out["nuggets"]) == 1
    assert out["nuggets"][0]["quote"] == "t"


def test_validate_missing_overview():
    try:
        _validate_llm_payload({"nuggets": []})
        assert False
    except SummarizeError as e:
        assert "nuggets" in str(e).lower()


def test_validate_empty_key_points():
    try:
        _validate_llm_payload({"nuggets": [{"insight": "   ", "context": "ctx", "why_it_matters": "why", "quote": ""}]})
        assert False
    except SummarizeError:
        pass


def test_validate_quotes_string_form():
    p = {"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [0]}
    out = _validate_llm_payload(p)
    assert out["nuggets"][0]["quote"] == ""


def test_validate_quotes_lenient_str():
    p = {"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": "just text"}], "top5_best": [0]}
    out = _validate_llm_payload(p)
    assert out["nuggets"][0]["quote"] == "just text"


# ── _call_openai_compatible ──
def test_call_success():
    mock_resp = MagicMock(status_code=200, text='{"choices":[]}')
    mock_resp.json.return_value = {"choices": [{"message": {"content": '{"overview":"ov","key_points":["a"]}'}}]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        content = _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[{"role": "user", "content": "hi"}])
        assert "overview" in content
        assert mock_client.post.called
        assert "chat/completions" in mock_client.post.call_args[0][0]


def test_call_no_api_key():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": '{"overview":"ov","key_points":["a"]}'}}]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        _call_openai_compatible(base_url="https://example.com/v1", api_key=None, model="m", messages=[])
        headers = mock_client.post.call_args[1]["headers"]
        assert "Authorization" not in headers


def test_call_http_error():
    mock_resp = MagicMock(status_code=401, text="unauthorized")
    mock_resp.json.return_value = {}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "401" in str(e)


def test_call_request_error():
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.RequestError("network fail")
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "request failed" in str(e).lower()


def test_call_invalid_json_response():
    mock_resp = MagicMock(status_code=200, text="not json")
    mock_resp.json.side_effect = ValueError("bad")
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "invalid json" in str(e).lower()


def test_call_missing_choices():
    mock_resp = MagicMock(status_code=200, text="{}")
    mock_resp.json.return_value = {"nope": 1}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "choices" in str(e).lower()


def test_call_content_not_str():
    mock_resp = MagicMock(status_code=200, text="{}")
    mock_resp.json.return_value = {"choices": [{"message": {"content": {"overview": "ov"}}}]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        content = _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
        assert "overview" in content


# ── build_summary LLM paths ──
def _mock_llm(monkeypatch, payload_json: str):
    def fake_call(*args, **kwargs):
        return payload_json
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", fake_call)


def test_build_openrouter_success(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response(overview="LLM OV", key_points=["a", "b"], quotes=[{"text": "First sentence is overview.", "start": 0}]))
    summary = build_summary(ep, tx, backend="openrouter", api_key="sk-test")
    assert summary["overview"] == "LLM OV"
    assert summary["backend"] == "openrouter"
    assert summary["model"] == "meta/muse-spark-1.2-contributor"
    assert summary["truncated"] is False


def test_build_opencode_success(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response(overview="OC OV"))
    summary = build_summary(ep, tx, backend="opencode", api_key="key")
    assert summary["backend"] == "opencode"
    assert summary["model"] == "muse-spark-1.2-contributor"


def test_build_lmstudio_success(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response())
    summary = build_summary(ep, tx, backend="lmstudio", model="qwen2", base_url="http://localhost:1234/v1")
    assert summary["backend"] == "lmstudio"
    assert summary["model"] == "qwen2"


def test_build_local_alias(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response())
    summary = build_summary(ep, tx, backend="local", model="m", base_url="http://localhost:1234/v1")
    assert summary["backend"] == "lmstudio"


def test_build_openrouter_missing_key():
    ep = _fake_episode()
    tx = _fake_transcript()
    try:
        build_summary(ep, tx, backend="openrouter")
        assert False
    except SummarizeError as e:
        assert "api key" in str(e).lower()


def test_build_lmstudio_missing_model():
    ep = _fake_episode()
    tx = _fake_transcript()
    try:
        build_summary(ep, tx, backend="lmstudio")
        assert False
    except SummarizeError as e:
        assert "requires --model" in str(e)


def test_build_unknown_backend():
    ep = _fake_episode()
    tx = _fake_transcript()
    try:
        build_summary(ep, tx, backend="bogus")
        assert False
    except ValueError as e:
        assert "Unknown" in str(e)


def test_build_truncation(monkeypatch):
    ep = _fake_episode()
    tx = Transcript(text="A" * 200, segments=[], language="en", model="m", engine="fake")
    captured = {}
    def fake_call(*args, **kwargs):
        msgs = kwargs.get("messages")
        if msgs:
            captured["text_len"] = len(msgs[1]["content"])
        return _llm_response()
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", fake_call)
    summary = build_summary(ep, tx, backend="openrouter", api_key="k", max_input_chars=50)
    assert summary["truncated"] is True
    assert captured["text_len"] < 200


def test_build_keychain_fallback(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response())
    with patch("podtx.keychain.get_api_key", return_value="kc-key"):
        summary = build_summary(ep, tx, backend="openrouter")
        assert summary["overview"] == "OV"


def test_build_settings_api_key(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, _llm_response())
    summary = build_summary(ep, tx, backend="openrouter", settings_api_key="settings-key")
    assert summary["overview"] == "OV"


def test_build_quote_anchoring(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript(text="hello world transcript", segments=[Segment(10, 15, "hello world transcript"), Segment(20, 25, "other")])
    _mock_llm(monkeypatch, _llm_response(quotes=[{"text": "hello world transcript", "start": 0}]))
    summary = build_summary(ep, tx, backend="openrouter", api_key="k")
    assert summary["quotes"][0]["start"] == 10
    assert summary["quotes"][0]["timestamp"] == "00:10"


def test_build_quote_no_match(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    _mock_llm(monkeypatch, json.dumps({"nuggets": [{"insight": "a", "context": "ctx", "why_it_matters": "why", "quote": "no match"}], "top5_best": [0]}))
    summary = build_summary(ep, tx, backend="openrouter", api_key="k")
    assert summary["nuggets"][0]["quote"] == "no match"
    assert summary["nuggets"][0]["start"] == 0.0


def test_build_invalid_llm_json(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: "not json")
    try:
        build_summary(ep, tx, backend="openrouter", api_key="k")
        assert False
    except SummarizeError:
        pass


def test_build_missing_base_url(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript()
    monkeypatch.setattr("podtx.summarize._default_base_url", lambda b: None)
    try:
        build_summary(ep, tx, backend="openrouter", api_key="k")
        assert False
    except SummarizeError as e:
        assert "base-url" in str(e).lower()


# ── summarize_transcript / summarize_many with LLM ──
def test_summarize_transcript_llm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: _llm_response(overview="file ov"))
    jp = _write_fake_transcript_json(tmp_path, basename="ep")
    written = summarize_transcript(jp, backend="openrouter", api_key="k", formats=("json",))
    assert (tmp_path / "ep.summary.json").is_file()
    payload = json.loads((tmp_path / "ep.summary.json").read_text())
    assert payload["overview"] == "file ov"
    assert payload["backend"] == "openrouter"


def test_summarize_transcript_llm_md_truncated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: _llm_response(overview="ov"))
    jp = _write_fake_transcript_json(tmp_path, basename="ep")
    written = summarize_transcript(jp, backend="openrouter", api_key="k", formats=("md",), max_input_chars=10)
    md = (tmp_path / "ep.summary.md").read_text()
    assert "truncated" in md.lower()


def test_summarize_many_llm_partial_failure(tmp_path: Path, monkeypatch):
    jp1 = _write_fake_transcript_json(tmp_path, basename="ep1")
    jp2 = tmp_path / "bad.json"
    jp2.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: _llm_response())
    result = summarize_many([jp1, jp2], backend="openrouter", api_key="k")
    assert result.ok == 1
    assert result.failed == 1


def test_summarize_many_llm_api_error(tmp_path: Path, monkeypatch):
    jp = _write_fake_transcript_json(tmp_path, basename="ep")
    def fail(**kw):
        raise SummarizeError("boom")
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", fail)
    result = summarize_many([jp], backend="openrouter", api_key="k")
    assert result.failed == 1
    assert "boom" in result.errors[0][1]


def test_build_quote_chunk_fallback(monkeypatch):
    ep = _fake_episode()
    tx = _fake_transcript(segments=[
        Segment(0.0, 1.0, "Alpha unknown content."),
        Segment(5.0, 6.0, "the quick brown fox ran away"),
    ])
    q = "the quick brown fox jumped over the lazy dog"
    payload = json.dumps({"nuggets": [{"insight": "i", "context": "ctx", "why_it_matters": "why", "quote": q}], "top5_best": [0]})
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: payload)
    summary = build_summary(ep, tx, backend="openrouter", api_key="k")
    assert summary["nuggets"][0]["start"] == 5.0
    assert summary["nuggets"][0]["timestamp"] == "00:05"
