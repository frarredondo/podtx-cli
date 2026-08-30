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
    _default_base_url,
    _default_model,
    _duration_minutes,
    _nugget_target_for_duration,
    _resolve_api_key,
    _validate_llm_payload,
    build_summary,
)

# _default_base_url
def test_default_base_lmstudio():
    assert _default_base_url("lmstudio") == "http://localhost:1234/v1"
    assert _default_base_url("local") == "http://localhost:1234/v1"
    assert _default_base_url("unknown") is None
    assert _default_model("lmstudio") is None
    assert _default_model("unknown") is None

# _resolve_api_key with service/account
def test_resolve_api_key_service_success():
    with patch("podtx.keychain.get_api_key", return_value="sec"):
        assert _resolve_api_key("openrouter", None, None, service="svc", account="acct") == "sec"

def test_resolve_api_key_service_exception():
    with patch("podtx.keychain.get_api_key", side_effect=Exception("boom")):
        # should not raise, returns None
        assert _resolve_api_key("openrouter", None, None, service="svc", account="acct") is None
        # also provider default exception path
        assert _resolve_api_key("openrouter", None, None) is None

def test_resolve_api_key_provider_default():
    with patch("podtx.keychain.get_api_key", return_value="provider-sec"):
        # provider default for openrouter
        assert _resolve_api_key("openrouter", None, None) == "provider-sec"
        # opencode
        assert _resolve_api_key("opencode", None, None) == "provider-sec"

# _duration_minutes
def test_duration_minutes_with_segments():
    tx = Transcript(text="hello", segments=[Segment(0, 60, "hi"), Segment(60, 120, "hi2")], language="en", model="m", engine="fake")
    assert _duration_minutes(tx) == 2.0

def test_duration_minutes_empty_segments_fallback_wordcount():
    tx = Transcript(text="hello world " * 150, segments=[], language="en", model="m", engine="fake")
    # 300 words /150 =2
    assert _duration_minutes(tx) == 2.0

def test_duration_minutes_exception_fallback():
    # segments[-1].end raises
    class BadSeg:
        @property
        def end(self):
            raise ValueError("bad")
    tx = Transcript(text="hello world", segments=[BadSeg()], language="en", model="m", engine="fake")  # type: ignore
    # should fallback to wordcount
    assert _duration_minutes(tx) == 2/150

def test_nugget_target_short_long():
    assert "1-3" in _nugget_target_for_duration(10)
    assert "3-7" in _nugget_target_for_duration(70)
    assert "3-5" in _nugget_target_for_duration(30)

# _validate_llm_payload branches
def test_validate_nuggets_not_list():
    try:
        _validate_llm_payload({"nuggets": "notalist"})
        assert False
    except SummarizeError as e:
        assert "nuggets" in str(e).lower()

def test_validate_nuggets_empty():
    try:
        _validate_llm_payload({"nuggets": []})
        assert False
    except SummarizeError:
        pass

def test_validate_nuggets_non_dict_entry():
    payload = {"nuggets": ["notadict", {"insight": "ok", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [1]}
    out = _validate_llm_payload(payload)
    assert len(out["nuggets"]) == 1
    assert out["nuggets"][0]["insight"] == "ok"

def test_validate_nuggets_empty_insight():
    payload = {"nuggets": [{"insight": "   ", "context": "ctx", "why_it_matters": "why", "quote": ""}, {"insight": "ok", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [1]}
    out = _validate_llm_payload(payload)
    assert len(out["nuggets"]) == 1
    assert out["nuggets"][0]["insight"] == "ok"

def test_validate_long_quote_truncated():
    long_quote = "word " * 40  # 40 words
    payload = {"nuggets": [{"insight": "ins", "context": "ctx", "why_it_matters": "why", "quote": long_quote}], "top5_best": [0]}
    out = _validate_llm_payload(payload)
    assert len(out["nuggets"][0]["quote"].split()) == 30

def test_validate_top5_not_list():
    payload = {"nuggets": [{"insight": "a", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": "notalist"}
    out = _validate_llm_payload(payload)
    # should fallback to default ranking
    assert out["top5_best"] == [0]

def test_validate_top5_invalid_indices():
    payload = {"nuggets": [{"insight": "a", "context": "ctx", "why_it_matters": "why", "quote": ""}, {"insight": "b", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": ["bad", 1, -1, 0, 0]}
    out = _validate_llm_payload(payload)
    # "bad" invalid, 1 valid, -1 invalid, 0 valid, duplicate 0 skipped => [1,0] in order
    assert 0 in out["top5_best"] and 1 in out["top5_best"]

def test_validate_legacy_fallback():
    payload = {"overview": "legacy overview", "key_points": ["kp1", "kp2"], "quotes": [{"text": "q1"}, "q2"]}
    out = _validate_llm_payload(payload)
    assert len(out["nuggets"]) == 2
    assert out["nuggets"][0]["insight"] == "kp1"
    # second quote is string "q2"
    assert "q2" in out["nuggets"][1]["quote"] or out["nuggets"][1]["quote"] == "q2"

def test_validate_legacy_no_key_points():
    payload = {"overview": "only overview", "key_points": [], "quotes": []}
    out = _validate_llm_payload(payload)
    assert out["nuggets"][0]["insight"] == "only overview"

def test_validate_missing_nuggets_and_overview():
    try:
        _validate_llm_payload({"foo": "bar"})
        assert False
    except SummarizeError as e:
        assert "nuggets" in str(e).lower()

def test_validate_why_alias():
    payload = {"nuggets": [{"insight": "a", "context": "ctx", "why": "alias why", "quote": ""}], "top5_best": [0]}
    out = _validate_llm_payload(payload)
    assert out["nuggets"][0]["why_it_matters"] == "alias why"

# _call_openai_compatible hints
def test_call_401_go_hint():
    mock_resp = MagicMock(status_code=401, text='{"error":"unauth"}')
    mock_resp.text = '{"error":"unauth"}'
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://opencode.ai/zen/go/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "Go key invalid" in str(e)

def test_call_401_meta_hint():
    mock_resp = MagicMock(status_code=401, text='{"error":"unauth"}')
    mock_resp.text = '{"error":"unauth"}'
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://api.meta.ai/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "Meta API key invalid" in str(e)

def test_call_500_go_hint():
    mock_resp = MagicMock(status_code=500, text='{"error":"internal"}')
    mock_resp.text = '{"error":"internal"}'
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://opencode.ai/zen/go/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "Go 500" in str(e)

def test_call_not_found_hint():
    mock_resp = MagicMock(status_code=200, text="Not Found")
    mock_resp.json.side_effect = ValueError("bad json")
    mock_resp.text = "Not Found"
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        try:
            _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
            assert False
        except SummarizeError as e:
            assert "check --base-url" in str(e)

def test_call_content_not_string():
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": {"key": "val"}}}]}
    mock_resp.text = '{"choices": [{"message": {"content": {"key": "val"}}}]}'
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    with patch("podtx.summarize.httpx.Client", return_value=mock_client):
        content = _call_openai_compatible(base_url="https://example.com/v1", api_key="k", model="m", messages=[])
        assert "key" in content

# _summarize_with_llm error branches
def test_summarize_missing_model_lmstudio():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="hi", segments=[], language="en", model="m", engine="fake")
    try:
        build_summary(ep, tx, backend="lmstudio")
        assert False
    except SummarizeError as e:
        assert "requires --model" in str(e)

def test_summarize_missing_base_url():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="hi", segments=[], language="en", model="m", engine="fake")
    from podtx.summarize import _summarize_with_llm
    with patch("podtx.summarize._default_base_url", return_value=None):
        try:
            _summarize_with_llm(ep, tx, backend="lmstudio", model="m", api_key=None, base_url=None, timeout=60, temperature=0.3, max_input_chars=None, basename="ep")
            assert False
        except SummarizeError as e:
            assert "base-url" in str(e).lower()

def test_summarize_missing_api_key():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="hi", segments=[], language="en", model="m", engine="fake")
    try:
        build_summary(ep, tx, backend="openrouter")
        assert False
    except SummarizeError as e:
        assert "api key" in str(e).lower()

def test_build_summary_unknown_backend():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="hi", segments=[], language="en", model="m", engine="fake")
    try:
        build_summary(ep, tx, backend="bogus")
        assert False
    except ValueError as e:
        assert "Unknown summary backend" in str(e)

# fake with empty text
def test_fake_empty_text():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com", show_title="Show")
    tx = Transcript(text="   ", segments=[], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx, backend="fake", basename="ep.json")
    assert summary["nuggets"][0]["insight"] == "No transcript" or summary["nuggets"][0]["insight"] == ""
    assert "No transcript" in summary["nuggets"][0]["insight"] or summary["overview"] == ""

def test_fake_with_segments():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com", show_title="Show")
    tx = Transcript(text="hello world. second.", segments=[Segment(0,1,"hello"), Segment(1,2,"world")], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx, backend="fake", basename="ep")
    assert len(summary["nuggets"]) >= 1
    assert summary["nuggets"][0]["timestamp"] == "00:00"

# Test markdown fallback when no nuggets but overview
def test_markdown_legacy_fallback():
    from podtx.summarize import _summary_to_markdown
    md = _summary_to_markdown({"title": "T", "overview": "ov", "key_points": ["kp"], "quotes": [{"text": "q", "timestamp": "00:00", "start": 0}], "backend": "fake"})
    assert "Overview" in md

def test_markdown_no_nuggets_no_overview():
    from podtx.summarize import _summary_to_markdown
    md = _summary_to_markdown({"title": "T", "backend": "fake"})
    assert "No overview" in md or "No nuggets" in md

# Test _truncate_text branches
def test_truncate_no_dot():
    from podtx.summarize import _truncate_text
    txt, trunc = _truncate_text("A"*100, 50)
    assert trunc is True
    assert txt.endswith("…")

def test_truncate_with_dot():
    from podtx.summarize import _truncate_text
    txt, trunc = _truncate_text("Sentence one. Sentence two. " + "A"*100, 50)
    assert trunc is True
    assert txt.endswith(".")

