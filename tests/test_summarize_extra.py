from __future__ import annotations

import json
from unittest.mock import patch

from podtx.summarize import (
    SummarizeError,
    _default_base_url,
    _default_model,
    _extract_json_content,
    _resolve_api_key,
    _validate_llm_payload,
    build_summary,
)
from podtx.models import Episode, Segment, Transcript
from datetime import datetime, timezone


def test_default_model_unknown():
    assert _default_model("fake") is None
    assert _default_model("bogus") is None


def test_default_base_unknown():
    assert _default_base_url("fake") is None
    assert _default_base_url("bogus") is None


def test_resolve_api_key_with_service():
    with patch("podtx.keychain.get_api_key", return_value="sec"):
        assert _resolve_api_key("openrouter", None, None, service="svc", account="acct") == "sec"


def test_resolve_api_key_exception_path():
    with patch("podtx.keychain.get_api_key", side_effect=Exception("boom")):
        assert _resolve_api_key("openrouter", None, None, service="svc", account="acct") is None
        assert _resolve_api_key("openrouter", None, None) is None


def test_validate_payload_non_dict():
    try:
        _validate_llm_payload("not a dict")  # type: ignore
        assert False
    except SummarizeError:
        pass


def test_validate_payload_missing_key_points_list():
    try:
        _validate_llm_payload({"nuggets": [{"insight": "", "context": "ctx", "why_it_matters": "why", "quote": ""}]})
        assert False
    except SummarizeError:
        pass


def test_validate_payload_quotes_not_list():
    out = _validate_llm_payload({"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [0]})
    assert out["nuggets"][0]["quote"] == ""


def test_validate_payload_quotes_none():
    out = _validate_llm_payload({"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [0]})
    assert out["nuggets"][0]["quote"] == ""


def test_extract_json_with_trailing():
    raw = 'xxx {"overview":"ov","key_points":["a"]} yyy'
    out = _extract_json_content(raw)
    assert out["overview"] == "ov"


def test_extract_invalid_json_block():
    raw = "{ not json }"
    try:
        _extract_json_content(raw)
        assert False
    except SummarizeError:
        pass


def test_build_summary_fake_no_text():
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="   ", segments=[], language="en", model="m", engine="fake")
    summary = build_summary(ep, tx, backend="fake")
    assert summary["overview"] == ""
    assert summary["key_points"] == []


def test_build_summary_llm_bad_start_type(monkeypatch):
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="hello world", segments=[], language="en", model="m", engine="fake")
    fake_payload = json.dumps({"nuggets": [{"insight": "a", "context": "ctx", "why_it_matters": "why", "quote": "hi"}], "top5_best": [0]})
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", lambda **kw: fake_payload)
    summary = build_summary(ep, tx, backend="openrouter", api_key="k")
    # quote "hi" not in segments, so start 0
    assert summary["nuggets"][0]["start"] == 0.0


def test_build_summary_llm_empty_text_fallback(monkeypatch):
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com")
    tx = Transcript(text="   ", segments=[Segment(0,1,"seg text")], language="en", model="m", engine="fake")
    fake_payload = json.dumps({"nuggets": [{"insight": "ov", "context": "ctx", "why_it_matters": "why", "quote": ""}], "top5_best": [0]})
    # capture prompt to ensure text fallback used
    captured = {}
    def fake_call(**kw):
        captured["messages"] = kw.get("messages")
        return fake_payload
    monkeypatch.setattr("podtx.summarize._call_openai_compatible", fake_call)
    summary = build_summary(ep, tx, backend="openrouter", api_key="k")
    assert "seg text" in captured["messages"][1]["content"]


def test_summary_markdown_with_model_and_truncated(tmp_path):
    from podtx.summarize import _summary_to_markdown
    md = _summary_to_markdown({"title": "T", "show": "S", "episode": 1, "overview": "ov", "key_points": ["kp"], "quotes": [], "backend": "openrouter", "model": "m", "truncated": True})
    assert "m" in md
    assert "truncated" in md.lower()


def test_resolve_api_key_falsy_keychain():
    with patch("podtx.keychain.get_api_key", return_value=""):
        assert _resolve_api_key("fake", None, None, service="svc", account="acct") is None


def test_validate_payload_legacy_blank_key_point_and_string_quote():
    out = _validate_llm_payload({
        "overview": "ov",
        "key_points": ["kp1", "", "kp2"],
        "quotes": ["q one", "q two"],
    })
    assert [n["insight"] for n in out["nuggets"]] == ["kp1", "kp2"]
    assert out["nuggets"][0]["quote"] == "q one"


def test_build_summary_fake_truncates_long_nugget_quote():
    from podtx.summarize import build_summary
    ep = Episode(guid="g", title="T", enclosure_url="https://example.com", show_title="S")
    tx = Transcript(
        text="First sentence here. Second sentence here. Third sentence here.",
        segments=[Segment(0.0, 60.0, " ".join(["word"] * 50))],
        language="en", model="m", engine="fake",
    )
    summary = build_summary(ep, tx, backend="fake")
    assert len(summary["nuggets"][0]["quote"].split()) == 30


def test_summarize_with_llm_unknown_backend_requires_model():
    from podtx.summarize import _summarize_with_llm, SummarizeError
    ep = Episode("g", "T", "https://example.com")
    tx = Transcript(text="some text here. more words.", segments=[], language="en", model="m", engine="fake")
    try:
        _summarize_with_llm(
            ep, tx,
            backend="bogus", model=None, api_key=None, base_url=None,
            timeout=60, temperature=0.3, max_input_chars=None, basename="",
        )
        assert False
    except SummarizeError:
        pass


def test_summary_markdown_no_model_backend():
    from podtx.summarize import _summary_to_markdown
    md = _summary_to_markdown({"title": "T", "backend": "openrouter", "nuggets": []})
    assert "Backend: openrouter" in md


def test_summary_markdown_nugget_variants():
    from podtx.summarize import _summary_to_markdown
    nuggets = [
        {"insight": "Full nugget", "context": "c", "why_it_matters": "w", "quote": "q", "timestamp": "00:10"},
        {"insight": "No meta", "quote": ""},
        {"insight": "Missing why", "context": "c2", "quote": "q2", "timestamp": "00:20"},
    ]
    md = _summary_to_markdown({"title": "T", "backend": "fake", "nuggets": nuggets, "top5_best": [0, 5, 1, 2]})
    assert "Best of Show" in md
    assert "Missing why" in md
    assert "Full nugget" in md


def test_validate_payload_legacy_quote_int_falls_fast():
    out = _validate_llm_payload({"overview": "ov", "key_points": ["kp1"], "quotes": [123]})
    assert out["nuggets"][0]["quote"] == ""
