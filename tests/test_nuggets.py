from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from podtx.format_cmd import TranscriptJsonError
from podtx.models import Episode, Segment, Transcript
from podtx.nuggets import (
    NUGGETS_PROMPT_VERSION,
    BatchNuggetsResult,
    NuggetsError,
    _chunk_segments,
    _clean_score,
    _demote_quote,
    _extract_chunk_with_retry,
    _fake_nuggets,
    _format_timestamp,
    _locate_quote,
    _merge_nuggets,
    _normalize,
    _nuggets_to_markdown,
    _parse_json,
    _rubric_messages,
    _sidecar_fresh,
    _split_chunks,
    _split_text,
    _transcript_text,
    _valid_backend,
    _validate_payload,
    _verify_quotes,
    _write_nugget_files,
    extract_nuggets_transcript,
    nuggets_many,
)
from podtx.providers import ProviderError
from podtx.writers import write_outputs


def _episode(**overrides) -> Episode:
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


def _transcript(
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


def _write_transcript(tmp_path: Path, basename: str = "ep", transcript: Transcript | None = None) -> Path:
    ep = _episode()
    tx = transcript or _transcript()
    write_outputs(
        out_dir=tmp_path,
        basename=basename,
        episode=ep,
        transcript=tx,
        formats=("txt", "json"),
        readable=False,
        cleanup=False,
    )
    return tmp_path / f"{basename}.json"


def _payload(*nuggets: dict) -> str:
    return json.dumps({"nuggets": list(nuggets)})


def _nugget(insight="Insight here.", quote="First sentence is overview.", **over) -> dict:
    n = {
        "insight": insight,
        "context": "Fake Show — ep",
        "why_it_matters": "why it matters",
        "quote": quote,
        "scores": {"T": 2, "S": 2, "E": 1, "A": 1},
        "tag": "eng",
    }
    n.update(over)
    return n


class _StubProvider:
    def __init__(self, results: list[str]) -> None:
        self.results = list(results)
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def complete(self, messages, *, timeout=120.0, temperature=0.3) -> str:
        self.timeout = timeout
        self.temperature = temperature
        self.calls += 1
        return self.results.pop(0)


def test_prompt_version_constant() -> None:
    assert NUGGETS_PROMPT_VERSION == "nuggets-rubric-1"


def test_valid_backend() -> None:
    assert _valid_backend("fake") == "fake"
    assert _valid_backend("local") == "lmstudio"
    assert _valid_backend("  OpEnRouTer ") == "openrouter"
    assert _valid_backend("openai") == "openai"
    with pytest.raises(NuggetsError):
        _valid_backend("bogus")
    with pytest.raises(NuggetsError):
        _valid_backend("")


def test_format_timestamp() -> None:
    assert _format_timestamp(0) == "00:00"
    assert _format_timestamp(1.9) == "00:01"
    assert _format_timestamp(61.0) == "01:01"
    assert _format_timestamp(3659.0) == "01:00:59"
    assert _format_timestamp(0.4) == "00:00"


def test_rubric_messages() -> None:
    msgs = _rubric_messages(_episode(), "transcript body here", "ep")
    assert len(msgs) == 2
    system, user = msgs
    assert system["role"] == "system"
    assert "Extract 3-7 nuggets" in system["content"]
    assert "Timestamp" not in system["content"]
    assert "Fake Show" in user["content"]
    assert "Fake Episode Title" in user["content"]
    assert "ep" in user["content"]
    assert "transcript body here" in user["content"]


def test_parse_json() -> None:
    assert _parse_json('{"nuggets": []}') == {"nuggets": []}
    raw = 'Sure, here you go:\n{"nuggets": [{"a": 1}]}\nHope that helps'
    assert _parse_json(raw) == {"nuggets": [{"a": 1}]}
    with pytest.raises(NuggetsError):
        _parse_json("not json at all")
    with pytest.raises(NuggetsError):
        _parse_json("A {not valid json} B")
    with pytest.raises(NuggetsError):
        _parse_json("[1, 2, 3]")


def test_clean_score() -> None:
    assert _clean_score(2) == 2
    assert _clean_score(3) == 2
    assert _clean_score(-5) == 0
    assert _clean_score("1") == 1
    assert _clean_score("x") == 0
    assert _clean_score(None) == 0


def test_validate_payload_basic() -> None:
    out = _validate_payload(
        {"nuggets": [_nugget(scores={"T": 2, "S": 2, "E": 2, "A": 2})]}
    )
    assert len(out) == 1
    assert out[0]["total"] == 8


def test_validate_payload_filters() -> None:
    payload = {
        "nuggets": [
            _nugget(insight="below bar", scores={"T": 1, "S": 1, "E": 1, "A": 1}),
            _nugget(insight="no scores"),
            _nugget(insight=""),
            "not-a-dict",
            _nugget(insight="general tag", tag="general"),
        ]
    }
    out = _validate_payload(payload)
    assert [n["insight"] for n in out] == ["no scores", "general tag"]


def test_validate_payload_normalizes() -> None:
    long_quote = " ".join(f"w{i}" for i in range(40))
    out = _validate_payload(
        {
            "nuggets": [
                _nugget(
                    insight="kept",
                    quote=long_quote,
                    tag="BUSINESS",
                    scores={"T": 2, "S": 2, "E": 1, "A": 1},
                ),
                _nugget(insight="dropped", scores="oops"),
            ]
        }
    )
    assert len(out) == 1
    n = out[0]
    assert len(n["quote"].split()) == 30
    assert n["tag"] == "eng"
    assert n["scores"] == {"T": 2, "S": 2, "E": 1, "A": 1}


def test_validate_payload_caps_count() -> None:
    payload = {
        "nuggets": [
            _nugget(insight=f"n{i}", scores={"T": 2, "S": 2, "E": 1, "A": 1})
            for i in range(10)
        ]
    }
    out = _validate_payload(payload)
    assert len(out) == 7


def test_validate_payload_missing_array() -> None:
    with pytest.raises(NuggetsError):
        _validate_payload({})
    with pytest.raises(NuggetsError):
        _validate_payload({"nuggets": []})


def test_validate_payload_all_dropped() -> None:
    with pytest.raises(NuggetsError):
        _validate_payload({"nuggets": [_nugget(scores={"T": 0, "S": 0, "E": 0, "A": 0})]})


def test_validate_payload_sorts_desc() -> None:
    out = _validate_payload(
        {
            "nuggets": [
                _nugget(insight="low", scores={"T": 2, "S": 1, "E": 1, "A": 1}),
                _nugget(insight="high", scores={"T": 2, "S": 2, "E": 2, "A": 2}),
            ]
        }
    )
    assert [n["insight"] for n in out] == ["high", "low"]


def test_normalize() -> None:
    assert _normalize("  Hello   World!\nagain ") == "hello world! again"


def test_demote_quote() -> None:
    n = _nugget(
        quote="some quote",
        scores={"T": 2, "S": 2, "E": 2, "A": 2},
        start=5.0,
        end=6.0,
        timestamp="00:05",
    )
    out = _demote_quote(n)
    assert out["quote"] == ""
    assert out["scores"]["E"] == 1
    assert out["total"] == 7
    assert out["timestamp"] == ""
    assert out["start"] == 0.0 and out["end"] == 0.0


def test_verify_quotes_passthrough_when_empty() -> None:
    tx = _transcript()
    n = _nugget(quote="")
    out = _verify_quotes([n], tx)
    assert out[0] is n


def test_verify_quotes_match_segment() -> None:
    tx = _transcript()
    out = _verify_quotes([_nugget(quote="Second sentence also overview.")], tx)
    assert out[0]["timestamp"] == "00:02"
    assert out[0]["start"] == 2.0
    assert out[0]["scores"]["E"] == 1


def test_verify_quotes_demotes_fabricated() -> None:
    tx = _transcript()
    n = _nugget(quote="This sentence is fabricated and appears nowhere.", scores={"T": 2, "S": 2, "E": 2, "A": 2})
    out = _verify_quotes([n], tx)
    assert out[0]["quote"] == ""
    assert out[0]["total"] == 7
    assert out[0]["timestamp"] == ""


def test_verify_quotes_demotes_whitespace() -> None:
    tx = _transcript()
    n = _nugget(quote="   ")
    out = _verify_quotes([n], tx)
    assert out[0] is n


def test_verify_quotes_chunk_prefix_fallback() -> None:
    segments = [
        Segment(0.0, 3.0, "beta gamma delta epsilon zeta eta"),
        Segment(3.0, 6.0, "theta iota kappa lambda mu"),
    ]
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    tx = Transcript(text=text, segments=segments, language="en", model="m", engine="fake")
    quote = "gamma delta epsilon zeta eta theta iota kappa"
    out = _verify_quotes([_nugget(quote=quote)], tx)
    assert out[0]["timestamp"] == "00:00"
    assert out[0]["start"] == 0.0


def test_verify_quotes_timestamps_boundary_crossing() -> None:
    segments = [
        Segment(0.0, 3.0, "one tiny"),
        Segment(3.0, 6.0, "small phrase two"),
    ]
    text = "one tiny small phrase two"
    tx = Transcript(text=text, segments=segments, language="en", model="m", engine="fake")
    quote = "tiny small"
    out = _verify_quotes([_nugget(quote=quote, scores={"T": 2, "S": 2, "E": 2, "A": 2})], tx)
    assert out[0]["quote"] == quote
    assert out[0]["timestamp"] == "00:00"
    assert out[0]["start"] == 0.0


def test_verify_quotes_timestamps_offset_fallback() -> None:
    segments = [
        Segment(0.0, 3.0, "one two three four five"),
        Segment(3.0, 6.0, "six seven eight nine ten"),
    ]
    text = "one two three four five six seven eight nine ten"
    tx = Transcript(text=text, segments=segments, language="en", model="m", engine="fake")
    quote = "four five six seven"
    out = _verify_quotes([_nugget(quote=quote, scores={"T": 2, "S": 2, "E": 2, "A": 2})], tx)
    assert out[0]["quote"] == quote
    assert out[0]["timestamp"] == "00:00"
    assert out[0]["start"] == 0.0


def test_verify_quotes_demotes_without_segments() -> None:
    tx = Transcript(text="alpha wave", segments=[], language="en", model="m", engine="fake")
    out = _verify_quotes([_nugget(quote="alpha")], tx)
    assert out[0]["quote"] == ""
    assert out[0]["timestamp"] == ""


def test_verify_quotes_timestamps_later_segment() -> None:
    segments = [
        Segment(0.0, 3.0, "a b c"),
        Segment(3.0, 6.0, "d e f"),
        Segment(6.0, 9.0, "g h i"),
    ]
    text = "a b c d e f g h i"
    tx = Transcript(text=text, segments=segments, language="en", model="m", engine="fake")
    out = _verify_quotes([_nugget(quote="e f g")], tx)
    assert out[0]["quote"] == "e f g"
    assert out[0]["timestamp"] == "00:03"
    assert out[0]["start"] == 3.0


def test_locate_quote_no_match_in_full() -> None:
    segs = [Segment(0.0, 3.0, "one tiny")]
    assert _locate_quote("totally different", segs, "one tiny") is None


def test_chunk_segments_no_segments() -> None:
    tx = Transcript(text="", segments=[], language="en", model="m", engine="fake")
    assert _chunk_segments(tx, budget_chars=20, overlap_chars=2) == [[0]]


def test_split_chunks_single() -> None:
    tx = _transcript()
    assert _split_chunks(tx, max_input_chars=10_000) == [tx.text]
    tx2 = Transcript(text="no segments", segments=[], language="en", model="m", engine="fake")
    assert _split_chunks(tx2, max_input_chars=2) == ["no", "segments"]


def test_split_chunks_multi_with_overlap() -> None:
    segments = [Segment(float(i), float(i) + 1, str(i)) for i in range(10)]
    tx = Transcript(
        text=" ".join(str(i) for i in range(10)),
        segments=segments,
        language="en",
        model="m",
        engine="fake",
    )
    chunks = _split_chunks(tx, max_input_chars=2)
    assert chunks == [str(i) for i in range(10)]


def test_chunk_segments_carry_overlap() -> None:
    segments = [Segment(float(i), float(i) + 1, str(i)) for i in range(10)]
    tx = Transcript(
        text=" ".join(str(i) for i in range(10)),
        segments=segments,
        language="en",
        model="m",
        engine="fake",
    )
    idx = _chunk_segments(tx, budget_chars=2, overlap_chars=2)
    assert idx == [[0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]


def test_split_chunks_oversized_segment() -> None:
    segments = [Segment(0.0, 1.0, "aa bb cc dd ee")]
    tx = Transcript(text="aa bb cc dd ee", segments=segments, language="en", model="m", engine="fake")
    pieces = _split_chunks(tx, max_input_chars=4)
    assert pieces == ["aa", "bb", "cc", "dd", "ee"]
    assert all(len(p) <= 4 for p in pieces)


def test_split_text_carries_overlap() -> None:
    assert _split_text("aa bb cc dd", max_chars=6, overlap_chars=3) == [
        "aa bb",
        "bb cc",
        "cc dd",
    ]


def test_transcript_text() -> None:
    tx = _transcript(text="   ")
    assert _transcript_text(tx) == " ".join(s.text for s in tx.segments)


def test_extract_chunk_with_retry_first_try() -> None:
    provider = _StubProvider([_payload(_nugget())])
    out = _extract_chunk_with_retry(provider, _episode(), "text", "ep", timeout=10.0, temperature=0.2)
    assert out[0]["insight"] == "Insight here."
    assert provider.calls == 1
    assert provider.timeout == 10.0
    assert provider.temperature == 0.2


def test_extract_chunk_with_retry_second_try() -> None:
    provider = _StubProvider(["not json", _payload(_nugget(insight="after retry"))])
    out = _extract_chunk_with_retry(provider, _episode(), "text", "ep", timeout=10.0, temperature=0.2)
    assert out[0]["insight"] == "after retry"
    assert provider.calls == 2


def test_extract_chunk_with_retry_fails_both() -> None:
    provider = _StubProvider(["not json", "also not json"])
    with pytest.raises(NuggetsError, match="after 1 retry"):
        _extract_chunk_with_retry(provider, _episode(), "text", "ep", timeout=10.0, temperature=0.2)
    assert provider.calls == 2


def test_merge_nuggets() -> None:
    def total(insight, t=0, s=0, e=0, a=0):
        return {
            "insight": insight,
            "scores": {"T": t, "S": s, "E": e, "A": a},
            "total": t + s + e + a,
        }

    merged = _merge_nuggets(
        [
            total("dup", 2, 1, 1, 1),
            total("  DUP ", 2, 2, 1, 1),
            total("", 2, 2, 2, 2),
        ]
    )
    assert [n["insight"] for n in merged] == ["  DUP "]


def test_fake_nuggets_full() -> None:
    segs = [
        Segment(0.0, 1.0, "zero"),
        Segment(1.0, 2.0, "one"),
        Segment(2.0, 3.0, "two"),
        Segment(3.0, 4.0, "three"),
        Segment(4.0, 5.0, "four"),
    ]
    tx = Transcript(text="zero one two three four", segments=segs, language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert len(out) == 3
    assert {n["total"] for n in out} == {6}
    assert out[0]["timestamp"] == "00:00"
    assert out[0]["tag"] == "general"


def test_fake_nuggets_single_segment() -> None:
    segs = [Segment(0.0, 1.0, "only one")]
    tx = Transcript(text="only one", segments=segs, language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert len(out) == 1
    assert out[0]["quote"] == "only one"


def test_fake_nuggets_three_segments() -> None:
    segs = [
        Segment(0.0, 1.0, "one"),
        Segment(1.0, 2.0, "two"),
        Segment(2.0, 3.0, "three"),
    ]
    tx = Transcript(text="one two three", segments=segs, language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert [n["quote"] for n in out] == ["one", "two", "three"]


def test_fake_nuggets_skips_blank_and_truncates() -> None:
    long_text = " ".join(f"word{i}" for i in range(45))
    segs = [
        Segment(0.0, 1.0, "   "),
        Segment(1.0, 2.0, long_text),
    ]
    tx = Transcript(text="", segments=segs, language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert len(out) == 1
    assert len(out[0]["quote"].split()) == 30
    assert out[0]["insight"].endswith("...")


def test_fake_nuggets_empty_fallback() -> None:
    tx = Transcript(text="", segments=[], language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert len(out) == 0


def test_fake_nuggets_text_fallback() -> None:
    tx = Transcript(text="a transcript with no segments", segments=[], language="en", model="m", engine="fake")
    out = _fake_nuggets(_episode(), tx, "ep")
    assert len(out) == 1
    assert out[0]["total"] == 5
    assert out[0]["tag"] == "eng"


def test_nuggets_to_markdown_full() -> None:
    data = {
        "title": "T",
        "show": "S",
        "episode": 3,
        "backend": "fake",
        "model": None,
        "prompt_version": "nuggets-rubric-1",
        "chunked": False,
        "nuggets": [_nugget(quote="First sentence is overview.", total=8, timestamp="00:00")],
    }
    md = _nuggets_to_markdown(data)
    assert "# Nuggets: T" in md
    assert "Show: S" in md
    assert "Episode: 3" in md
    assert "fake (offline, no network)" in md
    assert "Rubric: nuggets-rubric-1" in md
    assert "> \"First sentence is overview.\" — [00:00]" in md
    assert "*Why:* why it matters" in md
    assert "*Context:* Fake Show — ep" in md


def test_nuggets_to_markdown_no_quote() -> None:
    data = {
        "title": "T",
        "backend": "fake",
        "prompt_version": "v",
        "nuggets": [
            {
                "insight": "no quote insight",
                "tag": "eng",
                "total": 5,
                "quote": "",
                "timestamp": "",
                "why_it_matters": "still has why",
                "context": "ctx",
            }
        ],
    }
    md = _nuggets_to_markdown(data)
    assert "no quote insight" in md
    assert '">' not in md.split("###")[1]
    assert "*Why:* still has why" in md


def test_nuggets_to_markdown_model_and_chunked() -> None:
    data = {
        "backend": "openrouter",
        "model": "muse",
        "prompt_version": "v",
        "chunked": True,
        "nuggets": [
            {
                "insight": "  spaced  insight ",
                "tag": "general",
                "total": 7,
                "quote": "no ts quote",
                "timestamp": "",
                "why_it_matters": "",
                "context": "",
            }
        ],
    }
    md = _nuggets_to_markdown(data)
    assert "Nuggets" in md
    assert "Backend: openrouter (muse)" in md
    assert "Chunked: yes (episode exceeded input budget)" in md
    assert "spaced  insight" in md
    assert '> "no ts quote"' in md


def test_nuggets_to_markdown_plain_backend_no_model() -> None:
    data = {
        "backend": "openrouter",
        "prompt_version": "v",
        "nuggets": [],
    }
    md = _nuggets_to_markdown(data)
    assert "Backend: openrouter" in md
    assert "offline" not in md


def test_nuggets_to_markdown_empty() -> None:
    md = _nuggets_to_markdown({"nuggets": [], "backend": "fake"})
    assert "_No nuggets_" in md


def test_write_nugget_files() -> None:
    data = {"nuggets": [], "backend": "fake"}
    written = _write_nugget_files(data, out_dir=Path("/tmp/nug-write-test"), basename="ep", formats=("json", "md"))
    assert len(written) == 2
    assert written[0].name == "ep.nuggets.json"
    assert written[1].name == "ep.nuggets.md"
    assert json.loads(written[0].read_text(encoding="utf-8")) == data


def test_write_nugget_files_unsupported() -> None:
    with pytest.raises(NuggetsError):
        _write_nugget_files({"nuggets": []}, out_dir=Path("/tmp/nug-write-test"), basename="ep", formats=("vcf",))


def test_sidecar_fresh(tmp_path: Path) -> None:
    p = tmp_path / "ep.nuggets.json"
    fresh = {
        "prompt_version": NUGGETS_PROMPT_VERSION,
        "backend": "openrouter",
        "model": "muse",
    }
    p.write_text(json.dumps(fresh), encoding="utf-8")
    assert _sidecar_fresh(p, backend="openrouter", model="muse", prompt_version=NUGGETS_PROMPT_VERSION)
    assert not _sidecar_fresh(p, backend="fake", model="muse", prompt_version=NUGGETS_PROMPT_VERSION)
    assert not _sidecar_fresh(p, backend="openrouter", model=None, prompt_version=NUGGETS_PROMPT_VERSION)
    assert not _sidecar_fresh(p, backend="openrouter", model="muse", prompt_version="other")
    assert not _sidecar_fresh(tmp_path / "missing.json", backend="openrouter", model="muse", prompt_version=NUGGETS_PROMPT_VERSION)
    p.write_text("{corrupt", encoding="utf-8")
    assert not _sidecar_fresh(p, backend="openrouter", model="muse", prompt_version=NUGGETS_PROMPT_VERSION)
    p.write_text('["not a dict"]', encoding="utf-8")
    assert not _sidecar_fresh(p, backend="openrouter", model=None, prompt_version=NUGGETS_PROMPT_VERSION)


def test_extract_fake_writes_sidecars(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    run = extract_nuggets_transcript(path)
    assert not run.skipped
    assert {w.name for w in run.written} == {"ep.nuggets.json"}
    data = json.loads((tmp_path / "ep.nuggets.json").read_text(encoding="utf-8"))
    assert data["backend"] == "fake"
    assert data["model"] is None
    assert data["prompt_version"] == NUGGETS_PROMPT_VERSION
    assert data["chunked"] is False
    assert data["nuggets"]


def test_extract_skips_when_fresh(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    extract_nuggets_transcript(path)
    run = extract_nuggets_transcript(path)
    assert run.skipped
    assert run.written == []

    run = extract_nuggets_transcript(path, force=True)
    assert not run.skipped


def test_extract_fake_multiple_formats(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    run = extract_nuggets_transcript(path, formats=("json", "md"))
    assert {w.name for w in run.written} == {"ep.nuggets.json", "ep.nuggets.md"}


def test_extract_out_dir(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    out = tmp_path / "out"
    run = extract_nuggets_transcript(path, out_dir=out)
    assert not run.skipped
    assert (out / "ep.nuggets.json").is_file()


def test_extract_provider_path(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    payload = _payload(
        _nugget(insight="LLM insight", quote="Second sentence also overview."),
        _nugget(insight="spans", quote="not present"),
    )
    with patch("podtx.nuggets.build_provider", return_value=_StubProvider([payload])) as bp:
        run = extract_nuggets_transcript(path, backend="openrouter", api_key="k")
    data = json.loads((tmp_path / "ep.nuggets.json").read_text(encoding="utf-8"))
    assert data["backend"] == "openrouter"
    assert data["model"] == "meta/muse-spark-1.2-contributor"
    assert data["chunked"] is False
    insights = {n["insight"] for n in data["nuggets"]}
    assert "LLM insight" in insights
    assert not any(n["quote"] for n in data["nuggets"] if n["insight"] == "spans")
    bp.assert_called_once()


def test_extract_provider_chunked(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    payload = _payload(
        _nugget(insight="c1", quote="First sentence is overview."),
        _nugget(insight="c2", quote="Fifth extra."),
    )
    provider = _StubProvider([payload, payload])
    with patch("podtx.nuggets.build_provider", return_value=provider):
        run = extract_nuggets_transcript(path, backend="openrouter", api_key="k", max_input_chars=60)
    assert provider.calls == 2
    data = json.loads((tmp_path / "ep.nuggets.json").read_text(encoding="utf-8"))
    assert data["chunked"] is True
    insights = {n["insight"] for n in data["nuggets"]}
    assert insights == {"c1", "c2"}


def test_extract_provider_requires_keychain_args(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    with patch("podtx.nuggets.build_provider", return_value=_StubProvider([_payload(_nugget())])) as bp:
        extract_nuggets_transcript(
            path,
            backend="openrouter",
            model="custom",
            base_url="https://custom/v1",
            settings_api_key="sk-set",
            service="s",
            account="a",
        )
    bp.assert_called_once()
    kwargs = bp.call_args.kwargs
    assert kwargs["model"] == "custom"
    assert kwargs["base_url"] == "https://custom/v1"
    assert kwargs["settings_api_key"] == "sk-set"
    assert kwargs["service"] == "s"
    assert kwargs["account"] == "a"


def test_extract_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(TranscriptJsonError):
        extract_nuggets_transcript(bad)


def test_extract_unknown_backend(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    with pytest.raises(NuggetsError):
        extract_nuggets_transcript(path, backend="bogus")


def test_extract_openai_requires_model(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    with pytest.raises(ProviderError):
        extract_nuggets_transcript(path, backend="openai", api_key="k")


def test_nuggets_many_continues_on_errors(tmp_path: Path) -> None:
    good = _write_transcript(tmp_path, basename="one")
    good2 = _write_transcript(tmp_path, basename="two")
    bad = tmp_path / "bad.json"
    bad.write_text("oops", encoding="utf-8")
    result = nuggets_many([good, bad, good2])
    assert result.ok == 2
    assert result.failed == 1
    assert result.skipped == 0
    assert len(result.written) == 2
    assert result.errors[0][0] == bad

    result2 = nuggets_many([good, good2])
    assert result2.ok == 0
    assert result2.skipped == 2


def test_nuggets_many_counts_provider_errors(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    result = nuggets_many([path], backend="openai", api_key="k")
    assert result.failed == 1
    assert "requires --model" in result.errors[0][1]


def test_batch_result_defaults() -> None:
    r = BatchNuggetsResult()
    assert r.ok == 0 and r.failed == 0 and r.skipped == 0
    assert r.written == [] and r.errors == []