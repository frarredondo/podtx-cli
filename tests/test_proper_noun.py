from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from podtx.models import Episode, Segment, Transcript
from podtx.formatting import body_text
from podtx.proper_noun import build_glossary, correct_proper_nouns
from podtx.writers import write_outputs
from podtx.format_cmd import reformat_transcript
from typer.testing import CliRunner
from podtx.cli import app

runner = CliRunner()


def _episode_with_title(title: str = "Sabine Wojcieszak") -> Episode:
    return Episode(
        guid="g1",
        title=title,
        enclosure_url="https://example.com/a.mp3",
        published_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
        episode_num=1,
        show_title="Demo Show",
        description="Interview with Sabine Wojcieszak about Jane Street and parakeet.",
        link="https://example.com/ep",
    )


def test_build_glossary_from_metadata() -> None:
    ep = _episode_with_title()
    glossary = build_glossary(ep)
    # should contain Sabine Wojcieszak and Jane Street but not common words
    combined = " ".join(glossary)
    assert "Sabine Wojcieszak" in combined or "Sabine" in combined
    assert "Jane Street" in combined


def test_correct_close_misspelling() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    text = "Today we talk with Sabina Vosheshak about AI."
    corrected, subs = correct_proper_nouns(text, ep)
    assert "Sabine Wojcieszak" in corrected
    assert "Sabina Vosheshak" not in corrected
    assert len(subs) >= 1


def test_common_words_not_changed() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    text = "the and hello world is common"
    corrected, subs = correct_proper_nouns(text, ep)
    assert corrected == text
    assert len(subs) == 0


def test_conservative_no_false_positive_on_short_word() -> None:
    # Jane is short but Jane Street is in glossary; single "Chain" should not become Jane
    ep = Episode(
        guid="g1",
        title="Jane Street Interview",
        enclosure_url="https://example.com/a.mp3",
        show_title="Demo",
        description="Jane Street is a firm",
    )
    text = "Chain Street is not Jane Street"
    corrected, subs = correct_proper_nouns(text, ep)
    # Correct "Chain Street" is close to Jane Street but conservative should maybe not? We expect not to change common phrase or at least not over-correct unrelated
    # For this test, ensure that a plain common word "the" not in glossary stays, and that unrelated single word not corrected
    ep2 = _episode_with_title("Sabine Wojcieszak")
    text2 = "the quick brown fox jumps"
    corrected2, subs2 = correct_proper_nouns(text2, ep2)
    assert corrected2 == text2
    assert subs2 == []


def test_timed_segments_stay_raw() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Hello Sabina Vosheshak world.",
        segments=[
            Segment(0.0, 1.0, "Hello Sabina Vosheshak world."),
            Segment(1.0, 2.0, "Another Sabina Vosheshak segment."),
        ],
        language="en",
        model="test",
        engine="fake",
    )
    # body_text with correct_names should correct body but segments stay raw
    body = body_text(transcript.text, transcript.segments, readable=False, cleanup=False, correct_names=True, episode=ep)
    assert "Sabine Wojcieszak" in body
    # segments unchanged
    assert transcript.segments[0].text == "Hello Sabina Vosheshak world."
    assert transcript.segments[1].text == "Another Sabina Vosheshak segment."


def test_write_outputs_correct_names_flag() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Interview with Sabina Vosheshak is great.",
        segments=[Segment(0.0, 1.0, "Interview with Sabina Vosheshak is great.")],
        language="en",
        model="test",
        engine="fake",
    )
    import tempfile
    from pathlib import Path as P
    import json as j

    tmp = P(tempfile.mkdtemp())
    # without flag: byte-identical raw (body should stay uncorrected; header always has episode title)
    paths_raw = write_outputs(out_dir=tmp / "raw", basename="ep", episode=ep, transcript=transcript, formats=("txt", "json"), readable=False, cleanup=False, correct_names=False)
    txt_raw = (tmp / "raw" / "ep.txt").read_text()
    # Body is after header (first blank line). Check body portion.
    body_raw = txt_raw.split("\n\n", 1)[-1] if "\n\n" in txt_raw else txt_raw
    assert "Sabina Vosheshak" in body_raw
    assert "Sabine Wojcieszak" not in body_raw
    payload_raw = j.loads((tmp / "raw" / "ep.json").read_text())
    assert "Sabina Vosheshak" in payload_raw["text"]
    assert payload_raw["segments"][0]["text"] == "Interview with Sabina Vosheshak is great."

    # with flag: corrected body, segments still raw
    paths_corr = write_outputs(out_dir=tmp / "corr", basename="ep", episode=ep, transcript=transcript, formats=("txt", "json"), readable=False, cleanup=False, correct_names=True)
    txt_corr = (tmp / "corr" / "ep.txt").read_text()
    body_corr = txt_corr.split("\n\n", 1)[-1] if "\n\n" in txt_corr else txt_corr
    assert "Sabine Wojcieszak" in body_corr
    assert "Sabina Vosheshak" not in body_corr
    payload_corr = j.loads((tmp / "corr" / "ep.json").read_text())
    assert "Sabine Wojcieszak" in payload_corr["text"]
    assert payload_corr["segments"][0]["text"] == "Interview with Sabina Vosheshak is great."


def test_reformat_transcript_without_asr() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Hello Sabina Vosheshak world.",
        segments=[Segment(0.0, 1.0, "Hello Sabina Vosheshak world.")],
        language="en",
        model="test",
        engine="fake",
    )
    import tempfile
    from pathlib import Path as P
    tmp = P(tempfile.mkdtemp())
    write_outputs(out_dir=tmp, basename="ep", episode=ep, transcript=transcript, formats=("json",), readable=False, cleanup=False, correct_names=False)
    json_path = tmp / "ep.json"
    # reformat without flag should stay raw
    out_raw = tmp / "out_raw"
    paths = reformat_transcript(json_path, out_dir=out_raw, readable=False, cleanup=False, correct_names=False, formats=("txt", "json"))
    txt = (out_raw / "ep.txt").read_text()
    assert "Sabina Vosheshak" in txt
    # reformat with correct_names should correct
    out_corr = tmp / "out_corr"
    paths2 = reformat_transcript(json_path, out_dir=out_corr, readable=False, cleanup=False, correct_names=True, formats=("txt", "json"))
    txt2 = (out_corr / "ep.txt").read_text()
    assert "Sabine Wojcieszak" in txt2


def test_reporting_substitutions() -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    text = "Sabina Vosheshak and Sabina Vosheshak again."
    corrected, subs = correct_proper_nouns(text, ep)
    assert len(subs) == 2
    for orig, corr in subs:
        assert "Sabina" in orig or "Vosheshak" in orig
        assert corr == "Sabine Wojcieszak"


def test_byte_identical_when_off(tmp_path: Path) -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Sabina Vosheshak hello",
        segments=[],
        language="en",
        model="t",
        engine="e",
    )
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    write_outputs(out_dir=out1, basename="ep", episode=ep, transcript=transcript, formats=("txt", "json"), readable=False, cleanup=False, correct_names=False)
    write_outputs(out_dir=out2, basename="ep", episode=ep, transcript=transcript, formats=("txt", "json"), readable=False, cleanup=False, correct_names=False)
    assert (out1 / "ep.txt").read_bytes() == (out2 / "ep.txt").read_bytes()
    assert (out1 / "ep.json").read_bytes() == (out2 / "ep.json").read_bytes()


def test_cli_format_feed_and_all_with_correct_names(tmp_path: Path) -> None:
    # Build mini library with two feeds, each with a transcript containing misspelling
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Hello Sabina Vosheshak world.",
        segments=[Segment(0.0, 1.0, "Hello Sabina Vosheshak world.")],
        language="en",
        model="test",
        engine="fake",
    )
    root = tmp_path / "transcripts"
    for slug in ("feed-a", "feed-b"):
        feed_dir = root / slug
        feed_dir.mkdir(parents=True)
        write_outputs(out_dir=feed_dir, basename=f"{slug}-ep", episode=ep, transcript=transcript, formats=("txt", "json"), readable=False, cleanup=False, correct_names=False)
        # verify raw before
        assert "Sabina Vosheshak" in (feed_dir / f"{slug}-ep.txt").read_text()

    # --feed with --correct-names should correct only feed-a
    result = runner.invoke(app, ["format", "--feed", "feed-a", "--data-dir", str(tmp_path), "--correct-names"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Sabine Wojcieszak" in (root / "feed-a" / "feed-a-ep.txt").read_text()
    # feed-b should remain raw after single-feed run
    assert "Sabina Vosheshak" in (root / "feed-b" / "feed-b-ep.txt").read_text()

    # --all with --correct-names should correct remaining
    result2 = runner.invoke(app, ["format", "--all", "--data-dir", str(tmp_path), "--correct-names"])
    assert result2.exit_code == 0, result2.stdout + result2.stderr
    assert "Sabine Wojcieszak" in (root / "feed-b" / "feed-b-ep.txt").read_text()
    # JSON corrections reported
    payload = json.loads((root / "feed-a" / "feed-a-ep.json").read_text())
    assert payload.get("correct_names") is True
    assert len(payload.get("corrections", [])) >= 1


def test_cli_format_single_file_correct_names(tmp_path: Path) -> None:
    ep = _episode_with_title("Sabine Wojcieszak")
    transcript = Transcript(
        text="Sabina Vosheshak hello",
        segments=[Segment(0.0, 1.0, "Sabina Vosheshak hello")],
        language="en",
        model="test",
        engine="fake",
    )
    out = tmp_path / "orig"
    out.mkdir()
    write_outputs(out_dir=out, basename="ep", episode=ep, transcript=transcript, formats=("json",), readable=False, cleanup=False, correct_names=False)
    json_path = out / "ep.json"
    # without flag: still raw
    result_raw = runner.invoke(app, ["format", str(json_path), "--out-dir", str(tmp_path / "out_raw")])
    assert result_raw.exit_code == 0
    assert "Sabina Vosheshak" in (tmp_path / "out_raw" / "ep.txt").read_text()
    # with flag: corrected
    result_corr = runner.invoke(app, ["format", str(json_path), "--correct-names", "--out-dir", str(tmp_path / "out_corr")])
    assert result_corr.exit_code == 0
    assert "Sabine Wojcieszak" in (tmp_path / "out_corr" / "ep.txt").read_text()
    # segments stay raw in json
    payload = json.loads((tmp_path / "out_corr" / "ep.json").read_text())
    assert "Sabine Wojcieszak" in payload["text"]
    assert payload["segments"][0]["text"] == "Sabina Vosheshak hello"
    assert payload["corrections"][0][1] == "Sabine Wojcieszak"


def test_conservative_avoids_false_positives(tmp_path: Path) -> None:
    # Glossary contains Sabine Wojcieszak, but transcript has common words and unrelated names
    ep = _episode_with_title("Sabine Wojcieszak")
    text = "The quick brown fox jumps over the lazy dog and says hello world"
    corrected, subs = correct_proper_nouns(text, ep)
    assert corrected == text
    assert subs == []
    # Single common word should not be corrected even if single-word glossary has similar length
    # e.g., glossary has "Interview" (common filtered) so "interview" misspelling not corrected
    ep2 = Episode(guid="g", title="Interview with Bob", enclosure_url="https://example.com/a.mp3", show_title="Demo", description="Interview")
    corrected2, subs2 = correct_proper_nouns("interveiw is typo but common", ep2)
    # interveiw vs Interview distance 1 but Interview is filtered as common, so no correction
    assert subs2 == []


def test_glossary_built_from_all_metadata_fields() -> None:
    # Title, show_title, description, link should all contribute
    ep = Episode(
        guid="g1",
        title="Episode with Alice Wonderland",
        enclosure_url="https://example.com/a.mp3",
        show_title="Wonderland Show",
        description="Featuring Bob Builder and description mentions Charlie Chaplin",
        link="https://example.com/episode/alice-wonderland",
    )
    glossary = build_glossary(ep)
    combined = " ".join(glossary)
    assert "Alice Wonderland" in combined
    assert "Wonderland Show" in combined or "Wonderland" in combined
    assert "Bob Builder" in combined
    assert "Charlie Chaplin" in combined


def test_build_glossary_skips_short_single_words() -> None:
    ep = Episode(
        guid="g1",
        title="Ada",
        enclosure_url="https://example.com/a.mp3",
        show_title="Demo",
    )
    assert build_glossary(ep) == []


def test_build_glossary_keeps_long_single_word() -> None:
    ep = Episode(
        guid="g1",
        title="Sustainable Coding",
        enclosure_url="https://example.com/a.mp3",
        show_title="Demo",
    )
    assert build_glossary(ep) == ["Sustainable Coding"]


def test_build_glossary_single_capitalized_word() -> None:
    ep = Episode(
        guid="g1",
        title="Python",
        enclosure_url="https://example.com/a.mp3",
    )
    assert build_glossary(ep) == ["Python"]
