from __future__ import annotations

from podtx.cleanup import clean_text


def test_removes_uh_um_fillers() -> None:
    raw = "So uh yeah, what did I um send you?"
    assert clean_text(raw) == "So yeah, what did I send you?"


def test_collapses_immediate_word_doubles() -> None:
    raw = "I'll I'll I'm building the things and the the array."
    assert clean_text(raw) == "I'll I'm building the things and the array."


def test_collapses_two_word_phrase_doubles() -> None:
    assert clean_text("I think I think this is right.") == "I think this is right."
    assert clean_text("when you when you run the tests") == "when you run the tests"


def test_collapses_three_and_four_word_phrase_doubles() -> None:
    assert (
        clean_text("He covers a he covers a wide breadth of topics.")
        == "He covers a wide breadth of topics."
    )
    assert (
        clean_text("I would like to I would like to emphasize that.")
        == "I would like to emphasize that."
    )


def test_collapses_repeated_phrase_triples() -> None:
    # Same phrase stuttered three times should collapse to one.
    assert clean_text("I think I think I think so.") == "I think so."


def test_does_not_collapse_non_adjacent_similar_phrases() -> None:
    # Only immediate repeats; intervening words must block collapse.
    raw = "I think that I think this is fine."
    assert clean_text(raw) == raw


def test_preserves_paragraph_breaks() -> None:
    raw = "First paragraph uh here.\n\nSecond um paragraph."
    assert clean_text(raw) == "First paragraph here.\n\nSecond paragraph."


def test_cleanup_is_idempotent() -> None:
    raw = "So uh the the quick brown fox."
    once = clean_text(raw)
    assert clean_text(once) == once


def test_empty_and_whitespace() -> None:
    assert clean_text("") == ""
    assert clean_text("   ") == ""
