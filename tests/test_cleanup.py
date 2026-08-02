from __future__ import annotations

from podtx.cleanup import clean_text


def test_removes_uh_um_fillers() -> None:
    raw = "So uh yeah, what did I um send you?"
    assert clean_text(raw) == "So yeah, what did I send you?"


def test_collapses_immediate_word_doubles() -> None:
    raw = "I'll I'll I'm building the things and the the array."
    assert clean_text(raw) == "I'll I'm building the things and the array."


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
