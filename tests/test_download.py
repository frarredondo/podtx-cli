from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podtx.download import convert_to_wav


# Bytes that are invalid UTF-8 (0xc3 without a valid continuation) — mirrors
# real ffmpeg stderr when an MP3 has mangled Latin-1/ID3 metadata.
_BAD_STDERR = b"Metadata:\n  comment: can\xc3\n  title: I\xc3\xa2ve always\n"


def test_convert_to_wav_tolerates_non_utf8_ffmpeg_stderr(tmp_path: Path) -> None:
    src = tmp_path / "ep.mp3"
    src.write_bytes(b"fake")
    dest = tmp_path / "ep.wav"

    completed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=b"",
        stderr=_BAD_STDERR,
    )
    with patch("podtx.download.require_ffmpeg", return_value="ffmpeg"), patch(
        "podtx.download.subprocess.run", return_value=completed
    ) as run:
        out = convert_to_wav(src, dest)

    assert out == dest
    kwargs = run.call_args.kwargs
    assert kwargs.get("text") is not True
    assert kwargs.get("capture_output") is True


def test_convert_to_wav_error_message_handles_non_utf8_stderr(tmp_path: Path) -> None:
    src = tmp_path / "ep.mp3"
    src.write_bytes(b"fake")

    completed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=1,
        stdout=b"",
        stderr=_BAD_STDERR + b"Error while decoding\n",
    )
    with patch("podtx.download.require_ffmpeg", return_value="ffmpeg"), patch(
        "podtx.download.subprocess.run", return_value=completed
    ):
        with pytest.raises(RuntimeError, match="ffmpeg failed") as exc_info:
            convert_to_wav(src)

    # Must surface a str error, not crash with UnicodeDecodeError
    assert "Error while decoding" in str(exc_info.value)
