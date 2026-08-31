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


def test_require_ffmpeg_missing(monkeypatch) -> None:
    from podtx.download import FFmpegNotFoundError, require_ffmpeg

    monkeypatch.setattr("podtx.download.shutil.which", lambda _: None)
    try:
        require_ffmpeg()
        assert False
    except FFmpegNotFoundError:
        pass


def test_require_ffmpeg_found(monkeypatch) -> None:
    from podtx.download import require_ffmpeg

    monkeypatch.setattr("podtx.download.shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert require_ffmpeg() == "/usr/bin/ffmpeg"


def test_filename_from_url_variants() -> None:
    from podtx.download import _filename_from_url

    assert _filename_from_url("https://x.com/audio/ep3.mp3") == "ep3.mp3"
    assert _filename_from_url("https://x.com/") == "episode.audio"
    assert _filename_from_url("https://x.com/audio%20file.mp3") == "audio file.mp3"


def test_download_file_streams_with_progress(tmp_path: Path) -> None:
    import httpx
    from podtx.download import download_file

    body = b"chunk-one-chunk-two"
    content_length = str(len(body))

    def handler(request):
        return httpx.Response(200, content=body, headers={"content-length": content_length})

    dest = tmp_path / "sub" / "out.mp3"
    seen = []
    with httpx.MockTransport(handler) as transport:
        client = httpx.Client(transport=transport)
        with patch("podtx.download.httpx.stream", client.stream):
            download_file("https://x.com/a.mp3", dest, on_progress=lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == body
    assert seen and seen[-1][1] == len(body)


def test_download_episode_audio_default_extension(tmp_path: Path) -> None:
    from podtx.download import download_episode_audio

    with patch(
        "podtx.download.download_file",
        lambda url, dest, on_progress=None: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"x"), dest)[2],
    ):
        out = download_episode_audio("https://x.com/stream", tmp_path, "abc123")
        assert out.name == "abc123.mp3"


def test_convert_to_wav_with_trim(tmp_path: Path, monkeypatch) -> None:
    from podtx.download import convert_to_wav

    src = tmp_path / "ep.mp3"
    src.write_bytes(b"fake")
    dest = tmp_path / "ep.wav"

    completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout=b"", stderr=b"")
    with patch("podtx.download.require_ffmpeg", return_value="ffmpeg"), patch(
        "podtx.download.subprocess.run", return_value=completed
    ) as run:
        out = convert_to_wav(src, dest, trim_start=5.0)
    assert out == dest
    assert "-ss" in run.call_args.args[0]


def test_download_file_without_progress_no_content_length(tmp_path: Path) -> None:
    import httpx
    from podtx.download import download_file

    body = b"data"

    def handler(request):
        return httpx.Response(200, content=body)

    dest = tmp_path / "plain.mp3"
    with httpx.MockTransport(handler) as transport:
        client = httpx.Client(transport=transport)
        with patch("podtx.download.httpx.stream", client.stream):
            download_file("https://x.com/a.mp3", dest)
    assert dest.read_bytes() == body
