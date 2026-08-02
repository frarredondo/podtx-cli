from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

USER_AGENT = "podtx/0.1 (+https://github.com/frarredondo/podtx-cli)"

ProgressHook = Callable[[int, int], None]


class FFmpegNotFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "ffmpeg is required but was not found on PATH. Install it with: brew install ffmpeg"
        )


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError()
    return path


def _filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    if not name or name in {".", "/"}:
        return "episode.audio"
    return name


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    on_progress: ProgressHook | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)
    return dest


def download_episode_audio(
    url: str,
    audio_dir: Path,
    guid_hash: str,
    *,
    on_progress: ProgressHook | None = None,
) -> Path:
    """Download enclosure into audio_dir using a stable temp name."""
    suffix = Path(_filename_from_url(url)).suffix or ".mp3"
    dest = audio_dir / f"{guid_hash}{suffix}"
    return download_file(url, dest, on_progress=on_progress)


def convert_to_wav(src: Path, dest: Path | None = None) -> Path:
    """Convert/normalize audio to 16kHz mono WAV via ffmpeg."""
    ffmpeg = require_ffmpeg()
    out = dest or src.with_suffix(".wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    # Capture as bytes: some MP3s ship mangled ID3 tags, and ffmpeg echoes
    # them on stderr. text=True would raise UnicodeDecodeError before we can
    # even inspect returncode (seen on CoRecursive ep. "When AI Codes...").
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed converting {src}: {err}")
    return out
