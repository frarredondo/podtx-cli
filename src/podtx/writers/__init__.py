from __future__ import annotations

from pathlib import Path

from podtx.models import Episode, Transcript
from podtx.writers import json as json_writer
from podtx.writers import md as md_writer
from podtx.writers import srt as srt_writer
from podtx.writers import txt as txt_writer
from podtx.writers import vtt as vtt_writer

_WRITERS = {
    "txt": ("txt", txt_writer.write_txt),
    "json": ("json", json_writer.write_json),
    "srt": ("srt", srt_writer.write_srt),
    "vtt": ("vtt", vtt_writer.write_vtt),
    "md": ("md", md_writer.write_md),
}


def write_outputs(
    *,
    out_dir: Path,
    basename: str,
    episode: Episode,
    transcript: Transcript,
    formats: tuple[str, ...] | list[str],
    readable: bool = False,
    cleanup: bool = False,
    correct_names: bool = False,
    diarize: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        key = fmt.lower().strip()
        if key not in _WRITERS:
            raise ValueError(f"Unsupported format {fmt!r}. Choose from: {', '.join(_WRITERS)}")
        ext, writer = _WRITERS[key]
        path = out_dir / f"{basename}.{ext}"
        if key in {"txt", "json", "md"}:
            paths.append(
                writer(path, episode, transcript, readable=readable, cleanup=cleanup, correct_names=correct_names, diarize=diarize)
            )
        else:
            paths.append(writer(path, episode, transcript))
    return paths
