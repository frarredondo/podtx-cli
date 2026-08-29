from __future__ import annotations

import hashlib
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from podtx.config import DEFAULT_LIMIT, Settings, ensure_data_dirs
from podtx.db import Database
from podtx.download import convert_to_wav, download_episode_audio, require_ffmpeg
from podtx.engines import get_engine
from podtx.models import Episode, Transcript
from podtx.naming import unique_basename
from podtx.writers import write_outputs

console = Console(stderr=True)


def _guid_hash(guid: str) -> str:
    return hashlib.sha1(guid.encode()).hexdigest()[:16]


def _log(settings: Settings, message: str) -> None:
    if not settings.quiet:
        console.print(message)


def download_only(episode: Episode, audio_dir: Path, *, quiet: bool = False) -> Path:
    require_ffmpeg()
    h = _guid_hash(episode.guid)

    if quiet:
        return download_episode_audio(episode.enclosure_url, audio_dir, h)

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(f"Download {episode.title[:40]}", total=None)

        def on_progress(downloaded: int, total: int) -> None:
            if total > 0:
                progress.update(task_id, total=total, completed=downloaded)
            else:
                progress.update(task_id, completed=downloaded)

        return download_episode_audio(
            episode.enclosure_url, audio_dir, h, on_progress=on_progress
        )


def transcribe_local_file(
    path: Path,
    *,
    settings: Settings,
    episode: Episode | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    ensure_data_dirs(settings)
    require_ffmpeg()
    engine = get_engine(settings.engine)
    model = settings.resolved_model()

    ep = episode or Episode(
        guid=str(path.resolve()),
        title=path.stem,
        enclosure_url=str(path.resolve()),
        show_title=None,
    )

    dest_dir = out_dir or Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)

    work_dir = settings.audio_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    h = _guid_hash(ep.guid)
    wav = work_dir / f"{h}.wav"
    convert_to_wav(path, wav)

    _log(settings, f"[cyan]Transcribing[/cyan] {ep.title} with {engine.name}/{model}")
    transcript = engine.transcribe(
        wav,
        model=model,
        language=settings.language,
        local_attention=settings.local_attention,
        local_attention_context_size=settings.local_attention_context_size,
    )
    basename = unique_basename(ep, existing=set())
    paths = write_outputs(
        out_dir=dest_dir,
        basename=basename,
        episode=ep,
        transcript=transcript,
        formats=settings.formats,
        readable=settings.readable,
        cleanup=settings.cleanup,
    )
    if not settings.keep_audio and wav.exists():
        wav.unlink(missing_ok=True)
    return paths


def process_episodes(
    episodes: Sequence[Episode],
    *,
    settings: Settings,
    out_dir: Path,
    db: Database | None = None,
    feed_id: int | None = None,
) -> list[list[Path]]:
    """Transcribe episodes sequentially, prefetching the next download."""
    ensure_data_dirs(settings)
    require_ffmpeg()
    engine = get_engine(settings.engine)
    model = settings.resolved_model()
    audio_dir = settings.audio_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[list[Path]] = []
    existing_bases = {p.stem for p in out_dir.glob("*") if p.is_file()}

    with ThreadPoolExecutor(max_workers=1) as pool:
        def enqueue(ep: Episode) -> Future[Path]:
            return pool.submit(download_only, ep, audio_dir, quiet=settings.quiet)

        prefetch: Future[Path] | None = None
        if episodes:
            prefetch = enqueue(episodes[0])

        for idx, episode in enumerate(episodes):
            if db is not None and feed_id is not None:
                db.upsert_episode(
                    feed_id=feed_id,
                    guid=episode.guid,
                    title=episode.title,
                    published_at=episode.published_at,
                    episode_num=episode.episode_num,
                    enclosure_url=episode.enclosure_url,
                )
                if db.is_done(feed_id, episode.guid):
                    _log(settings, f"[dim]Skipping already done:[/dim] {episode.title}")
                    if prefetch is not None:
                        try:
                            prefetch.result()
                        except Exception:
                            pass
                    prefetch = enqueue(episodes[idx + 1]) if idx + 1 < len(episodes) else None
                    continue

            _log(settings, f"[cyan]Downloading[/cyan] {episode.title}")
            assert prefetch is not None
            try:
                original = prefetch.result()
            except Exception as exc:
                _log(settings, f"[red]Download failed:[/red] {episode.title}: {exc}")
                if db is not None and feed_id is not None:
                    db.mark_error(feed_id=feed_id, guid=episode.guid, message=str(exc))
                prefetch = enqueue(episodes[idx + 1]) if idx + 1 < len(episodes) else None
                continue

            prefetch = enqueue(episodes[idx + 1]) if idx + 1 < len(episodes) else None

            h = _guid_hash(episode.guid)
            wav = audio_dir / f"{h}.wav"
            cleanup = [wav]
            if not settings.keep_audio:
                cleanup.append(original)

            try:
                convert_to_wav(original, wav)
                _log(
                    settings,
                    f"[cyan]Transcribing[/cyan] {episode.title} ({engine.name}/{model})",
                )
                transcript: Transcript = engine.transcribe(
                    wav,
                    model=model,
                    language=settings.language,
                    local_attention=settings.local_attention,
                    local_attention_context_size=settings.local_attention_context_size,
                )
                basename = unique_basename(episode, existing_bases)
                existing_bases.add(basename)
                paths = write_outputs(
                    out_dir=out_dir,
                    basename=basename,
                    episode=episode,
                    transcript=transcript,
                    formats=settings.formats,
                    readable=settings.readable,
                    cleanup=settings.cleanup,
                )
                results.append(paths)
                for p in paths:
                    _log(settings, f"[green]Wrote[/green] {p}")
                if db is not None and feed_id is not None:
                    db.mark_done(
                        feed_id=feed_id,
                        guid=episode.guid,
                        engine=transcript.engine,
                        model=transcript.model,
                        output_paths=paths,
                    )
                    # Incremental FTS indexing (offline search)
                    try:
                        feed = db.get_feed_by_id(feed_id)
                        if feed is not None:
                            txt_path = next((str(p) for p in paths if p.suffix == ".txt"), str(paths[0]) if paths else "")
                            json_path = next((str(p) for p in paths if p.suffix == ".json"), str(paths[0]) if paths else "")
                            db.upsert_search_entry(
                                feed_slug=feed.slug,
                                guid=episode.guid,
                                title=episode.title,
                                published_at=episode.published_at.isoformat() if episode.published_at else None,
                                text=transcript.text,
                                txt_path=txt_path,
                                json_path=json_path,
                            )
                    except Exception:  # pragma: no cover - best-effort FTS indexing
                        pass
            except Exception as exc:  # pragma: no cover - download/transcribe failure
                _log(settings, f"[red]Failed:[/red] {episode.title}: {exc}")
                if db is not None and feed_id is not None:
                    db.mark_error(feed_id=feed_id, guid=episode.guid, message=str(exc))
            finally:
                if not settings.keep_audio:
                    for p in cleanup:
                        p.unlink(missing_ok=True)

    return results


def select_episodes_for_sync(
    episodes: Sequence[Episode],
    *,
    done_guids: set[str],
    limit: int | None,
    process_all: bool,
) -> list[Episode]:
    pending = [e for e in episodes if e.guid not in done_guids]
    if process_all:
        return list(pending)
    n = DEFAULT_LIMIT if limit is None else limit
    return list(pending[:n])
