from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from podtx import __version__
from podtx.config import Settings, ensure_data_dirs, load_settings
from podtx.db import Database
from podtx.download import FFmpegNotFoundError, require_ffmpeg
from podtx.engines import available_engines
from podtx.models import Episode
from podtx.pipeline import process_episodes, select_episodes_for_sync, transcribe_local_file
from podtx.format_cmd import TranscriptJsonError, reformat_transcript
from podtx.rss import FeedParseError, parse_feed, suggest_unique_slug

app = typer.Typer(
    name="podtx",
    help="Pull and transcribe podcast episodes from RSS using local AI models.",
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)
console = Console()
err_console = Console(stderr=True)


def _settings_from_opts(
    *,
    engine: Optional[str],
    model: Optional[str],
    limit: Optional[int],
    formats: Optional[list[str]],
    keep_audio: Optional[bool],
    data_dir: Optional[Path],
    quiet: bool,
    local_attention: Optional[bool] = None,
    local_attention_context_size: Optional[int] = None,
    readable: Optional[bool] = None,
    cleanup: Optional[bool] = None,
) -> Settings:
    return load_settings(
        engine=engine,
        model=model,
        limit=limit,
        formats=formats,
        keep_audio=keep_audio,
        data_dir=data_dir,
        quiet=quiet,
        local_attention=local_attention,
        local_attention_context_size=local_attention_context_size,
        readable=readable,
        cleanup=cleanup,
    )


def _open_db(settings: Settings) -> Database:
    ensure_data_dirs(settings)
    return Database(settings.state_db_path())


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _looks_like_audio_url(value: str) -> bool:
    if not _looks_like_url(value):
        return False
    path = urlparse(value).path.lower()
    return any(path.endswith(ext) for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".mp4"))


def _merge_formats(base: tuple[str, ...], extra: Optional[list[str]]) -> tuple[str, ...]:
    if not extra:
        return base
    # If user passes formats, use those (still allow combining with defaults when only srt/vtt added)
    requested = [f.lower() for f in extra]
    if any(f in {"txt", "json"} for f in requested):
        return tuple(dict.fromkeys(requested))
    # Adding subtitle formats keeps default txt+json
    return tuple(dict.fromkeys([*base, *requested]))


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("add")
def add_feed(
    rss_url: str = typer.Argument(..., help="RSS feed URL"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Override data directory"),
) -> None:
    """Register an RSS feed."""
    settings = load_settings(data_dir=data_dir)
    try:
        title, slug_base, _episodes = parse_feed(rss_url)
    except FeedParseError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    with _open_db(settings) as db:
        existing = {f.slug for f in db.list_feeds()}
        if db.get_feed(rss_url):
            err_console.print(f"[yellow]Feed already registered:[/yellow] {rss_url}")
            raise typer.Exit(1)
        slug = suggest_unique_slug(slug_base, existing)
        feed = db.add_feed(rss_url, slug, title)
    console.print(f"[green]Added[/green] {feed.title} [dim]({feed.slug})[/dim]")


@app.command("remove")
def remove_feed(
    feed: str = typer.Argument(..., help="Feed slug or URL"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """Unregister a feed and its episode records."""
    settings = load_settings(data_dir=data_dir)
    with _open_db(settings) as db:
        ok = db.remove_feed(feed)
    if not ok:
        err_console.print(f"[red]Feed not found:[/red] {feed}")
        raise typer.Exit(1)
    console.print(f"[green]Removed[/green] {feed}")


@app.command("feeds")
def list_feeds(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """List registered feeds."""
    settings = load_settings(data_dir=data_dir)
    with _open_db(settings) as db:
        feeds = db.list_feeds()
    if not feeds:
        console.print("[dim]No feeds registered. Use `podtx add <rss-url>`.[/dim]")
        return
    table = Table(title="Registered feeds")
    table.add_column("Slug")
    table.add_column("Title")
    table.add_column("URL")
    for f in feeds:
        table.add_row(f.slug, f.title, f.url)
    console.print(table)


@app.command("show")
def show_feed(
    feed: str = typer.Argument(..., help="Feed slug or URL"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """List known episodes and transcription status for a feed."""
    settings = load_settings(data_dir=data_dir)
    with _open_db(settings) as db:
        f = db.get_feed(feed)
        if not f:
            err_console.print(f"[red]Feed not found:[/red] {feed}")
            raise typer.Exit(1)
        rows = db.list_episodes(f.id)

    table = Table(title=f"{f.title} ({f.slug})")
    table.add_column("Status")
    table.add_column("Date")
    table.add_column("Ep")
    table.add_column("Title")
    for row in rows:
        table.add_row(
            row["status"],
            (row["published_at"] or "")[:10],
            str(row["episode_num"] if row["episode_num"] is not None else "000"),
            row["title"],
        )
    if not rows:
        console.print("[dim]No episodes recorded yet. Run `podtx sync`.[/dim]")
    else:
        console.print(table)


@app.command("sync")
def sync_feeds(
    feed: Optional[str] = typer.Argument(None, help="Optional feed slug/URL (default: all)"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="parakeet or whisper"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="HF model id"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max new episodes per feed"),
    all_episodes: bool = typer.Option(False, "--all", help="Process all pending episodes"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Retain downloaded audio"),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Output format (repeatable): txt, json, srt, vtt"
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Override output directory"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    local_attention: Optional[bool] = typer.Option(
        None,
        "--local-attention/--full-attention",
        help="Parakeet attention mode (default: local; use full only for short audio)",
    ),
    local_attention_context_size: Optional[int] = typer.Option(
        None,
        "--local-attention-context-size",
        help="Local attention context size (default: 256)",
    ),
    readable: bool = typer.Option(
        False,
        "--readable",
        help="Human-friendly text: paragraph breaks on silence gaps (JSON timestamps always rounded)",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Strip fillers (uh/um) and collapse consecutive word doubles in text outputs",
    ),
) -> None:
    """Download and transcribe new episodes for registered feeds."""
    settings = _settings_from_opts(
        engine=engine,
        model=model,
        limit=limit,
        formats=None,
        keep_audio=keep_audio or None,
        data_dir=data_dir,
        quiet=quiet,
        local_attention=local_attention,
        local_attention_context_size=local_attention_context_size,
        readable=True if readable else None,
        cleanup=True if cleanup else None,
    )
    settings = replace(
        settings,
        formats=_merge_formats(settings.formats, format),
        keep_audio=keep_audio or settings.keep_audio,
        readable=readable or settings.readable,
        cleanup=cleanup or settings.cleanup,
    )

    try:
        require_ffmpeg()
    except FFmpegNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if settings.engine not in available_engines():
        err_console.print(f"[red]Unknown engine:[/red] {settings.engine}")
        raise typer.Exit(1)

    with _open_db(settings) as db:
        if feed:
            feeds = [db.get_feed(feed)]
            if feeds[0] is None:
                err_console.print(f"[red]Feed not found:[/red] {feed}")
                raise typer.Exit(1)
        else:
            feeds = db.list_feeds()
            if not feeds:
                err_console.print("[red]No feeds registered. Use `podtx add <rss-url>`.[/red]")
                raise typer.Exit(1)

        for f in feeds:
            assert f is not None
            try:
                _title, _slug, episodes = parse_feed(f.url)
            except FeedParseError as exc:
                err_console.print(f"[red]Failed to parse {f.slug}:[/red] {exc}")
                continue

            # Refresh show title if changed
            done = db.done_guids(f.id)
            selected = select_episodes_for_sync(
                episodes,
                done_guids=done,
                limit=settings.limit,
                process_all=all_episodes,
            )
            if not selected:
                if not quiet:
                    console.print(f"[dim]{f.slug}: nothing new[/dim]")
                continue

            dest = out_dir or settings.transcripts_dir(f.slug)
            if not quiet:
                console.print(
                    f"[bold]{f.title}[/bold]: transcribing {len(selected)} episode(s) "
                    f"with {settings.engine}/{settings.resolved_model()}"
                )
            # Attach show title
            selected = [
                Episode(
                    guid=e.guid,
                    title=e.title,
                    enclosure_url=e.enclosure_url,
                    published_at=e.published_at,
                    episode_num=e.episode_num,
                    description=e.description,
                    link=e.link,
                    show_title=f.title,
                )
                for e in selected
            ]
            process_episodes(
                selected,
                settings=settings,
                out_dir=dest,
                db=db,
                feed_id=f.id,
            )


@app.command("transcribe")
def transcribe_cmd(
    target: str = typer.Argument(..., help="RSS URL, audio URL, or local audio file"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="For RSS: max episodes"),
    all_episodes: bool = typer.Option(False, "--all", help="For RSS: all episodes"),
    keep_audio: bool = typer.Option(False, "--keep-audio"),
    format: Optional[list[str]] = typer.Option(None, "--format", "-f"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Default: current directory"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    local_attention: Optional[bool] = typer.Option(
        None,
        "--local-attention/--full-attention",
        help="Parakeet attention mode (default: local; use full only for short audio)",
    ),
    local_attention_context_size: Optional[int] = typer.Option(
        None,
        "--local-attention-context-size",
        help="Local attention context size (default: 256)",
    ),
    readable: bool = typer.Option(
        False,
        "--readable",
        help="Human-friendly text: paragraph breaks on silence gaps (JSON timestamps always rounded)",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Strip fillers (uh/um) and collapse consecutive word doubles in text outputs",
    ),
) -> None:
    """One-shot transcription of a feed, audio URL, or local file."""
    settings = _settings_from_opts(
        engine=engine,
        model=model,
        limit=limit,
        formats=None,
        keep_audio=keep_audio or None,
        data_dir=data_dir,
        quiet=quiet,
        local_attention=local_attention,
        local_attention_context_size=local_attention_context_size,
        readable=True if readable else None,
        cleanup=True if cleanup else None,
    )
    settings = replace(
        settings,
        formats=_merge_formats(settings.formats, format),
        keep_audio=keep_audio or settings.keep_audio,
        readable=readable or settings.readable,
        cleanup=cleanup or settings.cleanup,
    )

    try:
        require_ffmpeg()
    except FFmpegNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    dest = out_dir or Path.cwd()

    # Local file
    local = Path(target).expanduser()
    if local.is_file():
        paths = transcribe_local_file(local, settings=settings, out_dir=dest)
        for p in paths:
            console.print(f"[green]Wrote[/green] {p}")
        return

    if not _looks_like_url(target):
        err_console.print(f"[red]Not a file or URL:[/red] {target}")
        raise typer.Exit(1)

    # Direct audio URL
    if _looks_like_audio_url(target):
        episode = Episode(
            guid=target,
            title=Path(urlparse(target).path).stem or "episode",
            enclosure_url=target,
            show_title=None,
        )
        process_episodes([episode], settings=settings, out_dir=dest)
        return

    # RSS feed URL — default to latest episode only unless --limit / --all
    try:
        title, _slug, episodes = parse_feed(target)
    except FeedParseError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not episodes:
        err_console.print("[red]No episodes with audio enclosures found in feed.[/red]")
        raise typer.Exit(1)

    if all_episodes:
        selected = list(episodes)
    elif limit is not None:
        selected = list(episodes[:limit])
    else:
        selected = list(episodes[:1])

    selected = [
        Episode(
            guid=e.guid,
            title=e.title,
            enclosure_url=e.enclosure_url,
            published_at=e.published_at,
            episode_num=e.episode_num,
            description=e.description,
            link=e.link,
            show_title=title,
        )
        for e in selected
    ]
    if not quiet:
        console.print(f"[bold]{title}[/bold]: {len(selected)} episode(s)")
    process_episodes(selected, settings=settings, out_dir=dest)


@app.command("format")
def format_cmd(
    json_path: Path = typer.Argument(..., help="Existing podtx transcript .json file"),
    readable: bool = typer.Option(
        False,
        "--readable",
        help="Paragraph breaks on silence gaps",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Strip fillers (uh/um) and collapse consecutive word doubles",
    ),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Output format (repeatable): txt, json, srt, vtt"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Output directory (default: same as JSON)"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Re-format an existing transcript JSON without re-running ASR."""
    path = json_path.expanduser()
    if not path.is_file():
        err_console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    formats = tuple(format) if format else ("txt", "json")
    try:
        paths = reformat_transcript(
            path,
            out_dir=out_dir,
            readable=readable,
            cleanup=cleanup,
            formats=formats,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not quiet:
        for p in paths:
            console.print(f"[green]Wrote[/green] {p}")
