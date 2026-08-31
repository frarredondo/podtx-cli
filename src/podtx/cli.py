from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from podtx import __version__
from podtx.config import Settings, ensure_data_dirs, load_settings
from podtx.db import Database
from podtx.download import FFmpegNotFoundError, require_ffmpeg
from podtx.engines import available_engines
from podtx.models import Episode, Feed
from podtx.pipeline import process_episodes, select_episodes_for_sync, transcribe_local_file
from podtx.format_cmd import (
    TranscriptJsonError,
    discover_transcript_jsons,
    load_transcript_json,
    reformat_many,
    reformat_transcript,
)
from podtx.nuggets import (
    DryRunEstimate,
    NuggetsError,
    _checked_formats,
    _merge_corpus_markdown,
    _valid_backend,
    estimate_dry_run,
    extract_nuggets_transcript,
    merge_nugget_sidecars,
    nuggets_many,
)
from podtx.providers import (
    CatalogError,
    ModelInfo,
    ProviderError,
    available_providers,
    build_provider,
    catalog_providers,
    get_model,
    list_models,
    load_catalog,
)
from podtx.rename_cmd import rename_many_from_title
from podtx.summarize import SummarizeError, summarize_many, summarize_transcript
from podtx.rss import FeedParseError, parse_feed, suggest_unique_slug

app = typer.Typer(
    name="podtx",
    help="Pull and transcribe podcast episodes from RSS using local AI models.",
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)
auth_app = typer.Typer(help="Manage API keys in macOS Keychain for summarize backends.", no_args_is_help=True)
app.add_typer(auth_app, name="auth")
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
    correct_names: Optional[bool] = None,
    diarize: Optional[bool] = None,
    diarize_backend: Optional[str] = None,
    diarize_model: Optional[str] = None,
    diarize_base_url: Optional[str] = None,
    diarize_api_key: Optional[str] = None,
    diarize_api_key_service: Optional[str] = None,
    diarize_api_key_account: Optional[str] = None,
    diarize_timeout: Optional[float] = None,
    trim_start: Optional[float] = None,
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
        correct_names=correct_names,
        diarize=diarize,
        diarize_backend=diarize_backend,
        diarize_model=diarize_model,
        diarize_base_url=diarize_base_url,
        diarize_api_key=diarize_api_key,
        diarize_api_key_service=diarize_api_key_service,
        diarize_api_key_account=diarize_api_key_account,
        diarize_timeout=diarize_timeout,
        trim_start=trim_start,
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


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"


def _transcript_disk_size(settings: Settings, slug: str) -> int:
    root = settings.transcripts_dir(slug)
    if not root.is_dir():
        return 0
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:  # pragma: no cover - invariant: no_args_is_help=True shows help before the callback runs
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
    """List registered feeds with episode counts, health and disk usage."""
    settings = load_settings(data_dir=data_dir)
    with _open_db(settings) as db:
        feeds = db.list_feeds()
        if not feeds:
            console.print("[dim]No feeds registered. Use `podtx add <rss-url>`.[/dim]")
            return
        health_by_id = {r["feed_id"]: r for r in db.feed_health_summary()}
    table = Table(title="Registered feeds", show_lines=False)
    table.add_column("Slug", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("URL", overflow="fold")
    table.add_column("Episodes", no_wrap=True)
    table.add_column("Health", justify="center", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True)
    for f in feeds:
        h = health_by_id.get(
            f.id,
            {"total_episodes": 0, "done_count": 0, "pending_count": 0, "error_count": 0, "health_status": "empty"},
        )
        episodes_str = (
            f"{h['done_count']} done / {h['pending_count']} pending / "
            f"{h['error_count']} failed ({h['total_episodes']})"
        )
        size_bytes = _transcript_disk_size(settings, f.slug)
        size_str = _human_size(size_bytes) if size_bytes else "-"
        health = str(h["health_status"])
        if health == "healthy":
            health_disp = f"[green]{escape(health)}[/green]"
        elif health == "empty":
            health_disp = f"[dim]{escape(health)}[/dim]"
        else:
            health_disp = f"[yellow]{escape(health)}[/yellow]"
        table.add_row(f.slug, f.title, f.url, episodes_str, health_disp, size_str)
    console.print(table)


@app.command("show")
def show_feed(
    feed: str = typer.Argument(..., help="Feed slug or URL"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """List known episodes and transcription status for a feed, with summary header."""
    settings = load_settings(data_dir=data_dir)
    with _open_db(settings) as db:
        f = db.get_feed(feed)
        if not f:
            err_console.print(f"[red]Feed not found:[/red] {feed}")
            raise typer.Exit(1)
        rows = db.list_episodes(f.id)
        # Summary header
        summary = db.feed_health_summary()
        # find this feed's summary
        h = next((r for r in summary if r["feed_id"] == f.id), None)
        if h:
            size_bytes = _transcript_disk_size(settings, f.slug)
            size_str = _human_size(size_bytes) if size_bytes else "-"
            last = db.last_transcribed_at(f.id)
            last_str = last[:10] if last else "-"
            health = str(h["health_status"])
            if health == "healthy":
                health_disp = f"[green]{health}[/green]"
            elif health == "empty":
                health_disp = f"[dim]{health}[/dim]"
            else:
                health_disp = f"[yellow]{health}[/yellow]"
            console.print(
                f"[bold]{escape(f.title)}[/bold] ({escape(f.slug)}) — "
                f"Episodes: {h['total_episodes']} · {h['done_count']} done · {h['pending_count']} pending · {h['error_count']} failed | "
                f"Health: {health_disp} | Size: {size_str}"
            )
            console.print(f"Last transcribed: {escape(last_str)} | Last sync: {escape(last_str)}")
            pending = int(h["pending_count"])
            total = int(h["total_episodes"])
            if pending > 0:
                console.print(f"Pending queue: {pending} pending / {total} (showing {len(rows)} rows)")
            if h["health_status"] == "empty":
                console.print("[dim]No episodes recorded yet. (empty feed) — run `podtx sync`.[/dim]")

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
        if h is None or h["health_status"] != "empty":
            console.print("[dim]No episodes recorded yet. Run `podtx sync`.[/dim]")
    else:
        console.print(table)


def _episode_num_label(episode_num: object) -> str:
    return str(episode_num) if episode_num is not None else "000"


def _error_detail(paths_json: str | None) -> str:
    """Extract the recorded error message for an 'error' episode, if any."""
    if not paths_json:
        return "unknown error"
    try:
        payload = json.loads(paths_json)
    except json.JSONDecodeError:
        return "unknown error"
    if isinstance(payload, dict):
        message = payload.get("error")
        if message:
            return str(message)
    return "unknown error"


def _parse_output_paths(paths_json: str | None) -> list[str]:
    """Parse recorded output paths; unparseable records yield no paths."""
    if not paths_json:
        return []
    try:
        payload = json.loads(paths_json)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [str(p) for p in payload]
    return []


def collect_doctor_report(db: Database) -> list[dict[str, object]]:
    """Collect per-episode rows that need attention across all feeds.

    Row keys: feed_id, feed_title, feed_slug, guid, episode_num, title,
    issue ("failed" | "pending" | "missing outputs"), detail.
    """
    feeds = {f.id: f for f in db.list_feeds()}
    rows: list[dict[str, object]] = []
    for feed_id, feed in feeds.items():
        for row in db.list_episodes(feed_id):
            status = row["status"]
            if status == "error":
                issue, detail = "failed", _error_detail(row["output_paths_json"])
            elif status == "pending":
                issue, detail = "pending", "not transcribed"
            else:  # done: sanity-check that recorded outputs still exist
                # Only absolute records can be checked. Legacy rows hold paths
                # relative to the cwd of the sync that wrote them, which is
                # unknowable here — never report those as missing.
                missing = [
                    p
                    for p in _parse_output_paths(row["output_paths_json"])
                    if Path(p).is_absolute() and not Path(p).exists()
                ]
                if not missing:
                    continue
                issue = "missing outputs"
                detail = ", ".join(Path(p).name for p in missing)
            rows.append({
                "feed_id": feed_id,
                "feed_title": feed.title,
                "feed_slug": feed.slug,
                "guid": row["guid"],
                "episode_num": row["episode_num"],
                "title": row["title"],
                "issue": issue,
                "detail": detail,
            })
    return rows


@app.command("doctor")
def doctor_cmd(
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory"
    ),
) -> None:
    """Report library health: failed, stuck, and missing transcripts.

    Read-only check over the episode database. Lists episodes that failed
    or are still pending, feeds with no recorded episodes, and done
    episodes whose output files no longer exist. Never creates or modifies
    library state and always exits 0; it reports, it does not retry.
    """
    settings = load_settings(data_dir=data_dir)
    db_path = settings.state_db_path()
    if not db_path.exists():
        # Do not create a library here: a mistyped --data-dir must stay visible.
        console.print(
            f"[dim]No feeds registered — no library at {escape(str(db_path))}. "
            "Use `podtx add <rss-url>`.[/dim]"
        )
        return
    with Database(db_path) as db:
        feeds: dict[int, Feed] = {f.id: f for f in db.list_feeds()}
        if not feeds:
            console.print("[dim]No feeds registered. Use `podtx add <rss-url>`.[/dim]")
            return
        summaries = db.feed_health_summary()
        attention = collect_doctor_report(db)
        empty_feed_ids = {int(f["id"]) for f in db.empty_feeds()}

    by_feed: dict[int, list[dict[str, object]]] = {}
    for row in attention:
        by_feed.setdefault(int(row["feed_id"]), []).append(row)

    health_colors = {"healthy": "green", "unhealthy": "red", "empty": "yellow"}
    table = Table(title="Library health")
    table.add_column("Feed")
    table.add_column("Status")
    table.add_column("Episodes")
    table.add_column("Detail")
    # Escalate display status: a feed with missing outputs is 'done' in the
    # database, but missing transcripts still count as needing attention.
    effective: dict[int, str] = {}
    for s in summaries:
        feed_id = int(s["feed_id"])
        if feed_id in empty_feed_ids:
            effective[feed_id] = "empty"
        elif by_feed.get(feed_id):
            effective[feed_id] = "unhealthy"
        else:
            effective[feed_id] = str(s["health_status"])

    for s in summaries:
        feed_id = int(s["feed_id"])
        feed = feeds.get(feed_id)
        name = escape(f"{s['title']} ({feed.slug})" if feed else str(s["title"]))
        parts = [
            f"{n} {issue}"
            for issue in ("failed", "pending", "missing outputs")
            if (n := sum(1 for r in by_feed.get(feed_id, []) if r["issue"] == issue))
        ]
        total = int(s["total_episodes"])
        detail = (
            "no episodes yet — run `podtx sync`"
            if feed_id in empty_feed_ids
            else (", ".join(parts) if parts else "ok")
        )
        health = effective[feed_id]
        table.add_row(
            name,
            f"[{health_colors[health]}]{health}[/{health_colors[health]}]",
            f"{s['done_count']}/{total} done",
            detail,
        )
    console.print(table)

    if attention:
        issue_colors = {"failed": "red", "pending": "yellow", "missing outputs": "red"}
        at = Table(title="Needs attention")
        at.add_column("Feed")
        at.add_column("Issue")
        at.add_column("Ep")
        at.add_column("Title")
        at.add_column("Detail")
        for row in attention:
            color = issue_colors[str(row["issue"])]
            at.add_row(
                escape(str(row["feed_slug"])),
                f"[{color}]{row['issue']}[/{color}]",
                _episode_num_label(row["episode_num"]),
                escape(str(row["title"])),
                escape(str(row["detail"])),
            )
        console.print(at)

    # Keyed off displayed status, not off `attention`: an empty feed has no
    # episode rows to list but still needs attention.
    feeds_needing = sum(1 for h in effective.values() if h != "healthy")
    if feeds_needing:
        line = f"[red]{feeds_needing} of {len(summaries)} feed(s) need attention[/]"
        if attention:
            line += f" ({len(attention)} episode(s))"
        console.print(line)
    else:
        console.print(
            f"[green]All {len(summaries)} feed(s) healthy — nothing needs attention.[/green]"
        )


@app.command("sync")
def sync_feeds(
    feed: Optional[str] = typer.Argument(None, help="Optional feed slug/URL (default: all)"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="parakeet or whisper"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="HF model id"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max new episodes per feed"),
    all_episodes: bool = typer.Option(False, "--all", help="Process all pending episodes"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Retain downloaded audio"),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Output format (repeatable): txt, json, srt, vtt, md"
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
    correct_names: bool = typer.Option(
        False,
        "--correct-names",
        help="Conservative proper-noun correction: build per-episode glossary from title/show/description/link and fix close misspellings in body text only (segments stay raw). Reports substitutions; byte-identical when off.",
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Speaker diarization: label segments with SPEAKER_00/01 + reflect turns in text. Opt-in; off by default (single-speaker unchanged). Performance/memory impact: local, CPU-bound.",
    ),
    diarize_backend: Optional[str] = typer.Option(
        None,
        "--diarize-backend",
        help="Diarization backend: fake (round-robin), pyannote (local), hf, assemblyai, deepgram (default: fake)",
    ),
    diarize_model: Optional[str] = typer.Option(
        None, "--diarize-model", help="Diarization model id (default for pyannote/hf: pyannote/speaker-diarization-3.1)",
    ),
    diarize_api_key: Optional[str] = typer.Option(
        None, "--diarize-api-key", help="API key for diarize backend (or env HF_TOKEN/ASSEMBLYAI_API_KEY, or Keychain via `podtx auth set`)",
    ),
    diarize_base_url: Optional[str] = typer.Option(
        None, "--diarize-base-url", help="Override base URL for diarize backend",
    ),
    diarize_timeout: Optional[float] = typer.Option(
        None, "--diarize-timeout", help="Diarization request timeout seconds (default 120)",
    ),
    trim_start: Optional[float] = typer.Option(
        None,
        "--trim-start",
        help="Skip first N seconds of audio before transcription (e.g. --trim-start 20). "
        "Default 0 (no trimming). Caveat: can delete substantive opening content — use only "
        "when you know the feed's intro length. Requires re-transcription; `podtx format` "
        "without ASR does not re-apply trimming. seconds",
    ),
) -> None:
    """Download and transcribe new episodes for registered feeds."""
    if trim_start is not None and trim_start < 0:  # pragma: no cover
        err_console.print("[red]--trim-start must be >= 0[/red]")  # pragma: no cover
        raise typer.Exit(1)  # pragma: no cover
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
        correct_names=True if correct_names else None,
        diarize=True if diarize else None,
        diarize_backend=diarize_backend,
        diarize_model=diarize_model,
        diarize_base_url=diarize_base_url,
        diarize_api_key=diarize_api_key,
        diarize_timeout=diarize_timeout,
        trim_start=trim_start,
    )
    settings = replace(
        settings,
        formats=_merge_formats(settings.formats, format),
        keep_audio=keep_audio or settings.keep_audio,
        readable=readable or settings.readable,
        cleanup=cleanup or settings.cleanup,
        correct_names=correct_names or settings.correct_names,
        diarize=diarize or settings.diarize,
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
    format: Optional[list[str]] = typer.Option(None, "--format", "-f", help="Output format (repeatable): txt, json, srt, vtt, md"),
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
    correct_names: bool = typer.Option(
        False,
        "--correct-names",
        help="Conservative proper-noun correction: build per-episode glossary from title/show/description/link and fix close misspellings in body text only (segments stay raw). Reports substitutions; byte-identical when off.",
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Speaker diarization: label segments with SPEAKER_00/01 + reflect turns in text. Opt-in; off by default.",
    ),
    diarize_backend: Optional[str] = typer.Option(
        None,
        "--diarize-backend",
        help="Diarization backend: fake (round-robin), pyannote (local), hf, assemblyai, deepgram (default: fake)",
    ),
    diarize_model: Optional[str] = typer.Option(
        None, "--diarize-model", help="Diarization model id (default for pyannote/hf: pyannote/speaker-diarization-3.1)",
    ),
    diarize_api_key: Optional[str] = typer.Option(
        None, "--diarize-api-key", help="API key for diarize backend (or env HF_TOKEN/ASSEMBLYAI_API_KEY, or Keychain)",
    ),
    diarize_base_url: Optional[str] = typer.Option(
        None, "--diarize-base-url", help="Override base URL for diarize backend",
    ),
    diarize_timeout: Optional[float] = typer.Option(
        None, "--diarize-timeout", help="Diarization request timeout seconds (default 120)",
    ),
    trim_start: Optional[float] = typer.Option(
        None,
        "--trim-start",
        help="Skip first N seconds of audio before transcription (e.g. --trim-start 20). "
        "Default 0 (no trimming). Caveat: can delete substantive opening content — use only "
        "when you know the feed's intro length. Requires re-transcription; `podtx format` "
        "without ASR does not re-apply trimming. seconds",
    ),
) -> None:
    """One-shot transcription of a feed, audio URL, or local file."""
    if trim_start is not None and trim_start < 0:  # pragma: no cover
        err_console.print("[red]--trim-start must be >= 0[/red]")  # pragma: no cover
        raise typer.Exit(1)  # pragma: no cover
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
        correct_names=True if correct_names else None,
        diarize=True if diarize else None,
        diarize_backend=diarize_backend,
        diarize_model=diarize_model,
        diarize_base_url=diarize_base_url,
        diarize_api_key=diarize_api_key,
        diarize_timeout=diarize_timeout,
        trim_start=trim_start,
    )
    settings = replace(
        settings,
        formats=_merge_formats(settings.formats, format),
        keep_audio=keep_audio or settings.keep_audio,
        readable=readable or settings.readable,
        cleanup=cleanup or settings.cleanup,
        correct_names=correct_names or settings.correct_names,
        diarize=diarize or settings.diarize,
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
    json_path: Optional[Path] = typer.Argument(
        None,
        help="Existing podtx transcript .json file (omit when using --feed / --all)",
    ),
    feed: Optional[str] = typer.Option(
        None,
        "--feed",
        help="Re-format all transcript JSON files for a feed slug",
    ),
    all_feeds: bool = typer.Option(
        False,
        "--all",
        help="Re-format all transcript JSON files in the library",
    ),
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
    correct_names: bool = typer.Option(
        False,
        "--correct-names",
        help="Conservative proper-noun correction: build per-episode glossary from title/show/description/link and fix close misspellings in body text only (segments stay raw). Reports substitutions; byte-identical when off.",
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize",
        help="Speaker diarization: label segments with SPEAKER_00/01 + reflect turns in text. Opt-in; off by default.",
    ),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Output format (repeatable): txt, json, srt, vtt, md"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Output directory (default: same as JSON)"
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory (for --feed / --all)"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Re-format existing transcript JSON without re-running ASR.

    Target one file, one feed (`--feed`), or the whole library (`--all`).

    Note: --trim-start is an audio operation that requires re-transcription;
    `format` does not re-apply trimming.
    """
    modes = sum([json_path is not None, feed is not None, all_feeds])
    if modes != 1:
        err_console.print(
            "[red]Specify exactly one of:[/red] a JSON path, `--feed <slug>`, or `--all`"
        )
        raise typer.Exit(1)

    formats = tuple(format) if format else ("txt", "json")

    if json_path is not None:
        path = json_path.expanduser()
        if not path.is_file():
            err_console.print(f"[red]File not found:[/red] {path}")
            raise typer.Exit(1)
        try:
            paths = reformat_transcript(
                path,
                out_dir=out_dir,
                readable=readable,
                cleanup=cleanup,
                correct_names=correct_names,
                diarize=diarize,
                formats=formats,
            )
        except TranscriptJsonError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        # Single-file incremental indexing (best-effort)
        try:
            settings_sf = load_settings(data_dir=data_dir)
            jpath_sf = next((p for p in paths if p.suffix == ".json"), None)
            if jpath_sf and settings_sf.state_db_path().exists():
                # If custom data_dir, check file is under transcripts or just index anyway
                payload_sf = json.loads(jpath_sf.read_text(encoding="utf-8"))
                feed_slug_sf = jpath_sf.parent.name
                guid_sf = str(payload_sf.get("guid") or jpath_sf.stem)
                title_sf = str(payload_sf.get("title") or jpath_sf.stem)
                published_at_sf = str(payload_sf.get("date")) if payload_sf.get("date") else None
                text_sf = str(payload_sf.get("text") or "")
                if not text_sf and payload_sf.get("segments"):  # pragma: no cover - reformat fills text from segments, so written JSON never hits this
                    text_sf = " ".join(str(s.get("text","")) for s in payload_sf.get("segments") or [] if s.get("text"))
                txt_path_sf = str(jpath_sf.with_suffix(".txt"))
                try:
                    txt_path_sf = str(jpath_sf.with_suffix(".txt").resolve())
                    json_path_sf = str(jpath_sf.resolve())
                except Exception:  # pragma: no cover - best-effort resolve
                    json_path_sf = str(jpath_sf)
                with Database(settings_sf.state_db_path()) as sdb_sf:
                    sdb_sf.upsert_search_entry(
                        feed_slug=feed_slug_sf,
                        guid=guid_sf,
                        title=title_sf,
                        published_at=published_at_sf,
                        text=text_sf,
                        txt_path=txt_path_sf,
                        json_path=json_path_sf,
                    )
        except Exception:  # pragma: no cover - best-effort indexing
            pass
        if not quiet:
            for p in paths:
                console.print(f"[green]Wrote[/green] {p}")
            if correct_names:
                try:
                    for jp in [p for p in paths if p.suffix.lower() == ".json"]:
                        payload = json.loads(jp.read_text(encoding="utf-8"))
                        corr = payload.get("corrections") or []
                        if corr:
                            console.print(f"[dim]Corrected {len(corr)} name(s) in {jp.name}: {', '.join(f'{a} → {b}' for a, b in corr[:3])}[/dim]")
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover
        return

    settings = load_settings(data_dir=data_dir)
    transcripts_root = settings.transcripts_dir()
    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not targets:
        err_console.print("[dim]No transcript JSON files found.[/dim]")
        raise typer.Exit(1)

    if not quiet:
        scope = "all feeds" if all_feeds else f"feed {feed}"
        console.print(f"[bold]Formatting {len(targets)} transcript(s)[/bold] ({scope})")

    result = reformat_many(
        targets,
        out_dir=out_dir,
        readable=readable,
        cleanup=cleanup,
        correct_names=correct_names,
        diarize=diarize,
        formats=formats,
    )

    # Incremental FTS indexing for batch reformats
    if result.ok:
        try:
            with Database(settings.state_db_path()) as sdb:
                json_written = [p for p in result.written if p.suffix == ".json"]
                # When out_dir is custom, index written files; otherwise targets are overwritten in-place
                for jpath in json_written:
                    try:
                        feed_slug = jpath.parent.name
                        payload = json.loads(jpath.read_text(encoding="utf-8"))
                        guid = str(payload.get("guid") or jpath.stem)
                        title = str(payload.get("title") or jpath.stem)
                        published_at = str(payload.get("date")) if payload.get("date") else None
                        text = str(payload.get("text") or "")
                        if not text and payload.get("segments"):  # pragma: no cover - same as above, written JSON text already filled by reformat
                            text = " ".join(str(s.get("text", "")) for s in payload.get("segments") or [] if s.get("text"))
                        txt_path = str(jpath.with_suffix(".txt"))
                        try:
                            txt_path = str(jpath.with_suffix(".txt").resolve())
                            json_path_str = str(jpath.resolve())
                        except Exception:  # pragma: no cover - best-effort resolve
                            json_path_str = str(jpath)
                        sdb.upsert_search_entry(
                            feed_slug=feed_slug,
                            guid=guid,
                            title=title,
                            published_at=published_at,
                            text=text,
                            txt_path=txt_path,
                            json_path=json_path_str,
                        )
                    except Exception:  # pragma: no cover - per-file best-effort
                        continue
        except Exception:  # pragma: no cover - batch best-effort indexing
            pass

    if not quiet:
        for p in result.written:
            console.print(f"[green]Wrote[/green] {p}")
        for path, message in result.errors:
            err_console.print(f"[red]Failed[/red] {path}: {message}")
        console.print(
            f"[bold]Done[/bold]: {result.ok} ok, {result.failed} failed"
        )

    if result.failed:
        raise typer.Exit(1)


@app.command("summarize")
def summarize_cmd(
    json_path: Optional[Path] = typer.Argument(
        None,
        help="Existing podtx transcript .json file (omit when using --feed / --all)",
    ),
    feed: Optional[str] = typer.Option(
        None,
        "--feed",
        help="Summarize all transcript JSON files for a feed slug",
    ),
    all_feeds: bool = typer.Option(
        False,
        "--all",
        help="Summarize all transcript JSON files in the library",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Max transcripts to summarize (for --feed / --all)",
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory (for --feed / --all)"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Output directory (default: same as transcript JSON)"
    ),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Summary output format (repeatable): json, md (default: json)"
    ),
    backend: str = typer.Option(
        "fake",
        "--backend",
        help="Summary backend: fake (offline), openrouter, opencode, lmstudio/local (OpenAI-compatible)",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id (required for lmstudio/local, default for openrouter/opencode)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or env OPENROUTER_API_KEY/OPENCODE_API_KEY, or Keychain via `podtx auth set`)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override base URL (e.g. LM Studio custom port)"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Request timeout seconds (default 60)"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM temperature (default 0.3)"),
    max_input_chars: Optional[int] = typer.Option(None, "--max-input-chars", help="Truncate transcript to N chars (default: no truncation)"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Summarize existing transcript JSON without re-running ASR.

    Reads one transcript JSON file, or all JSON files for a feed (`--feed`)
    or the whole library (`--all`), and writes stable sidecar summaries alongside
    each transcript (e.g. `episode.summary.json` / `episode.summary.md` with
    overview, key points, and optional timestamped quotes). Default backend is
    `fake` (offline extractive, no network). Use `openrouter`, `opencode`, or
    `lmstudio`/`local` for LLM summaries (OpenAI-compatible).

    Invocation: `podtx summarize <path/to/episode.json>`, `podtx summarize --feed <slug> [--limit N]`,
    or `podtx summarize --all [--limit N]`. Output sidecars are written to the
    same directory as the transcript JSON unless `--out-dir` is given. Use
    `--format json` (default) and/or `--format md` to choose output format(s).

    API keys: prefer `podtx auth set --backend openrouter|opencode` (Keychain), or env `OPENROUTER_API_KEY`/`OPENCODE_API_KEY`, or `--api-key`.
    """
    modes = sum([json_path is not None, feed is not None, all_feeds])
    if modes != 1:
        err_console.print(
            "[red]Specify exactly one of:[/red] a JSON path, `--feed <slug>`, or `--all`"
        )
        raise typer.Exit(1)

    # Resolve settings for defaults / keychain service/account
    settings = load_settings(data_dir=data_dir)
    # Backend validation via summarize module (includes aliases)
    from podtx.summarize import _SUMMARY_BACKENDS, _ALIAS

    normalized = backend.lower().strip()
    normalized = _ALIAS.get(normalized, normalized)
    if normalized not in _SUMMARY_BACKENDS:
        err_console.print(f"[red]Unknown backend:[/red] {backend} (choose from: {', '.join(sorted(_SUMMARY_BACKENDS))})")
        raise typer.Exit(1)

    formats = tuple(format) if format else ("json",)
    for fmt in formats:
        if fmt.lower().strip() not in {"json", "md"}:
            err_console.print(f"[red]Unsupported format {fmt!r}. Choose from: json, md[/red]")
            raise typer.Exit(1)

    # Resolve summarize opts from settings if not provided via CLI
    resolved_model = model if model is not None else settings.summarize_model
    resolved_base = base_url if base_url is not None else settings.summarize_base_url
    resolved_timeout = timeout if timeout is not None else settings.summarize_timeout
    resolved_temp = temperature if temperature is not None else settings.summarize_temperature
    resolved_max = max_input_chars if max_input_chars is not None else settings.summarize_max_input_chars
    # api_key resolution deferred to summarize module (needs keychain)
    # Pass settings_api_key/service/account so summarize can fallback to keychain
    summary_kwargs = dict(
        backend=backend,
        model=resolved_model,
        api_key=api_key,
        base_url=resolved_base,
        timeout=resolved_timeout,
        temperature=resolved_temp,
        max_input_chars=resolved_max,
        settings_api_key=settings.summarize_api_key,
        service=settings.summarize_api_key_service,
        account=settings.summarize_api_key_account,
    )

    if json_path is not None:
        path = json_path.expanduser()
        if not path.is_file():
            err_console.print(f"[red]File not found:[/red] {path}")
            raise typer.Exit(1)
        try:
            paths = summarize_transcript(
                path,
                out_dir=out_dir,
                formats=formats,
                **summary_kwargs,
            )
        except TranscriptJsonError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except (ValueError, SummarizeError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if not quiet:
            for p in paths:
                console.print(f"[green]Wrote[/green] {p}")
        return

    transcripts_root = settings.transcripts_dir()
    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not targets:
        err_console.print("[dim]No transcript JSON files found.[/dim]")
        raise typer.Exit(1)

    if limit is not None:
        targets = targets[:limit]

    if not quiet:
        scope = "all feeds" if all_feeds else f"feed {feed}"
        console.print(f"[bold]Summarizing {len(targets)} transcript(s)[/bold] ({scope}, backend: {backend})")

    result = summarize_many(
        targets,
        out_dir=out_dir,
        formats=formats,
        **summary_kwargs,
    )

    if not quiet:
        for p in result.written:
            console.print(f"[green]Wrote[/green] {p}")
        for path, message in result.errors:
            err_console.print(f"[red]Failed[/red] {path}: {message}")
        console.print(f"[bold]Done[/bold]: {result.ok} ok, {result.failed} failed")

    if result.failed:
        raise typer.Exit(1)


@app.command("nuggets")
def nuggets_cmd(
    json_path: Optional[Path] = typer.Argument(
        None,
        help="Existing podtx transcript .json file (omit when using --feed / --all)",
    ),
    feed: Optional[str] = typer.Option(
        None,
        "--feed",
        help="Extract nuggets from all transcript JSON files for a feed slug",
    ),
    all_feeds: bool = typer.Option(
        False,
        "--all",
        help="Extract nuggets from all transcript JSON files in the library",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Max transcripts to process (for --feed / --all)",
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory (for --feed / --all)"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="Output directory (default: same as transcript JSON)"
    ),
    format: Optional[list[str]] = typer.Option(
        None, "--format", "-f", help="Nuggets output format (repeatable): json, md (default: json)"
    ),
    backend: str = typer.Option(
        "fake",
        "--backend",
        help="Nuggets backend: fake (offline), openrouter, opencode, openai, anthropic, lmstudio/local",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Alias for --backend"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id (required for openai/anthropic/lmstudio; default for openrouter/opencode)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or env OPENROUTER_API_KEY/OPENCODE_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY, or Keychain via `podtx auth set`)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override base URL (e.g. LM Studio custom port)"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Request timeout seconds (default 120)"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM temperature (default 0.3)"),
    max_input_chars: Optional[int] = typer.Option(None, "--max-input-chars", help="Chunk transcript at N chars, split on segment boundaries with overlap (default: 100000)"),
    force: bool = typer.Option(False, "--force", help="Re-extract even when a fresh sidecar exists"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate tokens and cost from the models.dev catalog without calling any backend or writing files",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Merge same-idea nuggets across existing sidecars for a feed or the whole library, writing corpus.nuggets.json/.md",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Extract durable insights ("nuggets") from existing transcript JSON.

    Reads one transcript JSON file, or all JSON files for a feed (`--feed`)
    or the whole library (`--all`), applies a versioned scored rubric, and
    writes stable sidecars alongside each transcript (e.g. `episode.nuggets.json`
    / `episode.nuggets.md`). Every quote is mechanically verified against the
    transcript and cited with a `[hh:mm:ss]` timestamp resolved from segments.

    Default backend is `fake` (offline extractive, no network). Use
    `openrouter`, `opencode`, `openai`, `anthropic`, or `lmstudio`/`local`
    for LLM extraction (OpenAI-compatible / Anthropic) — always opt-in.

    Re-runs skip episodes whose sidecar matches the current rubric +
    backend + model; use `--force` to re-extract.

    API keys: prefer `podtx auth set --backend openrouter|opencode|openai|anthropic`
    (Keychain), or env `OPENROUTER_API_KEY` / `OPENCODE_API_KEY` /
    `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or `--api-key`.
    """
    effective_backend = provider if provider is not None else backend
    modes = sum([json_path is not None, feed is not None, all_feeds])
    if modes != 1:
        err_console.print(
            "[red]Specify exactly one of:[/red] a JSON path, `--feed <slug>`, or `--all`"
        )
        raise typer.Exit(1)

    settings = load_settings(data_dir=data_dir)

    try:
        effective_backend = _valid_backend(effective_backend)
    except NuggetsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    formats = tuple(format) if format else ("json",)
    try:
        formats = _checked_formats(formats)
    except NuggetsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if merge and json_path is not None:
        err_console.print(
            "[red]For --merge specify exactly one of:[/red] `--feed <slug>` or `--all` (not a JSON path)"
        )
        raise typer.Exit(1)
    if merge and dry_run:
        err_console.print("[red]--merge and --dry-run cannot be combined[/red]")
        raise typer.Exit(1)

    resolved_model = model if model is not None else settings.nuggets_model
    resolved_base = base_url if base_url is not None else settings.nuggets_base_url
    resolved_timeout = timeout if timeout is not None else settings.nuggets_timeout
    resolved_temp = temperature if temperature is not None else settings.nuggets_temperature
    resolved_max = max_input_chars if max_input_chars is not None else settings.nuggets_max_input_chars
    nuggets_kwargs = dict(
        backend=effective_backend,
        model=resolved_model,
        api_key=api_key,
        base_url=resolved_base,
        timeout=resolved_timeout,
        temperature=resolved_temp,
        max_input_chars=resolved_max,
        force=force,
        settings_api_key=settings.nuggets_api_key,
        service=settings.nuggets_api_key_service,
        account=settings.nuggets_api_key_account,
    )

    if dry_run:
        _run_nuggets_dry_run(
            json_path=json_path,
            feed=feed,
            all_feeds=all_feeds,
            limit=limit,
            transcripts_root=settings.transcripts_dir(),
            data_dir=settings.data_dir,
            backend=effective_backend,
            model=resolved_model,
            max_input_chars=resolved_max,
            quiet=quiet,
        )
        return

    if merge:
        _run_nuggets_merge(
            feed=feed,
            all_feeds=all_feeds,
            out_dir=out_dir,
            transcripts_root=settings.transcripts_dir(),
            backend=effective_backend,
            model=resolved_model,
            api_key=api_key,
            base_url=resolved_base,
            timeout=resolved_timeout,
            temperature=resolved_temp,
            settings_api_key=settings.nuggets_api_key,
            service=settings.nuggets_api_key_service,
            account=settings.nuggets_api_key_account,
            quiet=quiet,
        )
        return

    if json_path is not None:
        path = json_path.expanduser()
        if not path.is_file():
            err_console.print(f"[red]File not found:[/red] {path}")
            raise typer.Exit(1)
        try:
            run = extract_nuggets_transcript(
                path,
                out_dir=out_dir,
                formats=formats,
                **nuggets_kwargs,
            )
        except TranscriptJsonError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except (ValueError, NuggetsError, ProviderError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        if run.skipped:
            if not quiet:
                console.print(f"[dim]Skipping[/dim] {path} (fresh sidecar for current rubric/backend/model)")
            return
        if not quiet:
            for p in run.written:
                console.print(f"[green]Wrote[/green] {p}")
        return

    transcripts_root = settings.transcripts_dir()
    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not targets:
        err_console.print("[dim]No transcript JSON files found.[/dim]")
        raise typer.Exit(1)

    if limit is not None:
        targets = targets[:limit]

    if not quiet:
        scope = "all feeds" if all_feeds else f"feed {feed}"
        console.print(f"[bold]Extracting nuggets from {len(targets)} transcript(s)[/bold] ({scope}, backend: {effective_backend})")

    result = nuggets_many(
        targets,
        out_dir=out_dir,
        formats=formats,
        **nuggets_kwargs,
    )

    if not quiet:
        for p in result.written:
            console.print(f"[green]Wrote[/green] {p}")
        for path, message in result.errors:
            err_console.print(f"[red]Failed[/red] {path}: {message}")
        console.print(f"[bold]Done[/bold]: {result.ok} ok, {result.skipped} skipped, {result.failed} failed")

    if result.failed:
        raise typer.Exit(1)


def _fmt_integer(value: int | None) -> str:
    return f"{value:,}" if value is not None else "—"


def _fmt_money(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "—"


def _print_dry_run(
    path: Path,
    episode: Episode,
    transcript: Transcript,
    *,
    backend: str,
    model: str | None,
    max_input_chars: int,
    raw: dict | None,
) -> DryRunEstimate:
    est = estimate_dry_run(
        episode,
        transcript,
        backend=backend,
        model=model,
        max_input_chars=max_input_chars,
        providers=raw or {},
    )
    console.print(
        f"[bold]Dry run[/bold]: {path} — {episode.title if episode else path.stem}"
    )
    console.print(
        f"  input: {est.input_chars:,} chars -> {est.input_tokens:,} tokens"
        f" | output est: {est.output_tokens:,} tokens"
        f" | total: {est.total_tokens:,} tokens"
    )
    if est.chunked:
        console.print(f"  plan: {est.chunk_count} chunks (over max-input-chars)")
    else:
        console.print("  plan: single pass")
    if backend == "fake":
        console.print("  backend fake: no inference call expected - token estimate only")
        return est
    if est.model_known:
        info = get_model(raw or {}, backend, model or "")
        console.print(f"  model: {model} ({info.name})")
        if est.cost_known:
            console.print(f"  cost: ${est.cost_usd:,.6f}")
        else:
            console.print("  cost: unknown (no pricing in catalog)")
    elif raw is not None:
        console.print(f"  cost: unknown (model '{model}' not in catalog)")
    else:
        console.print("  cost: unknown (catalog unavailable)")
    return est


def _run_nuggets_dry_run(
    *,
    json_path: Optional[Path],
    feed: Optional[str],
    all_feeds: bool,
    limit: Optional[int],
    transcripts_root: Path,
    data_dir: Path,
    backend: str,
    model: str | None,
    max_input_chars: int,
    quiet: bool,
) -> None:
    try:
        raw = load_catalog(data_dir)
    except CatalogError:
        raw = None
    if json_path is not None:
        path = json_path.expanduser()
        if not path.is_file():
            err_console.print(f"[red]File not found:[/red] {path}")
            raise typer.Exit(1)
        try:
            episode, transcript = load_transcript_json(path)
        except TranscriptJsonError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        _print_dry_run(
            path,
            episode,
            transcript,
            backend=backend,
            model=model,
            max_input_chars=max_input_chars,
            raw=raw,
        )
        return

    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not targets:
        err_console.print("[dim]No transcript JSON files found.[/dim]")
        raise typer.Exit(1)

    if limit is not None:
        targets = targets[:limit]

    total_tokens = 0
    total_cost = 0.0
    costed = 0
    ok = 0
    for target in targets:
        try:
            episode, transcript = load_transcript_json(target)
        except TranscriptJsonError as exc:
            err_console.print(f"[red]Skipping[/red] {target}: {exc}")
            continue
        est = _print_dry_run(
            target,
            episode,
            transcript,
            backend=backend,
            model=model,
            max_input_chars=max_input_chars,
            raw=raw,
        )
        ok += 1
        total_tokens += est.total_tokens
        if est.cost_known:
            costed += 1
            total_cost += est.cost_usd
    if not quiet:
        console.print(
            f"[bold]TOTAL[/bold]: {ok} episodes, {total_tokens:,} tokens, "
            f"${total_cost:,.6f} ({costed}/{ok} costed)"
        )
    if ok != len(targets):
        raise typer.Exit(1)


def _run_nuggets_merge(
    *,
    feed: str | None,
    all_feeds: bool,
    out_dir: Path | None,
    transcripts_root: Path,
    backend: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    timeout: float,
    temperature: float,
    settings_api_key: str | None,
    service: str | None,
    account: str | None,
    quiet: bool,
) -> None:
    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    sidecars: list[Path] = []
    for target in targets:
        side = target.with_name(target.stem + ".nuggets.json")
        if side.is_file():
            sidecars.append(side)

    if not sidecars:
        if not quiet:
            console.print("[dim]No nugget sidecar files found in scope; nothing to merge.[/dim]")
        return

    provider = None
    if backend != "fake":
        provider = build_provider(
            backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            settings_api_key=settings_api_key,
            service=service,
            account=account,
        )

    corpus = merge_nugget_sidecars(
        sidecars,
        in_scope=len(targets),
        provider=provider,
        timeout=timeout,
        temperature=temperature,
    )

    if out_dir is not None:
        dest = out_dir.expanduser()
    elif all_feeds:
        dest = transcripts_root
    else:
        dest = transcripts_root / feed
    dest.mkdir(parents=True, exist_ok=True)

    corpus_json = dest / "corpus.nuggets.json"
    corpus_md = dest / "corpus.nuggets.md"
    corpus_json.write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    corpus_md.write_text(_merge_corpus_markdown(corpus), encoding="utf-8")

    if quiet:
        return
    skipped = corpus["sidecars_skipped"]
    suffix = f" ({len(skipped)} malformed sidecar skipped)" if skipped else ""
    console.print(
        f"[green]Merged[/green] {corpus['episodes_processed']} episodes \u2192 "
        f"{len(corpus['groups'])} groups ({corpus['clustering']}){suffix}"
    )
    console.print(f"[dim]Wrote[/dim] {corpus_json}")
    console.print(f"[dim]Wrote[/dim] {corpus_md}")


def _show_provider_counts(raw: dict) -> None:
    supported = sorted(set(available_providers()) & set(catalog_providers(raw)))
    if not supported:
        console.print("[dim]No configured providers found in the models.dev catalog.[/dim]")
        return
    table = Table(title="models.dev catalog - configured providers")
    table.add_column("Provider")
    table.add_column("Name")
    table.add_column("Models")
    for pid in supported:
        entry = raw[pid]
        table.add_row(
            pid,
            entry.get("name", pid),
            str(len(list_models(raw, pid))),
        )
    console.print(table)


def _show_provider_models(raw: dict, *, provider: str, limit: Optional[int]) -> None:
    rows = list_models(raw, provider)
    if not rows:
        err_console.print(f"[red]Provider {provider!r} has no models in the models.dev catalog.[/red]")
        raise typer.Exit(1)
    if limit is not None:
        rows = rows[:limit]
    table = Table(title=f"models.dev catalog - {provider}")
    table.add_column("Model")
    table.add_column("Context")
    table.add_column("$/M in")
    table.add_column("$/M out")
    for model in rows:
        table.add_row(
            model.name,
            _fmt_integer(model.context_length),
            _fmt_money(model.cost_input_per_million),
            _fmt_money(model.cost_output_per_million),
        )
    console.print(table)


def _show_models(raw: dict, *, provider: Optional[str], model_id: str) -> None:
    matches: list[tuple[str, ModelInfo]] = []
    if provider is not None:
        info = get_model(raw, provider, model_id)
        if info is not None:
            matches.append((provider, info))
    else:
        for pid in catalog_providers(raw):
            info = get_model(raw, pid, model_id)
            if info is not None:
                matches.append((pid, info))
    if not matches:
        where = f" for provider {provider!r}" if provider is not None else ""
        err_console.print(f"[red]Model {model_id!r} is not in the models.dev catalog{where}.[/red]")
        raise typer.Exit(1)
    table = Table(title=f"models.dev catalog - {model_id}")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Context")
    table.add_column("$/M in")
    table.add_column("$/M out")
    for pid, model in matches:
        table.add_row(
            pid,
            model.name,
            _fmt_integer(model.context_length),
            _fmt_money(model.cost_input_per_million),
            _fmt_money(model.cost_output_per_million),
        )
    console.print(table)


@app.command("models")
def models_cmd(
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="List models for a single provider id (e.g. openrouter, lmstudio)",
    ),
    model_id: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Show a specific model id (searches all providers when --provider is omitted)",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Cap the number of models listed"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch the models.dev catalog, ignoring the cache"
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory (models.dev cache location)"
    ),
) -> None:
    """Inspect the models.dev catalog (metadata only, no inference).

    Shows which registered providers exist in the catalog and how many
    models each exposes, lists a provider's models with context window and
    USD pricing, or validates a model id (cross-provider search).
    """
    settings = load_settings(data_dir=data_dir)
    try:
        raw = load_catalog(settings.data_dir, refresh=refresh)
    except CatalogError as exc:
        err_console.print(f"[red]Failed to load models.dev catalog:[/red] {exc}")
        raise typer.Exit(1) from exc

    if model_id is not None:
        _show_models(raw, provider=provider, model_id=model_id)
        return
    if provider is not None:
        _show_provider_models(raw, provider=provider, limit=limit)
        return
    _show_provider_counts(raw)


@auth_app.command("set")
def auth_set(
    backend: str = typer.Argument(..., help="Backend: openrouter, opencode, anthropic, openai"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key value (if omitted, prompts securely)"),
    service: Optional[str] = typer.Option(None, "--service", help="Keychain service name (default: podtx-<backend>)"),
    account: Optional[str] = typer.Option(None, "--account", help="Keychain account name (default: api-key)"),
) -> None:
    """Save API key to macOS Keychain for a summarize/nuggets backend.

    Example: `podtx auth set openrouter` (prompts for key), or `podtx auth set opencode --api-key sk-...`.
    Stored under service `podtx-<backend>` and account `api-key` by default. Subsequent `podtx summarize --backend <backend>` / `podtx nuggets --backend <backend>` will read from Keychain if no --api-key / env key is provided.
    """
    b = backend.lower().strip()
    if b not in {"openrouter", "opencode", "anthropic", "openai"}:
        err_console.print(f"[red]Unknown backend for auth:[/red] {backend} (choose: openrouter, opencode)")
        raise typer.Exit(1)
    svc = service or f"podtx-{b}"
    acct = account or "api-key"
    secret = api_key
    if not secret:
        secret = typer.prompt(f"API key for {b}", hide_input=True)
        if not secret or not secret.strip():
            err_console.print("[red]No key provided[/red]")
            raise typer.Exit(1)
        secret = secret.strip()
    # Sanitize bracketed-paste artifacts (ESC[200~ / ESC[201~ etc. can leak as " [O [I" when hide_input=True)
    def _sanitize_key(raw: str) -> str:
        s = raw.strip()
        # strip surrounding quotes if pasted with them
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", "`"):
            s = s[1:-1].strip()
        # common bracketed-paste / escape fragments that leak with hide_input
        for seq in ("\x1b[200~", "\x1b[201~", "[200~", "[201~", "\x1b[O", "\x1b[I", "[O", "[I"):
            s = s.replace(seq, "")
        s = s.replace("\x1b", "").replace("\r", "").replace("\n", "")
        # leading garbage like " [" from broken paste
        s = s.lstrip(" [")
        return s.strip()
    sanitized = _sanitize_key(secret)
    if sanitized != secret:
        # warn but use sanitized version (common when pasting long keys with hide_input)
        console.print(f"[dim]Sanitized pasted key ({len(secret)} → {len(sanitized)} chars) — bracketed-paste artifacts removed[/dim]")
        secret = sanitized
    if not secret:
        err_console.print(f"[red]No key after sanitizing — try `podtx auth set {b} --api-key <key>` to avoid paste issues[/red]")
        raise typer.Exit(1)
    from podtx.keychain import save_api_key as _save

    try:
        _save(svc, acct, secret)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Saved[/green] {b} key to Keychain [dim]({svc} / {acct})[/dim]")


@auth_app.command("get")
def auth_get(
    backend: str = typer.Argument(..., help="Backend: openrouter, opencode, anthropic, openai, lmstudio"),
    service: Optional[str] = typer.Option(None, "--service"),
    account: Optional[str] = typer.Option(None, "--account"),
) -> None:
    """Check if API key exists in Keychain (does not print secret)."""
    b = backend.lower().strip()
    if b not in {"openrouter", "opencode", "lmstudio", "local", "anthropic", "openai"}:
        err_console.print(f"[red]Unknown backend:[/red] {backend}")
        raise typer.Exit(1)
    svc = service or f"podtx-{b}"
    acct = account or "api-key"
    from podtx.keychain import get_api_key as _get

    val = _get(svc, acct)
    if val:
        console.print(f"[green]Found[/green] {b} key in Keychain [dim]({svc} / {acct})[/dim] (length {len(val)})")
    else:
        console.print(f"[dim]No key found for {b} in Keychain ({svc} / {acct})[/dim]")
        raise typer.Exit(1)


@auth_app.command("delete")
def auth_delete(
    backend: str = typer.Argument(..., help="Backend: openrouter, opencode, anthropic, openai"),
    service: Optional[str] = typer.Option(None, "--service"),
    account: Optional[str] = typer.Option(None, "--account"),
) -> None:
    """Delete API key from Keychain."""
    b = backend.lower().strip()
    if b not in {"openrouter", "opencode", "anthropic", "openai"}:
        err_console.print(f"[red]Unknown backend for auth:[/red] {backend} (choose: openrouter, opencode)")
        raise typer.Exit(1)
    svc = service or f"podtx-{b}"
    acct = account or "api-key"
    from podtx.keychain import delete_api_key as _del

    try:
        ok = _del(svc, acct)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if ok:
        console.print(f"[green]Deleted[/green] {b} key from Keychain [dim]({svc} / {acct})[/dim]")
    else:
        console.print(f"[dim]No key to delete for {b} ({svc} / {acct})[/dim]")
        raise typer.Exit(1)


@app.command("search")
def search_cmd(
    query: Optional[str] = typer.Argument(None, help="Search query (FTS5)"),
    feed: Optional[str] = typer.Option(None, "--feed", help="Filter by feed slug"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    since: Optional[str] = typer.Option(None, "--since", help="Earliest date (ISO YYYY-MM-DD)"),
    until: Optional[str] = typer.Option(None, "--until", help="Latest date (ISO YYYY-MM-DD)"),
    reindex: bool = typer.Option(False, "--reindex", help="Rebuild search index from transcripts"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Override data directory"),
) -> None:
    """Search indexed transcripts (offline FTS5)."""
    settings = load_settings(data_dir=data_dir)
    db_path = settings.state_db_path()
    if reindex:
        ensure_data_dirs(settings)
        with Database(db_path) as db:
            count = db.reindex_search(settings.transcripts_dir())
        console.print(f"Reindexed {count} transcript(s)")
        if not query:
            return
    if not query:
        err_console.print("[red]Provide a search query or use --reindex[/red]")
        raise typer.Exit(1)
    if not db_path.exists():
        console.print("[dim]No results[/dim]")
        return
    with Database(db_path) as db:
        hits = db.search_transcripts(query, feed=feed, limit=limit, since=since, until=until)
    if not hits:
        console.print("[dim]No results[/dim]")
        return
    for h in hits:
        title = str(h.get("title") or "")
        feed_slug = str(h.get("feed_slug") or "")
        published_at = str(h.get("published_at") or "")
        date_str = published_at[:10] if published_at else ""
        snippet = str(h.get("snippet") or h.get("text") or "")[:400]
        txt_path = str(h.get("txt_path") or "")
        json_path = str(h.get("json_path") or "")
        header = f"[bold]{escape(title)}[/bold] [dim]({escape(feed_slug)})[/dim]"
        if date_str:
            header += f" [dim]{escape(date_str)}[/dim]"
        console.print(header)
        if snippet:
            console.print(escape(snippet))
        if txt_path:
            console.print(escape(txt_path))
        if json_path:
            console.print(escape(json_path))
        console.print("[dim]---[/dim]")


@app.command("rename")
def rename_cmd(
    from_title: bool = typer.Option(
        False,
        "--from-title",
        help="Infer episode numbers from JSON titles and rename sibling outputs",
    ),
    feed: Optional[str] = typer.Option(
        None,
        "--feed",
        help="Rename transcripts for a feed slug",
    ),
    all_feeds: bool = typer.Option(
        False,
        "--all",
        help="Rename transcripts across the whole library",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned renames without writing",
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Override data directory"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Rename transcript outputs to fix missing episode numbers (no ASR).

    Currently supports ``--from-title`` with ``--feed`` or ``--all``.
    """
    if not from_title:
        err_console.print("[red]Specify --from-title[/red] (only rename mode supported)")
        raise typer.Exit(1)

    if sum([feed is not None, all_feeds]) != 1:
        err_console.print("[red]Specify exactly one of:[/red] `--feed <slug>` or `--all`")
        raise typer.Exit(1)

    settings = load_settings(data_dir=data_dir)
    ensure_data_dirs(settings)
    transcripts_root = settings.transcripts_dir()
    try:
        targets = discover_transcript_jsons(
            transcripts_root,
            feed=None if all_feeds else feed,
        )
    except TranscriptJsonError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not targets:
        err_console.print("[dim]No transcript JSON files found.[/dim]")
        raise typer.Exit(1)

    if not quiet:
        scope = "all feeds" if all_feeds else f"feed {feed}"
        mode = "Dry-run rename" if dry_run else "Renaming"
        console.print(f"[bold]{mode} {len(targets)} transcript(s)[/bold] ({scope})")

    db = Database(settings.state_db_path())
    try:
        result = rename_many_from_title(targets, dry_run=dry_run, db=None if dry_run else db)
    finally:
        db.close()

    if not quiet:
        for old, new in result.renames:
            verb = "Would rename" if dry_run else "Renamed"
            console.print(f"[green]{verb}[/green] {old.name} → {new.name}")
        for path, message in result.skips:
            console.print(f"[dim]Skipped[/dim] {path.name}: {message}")
        for path, message in result.errors:
            err_console.print(f"[red]Failed[/red] {path.name}: {message}")
        prefix = "Dry-run done" if dry_run else "Done"
        console.print(
            f"[bold]{prefix}[/bold]: {result.ok} ok, "
            f"{result.skipped} skipped, {result.failed} failed"
        )

    if result.failed:
        raise typer.Exit(1)
