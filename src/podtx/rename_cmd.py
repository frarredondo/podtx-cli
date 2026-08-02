from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from podtx.db import Database
from podtx.format_cmd import TranscriptJsonError, load_transcript_json
from podtx.models import Episode
from podtx.naming import parse_episode_number_from_title, transcript_basename

# Sibling outputs that share a transcript basename.
_SIBLING_EXTS = (".json", ".txt", ".srt", ".vtt")


class RenameError(ValueError):
    pass


@dataclass
class RenameAction:
    old_json: Path
    new_basename: str
    episode_num: int
    moves: list[tuple[Path, Path]]


@dataclass
class BatchRenameResult:
    ok: int = 0
    skipped: int = 0
    failed: int = 0
    renames: list[tuple[Path, Path]] = field(default_factory=list)
    skips: list[tuple[Path, str]] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def resolve_episode_number_for_rename(episode: Episode) -> int | None:
    """Prefer a positive JSON/RSS episode number; else parse a clear title number."""
    if episode.episode_num is not None and episode.episode_num > 0:
        return episode.episode_num
    parsed = parse_episode_number_from_title(episode.title)
    if parsed is not None and parsed > 0:
        return parsed
    return None


def plan_rename_from_title(json_path: Path) -> RenameAction | None:
    """Build a rename plan from JSON title/episode metadata.

    Returns ``None`` when no rename is needed (already correct, or no clear number).
    Raises ``RenameError`` if a target basename already exists.
    """
    path = json_path.expanduser()
    episode, _ = load_transcript_json(path)
    num = resolve_episode_number_for_rename(episode)
    if num is None:
        return None

    updated = replace(episode, episode_num=num)
    new_basename = transcript_basename(updated)
    old_basename = path.stem
    if new_basename == old_basename:
        return None

    parent = path.parent
    moves: list[tuple[Path, Path]] = []
    for ext in _SIBLING_EXTS:
        src = parent / f"{old_basename}{ext}"
        if not src.is_file():
            continue
        dest = parent / f"{new_basename}{ext}"
        if dest.exists():
            raise RenameError(f"Target already exists: {dest}")
        moves.append((src, dest))

    if not moves:
        return None

    return RenameAction(
        old_json=path,
        new_basename=new_basename,
        episode_num=num,
        moves=moves,
    )


def apply_rename(
    action: RenameAction,
    *,
    dry_run: bool = False,
    db: Database | None = None,
) -> Path:
    """Apply a rename plan. Returns the new JSON path (even on dry-run)."""
    new_json = action.old_json.parent / f"{action.new_basename}.json"
    if dry_run:
        return new_json

    for src, dest in action.moves:
        src.rename(dest)

    payload = json.loads(new_json.read_text(encoding="utf-8"))
    payload["episode"] = action.episode_num
    new_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if db is not None:
        guid = str(payload.get("guid") or "")
        feed = db.get_feed(new_json.parent.name)
        if feed is not None and guid:
            db.update_episode_paths(
                feed_id=feed.id,
                guid=guid,
                episode_num=action.episode_num,
                output_paths=[dest for _, dest in action.moves],
            )

    return new_json


def rename_many_from_title(
    json_paths: list[Path],
    *,
    dry_run: bool = False,
    db: Database | None = None,
) -> BatchRenameResult:
    """Rename many transcript JSON trees; continue on per-file skip/error."""
    result = BatchRenameResult()
    for path in json_paths:
        try:
            plan = plan_rename_from_title(path)
        except (TranscriptJsonError, RenameError, OSError, ValueError) as exc:
            result.failed += 1
            result.errors.append((path, str(exc)))
            continue

        if plan is None:
            result.skipped += 1
            result.skips.append((path, "no rename needed"))
            continue

        try:
            new_json = apply_rename(plan, dry_run=dry_run, db=db)
        except (OSError, ValueError) as exc:
            result.failed += 1
            result.errors.append((path, str(exc)))
            continue

        result.ok += 1
        result.renames.append((path, new_json))
    return result
