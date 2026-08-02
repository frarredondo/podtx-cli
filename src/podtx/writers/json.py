from __future__ import annotations

import json
from pathlib import Path

from podtx.formatting import body_text, round_ts
from podtx.models import Episode, Transcript


def write_json(
    path: Path,
    episode: Episode,
    transcript: Transcript,
    *,
    readable: bool = False,
    cleanup: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body_text(
        transcript.text,
        transcript.segments,
        readable=readable,
        cleanup=cleanup,
    )
    payload = {
        "title": episode.title,
        "show": episode.show_title,
        "date": episode.published_at.isoformat() if episode.published_at else None,
        "episode": episode.episode_num,
        "guid": episode.guid,
        "source": episode.enclosure_url,
        "link": episode.link,
        "engine": transcript.engine,
        "model": transcript.model,
        "language": transcript.language,
        "readable": readable,
        "cleanup": cleanup,
        "text": text,
        "segments": [
            {
                "start": round_ts(s.start),
                "end": round_ts(s.end),
                "text": s.text,
            }
            for s in transcript.segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
