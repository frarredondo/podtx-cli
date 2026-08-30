from __future__ import annotations

import json
from pathlib import Path

from podtx.formatting import body_text, body_text_with_report, round_ts
from podtx.models import Episode, Transcript


def write_json(
    path: Path,
    episode: Episode,
    transcript: Transcript,
    *,
    readable: bool = False,
    cleanup: bool = False,
    correct_names: bool = False,
    diarize: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if correct_names:
        payload_text, subs = body_text_with_report(
            transcript.text,
            transcript.segments,
            readable=readable,
            cleanup=cleanup,
            correct_names=True,
            episode=episode,
        )
        payload_subs: list[list[str]] = [[a, b] for a, b in subs]
    else:
        payload_text = body_text(
            transcript.text,
            transcript.segments,
            readable=readable,
            cleanup=cleanup,
            correct_names=False,
            episode=episode,
            diarize=diarize,
        )
        payload_subs = []
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
        "diarize": diarize,
        "text": payload_text,
        "segments": [
            {
                "start": round_ts(s.start),
                "end": round_ts(s.end),
                "text": s.text,
                **({"speaker": s.speaker} if s.speaker else {}),
            }
            for s in transcript.segments
        ],
    }
    if correct_names:
        payload["correct_names"] = True
        payload["corrections"] = payload_subs
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
