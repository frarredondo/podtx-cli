from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment]
    language: str
    model: str
    engine: str


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    enclosure_url: str
    published_at: datetime | None = None
    episode_num: int | None = None
    description: str | None = None
    link: str | None = None
    show_title: str | None = None


@dataclass(frozen=True)
class Feed:
    id: int
    url: str
    slug: str
    title: str
    created_at: datetime


@dataclass
class EpisodeResult:
    episode: Episode
    transcript: Transcript
    output_paths: list[Path] = field(default_factory=list)
