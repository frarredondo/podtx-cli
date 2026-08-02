from __future__ import annotations

from podtx.engines.base import TranscriptionEngine
from podtx.engines.parakeet import ParakeetEngine
from podtx.engines.whisper import WhisperEngine

_REGISTRY: dict[str, type] = {
    "parakeet": ParakeetEngine,
    "whisper": WhisperEngine,
}


def available_engines() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_engine(name: str) -> TranscriptionEngine:
    key = name.strip().lower()
    if key not in _REGISTRY:
        known = ", ".join(available_engines())
        raise ValueError(f"Unknown engine {name!r}. Available: {known}")
    return _REGISTRY[key]()
