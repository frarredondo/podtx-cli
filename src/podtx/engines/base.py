from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from podtx.models import Transcript


@runtime_checkable
class TranscriptionEngine(Protocol):
    name: str
    default_model: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        language: str = "en",
        local_attention: bool = True,
        local_attention_context_size: int = 256,
    ) -> Transcript:
        """Transcribe audio and return a normalized Transcript."""
