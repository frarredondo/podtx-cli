from __future__ import annotations

from pathlib import Path

from podtx.engines.registry import available_engines, get_engine
from podtx.models import Segment, Transcript


class FakeEngine:
    name = "fake"
    default_model = "fake-model"

    def transcribe(self, audio_path: Path, *, model: str | None = None, language: str = "en") -> Transcript:
        return Transcript(
            text="hello",
            segments=[Segment(0.0, 1.0, "hello")],
            language=language,
            model=model or self.default_model,
            engine=self.name,
        )


def test_registry_known_engines() -> None:
    assert "parakeet" in available_engines()
    assert "whisper" in available_engines()
    engine = get_engine("parakeet")
    assert engine.name == "parakeet"
    assert engine.default_model.startswith("mlx-community/")
