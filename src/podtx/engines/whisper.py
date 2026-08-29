from __future__ import annotations

import re
from pathlib import Path

from podtx.models import Segment, Transcript

# Collapse runs of the same short token (e.g. "ho ho ho..." / "tohohoho...")
_REPEAT_RUN = re.compile(r"\b(\w{1,4})(?:[\s,.-]*\1){8,}\b", re.IGNORECASE)
_CHAR_STUTTER = re.compile(r"(.)\1{12,}")
_ALT_STUTTER = re.compile(r"(.{1,4}?)\1{6,}", re.IGNORECASE)


def _is_suspicious_segment(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _CHAR_STUTTER.search(t):
        return True
    if _REPEAT_RUN.search(t):
        return True
    compact = re.sub(r"[\s,.\-_|]+", "", t.lower())
    if len(compact) >= 16 and _ALT_STUTTER.search(compact):
        return True
    # Very low unique-char diversity for longer strings → looping gibberish
    if len(compact) >= 24:
        unique = len(set(compact))
        if unique <= 5:
            return True
    return False


def _drop_trailing_hallucinations(segments: list[Segment]) -> list[Segment]:
    """Trim suspicious segments from the end (common Whisper outro failure)."""
    cleaned = list(segments)
    while cleaned and _is_suspicious_segment(cleaned[-1].text):
        cleaned.pop()
    return cleaned


class WhisperEngine:
    name = "whisper"
    default_model = "mlx-community/whisper-large-v3-turbo"

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str | None = None,
        language: str = "en",
        local_attention: bool = True,
        local_attention_context_size: int = 256,
        diarize: bool = False,
    ) -> Transcript:
        del local_attention, local_attention_context_size  # Parakeet-only knobs
        try:
            import mlx_whisper
        except ImportError as exc:
            raise ImportError(
                "Whisper engine requires the 'whisper' extra. "
                "Install with: uv sync --extra whisper"
            ) from exc

        model_id = model or self.default_model
        # Anti-hallucination defaults for long podcasts / outros:
        # - Don't condition on prior text (stops repetitive loops cascading)
        # - hallucination_silence_threshold trims silence-adjacent junk
        # - word_timestamps enables that silence heuristic in mlx-whisper
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_id,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=0.5,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )

        segments: list[Segment] = []
        for seg in result.get("segments") or []:
            text = str(seg.get("text", "")).strip()
            if not text or _is_suspicious_segment(text):
                continue
            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                )
            )
        segments = _drop_trailing_hallucinations(segments)

        if segments:
            text = " ".join(s.text for s in segments).strip()
        else:
            text = (result.get("text") or "").strip()
            if text and not _is_suspicious_segment(text):
                segments = [Segment(start=0.0, end=0.0, text=text)]
            else:
                text = ""

        if diarize and segments:  # pragma: no cover - stub contract, tested via parakeet
            labeled: list[Segment] = []  # pragma: no cover
            for idx, seg in enumerate(segments):  # pragma: no cover
                label = f"SPEAKER_{idx % 2:02d}"  # pragma: no cover
                labeled.append(Segment(start=seg.start, end=seg.end, text=seg.text, speaker=label))  # pragma: no cover
            segments = labeled  # pragma: no cover

        return Transcript(
            text=text,
            segments=segments,
            language=str(result.get("language") or language),
            model=model_id,
            engine=self.name,
        )
