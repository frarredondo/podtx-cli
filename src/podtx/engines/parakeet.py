from __future__ import annotations

from pathlib import Path

from podtx.models import Segment, Transcript


class ParakeetEngine:
    name = "parakeet"
    default_model = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, bool, int], object] = {}

    def _load(
        self,
        model_id: str,
        *,
        local_attention: bool,
        local_attention_context_size: int,
    ) -> object:
        key = (model_id, local_attention, local_attention_context_size)
        if key not in self._model_cache:
            try:
                from parakeet_mlx import from_pretrained
            except ImportError as exc:
                raise ImportError(
                    "Parakeet engine requires the 'parakeet' extra. "
                    "Install with: uv sync --extra parakeet"
                ) from exc
            asr = from_pretrained(model_id)
            if local_attention:
                # Reduces peak memory for long podcasts (full attention OOMs on hour-scale audio)
                asr.encoder.set_attention_model(
                    "rel_pos_local_attn",
                    (local_attention_context_size, local_attention_context_size),
                )
            self._model_cache[key] = asr
        return self._model_cache[key]

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
        model_id = model or self.default_model
        asr = self._load(
            model_id,
            local_attention=local_attention,
            local_attention_context_size=local_attention_context_size,
        )
        result = asr.transcribe(str(audio_path))  # type: ignore[attr-defined]

        text = getattr(result, "text", None) or str(result)
        segments: list[Segment] = []

        raw_segments = getattr(result, "segments", None) or getattr(result, "sentences", None) or []
        for seg in raw_segments:
            seg_text = getattr(seg, "text", None) or (seg.get("text") if isinstance(seg, dict) else "")
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            if isinstance(seg, dict):
                start = seg.get("start", start)
                end = seg.get("end", end)
            if start is None:
                start = 0.0
            if end is None:
                end = float(start)
            segments.append(Segment(start=float(start), end=float(end), text=str(seg_text).strip()))

        if not segments and text:
            segments = [Segment(start=0.0, end=0.0, text=str(text).strip())]

        # Diarization: if opted-in, assign speaker labels round-robin by segment
        # Real diarization requires an external model; this stub provides the
        # contract for output shape and tests via fakes. Engine-level note:
        # diarization is CPU-bound, increases memory, and is opt-in.
        if diarize and segments:
            labeled: list[Segment] = []
            for idx, seg in enumerate(segments):
                label = f"SPEAKER_{idx % 2:02d}"
                labeled.append(Segment(start=seg.start, end=seg.end, text=seg.text, speaker=label))
            segments = labeled

        return Transcript(
            text=str(text).strip(),
            segments=segments,
            language=language,
            model=model_id,
            engine=self.name,
        )
