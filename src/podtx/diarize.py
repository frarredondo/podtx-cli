from __future__ import annotations

import os
from pathlib import Path

import httpx

from podtx.config import Settings
from podtx.keychain import get_api_key
from podtx.models import Segment, Transcript

_DIARIZE_BACKENDS = {"fake", "pyannote", "hf", "assemblyai", "deepgram"}
_ALIAS = {"local": "pyannote"}

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_HF_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_HF_BASE_URL = "https://api-inference.huggingface.co"
DEFAULT_ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"
DEFAULT_DEEPGRAM_BASE_URL = "https://api.deepgram.com"
DEFAULT_DIARIZE_TIMEOUT = 120.0


class DiarizeError(ValueError):
    """Raised for diarization backend failures."""


def _normalize_backend(backend: str) -> str:
    b = backend.lower().strip()
    return _ALIAS.get(b, b)


def _default_model(backend: str) -> str | None:
    b = _normalize_backend(backend)
    if b == "pyannote":
        return DEFAULT_PYANNOTE_MODEL
    if b == "hf":
        return DEFAULT_HF_MODEL
    if b == "assemblyai":
        return "assemblyai_default"
    if b == "deepgram":
        return "nova-2"
    return None


def _default_base_url(backend: str) -> str | None:
    b = _normalize_backend(backend)
    if b == "hf":
        return DEFAULT_HF_BASE_URL
    if b == "assemblyai":
        return DEFAULT_ASSEMBLYAI_BASE_URL
    if b == "deepgram":
        return DEFAULT_DEEPGRAM_BASE_URL
    return None


def _resolve_api_key(
    backend: str,
    api_key: str | None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> str | None:
    # CLI direct
    if api_key:
        return api_key
    # settings (from config)
    if settings_api_key:
        return settings_api_key
    # env aliases
    b = _normalize_backend(backend)
    # provider specific envs
    if b == "hf" and (v := os.environ.get("HF_TOKEN")) is not None:
        return v
    if b == "hf" and (v := os.environ.get("HUGGINGFACE_API_KEY")) is not None:
        return v
    if b == "assemblyai" and (v := os.environ.get("ASSEMBLYAI_API_KEY")) is not None:
        return v
    if b == "deepgram" and (v := os.environ.get("DEEPGRAM_API_KEY")) is not None:
        return v
    # generic
    if (v := os.environ.get("DIARIZE_API_KEY")) is not None:
        return v
    if (v := os.environ.get("PODCAST_TRANSCRIBER_DIARIZE_API_KEY")) is not None:
        return v
    # keychain fallback if service/account provided (mirrors summarize)
    if service and account:
        try:
            if (val := get_api_key(service, account)) is not None:
                return val
        except Exception:
            return None
    # also try default service names
    # podtx-hf / podtx-assemblyai / podtx-deepgram
    default_service = f"podtx-{b}"
    default_account = "api-key"
    # Try both explicit and default keychain entries
    # Only try default if not already tried with same values
    if not (service == default_service and account == default_account):
        try:
            if (val := get_api_key(default_service, default_account)) is not None:
                return val
        except Exception:
            return None
    # also try via Settings-like env? Already covered
    # try provider fallback for hf: also check generic HF_TOKEN already done
    # For pyannote local, HF_TOKEN may be needed for model download; allow same env
    if b == "pyannote" and (v := os.environ.get("HF_TOKEN")) is not None:
        return v
    return None


def align_segments(
    transcript_segments: list[Segment],
    diarization: list[tuple[float, float, str]],
) -> list[Segment]:
    """Assign speaker to each transcript segment based on max overlap with diarization turns.

    diarization: list of (start, end, speaker) where speaker already normalized like SPEAKER_00.
    Returns new list of Segments with speaker assigned (or None if no overlap).
    """
    if not diarization:
        return [Segment(s.start, s.end, s.text, speaker=None) for s in transcript_segments]
    # Normalize diarization speaker labels already like SPEAKER_00, but ensure format
    out: list[Segment] = []
    for seg in transcript_segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for d_start, d_end, d_speaker in diarization:
            # compute overlap
            overlap_start = max(seg.start, d_start)
            overlap_end = min(seg.end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)
            # handle zero-length segments (start==end): treat as point at start
            if seg.start == seg.end:
                # if diarization contains that point
                if d_start <= seg.start < d_end or d_start < seg.end <= d_end:
                    overlap = 1.0  # treat as overlap
                else:
                    overlap = 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_speaker
            # tie: keep first encountered (stable)
        out.append(Segment(start=seg.start, end=seg.end, text=seg.text, speaker=best_speaker))
    return out


def _load_pyannote_pipeline(model: str):
    """Load pyannote pipeline. Separated for test mocking."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizeError("pyannote backend requires pyannote.audio (install with: uv sync --extra pyannote)") from exc
    # Pipeline.from_pretrained handles HF_TOKEN via env
    try:
        pipeline = Pipeline.from_pretrained(model)
    except Exception as exc:
        raise DiarizeError(f"Failed to load pyannote pipeline {model}: {exc}") from exc
    return pipeline


def _call_pyannote(audio_path: Path, model: str, timeout: float) -> list[tuple[float, float, str]]:
    """Run local pyannote diarization, return list of (start, end, speaker)."""
    try:
        pipeline = _load_pyannote_pipeline(model)
    except ImportError as exc:
        raise DiarizeError(f"pyannote diarization requires pyannote.audio: {exc}") from exc
    except DiarizeError:
        raise
    # pyannote pipeline is callable with audio file
    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:
        raise DiarizeError(f"pyannote diarization failed: {exc}") from exc
    turns: list[tuple[float, float, str]] = []
    # diarization may be an Annotation object with .itertracks etc., or already list
    # Handle both: if it has itertracks, iterate; else assume iterable of dicts
    try:
        # pyannote Annotation
        if hasattr(diarization, "itertracks"):
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(speaker)))
            # Normalize speaker labels to SPEAKER_XX
            # pyannote returns SPEAKER_00 already, but ensure
            # If labels are arbitrary, map to SPEAKER_00,01...
            uniq = sorted(set(s for _, _, s in turns))
            mapping = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(uniq)}
            turns = [(s, e, mapping[spk]) for s, e, spk in turns]
        elif isinstance(diarization, list):
            for item in diarization:
                if isinstance(item, dict):
                    turns.append((float(item["start"]), float(item["end"]), str(item["speaker"])))
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    turns.append((float(item[0]), float(item[1]), str(item[2])))
                else:
                    continue
        else:
            # unknown format
            raise DiarizeError(f"Unexpected pyannote output type: {type(diarization)}")
    except DiarizeError:
        raise
    except Exception as exc:
        raise DiarizeError(f"Failed to parse pyannote output: {exc}") from exc
    return turns


def _call_hf(audio_path: Path, model: str, api_key: str, base_url: str, timeout: float) -> list[tuple[float, float, str]]:
    """Call HuggingFace Inference API for diarization."""
    if not api_key:
        raise DiarizeError("HF diarization requires an API key (HF_TOKEN env, --diarize-api-key, or Keychain)")
    if not base_url:
        base_url = DEFAULT_HF_BASE_URL
    url = f"{base_url.rstrip('/')}/models/{model}"
    # Read audio bytes (simple, for test we mock)
    try:
        data = audio_path.read_bytes() if audio_path.is_file() else b"fake-audio"
    except Exception:
        data = b"fake-audio"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.post(url, headers=headers, content=data, timeout=timeout)
    except Exception as exc:
        raise DiarizeError(f"HF diarization request failed: {exc}") from exc
    # Handle 401 etc.
    if resp.status_code == 401:
        raise DiarizeError("HF diarization failed (401): Invalid API key (check HF_TOKEN)")
    if resp.status_code == 404:
        raise DiarizeError(f"HF diarization failed (404): Model not found {model}")
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Try to extract body
        body = getattr(resp, "text", "") or str(exc)
        raise DiarizeError(f"HF diarization failed ({resp.status_code}): {body[:500]}") from exc
    try:
        payload = resp.json()
    except Exception as exc:
        raise DiarizeError(f"HF diarization returned invalid JSON: {resp.text[:500]}") from exc
    # Payload expected: list of {start, end, speaker} or {start, end, label}
    turns: list[tuple[float, float, str]] = []
    if isinstance(payload, dict) and "diarization" in payload:
        payload = payload["diarization"]
    if not isinstance(payload, list):
        raise DiarizeError(f"HF diarization unexpected response format: {type(payload)}")
    for item in payload:
        if not isinstance(item, dict):
            continue
        s = item.get("start")
        e = item.get("end")
        spk = item.get("speaker") or item.get("label") or item.get("speaker_label")
        if s is None or e is None or spk is None:
            continue
        turns.append((float(s), float(e), str(spk)))
    # Normalize speakers
    if turns:
        uniq = sorted(set(s for _, _, s in turns))
        mapping = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(uniq)}
        turns = [(s, e, mapping[spk]) for s, e, spk in turns]
    return turns


def _call_assemblyai(audio_path: Path, api_key: str, base_url: str, timeout: float) -> list[tuple[float, float, str]]:
    """Stub for AssemblyAI — mocked in tests; real would upload + poll."""
    if not api_key:
        raise DiarizeError("AssemblyAI diarization requires an API key (ASSEMBLYAI_API_KEY)")
    # For TDD, we just raise not implemented unless mocked; tests will patch this
    raise DiarizeError("AssemblyAI backend not fully implemented — use hf or pyannote")


def diarize_transcript(
    transcript: Transcript,
    audio_path: Path,
    backend: str = "fake",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    settings_api_key: str | None = None,
    service: str | None = None,
    account: str | None = None,
) -> Transcript:
    """Assign speaker labels to transcript segments via selected backend."""
    if not transcript.segments:
        return transcript
    b = _normalize_backend(backend)
    if b not in _DIARIZE_BACKENDS:
        raise DiarizeError(f"Unknown backend: {backend} (choose from: {', '.join(sorted(_DIARIZE_BACKENDS))})")
    # Resolve model/base_url
    resolved_model = model or _default_model(b)
    resolved_base = base_url or _default_base_url(b)
    resolved_timeout = timeout if timeout is not None else DEFAULT_DIARIZE_TIMEOUT
    # Resolve api key if needed for cloud backends
    needs_key = b in {"hf", "assemblyai", "deepgram"}
    resolved_key: str | None = None
    if needs_key:
        resolved_key = _resolve_api_key(b, api_key, settings_api_key, service, account)
        if not resolved_key:
            raise DiarizeError(f"Diarization backend '{b}' requires an API key (set via --diarize-api-key, env HF_TOKEN/ASSEMBLYAI_API_KEY, or Keychain)")

    # Fake: round-robin
    if b == "fake":
        labeled: list[Segment] = []
        for idx, seg in enumerate(transcript.segments):
            label = f"SPEAKER_{idx % 2:02d}"
            labeled.append(Segment(start=seg.start, end=seg.end, text=seg.text, speaker=label))
        return Transcript(
            text=transcript.text,
            segments=labeled,
            language=transcript.language,
            model=transcript.model,
            engine=transcript.engine,
        )

    # Real backends: get diarization turns then align
    turns: list[tuple[float, float, str]] = []
    if b == "pyannote":
        if not resolved_model:
            raise DiarizeError("pyannote backend requires a model (e.g. pyannote/speaker-diarization-3.1)")
        turns = _call_pyannote(audio_path, resolved_model, resolved_timeout)
    elif b == "hf":
        assert resolved_key is not None
        assert resolved_model is not None
        assert resolved_base is not None
        turns = _call_hf(audio_path, resolved_model, resolved_key, resolved_base, resolved_timeout)
    elif b == "assemblyai":
        assert resolved_key is not None
        assert resolved_base is not None
        turns = _call_assemblyai(audio_path, resolved_key, resolved_base, resolved_timeout)
    elif b == "deepgram":
        raise DiarizeError("deepgram backend not yet implemented — use hf or pyannote")
    else:
        raise DiarizeError(f"Unknown backend {b}")

    aligned = align_segments(transcript.segments, turns)
    return Transcript(
        text=transcript.text,
        segments=aligned,
        language=transcript.language,
        model=transcript.model,
        engine=transcript.engine,
    )
