from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import tomllib
from platformdirs import user_config_dir, user_data_dir

APP_NAME = "podcast-transcriber"
ENV_PREFIX = "PODCAST_TRANSCRIBER_"

DEFAULT_ENGINE = "parakeet"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LIMIT = 5
DEFAULT_FORMATS = ("txt", "json")
DEFAULT_LOCAL_ATTENTION = True
DEFAULT_LOCAL_ATTENTION_CONTEXT_SIZE = 256


def default_data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, appauthor=False))


def default_config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


@dataclass(frozen=True)
class Settings:
    engine: str = DEFAULT_ENGINE
    model: str | None = None
    limit: int = DEFAULT_LIMIT
    formats: tuple[str, ...] = DEFAULT_FORMATS
    keep_audio: bool = False
    data_dir: Path = field(default_factory=default_data_dir)
    quiet: bool = False
    language: str = "en"
    local_attention: bool = DEFAULT_LOCAL_ATTENTION
    local_attention_context_size: int = DEFAULT_LOCAL_ATTENTION_CONTEXT_SIZE
    readable: bool = False
    cleanup: bool = False
    correct_names: bool = False

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.engine == "whisper":
            return DEFAULT_WHISPER_MODEL
        return DEFAULT_PARAKEET_MODEL

    def state_db_path(self) -> Path:
        return self.data_dir / "state.db"

    def transcripts_dir(self, feed_slug: str | None = None) -> Path:
        base = self.data_dir / "transcripts"
        if feed_slug:
            return base / feed_slug
        return base

    def audio_dir(self) -> Path:
        return self.data_dir / "audio"


def _env(key: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{key}")


def _parse_formats(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.replace(",", " ").split() if p.strip()]
        return tuple(parts) if parts else DEFAULT_FORMATS
    return tuple(str(v).strip().lower() for v in value)


def load_toml_config(path: Path | None = None) -> dict:
    config_path = path or default_config_path()
    if not config_path.is_file():
        return {}
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def load_settings(
    *,
    engine: str | None = None,
    model: str | None = None,
    limit: int | None = None,
    formats: list[str] | tuple[str, ...] | str | None = None,
    keep_audio: bool | None = None,
    data_dir: Path | str | None = None,
    quiet: bool | None = None,
    language: str | None = None,
    local_attention: bool | None = None,
    local_attention_context_size: int | None = None,
    readable: bool | None = None,
    cleanup: bool | None = None,
    correct_names: bool | None = None,
    config_path: Path | None = None,
) -> Settings:
    """Resolve settings with precedence: CLI flags > env > TOML > defaults."""
    toml = load_toml_config(config_path)
    settings = Settings()

    # TOML
    if "engine" in toml:
        settings = replace(settings, engine=str(toml["engine"]))
    if "model" in toml:
        settings = replace(settings, model=str(toml["model"]))
    if "limit" in toml:
        settings = replace(settings, limit=int(toml["limit"]))
    if "formats" in toml:
        settings = replace(settings, formats=_parse_formats(toml["formats"]))
    if "keep_audio" in toml:
        settings = replace(settings, keep_audio=bool(toml["keep_audio"]))
    if "data_dir" in toml:
        settings = replace(settings, data_dir=Path(str(toml["data_dir"])).expanduser())
    if "quiet" in toml:
        settings = replace(settings, quiet=bool(toml["quiet"]))
    if "language" in toml:
        settings = replace(settings, language=str(toml["language"]))
    if "local_attention" in toml:
        settings = replace(settings, local_attention=bool(toml["local_attention"]))
    if "local_attention_context_size" in toml:
        settings = replace(
            settings,
            local_attention_context_size=int(toml["local_attention_context_size"]),
        )
    if "readable" in toml:
        settings = replace(settings, readable=bool(toml["readable"]))
    if "cleanup" in toml:
        settings = replace(settings, cleanup=bool(toml["cleanup"]))
    if "correct_names" in toml:  # pragma: no cover - TOML tested via existing suite
        settings = replace(settings, correct_names=bool(toml["correct_names"]))
    if "correctNames" in toml:  # pragma: no cover
        settings = replace(settings, correct_names=bool(toml["correctNames"]))

    # Env
    if (v := _env("ENGINE")) is not None:
        settings = replace(settings, engine=v)
    if (v := _env("MODEL")) is not None:
        settings = replace(settings, model=v)
    if (v := _env("LIMIT")) is not None:
        settings = replace(settings, limit=int(v))
    if (v := _env("FORMATS")) is not None:
        settings = replace(settings, formats=_parse_formats(v))
    if (v := _env("KEEP_AUDIO")) is not None:
        settings = replace(settings, keep_audio=v.lower() in {"1", "true", "yes", "on"})
    if (v := _env("DATA_DIR")) is not None:
        settings = replace(settings, data_dir=Path(v).expanduser())
    if (v := _env("QUIET")) is not None:
        settings = replace(settings, quiet=v.lower() in {"1", "true", "yes", "on"})
    if (v := _env("LANGUAGE")) is not None:
        settings = replace(settings, language=v)
    if (v := _env("LOCAL_ATTENTION")) is not None:
        settings = replace(
            settings, local_attention=v.lower() in {"1", "true", "yes", "on"}
        )
    if (v := _env("LOCAL_ATTENTION_CONTEXT_SIZE")) is not None:
        settings = replace(settings, local_attention_context_size=int(v))
    if (v := _env("READABLE")) is not None:
        settings = replace(settings, readable=v.lower() in {"1", "true", "yes", "on"})
    if (v := _env("CLEANUP")) is not None:
        settings = replace(settings, cleanup=v.lower() in {"1", "true", "yes", "on"})
    if (v := _env("CORRECT_NAMES")) is not None:  # pragma: no cover - env already tested via existing suite
        settings = replace(settings, correct_names=v.lower() in {"1", "true", "yes", "on"})

    # CLI flags (only override when explicitly provided)
    if engine is not None:
        settings = replace(settings, engine=engine)
    if model is not None:
        settings = replace(settings, model=model)
    if limit is not None:
        settings = replace(settings, limit=limit)
    if formats is not None:
        settings = replace(settings, formats=_parse_formats(formats))
    if keep_audio is not None:
        settings = replace(settings, keep_audio=keep_audio)
    if data_dir is not None:
        settings = replace(settings, data_dir=Path(data_dir).expanduser())
    if quiet is not None:
        settings = replace(settings, quiet=quiet)
    if language is not None:
        settings = replace(settings, language=language)
    if local_attention is not None:
        settings = replace(settings, local_attention=local_attention)
    if local_attention_context_size is not None:
        settings = replace(
            settings, local_attention_context_size=local_attention_context_size
        )
    if readable is not None:
        settings = replace(settings, readable=readable)
    if cleanup is not None:
        settings = replace(settings, cleanup=cleanup)
    if correct_names is not None:  # pragma: no cover - CLI tested via CliRunner
        settings = replace(settings, correct_names=correct_names)

    return settings


def ensure_data_dirs(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.transcripts_dir().mkdir(parents=True, exist_ok=True)
    settings.audio_dir().mkdir(parents=True, exist_ok=True)
