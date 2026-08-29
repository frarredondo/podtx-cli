# podtx-cli

[![CI](https://github.com/frarredondo/podtx-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/frarredondo/podtx-cli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/frarredondo/podtx-cli/graph/badge.svg)](https://codecov.io/gh/frarredondo/podtx-cli)

CLI (`podtx`) to pull podcast episodes from an RSS feed and transcribe them locally on Apple Silicon.

Default engine: **NVIDIA Parakeet TDT v3** via [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx).  
Optional: **OpenAI Whisper** via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples).

## Requirements

- macOS with Apple Silicon
- Python 3.11+ (managed by [uv](https://docs.astral.sh/uv/))
- [ffmpeg](https://ffmpeg.org/): `brew install ffmpeg`

## Install

```bash
uv sync --extra all
```

Or install a single engine:

```bash
uv sync --extra parakeet   # default engine
uv sync --extra whisper
```

Activate / run:

```bash
uv run podtx --help
```

## Quick start

```bash
# Register a feed and sync the latest 5 episodes (default)
podtx add https://example.com/podcast.rss
podtx sync

# One-shot: latest episode from a feed → files in the current directory
podtx transcribe https://example.com/podcast.rss

# Local file or direct audio URL
podtx transcribe ./episode.mp3
podtx transcribe https://example.com/audio/ep01.mp3 --engine whisper
```

## Commands

| Command | Description |
|---------|-------------|
| `podtx add <rss-url>` | Register a feed |
| `podtx remove <feed>` | Unregister by slug or URL |
| `podtx feeds` | List registered feeds |
| `podtx show <feed>` | Episode status for a feed |
| `podtx doctor` | Library health: failed / stuck episodes, empty feeds, missing outputs |
| `podtx sync [feed]` | Transcribe new episodes |
| `podtx transcribe <target>` | One-shot RSS / URL / file |
| `podtx format <json\|--feed\|--all>` | Re-format existing transcript JSON (no ASR) |
| `podtx rename --from-title --feed\|--all` | Fix `_000_` filenames from title episode numbers |
| `podtx search <query> [--feed] [--limit] [--since] [--until] [--reindex]` | Offline FTS5 search over transcripts |

### Useful flags

- `--engine parakeet|whisper` — ASR backend (default: `parakeet`)
- `--model <hf-repo>` — override model id
- `--limit N` / `--all` — how many pending episodes (`sync` default limit: 5)
- `--format srt` / `--format vtt` — add subtitle outputs (`.txt` + `.json` always by default)
- `--keep-audio` — retain downloaded audio
- `--local-attention` / `--full-attention` — Parakeet attention mode (default: local; required for long episodes)
- `--local-attention-context-size N` — local attention window (default: 256)
- `--readable` — paragraph breaks for human reading: silence gaps, then sentence
  boundaries after ~20s, with a max ~45s / ~120 words so long Parakeet runs
  don’t stay as one wall of text (default is raw continuous text; JSON timestamps
  are always rounded)
- `--cleanup` — strip `uh`/`um` and collapse consecutive word/phrase doubles
  (1–4 words, e.g. `the the`, `I think I think`) in text outputs (JSON segments stay raw)
- `--out-dir` / `--data-dir` — override output or app data paths
- `--quiet` — less terminal noise

Re-format an existing transcript without re-running ASR:

```bash
podtx format path/to/episode.json --readable --cleanup
podtx format --feed corecursive-coding-stories --readable --cleanup
podtx format --all --readable --cleanup
```

Rename already-transcribed files whose episode number was missing (`_000_`)
when the title embeds a clear number (same rules as filename inference above):

```bash
podtx rename --from-title --feed syntax-tasty-web-development-treats --dry-run
podtx rename --from-title --feed syntax-tasty-web-development-treats
podtx rename --from-title --all
```

Check library health — failed or still-pending episodes, feeds with no recorded
episodes, and done episodes whose transcript files no longer exist (read-only):

```bash
podtx doctor
```

## Data & config

- Data (SQLite state, transcripts, temp audio): `~/.local/share/podcast-transcriber/`
- Optional config: `~/.config/podcast-transcriber/config.toml`
- Transcript filenames: `{YYYY-MM-DD}_{episode:03d}_{slug}`. Episode comes from
  RSS `itunes:episode` when present; otherwise a clear leading number in the title
  is used (`860 - …`, `#860 …`, `Episode 860: …`). Section-style ids like `1.1 - …`
  are ignored (fallback `000`).

```toml
engine = "parakeet"
# model = "mlx-community/parakeet-tdt-0.6b-v3"
limit = 5
formats = ["txt", "json"]
keep_audio = false
local_attention = true
# local_attention_context_size = 256
# readable = true
# cleanup = true
```

Precedence: **CLI flags > environment (`PODCAST_TRANSCRIBER_*`) > config.toml > defaults**.

## Engines

| Engine | Default model | Extra |
|--------|---------------|-------|
| `parakeet` | `mlx-community/parakeet-tdt-0.6b-v3` | `--extra parakeet` |
| `whisper` | `mlx-community/whisper-large-v3-turbo` | `--extra whisper` |

New engines: implement `TranscriptionEngine` in `src/podtx/engines/` and register in `registry.py`.

## Model licenses

- **Parakeet TDT v3** weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribution required). See [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
- This project’s code is [GPL-3.0-only](https://www.gnu.org/licenses/gpl-3.0.html) (see `LICENSE`).

## Development

```bash
uv sync --extra all --extra dev
uv run pytest
```

CI runs the unit suite with coverage on pull requests via jobs **`test`** and **`coverage-ratchet`** (see [CONTRIBUTING.md](CONTRIBUTING.md)). Project statement/branch floors come from repo variables **`COVERAGE_RATCHET_MIN`** (default 65%) and **`COVERAGE_RATCHET_MIN_BRANCHES`** (default 45%); combined coverage is informational. PRs also get a sticky coverage comment (whole package, not patch).

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to file bugs and feature requests.
