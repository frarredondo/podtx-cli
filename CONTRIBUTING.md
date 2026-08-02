# Contributing

## Filing issues

Please use an [issue template](https://github.com/frarredondo/podtx-cli/issues/new/choose).

- **Bugs:** include the exact command, expected vs actual output, and environment.
- **Features / enhancements:** state the problem first, then a concrete proposal and acceptance criteria. Include examples from real usage when you can.
- Prefer one concern per issue. Split unrelated ideas (e.g. search + summarize + diarization) into separate issues.
- Search existing issues before opening a new one.

## Development

```bash
uv sync --extra all --extra dev
uv run pytest
```
