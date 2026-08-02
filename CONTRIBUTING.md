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

### CI

Pull requests and pushes to `main` run the **`test`** GitHub Actions job (`.github/workflows/ci.yml`):

1. `uv sync --extra dev`
2. `uv run pytest` with line + branch coverage (`pytest-cov`)
3. Statement-coverage ratchet (`scripts/check_coverage_ratchet.py`, currently **≥ 65%**)
4. Upload `coverage.xml` to Codecov (badge in README)

ML extras (`parakeet` / `whisper`) are not installed in CI; unit tests do not require them. Branch coverage is reported to Codecov but is not gated yet.

Set repository secret `CODECOV_TOKEN` (from [codecov.io](https://codecov.io)) so uploads succeed on the protected `main` branch.

Once the workflow has run at least once on `main`, you can mark **`test`** as a required status check on the Protect main ruleset.
