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

Pull requests and pushes to `main` run two GitHub Actions jobs (`.github/workflows/ci.yml`):

1. **`test`** — `uv sync --extra dev`, then `pytest` with line + branch coverage; upload `coverage.xml` to Codecov
2. **`coverage-ratchet`** — project coverage floors for package `podtx` (whole package on the branch tip, **not** Codecov patch/diff coverage):
   - Statements ≥ **`COVERAGE_RATCHET_MIN`** (default **65**)
   - Branches ≥ **`COVERAGE_RATCHET_MIN_BRANCHES`** (default **45**)
   - Combined is reported only (not gated)
   - On pull requests, posts/updates a sticky comment with the summary

ML extras (`parakeet` / `whisper`) are not installed in CI; unit tests do not require them.

Raise floors over time under *Settings → Secrets and variables → Actions → Variables* (no code change required). Locally:

```bash
COVERAGE_RATCHET_MIN=65 COVERAGE_RATCHET_MIN_BRANCHES=45 \
  uv run python scripts/check_coverage_ratchet.py
```

Repository secret `CODECOV_TOKEN` is required for Codecov uploads on protected branches.

Once the workflow has run at least once on `main`, mark **`test`** and **`coverage-ratchet`** as required status checks on the Protect main ruleset.
