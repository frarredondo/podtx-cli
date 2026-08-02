#!/usr/bin/env python3
"""Fail if project coverage drops below configured ratchet floors.

Gates:
  - statement (line) coverage vs ``COVERAGE_RATCHET_MIN`` (default 65)
  - branch coverage vs ``COVERAGE_RATCHET_MIN_BRANCHES`` (default 45)

Combined coverage is reported but not gated.

Metrics are **project** coverage for package ``podtx`` on the tested commit
(whole package), not Codecov patch / diff coverage.

Also writes ``coverage-ratchet.md`` for the CI sticky PR comment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_DEFAULT_STATEMENT_MIN = 65.0
_DEFAULT_BRANCH_MIN = 45.0
_MARKDOWN_PATH = Path("coverage-ratchet.md")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(
            f"error: {name} must be a number, got {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def _summary_line(
    *,
    statements: float,
    statement_floor: float,
    branches: float,
    branch_floor: float,
    combined: float,
) -> str:
    return (
        f"statements={statements:.2f}% (min {statement_floor:.0f}%) | "
        f"branches={branches:.2f}% (min {branch_floor:.0f}%) | "
        f"combined={combined:.2f}% (informational)"
    )


def write_markdown_report(
    path: Path,
    *,
    statements: float,
    statement_floor: float,
    branches: float,
    branch_floor: float,
    combined: float,
    passed: bool,
) -> None:
    """Write sticky-comment markdown for project (whole-package) coverage."""
    line = _summary_line(
        statements=statements,
        statement_floor=statement_floor,
        branches=branches,
        branch_floor=branch_floor,
        combined=combined,
    )
    status = "passed" if passed else "failed"
    path.write_text(
        "\n".join(
            [
                "## Coverage ratchet (project)",
                "",
                "Whole package (`podtx`) on this PR branch — "
                "**not** Codecov patch / diff coverage.",
                "",
                f"`{line}`",
                "",
                "| Metric | Value | Role |",
                "|--------|------:|------|",
                f"| Statements | {statements:.2f}% | Gated (min {statement_floor:.0f}%) |",
                f"| Branches | {branches:.2f}% | Gated (min {branch_floor:.0f}%) |",
                f"| Combined | {combined:.2f}% | Informational |",
                "",
                f"**Status:** {status}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def evaluate(
    totals: dict[str, object],
    *,
    statement_floor: float,
    branch_floor: float,
) -> tuple[bool, str, list[str]]:
    """Return (passed, summary_line, error_messages)."""
    statements = float(totals["percent_statements_covered"])  # type: ignore[arg-type]
    branches = float(totals.get("percent_branches_covered") or 0.0)  # type: ignore[arg-type]
    combined = float(totals.get("percent_covered") or 0.0)  # type: ignore[arg-type]

    line = _summary_line(
        statements=statements,
        statement_floor=statement_floor,
        branches=branches,
        branch_floor=branch_floor,
        combined=combined,
    )
    errors: list[str] = []
    if statements < statement_floor:
        errors.append(
            f"error: statement coverage {statements:.2f}% is below ratchet "
            f"{statement_floor:.0f}%"
        )
    if branches < branch_floor:
        errors.append(
            f"error: branch coverage {branches:.2f}% is below ratchet "
            f"{branch_floor:.0f}%"
        )
    return (not errors, line, errors)


def main() -> int:
    statement_floor = _env_float("COVERAGE_RATCHET_MIN", _DEFAULT_STATEMENT_MIN)
    branch_floor = _env_float(
        "COVERAGE_RATCHET_MIN_BRANCHES", _DEFAULT_BRANCH_MIN
    )
    report = Path("coverage.json")
    if not report.is_file():
        print(
            f"error: missing {report}; run pytest with --cov-report=json:coverage.json",
            file=sys.stderr,
        )
        return 2

    data = json.loads(report.read_text(encoding="utf-8"))
    totals = data["totals"]
    passed, line, errors = evaluate(
        totals,
        statement_floor=statement_floor,
        branch_floor=branch_floor,
    )

    print(f"coverage ratchet: {line}")
    write_markdown_report(
        _MARKDOWN_PATH,
        statements=float(totals["percent_statements_covered"]),
        statement_floor=statement_floor,
        branches=float(totals.get("percent_branches_covered") or 0.0),
        branch_floor=branch_floor,
        combined=float(totals.get("percent_covered") or 0.0),
        passed=passed,
    )

    for message in errors:
        print(message, file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
