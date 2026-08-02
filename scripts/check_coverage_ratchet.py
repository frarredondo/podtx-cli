#!/usr/bin/env python3
"""Fail if statement (line) coverage drops below the ratchet floor.

Branch coverage is measured in CI and uploaded to Codecov, but is not gated
here yet.

The floor comes from env ``COVERAGE_RATCHET_MIN`` (GitHub Actions repo variable
in CI). Default is 65 to match the 2026-08 unit-suite baseline without ML extras.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_DEFAULT_RATCHET_MIN = 65.0


def _ratchet_min() -> float:
    raw = os.environ.get("COVERAGE_RATCHET_MIN", "").strip()
    if not raw:
        return _DEFAULT_RATCHET_MIN
    try:
        return float(raw)
    except ValueError:
        print(
            f"error: COVERAGE_RATCHET_MIN must be a number, got {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def main() -> int:
    floor = _ratchet_min()
    report = Path("coverage.json")
    if not report.is_file():
        print(
            f"error: missing {report}; run pytest with --cov-report=json:coverage.json",
            file=sys.stderr,
        )
        return 2

    data = json.loads(report.read_text(encoding="utf-8"))
    totals = data["totals"]
    statements = float(totals["percent_statements_covered"])
    branches = float(totals.get("percent_branches_covered") or 0.0)
    combined = float(totals.get("percent_covered") or 0.0)

    print(
        f"coverage ratchet: statements={statements:.2f}% "
        f"(min {floor:.0f}%) | "
        f"branches={branches:.2f}% (informational) | "
        f"combined={combined:.2f}% (informational)"
    )

    if statements < floor:
        print(
            f"error: statement coverage {statements:.2f}% is below ratchet "
            f"{floor:.0f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
