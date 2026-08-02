#!/usr/bin/env python3
"""Fail if statement (line) coverage drops below the ratchet floor.

Branch coverage is measured in CI and uploaded to Codecov, but is not gated
here yet. Raise COVERAGE_RATCHET_MIN as the suite improves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Current baseline (2026-08): ~65% statements on the unit suite without ML extras.
COVERAGE_RATCHET_MIN = 65.0


def main() -> int:
    report = Path("coverage.json")
    if not report.is_file():
        print(f"error: missing {report}; run pytest with --cov-report=json:coverage.json", file=sys.stderr)
        return 2

    data = json.loads(report.read_text(encoding="utf-8"))
    totals = data["totals"]
    statements = float(totals["percent_statements_covered"])
    branches = float(totals.get("percent_branches_covered") or 0.0)
    combined = float(totals.get("percent_covered") or 0.0)

    print(
        f"coverage ratchet: statements={statements:.2f}% "
        f"(min {COVERAGE_RATCHET_MIN:.0f}%) | "
        f"branches={branches:.2f}% (informational) | "
        f"combined={combined:.2f}% (informational)"
    )

    if statements < COVERAGE_RATCHET_MIN:
        print(
            f"error: statement coverage {statements:.2f}% is below ratchet "
            f"{COVERAGE_RATCHET_MIN:.0f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
