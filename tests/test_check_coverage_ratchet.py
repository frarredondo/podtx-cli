from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_coverage_ratchet.py"


def _load_ratchet():
    spec = importlib.util.spec_from_file_location("check_coverage_ratchet", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def _totals(
    *,
    statements: float,
    branches: float,
    combined: float,
) -> dict[str, float]:
    return {
        "percent_statements_covered": statements,
        "percent_branches_covered": branches,
        "percent_covered": combined,
    }


def test_evaluate_passes_when_both_above_floor() -> None:
    passed, line, errors = ratchet.evaluate(
        _totals(statements=70.0, branches=50.0, combined=65.0),
        statement_floor=65.0,
        branch_floor=45.0,
    )
    assert passed is True
    assert errors == []
    assert "statements=70.00% (min 65%)" in line
    assert "branches=50.00% (min 45%)" in line
    assert "combined=65.00% (informational)" in line


def test_evaluate_fails_on_statement_or_branch_floor() -> None:
    passed, _, errors = ratchet.evaluate(
        _totals(statements=64.0, branches=50.0, combined=60.0),
        statement_floor=65.0,
        branch_floor=45.0,
    )
    assert passed is False
    assert any("statement coverage" in e for e in errors)

    passed2, _, errors2 = ratchet.evaluate(
        _totals(statements=70.0, branches=40.0, combined=60.0),
        statement_floor=65.0,
        branch_floor=45.0,
    )
    assert passed2 is False
    assert any("branch coverage" in e for e in errors2)


def test_write_markdown_report_mentions_project_not_patch(tmp_path: Path) -> None:
    path = tmp_path / "coverage-ratchet.md"
    ratchet.write_markdown_report(
        path,
        statements=69.35,
        statement_floor=65.0,
        branches=53.45,
        branch_floor=45.0,
        combined=65.49,
        passed=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "Whole package (`podtx`)" in text
    assert "Codecov patch" in text
    assert "statements=69.35% (min 65%)" in text
    assert "branches=53.45% (min 45%)" in text
    assert "**Status:** passed" in text


def test_main_writes_markdown_and_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "totals": _totals(
                    statements=70.0, branches=50.0, combined=65.0
                )
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COVERAGE_RATCHET_MIN", "65")
    monkeypatch.setenv("COVERAGE_RATCHET_MIN_BRANCHES", "45")
    monkeypatch.setattr(ratchet, "_MARKDOWN_PATH", tmp_path / "coverage-ratchet.md")

    assert ratchet.main() == 0
    assert (tmp_path / "coverage-ratchet.md").is_file()

    report.write_text(
        json.dumps(
            {
                "totals": _totals(
                    statements=70.0, branches=10.0, combined=50.0
                )
            }
        ),
        encoding="utf-8",
    )
    assert ratchet.main() == 1
