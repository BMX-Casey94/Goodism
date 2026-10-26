"""Continuous-integration gate. One command runs every check.

    python -m tools.kb.ci

Checks (all must pass):
  1. Corpus + manifest validation (schema, hashes, duplicates, links).
  2. Unit tests for the integrity gate and chunker.
  3. Evaluation harness (recall@k must be 1.0 on the question set).

Intended to run locally pre-commit and in CI. Exits non-zero on failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import config
from .validate import run_validation


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    proc = subprocess.run(cmd, cwd=config.ROOT)
    return proc.returncode == 0


def main() -> int:
    failures: list[str] = []

    print("=== corpus + manifest validation ===")
    report = run_validation()
    for warning in report.warnings:
        print(f"WARN  {warning}")
    for error in report.errors:
        print(f"ERROR {error}")
    print(f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    if not report.ok:
        failures.append("validation")

    if not _run(
        [sys.executable, "-m", "pytest", "tools/kb/tests/", "-q"], "unit tests"
    ):
        failures.append("unit tests")

    # Eval only makes sense once the index exists.
    if config.INDEX_DB_PATH.exists():
        if not _run([sys.executable, "eval/run_eval.py", "--k", "5"], "evaluation"):
            failures.append("evaluation")
    else:
        print("\n=== evaluation skipped (no index yet) ===")

    print("\n" + ("CI PASSED" if not failures else f"CI FAILED: {', '.join(failures)}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
