"""Every runnable script must be invocable as `python path/to/script.py`.

Regression guard (2026-09-22): task1/data/make_subset.py and
make_cue_conflicts.py imported from `common` without putting the repository
root on sys.path. They worked locally only because the repo root happened to
be the working directory; on Colab they failed with ModuleNotFoundError.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

ENTRY_POINTS = [
    "shared/make_splits.py",
    "task1/data/make_subset.py",
    "task1/data/make_cue_conflicts.py",
    "task1/scripts/run_task1.py",
    "task2/train.py",
    "task2/evaluate_final.py",
    "task3/train.py",
    "task3/evaluate_sketch.py",
    "task4/train.py",
    "task4/extract_outputs.py",
    "task4/evaluate_osr.py",
]


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_runs_from_an_unrelated_directory(script: str, tmp_path) -> None:
    """`--help` exercises every module-level import without needing data."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script} failed from a foreign cwd:\n{result.stderr[-800:]}"
    )
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_bootstraps_sys_path(script: str) -> None:
    """The bootstrap must be present, not merely working by accident."""
    source = (REPO_ROOT / script).read_text(encoding="utf-8")
    assert "sys.path.insert" in source, f"{script} lacks a sys.path bootstrap"
