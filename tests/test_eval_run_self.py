"""Tests for scripts/eval_run_self.py — the run-agnostic self-comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
def test_experiments_up_to_round_filters_by_filename_round(tmp_path):
    from scripts.eval_run_self import experiments_up_to_round

    names = [
        tmp_path / f"round_{n:03d}_obs_{m:02d}.txt"
        for n in range(25) for m in (0, 1)
    ]
    assert len(experiments_up_to_round(names, 2)) == 6
    assert len(experiments_up_to_round(names, 24)) == 50


