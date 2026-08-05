"""Tests for scripts/eval_run_self.py — the run-agnostic self-comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN25 = (
    REPO_ROOT
    / "results/heuristic_decision_making/centaur_corrected_theories"
    / "seeds_tallying_ttb_rounds_25"
    / "hdm_seeds_tallying_ttb_gemini-3.1-pro-preview_rundella_centaur"
)


@pytest.mark.skipif(not RUN25.is_dir(), reason="25-round run dir not present")
def test_detect_n_rounds(tmp_path):
    from scripts.eval_run_self import detect_n_rounds

    assert detect_n_rounds(RUN25) == 25


@pytest.mark.skipif(not RUN25.is_dir(), reason="25-round run dir not present")
def test_resolve_rounds_cutoffs_defaults_to_all(tmp_path):
    from scripts.eval_run_self import resolve_rounds_cutoffs

    rounds, cutoffs = resolve_rounds_cutoffs(RUN25, rounds=None, cutoffs=None)
    assert rounds == list(range(25))
    assert cutoffs == list(range(1, 26))
    # explicit args pass through unchanged
    r2, c2 = resolve_rounds_cutoffs(RUN25, rounds=[0, 1], cutoffs=[5])
    assert r2 == [0, 1] and c2 == [5]


def test_experiments_up_to_round_filters_by_filename_round(tmp_path):
    from scripts.eval_run_self import experiments_up_to_round

    names = [
        tmp_path / f"round_{n:03d}_obs_{m:02d}.txt"
        for n in range(25) for m in (0, 1)
    ]
    assert len(experiments_up_to_round(names, 2)) == 6
    assert len(experiments_up_to_round(names, 24)) == 50


@pytest.mark.skipif(not RUN25.is_dir(), reason="25-round run dir not present")
def test_main_smoke_both_views(tmp_path):
    from scripts.eval_run_self import main

    out = tmp_path / "analysis"
    rc = main([
        "--run-dir", str(RUN25),
        "--exp-out", str(tmp_path / "self_exp"),
        "--out", str(out),
        "--cutoffs", "5",
        "--rounds", "0", "1",
        "--n-subjects", "40",
        "--which", "both",
    ])
    assert rc == 0
    # suffix "all" (no centaur "all50")
    assert (out / "trajectory_all.csv").is_file()
    assert (out / "calibration_scatter_all.png").is_file()
    assert (out / "trajectory_all.png").is_file()
    assert (out / "trajectory_cumulative.csv").is_file()
    assert (out / "calibration_scatter_cumulative.png").is_file()
    a = pd.read_csv(out / "per_experiment_summary_all.csv")
    assert a[(a.cutoff_round == 5) & (a.model == "pi_6")].shape[0] == 50
    cum = pd.read_csv(out / "per_experiment_summary_cumulative.csv")
    assert cum[cum.cutoff_round == 1].groupby("model").size().max() == 4
