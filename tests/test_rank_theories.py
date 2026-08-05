"""Tests for `scripts/rank_theories.py`.

The ranking script reads the per-experiment `eval_hilbig_per_stimulus.csv`
files an autocog human-eval run produced and turns them into theory x
experiment tables of several fit metrics, plus one pooled cross-experiment
model-vs-human correlation. The invariants worth pinning:

1. Per-experiment metrics (MAE / RMSE / Pearson / Spearman / binomial
   log-likelihood) are computed correctly from per-stimulus human and
   model P(B). Hand-checked on a tiny input so a sign flip or a
   column-mixup is caught.
2. Ranking respects metric direction: error metrics rank low=best,
   correlation / log-likelihood rank high=best, and ranks are computed
   *within* each experiment (column), independently.
3. The pooled correlation concatenates every stimulus across all
   experiments for a model before correlating — it is NOT the mean of
   per-experiment correlations.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def _toy_per_stim() -> pd.DataFrame:
    """Two experiments, two models, hand-checkable numbers.

    e1/m1: human=[0.0, 1.0], model=[0.25, 0.75], 10 trials each.
    e1/m2: human=[0.0, 1.0], model=[1.0-eps, eps] (anti-correlated).
    e2/m1: human=[0.5], model=[0.5], 20 trials.
    """
    rows = [
        ("e1", "m1", "surfaced", 0.0, 0.25, 10),
        ("e1", "m1", "surfaced", 1.0, 0.75, 10),
        ("e1", "m2", "base", 0.0, 0.9, 10),
        ("e1", "m2", "base", 1.0, 0.1, 10),
        ("e2", "m1", "surfaced", 0.5, 0.5, 20),
        ("e2", "m2", "base", 0.5, 0.2, 20),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "experiment", "model", "role",
            "p_b_human", "p_b_model", "n_trials_human",
        ],
    )


def test_per_experiment_metrics_match_hand_computation():
    from scripts.rank_theories import compute_metrics_per_experiment

    m = compute_metrics_per_experiment(_toy_per_stim())
    row = m[(m["experiment"] == "e1") & (m["model"] == "m1")].iloc[0]

    # MAE = mean(|0.25-0|, |0.75-1|) = 0.25; MSE = 0.0625; RMSE = 0.25
    assert row["mae"] == pytest.approx(0.25)
    assert row["mse"] == pytest.approx(0.0625)
    assert row["rmse"] == pytest.approx(0.25)
    # Perfectly rank/linearly aligned -> Pearson = Spearman = 1.
    assert row["pearson_r"] == pytest.approx(1.0)
    assert row["spearman_r"] == pytest.approx(1.0)
    # Binomial LL: n_choose_b = [0, 10] over model [0.25, 0.75]:
    #   10*log(0.75) + 10*log(0.75) = 20*log(0.75); per-trial = log(0.75).
    assert row["loglik_total"] == pytest.approx(20 * math.log(0.75))
    assert row["loglik_per_trial"] == pytest.approx(math.log(0.75))
    assert int(row["n_stimuli"]) == 2


def test_ranking_respects_metric_direction_and_is_within_experiment():
    from scripts.rank_theories import (
        compute_metrics_per_experiment,
        metric_pivot,
        rank_pivot,
    )

    m = compute_metrics_per_experiment(_toy_per_stim())

    # MAE: lower is better. In e1, m1 (0.25) beats m2 (0.9) -> m1 rank 1.
    mae = metric_pivot(m, "mae")
    mae_rank = rank_pivot(mae, higher_is_better=False)
    assert mae_rank.loc["m1", "e1"] == 1
    assert mae_rank.loc["m2", "e1"] == 2

    # Pearson: higher is better. m2 is anti-correlated in e1 -> rank 2.
    pr = metric_pivot(m, "pearson_r")
    pr_rank = rank_pivot(pr, higher_is_better=True)
    assert pr_rank.loc["m1", "e1"] == 1
    assert pr_rank.loc["m2", "e1"] == 2


def test_pooled_correlation_concatenates_across_experiments():
    from scripts.rank_theories import pooled_correlation

    # m1 over ALL its stimuli: human=[0,1,0.5], model=[0.25,0.75,0.5].
    pooled = pooled_correlation(_toy_per_stim())
    r_m1 = pooled[pooled["model"] == "m1"]["pearson_r"].iloc[0]
    expected = np.corrcoef([0.0, 1.0, 0.5], [0.25, 0.75, 0.5])[0, 1]
    assert r_m1 == pytest.approx(expected)
    # Pooled point count = every stimulus the model was scored on.
    assert int(pooled[pooled["model"] == "m1"]["n_points"].iloc[0]) == 3
