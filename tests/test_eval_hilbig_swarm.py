"""Tests for the per-pair MSE swarm overlay (`plot_mse_swarm`).

The swarm shows, per model, every unique stimulus pair's squared error
(p_b_model - p_b_human)^2 — the quantity the MSE bar averages over. The
key invariant is analytic: the per-model mean of those squared errors must
equal the MSE as `eval_hilbig` defines it (mean of diff^2, line 504).
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from scripts.eval_hilbig import (
    per_model_mse_sem,
    per_pair_squared_error,
    plot_mse_swarm,
)


def _per_stim():
    """Two models, two unique pairs each, hand-picked errors:
      pi_1: errors 0.1, 0.3 -> sq 0.01, 0.09 -> MSE 0.05
      pi_2: errors 0.2, -0.2 -> sq 0.04, 0.04 -> MSE 0.04
    """
    return pd.DataFrame([
        dict(model="pi_1", role="base", p_b_human=0.5, p_b_model=0.6),
        dict(model="pi_1", role="base", p_b_human=0.5, p_b_model=0.8),
        dict(model="pi_2", role="surfaced", p_b_human=0.5, p_b_model=0.7),
        dict(model="pi_2", role="surfaced", p_b_human=0.5, p_b_model=0.3),
    ])


def test_per_pair_squared_error_exact_values():
    """Each pair's sq_err is exactly (p_b_model - p_b_human)^2."""
    d = per_pair_squared_error(_per_stim())
    np.testing.assert_allclose(
        sorted(d["sq_err"].to_numpy()), [0.01, 0.04, 0.04, 0.09],
    )


def test_per_model_mean_equals_mse_definition():
    """Per-model mean squared error reproduces the MSE bar height exactly,
    matching eval_hilbig's summary definition mean(diff*diff)."""
    df = _per_stim()
    means = per_pair_squared_error(df).groupby("model")["sq_err"].mean()
    np.testing.assert_allclose(means["pi_1"], 0.05)
    np.testing.assert_allclose(means["pi_2"], 0.04)
    for m, g in df.groupby("model"):
        diff = (g["p_b_model"] - g["p_b_human"]).to_numpy()
        assert means[m] == pytest.approx(float(np.mean(diff * diff)))


def test_per_model_mse_sem_analytic():
    """SEM of the per-pair squared errors = std(ddof=1)/sqrt(n_pairs).
      pi_1 sq errors {0.01, 0.09}: var=0.0032, std=sqrt(0.0032),
        sem=std/sqrt(2)=0.04 exactly.
      pi_2 sq errors {0.04, 0.04}: zero variance -> sem=0.
    """
    sem = per_model_mse_sem(_per_stim())
    np.testing.assert_allclose(sem["pi_1"], 0.04)
    np.testing.assert_allclose(sem["pi_2"], 0.0, atol=1e-12)


def test_plot_mse_swarm_writes_svg_and_png(tmp_path):
    """The figure writes both an SVG and a PNG, like the other eval figures."""
    out = tmp_path / "mse_swarm.png"
    plot_mse_swarm(_per_stim(), out, show_model_labels=False)
    assert out.with_suffix(".png").is_file()
    assert out.with_suffix(".svg").is_file()


def test_plot_mse_swarm_respects_exclude_models(tmp_path):
    """`exclude_models` drops a model before computing/plotting; the figure
    still writes with the remaining models."""
    out = tmp_path / "mse_swarm.png"
    plot_mse_swarm(
        _per_stim(), out, exclude_models=("pi_2",), show_model_labels=False,
    )
    assert out.with_suffix(".png").is_file()
