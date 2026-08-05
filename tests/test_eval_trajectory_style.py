"""Style tests for the shared eval plotting functions (calibration grid and
metric trajectory) used by eval_run_self / eval_centaur_self / eval_human."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from scripts.eval_human_trajectory import (
    plot_calibration_grid,
    plot_metric_trajectory,
)


def _metric_df():
    rows = []
    for c in (5, 10):
        rows += [
            dict(metric="mae", role="surfaced", cutoff_round=c,
                 mean_across_exps=0.1, sem_across_exps=0.02, model="pi_3",
                 n_experiments=10),
            dict(metric="mae", role="seed", cutoff_round=c,
                 mean_across_exps=0.2, sem_across_exps=0.0, model="pi_1",
                 n_experiments=10),
            dict(metric="mse", role="surfaced", cutoff_round=c,
                 mean_across_exps=0.05, sem_across_exps=0.01, model="pi_3",
                 n_experiments=10),
            dict(metric="pearson_r", role="surfaced", cutoff_round=c,
                 mean_across_exps=0.8, sem_across_exps=0.05, model="pi_3",
                 n_experiments=10),
        ]
    return pd.DataFrame(rows)


def _scatter_df():
    rows = []
    for c in (5, 10):
        for k in range(4):
            rows.append(dict(cutoff_round=c, model="pi_3", role="survivor",
                             p_b_human=0.2 * k, p_b_model=0.2 * k + 0.05,
                             p_b_human_sem=0.02, p_b_model_sem=0.02))
    return pd.DataFrame(rows)


def test_metric_trajectory_writes_svg_and_png(tmp_path):
    out = tmp_path / "traj.png"
    plot_metric_trajectory(_metric_df(), out, target="p_b_pooled")
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".svg").exists()


def test_calibration_grid_writes_svg_and_png(tmp_path):
    out = tmp_path / "calib.png"
    plot_calibration_grid(_scatter_df(), out, cutoffs=[5, 10])
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".svg").exists()
