"""Tests for scripts/eval_human_trajectory.py and the eval_human helpers
it depends on (round-cutoff survivor resolution, per-stimulus SEM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
def test_resolve_surfaced_missing_round_raises(tmp_path):
    from scripts.eval_human import resolve_surfaced_theories

    (tmp_path / "rounds").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_surfaced_theories(tmp_path, round_idx=99)


def test_compute_human_proportions_adds_across_subject_sem():
    from scripts.eval_human import compute_human_proportions

    # Pair X: 2 participants, per-subject P(B) = 0.0 and 1.0.
    #   mean across subjects = 0.5; sample SD (ddof=1) = 0.7071...;
    #   SEM = 0.7071 / sqrt(2) = 0.5 exactly.
    df = pd.DataFrame({
        "participant": [1, 1, 2, 2],
        "trial": [0, 1, 0, 1],
        "choice": [0, 0, 1, 1],
        "option_a": [(0, 0)] * 4,
        "option_b": [(1, 1)] * 4,
    })
    out = compute_human_proportions(df)
    row = out.iloc[0]
    assert "p_b_sem" in out.columns
    assert row["p_b_sem"] == pytest.approx(0.5)


def test_compute_human_proportions_sem_zero_for_single_subject():
    from scripts.eval_human import compute_human_proportions

    df = pd.DataFrame({
        "participant": [1, 1],
        "trial": [0, 1],
        "choice": [0, 1],
        "option_a": [(0, 0)] * 2,
        "option_b": [(1, 1)] * 2,
    })
    out = compute_human_proportions(df)
    assert out.iloc[0]["p_b_sem"] == 0.0


def test_summarize_fit_perfect_match():
    from scripts.eval_human_trajectory import summarize_fit

    human = np.array([0.1, 0.5, 0.9])
    mae, mse, r = summarize_fit(human.copy(), human.copy())
    assert mae == pytest.approx(0.0)
    assert mse == pytest.approx(0.0)
    assert r == pytest.approx(1.0)


def test_summarize_fit_constant_model_gives_nan_r():
    from scripts.eval_human_trajectory import summarize_fit

    human = np.array([0.1, 0.5, 0.9])
    model = np.array([0.5, 0.5, 0.5])
    mae, mse, r = summarize_fit(model, human)
    assert mae == pytest.approx(np.mean(np.abs(model - human)))
    assert np.isnan(r)


def test_replay_with_spread_shapes_and_bounds():
    from scripts.eval_human import load_canonical_theories
    from scripts.eval_human_trajectory import replay_with_spread

    ttb = load_canonical_theories()["ttb"]
    pairs = [((0, 0, 0, 0), (1, 1, 1, 1)), ((1, 0, 1, 0), (0, 1, 0, 1))]
    mean, sem = replay_with_spread(
        ttb, pairs, validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_subjects=50, base_seed=0,
    )
    assert mean.shape == (2,) and sem.shape == (2,)
    assert np.all((mean >= 0) & (mean <= 1))
    assert np.all(sem >= 0)


def test_replay_with_spread_matches_eval_human_mean():
    # The trajectory replay must reproduce eval_human's mean P(B) exactly
    # (same seeding), so cutoff-5 numbers match the verification baseline.
    from scripts.eval_human import (
        load_canonical_theories, model_proportions_for_theory,
    )
    from scripts.eval_human_trajectory import replay_with_spread

    ttb = load_canonical_theories()["ttb"]
    pairs = [((0, 0, 0, 0), (1, 1, 1, 1)), ((1, 0, 1, 0), (0, 1, 0, 1))]
    ref = model_proportions_for_theory(
        ttb, stimulus_pairs=pairs, validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_subjects=50, base_seed=0,
    )
    mean, _ = replay_with_spread(
        ttb, pairs, validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_subjects=50, base_seed=0,
    )
    for j, p in enumerate(pairs):
        assert mean[j] == pytest.approx(ref[p])


def test_gather_experiments_excludes_og_and_counts(tmp_path):
    from scripts.eval_human_trajectory import gather_experiments

    for name in ["a.txt", "b.txt", "b_og.txt"]:
        (tmp_path / name).write_text("x")
    got = gather_experiments(tmp_path, n_expected=2)
    assert [p.name for p in got] == ["a.txt", "b.txt"]

    with pytest.raises(SystemExit):
        gather_experiments(tmp_path, n_expected=10)


def test_aggregate_trajectory_mean_and_sem_across_experiments():
    from scripts.eval_human_trajectory import aggregate_trajectory

    # One cutoff, one model, two experiments: MAE 0.2 and 0.4.
    #   mean = 0.3; SD(ddof=1) = 0.1414...; SEM = 0.1414/sqrt(2) = 0.1.
    per_exp = pd.DataFrame([
        {"cutoff_round": 5, "experiment": "e0", "model": "pi_6",
         "role": "surfaced", "mae": 0.2, "mse": 0.04, "pearson_r": 0.8},
        {"cutoff_round": 5, "experiment": "e1", "model": "pi_6",
         "role": "surfaced", "mae": 0.4, "mse": 0.16, "pearson_r": 0.6},
    ])
    out = aggregate_trajectory(per_exp)
    mae_row = out[(out.model == "pi_6") & (out.metric == "mae")].iloc[0]
    assert mae_row["cutoff_round"] == 5
    assert mae_row["role"] == "surfaced"
    assert mae_row["n_experiments"] == 2
    assert mae_row["mean_across_exps"] == pytest.approx(0.3)
    assert mae_row["sem_across_exps"] == pytest.approx(0.1)
    # All three metrics present.
    assert set(out["metric"]) == {"mae", "mse", "pearson_r"}


def test_aggregate_trajectory_skips_nan_pearson():
    from scripts.eval_human_trajectory import aggregate_trajectory

    per_exp = pd.DataFrame([
        {"cutoff_round": 5, "experiment": "e0", "model": "pi_2",
         "role": "seed", "mae": 0.3, "mse": 0.1, "pearson_r": float("nan")},
        {"cutoff_round": 5, "experiment": "e1", "model": "pi_2",
         "role": "seed", "mae": 0.3, "mse": 0.1, "pearson_r": 0.5},
    ])
    out = aggregate_trajectory(per_exp)
    r_row = out[(out.model == "pi_2") & (out.metric == "pearson_r")].iloc[0]
    # NaN experiment is skipped: mean over the one finite value.
    assert r_row["mean_across_exps"] == pytest.approx(0.5)
    assert r_row["n_experiments"] == 1


def _toy_scatter_and_traj():
    scatter = pd.DataFrame([
        {"cutoff_round": 5, "experiment": "e0", "model": "pi_6",
         "role": "survivor", "option_a": "[0, 0]", "option_b": "[1, 1]",
         "p_b_human": 0.9, "p_b_human_sem": 0.05,
         "p_b_model": 0.8, "p_b_model_sem": 0.03},
        {"cutoff_round": 5, "experiment": "e0", "model": "pi_7",
         "role": "replacement", "option_a": "[0, 0]", "option_b": "[1, 1]",
         "p_b_human": 0.9, "p_b_human_sem": 0.05,
         "p_b_model": 0.5, "p_b_model_sem": 0.02},
    ])
    traj = pd.DataFrame([
        {"cutoff_round": 5, "model": "pi_6", "role": "surfaced",
         "metric": "mae", "mean_across_exps": 0.16,
         "sem_across_exps": 0.02, "n_experiments": 10},
        {"cutoff_round": 5, "model": "pi_1", "role": "seed",
         "metric": "mae", "mean_across_exps": 0.33,
         "sem_across_exps": 0.03, "n_experiments": 10},
    ])
    return scatter, traj


def test_plot_calibration_grid_writes_png(tmp_path):
    from scripts.eval_human_trajectory import plot_calibration_grid

    scatter, _ = _toy_scatter_and_traj()
    out = tmp_path / "calibration_scatter.png"
    plot_calibration_grid(scatter, out, cutoffs=[5])
    assert out.is_file() and out.stat().st_size > 0


def test_calibration_dot_color_matches_trajectory_scheme():
    from scripts.eval_human_trajectory import _calibration_dot_color
    from scripts.figure_style import GRAY, ROLE_COLOR

    # No theory_color map -> existing role-based default colour.
    assert _calibration_dot_color("pi_6", None) == ROLE_COLOR["surfaced"]
    # With a map, a known label gets its mapped colour ...
    assert _calibration_dot_color("pi_6", {"pi_6": "#123456"}) == "#123456"
    # ... and an unmapped label falls back to gray (matches surfaced_color).
    assert _calibration_dot_color("pi_x", {"pi_6": "#123456"}) == GRAY


def test_draw_calibration_panel_can_suppress_title():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.eval_human_trajectory import _draw_calibration_panel

    scatter, _ = _toy_scatter_and_traj()
    fig, ax = plt.subplots()
    _draw_calibration_panel(
        ax, scatter, "pi_6", "surfaced", 5, theory_color=None,
        ground_truth_label="human", ylabel=True, xlabel=True, show_title=False,
    )
    assert ax.get_title() == ""
    # Default still annotates the panel with the MAE/r title.
    fig2, ax2 = plt.subplots()
    _draw_calibration_panel(
        ax2, scatter, "pi_6", "surfaced", 5, theory_color=None,
        ground_truth_label="human", ylabel=True, xlabel=True,
    )
    assert ax2.get_title() != ""
    plt.close(fig)
    plt.close(fig2)


def test_plot_calibration_per_model_can_suppress_titles(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.eval_human_trajectory import plot_calibration_per_model

    scatter, _ = _toy_scatter_and_traj()
    plot_calibration_per_model(
        scatter, tmp_path, "calib_notitle", cutoffs=[5], show_title=False,
    )
    assert (tmp_path / "calib_notitle_pi_6.png").is_file()
    plt.close("all")


def test_plot_calibration_per_model_writes_one_file_per_model(tmp_path):
    from scripts.eval_human_trajectory import plot_calibration_per_model

    scatter, _ = _toy_scatter_and_traj()
    written = plot_calibration_per_model(
        scatter, tmp_path, "calibration_scatter_all", cutoffs=[5],
        theory_color={"pi_6": "#123456", "pi_7": "#654321"},
    )
    pi6 = tmp_path / "calibration_scatter_all_pi_6.png"
    pi7 = tmp_path / "calibration_scatter_all_pi_7.png"
    assert pi6.is_file() and pi6.stat().st_size > 0
    assert pi7.is_file() and pi7.stat().st_size > 0
    assert set(written) == {pi6, pi7}


def test_plot_calibration_grid_accepts_theory_color(tmp_path):
    from scripts.eval_human_trajectory import plot_calibration_grid

    scatter, _ = _toy_scatter_and_traj()
    out = tmp_path / "calibration_scatter_colored.png"
    plot_calibration_grid(
        scatter, out, cutoffs=[5],
        theory_color={"pi_6": "#123456", "pi_7": "#654321"},
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_metric_trajectory_writes_png(tmp_path):
    from scripts.eval_human_trajectory import plot_metric_trajectory

    _, traj = _toy_scatter_and_traj()
    out = tmp_path / "trajectory_p_b_pooled.png"
    plot_metric_trajectory(traj, out, target="p_b_pooled")
    assert out.is_file() and out.stat().st_size > 0


def _toy_traj_all_metrics():
    rows = []
    for metric in ("mae", "mse", "pearson_r"):
        rows.append(
            {"cutoff_round": 5, "model": "pi_6", "role": "surfaced",
             "metric": metric, "mean_across_exps": 0.2,
             "sem_across_exps": 0.02, "n_experiments": 10})
        rows.append(
            {"cutoff_round": 5, "model": "pi_1", "role": "seed",
             "metric": metric, "mean_across_exps": 0.3,
             "sem_across_exps": 0.03, "n_experiments": 10})
    return pd.DataFrame(rows)


def test_draw_metric_trajectory_panel_publication_styling():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.eval_human_trajectory import _draw_metric_trajectory_panel

    traj = pd.DataFrame([
        {"cutoff_round": c, "model": "pi_6", "role": "surfaced",
         "metric": "mse", "mean_across_exps": 0.2,
         "sem_across_exps": 0.02, "n_experiments": 10}
        for c in (1, 2, 3, 4, 5)
    ])
    fig, ax = plt.subplots()
    _draw_metric_trajectory_panel(
        ax, traj, "mse", "MSE", seeds_as_line=False, theory_color=None,
        with_legend=False, xlabel="cycles", show_title=False,
        integer_xticks=True,
    )
    # Title removed, x-axis relabelled, integer cycle ticks in steps of one.
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "cycles"
    assert ax.get_ylabel() == "MSE"
    assert list(ax.get_xticks()) == [1.0, 2.0, 3.0, 4.0, 5.0]
    plt.close(fig)


def test_draw_metric_trajectory_panel_defaults_unchanged():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.eval_human_trajectory import _draw_metric_trajectory_panel

    _, traj = _toy_scatter_and_traj()
    fig, ax = plt.subplots()
    _draw_metric_trajectory_panel(
        ax, traj, "mae", "MAE", seeds_as_line=False, theory_color=None,
        with_legend=False,
    )
    # Combined-figure defaults: title kept, original x-axis label.
    assert ax.get_title() == "MAE"
    assert ax.get_xlabel() == "cutoff round"
    plt.close(fig)


def test_plot_metric_trajectory_per_metric_writes_one_file_each(tmp_path):
    from scripts.eval_human_trajectory import plot_metric_trajectory_per_metric

    traj = _toy_traj_all_metrics()
    written = plot_metric_trajectory_per_metric(
        traj, tmp_path, "trajectory_all", target="p_b_pooled",
        theory_color={"pi_6": "#123456", "pi_1": "#654321"},
    )
    expected = {
        tmp_path / "trajectory_all_mae.png",
        tmp_path / "trajectory_all_mse.png",
        tmp_path / "trajectory_all_pearson_r.png",
    }
    for p in expected:
        assert p.is_file() and p.stat().st_size > 0
    assert set(written) == expected


HUMANS = (
    REPO_ROOT
    / "results/human_decision_making_cardinal"
    / "ttb+tallying/prolific_exp"
)


def test_replay_matched_to_participants_slices_shared_pool():
    # Each pair uses as many model draws as its participant count; the
    # per-pair mean must equal replay_with_spread run with n_subjects=n_j
    # (same seeding, same draw order), and a 1-participant pair gives SEM 0.
    from scripts.eval_human import load_canonical_theories
    from scripts.eval_human_trajectory import (
        replay_matched_to_participants, replay_with_spread,
    )

    ttb = load_canonical_theories()["ttb"]
    pairs = [((0, 0, 0, 0), (1, 1, 1, 1)), ((1, 0, 1, 0), (0, 1, 0, 1))]
    mean, sem = replay_matched_to_participants(
        ttb, pairs, n_per_pair=[1, 10],
        validities=[0.9, 0.8, 0.7, 0.6], rating_max=1, base_seed=0,
    )
    assert mean.shape == (2,) and sem.shape == (2,)
    assert sem[0] == 0.0  # single participant -> no spread

    ref_mean, _ = replay_with_spread(
        ttb, [pairs[1]], validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_subjects=10, base_seed=0,
    )
    assert mean[1] == pytest.approx(ref_mean[0])


def test_replay_matched_sem_larger_than_200_draw_sem():
    # Matching to a small participant count must yield a LARGER model SEM
    # than the over-precise 200-draw replay (the whole point of the plot).
    from scripts.eval_human import load_canonical_theories
    from scripts.eval_human_trajectory import (
        replay_matched_to_participants, replay_with_spread,
    )

    wadd = load_canonical_theories()["wadd"]
    pairs = [((1, 1, 0, 0), (0, 0, 1, 1)), ((0, 1, 1, 1), (1, 0, 0, 0))]
    _, sem_matched = replay_matched_to_participants(
        wadd, pairs, n_per_pair=[8, 8],
        validities=[0.9, 0.8, 0.7, 0.6], rating_max=1, base_seed=0,
    )
    _, sem_200 = replay_with_spread(
        wadd, pairs, validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_subjects=200, base_seed=0,
    )
    assert np.mean(sem_matched) > np.mean(sem_200)


def test_plot_metric_trajectory_accepts_ground_truth_label(tmp_path):
    from scripts.eval_human_trajectory import plot_metric_trajectory

    traj = pd.DataFrame([
        {"cutoff_round": 5, "model": "pi_1", "role": "seed",
         "metric": "mae", "mean_across_exps": 0.33,
         "sem_across_exps": 0.03, "n_experiments": 50},
        {"cutoff_round": 5, "model": "pi_6", "role": "surfaced",
         "metric": "mae", "mean_across_exps": 0.16,
         "sem_across_exps": 0.02, "n_experiments": 50},
    ])
    out = tmp_path / "traj_label.png"
    plot_metric_trajectory(
        traj, out, target="p_b_pooled", ground_truth_label="Centaur",
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_calibration_grid_accepts_ground_truth_label(tmp_path):
    from scripts.eval_human_trajectory import plot_calibration_grid

    scatter = pd.DataFrame([
        {"cutoff_round": 5, "experiment": "e0", "model": "pi_6",
         "role": "survivor", "option_a": "[0, 0]", "option_b": "[1, 1]",
         "p_b_human": 0.9, "p_b_human_sem": 0.05,
         "p_b_model": 0.8, "p_b_model_sem": 0.03},
    ])
    out = tmp_path / "grid_label.png"
    plot_calibration_grid(
        scatter, out, cutoffs=[5], ground_truth_label="Centaur",
    )
    assert out.is_file() and out.stat().st_size > 0
