"""Tests for scripts/recovery_correlation.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = REPO_ROOT / "theories" / "heuristic_decision_making"

# Make src importable (conftest.py adds repo root to sys.path already, but
# we need Theory for the test fixtures).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.theory import Theory  # noqa: E402


def test_unique_stimulus_pairs_returns_int_tuple_pairs():
    from scripts.recovery_correlation import unique_stimulus_pairs

    pairs = unique_stimulus_pairs()  # default: real exp1.txt
    assert len(pairs) > 0
    # no duplicates
    assert len(pairs) == len(set(pairs))
    a, b = pairs[0]
    assert isinstance(a, tuple) and isinstance(b, tuple)
    assert all(isinstance(x, int) for x in a + b)
    # 4-feature Hilbig task
    assert len(a) == 4 and len(b) == 4


def test_choice_proportions_in_unit_interval():
    from scripts.recovery_correlation import (
        simulate_choice_proportions,
        unique_stimulus_pairs,
    )

    theory = Theory.from_yaml(YAML_DIR / "wadd.yaml")
    pairs = unique_stimulus_pairs()
    props = simulate_choice_proportions(
        theory, stimulus_pairs=pairs,
        validities=[0.9, 0.8, 0.7, 0.6], rating_max=1,
        n_draws=20, seed=0, action_noise=0.0,
    )
    assert props.shape == (len(pairs),)
    assert np.all(props >= 0.0) and np.all(props <= 1.0)


def test_full_action_noise_pushes_proportions_to_half():
    from scripts.recovery_correlation import (
        simulate_choice_proportions,
        unique_stimulus_pairs,
    )

    theory = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    pairs = unique_stimulus_pairs()
    props = simulate_choice_proportions(
        theory, stimulus_pairs=pairs,
        validities=[0.9, 0.8, 0.7, 0.6], rating_max=1,
        n_draws=400, seed=1, action_noise=1.0,
    )
    # action_noise=1.0 => every choice is a uniform 50/50 coin.
    assert np.allclose(props.mean(), 0.5, atol=0.05)


def test_determinism_same_seed_same_result():
    from scripts.recovery_correlation import (
        simulate_choice_proportions,
        unique_stimulus_pairs,
    )

    theory = Theory.from_yaml(YAML_DIR / "wadd.yaml")
    pairs = unique_stimulus_pairs()
    kw = dict(
        stimulus_pairs=pairs, validities=[0.9, 0.8, 0.7, 0.6],
        rating_max=1, n_draws=15, seed=7, action_noise=0.05,
    )
    a = simulate_choice_proportions(theory, **kw)
    b = simulate_choice_proportions(theory, **kw)
    assert np.array_equal(a, b)


def test_pearson_r_perfect_and_zero_variance():
    from scripts.recovery_correlation import pearson_r

    x = np.array([0.1, 0.4, 0.9, 0.5])
    assert pearson_r(x, x) == pytest.approx(1.0)
    # zero variance in one vector -> NaN (undefined correlation)
    assert np.isnan(pearson_r(x, np.ones_like(x)))


def test_matching_theory_recovers_better_than_wrong_family():
    """A TTB target is recovered better by TTB than by WADD."""
    from scripts.recovery_correlation import (
        simulate_choice_proportions,
        correlation_for_theory,
        unique_stimulus_pairs,
    )

    pairs = unique_stimulus_pairs()
    vk = dict(validities=[0.9, 0.8, 0.7, 0.6], rating_max=1)
    ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")
    wadd = Theory.from_yaml(YAML_DIR / "wadd.yaml")

    target = simulate_choice_proportions(
        ttb, stimulus_pairs=pairs, n_draws=200, seed=900_000,
        action_noise=0.0, **vk,
    )
    r_ttb = correlation_for_theory(
        ttb, target, stimulus_pairs=pairs, n_draws=200, seed=100_001,
        action_noise=0.0, **vk,
    )
    r_wadd = correlation_for_theory(
        wadd, target, stimulus_pairs=pairs, n_draws=200, seed=300_001,
        action_noise=0.0, **vk,
    )
    assert r_ttb > r_wadd
    assert r_ttb > 0.9  # Fix 3: ceiling ≈1 at ε=0 (spec §9.1)


def test_empty_noises_normalises_to_all_dirs():
    """Fix 1: noises=None and noises=[] (normalised to None) both discover ≥1 dir."""
    from scripts.recovery_correlation import discover_run_dirs

    root = REPO_ROOT / "results" / "recovery"
    found_none = discover_run_dirs(root, families=["ttb_sampling"], noises=None)
    # Simulate what main() does: empty list -> None
    empty_list_normalised = None  # [] normalised to None
    found_normalised = discover_run_dirs(root, families=["ttb_sampling"], noises=empty_list_normalised)
    assert len(found_none) >= 1
    assert found_none == found_normalised


def test_discover_run_dirs_parses_family_and_noise():
    from scripts.recovery_correlation import discover_run_dirs

    root = REPO_ROOT / "results" / "recovery"
    found = discover_run_dirs(root, families=["ttb_sampling"], noises=[0.0])
    assert len(found) >= 1
    for family, noise, run_dir in found:
        assert family == "ttb_sampling"
        assert noise == 0.0
        assert run_dir.is_dir()
        assert (run_dir / "rounds").is_dir()


def test_gt_bar_degrades_monotonically_with_noise():
    """Correlation of the GT model to the noise-less GT target should not
    increase as action noise grows."""
    from scripts.recovery_correlation import (
        simulate_choice_proportions,
        correlation_for_theory,
        unique_stimulus_pairs,
    )

    pairs = unique_stimulus_pairs()
    vk = dict(validities=[0.9, 0.8, 0.7, 0.6], rating_max=1)
    gt = Theory.from_yaml(YAML_DIR / "wadd.yaml")
    target = simulate_choice_proportions(
        gt, stimulus_pairs=pairs, n_draws=300, seed=900_000,
        action_noise=0.0, **vk,
    )
    rs = [
        correlation_for_theory(
            gt, target, stimulus_pairs=pairs, n_draws=300,
            seed=700_000, action_noise=eps, **vk,
        )
        for eps in (0.0, 0.05, 0.3)
    ]
    # non-increasing (allow tiny finite-sample wiggle)
    assert rs[0] >= rs[1] - 0.05
    assert rs[1] >= rs[2] - 0.05
    assert rs[0] > rs[2]


def test_reference_target_is_sampled_with_run_noise():
    """The recovery reference is the GT theory sampled WITH each run's action
    noise (not the noise-less GT). Consequence: the gt bar (GT replayed at the
    run's ε, independent seed) is compared against a reference contracted by the
    SAME ε, so its MSE stays at the finite-sample floor across noise levels
    instead of growing with the contraction ε²(0.5-p)².

    Analytic anchor: as n_draws->inf both the gt bar and the reference converge
    to the same contracted vector p' = (1-ε)p + ε·0.5, so MSE(gt_bar, ref)->0 at
    every ε. Against a NOISE-LESS reference the gt MSE at ε=0.3 sits near
    ε²·E[(0.5-p)²] ≈ 0.09·0.12 ≈ 0.0075 for ttb (empirically 0.0075); the noisy
    reference must drive it down to the ~0.002 sampling-variance floor."""
    from scripts.recovery_correlation import (
        build_results_long, unique_stimulus_pairs,
    )

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["ttb_sampling"], noises=[0.0, 0.5],
        stimulus_pairs=pairs, n_draws=200, base_seed=0,
    )
    mse_by_noise = df[df.role == "gt"].groupby("noise")["mse"].mean()
    # gt bar vs same-noise reference: at the sampling-variance floor at BOTH
    # noise levels, well below the ~0.0075 contraction floor of a noiseless ref.
    assert mse_by_noise.loc[0.5] < 0.005
    # and barely above the ε=0 case (both sampling-only, no contraction term).
    assert mse_by_noise.loc[0.5] < mse_by_noise.loc[0.0] + 0.003


def test_gt_clean_role_is_mse_between_clean_and_noisy_gt_rises_with_noise():
    """The 'gt_clean' role is the MSE between the CLEAN gt (gt@ε=0 predictions)
    and its NOISY version (the gt@ε reference). Unlike the 'gt' row (gt@ε vs
    gt@ε, flat at the sampling floor), gt_clean RISES with ε because the noisy
    version contracts toward 0.5 while the clean prediction does not.

    Analytic anchor: at ε=1 the noisy version is a uniform coin (≈0.5
    everywhere), so gt_clean MSE → mean((p_clean - 0.5)²), the variance of the
    clean gt proportions about 0.5 (plus small finite-sample terms). Checked
    within a Monte-Carlo tolerance."""
    from scripts.recovery_correlation import (
        build_results_long, unique_stimulus_pairs, simulate_choice_proportions,
    )
    from scripts.eval_hilbig import (
        CANONICAL_YAML_DIR, HUMAN_VALIDITIES, HUMAN_RATING_MAX,
    )

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["ttb_sampling"], noises=[0.0, 1.0],
        stimulus_pairs=pairs, n_draws=150, base_seed=0,
    )
    assert "gt_clean" in set(df["role"])
    gtc = df[df.role == "gt_clean"].groupby("noise")["mse"].mean()
    gt = df[df.role == "gt"].groupby("noise")["mse"].mean()
    # rises with noise, and far above the flat gt ceiling at ε=1
    assert gtc.loc[1.0] > gtc.loc[0.0] + 0.02
    assert gtc.loc[1.0] > gt.loc[1.0] + 0.02
    # analytic: at ε=1 the noisy version ≈ 0.5 const -> gt_clean ≈ var of clean
    # gt proportions about 0.5 (within Monte-Carlo tolerance).
    gt_theory = Theory.from_yaml(CANONICAL_YAML_DIR / "ttb_sampling.yaml")
    clean = simulate_choice_proportions(
        gt_theory, stimulus_pairs=pairs, validities=HUMAN_VALIDITIES,
        rating_max=HUMAN_RATING_MAX, n_draws=400, seed=5, action_noise=0.0,
    )
    expected = float(((clean - 0.5) ** 2).mean())
    assert abs(gtc.loc[1.0] - expected) < 0.025


def test_reference_action_noise_pins_the_reference_to_a_fixed_noise():
    """`reference_action_noise` pins the reference's noise instead of the run's
    ε. Set to 0.0 the target is the CLEAN gt@0 for every column, so the gt /
    gt_clean roles swap relative to the default (per-ε noisy reference):
      * gt       (gt@ε vs clean gt@0)  RISES with ε,
      * gt_clean (gt@0 vs clean gt@0)  stays at the sampling floor.
    The default (reference_action_noise=None) gives the opposite (gt flat,
    gt_clean rising) — covered by the sibling gt_clean test."""
    from scripts.recovery_correlation import (
        build_results_long, unique_stimulus_pairs,
    )

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["ttb_sampling"], noises=[0.0, 1.0],
        stimulus_pairs=pairs, n_draws=150, base_seed=0,
        reference_action_noise=0.0,
    )
    gt = df[df.role == "gt"].groupby("noise")["mse"].mean()
    gtc = df[df.role == "gt_clean"].groupby("noise")["mse"].mean()
    assert gt.loc[1.0] > gt.loc[0.0] + 0.02     # gt@ε drifts from the clean gt
    assert gtc.loc[1.0] < 0.01                  # clean-vs-clean stays at floor


def test_plot_grid_share_y_threads_to_subplots(monkeypatch, tmp_path):
    """`share_y` controls whether panels share a y-axis. Default True keeps the
    existing shared-scale behaviour; False gives each family panel its own
    y-axis (seed magnitudes differ by family and aren't comparable)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.recovery_correlation import plot_grid

    summary = pd.DataFrame([
        dict(family=f, noise=n, role=role, mean=0.5, std=0.1, n_runs=3, sem=0.05)
        for f in ("ttb", "wadd")
        for n in (0.0, 0.5)
        for role in ("seed", "surfaced")
    ])
    captured = {}
    real_subplots = plt.subplots

    def fake_subplots(*a, **k):
        captured["sharey"] = k.get("sharey")
        return real_subplots(*a, **k)

    monkeypatch.setattr(plt, "subplots", fake_subplots)
    plot_grid(summary, tmp_path / "indep.png", share_y=False)
    assert captured["sharey"] is False
    plot_grid(summary, tmp_path / "shared.png")          # default
    assert captured["sharey"] is True


def test_legend_label_override_falls_back_to_role_label():
    """`_legend_label` returns the per-figure override when present, else the
    global role_label. Lets one figure relabel a role (e.g. the clean-reference
    figure where the gt / gt_clean lines swap meaning) without touching the
    global ROLE_DISPLAY map."""
    from scripts.recovery_correlation import _legend_label, role_label

    assert _legend_label("gt", {"gt": "Custom"}) == "Custom"
    assert _legend_label("gt", None) == role_label("gt")          # no override
    assert _legend_label("seed", {"gt": "Custom"}) == role_label("seed")  # miss


def test_build_results_long_has_all_three_roles():
    from scripts.recovery_correlation import build_results_long, unique_stimulus_pairs

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["ttb_sampling"], noises=[0.0],
        stimulus_pairs=pairs, n_draws=5, base_seed=0,
    )
    assert set(["family", "noise", "run_dir", "role", "theory_label",
                "n_stimuli", "pearson_r"]).issubset(df.columns)
    assert {"seed", "surfaced", "gt"}.issubset(set(df["role"].unique()))
    assert (df["family"] == "ttb_sampling").all()
    assert (df["noise"] == 0.0).all()
    # exactly one gt row per run-dir
    n_run_dirs = df["run_dir"].nunique()
    assert (df["role"] == "gt").sum() == n_run_dirs


def test_surfaced_best_per_run_vs_across_runs():
    """`surfaced_best_rows` keeps the best surfaced theory PER run-dir
    (one row per run, labelled 'surfaced (best per run)'), while
    `surfaced_best_across_runs_rows` keeps the SINGLE globally-best surfaced
    theory per (family, noise) across all run-dirs (one row, labelled
    'surfaced (best across runs)')."""
    from scripts.recovery_correlation import (
        surfaced_best_rows,
        surfaced_best_across_runs_rows,
    )

    long = pd.DataFrame([
        # run A: best surfaced r = 0.7
        dict(family="ttb", noise=0.0, run_dir="A", role="surfaced",
             theory_label="a1", n_stimuli=10, pearson_r=0.5, mse=0.5,
             lag1_autocorr=0.0),
        dict(family="ttb", noise=0.0, run_dir="A", role="surfaced",
             theory_label="a2", n_stimuli=10, pearson_r=0.7, mse=0.3,
             lag1_autocorr=0.0),
        # run B: best surfaced r = 0.9 (also the global best)
        dict(family="ttb", noise=0.0, run_dir="B", role="surfaced",
             theory_label="b1", n_stimuli=10, pearson_r=0.6, mse=0.4,
             lag1_autocorr=0.0),
        dict(family="ttb", noise=0.0, run_dir="B", role="surfaced",
             theory_label="b2", n_stimuli=10, pearson_r=0.9, mse=0.1,
             lag1_autocorr=0.0),
    ])

    per_run = surfaced_best_rows(long, metric="pearson_r", higher_is_better=True)
    assert set(per_run["role"]) == {"surfaced (best per run)"}
    assert len(per_run) == 2  # one per run-dir
    assert sorted(per_run["pearson_r"].tolist()) == [0.7, 0.9]

    across = surfaced_best_across_runs_rows(
        long, metric="pearson_r", higher_is_better=True
    )
    assert set(across["role"]) == {"surfaced (best across runs)"}
    assert len(across) == 1  # one per (family, noise)
    assert across["pearson_r"].iloc[0] == 0.9
    assert across["theory_label"].iloc[0] == "b2"

    # MSE: lower is better -> global min mse = 0.1 (b2).
    across_mse = surfaced_best_across_runs_rows(
        long, metric="mse", higher_is_better=False
    )
    assert across_mse["mse"].iloc[0] == 0.1
    assert across_mse["theory_label"].iloc[0] == "b2"


def test_summarise_pools_theories_per_run_then_sem_across_runs():
    from scripts.recovery_correlation import summarise

    # Two run-dirs, a role with two theories per run-dir: per-run mean of
    # the two theories, then mean/SEM across the two run-dirs.
    long = pd.DataFrame([
        # run A, seed role: theories 0.6 and 0.8 -> per-run mean 0.7
        dict(family="ttb", noise=0.0, run_dir="A", role="seed",
             theory_label="x", n_stimuli=10, pearson_r=0.6),
        dict(family="ttb", noise=0.0, run_dir="A", role="seed",
             theory_label="y", n_stimuli=10, pearson_r=0.8),
        # run B, seed role: theories 0.8 and 1.0 -> per-run mean 0.9
        dict(family="ttb", noise=0.0, run_dir="B", role="seed",
             theory_label="x", n_stimuli=10, pearson_r=0.8),
        dict(family="ttb", noise=0.0, run_dir="B", role="seed",
             theory_label="y", n_stimuli=10, pearson_r=1.0),
    ])
    s = summarise(long)
    row = s[(s.family == "ttb") & (s.noise == 0.0) & (s.role == "seed")].iloc[0]
    # mean of per-run means (0.7, 0.9) = 0.8
    assert row["mean"] == pytest.approx(0.8)
    # SEM across the two run-dir means = std([0.7,0.9],ddof=1)/sqrt(2)
    import numpy as _np
    expected_sem = _np.std([0.7, 0.9], ddof=1) / _np.sqrt(2)
    assert row["sem"] == pytest.approx(expected_sem)
    assert int(row["n_runs"]) == 2


def test_plot_grid_writes_png(tmp_path):
    from scripts.recovery_correlation import plot_grid

    summary = pd.DataFrame([
        dict(family=f, noise=n, role=role, mean=0.8, std=0.1, n_runs=3, sem=0.05)
        for f in ("ttb", "wadd", "tallying")
        for n in (0.0, 0.05, 0.3)
        for role in ("seed", "surfaced", "gt")
    ])
    out = tmp_path / "recovery_correlation.png"
    plot_grid(summary, out, title="test")
    assert out.is_file() and out.stat().st_size > 0


def test_build_random_family_rows_pools_canonical_eps1_vs_half_reference():
    """The synthetic 'random' family pools the canonical families' ε=1 run-dirs
    (GT behaves randomly there) and scores their seed/surfaced theories against
    a FIXED 0.5 reference. Signatures of the constant-0.5 reference: Pearson r is
    NaN for every row, and mse is bounded by 0.25 = max mean((p-0.5)²). Rows are
    labelled family='random', noise=0.0 (so the panel renders single-noise)."""
    from scripts.recovery_correlation import (
        build_random_family_rows, unique_stimulus_pairs,
    )

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_random_family_rows(
        results_root=root,
        canonical_families=["ttb_sampling", "wadd_sampling", "tallying_sampling"],
        stimulus_pairs=pairs, n_draws=5, base_seed=0, eps=1.0,
    )
    assert not df.empty
    assert (df["family"] == "random").all()
    assert {"seed", "surfaced"}.issubset(set(df["role"].unique()))
    assert (df["noise"] == 0.0).all()
    # constant 0.5 reference -> Pearson r undefined (NaN), mse in [0, 0.25].
    assert df["pearson_r"].isna().all()
    assert (df["mse"] >= 0.0).all() and (df["mse"] <= 0.25).all()
    # pooled across all THREE canonical families' ε=1 run-dirs.
    names = " ".join(df["run_dir"].unique())
    assert "ttb_sampling" in names
    assert "wadd_sampling" in names
    assert "tallying_sampling" in names


def test_plot_grid_bar_edgecolor_drops_black_stroke(tmp_path):
    """bar_edgecolor='none' removes the black bar outline (used by the pool /
    per-model figures so chunky single-noise bars read clean); the default
    keeps it (the combined grid needs it for the cream best-across-runs bar)."""
    from scripts.recovery_correlation import plot_grid

    summary = pd.DataFrame([
        dict(family="ttb_sampling", noise=0.0, role=r, mean=0.1, sem=0.01)
        for r in ("seed", "surfaced")
    ])
    plot_grid(summary, tmp_path / "edged.svg", roles=("seed", "surfaced"))
    plot_grid(summary, tmp_path / "clean.svg", roles=("seed", "surfaced"),
              bar_edgecolor="none")
    black_bar = "fill: #3d405b; stroke: #000000"
    assert black_bar in (tmp_path / "edged.svg").read_text()
    assert black_bar not in (tmp_path / "clean.svg").read_text()


def test_plot_grid_empty_summary_does_not_crash(tmp_path):
    """An empty summary (e.g. every pooled family is all-NaN for the metric, so
    summarise drops every row) must NOT crash plot_grid with a 1x0 subplots
    ValueError — it writes nothing and returns."""
    from scripts.recovery_correlation import plot_grid

    empty = pd.DataFrame(
        columns=["family", "noise", "role", "mean", "std", "n_runs", "sem"]
    )
    out = tmp_path / "empty.png"
    plot_grid(empty, out)  # must not raise
    assert not out.exists()


def test_mse_zero_for_identical_and_positive_for_shifted():
    from scripts.recovery_correlation import mse
    x = np.array([0.1, 0.4, 0.9, 0.5])
    assert mse(x, x) == 0.0
    assert mse(x, x + 0.2) == pytest.approx(0.04)


def test_random_model_has_nan_corr_and_positive_mse():
    from scripts.recovery_correlation import pearson_r, mse
    target = np.array([0.1, 0.8, 0.3, 0.95, 0.5])
    rand = np.full_like(target, 0.5)
    assert np.isnan(pearson_r(rand, target))   # constant vector -> undefined
    assert mse(rand, target) > 0.0


def test_build_results_long_has_mse_column_and_random_role():
    from scripts.recovery_correlation import build_results_long, unique_stimulus_pairs
    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["ttb_sampling"], noises=[0.0],
        stimulus_pairs=pairs, n_draws=5, base_seed=0,
    )
    assert "mse" in df.columns
    assert "random" in set(df["role"].unique())
    rand_rows = df[df["role"] == "random"]
    assert rand_rows["pearson_r"].isna().all()      # NaN corr
    assert (rand_rows["mse"] > 0).all()             # finite, positive mse
    # one random row per run-dir
    assert len(rand_rows) == df["run_dir"].nunique()


def test_summarise_works_for_mse_metric():
    from scripts.recovery_correlation import summarise
    long = pd.DataFrame([
        dict(family="ttb", noise=0.0, run_dir="A", role="gt",
             theory_label="gt", n_stimuli=10, pearson_r=0.9, mse=0.02),
        dict(family="ttb", noise=0.0, run_dir="B", role="gt",
             theory_label="gt", n_stimuli=10, pearson_r=0.8, mse=0.04),
    ])
    s = summarise(long, "mse")
    row = s[(s.role == "gt")].iloc[0]
    assert row["mean"] == pytest.approx(0.03)


def test_main_end_to_end_writes_csv_and_png(tmp_path):
    from scripts.recovery_correlation import main

    csv_out = tmp_path / "recovery_correlation.csv"
    png_out = tmp_path / "recovery_correlation.png"
    mse_out = tmp_path / "recovery_mse.png"
    # No --results-root: this exercises RESULTS_ROOT_DEFAULT (results/recovery),
    # whose canonical family directories carry the `_sampling` suffix.
    rc = main([
        "--families", "ttb_sampling",
        "--noises", "0.0",
        "--n-draws", "3",
        "--seed", "0",
        "--csv", str(csv_out),
        "--out", str(png_out),
        "--mse-out", str(mse_out),
    ])
    assert rc == 0
    assert csv_out.is_file() and csv_out.stat().st_size > 0
    assert png_out.is_file() and png_out.stat().st_size > 0
    assert (tmp_path / "recovery_mse.png").is_file()
    df = pd.read_csv(csv_out)
    assert {"family", "noise", "run_dir", "role", "theory_label",
            "pearson_r"}.issubset(df.columns)
    assert "mse" in df.columns
    assert "random" in set(df["role"].unique())


def test_surfaced_best_rows_picks_max_corr_and_min_mse():
    from scripts.recovery_correlation import surfaced_best_rows
    long = pd.DataFrame([
        dict(family="ttb", noise=0.0, run_dir="A", role="surfaced",
             theory_label="p1", n_stimuli=10, pearson_r=0.6, mse=0.01),
        dict(family="ttb", noise=0.0, run_dir="A", role="surfaced",
             theory_label="p2", n_stimuli=10, pearson_r=0.9, mse=0.02),
        dict(family="ttb", noise=0.0, run_dir="A", role="seed",
             theory_label="s", n_stimuli=10, pearson_r=0.3, mse=0.1),
    ])
    best_corr = surfaced_best_rows(long, metric="pearson_r", higher_is_better=True)
    assert len(best_corr) == 1
    assert best_corr.iloc[0]["role"] == "surfaced (best per run)"
    assert best_corr.iloc[0]["theory_label"] == "p2"   # max r
    best_mse = surfaced_best_rows(long, metric="mse", higher_is_better=False)
    assert best_mse.iloc[0]["theory_label"] == "p1"    # min mse
    assert best_mse.iloc[0]["role"] == "surfaced (best per run)"


def test_surfaced_best_ignores_nan_and_empty():
    from scripts.recovery_correlation import surfaced_best_rows
    # no surfaced rows -> empty result, same columns
    long = pd.DataFrame([
        dict(family="ttb", noise=0.0, run_dir="A", role="seed",
             theory_label="s", n_stimuli=10, pearson_r=0.3, mse=0.1),
    ])
    out = surfaced_best_rows(long, metric="pearson_r", higher_is_better=True)
    assert out.empty
    assert list(out.columns) == list(long.columns)


def test_surfaced_best_at_least_mean_in_pipeline():
    from scripts.recovery_correlation import (
        build_results_long, surfaced_best_rows, summarise, unique_stimulus_pairs,
    )
    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    long = build_results_long(
        results_root=root, families=["wadd_sampling"], noises=[0.0],
        stimulus_pairs=pairs, n_draws=5, base_seed=0,
    )
    corr_long = pd.concat(
        [long, surfaced_best_rows(long, metric="pearson_r", higher_is_better=True)],
        ignore_index=True,
    )
    s = summarise(corr_long, "pearson_r")
    assert "surfaced (best per run)" in set(s["role"])
    best = s[s.role == "surfaced (best per run)"]["mean"].iloc[0]
    mean = s[s.role == "surfaced"]["mean"].iloc[0]
    assert best >= mean - 1e-9   # best is at least the mean (corr: higher better)


# --------------------------------------------------------------------------
# Sequence-aware recovery for history-dependent ground truths
# (alternating / anti_majority). See scripts/recovery_correlation.py and
# scripts/eval_hilbig.py:simulate_sequence / lag1_autocorr.
# --------------------------------------------------------------------------

VALIDITIES = [0.9, 0.8, 0.7, 0.6]


def test_theory_reads_history_detects_use_not_signature():
    """The detector must fire on actual history USE (history. / history[),
    not on the `history` parameter name that every predict signature has."""
    from scripts.recovery_correlation import theory_reads_history

    alternating = Theory.from_yaml(YAML_DIR / "alternating.yaml")
    anti_majority = Theory.from_yaml(YAML_DIR / "anti_majority.yaml")
    ttb = Theory.from_yaml(YAML_DIR / "ttb.yaml")

    assert theory_reads_history(alternating) is True      # uses history.get(...)
    assert theory_reads_history(anti_majority) is False   # memoryless body
    assert theory_reads_history(ttb) is False             # memoryless body


def test_simulate_sequence_runs_history_theory_without_crash():
    """simulate_sequence must drive a history-reading theory through the
    real experiment loop (populating history['response']) without KeyError,
    returning per-pair proportions and per-subject response sequences."""
    from scripts.recovery_correlation import simulate_sequence, unique_stimulus_pairs

    alternating = Theory.from_yaml(YAML_DIR / "alternating.yaml")
    pairs = unique_stimulus_pairs()
    props, sequences = simulate_sequence(
        alternating, stimulus_pairs=pairs, validities=VALIDITIES,
        n_runs=20, seed=0, action_noise=0.0,
    )
    assert props.shape == (len(pairs),)
    assert np.all(props >= 0.0) and np.all(props <= 1.0)
    assert len(sequences) == 20
    # each subject sees >= len(pairs) trials (pairs are repeated to fill MAX_TRIALS)
    assert all(len(seq) >= len(pairs) for seq in sequences)
    assert all(set(seq) <= {0, 1} for seq in sequences)


def test_alternating_lag1_autocorr_strongly_negative_at_no_noise():
    """A pure alternator's realized responses flip every trial -> lag-1
    autocorrelation ~= -1 with no action noise."""
    from scripts.recovery_correlation import (
        simulate_sequence, lag1_autocorr, unique_stimulus_pairs,
    )

    alternating = Theory.from_yaml(YAML_DIR / "alternating.yaml")
    pairs = unique_stimulus_pairs()
    _, sequences = simulate_sequence(
        alternating, stimulus_pairs=pairs, validities=VALIDITIES,
        n_runs=30, seed=1, action_noise=0.0,
    )
    assert lag1_autocorr(sequences) < -0.9


def test_full_action_noise_erodes_alternation_autocorr():
    """At action_noise=1.0 every choice is a uniform coin, so the serial
    dependency (and thus lag-1 autocorrelation) collapses toward 0."""
    from scripts.recovery_correlation import (
        simulate_sequence, lag1_autocorr, unique_stimulus_pairs,
    )

    alternating = Theory.from_yaml(YAML_DIR / "alternating.yaml")
    pairs = unique_stimulus_pairs()
    _, sequences = simulate_sequence(
        alternating, stimulus_pairs=pairs, validities=VALIDITIES,
        n_runs=60, seed=2, action_noise=1.0,
    )
    assert abs(lag1_autocorr(sequences)) < 0.2


def test_memoryless_sequence_autocorr_near_zero():
    """A memoryless stimulus-dependent model on shuffled trials has no
    serial dependence -> lag-1 autocorrelation ~= 0."""
    from scripts.recovery_correlation import (
        simulate_sequence, lag1_autocorr, unique_stimulus_pairs,
    )

    wadd = Theory.from_yaml(YAML_DIR / "wadd.yaml")
    pairs = unique_stimulus_pairs()
    _, sequences = simulate_sequence(
        wadd, stimulus_pairs=pairs, validities=VALIDITIES,
        n_runs=60, seed=3, action_noise=0.0,
    )
    assert abs(lag1_autocorr(sequences)) < 0.2


def test_row_metrics_routes_by_history_use_and_reports_autocorr():
    """_row_metrics: history-reading theories go through the sequence sim and
    get an empirical autocorr; memoryless theories keep the independent path
    (byte-identical pearson_r) and an analytic 0.0 autocorr."""
    from scripts.recovery_correlation import (
        _row_metrics, unique_stimulus_pairs,
        simulate_choice_proportions, metrics_for_theory,
    )

    pairs = unique_stimulus_pairs()
    target = simulate_choice_proportions(
        Theory.from_yaml(YAML_DIR / "wadd.yaml"), stimulus_pairs=pairs,
        validities=VALIDITIES, rating_max=1, n_draws=50, seed=900_000,
        action_noise=0.0,
    )

    # memoryless: identical to the existing independent path, autocorr == 0.0
    wadd = Theory.from_yaml(YAML_DIR / "wadd.yaml")
    m_ref = metrics_for_theory(
        wadd, target, stimulus_pairs=pairs, validities=VALIDITIES,
        rating_max=1, n_draws=50, seed=123, action_noise=0.0,
    )
    m = _row_metrics(
        wadd, target, stimulus_pairs=pairs, n_draws=50, seed=123,
        action_noise=0.0,
    )
    assert m["lag1_autocorr"] == 0.0
    assert m["pearson_r"] == pytest.approx(m_ref["pearson_r"])
    assert m["mse"] == pytest.approx(m_ref["mse"])

    # history-reading: routed through the sequence sim, finite negative autocorr
    alternating = Theory.from_yaml(YAML_DIR / "alternating.yaml")
    m_alt = _row_metrics(
        alternating, target, stimulus_pairs=pairs, n_draws=30, seed=123,
        action_noise=0.0,
    )
    assert "lag1_autocorr" in m_alt
    assert m_alt["lag1_autocorr"] < -0.5


def test_surfaced_best_autocorr_picks_closest_to_gt():
    """For autocorr, 'best' = the surfaced theory whose lag-1 autocorr is
    closest to that run-dir's gt autocorr (not the most extreme value)."""
    from scripts.recovery_correlation import surfaced_best_autocorr_rows

    long = pd.DataFrame([
        # alternating run A: gt = -1.0; surfaced x=-0.9 (close), y=+0.1 (far)
        dict(family="alternating", noise=0.0, run_dir="A", role="gt",
             theory_label="gt", n_stimuli=24, pearson_r=np.nan, mse=0.0,
             lag1_autocorr=-1.0),
        dict(family="alternating", noise=0.0, run_dir="A", role="surfaced",
             theory_label="x", n_stimuli=24, pearson_r=0.0, mse=0.1,
             lag1_autocorr=-0.9),
        dict(family="alternating", noise=0.0, run_dir="A", role="surfaced",
             theory_label="y", n_stimuli=24, pearson_r=0.0, mse=0.1,
             lag1_autocorr=0.1),
    ])
    best = surfaced_best_autocorr_rows(long)
    assert len(best) == 1
    assert (best["role"] == "surfaced (best per run)").all()
    assert best.iloc[0]["theory_label"] == "x"      # -0.9 is closest to gt -1.0
    assert best.iloc[0]["lag1_autocorr"] == pytest.approx(-0.9)


def test_build_results_long_handles_sequential_families():
    """End-to-end: alternating and anti_majority must build without crashing,
    carry a lag1_autocorr column, and the alternating gt bar must show the
    strong negative serial dependence."""
    from scripts.recovery_correlation import build_results_long, unique_stimulus_pairs

    root = REPO_ROOT / "results" / "recovery"
    pairs = unique_stimulus_pairs()
    df = build_results_long(
        results_root=root, families=["alternating", "anti_majority"],
        noises=[0.0], stimulus_pairs=pairs, n_draws=5, base_seed=0,
    )
    assert "lag1_autocorr" in df.columns
    assert {"seed", "surfaced", "gt"}.issubset(set(df["role"].unique()))
    # alternating gt rows carry a strongly negative autocorr
    alt_gt = df[(df.family == "alternating") & (df.role == "gt")]
    assert not alt_gt.empty
    assert (alt_gt["lag1_autocorr"] < -0.5).all()


def test_plot_grid_drops_noise_axis_when_single_noise_level(tmp_path):
    """For single-noise figures (e.g. non-canonical families that only have
    ε=0), the x-axis 'noise' label and tick labels are meaningless and should
    be suppressed; multi-noise figures keep them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scripts.recovery_correlation import plot_grid

    def make(noises):
        rows = []
        for nz in noises:
            for role in ("seed", "surfaced", "gt"):
                rows.append(dict(family="alternating", noise=nz, role=role,
                                 mean=0.5, sem=0.1))
        return pd.DataFrame(rows)

    single = tmp_path / "single.png"
    plot_grid(make([0.0]), single)
    multi = tmp_path / "multi.png"
    plot_grid(make([0.0, 0.5]), multi)
    assert single.exists() and multi.exists()

    # Re-render single to inspect the live axes (plot_grid closes its own fig).
    fig, ax = plt.subplots()
    sub = make([0.0])
    # Mirror the single-noise branch: no xlabel, no tick labels.
    from scripts.recovery_correlation import _noise_axis_is_informative
    assert _noise_axis_is_informative(make([0.0, 0.5])) is True
    assert _noise_axis_is_informative(make([0.0])) is False
    plt.close(fig)
