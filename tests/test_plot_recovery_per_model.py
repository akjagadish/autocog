"""Tests for scripts/plot_recovery_per_model.py.

The per-model variant takes the same long-format recovery table that
scripts/recovery_correlation.py builds and emits ONE single-panel figure per
ground-truth family (instead of the combined 1xN grid). The risk a unit test
must catch is cross-family contamination: each family's figure — including its
two best-surfaced bars — must be computed from that family's rows ONLY, never
borrowing another family's better/worse value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _two_family_long() -> pd.DataFrame:
    """Synthetic long table with two families whose best surfaced theories
    differ, so a leak between families changes the analytic answer.

    ttb_sampling surfaced Pearson r: run A {0.5, 0.7}, run B {0.6, 0.65}
        -> best per run   = mean(0.7, 0.65) = 0.675
        -> best across run = 0.7 (single global best, in run A)
    wadd_sampling surfaced Pearson r: run A {0.2, 0.95}
        -> best across run = 0.95  (must NOT appear in ttb's figure)
    """
    rows: list[dict] = []

    def add(family, run, role, label, r, m):
        rows.append(dict(
            family=family, noise=0.0, run_dir=run, role=role,
            theory_label=label, n_stimuli=24, pearson_r=r, mse=m,
            lag1_autocorr=0.0,
        ))

    # ttb_sampling
    add("ttb_sampling", "A", "seed", "s", 0.30, 0.20)
    add("ttb_sampling", "A", "surfaced", "a1", 0.50, 0.30)
    add("ttb_sampling", "A", "surfaced", "a2", 0.70, 0.10)
    add("ttb_sampling", "A", "gt", "gt", 0.99, 0.01)
    add("ttb_sampling", "A", "random", "random", float("nan"), 0.25)
    add("ttb_sampling", "B", "seed", "s", 0.30, 0.20)
    add("ttb_sampling", "B", "surfaced", "b1", 0.60, 0.15)
    add("ttb_sampling", "B", "surfaced", "b2", 0.65, 0.12)
    add("ttb_sampling", "B", "gt", "gt", 0.99, 0.01)
    add("ttb_sampling", "B", "random", "random", float("nan"), 0.25)
    # wadd_sampling (only its OWN best should ever surface in its own figure)
    add("wadd_sampling", "A", "seed", "s", 0.10, 0.40)
    add("wadd_sampling", "A", "surfaced", "w1", 0.20, 0.35)
    add("wadd_sampling", "A", "surfaced", "w2", 0.95, 0.02)
    add("wadd_sampling", "A", "gt", "gt", 0.99, 0.01)
    add("wadd_sampling", "A", "random", "random", float("nan"), 0.25)
    return pd.DataFrame(rows)


def test_family_plot_order_canonical_first():
    from scripts.plot_recovery_per_model import family_plot_order

    long = _two_family_long()
    # Add a non-canonical family that is not in GROUND_TRUTHS_PLOT.
    extra = long.iloc[[0]].copy()
    extra["family"] = "alternating"
    long = pd.concat([long, extra], ignore_index=True)
    order = family_plot_order(long["family"])
    # canonical-sampling families keep GROUND_TRUTHS_PLOT order, extras appended
    assert order == ["ttb_sampling", "wadd_sampling", "alternating"]


def test_family_plot_order_respects_priority():
    """family_plot_order orders present families by a caller-supplied priority
    list (default GROUND_TRUTHS_PLOT); unknown families append in first-seen
    order."""
    from scripts.recovery_correlation import family_plot_order

    present = ["anti_majority", "random", "alternating", "extra"]
    pri = ("alternating", "random", "anti_majority")
    assert family_plot_order(present, priority=pri) == [
        "alternating", "random", "anti_majority", "extra",
    ]


def test_noncanonical_panels_use_requested_order():
    """The non-canonical pool panels follow FAMILY_PLOT_ORDER:
    Alternating, Perseveration, Random, Take the worst, Single cue,
    Anti majority (cue_parity removed from the pool)."""
    from scripts.plot_recovery_per_model import (
        FAMILY_PLOT_ORDER, NONCANONICAL_FAMILIES,
    )
    from scripts.recovery_correlation import family_plot_order

    assert "cue_parity" not in NONCANONICAL_FAMILIES
    present = ["anti_majority", "alternating", "random",
               "take_the_worst", "single_cue", "perseveration"]
    assert family_plot_order(present, priority=FAMILY_PLOT_ORDER) == [
        "alternating", "perseveration", "random", "take_the_worst",
        "single_cue", "anti_majority",
    ]


def test_per_model_summary_isolates_family_and_selects_own_best():
    """Building one family's plot table must use only that family's rows: its
    'surfaced (best across runs)' bar equals THAT family's global best, and the
    other family's stronger value never leaks in."""
    from scripts.plot_recovery_per_model import per_model_summary

    long = _two_family_long()
    summ = per_model_summary(
        long, "ttb_sampling", metric="pearson_r", higher_is_better=True,
    )
    # only ttb_sampling rows
    assert set(summ["family"]) == {"ttb_sampling"}

    def mean_of(role):
        sub = summ[summ.role == role]["mean"]
        assert len(sub) == 1
        return float(sub.iloc[0])

    # ttb's own analytic bests, NOT wadd's 0.95
    assert mean_of("surfaced (best across runs)") == pytest.approx(0.70)
    assert mean_of("surfaced (best per run)") == pytest.approx(0.675)
    # plain surfaced mean = mean of per-run surfaced means
    #   run A mean(0.5,0.7)=0.6 ; run B mean(0.6,0.65)=0.625 ; mean=0.6125
    assert mean_of("surfaced") == pytest.approx(0.6125)
    # the leaked value must be absent everywhere
    assert not np.isclose(summ["mean"], 0.95).any()


def test_per_model_summary_include_best_false_excludes_best_roles():
    """With include_best=False the two metric-specific best-surfaced bars are
    NOT added, so only the raw roles (random/seed/surfaced/gt) survive."""
    from scripts.plot_recovery_per_model import per_model_summary

    long = _two_family_long()
    summ = per_model_summary(
        long, "ttb_sampling", metric="mse", higher_is_better=False,
        include_best=False,
    )
    roles = set(summ["role"])
    assert "surfaced (best per run)" not in roles
    assert "surfaced (best across runs)" not in roles
    assert {"random", "seed", "surfaced", "gt"}.issubset(roles)


def test_panel_title_respects_show_title():
    """plot_grid's per-panel title is the uppercased family when show_title is
    True, and suppressed (None) when False — so per-model figures can drop it."""
    from scripts.recovery_correlation import _panel_title

    assert _panel_title("ttb_sampling", True) == "TTB_SAMPLING"
    assert _panel_title("ttb_sampling", False) is None


def test_heuristic_title_mapping():
    """Canonical heuristics get their acronym/name (TTB / WADD / Tallying,
    sampling suffix stripped); every other family is first-letter-capitalised
    with underscores rendered as spaces."""
    from scripts.plot_recovery_per_model import heuristic_title

    assert heuristic_title("ttb") == "TTB"
    assert heuristic_title("ttb_sampling") == "TTB"
    assert heuristic_title("wadd_sampling") == "WADD"
    assert heuristic_title("tallying_sampling") == "Tallying"
    assert heuristic_title("alternating") == "Alternating"
    assert heuristic_title("anti_majority") == "Anti majority"
    assert heuristic_title("take_the_worst") == "Take the worst"
    assert heuristic_title("single_cue") == "Single cue"


def test_baseline_roles_for_family_drops_gt_for_noncanonical():
    """gt is a reference line only for canonical sampling families; the
    non-canonical/sequential baselines drop it (it is not a meaningful per-pair
    ceiling for them). random is unaffected."""
    from scripts.plot_recovery_per_model import baseline_roles_for_family

    assert baseline_roles_for_family("ttb_sampling", ("random", "gt")) == \
        ("random", "gt")
    assert baseline_roles_for_family("wadd_sampling", ("gt",)) == ("gt",)
    assert baseline_roles_for_family("alternating", ("random", "gt")) == \
        ("random",)
    assert baseline_roles_for_family("perseveration", ("gt",)) == ()
    assert baseline_roles_for_family("cue_parity", ("random",)) == ("random",)


def test_pooled_specs_drop_gt_only_for_noncanonical():
    """The canonical pool keeps the gt line; the non-canonical pool drops it
    from every figure. Each pool has five figures (mse, mse-no-random, corr,
    mse-with-best, mse-with-best-no-random); only the with-best ones augment
    the data (include_best=True)."""
    from scripts.plot_recovery_per_model import (
        pooled_specs, CANONICAL_SAMPLING_FAMILIES, NONCANONICAL_FAMILIES,
    )

    canon = {stem: (br, inc) for _m, _y, _r, br, stem, inc in
             pooled_specs(CANONICAL_SAMPLING_FAMILIES, "canonical_pooled")}
    assert canon["recovery_mse_canonical_pooled"] == (("random", "gt"), False)
    assert canon["recovery_mse_canonical_pooled_no_random"] == (("gt",), False)
    assert canon["recovery_correlation_canonical_pooled"] == (("gt",), False)
    assert canon["recovery_mse_canonical_pooled_with_best"] == \
        (("random", "gt"), True)
    assert canon["recovery_mse_canonical_pooled_with_best_no_random"] == \
        (("gt",), True)

    noncanon = {stem: (br, inc) for _m, _y, _r, br, stem, inc in
                pooled_specs(NONCANONICAL_FAMILIES, "noncanonical_pooled")}
    for (br, _inc) in noncanon.values():
        assert "gt" not in br      # gt dropped everywhere for non-canonical
    assert noncanon["recovery_mse_noncanonical_pooled"] == (("random",), False)
    assert noncanon["recovery_mse_noncanonical_pooled_no_random"] == ((), False)
    assert noncanon["recovery_correlation_noncanonical_pooled"] == ((), False)
    assert noncanon["recovery_mse_noncanonical_pooled_with_best"] == \
        (("random",), True)
    assert noncanon["recovery_mse_noncanonical_pooled_with_best_no_random"] == \
        ((), True)


def test_panel_title_uses_formatter_else_upper():
    """_panel_title applies a custom title_formatter when given, else falls
    back to str.upper (the combined-grid behaviour); None when suppressed."""
    from scripts.recovery_correlation import _panel_title
    from scripts.plot_recovery_per_model import heuristic_title

    assert _panel_title("ttb_sampling", True) == "TTB_SAMPLING"        # default
    assert _panel_title("ttb_sampling", True, heuristic_title) == "TTB"  # custom
    assert _panel_title("ttb_sampling", False, heuristic_title) is None  # off


def test_plot_grid_baseline_roles_drawn_as_lines_not_bars(tmp_path):
    """baseline_roles are drawn as per-noise reference lines, so each must be
    disjoint from the bar `roles` (else it would be drawn twice). Multiple
    baseline roles (random floor + gt ceiling) are allowed together."""
    from scripts.recovery_correlation import plot_grid

    summary = pd.DataFrame([
        dict(family="ttb_sampling", noise=n, role=role, mean=0.05, sem=0.01)
        for n in (0.0, 0.5)
        for role in ("seed", "surfaced", "gt", "random")
    ])
    # gt is both a bar role and a baseline role -> double-draw -> error
    with pytest.raises(ValueError):
        plot_grid(summary, tmp_path / "bad.png",
                  roles=("seed", "surfaced", "gt"),
                  baseline_roles=("gt",))

    # disjoint: seed/surfaced bars, random + gt as lines
    out = tmp_path / "ok.png"
    plot_grid(summary, out, roles=("seed", "surfaced"),
              baseline_roles=("random", "gt"))
    assert out.is_file() and out.stat().st_size > 0


def test_roles_and_baseline_lines_per_metric():
    """Both metrics draw only seed/surfaced as bars. MSE adds random + gt
    reference lines (plus a no-random variant); correlation adds only the gt
    line (random is undefined there)."""
    from scripts.plot_recovery_per_model import (
        ROLES_MSE, ROLES_CORR, BASELINE_ROLES_MSE,
        BASELINE_ROLES_MSE_NO_RANDOM, BASELINE_ROLES_CORR,
    )

    assert ROLES_MSE == ("seed", "surfaced")
    assert ROLES_CORR == ("seed", "surfaced")
    assert set(BASELINE_ROLES_MSE) == {"random", "gt"}
    assert BASELINE_ROLES_MSE_NO_RANDOM == ("gt",)
    assert BASELINE_ROLES_CORR == ("gt",)


def test_per_model_summary_mse_picks_min_not_max():
    """For MSE the best is the MINIMUM; the random floor must survive too."""
    from scripts.plot_recovery_per_model import per_model_summary

    long = _two_family_long()
    summ = per_model_summary(
        long, "ttb_sampling", metric="mse", higher_is_better=False,
    )
    best_across = summ[summ.role == "surfaced (best across runs)"]["mean"].iloc[0]
    # global min surfaced mse for ttb_sampling = 0.10 (theory a2)
    assert best_across == pytest.approx(0.10)
    assert "random" in set(summ["role"])


def test_plot_per_model_writes_one_named_figure_per_family(tmp_path):
    from scripts.plot_recovery_per_model import plot_per_model

    long = _two_family_long()
    out = plot_per_model(
        long, out_dir=tmp_path, metric="mse", higher_is_better=False,
        roles=("random", "seed", "surfaced", "surfaced (best per run)",
               "surfaced (best across runs)", "gt"),
        ylabel="MSE", filename_prefix="recovery_mse",
    )
    # one figure per family, named recovery_mse_<family>.png (+ .svg sibling)
    assert {p.name for p in out} == {
        "recovery_mse_ttb_sampling.png", "recovery_mse_wadd_sampling.png",
    }
    for p in out:
        assert p.is_file() and p.stat().st_size > 0
        assert p.with_suffix(".svg").is_file()


def test_figure_width_default_and_override():
    """plot_grid's figure width is 6in/panel by default, or a fixed total when
    `fig_width` is given (so the pooled non-canonical figure can be sized to
    ~2 canonical panels regardless of panel count)."""
    from scripts.recovery_correlation import _figure_width

    assert _figure_width(3, None) == 18.0      # 6in * 3 panels
    assert _figure_width(6, 12.0) == 12.0      # fixed total, 6 panels
    assert _figure_width(8, 12.0) == 12.0      # same total, 8 panels


def _noncanonical_long(families=("alternating", "cue_parity")) -> pd.DataFrame:
    """Synthetic single-noise long table for a few non-canonical families,
    including the random (NaN-corr) rows the MSE baseline line reads."""
    rows = []
    roles = {
        "seed": (0.30, 0.20), "surfaced": (0.50, 0.10),
        "gt": (0.90, 0.02), "random": (float("nan"), 0.25),
    }
    for fam in families:
        for run in ("A", "B"):
            for role, (r, m) in roles.items():
                rows.append(dict(
                    family=fam, noise=0.0, run_dir=run, role=role,
                    theory_label=role, n_stimuli=24, pearson_r=r, mse=m,
                    lag1_autocorr=0.0,
                ))
    return pd.DataFrame(rows)


def test_plot_pooled_writes_all_variants(tmp_path):
    """plot_pooled emits FOUR combined figures over the pooled families: MSE
    (random + gt lines), a no-random MSE variant, the correlation figure, and
    a with-best MSE figure (adds the best-surfaced-per-run bar)."""
    from scripts.plot_recovery_per_model import plot_pooled

    out = plot_pooled(_noncanonical_long(), out_dir=tmp_path, fig_width=12.0)
    assert {p.name for p in out} == {
        "recovery_mse_noncanonical_pooled.png",
        "recovery_mse_noncanonical_pooled_no_random.png",
        "recovery_correlation_noncanonical_pooled.png",
        "recovery_mse_noncanonical_pooled_with_best.png",
        "recovery_mse_noncanonical_pooled_with_best_no_random.png",
    }
    for p in out:
        assert p.is_file() and p.stat().st_size > 0
        assert p.with_suffix(".svg").is_file()


def test_role_label_display_names():
    """Legend display names: capitalised, gt -> 'Ground truth', the best-per-run
    role -> 'Surfaced (Best)'; unknown roles fall back to themselves."""
    from scripts.figure_style import role_label

    assert role_label("gt") == "Ground truth"
    assert role_label("seed") == "Seed"
    assert role_label("surfaced") == "Surfaced"
    assert role_label("surfaced (best per run)") == "Surfaced (Best)"
    assert role_label("random") == "Random"
    assert role_label("unmapped_role") == "unmapped_role"


def test_plot_pooled_skips_when_no_matching_families(tmp_path):
    """If the long table has no non-canonical rows, plot_pooled writes nothing
    (rather than crashing on an empty 1x0 subplot grid)."""
    from scripts.plot_recovery_per_model import plot_pooled

    canonical_only = _noncanonical_long(families=("ttb_sampling",))
    out = plot_pooled(canonical_only, out_dir=tmp_path)
    assert out == []
    assert not list(tmp_path.glob("*.png"))


def test_main_from_csv_emits_canonical_and_noncanonical_pools(tmp_path):
    """main(--from-csv) writes BOTH pooled figures: the non-canonical pool and
    the canonical pool (multi-noise, full per-panel width)."""
    from scripts.plot_recovery_per_model import main

    rows = []
    roles = {
        "seed": (0.3, 0.2), "surfaced": (0.5, 0.1),
        "gt": (0.9, 0.02), "random": (float("nan"), 0.25),
    }
    for fam, noises in (("ttb_sampling", [0.0, 0.5]), ("alternating", [0.0])):
        for nz in noises:
            for run in ("A", "B"):
                for role, (r, m) in roles.items():
                    rows.append(dict(
                        family=fam, noise=nz, run_dir=run, role=role,
                        theory_label=role, n_stimuli=24, pearson_r=r, mse=m,
                        lag1_autocorr=0.0,
                    ))
    csv = tmp_path / "long.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out_dir = tmp_path / "per_model"

    rc = main(["--from-csv", str(csv), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "recovery_mse_noncanonical_pooled.png").is_file()
    assert (out_dir / "recovery_mse_canonical_pooled.png").is_file()
    assert (out_dir / "recovery_correlation_canonical_pooled.png").is_file()


def test_random_in_noncanonical_families():
    """The synthetic 'random' family is part of the non-canonical pool."""
    from scripts.plot_recovery_per_model import (
        NONCANONICAL_FAMILIES, CANONICAL_SAMPLING_FAMILIES,
    )

    assert "random" in NONCANONICAL_FAMILIES
    assert "random" not in CANONICAL_SAMPLING_FAMILIES


def test_main_assembles_random_family_into_long_df(tmp_path):
    """main() concatenates the synthetic 'random' family (sourced from the
    canonical ε=1 runs vs a 0.5 reference) into the long table and writes its
    per-model MSE figure. Correlation for random is all-NaN, so no
    recovery_correlation_random figure is written."""
    from scripts.plot_recovery_per_model import main

    root = REPO_ROOT / "results" / "recovery"
    csv = tmp_path / "long.csv"
    out_dir = tmp_path / "per_model"
    rc = main([
        "--results-root", str(root),
        "--families", "ttb_sampling", "--noises", "1.0",
        "--n-draws", "3", "--seed", "0",
        "--out-dir", str(out_dir), "--csv", str(csv),
    ])
    assert rc == 0
    df = pd.read_csv(csv)
    assert "random" in set(df["family"])
    # random rows: constant-0.5 reference -> NaN correlation, finite mse.
    rand = df[df["family"] == "random"]
    assert rand["pearson_r"].isna().all()
    assert (out_dir / "recovery_mse_random.png").is_file()
    assert not (out_dir / "recovery_correlation_random.png").is_file()


def test_main_end_to_end_per_model(tmp_path):
    """End-to-end on a real (small) family: writes per-model mse + corr
    figures into the requested out-dir, one each per family."""
    from scripts.plot_recovery_per_model import main

    out_dir = tmp_path / "per_model"
    rc = main([
        "--results-root",
        str(REPO_ROOT / "results" / "recovery"),
        "--families", "ttb_sampling",
        "--noises", "0.0",
        "--n-draws", "3",
        "--seed", "0",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "recovery_mse_ttb_sampling.png").is_file()
    assert (out_dir / "recovery_correlation_ttb_sampling.png").is_file()
    assert (out_dir / "recovery_mse_ttb_sampling.svg").is_file()
