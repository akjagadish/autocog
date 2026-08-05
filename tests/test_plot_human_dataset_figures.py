"""Tests for the human-dataset figure assembly script.

The script is a thin curation layer over already-tested drawers, so the
meaningful new logic is: (1) deriving headline/other surfaced roles from the
lineage colours, (2) picking the final cycle, and (3) handing each drawer the
publication contract (no titles, exclude the 'other' model from the MSE bar,
hide bar x-labels). We assert those contracts analytically (exact role labels,
exact kwargs) and smoke-test that every figure writes PNG+SVG.
"""
import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import pandas as pd
import pytest

from scripts import plot_human_dataset_figures as H
from scripts.figure_style import NEUTRAL_HARMONY

SANDY = NEUTRAL_HARMONY["sandy"]
TERRA = NEUTRAL_HARMONY["terracotta"]
SLATE = NEUTRAL_HARMONY["slate"]

WADD = Path("results/online/dmb/firebase_prolific/prolific1/ttb+wadd")
TALLY = Path(
    "results/heuristic_decision_making/humans/hdm_full_prolific_run_full/"
    "ttb+tallying"
)


# --- synthetic frames matching the committed CSV schemas -------------------
def _traj_df() -> pd.DataFrame:
    rows = []
    for c in range(1, 6):
        for metric in ("mse", "pearson_r"):
            rows.append(dict(cutoff_round=c, model="pi_4", role="surfaced",
                             metric=metric, mean_across_exps=0.05 * c,
                             sem_across_exps=0.01))
            rows.append(dict(cutoff_round=c, model="pi_1", role="seed",
                             metric=metric, mean_across_exps=0.1,
                             sem_across_exps=0.0))
    return pd.DataFrame(rows)


def _calib_df() -> pd.DataFrame:
    rows = []
    for c in (1, 3, 5):
        for k in range(4):
            rows.append(dict(cutoff_round=c, model="pi_4", role="surfaced",
                             p_b_human=0.25 * k, p_b_human_sem=0.02,
                             p_b_model=0.25 * k + 0.03, p_b_model_sem=0.02))
    return pd.DataFrame(rows)


def _per_stim() -> pd.DataFrame:
    rows = []
    for m, role in (("pi_1", "seed"), ("pi_4", "surfaced")):
        for k in range(5):
            rows.append(dict(model=m, role=role,
                             p_b_human=0.2 * k, p_b_model=0.2 * k + 0.03))
    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    return pd.DataFrame([
        dict(model="pi_1", role="seed", mae=0.20, mse=0.06, pearson_r=0.78),
        dict(model="pi_2", role="seed", mae=0.20, mse=0.05, pearson_r=0.91),
        dict(model="pi_4", role="surfaced", mae=0.19, mse=0.04, pearson_r=0.96),
        dict(model="pi_7", role="surfaced", mae=0.10, mse=0.03, pearson_r=0.93),
    ])


# --- role derivation (analytic, exact) -------------------------------------
def test_roles_from_colors_sandy_is_headline_terracotta_is_other():
    cmap = {"pi_1": SLATE, "pi_2": SLATE, "pi_4": SANDY, "pi_7": TERRA}
    assert H._roles_from_colors(cmap) == ("pi_4", "pi_7")


def test_roles_from_colors_independent_of_label_order():
    cmap = {"pi_1": SLATE, "pi_6": TERRA, "pi_7": SANDY}
    assert H._roles_from_colors(cmap) == ("pi_7", "pi_6")


def test_roles_from_colors_requires_one_headline_and_one_other():
    with pytest.raises(ValueError):
        H._roles_from_colors({"pi_1": SLATE, "pi_4": SANDY})  # no terracotta


def test_final_cutoff_is_the_max_round():
    assert H._final_cutoff(pd.DataFrame({"cutoff_round": [1, 3, 5, 3]})) == 5


@pytest.mark.skipif(not WADD.exists(), reason="run folder absent")
def test_roles_on_real_wadd_run():
    from scripts.plot_autocog_convergence import theory_colors
    assert H._roles_from_colors(theory_colors(WADD)) == ("pi_4", "pi_7")


@pytest.mark.skipif(not TALLY.exists(), reason="run folder absent")
def test_roles_on_real_tallying_run():
    # Different surfaced labels; pi_7 is the gold headline there.
    from scripts.plot_autocog_convergence import theory_colors
    assert H._roles_from_colors(theory_colors(TALLY)) == ("pi_7", "pi_6")


# --- per-plot contracts (exact kwargs handed to the drawers) ---------------
def test_metric_over_cycles_requests_publication_contract(tmp_path, monkeypatch):
    seen = {}

    def fake(ax, traj_df, metric, nice, **kw):
        seen.update(kw)
        seen["metric"] = metric

    monkeypatch.setattr(H, "_draw_metric_trajectory_panel", fake)
    H.plot_metric_over_cycles(
        _traj_df(), "mse", "MSE", tmp_path / "mse.png", theory_color={},
    )
    assert seen["metric"] == "mse"
    assert seen["show_title"] is False
    assert seen["with_legend"] is False
    assert seen["xlabel"] == "cycles"
    assert seen["integer_xticks"] is True


def test_final_calibration_uses_only_final_cutoff_and_no_title(
    tmp_path, monkeypatch,
):
    seen = {}

    def fake(df, out_dir, prefix, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(H, "plot_calibration_per_model", fake)
    H.plot_final_calibration_scatter(
        _calib_df(), tmp_path, "calibration_final", theory_color={},
    )
    assert seen["cutoffs"] == [5]
    assert seen["show_title"] is False


def test_pb_scatter_drops_per_panel_titles(tmp_path, monkeypatch):
    seen = {}

    def fake(per_stim_df, out_dir, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(H, "plot_scatter_per_model", fake)
    H.plot_pb_scatter(_per_stim(), tmp_path, color_map={}, name_map={})
    assert seen["show_title"] is False
    assert seen["stem"] == "pb_scatter"


def test_mse_bar_excludes_other_and_hides_labels(tmp_path, monkeypatch):
    seen = {}

    def fake(summary_df, path, **kw):
        seen.update(kw)

    monkeypatch.setattr(H, "plot_bars", fake)
    H.plot_mse_bar(
        _summary(), _per_stim(), tmp_path / "mse_bar.png", other_label="pi_7",
        color_map={}, name_map={},
    )
    assert seen["metric"] == "mse"
    assert seen["exclude_models"] == ("pi_7",)
    assert seen["show_model_labels"] is False


# --- smoke: real drawers end-to-end write PNG + SVG ------------------------
def test_metric_over_cycles_writes_png_and_svg(tmp_path):
    H.plot_metric_over_cycles(
        _traj_df(), "pearson_r", "Pearson r",
        tmp_path / "pearson_r_over_cycles.png", theory_color={"pi_4": SANDY},
    )
    assert (tmp_path / "pearson_r_over_cycles.png").exists()
    assert (tmp_path / "pearson_r_over_cycles.svg").exists()


def test_final_calibration_writes_one_file_per_surfaced_model(tmp_path):
    H.plot_final_calibration_scatter(
        _calib_df(), tmp_path, "calibration_final",
        theory_color={"pi_4": SANDY},
    )
    assert (tmp_path / "calibration_final_pi_4.png").exists()
    assert (tmp_path / "calibration_final_pi_4.svg").exists()


def test_pb_scatter_writes_a_file_per_model(tmp_path):
    H.plot_pb_scatter(
        _per_stim(), tmp_path,
        color_map={"pi_1": SLATE, "pi_4": SANDY}, name_map={},
    )
    for m in ("pi_1", "pi_4"):
        assert (tmp_path / f"pb_scatter_{m}.png").exists()
        assert (tmp_path / f"pb_scatter_{m}.svg").exists()


def test_mse_bar_writes_png_and_svg(tmp_path):
    H.plot_mse_bar(
        _summary(), _per_stim(), tmp_path / "mse_bar.png", other_label="pi_7",
        color_map={"pi_1": SLATE, "pi_2": SLATE, "pi_4": SANDY}, name_map={},
    )
    assert (tmp_path / "mse_bar.png").exists()
    assert (tmp_path / "mse_bar.svg").exists()


# --- integration: the full per-folder file set on the real run -------------
@pytest.mark.skipif(not WADD.exists(), reason="run folder absent")
def test_build_figures_emits_the_curated_set(tmp_path):
    H.build_figures("firebase_prolific", WADD, tmp_path, refresh=False)
    expected = [
        "mse_over_cycles", "pearson_r_over_cycles",
        "calibration_final_pi_4", "calibration_final_pi_7",
        "pb_scatter_pi_1", "pb_scatter_pi_2",
        "pb_scatter_pi_4", "pb_scatter_pi_7",
        "mse_bar",
    ]
    for stem in expected:
        assert (tmp_path / f"{stem}.png").exists(), stem
        assert (tmp_path / f"{stem}.svg").exists(), stem
    # MSE bar drops the 'other' surfaced model (pi_7); not assertable from the
    # PNG, but the kwarg contract is covered by the unit test above.
