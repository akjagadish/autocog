"""Assemble the curated publication figures for the human-dataset autocog runs.

One entry point that regenerates, for each human-data run, the figures we show
in the paper: MSE and Pearson-r over cycles, the final-cycle calibration scatter
for the two surfaced theories, the eval_hilbig P(B) scatter per model, and the
MSE bar (seeds + the headline surfaced model only). It is a thin curation layer:
every figure is drawn by an already-tested drawer from `eval_human_trajectory`
/ `eval_hilbig`, so styling stays identical to the rest of the project. By
default it replots from the committed CSVs (fast); `--refresh` re-runs the eval
pipelines first.

The headline (gold/sandy) and other (peach/terracotta) surfaced models are
derived per run from the lineage colours (`theory_colors`), so nothing is
hardcoded to a particular `pi_N`:

    firebase_prolific (ttb+wadd):          headline pi_4 NLSWM, other pi_7 PSMM
    hdm_full_prolific_run_full (ttb+tallying):
        headline pi_7 Diminishing Returns WADD, other pi_6 Satisficing WADD

Usage:
    python scripts/plot_human_dataset_figures.py            # both runs, replot
    python scripts/plot_human_dataset_figures.py --refresh  # re-run eval first
    python scripts/plot_human_dataset_figures.py --dataset firebase_prolific
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_human_trajectory import (  # noqa: E402
    _draw_metric_trajectory_panel,
    plot_calibration_per_model,
)
from scripts.eval_hilbig import (  # noqa: E402
    build_name_map,
    per_model_mse_sem,
    plot_bars,
    plot_mse_swarm,
    plot_scatter_per_model,
    role_color_map,
)
from scripts.figure_style import NEUTRAL_HARMONY, save_figure  # noqa: E402
from scripts.plot_autocog_convergence import theory_colors  # noqa: E402

SANDY = NEUTRAL_HARMONY["sandy"]        # headline (gold) survivor
TERRACOTTA = NEUTRAL_HARMONY["terracotta"]  # other (peach) survivor

# (name, run_dir) for each human dataset.
HUMAN_DATASETS: list[tuple[str, str]] = [
    ("firebase_prolific",
     "results/human_decision_making_binary/ttb+wadd"),
    ("hdm_full_prolific_run_full",
     "results/human_decision_making_cardinal/ttb+tallying"),
]


# ---------------------------------------------------------------------------
# Role / cycle derivation.
# ---------------------------------------------------------------------------
def _roles_from_colors(color_map: dict[str, str]) -> tuple[str, str]:
    """`(headline, other)` surfaced labels from a lineage colour map: the
    sandy (gold) label is the headline, the terracotta (peach) label is the
    other final survivor. Exactly one of each is expected."""
    headline = [lbl for lbl, c in color_map.items() if c == SANDY]
    other = [lbl for lbl, c in color_map.items() if c == TERRACOTTA]
    if len(headline) != 1 or len(other) != 1:
        raise ValueError(
            f"expected exactly one headline (sandy) and one other "
            f"(terracotta) surfaced model, got headline={headline} "
            f"other={other}"
        )
    return headline[0], other[0]


def _final_cutoff(df: pd.DataFrame) -> int:
    """The last cycle present in a trajectory / calibration frame."""
    return int(df["cutoff_round"].max())


# ---------------------------------------------------------------------------
# One function per plot type (each delegates to an existing tested drawer).
# ---------------------------------------------------------------------------
def plot_metric_over_cycles(
    traj_df: pd.DataFrame, metric: str, nice: str, out_path: Path, *,
    theory_color: dict[str, str],
) -> None:
    """One metric's trajectory over cycles: no title, "cycles" x-axis, integer
    ticks, no legend (publication styling)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    _draw_metric_trajectory_panel(
        ax, traj_df, metric, nice,
        seeds_as_line=False, theory_color=theory_color,
        with_legend=False, xlabel="cycles", show_title=False,
        integer_xticks=True,
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_final_calibration_scatter(
    calib_df: pd.DataFrame, out_dir: Path, prefix: str, *,
    theory_color: dict[str, str],
) -> list[Path]:
    """Final-cycle calibration scatter (model vs human P(B)) per surfaced model,
    one standalone figure each, no title."""
    return plot_calibration_per_model(
        calib_df, out_dir, prefix, cutoffs=[_final_cutoff(calib_df)],
        theory_color=theory_color, show_title=False,
        ground_truth_label="Human",
    )


def plot_pb_scatter(
    per_stim_df: pd.DataFrame, out_dir: Path, *,
    color_map: dict[str, str], name_map: dict[str, str],
) -> list[Path]:
    """eval_hilbig P(B) model-vs-human scatter, one standalone figure per
    evaluated model (surfaced + seeds), no title."""
    return plot_scatter_per_model(
        per_stim_df, out_dir, color_map=color_map, name_map=name_map,
        stem="pb_scatter", show_title=False,
    )


def plot_mse_bar(
    summary_df: pd.DataFrame, per_stim_df: pd.DataFrame, out_path: Path, *,
    other_label: str, color_map: dict[str, str], name_map: dict[str, str],
) -> None:
    """MSE bar vs humans, keeping seeds + the headline surfaced model only
    (the `other` surfaced model is dropped), with no model-name x labels.
    Error bars are the SEM of each model's per-pair squared errors."""
    sem = per_model_mse_sem(per_stim_df).to_dict()
    plot_bars(
        summary_df, out_path, metric="mse", title=None,
        color_map=color_map, name_map=name_map,
        exclude_models=(other_label,), show_model_labels=False, yerr=sem,
    )


def plot_mse_swarm_fig(
    per_stim_df: pd.DataFrame, out_path: Path, *,
    other_label: str, color_map: dict[str, str], name_map: dict[str, str],
) -> None:
    """MSE bar with the underlying per-pair squared errors overlaid as a
    swarm — same model selection (seeds + headline surfaced, the `other`
    surfaced dropped) and styling as `plot_mse_bar`."""
    plot_mse_swarm(
        per_stim_df, out_path, color_map=color_map, name_map=name_map,
        exclude_models=(other_label,), show_model_labels=False,
    )


# ---------------------------------------------------------------------------
# Per-folder driver.
# ---------------------------------------------------------------------------
def _refresh(run_dir: Path) -> None:
    """Re-run the eval pipelines so the CSVs are current before replotting."""
    from scripts import eval_hilbig, eval_run_self
    eval_run_self.main([
        "--run-dir", str(run_dir), "--which", "all",
        "--ground-truth-label", "Human",
    ])
    eval_hilbig.main(["--run-dir", str(run_dir)])


def build_figures(
    name: str, run_dir: Path, out_dir: Path, *, refresh: bool = False,
) -> None:
    """Write the curated figure set for one human-dataset run into `out_dir`."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if refresh:
        _refresh(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theory_color = theory_colors(run_dir)
    _, other = _roles_from_colors(theory_color)
    name_map = build_name_map(run_dir)

    self_dir = run_dir / "analysis" / "eval_run_self"
    traj = pd.read_csv(self_dir / "trajectory_all.csv")
    calib = pd.read_csv(self_dir / "calibration_points_all.csv")
    plot_metric_over_cycles(
        traj, "mse", r"$MSE_{\hat{p}(B)}$", out_dir / "mse_over_cycles.png",
        theory_color=theory_color,
    )
    plot_metric_over_cycles(
        traj, "pearson_r", r"$r_{\hat{p}(B)}$",
        out_dir / "pearson_r_over_cycles.png",
        theory_color=theory_color,
    )
    plot_final_calibration_scatter(
        calib, out_dir, "calibration_final", theory_color=theory_color,
    )

    hilbig_dir = run_dir / "analysis" / "eval_hilbig"
    per_stim = pd.read_csv(hilbig_dir / "eval_hilbig_per_stimulus.csv")
    summary = pd.read_csv(hilbig_dir / "eval_hilbig_summary.csv")
    color_map = role_color_map(list(summary["model"]), theory_color)
    plot_pb_scatter(
        per_stim, out_dir, color_map=color_map, name_map=name_map,
    )
    plot_mse_bar(
        summary, per_stim, out_dir / "mse_bar.png", other_label=other,
        color_map=color_map, name_map=name_map,
    )
    plot_mse_swarm_fig(
        per_stim, out_dir / "mse_swarm.png", other_label=other,
        color_map=color_map, name_map=name_map,
    )
    print(f"[human-figs] {name}: wrote curated figures under {out_dir}/")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh", action="store_true",
        help="Re-run eval_run_self + eval_hilbig before replotting.",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Only build this dataset (by name); default builds both.",
    )
    args = p.parse_args(argv)

    for name, run_dir in HUMAN_DATASETS:
        if args.dataset and name != args.dataset:
            continue
        out_dir = Path(run_dir) / "analysis" / "figures"
        build_figures(name, Path(run_dir), out_dir, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
