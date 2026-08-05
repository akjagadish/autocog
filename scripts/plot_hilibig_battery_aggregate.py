"""Plot base vs surfaced MAE / MSE to ground truth, across noise levels.

Reads `results/hilibig_battery_aggregate.csv` (the output of
`scripts/aggregate_hilibig_battery.py`) and produces a 2×3 bar-chart grid:

    rows = {MAE, MSE}
    cols = ground_truth ∈ {ttb, wadd, tallying}
    x    = noise level
    bars = role ∈ {base, surfaced, gt-floor}

The semicolon-joined `mae_subjects` / `mse_subjects` columns are exploded
to one row per subject before grouping, so the error bars (SEM) pool
variance across runs × slots × subjects rather than treating each run
as a single datum. The `gt-floor` role is included as a visual reference
for the irreducible within-family variability.

CLI:
    python scripts/plot_hilibig_battery_aggregate.py \\
        [--aggregate-csv results/hilibig_battery_aggregate.csv] \\
        [--out results/hilibig_battery_base_vs_surfaced.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import FONTSIZE, ROLE_COLOR, save_figure, style_axes


GROUND_TRUTHS: tuple[str, ...] = ("ttb", "wadd", "tallying")
ROLES: tuple[str, ...] = ("base", "surfaced", "gt-floor")


def explode_subjects(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Split a `';'`-joined per-subject column into one row per subject.

    The aggregate CSV stores `mae_subjects` / `mse_subjects` as
    semicolon-joined floats (length = n_subjects per run). We split +
    explode so the downstream groupby averages over runs × slots ×
    subjects naturally.
    """
    out = df[df[col].notna() & (df[col] != "")].copy()
    out[col] = out[col].str.split(";").apply(
        lambda xs: [float(x) for x in xs]
    )
    out = out.explode(col, ignore_index=True)
    out[col] = out[col].astype(float)
    return out


def summarise(
    df: pd.DataFrame, metric: str,
) -> pd.DataFrame:
    """Return per (ground_truth, noise, role) mean and SEM of `metric`.

    Pools per-subject values across runs and positional slots.
    """
    col = f"{metric}_subjects"
    long = explode_subjects(
        df[["ground_truth", "noise", "role", col]], col,
    )
    long = long.rename(columns={col: metric})
    grouped = (
        long
        .groupby(["ground_truth", "noise", "role"], observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"]).where(
        grouped["count"] > 1, other=np.inf,
    )
    return grouped


def plot_grid(
    df: pd.DataFrame, out_path: Path, *, title: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(GROUND_TRUTHS),
                             figsize=(6 * len(GROUND_TRUTHS), 8),
                             sharey="row", sharex=True)

    for row, metric in enumerate(("mae", "mse")):
        summary = summarise(df, metric)
        for col, gt in enumerate(GROUND_TRUTHS):
            ax = axes[row][col]
            sub = summary[summary.ground_truth == gt]
            noises = sorted(sub.noise.unique())

            x = np.arange(len(noises))
            width = 0.8 / len(ROLES)

            for i, role in enumerate(ROLES):
                role_rows = sub[sub.role == role].set_index("noise")
                means = np.array(
                    [role_rows.loc[n, "mean"] if n in role_rows.index else np.nan
                     for n in noises], dtype=float,
                )
                sems = np.array(
                    [role_rows.loc[n, "sem"] if n in role_rows.index else 0.0
                     for n in noises], dtype=float,
                )
                ax.bar(
                    x - 0.4 + width * (i + 0.5), means, width,
                    yerr=sems,
                    color=ROLE_COLOR[role],
                )

            ax.set_xticks(x)
            ax.set_xticklabels([f"{n:g}" for n in noises])
            style_axes(
                ax,
                title=gt.upper() if row == 0 else None,
                xlabel="ground-truth action noise (ε)" if row == 1 else None,
            )

        axes[row][0].set_ylabel(metric.upper(), fontsize=FONTSIZE)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ROLE_COLOR[r], label=r) for r in ROLES]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(ROLES),
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=FONTSIZE - 2,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Plot base vs surfaced MAE/MSE to ground truth, across noise "
            "levels, from the hilibig_battery aggregate CSV."
        )
    )
    p.add_argument(
        "--aggregate-csv", type=Path,
        default=Path("results/hilibig_battery_aggregate.csv"),
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/hilibig_battery_base_vs_surfaced.png"),
    )
    p.add_argument(
        "--title", type=str,
        default=(
            "Base vs surfaced vs ground truth — "
            "per-subject MAE / MSE pooled across runs and slots"
        ),
    )
    args = p.parse_args(argv)

    df = pd.read_csv(args.aggregate_csv)
    # Noise column is numeric (0.0, 0.05, 0.3); keep it float so sorted
    # ordering matches the sweep's noise levels.
    df["noise"] = df["noise"].astype(float)
    plot_grid(df, args.out, title=args.title)
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
