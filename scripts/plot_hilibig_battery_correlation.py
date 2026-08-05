"""Scatter-plot base-model vs surfaced-model error at two granularities.

Reads `results/hilibig_battery_aggregate.csv` and, for each metric
(MAE, MSE), produces two scatters:

  Row 1 — per-run:      one dot per run (base and surfaced pooled over
                        subjects × slots within the run).
  Row 2 — per-subject:  one dot per (run × subject), pooled across all
                        runs and all model slots within each role.
                        Shows the full cloud of paired per-subject
                        datapoints, not just run-level means.

Color = ground-truth family. Marker = noise level. Dotted y = x line
included for reference (points below = surfaced beats base).

CLI:
    python scripts/plot_hilibig_battery_correlation.py \\
        [--aggregate-csv results/hilibig_battery_aggregate.csv] \\
        [--out results/hilibig_battery_correlation.png]
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

from scripts.figure_style import (
    FAMILY_COLOR,
    FONTSIZE,
    GRAY,
    save_figure,
    style_axes,
)

NOISE_MARKER: dict[float, str] = {
    0.0: "o",
    0.05: "s",
    0.3: "^",
}


def _explode_subjects(
    df: pd.DataFrame, col: str, *, keep_subject_idx: bool = False,
) -> pd.DataFrame:
    """Split semicolon-joined per-subject arrays into one row per subject.

    If `keep_subject_idx=True`, also attach a `subject_idx` column, where
    subject_idx is the 0-based position within each group of rows that
    share (ground_truth, noise, run_id, role, slot). Subjects are seeded
    deterministically in the battery, so subject_idx is comparable across
    roles within a run (subject 0 sees the same trials as subject 0 in
    the other role).
    """
    out = df[df[col].notna() & (df[col] != "")].copy()
    out[col] = out[col].str.split(";").apply(
        lambda xs: [float(x) for x in xs]
    )
    out = out.explode(col, ignore_index=True)
    out[col] = out[col].astype(float)
    if keep_subject_idx:
        out["subject_idx"] = out.groupby(
            ["ground_truth", "noise", "run_id", "role", "slot"],
            observed=True,
        ).cumcount()
    return out


def per_run_pairs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (ground_truth, noise, run_id): base and surfaced as columns,
    each the mean of `metric` pooled over that run's subjects × slots."""
    col = f"{metric}_subjects"
    keep = df[df["role"].isin(["base", "surfaced"])][
        ["ground_truth", "noise", "run_id", "role", col]
    ]
    long = _explode_subjects(keep, col).rename(columns={col: metric})
    wide = (
        long
        .groupby(["ground_truth", "noise", "run_id", "role"], observed=True)[metric]
        .mean()
        .unstack("role")
        .reset_index()
    )
    return wide.dropna(subset=["base", "surfaced"])


def per_subject_pairs(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (run × subject): base and surfaced each averaged over
    that role's model slots for that subject. Subjects are aligned across
    roles by `subject_idx` (shared seed → shared trajectory sampling)."""
    col = f"{metric}_subjects"
    keep = df[df["role"].isin(["base", "surfaced"])][
        ["ground_truth", "noise", "run_id", "role", "slot", col]
    ]
    long = _explode_subjects(keep, col, keep_subject_idx=True).rename(
        columns={col: metric},
    )
    # Mean over slots within (run, role, subject_idx).
    per_subj = (
        long
        .groupby(
            ["ground_truth", "noise", "run_id", "role", "subject_idx"],
            observed=True,
        )[metric]
        .mean()
        .reset_index()
    )
    wide = per_subj.pivot_table(
        index=["ground_truth", "noise", "run_id", "subject_idx"],
        columns="role", values=metric,
    ).reset_index()
    return wide.dropna(subset=["base", "surfaced"])


def _corr_text(x: pd.Series, y: pd.Series) -> str:
    """Pearson + Spearman via pandas (avoids scipy dep)."""
    r = x.corr(y, method="pearson")
    rho = x.corr(y, method="spearman")
    return f"Pearson r={r:+.2f}   Spearman ρ={rho:+.2f}   n={len(x)}"


def _scatter_one(
    ax, pairs: pd.DataFrame, *, metric: str, subtitle: str, marker_size: float,
    alpha: float,
) -> None:
    for family in sorted(pairs.ground_truth.unique()):
        for noise in sorted(pairs.noise.unique()):
            sub = pairs[
                (pairs.ground_truth == family) & (pairs.noise == noise)
            ]
            if sub.empty:
                continue
            ax.scatter(
                sub["base"], sub["surfaced"],
                c=FAMILY_COLOR.get(family, GRAY),
                marker=NOISE_MARKER.get(float(noise), "o"),
                s=marker_size, alpha=alpha,
                edgecolor="black", linewidth=0.3,
            )

    lo = float(min(pairs["base"].min(), pairs["surfaced"].min()))
    hi = float(max(pairs["base"].max(), pairs["surfaced"].max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        color=GRAY, linestyle=":", linewidth=1.0, zorder=0,
    )

    style_axes(
        ax,
        xlabel=f"base {metric.upper()}",
        ylabel=f"surfaced {metric.upper()}",
    )
    ax.set_title(
        f"{subtitle}\n{_corr_text(pairs['base'], pairs['surfaced'])}",
        fontsize=FONTSIZE - 6,
    )


def plot_correlation(
    df: pd.DataFrame, out_path: Path, *, title: str | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, metric in enumerate(("mae", "mse")):
        _scatter_one(
            axes[0][col], per_run_pairs(df, metric),
            metric=metric, subtitle="per-run mean (one dot = one run)",
            marker_size=80, alpha=0.85,
        )
        _scatter_one(
            axes[1][col], per_subject_pairs(df, metric),
            metric=metric, subtitle="per-subject (pooled across runs × slots)",
            marker_size=18, alpha=0.5,
        )

    family_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=FAMILY_COLOR[f], markeredgecolor="black",
            markersize=9, linestyle="None", label=f.upper(),
        )
        for f in FAMILY_COLOR
    ]
    noise_handles = [
        Line2D(
            [0], [0], marker=NOISE_MARKER[n], color="black",
            markerfacecolor="white", markersize=8,
            linestyle="None", label=f"ε={n:g}",
        )
        for n in NOISE_MARKER
    ]
    fig.legend(
        handles=family_handles + noise_handles,
        loc="lower center", ncol=len(family_handles) + len(noise_handles),
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=FONTSIZE - 4,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Scatter base vs surfaced error, per-run and per-subject, with "
            "Pearson/Spearman correlation. Reads the hilibig_battery "
            "aggregate CSV."
        ),
    )
    p.add_argument(
        "--aggregate-csv", type=Path,
        default=Path("results/hilibig_battery_aggregate.csv"),
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/hilibig_battery_correlation.png"),
    )
    p.add_argument(
        "--title", type=str,
        default="Base vs surfaced error — per-run and pooled per-subject",
    )
    args = p.parse_args(argv)

    df = pd.read_csv(args.aggregate_csv)
    df["noise"] = df["noise"].astype(float)
    plot_correlation(df, args.out, title=args.title)
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
