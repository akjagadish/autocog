"""Empirical H1 results: human–model choice agreement by comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from data_io import group_by, read_csv
from style import (
    COLORS,
    DATA_DIR,
    FONTS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
    save_figure,
    style_axes,
)

COMPARISON_ORDER = ["wadd", "tallying", "ttb"]
COMPARISON_LABELS = {
    "wadd": "vs WADD",
    "tallying": "vs Tallying",
    "ttb": "vs TTB",
}
TCRIT_49 = 1.984  # two-sided 95%, df = 49


def load_h1_results(data_dir: Path | None = None) -> dict[str, dict[str, dict]]:
    rows = read_csv((data_dir or DATA_DIR) / "h1_results.csv")
    by_comp = group_by(rows, "comparison")
    out = {}
    for comp, comp_rows in by_comp.items():
        out[comp] = {r["model"]: r for r in comp_rows}
    return out


def plot_h1(out_dir: Path | None = None, basename: str = "h1_model_match") -> list[Path]:
    """Match rate with Diminishing Returns WADD vs alternative, per comparison."""
    apply_style()
    data = load_h1_results()
    n = len(COMPARISON_ORDER)
    x = np.arange(n)
    bar_w = 0.34
    pi7_rates, alt_rates = [], []
    pi7_ci, alt_ci = [], []
    alt_colors = []

    for comp in COMPARISON_ORDER:
        pi7 = data[comp]["pi_7"]
        alt = data[comp][comp]
        pi7_rates.append(float(pi7["match_rate"]))
        alt_rates.append(float(alt["match_rate"]))
        pi7_ci.append(TCRIT_49 * float(pi7["se"]))
        alt_ci.append(TCRIT_49 * float(alt["se"]))
        alt_colors.append(MODEL_COLORS[comp])

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.axhline(0.5, color=COLORS["chance"], linewidth=1.2, linestyle="--", zorder=1)
    ax.bar(
        x - bar_w / 2, pi7_rates, bar_w,
        color=MODEL_COLORS["pi_7"], edgecolor="white", linewidth=0.8, zorder=2,
    )
    ax.bar(
        x + bar_w / 2, alt_rates, bar_w,
        color=alt_colors, edgecolor="white", linewidth=0.8, zorder=2,
    )
    ax.errorbar(
        x - bar_w / 2, pi7_rates, yerr=pi7_ci, fmt="none",
        ecolor=COLORS["error_bar"], elinewidth=1.8, capsize=5, zorder=3,
    )
    ax.errorbar(
        x + bar_w / 2, alt_rates, yerr=alt_ci, fmt="none",
        ecolor=COLORS["error_bar"], elinewidth=1.8, capsize=5, zorder=3,
    )

    ax.text(n - 0.52, 0.505, "chance", va="bottom", fontsize=FONTS["size_tick"], color=COLORS["tick"])
    ax.set_xticks(x, [COMPARISON_LABELS[c] for c in COMPARISON_ORDER], fontsize=FONTS["size_tick"])
    ax.set_ylabel("Proportion matching model's choice", labelpad=6)
    ax.set_ylim(0.2, 0.82)
    ax.set_title(
        "H1: model-discriminating trials",
        fontsize=FONTS["size_title"], weight="bold", color=COLORS["title"],
    )

    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor="white",
                       label=MODEL_LABELS[m].replace("\n", " "))
        for m in ["pi_7", "wadd", "tallying", "ttb"]
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=FONTS["size_tick"], loc="upper right")
    style_axes(ax)

    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for path in plot_h1():
        print(f"Saved {path}")
