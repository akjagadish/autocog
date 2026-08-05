"""Empirical H2 (steep-vs-flat) and H3 (level-shift) results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_io import read_csv
from style import COLORS, DATA_DIR, FONTS, apply_style, save_figure, style_axes


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def load_h2(data_dir: Path | None = None) -> tuple[list[float], float, float]:
    rows = read_csv((data_dir or DATA_DIR) / "h2_participants.csv")
    rates = [float(r["steep_rate"]) for r in rows]
    mean, se = _mean_se(rates)
    tcrit = 1.984  # df=99, two-sided 95%
    return rates, mean, tcrit * se


def load_h3(data_dir: Path | None = None) -> tuple[list[float], list[float], list[float]]:
    rows = read_csv((data_dir or DATA_DIR) / "h3_participants.csv")
    low = [float(r["offset_lower"]) for r in rows]
    high = [float(r["offset_shifted"]) for r in rows]
    effect = [float(r["offset_effect"]) for r in rows]
    return low, high, effect


def plot_h2(out_dir: Path | None = None, basename: str = "h2_steep_vs_flat") -> list[Path]:
    """Steep-vs-flat as an explicit trade-off: low-range vs high-range advantage."""
    apply_style()
    _, low_mean, low_ci = load_h2()
    high_mean = 1.0 - low_mean

    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    labels = ["Low-range\nadvantage", "High-range\nadvantage"]
    means = [low_mean, high_mean]
    colors = [COLORS["h3_low"], COLORS["h3_high"]]
    x = np.arange(2)

    ax.axhline(0.5, color=COLORS["chance"], linewidth=1.2, linestyle="--", zorder=1)
    ax.bar(x, means, color=colors, width=0.55, zorder=2, edgecolor="white", linewidth=0.8)
    ax.errorbar(
        x, means, yerr=low_ci, fmt="none", ecolor=COLORS["error_bar"],
        elinewidth=1.8, capsize=5, zorder=3,
    )
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.025, f"{m:.1%}", ha="center", va="bottom",
                fontsize=FONTS["size_annotation"], weight="bold", color=COLORS["title"])

    ax.text(1.45, 0.5, "no preference", va="center", fontsize=FONTS["size_tick"], color=COLORS["tick"])
    ax.set_xticks(x, labels, fontsize=FONTS["size_tick"])
    ax.set_ylabel("Proportion of choices", labelpad=6)
    ax.set_ylim(0.35, 0.72)
    ax.set_title(
        "H2: steep-vs-flat\n(equal-sized objective value steps)",
        fontsize=FONTS["size_title"], weight="bold", color=COLORS["title"],
    )
    style_axes(ax)

    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


def plot_h3(out_dir: Path | None = None, basename: str = "h3_level_shift") -> list[Path]:
    """Level-shift as paired within-subject lines: same structure, weaker when shifted up."""
    apply_style()
    low, high, effect = load_h3()
    effect_mean, effect_se = _mean_se(effect)
    effect_ci = 1.984 * effect_se
    low_mean, low_se = _mean_se(low)
    high_mean, high_se = _mean_se(high)
    low_ci = 1.984 * low_se
    high_ci = 1.984 * high_se

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), constrained_layout=True, width_ratios=[1.2, 1])

    # Left: paired slopegraph
    ax = axes[0]
    x = [0, 1]
    for lo, hi in zip(low, high):
        ax.plot(x, [lo, hi], color=COLORS["axis_spine"], alpha=0.12, linewidth=0.8, zorder=1)
    ax.errorbar(
        x, [low_mean, high_mean], yerr=[low_ci, high_ci], fmt="o-",
        color=COLORS["h3_low"], ecolor=COLORS["error_bar"], elinewidth=1.8,
        capsize=5, markersize=8, linewidth=2.2, zorder=3,
    )
    for xi, m in zip(x, [low_mean, high_mean]):
        ax.text(xi, m + 0.02, f"{m:.1%}", ha="center", va="bottom",
                fontsize=FONTS["size_annotation"], weight="bold", color=COLORS["title"])

    ax.set_xticks(x, ["Low-value\nrange", "Shifted-up\nrange"], fontsize=FONTS["size_tick"])
    ax.set_ylabel("Proportion choosing target option", labelpad=6)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.25, 1.05)
    ax.set_title("Same objective structure", fontsize=FONTS["size_title"], weight="bold", color=COLORS["title"])
    style_axes(ax)

    # Right: offset effect (low − shifted) per participant
    ax = axes[1]
    jitter = (np.random.default_rng(0).random(len(effect)) - 0.5) * 0.08
    ax.scatter(jitter, effect, s=22, color=COLORS["h3_low"], alpha=0.35, edgecolors="none", zorder=1)
    ax.axhline(0, color=COLORS["chance"], linewidth=1.2, linestyle="--", zorder=1)
    ax.errorbar(
        0, effect_mean, yerr=effect_ci, fmt="D", color=COLORS["subjective_arrow"],
        ecolor=COLORS["error_bar"], elinewidth=1.8, capsize=5, markersize=7, zorder=3,
    )
    ax.text(0, effect_mean + effect_ci + 0.012, f"{effect_mean:+.1%}", ha="center", va="bottom",
            fontsize=FONTS["size_annotation"], weight="bold", color=COLORS["subjective_label"])

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Offset effect\n(low − shifted-up)", labelpad=6)
    ax.set_ylim(-0.12, 0.14)
    ax.set_title("Within-subject difference", fontsize=FONTS["size_title"], weight="bold", color=COLORS["title"])
    style_axes(ax)

    fig.suptitle(
        "H3: level-shift",
        fontsize=FONTS["size_title"], weight="bold", color=COLORS["title"], y=1.04,
    )

    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for path in plot_h2():
        print(f"Saved {path}")
    for path in plot_h3():
        print(f"Saved {path}")
