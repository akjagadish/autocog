"""Flagship figure A: the concave curve is the star, results sit beneath it.

Top:    large concave-vs-linear curve (the discovery).
Bottom: three preregistered effects shown on one common scale (forest style),
        each as a point estimate +/- SEM measured in percentage points away from
        the no-effect baseline.

The hero curve and the result drawers are reused by the column-layout variant.
"""

import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from plot_h1 import COMPARISON_ORDER, load_h1_results
from plot_h2_h3 import load_h2, load_h3
from style import (
    COLORS,
    CONCAVE,
    FLAT,
    FLAT_REGION_X,
    RIVAL,
    STEEP,
    STEEP_REGION_X,
    apply_style,
    save_figure,
    style_axes,
    value,
)


# ===========================================================================
# Hero curve
# ===========================================================================
def draw_hero_curve(ax) -> None:
    x = np.linspace(0, 5, 400)
    y_concave = value(x)

    ax.fill_between(x, 0, y_concave, where=(x <= STEEP_REGION_X[1]),
                    color=COLORS["steep_region_fill"], alpha=0.85, zorder=0)
    ax.fill_between(x, 0, y_concave, where=(x >= FLAT_REGION_X[0]),
                    color=COLORS["flat_region_fill"], alpha=0.85, zorder=0)
    ax.plot(x, y_concave, color=COLORS["curve"], linewidth=3.4, zorder=4)

    for start, end in ((0, 1), (4, 5)):
        ax.annotate("", xy=(end, -0.10), xytext=(start, -0.10),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["objective_arrow"], lw=1.5))
        ax.text((start + end) / 2, -0.16, "+1 step", ha="center", va="top",
                fontsize=9, color=COLORS["objective_label"])

    # Gain arrows sit inside their regions, so they take the region colours.
    ax.annotate("", xy=(1.14, value(1)), xytext=(1.14, value(0)),
                arrowprops=dict(arrowstyle="<->", color=STEEP, lw=2.0))
    ax.text(1.30, (value(0) + value(1)) / 2, "big gain", va="center",
            fontsize=10, weight="bold", color=STEEP, linespacing=1.0)
    ax.annotate("", xy=(5.12, value(5)), xytext=(5.12, value(4)),
                arrowprops=dict(arrowstyle="<->", color=FLAT, lw=2.0))
    ax.text(5.26, (value(4) + value(5)) / 2, "small gain", va="center",
            fontsize=10, weight="bold", color=FLAT, linespacing=1.0)

    ax.text(np.mean(STEEP_REGION_X), 1.07, "steep region", ha="center", va="bottom",
            fontsize=10, weight="bold", color=STEEP)
    ax.text(np.mean(FLAT_REGION_X), 1.07, "flat region", ha="center", va="bottom",
            fontsize=10, weight="bold", color=FLAT)

    ax.set_xlabel("Objective feature value")
    ax.set_ylabel("Subjective value")
    ax.set_xlim(-0.2, 6.05)
    ax.set_ylim(-0.26, 1.16)
    ax.set_xticks(range(0, 6))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axes(ax)


# ===========================================================================
# Results: standard hypothesis-testing bars.
#   * point estimate (bar) with 95% confidence interval
#   * dashed null/chance reference line at 0.5
#   * significance stars (vs the relevant null) so the reader can judge the test
# ===========================================================================
ERR_COLOR = "#222222"
RESULT_YLIM = (0.0, 1.0)
RESULT_YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TCRIT = 1.984  # two-sided 95% critical value, df ~ 49-99
ERRBAR = dict(ecolor=ERR_COLOR, elinewidth=1.6, capsize=0, zorder=4)

# --- Result bar-chart toggles (flip these to restyle every result panel) ----
SHOW_BAR_VALUES = False   # print the "%" value label inside each bar
BAR_EDGE = True           # draw an outline around each individual bar
BAR_EDGE_COLOR = "#222222"
BAR_EDGE_WIDTH = 1.0


def _sem(values) -> float:
    arr = np.asarray(values, float)
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def _p_one_sided(t_stat: float) -> float:
    """One-sided p-value in the predicted direction (the preregistered test).

    Normal approximation; fine for df ~ 50-100 and matches Table 1 to the
    reported significance levels.
    """
    return 1.0 - 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _bar_label(ax, x, h):
    if not SHOW_BAR_VALUES:
        return
    ax.text(x, h - 0.02, f"{h:.0%}", ha="center", va="top", fontsize=9,
            weight="bold", color="white", zorder=6)


def _bar_edge() -> dict:
    """Outline each bar (an edge around the bar itself), if enabled."""
    if BAR_EDGE:
        return dict(edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
    return dict(edgecolor="none")


def _sig_marker(ax, x, top, p) -> None:
    txt = _stars(p)
    ax.text(x, top + 0.012, txt, ha="center", va="bottom",
            fontsize=10.5 if txt != "n.s." else 8.0, weight="bold",
            color=COLORS["title"], zorder=6)


H1_RESULT_YLIM = RESULT_YLIM


def draw_result_h1(ax) -> None:
    # One paired comparison per rival: what share chose the concave-predicted
    # option (DR colour) vs the rival-predicted option (that rival's colour).
    # Same colours as the three H1 example trials above.
    data = load_h1_results()
    x = np.arange(len(COMPARISON_ORDER)) * 0.78
    bw = 0.36
    for xi, comp in zip(x, COMPARISON_ORDER):
        dr = float(data[comp]["pi_7"]["match_rate"])
        rv = float(data[comp][comp]["match_rate"])
        # CSV SE is the SE of the difference (= 2*rate - 1); rate SE is half.
        se_rate = float(data[comp]["pi_7"]["se"]) / 2.0
        ci = TCRIT * se_rate
        ax.bar(xi - bw / 2, dr, bw, color=CONCAVE, zorder=2, **_bar_edge())
        ax.bar(xi + bw / 2, rv, bw, color=RIVAL, zorder=2, **_bar_edge())
        ax.errorbar(xi - bw / 2, dr, yerr=ci, fmt="none", **ERRBAR)
        _bar_label(ax, xi - bw / 2, dr)
        _bar_label(ax, xi + bw / 2, rv)
        _sig_marker(ax, xi - bw / 2, dr + ci, _p_one_sided((dr - 0.5) / se_rate))
    ax.set_xticks(x, ["vs WADD", "vs Tallying", "vs TTB"], fontsize=9)
    ax.set_ylabel("Share choosing each model's option")
    ax.set_ylim(*RESULT_YLIM)
    ax.set_yticks(RESULT_YTICKS)
    ax.set_xlim(x[0] - bw - 0.08, x[-1] + bw + 0.08)
    style_axes(ax)


def draw_result_h2(ax) -> None:
    # Paired bars (same grammar as H1): the steep-region option is the choice
    # ONLY the concave model favours (CONCAVE colour); linear is indifferent, so
    # the flat-region option is the "rival" choice (RIVAL colour).
    rates, mean, ci = load_h2()
    se = _sem(rates)
    other = 1.0 - mean
    x = np.array([0.0, 0.40])
    bw = 0.40
    ax.bar(x[0], mean, bw, color=CONCAVE, zorder=2, **_bar_edge())
    ax.bar(x[1], other, bw, color=RIVAL, zorder=2, **_bar_edge())
    ax.errorbar(x[0], mean, yerr=ci, fmt="none", **ERRBAR)
    _bar_label(ax, x[0], mean)
    _bar_label(ax, x[1], other)
    _sig_marker(ax, x[0], mean + ci, _p_one_sided((mean - 0.5) / se))
    ax.set_xticks(x, ["Chose steep\noption", "Chose flat\noption"], fontsize=9)
    ax.set_ylabel("Share of choices")
    ax.set_ylim(*RESULT_YLIM)
    ax.set_yticks(RESULT_YTICKS)
    ax.set_xlim(x[0] - bw / 2 - 0.08, x[-1] + bw / 2 + 0.08)
    style_axes(ax)


def draw_result_h3(ax) -> None:
    # Two within-subject conditions; the preregistered test is the paired drop.
    low, high, effect = load_h3()
    means = [float(np.mean(low)), float(np.mean(high))]
    cis = [TCRIT * _sem(low), TCRIT * _sem(high)]
    eff_mean, eff_se = float(np.mean(effect)), _sem(effect)
    p = _p_one_sided(eff_mean / eff_se)
    x = np.array([0.0, 0.40])
    bw = 0.40
    # Bars carry the region colours: the low range sits in the steep region
    # (gold), the shifted-up range in the flat region (cool grey) -- the same
    # two values circled in the H3 example (+7.0 vs +1.5 concave preference).
    ax.bar(x, means, bw, color=[STEEP, FLAT], zorder=2, **_bar_edge())
    ax.errorbar(x, means, yerr=cis, fmt="none", **ERRBAR)
    for xi, h in zip(x, means):
        _bar_label(ax, xi, h)
    # Paired-difference bracket (the actual H3 test).
    y_br = max(m + c for m, c in zip(means, cis)) + 0.035
    ax.plot([x[0], x[0], x[1], x[1]],
            [means[0] + cis[0] + 0.012, y_br, y_br, means[1] + cis[1] + 0.012],
            color=COLORS["title"], lw=1.2, zorder=5)
    ax.text((x[0] + x[1]) / 2, y_br + 0.008, f"\u2212{abs(eff_mean):.1%}  {_stars(p)}",
            ha="center", va="bottom", fontsize=9, weight="bold", color=COLORS["title"])
    ax.set_xticks(x, ["Steep\nregion", "Flat\nregion"], fontsize=9)
    ax.set_ylabel("Chose the stronger option")
    ax.set_ylim(0.35, 1.02)
    ax.set_xlim(x[0] - bw / 2 - 0.08, x[-1] + bw / 2 + 0.08)
    style_axes(ax)


RESULT_TITLES = {
    "h1": "H1  Beats every rival model",
    "h2": "H2  Steep-region gains win",
    "h3": "H3  Preference flattens when shifted up",
}


# ===========================================================================
# Version A assembly: hero curve + forest results
# ===========================================================================
def build_flagship_figure():
    apply_style()
    fig = plt.figure(figsize=(13.0, 9.4))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, height_ratios=[1.7, 1.0],
        hspace=0.55, wspace=0.40,
        left=0.07, right=0.965, top=0.88, bottom=0.10,
    )

    ax_hero = fig.add_subplot(gs[0, :])
    draw_hero_curve(ax_hero)

    for col, (key, draw_fn) in enumerate(
        [("h1", draw_result_h1), ("h2", draw_result_h2), ("h3", draw_result_h3)]
    ):
        ax = fig.add_subplot(gs[1, col])
        draw_fn(ax)
        ax.set_title(RESULT_TITLES[key], fontsize=10.5, weight="bold",
                     color=COLORS["title"], pad=8)

    fig.suptitle(
        "Diminishing Returns WADD: a concave theory of choice, confirmed",
        fontsize=16, weight="bold", color=COLORS["title"], y=0.965,
    )
    return fig


def plot_flagship(out_dir: Path | None = None, basename: str = "theory_results") -> list[Path]:
    fig = build_flagship_figure()
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for path in plot_flagship():
        print(f"Saved {path}")
