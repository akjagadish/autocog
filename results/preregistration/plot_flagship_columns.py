"""Flagship figure B: three matched columns (H1 / H2 / H3).

Each column tells a self-contained story:
  top    - a small prediction curve highlighting one feature of the concave shape
  bottom - the preregistered result that confirms it
A shared header reminds the reader of the concave-vs-linear contrast.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from plot_flagship import (
    draw_hero_curve,
    draw_result_h1,
    draw_result_h2,
    draw_result_h3,
)
from style import (
    CONCAVE,
    COLORS,
    FLAT,
    GRAY,
    INK,
    LABEL,
    OPTION_A,
    OPTION_B,
    RIVAL,
    SHIFT,
    STEEP,
    WADD,
    apply_style,
    save_figure,
)
from trial_diagram import load_json, option_scores, rival_scores


# ---------------------------------------------------------------------------
# COLOUR CONTRACT (all colours are the named constants from style.py):
#   * each MODEL owns one colour; the concave "+" prediction is ALWAYS CONCAVE,
#     and each rival "-" prediction is that rival's colour. The very same
#     colours reappear on the result bars below.
#   * option tiles are neutral grey (OPTION_A / OPTION_B) so model colours are
#     never confused with "Option A / Option B".
#   * region identity (STEEP / FLAT) is a third dimension.
# Layout-only constants:
# ---------------------------------------------------------------------------
TILE = 0.8
SUM_W = 0.86
CONC_X = 4.05          # concave column (Diminishing Returns)
LIN_X = 5.95           # rival column (WADD / linear)
EX_XLIM = (-1.3, 5.7)
EX_YLIM = (-0.2, 7.4)

RIVAL_NAMES = {"wadd": "WADD", "tallying": "Tallying", "ttb": "TTB"}


def _tiles(ax, x0, y, vals, fill, *, size=TILE, step=1.0, badges=None):
    for j, v in enumerate(vals):
        x = x0 + j * step
        ax.add_patch(mpatches.Rectangle((x, y), size, size, facecolor=fill,
                                        edgecolor="white", lw=1.6, zorder=3))
        ax.text(x + size / 2, y + size / 2, str(int(v)), ha="center", va="center",
                color="white", fontsize=12 * size / TILE, weight="bold", zorder=4)
        if badges and j in badges:
            text, bcolor = badges[j]
            ax.text(x + size * 0.85, y + size * 0.85, text, ha="center", va="center",
                    fontsize=7.5, weight="bold", color=bcolor, zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.14", facecolor="white",
                              edgecolor=bcolor, linewidth=1.3))


def _row_label(ax, y, text, *, size=TILE, x=-0.25):
    ax.text(x, y + size / 2, text, ha="right", va="center",
            fontsize=12 * size / TILE, weight="bold", color=LABEL)


def _sum_cell(ax, x, y, total, *, winner, decimals):
    txt = f"{total:.1f}" if decimals else f"{total:.0f}"
    ax.text(x + SUM_W / 2, y + TILE / 2, txt, ha="center", va="center",
            fontsize=12 if winner else 11, weight="bold" if winner else "normal",
            color=INK if winner else GRAY, zorder=4)


def _score_value(ax, xc, yc, val, *, winner, decimals):
    txt = f"{val:.1f}" if decimals else f"{val:.0f}"
    ax.text(xc, yc, txt, ha="center", va="center",
            fontsize=12 if winner else 11, weight="bold" if winner else "normal",
            color=INK if winner else GRAY, zorder=4)


def _sum_headers(ax, top_y):
    ax.text(CONC_X + SUM_W / 2, top_y + 0.14, "concave", ha="center", va="bottom",
            fontsize=7.8, weight="bold", color=CONCAVE)
    ax.text(LIN_X + SUM_W / 2, top_y + 0.14, "linear", ha="center", va="bottom",
            fontsize=7.8, weight="bold", color=WADD)


def _winners(sc):
    lin = ("A" if sc["linear_a"] > sc["linear_b"]
           else "B" if sc["linear_b"] > sc["linear_a"] else None)
    con = "A" if sc["concave_a"] > sc["concave_b"] else "B"
    return lin, con


def _sums_block(ax, ya, yb, sc, *, headers=True):
    lin_win, con_win = _winners(sc)
    _sum_cell(ax, CONC_X, ya, sc["concave_a"], winner=con_win == "A", decimals=True)
    _sum_cell(ax, CONC_X, yb, sc["concave_b"], winner=con_win == "B", decimals=True)
    _sum_cell(ax, LIN_X, ya, sc["linear_a"], winner=lin_win == "A", decimals=False)
    _sum_cell(ax, LIN_X, yb, sc["linear_b"], winner=lin_win == "B", decimals=False)
    if headers:
        _sum_headers(ax, max(ya, yb) + TILE)


def _frame(ax):
    ax.set_xlim(*EX_XLIM)
    ax.set_ylim(*EX_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")


def _signed(v, dec):
    return "0" if abs(v) < 0.05 else f"{v:+.{dec}f}"


def _circle(ax, cx, cy, color, w=1.22, h=0.74):
    """Ring the compared quantity (generous padding), in its model colour."""
    ax.add_patch(mpatches.Ellipse((cx, cy), w, h, fill=False,
                                  edgecolor=color, lw=1.7, zorder=5))


def _pref_row(ax, y, con_pref, lin_pref, *, lin_color=WADD,
              circle_con=True, circle_lin=True):
    """B - A row: concave always the CONCAVE colour, rival in its own colour."""
    ax.text(-0.25, y, "B - A", ha="right", va="center", fontsize=8,
            weight="bold", color=INK)
    ax.text(CONC_X + SUM_W / 2, y, _signed(con_pref, 1), ha="center", va="center",
            fontsize=12.5, weight="bold", color=CONCAVE, zorder=6)
    ax.text(LIN_X + SUM_W / 2, y, _signed(lin_pref, 0), ha="center", va="center",
            fontsize=10.5, weight="bold", color=lin_color, zorder=6)
    if circle_con:
        _circle(ax, CONC_X + SUM_W / 2, y, CONCAVE)
    if circle_lin:
        _circle(ax, LIN_X + SUM_W / 2, y, lin_color)


# ---------------------------------------------------------------------------
# H1: three model-discriminating trials stacked in the column. Each block uses
# the SAME layout as H2/H3 (per-option value for concave + for the one rival),
# with the compared B - A values side by side so the circles never overlap.
# The concave "+" is the DR colour; the rival "-" is that rival's colour --
# the same colours return on the result bars.
# ---------------------------------------------------------------------------
H1_TRIALS = [
    {"rival": "wadd", "a": [5, 5, 4, 0], "b": [4, 4, 3, 2]},
    {"rival": "tallying", "a": [3, 3, 3, 0], "b": [2, 2, 2, 3]},
    {"rival": "ttb", "a": [5, 2, 1, 0], "b": [4, 1, 2, 1]},
]
H1_TILE = 0.56
H1_STEP = 0.62
H1_CX = 3.00            # centre of the concave value column
H1_RX = 4.60           # centre of the rival value column
H1_YA = [6.5, 3.85, 1.2]   # bottom of each block's Option-A tile row (extra gap)


def _h1_block(ax, ya, trial) -> None:
    a, b = trial["a"], trial["b"]
    rk = trial["rival"]
    rcolor = RIVAL
    sc = option_scores({"option_a": a, "option_b": b})
    ca, cb = sc["concave_a"], sc["concave_b"]
    ra, rb = rival_scores(a, b, rk)
    yb = ya - (H1_TILE + 0.1)
    _tiles(ax, 0, ya, a, OPTION_A, size=H1_TILE, step=H1_STEP)
    _tiles(ax, 0, yb, b, OPTION_B, size=H1_TILE, step=H1_STEP)
    _row_label(ax, ya, "A", size=H1_TILE, x=-0.18)
    _row_label(ax, yb, "B", size=H1_TILE, x=-0.18)
    # Column headers (model names, not "sum").
    hy = ya + H1_TILE + 0.14
    ax.text(H1_CX, hy, "concave", ha="center", va="bottom", fontsize=7.4,
            weight="bold", color=CONCAVE)
    ax.text(H1_RX, hy, RIVAL_NAMES[rk], ha="center", va="bottom", fontsize=7.4,
            weight="bold", color=rcolor)
    # Per-option calculated values.
    _score_value(ax, H1_CX, ya + H1_TILE / 2, ca, winner=ca > cb, decimals=True)
    _score_value(ax, H1_CX, yb + H1_TILE / 2, cb, winner=cb > ca, decimals=True)
    _score_value(ax, H1_RX, ya + H1_TILE / 2, ra, winner=ra > rb, decimals=False)
    _score_value(ax, H1_RX, yb + H1_TILE / 2, rb, winner=rb > ra, decimals=False)
    # B - A row: the compared quantities, side by side and circled.
    yd = yb - 0.40
    ax.text(-0.18, yd, "B-A", ha="right", va="center", fontsize=7,
            weight="bold", color=INK)
    ax.text(H1_CX, yd, _signed(cb - ca, 1), ha="center", va="center",
            fontsize=11.5, weight="bold", color=CONCAVE, zorder=6)
    ax.text(H1_RX, yd, _signed(rb - ra, 0), ha="center", va="center",
            fontsize=11.5, weight="bold", color=rcolor, zorder=6)
    _circle(ax, H1_CX, yd, CONCAVE, w=0.96, h=0.58)
    _circle(ax, H1_RX, yd, rcolor, w=0.96, h=0.58)


def draw_example_h1(ax) -> None:
    for ya, trial in zip(H1_YA, H1_TRIALS):
        _h1_block(ax, ya, trial)
    _frame(ax)


# ---------------------------------------------------------------------------
# H2: two steep-vs-flat trials stacked like the H1 column. Each trial trades a
# steep-region gain (gold +1 badge) against a flat-region gain (grey -1 badge)
# so the LINEAR sums tie (rival B - A = 0) but the concave model still prefers
# the steep-region option (CONCAVE B - A > 0). The two trials place the steep /
# flat features in DIFFERENT columns to show the effect is not position-bound.
# ---------------------------------------------------------------------------
H2_TRIALS = [
    {"a": [2, 0, 5, 2], "b": [2, 1, 4, 2], "steep_i": 1, "flat_i": 2},
    {"a": [0, 3, 3, 5], "b": [1, 3, 3, 4], "steep_i": 0, "flat_i": 3},
]
H2_YA = [5.55, 2.35]


def _h2_block(ax, ya, trial) -> None:
    a, b = trial["a"], trial["b"]
    steep_i, flat_i = trial["steep_i"], trial["flat_i"]
    sc = option_scores({
        "option_a": a, "option_b": b,
        "steep_advantage": {"feature_index": steep_i},
        "flat_advantage": {"feature_index": flat_i},
    })
    ca, cb = sc["concave_a"], sc["concave_b"]
    la, lb = sc["linear_a"], sc["linear_b"]
    yb = ya - (H1_TILE + 0.1)
    b_badges = {
        steep_i: (f"+{int(b[steep_i] - a[steep_i])}", STEEP),
        flat_i: (f"{int(b[flat_i] - a[flat_i])}", FLAT),
    }
    _tiles(ax, 0, ya, a, OPTION_A, size=H1_TILE, step=H1_STEP)
    _tiles(ax, 0, yb, b, OPTION_B, size=H1_TILE, step=H1_STEP, badges=b_badges)
    _row_label(ax, ya, "A", size=H1_TILE, x=-0.18)
    _row_label(ax, yb, "B", size=H1_TILE, x=-0.18)
    # Steep / flat region labels over the two traded features.
    top = ya + H1_TILE
    for idx, txt, color in ((steep_i, "steep", STEEP), (flat_i, "flat", FLAT)):
        cx = idx * H1_STEP + H1_TILE / 2
        ax.annotate("", xy=(cx, top + 0.04), xytext=(cx, top + 0.26),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                    mutation_scale=7))
        ax.text(cx, top + 0.30, txt, ha="center", va="bottom", fontsize=6.8,
                weight="bold", color=color)
    hy = ya + H1_TILE + 0.14
    ax.text(H1_CX, hy, "concave", ha="center", va="bottom", fontsize=7.4,
            weight="bold", color=CONCAVE)
    ax.text(H1_RX, hy, "linear", ha="center", va="bottom", fontsize=7.4,
            weight="bold", color=WADD)
    _score_value(ax, H1_CX, ya + H1_TILE / 2, ca, winner=ca > cb, decimals=True)
    _score_value(ax, H1_CX, yb + H1_TILE / 2, cb, winner=cb > ca, decimals=True)
    _score_value(ax, H1_RX, ya + H1_TILE / 2, la, winner=la > lb, decimals=False)
    _score_value(ax, H1_RX, yb + H1_TILE / 2, lb, winner=lb > la, decimals=False)
    yd = yb - 0.40
    ax.text(-0.18, yd, "B-A", ha="right", va="center", fontsize=7,
            weight="bold", color=INK)
    ax.text(H1_CX, yd, _signed(cb - ca, 1), ha="center", va="center",
            fontsize=11.5, weight="bold", color=CONCAVE, zorder=6)
    ax.text(H1_RX, yd, _signed(lb - la, 0), ha="center", va="center",
            fontsize=11.5, weight="bold", color=WADD, zorder=6)
    _circle(ax, H1_CX, yd, CONCAVE, w=0.96, h=0.58)
    _circle(ax, H1_RX, yd, WADD, w=0.96, h=0.58)


def draw_example_h2(ax) -> None:
    for ya, trial in zip(H2_YA, H2_TRIALS):
        _h2_block(ax, ya, trial)
    _frame(ax)


def _value_block(ax, ya, a, b, ca, cb, ra, rb, *, rname, rcolor,
                 badges_a=None, badges_b=None, header=True):
    """Compact A/B trial block (shared with H1/H2): tiles + concave & rival
    per-option values + a circled B - A row. Returns the B - A row's y."""
    yb = ya - (H1_TILE + 0.1)
    _tiles(ax, 0, ya, a, OPTION_A, size=H1_TILE, step=H1_STEP, badges=badges_a)
    _tiles(ax, 0, yb, b, OPTION_B, size=H1_TILE, step=H1_STEP, badges=badges_b)
    _row_label(ax, ya, "A", size=H1_TILE, x=-0.18)
    _row_label(ax, yb, "B", size=H1_TILE, x=-0.18)
    if header:
        hy = ya + H1_TILE + 0.14
        ax.text(H1_CX, hy, "concave", ha="center", va="bottom", fontsize=7.4,
                weight="bold", color=CONCAVE)
        ax.text(H1_RX, hy, rname, ha="center", va="bottom", fontsize=7.4,
                weight="bold", color=rcolor)
    _score_value(ax, H1_CX, ya + H1_TILE / 2, ca, winner=ca > cb, decimals=True)
    _score_value(ax, H1_CX, yb + H1_TILE / 2, cb, winner=cb > ca, decimals=True)
    _score_value(ax, H1_RX, ya + H1_TILE / 2, ra, winner=ra > rb, decimals=False)
    _score_value(ax, H1_RX, yb + H1_TILE / 2, rb, winner=rb > ra, decimals=False)
    yd = yb - 0.40
    ax.text(-0.18, yd, "B-A", ha="right", va="center", fontsize=7,
            weight="bold", color=INK)
    ax.text(H1_CX, yd, _signed(cb - ca, 1), ha="center", va="center",
            fontsize=11.5, weight="bold", color=CONCAVE, zorder=6)
    ax.text(H1_RX, yd, _signed(rb - ra, 0), ha="center", va="center",
            fontsize=11.5, weight="bold", color=rcolor, zorder=6)
    _circle(ax, H1_CX, yd, CONCAVE, w=0.96, h=0.58)
    _circle(ax, H1_RX, yd, rcolor, w=0.96, h=0.58)
    return yd


def _h3_region_tag(ax, top_y, bot_y, text, color):
    """Rotated section label + vertical accent marking one region's block."""
    cy = (top_y + bot_y) / 2
    ax.plot([-0.78, -0.78], [bot_y, top_y], color=color, lw=2.6,
            solid_capstyle="round", zorder=3)
    ax.text(-1.04, cy, text, rotation=90, ha="center", va="center",
            fontsize=8.6, weight="bold", color=color)


def draw_example_h3(ax) -> None:
    ex = load_json("h3_example.json")
    low, shifted, shift = ex["low"], ex["shifted"], ex["shift"]
    slo, ssh = option_scores(low), option_scores(shifted)

    # Top block: low (steep) region -- same compact block as H1/H2.
    ya1 = 5.55
    yd1 = _value_block(ax, ya1, low["option_a"], low["option_b"],
                       slo["concave_a"], slo["concave_b"],
                       slo["linear_a"], slo["linear_b"],
                       rname="linear", rcolor=WADD, header=True)
    yb1 = ya1 - (H1_TILE + 0.1)
    _h3_region_tag(ax, ya1 + H1_TILE + 0.05, yd1 - 0.30, "steep region", STEEP)

    # Bottom block: shifted-up (flat) region. Every feature is lifted by +shift,
    # marked with a small green "+shift" badge on each tile of both options.
    ya2 = 2.35
    shift_badges = {j: (f"+{shift}", SHIFT) for j in range(len(shifted["option_a"]))}
    yd2 = _value_block(ax, ya2, shifted["option_a"], shifted["option_b"],
                       ssh["concave_a"], ssh["concave_b"],
                       ssh["linear_a"], ssh["linear_b"],
                       rname="linear", rcolor=WADD,
                       badges_a=shift_badges, badges_b=shift_badges, header=False)
    yb2 = ya2 - (H1_TILE + 0.1)
    _h3_region_tag(ax, ya2 + H1_TILE + 0.05, yd2 - 0.30, "flat region", FLAT)

    # Shift arrow connecting the two blocks (green = the act of shifting).
    mid_x = 1.0
    ax.annotate("", xy=(mid_x, ya2 + H1_TILE + 0.10), xytext=(mid_x, yb1 - 0.18),
                arrowprops=dict(arrowstyle="-|>", color=SHIFT, lw=2.0, mutation_scale=12))
    ax.text(mid_x + 0.22, (yb1 - 0.18 + ya2 + H1_TILE) / 2, f"+{shift} shift", ha="left",
            va="center", fontsize=8, weight="bold", color=SHIFT)

    # The tested comparison: concave preference shrinks (the two circled concave
    # B - A values), while linear stays put. A curved connector bows to the
    # right so it never crosses the printed numbers.
    top_c, bot_c = yd1, yd2
    ax.annotate("", xy=(H1_CX + 0.40, bot_c), xytext=(H1_CX + 0.40, top_c),
                arrowprops=dict(arrowstyle="-|>", color=CONCAVE, lw=1.5,
                                mutation_scale=10, linestyle=(0, (4, 2)),
                                connectionstyle="arc3,rad=-0.30",
                                shrinkA=6, shrinkB=6))
    ax.text(H1_CX + 0.78, (top_c + bot_c) / 2, "shrinks", rotation=90,
            ha="center", va="center", fontsize=7.4, weight="bold", color=CONCAVE)

    # Linear says nothing changes. A matching (head-less) connector on the
    # linear column, labelled "invariant", makes the contrast explicit.
    ax.annotate("", xy=(H1_RX + 0.40, bot_c), xytext=(H1_RX + 0.40, top_c),
                arrowprops=dict(arrowstyle="-", color=WADD, lw=1.5,
                                linestyle=(0, (4, 2)),
                                connectionstyle="arc3,rad=-0.30",
                                shrinkA=6, shrinkB=6))
    ax.text(H1_RX + 0.78, (top_c + bot_c) / 2, "invariant", rotation=90,
            ha="center", va="center", fontsize=7.4, weight="bold", color=WADD)
    _frame(ax)

COLUMNS = [
    {
        "key": "h1",
        "title": "H1 · Model discrimination",
        "example": draw_example_h1,
        "result": draw_result_h1,
        "result_caption": "Beats WADD, Tallying & TTB",
    },
    {
        "key": "h2",
        "title": "H2 · Steep-vs-flat",
        "example": draw_example_h2,
        "result": draw_result_h2,
        "result_caption": "Steep-region gains are chosen",
    },
    {
        "key": "h3",
        "title": "H3 · Diminishing preference",
        "example": draw_example_h3,
        "result": draw_result_h3,
        "result_caption": "Preference shrinks when shifted up",
    },
]

SUPTITLE = "Diminishing Returns WADD: a concave theory of multi-attribute choice"


def build_columns_figure():
    apply_style()
    fig = plt.figure(figsize=(14.5, 12.8))

    # Two bands: the hero curve on top, then a row of three self-contained
    # hypothesis groups. Each group stacks its example trial directly above its
    # result (small vertical gap = "these belong together"); a wide horizontal
    # gap separates the three groups.
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 2.4], hspace=0.14,
        left=0.06, right=0.975, top=0.92, bottom=0.05,
    )
    ax_hero = fig.add_subplot(outer[0])
    draw_hero_curve(ax_hero)

    body = outer[1].subgridspec(1, 3, wspace=0.55)
    for col, spec in enumerate(COLUMNS):
        group = body[0, col].subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.1)
        ax_ex = fig.add_subplot(group[0])
        spec["example"](ax_ex)
        ax_ex.set_anchor("S")  # drop the (square) example down toward its result
        ax_ex.set_title(spec["title"], fontsize=12.5, weight="bold",
                        color=COLORS["title"], pad=6)
        ax_res = fig.add_subplot(group[1])
        spec["result"](ax_res)
        ax_res.set_anchor("N")

    fig.suptitle(SUPTITLE, fontsize=16, weight="bold",
                 color=COLORS["title"], y=0.965)
    return fig


def plot_flagship_columns(out_dir: Path | None = None,
                          basename: str = "columns") -> list[Path]:
    fig = build_columns_figure()
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


def build_rows_figure():
    apply_style()
    fig = plt.figure(figsize=(10.5, 17.0))

    # Hero curve on top; then one ROW per hypothesis. In each row the example
    # trial sits beside its result, both the same size; a larger vertical gap
    # separates the three hypotheses.
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.62, 3.0], hspace=0.09,
        left=0.075, right=0.965, top=0.945, bottom=0.035,
    )
    ax_hero = fig.add_subplot(outer[0])
    draw_hero_curve(ax_hero)

    body = outer[1].subgridspec(3, 1, hspace=0.22)
    for r, spec in enumerate(COLUMNS):
        row = body[r].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.1)
        ax_ex = fig.add_subplot(row[0])
        spec["example"](ax_ex)
        ax_ex.set_title(spec["title"], fontsize=12.5, weight="bold",
                        color=COLORS["title"], pad=6)
        ax_res = fig.add_subplot(row[1])
        spec["result"](ax_res)

    fig.suptitle(SUPTITLE, fontsize=16, weight="bold",
                 color=COLORS["title"], y=0.972)
    return fig


def plot_flagship_rows(out_dir: Path | None = None,
                       basename: str = "rows") -> list[Path]:
    fig = build_rows_figure()
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


def _draw_group_stack(fig, cell, spec):
    """One hypothesis as an example trial stacked directly above its result."""
    group = cell.subgridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.12)
    ax_ex = fig.add_subplot(group[0])
    spec["example"](ax_ex)
    ax_ex.set_anchor("S")
    ax_ex.set_title(spec["title"], fontsize=11.5, weight="bold",
                    color=COLORS["title"], pad=6)
    ax_res = fig.add_subplot(group[1])
    spec["result"](ax_res)
    ax_res.set_anchor("N")


def build_single_row_figure():
    apply_style()
    fig = plt.figure(figsize=(19.5, 7.4))

    # Everything in ONE row of four: the hero curve, then the three hypotheses,
    # each stacking its example trial above its result.
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.32,
        left=0.045, right=0.99, top=0.88, bottom=0.085,
    )
    ax_hero = fig.add_subplot(gs[0, 0])
    draw_hero_curve(ax_hero)
    ax_hero.set_box_aspect(1)
    ax_hero.set_anchor("C")

    for i, spec in enumerate(COLUMNS):
        _draw_group_stack(fig, gs[0, i + 1], spec)

    fig.suptitle(SUPTITLE, fontsize=16, weight="bold",
                 color=COLORS["title"], y=0.965)
    return fig


def plot_flagship_single_row(out_dir: Path | None = None,
                             basename: str = "single_row") -> list[Path]:
    fig = build_single_row_figure()
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


def build_single_row_h1_first_figure():
    """Single-row variant ordered H1 -> theory (hero) -> H2 -> H3, so the model
    discrimination motivates the concave curve before the two follow-ups."""
    apply_style()
    fig = plt.figure(figsize=(19.5, 7.4))
    cols = {c["key"]: c for c in COLUMNS}
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.35, 1.0, 1.0], wspace=0.32,
        left=0.045, right=0.99, top=0.88, bottom=0.085,
    )
    _draw_group_stack(fig, gs[0, 0], cols["h1"])
    ax_hero = fig.add_subplot(gs[0, 1])
    draw_hero_curve(ax_hero)
    ax_hero.set_box_aspect(1)
    ax_hero.set_anchor("C")
    _draw_group_stack(fig, gs[0, 2], cols["h2"])
    _draw_group_stack(fig, gs[0, 3], cols["h3"])

    fig.suptitle(SUPTITLE, fontsize=16, weight="bold",
                 color=COLORS["title"], y=0.965)
    return fig


def plot_flagship_single_row_h1_first(
        out_dir: Path | None = None,
        basename: str = "single_row_h1_first") -> list[Path]:
    fig = build_single_row_h1_first_figure()
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Standalone panels: every plot is also saved on its own so the final figure
# can be assembled (in LaTeX / a layout tool) from independent pieces.
# ---------------------------------------------------------------------------
def _save_panel(draw_fn, basename, figsize, out_dir, *, title=None) -> list[Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    if title:
        ax.set_title(title, fontsize=13, weight="bold", color=COLORS["title"], pad=6)
    paths = save_figure(fig, basename, out_dir)
    plt.close(fig)
    return paths


def plot_standalone_panels(out_dir: Path | None = None) -> list[Path]:
    paths = _save_panel(draw_hero_curve, "panel_hero", (9.5, 4.2), out_dir)
    for spec in COLUMNS:
        paths += _save_panel(spec["example"], f"panel_{spec['key']}_example",
                             (5.4, 4.6), out_dir, title=spec["title"])
        paths += _save_panel(spec["result"], f"panel_{spec['key']}_result",
                             (4.6, 3.8), out_dir)
    return paths


if __name__ == "__main__":
    for path in plot_flagship_columns():
        print(f"Saved {path}")
    for path in plot_flagship_rows():
        print(f"Saved {path}")
    for path in plot_flagship_single_row():
        print(f"Saved {path}")
    for path in plot_flagship_single_row_h1_first():
        print(f"Saved {path}")
    for path in plot_standalone_panels():
        print(f"Saved {path}")
