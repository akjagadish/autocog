"""Shared block-diagram drawing for choice-trial examples."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from style import (
    DATA_DIR,
    GRAY,
    INK,
    NH,
    TRIAL_COLOR_A,
    TRIAL_COLOR_B,
    TRIAL_FLAT_LABEL,
    TRIAL_SCORE_FILL,
    TRIAL_STEEP_LABEL,
    VALUE_K,
    value,
)

ROW_GAP = 0.55
OBJ_SCALE = 5.0

# Two uniform block-diagram font tiers (tiles here are width 1.0):
#   * BLOCK_NUM_FS   -- every NUMBER (tile digits, sums, scores, B - A values)
#   * BLOCK_LABEL_FS -- every small WORD label (headers, option labels, tags)
# Using just two sizes keeps all numbers visually equal (no "some bigger than
# others") while leaving descriptive labels a consistent step smaller.
BLOCK_NUM_FS = 15
BLOCK_LABEL_FS = 10


def _is_h3_example(example: dict) -> bool:
    return "steep_advantage" not in example


def _trade_columns(example: dict, a=None, b=None) -> list[int]:
    if _is_h3_example(example):
        if a is not None and b is not None:
            return [j for j in range(len(a)) if a[j] != b[j]]
        return list(example.get("trade_features", []))
    steep_i = example["steep_advantage"]["feature_index"]
    flat_i = example["flat_advantage"]["feature_index"]
    return [steep_i, flat_i]


def trade_subjective_weight(winner_val: float, loser_val: float, k: float = VALUE_K) -> float:
    """Subjective value of an objective gap (in display units)."""
    return float(OBJ_SCALE * (value(winner_val, k) - value(loser_val, k)))


def option_scores(example: dict, k: float = VALUE_K) -> dict:
    a = np.array(example["option_a"], float)
    b = np.array(example["option_b"], float)
    trade_cols = _trade_columns(example, a, b)
    h3 = _is_h3_example(example)

    linear_a = float(a.sum())
    linear_b = float(b.sum())

    def trade_adjustment(winner_val: float, loser_val: float) -> float:
        linear_diff = winner_val - loser_val
        subj_diff = OBJ_SCALE * (value(winner_val, k) - value(loser_val, k))
        return float(subj_diff - linear_diff)

    concave_a = linear_a
    concave_b = linear_b

    if h3:
        for j in range(len(a)):
            if a[j] > b[j]:
                concave_a += trade_adjustment(a[j], b[j])
            elif b[j] > a[j]:
                concave_b += trade_adjustment(b[j], a[j])
    else:
        steep_i, flat_i = trade_cols[0], trade_cols[1]
        if a[steep_i] > b[steep_i]:
            concave_a += trade_adjustment(a[steep_i], b[steep_i])
        if a[flat_i] > b[flat_i]:
            concave_a += trade_adjustment(a[flat_i], b[flat_i])
        if b[steep_i] > a[steep_i]:
            concave_b += trade_adjustment(b[steep_i], a[steep_i])
        if b[flat_i] > a[flat_i]:
            concave_b += trade_adjustment(b[flat_i], a[flat_i])

    return {
        "a": a,
        "b": b,
        "linear_a": linear_a,
        "linear_b": linear_b,
        "concave_a": concave_a,
        "concave_b": concave_b,
        "trade_cols": trade_cols,
        "h3": h3,
    }


def _tile(ax, x, y, w, h, facecolor, value_text, edgecolor="white", linewidth=2.0,
          zorder=3, relative=None, badge_color=None, shift_badge=None,
          shift_badge_color=None):
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder,
    ))
    badge = relative or shift_badge
    if badge:
        badge_edge = shift_badge_color or badge_color or facecolor
        badge_text_color = shift_badge_color or badge_color or facecolor
        ax.text(x + w * 0.35, y + h / 2, value_text, ha="center", va="center",
                fontsize=BLOCK_NUM_FS, weight="bold", color="white", zorder=zorder + 1)
        ax.text(
            x + w * 0.78, y + h / 2, badge,
            ha="center", va="center", fontsize=BLOCK_LABEL_FS, weight="bold",
            color=badge_text_color, zorder=zorder + 2,
            bbox=dict(boxstyle="circle,pad=0.28", facecolor="white",
                      edgecolor=badge_edge, linewidth=1.8),
        )
    else:
        ax.text(x + w / 2, y + h / 2, value_text, ha="center", va="center",
                fontsize=BLOCK_NUM_FS, weight="bold", color="white", zorder=zorder + 1)


def _score_cell(ax, x, y, w, h, total, edgecolor, zorder=3, fmt=".2f"):
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, facecolor=TRIAL_SCORE_FILL, edgecolor=edgecolor, linewidth=2.5,
        zorder=zorder,
    ))
    text = f"{total:.0f}" if fmt == ".0f" else f"{total:{fmt}}"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=BLOCK_NUM_FS, weight="bold", color=INK, zorder=zorder + 1)


def _relative_label(col_j, val_self, val_other, trade_cols):
    if col_j not in trade_cols:
        return None
    diff = int(round(val_self - val_other))
    if diff > 0:
        return f"+{diff}"
    if diff < 0:
        return str(diff)
    return None


def draw_trial_block(
    ax,
    example: dict,
    x0: float = 0.0,
    *,
    left_margin: float = -0.55,
    show_scores: bool = True,
    show_b_relative: bool = True,
    show_option_labels: bool = True,
    show_preference: bool = False,
    preference_under_options: bool = False,
    emphasize_concave_preference: bool = False,
    emphasize_linear_preference: bool = False,
    concave_preference_color: str | None = None,
    panel_label: str | None = None,
    panel_label_color: str | None = None,
    panel_label_position: str = "bottom",
    panel_background: str | None = None,
    tile_shift_badge: str | None = None,
    tile_shift_badge_color: str | None = None,
) -> dict:
    """Draw one A/B trial block; returns option_scores dict."""
    scores = option_scores(example)
    a, b = scores["a"], scores["b"]
    n = len(a)
    trade_cols = scores["trade_cols"]
    h3 = scores["h3"]

    cell_w = 1.0
    score_w = 1.1
    score_gap = 0.15
    row_h = 1.0
    row_gap = ROW_GAP
    y_b = 0.0
    y_a = y_b + row_h + row_gap
    concave_x = x0 + n * cell_w + 0.25
    linear_x = concave_x + score_w + score_gap
    tag_y = y_a + row_h + 0.55
    label_x = left_margin + x0 + 0.2
    block_right = linear_x + score_w + 0.05
    pref_y = y_b - 0.32

    if panel_background:
        bg_bottom = pref_y - 0.58 if show_preference else y_b - 0.12
        bg_left = min(x0 - 0.08, label_x - 0.25) if show_preference else x0 - 0.08
        ax.add_patch(mpatches.Rectangle(
            (bg_left, bg_bottom), block_right - bg_left + 0.08,
            tag_y - bg_bottom + 0.12,
            facecolor=panel_background, edgecolor="none", zorder=0, alpha=0.55,
        ))

    if panel_label and panel_label_position == "top":
        ax.text(
            x0 + (block_right - x0) / 2, tag_y + 0.38, panel_label,
            ha="center", va="bottom", fontsize=BLOCK_LABEL_FS, weight="bold",
            color=panel_label_color or INK,
        )

    if not h3:
        for j in range(n):
            cx = x0 + j * cell_w + cell_w / 2
            steep_i, flat_i = trade_cols[0], trade_cols[1]
            if j == steep_i:
                tag = "increase in\nsteep region"
                tag_color = TRIAL_STEEP_LABEL
            elif j == flat_i:
                tag = "decrease in\nflat region"
                tag_color = TRIAL_FLAT_LABEL
            else:
                tag, tag_color = "tied", INK
            ax.text(cx, tag_y, tag, ha="center", va="bottom", fontsize=BLOCK_LABEL_FS,
                    weight="bold", color=tag_color, linespacing=1.12)

    if show_scores:
        ax.text(concave_x + score_w / 2, tag_y, "concave sum",
                ha="center", va="bottom", fontsize=BLOCK_LABEL_FS, color=INK)
        ax.text(linear_x + score_w / 2, tag_y, "linear sum",
                ha="center", va="bottom", fontsize=BLOCK_LABEL_FS, color=INK)

    def draw_option_row(y, ratings, other, option_color, option_label, concave_total,
                        linear_total, show_relative=False, draw_label=True):
        if draw_label and show_option_labels:
            ax.text(label_x, y + row_h / 2, option_label, ha="right", va="center",
                    fontsize=BLOCK_LABEL_FS, weight="bold", color=option_color)
        for j in range(n):
            x = x0 + j * cell_w
            rel = (_relative_label(j, ratings[j], other[j], trade_cols)
                   if show_relative else None)
            _tile(ax, x, y, cell_w, row_h, option_color, str(int(ratings[j])),
                  relative=rel, badge_color=option_color,
                  shift_badge=tile_shift_badge if rel is None else None,
                  shift_badge_color=tile_shift_badge_color)
        if show_scores:
            _score_cell(ax, concave_x, y, score_w, row_h, concave_total, option_color, fmt=".1f")
            _score_cell(ax, linear_x, y, score_w, row_h, linear_total, option_color, fmt=".0f")

    draw_option_row(y_a, a, b, TRIAL_COLOR_A, "Option A",
                    scores["concave_a"], scores["linear_a"])
    draw_option_row(y_b, b, a, TRIAL_COLOR_B, "Option B",
                    scores["concave_b"], scores["linear_b"],
                    show_relative=show_b_relative)

    if show_preference and show_scores:
        lin_pref = scores["linear_b"] - scores["linear_a"]
        con_pref = scores["concave_b"] - scores["concave_a"]
        lin_fontsize = BLOCK_NUM_FS
        lin_kw = dict(ha="center", va="top", fontsize=lin_fontsize, weight="bold", zorder=5)
        if emphasize_linear_preference:
            lin_kw["color"] = INK
        else:
            lin_kw["color"] = GRAY
        ax.text(linear_x + score_w / 2, pref_y - 0.08, f"{lin_pref:.0f}", **lin_kw)
        con_color = concave_preference_color or INK
        con_fontsize = BLOCK_NUM_FS
        ax.text(
            concave_x + score_w / 2, pref_y - 0.08, f"{con_pref:.1f}",
            ha="center", va="top", fontsize=con_fontsize, weight="bold",
            color=con_color, zorder=5,
        )

        if preference_under_options and show_option_labels:
            caption_y = pref_y - 0.12
            ax.text(
                label_x, caption_y + 0.14, "Preference",
                ha="right", va="center", fontsize=BLOCK_LABEL_FS, weight="bold", color=INK,
            )
            ax.text(
                label_x, caption_y - 0.14, "(Option B vs A)",
                ha="right", va="center", fontsize=BLOCK_LABEL_FS, weight="bold", color=INK,
            )

    if panel_label and panel_label_position == "bottom":
        label_y = pref_y - 0.55 if show_preference else y_b - 0.42
        ax.text(x0 + (block_right - x0) / 2, label_y, panel_label,
                ha="center", va="top", fontsize=BLOCK_LABEL_FS, weight="bold",
                color=panel_label_color or INK)

    scores["layout"] = {
        "x0": x0,
        "total_w": block_right - x0 + 0.15,
        "tag_y": tag_y,
        "region_y": tag_y + 0.38,
        "y_a": y_a,
        "y_b": y_b,
        "pref_y": pref_y if show_preference else y_b - 0.32,
        "concave_x": concave_x,
        "linear_x": linear_x,
        "score_w": score_w,
        "label_x": label_x,
        "block_right": block_right,
        "panel_center_x": x0 + (block_right - x0) / 2,
    }
    return scores


def model_diffs(a, b) -> dict:
    """Each rival model's B - A decision variable (negative => prefers A).

    * wadd     : linear (unweighted) sum difference
    * tallying : (# features B wins) - (# features A wins)
    * ttb      : the most-valid DIFFERING feature decides (features assumed
                 ordered left-to-right by validity), reported as that feature's
                 B - A gap
    * concave  : Diminishing Returns sum difference (from option_scores)
    """
    a = [int(v) for v in a]
    b = [int(v) for v in b]
    sc = option_scores({"option_a": a, "option_b": b})
    ttb = 0
    for ai, bi in zip(a, b):
        if ai != bi:
            ttb = bi - ai
            break
    return {
        "concave": sc["concave_b"] - sc["concave_a"],
        "wadd": sum(b) - sum(a),
        "tallying": sum(bi > ai for ai, bi in zip(a, b))
        - sum(ai > bi for ai, bi in zip(a, b)),
        "ttb": ttb,
    }


def rival_scores(a, b, rival: str):
    """Each rival's per-option decision value (A, B) -- NOT all "sums":
      * wadd     -> weighted (here unit-weight) additive total
      * tallying -> number of features that option wins
      * ttb      -> value on the most-valid DIFFERING feature
    """
    a = [int(v) for v in a]
    b = [int(v) for v in b]
    if rival == "wadd":
        return float(sum(a)), float(sum(b))
    if rival == "tallying":
        return (float(sum(ai > bi for ai, bi in zip(a, b))),
                float(sum(bi > ai for ai, bi in zip(a, b))))
    if rival == "ttb":
        for ai, bi in zip(a, b):
            if ai != bi:
                return float(ai), float(bi)
        return 0.0, 0.0
    raise ValueError(rival)


# What each model's per-option number is called (never "sum" for tally/ttb).
RIVAL_VALUE_NAME = {"wadd": "WADD", "tallying": "tally", "ttb": "TTB cue"}


def load_json(name: str, data_dir: Path | None = None) -> dict:
    path = (data_dir or DATA_DIR) / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)
