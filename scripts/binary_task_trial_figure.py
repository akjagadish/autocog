"""Standalone illustration of a single binary-feature decision trial.

Reproduces the side-by-side task cartoon (validity bars over Option A on the
left, Option B on the right, feature labels under B) but with BINARY feature
values in the option boxes instead of graded ratings.

The binary values use the "per-feature winner" convention: on each feature the
option with the higher original rating gets 1, the other 0. With the canonical
illustrative ratings (A = 4,2,2,2,2 / B = 3,4,4,4,4) this gives A = 1,0,0,0,0
and B = 0,1,1,1,1 (A wins f1; B wins f2-f5).

Run:  python scripts/binary_task_trial_figure.py
Out:  binary_task_trial.png next to this script's --out (default: repo root).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import INK, NEUTRAL_HARMONY, save_figure

# Neutral-harmony palette (shared via scripts/figure_style.py).
COLOR_A = NEUTRAL_HARMONY["terracotta"]  # Option A
COLOR_B = NEUTRAL_HARMONY["slate"]       # Option B
COLOR_TEXT = INK
COLOR_BAR = NEUTRAL_HARMONY["sage"]      # validity bars (distinct from A/B)

# Illustrative trial.
VALIDITIES = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
A_BINARY = np.array([1, 0, 0, 0, 0])
B_BINARY = np.array([0, 1, 1, 1, 1])


def render_binary_trial(validities=VALIDITIES, a=A_BINARY, b=B_BINARY):
    n_feat = len(a)
    cell_w = 1.0
    gap = 2.0  # horizontal gap between the Option A and Option B blocks

    a_x0 = 0.0
    b_x0 = n_feat * cell_w + gap
    ax_of = lambda j: a_x0 + j * cell_w
    bx_of = lambda j: b_x0 + j * cell_w

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_axis_off()

    y_opt = 0.0          # both option rows share this baseline (side by side)
    bar_base = 1.5       # validity bars sit above Option A
    bar_max_h = 0.55
    max_val = max(1.0, float(validities.max()))

    # Validity bars + value labels (above Option A only).
    for j in range(n_feat):
        h = bar_max_h * (validities[j] / max_val)
        ax.add_patch(mpatches.Rectangle(
            (ax_of(j), bar_base), cell_w, h,
            facecolor=COLOR_BAR, edgecolor="white", linewidth=2.0,
        ))
        ax.text(ax_of(j) + cell_w / 2, bar_base + h + 0.06,
                f"{validities[j]:.2f}", ha="center", va="bottom",
                fontsize=18, color=COLOR_TEXT)
    ax.text(a_x0 + n_feat * cell_w / 2, bar_base + bar_max_h + 0.42,
            "Validity", ha="center", va="bottom",
            fontsize=20, weight="bold", color=COLOR_TEXT)

    # Option A tiles (tan) with binary values.
    for j in range(n_feat):
        ax.add_patch(mpatches.Rectangle(
            (ax_of(j), y_opt), cell_w, 1.0,
            facecolor=COLOR_A, edgecolor="white", linewidth=2.0,
        ))
        ax.text(ax_of(j) + cell_w / 2, y_opt + 0.5, str(int(a[j])),
                ha="center", va="center", fontsize=18, weight="bold",
                color="white")

    # Option B tiles (blue) with binary values.
    for j in range(n_feat):
        ax.add_patch(mpatches.Rectangle(
            (bx_of(j), y_opt), cell_w, 1.0,
            facecolor=COLOR_B, edgecolor="white", linewidth=2.0,
        ))
        ax.text(bx_of(j) + cell_w / 2, y_opt + 0.5, str(int(b[j])),
                ha="center", va="center", fontsize=18, weight="bold",
                color="white")

    # Option labels (left of each block).
    ax.text(a_x0 - 0.3, y_opt + 0.5, "Product A", ha="right", va="center",
            fontsize=12, weight="bold", color=COLOR_A)
    ax.text(b_x0 - 0.3, y_opt + 0.5, "Product B", ha="right", va="center",
            fontsize=12, weight="bold", color=COLOR_B)

    ax.set_xlim(a_x0 - 2.6, b_x0 + n_feat * cell_w + 0.4)
    ax.set_ylim(y_opt - 0.8, bar_base + bar_max_h + 0.9)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=Path("binary_task_trial.png"),
                   help="Output PNG path.")
    args = p.parse_args(argv)
    fig = render_binary_trial()
    written = save_figure(fig, args.out)
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
