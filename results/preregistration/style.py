"""Shared style configuration for all figures in this bundle."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================================================================
# Edit these to change the look of every figure
# =============================================================================

FONTS = {
    "family": "sans-serif",
    "sans_serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "size_base": 10,
    "size_annotation": 9.5,
    "size_region": 10,
    "size_title": 12,
    "size_axis_label": 11,
    "size_tick": 9,
}

# Neutral-harmony palette (from style_example.py / figure_style.py)
NEUTRAL_HARMONY = {
    "cream": "#f4f1de",
    "terracotta": "#e07a5f",
    "slate": "#3d405b",
    "sage": "#81b29a",
    "sandy": "#f2cc8f",
}
NH = NEUTRAL_HARMONY
INK = "#111111"
GRAY = "#999999"

# =============================================================================
# FIGURE COLOURS  -- SINGLE SOURCE OF TRUTH. One named constant == one meaning.
# Edit a colour here and it updates the example values AND the result bars.
# =============================================================================
# Decision models (each owns one colour; concave is the protagonist):
CONCAVE = "#e07a5f"    # Diminishing Returns (concave)
WADD = "#3d405b"       # WADD / linear
TALLYING = "#577590"   # Tallying
TTB = "#b5838d"        # TTB
# In the flagship H1 every rival shares ONE colour so only the concave
# protagonist stands out; edit this to recolour all three at once.
RIVAL = "#3d405b"      # WADD / Tallying / TTB (H1)
# Within-trial option identity (kept NEUTRAL so model colours stay unambiguous):
OPTION_A = "#abb0bc"   # light grey tile (Option A)
OPTION_B = "#6b7280"   # dark grey tile (Option B)
# Region identity (a separate dimension from models / options):
STEEP = "#c8862a"      # gold
FLAT = "#9aa0ad"       # cool grey
# Region washes under the hero curve (pale tints of the region colours):
STEEP_FILL = "#f2cc8f"  # pale gold wash (steep region)
FLAT_FILL = "#dfe2e8"   # pale cool-grey wash (flat region)
# Misc roles:
SHIFT = "#81b29a"      # green: the act of shifting the range (H3)
CHANCE = GRAY          # chance / null reference line
LABEL = "#3f4350"      # neutral row labels

# Keyed lookups derived from the constants above (for code that loops by model).
MODEL_KEYS = ["pi_7", "wadd", "tallying", "ttb"]
PALETTE = {
    "concave": CONCAVE, "wadd": WADD, "tallying": TALLYING, "ttb": TTB,
    "option_a": OPTION_A, "option_b": OPTION_B,
    "steep": STEEP, "flat": FLAT, "shift": SHIFT, "chance": CHANCE,
}
# Back-compat aliases used across the bundle.
DR_COLOR = CONCAVE
REGION_STEEP = STEEP
REGION_FLAT = FLAT

# Semantic colors — same palette, different roles per figure type.
COLORS = {
    # Value function: slate curve; warm steep / cool flat regions.
    "curve": NH["slate"],
    "point": NH["slate"],
    "point_edge": "white",
    "guide": NH["sage"],
    "objective_arrow": NH["slate"],
    "objective_label": INK,
    "subjective_arrow": INK,
    "subjective_label": INK,
    "steep_region_fill": STEEP_FILL,
    "flat_region_fill": FLAT_FILL,
    "steep_region_label": STEEP,
    "flat_region_label": FLAT,
    # Shared text / axes (dark, to match the paper's other result figures)
    "title": INK,
    "axis_spine": "#222222",
    "tick": "#222222",
    "background": "white",
    "chance": GRAY,
    "error_bar": INK,
    # H2 / H3: low-range & steep = terracotta; high-range & shifted = slate.
    "h2": NH["terracotta"],
    "h3_low": NH["terracotta"],
    "h3_high": NH["slate"],
    # H1: Diminishing Returns vs alternative model.
    "h1_dr": NH["terracotta"],
    "h1_alt": NH["slate"],
}

# Trial block diagram (steep example)
TRIAL_COLOR_A = NH["terracotta"]
TRIAL_COLOR_B = NH["slate"]
TRIAL_COLOR_BAR = NH["sage"]
TRIAL_STEEP_LABEL = NH["terracotta"]
TRIAL_FLAT_LABEL = NH["slate"]
TRIAL_SCORE_FILL = NH["cream"]
TRIAL_HIGHLIGHT_LOW = NH["sandy"]
TRIAL_HIGHLIGHT_HIGH = NH["cream"]
FIGURE_FONTSIZE = 14

MODEL_ORDER = ["pi_7", "wadd", "tallying", "ttb"]
MODEL_LABELS = {
    "pi_7": "Diminishing\nReturns WADD",
    "wadd": "WADD",
    "tallying": "Tallying",
    "ttb": "TTB",
}
MODEL_COLORS = {
    "pi_7": CONCAVE,
    "wadd": WADD,
    "tallying": TALLYING,
    "ttb": TTB,
}

BUNDLE_DIR = Path(__file__).resolve().parent
DATA_DIR = BUNDLE_DIR / "data"
OUTPUT_DIR = BUNDLE_DIR / "output"

VALUE_K = 0.52
STEEP_REGION_X = (0, 1.2)
FLAT_REGION_X = (3.8, 5)


def value(v, k: float = VALUE_K):
    import numpy as np
    v = np.asarray(v, float)
    return (1 - np.exp(-k * v)) / (1 - np.exp(-k * 5))


def apply_style() -> None:
    rcParams.update(
        {
            "font.family": FONTS["family"],
            "font.sans-serif": FONTS["sans_serif"],
            "font.size": FONTS["size_base"],
            "axes.linewidth": 1.1,
            "axes.labelsize": FONTS["size_axis_label"],
            "axes.titlesize": FONTS["size_title"],
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis_spine"])
    ax.spines["bottom"].set_color(COLORS["axis_spine"])
    ax.tick_params(colors=COLORS["tick"], labelsize=FONTS["size_tick"],
                   direction="out", length=3.5, width=1.0)


def save_figure(fig, basename: str, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kw = dict(bbox_inches="tight", facecolor=COLORS["background"])
    paths = []
    for fmt in ("svg", "pdf"):
        path = out_dir / f"{basename}.{fmt}"
        fig.savefig(path, format=fmt, **save_kw)
        paths.append(path)
    return paths
