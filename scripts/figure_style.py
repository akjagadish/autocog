"""Shared publication figure style for autopi plots.

One palette (the user's "neutral harmony" scheme), one base font size, and
small helpers so every plotting script styles axes and saves figures the
same way (SVG + PNG, dpi=300, tight bbox).
"""
from __future__ import annotations

from pathlib import Path

FONTSIZE = 14

# Neutral-harmony palette (user-specified, 2026-06-10).
NEUTRAL_HARMONY: dict[str, str] = {
    "cream": "#f4f1de",
    "terracotta": "#e07a5f",
    "slate": "#3d405b",
    "sage": "#81b29a",
    "sandy": "#f2cc8f",
}

# Off-palette neutrals: chance baselines and text/ink.
GRAY = "#999999"
INK = "#111111"

# Model roles (base/seed model vs surfaced theory vs ground truth).
ROLE_COLOR: dict[str, str] = {
    "base": NEUTRAL_HARMONY["slate"],
    "seed": NEUTRAL_HARMONY["slate"],
    "surfaced": NEUTRAL_HARMONY["terracotta"],
    "surfaced (best)": NEUTRAL_HARMONY["sandy"],
    "surfaced (best per run)": NEUTRAL_HARMONY["sandy"],
    "surfaced (best across runs)": NEUTRAL_HARMONY["cream"],
    "gt": NEUTRAL_HARMONY["sage"],
    "gt-floor": NEUTRAL_HARMONY["sage"],
    # gt_clean: clean gt vs its noisy version — same green as gt.
    "gt_clean": NEUTRAL_HARMONY["sage"],
    "random": GRAY,
}

# Legend display names for model roles (capitalised, human-readable).
ROLE_DISPLAY: dict[str, str] = {
    "base": "Seed",
    "seed": "Seed",
    "surfaced": "Surfaced",
    "surfaced (best per run)": "Surfaced (Best)",
    "surfaced (best across runs)": "Surfaced (Best across runs)",
    "gt": "Ground truth",
    "gt_clean": "GT vs noisy GT",
    "random": "Random",
}


def role_label(role: str) -> str:
    """Legend label for a model role, falling back to the role name itself."""
    return ROLE_DISPLAY.get(role, role)


# Ground-truth heuristic families.
FAMILY_COLOR: dict[str, str] = {
    "ttb": NEUTRAL_HARMONY["terracotta"],
    "wadd": NEUTRAL_HARMONY["slate"],
    "tallying": NEUTRAL_HARMONY["sage"],
}

# Judge metrics (family vs algorithm match).
METRIC_COLOR: dict[str, str] = {
    "gt_family_match": NEUTRAL_HARMONY["slate"],
    "gt_algorithm_match": NEUTRAL_HARMONY["terracotta"],
}

# Categorical cycle for unkeyed series (e.g. one color per ground truth).
CYCLE: tuple[str, ...] = (
    NEUTRAL_HARMONY["terracotta"],
    NEUTRAL_HARMONY["slate"],
    NEUTRAL_HARMONY["sage"],
    NEUTRAL_HARMONY["sandy"],
    GRAY,
)


def style_axes(ax, *, xlabel: str | None = None, ylabel: str | None = None,
               title: str | None = None) -> None:
    """Apply the house style to one axes: despine, big labels, styled ticks."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path_base: Path | str) -> list[Path]:
    """Save `fig` as both SVG and PNG next to each other; returns the paths.

    `path_base` may carry a suffix (it is replaced); parents are created.
    Callers run `fig.tight_layout()` themselves (some figures need a `rect`).
    """
    base = Path(path_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".svg", ".png"):
        out = base.with_suffix(suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    return written
