"""Plot how AutoPI converged to its final two theories over rounds.

Focused on one run — the human decision-making task in
`online/dmb/firebase_prolific/prolific1/ttb+wadd`. AutoPI keeps **two theory
slots**. Each round it proposes one new theory and an arbiter either kills a
slot's occupant and admits the new one (`new_theory`) or keeps both. Over five
rounds the slots evolve:

    slot 1 (churns):   TTB -> Heuristic-Selection -> Rank-Decay
                           -> Threshold-Gate -> Strategy-Mixture   (final)
    slot 2 (stabilises): WADD -> Non-linear Subjective Weighting   (final)
                           (survives every round after it is admitted)

The script writes two standalone figures (no panel labels, no titles — each
fills its own canvas):

* **autopi_lineage — theory lineage.** A slot x round grid of theory boxes
  (full names), with a one-line *reasoning step* on each round's kill->admit
  transition. The two final theories are starred and colour-highlighted.
* **autopi_leaderboard — leaderboard-score trajectory.** Per-round
  end-of-round (post-admit) score of every theory. The score is a *relative*
  cross-experiment fit to the human data (higher = better; worst theory = 0),
  recomputed each round as experiments accumulate — so the signal is *rank
  persistence*, not the absolute value. The final survivors are highlighted;
  the rest are greyed.

This script is written for `new_theory`-verdict runs (every round admits a
replacement). Only the data extraction is unit-tested (see
tests/test_plot_autopi_convergence); the matplotlib drawing is iterated visually.

Run:  python scripts/plot_autopi_convergence.py
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    GRAY,
    NEUTRAL_HARMONY,
    ROLE_COLOR,
    save_figure,
    style_axes,
)
from scripts.plot_seed_gt_control import parse_post_admit_scores  # noqa: E402

DEFAULT_RUN_DIR = Path(
    "results/online/dmb/firebase_prolific/prolific1/ttb+wadd"
)

# These are full-page figures meant for print (PNAS): the shared house base
# (FONTSIZE = 14) is too small once a panel is scaled to column width, so this
# script works from a larger base and keeps the usual relative hierarchy.
FS = FONTSIZE + 4

# One-line reasoning step per round transition, hand-distilled from each round's
# arbiter rationale (theories.md "## Replacement" section). The text is the
# conceptual move, not the parameter-tweak detail, and is run-specific — so it is
# keyed per run (see `_run_key` / `reasoning_for`). A run with no curated entry
# gets NO annotations, never another run's (mismatched) text.
REASONING_BY_RUN: dict[str, dict[int, str]] = {
    # online/dmb/firebase_prolific/prolific1/ttb+wadd
    "prolific1/ttb+wadd": {
        0: "fixed rules mis-fit adherence\n→ mix TTB & Tallying",
        1: "unify strategies via one\npower exponent on validities",
        2: "stronger competitor:\nrank-decay weighting",
        3: "discrete shift when\nvalidity gaps are large",
        4: "make mixture explicit: TTB×WADD\n(WADD arm = power transform)",
    },
    # heuristic_decision_making/humans/hdm_full_prolific_run_full/ttb+tallying
    "hdm_full_prolific_run/ttb+tallying": {
        0: "drop TTB → weighted-additive\nintegration of all cues",
        1: "replace Tallying with stimulus-\ndriven WADD/heuristic switching",
        2: "concave (power) utility on\ncardinal feature values",
        3: "binarize ratings (satisfice),\nthen weight by validity",
        4: "flexible concave utility\n(free-shift diminishing returns)",
    },
}


def _run_key(run_dir: Path) -> str:
    """Stable per-run key `"<run_id>/<seed1>+<seed2>"` from run_meta."""
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    return f"{meta['run_id']}/{'+'.join(meta['seeds'])}"


def reasoning_for(run_dir: Path) -> dict[int, str]:
    """Curated `{round_idx: reasoning}` for this run, or `{}` if uncurated."""
    return REASONING_BY_RUN.get(_run_key(run_dir), {})


# The per-run "headline" theory: the discovery the figure foregrounds. It is
# starred and coloured yellow (sandy) everywhere; the other final survivor is
# orange (terracotta). Keyed per run (see `_run_key`); an uncurated run falls
# back to the persistent survivor.
HIGHLIGHT_BY_RUN: dict[str, str] = {
    "prolific1/ttb+wadd": "pi_4",                  # Non-linear Subjective Weighting
    "hdm_full_prolific_run/ttb+tallying": "pi_7",  # Diminishing Returns WADD
}


def highlight_label(run_dir: Path) -> str | None:
    """Curated headline-theory label for this run, or None if uncurated."""
    return HIGHLIGHT_BY_RUN.get(_run_key(run_dir))


# --- data structures ----------------------------------------------------------
class SlotState(NamedTuple):
    label: str
    name: str
    killed: bool


class Replacement(NamedTuple):
    label: str
    name: str
    slot: int


class RoundLineage(NamedTuple):
    round_idx: int
    slots: dict[int, SlotState]  # {slot_number: SlotState} at round start
    replacement: Replacement  # theory admitted into `slot` for the next round


# --- extraction ---------------------------------------------------------------
def theory_title(description: str) -> str:
    """Short theory name from an AutoPI theory description.

    Descriptions name the theory up front in one of two formats, then launch
    into prose: "Name: ..." or "Name [theory] posits that ...". Take the text
    before whichever delimiter comes first, and drop a trailing "theory".
    Seeds (TTB/Tallying/WADD) are bare prose with neither delimiter; use
    `seed_names` for their canonical labels instead.
    """
    text = description.strip()
    cut = len(text)
    for sep in (":", " posits "):
        i = text.find(sep)
        if i != -1:
            cut = min(cut, i)
    title = text[:cut].strip()
    if title.endswith(" theory"):
        title = title[: -len(" theory")].strip()
    return title


def _round_dirs(run_dir: Path) -> list[Path]:
    return sorted((Path(run_dir) / "rounds").glob("round_*"))


def parse_lineage(run_dir: Path) -> list[RoundLineage]:
    """Per-round slot occupants, kills, and the admitted replacement.

    Assumes every round has a `replacement` (true for `new_theory` runs).
    """
    lineage: list[RoundLineage] = []
    for rd in _round_dirs(run_dir):
        d = json.loads((rd / "theories.json").read_text())
        slots = {
            int(s["slot"]): SlotState(
                label=s["label"],
                name=theory_title(s["theory"]["description"]),
                killed=bool(s["killed"]),
            )
            for s in d["starting_theories"]
        }
        rep = d["replacement"]
        lineage.append(
            RoundLineage(
                round_idx=int(d["round_idx"]),
                slots=slots,
                replacement=Replacement(
                    label=rep["label"],
                    name=theory_title(rep["theory"]["description"]),
                    slot=int(rep["slot"]),
                ),
            )
        )
    return lineage


def final_survivors(lineage: list[RoundLineage]) -> set[str]:
    """Labels of the two theories standing after the last round's admission.

    The surviving slot keeps its occupant; the killed slot is taken by the
    admitted theory.
    """
    return set(survivor_names(lineage))


def survivor_names(lineage: list[RoundLineage]) -> dict[str, str]:
    """`{label: full_name}` for the two final theories.

    The surviving slot keeps its occupant; the killed slot is taken by the
    admitted theory (the newcomer).
    """
    last = lineage[-1]
    names = {s.label: s.name for s in last.slots.values() if not s.killed}
    names[last.replacement.label] = last.replacement.name
    return names


# Seeds are stored lower-case in run_meta. Display capitalises the acronyms
# (TTB, WADD) and title-cases the word "Tallying"; any unlisted seed falls back
# to upper-case.
SEED_DISPLAY: dict[str, str] = {"ttb": "TTB", "wadd": "WADD", "tallying": "Tallying"}


def _seed_display(seed: str) -> str:
    return SEED_DISPLAY.get(seed.lower(), seed.upper())


# Verbose theory names shortened to their canonical acronym in the lineage
# boxes (keyed case-insensitively; unlisted names pass through unchanged).
DISPLAY_ABBREV: dict[str, str] = {"weighted additive (wadd)": "WADD"}


def _abbrev(name: str) -> str:
    return DISPLAY_ABBREV.get(name.lower(), name)


def seed_names(run_dir: Path) -> dict[str, str]:
    """Map each round-0 slot occupant label to its canonical seed name.

    run_meta `seeds` lists the seeds in slot order (slot 1, slot 2, ...).
    """
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    seeds = [_seed_display(s) for s in meta["seeds"]]
    round0 = json.loads(
        (Path(run_dir) / "rounds" / "round_000" / "theories.json").read_text()
    )
    starting = sorted(round0["starting_theories"], key=lambda s: s["slot"])
    return {s["label"]: seeds[i] for i, s in enumerate(starting)}


def load_post_admit_scores(run_dir: Path) -> list[dict[str, float]]:
    """Per-round end-of-round (post-admit) leaderboard scores."""
    return parse_post_admit_scores((Path(run_dir) / "leaderboard.md").read_text())


# --- colours ------------------------------------------------------------------
def survivor_colors(
    lineage: list[RoundLineage], highlight: str | None = None
) -> dict[str, str]:
    """Highlight colours for the two final theories.

    The headline theory (`highlight`, or the persistent survivor when it is None
    or not a survivor) gets sandy (yellow); the other final survivor gets
    terracotta (orange).
    """
    survivors = final_survivors(lineage)
    hi = highlight if highlight in survivors else persistent_survivor(lineage)
    other = next(lbl for lbl in survivors if lbl != hi)
    return {
        hi: NEUTRAL_HARMONY["sandy"],
        other: NEUTRAL_HARMONY["terracotta"],
    }


def persistent_survivor(lineage: list[RoundLineage]) -> str:
    """Label of the slot still occupied (not killed) after the last round.

    The default headline theory when a run has no curated `highlight`.
    """
    last = lineage[-1]
    return next(s.label for s in last.slots.values() if not s.killed)


# Per-run seed colour pins: force a named seed to a curated colour in THIS
# run's figures only (keyed by `_run_key`, then canonical seed name). Other
# theories and other runs are unaffected, so the override is scoped per folder.
SEED_COLOR_OVERRIDE_BY_RUN: dict[str, dict[str, str]] = {
    # TTB -> sage (teal) in both human-dataset runs.
    "prolific1/ttb+wadd": {"TTB": NEUTRAL_HARMONY["sage"]},
    "hdm_full_prolific_run/ttb+tallying": {"TTB": NEUTRAL_HARMONY["sage"]},
}


def seed_color_overrides(run_dir: Path) -> dict[str, str]:
    """`{label: colour}` pinning specific seeds to a curated colour for this
    run only (resolved from canonical seed name). Empty for uncurated runs."""
    by_name = SEED_COLOR_OVERRIDE_BY_RUN.get(_run_key(run_dir), {})
    return {lbl: by_name[name]
            for lbl, name in seed_names(run_dir).items() if name in by_name}


def theory_colors(run_dir: Path) -> dict[str, str]:
    """`{label: colour}` for every theory in the run, matching the convergence
    palette: seeds -> slate, highlight -> sandy (yellow), other survivor ->
    terracotta (orange), transient -> gray. A per-run `seed_color_overrides`
    pin (e.g. TTB -> sage here) takes precedence for the named seeds.

    Single source of truth so other figures (e.g. the human MSE/Pearson
    trajectory) colour theories identically to this convergence plot.
    """
    lineage = parse_lineage(run_dir)
    seeds = seed_names(run_dir)
    survivors = survivor_colors(lineage, highlight_label(run_dir))
    overrides = seed_color_overrides(run_dir)
    labels = {s.label for r in lineage for s in r.slots.values()}
    labels |= {r.replacement.label for r in lineage}
    return {lbl: _label_color(lbl, seeds, survivors, overrides) for lbl in labels}


def _label_color(
    label: str, seeds: dict[str, str], survivors: dict[str, str],
    overrides: dict[str, str] | None = None,
) -> str:
    if overrides and label in overrides:
        return overrides[label]  # curated per-run pin (e.g. TTB -> sage)
    if label in survivors:
        return survivors[label]
    if label in seeds:
        return ROLE_COLOR["seed"]  # slate
    return GRAY  # transient (proposed then killed)


def _seed_legend_label(seeds: dict[str, str]) -> str:
    """Legend text naming this run's seeds, e.g. "seeds (TTB, Tallying)"."""
    return "seeds (" + ", ".join(seeds[lbl] for lbl in sorted(seeds)) + ")"


def _text_on(facecolor: str) -> str:
    """Readable text colour (dark on light fills, white on dark fills)."""
    r, g, b = to_rgb(facecolor)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 0.6 else "white"


def _column_label(position: int) -> str:
    """Lineage column header for the column at `position` (0-indexed, left→right).

    Position 0 holds the seeds (the state before any proposal); positions 1..N
    are the end states of cycles 1..N — one per proposal/admission round, with
    the right-most column being the final post-admission state (cycle N).
    """
    return "seed" if position == 0 else f"cycle {position}"


# --- panel A: lineage grid ----------------------------------------------------
def _occupancy(lineage: list[RoundLineage]):
    """Grid `{(slot, col): (label, name, killed)}`.

    Columns are the round indices 0..N-1 plus the string "final" (the slot
    state after the last round's admission).
    """
    occ: dict[tuple[int, object], tuple[str, str, bool]] = {}
    for r in lineage:
        for slot, s in r.slots.items():
            occ[(slot, r.round_idx)] = (s.label, s.name, s.killed)
    last = lineage[-1]
    for slot, s in last.slots.items():
        if not s.killed:
            occ[(slot, "final")] = (s.label, s.name, False)
    occ[(last.replacement.slot, "final")] = (
        last.replacement.label, last.replacement.name, False,
    )
    return occ


def draw_lineage(
    ax, lineage, seeds, survivors, reasoning, highlight=None, overrides=None,
) -> None:
    columns: list[object] = [r.round_idx for r in lineage] + ["final"]
    col_x = {c: i for i, c in enumerate(columns)}
    slot_y = {1: 1.5, 2: 0.0}  # slot 1 on top
    bw, bh = 0.47, 0.46  # box half-width / half-height

    for (slot, col), (label, name, killed) in _occupancy(lineage).items():
        x, y = col_x[col], slot_y[slot]
        color = _label_color(label, seeds, survivors, overrides)
        is_final = label in survivors and col == "final"
        is_star = is_final and label == highlight  # star only the headline theory
        is_highlight = label in survivors or label in seeds
        ax.add_patch(FancyBboxPatch(
            (x - bw, y - bh), 2 * bw, 2 * bh,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=2.4 if is_final else 1.0,
            edgecolor="#111111" if is_final else color,
            facecolor=color, alpha=1.0 if is_highlight else 0.6, zorder=2,
        ))
        display = _abbrev(seeds.get(label, name))
        txt = textwrap.fill(display, width=15)
        ax.text(
            x, y, ("★ " + txt) if is_star else txt, ha="center", va="center",
            fontsize=FS - 5, color=_text_on(color), zorder=3,
            fontweight="bold" if is_final else "normal",
        )
        if killed:
            ax.text(x + bw - 0.09, y + bh - 0.09, "✗", ha="center",
                    va="center", fontsize=FS + 2, color="#7a1f1f",
                    fontweight="bold", zorder=4)

    # Kill -> admit arrow on each transition (within the killed slot), with the
    # curated reasoning step above/below it when available for this run.
    for r in lineage:
        slot = r.replacement.slot
        c0, y = col_x[r.round_idx], slot_y[slot]
        ax.annotate("", xy=(c0 + 1 - bw - 0.02, y), xytext=(c0 + bw + 0.02, y),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4),
                    zorder=1)
        reason = reasoning.get(r.round_idx)
        if reason:
            ty = y + (bh + 0.40) if slot == 1 else y - (bh + 0.40)
            # Re-wrap to one column's width; the hand-authored line breaks were
            # tuned for a smaller font and overflow into neighbours at this size.
            wrapped = textwrap.fill(reason.replace("\n", " "), width=16)
            ax.text(c0 + 0.5, ty, wrapped, ha="center", va="center",
                    fontsize=FS - 7, color="#333333", style="italic",
                    zorder=3)

    for slot, y in slot_y.items():
        ax.text(-0.85, y, f"slot {slot}", ha="right", va="center",
                fontsize=FS - 3, fontweight="bold", color="#333333")
    # Left-most column is the seed state; the rest are cycles 1..N.
    for c in columns:
        ax.text(col_x[c], -1.45, _column_label(col_x[c]),
                ha="center", va="center", fontsize=FS - 3, color="#333333")

    ax.set_xlim(-1.5, len(columns) - 0.4)
    ax.set_ylim(-1.75, 2.8)
    ax.axis("off")


# --- panel B: score trajectory ------------------------------------------------
def draw_scores(ax, rounds_scores, seeds, survivors, survivor_full,
                highlight=None, overrides=None) -> None:
    labels = sorted({lbl for sc in rounds_scores for lbl in sc})
    xs = list(range(len(rounds_scores)))

    def series(label):
        return [sc.get(label, float("nan")) for sc in rounds_scores]

    # Transient/seed lines first (thin, behind); survivors on top (thick).
    for label in labels:
        if label in survivors:
            continue
        ax.plot(xs, series(label),
                color=_label_color(label, seeds, survivors, overrides),
                lw=1.4, alpha=0.55, marker="o", markersize=3, zorder=2)

    # Draw the highlighted survivor last so it sits on top; only it is starred.
    for label in sorted(survivors, key=lambda l: l == highlight):
        ys = series(label)
        ax.plot(xs, ys, color=survivors[label], lw=3.0, marker="o",
                markersize=8, zorder=4, markeredgecolor="#111111",
                markeredgewidth=0.8)
        if label == highlight:
            fx = max(i for i, v in enumerate(ys) if v == v)
            ax.plot([fx], [ys[fx]], marker="*", markersize=16,
                    color=survivors[label], markeredgecolor="#111111",
                    markeredgewidth=0.8, zorder=5)

    handles = [
        Line2D([0], [0], color=survivors[lbl], lw=3,
               marker="*" if lbl == highlight else "o",
               markersize=12 if lbl == highlight else 8,
               label=survivor_full[lbl] + ("  ★" if lbl == highlight else ""))
        for lbl in sorted(survivors, key=lambda l: l != highlight)
    ]
    handles.append(Line2D([0], [0], color=ROLE_COLOR["seed"], lw=1.4, alpha=0.7,
                          marker="o", markersize=4,
                          label=_seed_legend_label(seeds)))
    handles.append(Line2D([0], [0], color=GRAY, lw=1.4, alpha=0.6, marker="o",
                          markersize=4, label="other (proposed then killed)"))
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=FS - 7, handlelength=1.6)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"cycle {i + 1}" for i in xs])
    ax.set_ylim(-0.03, 1.08)
    style_axes(ax, xlabel="Cycle",
               ylabel="Leaderboard score\n(relative fit; higher = better)")
    # Enlarge axis text for print; style_axes used the smaller house base.
    ax.xaxis.label.set_size(FS)
    ax.yaxis.label.set_size(FS)
    ax.tick_params(axis="both", which="major", labelsize=FS - 3)


def main(run_dir: Path = DEFAULT_RUN_DIR) -> list[Path]:
    lineage = parse_lineage(run_dir)
    seeds = seed_names(run_dir)
    # Curated headline theory, or the persistent survivor for uncurated runs,
    # so the sandy (yellow) theory is also the one that gets the star.
    highlight = highlight_label(run_dir) or persistent_survivor(lineage)
    survivors = survivor_colors(lineage, highlight)
    survivor_full = survivor_names(lineage)
    overrides = seed_color_overrides(run_dir)
    scores = load_post_admit_scores(run_dir)

    out_dir = Path(run_dir) / "analysis" / "convergence"
    written: list[Path] = []

    # The lineage grid and the score trajectory are saved as two standalone
    # figures (no A/B labels, no titles) so each one fills its whole canvas.
    fig_lineage, ax_lineage = plt.subplots(figsize=(14, 7.5))
    draw_lineage(ax_lineage, lineage, seeds, survivors, reasoning_for(run_dir),
                 highlight, overrides)
    fig_lineage.tight_layout(pad=0.4)
    written += save_figure(fig_lineage, out_dir / "autopi_lineage")
    plt.close(fig_lineage)

    fig_leaderboard, ax_leaderboard = plt.subplots(figsize=(9, 6))
    draw_scores(ax_leaderboard, scores, seeds, survivors, survivor_full,
                highlight, overrides)
    fig_leaderboard.tight_layout(pad=0.6)
    written += save_figure(fig_leaderboard, out_dir / "autopi_leaderboard")
    plt.close(fig_leaderboard)

    print("wrote:", *(str(p) for p in written), sep="\n  ")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR,
        help="AutoPI run directory (holds rounds/, leaderboard.md, run_meta.json)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args().run_dir)
