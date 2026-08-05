"""Plot mechanism-similarity vs ground-truth noise for the canonical families.

Reads the per-cell `judge_similarity_desc.csv` / `judge_similarity_joint.csv`
written by `scripts/judge_similarity.py` under

    <root>/<family>_sampling/noise=<x>/

draws TWO figures per judge input mode (description, and description + code):

  similarity_vs_noise_<mode>  - faint per-family mean lines, overlaid with a
                                bold BLACK pooled mean ± SEM line.
  similarity_swarm_<mode>     - every individual theory score as a faint
                                jittered point (pooled across families), with
                                the same pooled mean ± SEM line on top.
  similarity_bar_swarm_<mode> - the pooled mean ± SEM as bars per noise level,
                                with the same jittered points behind the bars.

The pooled mean and SEM are over every per-(run, slot) score at a noise level,
across all families (a single model, gemini-3.1-pro-preview, in this sweep).

Usage:
    python scripts/plot_similarity.py [--root <dir>] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.figure_style import (  # noqa: E402
    FAMILY_COLOR, FONTSIZE, GRAY, INK, NEUTRAL_HARMONY, save_figure, style_axes,
)

DEFAULT_ROOT = (
    ROOT / "results" / "heuristic_decision_making"
    / "synthetic_corrected_theories_binary_sampling"
)
FAMILIES: tuple[str, ...] = ("ttb", "tallying", "wadd")
MODE_FILES: dict[str, str] = {
    "description": "judge_similarity_desc.csv",
    "joint": "judge_similarity_joint.csv",
}
MODE_LABEL: dict[str, str] = {"description": "description", "joint": "description + code"}
# Acronyms stay capitalized; 'tallying' is a word, so title-case it.
FAMILY_LABEL: dict[str, str] = {"ttb": "TTB", "wadd": "WADD", "tallying": "Tallying"}
FAMILY_ALPHA = 0.40  # faint family lines behind the bold black pooled line


@dataclass(frozen=True)
class CellStat:
    mean: float
    sem: float
    n: int


def summarize(scores: list[float]) -> CellStat:
    """Mean and standard error of the mean over a cell's similarity scores.

    For n == 1 (no SEM) or identical scores (zero variance) the SEM collapses
    to 0 rather than producing a NaN or a spurious tiny float.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("summarize requires at least one score")
    mean = sum(scores) / n
    if n == 1 or max(scores) == min(scores):
        return CellStat(mean=mean, sem=0.0, n=n)
    sd = statistics.stdev(scores)  # sample std, ddof=1
    return CellStat(mean=mean, sem=sd / math.sqrt(n), n=n)


def read_scores(csv_path: Path) -> list[float]:
    with csv_path.open() as f:
        return [float(r["similarity"]) for r in csv.DictReader(f)
                if r.get("similarity", "") != ""]


def _parse_noise(noise_dir: Path) -> float:
    return float(noise_dir.name.replace("noise=", ""))


def collect_scores(
    root: Path, families: list[str]
) -> dict[tuple[str, float, str], list[float]]:
    """Walk root/<family>_sampling/noise=*/ and return the raw per-(run, slot)
    similarity scores for each (family, noise, mode) cell. Cells with no CSV
    (or a stray non-numeric noise=* dir) are simply absent from the result."""
    out: dict[tuple[str, float, str], list[float]] = {}
    for family in families:
        family_dir = root / f"{family}_sampling"
        if not family_dir.is_dir():
            continue
        for noise_dir in sorted(family_dir.glob("noise=*")):
            if not noise_dir.is_dir():
                continue
            try:
                noise = _parse_noise(noise_dir)
            except ValueError:
                continue  # stray non-numeric noise=* dir: skip like a missing cell

            for mode, fname in MODE_FILES.items():
                csv_path = noise_dir / fname
                if not csv_path.exists():
                    continue
                scores = read_scores(csv_path)
                if scores:
                    out[(family, noise, mode)] = scores
    return out


def collect(
    root: Path, families: list[str]
) -> dict[tuple[str, float, str], CellStat]:
    """Per-(family, noise, mode) summary statistics."""
    return {k: summarize(v) for k, v in collect_scores(root, families).items()}


def filter_max_noise(
    scores: dict[tuple[str, float, str], list[float]], max_noise: float | None
) -> dict[tuple[str, float, str], list[float]]:
    """Keep only cells with noise <= max_noise (all cells when max_noise None)."""
    if max_noise is None:
        return scores
    return {k: v for k, v in scores.items() if k[1] <= max_noise}


def pool_raw_by_noise(
    scores: dict[tuple[str, float, str], list[float]], mode: str
) -> dict[float, list[float]]:
    """For one mode, concatenate every family's raw per-theory scores at each
    noise level (each theory weighted equally, not each family)."""
    by_noise: dict[float, list[float]] = {}
    for (_family, noise, m), vals in scores.items():
        if m != mode:
            continue
        by_noise.setdefault(noise, []).extend(vals)
    return by_noise


def pool_across_families(
    scores: dict[tuple[str, float, str], list[float]], mode: str
) -> dict[float, CellStat]:
    """Mean + dispersion of the pooled per-theory scores at each noise level."""
    return {noise: summarize(vals)
            for noise, vals in pool_raw_by_noise(scores, mode).items()}


def _jitter(n: int, rng, width: float = 0.012):
    """`n` reproducible horizontal offsets uniform in [-width, width]."""
    return rng.uniform(-width, width, size=n)


FIGSIZE = (7, 4)  # match the recovery (MSE / Pearson-r) figures' aspect
XLABEL = r"action noise ($\epsilon$)"
YLABEL = "similarity to ground truth theory"
# A little headroom past [0, 1] so markers at the bounds aren't clipped.
YLIM = (-0.03, 1.05)
# Bars are anchored at 0, so they keep a flush bottom (only top headroom).
BAR_YLIM = (0.0, 1.05)


def _overlay_pooled_mean_sem(ax, scores, mode: str) -> None:
    """Bold black pooled mean across all families with SEM error bars."""
    pooled = sorted(pool_across_families(scores, mode).items())
    if not pooled:
        return
    xs = [noise for noise, _ in pooled]
    means = [s.mean for _, s in pooled]
    yerr = [s.sem for _, s in pooled]
    ax.errorbar(
        xs, means, yerr=yerr, color=INK, linewidth=2.5, marker="o",
        markersize=6, capsize=0, zorder=5,
    )


def _finish(fig, ax, *, legend: bool = True, ylim: tuple[float, float] = YLIM) -> None:
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="lower left", fontsize=FONTSIZE - 2, frameon=False)
    style_axes(ax, xlabel=XLABEL, ylabel=YLABEL)
    fig.tight_layout()


def make_figure(scores: dict[tuple[str, float, str], list[float]], mode: str):
    """Faint per-family mean lines + bold black pooled mean ± SEM."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for family in FAMILIES:
        points = sorted(
            (noise, summarize(vals))
            for (fam, noise, m), vals in scores.items()
            if fam == family and m == mode
        )
        if not points:
            continue
        xs = [noise for noise, _ in points]
        means = [s.mean for _, s in points]
        ax.plot(
            xs, means, color=FAMILY_COLOR[family], alpha=FAMILY_ALPHA,
            linewidth=2, marker="o", markersize=4, label=FAMILY_LABEL[family],
        )
    _overlay_pooled_mean_sem(ax, scores, mode)
    _finish(fig, ax)
    return fig


def make_swarm_figure(
    scores: dict[tuple[str, float, str], list[float]], mode: str, seed: int = 0
):
    """Every individual theory score as a faint jittered point (one neutral
    color, pooled across families/models), with the pooled mean ± SEM on top."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(seed)
    for noise, vals in sorted(pool_raw_by_noise(scores, mode).items()):
        x = noise + _jitter(len(vals), rng)
        ax.scatter(x, vals, color=GRAY, alpha=0.35, s=18, linewidths=0, zorder=1)
    _overlay_pooled_mean_sem(ax, scores, mode)
    _finish(fig, ax, legend=False)
    return fig


def make_bar_swarm_figure(
    scores: dict[tuple[str, float, str], list[float]], mode: str, seed: int = 0
):
    """Pooled mean ± SEM as bars at each noise level, with every individual
    theory score as a faint jittered point behind the bars."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rng = np.random.default_rng(seed)
    by_noise = sorted(pool_raw_by_noise(scores, mode).items())
    positions = list(range(len(by_noise)))
    for pos, (_noise, vals) in zip(positions, by_noise):
        x = pos + _jitter(len(vals), rng, width=0.12)
        ax.scatter(x, vals, color=GRAY, alpha=0.35, s=18, linewidths=0, zorder=1)
    stats = [summarize(vals) for _, vals in by_noise]
    ax.bar(
        positions, [s.mean for s in stats], yerr=[s.sem for s in stats],
        width=0.6, color=NEUTRAL_HARMONY["sage"], alpha=0.70, edgecolor=INK,
        linewidth=1.2, capsize=0, zorder=2,
        error_kw={"ecolor": INK, "elinewidth": 1.5},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{noise:g}" for noise, _ in by_noise])
    _finish(fig, ax, legend=False, ylim=BAR_YLIM)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Parent dir of <family>_sampling/noise=*/ cells.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for the figures; three PNG+SVG per mode "
                        "(similarity_vs_noise, similarity_swarm, "
                        "similarity_bar_swarm). Defaults to "
                        "<root>/analysis/llmasjudge/.")
    parser.add_argument("--max-noise", type=float, default=None,
                        help="Drop noise levels above this value (e.g. 0.75 to "
                        "exclude the fully-random epsilon=1.0 cells).")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    scores = filter_max_noise(
        collect_scores(root, families=list(FAMILIES)), args.max_noise
    )
    if not scores:
        parser.error(f"no judge_similarity_*.csv found under {root}")

    out_dir = args.out_dir or (root / "analysis" / "llmasjudge")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODE_FILES:
        if not any(m == mode for (_f, _n, m) in scores):
            continue
        for builder, stem in (
            (make_figure, "similarity_vs_noise"),
            (make_swarm_figure, "similarity_swarm"),
            (make_bar_swarm_figure, "similarity_bar_swarm"),
        ):
            fig = builder(scores, mode)
            paths = save_figure(fig, out_dir / f"{stem}_{mode}")
            for p in paths:
                print(f"wrote {p}")


if __name__ == "__main__":
    main()
