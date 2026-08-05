"""Plot aggregated LLM-judge results.

Reads <aggregated-csv> (output of scripts/aggregate_runs.sh) and writes:

  - gt_theory_recovery.png       (per-run rate that any slot matches GT theory)
  - family_vs_theory_match.png   (family vs theory match — same algorithm vs same family)
  - novelty_vs_seeds.png         (theory and algorithmic novelty vs the two seed theories)
  - novelty_vs_domain.png        (novelty vs the broader domain literature)

GT-dependent plots (the first two) are skipped when GT columns are blank
(real-data runs without --ground-truth).

Each plot uses the same axes convention: x = noise level, hue = ground-truth
heuristic. Ground truth is parsed from `run_dir` (looking for tokens
'tallying', 'ttb', 'wadd', 'ew') and noise level from `run_group`
(matching 'noise=<x>'). Rows that don't yield a GT or a noise label are
bucketed under "unknown".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figure_style import CYCLE, FONTSIZE, save_figure, style_axes


_GT_TOKENS = ("tallying", "ttb", "wadd", "ew", "gcm", "rulex", "sustain")
_NOISE_RE = re.compile(r"noise=([\d.]+)")
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]")
_MODEL_RE = re.compile(r"noise=[\d.]+_(.+?)_run\d+$")


def extract_ground_truth(run_dir: str) -> str | None:
    """Pick a ground-truth heuristic name out of a run directory name."""
    for tok in _TOKEN_SPLIT.split(run_dir):
        if tok.lower() in _GT_TOKENS:
            return tok.lower()
    return None


def parse_noise(run_group: str) -> str | None:
    """Pull the noise level out of a run_group like 'noise=0.3'."""
    m = _NOISE_RE.match(run_group)
    return m.group(1) if m else None


def extract_model(run_dir: str) -> str | None:
    """Pull the model name out of a run dir like '..._noise=0.0_<model>_run1'."""
    m = _MODEL_RE.search(run_dir)
    return m.group(1) if m else None


def per_run_match_rates(
    rows: list[dict[str, str]],
    col: str,
    *,
    key_fn: Callable[[dict[str, str]], str],
) -> dict[str, float]:
    """Per-cell fraction of *runs* where any slot has `col == "True"`.

    A run is counted as a match if either of its surfaced theories matches.
    Runs whose `col` cells are all blank are excluded from the denominator.
    """
    by_run: dict[tuple[str, str], bool] = {}
    for r in rows:
        v = r.get(col, "")
        if v == "":
            continue
        run_key = (key_fn(r), r["run"])
        by_run[run_key] = by_run.get(run_key, False) or (v == "True")

    buckets: dict[str, dict[str, int]] = {}
    for (cell_key, _run), matched in by_run.items():
        b = buckets.setdefault(cell_key, {"matches": 0, "total": 0})
        b["total"] += 1
        if matched:
            b["matches"] += 1

    return {k: v["matches"] / v["total"] for k, v in buckets.items() if v["total"]}


def _has_gt_data(rows: list[dict[str, str]]) -> bool:
    return any(r.get("gt_family_match") or r.get("gt_algorithm_match") for r in rows)


def _floats(rows: list[dict[str, str]], col: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        v = r.get(col, "")
        if v == "":
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def _annotate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Tag each row with parsed `_gt` and `_noise` fields."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["_gt"] = extract_ground_truth(r.get("run_dir", "")) or "unknown"
        r2["_noise"] = parse_noise(r.get("run_group", "")) or "?"
        r2["_model"] = extract_model(r.get("run_dir", "")) or "unknown"
        out.append(r2)
    return out


def _sorted_noise_levels(rows: list[dict[str, str]]) -> list[str]:
    levels = sorted({r["_noise"] for r in rows}, key=lambda s: float(s) if s != "?" else float("inf"))
    return levels


def _sorted_gts(rows: list[dict[str, str]]) -> list[str]:
    return sorted({r["_gt"] for r in rows})


def _bar_positions(n_cells: int, n_groups: int, bar_width: float) -> list[list[float]]:
    """Compute x-offsets for grouped bars across `n_cells` x positions."""
    return [
        [xi + (j - (n_groups - 1) / 2) * bar_width for xi in range(n_cells)]
        for j in range(n_groups)
    ]


def _grouped_bar(
    ax,
    *,
    rows: list[dict[str, str]],
    metric_fn: Callable[[list[dict[str, str]]], tuple[float, float]],
    title: str,
    ylabel: str,
    ymax: float = 1.05,
) -> None:
    """Draw x = noise, hue = GT bars on the given axes.

    `metric_fn` takes the list of rows for one (gt, noise) cell and returns
    a (mean, stderr-like-spread) pair. stderr is shown as an error bar; if
    the metric is rate-based, return 0.0.
    """
    noises = _sorted_noise_levels(rows)
    gts = _sorted_gts(rows)
    width = 0.8 / max(len(gts), 1)

    cmap = _cmap_for(len(gts))
    offsets = _bar_positions(len(noises), len(gts), width)

    for j, gt in enumerate(gts):
        means: list[float] = []
        spreads: list[float] = []
        for noise in noises:
            cell_rows = [r for r in rows if r["_gt"] == gt and r["_noise"] == noise]
            m, s = metric_fn(cell_rows) if cell_rows else (0.0, 0.0)
            means.append(m)
            spreads.append(s)
        ax.bar(offsets[j], means, width=width, yerr=spreads,
               color=cmap(j), label=gt)

    ax.set_xticks(list(range(len(noises))))
    ax.set_xticklabels([f"{n}" for n in noises])
    ax.set_ylim(0, ymax)
    style_axes(ax, xlabel="noise level", ylabel=ylabel, title=title)
    ax.legend(title="ground truth", frameon=False, fontsize=FONTSIZE - 6,
              title_fontsize=FONTSIZE - 6, loc="best")


def _cmap_for(n: int):
    return lambda i: CYCLE[i % len(CYCLE)]


def _plot_gt_theory_recovery(rows: list[dict[str, str]], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = per_run_match_rates(
        rows, "gt_algorithm_match",
        key_fn=lambda r: f"{r['_gt']}|{r['_noise']}",
    )

    def metric(cell_rows: list[dict[str, str]]) -> tuple[float, float]:
        if not cell_rows:
            return 0.0, 0.0
        key = f"{cell_rows[0]['_gt']}|{cell_rows[0]['_noise']}"
        return rates.get(key, 0.0), 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(
        ax,
        rows=rows,
        metric_fn=metric,
        title="GT theory recovery (per run)",
        ylabel="runs with any\nmatching slot",
    )
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_family_vs_theory_match(rows: list[dict[str, str]], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam_rates = per_run_match_rates(
        rows, "gt_family_match", key_fn=lambda r: f"{r['_gt']}|{r['_noise']}",
    )
    thy_rates = per_run_match_rates(
        rows, "gt_algorithm_match", key_fn=lambda r: f"{r['_gt']}|{r['_noise']}",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    def make_metric(rates: dict[str, float]):
        def metric(cell_rows: list[dict[str, str]]) -> tuple[float, float]:
            if not cell_rows:
                return 0.0, 0.0
            key = f"{cell_rows[0]['_gt']}|{cell_rows[0]['_noise']}"
            return rates.get(key, 0.0), 0.0
        return metric

    _grouped_bar(
        axes[0], rows=rows, metric_fn=make_metric(fam_rates),
        title="Same family (looser)",
        ylabel="per-run match rate",
    )
    _grouped_bar(
        axes[1], rows=rows, metric_fn=make_metric(thy_rates),
        title="Same algorithm (tighter)",
        ylabel="per-run match rate",
    )
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _pooled_run_matches(
    rows: list[dict[str, str]], col: str, noise: str,
) -> list[float]:
    """For a given noise level, return one 0/1 per run pooled across models.

    Each `run_dir` (a single seed/model/GT combination) contributes one
    data point: 1 if any slot has `col == "True"`, else 0. Models and GTs
    are pooled.
    """
    by_run: dict[str, bool] = {}
    for r in rows:
        if r["_noise"] != noise:
            continue
        v = r.get(col, "")
        if v == "":
            continue
        by_run[r["run_dir"]] = by_run.get(r["run_dir"], False) or (v == "True")
    return [1.0 if matched else 0.0 for matched in by_run.values()]


def _plot_pooled_recovery_vs_noise(rows: list[dict[str, str]], out: Path) -> Path:
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noises = _sorted_noise_levels(rows)
    n_models = len({r["_model"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, col, title in [
        (axes[0], "gt_family_match", "Family recovery (looser)"),
        (axes[1], "gt_algorithm_match", "Algorithmic recovery (tighter)"),
    ]:
        means: list[float] = []
        sems: list[float] = []
        for noise in noises:
            vals = _pooled_run_matches(rows, col, noise)
            if not vals:
                means.append(0.0)
                sems.append(0.0)
                continue
            means.append(mean(vals))
            sems.append(pstdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        ax.bar(range(len(noises)), means, width=0.5, yerr=sems,
               color=CYCLE[0])
        ax.set_xticks(range(len(noises)))
        ax.set_xticklabels([f"{n}" for n in noises])
        ax.set_ylim(0, 1.05)
        style_axes(ax, xlabel="noise level",
                   ylabel="per-run match rate", title=title)

    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_novelty_vs_seeds(rows: list[dict[str, str]], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_metric(col: str):
        def metric(cell_rows: list[dict[str, str]]) -> tuple[float, float]:
            vals = _floats(cell_rows, col)
            if not vals:
                return 0.0, 0.0
            return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0
        return metric

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    _grouped_bar(
        axes[0], rows=rows, metric_fn=make_metric("theory_novelty_vs_seeds"),
        title="Theory description novelty",
        ylabel="novelty vs seeds",
    )
    _grouped_bar(
        axes[1], rows=rows, metric_fn=make_metric("algo_novelty_vs_seeds"),
        title="Algorithm (predict source) novelty",
        ylabel="novelty vs seeds",
    )
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def _plot_novelty_vs_domain(rows: list[dict[str, str]], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def metric(cell_rows: list[dict[str, str]]) -> tuple[float, float]:
        vals = _floats(cell_rows, "novelty_vs_domain_literature")
        if not vals:
            return 0.0, 0.0
        return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(
        ax, rows=rows, metric_fn=metric,
        title="Novelty vs domain literature",
        ylabel="novelty",
    )
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_aggregated(csv_path: Path, out_dir: Path) -> list[Path]:
    raw = list(csv.DictReader(csv_path.open()))
    rows = _annotate(raw)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if _has_gt_data(rows):
        paths.append(_plot_gt_theory_recovery(rows, out_dir / "gt_theory_recovery.png"))
        paths.append(_plot_family_vs_theory_match(rows, out_dir / "family_vs_theory_match.png"))
        paths.append(_plot_pooled_recovery_vs_noise(rows, out_dir / "pooled_recovery_vs_noise.png"))
    paths.append(_plot_novelty_vs_seeds(rows, out_dir / "novelty_vs_seeds.png"))
    paths.append(_plot_novelty_vs_domain(rows, out_dir / "novelty_vs_domain.png"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregated_csv", type=Path,
                        help="Path to all_judge_results.csv from aggregate_runs.sh")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for PNGs "
                        "(default: <aggregated_csv>.parent/plots/)")
    args = parser.parse_args()

    out_dir = args.out or args.aggregated_csv.parent / "plots"
    paths = plot_aggregated(args.aggregated_csv, out_dir)
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
