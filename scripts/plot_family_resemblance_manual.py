"""Family-resemblance plot from MANUAL (human) family-match scores.

These are hand-scored counts of "did at least one surfaced theory match the
ground-truth family, per run" out of N_RUNS=3 runs, for the three canonical
heuristics across the binary-sampling action-noise sweep. They are the human
ground truth used to cross-check (and correct) the LLM judge, which
systematically under-counts here (notably Tallying: Equal-Weight is identical
to Tallying on binary cues but the judge marked it a different family).

This script pools ACROSS the three models at each noise level and reports the
pooled family-resemblance rate (fraction of matching runs) with SEM across
models. Output: a single-panel rate-vs-noise figure + a CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import CYCLE, FONTSIZE, INK, save_figure, style_axes
RESULTS_ROOT_DEFAULT = (
    _REPO_ROOT / "results" / "heuristic_decision_making"
    / "synthetic_corrected_theories_binary_sampling"
)

N_RUNS: int = 3
# Noise levels (the last is the "detect random" condition, epsilon = 1.0).
NOISES: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)

# Manual per-(model, noise) counts of runs (out of N_RUNS) with >=1 family
# match. Source: user hand-scoring (2026-06-09).
MANUAL_SCORES: dict[str, tuple[float, ...]] = {
    #            noise:  0.0  0.5  0.75 1.0
    "Tallying": (3.0, 2.5, 1.5, 2.0),
    "TTB":      (3.0, 2.5, 3.0, 2.0),
    "WADD":     (3.0, 2.5, 2.0, 1.5),
}


# Canonical family -> display label (matches MANUAL_SCORES keys), so the manual
# and judge plots use the same model ordering/labels.
FAMILY_TO_LABEL: dict[str, str] = {
    "ttb_sampling": "TTB",
    "wadd_sampling": "WADD",
    "tallying_sampling": "Tallying",
}


def judge_scores(
    results_root: Path, csv_name: str, *,
    noises: tuple[float, ...] = NOISES,
    family_to_label: dict[str, str] = FAMILY_TO_LABEL,
) -> dict[str, tuple[float, ...]]:
    """Build the same (model -> per-noise counts) score table the manual plot
    uses, but FROM an LLM-judge run: for each (family, noise) cell, count the
    runs with AT LEAST ONE surfaced theory whose `gt_family_match` is True.
    Reads `<results_root>/<family>/noise=<nz>/<csv_name>`."""
    import pandas as pd

    out: dict[str, tuple[float, ...]] = {}
    for family, label in family_to_label.items():
        counts: list[float] = []
        for nz in noises:
            csv = results_root / family / f"noise={nz}" / csv_name
            df = pd.read_csv(csv)
            any_per_run = df.groupby("run_dir")["gt_family_match"].apply(
                lambda s: bool(s.astype(bool).any())
            )
            counts.append(float(int(any_per_run.sum())))
        out[label] = tuple(counts)
    return out


def pool_across_models(
    scores: dict[str, tuple[float, ...]], n_runs: int = N_RUNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool the per-model family-match counts at each noise level.

    Returns (mean_rate, sem_rate, per_model_rates) where rates are fractions in
    [0, 1] (count / n_runs). `mean_rate`/`sem_rate` are the mean and standard
    error ACROSS models at each noise level; `per_model_rates` is the
    (n_models, n_noise) matrix of individual rates for overlay."""
    rates = np.array([np.asarray(v, dtype=float) / n_runs
                      for v in scores.values()])  # (n_models, n_noise)
    mean = rates.mean(axis=0)
    n = rates.shape[0]
    sd = rates.std(axis=0, ddof=1) if n > 1 else np.zeros(rates.shape[1])
    sem = sd / np.sqrt(n) if n > 1 else np.zeros(rates.shape[1])
    return mean, sem, rates


def plot(out_path: Path, *, noises=NOISES, scores=MANUAL_SCORES,
         n_runs=N_RUNS, source_label: str = "manual scores") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, sem, rates = pool_across_models(scores, n_runs)
    x = np.arange(len(noises))

    fig, ax = plt.subplots(figsize=(6, 4))
    # Per-model rates as faint markers (shows the spread that the SEM summarises).
    for i, ((model, _), row) in enumerate(zip(scores.items(), rates)):
        ax.scatter(x, row, s=64, alpha=0.5, color=CYCLE[i % len(CYCLE)],
                   label=model, zorder=2)
    # Pooled mean +/- SEM across models.
    ax.errorbar(x, mean, yerr=sem, fmt="o-", color=INK,
                zorder=3, label="pooled (mean ± SEM)")

    ax.set_xticks(x)
    labels = [f"{nz:g}" for nz in noises]
    labels[-1] = f"{noises[-1]:g}\n(random)"
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    style_axes(
        ax,
        xlabel="ground-truth action noise (ε)",
        ylabel="family resemblance\n(runs with ≥1 match)",
    )
    ax.legend(frameon=False, fontsize=FONTSIZE - 6, loc="lower left")
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT_DEFAULT)
    p.add_argument("--source", choices=["manual", "judge"], default="manual",
                   help="'manual' = hardcoded human scores; 'judge' = build "
                   "scores from an LLM-judge run's per-cell CSVs.")
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT,
                   help="Root holding <family>/noise=<nz>/<judge-csv> (for "
                   "--source judge).")
    p.add_argument("--judge-csv", default="judge_results_majority5_v2.csv",
                   help="Per-cell judge CSV name (for --source judge).")
    args = p.parse_args(argv)

    if args.source == "judge":
        scores = judge_scores(args.results_root, args.judge_csv)
        source_label = f"LLM judge: {args.judge_csv}"
        tag = "judge"
    else:
        scores = MANUAL_SCORES
        source_label = "manual scores"
        tag = "manual"

    mean, sem, rates = pool_across_models(scores)
    # Long CSV: one row per noise (pooled) for downstream use.
    import csv as _csv
    csv_path = args.out_dir / f"family_resemblance_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["noise", "pooled_rate", "sem"]
                   + [f"rate_{m}" for m in scores])
        for j, nz in enumerate(NOISES):
            w.writerow([nz, round(float(mean[j]), 4), round(float(sem[j]), 4)]
                       + [round(float(rates[i, j]), 4)
                          for i in range(rates.shape[0])])

    png_path = args.out_dir / f"family_resemblance_{tag}_pooled.png"
    plot(png_path, scores=scores, source_label=source_label)

    print(f"[{tag}] noise  pooled_rate  sem")
    for j, nz in enumerate(NOISES):
        print(f"{nz:<6g} {mean[j]:.3f}        {sem[j]:.3f}")
    print(f"[resemblance] wrote {csv_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
