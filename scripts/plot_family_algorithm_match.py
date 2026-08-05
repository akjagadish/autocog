"""Family vs. algorithm recovery of surfaced theories, from LLM-judge verdicts.

Reads the per-cell `judge_results_majority5.csv` files (output of
`judge_runs.py --n-votes 5 --gt-only`) under a results root laid out as
`<root>/<family>/noise=<eps>/judge_results_majority5.csv`, and measures, per
ground-truth family and action-noise level, two judgements on each surfaced
theory:

  gt_family_match     - same family as the ground truth, IRRESPECTIVE of the
                        algorithmic instantiation.
  gt_algorithm_match  - same family AND the same algorithmic instantiation.

Two counting units are reported, and are NEVER mixed in one figure:

  per theory - each surfaced theory is one data point; rate = fraction of
               surfaced theories judged True.
  per run    - each run is one data point; a run counts as a match if ANY of
               its surfaced slots matched; rate = fraction of matching runs.

Outputs (under --out-dir, default the results root). By default only the
`per_run` figures are written (a run matches if AT LEAST ONE surfaced slot
matched, not both); pass `--units per_theory per_run` to also emit the
per-theory figures:
  family_algorithm_match.csv                      - long format, one row per
                                                    (family, noise, unit, metric)
  family_algorithm_match_pooled_per_run.png       - Fig 1, per run  (default)
  family_algorithm_match_per_model_per_run.png    - Fig 2, per run  (default)
  family_algorithm_match_pooled_per_theory.png    - Fig 1, per theory (--units)
  family_algorithm_match_per_model_per_theory.png - Fig 2, per theory (--units)

Fig 1 pools across families and runs: x = noise, bars = {family, algorithm}.
Fig 2 is a 1x3 grid (one column per family, recovery-plot style): x = noise,
bars = {family, algorithm}. Error bars are SEM across the unit of replication
(run_dir), computed by first reducing each run_dir to a scalar (mean over its
theories for the per-theory unit, ANY over its theories for the per-run unit)
and then taking mean +/- SEM across run_dirs within the group.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import FONTSIZE, METRIC_COLOR, save_figure, style_axes

RESULTS_ROOT_DEFAULT = (
    _REPO_ROOT / "results" / "heuristic_decision_making"
    / "synthetic_corrected_theories"
)

FAMILIES_PLOT: tuple[str, ...] = (
    "ttb", "wadd", "tallying",
    "ttb_sampling", "wadd_sampling", "tallying_sampling",
    "alternating", "anti_majority",
    "perseveration", "take_the_worst", "single_cue", "cue_parity",
)
METRICS: tuple[str, ...] = ("gt_family_match", "gt_algorithm_match")
METRIC_LABEL = {
    "gt_family_match": "family match",
    "gt_algorithm_match": "algorithm match",
}
JUDGE_CSV_NAME = "judge_results_majority5.csv"


# --- pure aggregation helpers (unit-tested) ---------------------------------

def per_theory_rate(df: pd.DataFrame, metric: str) -> float:
    """Fraction of surfaced theories (rows) whose `metric` is True."""
    return float(df[metric].astype(bool).mean())


def per_run_rate(df: pd.DataFrame, metric: str) -> float:
    """Fraction of runs (unique `run_dir`) with at least one matching slot."""
    any_per_run = df.groupby("run_dir")[metric].apply(
        lambda s: bool(s.astype(bool).any())
    )
    return float(any_per_run.mean())


def _run_dir_scalars(df: pd.DataFrame, metric: str, unit: str) -> pd.Series:
    """Reduce each run_dir to one scalar: mean over its theories (per-theory
    unit) or ANY over its theories (per-run unit)."""
    grp = df.groupby("run_dir")[metric]
    if unit == "per_theory":
        return grp.apply(lambda s: float(s.astype(bool).mean()))
    if unit == "per_run":
        return grp.apply(lambda s: float(bool(s.astype(bool).any())))
    raise ValueError(f"unknown unit {unit!r}")


# --- loading ----------------------------------------------------------------

def load_judgements(
    results_root: Path, families: list[str], csv_name: str = JUDGE_CSV_NAME,
) -> pd.DataFrame:
    """Concatenate every `judge_results_majority5.csv` under the root into a
    long frame with parsed `family` and `noise` columns."""
    frames: list[pd.DataFrame] = []
    for family in families:
        fam_root = results_root / family
        if not fam_root.is_dir():
            print(f"[match] no dir for family {family!r}: {fam_root}",
                  file=sys.stderr)
            continue
        for noise_dir in sorted(fam_root.glob("noise=*")):
            csv = noise_dir / csv_name
            if not csv.is_file():
                print(f"[match] missing {csv}", file=sys.stderr)
                continue
            d = pd.read_csv(csv)
            d["family"] = family
            d["noise"] = float(noise_dir.name.removeprefix("noise="))
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for m in METRICS:
        out[m] = out[m].astype(bool)
    return out


# --- summary ----------------------------------------------------------------

def summarise(
    df: pd.DataFrame, *, unit: str, group_cols: list[str]
) -> pd.DataFrame:
    """Mean +/- SEM of each metric across run_dirs, after reducing each
    run_dir to a scalar per `unit`. Returns long format with one row per
    (group_cols..., metric)."""
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in METRICS:
            scalars = _run_dir_scalars(g, metric, unit).to_numpy()
            n = len(scalars)
            mean = float(scalars.mean()) if n else float("nan")
            sd = float(scalars.std(ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            row = dict(zip(group_cols, keys))
            row.update(unit=unit, metric=metric, mean=mean, sem=sem, n_runs=n)
            rows.append(row)
    return pd.DataFrame(rows)


# --- plotting ---------------------------------------------------------------

def _bar_panel(ax, sub: pd.DataFrame, noises: list[float]) -> None:
    x = np.arange(len(noises))
    # Plot only the metrics present in `sub` (in canonical METRICS order), so a
    # family-only summary yields a single bar group per noise.
    present = [m for m in METRICS if m in set(sub["metric"])]
    width = 0.8 / max(len(present), 1)
    for i, metric in enumerate(present):
        rrows = sub[sub.metric == metric].set_index("noise")
        means = np.array([rrows.loc[n, "mean"] if n in rrows.index else np.nan
                          for n in noises], dtype=float)
        sems = np.array([rrows.loc[n, "sem"] if n in rrows.index else 0.0
                         for n in noises], dtype=float)
        ax.bar(x - 0.4 + width * (i + 0.5), means, width, yerr=sems,
               color=METRIC_COLOR[metric], label=METRIC_LABEL[metric])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:g}" for n in noises])
    ax.set_ylim(0, 1.05)
    style_axes(ax, xlabel="ground-truth action noise (ε)")


def plot_pooled(summary: pd.DataFrame, out_path: Path, *, unit: str) -> None:
    """Fig 1: pooled across families and runs. Single panel, x = noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noises = sorted(summary.noise.unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_panel(ax, summary, noises)
    unit_lbl = "per theory" if unit == "per_theory" else "per run"
    style_axes(ax, ylabel=f"match rate ({unit_lbl})")
    ax.legend(frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), fontsize=FONTSIZE - 4)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_model(summary: pd.DataFrame, out_path: Path, *, unit: str) -> None:
    """Fig 2: 1x3 grid (one column per family), recovery-plot style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILIES_PLOT if f in set(summary["family"])]
    fig, axes = plt.subplots(
        1, len(families), figsize=(6 * len(families), 4),
        sharey=True, squeeze=False,
    )
    for col, fam in enumerate(families):
        ax = axes[0][col]
        sub = summary[summary.family == fam]
        noises = sorted(sub.noise.unique())
        _bar_panel(ax, sub, noises)
        ax.set_title(fam.upper(), fontsize=FONTSIZE)
    unit_lbl = "per surfaced theory" if unit == "per_theory" else "per run (any slot)"
    axes[0][0].set_ylabel(f"match rate ({unit_lbl})", fontsize=FONTSIZE)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METRICS),
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               fontsize=FONTSIZE - 2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_DEFAULT)
    p.add_argument("--families", nargs="+", default=list(FAMILIES_PLOT))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write CSV + PNGs (default: results root).")
    p.add_argument("--csv-name", default=JUDGE_CSV_NAME,
                   help=f"Per-cell judge CSV filename to read (default "
                   f"{JUDGE_CSV_NAME}). Use to plot an alternate judge run, "
                   "e.g. a flash-model rerun.")
    p.add_argument("--metrics", nargs="+", default=list(METRICS),
                   choices=list(METRICS),
                   help="Which judgement(s) to plot. Default both; pass "
                   "'gt_family_match' to show family resemblance only "
                   "(ignore algorithmic structure).")
    p.add_argument("--units", nargs="+", default=["per_run"],
                   choices=["per_theory", "per_run"],
                   help="Counting unit(s) to plot. Default 'per_run' = a run "
                   "matches if AT LEAST ONE surfaced slot matched (not both). "
                   "Pass 'per_theory per_run' for both figure sets.")
    args = p.parse_args(argv)

    out_dir = args.out_dir or args.results_root
    df = load_judgements(args.results_root, args.families, args.csv_name)
    if df.empty:
        print("[match] no judge CSVs found — check --results-root.",
              file=sys.stderr)
        return 1

    print(f"[match] {len(df)} surfaced-theory judgements across "
          f"{df.run_dir.nunique()} runs, families={sorted(df.family.unique())}, "
          f"noises={sorted(df.noise.unique())}")

    long_rows = []
    for unit in args.units:
        # Fig 1: pooled across families (group by noise only).
        pooled = summarise(df, unit=unit, group_cols=["noise"])
        pooled["family"] = "pooled"
        # Fig 2: per family.
        per_model = summarise(df, unit=unit, group_cols=["family", "noise"])
        # Restrict to the requested metric(s) (e.g. family-only).
        pooled = pooled[pooled["metric"].isin(args.metrics)]
        per_model = per_model[per_model["metric"].isin(args.metrics)]
        long_rows.append(pooled)
        long_rows.append(per_model)

        plot_pooled(
            pooled, out_dir / f"family_algorithm_match_pooled_{unit}.png",
            unit=unit)
        plot_per_model(
            per_model, out_dir / f"family_algorithm_match_per_model_{unit}.png",
            unit=unit)

    long_df = pd.concat(long_rows, ignore_index=True)
    csv_path = out_dir / "family_algorithm_match.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(csv_path, index=False)
    print(long_df.to_string(index=False))
    print(f"[match] wrote {csv_path} and {2 * len(args.units)} PNG(s) "
          f"(units={args.units}) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
