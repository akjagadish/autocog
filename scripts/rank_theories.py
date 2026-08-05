"""Rank base / surfaced / extra theories from an autopi human-eval run.

`eval_human.py` writes, per experiment, an `eval_hilbig_per_stimulus.csv`
holding each theory's predicted P(B) alongside the human pooled P(B) for
every stimulus. This script collects all of those across the run's
experiments and produces, **per metric separately** (no single aggregate
score — by request, since which metric is "most meaningful" is itself the
open question):

  - a theory x experiment table of the metric, and
  - a theory x experiment table of within-experiment ranks (1 = best),

for MAE, RMSE, Pearson r, Spearman r, and binomial log-likelihood (total
and per-trial). Every metric is recomputed from the per-stimulus P(B)
vectors so there is a single source of truth (the recomputed MAE / MSE /
Pearson match `eval_hilbig_summary.csv` to floating point).

It also produces the one cross-experiment view that *is* wanted: each
theory's predicted P(B) pooled against human pooled P(B) over **every
stimulus in every experiment at once** — a single correlation and scatter
panel per theory, summarising how well the theory tracks humans across the
whole stimulus space the run probed.

Outputs (under `--out`, default `<eval-dir>/ranking/`):

  metric_<name>_values.csv   — theory x experiment, the metric
  metric_<name>_ranks.csv    — theory x experiment, within-experiment rank
  pooled_correlation.csv     — per-theory Pearson/Spearman over all stimuli
  pooled_scatter.png         — per-theory pooled human-vs-model P(B) scatter

Usage:
    python scripts/rank_theories.py --eval-dir <run>/analysis/eval_hilbig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Metric name -> whether a larger value means a better fit. Drives the
# direction of `rank_pivot` and is the single place ranking direction is
# defined, so adding a metric is a one-line change.
METRIC_DIRECTION: dict[str, bool] = {
    "mae": False,
    "mse": False,
    "rmse": False,
    "pearson_r": True,
    "spearman_r": True,
    "loglik_total": True,
    "loglik_per_trial": True,
}

# Clip model P(B) away from {0, 1} before taking logs so a single
# confident-and-wrong prediction can't send the whole log-likelihood to
# -inf. 1e-6 is well below any resolution the human proportions carry.
_LL_EPS = 1e-6


# ---------------------------------------------------------------------------
# Load.
# ---------------------------------------------------------------------------


def load_all_per_stimulus(eval_dir: Path) -> pd.DataFrame:
    """Concatenate every `round_*/eval_hilbig_per_stimulus.csv` under
    `eval_dir` into one long DataFrame, tagging each row with the
    experiment (the subdirectory name)."""
    subdirs = sorted(
        d for d in eval_dir.iterdir()
        if d.is_dir() and (d / "eval_hilbig_per_stimulus.csv").is_file()
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No */eval_hilbig_per_stimulus.csv under {eval_dir!s}"
        )
    frames: list[pd.DataFrame] = []
    for d in subdirs:
        df = pd.read_csv(d / "eval_hilbig_per_stimulus.csv")
        df.insert(0, "experiment", d.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-experiment metrics.
# ---------------------------------------------------------------------------


def _binomial_loglik(
    p_human: np.ndarray, p_model: np.ndarray, n_trials: np.ndarray,
) -> float:
    """Total binomial log-likelihood of the human choices under the model.

    Each stimulus contributed `n_trials` independent choices of which
    `round(p_human * n_trials)` chose B; the model assigns each a
    probability `p_model`. This weights stimuli by how much human data
    backs them and penalises confident-but-wrong predictions — neither of
    which MAE/MSE do."""
    pm = np.clip(p_model, _LL_EPS, 1.0 - _LL_EPS)
    k_b = np.round(p_human * n_trials)
    k_a = n_trials - k_b
    return float(np.sum(k_b * np.log(pm) + k_a * np.log(1.0 - pm)))


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    """Pearson or Spearman correlation, NaN when either side is constant
    (correlation is undefined, not zero)."""
    if x.std() == 0 or y.std() == 0 or len(x) < 2:
        return float("nan")
    if method == "spearman":
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics_per_experiment(per_stim: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, model, role) with MAE / MSE / RMSE /
    Pearson / Spearman / binomial log-likelihood, all derived from the
    per-stimulus human and model P(B) vectors."""
    rows: list[dict[str, Any]] = []
    grouped = per_stim.groupby(["experiment", "model", "role"], sort=True)
    for (exp, model, role), g in grouped:
        h = g["p_b_human"].to_numpy(dtype=float)
        m = g["p_b_model"].to_numpy(dtype=float)
        n = g["n_trials_human"].to_numpy(dtype=float)
        diff = m - h
        mse = float(np.mean(diff * diff))
        ll = _binomial_loglik(h, m, n)
        total_trials = float(np.sum(n))
        rows.append({
            "experiment": exp,
            "model": model,
            "role": role,
            "n_stimuli": int(len(g)),
            "mae": float(np.mean(np.abs(diff))),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "pearson_r": _safe_corr(m, h, method="pearson"),
            "spearman_r": _safe_corr(m, h, method="spearman"),
            "loglik_total": ll,
            "loglik_per_trial": ll / total_trials if total_trials else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pivot + rank.
# ---------------------------------------------------------------------------


def metric_pivot(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """theory (index) x experiment (columns) table of one metric."""
    if metric not in METRIC_DIRECTION:
        raise ValueError(f"Unknown metric {metric!r}.")
    return metrics.pivot(index="model", columns="experiment", values=metric)


def rank_pivot(pivot: pd.DataFrame, *, higher_is_better: bool) -> pd.DataFrame:
    """Within-experiment ranks (1 = best) of a metric pivot. Each column
    (experiment) is ranked independently; NaNs keep rank NaN."""
    return pivot.rank(
        axis=0, ascending=not higher_is_better, method="min", na_option="keep",
    ).astype("Int64")


# ---------------------------------------------------------------------------
# Pooled cross-experiment correlation.
# ---------------------------------------------------------------------------


def pooled_correlation(per_stim: pd.DataFrame) -> pd.DataFrame:
    """Per theory, correlate its predicted P(B) against human pooled P(B)
    over **every stimulus in every experiment at once**. This is a single
    correlation per theory across the whole probed stimulus space — not
    the mean of the per-experiment correlations."""
    rows: list[dict[str, Any]] = []
    for (model, role), g in per_stim.groupby(["model", "role"], sort=True):
        h = g["p_b_human"].to_numpy(dtype=float)
        m = g["p_b_model"].to_numpy(dtype=float)
        rows.append({
            "model": model,
            "role": role,
            "n_points": int(len(g)),
            "pearson_r": _safe_corr(m, h, method="pearson"),
            "spearman_r": _safe_corr(m, h, method="spearman"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------


def plot_pooled_scatter(
    per_stim: pd.DataFrame, pooled: pd.DataFrame, path: Path,
) -> None:
    """One panel per theory: human pooled P(B) (x) vs model P(B) (y) over
    all stimuli of all experiments, points coloured by experiment, with
    the pooled Pearson/Spearman r in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = pooled["model"].tolist()
    experiments = sorted(per_stim["experiment"].unique())
    cmap = matplotlib.colormaps["tab10"]
    exp_color = {e: cmap(i % 10) for i, e in enumerate(experiments)}

    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False,
    )
    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = per_stim[per_stim["model"] == model]
        for e in experiments:
            s = sub[sub["experiment"] == e]
            if len(s):
                ax.scatter(
                    s["p_b_human"], s["p_b_model"],
                    color=exp_color[e], alpha=0.75, s=34,
                    edgecolor="black", linewidth=0.3, label=e,
                )
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Human pooled P(B)")
        ax.set_ylabel("Model P(B)")
        prow = pooled[pooled["model"] == model].iloc[0]
        ax.set_title(
            f"{model} [{prow['role']}]\n"
            f"pooled r={prow['pearson_r']:.3f}  "
            f"rho={prow['spearman_r']:.3f}  (n={int(prow['n_points'])})",
            fontsize=9,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(len(experiments), 5),
        fontsize=8, frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir", type=Path, required=True,
        help="Directory holding round_*/eval_hilbig_per_stimulus.csv.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory. Default: <eval-dir>/ranking/.",
    )
    args = p.parse_args(argv)

    out = args.out or (args.eval_dir / "ranking")
    out.mkdir(parents=True, exist_ok=True)

    per_stim = load_all_per_stimulus(args.eval_dir)
    experiments = sorted(per_stim["experiment"].unique())
    models = sorted(per_stim["model"].unique())
    print(
        f"[rank_theories] {len(experiments)} experiments, "
        f"{len(models)} theories: {models}"
    )

    metrics = compute_metrics_per_experiment(per_stim)
    for metric, higher in METRIC_DIRECTION.items():
        values = metric_pivot(metrics, metric)
        ranks = rank_pivot(values, higher_is_better=higher)
        values.to_csv(out / f"metric_{metric}_values.csv")
        ranks.to_csv(out / f"metric_{metric}_ranks.csv")

    pooled = pooled_correlation(per_stim)
    pooled.to_csv(out / "pooled_correlation.csv", index=False)

    try:
        plot_pooled_scatter(per_stim, pooled, out / "pooled_scatter.png")
        plot_msg = f"and pooled_scatter.png under {out}/"
    except ImportError as e:
        print(
            f"[rank_theories] WARNING: skipping scatter ({e}). "
            f"Install matplotlib to enable.",
            file=sys.stderr,
        )
        plot_msg = "(scatter skipped — matplotlib missing)"

    print("\n[rank_theories] pooled model-vs-human P(B) across all experiments:")
    print(pooled.to_string(index=False))
    print(f"\n[rank_theories] wrote per-metric value/rank tables {plot_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
