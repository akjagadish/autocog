"""Measure metric stability under source models across cohort sizes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure the repo root is on sys.path so `src.*` imports work when the
# script is invoked directly (e.g. `python scripts/metric_stability.py`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.category_learning.experiment import CategoryLearningExperiment
from src.experiment import Experiment
from src.metric import Metric
from src.observation import Observations
from src.theory import Theory


SOURCE_MODELS = ("gcm", "sustain", "rulex")


def load_source_model(name: str) -> Theory:
    """Load a canonical source model YAML as a `Theory`.

    `name` must be one of `SOURCE_MODELS`. Resolves to
    `theories/category_learning/<name>.yaml` relative to the repo root.
    """
    if name not in SOURCE_MODELS:
        raise ValueError(
            f"unknown model {name!r}; expected one of {SOURCE_MODELS}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "theories" / "category_learning" / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"source model YAML not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Theory(**raw)


def run_stability(
    *,
    experiment: Experiment,
    metric: Metric,
    theory: Theory,
    n_participants_list: list[int],
    n_sampling_runs: int,
) -> pd.DataFrame:
    """Simulate `theory` on `experiment`, reduce via `metric`, repeat.

    For each n in n_participants_list, runs n_sampling_runs replicates.
    Each replicate simulates a cohort of n participants (fresh parameter
    draw each), scores pooled data via metric, records the scalar.

    Metric evaluation failures record value=NaN and continue.

    Returns DataFrame with columns [n_participants, replicate, value].
    """
    rows: list[dict[str, Any]] = []
    for n in n_participants_list:
        for r in range(n_sampling_runs):
            df = experiment.simulate(theory, n_runs=n)
            try:
                value: float = float(metric(df.copy()))
            except Exception as e:
                print(
                    f"[metric_stability] metric eval failed at "
                    f"n={n} replicate={r} ({type(e).__name__}: {e}); value=NaN"
                )
                value = float("nan")
            rows.append({"n_participants": n, "replicate": r, "value": value})
    return pd.DataFrame(rows, columns=["n_participants", "replicate", "value"])


def load_observation(
    run_dir: Path,
    *,
    round_idx: int,
    slot: int,
) -> tuple[CategoryLearningExperiment, Metric]:
    """Load `(experiment, metric)` from an autopi run-dir.

    `slot` is 1-indexed (matches the autopi pi_1 / pi_2 convention).
    Raises `ValueError` with a readable message when round/slot are out of
    range.
    """
    if slot not in (1, 2):
        raise ValueError(f"slot must be 1 or 2, got {slot}")
    pool = Observations.load(
        run_dir / "observations",
        experiment_class=CategoryLearningExperiment,
    )
    if round_idx < 0 or round_idx >= len(pool):
        raise ValueError(
            f"round_idx={round_idx} out of range; pool has {len(pool)} rounds"
        )
    round_obj = pool.rounds[round_idx]
    if len(round_obj) < slot:
        raise ValueError(
            f"round {round_idx} has {len(round_obj)} observations; "
            f"slot={slot} not available"
        )
    obs = round_obj.observations[slot - 1]
    return obs.experiment, obs.metric


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-replicate values into per-cohort summary rows.

    Columns returned (in order): `n_participants, mean, std, sem, min, max,
    n_ok`. `n_ok` counts non-NaN values. `std` uses ddof=1 (sample std).
    `sem = std / sqrt(n_ok)` when `n_ok > 0` else NaN.
    """
    rows: list[dict[str, Any]] = []
    for n, grp in df.groupby("n_participants", sort=True):
        vals = grp["value"].to_numpy()
        ok_mask = ~np.isnan(vals)
        ok = vals[ok_mask]
        n_ok = int(ok_mask.sum())
        if n_ok == 0:
            mean = std = sem = vmin = vmax = float("nan")
        else:
            mean = float(np.mean(ok))
            std = float(np.std(ok, ddof=1)) if n_ok > 1 else 0.0
            sem = std / (n_ok ** 0.5) if n_ok > 0 else float("nan")
            vmin = float(np.min(ok))
            vmax = float(np.max(ok))
        rows.append({
            "n_participants": int(n),
            "mean": mean,
            "std": std,
            "sem": sem,
            "min": vmin,
            "max": vmax,
            "n_ok": n_ok,
        })
    return pd.DataFrame(
        rows,
        columns=["n_participants", "mean", "std", "sem", "min", "max", "n_ok"],
    )


def default_output_path(
    *,
    model: str,
    run_dir: Path,
    round_idx: int,
    slot: int,
) -> Path:
    """`results/stability/metric_stability__<model>__<run>__r<round>s<slot>.csv`."""
    stem = (
        f"metric_stability__{model}__{run_dir.name}"
        f"__r{round_idx}s{slot}.csv"
    )
    return Path("results") / "stability" / stem


def plot_stability(
    df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> None:
    """Two-panel figure: left = boxplot per N, right = std/sem vs N on log-log.

    The right panel draws a dashed reference slope of `-1/2` through the
    leftmost std point (expected scaling under i.i.d. replicates) when there
    are >= 2 cohort sizes; skipped for a single N.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(df["n_participants"].unique().tolist())
    groups = [df.loc[df["n_participants"] == n, "value"].dropna().to_numpy() for n in ns]

    summary = summarize(df)
    summary_sorted = summary.sort_values("n_participants")

    n_title_lines = title.count("\n") + 1
    fig_height = 4 + 0.25 * max(0, n_title_lines - 1)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, fig_height))

    ax_left.boxplot(groups, positions=range(len(ns)), widths=0.5)
    ax_left.set_xticks(range(len(ns)))
    ax_left.set_xticklabels([str(n) for n in ns])
    ax_left.set_xlabel("n_participants")
    ax_left.set_ylabel("metric value")
    ax_left.set_title("per-cohort distribution")

    ax_right.loglog(
        summary_sorted["n_participants"],
        summary_sorted["std"],
        marker="o",
        label="std",
    )
    ax_right.loglog(
        summary_sorted["n_participants"],
        summary_sorted["sem"],
        marker="s",
        linestyle="--",
        label="sem",
    )
    if len(ns) >= 2:
        n_ref = summary_sorted["n_participants"].iloc[0]
        std_ref = summary_sorted["std"].iloc[0]
        xs = summary_sorted["n_participants"].to_numpy()
        ys = std_ref * (xs / n_ref) ** (-0.5)
        ax_right.loglog(xs, ys, linestyle=":", color="grey", label="slope=-1/2")
    ax_right.set_xlabel("n_participants")
    ax_right.set_ylabel("variability")
    ax_right.set_title("scaling with N")
    ax_right.legend()

    fig.suptitle(title, fontsize=10)
    top_margin = max(0.78, 1.0 - 0.05 * n_title_lines)
    fig.tight_layout(rect=[0, 0, 1, top_margin])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _parse_ns(arg: str) -> list[int]:
    try:
        ns = [int(x.strip()) for x in arg.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--n-participants must be a comma-separated list of ints; got {arg!r}"
        ) from e
    if not ns or any(n <= 0 for n in ns):
        raise argparse.ArgumentTypeError(
            f"--n-participants must be positive integers; got {ns}"
        )
    return ns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how stable an autopi-proposed metric is under a source "
            "model as the participant cohort size varies."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int, dest="round_idx")
    parser.add_argument("--slot", required=True, type=int, choices=(1, 2))
    parser.add_argument("--model", required=True, choices=SOURCE_MODELS)
    parser.add_argument(
        "--n-participants",
        required=True,
        type=_parse_ns,
        dest="n_participants_list",
        help="Comma-separated cohort sizes, e.g. 10,25,50,100",
    )
    parser.add_argument("--n-sampling-runs", required=True, type=int)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    experiment, metric = load_observation(
        args.run_dir, round_idx=args.round_idx, slot=args.slot
    )
    theory = load_source_model(args.model)

    raw = run_stability(
        experiment=experiment,
        metric=metric,
        theory=theory,
        n_participants_list=args.n_participants_list,
        n_sampling_runs=args.n_sampling_runs,
    )
    raw.insert(0, "model", args.model)
    raw.insert(1, "round", args.round_idx)
    raw.insert(2, "slot", args.slot)

    summary = summarize(raw.drop(columns=["model", "round", "slot"]))

    csv_path = args.output or default_output_path(
        model=args.model,
        run_dir=args.run_dir,
        round_idx=args.round_idx,
        slot=args.slot,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(csv_path, index=False)

    png_path = csv_path.with_suffix(".png")
    header = (
        f"{args.model} / {args.run_dir.name} / "
        f"round {args.round_idx} / slot {args.slot}"
    )
    rationale = (metric.rationale or "").strip()
    if rationale:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(rationale, width=110, max_lines=3, placeholder="..."))
        title = f"{header}\nmetric: {wrapped}"
    else:
        title = header
    plot_stability(
        raw.drop(columns=["model", "round", "slot"]),
        output_path=png_path,
        title=title,
    )

    print(f"[metric_stability] wrote {csv_path}")
    print(f"[metric_stability] wrote {png_path}")
    print("[metric_stability] summary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
