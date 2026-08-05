"""Re-draw the eval_hilbig figures from an existing analysis/eval_hilbig
directory, colour-matched to the run's convergence & trajectory palette.

`eval_hilbig.py` already writes `eval_hilbig_summary.csv` and
`eval_hilbig_per_stimulus.csv` — and those CSVs are the source of truth. This
script re-plots from them *without* re-running the (seeded, deterministic)
model evaluation, so figures can be restyled without recomputing predictions
or needing the original pooled human-data file.

Figures written (each as SVG + PNG):

  eval_hilbig_scatter.png        — multi-panel human-vs-model P(B) scatter grid
  eval_hilbig_scatter_<model>.png — one standalone scatter per model
  eval_hilbig_bars_<metric>.png  — bar chart per metric (mae, mse, pearson_r)

Colours follow the lineage role (seeds -> slate / indigo, winning survivor ->
sandy / gold, other survivor -> terracotta, transient -> gray), identical to
the convergence and human-trajectory plots.

Usage:
  python scripts/plot_eval_hilbig_figures.py \
      --run-dir results/.../ttb+tallying \
      [--eval-dir results/.../analysis/eval_hilbig] \
      [--target p_b_pooled]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_hilbig import (  # noqa: E402
    build_name_map,
    plot_bars,
    plot_scatter,
    plot_scatter_per_model,
    role_color_map,
    _lineage_color_map,
)

# Per-metric bar charts. MAE/MSE are lower-is-better; Pearson r is
# higher-is-better, so sort it descending so the best model leads.
_BAR_METRICS: tuple[tuple[str, bool], ...] = (
    ("mae", True), ("mse", True), ("pearson_r", False),
)


def replot(
    eval_dir: Path, run_dir: Path, *, target: str = "p_b_pooled",
) -> list[Path]:
    """Read the eval CSVs under `eval_dir` and re-draw every figure with the
    lineage-role palette. `run_dir` supplies the lineage colours and readable
    theory names (both degrade gracefully if it can't be parsed). Returns the
    list of figure base paths written."""
    summary_df = pd.read_csv(eval_dir / "eval_hilbig_summary.csv")
    per_stim_df = pd.read_csv(eval_dir / "eval_hilbig_per_stimulus.csv")

    models = list(summary_df["model"])
    color_map = role_color_map(models, _lineage_color_map(run_dir))
    name_map = build_name_map(run_dir)

    written: list[Path] = []
    grid = eval_dir / "eval_hilbig_scatter.png"
    plot_scatter(
        per_stim_df, grid,
        title=f"Model vs human P(B) — run: {run_dir.name} — target={target}",
        color_map=color_map, name_map=name_map, show_title=False,
    )
    written.append(grid)
    written.extend(plot_scatter_per_model(
        per_stim_df, eval_dir, color_map=color_map, name_map=name_map,
        show_title=False,
    ))
    for metric, ascending in _BAR_METRICS:
        bars = eval_dir / f"eval_hilbig_bars_{metric}.png"
        plot_bars(
            summary_df, bars, metric=metric, ascending=ascending,
            title=f"{metric.upper()} vs humans ({target}) — run: {run_dir.name}",
            color_map=color_map, name_map=name_map,
        )
        written.append(bars)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--eval-dir", type=Path, default=None,
        help="Default: <run-dir>/analysis/eval_hilbig/.",
    )
    p.add_argument("--target", default="p_b_pooled")
    args = p.parse_args(argv)

    eval_dir = args.eval_dir or (args.run_dir / "analysis" / "eval_hilbig")
    if not (eval_dir / "eval_hilbig_summary.csv").is_file():
        raise SystemExit(
            f"[plot_eval_hilbig_figures] no eval_hilbig_summary.csv under "
            f"{eval_dir} — run scripts/eval_hilbig.py first."
        )
    written = replot(eval_dir, args.run_dir, target=args.target)
    print(
        f"[plot_eval_hilbig_figures] wrote {len(written)} figures "
        f"(SVG + PNG) under {eval_dir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
