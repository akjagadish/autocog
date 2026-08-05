"""Recovery against the CLEAN ground truth, canonical sampling families.

The clean-reference complement of plot_recovery_vs_noisy_gt.py. For each
canonical sampling family (TTB / WADD / Tallying) and action-noise level
ε ∈ {0, 0.5, 0.75, 1.0}, score seed / surfaced / best-surfaced choice
proportions against the gt theory sampled at ε=0 (the CLEAN, true signal) — the
honest recovery metric, i.e. distance to the true generator rather than to the
noisy data autopi saw. Bars only (seed / surfaced / best-surfaced); no
reference line.

Unlike the noisy-reference figure, the correlation panel keeps ALL noise columns
(the clean reference is never a constant vector, so Pearson r is defined at
every ε). Use `--noises` to drop levels (e.g. the degenerate ε=1 column); it
filters both a fresh build and a `--from-csv` re-plot.

Two-step workflow: build the long table ONCE (the expensive simulation), then
re-plot from it with `--from-csv` (no re-simulation).

Outputs (under --out-dir, default <results-root>/recovery_vs_clean_gt):
  recovery_vs_clean_gt_mse.{svg,png}
  recovery_vs_clean_gt_correlation.{svg,png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_hilbig import HUMAN_DATA_DEFAULT  # noqa: E402
from scripts.recovery_correlation import (  # noqa: E402
    N_DRAWS_DEFAULT,
    build_results_long,
    plot_grid,
    summarise,
    surfaced_best_rows,
    unique_stimulus_pairs,
)
from scripts.plot_recovery_per_model import (  # noqa: E402
    CANONICAL_SAMPLING_FAMILIES,
    RESULTS_ROOT_SAMPLING,
    heuristic_title,
)

# Bars (left to right). No reference line: just the recovery roles scored
# against the clean gt@0.
BAR_ROLES: tuple[str, ...] = (
    "seed", "surfaced", "surfaced (best per run)",
)
YLABEL_MSE: str = r"$\mathrm{MSE}_{\hat{p(B)}}$ to clean gt (lower = better)"
YLABEL_CORR: str = r"$r_{\hat{p(B)}}$ to clean gt"


def select_noises(long_df: pd.DataFrame, noises) -> pd.DataFrame:
    """Restrict the long table to `noises` (None/empty -> keep all). Lets a
    `--from-csv` re-plot drop noise levels (e.g. the degenerate ε=1 column)
    without re-simulating."""
    if not noises:
        return long_df
    return long_df[long_df["noise"].isin(noises)].copy()


def _with_best(long_df: pd.DataFrame, *, metric: str, higher_is_better: bool):
    """Append the per-run best-surfaced rows (metric-specific) to `long_df`."""
    return pd.concat(
        [long_df,
         surfaced_best_rows(long_df, metric=metric,
                            higher_is_better=higher_is_better)],
        ignore_index=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Recovery against the CLEAN ground truth (gt@0) for the canonical "
            "sampling families: seed / surfaced / best-surfaced bars plus the "
            "ground-truth floor and the noisy-gt drift line, across noise."
        )
    )
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_SAMPLING)
    p.add_argument("--families", nargs="+",
                   default=list(CANONICAL_SAMPLING_FAMILIES))
    p.add_argument("--noises", nargs="*", type=float, default=None,
                   help="Noise levels to include. Empty = all found.")
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <results-root>/recovery_vs_clean_gt")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional path to also write the long-format table")
    p.add_argument(
        "--from-csv", type=Path, default=None,
        help="Load the long-format table from this CSV instead of rebuilding "
             "from the run-dirs — a fast re-plot with no re-simulation.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.results_root / "recovery_vs_clean_gt")
    noises = args.noises if args.noises else None

    if args.from_csv is not None:
        long_df = pd.read_csv(args.from_csv)
        print(f"[vs-clean-gt] loaded long table {args.from_csv} "
              f"({len(long_df)} rows, no re-simulation)")
    else:
        pairs = unique_stimulus_pairs(args.human_data)
        print(f"[vs-clean-gt] stimuli={len(pairs)} families={args.families} "
              f"noises={noises} n_draws={args.n_draws} seed={args.seed}")
        # reference_action_noise=0.0 -> score every column against clean gt@0.
        long_df = build_results_long(
            results_root=args.results_root, families=args.families,
            noises=noises, stimulus_pairs=pairs,
            n_draws=args.n_draws, base_seed=args.seed,
            reference_action_noise=0.0,
        )
    # Drop excluded noise levels (also filters a --from-csv re-plot).
    long_df = select_noises(long_df, noises)
    if long_df.empty:
        print("[vs-clean-gt] no results — check --results-root / --families.",
              file=sys.stderr)
        return 1
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.csv, index=False)
        print(f"[vs-clean-gt] wrote long table {args.csv}")

    mse_summary = summarise(
        _with_best(long_df, metric="mse", higher_is_better=False), "mse")
    plot_grid(
        mse_summary, out_dir / "recovery_vs_clean_gt_mse.png",
        roles=BAR_ROLES, ylabel=YLABEL_MSE,
        show_title=True, title_formatter=heuristic_title, share_y=False,
        title="Recovery vs clean ground truth (MSE)",
    )
    corr_summary = summarise(
        _with_best(long_df, metric="pearson_r", higher_is_better=True),
        "pearson_r")
    plot_grid(
        corr_summary, out_dir / "recovery_vs_clean_gt_correlation.png",
        roles=BAR_ROLES, ylabel=YLABEL_CORR,
        show_title=True, title_formatter=heuristic_title, share_y=False,
        title="Recovery vs clean ground truth (Pearson r)",
    )
    print(f"[vs-clean-gt] wrote MSE + correlation figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
