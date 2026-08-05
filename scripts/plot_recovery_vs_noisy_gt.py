"""Recovery against the NOISY ground truth, canonical sampling families.

For each canonical sampling family (TTB / WADD / Tallying) and action-noise
level ε ∈ {0, 0.5, 0.75, 1.0}, score seed / surfaced / best-surfaced choice
proportions against the gt theory sampled AT THAT ε (the noisy reference). The
ground truth is drawn as a dashed reference line per noise level (its own
ceiling: gt@ε vs gt@ε, flat at the sampling floor):

  bars (vs noisy gt@ε):
    seed                       - round-0 starting (competitor) theories.
    surfaced                   - theories autocog discovered.
    surfaced (best per run)    - each run-dir's best surfaced theory, averaged.
  reference lines (one segment per ε):
    gt (dashed)                - the noisy gt@ε vs an independent noisy draw of
                                 itself: the best ANY model can do against the
                                 noisy data (the floor). Flat at the sampling
                                 floor across ε; nothing beats it.
    gt_clean (dotted)          - MSE between the CLEAN gt and its noisy version
                                 (gt@0 vs gt@ε). A baseline, NOT a ceiling: it
                                 RISES with ε (the noisy version contracts toward
                                 0.5) and is beatable by noise-fit theories.

Two-step workflow: build the long-format table ONCE (the expensive simulation),
then re-plot from it with `--from-csv` (no re-simulation). The MSE figure is the
primary one; a correlation figure is also written, but its ε=1 column drops out
(at ε=1 the reference is ≈0.5 everywhere, a constant vector, so Pearson r is
undefined and `summarise` removes it).

Outputs (under --out-dir, default <results-root>/recovery_vs_noisy_gt):
  recovery_vs_noisy_gt_mse.{svg,png}
  recovery_vs_noisy_gt_correlation.{svg,png}
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

# Bars (left to right) and two reference lines: the flat gt floor (best
# achievable: noisy gt@ε vs itself, dashed) and the rising gt-vs-noisy-gt
# baseline (clean gt@0 vs noisy gt@ε, dotted).
BAR_ROLES: tuple[str, ...] = (
    "seed", "surfaced", "surfaced (best per run)",
)
BASELINE_ROLES: tuple[str, ...] = ("gt", "gt_clean")
YLABEL_MSE: str = r"$\mathrm{MSE}_{\hat{p(B)}}$ to noisy gt (lower = better)"
YLABEL_CORR: str = r"$r_{\hat{p(B)}}$ to noisy gt"


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
            "Recovery against the NOISY ground truth (gt@ε) for the canonical "
            "sampling families: seed / surfaced / best-surfaced / gt bars plus "
            "a noiseless-gt reference line, across action-noise levels."
        )
    )
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_SAMPLING)
    p.add_argument("--families", nargs="+",
                   default=list(CANONICAL_SAMPLING_FAMILIES))
    p.add_argument(
        "--noises", nargs="*", type=float, default=None,
        help="Noise levels to include (e.g. 0.0 0.5 0.75 1.0). Empty = all.",
    )
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <results-root>/recovery_vs_noisy_gt")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional path to also write the long-format table")
    p.add_argument(
        "--from-csv", type=Path, default=None,
        help="Load the long-format table from this CSV instead of rebuilding "
             "from the run-dirs — a fast re-plot with no re-simulation.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.results_root / "recovery_vs_noisy_gt")

    if args.from_csv is not None:
        long_df = pd.read_csv(args.from_csv)
        print(f"[vs-noisy-gt] loaded long table {args.from_csv} "
              f"({len(long_df)} rows, no re-simulation)")
    else:
        noises = args.noises if args.noises else None
        pairs = unique_stimulus_pairs(args.human_data)
        print(f"[vs-noisy-gt] stimuli={len(pairs)} families={args.families} "
              f"noises={noises} n_draws={args.n_draws} seed={args.seed}")
        long_df = build_results_long(
            results_root=args.results_root, families=args.families,
            noises=noises, stimulus_pairs=pairs,
            n_draws=args.n_draws, base_seed=args.seed,
        )
    if long_df.empty:
        print("[vs-noisy-gt] no results — check --results-root / --families.",
              file=sys.stderr)
        return 1
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.csv, index=False)
        print(f"[vs-noisy-gt] wrote long table {args.csv}")

    # MSE (primary): lower is better -> best surfaced = min mse.
    mse_summary = summarise(
        _with_best(long_df, metric="mse", higher_is_better=False), "mse")
    plot_grid(
        mse_summary, out_dir / "recovery_vs_noisy_gt_mse.png",
        roles=BAR_ROLES, baseline_roles=BASELINE_ROLES,
        ylabel=YLABEL_MSE, show_title=True, title_formatter=heuristic_title,
        title="Recovery vs noisy ground truth (MSE)",
    )
    # Correlation: higher is better -> best surfaced = max r. The ε=1 column
    # drops out (constant reference -> NaN Pearson r), handled by summarise.
    corr_summary = summarise(
        _with_best(long_df, metric="pearson_r", higher_is_better=True),
        "pearson_r")
    plot_grid(
        corr_summary, out_dir / "recovery_vs_noisy_gt_correlation.png",
        roles=BAR_ROLES, baseline_roles=BASELINE_ROLES,
        ylabel=YLABEL_CORR, show_title=True, title_formatter=heuristic_title,
        title="Recovery vs noisy ground truth (Pearson r)",
    )
    print(f"[vs-noisy-gt] wrote MSE + correlation figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
