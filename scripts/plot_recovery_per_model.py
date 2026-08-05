"""Per-model ground-truth recovery figures for the Hilbig (2014) task.

Same recovery analysis as ``scripts/recovery_correlation.py`` (seed / surfaced
/ gt / random choice-proportion recovery), but instead of the combined 1xN
grid this emits ONE single-panel figure per ground-truth family — including
the non-canonical sampling families (alternating, anti_majority, cue_parity,
perseveration, single_cue, take_the_worst) alongside the canonical sampling
ones (ttb_sampling, wadd_sampling, tallying_sampling).

Each family's figure (and its two best-surfaced bars) is computed from that
family's rows ONLY, so families never borrow one another's best/worst values.
Canonical families that span several action-noise levels keep the noise
x-axis; families with a single noise level (ε=0) drop it — both handled by the
reused ``plot_grid``.

Outputs (under ``--out-dir``, default ``<results-root>/per_model``):
  recovery_mse_<family>.{svg,png}          - MSE recovery, one per family
  recovery_correlation_<family>.{svg,png}  - Pearson r recovery, one per family

The long-format table is rebuilt fresh from the current run-dirs (it is NOT
read from a possibly-stale CSV), matching the recovery pipeline's defaults.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recovery_correlation import (  # noqa: E402
    N_DRAWS_DEFAULT,
    build_random_family_rows,
    build_results_long,
    family_plot_order,
    plot_grid,
    summarise,
    surfaced_best_rows,
    surfaced_best_across_runs_rows,
    unique_stimulus_pairs,
)
from scripts.eval_hilbig import HUMAN_DATA_DEFAULT  # noqa: E402

# All nine families produced under the binary-sampling results tree: the three
# canonical sampling heuristics first, then the six non-canonical baselines.
ALL_FAMILIES_DEFAULT: tuple[str, ...] = (
    "ttb_sampling", "wadd_sampling", "tallying_sampling",
    "alternating", "anti_majority", "cue_parity",
    "perseveration", "single_cue", "take_the_worst",
)
# The three canonical sampling heuristics, pooled at full per-panel width.
CANONICAL_SAMPLING_FAMILIES: tuple[str, ...] = (
    "ttb_sampling", "wadd_sampling", "tallying_sampling",
)
# The non-canonical baselines, pooled into one compact combined figure, in the
# requested panel order (cue_parity is intentionally excluded from the pool).
# The synthetic `random` family is sourced from the canonical ε=1 runs (see
# build_random_family_rows) and scored against a 0.5 reference; it has no real
# run-dir, so it is NOT in ALL_FAMILIES_DEFAULT (only here, for pooling).
NONCANONICAL_FAMILIES: tuple[str, ...] = (
    "alternating", "perseveration", "random",
    "take_the_worst", "single_cue", "anti_majority",
)
# Global panel order across the pooled figures: canonical first, then the
# non-canonical families in the requested order. Passed to plot_grid so panels
# follow this order rather than first-seen.
FAMILY_PLOT_ORDER: tuple[str, ...] = (
    CANONICAL_SAMPLING_FAMILIES + NONCANONICAL_FAMILIES
)
# Total width (inches) of the pooled non-canonical figure: ~2 canonical panels
# (6in each) wide regardless of how many panels it holds, so it sits next to a
# two-panel canonical figure. Per-panel width = POOLED_FIG_WIDTH / n_panels.
POOLED_FIG_WIDTH: float = 12.0
RESULTS_ROOT_SAMPLING: Path = _REPO_ROOT / "results" / "recovery"

# Bar roles per metric: the raw recovery roles only — the two metric-specific
# "best surfaced" bars are intentionally dropped. On BOTH metrics only
# seed/surfaced are bars; `random` (floor) and `gt` (ceiling) are reference
# LINES. Correlation has no random line (a constant 0.5 vector has NaN corr).
ROLES_MSE: tuple[str, ...] = ("seed", "surfaced")
ROLES_CORR: tuple[str, ...] = ("seed", "surfaced")
# Bars for the "with best" MSE figure: adds the best-surfaced-per-run bar.
ROLES_MSE_WITH_BEST: tuple[str, ...] = (
    "seed", "surfaced", "surfaced (best per run)",
)
# MSE reference lines: floor (random) + ceiling (gt), plus a no-random variant
# (gt only). Correlation: gt ceiling only.
BASELINE_ROLES_MSE: tuple[str, ...] = ("random", "gt")
BASELINE_ROLES_MSE_NO_RANDOM: tuple[str, ...] = ("gt",)
BASELINE_ROLES_CORR: tuple[str, ...] = ("gt",)

# Compact LaTeX y-labels: r / MSE of the predicted choice proportion p̂(B).
YLABEL_MSE: str = r"$\mathrm{MSE}_{\hat{p(B)}}$ (lower = better)"
YLABEL_CORR: str = r"$r_{\hat{p(B)}}$"

# Panel titles: canonical heuristics keep their acronym/name (sampling suffix
# stripped); every other family is first-letter-capitalised, underscores → spaces.
_TITLE_OVERRIDES: dict[str, str] = {
    "ttb": "TTB", "ttb_sampling": "TTB",
    "wadd": "WADD", "wadd_sampling": "WADD",
    "tallying": "Tallying", "tallying_sampling": "Tallying",
}


def heuristic_title(family: str) -> str:
    """Human-readable panel title for a family. Canonical heuristics map to
    TTB / WADD / Tallying (sampling suffix dropped); any other family becomes
    its name with underscores as spaces and only the first letter capitalised
    (e.g. ``anti_majority`` → ``Anti majority``)."""
    if family in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[family]
    return family.replace("_", " ").capitalize()


def _without_gt(baseline_roles: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the gt reference line from a baseline_roles tuple."""
    return tuple(r for r in baseline_roles if r != "gt")


def baseline_roles_for_family(
    family: str, baseline_roles: tuple[str, ...],
) -> tuple[str, ...]:
    """Reference lines for ONE family: gt is kept only for the canonical
    sampling families; the non-canonical/sequential baselines drop the gt
    ceiling (it is not a meaningful per-pair reference for them). random is
    unaffected."""
    if family in CANONICAL_SAMPLING_FAMILIES:
        return baseline_roles
    return _without_gt(baseline_roles)


def _gt_line_filter(families: tuple[str, ...]):
    """A baseline_roles filter for a pool: identity when EVERY family is
    canonical (keep the gt ceiling line), else drop gt (non-canonical pools
    show no gt ceiling)."""
    keep_gt = all(f in CANONICAL_SAMPLING_FAMILIES for f in families)
    return (lambda br: br) if keep_gt else _without_gt


def pooled_specs(
    families: tuple[str, ...], filename_suffix: str,
) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...], str, bool]]:
    """The (metric, ylabel, bar-roles, baseline-roles, filename-stem,
    include_best) specs for a pooled figure set: the MSE figure (floor +
    ceiling lines), a no-random MSE variant, the correlation figure, and two
    with-best MSE figures that add the best-surfaced-per-run bar (with and
    without the random floor). `include_best` tells `plot_pooled` to augment
    the data. gt lines are dropped unless EVERY family is canonical."""
    line = _gt_line_filter(families)
    return [
        ("mse", YLABEL_MSE, ROLES_MSE, line(BASELINE_ROLES_MSE),
         f"recovery_mse_{filename_suffix}", False),
        ("mse", YLABEL_MSE, ROLES_MSE, line(BASELINE_ROLES_MSE_NO_RANDOM),
         f"recovery_mse_{filename_suffix}_no_random", False),
        ("pearson_r", YLABEL_CORR, ROLES_CORR, line(BASELINE_ROLES_CORR),
         f"recovery_correlation_{filename_suffix}", False),
        ("mse", YLABEL_MSE, ROLES_MSE_WITH_BEST, line(BASELINE_ROLES_MSE),
         f"recovery_mse_{filename_suffix}_with_best", True),
        ("mse", YLABEL_MSE, ROLES_MSE_WITH_BEST,
         line(BASELINE_ROLES_MSE_NO_RANDOM),
         f"recovery_mse_{filename_suffix}_with_best_no_random", True),
    ]


def per_model_summary(
    long_df: pd.DataFrame, family: str, *, metric: str, higher_is_better: bool,
    include_best: bool = True,
) -> pd.DataFrame:
    """Plot-ready summary (mean/sem per role) for a SINGLE family. Restricts to
    that family's rows first; when `include_best` it then augments with the two
    metric-specific best-surfaced bars (max r / min MSE) before the two-stage
    `summarise`. Filtering before selection guarantees a family's bars are
    derived from its own rows only — no cross-family leakage."""
    fam_long = long_df[long_df["family"] == family]
    if include_best:
        fam_long = pd.concat(
            [
                fam_long,
                surfaced_best_rows(
                    fam_long, metric=metric, higher_is_better=higher_is_better,
                ),
                surfaced_best_across_runs_rows(
                    fam_long, metric=metric, higher_is_better=higher_is_better,
                ),
            ],
            ignore_index=True,
        )
    return summarise(fam_long, metric)


def plot_per_model(
    long_df: pd.DataFrame, *, out_dir: Path, metric: str,
    higher_is_better: bool, roles: tuple[str, ...], ylabel: str,
    filename_prefix: str, include_best: bool = True, show_title: bool = True,
    title_formatter=None, baseline_roles: tuple[str, ...] = (),
) -> list[Path]:
    """Write one single-panel `metric` figure per family to
    ``out_dir/<filename_prefix>_<family>.png`` (plus the .svg sibling via
    `save_figure`). `include_best` toggles the two best-surfaced bars,
    `show_title`/`title_formatter` the per-panel title, and `baseline_roles`
    the per-noise reference lines. Returns the .png paths in family-plot
    order."""
    out_dir = Path(out_dir)
    out_paths: list[Path] = []
    for family in family_plot_order(long_df["family"], priority=FAMILY_PLOT_ORDER):
        summary = per_model_summary(
            long_df, family, metric=metric, higher_is_better=higher_is_better,
            include_best=include_best,
        )
        out_path = out_dir / f"{filename_prefix}_{family}.png"
        plot_grid(summary, out_path, roles=roles, ylabel=ylabel,
                  show_title=show_title, title_formatter=title_formatter,
                  baseline_roles=baseline_roles_for_family(family, baseline_roles),
                  bar_edgecolor="none")
        out_paths.append(out_path)
    return out_paths


def plot_pooled(
    long_df: pd.DataFrame, *, out_dir: Path,
    families: tuple[str, ...] = NONCANONICAL_FAMILIES,
    fig_width: float | None = POOLED_FIG_WIDTH,
    filename_suffix: str = "noncanonical_pooled",
    bar_edgecolor: str = "none",
) -> list[Path]:
    """One COMBINED MSE figure and one combined correlation figure pooling the
    given `families` into a single panel row. `fig_width` (inches total) shrinks
    the panels as more families are added; pass ``None`` for the default
    6in/panel (used by the canonical pool). Same styling as the per-model
    figures: heuristic titles, the two best-surfaced bars dropped, and (MSE
    only) `random` as a dotted per-noise reference line. `bar_edgecolor` is the
    outline drawn around each bar ("none" = no outline, the default; "black"
    for the paper figure). Returns the .png paths; writes nothing and returns
    ``[]`` if no rows match `families`."""
    out_dir = Path(out_dir)
    sub = long_df[long_df["family"].isin(families)]
    if sub.empty:
        print(f"[per-model] pooled: no rows for families {families} — skipped.",
              file=sys.stderr)
        return []

    # Five figures: MSE (random floor + gt ceiling lines), a no-random MSE
    # variant, correlation, and two with-best MSE figures (adding the
    # best-surfaced-per-run bar, with and without the random floor). gt lines
    # are dropped for non-canonical pools. The with-best figures augment the
    # data with the min-MSE / max-r theory per run-dir, averaged across runs.
    out_paths: list[Path] = []
    for metric, ylabel, roles, baseline_roles, stem, include_best in (
        pooled_specs(families, filename_suffix)
    ):
        data = sub
        if include_best:
            data = pd.concat(
                [sub, surfaced_best_rows(
                    sub, metric=metric, higher_is_better=metric == "pearson_r")],
                ignore_index=True,
            )
        out_path = out_dir / f"{stem}.png"
        plot_grid(
            summarise(data, metric), out_path, roles=roles, ylabel=ylabel,
            show_title=True, title_formatter=heuristic_title,
            baseline_roles=baseline_roles, fig_width=fig_width,
            family_order=FAMILY_PLOT_ORDER, bar_edgecolor=bar_edgecolor,
        )
        out_paths.append(out_path)
    return out_paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Per-model ground-truth recovery on the Hilbig task: one "
            "single-panel MSE and correlation figure per family (canonical "
            "sampling + non-canonical), rebuilt fresh from the run-dirs."
        )
    )
    p.add_argument("--results-root", type=Path, default=RESULTS_ROOT_SAMPLING)
    p.add_argument("--families", nargs="+", default=list(ALL_FAMILIES_DEFAULT))
    p.add_argument(
        "--noises", nargs="*", type=float, default=None,
        help="Noise levels to include (e.g. 0.0 0.5 0.75 1.0). Empty = all "
             "found per family.",
    )
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-draws", type=int, default=N_DRAWS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for the per-model figures "
             "(default <results-root>/per_model).",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Optional path to also write the rebuilt long-format table.",
    )
    p.add_argument(
        "--from-csv", type=Path, default=None,
        help="Load the long-format table from this CSV instead of rebuilding "
             "from the run-dirs — a fast re-plot with no re-simulation.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.results_root / "per_model")

    if args.from_csv is not None:
        long_df = pd.read_csv(args.from_csv)
        print(f"[per-model] loaded long table {args.from_csv} "
              f"({len(long_df)} rows, no re-simulation)")
    else:
        # Empty list (--noises with no values) -> None so every noise dir kept.
        noises = args.noises if args.noises else None
        pairs = unique_stimulus_pairs(args.human_data)
        print(f"[per-model] stimuli={len(pairs)} families={args.families} "
              f"noises={noises} n_draws={args.n_draws} seed={args.seed}")
        long_df = build_results_long(
            results_root=args.results_root, families=args.families,
            noises=noises, stimulus_pairs=pairs,
            n_draws=args.n_draws, base_seed=args.seed,
        )
        # Append the synthetic 'random' family: the canonical families' ε=1
        # runs (GT behaves randomly) scored against a fixed 0.5 reference.
        random_rows = build_random_family_rows(
            results_root=args.results_root,
            canonical_families=list(CANONICAL_SAMPLING_FAMILIES),
            stimulus_pairs=pairs, n_draws=args.n_draws, base_seed=args.seed,
        )
        long_df = pd.concat([long_df, random_rows], ignore_index=True)
    if long_df.empty:
        print("[per-model] no results — check --results-root / --families.",
              file=sys.stderr)
        return 1

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.csv, index=False)
        print(f"[per-model] wrote long table {args.csv}")

    # Per-model figures: drop the two best-surfaced bars, show heuristic titles,
    # and (MSE only) draw `random` and `gt` as per-noise reference lines.
    mse_paths = plot_per_model(
        long_df, out_dir=out_dir, metric="mse", higher_is_better=False,
        roles=ROLES_MSE, ylabel=YLABEL_MSE, filename_prefix="recovery_mse",
        include_best=False, show_title=True, title_formatter=heuristic_title,
        baseline_roles=BASELINE_ROLES_MSE,
    )
    corr_paths = plot_per_model(
        long_df, out_dir=out_dir, metric="pearson_r", higher_is_better=True,
        roles=ROLES_CORR, ylabel=YLABEL_CORR,
        filename_prefix="recovery_correlation",
        include_best=False, show_title=True, title_formatter=heuristic_title,
        baseline_roles=BASELINE_ROLES_CORR,
    )
    # Two combined figures: the non-canonical families compressed to ~2
    # canonical panels wide, and the canonical sampling families at full width.
    pooled_noncanon = plot_pooled(long_df, out_dir=out_dir, bar_edgecolor="black")
    pooled_canon = plot_pooled(
        long_df, out_dir=out_dir, families=CANONICAL_SAMPLING_FAMILIES,
        fig_width=None, filename_suffix="canonical_pooled",
    )
    print(f"[per-model] wrote {len(mse_paths)} MSE + {len(corr_paths)} "
          f"correlation per-model figures + {len(pooled_noncanon)} "
          f"non-canonical + {len(pooled_canon)} canonical pooled figures "
          f"to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
