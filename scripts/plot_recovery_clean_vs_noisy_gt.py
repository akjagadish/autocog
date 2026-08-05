"""Recovery scored against BOTH the clean and the noisy ground truth.

For each canonical sampling family (TTB / WADD / Tallying) and action-noise
level ε ∈ {0, 0.5, 0.75, 1.0}, each role (seed / surfaced / best-surfaced / gt)
gets TWO bars side by side:

  vs clean gt   - MSE of the role's (noiseless) predictions to the CLEAN gt@0
                  reference. Distance to the TRUE signal (recovery quality).
  vs noisy gt   - MSE of the same predictions to the NOISY gt@ε reference, i.e.
                  to the data autocog actually saw at that noise level.

The `gt` role is the ground-truth reference itself, shown as paired bars too:
its "vs clean gt" bar is gt@ε vs clean gt@0 (RISES with ε — the noisy
generator's own distance from the true signal) and its "vs noisy gt" bar is
gt@ε vs its own noisy draw (the FLOOR). The two bars share a role colour;
"vs clean" is hatched, "vs noisy" is solid.

Implementation: build the long table twice — once with the per-ε noisy
reference (`reference_action_noise=None`) and once with the clean reference
(`reference_action_noise=0.0`). Every role's own predictions are identical
across the two builds (same seeds); only the reference each row is scored
against differs, so the two bars are a like-for-like contrast.

Two-step workflow: build once (writes a merged long table with a `reference`
column), then re-plot from it with `--from-csv` (no re-simulation).

Outputs (under --out-dir, default <results-root>/recovery_clean_vs_noisy_gt):
  recovery_clean_vs_noisy_gt_mse.{svg,png}
  recovery_clean_vs_noisy_gt_correlation.{svg,png}
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

from scripts.eval_hilbig import HUMAN_DATA_DEFAULT  # noqa: E402
from scripts.recovery_correlation import (  # noqa: E402
    N_DRAWS_DEFAULT,
    build_results_long,
    family_plot_order,
    summarise,
    surfaced_best_rows,
    unique_stimulus_pairs,
)
from scripts.plot_recovery_per_model import (  # noqa: E402
    CANONICAL_SAMPLING_FAMILIES,
    RESULTS_ROOT_SAMPLING,
    heuristic_title,
)
from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    ROLE_COLOR,
    role_label,
    save_figure,
    style_axes,
)

# Roles shown as paired bars (left to right within each noise group). `gt` is
# the ground-truth reference itself: its two bars are gt@ε vs clean gt (hatched,
# RISES with ε) and gt@ε vs noisy gt (solid, the FLOOR — gt vs its own noisy
# draw), each with error bars like the recovery roles.
PLOT_ROLES: tuple[str, ...] = (
    "seed", "surfaced", "surfaced (best per run)", "gt",
)
# Reference labels and bar styling: clean is hatched, noisy is solid.
REF_NOISY: str = "noisy"
REF_CLEAN: str = "clean"
_REF_HATCH: dict[str, str | None] = {REF_CLEAN: "///", REF_NOISY: None}
YLABEL_MSE: str = r"$\mathrm{MSE}_{\hat{p(B)}}$ (lower = better)"
YLABEL_CORR: str = r"$r_{\hat{p(B)}}$"


def _with_best(long_df: pd.DataFrame, *, metric: str, higher_is_better: bool):
    """Append the per-run best-surfaced rows (metric-specific) to `long_df`."""
    return pd.concat(
        [long_df,
         surfaced_best_rows(long_df, metric=metric,
                            higher_is_better=higher_is_better)],
        ignore_index=True,
    )


def build_clean_and_noisy(
    *, results_root: Path, families: list[str], noises: list[float] | None,
    stimulus_pairs, n_draws: int, base_seed: int,
) -> pd.DataFrame:
    """Long table scored against BOTH references, tagged by a `reference`
    column ('clean' = gt@0, 'noisy' = gt@ε). The two builds share every role's
    predictions; only the target differs."""
    noisy = build_results_long(
        results_root=results_root, families=families, noises=noises,
        stimulus_pairs=stimulus_pairs, n_draws=n_draws, base_seed=base_seed,
        reference_action_noise=None,
    )
    clean = build_results_long(
        results_root=results_root, families=families, noises=noises,
        stimulus_pairs=stimulus_pairs, n_draws=n_draws, base_seed=base_seed,
        reference_action_noise=0.0,
    )
    noisy["reference"] = REF_NOISY
    clean["reference"] = REF_CLEAN
    return pd.concat([noisy, clean], ignore_index=True)


def _summary_lookup(summary: pd.DataFrame) -> dict:
    """{(family, noise, role): (mean, sem)} from a `summarise` frame."""
    return {
        (r["family"], r["noise"], r["role"]): (r["mean"], r["sem"])
        for _, r in summary.iterrows()
    }


def plot_paired_bars(
    long_both: pd.DataFrame, out_path: Path, *,
    metric: str, higher_is_better: bool, ylabel: str, title: str | None = None,
) -> None:
    """One panel per family; x = noise; per role a CLEAN (hatched) and NOISY
    (solid) bar. Best-surfaced is selected per reference (metric-specific)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Per-reference summary (best-surfaced chosen within each reference).
    lookups = {}
    for ref in (REF_CLEAN, REF_NOISY):
        sub = long_both[long_both["reference"] == ref]
        s = summarise(_with_best(sub, metric=metric,
                                 higher_is_better=higher_is_better), metric)
        lookups[ref] = _summary_lookup(s)

    families = family_plot_order(long_both["family"])
    refs = (REF_CLEAN, REF_NOISY)
    fig, axes = plt.subplots(
        1, len(families), figsize=(7.0 * len(families), 4),
        sharey=True, squeeze=False,
    )
    for col, fam in enumerate(families):
        ax = axes[0][col]
        noises = sorted(long_both[long_both.family == fam].noise.unique())
        x = np.arange(len(noises))
        # 6 sub-bars per noise group: 3 roles x 2 references.
        sub_w = 0.8 / (len(PLOT_ROLES) * len(refs))
        for j, role in enumerate(PLOT_ROLES):
            for k, ref in enumerate(refs):
                means = np.array(
                    [lookups[ref].get((fam, n, role), (np.nan, 0.0))[0]
                     for n in noises], dtype=float)
                sems = np.array(
                    [lookups[ref].get((fam, n, role), (np.nan, 0.0))[1]
                     for n in noises], dtype=float)
                offset = -0.4 + (j * len(refs) + k + 0.5) * sub_w
                ax.bar(
                    x + offset, means, sub_w, yerr=sems,
                    color=ROLE_COLOR[role], hatch=_REF_HATCH[ref],
                    edgecolor="black", linewidth=0.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:g}" for n in noises])
        style_axes(ax, xlabel="action noise (ε)", title=heuristic_title(fam))
    axes[0][0].set_ylabel(ylabel, fontsize=FONTSIZE)

    # Legend: role colours + the two reference patterns (hatch vs solid).
    role_handles = [Patch(facecolor=ROLE_COLOR[r], edgecolor="black",
                          label=role_label(r)) for r in PLOT_ROLES]
    ref_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=_REF_HATCH[REF_CLEAN],
              label="vs clean gt"),
        Patch(facecolor="white", edgecolor="black", label="vs noisy gt"),
    ]
    fig.legend(handles=role_handles + ref_handles, loc="lower center",
               ncol=len(role_handles) + len(ref_handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=FONTSIZE - 2)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    save_figure(fig, out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Recovery scored against BOTH the clean (gt@0) and noisy (gt@ε) "
            "ground truth: paired bars per role across action-noise levels."
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
                   help="default <results-root>/recovery_clean_vs_noisy_gt")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional path to also write the merged long table")
    p.add_argument(
        "--from-csv", type=Path, default=None,
        help="Load the merged long table (with a `reference` column) instead of "
             "rebuilding from the run-dirs — a fast re-plot with no re-sim.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.results_root / "recovery_clean_vs_noisy_gt")

    if args.from_csv is not None:
        long_both = pd.read_csv(args.from_csv)
        print(f"[clean-vs-noisy] loaded {args.from_csv} "
              f"({len(long_both)} rows, no re-simulation)")
    else:
        noises = args.noises if args.noises else None
        pairs = unique_stimulus_pairs(args.human_data)
        print(f"[clean-vs-noisy] stimuli={len(pairs)} families={args.families} "
              f"noises={noises} n_draws={args.n_draws} seed={args.seed}")
        long_both = build_clean_and_noisy(
            results_root=args.results_root, families=args.families,
            noises=noises, stimulus_pairs=pairs,
            n_draws=args.n_draws, base_seed=args.seed,
        )
    if long_both.empty:
        print("[clean-vs-noisy] no results — check --results-root / --families.",
              file=sys.stderr)
        return 1
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        long_both.to_csv(args.csv, index=False)
        print(f"[clean-vs-noisy] wrote merged long table {args.csv}")

    plot_paired_bars(
        long_both, out_dir / "recovery_clean_vs_noisy_gt_mse.png",
        metric="mse", higher_is_better=False, ylabel=YLABEL_MSE,
        title="Recovery vs clean and noisy ground truth (MSE)",
    )
    plot_paired_bars(
        long_both, out_dir / "recovery_clean_vs_noisy_gt_correlation.png",
        metric="pearson_r", higher_is_better=True, ylabel=YLABEL_CORR,
        title="Recovery vs clean and noisy ground truth (Pearson r)",
    )
    print(f"[clean-vs-noisy] wrote MSE + correlation figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
