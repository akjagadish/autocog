"""Paper-style (Figure 3C) recovery bars comparing discovery LLMs side by side.

For each ground-truth family, one panel; x = discovery model (e.g. Gemini
3.1 Pro vs Claude Opus 5), bars = model roles (Seed / Surfaced / Surfaced
best), y = MSE of choice proportions against the noiseless ground truth on
the Hilbig stimuli (or Pearson r with ``--metric pearson_r``). Error bars are
SEM over runs, aggregated exactly as ``scripts/recovery_correlation.py``:
theories pooled within a run-dir first, then mean +/- SEM across run-dirs.

Inputs are the long-format tables written by ``plot_recovery_per_model.py
--csv`` (or ``recovery_correlation.py --csv``), one per discovery model::

    python scripts/plot_recovery_model_comparison.py \\
        --inputs "Gemini 3.1 Pro (20 cycles)=results/recovery/analysis/per_model/recovery_long.csv" \\
                 "Claude Opus 5 (10 cycles)=results/recovery_anthropic/analysis/per_model/recovery_long.csv" \\
        --families single_cue anti_majority --metric mse \\
        --out results/recovery_anthropic/analysis/recovery_mse_model_comparison

Writes ``<out>.svg`` and ``<out>.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    ROLE_COLOR,
    role_label,
    save_figure,
    style_axes,
)
from scripts.plot_recovery_per_model import heuristic_title  # noqa: E402
from scripts.recovery_correlation import (  # noqa: E402
    summarise,
    surfaced_best_rows,
)

# Bars drawn per model, in this order (matches the paper's Figure 3C).
ROLES: tuple[str, ...] = ("seed", "surfaced", "surfaced (best per run)")
_HIGHER_IS_BETTER = {"mse": False, "pearson_r": True}
_YLABEL = {"mse": r"MSE $\hat{p}(B)$ vs. ground truth", "pearson_r": "Pearson r vs. ground truth"}


def build_comparison_table(
    inputs: dict[str, Path], *, metric: str = "mse", noise: float = 0.0,
) -> pd.DataFrame:
    """Mean/SEM per (model, family, role) from one long-format CSV per model.

    ``inputs`` maps display label -> CSV path; label order is preserved as the
    categorical order of the ``model`` column (left-to-right on the x-axis).
    Only rows at action-noise ``noise`` are used (one bar group per model
    needs a single noise level). The best-surfaced role is recomputed per
    metric (min MSE / max r per run); any best rows already in the CSV are
    dropped first so they are not double-counted.
    """
    frames = []
    for label, path in inputs.items():
        long_df = pd.read_csv(path)
        at_noise = np.isclose(long_df.noise.astype(float), noise)
        long_df = long_df[at_noise & (long_df.role != ROLES[2])]
        if long_df.empty:
            raise ValueError(f"{path}: no rows at noise={noise} (levels present: "
                             f"{sorted(pd.read_csv(path).noise.unique())})")
        long_df = pd.concat(
            [long_df, surfaced_best_rows(
                long_df, metric=metric, higher_is_better=_HIGHER_IS_BETTER[metric],
            )],
            ignore_index=True,
        )
        summary = summarise(long_df, metric)
        summary = summary[summary.role.isin(ROLES)].copy()
        summary["model"] = label
        frames.append(summary)
    table = pd.concat(frames, ignore_index=True)
    table["model"] = pd.Categorical(table["model"], categories=list(inputs), ordered=True)
    return table[["model", "family", "noise", "role", "mean", "sem", "n_runs"]]


def plot_comparison(
    table: pd.DataFrame, *, families: list[str], metric: str,
) -> matplotlib.figure.Figure:
    """One panel per family; grouped bars (roles) per model with SEM error bars."""
    models = list(table["model"].cat.categories)
    # Panels widen with the number of model groups so wrapped tick labels
    # (two lines each) do not collide; ticks step down one size beyond three.
    panel_w = max(5.0, 1.8 * len(models))
    tick_fs = FONTSIZE - 2 if len(models) <= 3 else FONTSIZE - 4
    fig, axes = plt.subplots(1, len(families), figsize=(panel_w * len(families), 4), squeeze=False)
    width = 0.8 / len(ROLES)
    for col, family in enumerate(families):
        ax = axes[0][col]
        fam = table[table.family == family]
        for i, role in enumerate(ROLES):
            rows = fam[fam.role == role].set_index("model").reindex(models)
            x = [m - 0.4 + width * (i + 0.5) for m in range(len(models))]
            ax.bar(
                x, rows["mean"].to_numpy(), width, yerr=rows["sem"].to_numpy(),
                color=ROLE_COLOR[role], edgecolor="none",
                label=role_label(role) if col == 0 else None,
            )
        ax.set_xticks(range(len(models)))
        # Long labels like "Gemini 3.1 Pro (20 cycles)" wrap at the parenthesis.
        ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=tick_fs)
        style_axes(ax, title=heuristic_title(family),
                   ylabel=_YLABEL[metric] if col == 0 else None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=FONTSIZE - 2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _parse_inputs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        label, _, path = item.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"--inputs entries are LABEL=PATH; got {item!r}")
        if label in out:
            raise argparse.ArgumentTypeError(f"duplicate --inputs label {label!r}")
        out[label] = Path(path)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="LABEL=long_csv, one per discovery model.")
    p.add_argument("--families", nargs="+", default=["single_cue", "anti_majority"])
    p.add_argument("--metric", choices=sorted(_HIGHER_IS_BETTER), default="mse")
    p.add_argument("--noise", type=float, default=0.0, help="Action-noise level to plot.")
    p.add_argument("--out", type=Path, required=True, help="Output path base (.svg and .png written).")
    p.add_argument("--csv", type=Path, default=None, help="Optional path for the summary table.")
    args = p.parse_args(argv)

    try:
        inputs = _parse_inputs(args.inputs)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    table = build_comparison_table(inputs, metric=args.metric, noise=args.noise)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
    fig = plot_comparison(table, families=args.families, metric=args.metric)
    written = save_figure(fig, args.out)
    print(f"[model-comparison] wrote {', '.join(str(w) for w in written)}")


if __name__ == "__main__":
    main()
