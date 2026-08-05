"""Trajectory of how well an autopi run's surfaced theories capture human
P(B), evaluated at the end of several round cutoffs (default 5/10/15/20/25)
against the 10 Prolific decision-making experiments.

At each cutoff N the surviving theories (un-killed survivor + replacement
from `rounds/round_{N-1:03d}`) are replayed on every experiment's unique
stimulus pairs, scored against the pooled human P(B) via MAE / MSE /
Pearson r, then averaged across experiments (mean +/- SEM). The run's seed
theories (round-0 base: tallying, ttb) are evaluated as round-invariant
reference lines.

Outputs (under <run-dir>/analysis/eval_human_trajectory/):
  trajectory.csv             — per (cutoff, model, metric) mean +/- SEM across experiments
  per_experiment_summary.csv — raw per (cutoff, experiment, model) MAE/MSE/r
  calibration_scatter.png    — model-vs-human P(B), dot=stimulus, both-axis SEM bars,
                               grid rows=survivor slot x cols=cutoff
  trajectory_<target>.png    — 3 subplots (MAE/MSE/r) vs cutoff round
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_human import (  # noqa: E402
    HUMAN_DATA_DEFAULT,
    _predict_p_b,
    _sample_hilibig_params,
    compute_human_proportions,
    extract_human_design,
    load_human_choices,
    resolve_base_theories,
    resolve_surfaced_theories,
)
from src.theory import Theory  # noqa: E402
from scripts.figure_style import (  # noqa: E402
    FONTSIZE,
    GRAY,
    NEUTRAL_HARMONY,
    ROLE_COLOR,
    save_figure,
    style_axes,
)

CUTOFFS_DEFAULT: tuple[int, ...] = (5, 10, 15, 20, 25)


def gather_experiments(human_dir: Path, *, n_expected: int = 10) -> list[Path]:
    """All `*.txt` under `human_dir` except `*_og.txt`, sorted. Aborts if
    the count != `n_expected` so we never silently score the wrong set."""
    if not human_dir.is_dir():
        raise FileNotFoundError(f"--human-data must be a directory: {human_dir!s}")
    files = sorted(
        p for p in human_dir.glob("*.txt") if not p.stem.endswith("_og")
    )
    if len(files) != n_expected:
        raise SystemExit(
            f"[trajectory] expected {n_expected} experiments under "
            f"{human_dir} (excluding *_og.txt), found {len(files)}: "
            f"{[p.name for p in files]}"
        )
    return files


def summarize_fit(
    model_mean: np.ndarray, human_pb: np.ndarray,
) -> tuple[float, float, float]:
    """MAE / MSE / Pearson r of model vs human P(B). r is NaN if either
    vector is constant. Matches `eval_human.evaluate_models` exactly."""
    diff = model_mean - human_pb
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    if model_mean.std() > 0 and human_pb.std() > 0:
        r = float(np.corrcoef(model_mean, human_pb)[0, 1])
    else:
        r = float("nan")
    return mae, mse, r


def replay_with_spread(
    theory: Theory,
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    validities: list[float],
    rating_max: int,
    n_subjects: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair (mean P(B), SEM across parameter draws). The mean matches
    `eval_human.model_proportions_for_theory` (identical seeding); the SEM
    is the across-draws standard error kept for error bars."""
    n_features = len(validities)
    random.seed(base_seed)
    probs = np.empty((n_subjects, len(pairs)), dtype=float)
    for s in range(n_subjects):
        params = _sample_hilibig_params(
            theory,
            validities=validities,
            rating_max=rating_max,
            n_features=n_features,
            seed_override=base_seed + s,
        )
        for j, (a, b) in enumerate(pairs):
            probs[s, j] = _predict_p_b(theory, params, a, b)
    mean = probs.mean(axis=0)
    if n_subjects > 1:
        sem = probs.std(axis=0, ddof=1) / np.sqrt(n_subjects)
    else:
        sem = np.zeros(len(pairs))
    return mean, sem


def replay_matched_to_participants(
    theory: Theory,
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    n_per_pair: list[int],
    validities: list[float],
    rating_max: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Like `replay_with_spread`, but each pair draws as many model
    "subjects" (parameter sets) as there were human participants for that
    pair, so the model's sampling SEM is on the same footing as the human
    across-subject SEM (instead of an over-precise fixed 200). A single
    pool of `max(n_per_pair)` draws is sampled once and sliced per pair, so
    pair j uses draws `0..n_per_pair[j]`; this keeps the s-th model subject
    identical to `replay_with_spread`'s s-th subject for the same seed."""
    n_per_pair = [int(n) for n in n_per_pair]
    n_features = len(validities)
    n_max = max(n_per_pair)
    random.seed(base_seed)
    probs = np.empty((n_max, len(pairs)), dtype=float)
    for s in range(n_max):
        params = _sample_hilibig_params(
            theory,
            validities=validities,
            rating_max=rating_max,
            n_features=n_features,
            seed_override=base_seed + s,
        )
        for j, (a, b) in enumerate(pairs):
            probs[s, j] = _predict_p_b(theory, params, a, b)
    mean = np.empty(len(pairs), dtype=float)
    sem = np.empty(len(pairs), dtype=float)
    for j, n_j in enumerate(n_per_pair):
        col = probs[:n_j, j]
        mean[j] = float(col.mean())
        sem[j] = float(col.std(ddof=1) / np.sqrt(n_j)) if n_j > 1 else 0.0
    return mean, sem


_METRICS: tuple[str, ...] = ("mae", "mse", "pearson_r")


def eval_cutoff(
    cutoff_round: int,
    theories: dict[str, Theory],
    role: str,
    experiments: list[Path],
    *,
    n_subjects: int,
    seed: int,
    target: str,
    match_participants: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every theory in `theories` on every experiment. Returns
    (summary_rows, scatter_rows): one summary row per (experiment, model)
    with MAE/MSE/r, and one scatter row per (experiment, model, stimulus)
    with human/model P(B) and their SEMs. When `match_participants` is set,
    each pair is replayed with as many model draws as it had human
    participants (so model SEM is comparable to the human across-subject
    SEM) instead of a fixed `n_subjects`."""
    summary_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    for exp in experiments:
        df_human = load_human_choices(exp)
        validities, rating_max = extract_human_design(df_human)
        props = compute_human_proportions(df_human)
        pairs = list(zip(props["option_a"], props["option_b"]))
        human_pb = props[target].to_numpy(dtype=float)
        human_sem = props["p_b_sem"].to_numpy(dtype=float)
        n_per_pair = props["n_participants"].to_numpy(dtype=int)

        for label, theory in theories.items():
            if match_participants:
                model_mean, model_sem = replay_matched_to_participants(
                    theory, pairs, n_per_pair=list(n_per_pair),
                    validities=validities, rating_max=rating_max,
                    base_seed=seed,
                )
            else:
                model_mean, model_sem = replay_with_spread(
                    theory, pairs,
                    validities=validities, rating_max=rating_max,
                    n_subjects=n_subjects, base_seed=seed,
                )
            mae, mse, r = summarize_fit(model_mean, human_pb)
            summary_rows.append({
                "cutoff_round": cutoff_round, "experiment": exp.stem,
                "model": label, "role": role,
                "mae": mae, "mse": mse, "pearson_r": r,
            })
            for k, (a, b) in enumerate(pairs):
                scatter_rows.append({
                    "cutoff_round": cutoff_round, "experiment": exp.stem,
                    "model": label, "role": role,
                    "option_a": str(list(a)), "option_b": str(list(b)),
                    "p_b_human": float(human_pb[k]),
                    "p_b_human_sem": float(human_sem[k]),
                    "p_b_model": float(model_mean[k]),
                    "p_b_model_sem": float(model_sem[k]),
                })
    return summary_rows, scatter_rows


def aggregate_trajectory(per_exp: pd.DataFrame) -> pd.DataFrame:
    """Long-format mean +/- SEM of each metric across experiments, grouped
    by (cutoff_round, model, role, metric). NaN metric values (e.g. a
    constant-prediction Pearson r on one experiment) are skipped; SEM uses
    the count of finite values."""
    rows: list[dict[str, Any]] = []
    keys = ["cutoff_round", "model", "role"]
    for (cutoff, model, role), g in per_exp.groupby(keys, sort=True):
        for metric in _METRICS:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            mean = float(np.mean(vals)) if n else float("nan")
            if n > 1:
                sem = float(np.std(vals, ddof=1) / np.sqrt(n))
            else:
                sem = 0.0
            rows.append({
                "cutoff_round": int(cutoff), "model": model, "role": role,
                "metric": metric, "mean_across_exps": mean,
                "sem_across_exps": sem, "n_experiments": n,
            })
    return pd.DataFrame(rows)


def _calibration_dot_color(
    label: str, theory_color: dict[str, str] | None,
) -> str:
    """Dot colour for a calibration panel. With no `theory_color` map, keep
    the legacy single surfaced colour; with a map, mirror the trajectory's
    `surfaced_color` (mapped colour, gray fallback) so the two figures agree."""
    if theory_color is None:
        return ROLE_COLOR["surfaced"]
    return theory_color.get(label, GRAY)


def _draw_calibration_panel(
    ax, scatter_df: pd.DataFrame, label: str, role: str, c: int, *,
    theory_color: dict[str, str] | None, ground_truth_label: str,
    ylabel: bool, xlabel: bool, show_title: bool = True,
) -> None:
    """Render one model@cutoff calibration panel onto `ax`: pooled stimulus
    dots with both-axis SEM bars, a y=x diagonal, and a MAE/r title.
    `show_title=False` drops the title for publication figures."""
    sub = scatter_df[
        (scatter_df["cutoff_round"] == c)
        & (scatter_df["model"] == label)
    ]
    h = sub["p_b_human"].to_numpy(dtype=float)
    m = sub["p_b_model"].to_numpy(dtype=float)
    ax.errorbar(
        h, m,
        xerr=sub["p_b_human_sem"].to_numpy(dtype=float),
        yerr=sub["p_b_model_sem"].to_numpy(dtype=float),
        fmt="o", ms=6, alpha=0.6,
        color=_calibration_dot_color(label, theory_color),
    )
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    mae = float(np.mean(np.abs(m - h)))
    r = (
        float(np.corrcoef(m, h)[0, 1])
        if m.std() > 0 and h.std() > 0 else float("nan")
    )
    style_axes(
        ax,
        ylabel=r"Model $\hat{p}(B)$" if ylabel else None,
        xlabel=(
            ground_truth_label + r" $\hat{p}(B)$" if xlabel else None
        ),
    )
    if show_title:
        ax.set_title(
            f"r{c}  {label} [{role}]\nMAE={mae:.3f}  r={r:.2f}",
            fontsize=FONTSIZE - 6,
        )


def plot_calibration_per_model(
    scatter_df: pd.DataFrame, out_dir: Path, prefix: str, *,
    cutoffs: list[int], ground_truth_label: str = "human",
    theory_color: dict[str, str] | None = None, show_title: bool = True,
) -> list[Path]:
    """Write one standalone calibration figure per surfaced model, each a
    1-row strip of that model's panels across the cutoffs where it appears.
    Files are `{out_dir}/{prefix}_{model}.png`. `show_title=False` drops the
    per-panel MAE/r titles for publication figures. Returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for label in dict.fromkeys(scatter_df["model"]):
        ms = scatter_df[scatter_df["model"] == label]
        role = str(ms["role"].iloc[0])
        model_cutoffs = [c for c in cutoffs if c in set(ms["cutoff_round"])]
        if not model_cutoffs:
            continue
        n_cols = len(model_cutoffs)
        fig, axes = plt.subplots(
            1, n_cols, figsize=(4 * n_cols, 4), squeeze=False,
        )
        for ci, c in enumerate(model_cutoffs):
            _draw_calibration_panel(
                axes[0][ci], scatter_df, label, role, c,
                theory_color=theory_color,
                ground_truth_label=ground_truth_label,
                ylabel=ci == 0, xlabel=True, show_title=show_title,
            )
        fig.tight_layout()
        path = out_dir / f"{prefix}_{label}.png"
        save_figure(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def plot_calibration_grid(
    scatter_df: pd.DataFrame, path: Path, *, cutoffs: list[int],
    ground_truth_label: str = "human",
    theory_color: dict[str, str] | None = None,
    show_title: bool = True,
) -> None:
    """Grid of model-vs-human P(B) calibration panels. Rows = survivor
    slot (survivor first, replacement last), columns = cutoff round. Each
    panel pools all experiments' stimuli as dots with both-axis SEM bars
    and a y=x diagonal; the title annotates the model label and its
    mean-across-experiments MAE/r computed from the pooled dots. With
    `theory_color`, dots are coloured per model to match the trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    role_order = {"survivor": 0, "replacement": 1}

    # Build a per-cutoff ordered list of (row, label, role).
    cells: dict[tuple[int, int], tuple[str, str]] = {}
    n_rows = 1
    for c in cutoffs:
        sub = scatter_df[scatter_df["cutoff_round"] == c]
        labels = list(dict.fromkeys(zip(sub["model"], sub["role"])))
        labels.sort(key=lambda lr: role_order.get(lr[1], 99))
        for row, (label, role) in enumerate(labels):
            cells[(c, row)] = (label, role)
            n_rows = max(n_rows, row + 1)

    n_cols = len(cutoffs)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False,
    )
    for ci, c in enumerate(cutoffs):
        for row in range(n_rows):
            ax = axes[row][ci]
            cell = cells.get((c, row))
            if cell is None:
                ax.axis("off")
                continue
            label, role = cell
            _draw_calibration_panel(
                ax, scatter_df, label, role, c,
                theory_color=theory_color,
                ground_truth_label=ground_truth_label,
                ylabel=ci == 0, xlabel=row == n_rows - 1,
                show_title=show_title,
            )

    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


TRAJ_METRICS: tuple[tuple[str, str], ...] = (
    ("mae", "MAE"), ("mse", r"$MSE_{\hat{p}(B)}$"),
    ("pearson_r", r"$r_{\hat{p}(B)}$"),
)
# Seed reference colours when no convergence palette is supplied.
_DEFAULT_SEED_COLORS = {
    "pi_1": NEUTRAL_HARMONY["slate"], "pi_2": NEUTRAL_HARMONY["sage"],
}


def _seed_traj_color(label: str, theory_color: dict[str, str] | None) -> str:
    if theory_color is not None:
        return theory_color.get(label, GRAY)
    return _DEFAULT_SEED_COLORS.get(label, GRAY)


def _surfaced_traj_color(label: str, theory_color: dict[str, str] | None) -> str:
    if theory_color is not None:
        # Fail safe to gray (transient) for any label not in the map, never
        # to terracotta — that is the "other final survivor" colour.
        return theory_color.get(label, GRAY)
    return ROLE_COLOR["surfaced"]


def _draw_metric_trajectory_panel(
    ax, traj_df: pd.DataFrame, metric: str, nice: str, *,
    seeds_as_line: bool, theory_color: dict[str, str] | None,
    with_legend: bool, xlabel: str = "cutoff round", show_title: bool = True,
    integer_xticks: bool = False, legend_fontsize: float = FONTSIZE - 6,
) -> None:
    """Draw one metric's trajectory onto `ax`: surfaced theories as SEM
    error-bar points, seeds as dashed reference lines. Shared by the combined
    1x3 figure and the standalone per-metric figures.

    `xlabel`/`show_title`/`integer_xticks`/`legend_fontsize` let the standalone
    publication figures relabel the x-axis ("cycles"), drop the per-panel
    title, pin integer cycle ticks (no 1.5 etc.), and enlarge the legend, while
    the combined figure keeps the diagnostic defaults."""
    from matplotlib.lines import Line2D

    sub = traj_df[traj_df["metric"] == metric]
    surfaced = sub[sub["role"] != "seed"]
    seeds = sub[sub["role"] == "seed"]

    if theory_color is None:
        ax.errorbar(
            surfaced["cutoff_round"], surfaced["mean_across_exps"],
            yerr=surfaced["sem_across_exps"],
            fmt="o", ms=8, color=ROLE_COLOR["surfaced"],
            linestyle="none", label="surfaced",
        )
    else:
        for label, gm in surfaced.groupby("model"):
            ax.errorbar(
                gm["cutoff_round"], gm["mean_across_exps"],
                yerr=gm["sem_across_exps"], fmt="o", ms=8,
                color=_surfaced_traj_color(label, theory_color),
                linestyle="none",
            )
    for label, gs in seeds.groupby("model"):
        gs = gs.sort_values("cutoff_round")
        color = _seed_traj_color(label, theory_color)
        if seeds_as_line:
            # Cumulative view: a seed's fit changes as the eval set
            # grows, so draw the actual per-round line, not a flat mean.
            ax.plot(
                gs["cutoff_round"], gs["mean_across_exps"],
                ls="--", lw=3, marker=".", color=color,
                label=f"seed {label}",
            )
        else:
            y = float(gs["mean_across_exps"].mean())
            ax.axhline(
                y, ls="--", lw=3, color=color, label=f"seed {label}",
            )
    style_axes(
        ax, xlabel=xlabel, ylabel=nice, title=nice if show_title else None,
    )
    if integer_xticks:
        # Cycle counts are integers; pin ticks to the actual cycles so the
        # auto-locator can't add fractional ticks (1.0, 1.5, ...).
        xs = sorted({int(v) for v in sub["cutoff_round"]})
        ax.set_xticks(xs)
    if with_legend:
        if theory_color is None:
            ax.legend(frameon=False, fontsize=legend_fontsize, loc="best")
        else:
            # Role swatches matching the convergence palette.
            handles = [
                Line2D([0], [0], color=ROLE_COLOR["seed"], lw=3, ls="--",
                       label="seeds"),
                Line2D([0], [0], color=NEUTRAL_HARMONY["sandy"], lw=0,
                       marker="o", ms=8, label="surfaced (headline)"),
                Line2D([0], [0], color=NEUTRAL_HARMONY["terracotta"], lw=0,
                       marker="o", ms=8, label="surfaced (other final)"),
                Line2D([0], [0], color=GRAY, lw=0, marker="o", ms=8,
                       label="surfaced (killed)"),
            ]
            ax.legend(handles=handles, frameon=False,
                      fontsize=legend_fontsize, loc="best")


def plot_metric_trajectory_per_metric(
    traj_df: pd.DataFrame, out_dir: Path, prefix: str, *, target: str,
    seeds_as_line: bool = False, ground_truth_label: str = "human",
    theory_color: dict[str, str] | None = None,
) -> list[Path]:
    """Write each metric (MAE / MSE / Pearson r) as its own standalone
    single-panel figure under `{out_dir}/{prefix}_{metric}.png`. Each panel
    uses the calibration scatter's 4x4 canvas and FONTSIZE so axis/tick fonts
    render at the same size; the x-axis is relabelled "cycles" with integer
    ticks, the per-panel title is dropped, and the legend is enlarged for a
    publication-ready figure. Returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for metric, nice in TRAJ_METRICS:
        fig, ax = plt.subplots(figsize=(4, 4))
        # Only MAE carries the legend; MSE and Pearson r stay clean.
        _draw_metric_trajectory_panel(
            ax, traj_df, metric, nice,
            seeds_as_line=seeds_as_line, theory_color=theory_color,
            with_legend=metric == "mae", xlabel="cycles", show_title=False,
            integer_xticks=True, legend_fontsize=FONTSIZE - 2,
        )
        fig.tight_layout()
        path = out_dir / f"{prefix}_{metric}.png"
        save_figure(fig, path)
        plt.close(fig)
        written.append(path)
    return written


def plot_metric_trajectory(
    traj_df: pd.DataFrame, path: Path, *, target: str,
    seeds_as_line: bool = False, ground_truth_label: str = "human",
    theory_color: dict[str, str] | None = None,
) -> None:
    """3 subplots (MAE / MSE / Pearson r). x = cutoff round. Surfaced
    theories are scattered points with SEM error
    bars; seed theories are drawn as flat horizontal reference lines at
    their mean across cutoffs.

    When `theory_color` (`{pi_label: colour}`) is given, every theory is
    coloured by it so the figure matches the convergence plot's scheme
    (seeds -> slate, headline survivor -> sandy, other survivor ->
    terracotta, transient -> gray). Without it, the original two-tone seed
    colours and single-colour surfaced points are kept."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), squeeze=False)
    for i, (metric, nice) in enumerate(TRAJ_METRICS):
        _draw_metric_trajectory_panel(
            axes[0][i], traj_df, metric, nice,
            seeds_as_line=seeds_as_line, theory_color=theory_color,
            with_legend=i == 2,
        )

    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Trajectory of surfaced-theory fit to human P(B) across round "
            "cutoffs."
        ),
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--rounds", type=int, nargs="+", default=list(CUTOFFS_DEFAULT),
        help="Round cutoffs (1-indexed); cutoff N reads round_{N-1}.",
    )
    p.add_argument("--human-data", type=Path, default=HUMAN_DATA_DEFAULT)
    p.add_argument("--n-expected", type=int, default=10)
    p.add_argument("--n-subjects", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default="p_b_pooled",
                   choices=["p_b_pooled", "p_b_within_subj_mean"])
    p.add_argument(
        "--match-participants", action="store_true",
        help=(
            "Replay each stimulus pair with as many model draws as it had "
            "human participants, so model SEM is comparable to the human "
            "across-subject SEM. Outputs get a `_matched` suffix so they "
            "do not clobber the default (--n-subjects) run."
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    out = args.out or (args.run_dir / "analysis" / "eval_human_trajectory")
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_matched" if args.match_participants else ""

    experiments = gather_experiments(
        args.human_data, n_expected=args.n_expected,
    )
    seeds = resolve_base_theories(args.run_dir)
    print(
        f"[trajectory] run={args.run_dir.name} cutoffs={args.rounds} "
        f"seeds={list(seeds)} n_experiments={len(experiments)} "
        f"target={args.target}"
    )

    summary_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []

    # Seeds: evaluated once at the first cutoff (round-invariant) but
    # recorded under every cutoff so the trajectory has a value at each x.
    seed_summ, _ = eval_cutoff(
        args.rounds[0], seeds, "seed", experiments,
        n_subjects=args.n_subjects, seed=args.seed, target=args.target,
        match_participants=args.match_participants,
    )
    for c in args.rounds:
        for r in seed_summ:
            summary_rows.append({**r, "cutoff_round": c})

    for c in args.rounds:
        survivors = resolve_surfaced_theories(args.run_dir, round_idx=c - 1)
        roles = _surfaced_slot_roles(args.run_dir, c - 1)
        # Split into survivor / replacement so the scatter grid rows are
        # stable; metrics are role-agnostic ("surfaced").
        for label, theory in survivors.items():
            slot = roles.get(label, "survivor")
            s_summ, s_scat = eval_cutoff(
                c, {label: theory}, "surfaced", experiments,
                n_subjects=args.n_subjects, seed=args.seed,
                target=args.target,
                match_participants=args.match_participants,
            )
            summary_rows.extend(s_summ)
            # Re-tag scatter rows with the slot role for grid placement.
            for sr in s_scat:
                sr["role"] = slot
            scatter_rows.extend(s_scat)
        print(f"[trajectory] cutoff {c}: survivors={list(survivors)}")

    per_exp_df = pd.DataFrame(summary_rows)
    scatter_df = pd.DataFrame(scatter_rows)
    traj_df = aggregate_trajectory(per_exp_df)

    per_exp_df.to_csv(out / f"per_experiment_summary{suffix}.csv", index=False)
    traj_df.to_csv(out / f"trajectory{suffix}.csv", index=False)
    scatter_df.to_csv(out / f"calibration_points{suffix}.csv", index=False)

    try:
        plot_calibration_grid(
            scatter_df, out / f"calibration_scatter{suffix}.png",
            cutoffs=args.rounds,
        )
        plot_metric_trajectory(
            traj_df, out / f"trajectory_{args.target}{suffix}.png",
            target=args.target,
        )
    except ImportError as e:
        print(f"[trajectory] WARNING: plots skipped ({e}).", file=sys.stderr)

    print(traj_df.to_string(index=False))
    mode = "participant-matched" if args.match_participants else "n_subjects"
    print(f"[trajectory] wrote {mode} outputs under {out}/ (suffix={suffix!r})")
    return 0


def _surfaced_slot_roles(run_dir: Path, round_idx: int) -> dict[str, str]:
    """Map each surfaced label at `round_idx` to its slot: un-killed
    starting theories are 'survivor', the replacement is 'replacement'."""
    rd = run_dir / "rounds" / f"round_{round_idx:03d}" / "theories.json"
    data = json.loads(rd.read_text())
    roles: dict[str, str] = {}
    for s in data.get("starting_theories", []):
        if not s.get("killed", False):
            roles[s["label"]] = "survivor"
    repl = data.get("replacement")
    if repl is not None:
        roles[repl["label"]] = "replacement"
    return roles


if __name__ == "__main__":
    raise SystemExit(main())
