"""Run-agnostic self-comparison: score an autocog run's surfaced theories
against the run's OWN discovery behavior (the per-round JSONL observations),
for ANY run. Every autocog run shares the same on-disk layout
(rounds/round_NNN/, observations/data/round_NNN_obs_MM.jsonl,
observations/state.json), so nothing about run size is hardcoded.

Two views (generic naming):
  all         — cutoffs default to every round; theory@cutoff vs ALL
                experiments pooled; seeds flat.
  cumulative  — rounds default to every round; theory@R vs experiments from
                rounds 0..R (in-sample); seeds as per-round lines.

Each view writes a calibration scatter PNG + a MAE/MSE/r trajectory PNG plus
CSVs. Outputs under <run>/analysis/eval_run_self/; converted .txt under
<run>/observations/self_exp/. Self-contained: no dependency on
eval_centaur_self; reuses only the run-agnostic eval/plot infrastructure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_human import (  # noqa: E402
    resolve_base_theories,
    resolve_surfaced_theories,
)
from scripts.eval_human_trajectory import (  # noqa: E402
    _surfaced_slot_roles,
    aggregate_trajectory,
    eval_cutoff,
    plot_calibration_grid,
    plot_calibration_per_model,
    plot_metric_trajectory,
    plot_metric_trajectory_per_metric,
)
from scripts.jsonl_to_hilbig_exp import convert  # noqa: E402
from scripts.plot_autocog_convergence import theory_colors  # noqa: E402

GROUND_TRUTH_LABEL = "Centaur"


def detect_n_rounds(run_dir: Path) -> int:
    """Number of rounds in the run, from observations/state.json."""
    state = json.loads(
        (run_dir / "observations" / "state.json").read_text()
    )
    return len(state.get("rounds", []))


def resolve_rounds_cutoffs(
    run_dir: Path, rounds: list[int] | None, cutoffs: list[int] | None,
) -> tuple[list[int], list[int]]:
    """Fill omitted --rounds / --cutoffs from the run's round count:
    rounds default to 0..R-1, cutoffs default to 1..R."""
    R = detect_n_rounds(run_dir)
    out_rounds = rounds if rounds is not None else list(range(R))
    out_cutoffs = cutoffs if cutoffs is not None else list(range(1, R + 1))
    return out_rounds, out_cutoffs


def prepare_experiments(
    run_dir: Path, out_dir: Path | None = None,
) -> list[Path]:
    """Convert every observations/data/round_NNN_obs_MM.jsonl to a
    Hilbig-shaped .txt, pinning that obs's validities from state.json.
    Returns the sorted list of written paths."""
    out_dir = out_dir or (run_dir / "observations" / "self_exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(
        (run_dir / "observations" / "state.json").read_text()
    )
    data_dir = run_dir / "observations" / "data"
    paths: list[Path] = []
    for n, rnd in enumerate(state.get("rounds", [])):
        for m, obs in enumerate(rnd.get("observations", [])):
            jsonl = data_dir / f"round_{n:03d}_obs_{m:02d}.jsonl"
            if not jsonl.is_file():
                raise FileNotFoundError(f"Missing behavior file {jsonl!s}")
            validities = [float(v) for v in obs["experiment"]["validities"]]
            out = out_dir / f"round_{n:03d}_obs_{m:02d}.txt"
            convert([jsonl], out, validities)
            paths.append(out)
    return sorted(paths)


def experiments_up_to_round(exp_paths: list[Path], R: int) -> list[Path]:
    """Subset of `exp_paths` whose filename round index is <= R."""
    return [p for p in exp_paths if int(p.stem.split("_")[1]) <= R]


def _eval_surfaced(
    run_dir: Path, cutoff_label: int, resolve_idx: int,
    experiments: list[Path], *, n_subjects: int, seed: int, target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survivors = resolve_surfaced_theories(run_dir, round_idx=resolve_idx)
    roles = _surfaced_slot_roles(run_dir, resolve_idx)
    summ: list[dict[str, Any]] = []
    scat: list[dict[str, Any]] = []
    for label, theory in survivors.items():
        slot = roles.get(label, "survivor")
        s_summ, s_scat = eval_cutoff(
            cutoff_label, {label: theory}, "surfaced", experiments,
            n_subjects=n_subjects, seed=seed, target=target,
        )
        summ.extend(s_summ)
        for sr in s_scat:
            sr["role"] = slot
        scat.extend(s_scat)
    return summ, scat


def _convergence_theory_colors(run_dir: Path) -> dict[str, str] | None:
    """Per-theory colours matching the convergence figure, or None when the run
    lacks the artifacts to derive them (e.g. a Centaur self-eval run without
    run_meta.json). On None the trajectory keeps its default palette."""
    try:
        return theory_colors(run_dir)
    except (FileNotFoundError, IndexError, KeyError, StopIteration):
        return None


def _write_view(
    out: Path, suffix: str, per_exp_df: pd.DataFrame,
    scatter_df: pd.DataFrame, cutoffs: list[int], *, target: str,
    seeds_as_line: bool, title_note: str, run_dir: Path,
    ground_truth_label: str = GROUND_TRUTH_LABEL,
) -> None:
    traj_df = aggregate_trajectory(per_exp_df)
    theory_color = _convergence_theory_colors(run_dir)
    per_exp_df.to_csv(out / f"per_experiment_summary_{suffix}.csv", index=False)
    traj_df.to_csv(out / f"trajectory_{suffix}.csv", index=False)
    scatter_df.to_csv(out / f"calibration_points_{suffix}.csv", index=False)
    plot_calibration_grid(
        scatter_df, out / f"calibration_scatter_{suffix}.png", cutoffs=cutoffs,
        ground_truth_label=ground_truth_label, theory_color=theory_color,
        show_title=False,
    )
    plot_calibration_per_model(
        scatter_df, out, f"calibration_scatter_{suffix}", cutoffs=cutoffs,
        ground_truth_label=ground_truth_label, theory_color=theory_color,
        show_title=False,
    )
    plot_metric_trajectory(
        traj_df, out / f"trajectory_{suffix}.png", target=target,
        seeds_as_line=seeds_as_line, ground_truth_label=ground_truth_label,
        theory_color=theory_color,
    )
    plot_metric_trajectory_per_metric(
        traj_df, out, f"trajectory_{suffix}", target=target,
        seeds_as_line=seeds_as_line, ground_truth_label=ground_truth_label,
        theory_color=theory_color,
    )
    print(f"[run_self] wrote {suffix} view ({title_note}) under {out}/")


def run_all(
    run_dir: Path, experiments: list[Path], cutoffs: list[int],
    out: Path, *, n_subjects: int, seed: int, target: str,
    ground_truth_label: str = GROUND_TRUTH_LABEL,
) -> None:
    """theory@cutoff vs ALL experiments; seeds flat (eval set constant)."""
    seeds = resolve_base_theories(run_dir)
    summary_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    seed_summ, _ = eval_cutoff(
        cutoffs[0], seeds, "seed", experiments,
        n_subjects=n_subjects, seed=seed, target=target,
    )
    for c in cutoffs:
        for r in seed_summ:
            summary_rows.append({**r, "cutoff_round": c})
    for c in cutoffs:
        s_summ, s_scat = _eval_surfaced(
            run_dir, c, c - 1, experiments,
            n_subjects=n_subjects, seed=seed, target=target,
        )
        summary_rows.extend(s_summ)
        scatter_rows.extend(s_scat)
    _write_view(
        out, "all", pd.DataFrame(summary_rows), pd.DataFrame(scatter_rows),
        cutoffs, target=target, seeds_as_line=False,
        title_note="theory@cutoff vs all experiments", run_dir=run_dir,
        ground_truth_label=ground_truth_label,
    )


def run_cumulative(
    run_dir: Path, experiments: list[Path], rounds: list[int],
    out: Path, *, n_subjects: int, seed: int, target: str,
    ground_truth_label: str = GROUND_TRUTH_LABEL,
) -> None:
    """theory@R vs experiments from rounds 0..R (in-sample); seeds as lines."""
    seeds = resolve_base_theories(run_dir)
    summary_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    for R in rounds:
        cum = experiments_up_to_round(experiments, R)
        seed_summ, _ = eval_cutoff(
            R, seeds, "seed", cum,
            n_subjects=n_subjects, seed=seed, target=target,
        )
        summary_rows.extend(seed_summ)
        s_summ, s_scat = _eval_surfaced(
            run_dir, R, R, cum,
            n_subjects=n_subjects, seed=seed, target=target,
        )
        summary_rows.extend(s_summ)
        scatter_rows.extend(s_scat)
        print(f"[run_self] cumulative round {R}: {len(cum)} experiments")
    _write_view(
        out, "cumulative", pd.DataFrame(summary_rows),
        pd.DataFrame(scatter_rows), list(rounds), target=target,
        seeds_as_line=True, title_note="in-sample: theory@R vs rounds 0..R",
        run_dir=run_dir, ground_truth_label=ground_truth_label,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Self-comparison of an autocog run vs its own behavior.",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--cutoffs", type=int, nargs="+", default=None,
        help="all-view cutoffs (default: every round; cutoff N reads round N-1).",
    )
    p.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="cumulative-view rounds (default: every round, 0-indexed).",
    )
    p.add_argument("--exp-out", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-subjects", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default="p_b_pooled",
                   choices=["p_b_pooled", "p_b_within_subj_mean"])
    p.add_argument("--which", default="both",
                   choices=["all", "cumulative", "both"])
    p.add_argument(
        "--ground-truth-label", default=GROUND_TRUTH_LABEL,
        help=(
            "Label for the behavior source in plot titles/axes "
            f"(default {GROUND_TRUTH_LABEL!r}). Use e.g. 'Human' for a "
            "run whose discovery experiments were run on people."
        ),
    )
    args = p.parse_args(argv)

    out = args.out or (args.run_dir / "analysis" / "eval_run_self")
    out.mkdir(parents=True, exist_ok=True)
    rounds, cutoffs = resolve_rounds_cutoffs(
        args.run_dir, args.rounds, args.cutoffs,
    )

    experiments = prepare_experiments(args.run_dir, out_dir=args.exp_out)
    print(
        f"[run_self] run={args.run_dir.name} n_experiments={len(experiments)} "
        f"n_rounds={detect_n_rounds(args.run_dir)} cutoffs={cutoffs} "
        f"which={args.which}"
    )

    if args.which in ("all", "both"):
        run_all(
            args.run_dir, experiments, cutoffs, out,
            n_subjects=args.n_subjects, seed=args.seed, target=args.target,
            ground_truth_label=args.ground_truth_label,
        )
    if args.which in ("cumulative", "both"):
        run_cumulative(
            args.run_dir, experiments, rounds, out,
            n_subjects=args.n_subjects, seed=args.seed, target=args.target,
            ground_truth_label=args.ground_truth_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
