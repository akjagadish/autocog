"""Mechanism-recovery scoring via sequence-aware Jensen-Shannon divergence.

Static choice-proportion correlation (`recovery_correlation.py`) scores a
theory by its per-trial MARGINAL, so it is blind to history: a history-
dependent ground truth (perseveration) and a static surrogate with the same
marginal look identical. This script instead scores each surfaced theory
against the ground truth with JSD on the Hilbig design, in two variants:

  * static   — per-trial Bernoulli marginal (matches the correlation metric's
               blindness; sanity reference)
  * sequence — per-trial lag-1 joint over (response_{t-1}, response_t); SEES
               history, so it scores history-dependent mechanisms

JSD is in nats (0..ln 2); LOWER = closer to the GT predictive distribution.
The `gt_floor` role (gt vs an independent gt sample) is the plug-in bias floor;
the `seed` role (the run's non-GT canonical seeds) is the no-recovery
reference. Both sides of every JSD use the same `n_runs`, so the upward plug-in
bias cancels in comparisons (see src/jsd.py).

Usage:
  python scripts/jsd_recovery.py            # scores blind_design vs baseline
  python scripts/jsd_recovery.py --n-runs 300 --seed 0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.decision_making_binary_features.experiment import DecisionMakingBinaryExperiment  # noqa: E402
from src.jsd import _per_trial_bernoulli, _per_trial_lag1, choice_matrix, jsd  # noqa: E402
from src.theory import Theory  # noqa: E402
from scripts.eval_hilbig import (  # noqa: E402
    CANONICAL_YAML_DIR,
    HUMAN_DATA_DEFAULT,
    HUMAN_VALIDITIES,
    resolve_base_theories,
    resolve_surfaced_theories,
)
from scripts.recovery_correlation import (  # noqa: E402
    GROUND_TRUTH_YAML,
    discover_run_dirs,
    unique_stimulus_pairs,
)

FAMILIES_DEFAULT = ("ttb_sampling", "take_the_worst", "perseveration")
BLIND_ROOT_DEFAULT = Path("results/condition_blind_design")
BASELINE_ROOT_DEFAULT = Path(
    "results/heuristic_decision_making/synthetic_corrected_theories_binary_sampling"
)


def build_eval_experiment(
    human_data: Path = HUMAN_DATA_DEFAULT,
) -> DecisionMakingBinaryExperiment:
    """One experiment over the unique Hilbig stimulus pairs at the human
    validities. A single instance is reused for every JSD so both theories
    share the same (history-bearing) trial order."""
    pairs = unique_stimulus_pairs(human_data)
    return DecisionMakingBinaryExperiment(
        validities=list(HUMAN_VALIDITIES),
        trial_a_ratings=[list(a) for a, _ in pairs],
        trial_b_ratings=[list(b) for _, b in pairs],
    )


def jsd_to_gt(
    theory: Theory, gt: Theory, experiment: DecisionMakingBinaryExperiment, *, n_runs: int
) -> dict[str, float]:
    """Static and sequence-aware JSD of `theory` to `gt` on `experiment`.

    Each theory's choice matrix is simulated once and both variants are derived
    from it, so static and sequence JSD see the same samples."""
    m_t = choice_matrix(theory, experiment, n_runs=n_runs)
    m_g = choice_matrix(gt, experiment, n_runs=n_runs)
    static = np.mean([
        jsd(a, b) for a, b in zip(_per_trial_bernoulli(m_t), _per_trial_bernoulli(m_g))
    ])
    sequence = np.mean([
        jsd(a, b) for a, b in zip(_per_trial_lag1(m_t), _per_trial_lag1(m_g))
    ])
    return {"static_jsd": float(static), "sequence_jsd": float(sequence)}


def score_condition(
    *,
    condition: str,
    results_root: Path,
    families: list[str],
    noises: list[float] | None,
    experiment: DecisionMakingBinaryExperiment,
    n_runs: int,
) -> list[dict]:
    """One row per (run, role, theory): JSD of that theory to the family's GT."""
    rows: list[dict] = []
    gt_cache: dict[str, Theory] = {}
    for family, noise, run_dir in discover_run_dirs(
        results_root, families=families, noises=noises
    ):
        if family not in gt_cache:
            gt_cache[family] = Theory.from_yaml(
                CANONICAL_YAML_DIR / f"{GROUND_TRUTH_YAML.get(family, family)}.yaml"
            )
        gt = gt_cache[family]

        def add(role: str, label: str, theory: Theory) -> None:
            d = jsd_to_gt(theory, gt, experiment, n_runs=n_runs)
            rows.append({
                "condition": condition, "family": family, "run_dir": run_dir.name,
                "role": role, "label": label, **d,
            })

        add("gt_floor", "gt", gt)  # gt vs independent gt sample = plug-in bias floor
        for label, theory in resolve_base_theories(run_dir).items():
            add("seed", label, theory)
        for label, theory in resolve_surfaced_theories(run_dir).items():
            add("surfaced", label, theory)
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, family): gt_floor, seed mean, surfaced mean, and
    surfaced-best (min per run, averaged over runs), for both JSD variants."""
    out: list[dict] = []
    for (cond, fam), sub in df.groupby(["condition", "family"]):
        row = {"condition": cond, "family": fam}
        for metric in ("static_jsd", "sequence_jsd"):
            row[f"{metric}_gt_floor"] = sub[sub.role == "gt_floor"][metric].mean()
            row[f"{metric}_seed"] = sub[sub.role == "seed"][metric].mean()
            surf = sub[sub.role == "surfaced"]
            row[f"{metric}_surfaced"] = surf[metric].mean()
            # best (min) surfaced theory per run, averaged across runs
            row[f"{metric}_surfaced_best"] = (
                surf.groupby("run_dir")[metric].min().mean()
            )
        out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blind-root", type=Path, default=BLIND_ROOT_DEFAULT)
    p.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT_DEFAULT)
    p.add_argument("--families", nargs="+", default=list(FAMILIES_DEFAULT))
    p.add_argument("--noises", nargs="*", type=float, default=[0.0])
    p.add_argument("--n-runs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=Path, default=BLIND_ROOT_DEFAULT / "jsd_recovery.csv")
    args = p.parse_args(argv)

    # Seed BOTH RNGs: policy draws use numpy, but Theory parameter sampling
    # uses the stdlib `random` module — seeding only numpy leaves the run
    # non-reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    experiment = build_eval_experiment()
    noises = args.noises if args.noises else None

    rows: list[dict] = []
    for condition, root in (
        ("blind_design", args.blind_root),
        ("baseline", args.baseline_root),
    ):
        print(f"[jsd_recovery] scoring {condition} under {root}")
        rows += score_condition(
            condition=condition, results_root=root, families=args.families,
            noises=noises, experiment=experiment, n_runs=args.n_runs,
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[jsd_recovery] no run-dirs found.", file=sys.stderr)
        return 1
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    summary = summarise(df)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== sequence-aware JSD to GT (lower = better mechanism recovery) ===")
    seq_cols = ["condition", "family", "sequence_jsd_gt_floor", "sequence_jsd_seed",
                "sequence_jsd_surfaced", "sequence_jsd_surfaced_best"]
    print(summary[seq_cols].to_string(index=False))
    print("\n=== static JSD to GT (history-blind; sanity reference) ===")
    stat_cols = ["condition", "family", "static_jsd_gt_floor", "static_jsd_seed",
                 "static_jsd_surfaced", "static_jsd_surfaced_best"]
    print(summary[stat_cols].to_string(index=False))
    print(f"\n[jsd_recovery] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
