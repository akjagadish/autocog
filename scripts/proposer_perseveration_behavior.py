"""Behavioral recovery metrics for the perseveration GT, where every static
per-pair metric is degenerate (r NaN by definition; MSE chance ~= ceiling).

Per run (non-oracle selection via scripts/proposer_selection.py):
  * switch_rate of the capped-final best theory — GT = 0.0, coin flip = 0.5;
  * lock_split — fraction of simulated subjects locked onto option B; true
    perseveration ~0.5 (random first choice), a constant-side model 0 or 1;
  * per-round switch-rate trajectory + first round behaviorally "good"
    (switch_rate <= GOOD_SWITCH).

Determinism note: simulate_sequence's trial order is unseeded upstream, so
values carry small jitter; the quantities reported (switch rate ~0 vs ~0.5,
split ~0.5) are far from decision boundaries.

Usage: PYTHONPATH=. python scripts/proposer_perseveration_behavior.py
Writes results/controls/proposer_comparison/perseveration_behavior.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_hilbig import HUMAN_VALIDITIES  # noqa: E402
from scripts.proposer_selection import best_surviving_at  # noqa: E402
from scripts.recovery_correlation import (  # noqa: E402
    simulate_sequence,
    theory_reads_history,
    unique_stimulus_pairs,
)
from src.theory import Theory  # noqa: E402

CAP = 5
N_RUNS = 100
GOOD_SWITCH = 0.05  # GT = 0.0, coin flip = 0.5
GT = "perseveration"
THEORIES_DIR = Path("theories/heuristic_decision_making")
NP_ROOT = Path("results/controls/ablation_stage0/condition_neutral_proposer")
ADV_ROOT = Path("results/recovery")
OUT = Path("results/controls/proposer_comparison/perseveration_behavior.json")


def run_dir_for(cond: str, rid: str) -> Path:
    root, tag = (NP_ROOT, "run") if cond == "neutral" else (ADV_ROOT, "")
    return root / GT / "noise=0.0" / (
        f"dmb_ground_truth_{GT}_noise=0.0_gemini-3.1-pro-preview_{tag}{rid}"
    )


def behavior(theory: Theory, seed: int) -> tuple[float, float]:
    """(mean switch rate, fraction of subjects locked onto B)."""
    _, seqs = simulate_sequence(
        theory, stimulus_pairs=unique_stimulus_pairs(),
        validities=HUMAN_VALIDITIES, n_runs=N_RUNS, seed=seed, action_noise=0.0,
    )
    rates = [float(np.mean(np.abs(np.diff(np.asarray(s))))) for s in seqs if len(s) > 1]
    locked_b = [1.0 if float(np.mean(s)) > 0.5 else 0.0 for s in seqs]
    return float(np.mean(rates)), float(np.mean(locked_b))


def main() -> None:
    gt_theory = Theory.from_yaml(THEORIES_DIR / f"{GT}.yaml")
    gt_switch, gt_split = behavior(gt_theory, seed=901)
    results = {"gt_switch_rate": round(gt_switch, 4),
               "gt_lock_split": round(gt_split, 4),
               "good_switch_threshold": GOOD_SWITCH,
               "runs": []}
    for cond, ids in (("neutral", ["full1", "full2", "full3"]),
                      ("adversarial", ["run1", "run2", "run3"])):
        for rid in ids:
            run_dir = run_dir_for(cond, rid)
            traj = []
            for r in range(CAP):
                _, theory = best_surviving_at(run_dir, r)
                sr, _ = behavior(theory, seed=301)
                traj.append(round(sr, 3))
            best, final_theory = best_surviving_at(run_dir, CAP - 1)
            sr, split = behavior(final_theory, seed=301)
            good = next((r for r, v in enumerate(traj) if v <= GOOD_SWITCH), None)
            results["runs"].append(dict(
                cond=cond, rid=rid, best=best,
                reads_history=theory_reads_history(final_theory),
                switch_rate=round(sr, 4), lock_split=round(split, 4),
                behaviorally_recovered=bool(sr <= GOOD_SWITCH and 0.2 <= split <= 0.8),
                switch_traj=traj, rounds_to_good=good,
            ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"WROTE {len(results['runs'])} run records -> {OUT}")


if __name__ == "__main__":
    main()
