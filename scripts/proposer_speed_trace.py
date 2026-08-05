"""Per-round recovery trajectory for the neutral-vs-adversarial comparison:
at each round R (0..4), score the best-by-loss surviving theory in the pool
*as of round R* (non-oracle, scripts/proposer_selection.py), normalized to
  frac = (chance_mse - mse) / (chance_mse - ceiling_mse)
(=1 at the GT ceiling, 0 at the predict-0.5 chance baseline, <0 worse than
chance). Per run: the trajectory, rounds_to_good (first frac >= GOOD_FRAC),
and AUC (mean frac over the 5 rounds).

perseveration is excluded: its per-pair metric is degenerate (chance ~=
ceiling); use proposer_perseveration_behavior.py instead.

Usage: PYTHONPATH=. python scripts/proposer_speed_trace.py
Writes results/heuristic_decision_making/proposer_comparison/speed_trace.json
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_hilbig import HUMAN_RATING_MAX, HUMAN_VALIDITIES  # noqa: E402
from scripts.proposer_selection import best_surviving_at  # noqa: E402
from scripts.recovery_correlation import (  # noqa: E402
    _row_metrics,
    mse,
    simulate_choice_proportions,
    unique_stimulus_pairs,
)
from src.theory import Theory  # noqa: E402

CAP = 5
N_DRAWS = 100
GOOD_FRAC = 0.9  # threshold for "good"; full trajectories are reported so
                 # readers can re-threshold (sensitivity noted in review)
THEORIES_DIR = Path("theories/heuristic_decision_making")
NP_ROOT = Path("results/heuristic_decision_making/ablation_stage0/condition_neutral_proposer")
ADV_ROOT = Path("results/heuristic_decision_making/synthetic_corrected_theories_binary_sampling")
OUT = Path("results/heuristic_decision_making/proposer_comparison/speed_trace.json")

RUNS = {
    "ttb_sampling": ["full1", "full2", "full3"],
    "anti_majority": ["full1", "full2", "full3"],
    "take_the_worst": ["full1", "full2", "full3"],
}


def run_dir_for(gt: str, cond: str, rid: str) -> Path:
    root, tag = (NP_ROOT, "run") if cond == "neutral" else (ADV_ROOT, "")
    return root / gt / "noise=0.0" / (
        f"dmb_ground_truth_{gt}_noise=0.0_gemini-3.1-pro-preview_{tag}{rid}"
    )


def main() -> None:
    pairs = unique_stimulus_pairs()
    results = []
    for gt, neutral_ids in RUNS.items():
        gt_theory = Theory.from_yaml(THEORIES_DIR / f"{gt}.yaml")
        tgt = simulate_choice_proportions(
            gt_theory, stimulus_pairs=pairs, validities=HUMAN_VALIDITIES,
            rating_max=HUMAN_RATING_MAX, n_draws=N_DRAWS, seed=900_000,
            action_noise=0.0,
        )
        ceil = _row_metrics(gt_theory, tgt, stimulus_pairs=pairs,
                            n_draws=N_DRAWS, seed=700_000, action_noise=0.0)["mse"]
        chance = float(mse(np.full(len(tgt), 0.5), tgt))
        denom = chance - ceil

        for cond, ids in (("neutral", neutral_ids),
                          ("adversarial", ["run1", "run2", "run3"])):
            for rid in ids:
                run_dir = run_dir_for(gt, cond, rid)
                traj = []
                for r in range(CAP):
                    _, theory = best_surviving_at(run_dir, r)
                    m = _row_metrics(
                        theory, tgt, stimulus_pairs=pairs, n_draws=N_DRAWS,
                        seed=500_000 + (zlib.crc32(f"{gt}/{cond}/{rid}/{r}".encode()) % 1000),
                        action_noise=0.0,
                    )["mse"]
                    traj.append(round((chance - float(m)) / denom, 3))
                good = next((r for r, f in enumerate(traj) if f >= GOOD_FRAC), None)
                results.append(dict(gt=gt, cond=cond, rid=rid, traj=traj,
                                    rounds_to_good=good,
                                    auc=round(float(np.mean(traj)), 3)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"WROTE {len(results)} trajectories -> {OUT}")


if __name__ == "__main__":
    main()
