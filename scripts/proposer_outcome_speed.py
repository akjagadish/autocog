"""M1 (best-by-loss recovery) + M3 (rounds-to-best) per run for the
neutral-vs-adversarial proposer comparison, capped at round 5.

Selection is non-oracle (scripts/proposer_selection.py): the surviving theory
the pipeline itself ranks best at the capped final round. Recovery uses
recovery_correlation's history-aware _row_metrics (history-reading theories
go through the real trial loop; memoryless ones keep the static path), with
ceiling (GT vs itself) and chance (predict 0.5) MSE reference scales.

Known metric limits (see results/.../summary.md): perseveration's static r is
NaN by definition (zero target variance) and its MSE scale is degenerate
(chance ~= ceiling); use proposer_perseveration_behavior.py for that GT.

Usage: PYTHONPATH=. python scripts/proposer_outcome_speed.py
Writes results/heuristic_decision_making/proposer_comparison/outcome_speed.json
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_hilbig import HUMAN_RATING_MAX, HUMAN_VALIDITIES  # noqa: E402
from scripts.proposer_selection import (  # noqa: E402
    best_surviving_at,
    rounds_to_first_appearance,
)
from scripts.recovery_correlation import (  # noqa: E402
    _row_metrics,
    mse,
    simulate_choice_proportions,
    simulate_sequence,
    theory_reads_history,
    unique_stimulus_pairs,
)
from src.theory import Theory  # noqa: E402

CAP = 5
LAST = CAP - 1
N_DRAWS = 100
THEORIES_DIR = Path("theories/heuristic_decision_making")
NP_ROOT = Path("results/heuristic_decision_making/ablation_stage0/condition_neutral_proposer")
ADV_ROOT = Path("results/heuristic_decision_making/synthetic_corrected_theories_binary_sampling")
OUT = Path("results/heuristic_decision_making/proposer_comparison/outcome_speed.json")

RUNS = {
    "ttb_sampling": (["full1", "full2", "full3"], ["run1", "run2", "run3"]),
    "anti_majority": (["full1", "full2", "full3"], ["run1", "run2", "run3"]),
    "take_the_worst": (["full1", "full2", "full3"], ["run1", "run2", "run3"]),
    "perseveration": (["full1", "full2", "full3"], ["run1", "run2", "run3"]),
}


def run_dir_for(gt: str, cond: str, rid: str) -> Path:
    root, tag = (NP_ROOT, "run") if cond == "neutral" else (ADV_ROOT, "")
    return root / gt / "noise=0.0" / (
        f"dmb_ground_truth_{gt}_noise=0.0_gemini-3.1-pro-preview_{tag}{rid}"
    )


def target_and_ceiling(gt_theory: Theory, pairs):
    """Noise-less GT per-pair proportions + the GT-vs-itself ceiling metrics
    (independent seed). History-reading GTs go through the real trial loop;
    note its trial order is unseeded upstream, so perseveration's static
    numbers carry irreducible jitter (documented limitation)."""
    if theory_reads_history(gt_theory):
        props, _ = simulate_sequence(
            gt_theory, stimulus_pairs=pairs, validities=HUMAN_VALIDITIES,
            n_runs=N_DRAWS, seed=900_000, action_noise=0.0,
        )
    else:
        props = simulate_choice_proportions(
            gt_theory, stimulus_pairs=pairs, validities=HUMAN_VALIDITIES,
            rating_max=HUMAN_RATING_MAX, n_draws=N_DRAWS, seed=900_000,
            action_noise=0.0,
        )
    ceil = _row_metrics(gt_theory, props, stimulus_pairs=pairs,
                        n_draws=N_DRAWS, seed=700_000, action_noise=0.0)
    return props, ceil


def main() -> None:
    pairs = unique_stimulus_pairs()
    results = []
    for gt, (neutral_ids, adv_ids) in RUNS.items():
        gt_theory = Theory.from_yaml(THEORIES_DIR / f"{gt}.yaml")
        target, ceil = target_and_ceiling(gt_theory, pairs)
        ceiling_mse = round(float(ceil["mse"]), 5)
        chance_mse = round(float(mse(np.full(len(target), 0.5), target)), 5)
        for cond, ids in (("neutral", neutral_ids), ("adversarial", adv_ids)):
            for rid in ids:
                run_dir = run_dir_for(gt, cond, rid)
                rec = dict(gt=gt, cond=cond, rid=rid,
                           gt_history=theory_reads_history(gt_theory),
                           ceiling_r=round(float(ceil["pearson_r"]), 4),
                           ceiling_mse=ceiling_mse, chance_mse=chance_mse)
                try:
                    best, theory = best_surviving_at(run_dir, LAST)
                    m = _row_metrics(
                        theory, target, stimulus_pairs=pairs, n_draws=N_DRAWS,
                        seed=300_000 + (zlib.crc32(f"{gt}/{cond}/{rid}".encode()) % 1000),
                        action_noise=0.0,
                    )
                    rec.update(
                        best=best,
                        reads_history=theory_reads_history(theory),
                        r=round(float(m["pearson_r"]), 4),
                        mse=round(float(m["mse"]), 5),
                        m3=rounds_to_first_appearance(run_dir, best, max_round=CAP),
                    )
                except Exception as e:  # noqa: BLE001 — record, don't abort grid
                    rec.update(error=f"{type(e).__name__}: {e}")
                results.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"WROTE {len(results)} run records -> {OUT}")


if __name__ == "__main__":
    main()
