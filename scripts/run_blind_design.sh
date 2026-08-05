#!/usr/bin/env bash
# blind_design on 3 ground truths x 3 runs each (9 runs), n_rounds=5.
# One log per run; continues past failures. Check each log before trusting results.
#
# GT naming: ttb_sampling (canonical) is used instead of "ttb" so the seeds are
# the OTHER canonical theories — passing "ttb" would seed with ttb_sampling
# (the GT's own family) and contaminate recovery. take_the_worst and
# perseveration are non-canonical GTs, seeded by ttb_sampling + tallying_sampling.
# design_seed = run index so the three runs draw INDEPENDENT random designs
# (the design RNG keys on (design_seed, slot_label, round); run_id alone does not
# vary it).
set -u
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate
mkdir -p logs/blind_design

ROUNDS=5
for GT in ttb_sampling take_the_worst perseveration; do
    for RUN in 0 1 2; do
        LOG="logs/blind_design/${GT}_run${RUN}.log"
        echo "=== blind_design gt=${GT} run=${RUN} (n_rounds=${ROUNDS}, design_seed=${RUN}) -> ${LOG}"
        python main_ablation_binary.py \
            --condition blind_design \
            --ground_truth "${GT}" \
            --n_rounds "${ROUNDS}" \
            --design_seed "${RUN}" \
            --run_id "blind${RUN}" \
            >"${LOG}" 2>&1 \
            || echo "!!! ${GT} run${RUN} FAILED — see ${LOG}"
    done
done

echo "=== all runs done; compute summary ==="
python scripts/summarize_compute.py results/controls/condition_blind_design/*/*/dmb_ground_truth_*blind* || true
