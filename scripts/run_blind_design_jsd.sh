#!/usr/bin/env bash
# blind_design_jsd on 3 ground truths x 3 runs each (9 runs), n_rounds=5.
# Random experimental design (no LLM, no adaptivity) PAIRED WITH the fixed
# lag-1 JSD-to-self metric (no metric-proposal LLM calls, no separation gate).
# Sibling of run_blind_design.sh: same GTs/seeds/rounds so the two are directly
# comparable (does the JSD metric help random design?), and comparable to
# jsd_metric (does LLM design help over random search at the SAME metric?).
#
# The three ground-truth theories are independent (separate run dirs / logs / no
# shared state), so each GT runs as its OWN background subshell (3 concurrent),
# with its 3 seed-runs sequential inside. Idempotent + re-runnable: a run that
# already has >= ROUNDS admitted rounds is SKIPPED; any partial run is wiped and
# restarted clean (--n_rounds is ADDITIVE, so resuming a complete run would
# double its rounds — restarting fresh is the safe, simple choice).
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
mkdir -p logs/blind_design_jsd

ROUNDS=5
LLM_MODEL="gemini-3.1-pro-preview"

run_gt() {
    local GT="$1"
    for RUN in 0 1 2; do
        local LOG="logs/blind_design_jsd/${GT}_run${RUN}.log"
        local DIR="results/controls/condition_blind_design_jsd/${GT}/noise=0.0/dmb_ground_truth_${GT}_noise=0.0_${LLM_MODEL}_runblindjsd${RUN}"
        # A finished round writes one "— post-admit" leaderboard entry; >= ROUNDS
        # of them means the run is complete.
        local done_rounds
        done_rounds=$(grep -cE "— post-admit" "${DIR}/leaderboard.md" 2>/dev/null || true)
        done_rounds=${done_rounds:-0}
        if [ "${done_rounds}" -ge "${ROUNDS}" ]; then
            echo "[${GT} run${RUN}] SKIP (already ${done_rounds} admitted rounds)"
            continue
        fi
        # Partial (or absent): start clean so --n_rounds is not added on top of a
        # half-finished pool. rm only ever touches an incomplete run dir.
        rm -rf "${DIR}"
        echo "=== blind_design_jsd gt=${GT} run=${RUN} (n_rounds=${ROUNDS}, design_seed=${RUN}) -> ${LOG}"
        python main_ablation_binary.py \
            --condition blind_design_jsd \
            --ground_truth "${GT}" \
            --n_rounds "${ROUNDS}" \
            --design_seed "${RUN}" \
            --run_id "blindjsd${RUN}" \
            >"${LOG}" 2>&1 \
            || echo "!!! ${GT} run${RUN} FAILED — see ${LOG}"
    done
}

# Launch the three ground-truth theories in parallel; wait for all to finish.
for GT in ttb_sampling take_the_worst perseveration; do
    run_gt "${GT}" &
done
wait

echo "=== all runs done; compute summary ==="
python scripts/summarize_compute.py results/controls/condition_blind_design_jsd/*/*/dmb_ground_truth_*blindjsd* || true
