#!/usr/bin/env bash
# Seed-with-ground-truth control (SOFT control, no code change to the binary).
#
# Idea: the binary seeds its two slots with the two canonical theories that are
# NOT the ground truth. We temporarily overwrite those two seed files so that
#   slot 1 = the ground-truth theory itself   (the "correct seed" under test)
#   slot 2 = tallying_sampling                (a fixed canonical adversary)
# run, then restore both. The GT's own simulation file is never the slot-1
# target (slot-1 competitor is always != GT), so the ground-truth data is
# unaffected. The GT-seeded slot keeps re-reading the swapped seed every round
# UNTIL the arbiter kills it — i.e. normal dynamics decide whether the correct
# seed survives (soft control, no forcing).
#
# Usage:
#   ./run_seed_gt_control.sh ttb_sampling            # real run (gemini, billed)
#   ./run_seed_gt_control.sh anti_majority
#   LLM_PROVIDER=mock ./run_seed_gt_control.sh cue_parity   # free harness check
set -euo pipefail
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate

GT="${1:?usage: run_seed_gt_control.sh <ground_truth>}"
N_ROUNDS="${N_ROUNDS:-5}"
LLM_PROVIDER="${LLM_PROVIDER:-gemini}"
LLM_MODEL="${LLM_MODEL:-gemini-3.1-pro-preview}"
RUN_ID="${RUN_ID:-seedgt0}"
OUT_DIR="${OUT_DIR:-results/heuristic_decision_making/seed_gt_control}"
THEORIES_DIR="theories/heuristic_decision_making"
ADVERSARY="tallying_sampling"   # fixed slot-2 seed for every run

# The binary's two competitor seeds = the canonical theories that are not GT, in
# this fixed order. SLOT1_FILE/SLOT2_FILE are the files it will read for slots
# 1 and 2; we overwrite their CONTENT (not the binary's choice of which files).
COMPETS=()
for t in ttb_sampling tallying_sampling wadd_sampling; do
  [ "$t" != "$GT" ] && COMPETS+=("$t")
done
SLOT1_FILE="${THEORIES_DIR}/${COMPETS[0]}.yaml"
SLOT2_FILE="${THEORIES_DIR}/${COMPETS[1]}.yaml"
GTY="${THEORIES_DIR}/${GT}.yaml"
ADVY="${THEORIES_DIR}/${ADVERSARY}.yaml"
[ -f "$GTY" ] || { echo "no such ground-truth yaml: $GTY"; exit 1; }

# Capture desired contents BEFORE any overwrite (the GT case has
# SLOT1_FILE == tallying_sampling.yaml, so the adversary source must be read
# first or it would be clobbered). Back up the two targets for restore.
TMP="$(mktemp -d)"
cp "$GTY"  "$TMP/slot1_src"
cp "$ADVY" "$TMP/slot2_src"
cp "$SLOT1_FILE" "$TMP/slot1_bak"
cp "$SLOT2_FILE" "$TMP/slot2_bak"
restore() {
  cp "$TMP/slot1_bak" "$SLOT1_FILE"
  cp "$TMP/slot2_bak" "$SLOT2_FILE"
  rm -rf "$TMP"
  echo "[restore] ${SLOT1_FILE}, ${SLOT2_FILE}"
}
trap restore EXIT INT TERM

cp "$TMP/slot1_src" "$SLOT1_FILE"   # slot 1 <= ground truth
cp "$TMP/slot2_src" "$SLOT2_FILE"   # slot 2 <= tallying_sampling
echo "[seed-gt] gt=${GT} | slot1(${SLOT1_FILE}) <= ${GT} | slot2(${SLOT2_FILE}) <= ${ADVERSARY}"

mkdir -p logs/seed_gt_control
LOG="logs/seed_gt_control/${GT}_${RUN_ID}.log"
echo "=== seed-gt control (gt=${GT}, n_rounds=${N_ROUNDS}, llm=${LLM_MODEL}) -> ${LOG}"
python -u main_ablation_binary.py \
  --condition baseline \
  --ground_truth "${GT}" \
  --n_rounds "${N_ROUNDS}" \
  --llm_provider "${LLM_PROVIDER}" \
  --llm_model "${LLM_MODEL}" \
  --run_id "${RUN_ID}" \
  --out_path "${OUT_DIR}" \
  2>&1 | tee "${LOG}"

# restore() runs via the EXIT trap.
