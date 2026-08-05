#!/usr/bin/env bash
# Stage-0 smoke: baseline / jsd_metric / neutral_proposer on ttb_sampling,
# n_rounds=5. One log per run; continues past failures. Check each log
# before trusting results.
#
# Usage:
#   ./run_ablation_smoke.sh                # real runs (gemini, billed)
#   LLM_PROVIDER=mock ./run_ablation_smoke.sh   # free harness check
set -u
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate
mkdir -p logs/ablation_smoke

GT="${GT:-ttb_sampling}"
N_ROUNDS="${N_ROUNDS:-5}"
LLM_PROVIDER="${LLM_PROVIDER:-gemini}"
LLM_MODEL="${LLM_MODEL:-gemini-3.1-pro-preview}"
OUT_DIR="${OUT_DIR:-results/heuristic_decision_making/ablation_stage0}"

for COND in baseline jsd_metric neutral_proposer; do
  LOG="logs/ablation_smoke/${COND}_${GT}.log"
  DIR="${OUT_DIR}/condition_${COND}/${GT}/noise=0.0/dmb_ground_truth_${GT}_noise=0.0_${LLM_MODEL}_runsmoke0"
  if [ -d "$DIR" ]; then
    echo "[${COND}] SKIP (exists: ${DIR})"
    continue
  fi
  echo "=== ${COND} (gt=${GT}, n_rounds=${N_ROUNDS}, llm=${LLM_MODEL}) -> ${LOG}"
  python -u main_ablation_binary.py \
    --condition "${COND}" \
    --ground_truth "${GT}" \
    --n_rounds "${N_ROUNDS}" \
    --llm_provider "${LLM_PROVIDER}" \
    --llm_model "${LLM_MODEL}" \
    --run_id "smoke0" \
    --out_path "${OUT_DIR}" \
    > "${LOG}" 2>&1 \
    || echo "!!! ${COND} FAILED — see ${LOG}"
done

echo "=== compute summary"
python scripts/summarize_compute.py "${OUT_DIR}"/condition_*/"${GT}"/noise=0.0/*runsmoke0
