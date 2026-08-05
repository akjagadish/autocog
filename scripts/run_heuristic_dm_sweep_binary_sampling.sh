#!/bin/bash
# Run N replications of the heuristic-decision-making held-out analysis for
# ONE model class. The other two classes are used as competitor seeds, so
# the adversarial pair has to rediscover the held-out class from scratch.
#
# Run once per ground truth — launch three terminals for parallelism:
#   bash scripts/run_heuristic_dm_sweep.sh ttb
#   bash scripts/run_heuristic_dm_sweep.sh tallying
#   bash scripts/run_heuristic_dm_sweep.sh wadd
#
# Skips any (run_id, gt_epsilon) whose results folder already exists.
# Per-run output lands in logs/hdm_<gt>[_gteps<eps>]_run<id>.log.
#
# Env overrides:
#   START_RUN=1 END_RUN=5 N_ROUNDS=5 LLM_MODEL=gemini-3.1-pro-preview
#   GT_EPSILONS="0.0 0.1 0.2 0.4"  — sweep policy-level action-noise on the
#     ground-truth simulator. Default "0.0" reproduces the prior noise-free
#     path (and the original, untagged results folder). GT_SEED (default 0)
#     seeds the noise RNG so the sweep is reproducible.

set -u

GROUND_TRUTH="${1:?usage: $0 <ground_truth: ttb_sampling|tallying_sampling|wadd_sampling>}"
START_RUN="${START_RUN:-1}"
END_RUN="${END_RUN:-3}"
N_ROUNDS="${N_ROUNDS:-5}"
LLM_PROVIDER="${LLM_PROVIDER:-gemini}"
LLM_MODEL="${LLM_MODEL:-gemini-3.1-pro-preview}"
read -r -a GT_EPSILONS <<< "${GT_EPSILONS:-0.0 0.5}"
GT_SEED="${GT_SEED:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-${REPO_ROOT}/results/recovery}"
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate
mkdir -p logs

echo "[${GROUND_TRUTH}] runs ${START_RUN}..${END_RUN}  n_rounds=${N_ROUNDS}  llm=${LLM_MODEL}  gt_epsilons=${GT_EPSILONS[*]}  gt_seed=${GT_SEED}"

for run_id in $(seq "$START_RUN" "$END_RUN"); do
  for eps in "${GT_EPSILONS[@]}"; do
    # Python always tags the run dir with noise=<eps> (even 0.0), see
    # main_decision_making_binary.py: `noise={gt_epsilon}` is emitted for
    # every gt_epsilon >= 0.0. Mirror that exactly so the skip check below
    # matches the on-disk path.
    eps_tag="noise=${eps}"
    eps_log="_noise=${eps}"
    dir="${out_dir}/${GROUND_TRUTH}/${eps_tag}/dmb_ground_truth_${GROUND_TRUTH}_${eps_tag}_${LLM_MODEL}_run${run_id}"
    log="logs/hdm_${GROUND_TRUTH}${eps_log}_run${run_id}.log"

    if [ -d "$dir" ]; then
      echo "[${GROUND_TRUTH} eps=${eps}] run${run_id}: SKIP (exists)"
      continue
    fi

    echo "[${GROUND_TRUTH} eps=${eps}] run${run_id}: starting -> ${log}"
    python -u main_decision_making_binary.py \
      --ground_truth "$GROUND_TRUTH" \
      --run_id "$run_id" \
      --n_rounds "$N_ROUNDS" \
      --llm_provider "$LLM_PROVIDER" \
      --llm_model "$LLM_MODEL" \
      --gt_epsilon "$eps" \
      --gt_seed "$GT_SEED" \
      --out_path  "$out_dir" \
      > "$log" 2>&1

    if [ $? -eq 0 ]; then
      echo "[${GROUND_TRUTH} eps=${eps}] run${run_id}: ok"
    else
      echo "[${GROUND_TRUTH} eps=${eps}] run${run_id}: FAILED (see ${log})"
    fi
  done
done

echo "[${GROUND_TRUTH}] done"
