#!/usr/bin/env bash
#
# Analyse whatever data has been collected so far (no recruiting, no download).
#
#   ./run_analyse.sh            # both experiments
#   ./run_analyse.sh exp1       # just one
#   ./run_analyse.sh exp1 exp2
#
# Runs the preregistered confirmatory analysis on the current
# validation_online/data/<exp>/trials.csv. (Results are only meaningful once the
# sample is reasonably full; at small N this just shows the machinery + current
# direction.)

set -euo pipefail
cd "$(dirname "$0")"             # repo root

if [[ $# -gt 0 ]]; then EXPS=("$@"); else EXPS=(exp1 exp2); fi

for EXP in "${EXPS[@]}"; do
  CSV="validation_online/data/$EXP/trials.csv"
  echo "=================================================================="
  echo "  ANALYSE $EXP"
  echo "=================================================================="
  if [[ -f "$CSV" ]]; then
    uv run python validation_online/analyse.py --data "$CSV" --experiment "$EXP"
  else
    echo "[$EXP] no data yet ($CSV not found) — skipping."
  fi
  echo
done
