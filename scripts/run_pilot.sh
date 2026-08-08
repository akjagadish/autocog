#!/usr/bin/env bash
#
# PILOT — 2 real Prolific participants for EACH experiment, then the full
# download + analyse, to check the whole pipeline before the full run.
# The experiment is chosen with --experiment exp1 / exp2.
#
# Creates TWO small PAID Prolific studies (2 people each = 4 total). Requires
# prolific_settings.json (API token + completion_code) in the repo root.
# At N=2 the analysis is just a plumbing check, not a meaningful result.

set -euo pipefail
cd "$(dirname "$0")"             # repo root

N=2                             # participants per experiment
DURATION=7                      # minutes (also drives Prolific reward)
MIN_RT_MS=1500
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"

for EXP in exp1; do
  echo "=================================================================="
  echo "  PILOT $EXP — recruiting $N Prolific participant(s)"
  echo "=================================================================="
  uv run python scripts/preregistration/run_experiment.py \
    --experiment "$EXP" \
    --backend firebase_prolific \
    --n_sessions "$N" \
    --duration "$DURATION" \
    --min_rt_ms "$MIN_RT_MS" \
    --study_url "$STUDY_URL"

  uv run python scripts/preregistration/download_data.py --experiment "$EXP"

  uv run python scripts/preregistration/analyse.py \
    --data "scripts/preregistration/data/$EXP/trials.csv" --experiment "$EXP"
done

echo "PILOT complete. See scripts/preregistration/data/exp1/ and data/exp2/."
