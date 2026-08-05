#!/usr/bin/env bash
#
# Binary decision-making — ONLINE run on PROLIFIC (recruitment) + Firebase (data).
#
# This CREATES A REAL, PAID PROLIFIC STUDY and auto-recruits participants:
# 2 * REAL_N_SUBJECTS people per round. Payment is auto-computed upstream
# (~$12/hour from --online_duration); there is no reward flag on this runner.
#
# Requires prolific_settings.json (holds your Prolific API token) in the repo
# root — already present by default. COMPLETION_CODE must match the completion
# code configured for your Prolific study.
#
# Study URL participants are sent to (the hosted app):
#   https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app

set -euo pipefail
cd "$(dirname "$0")"

# ---- edit these ------------------------------------------------------------
RUN_ID="prolific1"        # a fresh run writes results/online/dmb/<backend>/$RUN_ID/<seeds>/
                          # (the committed copy of this run was later moved to
                          #  results/human_decision_making_binary/<seeds>/)
SEEDS=(ttb wadd)          # two distinct seeds from: ttb ew tallying wadd
REAL_N_SUBJECTS=25        # PER experiment -> 2*this people recruited per round
N_ROUNDS=5               # adversarial rounds this invocation
ONLINE_DURATION=20        # minutes; also drives auto-computed Prolific reward
MIN_RT_MS=1500            # per-trial gate: A/B prompt hidden this long each trial
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
COMPLETION_CODE="rxfks2"
# ----------------------------------------------------------------------------

if [[ "$COMPLETION_CODE" == "REPLACE_WITH_YOUR_PROLIFIC_COMPLETION_CODE" ]]; then
  echo "ERROR: set COMPLETION_CODE in this script before running a Prolific study." >&2
  exit 1
fi

uv run python main_decision_making_binary_online.py \
  --online_backend firebase_prolific \
  --seeds "${SEEDS[@]}" \
  --real_n_subjects "$REAL_N_SUBJECTS" \
  --n_rounds "$N_ROUNDS" \
  --online_duration "$ONLINE_DURATION" \
  --min_rt_ms "$MIN_RT_MS" \
  --study_url "$STUDY_URL" \
  --completion_code "$COMPLETION_CODE" \
  --run_id "$RUN_ID"
