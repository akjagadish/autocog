#!/usr/bin/env bash
#
# Binary decision-making — ONLINE run on Firebase only (no Prolific recruitment).
#
# Recruits nobody automatically: only people you send the study link to can
# participate. A round proposes 2 experiments, and REAL_N_SUBJECTS is PER
# experiment, so total participants = 2 * REAL_N_SUBJECTS (default 2 -> 4 people).
#
# Participants open this URL (one fresh session each — e.g. separate incognito
# windows). Each session pulls one of the uploaded conditions:
#
#   https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app
#
# WARNING: writes to (and resets) the `autora` collection in the Firestore
# project `autograd-4bbea` — do NOT run while a real study is live there.

set -euo pipefail
cd "$(dirname "$0")"

# ---- edit these ------------------------------------------------------------
RUN_ID="fbtest1"          # names results/human_decision_making_binary/<seeds>/
SEEDS=(ttb wadd)          # two distinct seeds from: ttb ew tallying wadd
REAL_N_SUBJECTS=2         # PER experiment -> 2*this = total people (2 -> 4)
N_ROUNDS=1               # adversarial rounds this invocation
ONLINE_DURATION=20        # minutes to wait for all completions before timing out
MIN_RT_MS=1500            # per-trial gate: A/B prompt hidden this long each trial
# ----------------------------------------------------------------------------

echo "Participants open: https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
echo "Expect $((2 * REAL_N_SUBJECTS)) total people (open that many incognito windows)."
echo

uv run python main_decision_making_binary_online.py \
  --online_backend firebase \
  --seeds "${SEEDS[@]}" \
  --real_n_subjects "$REAL_N_SUBJECTS" \
  --n_rounds "$N_ROUNDS" \
  --online_duration "$ONLINE_DURATION" \
  --min_rt_ms "$MIN_RT_MS" \
  --run_id "$RUN_ID"
