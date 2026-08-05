#!/usr/bin/env bash
#
# Experiment 2 (value curvature: steep-vs-flat + level-shift) — ONLINE on
# Firebase ONLY (no Prolific).
#
# Recruits nobody automatically: only people you send the link to participate.
# Builds N_SESSIONS sessions and waits for them to complete. Each session is the
# full 96-trial Experiment-2 task (~7 min). Open N_SESSIONS fresh/incognito
# windows (one per test participant); each pulls one uploaded condition.
#
# Participants open:
#   https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app
#
# WARNING: writes to (and resets) the `autora` collection in the Firestore
# project `autograd-4bbea` — do NOT run while a real study is live there.
#
# After completion, build the dataset and analyse:
#   uv run python validation_online/download_data.py --experiment exp2
#   uv run python validation_online/analyse.py \
#       --data validation_online/data/exp2/trials.csv --experiment exp2

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (so uv + credential paths resolve)

# ---- edit these ------------------------------------------------------------
N_SESSIONS=2              # number of test participants (each opens one link)
ONLINE_DURATION=20        # minutes to wait for all completions before timing out
MIN_RT_MS=1500            # per-trial gate: A/B prompt hidden this long each trial
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
# ----------------------------------------------------------------------------

echo "Experiment 2 (Firebase only). Participants open: $STUDY_URL"
echo "Open $N_SESSIONS fresh/incognito windows (one per test participant)."
echo

uv run python validation_online/run_experiment.py \
  --experiment exp2 \
  --backend firebase \
  --n_sessions "$N_SESSIONS" \
  --duration "$ONLINE_DURATION" \
  --min_rt_ms "$MIN_RT_MS" \
  --study_url "$STUDY_URL"
