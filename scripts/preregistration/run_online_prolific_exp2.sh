#!/usr/bin/env bash
#
# Experiment 2 (value curvature: steep-vs-flat + level-shift) — ONLINE on
# PROLIFIC (recruitment) + Firebase (data).
#
# This CREATES A REAL, PAID PROLIFIC STUDY and auto-recruits the preregistered
# sample: 100 completers, 20 per validity vector (the n_per_vector frozen in
# battery.json). Payment is auto-computed upstream (~$12/hour from
# --duration). Requires prolific_settings.json (your Prolific API token) at the
# repo root — present by default.
#
# Study URL participants are sent to:
#   https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app
#
# After completion, build the dataset and analyse:
#   uv run python validation_online/download_data.py --experiment exp2
#   uv run python validation_online/analyse.py \
#       --data validation_online/data/exp2/trials.csv --experiment exp2

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (so uv + credential paths resolve)

# ---- edit these ------------------------------------------------------------
ONLINE_DURATION=7         # minutes; also drives the auto-computed Prolific reward
MIN_RT_MS=1500            # per-trial gate: A/B prompt hidden this long each trial
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
COMPLETION_CODE=""        # empty -> read from prolific_settings.json
# Collect a different N: add --per_vector N (5*N total) or --n_sessions N below.
# ----------------------------------------------------------------------------

ARGS=( --experiment exp2 --backend firebase_prolific
       --duration "$ONLINE_DURATION" --min_rt_ms "$MIN_RT_MS"
       --study_url "$STUDY_URL" )
[[ -n "$COMPLETION_CODE" ]] && ARGS+=( --completion_code "$COMPLETION_CODE" )

echo "Experiment 2 — Prolific run (100 completers, 20/vector). Creating a PAID study."
uv run python validation_online/run_experiment.py "${ARGS[@]}"
