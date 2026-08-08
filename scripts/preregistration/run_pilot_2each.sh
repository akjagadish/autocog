#!/usr/bin/env bash
#
# PILOT — 2 REAL Prolific participants per experiment, then the FULL pipeline
# (download + analyse) for each, to confirm everything works end-to-end before
# the full preregistered run.
#
# This CREATES TWO SMALL, PAID PROLIFIC STUDIES (N participants each; default
# N=2, so 4 people total) and recruits real participants. Payment is
# auto-computed upstream (~$12/hour from DURATION). Requires
# prolific_settings.json (Prolific API token + completion_code) at the repo root.
#
# For each experiment it runs three stages, blocking until that experiment's N
# participants complete before downloading/analysing:
#   1. run_experiment.py  --> recruits N, collects, writes data/<exp>/raw + conditions
#   2. download_data.py   --> data/<exp>/trials.csv
#   3. analyse.py         --> the three confirmatory tests (NOTE: at N=2 these are
#                             just a plumbing check, not a meaningful result)
#
# Study URL participants are sent to:
#   https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app
#
# Re-run only the analysis later, without recruiting again:
#   uv run python scripts/preregistration/download_data.py --experiment exp1
#   uv run python scripts/preregistration/analyse.py --data scripts/preregistration/data/exp1/trials.csv --experiment exp1

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (so uv + credential paths resolve)

# ---- edit these ------------------------------------------------------------
N=2                       # participants per experiment for the pilot
DURATION=7                # minutes; also drives the auto-computed Prolific reward
MIN_RT_MS=1500            # per-trial gate: A/B prompt hidden this long each trial
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
COMPLETION_CODE=""        # empty -> read from prolific_settings.json
EXPERIMENTS=(exp1 exp2)   # set to (exp1) or (exp2) to pilot just one
# ----------------------------------------------------------------------------

run_one () {
  local exp="$1"
  echo
  echo "=================================================================="
  echo "  PILOT $exp — recruiting $N Prolific participant(s)"
  echo "=================================================================="
  local args=( --experiment "$exp" --backend firebase_prolific
               --n_sessions "$N" --duration "$DURATION"
               --min_rt_ms "$MIN_RT_MS" --study_url "$STUDY_URL" )
  [[ -n "$COMPLETION_CODE" ]] && args+=( --completion_code "$COMPLETION_CODE" )

  uv run python scripts/preregistration/run_experiment.py "${args[@]}"

  echo "--- $exp: download_data ---"
  uv run python scripts/preregistration/download_data.py --experiment "$exp"

  echo "--- $exp: analyse (N=$N is a plumbing check only) ---"
  uv run python scripts/preregistration/analyse.py \
    --data "scripts/preregistration/data/$exp/trials.csv" --experiment "$exp"
}

for exp in "${EXPERIMENTS[@]}"; do
  run_one "$exp"
done

echo
echo "PILOT complete. Artifacts: scripts/preregistration/data/exp1/ and data/exp2/"
echo "(trials.csv + raw_observations.json + conditions.json per experiment)."
