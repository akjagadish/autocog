#!/usr/bin/env bash
#
# Incremental BALANCED collection toward the preregistered sample.
#
# Each run recruits Prolific participants ONLY for the validity vectors still
# below their per-vector target (run_experiment --fill, reading the current
# counts from the accumulated trials.csv) and APPENDS the new completers into
# trials.csv (download_data --append, deduped by participant). So you can run it
# as many times as you like and it accumulates toward a balanced 50 (exp1) /
# 100 (exp2) without ever double-counting or over-recruiting a filled vector.
#
# Re-run until download_data prints "ALL CONDITIONS FILLED" for each experiment.
#
# BATCH = participants per experiment per run. Set BATCH="" to recruit ALL
# remaining for an experiment in a single study (finish in one go).
#
# Creates PAID Prolific studies. Requires prolific_settings.json at the repo root.

set -euo pipefail
cd "$(dirname "$0")"             # repo root

# ---- edit these ------------------------------------------------------------
BATCH=2                   # participants per experiment per run ("" = all remaining)
DURATION=7                # minutes (also drives Prolific reward)
MIN_RT_MS=1500
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
EXPERIMENTS=(exp1 exp2)   # set to (exp1) or (exp2) to collect just one
# ----------------------------------------------------------------------------

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=================================================================="
  echo "  COLLECT $EXP — fill under-target vectors, append to trials.csv"
  echo "=================================================================="
  args=( --experiment "$EXP" --backend firebase_prolific --fill
         --duration "$DURATION" --min_rt_ms "$MIN_RT_MS" --study_url "$STUDY_URL" )
  [[ -n "$BATCH" ]] && args+=( --n_sessions "$BATCH" )

  uv run python scripts/preregistration/run_experiment.py "${args[@]}"
  uv run python scripts/preregistration/download_data.py --experiment "$EXP" --append
done

echo
echo "Done. Re-run until every vector shows DONE / ALL CONDITIONS FILLED."
