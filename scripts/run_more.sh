#!/usr/bin/env bash
#
# Add more participants to an experiment, then append them to trials.csv.
#
#   ./run_more.sh <exp1|exp2> [N]      (N defaults to 5)
#   e.g.  ./run_more.sh exp1 5         # recruit 5 more for exp1
#         ./run_more.sh exp2           # recruit 5 more for exp2
#
# Recruits N Prolific participants for the validity vectors still under target
# (--fill, reading current counts from trials.csv), then merges them in
# (--append, deduped by participant). Re-run any time to keep topping up toward
# the balanced sample (exp1=50, exp2=100). Creates a PAID Prolific study.

set -euo pipefail
cd "$(dirname "$0")"             # repo root

EXP="${1:-}"
N="${2:-5}"
if [[ "$EXP" != "exp1" && "$EXP" != "exp2" ]]; then
  echo "usage: ./run_more.sh <exp1|exp2> [N]   (N defaults to 5)" >&2
  exit 1
fi

# ---- edit if needed --------------------------------------------------------
DURATION=7                # minutes (also drives Prolific reward)
MIN_RT_MS=1500
STUDY_URL="https://autograd-online-experiment--autograd-4bbea.us-east4.hosted.app"
# ----------------------------------------------------------------------------

echo ">>> $EXP: recruiting $N more participant(s) for under-target vectors"
uv run python scripts/preregistration/run_experiment.py \
  --experiment "$EXP" --backend firebase_prolific --fill --n_sessions "$N" \
  --duration "$DURATION" --min_rt_ms "$MIN_RT_MS" --study_url "$STUDY_URL"

echo ">>> $EXP: appending to trials.csv"
uv run python scripts/preregistration/download_data.py --experiment "$EXP" --append
