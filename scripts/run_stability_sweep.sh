#!/bin/bash
# Run `metric_stability.py` across all 10 (round, slot) metrics from a run-dir
# for both `gcm` and `sustain` source models. Per-run CSV + PNG land in a
# timestamped sweep directory; a `combined.csv` stacks every per-replicate
# row across all 20 invocations.
#
# Overridable via environment variables, e.g.:
#   N_SAMPLING_RUNS=5 N_PARTICIPANTS=10,25 bash scripts/run_stability_sweep.sh
set -euo pipefail

RUN_DIR="${RUN_DIR:-results/ground_truth_rulex_gemini-3.1-pro-preview}"
MODELS=(${MODELS:-rulex}) #gcm sustain
N_PARTICIPANTS="${N_PARTICIPANTS:-10,25,50}"
N_SAMPLING_RUNS="${N_SAMPLING_RUNS:-15}"
ROUNDS=(0 1 2 3 4)
SLOTS=(1 2)

# Activate the project virtualenv (per CLAUDE.md).
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate

SWEEP_DIR="results/stability/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"

{
  echo "[sweep] start $(date)"
  echo "[sweep] run_dir=$RUN_DIR models=${MODELS[*]}"
  echo "[sweep] n_participants=$N_PARTICIPANTS n_sampling_runs=$N_SAMPLING_RUNS"
  echo "[sweep] output=$SWEEP_DIR"
} | tee -a "$LOG"

total=$(( ${#MODELS[@]} * ${#ROUNDS[@]} * ${#SLOTS[@]} ))
i=0
for model in "${MODELS[@]}"; do
  for round in "${ROUNDS[@]}"; do
    for slot in "${SLOTS[@]}"; do
      i=$(( i + 1 ))
      out_csv="$SWEEP_DIR/${model}__r${round}s${slot}.csv"
      echo "[sweep] ($i/$total) model=$model round=$round slot=$slot -> $out_csv" | tee -a "$LOG"
      python scripts/metric_stability.py \
        --run-dir "$RUN_DIR" \
        --round "$round" \
        --slot "$slot" \
        --model "$model" \
        --n-participants "$N_PARTICIPANTS" \
        --n-sampling-runs "$N_SAMPLING_RUNS" \
        --output "$out_csv" 2>&1 | tee -a "$LOG"
    done
  done
done

echo "[sweep] done $(date)" | tee -a "$LOG"

# Stack all per-run CSVs into a single combined frame.
python - "$SWEEP_DIR" <<'EOF' | tee -a "$LOG"
import sys
from pathlib import Path
import pandas as pd

sweep_dir = Path(sys.argv[1])
files = sorted(p for p in sweep_dir.glob("*.csv") if p.name != "combined.csv")
if not files:
    print("[sweep] no CSVs to combine")
    raise SystemExit(0)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
out = sweep_dir / "combined.csv"
df.to_csv(out, index=False)
print(f"[sweep] combined {len(files)} CSVs -> {out} ({len(df)} rows)")
EOF
