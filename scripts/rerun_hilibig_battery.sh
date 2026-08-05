#!/bin/bash
# Wipe and re-run the full hilibig_battery pipeline end-to-end.
#
# Steps:
#   1. Delete per-run analysis/hilibig_battery/ dirs
#   2. Delete the global aggregate CSV + plot
#   3. Re-run scripts/run_hilibig_battery_sweep.sh (the per-run sweep)
#   4. Re-aggregate (scripts/aggregate_hilibig_battery.py)
#   5. Re-plot (scripts/plot_hilibig_battery_aggregate.py)
#
# Env overrides (all optional):
#   N_SUBJECTS=50                   # subjects per run
#   FAMILIES="ttb wadd tallying"    # ground-truth families
#   NOISES=""                       # filter, e.g. "noise=0.0 noise=0.3". Empty = all found.
#   N_BLOCKS=1 SEED=0 ROUND_IDX=last

set -u

N_SUBJECTS="${N_SUBJECTS:-50}"
FAMILIES="${FAMILIES:-ttb wadd tallying}"
NOISES="${NOISES:-}"

source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate

# 1) Delete per-run analysis dirs (scoped to FAMILIES × NOISES).
echo "[rerun] wiping per-run analysis/hilibig_battery/ dirs..."
for family in $FAMILIES; do
  [ -d "results/${family}" ] || continue
  if [ -n "$NOISES" ]; then
    noise_list="$NOISES"
  else
    noise_list="$(cd "results/${family}" && ls -d noise=* 2>/dev/null || true)"
  fi
  for noise in $noise_list; do
    rm -rf results/${family}/${noise}/hdm_ground_truth_${family}_${noise}_*_run*/analysis/hilibig_battery
  done
done

# 2) Delete global aggregate CSV + plot.
echo "[rerun] wiping aggregate CSV + plot..."
rm -f results/hilibig_battery_aggregate.csv
rm -f results/hilibig_battery_base_vs_surfaced.png

# 3) Re-run the per-run sweep with the requested subject count.
echo "[rerun] running hilibig_battery sweep (N_SUBJECTS=${N_SUBJECTS})..."
N_SUBJECTS="$N_SUBJECTS" FAMILIES="$FAMILIES" NOISES="$NOISES" \
  bash scripts/run_hilibig_battery_sweep.sh

# 4) Re-aggregate all per-run outputs into the long CSV.
echo "[rerun] aggregating..."
python scripts/aggregate_hilibig_battery.py --families $FAMILIES

# 5) Re-plot.
echo "[rerun] plotting..."
python scripts/plot_hilibig_battery_aggregate.py
python scripts/plot_hilibig_battery_correlation.py

echo "[rerun] done."
