#!/usr/bin/env bash
# Score blind_design_jsd recovery against the three ground truths, after
# run_blind_design_jsd.sh finishes. Writes a CSV + Pearson-r / MSE / autocorr
# plots under results/controls/condition_blind_design_jsd/. Pearson r and MSE are computed
# on the Hilbig task vs the noise-less ground truth (scripts/recovery_correlation.py).
set -u
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate

python scripts/recovery_correlation.py \
    --results-root results/controls/condition_blind_design_jsd \
    --families ttb_sampling take_the_worst perseveration \
    --csv results/controls/condition_blind_design_jsd/recovery_correlation.csv \
    --out results/controls/condition_blind_design_jsd/recovery_correlation.png \
    --mse-out results/controls/condition_blind_design_jsd/recovery_mse.png \
    --autocorr-out results/controls/condition_blind_design_jsd/recovery_autocorr.png \
    --title "blind_design_jsd recovery (Hilbig) — Pearson r to noise-less ground truth"
