#!/usr/bin/env bash
# Score blind_design recovery against the three ground truths, after
# run_blind_design.sh finishes. Writes a CSV + Pearson-r / MSE / autocorr plots
# under results/condition_blind_design/. Pearson r and MSE are computed on the
# Hilbig task vs the noise-less ground truth (scripts/recovery_correlation.py).
set -u
source /Users/aj9225/Local/autograd/.autograd-gecco/bin/activate

python scripts/recovery_correlation.py \
    --results-root results/condition_blind_design \
    --families ttb_sampling take_the_worst perseveration \
    --csv results/condition_blind_design/recovery_correlation.csv \
    --out results/condition_blind_design/recovery_correlation.png \
    --mse-out results/condition_blind_design/recovery_mse.png \
    --autocorr-out results/condition_blind_design/recovery_autocorr.png \
    --title "blind_design recovery (Hilbig) — Pearson r to noise-less ground truth"
