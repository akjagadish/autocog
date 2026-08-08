#!/usr/bin/env bash
# Smoke-test every analysis behind a result reported in the paper.
#
# This does NOT reproduce the paper's numbers — sample sizes are cut to the bone
# so the whole sweep finishes in a couple of minutes. It answers one question:
# can each analysis still find its inputs and run to completion on the committed
# `results/` tree? That is the check that rots when data is reorganised.
#
# Usage:
#   bash scripts/smoke_analyses.sh              # all groups
#   bash scripts/smoke_analyses.sh fig3         # one group (fig1|fig3|fig4|fig5|si|loop)
#   KEEP=1 bash scripts/smoke_analyses.sh       # keep the output dir for inspection
#
# Exit status is non-zero if any analysis fails.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
OUT="${OUT:-$(mktemp -d)}"
WANT="${1:-all}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Non-destructive guard.
#
# The committed `results/` tree IS the paper's record, so this script must leave
# it byte-identical. Every analysis below writes to $OUT where it accepts an
# output flag, but a few (stats.py, the proposer_* controls, jsd_recovery.py,
# plot_autocog_convergence.py, plot_eval_hilbig_figures.py) hardcode an output
# path inside results/. Rather than skip those, we snapshot git's view of
# results/ up front and restore exactly what this run touched on exit.
#
# Only paths that changed DURING this run are reverted: anything already dirty
# before we started is left alone.
# ---------------------------------------------------------------------------
SNAP_BEFORE="$(mktemp -t autocog_smoke_before.XXXXXX)"
git status --porcelain results/ 2>/dev/null | sort > "$SNAP_BEFORE" || : > "$SNAP_BEFORE"

restore_results() {
  local after
  after="$(mktemp -t autocog_smoke_after.XXXXXX)"
  git status --porcelain results/ 2>/dev/null | sort > "$after" || return 0
  local changed
  changed=$(comm -13 "$SNAP_BEFORE" "$after")
  rm -f "$after"
  [ -z "$changed" ] && { echo "results/ untouched (verified)"; rm -f "$SNAP_BEFORE"; return 0; }
  echo "restoring results/ files this run touched:"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local code path
    code="${line:0:2}"; path="${line:3}"
    path="${path%\"}"; path="${path#\"}"
    case "$code" in
      "??") rm -rf "$path"          && echo "  removed  $path" ;;
      *)    git checkout -- "$path" && echo "  reverted $path" ;;
    esac
  done <<< "$changed"
  rm -f "$SNAP_BEFORE"
}
trap restore_results EXIT


# Canonical inputs from the committed results tree.
RECOVERY="results/recovery"
BINARY_HUMAN="results/human_decision_making_binary/ttb+wadd"
CARDINAL_HUMAN="results/human_decision_making_cardinal/ttb+tallying"
PREREG_BUNDLE="$CARDINAL_HUMAN/preregistration_visualization"
TTB_RUN="$RECOVERY/ttb_sampling/noise=0.0/dmb_ground_truth_ttb_sampling_noise=0.0_gemini-3.1-pro-preview_run1"
WADD_RUN3="$RECOVERY/wadd_sampling/noise=0.0/dmb_ground_truth_wadd_sampling_noise=0.0_gemini-3.1-pro-preview_run3"
CANON3=(ttb_sampling wadd_sampling tallying_sampling)

PASS=0; FAIL=0; SKIP=0; FAILED_NAMES=(); SKIPPED_NAMES=()

group() { CURRENT_GROUP="$1"; }
want()  { [ "$WANT" = "all" ] || [ "$WANT" = "$CURRENT_GROUP" ]; }

# run <label> -- <command...>
run() {
  want || return 0
  local label="$1"; shift; [ "$1" = "--" ] && shift
  local log="$OUT/${label//\//_}.log"
  if "$@" >"$log" 2>&1; then
    printf '  OK    %s\n' "$label"; PASS=$((PASS+1))
  else
    printf '  FAIL  %s\n' "$label"
    sed -n '$p' "$log" | sed 's/^/          /'
    printf '        (full log: %s)\n' "$log"
    FAIL=$((FAIL+1)); FAILED_NAMES+=("$label")
  fi
}

# skip <label> <reason> -- for analyses that cannot run here (input data not
# shipped, or a real API key required).
# Recorded explicitly so a gap never reads as coverage.
skip() {
  want || return 0
  printf '  SKIP  %s\n          reason: %s\n' "$1" "$2"
  SKIP=$((SKIP+1)); SKIPPED_NAMES+=("$1")
}

echo "smoke: repo=$REPO_ROOT out=$OUT group=$WANT"

# ---------------------------------------------------------------- Figure 1 ---
group fig1
want && echo "[Figure 1] task illustrations"
run fig1/binary_task_trial      -- $PY scripts/binary_task_trial_figure.py     --out "$OUT/f1_binary"
run fig1/non_binary_task_trial  -- $PY scripts/non_binary_task_trial_figure.py --out "$OUT/f1_cardinal"

# ---------------------------------------------------------------- Figure 3 ---
# 3A canonical recovery, 3B LLM-judge mechanism similarity, 3C non-canonical.
group fig3
want && echo "[Figure 3] ground-truth recovery + mechanism similarity"
run fig3A/recovery_correlation -- $PY scripts/recovery_correlation.py \
      --results-root "$RECOVERY" --families "${CANON3[@]}" --noises 0.0 0.5 \
      --n-draws 3 --seed 0 \
      --csv "$OUT/f3a.csv" --out "$OUT/f3a.png" \
      --mse-out "$OUT/f3a_mse.png" --autocorr-out "$OUT/f3a_ac.png"
run fig3A/per_model            -- $PY scripts/plot_recovery_per_model.py \
      --results-root "$RECOVERY" --families "${CANON3[@]}" --noises 0.0 \
      --n-draws 3 --seed 0 --out-dir "$OUT/f3a_per_model"
run fig3A/vs_clean_gt          -- $PY scripts/plot_recovery_vs_clean_gt.py \
      --results-root "$RECOVERY" --families ttb_sampling --noises 0.0 \
      --n-draws 3 --out "$OUT/f3a_clean.png"
run fig3A/vs_noisy_gt          -- $PY scripts/plot_recovery_vs_noisy_gt.py \
      --results-root "$RECOVERY" --families ttb_sampling --noises 0.0 \
      --n-draws 3 --out "$OUT/f3a_noisy.png"
run fig3A/clean_vs_noisy_gt    -- $PY scripts/plot_recovery_clean_vs_noisy_gt.py \
      --results-root "$RECOVERY" --families ttb_sampling --noises 0.0 \
      --n-draws 3 --out "$OUT/f3a_cvn.png"
run fig3B/plot_similarity      -- $PY scripts/plot_similarity.py \
      --root "$RECOVERY" --max-noise 0.75 --out-dir "$OUT/f3b_similarity"

# ---------------------------------------------------------------- Figure 4 ---
# Case study 1 (binary cues, seeds TTB+WADD) + generalisation to Hilbig 2014.
group fig4
want && echo "[Figure 4] case study 1 (binary) + Hilbig generalisation"
# run_dir is positional; writes into <run-dir>/analysis/convergence/.
run fig4A/convergence      -- $PY scripts/plot_autocog_convergence.py "$BINARY_HUMAN"
run fig4CD/eval_run_self   -- $PY scripts/eval_run_self.py \
      --run-dir "$BINARY_HUMAN" --out "$OUT/f4cd_run_self"
run fig4GJ/eval_hilbig     -- $PY scripts/eval_hilbig.py \
      --run-dir "$BINARY_HUMAN" --out "$OUT/f4gj_eval_hilbig"
run fig4GJ/plot_eval_hilbig -- $PY scripts/plot_eval_hilbig_figures.py \
      --run-dir "$BINARY_HUMAN"
run fig4/human_datasets    -- $PY scripts/plot_human_dataset_figures.py \
      --dataset firebase_prolific

# ---------------------------------------------------------------- Figure 5 ---
# Case study 2 (cardinal cues, seeds TTB+Tallying) + preregistered follow-up.
group fig5
want && echo "[Figure 5] case study 2 (cardinal) + preregistered study"
run fig5A/convergence      -- $PY scripts/plot_autocog_convergence.py "$CARDINAL_HUMAN"
run fig5CD/eval_human      -- $PY scripts/eval_human.py \
      --run-dir "$CARDINAL_HUMAN" --out "$OUT/f5cd_eval_human"
# --rounds must be given: the default (5 10 15 20 25) was tuned for a 25-round
# Centaur run and overruns every shipped run, which have 5 cycles.
run fig5CD/trajectory      -- $PY scripts/eval_human_trajectory.py \
      --run-dir "$CARDINAL_HUMAN" --rounds 1 2 3 4 5 --n-subjects 5 \
      --out "$OUT/f5cd_traj"
run fig5/human_datasets    -- $PY scripts/plot_human_dataset_figures.py \
      --dataset hdm_full_prolific_run_full
run fig5FHI/preregistration -- $PY scripts/plot_preregistration_figures.py \
      --bundle-dir "$PREREG_BUNDLE"
skip fig5/prereg_eval_models \
      "needs per-participant scripts/preregistration/data/exp{1,2}/trials.csv, which is not committed (only the aggregated h1/h2/h3 CSVs are). Not a reported result."

# --------------------------------------------------------- Figure 2 + SI -----
# Case study of round-0 narrowing, controls, compute accounting, stats table.
group si
want && echo "[Figure 2 + SI] case study, controls, compute, stats"
run fig2/case_study_wadd_data -- $PY scripts/case_study_wadd_narrowing.py \
      --run-dir "$WADD_RUN3"
run fig2/case_study_wadd_plot -- $PY scripts/plot_case_study_wadd.py \
      --run-dir "$WADD_RUN3" --out-dir "$OUT/f2_case_study"
run si/seed_gt_control    -- $PY scripts/plot_seed_gt_control.py --out-dir "$OUT/si_seed_gt"
run si/proposer_selection -- $PY scripts/proposer_selection.py
# jsd_discriminability takes a standalone design JSON; none is committed, so
# lift a real one out of a run's observations/state.json rather than invent it.
$PY - "$TTB_RUN" "$OUT/si_design.json" <<'EXTRACT' >/dev/null 2>&1
import json, sys
st = json.load(open(sys.argv[1] + "/observations/state.json"))
e = st["rounds"][0]["observations"][0]["experiment"]
json.dump({k: e[k] for k in ("validities", "trial_a_ratings", "trial_b_ratings")},
          open(sys.argv[2], "w"))
EXTRACT
run si/jsd_discriminability -- $PY scripts/jsd_discriminability.py \
      --theory_1 theories/heuristic_decision_making/ttb_sampling.yaml \
      --theory_2 theories/heuristic_decision_making/wadd_sampling.yaml \
      --experiment "$OUT/si_design.json" --n_runs 5
run si/toy_jsd            -- $PY scripts/toy_jsd_illustration.py
run si/summarize_compute  -- $PY scripts/summarize_compute.py "$TTB_RUN"
run si/summarize_theories -- $PY scripts/summarize_final_theories.py "$TTB_RUN"
run si/hilibig_battery    -- $PY scripts/hilibig_battery.py \
      --run-dir "$CARDINAL_HUMAN" --ground-truth tallying \
      --n-blocks 1 --n-subjects 3 --out "$OUT/si_battery"
# AUTOCOG_STATS_OUT keeps the committed results/stats/ table (the paper's
# record) untouched while still exercising the whole script.
run si/stats_table        -- env AUTOCOG_STATS_OUT="$OUT/si_stats" $PY scripts/stats.py

# --- SI controls: proposer framing, JSD metric, blind design ------------------
run si/proposer_outcome_speed  -- $PY scripts/proposer_outcome_speed.py
run si/proposer_perseveration  -- $PY scripts/proposer_perseveration_behavior.py
run si/proposer_speed_trace    -- $PY scripts/proposer_speed_trace.py
run si/proposer_divergence     -- $PY scripts/explore_proposer_divergence.py "$TTB_RUN"
run si/calibrate_jsd_threshold -- $PY scripts/calibrate_jsd_threshold.py \
      --n_designs 2 --n_runs 5 --out "$OUT/si_jsd_threshold.json"
run si/jsd_recovery            -- $PY scripts/jsd_recovery.py \
      --blind-root results/controls/condition_blind_design
run si/performance_curve       -- $PY scripts/performance_curve_trajectory.py \
      --run-dir "$BINARY_HUMAN"
run si/rank_theories           -- $PY scripts/rank_theories.py \
      --eval-dir "$CARDINAL_HUMAN/analysis/eval_hilbig" --out "$OUT/si_rank.md"
run si/jsonl_to_hilbig_exp     -- $PY scripts/jsonl_to_hilbig_exp.py \
      --out "$OUT/si_exp.txt" \
      "$CARDINAL_HUMAN/observations/data/round_000_obs_00.jsonl"

# --- analyses whose inputs or credentials are not available here --------------
skip si/judge_similarity \
      "calls the Gemini LLM judge; needs a real API key. Its outputs (judge_similarity_{desc,joint}.csv) ARE committed under results/recovery/<family>/noise=*/, which is what plot_similarity reads."
skip si/hilibig_battery_aggregate \
      "aggregate_hilibig_battery.py + plot_hilibig_battery_{aggregate,correlation}.py consume per-run hilibig_battery outputs that are not committed; regenerate them with scripts/rerun_hilibig_battery.sh first."
skip prereg/pipeline \
      "scripts/preregistration/{build,search,run_experiment,download_data,download_firebase,analyse,model_preference}.py drive live Firebase/Prolific data collection and need credentials + the raw per-participant drop, which is not committed. The frozen battery (battery.json) and the aggregated h1/h2/h3 CSVs that the paper reports ARE committed."

# ------------------------------------------------------------- the loop ------
group loop
want && echo "[Loop] discovery loop entry points"
# The loop cannot be smoke-tested without a real API key, so it is recorded as
# skipped rather than quietly omitted:
#   * `make_client` builds MockClient with an EMPTY canned list, so
#     --llm_provider mock raises "exhausted canned responses" on call #1. It is
#     a unit-test fixture; the suite supplies its own responses.
#   * AutoCog.from_yaml builds its client eagerly from configs/default.yaml
#     before the CLI provider is applied, so a keyless run dies there first.
# Set SMOKE_LLM=1 (with a key in .env) to actually run these, billed.
if [ -n "${SMOKE_LLM:-}" ]; then
  run loop/decision_making_binary -- $PY main_decision_making_binary.py \
        --ground_truth ttb_sampling --n_rounds 1 --out_path "$OUT/loop_dmb"
  run loop/ablation_binary        -- $PY main_ablation_binary.py \
        --condition baseline --ground_truth ttb_sampling --n_rounds 1 \
        --out_path "$OUT/loop_ablation"
  run loop/heuristic_dm           -- $PY main_heuristic_decision_making.py \
        --ground_truth ttb --n_rounds 1 --out_path "$OUT/loop_hdm"
else
  skip loop/decision_making_binary "needs a real API key; --llm_provider mock is a unit-test fixture with no canned responses. Re-run with SMOKE_LLM=1 (billed)."
  skip loop/ablation_binary        "needs a real API key (see above). SMOKE_LLM=1 to run."
  skip loop/heuristic_dm           "needs a real API key (see above). SMOKE_LLM=1 to run."
fi

# Argparse wiring IS checkable for free — every entry point must build its
# parser and print --help without touching the network.
group loop
run loop/help_dmb      -- $PY main_decision_making_binary.py --help
run loop/help_ablation -- $PY main_ablation_binary.py --help
run loop/help_hdm      -- $PY main_heuristic_decision_making.py --help
run loop/help_online   -- $PY main_decision_making_binary_online.py --help
run loop/help_online_hdm -- $PY main_heuristic_decision_making_online.py --help

# ------------------------------------------------------------------ report ---
echo
echo "================================================================"
printf 'smoke: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -gt 0 ]; then
  printf 'failed:  %s\n' "${FAILED_NAMES[*]}"
fi
if [ "$SKIP" -gt 0 ]; then
  printf 'skipped: %s\n' "${SKIPPED_NAMES[*]}"
  echo "         (each skip printed its own reason above -- these are known gaps,"
  echo "          not silent coverage)"
fi
if [ -n "${KEEP:-}" ]; then
  echo "outputs kept in $OUT"
else
  [ "$FAIL" -eq 0 ] && rm -rf "$OUT" || echo "logs kept in $OUT (failures)"
fi
[ "$FAIL" -eq 0 ]
