# Neutral vs Adversarial experiment proposer — summary

**Date:** 2026-06-12 · **Branch:** ablation
**Design:** 4 ground truths × 3 runs per condition, 5 rounds each (`--max_round 5`
cap for longer runs), gemini-3.1-pro-preview, gt_epsilon=0, seeds = ttb_sampling +
tallying_sampling (verified GT-leak-free).
**Conditions:** adversarial = original `AutoPi` (runs in
`results/heuristic_decision_making/synthetic_corrected_theories_binary_sampling/`);
neutral = `NeutralAutoPi` (`--condition neutral_proposer`, runs in
`results/heuristic_decision_making/ablation_stage0/condition_neutral_proposer/`).

## Result table (N = neutral, A = adversarial; mean ± sd over 3 runs)

| | ttb_sampling | take_the_worst | anti_majority | perseveration |
|---|---|---|---|---|
| Design divergence (excess JSD)ᵃ | N 0.120±0.061<br>A 0.067±0.020 | N 0.091±0.016<br>A 0.095±0.040 | N 0.055±0.018<br>A 0.060±0.008 | N 0.097±0.030<br>A 0.076±0.029 |
| Outcome — MSE ↓ᵇ | N 0.005±0.002<br>A 0.006±0.001 | N 0.037±0.059<br>A 0.065±0.076 | **N 0.265±0.134<br>A 0.342±0.055** | N 0.018±0.029<br>A 0.006±0.007 ᶜ |
| (ceiling / chance MSE) | (.003 / .107) | (.002 / .108) | (.003 / **.092 — both worse than chance**) | (.0025 / .0016 — degenerate) ᶜ |
| Outcome — Pearson r | N 0.990 / A 0.990 | N 0.83 / A 0.69 | N −0.42 / A −0.77 | NaN by definition ᶜ |
| Outcome — behavioralᵈ | — | — | — | **N 1/3 vs A 3/3 recovered** |
| Speed — AUC (recovery-vs-round) | N 0.89 / A 0.89 | N 0.49 / A 0.56 | N −2.6 / A −2.8 | (switch-rate trajectory ↓) |
| Speed — first round "good"ᵉ | both 0.7 (3/3 each) | N 3.0 / A 1.5 (2/3 each) | never (0/6) | **N round 4 (1/3) vs A round 2 (3/3, monotone)** |
| Mechanism — history-reading code | 0/3 both | 0/3 both | 0/3 both | **N 0/3 vs A 2/3** |
| Proposal attempts (M9) | ≈1 both | ≈1 both | ≈1 both | ≈1 both |
| Verdict | tie — both at ceiling | tie — both noisy recovery | tie — **both fail (worse than chance)** | **adversarial wins all 3 levels** (n=3, underpowered) |

ᵃ Sequence-JSD of each FIRST proposed design above a size-matched random-design
baseline for the same theory pair (pre-acceptance-gate; isolates the proposer).
All GTs: difference < spread → tie. Source: `divergence_<gt>.jsonl`
(`scripts/explore_proposer_divergence.py`, n_runs=120, k_baseline=5, seed=0,
max_round=5).
ᵇ Best-by-loss *surviving* theory at the round-5 cap (non-oracle selection,
`scripts/proposer_selection.py`) scored vs noise-less GT choice proportions on
the Hilbig task; ceiling = GT vs itself, chance = predict 0.5. Source:
`outcome_speed.json` (`scripts/proposer_outcome_speed.py`).
ᶜ Perseveration's flat ≈0.5 marginals break static metrics: r is NaN by
definition (zero target variance); MSE is defined but degenerate (chance ≈
ceiling — a coin flip scores as well as the GT itself).
ᵈ Behavioral recovery (perseveration only): switch-rate ≈ 0 (GT 0.000, coin 0.5)
AND lock-side split ≈ 0.5. Mechanisms split into true history-readers
(A run1/run2) and behaviorally-equivalent random-side-bias models (A run3,
N full3). Source: `perseveration_behavior.json`
(`scripts/proposer_perseveration_behavior.py`).
ᵉ Memoryless GTs: first round with normalized recovery fraction ≥ 0.9
(`speed_trace.json`, `scripts/proposer_speed_trace.py`; full trajectories
reported — re-thresholdable). Perseveration: first round with switch-rate
≤ 0.05; all three adversarial trajectories are identical and monotone
([~.5, ~.5, 0, 0, 0]).

## Conclusions

1. **Memoryless GTs: neutral ≈ adversarial on every axis** (divergence, outcome,
   speed, attempts). De-adversarializing the experiment proposer costs nothing
   detectable at n=3.
2. **Both conditions fail anti_majority worse than chance** — they confidently
   select conventional heuristics that predict the opposite of the contrarian
   GT. A pipeline-level failure, not a proposer difference. (An earlier
   apparent recovery of anti_majority was traced to GT-in-seed contamination
   during concurrent theory-file edits and is excluded; these runs use
   verified clean seeds.)
3. **Perseveration (history-dependent GT) is the one separation:** adversarial
   recovered perseverative behavior in 3/3 runs by round 2 without regressing;
   neutral 1/3 at round 4. Adversarial alone produced history-reading code
   (2/3). Consistent across behavior, speed and mechanism, but n=3
   (Fisher p ≈ 0.4) — a strong lead, not yet a claim. Natural follow-ups:
   more perseveration seeds; the `alternating` GT.
4. **Methodological notes:** (i) loss-gate selection is non-monotone in
   recovery (runs reach "good" then regress — both conditions); (ii) the
   excess-JSD baseline samples `random_design`'s validity distribution
   (uniform 0.55–0.95, anchored), a documented methodological choice;
   (iii) one surfaced theory (anti_majority A run2) samples inside `predict`,
   giving third-decimal jitter; all other rows are exactly reproducible.

## Reproduce

```bash
export PYTHONPATH=.
# divergence (per run dir; ~1-2 min each):
python scripts/explore_proposer_divergence.py <run_dir> --n_runs 120 --max_round 5
# outcome + speed + perseveration behavior (full grid):
python scripts/proposer_outcome_speed.py
python scripts/proposer_speed_trace.py
python scripts/proposer_perseveration_behavior.py
```
