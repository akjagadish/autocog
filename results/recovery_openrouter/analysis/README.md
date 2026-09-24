# Recovery of single_cue / anti_majority with open-weight models via OpenRouter (2026-09-19/20)

Re-run of the non-canonical recovery battery (paper Figure 3C) with two open-weight
discovery LLMs served through OpenRouter (`--llm_provider openrouter`): proposer, arbiter,
improver, theory generator and ground-truth simulator all on the same model.

- `deepseek/deepseek-v4-pro-0813`, reasoning effort **max** (`--reasoning_effort max`).
- `z-ai/glm-5.3`, reasoning effort **high**. At `max`, GLM never leaves its thinking block on
  the free-text theory-generation prompt (128k reasoning tokens, no answer, on three backends),
  so the user chose `high` for all GLM calls.
- Routing pinned per request: `provider.require_parameters` (JSON-schema support),
  `provider.quantizations=["fp8"]`, 131072 output cap. Backend actually serving each call is
  recorded in the prompt log's `## Usage` block (`provider`), as is `reasoning_tokens`.
  In practice DeepSeek was served ~80% by Baidu and GLM ~85% by Sail Research.

Runs live in `results/recovery_openrouter/<family>/noise=0.0/`; both models share the family
directories, so the analysis scripts (which glob every run dir) were run on per-model
symlink views. Logs: `logs/hdm_<family>_noise=0.0_<model-tag>_run<N>.log` (model-tag = id with
`/` -> `-`), including relaunch markers.

## Design

- Families: `single_cue`, `anti_majority`; seeds TTB + Tallying (sampling); action noise 0.0;
  gt_seed 0. 3 replications per family x model, **10 cycles** each (launched at 20, capped at 10
  by the user on 2026-09-19).
- Exception: `anti_majority` GLM run 2 has **9 complete cycles**. Its 10th-cycle state defeated
  both retry layers: every experiment proposal was a 50-80-feature design with rating lists one
  element off (3 in-process re-proposals), and the metric prompt (~50k tokens) hit the output
  cap on 3 consecutive samples; the watchdog gave up after 5 relaunches.
- Scoring exactly as the paper: MSE of the proportion of B choices per unique Hilbig (2014)
  option pair, 500 simulated subjects, against the noiseless ground truth; theories pooled
  within a run, then mean +/- SEM across runs (`scripts/recovery_correlation.py`).
- Baselines at 10 cycles: Gemini 3.1 Pro (`results/recovery_anthropic/analysis/baselines/
  gemini_10cycles_recovery_long.csv`) and gpt-5.6-sol
  (`results/recovery_openai/analysis/paper_style_sol_10r/recovery_long.csv`). Claude Opus 5 is
  included at **10 cycles for single_cue** (its three runs were extended from 5 to 10 on
  2026-09-20; `results/recovery_anthropic/analysis/per_model/single_cue_10cycles_recovery_long.csv`)
  and **5 cycles for anti_majority** (never extended; `.../per_model/recovery_long.csv`);
  the merged input is `.../per_model/recovery_long_single_cue10_anti_majority5.csv`.

## Files

- `recovery_mse_model_comparison.{svg,png}` — Fig. 3C-style bars: Gemini @10, Claude Opus 5 @10 (single_cue) / @5
  (anti_majority), gpt-5.6-sol @10, DeepSeek V4 Pro @10, GLM 5.3 @10. `_table.csv` holds the plotted means/SEMs.
- `<model-tag>/recovery_long.csv`, `recovery_correlation.*`, `recovery_mse.*` — per-model outputs.
- Produced with `scripts/plot_recovery_model_comparison.py`.

## Results (surfaced MSE, mean +/- SEM over 3 runs; lower is better; seed ~ 0.166 / 0.365)

| family        | Gemini @10    | Opus 5 (@10 / @5) | gpt-5.6-sol @10 | DeepSeek V4 Pro @10 | GLM 5.3 @10   |
|---------------|---------------|-------------------|-----------------|---------------------|---------------|
| single_cue    | 0.149 +/- 0.022 | 0.084 +/- 0.061 (@10) | 0.087 +/- 0.010 | 0.060 +/- 0.026     | 0.051 +/- 0.016 |
| anti_majority | 0.197 +/- 0.036 | 0.053 +/- 0.040 (@5)  | 0.270 +/- 0.127 | 0.125 +/- 0.113     | 0.010 +/- 0.001 |

Opus 5 single_cue per run @10: 0.046 / 0.004 / 0.203 (run 3's 10th-cycle replacement is
anti-correlated with the ground truth); @5 it was 0.115 +/- 0.059.

Per-run surfaced MSE (pooled within run): DeepSeek single_cue 0.017 / 0.055 / 0.108,
anti_majority 0.35 / 0.016 / 0.008; GLM single_cue 0.082 / 0.032 / 0.039, anti_majority
0.009 / 0.010 / 0.011. The ground truth resampled against itself scores ~0.004.

## Mechanism check (read from the surfaced `predict` functions)

- **anti_majority** (ground truth flips the TTB/Tallying/WADD majority; on these stimuli this
  behaves like "an endorsement is a red flag"): GLM found the polarity inversion in 3/3 runs
  (validity-weighted "distrust"/"defect" integration); DeepSeek in 2/3 (run 3 states it
  literally: "0 = clean, 1 = defect"). DeepSeek run 1 built a primacy-ordered evidence
  accumulator instead and is anti-correlated with the ground truth (r ~ -0.7).
- **single_cue** (only the lowest-validity cue is read; tie -> guess): no run found the pure
  rule. Closest: GLM run 2, a one-reason cascade consulting cues in *ascending* validity order
  (= take-the-worst, a look-alike that falls back to further cues); DeepSeek run 1, a
  below-median-validity cue weighting with lone-cue amplification (MSE 0.012, compensatory).
  The rest are tallying variants. gpt-5.6-sol likewise. **Claude Opus 5 run 2 (at 10 cycles)
  is the one genuine mechanism recovery of single_cue across all models**: its final theory
  reads only the minimum-validity cue, follows it with a fixed probability, and guesses
  (with a ~1-point residual tilt) when that cue ties — "no fallback to a tally, no
  compensatory balance, no validity-ordered take-the-best" in its own words. Opus run 1 is a
  doubt-gated one-reason rule with compensatory fallback (partial); run 3 a weighted reason
  tally (not recovered).

## Cost (from prompt-log usage priced at each backend's rate; ~13% more on the account
from work redone after relaunches)

| model               | logged $ | $/run (10 cycles) | in / out tokens | reasoning share of out | cap hits |
|---------------------|----------|-------------------|-----------------|------------------------|----------|
| DeepSeek V4 Pro @max | 66       | 9-12              | 23.0M / 23.6M   | 89%                    | 5        |
| GLM 5.3 @high        | 146      | 16-31             | 35.3M / 32.3M   | 89%                    | 58 ($26) |

For reference: gpt-5.6-sol ~$47/run @10 cycles, Claude Opus 5 ~$100/run @5 cycles.
