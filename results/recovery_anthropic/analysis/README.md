# Recovery of single_cue / anti_majority with Claude Opus 5 (2026-09-05)

Re-run of the non-canonical recovery battery (paper Figure 3C) with Claude Opus 5
as the discovery LLM (proposer, arbiter, improver, theory generator and
ground-truth simulator all on `claude-opus-5`, adaptive thinking at default
effort, 128k output ceiling). Runs live in `results/recovery_anthropic/<family>/noise=0.0/`.

## Design

- Families: `single_cue`, `anti_majority`; seeds TTB + Tallying (sampling); action noise 0.0; gt_seed 0.
- 3 replications per family, **5 cycles** each. Exception: `single_cue` run 3 has **4 complete
  cycles** (its 5th cycle was stopped after the theory generator hit the 128k output cap on three
  consecutive samples of the same prompt; partial artifacts remain in `rounds/round_004/` without
  `theories.json` and are ignored by the analysis).
- Scoring exactly as the paper: MSE of the proportion of B choices per unique Hilbig (2014) option
  pair, 500 simulated subjects, against the noiseless ground truth; theories pooled within a run,
  then mean ± SEM across runs (`scripts/recovery_correlation.py` / `plot_recovery_per_model.py`).
- Gemini baselines at matched cycle counts (5 / 10 / 20) were computed from snapshot copies of the
  Gemini run dirs under `results/recovery/` truncated to the first N rounds; long tables in
  `baselines/`. Gemini at 20 cycles reproduces the paper's values.

## Files

- `recovery_mse_model_comparison.{svg,png}` — Fig. 3C-style bars: Gemini @5/10/20 vs Opus 5 @5
  and Opus 5 @10 (single_cue) / @5 (anti_majority) (regenerated 2026-09-21; see addendum).
- `recovery_mse_model_comparison_5cycles.{svg,png}` — matched 5-cycle comparison only.
- `recovery_mse_model_comparison_table.csv` — the plotted means/SEMs.
- `per_model/` — standard per-model figures and `recovery_long.csv` for the Opus runs.
- `judge_similarity_model_comparison_table.csv` — LLM-judge mechanism similarity
  (`scripts/judge_similarity.py`, gemini-3-flash-preview, 3 votes, description and joint modes);
  per-run CSVs sit next to the run dirs.
- Produced with `scripts/plot_recovery_model_comparison.py`.

## Results (surfaced MSE, mean ± SEM over 3 runs; lower is better; seed ≈ 0.172 / 0.349)

| family        | Gemini @5     | Gemini @10    | Gemini @20    | Opus 5 @5     |
|---------------|---------------|---------------|---------------|---------------|
| single_cue    | 0.149 ± 0.040 | 0.149 ± 0.022 | 0.147 ± 0.028 | 0.115 ± 0.059 |
| anti_majority | 0.311 ± 0.036 | 0.197 ± 0.036 | 0.211 ± 0.070 | 0.053 ± 0.040 |

Opus 5 per-run surfaced MSE: single_cue 0.041 / 0.073 / 0.232 (run 3 = 4 cycles);
anti_majority 0.132 / 0.011 / 0.015.

LLM-judge mechanism similarity (description / joint), 5 cycles:
single_cue Gemini 0.04 / 0.11 vs Opus 0.36 / 0.34; anti_majority Gemini 0.08 / 0.06 vs Opus 0.43 / 0.55.

Qualitative read of the surfaced `predict` functions: all three Opus anti_majority runs invert
polarity (a '1' is a warning flag; choose the option with fewer validity-weighted flags), which is
the anti-majority mechanism in substance; Gemini's 5-cycle theories are all forward-framed. For
single_cue, Opus run 1 surfaced a one-reason "maximum-doubt" rule that consults the least-valid cue
(recovered), run 2 a validity-inverted tallying rule (behavioural look-alike), run 3 a read-order
scanning model (not recovered).

## Cost

540 LLM calls, 13.9M input / 9.5M output tokens (output includes thinking), ≈ $306 on Opus 5,
including ≈ $75 lost to the 32k-cap and runaway-loop incidents fixed in `src/llm.py`.


## Addendum 2026-09-20: single_cue extended to 10 cycles

All three `single_cue` Opus 5 runs were extended from 5 (run 3: 4 + a half-done 5th) to 10
cycles with the current pipeline (which, unlike in September, re-proposes in-process when an
experiment proposal fails schema validation; run 1's 6th cycle crashed on exactly that before
the client fix and was resumed by hand). `anti_majority` was NOT extended and stays at 5.

- `per_model/single_cue_10cycles_recovery_long.csv` (+ `_recovery_correlation.*`, `_recovery_mse.*`)
  — single_cue at 10 cycles. `per_model/recovery_long_single_cue10_anti_majority5.csv` merges it
  with the 5-cycle anti_majority rows for the cross-model figure in
  `results/recovery_openrouter/analysis/`.
- Surfaced MSE single_cue @10: 0.084 +/- 0.061 (runs 0.046 / 0.004 / 0.203) vs 0.115 +/- 0.059 @5.
  Run 2's final theory is a true single-cue rule on the least-valid cue (mechanism recovered);
  run 1 a doubt-gated one-reason rule with compensatory fallback; run 3 a weighted reason tally
  whose 10th-cycle replacement is anti-correlated with the ground truth.
- Cost of the extension (prompt-log tokens at $5 / $25 per M): 316 calls, 15.0M in / 7.2M out,
  ~$255 for 16 new cycles (~$16/cycle; later cycles carry 30k+ token prompts).
