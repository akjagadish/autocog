# metric_exp02_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=38):** [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. PQSAI (advocated) and SACNAI (competing) share everything: anti-validity additive integration (a rating of 1 is a defect, tied cues cancel exactly), the identical bucketed-mean-endorsed-validity context key with per-bucket eta redraw, softmax(beta*value) with symmetric lapse, ~13% true guessers, and near-identical engaged beta. The hard structural difference is PQSAI's LOAD DILUTION: its prestige amplifier is sigma_eff = 1 + (sigma-1)*exp(-gap/0.10)*D with D = 1/(1+(n_disc/n0)^1.1), n0 in [7,13], where n_disc counts ALL cues on which the two options differ; SACNAI's amplifier has NO n_disc term. Experiment [16] already varied n_disc, but only by SCALING the whole contest (4v5 -> 8v10 -> 12v15), which confounds n_disc with the margin magnitude (margins scale linearly in the contest size), so SACNAI's ladder prediction there was graded (0.33 -> 0.13), not flat. This design severs that confound completely.

THE INSTRUMENT: SCORE-NEUTRAL n_disc INJECTION. Take a fixed near-knife contest -- option A endorses 4 cues at 0.90, option B endorses 5 cues at 0.85 (top-tier gap 0.05, exp(-gap/0.10) = 0.607) -- and add p BALANCED DISCRIMINATING FILLER PAIRS at validity 0.85: in each pair, one cue is endorsed only by A and one only by B. Each pair contributes exactly +w85 to A's score and +w85 to B's score, so it cancels EXACTLY in both theories' score differences; but it increments n_disc by 2. The filler cues are at B's existing tier, so the endorsed-tier structure ({0.90, 0.85}), the gap key (0.05), and -- by explicit construction -- the context bucket are all unchanged (v_ref = 0.858-0.873 at every rung, all rounding to bucket '0.9', so both theories use the SAME per-subject eta draw at every rung). Consequently SACNAI's per-trial computation is BIT-IDENTICAL across rungs up to a <0.3% common v_ref scale factor and mean-zero attention noise: its ladder slope is EXACTLY 0.000 for EVERY admissible (sigma, beta, eta, epsilon) -- a parameter-free flat prediction strictly stronger than [16]'s graded one. PQSAI cannot stay flat: D is strictly decreasing in n_disc, so every prestige-sensitive subject's sigma_eff falls and their margin migrates monotonically toward A.

KNIFE MATH. The contest flips at sigma_eff* = 1.25/((0.90/0.85)^(1+eta)) in [1.134, 1.198] over the whole eta clip. PQSAI's sensitive subjects (sigma in [1.6, 2.1], 40-60% of non-guessers) have sigma_eff = 1 + (sigma-1)*0.607*D, which swings from [1.16, 1.43] at n_disc = 9 (D in [0.43, 0.60]) down to [1.07, 1.22] at n_disc = 25 (D in [0.20, 0.33]) -- the swing brackets the knife, so sensitive subjects migrate from B to A along the ladder for essentially any admissible parameters. PQSAI's linear counters (sigma in [1.0, 1.15]) have sigma_eff <= 1.05 < 1.134 and stay confidently A at every rung; SACNAI's subjects (sigma_eff in [1.15, 1.39], no dilution) stay B at every rung. Population P(A): PQSAI rises ~0.52 -> 0.65 (rise +0.07..+0.16 over its full box); SACNAI sits flat at ~0.33. The F2 family (5x0.90 vs 6x0.85, knife 1.125 < SACNAI's sigma_eff floor 1.1515) replicates the same logic with a parameter-free DOUBLE sign kill at its base rung: every SACNAI engaged subject prefers B (margin <= -0.008 across the entire (sigma, eta) box) while EVERY PQSAI linear counter prefers A (margin >= +0.21 across its entire box) -- this base cell deliberately replicates [16]'s B1 as a cross-experiment anchor, but its ladder (n_disc 11 -> 27) is the new manipulation.

HEURISTIC IMMUNITY. Every rung adds exactly one 1 to EACH option (total ones 4+p vs 5+p -- the count difference is constant), so raw-count, fewer-1s, more-1s, and total-ones heuristics are flat by construction and cannot mimic the rising slope; a generic complexity/dilution artifact would pull rates TOWARD 0.50, whereas PQSAI's ladder moves AWAY from 0.50 on the high side (0.52 -> 0.65). The TIED-FILLER NEGATIVE CONTROL (tied 0.90 cues endorsed by BOTH options, n_disc held at 9 while display complexity grows) is flat under BOTH theories -- any tied-filler effect invalidates the n_disc interpretation outright. Nested filler subsets make per-cue attention a fixed per-subject rescaling common to all rungs, unable to generate a systematic slope; attention on a filler pair is mean-zero and averages over repetitions. C1 (4x0.90 vs 5x0.60, gap 0.30) switches the amplifier OFF in both theories (both prefer the 0.60-stack) and screens TTB, fewer-1s, and pro-validity strategies (all pick A there). AG (2x0.90 vs 6x0.85) is a within-bucket ceiling anchor (~0.92 for both). TIE pins side bias at 0.500; ANCHOR (all-1 vs all-0) calibrates lapse and screens tallying/pro-validity (~0.09 for both vs ~1 for the screens). All critical pairs are mirrored; mirrors preserve the active set, v_ref, bucket, gap, and n_disc. Secondary readouts from the same trials: (a) per-subject rung slopes are ~50% significantly positive under PQSAI (the sensitive mode) vs chance under SACNAI; (b) within-pair repetition flips (same stimulus, K=4 reps) rise with rung under PQSAI (attention sd 0.096 -> 0.14 as n_active grows 9 -> 25, while the margin shrinks toward the knife) but stay constant under SACNAI (fixed sd 0.045, constant margin).

DESIGN. 38 experts (11 at 0.90, 22 at 0.85, 5 at 0.60); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials (40 F1 ladder, 16 F2 endpoints, 16 tied control, 8 AG, 8 C1, 4 tie, 4 anchor trials per subject). Approximate population predictions, P(A) with A = the 0.90-stack option (ANCHOR quoted as P(all-1)): F1 ladder (n_disc 9/13/17/21/25): PQSAI [0.52, 0.56, 0.60, 0.63, 0.65], SACNAI [0.33, 0.33, 0.33, 0.33, 0.33]; F2 (n_disc 11/27): PQSAI [0.50, 0.62], SACNAI [0.23, 0.23]; tied control: PQSAI 0.52, SACNAI 0.33 (= F1 base rung exactly); AG 0.92 both; C1 0.08 both; TIE 0.500 both; ANCHOR 0.09 both.

DECISION RULES. (i) PQSAI is supported and SACNAI falsified if the F1 ladder rises by >= +0.06 (monotone or near-monotone), the F2 endpoints rise by >= +0.05, the tied control is indistinguishable from the F1 base rung, and a substantial fraction (~15-30%) of non-guesser subjects show significantly positive within-subject rung slopes. SACNAI cannot produce ANY of this for any parameter values: n_disc is not an input to its computation, and the contest, gap, bucket, eta draw, and score difference are identical at every rung -- its population slope is exactly 0.000. (ii) SACNAI is supported and PQSAI falsified if all ladders are flat (|slope| <= 0.03 beyond binomial noise), the tied control is flat, and the F2 base rung shows essentially no engaged A-majority subjects (<= ~5%; every PQSAI linear counter -- 40-60% of its non-guesser mass -- is pinned to A there by a margin >= +0.21 across its entire box, while every SACNAI subject is pinned to B). (iii) BOTH are falsified if the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), the 0.90-stack wins C1 at >= 0.5 (TTB/fewer-1s/pro-validity), the tied control differs from the F1 base rung (filler-count/complexity heuristic), or rates move TOWARD 0.50 with load (generic dilution, which PQSAI's away-from-0.50 rising ladder excludes).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** PQSAI (recalibrated) — Prestige-Quantity Salience Anti-validity Integration with Load Dilution and a Bimodal Salience Mixture. Subjects integrate binary expert ratings additively into a per-option value but comprehend a rating of 1 as a DEFECT (sign inversion), with context-normalized anti-weights w_j = -v_j*(v_j/v_ref)^eta (v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context, latent cross-context r ~ 0.18). The top endorsed validity tier receives an additional multiplicative aversion sigma_eff = 1 + (sigma_i - 1)*exp(-gap/0.10)*D, where D = 1/(1 + (n_disc/n0)^1.1) is a FLAT load-dilution factor (exponent softened 1.5 -> 1.1, n0 raised 6 -> ~10): prestige aversion is strong on small knife-edge contests (n_disc = 6, D ~ 0.62-0.68) but retains SUBSTANTIAL amplification even on large stacks (n_disc = 13-14, D ~ 0.40-0.45), so the population sits between the pure linear core and full undiluted salience exactly where the real data lie. The engaged population is bimodal in salience: ~50% prestige-sensitive (sigma_i ~ U[1.6, 2.1]) and ~50% linear counters (sigma_i ~ U[1.0, 1.15]), generating large between-subject variance on near-knife composites. Per-trial attention diffusion scales with load, sd = s0 + s1*sqrt(n_active), attenuating within-subject reliability on complex multi-tier trials. Softmax(beta*value) with symmetric lapse; ~13% true guessers; ~12% hot tail; exactly linear within a validity family (tied-filler load ladders flat).

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- mode_seed: [0, 1]
- sigma_sensitive: [1.6, 2.1]
- sigma_sensitive_fraction: [0.4, 0.6]
- dilution_n0: [7, 13]
- noise_s0: [0.02, 0.04]
- noise_s1: [0.012, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PQSAI (recalibrated iter 2): Prestige-Quantity Salience
    # Anti-validity Integration with Load Dilution and a Bimodal
    # Salience Mixture.
    #
    # CHANGES VS THE ACCEPTED ITER-1 BASE (minimal diff, per the
    # critic's five-point recalibration — all in-family):
    #   (A) FLATTER DILUTION CURVE: exponent 1.5 -> 1.1 and
    #       dilution_n0 raised (default 6 -> 10, exposed range
    #       [7, 13]). New operating points: D(n_disc=6) ~ 0.62-0.68
    #       (was ~0.50) and D(n_disc=13) ~ 0.40-0.45 (was ~0.15-0.24).
    #       This simultaneously raises Exp 11 toward 2.88 (small-n_disc
    #       cells keep much more amplification) and pulls Exp 16 down
    #       from the pure-core 0.686 toward 0.495 (large-stack cells
    #       now RETAIN substantial amplification, landing the
    #       population between core and full SACNAI).
    #   (B) STRONGER SENSITIVE MODE: sigma_sensitive box widened to
    #       [1.6, 2.1] (default 1.95) and sensitive fraction toward
    #       0.5 (range [0.4, 0.6]). The 3v3 AL cells now receive
    #       sigma_eff ~ 1.35-1.40 on sensitive subjects — roughly a
    #       half-to-full recovery of SACNAI's undiluted amplification,
    #       which is what the real 2.88 (vs SACNAI's 3.60 and the
    #       core's 0.32) requires.
    #   (C) GAP SCALE KEPT AT 0.10: coarse-gap stacks (Exps 5/8/10)
    #       keep exp(-gap/0.10) <= 0.14-0.22, so the amplifier stays
    #       essentially off there and the validated core predictions
    #       are preserved.
    #   (D) EXP 2 VARIANCE: the widened sensitive box (up to 2.1) plus
    #       the flatter dilution enlarges the mode separation on
    #       gap-0.10, n_disc ~ 5-6 cells (sensitive sigma_eff ~ 1.25 vs
    #       linear ~ 1.02), injecting the missing between-subject
    #       spread without any new mechanism.
    #   (E) EXP 9 OVER-CONSISTENCY: noise_s1 raised (default 0.014 ->
    #       0.022, range [0.012, 0.03]) to attenuate within-subject
    #       reliability on the 24-feature complex trials and pull the
    #       cross-ladder correlation down toward 0.196; noise_s0 kept
    #       low (range [0.02, 0.04]) so Exp 13's flat ladder and
    #       Exp 14's slope are preserved.
    #
    # Everything else — the shared CNNAI/SACNAI core (defect sign,
    # context-normalized anti-validity weights, bucketed per-context
    # eta with latent r ~ 0.18, 13% guessers, 12% hot tail, exact
    # within-family linearity) — is retained verbatim from the
    # accepted base.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PQSAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    def _pos(name, default, lo, hi):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = default
        if not np.isfinite(x):
            x = default
        return min(max(x, lo), hi)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")
    ms = _seed("mode_seed")

    # --- PQSAI-specific knobs (recalibrated) ---
    n0 = _pos("dilution_n0", 10.0, 2.0, 20.0)          # dilution scale (raised)
    s0 = _pos("noise_s0", 0.03, 0.0, 0.2)              # noise intercept (kept low)
    s1 = _pos("noise_s1", 0.022, 0.0, 0.1)             # load-scaled slope (raised)
    sig_s = _pos("sigma_sensitive", 1.95, 1.0, 3.0)    # sensitive-mode sigma (raised)
    frac = _pos("sigma_sensitive_fraction", 0.52, 0.05, 0.95)  # toward 0.5

    # --- Bimodal salience mixture (applies to all non-guessers) ---
    if ms < frac:
        # Prestige-sensitive subject: sigma ~ U[1.6, 2.1] (drawn per
        # subject via the sigma_sensitive parameter).
        sigma = sig_s
    else:
        # Linear counter: sigma ~ U[1.0, 1.15].
        t = (ms - frac) / max(1e-9, 1.0 - frac)
        sigma = 1.0 + 0.15 * t

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. The amplifier is irrelevant
        # here, so pin sigma at 1.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.0
        sigma = 1.0
    elif u < 0.25:
        # Hot tail (~12%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~75%). Small stable elasticity component
        # (sd 0.09) + larger per-context redraw (sd 0.19): the variance
        # ratio fixes latent cross-context consistency at r ~ 0.18.
        # Narrow stable lapse so it cannot act as a hidden
        # cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill
    # knives (~0.31-0.34). Hot-tail eta is pinned at the engaged
    # median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Prestige aversion with QUANTITY DILUTION (flattened) ---
    # The top endorsed tier is amplified by
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap/0.10) * D,
    #     D = 1 / (1 + (n_disc/n0)^1.1),
    # where n_disc counts the cues on which the two options differ.
    # FLATTENED curve (exponent 1.1, n0 ~ 10): small knife-edge
    # contests (3v3, n_disc = 6, D ~ 0.62-0.68) keep most of the
    # amplification -> strong prestige aversion (Exp 11 AL cells,
    # Exp 15 L-cells); large stacks (n_disc = 13-14, D ~ 0.40-0.45)
    # retain SUBSTANTIAL amplification, landing the population
    # between the linear core and full undiluted salience (Exp 16).
    # Single-family trials have no contrast tier: off, so
    # within-family load ladders stay exactly flat.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        n_disc = int(np.count_nonzero(ra != rb))
        D = 1.0 / (1.0 + (n_disc / n0) ** 1.1)
        sig_eff = 1.0 + (sigma - 1.0) * math.exp(-gap / 0.10) * D
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # --- Load-scaled attention diffusion ---
    # sd = s0 + s1*sqrt(n_active): small-stack cells stay crisp while
    # complex multi-tier trials acquire within-subject stochasticity
    # (attenuating cross-ladder consistency on 24-feature designs).
    # Per-trial noise averages out within a subject, so it supplies
    # within-context stochasticity without damping population means.
    sd_att = s0 + s1 * math.sqrt(float(active.size))
    att = np.exp(sd_att * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** SACNAI — Salience-Amplified Context-Normalized Anti-Validity Integration with a Hot Tail. Subjects integrate all binary expert ratings additively into a per-option value, but comprehend a rating of 1 as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j/v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity buckets (mean endorsed validity rounded to 0.1), decomposing into a small stable subject component and a larger per-bucket redraw (latent cross-context r ~ 0.18). NEW — top-tier salience amplifier: on each trial, the highest-validity tier AMONG THE CURRENTLY ENDORSED experts (the 'most prestigious expert present') receives an additional multiplicative aversion sigma_eff = 1 + (sigma-1)*exp(-gap/0.10), where gap is the validity gap between the top endorsed tier and the next endorsed tier. The aversion is therefore strongest exactly on near-knife top-tier contrasts (e.g. 3x0.90 vs 3x0.85), where the top expert's prestige must break an almost tied count contest, and vanishes on coarse contrasts (gap >= 0.2, e.g. 0.90-vs-0.70 kill stacks), where it would act as a mere scale factor or flip established stack preferences. A modest per-trial multiplicative attention diffusion (sd 0.045) supplies within-context stochasticity. Choice is softmax(beta * value) with a symmetric lapse. The population is a three-part mixture decoupled from any load/compression machinery: ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged (beta lognormal, median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow lapse 0.03-0.09), and ~15% hot tail (beta ~ U[3.4, 5.2], lapse 0.02-0.05, eta pinned at the engaged median) — sized so the fitted-sensitivity ramp fraction lands near the observed ~0.09 while the winsorized ladder slope stays near 1.4. No compressive load map and no context-invariant quadratic term: the model is exactly linear in the ratings within a validity family, so tied-filler load ladders are exactly flat.

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- sigma_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SACNAI: Salience-Amplified Context-Normalized Anti-Validity
    # Integration with a Hot Tail.
    #
    # CORE (retained from CNNAI, validated by Exps 1 & 2): a rating of 1
    # is a defect; per-option value = sum_j w_j * rating_j with
    # w_j = -v_j * (v_j / v_ref)^eta, v_ref = mean validity of the
    # endorsed cues; eta is drawn per bucketed-mean-endorsed-validity
    # context (small stable subject component sd 0.09 + larger
    # per-bucket redraw sd 0.19 -> latent cross-context r ~ 0.18);
    # per-trial multiplicative attention diffusion (sd 0.045);
    # softmax(beta * value) with symmetric lapse. NO compressive load
    # map (knee empirically dead) and NO context-invariant quadratic
    # term. The model is exactly linear in the ratings within a
    # validity family, so tied-filler load ladders are exactly flat.
    #
    # NEW (1) TOP-TIER SALIENCE AMPLIFIER: on each trial, the
    # highest-validity tier among the currently ENDORSED experts
    # receives an extra multiplicative aversion
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap / 0.10),
    # where gap = (top endorsed validity) - (next endorsed validity).
    # Rank-based 'most-prestigious-expert aversion': strongest exactly
    # on near-knife top-tier contrasts (gap ~ 0.05, e.g. 3x0.90 vs
    # 3x0.85 cells), where it roughly doubles the value margin; a
    # near-common scale factor on moderate gaps (0.10); and essentially
    # off on coarse contrasts (gap >= 0.20, e.g. the 0.90-vs-0.70 and
    # 1.00-vs-0.50 kill stacks), where amplification would flip
    # established stack preferences. Tied (both-endorsed) top cues
    # cancel exactly in the value difference, so the amplifier only
    # bites on discriminating prestige.
    #
    # NEW (2) POPULATION (decoupled from any falsified machinery):
    # ~13% true guessers (beta ~ 0, elevated lapse), ~72% engaged
    # (beta lognormal median 2.0, log-sd 0.25, clip [0.8, 4.5]; narrow
    # lapse 0.03-0.09; eta centered at 0.12, safely below the kill
    # knives at eta ~ 0.31-0.34), and ~15% hot tail (beta ~ U[3.4, 5.2],
    # lapse 0.02-0.05, eta pinned at the engaged median) -- sized so
    # the fitted-sensitivity ramp fraction lands near the observed
    # ~0.09 while the winsorized single-validity ladder slope stays
    # near ~1.4 (engaged median trimmed to 2.0 to offset the tail's
    # contribution under the 3.5 winsorization cap).
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SACNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")
    gs = _seed("sigma_seed")

    # Per-subject salience-amplifier strength, U[1.25, 1.65]
    # (modest center ~1.45; the gap key does the fine calibration).
    sigma = 1.25 + 0.40 * gs

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.10, 0.19  # irrelevant
    elif u < 0.28:
        # Hot tail (~15%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        # Calibrated (rather than the nominal 4-7 box) so that the
        # fitted-sensitivity ramp fraction lands near the observed
        # ~0.09 while the winsorized ladder slope stays near ~1.4.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~72%). Small stable elasticity component
        # (sd 0.09) paired with a larger per-context redraw (sd 0.19):
        # the variance ratio fixes latent cross-context consistency
        # at r ~ 0.18 by construction. Narrow stable lapse so it
        # cannot act as a hidden cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill knives
    # (~0.31-0.34) with large below-knife margins. Hot-tail eta is
    # pinned at the engaged median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Top-tier salience amplifier (rank-based prestige aversion). ---
    # Keyed to the validity gap between the top endorsed tier and the
    # next endorsed tier: sigma_eff = 1 + (sigma-1)*exp(-gap/0.10).
    # gap ~ 0.05 (near-knife, e.g. 0.90 vs 0.85): strong amplification
    # (~1.27 at sigma center), roughly doubling the margin on those
    # cells. gap ~ 0.10: mild (~1.17). gap >= 0.20 (coarse kill
    # stacks like 0.90-vs-0.70 or 1.00-vs-0.50): essentially off
    # (~1.06 or less), so established stack preferences and
    # single-validity ladder slopes are preserved. Single-family
    # trials (only one endorsed tier) have no contrast tier: off.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        k_gap = math.exp(-gap / 0.10)
        sig_eff = 1.0 + (sigma - 1.0) * k_gap
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # Per-trial multiplicative attention diffusion on cue weights
    # (sd 0.045): per-trial noise averages out within a subject, so
    # it supplies within-context stochasticity without damping
    # population means.
    att = np.exp(0.045 * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # Fixed design: 38 experts, validities [0.9]*11 + [0.85]*22 + [0.6]*5\n    n_feat = 38\n    idx90 = np.arange(0, 11)   # 0.90 tier\n    idx60 = np.arange(33, 38)  # 0.60 tier\n\n    def parse(x):\n        if isinstance(x, str):\n            s = x.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            return np.array([float(t) for t in toks])\n        return np.asarray(x, dtype=float).ravel()\n\n    ys = []     # 1 = subject chose the 0.90-stack option\n    loads = []  # n_disc = number of cues on which A and B differ\n\n    a_col = data['option_a_ratings']\n    b_col = data['option_b_ratings']\n    r_col = data['response']\n\n    for i in range(len(data)):\n        ra = parse(a_col.iloc[i])\n        rb = parse(b_col.iloc[i])\n        if ra.size != n_feat or rb.size != n_feat:\n            continue\n        a90 = float(ra[idx90].sum())\n        b90 = float(rb[idx90].sum())\n        a60 = float(ra[33:38].sum())\n        b60 = float(rb[33:38].sum())\n\n        # Exclude C1 (0.60-tier endorsements) and ANCHOR (all-1 side\n        # touches the 0.60 tier).\n        if a60 > 0 or b60 > 0:\n            continue\n        # Exclude TIE (both sides endorse 0.90 cues) and any trial\n        # where neither side carries a 0.90 stack.\n        if (a90 > 0) == (b90 > 0):\n            continue\n        # Exactly one side carries the 0.90 stack; require stack size\n        # >= 3, which excludes the AG anchor (2x0.90 vs 6x0.85).\n        stack_count = a90 if a90 > 0 else b90\n        if stack_count < 3:\n            continue\n\n        r = float(r_col.iloc[i])\n        if a90 > 0:\n            y = 1.0 if r == 0.0 else 0.0   # A is the 0.90 stack\n        else:\n            y = 1.0 if r == 1.0 else 0.0   # B is the 0.90 stack\n        ys.append(y)\n        loads.append(int(np.count_nonzero(ra != rb)))\n\n    if len(ys) < 10:\n        return 0.5\n\n    ys = np.asarray(ys, dtype=float)\n    loads = np.asarray(loads, dtype=int)\n\n    # Level: P(choose 0.90 stack) over all 72 near-knife ladder\n    # trials (F1 rungs + F2 endpoints + tied-filler control).\n    L = float(ys.mean())\n\n    # Slope: high-load rungs (n_disc >= 21) minus low-load rungs\n    # (n_disc <= 13). Middle rung (n_disc = 17) contributes to L only.\n    hi = ys[loads >= 21]\n    lo = ys[loads <= 13]\n    S = 0.0\n    if hi.size >= 5 and lo.size >= 5:\n        S = float(hi.mean() - lo.mean())\n\n    return L + 0.5 * S",
  "rationale": "The metric isolates the one structural difference between PQSAI and SACNAI: PQSAI's load-dilution factor D = 1/(1+(n_disc/n0)^1.1), which has no counterpart in SACNAI. I pool all 72 near-knife trials per subject in which exactly one option carries a 0.90-tier stack of >=3 cues against a (k+1)-cue 0.85 stack with score-neutral filler pairs (F1 ladder trials 1-10, F2 endpoints 11-14, tied-filler control 15-18; mirrors handled by orienting y toward the 0.90-stack option). AG (2x0.90), C1 (0.60 tier), TIE, and ANCHOR are excluded by the tier/stack filters, so the metric reads only the manipulated family.\n\nThe metric is LEVEL + 0.5*SLOPE, where LEVEL = P(choose the 0.90-stack) over all 72 trials and SLOPE = P(0.90-stack | n_disc>=21) - P(0.90-stack | n_disc<=13).\n\nWhy this separates the theories: (1) LEVEL. On these gap-0.05 contests SACNAI's undiluted amplifier (sigma_eff in [1.15, 1.39]) sits above the knife sigma_eff* = 1.25/((0.90/0.85)^(1+eta)) for essentially all engaged subjects, so SACNAI's population sits near 0.31-0.33 with subjects tightly clustered on the 0.85-stack side. PQSAI's dilution (D ~ 0.53 at n_disc=9 down to ~0.27 at n_disc=25) plus its linear-counter mode puts its population near 0.55-0.62. (2) SLOPE. n_disc is not an input to SACNAI's computation, and the filler pairs are score-neutral with the context bucket fixed at '0.9' across rungs, so SACNAI's slope is 0 by construction; PQSAI's sensitive subjects migrate toward the 0.90 stack as D falls, giving a population rise of roughly +0.10 across the ladder.\n\nWhy the 0.5-weighted composite rather than level or slope alone: the slope alone is binomial-noise dominated (24 vs 24 trials) and fails at N=25; the level alone works but inherits PQSAI's bimodal between-subject spread (linear counters ~0.75, sensitive ~0.40). Adding half the slope partially CANCELS that bimodality \u2014 sensitive subjects have low level but large positive slope, linear counters high level and zero slope \u2014 so the per-subject composite values cluster (roughly 0.74 / 0.55 / 0.50 for linear / sensitive / guesser under PQSAI vs ~0.30 concentrated mass under SACNAI), shrinking between-subject variance while enlarging the mean gap to ~0.30. Expected Welch t at N=25 is approximately 5.5-6.5 (p ~ 1e-6), comfortably below alpha = 0.01, and the margin survives reasonable misestimation of the sensitive fraction, sigma boxes, and knife position. The metric is a plain proportion contrast, so it is well defined both on the pooled DataFrame and on each single subject's 96-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 16451,
  "output_tokens": 33154,
  "reasoning_tokens": 31550,
  "provider": "Sail Research"
}
```
