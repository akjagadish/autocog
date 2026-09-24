# metric_exp00_attempt_00

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
**Validities (n_features=28):** [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 23: A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** STRUCTURAL SETUP. RSD-vgp and JU-AEI share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5) with E = sum_j v_j^gamma*(b_j - a_j), RAW (unnormalized) weights, and per-cell trials are i.i.d. under BOTH theories (JU-AEI's per-trial gamma_t jitter and both theories' m-mixing marginalize into each cell's Bernoulli p, so there is NO supra-binomial trial-to-trial signature -- correctly not claimed). The beta boxes are IDENTICAL (log-uniform [1.8, 6]) and the A_eff boxes are IDENTICAL (m in [0.74, 1], epsilon in [0.03, 0.11]), so precision and extremity offer NO levers: exact-E beta cells cannot dissociate the populations and are included only as measurement anchors. The theories differ in exactly two places. (i) GAMMA SUPPORT: RSD-vgp's two-component gamma mixture is algebraically EXACTLY Uniform[0, 2] (density 0.5 throughout, hard edge at 2.0); JU-AEI's habitual gamma is Uniform[0, 2.2] (density 1/2.2 = 0.4545, hard edge at 2.2). JU-AEI's gamma distribution FIRST-ORDER STOCHASTICALLY DOMINATES RSD's, and places 9.1% of subjects in (2.0, 2.2] where RSD places exactly ZERO. (ii) PER-TRIAL GAMMA JITTER: with probability delta in [0.13, 0.16] JU-AEI redraws gamma_t uniformly from [gamma_habit - 0.50, gamma_habit + 0.50] clipped to [0, 2.4] -- a ~0.145-weighted boxcar smoothing (half-width 0.5 in gamma-space) of each subject's rung psychometric that fixed-gamma RSD cannot produce. The design makes both boundaries decisive.

ARCHITECTURE (28 experts): five cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- a gamma-invariant reference block of weight exactly 5) and 23 cues at 0.50 (weights halve exactly per unit gamma). On every rung trial R(m) option A is endorsed on the five unit cues (endorsement mass exactly 5 for all gamma) and option B on m of the 0.50-cues, so the evidence is EXACTLY E(gamma) = m*2^{-gamma} - 5, strictly decreasing with a single sign change at flip point g* = log2(m/5): gamma < g* subjects prefer the count-sparser unit-cue side A; gamma > g* prefer the low-validity-dense side B. Shared/tied cues contribute exactly 0. Rungs: m=12 (g*=1.2630), 14 (1.4854), 16 (1.6781), 18 (1.8480), 19 (1.9260), 20 (2.0000), 21 (2.0704), 23 (2.2016) -- deliberately concentrated in the upper region where the two survival curves diverge most, with the top three rungs straddling RSD's 2.0 edge and JU-AEI's 2.2 edge.

DISSOCIATION 1 -- HARD TAIL FLOOR (headline). At R21, E(gamma) = 21*2^{-gamma} - 5 >= 21*0.25 - 5 = +0.25 > 0 for EVERY gamma in [0, 2], and m >= 0.74 > 0.5 keeps p(unit side) >= 0.5 strictly, so NO RSD subject can prefer the dense side on R20 or R21 (R20: E >= 0 with equality only at the measure-zero point gamma = 2; the most extreme RSD subjects, gamma within 0.01 of 2.0, sit within 0.01 of exactly 0.5, and even the worst case at R21 -- gamma = 2, beta = 1.8, A_eff = 0.427 -- gives p = 0.547). JU-AEI REQUIRES (2.2 - g*)/2.2 of subjects with true p(unit side) < 0.5: 9.1% at R20, 5.9% at R21. Crucially these tail subjects are individually identifiable NOT by the shallow tail cells alone but by their FULL-LADDER pattern: a subject with gamma_h > 2.0704 inverts EVERY rung R12-R21 (deeply at low rungs: gamma_h = 2.1 gives E = -2.20 at R12, p(unit side) ~ 0.15-0.25) while still NOT inverting the double-floor control R23 -- a 56-observation fingerprint no RSD subject can produce at any parameter. With N = 100, JU-AEI expects ~6 such subjects; a hard zero falsifies JU-AEI; any credible one falsifies RSD-vgp categorically. Depth of the deepest tail subjects (gamma_h = 2.2, beta = 6, A_eff = 0.7, jitter-marginalized): p(unit side) ~ 0.20 at R19, ~ 0.25-0.33 at R20/R21 -- well below RSD's hard floor.

DISSOCIATION 2 -- SURVIVAL-CURVE / GAMMA-DENSITY PROFILOMETRY. The population inversion rate at each rung equals the survival function S(g*) = P(gamma > g*): RSD = (2 - g*)/2 = 0.369/0.257/0.161/0.076/0.037/0.000/0.000/0.000 over g* = 1.263/1.485/1.678/1.848/1.926/2.000/2.070/2.202 (exactly linear, slope -0.5, hard edge at 2.0); JU-AEI = (2.2 - g*)/2.2 = 0.426/0.325/0.237/0.160/0.125/0.091/0.059/0.000 (slope -0.4545, hard edge at 2.2). JU-AEI is higher at EVERY rung (stochastic dominance -- a one-sided ordering test across the whole ladder), with the ratio growing 1.16 -> 1.26 -> 1.47 -> 2.11 -> 3.38 -> infinity -> infinity down the ladder. Per-subject crossing points (from 8 rungs x 8 observations = 64 rung trials) give the gamma-hat histogram: RSD = flat density 0.5 with hard edge at 2.0; JU-AEI = flat density 0.4545 with hard edge at 2.2. The decisive bin contrast: mass in (2.0, 2.2] = 9.1% (JU) vs 0% HARD (RSD); the 10% density difference below 2.0 is a secondary, N-hungry slope test honestly flagged as weak. Caveat: JU-AEI's jitter shifts the effective marginal-inversion threshold up by ~0.01-0.02 gamma and leaves inverters within ~0.1 of g* at p in (0.45, 0.5) -- shallow inverters contribute to pooled rates but are not individually credible, so the survival curve must be estimated hierarchically, not by naive per-subject thresholds.

DISSOCIATION 3 -- THE DELTA JITTER (tertiary). With per-subject (beta, A_eff) pinned independently by the B1/B2 exact-E cells and the saturated U cells, each subject's rung psychometric under RSD is a SINGLE-gamma logistic in E; under JU-AEI it is a 0.145-weighted boxcar smoothing of half-width 0.5 in gamma-space (clipped at [0, 2.4]). The boxcar produces 'leakage': a high-beta JU subject who is near-deterministic on one side of the ladder still shows ~7-14% opposite-side choices on rungs within 0.5 gamma of its habitual crossing, where a high-beta RSD subject is near-deterministic. Maximum per-cell leakage is ~0.05 in p (delta * A_eff / 2), so this signature is population-level only (pooled SE ~ 0.016 at N = 100); the population delta-hat is 0 under RSD and ~0.145 under JU-AEI. Honest power note: this is a corroborating, not primary, weapon.

CONTROLS. R23 (g* = 2.2016 > 2.2) is a DOUBLE-FLOOR adversarial cell: E(2.2) = +0.0057 > 0 and E is decreasing, so no JU-AEI subject's marginal p can cross 0.5 (verified worst case: gamma_habit = 2.2, beta = 6 gives p = 0.512) and no RSD subject's either (E(2.0) = +0.75) -- any credible R23 inversion falsifies BOTH theories. B1/B2 use only unit cues with exactly canceling shared endorsements, giving E = +1 and +2 EXACTLY for every subject at every gamma; the per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is A_eff-free, lapse-free, gamma-free, and -- because the beta boxes are identical -- lies in [1.005, 1.322] under BOTH theories (no dissociation; used only to pin per-subject beta for the delta analysis). U/Um are conflict-style saturated cells (A endorsed on a single 0.50-cue, B on all 27 others; |E| = 5 + 21*0.5^gamma >= 7.4 at gamma = 2.4, beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 with both bars nonempty (no empty-bar artifact). T1/T2 are identical non-trivial vectors: both theories predict exactly 50% (key-bias check). The ladder itself rules out pure counting (predicts the 5-endorsement side always, flat) and conventional pro-endorsement polarity (predicts the m-endorsement side always, flat). Every informative trial has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs) or sparse side (B cells, U).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject; 64 rung observations per subject. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff, delta) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6], delta = 0; JU-AEI: gamma_habit U[0,2.2], delta in [0.13,0.16] with the +/-0.50 window clipped to [0,2.4]), plus simulation-based model recovery. Pooled mid-box planning values (approximate, A_eff = 0.69, beta = 3.3), p(unit-cue side), RSD vs JU-AEI: R12 0.62/0.57; R14 0.68/0.63; R16 0.73/0.68; R18 0.76/0.71; R19 0.79/0.74; R20 0.80/0.76; R21 0.80/0.76; R23 0.82/0.79; B1 ~0.82 both; B2 ~0.83 both; U ~0.845 both; ties 0.50 both. Pooled differences (0.03-0.05) are modest and honestly acknowledged: the populations' central tendencies nearly coincide, and the decisive evidence is SUBJECT-LEVEL population structure.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly preferring the low-validity-dense side on R20 or R21 categorically falsifies RSD-vgp (E > 0 over its entire gamma box; hard floor at 0.5), while JU-AEI requires 9.1%/5.9% of subjects there, identifiable by the full-ladder inverter pattern (inverts R12-R21, deep at low rungs, does NOT invert R23); (b) inversion-rate survival curve over g* = 1.263/1.485/1.678/1.848/1.926/2.000/2.070/2.202: RSD 0.369/0.257/0.161/0.076/0.037/0/0/0 (linear, slope -0.5, hard edge at 2.0) vs JU-AEI 0.426/0.325/0.237/0.160/0.125/0.091/0.059/0 (higher at EVERY rung, slope -0.4545, hard edge at 2.2); (c) per-subject gamma-hat histogram: 9.1% of mass in (2.0, 2.2] (JU) vs 0% hard (RSD), density 0.4545 vs 0.5 below 2.0; (d) population delta-hat = 0 (RSD) vs ~0.145 with boxcar shape (JU-AEI), given per-subject beta pinned by r12; (e) any credible R23 inversion falsifies BOTH theories (adversarial double-floor); (f) r12 in [1.005, 1.322] under both (shared, no dissociation -- honest); (g) U reads A_eff in [0.714, 0.985] under both, ties at exactly 50% (shared controls).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI, clip-restoration pass). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with A_eff = (1-eps)(2m-1) near 0.69 (steep normalized slope x moderate asymptote). The population: habitual gamma ~ U[0, 2.2] (single continuous uniform — no gaps, no cliffs, no narrow tail segment), beta log-uniform on [1.8, 6.0] independent of gamma. The novel mechanism is per-trial stochastic validity-attention jitter: with per-trial probability delta ~ 0.13-0.16, the effective gamma for that trial is redrawn uniformly from a SYMMETRIC window [gamma_habit - 0.50, gamma_habit + 0.50] clipped to [0, 2.4] — the arbiter's own prescribed geometry. The critical factorization this theory embodies: TRUE habitual mass above 2.2 is poison (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment and destroys Exp 18's negative offset), while per-trial EXCURSIONS reaching 2.4 from habits below 2.2 are safe and are the only channel that moves the apparent-tail estimator (Exp 14) above 0.17. There is NO per-trial log-beta jitter: five passes of loop evidence establish that any per-trial precision noise flattens per-subject gamma posteriors toward the estimator's EB prior mean and caps the tail estimate at ~0.12, while gamma-only jitter at delta ~0.14 simultaneously satisfies the BF softening (Exps 15/9/12) and the flip-rate anchor (Exp 16). Falsifiable signatures: apparent tail-mass estimates ~0.17-0.20 on shallow-slope designs; positive RSD-vs-EPAI log BF ~[4, 10]; flip rates in [0.43, 0.45]; crossing statistic in [-0.15, -0.10]; steep-slope posterior >= 0.88; no plateau in recovered gamma rung ladders.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- delta: [0.13, 0.16]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI),
    # clip-restoration pass: habitual support held at U[0, 2.2], the
    # per-trial gamma-excursion clip restored to the arbiter's own [0, 2.4],
    # gamma-only jitter (NO per-trial log-beta noise), symmetric +/-0.50
    # window, delta in [0.13, 0.16].
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"JU-AEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- (1) GAMMA SUPPORT: single continuous uniform on [0, 2.2] ----
    # Held exactly at [0, 2.2]: TRUE habitual mass above 2.2 is poison
    # (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment on
    # Exp 13 and inflates Exp 18's p_hi term). Apparent tail mass is
    # manufactured purely by the per-trial excursions below.
    gamma_habit = 2.2 * float(parameters["gamma_u"])

    # ---- (2) PRECISION: log-uniform raw inverse temperature, ----
    # ---- independent of gamma (no segment-coupled beta boxes)  ----
    # NO per-trial log-beta jitter: five loop passes establish that any
    # per-trial precision noise flattens per-subject gamma posteriors
    # toward the Exp 14 estimator's EB prior mean (0.075), capping the
    # tail estimate at ~0.12, and pushes Exp 16's flip rate toward 0.5.
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # ---- (3) PER-TRIAL STOCHASTIC VALIDITY-ATTENTION JITTER ----
    # Gamma-only. With per-trial probability delta in [0.13, 0.16] the
    # effective gamma is redrawn uniformly from a SYMMETRIC window
    # [gamma_habit - 0.50, gamma_habit + 0.50], clipped to [0, 2.4]
    # (the arbiter's own prescribed geometry). The 2.4 up-clip is the
    # single change most implicated in the previous pass's rejection:
    # excursions reaching 2.4 from habits in [1.7, 2.2] are what the
    # fixed-gamma per-subject MLE reads as apparent tail mass (the only
    # channel that has ever moved Exp 14 above 0.17), while habitual
    # RESIDENCE above 2.2 (removed here by keeping the support at 2.2)
    # is what flipped Exp 13's sign in the iter-6 pass.
    delta = float(parameters["delta"])
    if np.random.random() < delta:
        lo = max(0.0, gamma_habit - 0.50)
        hi = min(2.4, gamma_habit + 0.50)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma_habit

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma_t)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster. UNCHANGED (Exps 7/8 anchors are exact).
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Top-rung dense-preference ceiling probe (RSD-vgp vs JU-AEI).\n\n    Rung trials: one option ('sparse' side) endorsed on exactly the five\n    unit-validity cues (features 0-4) and nothing else; the other ('dense'\n    side) endorsed on m of the 23 half-validity cues and nothing else.\n    The anti-endorsement evidence for the sparse side is\n        E(gamma) = m * 2**(-gamma) - 5,\n    strictly decreasing in gamma (unit cues have weight 1 for every\n    gamma; half cues have weight 2**(-gamma)).\n\n    Per-subject score:  s = r(19) + r(20) + r(21), the summed rate of\n    dense-side choices on the three highest non-control rungs (m = 19,\n    20, 21; the m = 23 cell is the double-floor control and is excluded).\n\n    HARD CEILING UNDER RSD-vgp: every RSD-vgp subject has gamma <= 2, and\n    at gamma = 2 the three evidences are exactly (-0.25, 0, +0.25); by the\n    odd symmetry of the logistic,\n        p19 + p20 + p21 = 1.5  EXACTLY at gamma = 2,\n    and strictly less for every gamma < 2, for every beta, lapse and\n    polarity parameter (A_eff >= 0 multiplies a nonpositive sum). So\n    s <= 1.5 is a hard, parameter-free, per-subject ceiling under\n    RSD-vgp; observed values above it are pure binomial noise.\n\n    UNDER JU-AEI: 9.1% of subjects have habitual gamma in (2.0, 2.2];\n    all three top-rung evidences are then negative and s rises up to\n    ~2.6 (gamma_h = 2.2, high beta, high A_eff, jitter-marginalized).\n\n    Applied to a single subject's slice: returns that subject's s.\n    Applied to pooled multi-subject data: returns the population MAXIMUM\n    of s -- the natural extreme-value test of a hard bound.\n    \"\"\"\n    df = data.reset_index(drop=True)\n    n = len(df)\n    if n == 0:\n        return 0.0\n\n    subj = df[\"subject_id\"].to_numpy()\n    resp = df[\"response\"].to_numpy()\n\n    m_arr = np.full(n, -1, dtype=int)\n    dense_arr = np.full(n, -1, dtype=int)\n\n    for i in range(n):\n        try:\n            va = np.asarray(list(df.at[i, \"option_a_ratings\"]), dtype=int)\n            vb = np.asarray(list(df.at[i, \"option_b_ratings\"]), dtype=int)\n        except Exception:\n            continue\n        if va.size != vb.size or va.size < 6:\n            continue\n        ua = int(va[:5].sum())\n        ha = int(va[5:].sum())\n        ub = int(vb[:5].sum())\n        hb = int(vb[5:].sum())\n        # rung trial: sparse side = exactly the 5 unit cues, dense side =\n        # only half-cue endorsements (m of them). Mirrored twins both match.\n        if ua == 5 and ha == 0 and ub == 0 and hb >= 6:\n            m_arr[i] = hb                      # B is the dense side\n            dense_arr[i] = 1 if int(resp[i]) == 1 else 0\n        elif ub == 5 and hb == 0 and ua == 0 and ha >= 6:\n            m_arr[i] = ha                      # A is the dense side\n            dense_arr[i] = 1 if int(resp[i]) == 0 else 0\n\n    sel = m_arr >= 0\n    if not sel.any():\n        return 0.0\n\n    frame = pd.DataFrame({\n        \"subject_id\": subj[sel],\n        \"m\": m_arr[sel],\n        \"dense\": dense_arr[sel],\n    })\n\n    observed = sorted(int(x) for x in pd.unique(frame[\"m\"]))\n    target = [x for x in (19, 20, 21) if x in observed]\n    if len(target) < 3:\n        # fallback: three largest rung levels strictly below the top rung\n        below = [x for x in observed if x < observed[-1]]\n        pool = below if len(below) >= 3 else observed\n        target = pool[-3:]\n\n    sub = frame[frame[\"m\"].isin(target)]\n    if len(sub) == 0:\n        return 0.0\n\n    per = (\n        sub.groupby([\"subject_id\", \"m\"])[\"dense\"].mean()\n           .groupby(level=0).sum()\n    )\n    if len(per) == 0:\n        return 0.0\n    return float(per.max())",
  "rationale": "RATIONALE. The two theories share the anti-endorsement core, the same beta box (log-uniform 1.8-6), the same A_eff box, and per-cell i.i.d. sampling, so the ONLY genuine population difference is the validity-exponent support: RSD-vgp places gamma on [0, 2] (its two-component mixture is exactly Uniform[0,2]), while JU-AEI places habitual gamma on [0, 2.2]. The rung family was built so that the top rungs convert this support difference into a HARD, parameter-free behavioral bound.\n\nTHE HARD CEILING. On a rung trial with m half-cue endorsements versus 5 unit-cue endorsements, the sparse-side evidence is E(gamma) = m*2^(-gamma) - 5. For the three top non-control rungs m = 19, 20, 21 at gamma = 2 the evidences are exactly (-0.25, 0, +0.25). Since p_m = 0.5 - A_eff*(sigmoid(beta*E_m) - 0.5) with A_eff = (1-eps)(2m-1) >= 0.427 > 0, odd symmetry of the logistic gives p19 + p20 + p21 = 1.5 EXACTLY at gamma = 2, and monotonicity of E in gamma gives s = p19+p20+p21 < 1.5 strictly for every gamma < 2, for EVERY beta, lapse and polarity parameter in both theories' boxes. So under RSD-vgp the per-subject score s = r(19)+r(20)+r(21) (each r the dense-side choice rate over the 8 mirrored repetitions) is bounded above by 1.5 as a mathematical identity; any observed excess is binomial noise (per-subject noise SD ~ 0.3 for near-ceiling subjects, ~ 0.22 for typical subjects whose true s ~ 0.47).\n\nWHY JU-AEI MUST BREAK THE CEILING. JU-AEI's habitual gamma ~ U[0, 2.2] puts 9.1% of subjects in (2.0, 2.2], where all three top-rung evidences are negative and s exceeds 1.5 - e.g. gamma_h = 2.1 with median beta/A_eff gives s ~ 1.85, and the corner (gamma_h = 2.2, beta = 6, A_eff = 0.97, jitter-marginalized) gives s ~ 2.6. The m = 23 double-floor control is deliberately excluded (both theories are bounded there), and m = 18 is excluded because RSD-vgp subjects with gamma in (1.848, 2] legitimately prefer the dense side there. Jitter cannot manufacture false violations: for gamma_h <= 2 the boxcar window average of the three rates stays below 1.5, so ceiling violations identify true habitual gamma > 2 - exactly the mass RSD-vgp assigns probability zero.\n\nWHY THE MAXIMUM. A hard bound is tested by an extreme-value statistic, not a mean: the population maximum of s is the canonical readout of 'does any subject violate the ceiling, and by how much'. Pooled over the sample: RSD-vgp predicts max_i s_i ~ 1.8-2.1 (true values capped at 1.5 plus binomial extremes over the ~5% of subjects with gamma in [1.9, 2]); JU-AEI predicts max_i s_i ~ 2.4-2.9 (its 9.1% tail contains subjects with true s up to 2.6). The gap (~0.6-1.0) is large relative to the between-subject spread of s (per-subject SD ~ 0.36 under RSD-vgp, ~ 0.53 under JU-AEI; per-subject values are well-defined on a single 96-trial slice, so the variance is computable). At N = 25 this gives a Welch t of roughly 4-8, i.e. the two theories produce visibly different numbers on this metric, whereas every mean-based pooled statistic on this design is capped near a signal-to-noise ratio of ~0.4 because the discriminating subpopulation is only ~9% of subjects.\n\nPREDICTIONS. RSD-vgp (advocated): point estimate ~1.8-2.1, hard-limited; any human subject with s credibly above ~2.1 (e.g. 19+ dense choices out of the 24 top-rung trials) falsifies RSD-vgp categorically. JU-AEI: point estimate ~2.4-2.9, driven by 2-3 ceiling-violating subjects per 25 whose full-ladder fingerprint (dense-preferring at ALL rungs R12-R21 yet NOT inverting the R23 double-floor) has no RSD-vgp explanation at any legal parameter. This is the designed Dissociation 1 (hard tail floor) rendered as a single scalar."
}
```

## Usage

```json
{
  "input_tokens": 12509,
  "output_tokens": 105445,
  "reasoning_tokens": 103146,
  "provider": "Sail Research"
}
```
