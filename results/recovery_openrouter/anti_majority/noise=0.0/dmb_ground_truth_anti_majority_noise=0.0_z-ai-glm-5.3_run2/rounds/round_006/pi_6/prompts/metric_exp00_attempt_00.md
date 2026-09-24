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
**Validities (n_features=25):** [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.75, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 21: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. CNNAI (advocated) and CAVI-K (competing) are near-degenerate in ordinary mixed-validity trials: both are anti-validity additive integrators (a 1 is a defect, tied cues cancel), both use the identical bucketed-mean-endorsed-validity context key with a per-bucket eta redraw, both have softmax + symmetric lapse + a ~13-14% guesser subpopulation, and their effective validity exponents overlap almost everywhere (CNNAI's 1+eta ~ N(1.12, 0.21) vs CAVI-K's tau-implied ~1.26-1.42 plus zero-centered eta noise). Experiments [10]-[12] already took the context-redraw, beta-bloc, and compression-knee wedges. The ONE hard structural difference no prior design has isolated is WHERE each theory's supralinearity lives. CNNAI's curvature is ENTIRELY context-relative: w_j = -v_j*(v_j/v_ref)^eta, so when every cue endorsed on a trial shares a single validity v, v_ref = v and (v_j/v_ref)^eta = 1 for EVERY eta — CNNAI's per-defect weight collapses to c(v) = v EXACTLY, with zero free parameters: the entire eta machinery (bucket redraws, stable component, knife-avoidance centering at 0.12) is provably inert. CAVI-K's supralinearity is context-INVARIANT: d_j = v_j + tau*v_j^2, and the tau*v^2 term never vanishes in any context — tau in [0.35,0.65] (engaged) and [0.55,0.65] (deterministic tail) is always engaged. THE INSTRUMENT: a validity ladder of pure count-discrimination trials. On each trial, option A endorses 2 cues and option B endorses 1 cue, ALL from a single validity family v in {1.00, 0.90, 0.75, 0.60, 0.50} (all other experts rated 0 by both options). Both theories prefer B (fewer defects); the measurement is CONFIDENCE as a function of v. Per-trial core logit = beta*c(v). Within-subject logit ratios cancel beta and lapse, isolating c(v). KILL RATIOS (parameter-free): c(1.0)/c(0.5) = 2.000 EXACTLY for every engaged CNNAI subject (any beta, any eta, any bucket draw, any attention realization — attention is mean-1 multiplicative and averages over 16 trials/family), versus (1+tau)/(0.5+0.25*tau) in [2.298, 2.490] for EVERY non-guesser CAVI-K subject (monotone increasing in tau; deterministic tail >= 2.43). The band (2.05, 2.25) is EMPTY for both theories — a per-subject and population classifier with no free parameters. Mid rungs replicate the gap: c(0.9)/c(0.5) = 1.80 (CNNAI) vs [2.02, 2.15] (CAVI-K), empty band (1.85, 2.00); c(0.75)/c(0.5) = 1.50 vs [1.61, 1.68], empty band (1.53, 1.60). SUB-KNEE GUARANTEE: maximum raw defect load on any trial is 2*(1+0.65) = 3.30 < 4.45, safe by >= 24% even under a +4-SD adverse attention draw, so CAVI-K's two-regime compression is ALSO exactly inert (identity regime) — the dissociation isolates the weight map alone, with compression, eta redraws, bucket keys, and v_ref rescaling all neutralized in BOTH theories. POPULATION PREDICTIONS (fully specified mixtures, integrating each theory's stated parameter distributions; P(choose the fewer-defect option) on the (2,1) ladder): CNNAI ~ [0.69, 0.73, 0.77, 0.80, 0.82] for v = [0.50, 0.60, 0.75, 0.90, 1.00]; CAVI-K ~ [0.79, 0.83, 0.86, 0.88, 0.89]. The curves are separated by 0.07-0.10 at EVERY rung with non-overlapping population ranges (CNNAI's ceiling at v=0.50 is ~0.74 even if all engaged subjects sat at their 84th-percentile beta; CAVI-K's floor is ~0.76): CAVI-K is pinned more confident everywhere because its c(0.5) >= 0.5875 > 0.50 and its beta median 2.65 > 2.14 are hard-coded. SECONDARY STATISTICS from the same trials: (S1) DETERMINISTIC-TAIL BLOC: on the v=0.60 (2,1) rung (16 trials/subject), CAVI-K's 16% deterministic tail (beta 4.0-6.5, eps 0.02-0.04) sits at per-trial p ~ 0.96-0.99, so ~13-17% of subjects choose B on >= 15/16 trials; CNNAI's homogeneous beta (98% below 3.2, clip 4.5) caps engaged per-trial p at ~0.91, predicting a bloc fraction of ~0.01. (S2) LAPSE/CEILING CELL: the (2,0) pair at v=1.00 (A shows two 1.00-cue 1s, B shows none) drives both theories near ceiling (CNNAI pop ~0.89, CAVI-K ~0.91), pinning each subject's lapse for the ratio correction, and gives a Delta=2 replication of the v-ordering. (S3) RATIO DISPERSION: CNNAI's engaged subjects ALL have ratio exactly 2.00 (between-subject sd ~ 0 from the map; only binomial/attention noise), whereas CAVI-K's tau spread induces sd ~ 0.06 plus a visible deterministic-tail mode near 2.45. CONTROLS AND THIRD-PARTY SCREENS: the ladder's RISING profile itself screens validity-blind strategies — an unweighted counter (c(v) constant) predicts a FLAT ladder (ratio 1.0); a pro-validity integrator prefers the option with MORE high-validity 1s (P < 0.5 at every rung) and is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(choose all-1) ~ 0.09 (guessers at 0.5) while tallying/pro-validity predicts ~1; TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both; every critical pair is mirrored (swapping A/B leaves counts, v_ref, and the bucket key unchanged), absorbing side bias. ROBUSTNESS: per-trial attention diffusion (CNNAI sd 0.05, CAVI-K sd 0.035) is mean-1 multiplicative on weights and averages out over 16 trials/family (second-order softening ~1-2%, applied to both theories' quoted numbers); no feedback keeps weights anchored to the instructed validities; every trial requires scanning all 25 expert rows to locate and compare the sparse 1s, satisfying the min-RT viewing requirement. NOVELTY: [10] probed context redraws (mixed contexts), [11] probed the beta bloc and a mixed-validity knife ladder, [12] probed above-knee compression via tied-filler load. This design neutralizes ALL of those mechanisms and attacks the one remaining hard difference — the context-invariant quadratic term — in the exact regime (single-validity active sets, sub-knee loads) where CNNAI is parameter-free linear and CAVI-K is pinned quadratic. DESIGN: 25 experts (5 at each of 1.00, 0.90, 0.75, 0.60, 0.50); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials (16 ladder trials per validity family, 8 ceiling/lapse trials, 4 tie, 4 anchor). DECISION RULES: (i) CNNAI is supported and CAVI-K falsified if the population (2,1) ladder matches [0.69, 0.73, 0.77, 0.80, 0.82] (+/- 0.045), the lapse-corrected hierarchical weight estimate satisfies c_hat(1.0)/c_hat(0.5) ~ 2.00 with CI excluding 2.25 and c_hat(0.9)/c_hat(0.5) ~ 1.80 below 2.0, the v=0.60 bloc fraction is <= 0.05, and between-subject ratio dispersion is near zero — CAVI-K cannot produce a ratio below 2.298 for ANY admissible (tau, kappa, beta, epsilon, eta) in any regime. (ii) CAVI-K is supported and CNNAI falsified if the ladder matches [0.79, 0.83, 0.86, 0.88, 0.89], the ratio is >= 2.28 with c_hat(0.9)/c_hat(0.5) >= 2.0, and ~12-18% of subjects show the near-ceiling bloc on the v=0.60 rung — CNNAI's exact-linear map (c(v) = v with zero free parameters in these contexts) cannot produce ratio 2.3, and its homogeneous beta cannot produce the bloc. (iii) BOTH are falsified if the ladder is FLAT (validity-blind counting), P(fewer-defect option) < 0.5 (pro-validity), the all-1 option is chosen well above 0.15 on ANCHOR (tallying/pro-validity), TIE deviates from 0.5 (side bias), or the recovered weight profile fits neither a line through the origin nor the quadratic family.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

**Parameters:**
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 6 -- knife-avoidance recalibration of the accepted iter-5 base
    # per the iter-5 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta with v_ref = mean validity of the
    # endorsed cues, per-trial attention diffusion, softmax + symmetric
    # lapse, guesser mixture. Changes vs the accepted iter-5 base:
    #   (A) ETA CENTER LOWERED 0.23 -> 0.12 (eta_base ~ N(0.02, 0.09) plus
    #       per-context redraw ~ N(0.10, 0.19), total clipped [-0.25, 0.70]).
    #       The kill composites (Exps 5/6/8) are near-knife statistics:
    #       their flip knives sit at eta ~ 0.30-0.34, and on the
    #       below-knife side the score margins GROW as eta decreases
    #       (e.g. Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23
    #       vs 0.33 at eta 0.10). Centering at 0.12 both moves engaged
    #       mass below the knives (P(eta < 0.33) ~ 0.83) and enlarges
    #       the margins, lifting the undershooting composites.
    #   (B) ATTENTION DIFFUSION CUT 0.10 -> 0.05: at 0.10 the per-trial
    #       margin noise (sd ~ 0.25 on a 12-cue cell) was comparable to
    #       the near-knife signals (~0.15-0.35), damping below-knife
    #       subjects to p ~ 0.55-0.6 -- the main softening channel on
    #       Exps 5/6/8. Per-trial noise averages out within a subject
    #       (iter-2 critic), so cutting it lifts means without touching
    #       between-subject variance.
    #   (C) CONTEXT KEY COARSENED to the mean endorsed validity rounded
    #       to 0.1 (a bucketed v_ref / ladder identifier). The critic's
    #       'rounded MAX endorsed validity' variant was NOT used: both
    #       Exp-9 ladders ({1.00,0.50} and {1.00,0.60} cells) have max
    #       validity 1.0, so a max-based key would map them to the SAME
    #       draw and re-couple them -- recreating the iter-1 r = 0.44
    #       failure the critic itself warns against. Bucketed mean keeps
    #       them distinct (0.6 vs 0.7) while merging within-experiment
    #       fragmentation (Exp 2: ~2-3 keys instead of ~6; Exp 5: 2 keys),
    #       so per-subject composites stop averaging over many
    #       independent eta draws.
    #   (D) EXP-9 COUPLING CHANNELS CLOSED: engaged epsilon narrowed
    #       0.02-0.17 -> 0.03-0.09 (stable lapse was the dominant
    #       residual cross-ladder trait), eta_base sd cut 0.13 -> 0.09
    #       (latent r = 0.0081/(0.0081+0.0361) ~ 0.18), beta median
    #       trimmed 2.34 -> 2.14 with log-sd 0.25 -> 0.20 (guards Exps
    #       1/3/10 against the noise cut, and removes the last sizeable
    #       stable-trait correlation channel).
    #   (E) v_ref STAYS MEAN-REFERENCED (not max): the weight ratio
    #       w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is invariant to v_ref, and
    #       max-referencing multiplies every margin by v_max^(-eta) <= 1,
    #       which would DEFLATE exactly the near-knife margins the
    #       composites need. The critic's goal (steeper high-validity
    #       region) is achieved via (A)+(B) instead.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). Small stable elasticity
        # component (sd 0.09) paired with a larger per-context redraw
        # (sd 0.19): the variance ratio fixes latent cross-context
        # consistency at r ~ 0.18 by construction. The stable lapse
        # range is NARROW (0.03-0.09) so it can no longer act as a
        # hidden coupling channel across contexts, and beta is nearly
        # homogeneous (lognormal sd 0.20).
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # v_ref = mean validity of the cues actually displayed (endorsed)
    # this trial; the context key is v_ref rounded to one decimal, so
    # all cells of one validity ladder share a single per-subject eta
    # draw while different ladders (e.g. Exp 9's 0.50-ladder at bucket
    # 0.6 vs its 0.60-ladder at bucket 0.7) draw independently.
    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, centered low so the engaged bulk sits below
    # the kill knives (~0.31-0.34) with large below-knife margins:
    # engaged eta ~ N(0.12, 0.21), P(eta < 0.33) ~ 0.83.
    eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights,
    # sd cut 0.10 -> 0.05: at 0.10 the margin noise on near-knife cells
    # was comparable to the signal and was the main softening channel
    # on the Exp 5/6/8 composites.
    att = np.exp(0.05 * np.random.randn(n_features))
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
**Description:** Kneed Compressive Anti-Validity Integration with a Moderate Deterministic Tail (CAVI-K), beta-homogenization refinement. Subjects integrate all binary expert ratings into a per-option subjective DEFECT LOAD, comprehending a rating of 1 as a defect (sign inversion). Per-cue defect magnitude is supralinear in the instructed validity, d_j = v_j + tau*v_j^2 (tau in [0.35, 0.65]), modulated by a context elasticity (v_j/v_ref)^eta keyed on the bucketed mean endorsed validity, with a zero-centered stable component (sd 0.10) plus a per-context redraw (sd 0.20), fixing latent cross-context consistency at r ~ 0.20. The defect load is compressed by a two-regime map with the knee at 4.45 — above every near-knife small stack (AL cells, 2-top-cue adherence cells, which keep full raw margins) and below the large diagnostic stacks (which grow sub-proportionally aversive under kappa in [0.55, 0.68]). Population: ~14% true guessers, ~70% engaged with a HOMOGENIZED lognormal beta (median 2.65, log-sd 0.29, clip [0.8, 6.0]) — the tightened spread removes both the saturating hot-engaged bloc (which was driving the Exp-9 cross-ladder correlation to 0.388 via beta-saturation, not latent consistency) and the soft low-beta tail (which was diluting anti-validity adherence on Exps 2/5/6/8/11) — and ~16% deterministic tail (beta 4.0-6.5, near-linear kappa, pinned eta). Choice is softmax(beta*(D_B - D_A)) with symmetric lapse and per-trial attention diffusion (sd 0.035).

**Parameters:**
- regime_seed: [0, 1]
- tau_seed: [0, 1]
- kappa_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- eta_base_seed: [0, 1]
- subject_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Kneed Compressive Anti-Validity Integration (CAVI-K), iter 5.
    # Minimal single-lever diff on the accepted iter-4 base, applying
    # the iter-4 critic diagnosis. Changes vs the accepted base:
    #   (A) ENGAGED BETA LOG-SD TRIMMED 0.35 -> 0.29 with the median
    #       nudged 2.5 -> 2.65 (clip unchanged [0.8, 6.0]). The Exp-9
    #       trajectory isolated the failure: at a CONSTANT eta split
    #       (0.10/0.20), raising the beta median 2.0 -> 2.5 swung the
    #       cross-ladder correlation 0.067 -> 0.388, i.e. the correlation
    #       is driven by the saturating HOT END of the engaged beta
    #       distribution, not by latent context consistency. Trimming the
    #       log-sd removes that bloc (pulling Exp 9 back toward its
    #       [0.15, 0.30] window) while ALSO removing the soft low-beta
    #       tail that was diluting anti-validity adherence on
    #       Exps 2/5/6/8/11 — the shared undershoot of all four prior
    #       iterations. The small median nudge (2.5 -> 2.65, well short
    #       of the critic's 3.0+ that the AL-sigmoid arithmetic rules
    #       out) targets Exp 11's remaining shortfall (2.24 vs the
    #       [2.4, 3.2] window).
    #   (B) ENGAGED KAPPA UPPER BOUND 0.70 -> 0.68 (range now
    #       [0.55, 0.68]): a single small pre-emptive guardrail step.
    #       Removing the low-beta tail raises mean adherence broadly,
    #       which will push Exp 10's rise (0.350 at iter 4, window
    #       [0.22, 0.36]) toward its ceiling; slightly stronger
    #       compression on the large diagnostic stacks holds it inside
    #       the window. This is deliberately NOT the iter-3 confounded
    #       trim (which fought a simultaneous knee raise) — the knee
    #       stays at 4.45 and only the upper bound moves by 0.02.
    #   Everything else is held verbatim from the accepted iter-4 base:
    #   knee 4.45 (above the max AL stack 4.28 for every tau), tau
    #   [0.35, 0.65], the 0.10/0.20 eta split with pinned tail eta,
    #   attention diffusion 0.035, the 14/70/16 mixture, and all lapse
    #   ranges — each either in-window or explicitly validated at
    #   iter 4.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CAVI-K expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    ts = _seed("tau_seed")
    ks = _seed("kappa_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")
    ebs = min(max(_seed("eta_base_seed"), 1e-6), 1.0 - 1e-6)
    ss = _seed("subject_seed")

    # --- Population mixture ---
    if u < 0.14:
        # True guessers (~14%).
        t = u / 0.14
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es              # 0.12 .. 0.25
        tau = 0.50
        kappa = 0.75
        eta_base = 0.00
    elif u < 0.84:
        # Engaged bulk (~70%): HOMOGENIZED beta (median 2.65, log-sd
        # 0.29), kappa acting only above the knee, zero-centered stable
        # elasticity. The tightened beta spread removes both the
        # saturating hot bloc (Exp 9) and the soft low-beta tail
        # (adherence dilution on Exps 2/5/6/8/11).
        tau = 0.35 + 0.30 * ts                 # 0.35 .. 0.65
        kappa = 0.55 + 0.13 * ks                # 0.55 .. 0.68
        beta = math.exp(math.log(2.65) + 0.29 * float(ndtri(bs)))
        beta = min(max(beta, 0.8), 6.0)
        epsilon = 0.03 + 0.06 * es             # 0.03 .. 0.09
        eta_base = 0.10 * float(ndtri(ebs))
        eta_base = min(max(eta_base, -0.15), 0.15)
    else:
        # Deterministic tail (~16%): hot, low-lapse, strongly inverted,
        # least compressive, PINNED stable elasticity (no ndtri draw) so
        # the tail cannot act as a hidden cross-context coupling channel.
        tau = 0.55 + 0.10 * ts                 # 0.55 .. 0.65
        kappa = 0.78 + 0.07 * ks                # 0.78 .. 0.85
        beta = 4.0 + 2.5 * bs                   # 4.0 .. 6.5
        epsilon = 0.02 + 0.02 * es              # 0.02 .. 0.04
        eta_base = 0.05                         # pinned

    # --- Context: bucketed mean endorsed validity ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, zero-centered; stable/redraw sd ratio 0.10/0.20
    # fixes latent cross-context consistency at r ~ 0.20 by construction.
    eta_ctx = eta_base + 0.20 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.30), 0.50)

    # --- Defect weights: supralinear anti-validity, context-normalized ---
    d = val + tau * np.square(val)
    w = d * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion (sd 0.035, the
    # iter-3-proven level that lets Exp-12 fitted bhat reach the ramp).
    att = np.exp(0.035 * np.random.randn(n_features))
    w = w * att

    # Raw defect loads.
    s_a = float(np.dot(w, ra))
    s_b = float(np.dot(w, rb))

    # TWO-REGIME (offset) compression: identity below the knee, power
    # compression above. Knee 4.45 > max AL stack 4.28 (at tau = 0.65),
    # so near-knife small stacks keep full raw margins for every tau;
    # the large Exp-10/12 stacks (4.7-11.5) are compressed.
    S_KNEE = 4.45

    def _compress(s):
        if s <= S_KNEE:
            return s
        return S_KNEE * ((s / S_KNEE) ** kappa)

    d_a = _compress(s_a)
    d_b = _compress(s_b)

    scores = np.array([-d_a, -d_b])

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # Fixed design: 25 experts, 5 at each validity 1.00/0.90/0.75/0.60/0.50\n    VALS = np.array([1.0] * 5 + [0.9] * 5 + [0.75] * 5 + [0.6] * 5 + [0.5] * 5)\n\n    def _parse(cell):\n        if isinstance(cell, str):\n            toks = [t for t in cell.strip().strip('[]() ').replace(',', ' ').split() if t != '']\n            return np.array([float(t) for t in toks])\n        return np.asarray(cell, dtype=float).ravel()\n\n    def _subject_slope(df):\n        gs, ys = [], []\n        for _, row in df.iterrows():\n            try:\n                a = _parse(row['option_a_ratings'])\n                b = _parse(row['option_b_ratings'])\n            except Exception:\n                continue\n            if a.size != VALS.size or b.size != VALS.size:\n                continue\n            na = int(np.sum(a > 0.5))\n            nb = int(np.sum(b > 0.5))\n            # pure count-discrimination ladder trial: a {2,1} split of ones\n            if (na + nb) != 3 or na * nb != 2:\n                continue\n            active = np.flatnonzero((a > 0.5) | (b > 0.5))\n            if active.size == 0:\n                continue\n            v = VALS[active]\n            # all endorsed cues must come from one validity family\n            if (v.max() - v.min()) > 1e-9:\n                continue\n            vv = float(v[0])\n            # outcome: 1 = chose the option with FEWER endorsed cues (fewer defects)\n            if na == 2:\n                y = 1 if int(row['response']) == 1 else 0\n            else:\n                y = 1 if int(row['response']) == 0 else 0\n            # theory-neutral severity index: midpoint of linear (v) and\n            # fully quadratic (v + v^2) anti-validity maps\n            gs.append(vv + 0.5 * vv * vv)\n            ys.append(float(y))\n        if len(gs) < 12:\n            return 0.5\n        g = np.asarray(gs, dtype=float)\n        y = np.asarray(ys, dtype=float)\n        # logistic regression THROUGH THE ORIGIN: P(fewer-defect) = sigmoid(b * g(v))\n        b_hat = 1.0\n        for _ in range(200):\n            z = np.clip(b_hat * g, -30.0, 30.0)\n            p = 1.0 / (1.0 + np.exp(-z))\n            w = p * (1.0 - p)\n            grad = float(np.sum(g * (y - p)))\n            hess = float(np.sum(g * g * w))\n            if hess < 1e-10:\n                break\n            step = grad / hess\n            b_hat += step\n            if b_hat > 10.0:\n                b_hat = 10.0\n            elif b_hat < -10.0:\n                b_hat = -10.0\n            if abs(step) < 1e-10:\n                break\n        # winsorize: floor bounds pure-noise slopes of chance-level subjects,\n        # cap bounds ceiling-separated (near-deterministic) subjects\n        return float(min(max(b_hat, 0.5), 3.5))\n\n    vals = []\n    for _, df in data.groupby('subject_id', sort=False):\n        vals.append(_subject_slope(df))\n    if not vals:\n        return 0.5\n    return float(np.mean(vals))",
  "rationale": "TARGET: per-subject anti-validity sensitivity estimated from the 80 pure count-discrimination ladder trials (the {2,1}-cue splits within a single validity family), then averaged over subjects.\n\nWHY THIS QUANTITY: On these trials both theories are exactly parameter-free in their context machinery (v_ref = v, so (v/v_ref)^eta = 1 for every eta draw; CAVI-K's compression knee 4.45 is never reached since max raw load is 2*(1+0.65)=3.3). The per-trial core logit is therefore EXACTLY beta*c(v): CNNAI c(v)=v, CAVI-K c(v)=v+tau*v^2. I fit per subject a through-origin logistic model P(choose fewer-defect option) = sigmoid(b * g(v)) with the theory-neutral severity index g(v) = v + v^2/2 (midpoint of the linear and quadratic maps). This pools all 80 ladder trials with roughly optimal (inverse-information) weighting instead of wasting power on noisy per-rung logits: with only 16 trials/rung, empirical logit ratios like c(1.0)/c(0.5) have per-subject se >= 0.5-1.0 and blow up at the lapse-limited ceiling, so the design's raw 'kill ratios' are unusable as per-subject statistics at N=25. The slope b_hat is a single well-identified scalar per subject. Anchor/tie/all-none trials are deliberately excluded: the (2,0) anchor is lapse-ceiling-limited for both theories and would collapse the contrast.\n\nWINSORIZATION: b_hat is clipped to [0.5, 3.5]. The floor bounds the pure estimation noise of chance-level (guesser) subjects, whose slope is unidentified (raw b_hat ~ N(0, 0.46)); the cap bounds ceiling-separated near-deterministic subjects whose MLE diverges. Both bounds are far from the engaged CNNAI mass, so they act only on the degenerate subpopulations.\n\nPREDICTED SEPARATION (analytic MLE calculations using each theory's hard-coded parameter distributions): CNNAI engaged subjects have b_hat = 0.65*beta with beta ~ lognormal(median 2.14, log-sd 0.20), i.e. b_hat ~ 1.39, per-subject sd ~ 0.40 (beta spread + binomial estimation noise se ~ 0.28); with 13% guessers at the floor, population point ~ 1.28, between-subject variance ~ 0.21. CAVI-K engaged subjects have b_hat = 0.84*beta with beta ~ lognormal(median 2.65, log-sd 0.29), i.e. ~ 2.22, plus a 16% deterministic tail (beta 4-6.5) at ~ 3.3 after the cap; population point ~ 2.16, between-subject variance ~ 1.0. Welch's t at N=25: t = (2.16 - 1.28)/sqrt((0.21+1.00)/25) ~ 4.0, p ~ 2e-4, comfortably below alpha = 0.01, with margin against +-30% errors in either the gap or the variances. The contrast is driven by hard-coded, non-fudgeable structural facts: CAVI-K's larger effective defect weights c(v) = v + tau*v^2 (>= 1.25x CNNAI's c(v) = v at every rung), its higher beta median (2.65 vs 2.14), and its 16% deterministic tail \u2014 none of which CNNAI can reproduce (its homogeneous beta, clip 4.5, caps engaged slopes near 2.9, and its exact-linear weight map gives engaged slopes near 0.65*beta only).\n\nINTERPRETATION: values near ~1.3 support CNNAI (context-normalized linear anti-validity, no hot tail); values near ~2.2 with a visible upper mode near the 3.5 cap support CAVI-K (context-invariant quadratic defect term plus deterministic tail). The metric is also diagnostic against third parties: a validity-blind counter gives b_hat ~ 0 (floored), a pro-validity integrator gives negative slopes (floored to 0.5)."
}
```

## Usage

```json
{
  "input_tokens": 13434,
  "output_tokens": 48170,
  "reasoning_tokens": 46295,
  "provider": "Sail Research"
}
```
