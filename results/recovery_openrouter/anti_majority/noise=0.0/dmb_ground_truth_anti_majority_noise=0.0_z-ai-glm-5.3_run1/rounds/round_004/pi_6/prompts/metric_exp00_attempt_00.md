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
**Validities (n_features=20):** [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 25: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 26: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 27: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 29: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 31: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 32: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Rationale:** NOVEL AXIS (vs Exps 0-7): the first head-to-head between the two CONFIDENCE-DISTRUST theories, DCD (advocated) and VPS (competing). Both treat endorsements as liabilities increasing in the endorser's validity, so count-conflict cells and high-validity cells largely produce AGREEMENT; the dissociation lives in the SHAPE of the distrust weight near chance validity plus DCD's residual trust term. Formally, DCD's effective per-endorsement score s(v) = c0 + c1*(2v-1)^kappa - alpha*logit(v) is NON-MONOTONE in v (at mid parameters s(0.55)=0.268 < s(0.50)=0.280 < s(0.60)=0.271... < s(0.65)=0.288: the LEAST suspicious endorser is a 55-60% expert, because the (2v-1)^kappa dampening collapses the distrust charge near chance while alpha*logit(v) actively rewards validity), whereas VPS's penalty p(v) = c0 + c1*v^kappa is STRICTLY increasing (a 0.60 expert's endorsement costs c1*(0.6^kappa - 0.5^kappa) = 0.07-0.17 units more than a coin-flip's; DCD's corresponding charge is <= 0.043 units and is OPPOSED by alpha*logit(0.60) >= 0.032). At MATCHED endorsement counts VPS's flat term c0 cancels exactly, so VPS sign-robustly prefers the LOWER-validity endorser on every matched-count pair across its entire box (P(A) < 0.5 always); DCD prefers the 0.55/0.60 endorser (sign-robustly ABOVE 0.5 for v=0.55; within +/-0.06 of 0.5 for v=0.60). The design therefore concentrates on matched-count composition among near-chance experts and avoids the moderate/high-validity count cells where the two distrust theories agree. Validities (20 experts): 0.50 x4, 0.55 x4, 0.60 x4, 0.65 x3, 0.70 x3, 0.90 x2. 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject), perfectly side-balanced. Predictions are P(A) (A = higher-validity-composition side) at DCD mid (c0=.28, c1=.78, kappa=2, alpha=.10, beta=2.4, eps=.10) and VPS mid (c0=.10, c1=1.0, kappa=2, beta=2.0, eps=.10); box ranges include the full parameter boxes AND the lapse. FAMILY 1 -- k-GRADIENT AT 0.55 vs 0.50 (pairs 1-4, k=1..4): DCD 0.51, 0.51, 0.52, 0.53 (flat-rising, ABOVE 0.5, sign-robust: -c1*0.1^kappa + alpha*0.2007 >= +0.004 per endorsement); VPS 0.48, 0.45, 0.43, 0.41 (descending; per-unit D = -c1*(0.55^kappa - 0.5^kappa) in [-0.070, -0.034], remarkably stable across kappa). OPPOSITE-SIGNED SLOPES in k; boxes disjoint including lapse (k=4: DCD [0.51, 0.55] vs VPS [0.36, 0.45]). FAMILY 2 -- k-GRADIENT AT 0.60 vs 0.50 (pairs 5-8): DCD 0.51, 0.51, 0.52, 0.52 (flat; per-unit D in [-0.011, +0.027], bounded by the dampening and opposed by the trust term); VPS 0.45, 0.40, 0.36, 0.31 (diving; per-unit in [-0.146, -0.074]). At k=4: DCD [0.48, 0.56] vs VPS [0.24, 0.39] -- disjoint by 9+ points. FAMILY 3 -- VALIDITY LADDER AT k=3 (pairs 3, 7, 9, 10: 3x{0.55, 0.60, 0.65, 0.70} vs 3x0.50): VPS is a monotone steep descent 0.43, 0.36, 0.29, 0.22 (box ranges [0.39,0.46], [0.29,0.42], [0.21,0.37], [0.15,0.33]); DCD is HUMP-SHAPED 0.52, 0.52, 0.49, 0.44 (boxes [0.52,0.53], [0.50,0.55], [0.44,0.54], [0.38,0.51]), crossing 0.5 between v=0.60 and v=0.65 -- DCD's trust-distrust equilibrium validity (~0.63) that VPS structurally cannot exhibit. Max mid separation 22 points at v=0.70; all four rungs box-disjoint. FAMILY 4 -- ONE-STEP GAP CELLS (pairs 13, 11, 14, 12: {2x0.60, 3x0.60, 2x0.65, 3x0.65} vs {2x0.55, 3x0.55}): a single 0.05-validity step among near-chance experts moves VPS 0.45, 0.42, 0.39, 0.35 (boxes down to [0.28,0.41]) but leaves DCD blind at 0.50, 0.50, 0.48, 0.47 (all boxes within [0.43, 0.51]); all four disjoint. FAMILY 5 -- HIGH-VALIDITY ANCHORS (pairs 15, 16): pair 15 (1x0.90 vs 1x0.50): both avoid the 0.90 endorser but DCD is systematically softer (DCD 0.35, box [0.31, 0.39] vs VPS 0.27, box [0.21, 0.34]) -- DCD's softening of VPS's extremes. Pair 16 (1x0.90 vs 2x0.50) is the c0-MAGNITUDE assay: DCD's large flat charge exactly offsets its dampened 0.90 distrust (2*s(0.50) = 0.560 = s(0.90) at mid; P(A) = 0.50), while VPS's small c0 (2*0.35 = 0.70 << 0.91) leaves the validity-distrust gap exposed (P(A) = 0.41) -- a 10-point mid separation testing whether the count floor can rescue the high-validity endorser. UNFAKEABILITY IN BOTH DIRECTIONS: (i) no VPS parameterization can produce P(A) >= 0.5 on ANY matched-count cell (c0 cancels, c1 > 0, v^kappa strictly increasing), so the above-chance preference for the 0.55 endorser (Family 1) or even chance-level performance at k=4 where VPS sits at 0.24-0.31 (Family 2) falsifies VPS outright; (ii) no DCD parameterization can reproduce VPS's low-validity ladder: DCD's matched-count decision variable between 0.60 and 0.50 experts is bounded per endorsement in [-0.011, +0.027] (dampened distrust <= 0.043, opposed by alpha*logit >= 0.032), so at any k <= 4 it cannot leave [0.44, 0.57], while VPS's is <= -0.073 per endorsement (P <= 0.39 at k=4 including lapse); between 0.55 and 0.50 DCD is bounded BELOW by +0.004 per endorsement while VPS is bounded ABOVE by -0.034. The two theories are thus placed in orthogonal sensitivity regimes exactly at the validity range where their weight profiles diverge maximally: VPS is most sensitive precisely where DCD is dampened. 14 of 16 base pairs have fully disjoint predicted intervals across the ENTIRE parameter boxes including lapse; the two high-validity anchors (pairs 15, 16) are secondary magnitude cells (10-11-point mid separations, same direction). All 32 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- every cell has at least one theory off-chance by >= 4 mid-points, and the profile statistics use all cells), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in instructions), so the K=3 repetition is harmless.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Dampened Confidence-Distrust Integration (DCD). Every expert endorsement is treated as a liability whose magnitude grows with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. The per-endorsement distrust charge is d_j = c0 + c1*(2*v_j - 1)^kappa, where the (2v-1) rescaling DAMPENS the charge near chance validity (a 55% expert's endorsement costs barely more than a coin flip's, i.e. only the flat baseline c0) while remaining steeply increasing toward confident experts. Two additions repair the incumbents' failures: (i) a large flat charge c0 preserves pure endorsement-count avoidance, so when the low-validity option carries MORE endorsements the count signal can overcome validity-distrust (the count-conflict cells where VPS is wrong); (ii) a small residual validity-weighted trust term alpha * sum_j logit(v_j)*(a_j - b_j) captures residual endorsement-following that pulls matched-count choices back toward and slightly past chance in the direction of the higher-validity endorser (softening VPS's extremes). The decision variable is D = (P(B) - P(A)) + alpha * sum_j logit(v_j)*(a_j - b_j), where P(X) = sum_j x_j*d_j is the accumulated suspicion for option X; the subject chooses via P(A) = sigmoid(beta*D) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so distrust weights are fixed for the whole block. DCD nests VPS as a limiting case (alpha -> 0, rescaling absorbed) and SDI as another (c1 -> 0), while filling the empirically human region between them.

**Parameters:**
- kappa: [1.85, 2.15]
- c0: [0.26, 0.30]
- c1: [0.70, 0.85]
- alpha: [0.08, 0.12]
- beta: [2.2, 2.6]
- epsilon: [0.08, 0.12]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Dampened Confidence-Distrust Integration (DCD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Per-endorser liability (dampened confidence-distrust charge):
    #   d_j = c0 + c1 * (2*v_j - 1)^kappa
    #     c0            -> flat suspicion floor: every endorsement is a
    #                      defect, and a near-chance expert's praise
    #                      costs essentially only this floor (the
    #                      (2v-1) rescaling dampens distrust near
    #                      v = 0.5 relative to a raw v^kappa scaling)
    #     c1*(2v-1)^kap -> steeply INCREASING in claimed validity:
    #                      a confident expert's endorsement is a big
    #                      suspected defect
    # Accumulated suspicion:  P(X) = sum_j x_j * d_j
    # Residual trust term:    E = sum_j logit(v_j) * (a_j - b_j)
    # Decision variable:      D = (P(B) - P(A)) + alpha * E
    #   (positive favors A: B carries more accumulated suspicion,
    #    plus a small validity-weighted endorsement edge for A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so
    # the distrust weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the logit against v = 1 and v < 0.5. The (2v-1) power
    # is safe for v = 0.5 (charge collapses to the flat floor c0).
    v = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])

    # Dampened confidence-distrust charge per endorser: flat
    # suspicion floor plus a steeply validity-scaled distrust term
    # that vanishes toward chance validity.
    d = c0 + c1 * np.power(2.0 * v - 1.0, kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual validity-weighted endorsement (trust) term: a small
    # tendency to follow the more credibly endorsed option that
    # pulls matched-count choices toward (and slightly past)
    # chance in favor of the higher-validity endorser.
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

**Parameters:**
- kappa: [1.6, 2.4]
- beta: [1.6, 2.4]
- epsilon: [0.07, 0.13]
- c0: [0.05, 0.15]
- c1: [0.7, 1.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Weighted proportion of choosing the side whose endorsements come from\n    # MORE VALID experts, on matched-endorsement-count trials.\n    #\n    # Design validities (n_features=20, fixed): indices 0-5 are 0.50,\n    # indices 6-19 are 0.60. On every trial where both options carry the\n    # SAME number of endorsements, the flat count terms (c0) of both\n    # DCD and VPS cancel exactly, so the choice is driven purely by the\n    # validity composition of the endorsers:\n    #   - VPS: penalty strictly increasing in v  -> always avoids the\n    #     0.60 experts -> weighted P(hi-validity side) well below 0.5\n    #     (mid ~0.38, box <= ~0.45).\n    #   - DCD: dampened distrust (2v-1)^kappa collapses near chance and\n    #     is opposed by the residual trust term alpha*logit(v) ->\n    #     essentially at/above chance (mid ~0.51, box >= ~0.49).\n    # Trials are weighted by m = |#0.60-endorsements difference|, which\n    # emphasizes the cells (m=2,3,4) where the two theories diverge most.\n    V = np.array([0.5] * 6 + [0.6] * 14, dtype=float)\n    hi = (V > 0.55).astype(float)  # indicator of the 0.60-validity experts\n\n    if data is None or len(data) == 0:\n        return 0.5\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    resp = data['response'].to_numpy()\n\n    num = 0.0\n    den = 0.0\n    for i in range(len(data)):\n        a = np.asarray(a_col[i], dtype=float)\n        b = np.asarray(b_col[i], dtype=float)\n        if a.shape[0] != V.shape[0] or b.shape[0] != V.shape[0]:\n            continue\n        tA = float(np.sum(a))\n        tB = float(np.sum(b))\n        # Only matched-count trials: the flat c0 terms cancel and the\n        # choice isolates the validity-composition sensitivity.\n        if tA != tB:\n            continue\n        hA = float(np.dot(hi, a))  # number of 0.60-expert endorsements on A\n        hB = float(np.dot(hi, b))\n        if hA == hB:\n            continue  # pure tie in validity composition\n        m = abs(hA - hB)\n        r = int(resp[i])\n        chose_hi = (r == 0) if (hA > hB) else (r == 1)\n        den += m\n        if chose_hi:\n            num += m\n\n    if den <= 0:\n        return 0.5\n    return num / den",
  "rationale": "The design's actual validities are 0.50 (features 0-5) and 0.60 (features 6-19), so the entire DCD-vs-VPS dissociation lives on matched-endorsement-count trials: there the flat count term c0 cancels in BOTH theories and the choice is driven purely by the shape of the distrust weight near chance validity. VPS's penalty c0 + c1*v^kappa is strictly increasing in v, so on every matched-count pair it sign-robustly avoids the 0.60 experts (per-extra-0.60-endorsement decision contribution between -0.146 and -0.074 across its whole box, giving a weighted P(higher-validity side) around 0.38 at mid parameters, <= ~0.45 even at the most forgiving box corner). DCD's dampened charge c1*(2v-1)^kappa nearly vanishes at v=0.6 (0.022-0.043) and is actively opposed by the residual trust term alpha*logit(0.6) (0.032-0.049), so its matched-count decision variable stays within about +/-0.011 per endorsement and P(higher-validity side) stays at/above chance (~0.51 at mid, >= ~0.49 across the box). The metric is the m-weighted proportion of choosing the side with more 0.60-expert endorsements over all matched-count, non-tie trials (78 of 96 trials per subject; weights m = 1,2,3,4 concentrate on the cells with the largest predicted separation, e.g. m=4: DCD ~0.52 vs VPS ~0.31). It is a simple bounded proportion, so per-subject binomial noise is small (SD ~0.06 over 78 trials with these weights) and parameter heterogeneity barely moves DCD off 0.5 while VPS stays pinned below 0.45 \u2014 giving a mean gap of ~0.13 with tight per-subject values, far beyond what N=25 needs for a Welch t-test at alpha=0.01. Count-conflict trials (pairs 16/32) and pure validity ties (pairs 1/2/17/18) are excluded because both theories agree there, so including them would only add noise."
}
```

## Usage

```json
{
  "input_tokens": 10471,
  "output_tokens": 11245,
  "reasoning_tokens": 10025,
  "provider": "Sail Research"
}
```
