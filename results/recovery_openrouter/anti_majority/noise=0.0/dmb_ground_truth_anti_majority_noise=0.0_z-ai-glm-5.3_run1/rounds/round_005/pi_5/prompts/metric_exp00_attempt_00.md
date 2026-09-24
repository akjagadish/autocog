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
**Validities (n_features=15):** [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 4: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 18: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-10): PERFECT EXPERTS (v = 1.0) combined with MINIMAL COUNT ASYMMETRY (+/-1, +/-2 endorsements). Exp 10, the only prior SVD-vs-VPS design, used 0.95/0.75/0.50 experts and 2:1 count ratios; its conflict cells left VPS's weak box corners unsaturated (its Family-3 VPS intervals reached up to [0.05, 0.47]), so VPS could mimic SVD's levels at extreme corners. The v = 1.0 expert closes that escape route STRUCTURALLY: its distrust charge is c0 + c1*1^kappa = c0 + c1 -- the maximum possible and KAPPA-INVARIANT -- so no parameterization of either theory can discount it, while the coin-flip expert's charge c0 + c1*0.5^kappa (in [0.18, 0.58] on VPS's box) is the cheapest suspicion unit; the charge ratio (c0+c1)/(c0+c1*0.5^kappa) is >= 2.23 at every VPS corner. On a 6-perfect-vs-7-coin cell VPS's decision variable is D = c0 + c1*(7*0.5^kappa - 6) <= -2.43 at EVERY corner of its box: VPS is pinned at its floor (or ceiling) by arithmetic, not by favorable parameters. Meanwhile a SINGLE extra coin-flip endorsement -- worth at most 0.58 penalty units against a 2-6 unit saturated gap -- flips SVD's deterministic counter branch from voting one way, to indifferent, to voting the other way. Validities (15 experts): 1.0 x 6, 0.50 x 9. 12 base stimuli, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. Mid parameters quoted: VPS (kappa=2, beta=2, eps=.10, c0=.10, c1=1.0; charges: perfect 1.10, coin 0.35); SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18; charges: perfect 1.21, coin 0.443); box ranges verified at box corners. FAMILY 1 -- SATURATED FLOOR (pairs 1-3): A carries j perfect endorsements (j=4,6) vs B's j+1 (pair 3: j+2) coin-flips. VPS: P(A) in [0.050, 0.138] (pair 1), [0.050, 0.082] (pair 2), [0.050, 0.096] (pair 3) -- near-certain avoidance of the perfect-expert side at every corner. SVD: its skeptic branch is equally saturated toward B, but its counter branch (prob 1-w >= 0.08) deterministically votes A (fewer endorsements): P(A) = (1-eps)(w*s + (1-w)) + eps/2 >= 0.130 on all three (boxes [0.130, 0.294], [0.130, 0.285], [0.130, 0.293]; mid 0.210, 0.205, 0.205). DISJOINT on pairs 2-3 (VPS max 0.082/0.096 < SVD min 0.130); pair 1 is a near-disjoint gradient rung (15.6 mid points, 8-point box overlap). An A-rate >= 0.12 on pairs 2-3 falsifies VPS outright; <= 0.11 falsifies SVD outright. This floor-level pinning is what Exp 10's 0.95-expert cells could not achieve. FAMILY 2 -- SATURATED CEILING, COUNTER-OPPOSED (pairs 14-15, mirrors of 2-3): A = 7 (or 8) coin endorsements vs B = 6 perfect. Both cores saturate toward A; SVD's counter votes B. VPS: P(A) >= 0.919 (pair 14), >= 0.904 (pair 15), up to 0.951. SVD: P(A) = (1-eps)*w*s + eps/2 <= 0.870 (boxes [0.715, 0.870], [0.710, 0.870]; mid 0.795 both). DISJOINT: an A-rate >= 0.91 falsifies SVD; <= 0.88 falsifies VPS. Unlike Exp 10's matched-count ceiling cells (SVD cap 0.905), the counter branch here actively drags SVD 3-8 points further from VPS. FAMILY 3 -- THE COUNT-PARITY STAIRCASE (pairs 2 -> 4 -> 5 floor side; 14 -> 16 -> 17 ceiling side): the perfect-expert side's endorsement count climbs 6 -> 7 -> 8 (one added coin-flip expert per rung) against a FIXED 7-coin opponent, so the skeptic evidence stays saturated (|D| >= 2.05 at every corner of both boxes) while count parity passes through fewer -> matched -> more. SVD is a STAIRCASE: mid P(A) = 0.205 -> 0.147 -> 0.090 (floor side) and 0.795 -> 0.853 -> 0.910 (ceiling side), with a step of exactly (1-w)(1-eps)/2 in [3.0, 8.8] points at EVERY parity crossing -- the step size is a direct read on SVD's mixture and lapse parameters. VPS is FLAT: a coin endorsement shifts D by at most 0.58 units against |D| >= 2.05, so P(A) moves <= 0.7 points box-wide (floor side [0.050,0.082] -> [0.050,0.075] -> [0.050,0.071]; ceiling side [0.919,0.951] -> [0.926,0.951] -> [0.929,0.951]). The change intervals are DISJOINT at every step (SVD >= 3.0 points where VPS <= 0.7). Any single coin-flip endorsement moving choice by >= 2 points on these saturated cells falsifies VPS (no beta, kappa, c0, c1, or epsilon in its box can produce it); flatness (<= 2 points) falsifies SVD outright. Pair 4's LEVEL is additionally disjoint by itself: VPS [0.050, 0.075] vs SVD [0.150, 0.156] -- SVD's matched-count indifference cap (1-eps)(1+w)/2 + eps/2 keeps it 7.5+ points above VPS's floor even though its skeptic branch, like VPS's, is fully saturated. FAMILY 4 -- THE EXCHANGE LOCUS (pairs 7-10): one perfect endorsement vs m = 1, 2, 3, 4 coin-flips. VPS mid: 0.214, 0.329, 0.478, 0.631 -- crossing 50% between m=3 and m=4 (internal exchange rate (c0+c1)/(c0+c1*0.5^kappa) = 3.14 at mid, in [2.23, 4.56] across the box). SVD mid: 0.280, 0.452, 0.598, 0.730 -- crossing between m=2 and m=3 (skeptic rate 2.73 at mid, in [2.59, 2.88], PLUS the counter branch voting for the single-endorsement side at m >= 2, pulling the crossing a full rung earlier). SIGN CONFLICT at m=3 (pair 8): VPS 0.478 (slight B-majority) vs SVD 0.598 (clear A-majority), 12 mid points; at m=2 (pair 7) VPS is at a clear B-majority (0.329) where SVD sits near chance (0.452). Boxes overlap here (VPS's exchange-rate range is wide), so this family is a SECONDARY crossing-rung signature: the observed indifference rung is m=3-4 for VPS vs m=2-3 for SVD, and the joint profile adjudicates corner ambiguity -- the VPS corner that crosses earliest (c0=0.15, c1=0.7, kappa=1.6, giving P(pair 8) = 0.598) is still pinned flat on the staircase and <= 0.138 on every floor cell, so a subject showing SVD's crossing rung TOGETHER WITH any staircase step >= 2 points or any floor-cell A-rate >= 0.12 lies outside VPS's entire box. FAMILY 5 -- MIXED-COALITION EXCHANGE (pairs 11-12): two perfect endorsements vs 5 (then 6) coin-flips: VPS 0.310, 0.455 vs SVD 0.489, 0.636 -- 18-point mid gaps in the same direction (SVD's counter votes the 2-endorsement side), secondary magnitude cells. FAMILY 6 -- MATCHED-COUNT CEILING BASELINE (pairs 6, 18): 6 perfect vs 6 coin, counts matched. VPS [0.926, 0.951] vs SVD [0.841, 0.905] (pair 6; mirror flipped) -- disjoint; the v=1.0 sharpening of Exp 10's pair-1 level assay (VPS's matched-count D = 6*c1*(1-0.5^kappa) >= 2.81 across its box). UNFAKEABILITY IN BOTH DIRECTIONS: (i) No VPS parameterization can produce the staircase: on staircase cells |D| >= 2.05 at every VPS corner while a coin endorsement is worth <= 0.58 units, bounding VPS's per-rung change at 0.7 points vs SVD's structural step (1-w)(1-eps)/2 >= 3.0. (ii) No VPS parameterization can reach SVD's floor levels on pairs 2-3 (box max 0.082/0.096 < SVD box min 0.130) or pair 4 (0.075 < 0.150). (iii) No SVD parameterization can reach VPS's ceiling on pairs 14-15 (counter-opposed cap (1-eps)w + eps/2 <= 0.870 < VPS min 0.904) or the matched-count cells 6/16/18 (indifference cap <= 0.905 < VPS min 0.926). (iv) The parameters trade off oppositely: SVD's most VPS-like corner (eps=0.12, w=0.92) simultaneously gives floor cells at 0.130 (still 3.4+ points above VPS's pairs-2/3 ceiling of 0.096), ceiling cells at 0.870 (still below VPS's 0.904 floor), and staircase steps of 3.5 points (still 2.8 points above VPS's 0.7 maximum) -- every SVD corner is disjoint from every VPS corner on at least two families, and the conjunction {floor level, ceiling level, staircase step, exchange crossing} is a profile neither box can approximate: a VPS subject is a flat line at 0.05/0.95 across the staircase with a 3-4-rung exchange crossing; an SVD subject is a 0.13-0.29 floor, a 0.71-0.87 counter-opposed ceiling, a 3-9-point staircase whose step size estimates (1-w)(1-eps)/2, and a 2-3-rung exchange crossing. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 7 non-endorsements, no all-tie pairs, and no pair leaves both theories guessing -- the closest cells, pairs 5 and 17, are exactly the staircase rungs where the step signature lives), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

**Parameters:**
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.92]
- epsilon: [0.12, 0.24]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
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
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Saturated-cell PERFECT-SIDE CHOICE RATE.\n\n    Uses the 10 saturated pairs (1,2,3,4,6 and their A/B mirrors 13,14,15,16,18;\n    40 of the 96 trials per subject) in which one option carries >=4\n    validity-1.0 (\"perfect\") endorsements with <=7 total endorsements,\n    while the other option carries 0 perfect endorsements and >=5\n    coin-flip (v=0.5) endorsements. The metric is the subject's rate of\n    CHOOSING the perfect-endorser side on these trials.\n\n    VPS: the perfect expert's distrust charge c0+c1 is kappa-invariant and\n    maximal, so the skeptic evidence is saturated (|D| >= 2.4 at every\n    corner of VPS's box) -> P(perfect side) ~= eps/2 in [0.035, 0.07].\n\n    SVD: its skeptic branch is equally saturated, but its counter branch\n    (prob 1-w >= 0.08) votes by raw endorsement COUNT -- for the perfect\n    side on pairs 1-3 (fewer endorsements) and indifferent on the\n    matched-count pairs 4/6 -- giving P(perfect side) = (1-eps)(1-w)+eps/2\n    in [0.095, 0.27].\n\n    The two per-subject rate intervals are essentially disjoint\n    (VPS <= ~0.09 vs SVD >= ~0.10 even at the most favorable corners),\n    and pooling floor trials with their mirrors cancels any side bias.\n    \"\"\"\n    N_PERF = 6  # features 0..5 are the validity-1.0 (\"perfect\") experts\n\n    df = data.reset_index(drop=True)\n\n    def counts(ratings):\n        r = [int(x) for x in ratings]\n        perf = 0\n        coin = 0\n        for i, x in enumerate(r):\n            if x == 1:\n                if i < N_PERF:\n                    perf += 1\n                else:\n                    coin += 1\n        return perf, coin\n\n    hits = []\n    for a_raw, b_raw, resp in zip(\n        df[\"option_a_ratings\"], df[\"option_b_ratings\"], df[\"response\"]\n    ):\n        perfA, coinA = counts(a_raw)\n        perfB, coinB = counts(b_raw)\n\n        # A is the perfect-heavy saturated side\n        if perfA >= 4 and (perfA + coinA) <= 7 and perfB == 0 and coinB >= 5:\n            hits.append(1.0 if int(resp) == 0 else 0.0)\n        # B is the perfect-heavy saturated side (mirror trials)\n        elif perfB >= 4 and (perfB + coinB) <= 7 and perfA == 0 and coinA >= 5:\n            hits.append(1.0 if int(resp) == 1 else 0.0)\n        # all other trials (pairs 5/17 staircase top rung, exchange locus\n        # pairs 7-12/19-24) are deliberately excluded: their theory gap is\n        # small or their between-subject spread is large, so they would\n        # only dilute the contrast.\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))\n```",
  "rationale": "I advocate VPS. The sharpest, lowest-variance contrast in this design is the LEVEL of the saturated cells, not the staircase slope or the exchange crossing (both are difference-based and noisy at N=25). On the 10 saturated pairs used here (perfect side has >=4 perfect experts and <=7 total endorsements; coin side has 0 perfect and >=5 coins), the perfect expert's distrust charge c0+c1*v^kappa = c0+c1 is kappa-invariant and maximal, so VPS's decision variable is saturated at every corner of its parameter box: P(choose perfect side) collapses to eps/2 in [0.035, 0.065] (plus a negligible sigmoid residual; worst case ~0.09 only on pair 1's extreme corner). SVD's skeptic branch is equally saturated toward the coin side, but its counter branch (probability 1-w >= 0.08) votes by raw endorsement count: FOR the perfect side on pairs 1-3 (4-6 vs 5-8 endorsements) and INDIFFERENT on matched-count pairs 4/6, yielding P(choose perfect side) = (1-eps)(1-w)+eps/2 or (1-eps)(1-w)/2+eps/2, i.e. [0.095, 0.27] across its box \u2014 even the most VPS-like SVD corner (w=0.92, eps=0.12) sits at ~0.115 on average, above the most SVD-like VPS corner. Expected values: VPS ~0.055, SVD ~0.18, a gap of ~0.12 with per-subject sd of ~0.04 (VPS) and ~0.07 (SVD, dominated by binomial noise over 40 trials plus w/eps heterogeneity), giving Welch t ~7 at N=25, far below alpha=0.01. Three design choices maximize robustness: (i) pooling floor pairs with their A/B mirrors makes the metric a pure 'avoid the perfect endorser' rate, cancelling any side bias; (ii) excluding the staircase top rung (pairs 5/17, where both theories collapse to eps/2 and the gap comes only from the epsilon ranges) and the exchange-locus pairs (near 0.5, maximal binomial variance) removes dilution; (iii) the metric is a plain mean of binary indicators, so it is well defined both on the pooled DataFrame and on each single subject's 96-trial slice, giving a usable between-subject variance. The metric is not trivially true for VPS \u2014 it pins the theory at its lapse floor via arithmetic (kappa-invariant charge of a v=1.0 expert), and any human subject choosing the perfect-endorser side on >=12% of these trials falsifies VPS outright while matching SVD's counter-branch signature."
}
```

## Usage

```json
{
  "input_tokens": 9228,
  "output_tokens": 18308,
  "reasoning_tokens": 16871,
  "provider": "Sail Research"
}
```
