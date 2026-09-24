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
**Validities (n_features=15):** [0.95, 0.9, 0.9, 0.9, 0.85, 0.8, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 19: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 22: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-5): the first head-to-head between the two DEFECT-INTEGRATION theories. SDI and VPS agree that every endorsement is a liability, but they disagree about the WEIGHT PROFILE attached to the endorser's validity: SDI's skepticism weight (1-v)^delta is nearly FLAT across the whole validity range (0.86-0.98 at delta <= 0.05, a <14% spread; at delta=0 exactly flat), so SDI is essentially a validity-blind endorsement COUNTER; VPS's weight c0 + c1*v^kappa INCREASES steeply in validity (a 0.95 expert's endorsement costs ~2.9x a 0.50 expert's at mid parameters), so VPS actively prefers options endorsed by WEAKER experts. The design places the two theories in orthogonal indifference regimes: SDI is pinned at chance exactly when endorsement counts match but validity composition differs; VPS is pinned at (near) chance exactly when counts differ but its validity-scaled penalties cancel. Validities (15 experts): [0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65, 0.60, 0.60, 0.60, 0.55, 0.50, 0.50, 0.50, 0.50] -- duplicates are required to build coalitions; the spread 0.50-0.95 spans the full range. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below are given as P(A) at SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025) and VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1), with full-box ranges verified sign-robust. FAMILY 1 -- VALIDITY-GAP LADDER AT MATCHED COUNTS (pairs 1-5): 1v1 pairs with shrinking endorser gaps (.95/.50, .90/.55, .85/.60, .80/.65) plus the doubled 2v2 max-gap pair (.95+.90 vs .55+.50). VPS predicts a monotone ladder BELOW 50%: 0.24, 0.29, 0.34, 0.40, 0.13 -- it always prefers the LOWER-validity endorser, structurally (its penalty is strictly increasing in v for any c1>0, so D<0 at matched counts for EVERY parameterization). SDI is pinned at 0.51 +/- 0.02 (box 0.50-0.56): its penalty weights vary <14% and its only tilt, the alpha<=0.05 tie-break, pushes TOWARD the high-validity endorser -- i.e., AWAY from VPS's prediction, so no SDI parameterization can mimic the ladder. A monotone descending P(A) in gap falsifies SDI; a flat ~0.51 line falsifies VPS. FAMILY 2 -- SIGN CONFLICTS, OPPOSITE MAJORITIES (pairs 6-8): (a) 3x0.50 vs 2x0.90: VPS 0.79 (box 0.60-0.90) vs SDI 0.26 (box 0.24-0.30); (b) 4x0.50 vs 3x0.90: VPS 0.89 (box 0.75-0.93) vs SDI 0.25 (box 0.26-0.31); (c) 2x0.90 vs 3x0.60: VPS 0.31 (box 0.17-0.45) vs SDI 0.72 (box 0.68-0.75). Every conflict is sign-robust across BOTH entire parameter boxes (e.g., pair 6: VPS D = -c0 + c1(2*0.9^kappa - 3*0.5^kappa) > 0 always since c0/c1 <= 0.214 < 0.700; pair 8: D < 0 always since c0/c1 <= 0.214 < 0.365). A single majority choice on any conflict cell adjudicates the theories with no free parameters able to reverse it. FAMILY 3 -- JUNK-COUNT GRADIENT VS FIXED 2x0.90 (pairs 9, 10, 6): k = 1, 2, 3 endorsements from 50% experts on A against the same two 90% endorsements on B. SDI declines 0.66 -> 0.45 -> 0.26, CROSSING 50% between k=1 and k=2 (each junk endorsement adds ~0.97 penalty units; crossing at count parity is structural). VPS declines only 0.91 -> 0.86 -> 0.79 and does NOT cross through k=3, because each 0.50 endorsement costs only ~0.35 penalty units vs ~0.91 for a 0.90 endorsement -- it takes ~2.5 junk endorsements to offset one 90% endorsement for VPS but only 1 for SDI. The LOCATION OF THE CROSSING POINT is the quantitative signature. FAMILY 4 -- INDIFFERENCE ASYMMETRY (pairs 11-12): pair 11 (one 0.90 vs two 0.60s) is engineered so 2*0.6^kappa ~= 0.9^kappa (within 0.04-0.19 across kappa in [1.6, 2.4]), with the c0 count term partially cancelling the residual: VPS's D ranges only [-0.20, +0.20] over its ENTIRE box, keeping P(A) in [0.40, 0.61] (mid 0.50) -- near chance -- while SDI delivers a clear majority, P(A) = 0.70 (box 0.65-0.71), because B carries one extra endorsement. Pair 12 (3x0.90 vs 2x0.50) is the floor test: both theories prefer B, but SDI only moderately (P(A) = 0.35, box 0.28-0.42) while VPS is pinned at the floor (P(A) = 0.07, box 0.04-0.10) -- a 25+-point magnitude gap where sign cannot adjudicate. THE DISSOCIATION IS UN-FAKEABLE IN BOTH DIRECTIONS: a VPS subject is a descending ladder in Family 1 (all rungs < 0.47), stays above 0.60 through k=3 in Family 3, and sits at chance on pair 11; an SDI subject is a flat ~0.51 line across the entire ladder (its alpha tie-break can only push it ABOVE 0.5, opposite to VPS), crosses 50% at k=2 in Family 3, and shows a 0.70 majority on pair 11. Because SDI's stimulus-driven variation at matched counts is bounded by alpha*delta-logit <= 0.25 penalty units while VPS's ladder spans D in [-1.6, -0.41], and because VPS's indifference on pair 11 is arithmetic (2*0.6^kappa ~= 0.9^kappa across its kappa box) while SDI's count signal there is a full penalty unit, neither theory's parameter box can approximate the other's profile. All pairs are non-degenerate (every pair has discriminating endorsements, no all-tie pairs, no pair where both theories guess -- the closest, pair 11, is precisely the cell where they maximally diverge), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Skeptical Defect Integration (SDI). Subjects do not treat an expert's positive rating ('1') as an asset; they treat it as a liability — an endorsement from a fallible expert is noise-laden, adverse evidence. On each trial the decision maker computes a penalty for each option, P(X) = sum_j d_j * x_j, where d_j = (1 - v_j)^delta is the skepticism weight attached to expert j (delta = 0 recovers pure unweighted defect counting — every endorsement is one unit of suspicion; larger delta discounts endorsements from low-validity experts more steeply, since a bad expert's praise is especially untrustworthy). The subject chooses the option with the LOWER total penalty, i.e., the option that attracted less endorsement. Choice probability is P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. Exact penalty ties are broken by a weak validity-weighted endorsement term alpha * sum_j logit(v_j) * (a_j - b_j), which captures residual heterogeneity: as alpha grows, SDI smoothly recovers VWEI-like endorsement-following behavior. The empirically human region sits at delta near 0 (near-pure defect counting) with moderate noise. SDI is the only account in this domain that reproduces the SIGN of every observed metric: it follows the TTB winner whenever the tally winner carries a larger endorsing coalition (Exp 1 metric positive), follows the tally loser / top-cue favorite when the tally winner has more endorsements (Exp 2 metric negative), prefers the option with fewer endorsements even when a dominant high-validity cue endorses the rival (Exp 3 metric strongly negative, HIGH minus LOW), and produces a negative signed-evidence slope because P(choose TTB winner) DECREASES as validity-weighted evidence accumulates behind the more-endorsed option (Exp 4). It makes a novel falsifiable prediction: on designs where the high-validity option carries MORE positive ratings than the low-validity option, SDI predicts majority choices OPPOSITE to both TTB and VWEI.

**Parameters:**
- delta: [0.0, 0.05]
- beta: [0.9, 1.1]
- epsilon: [0.09, 0.14]
- alpha: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Skeptical Defect Integration (SDI).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Penalty for option X:  P(X) = sum_j d_j * x_j,
    #   d_j = (1 - v_j)^delta  (skepticism weight; delta = 0 -> pure
    #   unweighted defect counting).
    # Decision variable:      D = (P(B) - P(A)) + alpha * E,
    #   E = sum_j logit(v_j) * (a_j - b_j)  (weak validity-weighted
    #   endorsement tie-break).
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so skepticism weights are fixed for the block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SDI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard against v = 1 (infinite logit) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)

    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])

    # Skepticism weights: an endorsement from expert j is a liability,
    # scaled by how fallible the expert is.
    d = (1.0 - v) ** delta

    # Total penalty (accumulated suspicion) for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Weak validity-weighted endorsement term (tie-break only).
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A (A carries fewer / more-heavily-discounted
    # liabilities than B, plus a small endorsement edge).
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties and endorsement) -> exactly 0.5.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # \"Validity-following rate in theory-conflict cells\"\n    #\n    # The 15 experts split into a HIGH-validity group (features 0-6:\n    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a\n    # LOW-validity group (features 7-14: validity 0.60). Every trial in\n    # this design pits a \"pure high\" option against a \"pure low\"\n    # option. Two families of cells separate SDI from VPS:\n    #\n    # Family 1 (matched endorsement counts, validity gap): SDI's\n    #   penalty is a near-flat function of validity (delta ~ 0), so it\n    #   is pinned at ~0.51 for the high-validity option (its weak\n    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's\n    #   penalty rises steeply in validity, so it systematically picks\n    #   the LOW-validity option (~0.28).\n    #\n    # Family 2 (count conflict where the FEWER-endorsed option is the\n    #   high-validity one): SDI follows the endorsement count and\n    #   picks the high-validity option (~0.72). VPS's validity-scaled\n    #   distrust makes the low-validity majority cheaper, so it picks\n    #   the high-validity option only ~0.30 of the time.\n    #\n    # The metric is the average of P(choose the high-validity-group\n    #   option) over these two families. SDI predicts ~0.62; VPS\n    #   predicts ~0.29.\n\n    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,\n                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])\n\n    f1_hits = 0.0\n    f1_n = 0\n    f2_hits = 0.0\n    f2_n = 0\n\n    for a_raw, b_raw, r in zip(data['option_a_ratings'],\n                               data['option_b_ratings'],\n                               data['response']):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        m = int(min(a.size, b.size, V.size))\n        if m == 0:\n            continue\n        a = a[:m]\n        b = b[:m]\n\n        top = V[:m] >= 0.65  # high-validity expert group\n        a_top = float(np.sum(a[top]))\n        a_bot = float(np.sum(a[~top]))\n        b_top = float(np.sum(b[top]))\n        b_bot = float(np.sum(b[~top]))\n\n        # Require each option to be endorsed purely by one group\n        # (all trials in this design satisfy this; guards oddities).\n        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):\n            continue\n        ca = a_top + a_bot\n        cb = b_top + b_bot\n        if ca == 0 or cb == 0:\n            continue\n\n        a_is_top = a_top > 0\n        b_is_top = b_top > 0\n        if a_is_top == b_is_top:\n            continue\n\n        chose_a = (int(round(float(r))) == 0)\n        chose_top = chose_a if a_is_top else (not chose_a)\n\n        if ca == cb:\n            # Family 1: matched counts, validity gap.\n            f1_n += 1\n            f2_unused = None\n            f1_hits += 1.0 if chose_top else 0.0\n        else:\n            # Family 2: count conflict; keep only cells where the\n            # FEWER-endorsed option is the high-validity one\n            # (the cells where SDI and VPS point in opposite\n            # directions). Cells where the fewer-endorsed option is\n            # the low-validity one are agreed upon by both theories\n            # and are excluded.\n            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)\n            if fewer_is_top:\n                f2_n += 1\n                f2_hits += 1.0 if chose_top else 0.0\n\n    p1 = f1_hits / f1_n if f1_n > 0 else np.nan\n    p2 = f2_hits / f2_n if f2_n > 0 else np.nan\n    vals = [p for p in (p1, p2) if not np.isnan(p)]\n    if not vals:\n        return 0.5\n    return float(np.mean(vals))",
  "rationale": "The design's experts split cleanly into a high-validity group (features 0-6, validities 0.65-0.95) and a low-validity group (features 7-14, validity 0.60), and every trial pits a pure-high option against a pure-low option. This yields two orthogonal conflict families. FAMILY 1 (matched endorsement counts, 12 unique pairs, 48 trials/subject: e.g. one 0.95 endorser vs one 0.60 endorser; 2x0.90 vs 2x0.60): SDI's skepticism weight (1-v)^delta is nearly flat over the validity range (delta in [0, 0.05]), so its penalty difference is ~0 and its weak alpha tie-break pushes it slightly TOWARD the high-validity option (P ~ 0.51-0.55). VPS's penalty c0 + c1*v^kappa rises steeply in validity, so it systematically prefers the LOW-validity option (P(high) ~ 0.28 at mid parameters, and < 0.47 across its entire parameter box since D = c1*(v_lo^kappa - v_hi^kappa) < 0 structurally). FAMILY 2 (count conflicts where the FEWER-endorsed option is the high-validity one, 8 unique pairs, 32 trials/subject: 3x0.60 vs 2x0.90; 4x0.60 vs 3x0.90; 2x0.90 vs 3x0.60; 1x0.90 vs 2x0.60): SDI follows the endorsement count and picks the high-validity option with P ~ 0.70-0.73; VPS finds the low-validity majority cheaper per endorsement and picks the high-validity option only ~0.14-0.50 of the time (mean ~0.30). Cells where the fewer-endorsed option is the LOW-validity one (trials 9, 12, 21, 24) are agreed upon by both theories and are deliberately excluded so they cannot dilute the contrast. The metric averages P(choose the high-validity-group option) over the two families: SDI predicts ~0.62, VPS predicts ~0.29 \u2014 a gap of ~0.33 with per-subject standard deviation of only ~0.05 (binomial noise over 48 + 32 trials), giving a Welch t-statistic around 20 at N=25 (p << 0.001). The contrast is sign-robust across both theories' full parameter boxes: even at VPS's most SDI-like corner (c0=0.15, c1=0.7, kappa=1.6, beta=1.6, eps=0.13) the composite only rises to ~0.40, while SDI's least favorable corner stays ~0.59. The metric uses only the high/low validity grouping (threshold 0.65), which is invariant to the exact validities assigned to features 7-14, and it is well-defined both on the pooled DataFrame and on any single subject's 96-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 8143,
  "output_tokens": 19354,
  "reasoning_tokens": 17497,
  "provider": "Sail Research"
}
```
