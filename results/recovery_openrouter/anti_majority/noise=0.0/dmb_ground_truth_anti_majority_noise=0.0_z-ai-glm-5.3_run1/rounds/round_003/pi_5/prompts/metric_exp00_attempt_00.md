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
**Validities (n_features=20):** [0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.75, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.65, 0.65, 0.65]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 18: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 19: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** NOVEL AXIS (vs. Exps 0-6): MATCHED-COUNT SIGNAL AMPLIFICATION plus the full COUNT-VALIDITY EXCHANGE ECONOMICS. Exp 6 probed VPS vs SDI at unit matched count (1v1 validity ladder) and against one fixed high side (k x 0.50 vs 2x0.90). This design instead manipulates the COUNT dimension itself, exploiting a structural fact neither prior design used: at matched endorsement counts the two theories' decision variables have OPPOSITE SIGNS and IDENTICAL linear scaling in count. VPS: D = pen_B - pen_A = k*c1*(v_hi^kappa - v_lo^kappa) > 0 (the flat c0 term cancels EXACTLY, so matched-count trials are a pure assay of the confidence-distrust term c1*v^kappa), growing linearly in k, favoring the LOW-validity coalition ever more strongly. SDI: D = k*[(w_hi - w_lo) + alpha*Delta-logit] < 0 (its skepticism weight (1-v)^delta DECREASES in v, and the alpha tie-break also favors the high-validity side), also growing linearly in k, favoring the HIGH-validity coalition. Matched count k is therefore a signal AMPLIFIER for a sign conflict: at k=1 the theories differ by ~20 points; at k=4 they are at floor-vs-ceiling in opposite directions. Validity pool (20 experts): [0.95 x3, 0.90 x4, 0.85 x2, 0.75 x1, 0.65 x3, 0.60 x4, 0.55 x3] -- duplicates are required to build matched coalitions. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below: P(A) at VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1) and SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025), with full-box ranges [min,max] verified at all box corners; CRITICALLY, the VPS and SDI prediction intervals are DISJOINT on every single pair. FAMILY 1 -- MATCHED-COUNT GAP AMPLIFICATION (pairs 1-6): k x 0.60 vs k x 0.90 at k=1,2,3,4, and k x 0.55 vs k x 0.95 at k=2,3. VPS climbs toward ceiling: 0.69, 0.82, 0.89, 0.93 (gap .90/.60) and 0.87, 0.93 (gap .95/.55); box ranges [0.60,0.80], [0.68,0.92], [0.76,0.96], [0.81,0.96], [0.73,0.96], [0.81,0.96]. SDI stays pinned at/below chance, drifting slightly TOWARD the high-validity side as k grows: 0.48, 0.47, 0.45, 0.43 and 0.45, 0.42; box ranges [0.46,0.50] down to [0.34,0.50]. The signature is a sign x count interaction: VPS's slope in k is positive and large (+24 points over k=1..4 at mid params, and >= +21 points even between worst-case corners); SDI's slope is NEGATIVE or zero (its matched-count D is bounded in [-k*(0.064+2.74*alpha), 0], so P(A) <= 0.50 always and >= 0.34 at k=4). No SDI parameterization can produce P(A) > 0.5 on ANY matched-count pair (its validity sensitivity has the wrong sign and is bounded by 12% + alpha<=0.05), while no VPS parameterization can stay below 0.60 at k=4 (its D >= 4*0.7*0.403 = 1.13 across the entire box). FAMILY 2 -- EXCHANGE-RATE LOCUS (pairs 7-9): 2x0.55 vs 1 x v_hi with v_hi = 0.95, 0.85, 0.75. SDI is a validity-blind endorsement COUNTER here: the single endorser always carries fewer penalties, and its D varies by < 0.1 across the whole v_hi range, so SDI is FLAT at 0.28 / 0.29 / 0.29 (box [0.25,0.31]). VPS's preference tracks its internal exchange rate (v_hi/v_lo)^kappa ~ 2: it PREFERS two 55% endorsements over one 95% endorsement (0.59, box [0.49,0.74] -- a sign conflict with SDI on every parameterization), is indifferent at 0.85 (0.51, box [0.45,0.62]), and leans single at 0.75 (0.44, box [0.42,0.49]). VPS's D is strictly increasing in v_hi (slope sign parameter-independent), producing a monotone 15-point descending curve that crosses chance exactly where the exchange rate predicts; SDI's flat line cannot cross chance at any v_hi. FAMILY 3 -- VPS INDIFFERENCE RIDGE vs SDI COUNT GRADIENT (pairs 10-12): unmatched-count cells engineered so k*(c0 + c1*v_lo^kappa) ~= m*(c0 + c1*v_hi^kappa) at kappa~2 (the box center), pinning VPS at chance by arithmetic while SDI delivers a count-difference-graded majority for the fewer-endorsement option. Pair 10 (3x0.65 vs 2x0.85, count diff 1): VPS 0.54 (box [0.46,0.67]) vs SDI 0.29 (box [0.25,0.30]). Pair 11 (4x0.60 vs 2x0.90, count diff 2, since 2*0.9^2 = 4*0.6^2 - 0.18 with the c0 term closing the gap): VPS 0.49 (box [0.38,0.70]) vs SDI 0.16 (box [0.14,0.17]) -- SDI's majority deepens with count difference while VPS stays at chance, and even at VPS's most extreme box corner (0.38) it exceeds SDI's ceiling (0.17) by 21 points. Pair 12 (3x0.55 vs 2x0.85) is the fixed-count-ratio validity probe: same 3-vs-2 structure as pair 10 but with a larger coalition validity gap, VPS jumps from 0.54 to 0.68 (box [0.54,0.84], sign-robust) while SDI stays at 0.28 (box [0.25,0.31]) -- at fixed count difference, VPS's preference grows with the validity gap and SDI is structurally flat. UN-FAKEABILITY IN BOTH DIRECTIONS: (i) SDI cannot mimic Family 1 because its per-endorsement matched-count signal is bounded by |w_hi - w_lo| <= 0.064 plus alpha*Delta-logit <= 0.14, all of the WRONG SIGN (toward the high-validity side), so its P(A) is <= 0.50 on every matched-count pair while VPS's is >= 0.60 at k>=1 (>= 0.68 at k>=2); (ii) VPS cannot mimic Families 2-3 because on those cells its D is pinned within [-0.36, +0.39] (pair 11) and [-0.25, +0.48] (pair 7) by the arithmetic of k*v_lo^kappa ~= m*v_hi^kappa across its whole kappa box, keeping P(A) in [0.38,0.70] where SDI sits at 0.14-0.31 -- VPS has no parameterization that produces a 0.16 majority for the 2-endorsement option on pair 11, and SDI has none that produces a 0.59 majority for the 2-endorsement option on pair 7. The design also isolates VPS's two components: Family 1 cancels c0 exactly (pure c1*v^kappa assay), while Families 2-3 force c0 (the count term) to trade off against c1*v^kappa, with the ridge cells showing where they cancel. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- the closest cells, pairs 8 and 11, are precisely where the boxes are maximally separated), and every trial requires actually comparing the two rating vectors.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Matched-count validity-reversal index.\n    #\n    # On trials where A and B carry the SAME number of endorsements (k vs k),\n    # the two theories make opposite-sign predictions:\n    #   VPS: the coalition of LOWER-validity endorsers is preferred\n    #        (a high-validity endorsement is a bigger suspected defect),\n    #        with preference strengthening in k.\n    #   SDI: at matched count its validity sensitivity has the wrong sign\n    #        and is bounded, so it stays at or below chance for the\n    #        low-validity coalition on every such trial.\n    #\n    # The metric is the empirical proportion of choices of the\n    # LOW-validity coalition across all matched-count trials\n    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96\n    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.\n    # It is well-defined on a single subject's slice (48 Bernoulli\n    # trials) and on the pooled data.\n\n    v = np.array([\n        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,\n        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,\n    ], dtype=float)\n\n    hits = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:\n            continue\n\n        ca = float(a.sum())\n        cb = float(b.sum())\n        # Keep only matched-count trials with at least one endorsement\n        # on each side (the validity-reversal assay cells).\n        if ca == 0.0 or cb == 0.0 or ca != cb:\n            continue\n\n        # Mean validity of the endorsing coalition on each side.\n        va = float(np.mean(v[a > 0]))\n        vb = float(np.mean(v[b > 0]))\n        if va == vb:\n            continue\n\n        low_is_a = va < vb\n        chose_a = (int(row[\"response\"]) == 0)\n        hits.append(1.0 if (chose_a == low_is_a) else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The design's Family 1 (pairs 1-6 and their mirrors 13-18) is a pure matched-count assay: both options carry exactly k endorsements, so every pure count term (SDI's near-uniform skepticism weights at delta near 0, and VPS's flat c0 term) cancels exactly, leaving only the validity-sensitive component. There the two theories make strictly opposite-sign predictions that are parameter-robust across their entire boxes: VPS prefers the LOWER-validity coalition (mid-parameter P ~ 0.69, 0.82, 0.89, 0.93, 0.87, 0.93 for k = 1..4 ladders; box floors still >= 0.60 and rising with k), while SDI is pinned at or below 0.50 on every matched-count pair (its validity sensitivity has the wrong sign and is bounded by ~0.064 + alpha*Delta-logit, giving ~0.42-0.48 at mid parameters). The metric therefore is simply the proportion of choices of the low-validity coalition over all matched-count trials, identified stimulus-side from the rating vectors and the fixed validity list (order randomization and A/B mirroring are handled automatically because the side is computed per trial from content, not position). Expected values: VPS ~0.85, SDI ~0.45 \u2014 a mean gap of ~0.40. Per subject this is a proportion over 48 Bernoulli trials (binomial SD ~0.05-0.07), and even adding parameter heterogeneity across the boxes the per-subject SD stays ~0.08-0.09 on each side, so at N=25 the Welch t-statistic is on the order of 15+, far beyond the alpha=0.01 threshold. The metric works identically on the pooled DataFrame (canonical point estimate) and on each subject's slice (for the between-subject variance), and uses only scalar groupings (row iteration), avoiding any hashing of list-valued cells."
}
```

## Usage

```json
{
  "input_tokens": 9176,
  "output_tokens": 3837,
  "reasoning_tokens": 2742,
  "provider": "Sail Research"
}
```
