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
**Validities (n_features=20):** [0.92, 0.9, 0.9, 0.9, 0.85, 0.77, 0.77, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-8): COALITION COMPOSITION at matched endorsement count with arithmetically CANCELED trust, plus the COIN-FLIP EXCHANGE LOCUS. Exp 8 dissociated VPS from DCD using matched-count gradients among near-chance experts (0.55-0.70 vs 0.50) and two 0.90 anchors. It never pitted a MIXED coalition {v_hi, 0.50} against a HOMOGENEOUS coalition {v_mid, v_mid}, and never mapped how many coin-flip endorsements balance mid-validity endorsements. Both are direct assays of the one structural difference between the two confidence-distrust theories: the GEOMETRY of the distrust charge. VPS's charge c0 + c1*v^kappa uses RAW validity, so a 0.50 endorsement still costs c1*0.25-0.33 distrust units; DCD's charge c0 + c1*(2v-1)^kappa is DAMPENED to the bare floor c0 at v=0.50, and its residual trust term alpha*logit(v) partially refunds validity. Validities (20 experts): [0.92, 0.90x3, 0.85, 0.77x2, 0.75x6, 0.72x2, 0.50x5]. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- COMPOSITION SIGN CONFLICT (pairs 1-5): A = {v_hi, 0.50} vs B = {v_mid, v_mid} where v_mid is the LOGIT-MEAN of v_hi and 0.50, so DCD's trust term cancels by arithmetic identity: logit(0.90) = 2*logit(0.75) exactly (ln 9 = 2 ln 3; rungs 0.92/0.77 and 0.85/0.72 have residuals of only +0.003 and -0.015 at mid alpha). With trust gone, the pure charge shapes make OPPOSITE predictions, sign-robustly across the ENTIRE parameter boxes: VPS satisfies 2*v_mid^kappa > v_hi^kappa + 0.5^kappa for every kappa in [1.6, 2.4] at every rung (e.g. 0.90/0.75: 2*0.75^kappa - 0.9^kappa - 0.5^kappa in [+0.037, +0.087]), so the MIXED coalition is LESS suspicious and P(A) > 0.5 always (D >= 0.7*0.037 > 0). DCD satisfies (2v_hi-1)^kappa > 2*(2v_mid-1)^kappa for every kappa in [1.85, 2.15] (e.g. 0.8^kappa - 2*0.5^kappa in [0.122, 0.181]), so the mixed coalition is MORE suspicious and P(A) < 0.5 always (D <= -0.068 with trust residual <= +0.003). Mid-parameter predictions (VPS: kappa=2, beta=2, eps=.10, c0=.1, c1=1; DCD: kappa=2, beta=2.4, eps=.10, c0=.28, c1=.78, alpha=.10), with full-box ranges including lapse: pair 1 {90,50}v{75,75}: VPS 0.53 [0.51,0.57] vs DCD 0.44 [0.44,0.48] -- DISJOINT. Pair 2 (amplified x2): VPS 0.56 [0.52,0.62] vs DCD 0.39 [0.37,0.43] -- DISJOINT. Pair 3 (amplified x3): VPS 0.59 [0.52,0.68] vs DCD 0.33 [0.31,0.35] -- DISJOINT with an 18-point gap between interval edges. The amplification gradient is itself a signature: opposite-signed slopes in k (VPS rises 0.53->0.56->0.59; DCD falls 0.44->0.39->0.33), because both decision variables scale linearly in the number of replicated coalitions. Pairs 4-5 replicate the composition conflict at v_hi = 0.85 (vs 0.72) and 0.92 (vs 0.77): VPS 0.53 [0.51,0.55] and 0.54 [0.52,0.58] vs DCD 0.45 [0.44,0.47] and 0.45 [0.45,0.47] -- both disjoint, showing the effect spans the whole upper-validity range. FAMILY 2 -- EXCHANGE LOCUS (pairs 6-10): A = {75,75} (or {75x3}) vs B = k coin-flip endorsements, k = 2..5. VPS's internal exchange rate is (c0+c1*0.75^kappa)/(c0+0.25c1) ~ 1.9 coin-flips per 0.75 (it must 'pay' the substantial raw-v distrust of a 0.50 endorsement); DCD's is (c0+c1*0.5^kappa-alpha*1.0986)/c0 ~ 1.3 (a 0.50 endorsement costs only the floor, and trust refunds part of the 0.75 charge). The indifference point therefore differs by a full rung: VPS crosses 50% between k=3 and k=4 per two 0.75s; DCD crosses between k=2 and k=3. Pair 6 (k=2): VPS 0.25 [0.22,0.35] vs DCD 0.41 [0.38,0.48] -- disjoint magnitude conflict. Pair 7 (k=3): SIGN CONFLICT, VPS 0.38 [0.26,0.48] vs DCD 0.56 [0.48,0.64]. Pair 8 (k=4): VPS 0.53 [0.41,0.67] vs DCD 0.70 [0.60,0.76] -- DCD robustly prefers TWO 0.75 endorsements over FOUR coin-flips; VPS calls it near-chance. Pair 9 (k=5): agreement baseline (VPS 0.68, DCD 0.80). Pair 10 ({75x3} vs {50x5}, ratio 1.67 -- squarely between the two exchange rates): the strongest cell in the design, SIGN CONFLICT with VPS 0.40 [0.26,0.55] vs DCD 0.66 [0.53,0.71]; DCD's D > 0 by >= 0.076 at every box corner while VPS's mid D = -0.24. FAMILY 3 -- CALIBRATION (pairs 11-12): pair 11 ({90} vs {50}) both avoid the 0.90 endorser with DCD systematically softer (VPS 0.27 [0.20,0.34] vs DCD 0.35 [0.32,0.45]); pair 12 ({50,50} vs {90,90}) is a ceiling agreement cell (VPS 0.86, DCD 0.76) that scales each theory's slope where they concur. UNFAKEABILITY IN BOTH DIRECTIONS: (i) on all five composition cells, no DCD parameterization can reach P(A) = 0.5 (its D is negative by >= 0.068 with the trust residual bounded in [-0.019, +0.003]), and no VPS parameterization can fall to 0.5 (its D is positive by >= 0.026) -- the sign of the majority choice on any composition cell adjudicates the theories with no free parameters able to reverse it, and the k-amplification makes the P3 cell a 26-point mid separation; (ii) on the exchange family, the only VPS corner that reaches a positive decision variable on pair 10 (c0=0.15, c1=0.7, kappa=2.4) simultaneously predicts pair 6 at 0.34 (vs DCD 0.43) and pair 3 at 0.53 (vs DCD 0.31) -- a joint profile across cells that neither theory's box can mimic, since the composition conflict and the exchange conflict are two faces of the same structural difference (raw-v vs dampened charge) and no single reparameterization flips both. As the VPS advocate, the predicted profile is: majority for the MIXED coalition on all composition cells, growing with amplification (0.53 -> 0.59), and coin-flip avoidance that persists through k=3 (0.38) with an indifference point near k=4. A DCD subject instead shows majority-HOMOGENEOUS on all composition cells (falling to 0.33 at k=3), near-chance already at k=3 on the exchange gradient (0.56), and a robust preference for two 0.75s over four coin-flips (0.70). All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # COIN-FLIP ENDORSEMENT TOLERANCE.\n    #\n    # For every trial, count how many endorsements each option receives\n    # from the five chance-validity (v = 0.50) experts (feature indices\n    # 15-19 in this experiment's fixed validity vector). Every one of the\n    # 24 unique pairs pits an option carrying coin-flip endorsements\n    # against an option carrying none, so the trial is scored:\n    #   1.0  if the subject chose the option with MORE 0.50-expert\n    #        endorsements (the 'coin-flip tolerant' choice),\n    #   0.0  otherwise.\n    # The metric is the mean of this score over all trials.\n    #\n    # This is a direct assay of the one structural difference between the\n    # two confidence-distrust theories: the geometry of the per-endorsement\n    # distrust charge. VPS charges c0 + c1*v^kappa, so a 0.50 endorsement\n    # still costs a substantial c1*0.25-0.33 distrust units and the\n    # internal exchange rate between mid-validity (0.75) endorsements and\n    # coin-flips is ~1.9 coin-flips per 0.75. DCD charges\n    # c0 + c1*(2v-1)^kappa, which collapses to the bare floor c0 at\n    # v = 0.50 (and refunds part of the 0.75 charge via the residual trust\n    # term), yielding an exchange rate of only ~1.3. Hence VPS subjects\n    # are systematically MORE willing to accept coin-flip endorsements,\n    # across every cell family in the design:\n    #   composition cells (mixed {v_hi, 0.50} vs homogeneous mid): VPS\n    #     picks the mixed side (0.53-0.59), DCD avoids it (0.33-0.45);\n    #   exchange cells (0.75-coalition vs k coin-flips): VPS keeps\n    #     choosing the coin-flip side through k=3-4, DCD abandons it;\n    #   calibration cells ({0.90} vs {0.50}, {0.50,0.50} vs {0.90,0.90}):\n    #     VPS 0.73/0.86 vs DCD 0.65/0.76.\n    # Mid-parameter predictions: VPS ~0.59, DCD ~0.45 (disjoint across\n    # their full parameter boxes), with per-subject SD ~0.05-0.06 from\n    # 96 Bernoulli trials.\n\n    validities = np.array([\n        0.92, 0.90, 0.90, 0.90, 0.85, 0.77, 0.77,\n        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72,\n        0.50, 0.50, 0.50, 0.50, 0.50,\n    ])\n    coin_mask = np.isclose(validities, 0.5)\n\n    responses = data[\"response\"].to_numpy()\n    ratings_a = data[\"option_a_ratings\"].tolist()\n    ratings_b = data[\"option_b_ratings\"].tolist()\n\n    scores = []\n    for i in range(len(ratings_a)):\n        a = np.asarray(ratings_a[i], dtype=float).ravel()\n        b = np.asarray(ratings_b[i], dtype=float).ravel()\n        n = min(a.shape[0], b.shape[0], validities.shape[0])\n        mask = coin_mask[:n]\n        ca = float(np.sum(a[:n][mask]))\n        cb = float(np.sum(b[:n][mask]))\n        if ca == cb:\n            # No coin-flip asymmetry on this trial; skip defensively.\n            continue\n        chose_a = (int(responses[i]) == 0)\n        c_chosen = ca if chose_a else cb\n        c_other = cb if chose_a else ca\n        scores.append(1.0 if c_chosen > c_other else 0.0)\n\n    if len(scores) == 0:\n        return 0.5\n    return float(np.mean(scores))",
  "rationale": "The metric is the proportion of trials on which the subject chose the option carrying MORE endorsements from the five v=0.50 (coin-flip) experts. Every one of the 24 unique pairs (96 trials/subject, 4 reps each) has a strict coin-flip asymmetry, so all trials contribute, and the metric is a simple Bernoulli proportion that works identically on the pooled data and on a single subject's 96-trial slice. It targets the single structural difference between the theories: VPS's raw-v charge c0+c1*v^kappa makes a 0.50 endorsement cost substantial distrust (exchange rate ~1.9 coin-flips per 0.75 endorsement), while DCD's dampened charge c0+c1*(2v-1)^kappa collapses to the floor c0 at v=0.50 with a trust refund (exchange rate ~1.3). Consequently, on every cell family the VPS subject is more coin-flip tolerant: composition cells (mixed {v_hi,0.50} vs homogeneous mid-validity) give VPS 0.53/0.56/0.59 (pairs 1-3, amplified) vs DCD 0.44/0.39/0.33 \u2014 opposite-signed majorities that no parameterization in either box can reverse; exchange cells (75-coalition vs k coin-flips) give VPS 0.75/0.62/0.47/0.32/0.60 vs DCD 0.59/0.44/0.30/0.20/0.34; calibration cells give VPS 0.73/0.86 vs DCD 0.65/0.76. Averaging over all 24 pairs: VPS mid-parameter value \u2248 0.59, DCD \u2248 0.45 \u2014 a 0.146 mean gap that is disjoint across the two theories' full parameter boxes (VPS per-cell box ranges stay above DCD's on essentially every cell, with the composition cells sign-robust). Per-subject standard deviation is ~0.05 (binomial over 96 trials) plus modest parameter heterogeneity (~0.02-0.04), giving Welch t \u2248 8-9 at N=25, far below alpha=0.01. Averaging over all cells (rather than only the sign-conflict cells) maximizes the standardized gap because the added trials shrink the binomial variance faster than the moderate cells dilute the mean difference, and it makes the metric a single interpretable behavioral quantity \u2014 coin-flip endorsement tolerance \u2014 rather than a hand-picked subset of trials."
}
```

## Usage

```json
{
  "input_tokens": 9521,
  "output_tokens": 17069,
  "reasoning_tokens": 15476,
  "provider": "Sail Research"
}
```
