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
**Validities (n_features=10):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 25: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 26: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 27: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 29: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 31: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-3): ZERO-WEIGHT EXPERTS. All prior designs dissociated theories that all read endorsements as positive evidence. SDI versus VWEI is a clash of SIGN and of WEIGHTING PROFILE: SDI treats every endorsement as a liability (penalty d_j = (1-v_j)^delta, nearly uniform at ~0.86-1.0 across the whole validity range for delta in [0, 0.05]) and prefers the option with FEWER endorsements; VWEI weights endorsements by amplified log-odds (w_j = logit(v_j)^gamma, enormous spread: 25.5 at v=.95 down to 0.61 at v=.70) and prefers the option with MORE validity-weighted wins. The design's keystone is four experts at validity EXACTLY 0.50: VWEI's weight for a 50% expert is logit(0.5)^gamma = 0 for EVERY gamma — a 50% expert's endorsement is structurally invisible to VWEI under any parameterization — while SDI's skepticism weight is (1-0.5)^delta in [0.966, 1], its NEAR-MAXIMUM penalty. A coin-flip expert's praise is, for SDI, a full unit of suspicion. Validities: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50, 0.50, 0.50, 0.50] (10 experts; VWEI weights at gamma=3: 25.5, 10.6, 5.2, 2.7, 1.3, 0.6, 0, 0, 0, 0; SDI penalty weights at delta=0.025: 0.93, 0.94, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98). 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject). In every base pair VWEI's winner is A, so the design is perfectly side-balanced. FOUR QUANTITATIVE SIGNATURES (P(A) at SDI mid-params delta=.025, beta=1, eps=.115, alpha=.025; VWEI mid-params gamma=3, beta=.205, eps=.015; all signs verified robust across the FULL parameter boxes): (1) JUNK-ENDORSEMENT GRADIENT, VWEI STRUCTURALLY FLAT (pairs 1-5): the 95% expert endorses A; k = 0..4 of the 50% experts also endorse A. VWEI: E = 25.5 regardless of k (50% weight is exactly 0 at any gamma), so P(A) is flat at 0.98-0.99. SDI: each junk endorsement adds ~0.98 penalty units to A, so P(A) declines monotonone 0.32 -> 0.18 -> 0.11 -> 0.08 -> 0.06 (~6-7 points per junk endorsement; box ranges 0.32-0.33 down to 0.05-0.08). ANY measurable effect of 50%-expert endorsements on choice falsifies VWEI outright — it has no mechanism, at any gamma, beta, or epsilon, for a zero-weight cue to move its prediction off a constant; flatness at ~0.99 with no decline falsifies SDI. Pairs 6-7 run the mirror logic (junk endorsements on B): VWEI flat 0.99; SDI rises 0.53 -> 0.72 as B accumulates 'suspicion'. (2) OPPOSITE-SLOPE DOUBLE DISSOCIATION (pairs 10-14): cue 1 (95%) endorses A while the opposing coalition among cues 2-6 grows from 1 to 5. VWEI: E = 25.5 - cumulative weight, so P(A) DECLINES 0.95 -> 0.87 -> 0.81 -> 0.76 -> 0.74. SDI: each opposing endorsement adds a penalty to B, so P(A) RISES 0.51 -> 0.69 -> 0.82 -> 0.89 -> 0.92. The two profiles are anti-correlated and CROSS between k=2 and k=3 — slope signs are parameter-independent (VWEI's E strictly decreases in k because every added cue has positive weight; SDI's D strictly increases because every added endorsement adds positive penalty), so the sign of the empirical slope across this 5-point gradient adjudicates the theories with no free parameters able to reverse it. (3) SIGN CONFLICT AT CEILING (pairs 8-9, plus mirrors 24-25): A endorsed by 3 (then 5) high-validity experts, B by none. VWEI: P(A) ~ 0.99. SDI: P(A) = 0.12 (3 endorsements) falling to 0.07 (5 endorsements) — majority choices OPPOSITE to VWEI on stimuli where VWEI is at ceiling, which is SDI's novel falsifiable prediction (high-validity option carrying MORE endorsements). SDI additionally predicts a within-family decline (0.12 -> 0.07) where VWEI is flat at ceiling. (4) VALIDITY-BLINDNESS AT MATCHED ENDORSEMENT COUNTS (pairs 15-16): exactly one endorsement per option, varying expert identity (95% vs 70%; 90% vs 70%). SDI: P(A) = 0.51-0.52 regardless — its penalty weights span only ~9% across the entire validity range at delta <= 0.05, so at matched counts SDI is essentially indifferent to WHO endorses (its only tilt, +1-2 points toward the higher-validity endorser, comes from delta and the alpha tie-break). VWEI: 0.99 and 0.88 — a 40+ point sensitivity to expert identity that SDI cannot produce under any delta in [0, 0.05]. No pair is degenerate: every pair has discriminating cues, no pair leaves both theories guessing (the closest SDI gets to 0.5 is on D-1/E pairs where VWEI is at 0.88-0.95), and every pair requires actually comparing the two rating vectors. The conjunction is un-fakeable in both directions: a VWEI subject is a flat line at ~0.99 across the entire junk gradient and a declining line across the coalition gradient; an SDI subject is a specific descending curve on junk endorsements, a specific ascending curve on opposing coalitions, near-50% at matched counts, and B-majorities on the ceiling-conflict pairs. Neither theory's parameter box can approximate the other's profile because the key contrasts (zero vs. near-maximal weight on 50% experts; fewer- vs. more-endorsements preference) are structural, not parametric.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



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
**Description:** Validity-Weighted Evidence Integration (VWEI) with subjective validity amplification. On each trial the decision maker computes, for every rating dimension j, the discriminant d_j = sign(a_j - b_j) in {-1, 0, +1}, weights it by a subjectively amplified validity w_j = [log(v_j/(1-v_j))]^gamma, and accumulates evidence E = sum_j w_j * d_j. Choice probability is a sigmoid of beta * E mixed with a small uniform lapse epsilon. gamma = 1 recovers Bayes-optimal log-odds integration; gamma -> infinity recovers Take-The-Best; the empirically human region sits at gamma near 3, where the top cue dominates coalitions of weaker cues on steep validity gradients (Exp 2) but lower cues still attenuate allegiance on shallow gradients (Exp 1). This round's refinement is purely parametric: the mechanism is unchanged, but the sampling region is re-centered on the hand-verified sweet spot (gamma = 3, beta = 0.2, epsilon = 0), the lapse range is restored to (a tightened version of) the arbiter's [0, 0.2], and the beta floor is raised so that generic flattening noise no longer competes with the gamma-structured weight profile for attenuation work.

**Parameters:**
- gamma: [2.8, 3.2]
- beta: [0.18, 0.23]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Weighted Evidence Integration (VWEI) with subjective
    # validity amplification. Stimulus is the pair of option feature
    # vectors for the current trial: array-like of shape
    # (2, n_features), row 0 = option A, row 1 = option B.
    #
    # Evidence for A over B:  E = sum_j w_j * sign(a_j - b_j),
    # with w_j = [log(v_j / (1 - v_j))]^gamma.
    #   gamma = 1  -> Bayes-optimal log-odds weighting (full integration)
    #   gamma -> inf -> Take-The-Best (top cue dominates every coalition)
    # P(A) = sigmoid(beta * E), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the log-odds against v = 1 (infinite weight) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)
    log_odds = np.log(v / (1.0 - v))

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Subjective cue weights: amplified log-odds validity.
    w = np.sign(log_odds) * np.abs(log_odds) ** gamma

    # Per-cue discriminant: +1 favors A, -1 favors B, 0 is uninformative.
    d = np.sign(a - b)

    # Accumulated evidence for A (positive) vs B (negative).
    E = float(np.dot(w, d))

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # When E == 0 this is exactly 0.5 (uniform guessing), for any beta.
    x = beta * E
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Top-cue allegiance under endorsement-majority conflict.\n\n    Selected trials: the 95%-validity expert (feature index 0) endorses\n    exactly one option, AND that option carries strictly MORE total\n    endorsements (sum of ratings) than its rival.  On these trials the\n    metric is the empirical rate of choosing the option endorsed by the\n    top expert.\n\n    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98\n    here: the amplified log-odds weight of v=.95 dominates every\n    possible opposing coalition of the remaining cues, and 50%-expert\n    endorsements contribute exactly zero weight.\n    SDI must AVOID the top-cue option here (~0.15-0.25): every\n    endorsement is a liability, and the top-cue option is also the\n    endorsement-majority option.\n    \"\"\"\n    df = data.reset_index(drop=True)\n\n    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)\n    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)\n    resp = df['response'].to_numpy(dtype=float)\n\n    if a.ndim != 2 or a.shape[0] == 0:\n        return 0.5\n\n    chose_a = resp < 0.5\n\n    # Which option does the top (95%) expert endorse?\n    top_a = a[:, 0] > 0.5\n    top_b = b[:, 0] > 0.5\n    # Keep only trials where exactly one option gets the top expert's\n    # endorsement (guards against degenerate rows).\n    xor_top = np.logical_xor(top_a, top_b)\n\n    # Total endorsement counts per option.\n    cnt_a = a.sum(axis=1)\n    cnt_b = b.sum(axis=1)\n\n    # Is the top-cue-endorsed option also the endorsement-majority\n    # option (strictly more endorsements)?\n    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)\n\n    sel = xor_top & top_option_more\n    if not np.any(sel):\n        return 0.5\n\n    # Did the subject choose the option endorsed by the top expert?\n    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])\n    return float(np.mean(followed))",
  "rationale": "This metric isolates the sharpest structural sign clash in the design. Ten of the 32 base pairs (trials 1-5, 8-9 and their A/B mirrors 17-21, 24-25; 30 observations per subject after 3 repetitions) have the property that the 95%-validity expert endorses one option and that same option carries strictly more total endorsements than its rival (which is either empty or carries only junk 50%-expert endorsements). On exactly these trials the two theories make opposite majority predictions with no parameter able to reverse the sign. VWEI: the v=.95 cue's weight logit(.95)^gamma (~25.5 at gamma=3, still the largest weight for any gamma in [2.8,3.2]) dominates every possible opposing coalition of cues 2-6, and 50%-expert endorsements carry exactly zero weight at every gamma, so P(follow top cue) sits at ~0.97-0.99 (sigmoid(0.205*25.5) with the small lapse). SDI: every endorsement adds a near-unit penalty (d_j in [0.93,0.98] for delta in [0,0.05]), and the top-cue option is always the endorsement-majority option here, so P(follow top cue) averages ~0.18 pre-lapse, ~0.22 after the epsilon lapse \u2014 i.e., the majority choice is the LESS-endorsed option. Expected gap ~0.75 (SDI ~0.22 vs VWEI ~0.98) with per-subject binomial standard error of only ~0.07 (30 trials, p near 0.2 or 0.98), so at N=25 the Welch t-test separates the theories by dozens of standard errors. The metric is a plain choice rate, so it is well defined both on the pooled DataFrame and on each single subject's slice, and it uses only scalar hashing of the response column (sequence columns are converted to arrays, never hashed). Unlike a slope estimate over the junk-endorsement gradient (which is noisy at 6 samples per cell), this is a simple proportion over 30 trials per subject, keeping between-subject variance tight. It is falsifiable in both directions: near-ceiling following of the top cue under endorsement-majority conflict refutes SDI; majority avoidance of the top cue refutes VWEI, which has no mechanism (at any gamma, since gamma only rescales positive log-odds weights) for preferring the option with fewer endorsements."
}
```

## Usage

```json
{
  "input_tokens": 7649,
  "output_tokens": 7481,
  "reasoning_tokens": 6370,
  "provider": "Sail Research"
}
```
