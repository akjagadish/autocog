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
**Validities (n_features=10):** [0.9, 0.85, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.65, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-4): a CARDINAL WEIGHT-MATCHING RIDGE built from duplicate moderate validities, with no zero-validity experts and no dominant top cue. Exp 4 dissociated VWEI from SDI structurally (50% experts are invisible to VWEI, penalized by SDI) and via 95%-anchored ceiling conflicts. This design instead engineers trials where VWEI's evidence sum is arithmetically pinned to ~0 by matching coalitions of moderate experts against single stronger experts, and dual trials where SDI's penalty difference is pinned to ~0 by matching endorsement counts. The validities [0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60] are chosen so that VWEI's amplified log-odds weights satisfy 2*w(0.75) = w(0.80) exactly at gamma = ln2/ln(1.2616) = 2.98 and 4*w(0.75) = w(0.85) exactly at gamma = ln4/ln(1.579) = 3.03 -- both crossing points sit at the CENTER of VWEI's own gamma box [2.8, 3.2], so the weight-matching holds (|E| <= 0.53, hence P within +/-2.5 points of 0.50) across the ENTIRE parameter box, verified at gamma in {2.8, 3.0, 3.2} x beta in {0.18, 0.23}. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- EVIDENCE-MATCHED COUNT GRADIENT (pairs 1-5): one option carries 1 (or 2) endorsements, the other carries a weight-matched coalition of 2-4 moderate experts (e.g., A = {85%} vs B = {80%, 75%, 75%}: E = 5.22 - 5.32 = -0.10). VWEI is pinned at chance: P(fewer-endorsement option) = 0.49-0.50 on all five pairs (box range 0.48-0.52) -- it has no mechanism, at any gamma/beta/epsilon in its box, for these stimuli to move it off a guess, because its cardinal weights cancel by construction. SDI climbs monotonically with endorsement-count difference: 0.69 (1v2), 0.70 (1v2, second realization), 0.83 (1v3), 0.83 (2v4), 0.89 (1v4) (box ranges 0.65-0.73 up to 0.86-0.92), since every net endorsement adds ~0.96 penalty units and its D strictly increases with count difference -- the slope sign is parameter-independent. Any systematic departure from 50% toward the fewer-endorsement option on pairs 1-5 falsifies VWEI; flatness at 50% falsifies SDI. FAMILY 2 -- COUNT-MATCHED VALIDITY-GAP GRADIENT (pairs 6-9): exactly one endorsement per side, with the endorser validity gap shrinking (.90 vs .60; .85 vs .65; .85 vs .80; .80 vs .75). SDI is pinned at 0.50-0.52 on all four (its penalty weights (1-v)^delta span <5% across the whole validity range at delta <= 0.05, and the alpha tie-break adds at most 3 points) -- structurally near-blind to WHO endorses at matched counts. VWEI sweeps 0.89 -> 0.73 -> 0.62 -> 0.56 (box ranges 0.84-0.95 down to 0.55-0.59), a 33-point monotone gradient in endorser validity that no SDI parameterization can produce. FAMILY 3 -- SIGN CONFLICT AT THE RIDGE CREST (pair 10): four 75% experts (total weight 5.30) vs one 80% expert (weight 2.66). VWEI: majority for the 4-endorsement coalition, P(A) = 0.63 (box 0.61-0.65). SDI: majority for the single endorsement, P(A) = 0.10 (box 0.08-0.12) -- opposite majorities, both far from chance, sign-robust across both boxes. BASELINE/SUB-RIDGE (pairs 11-12): pair 11 ({90%} vs {65%, 60%}) is the one agreement cell (VWEI 0.89, SDI 0.71 -- both prefer A, calibrating each theory's scale where they concur); pair 12 ({80%, 75%} vs {85%}) sits just BELOW the matching ridge (VWEI 0.43, SDI 0.30 -- both prefer B but at parametrically distinct depths, gap >= 9 points across boxes). THE DOUBLE DISSOCIATION IS UN-FAKEABLE: the two theories are placed in orthogonal indifference regimes -- VWEI is indifferent exactly when endorsement counts differ at matched evidence (Family 1), SDI is indifferent exactly when counts match but validity composition differs (Family 2) -- and no reparameterization of either theory can escape its own indifference condition (VWEI's weights cancel by arithmetic fixed at gamma ~ 3; SDI's penalty weights are count-driven and validity-flat by delta <= 0.05). A VWEI subject is a flat 50% line across the entire count gradient and a steep monotone curve across the gap gradient; an SDI subject is a steep monotone climb across the count gradient and a flat ~51% line across the gap gradient, plus a strong majority for the single endorser on pair 10 where VWEI favors the coalition. All pairs are non-degenerate (every pair has discriminating cues; no all-tie pairs; no pair leaves both theories guessing), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Double-dissociation composite built on the two strongest contrasts\n    # engineered into this design:\n    #\n    # COMPONENT 1 (\"big-gap single endorsers\", pairs 6/18, 8 trials/subject):\n    #   both options carry exactly ONE endorsement; the endorsers' validities\n    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the\n    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);\n    #   SDI is structurally near-blind at matched endorsement counts\n    #   (P ~ 0.50-0.55 across its whole box).\n    #\n    # COMPONENT 2 (\"ridge-crest coalition\", pairs 10/22, 8 trials/subject):\n    #   one option carries a single 0.80-validity endorsement, the other a\n    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing\n    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly\n    #   favors the single endorser (coalition P ~ 0.10).\n    #\n    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.\n    # Both components use only 8 trials each but their gaps point the SAME\n    # way after sign alignment, so the composite has a large mean gap and\n    # modest binomial-only between-subject variance.\n\n    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])\n\n    a_cells = data['option_a_ratings'].tolist()\n    b_cells = data['option_b_ratings'].tolist()\n    resp = data['response'].to_numpy()\n\n    scores_gap = []    # 1 = chose the higher-validity single endorser\n    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser\n\n    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        if a.shape != V.shape or b.shape != V.shape:\n            continue\n        ca = int(round(a.sum()))\n        cb = int(round(b.sum()))\n        chose_a = (int(r) == 0)\n\n        if ca == 1 and cb == 1:\n            # Single endorsement on each side: who endorsed, and how good\n            # is that expert?\n            ia = int(np.argmax(a))\n            ib = int(np.argmax(b))\n            va = V[ia]\n            vb = V[ib]\n            if abs(va - vb) >= 0.25:\n                hi_is_a = (va > vb)\n                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)\n\n        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):\n            # Ridge-crest conflict: verify it is exactly the designed cell\n            # (single 0.80 expert vs four 0.75 experts) and score the\n            # coalition choice.\n            if ca == 1:\n                single_vec, coal_vec = a, b\n                single_is_a = True\n            else:\n                single_vec, coal_vec = b, a\n                single_is_a = False\n            i_single = int(np.argmax(single_vec))\n            coal_idx = np.where(coal_vec > 0.5)[0]\n            if (\n                abs(V[i_single] - 0.80) < 1e-9\n                and len(coal_idx) == 4\n                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))\n            ):\n                # +1 when the subject chose the coalition side\n                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)\n\n    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5\n    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5\n    return 0.5 * (r_gap + r_coal)",
  "rationale": "The design's double dissociation is exploited at its two strongest, sign-aligned points. COMPONENT 1 uses the count-matched big-validity-gap trials (pairs 6/18: a single 0.90 endorsement vs a single 0.60 endorsement, 8 trials per subject). Here VWEI's amplified log-odds weights produce P(follow the 0.90 expert) \u2248 0.84-0.94 across its entire (gamma, beta, epsilon) box, while SDI is pinned near chance (0.50-0.55) because its penalty difference at equal endorsement counts is driven only by delta \u2264 0.05 and alpha \u2264 0.05 \u2014 it is structurally near-blind to WHO endorses when counts match. COMPONENT 2 uses the ridge-crest conflict (pairs 10/22: four 0.75 experts vs one 0.80 expert, 8 trials per subject), scored +1 for choosing the coalition. VWEI's evidence sum favors the coalition (P \u2248 0.61-0.65 box-wide, since 4*w(0.75) > w(0.80) at gamma \u2248 3), while SDI's penalty difference of ~3 units at beta \u2248 1 drives P(coalition) \u2248 0.08-0.12 box-wide. Averaging the two aligned rates yields an expected pooled value of ~0.75 under VWEI vs ~0.31 under SDI \u2014 a ~0.44 mean gap. Between-subject variance is essentially binomial (each component uses 8 trials; var of the average \u2248 0.01 for both theories), so at N = 25 the Welch t-statistic is expected around 15 (p << 0.001). The metric is a plain choice rate on structurally identified trial cells (no theory fitting, no likelihood evaluation), works identically on the pooled DataFrame and on single-subject slices, avoids hashing any sequence-valued column, and degrades gracefully to 0.5 if a cell is somehow absent. Note the sign alignment is essential: both components are scored in the direction VWEI predicts (follow the strong expert; follow the weight-matched coalition), so the two gaps add rather than cancel."
}
```

## Usage

```json
{
  "input_tokens": 6913,
  "output_tokens": 8982,
  "reasoning_tokens": 7438,
  "provider": "Sail Research"
}
```
