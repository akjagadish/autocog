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
**Validities (n_features=12):** [1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. SCWI and PAHI are structurally near-identical: both are anti-validity additive integrators (a rating of 1 counts AGAINST an option, tied cues cancel exactly, choice is a softmax plus symmetric lapse). The ONLY substantive difference is the validity-to-weight mapping: SCWI uses w(v) = v + tau*v^2 with tau in [0.25, 0.45]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Two mathematical facts drive the whole design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-elasticity is (1+2*tau*v)/(1+tau*v) <= 1.9/1.45 = 1.31 everywhere on v in [0.5, 1], while PAHI's elasticity is gamma >= 1.4. Therefore for ANY two validities, PAHI's relative weight w(v_hi)/w(v_lo) strictly exceeds SCWI's: PAHI is a uniformly steeper validity gradient. (2) LEVEL CROSSING: at v = 1.0 PAHI's weight is EXACTLY 1 (gamma-invariant) versus SCWI's 1.25-1.45, while at v = 0.5 SCWI's weight is 0.56-0.61 versus PAHI's 0.19-0.38. Consequence: on 'concentration' trials pitting a few high-validity endorsements against many low-validity endorsements, the preference flips at a knife-edge count ratio that differs between the theories, and integer ratios can be placed in the gap between the two knife-edges. THE KILL ZONE: with validities exactly 1.00 (two cues) and 0.50 (eight cues), let d = (number of 0.5-cue 1s on the low-cue option) - (number of 0.5-cue 1s on the top-cue option). SCWI prefers the top-cue option H iff d > 2(1+tau)/(0.5+0.25*tau), i.e., d in (4.44, 4.74) over the whole tau box -> flips strictly between d=4 and d=5. PAHI prefers H iff d > 2/0.5^gamma = 2^(gamma+1) in (5.28, 10.55) -> flips at d=6 or LATER, never before 5.28. So at d=5 the two theories make OPPOSITE predictions for every admissible parameter value: SCWI P(H) in [0.54, 0.62] (its evidence is 0.5-0.75*tau > 0 since tau < 2/3), PAHI P(H) in [0.04, 0.46] (its evidence is 5*0.5^gamma - 2 < 0 since gamma > log2(2.5) = 1.32). Note v = 1.0 exactly is essential: with v = 0.99, SCWI's flip condition tau < 0.389 fails inside its own box, destroying the parameter-free separation. PRIMARY INSTRUMENT (pairs 1-12): a d-sweep from 3 to 8, fully mirrored. SCWI predicts a psychometric curve crossing 0.5 strictly between d=4 (evidence = -tau < 0 always) and d=5, then saturating (P(H) ~ 0.67/0.77/0.84 at d=6/7/8; at d=8 its evidence is exactly +2.0 for ALL tau - a parameter-free anchor). PAHI predicts a crossing at d* = 2^(gamma+1) in (5, 10.6): at or beyond d=6 for gamma >= 1.59, and for gamma > 2 the curve never exceeds 0.5 within the observable range d <= 8. Three parameter-free statistics separate them: (i) the SIGN of P(H) at d=5 (16 trials per subject: pairs 3, 9, 13, 14 x 4 reps); (ii) the CROSSING LOCATION (SCWI: 4-5 for every subject, since the tau range is too narrow to move it; PAHI: >= 5.3, heterogeneous across subjects - SCWI predicts essentially zero between-subject variance in crossing point, PAHI predicts large variance); (iii) the saturation level at d=8 (SCWI >= 0.84; PAHI can sit below 0.5). SECONDARY KILLS at other validity mixtures, so the dissociation is not an artifact of the 1.0/0.5 pair alone: pair 15/16 (G): {1.0, 1.0, 0.85} vs seven 0.5-cues - SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option, PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.14, 1.34] favors the low option; P(H) in [0.55, 0.66] vs [0.10, 0.46]. Pair 17/18 (S): two 0.85-cues vs four 0.5-cues - SCWI margin 0.3-0.445*tau > 0 always (tau < 0.674) favors the 0.85 option, PAHI margin 0.5^gamma*(1.7^gamma - 2) > 0 always (gamma > 1.305) favors the 0.5 option. Pair 19/20 (M2): two 1.0-cues vs {0.85, 0.85, 0.5} - SCWI margin 0.2-0.305*tau > 0 always (tau < 0.656), PAHI margin 2-2*0.85^gamma-0.5^gamma in [0.03, 0.48] > 0 always. GAMMA-LADDER (pairs 21/22 and d=6/7/8): pair W ({1.0, 0.85} vs five 0.5-cues) splits PAHI at gamma ~= 1.47; d=6 splits at 1.59; d=7 at 1.81; d=8 at 2.0. SCWI predicts uniformly high P(H) on all of W/d6/d7/d8 (>= 0.55/0.67/0.77/0.84). If PAHI is correct, the observed flips across this ladder must recover a single consistent gamma AND that gamma must place the d=5 choice below 0.5; if SCWI is correct, all ladder points sit high while d=5 sits above 0.5. ROBUSTNESS VARIANT (pairs 13/14): d=5 realized with an extra 1 on BOTH options (3 vs 6 total 1s instead of 2 vs 5) - identical difference, different totals; both theories predict behavior identical to pair 3, ruling out raw count-of-1s heuristics. CONTROLS: pair 23 (all-1 vs all-0) is a maximum-separation anti-validity anchor where both theories predict P(all-1 option) ~= epsilon/2, calibrating the lapse; pair 24 (identical mixed vectors) is a full tie where both predict exactly 0.500, calibrating side/response bias. All critical pairs are mirrored to absorb side bias. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. DECISION RULES: (i) If the d=5 composite H-choice rate exceeds 0.5 (with d=4 below and d=6 above), SCWI is supported and PAHI is falsified - PAHI cannot exceed 0.463 there for any (beta, gamma, epsilon) in its box. (ii) If the d=5 rate is below 0.5, or the psychometric crossing sits at d >= 6 (or never), PAHI is supported and SCWI is falsified - SCWI cannot cross later than 4.74 low-cue units. (iii) If the crossing point varies widely across subjects (some never crossing by d=8), that specifically favors PAHI's heterogeneous gamma over SCWI's narrow tau. (iv) If the secondary kills G/S/M2 flip sign relative to the primary kill (e.g., high option preferred at 1.0/0.5 but not at 0.85/0.5), BOTH pure forms are falsified, since each theory's weight-shape prediction is sign-consistent across all these mixtures by construction.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Parameters:**
- beta: [1.1, 1.7]
- tau: [0.25, 0.45]
- epsilon: [0.05, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

**Parameters:**
- beta: [1.5, 4.0]
- gamma: [1.4, 2.4]
- epsilon: [0.06, 0.20]
- sign_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Instructed validities of this experiment (fixed at design time).\n    VAL = np.array([1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])\n    n_f = VAL.size\n\n    # Extreme points of each theory's ADMISSIBLE parameter box.\n    # SCWI: w(v) = -(v + tau*v^2), tau in [0.25, 0.45].\n    W_SC = [-(VAL + t * VAL ** 2) for t in (0.25, 0.45)]\n    # PAHI: w(v) = -v^gamma, gamma in [1.4, 2.4].\n    W_PA = [-(VAL ** g) for g in (1.4, 2.4)]\n\n    # Minimum |evidence margin| required to call a preference \"guaranteed\"\n    # across the whole parameter box of a theory.\n    M_SC = 0.08\n    M_PA = 0.05\n\n    def parse_vec(cell):\n        if isinstance(cell, str):\n            s = cell.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            return np.array([float(t) for t in toks], dtype=float)\n        return np.asarray(cell, dtype=float).ravel()\n\n    def parse_resp(r):\n        if isinstance(r, str):\n            rs = r.strip().upper()\n            if rs == 'A':\n                return 0\n            if rs == 'B':\n                return 1\n            return int(float(rs))\n        return int(round(float(r)))\n\n    hits = []\n    for a_cell, b_cell, r_cell in zip(data['option_a_ratings'],\n                                       data['option_b_ratings'],\n                                       data['response']):\n        try:\n            a = parse_vec(a_cell)\n            b = parse_vec(b_cell)\n            if a.size != n_f or b.size != n_f:\n                continue\n            d_sc = [float(np.dot(w, a) - np.dot(w, b)) for w in W_SC]\n            d_pa = [float(np.dot(w, a) - np.dot(w, b)) for w in W_PA]\n            resp = parse_resp(r_cell)\n        except Exception:\n            continue\n\n        sc_lo, sc_hi = min(d_sc), max(d_sc)\n        if sc_lo > M_SC:\n            sc_pref = 1      # SCWI prefers A for EVERY admissible tau\n        elif sc_hi < -M_SC:\n            sc_pref = -1     # SCWI prefers B for EVERY admissible tau\n        else:\n            continue         # preference not guaranteed -> skip\n\n        pa_lo, pa_hi = min(d_pa), max(d_pa)\n        if pa_lo > M_PA:\n            pa_pref = 1      # PAHI prefers A for EVERY admissible gamma\n        elif pa_hi < -M_PA:\n            pa_pref = -1     # PAHI prefers B for EVERY admissible gamma\n        else:\n            continue\n\n        if sc_pref == pa_pref:\n            continue         # theories agree -> no diagnostic content\n\n        # Trial is a guaranteed sign-opposition trial: SCWI's preferred\n        # option is exactly the one PAHI rejects (and vice versa).\n        scwi_option_is_a = (sc_pref == 1)\n        chose_scwi = (resp == 0) if scwi_option_is_a else (resp == 1)\n        hits.append(1.0 if chose_scwi else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "METRIC: the choice rate of SCWI's preferred option on the subset of trials where the two theories make OPPOSITE predictions for EVERY admissible parameter value (a parameter-free sign separation). The selection is computed inside the metric from the ratings and the known validities: a trial is kept iff (i) SCWI's evidence difference has a fixed sign with margin >= 0.08 at BOTH extremes of its tau box [0.25, 0.45], (ii) PAHI's evidence difference has a fixed sign with margin >= 0.05 at BOTH extremes of its gamma box [1.4, 2.4], and (iii) the two guaranteed signs are OPPOSED. This rule provably selects exactly the 8 designed kill pairs x 4 reps = 32 trials per subject: the d=5 trials (pairs 3, 9 and the robustness variants 13, 14), the G trials (15, 16: {1.0,1.0,0.85} vs seven 0.5-cues), and the S trials (17, 18: two 0.85-cues vs four 0.5-cues). WHY THESE TRIALS: (a) d=5: SCWI's evidence for the top-cue option is 0.5-0.75*tau in [0.16, 0.31] > 0 for all tau < 2/3, while PAHI's is 5*0.5^gamma - 2 in [-1.05, -0.11] < 0 for all gamma > log2(2.5) = 1.32 \u2014 SCWI P(H) in [0.54, 0.61], PAHI P(H) <= 0.47. (b) G: SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option; PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.15, 1.35] favors the low option. (c) S: SCWI margin 0.3-0.445*tau in [0.10, 0.19] favors the 0.85 option; PAHI margin 2*0.85^gamma-4*0.5^gamma in [0.08, 0.60] favors the 0.5 option. All other trials are excluded precisely because they cannot separate the theories for all parameter values: d=3,4 (both prefer the low-cue option), d=6,7,8 and W (PAHI's preference flips inside its gamma box, so no guaranteed sign), M2 (margins too thin), and the controls (ties / both agree). EXPECTED VALUES: under SCWI the metric is a weighted average of per-type probabilities all in [0.52, 0.60] -> population mean ~0.57, and because SCWI's parameter box is narrow the per-subject true rate varies only over ~[0.55, 0.58]; the dominant per-subject noise is binomial (32 trials, sd ~ 0.09), giving between-subject variance ~0.008. Under PAHI every selected trial has P(H) < 0.5 for every (beta, gamma, epsilon), so the population mean is ~0.20-0.25; its wider box produces per-subject spread roughly [0.05, 0.47] (var ~0.017) plus binomial noise (~0.005), total ~0.022. Welch's t-test at N=25 on (0.57, 0.008) vs (0.21, 0.022) gives t ~ 10 (p << 0.001): a large mean gap that is NOT drowned by per-subject variance, because the metric deliberately (i) uses only guaranteed-sign trials (so PAHI's heterogeneity cannot push any subject above SCWI's range on the selected set) and (ii) pools 32 trials per subject to shrink binomial noise. On real human data, if SCWI is correct the metric should land near 0.55-0.60; PAHI predicts at most ~0.47 and realistically ~0.2, so the metric cleanly adjudicates between them."
}
```

## Usage

```json
{
  "input_tokens": 7271,
  "output_tokens": 31505,
  "reasoning_tokens": 29655,
  "provider": "Sail Research"
}
```
