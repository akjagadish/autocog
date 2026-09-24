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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 14: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 15: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 17: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 18: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 19: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 20: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 22: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 24: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. The advocated theory (SCWI) and the competitor (Tallying) have a deep structural relationship on binary cues: because SCWI's signed value difference between options is sum_j w_j*(a_j - b_j) with w_j = -(v_j + tau*v_j^2), tied cues cancel exactly, and SCWI effectively computes an ANTI-VALIDITY-WEIGHTED tally: it prefers the option that LOSES on more (and more heavily weighted) cues. Tallying prefers the option that WINS on more cues, ignoring validity entirely. Two things follow. (1) SIGN: on every trial with a nonzero tally margin the two theories prefer OPPOSITE options. (2) SLOPE: Tallying's evidence is the raw count margin; SCWI's evidence is the validity-weighted (inverted) margin, so the two theories order trials differently. The design exploits both, with a parameter-free kill shot: on any trial where the feature-wise counts are TIED (a_wins = b_wins), Tallying's softmax is exactly uniform for ANY beta and ANY epsilon (lapse is symmetric), pinning P(A) = 0.500 exactly. SCWI is not so pinned: a count tie with unequal validity composition yields a nonzero weighted margin. PROBE A (pairs 1-8, count ties): I construct 8 count-tie trials whose SCWI weighted margin M = sum|w| over B's winning cues minus sum|w| over A's winning cues sweeps from -0.92 to +0.92 (computed at tau=0.35; the sweep is robust across tau in [0.25,0.45]). Tallying predicts EXACTLY 0.500 on all eight, a horizontal line with zero free parameters. SCWI predicts a monotone psychometric sweep of P(A) from ~0.25 up to ~0.75, with the specific ordering implied by the supralinear anti-validity weights (e.g., pair 3, where A wins the two top experts and B wins the two worst, yields P(A)~0.25; pair 4, the cue-label swap, yields ~0.75). Any reliable deviation from 0.5 on these trials falsifies Tallying outright; the graded M-ordering then tests SCWI quantitatively. PROBE B (pairs 9-15, tally margin fixed at +1 for A: A wins exactly 3 cues, B wins 2): Tallying predicts a FLAT P(A) = sigmoid(beta)*(1-eps)+eps/2 on all seven, while SCWI predicts a strictly graded function of which cues A wins — from P(A)~0.11 when A wins the three most trusted experts (its 'defect load' is maximal) to P(A)~0.49 when A wins the three least trusted. This is a flat-vs-graded slope dissociation at fixed tally evidence, and every SCWI prediction sits BELOW 0.5 while Tallying's sits ABOVE 0.5: a double (sign + slope) dissociation. PROBE D (pairs 16-19) mirrors four Probe-B trials (A/B swapped) to absorb side and response bias; Tallying stays flat at sigmoid(-beta), SCWI stays graded above 0.5. ANCHORS (pairs 20-23): extreme trials where A wins all cues (pair 20), B wins all (21), or A wins four low-validity cues against B's single top cue (22/23). Tallying approaches deterministic choice of the tally winner (sigmoid(5*beta), sigmoid(3*beta)); SCWI approaches deterministic choice of the tally LOSER, with confidence tracking the weighted margin M rather than the count margin — plotting P(A) against count margin gives a rising function under Tallying and a falling function under SCWI. Pair 24 is a full tie (identical vectors) where both theories predict exactly 0.5, providing a pure lapse/side-bias baseline. With 24 unique pairs, K = 96//24 = 4 repetitions (96 trials). DECISION RULE: (i) if P(A) on pairs 1-8 is reliably different from 0.5 in either direction, Tallying is falsified (it cannot produce this for any parameter values); (ii) if P(A) on pairs 9-15 is flat, Tallying is supported and SCWI is falsified; (iii) if P(A) across pairs 1-15 is a monotone function of the anti-validity weighted margin M (increasing in M, crossing 0.5 near M=0, with the tally loser preferred), SCWI is supported; (iv) a preference for the tally WINNER that is nonetheless graded by validity weights would falsify both pure forms (it would indicate pro-validity weighted integration).

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
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Slope of P(choose A) regressed on the anti-validity weighted cue\n    margin D = sum_j u_j * (b_j - a_j), with u_j = v_j + tau * v_j^2\n    (tau fixed at 0.35, the midpoint of SCWI's tau range).\n\n    D > 0 means option A carries fewer/smaller anti-validity 'defects'\n    and is therefore preferred by SCWI. SCWI predicts a POSITIVE slope\n    (~0.2-0.3 with beta in [1.1,1.7], epsilon in [0.05,0.15]).\n    Tallying cannot produce a positive slope: on count-tied trials\n    (pairs 1-8, 24) it outputs exactly 0.5 for any beta/epsilon, and on\n    all remaining trials its choice probability tracks the raw count\n    margin, which is strongly ANTI-correlated with D across this design\n    (the tally winner is almost always the weighted-margin loser). So\n    Tallying's slope is <= 0 (approximately 0 for small beta, clearly\n    negative for large beta). The sign separation is parameter-free.\n    \"\"\"\n    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])\n    tau = 0.35\n    u = validities + tau * np.square(validities)\n\n    def _to_arr(cell):\n        arr = np.asarray(cell, dtype=float).ravel()\n        if arr.size != u.size:\n            raise ValueError(\n                f\"Expected {u.size} ratings, got {arr.size}\"\n            )\n        return arr\n\n    a = np.vstack([_to_arr(c) for c in data[\"option_a_ratings\"]])\n    b = np.vstack([_to_arr(c) for c in data[\"option_b_ratings\"]])\n\n    # Signed SCWI evidence in favor of A (positive -> A preferred).\n    D = (b - a) @ u\n\n    # response: 0 = chose A, 1 = chose B  ->  indicator of choosing A.\n    choice_a = 1.0 - np.asarray(data[\"response\"], dtype=float).ravel()\n\n    Dc = D - D.mean()\n    Yc = choice_a - choice_a.mean()\n    denom = float(np.sum(Dc * Dc))\n    if denom <= 1e-12:\n        return 0.0\n    return float(np.sum(Dc * Yc) / denom)",
  "rationale": "The metric is the OLS slope of the binary choice (indicator of choosing A) on the fixed, parameter-free signed evidence D = sum_j (v_j + 0.35 v_j^2)(b_j - a_j), computed over all 96 trials. This exploits both structural knives in the design simultaneously. (1) SIGN: SCWI's evidence for A is exactly the anti-validity weighted margin; with beta in [1.1,1.7] and epsilon in [0.05,0.15], the average derivative of its choice probability w.r.t. D is about 0.20-0.31 (softmax gain times average p(1-p) ~ 0.2), so the slope is robustly positive. (2) Tallying is pinned at slope <= 0 for ANY parameter values: on the eight count-tied probe trials (pairs 1-8) its softmax is exactly uniform (P(A)=0.5 regardless of beta and epsilon), contributing zero slope regardless of how D sweeps there (D ranges -0.98 to +0.98 across those trials); on all remaining trials its choice probability is a function of the raw count margin, which is strongly anti-correlated with D in this stimulus set (the tally winner is the weighted-margin loser on essentially every non-tied trial, e.g. pair 20: count margin +5 for A but D = -4.70). Hence Tallying's slope is ~0 for small beta and clearly negative for large beta \u2014 never positive. The per-subject sampling noise of the slope is small: with Var(D) ~ 2.9 across the 96 trials and binary residuals, the binomial standard error is ~0.03, and parameter heterogeneity adds only ~0.03 more (SCWI slope varies within [0.20, 0.31] across its parameter box; the sign of D is invariant over tau in [0.25,0.45], so the fixed tau=0.35 in the metric only mildly attenuates the slope). A mean gap of at least ~0.2 against a per-subject SD of ~0.04 gives a Welch t around 15-25 at N=25, far below alpha=0.01. The metric works identically on the pooled DataFrame and on a single subject's 96-trial slice, so between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 5038,
  "output_tokens": 11168,
  "reasoning_tokens": 10059,
  "provider": "Sail Research"
}
```
