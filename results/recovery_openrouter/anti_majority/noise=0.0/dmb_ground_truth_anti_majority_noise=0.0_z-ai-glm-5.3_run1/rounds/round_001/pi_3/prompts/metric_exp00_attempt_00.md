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
**Validities (n_features=9):** [0.74, 0.73, 0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]

**Trial pairs (n=23):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 13: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 14: A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 15: A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-2): a SHALLOW validity gradient with a NEAR-TIE at the top. All prior dissociation designs used a dominant top cue; here the maximum validity is 0.74 and the top two experts differ by only 0.01 (0.74 vs 0.73). This is the sharpest possible probe of TTB's ordinal cascade against VWEI's cardinal weights: TTB commits HARD to the 74% expert whenever it discriminates (a 1-point validity edge produces a full-strength decision), while VWEI's amplified weights for the top two differ by only w1-w2 = 0.16 (at gamma=3), so a single 70% expert (w=0.61) outweighs the entire top-two gap. Validities: [0.74, 0.73, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64] (9 experts; weights at gamma=3: 1.145, 0.984, 0.609, 0.512, 0.428, 0.356, 0.292, 0.237, 0.190; no cue can dominate: even all 8 lower weights sum to only 3.61 vs w1=1.145, so coalitions of 2-4 lower cues can override any single cue). 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K=96//24=4, 96 trials, 8 samples per base cell per subject). In every base pair TTB's winner is A, so TTB predicts ONE constant P(A) = (1-eps)*sigmoid(beta)+eps/2 on all 12 base pairs (and its complement on mirrors) -- stimulus-invariant by construction. VWEI predicts a specific 25-point graded, anti-symmetric profile (verified robust across the full parameter box gamma in [2.8,3.2], beta in [0.18,0.23]; all evidence signs stable). FOUR QUANTITATIVE SIGNATURES: (1) NEAR-TIE ATTENUATION GRADIENT (pairs 1-8): cue 1 favors A, cue 2 favors B, and the net lower-cue support sweeps from +7 to -7. TTB: flat (cue 1 always decides for A). VWEI: monotone P(A) = 0.636, 0.585, 0.538, 0.508, 0.478, 0.452, 0.429, 0.388 (range across the parameter box: 0.617-0.670 down to 0.380-0.396), crossing 50% between zero and one opposing cue. Any nonzero slope of P(A) on coalition weight falsifies TTB; a flat profile falsifies VWEI. (2) CONTINUITY AT THE VALIDITY TIE (pair 4): the 0.74-vs-0.73 conflict alone. TTB commits fully to A at sigmoid(beta); VWEI predicts 0.508 (range 0.507-0.510) -- statistically a guess -- because a 1-point validity edge is worth only 0.16 evidence units. (3) CASCADE-INVARIANCE VIOLATION (pair 1 vs pair 11): pair 1 is decided by cue 1 with all lower cues congruent; pair 11 is decided by cue 2 (cue 1 ties) with all lower cues opposed. TTB predicts IDENTICAL P(A) on both (first discriminating cue favors A in both; its softmax never sees anything else). VWEI predicts 0.636 vs 0.420 -- a 22-point gap in opposite directions of deviation that no constant can fit. Similarly pair 8 vs 11 (0.388 vs 0.420) shows TTB-equivalent pairs differing by deciding-cue identity. (4) SIGN CONFLICTS: on 7 of 12 base pairs (5-8, 10-12) VWEI predicts a B-majority (P(A) = 0.388-0.478) where TTB predicts an A-majority; majority-B choices there directly falsify TTB at any beta (TTB's winner probability is always >= 0.5). Family C (pair 12) additionally tests deep-cascade skipping: cues 1-3 all tie, cue 4 decides for A against a 5-cue coalition; TTB says A at the same constant; VWEI says 0.451. Because TTB's prediction on this entire design is a single stimulus-invariant constant (plus its mirror), VWEI's 25-point stimulus-driven spread -- spanning both sides of 50% with a specific monotone shape -- is a profile no TTB parameterization can approximate, while every VWEI prediction is sign-robust across its entire parameter box. All pairs are non-degenerate (every pair has discriminating cues; no all-tie pairs; no pair where both theories guess), the design is perfectly side-balanced (12 TTB-winner-A, 12 TTB-winner-B), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 23 unique pairs × 4 reps = 92 trials per subject.



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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Signed-evidence slope.\n\n    For every trial we compute the VWEI evidence\n        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3\n    and the TTB winner (first discriminating cue in validity order;\n    validities are strictly descending so cue order = feature order).\n\n    Define the signed regressor  x = +E if TTB's winner is A, x = -E if\n    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's\n    winner.  Return the OLS slope of y on x.\n\n    TTB's probability of choosing its own winner is a single constant\n    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected\n    slope is exactly 0.  VWEI's probability of choosing TTB's winner is\n    sigmoid(beta * x) on every trial, giving a clearly positive slope\n    (~0.05 at the design center, sign-stable over the whole parameter\n    box).  The two theories are maximally separated on this statistic.\n    \"\"\"\n    if data is None or len(data) == 0:\n        return 0.0\n\n    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,\n                           0.67, 0.66, 0.65, 0.64], dtype=float)\n    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)\n    w = np.log(v / (1.0 - v)) ** 3.0\n\n    def _to_vec(cell):\n        if isinstance(cell, str):\n            return [float(ch) for ch in cell if ch in '01']\n        return [float(val) for val in cell]\n\n    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)\n    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)\n    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):\n        return 0.0\n\n    resp = data['response'].to_numpy(dtype=float)\n\n    d = np.sign(a - b)                     # +1 favors A, -1 favors B\n    E = d @ w                              # VWEI evidence for A\n\n    disc = d != 0\n    keep = disc.any(axis=1)                # drop degenerate all-tie rows\n    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]\n    if d.shape[0] == 0:\n        return 0.0\n\n    # First discriminating cue (validities strictly descending -> order 0..8)\n    first_idx = disc.argmax(axis=1)\n    rows = np.arange(d.shape[0])\n    d_first = d[rows, first_idx]\n    winner_a = d_first > 0                  # TTB's winner is option A\n\n    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner\n    y = np.where(winner_a, 1.0 - resp, resp)\n    x = np.where(winner_a, E, -E)\n\n    xc = x - x.mean()\n    denom = float(np.dot(xc, xc))\n    if denom <= 1e-12:\n        return 0.0\n    yc = y - y.mean()\n    slope = float(np.dot(xc, yc)) / denom\n    return slope",
  "rationale": "Core logic. On this design TTB's prediction collapses to a single stimulus-invariant constant: its probability of choosing its own winner is q = (1-eps)*sigmoid(beta) + eps/2 on every one of the 23 pairs (softmax over a binary score never sees anything but the winner identity). VWEI instead predicts P(choose TTB's winner) = sigmoid(beta * x), where x is the VWEI evidence signed toward TTB's winner \u2014 I verified this identity trial-by-trial (e.g., trial 1: x=+2.78, P=0.636; trial 8: x=-2.46, P=0.380; trial 15: x=-2.46, P=0.380; trial 19: x=-2.46, P=0.620... wait, trial 19's winner is B with x=-2.46 so P(choose winner)=0.620). So the OLS slope of 'chose TTB's winner' on signed evidence x is exactly 0 under TTB (for ANY beta, epsilon) and clearly positive (~0.049 at gamma=3, beta=0.2, eps=0; still >=0.043 across the full VWEI box gamma in [2.8,3.2], beta in [0.18,0.23], eps in [0,0.03]) under VWEI. The metric needs no trial lookup \u2014 x and the TTB winner are recomputed from the raw rating vectors with the design's fixed validities, so it is robust to trial ordering, mirroring, and rep counts. Power analysis: with 92 trials/subject the per-trial x spread gives Sxx ~= 202, so the per-subject slope SD is about sqrt(0.25/202) ~= 0.035 under the noisiest TTB parameterization (q=0.5) and similar under VWEI, while the VWEI mean slope is ~0.049. At N=25 that yields Welch t ~= 5 (p ~ 1e-5), comfortably below alpha=0.01, with margin even at the worst-corner VWEI parameters (t ~= 4.3). The metric works identically on the pooled DataFrame (common x distribution per subject) and on single-subject slices, so between_subject_variance is well-defined; degenerate cases (all-tie rows, zero x variance, constant y) return finite values rather than NaN."
}
```

## Usage

```json
{
  "input_tokens": 6060,
  "output_tokens": 20032,
  "reasoning_tokens": 18637,
  "provider": "Sail Research"
}
```
