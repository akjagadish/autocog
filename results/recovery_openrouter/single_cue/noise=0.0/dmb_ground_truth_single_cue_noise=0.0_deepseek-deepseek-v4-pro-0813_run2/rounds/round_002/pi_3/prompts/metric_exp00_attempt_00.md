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
**Validities (n_features=6):** [0.9, 0.82, 0.74, 0.66, 0.58, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 8: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 10: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 14: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 15: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 17: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 18: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 19: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 20: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 21: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 23: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 24: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 25: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 26: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 27: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 28: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 29: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 30: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 31: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 32: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Rationale:** This design isolates a direction-of-lexicographic-scanning dissociation. All 32 unique pairs hold the feature-win tally difference at d = 0, so both theories' tallying routes are neutral and cannot drive the choice. On every trial the highest-validity cue (feature 0, validity 0.90) discriminates, and the lowest-validity cue (feature 5, validity 0.50) discriminates in the opposite direction. The advocated TTB/Tally mixture has only a descending lexicographic route, so its TTB component stops at feature 0; with q near 0.25 it predicts P(A) ≈ 0.62 when feature 0 favors A and ≈ 0.38 when feature 0 favors B. The competing model instead has a strong backward (ascending) lexicographic route (weight ≈ 0.42-0.54, beta ≈ 1.3-1.7) plus a weak forward route and neutral tallying, so on the same conflict trials it predicts the mirror pattern: P(A) ≈ 0.32-0.40 when feature 0 favors A/feature 5 favors B, and ≈ 0.60-0.68 when feature 0 favors B/feature 5 favors A. The middle four features are varied with equal A/B wins or ties to keep d = 0 while generating 32 unique pairs, 16 in each top-cue direction, repeated K = 3 for 96 trials.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** On each choice, a decision maker probabilistically selects between two comparison strategies. With probability q, centered near 0.25, the choice is made by a one-reason stopping rule: cues are inspected in descending validity order and the first discriminating cue determines the preferred option, with no further cues consulted. With probability 1 - q, the choice is made by full feature-win tallying: each option receives one point for every feature on which it strictly beats the other option, ties contribute nothing, and the option with the larger tally is preferred. Each strategy's preference is passed through its own softmax choice rule, giving nearly deterministic but not perfectly errorless responding, and there is no additional uniform lapse. The mixture probability q is a stable subject-level parameter applied independently on every trial, so the model recovers pure Take The Best when q = 1, pure Tallying when q = 0, and an intermediate strategy mixture when q is near 0.25.

**Parameters:**
- q: [0.22, 0.28]
- beta_tally: [5.44, 5.50]
- beta_ttb: [5.22, 5.28]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    def _softmax(scores, beta):
        z = beta * (scores - np.max(scores))
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    validities = np.asarray(parameters['validities'], dtype=float)
    if validities.shape[0] == n_features:
        cue_order = np.argsort(-validities, kind='stable').tolist()
    else:
        cue_order = list(range(n_features))

    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        p_ttb = np.ones(2) / 2.0
    else:
        ttb_scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        p_ttb = _softmax(ttb_scores, float(parameters['beta_ttb']))

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    p_tally = _softmax(
        np.array([a_wins, b_wins]),
        float(parameters['beta_tally'])
    )

    q = float(parameters['q'])
    p = q * p_ttb + (1.0 - q) * p_tally

    epsilon = float(parameters['epsilon'])
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2) / 2.0

    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

**Parameters:**
- w_forward: [0.22, 0.32]
- w_backward: [0.42, 0.54]
- beta_forward: [0.08, 0.20]
- beta_backward: [1.30, 1.70]
- beta_tally: [1.80, 2.60]
- epsilon: {0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data):\n    import numpy as np\n    a0 = np.array([int(r[0]) for r in data['option_a_ratings']])\n    b0 = np.array([int(r[0]) for r in data['option_b_ratings']])\n    choose_a = (np.asarray(data['response'], dtype=int) == 0).astype(float)\n    top_a = a0 > b0\n    top_b = ~top_a\n    if not np.any(top_a) or not np.any(top_b):\n        return 0.0\n    return float(choose_a[top_a].mean() - choose_a[top_b].mean())",
  "rationale": "The metric measures the signed effect of the highest-validity cue: mean P(choose A) on trials where feature 0 favors A minus mean P(choose A) on trials where feature 0 favors B. All pairs in this design are feature-win tally neutral and feature 0 always discriminates, so tallying contributes 0.5 in both directions. The advocated forward TTB mixture therefore predicts a positive value near 0.25, while the competing backward-heavy mixture predicts a negative value near -0.35. Each subject has 48 trials in each top-cue direction, so this contrast has a large between-theory gap and only modest per-subject binomial variability."
}
```

## Usage

```json
{
  "input_tokens": 5653,
  "output_tokens": 8861,
  "reasoning_tokens": 8553,
  "provider": "Baidu"
}
```
