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
  trial 1: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 16: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 17: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 18: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 20: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 21: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 22: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 24: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 25: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 26: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 27: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 28: A=[0, 1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 29: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 30: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 31: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 32: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** All 32 unique pairs are top-vs-bottom conflict trials: feature 0 (validity 0.90) and feature 5 (validity 0.50) always discriminate in opposite directions. Therefore the descending lexicographic route (TTB) stops immediately at feature 0, while the ascending lexicographic route (backward) stops immediately at feature 5. The four middle cues are varied only to set the feature-win tally difference d = A_wins - B_wins. In the 16 top-A/low-B trials d ranges 0..+4; in the 16 top-B/low-A trials d ranges -3..0. The competing model (q*TTB + tally, no backward route) predicts a step-like curve: for top-A/low-B it gives P(A) near 0.62 at d=0 and near 1.0 for d>=1; for top-B/low-A it gives P(A) near 0.38 at d=0 and near 0.0 for d<=-1. The advocated model has a strong backward route (w_backward about 0.48, beta about 1.5) that cancels the top-cue signal: for top-A/low-B P(A) stays in a compressed band near 0.36-0.48 across d, and for top-B/low-A P(A) stays near 0.52-0.64. The key quantitative dissociations are therefore top-A/low-B d=+1 (competing about 1.0 vs advocated about 0.46) and top-B/low-A d=-1 (competing about 0.005 vs advocated about 0.54). This d-response pattern cannot be produced by any q or beta in the competing model because the competing model lacks the backward lexicographic route.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    if len(data) == 0:\n        return 0.0\n\n    a = np.array([list(x) for x in data['option_a_ratings']], dtype=int)\n    b = np.array([list(x) for x in data['option_b_ratings']], dtype=int)\n    diff = a - b\n\n    top_cue = diff[:, 0]\n    tally_diff = (diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)\n\n    chose_a = (data['response'].to_numpy() == 0).astype(float)\n\n    top_a_plus = (top_cue > 0) & (tally_diff >= 1)\n    top_b_minus = (top_cue < 0) & (tally_diff <= -1)\n\n    mean_top_a = np.mean(chose_a[top_a_plus]) if np.any(top_a_plus) else 0.0\n    mean_top_b = np.mean(chose_a[top_b_minus]) if np.any(top_b_minus) else 0.0\n\n    return float(mean_top_a - mean_top_b)",
  "rationale": "This contrast uses the top-vs-bottom conflict trials where the first validity cue and the full tally point in the same direction. In the competing model, the TTB route and tallying both select the top-cued option on these trials, so choice of A is near 1 for top-A trials and near 0 for top-B trials, giving a pooled difference near +1. In the advocated model, the backward lexicographic route partially cancels the top cue: top-A trials stay near 0.47 and top-B trials stay near 0.53, so the same difference is near -0.06. The metric uses 36 top-A and 36 top-B trials per subject, making per-subject estimates reasonably tight."
}
```

## Usage

```json
{
  "input_tokens": 5721,
  "output_tokens": 24542,
  "reasoning_tokens": 24094,
  "provider": "Baidu"
}
```
