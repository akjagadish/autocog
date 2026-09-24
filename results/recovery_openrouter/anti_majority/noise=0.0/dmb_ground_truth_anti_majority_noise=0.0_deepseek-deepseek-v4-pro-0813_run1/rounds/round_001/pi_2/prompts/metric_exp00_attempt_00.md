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
**Validities (n_features=8):** [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 17: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 18: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Rationale:** Eight experts with non-monotonic validities [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]. The first three display positions form a first-discriminator validity staircase (0.53, 0.91, 0.82). Leading ties place the first discriminator at position 0, 1, or 2. The design is a 3 (first-discriminator validity) x 3 (tally margin d = +4/+2/0 for B-first; -4/-2/0 for A-first) x 2 (first-discriminator direction) factorial. Under Tallying, P(A) depends only on d: high for d=+4, above chance for d=+2, exactly chance for d=0, below chance for d=-2, and low for d=-4; it is identical across the three validity levels for a given d. Under display-order TTB, P(A) depends only on the first displayed discriminator's direction and validity: low for B-first (lowest for v=0.91, highest for v=0.53) and high for A-first, and it is flat across d. Thus all 18 pairs are dissociative: d!=0 trials give opposite directional predictions, and d=0 trials give chance (Tallying) versus directional (TTB). The factorial also provides quantitative tests: Tallying predicts a d effect with no validity effect, while TTB predicts a validity effect with no d effect. The 18 unique pairs are repeated 5 times for 90 trials, close to the 96-trial target.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** People execute a display-order Take The Best rule: they scan the physical feature vector left-to-right and stop at the first feature that discriminates between the two options. That first displayed discriminator determines the choice, and later features are ignored. If no feature discriminates, the person guesses. Stated validities do not reorder the scan, but they weakly modulate salience: a first discriminator with lower stated validity is used less decisively than one with higher stated validity. Response noise enters through softmax with inverse temperature beta plus a small lapse probability epsilon.

**Parameters:**
- beta: [0.75, 0.85]
- lambda: [6.0, 8.0]
- epsilon: [0.0, 0.03]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Display-order TTB expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    # Optional validities; they affect decisiveness, not cue order.
    if 'validities' in parameters and parameters['validities'] is not None:
        val = np.asarray(parameters['validities'], dtype=float)
        if val.shape[0] != n_features:
            raise ValueError(
                f'validities length {val.shape[0]} != n_features {n_features}.'
            )
    else:
        val = np.ones(n_features, dtype=float)

    # Scan features in physical display order: 0, 1, 2, ..., n-1.
    winner = None
    discriminator = None
    for j in range(n_features):
        if a[j] > b[j]:
            winner = 0
            discriminator = j
            break
        if b[j] > a[j]:
            winner = 1
            discriminator = j
            break

    # Complete tie: no discriminator, so guess.
    if winner is None:
        return np.ones(2, dtype=float) / 2.0

    beta = float(parameters['beta'])
    lam = float(parameters['lambda'])
    epsilon = float(parameters['epsilon'])

    # The first displayed discriminator is the only cue used for the choice.
    # Its stated validity weakly scales the effective inverse temperature.
    v_j = float(val[discriminator])
    salience = 1.0 + lam * max(0.0, v_j - 0.5)
    beta_eff = beta * salience

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax.
    z = beta_eff * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts, dtype=float) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_vals = np.asarray(data['option_a_ratings'].tolist(), dtype=float)\n    b_vals = np.asarray(data['option_b_ratings'].tolist(), dtype=float)\n\n    a_wins = np.sum(a_vals > b_vals, axis=1)\n    b_wins = np.sum(b_vals > a_vals, axis=1)\n    tally_tie = a_wins == b_wins\n\n    differ = a_vals != b_vals\n    has_disc = differ.any(axis=1)\n    first_disc = np.argmax(differ.astype(int), axis=1)\n\n    row_idx = np.arange(a_vals.shape[0])\n    first_a_wins = a_vals[row_idx, first_disc] > b_vals[row_idx, first_disc]\n\n    response = data['response'].to_numpy(dtype=float)\n\n    selected = tally_tie & has_disc\n    if not np.any(selected):\n        return 0.0\n\n    disc_positions = np.unique(first_disc[selected])\n    per_pos_diff = []\n    for pos in disc_positions:\n        in_pos = selected & (first_disc == pos)\n        b_first = in_pos & ~first_a_wins\n        a_first = in_pos & first_a_wins\n        if np.sum(b_first) == 0 or np.sum(a_first) == 0:\n            continue\n        per_pos_diff.append(\n            float(np.mean(response[b_first]) - np.mean(response[a_first]))\n        )\n\n    if not per_pos_diff:\n        return 0.0\n    return float(np.mean(per_pos_diff))",
  "rationale": "This metric isolates display-order directional responding on exactly the trials where Tallying has no preference. It selects tally-tie trials, where the feature-wise win counts for A and B are equal. For each first displayed discriminator position, it computes the average B choice rate when the B option is the first discriminator minus the average B choice rate when the A option is the first discriminator, then averages across those positions. Under Tallying, tally ties are uninformative and choice probability is 0.5 in both directions for every position, so this signed difference is close to 0. Under display-order Take The Best, the first discriminator controls choice: B-first trials produce mostly B responses and A-first trials produce mostly A responses. With the design validities 0.53, 0.91, 0.82 and the competing theory's beta/lambda ranges, the per-position differences are approximately 0.4 to 0.9, yielding a pooled value near 0.7. The within-subject signed contrast uses all 30 tally-tie trials per subject, keeping binomial estimation noise small, so the between-theory gap is large relative to between-subject variance."
}
```

## Usage

```json
{
  "input_tokens": 4880,
  "output_tokens": 14855,
  "reasoning_tokens": 14193,
  "provider": "Baidu"
}
```
