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
**Validities (n_features=5):** [0.95, 0.9, 0.65, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** This design extends the core TTB vs. Tallying dissociation by introducing a NEW dimension: feature-validity ordering effects under TIED high-validity cues. The key insight is that when the highest-validity cue ties between options, TTB must cascade to the next-highest cue, whereas Tallying ignores the hierarchy entirely and counts all features equally. We propose 5 features with validities [0.95, 0.90, 0.65, 0.60, 0.55] to create a sharp validity cliff after the first two cues. Trial design: (1) TIED high-validity cues (features 0 and 1 both tie), forcing TTB to consult feature 2 (validity 0.65). Feature 2 discriminates for option A. Meanwhile, features 3 and 4 (lower validity) discriminate for B. TTB picks A (follows feature 2 after tie-breaking). Tallying: A wins on feature 2 only; B wins on features 3 and 4. Tallying picks B. (2) Mirror: features 0 and 1 tie; feature 2 discriminates for B; features 3 and 4 discriminate for A. TTB picks B; Tallying picks A. (3) Control trials where all theories agree (e.g., feature 0 discriminates for A; no other discriminations, or all lower cues also favor A). This design isolates the validity-ordering cascade effect in TTB versus the egalitarian aggregation in Tallying, creating a quantitative dissociation that is orthogonal to the existing experiment [0]. The tie at the top forces TTB into a specific cascade behavior that Tallying's equal-weight rule directly contradicts.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Measure the degree to which subjects employ a validity-ordered cascade\n    (TTB) versus equal-weight tallying (Tallying).\n    \n    Key insight: On trials where the two highest-validity features (0, 1) tie\n    between options, TTB must consult feature 2 (validity 0.65) to break the tie,\n    while Tallying counts all features equally. This creates a sharp dissociation:\n    - TTB predicts choice based on feature 2 after tie-breaking at features 0-1.\n    - Tallying predicts choice based on the majority vote across all features.\n    \n    We compute a \"cascade fidelity\" score: the proportion of trials where the\n    subject's choice aligns with TTB's prediction (given the validity cascade)\n    minus the proportion where it aligns with Tallying's prediction.\n    \n    On the critical tie-breaking trials (1-4, 7-8), TTB and Tallying diverge.\n    On control trials (9-12), both theories largely agree. On low-validity trials\n    (5-6), both theories agree. The metric isolates the cascade effect.\n    \n    Validities: [0.95, 0.90, 0.65, 0.60, 0.55]\n    \"\"\"\n    validities = np.array([0.95, 0.90, 0.65, 0.60, 0.55])\n    \n    # Define the 12 trial pairs (from the experimental design)\n    trial_pairs = [\n        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 1\n        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 2\n        ([1, 1, 1, 0, 0], [1, 1, 0, 1, 1]),  # trial 3\n        ([1, 1, 0, 1, 1], [1, 1, 1, 0, 0]),  # trial 4\n        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 5\n        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 6\n        ([0, 0, 1, 0, 0], [0, 0, 0, 1, 1]),  # trial 7\n        ([0, 0, 0, 1, 1], [0, 0, 1, 0, 0]),  # trial 8\n        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 9\n        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 10\n        ([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),  # trial 11\n        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),  # trial 12\n    ]\n    \n    # Helper: compute TTB prediction (0 = choose A, 1 = choose B, None = tie)\n    def ttb_predict(a, b, vals):\n        cue_order = np.argsort(-vals, kind='stable')\n        for j in cue_order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return None  # All tied\n    \n    # Helper: compute Tallying prediction (0 = choose A, 1 = choose B, None = tie)\n    def tallying_predict(a, b):\n        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])\n        b_wins = sum(1 for i in range(len(b)) if b[i] > a[i])\n        if a_wins > b_wins:\n            return 0\n        elif b_wins > a_wins:\n            return 1\n        else:\n            return None  # Tied\n    \n    # Count agreement with each theory, excluding ties\n    ttb_matches = 0\n    tallying_matches = 0\n    valid_trials = 0\n    \n    for idx, row in data.iterrows():\n        a = np.array(row['option_a_ratings'], dtype=int)\n        b = np.array(row['option_b_ratings'], dtype=int)\n        response = int(row['response'])\n        \n        ttb_pred = ttb_predict(a, b, validities)\n        tallying_pred = tallying_predict(a, b)\n        \n        # Only count trials where both theories make a definite prediction\n        if ttb_pred is not None and tallying_pred is not None:\n            valid_trials += 1\n            if response == ttb_pred:\n                ttb_matches += 1\n            if response == tallying_pred:\n                tallying_matches += 1\n    \n    if valid_trials == 0:\n        return 0.0\n    \n    # Cascade fidelity: TTB agreement rate minus Tallying agreement rate\n    # Positive values favor TTB; negative values favor Tallying.\n    ttb_rate = ttb_matches / valid_trials\n    tallying_rate = tallying_matches / valid_trials\n    \n    return ttb_rate - tallying_rate",
  "rationale": "This metric directly operationalizes the core theoretical dissociation: TTB's validity-ordered cascade versus Tallying's equal-weight aggregation. On tie-breaking trials (1-4, 7-8), the two theories make divergent predictions. TTB cascades to feature 2 after features 0-1 tie, while Tallying counts all features equally. By measuring the subject's agreement rate with each theory's predictions and computing the difference, we obtain a signed metric that is large and stable when subjects follow one theory consistently. Subjects who follow TTB will show positive values; subjects who follow Tallying will show negative values. The metric is robust to individual differences in beta and epsilon because it measures choice consistency with the underlying decision process, not the parameters themselves. Critical trials (1-4, 7-8) dominate the signal, while control and agreement trials provide stability. This design should yield a large between-theory gap with tight within-subject variance, satisfying both the mean and variance acceptance criteria."
}
```

## Usage

```json
{
  "input_tokens": 4756,
  "output_tokens": 1717
}
```
