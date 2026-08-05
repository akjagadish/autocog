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
**Validities (n_features=4):** [0.95, 0.8, 0.78, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 16: A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by computing a weighted sum of the features for each option, using the cue validities as the weights. This Weighted Additive Model (WADD) is fully compensatory. Instead of a translation-invariant Softmax mapping, choice probabilities are generated using a ratio-based Luce choice rule (power rule) over the options' weighted sums, along with a baseline rate of random guessing. This allows choices to be sensitive to the ratio of evidence rather than just the absolute difference.

**Parameters:**
- theta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Ratio-based Luce choice rule (power rule)
    scores = np.maximum(scores, 1e-9)
    v = scores ** theta
    p_core = v / np.sum(v)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify specific trials by their feature string representations\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # High WADD ratio trials (Tallying difference is exactly 1)\n    # Trial 4: A=[0,0,0,1], B=[0,1,0,1] -> WADD ratio B/A = 1.35 / 0.55 = 2.45\n    t4 = data[(data['a_str'] == '0001') & (data['b_str'] == '0101')]\n    # Trial 8: A=[0,0,1,1], B=[0,0,0,1] -> WADD ratio A/B = 1.33 / 0.55 = 2.41\n    t8 = data[(data['a_str'] == '0011') & (data['b_str'] == '0001')]\n    \n    # Low WADD ratio trials (Tallying difference is exactly 1)\n    # Trial 1: A=[1,0,1,0], B=[1,0,1,1] -> WADD ratio B/A = 2.28 / 1.73 = 1.31\n    t1 = data[(data['a_str'] == '1010') & (data['b_str'] == '1011')]\n    # Trial 12: A=[1,1,0,0], B=[1,1,0,1] -> WADD ratio B/A = 2.30 / 1.75 = 1.31\n    t12 = data[(data['a_str'] == '1100') & (data['b_str'] == '1101')]\n    \n    p_b_t4 = t4['response'].mean() if len(t4) > 0 else 0.5\n    p_a_t8 = 1.0 - t8['response'].mean() if len(t8) > 0 else 0.5\n    \n    p_b_t1 = t1['response'].mean() if len(t1) > 0 else 0.5\n    p_b_t12 = t12['response'].mean() if len(t12) > 0 else 0.5\n    \n    high_ratio_acc = (p_b_t4 + p_a_t8) / 2.0\n    low_ratio_acc = (p_b_t1 + p_b_t12) / 2.0\n    \n    return float(high_ratio_acc - low_ratio_acc)",
  "rationale": "Under the Tallying heuristic, the number of strictly winning features for the favored option is exactly 1 (and 0 for the other) across all four of these trials. Thus, Tallying predicts identical choice probabilities for the better option in all four trials, making the difference metric zero in expectation. Under the Advocated WADD theory (using a ratio-based Luce choice rule), the magnitude of the weighted sum heavily impacts choice probabilities. Trials 4 and 8 have a much larger ratio of evidence (~2.4) compared to Trials 1 and 12 (~1.3), so WADD predicts a significantly higher accuracy on the former pair, making the metric strongly positive."
}
```

## Usage

```json
{
  "prompt_token_count": 3356,
  "candidates_token_count": 865,
  "total_token_count": 8374
}
```
