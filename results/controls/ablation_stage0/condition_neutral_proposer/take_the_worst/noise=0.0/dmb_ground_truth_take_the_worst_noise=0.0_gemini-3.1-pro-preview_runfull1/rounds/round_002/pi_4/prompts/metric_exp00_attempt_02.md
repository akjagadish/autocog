# metric_exp00_attempt_02

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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** Equal-Weight Additive (Theory 1) and Tallying (Theory 2) are mathematically equivalent when applied to binary features. Both models compute unweighted sums of feature values (or wins) and pass the difference through a softmax. Since the difference in total positive features between two options is exactly equal to the difference in the number of winning features, the models yield identical choice probabilities for any binary stimulus pair. Therefore, this design uses a diverse set of feature vectors to confirm this equivalence empirically across varying degrees of feature overlap and score differences.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Equal-Weight Additive Rule: People integrate all available cues but ignore their differential validities, assigning equal weight to each cue. They compute an overall score for each option by summing its positive features. This is a fully compensatory process that discards validity magnitudes but retains the quantity of evidence. The option with the higher total score is favored, with response noise introduced via a softmax function and a uniform lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal-Weight expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Sum all features (equal weighting)
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
[0] rationale: Equal-Weight Additive and Tallying are mathematically identical when applied to binary features. Both sum up the positive features (or wins) and compute the difference, which is algebraically equivalent for binary feature vectors. Because their choice probabilities are exactly the same, no metric can reliably distinguish them on this experimental design. This metric computes the probability of choosing the option with more positive features, which will be identical under both models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the difference in the number of positive features between A and B
    data['score_a'] = data['option_a_ratings'].apply(sum)
    data['score_b'] = data['option_b_ratings'].apply(sum)
    
    # Focus on trials where A has strictly more positive features than B
    mask = data['score_a'] > data['score_b']
    if not mask.any():
        return 0.5
    
    # Return the proportion of times the subject chose option A (response == 0)
    # in these cases. Since both models are mathematically equivalent for binary
    # features, this metric serves as a baseline consistency check.
    return (data.loc[mask, 'response'] == 0).mean()

outcome: self_sim=0.8697 (var=0.0073) adversary_sim=0.8547 (var=0.0119) welch_t=+0.541 p=0.5909 (N=25, alpha=0.01) -> reject

[1] rationale: Since Equal-Weight Additive and Tallying are mathematically equivalent for binary features, their choice probabilities are perfectly identical across all possible binary stimuli. The prompt correctly notes that they both compute the same score differences. Because the data-generating distributions are indistinguishable, no valid statistical metric can reliably separate them. This metric measures the choice consistency on the most extreme trials (5 vs 0 positive features), which will yield the same high expected value under both models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Focus on trials where the options have a large difference in positive features
    data['score_a'] = data['option_a_ratings'].apply(sum)
    data['score_b'] = data['option_b_ratings'].apply(sum)
    data['score_diff'] = (data['score_a'] - data['score_b']).abs()
    
    # Mask for the most extreme difference (e.g., 5 vs 0)
    mask = data['score_diff'] == 5
    if not mask.any():
        return 0.5
    
    # Calculate the rate of choosing the option with more features
    # Since the models are mathematically equivalent for binary features,
    # they will yield identical predictions. We provide a metric that captures
    # the deterministic nature of choices on extreme trials.
    correct_choice = ((data['score_a'] > data['score_b']) & (data['response'] == 0)) | \
                     ((data['score_a'] < data['score_b']) & (data['response'] == 1))
    return correct_choice[mask].mean()
outcome: self_sim=0.8944 (var=0.0068) adversary_sim=0.8788 (var=0.0095) welch_t=+0.612 p=0.5437 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Since Equal-Weight and Tallying are mathematically equivalent for binary features,\n    # their simulated datasets are drawn from the exact same distribution.\n    # We compute the proportion of trials where the subject chose option A when A had more positive features.\n    data['score_a'] = data['option_a_ratings'].apply(sum)\n    data['score_b'] = data['option_b_ratings'].apply(sum)\n    mask = data['score_a'] > data['score_b']\n    if not mask.any():\n        return 0.5\n    return (data.loc[mask, 'response'] == 0).mean()",
  "rationale": "Equal-Weight Additive and Tallying are mathematically identical when applied to binary features. Both models compute unweighted sums of feature values (or wins) and pass the difference through a softmax. Since the difference in total positive features between two options is exactly equal to the difference in the number of winning features, the models yield identical choice probabilities for any binary stimulus pair. Therefore, no metric can reliably distinguish them on this experimental design."
}
```

## Usage

```json
{
  "prompt_token_count": 3919,
  "candidates_token_count": 266,
  "total_token_count": 9321
}
```
