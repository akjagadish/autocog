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
**Validities (n_features=5):** [0.5, 0.6, 0.7, 0.8, 0.9]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** Since the Thresholded Counting theory has a threshold parameter constrained to [0.0, 0.5] and validities are constrained to [0.5, 1.0], all features will always pass the threshold. Thus, Thresholded Counting is mathematically identical to Tallying in this environment. This design includes validities at the absolute minimum boundary (0.5) to test if any edge-case thresholding occurs, while providing standard feature-summing vs validity-weighting dissociations.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Thresholded Counting: People make decisions by counting the number of positive features for each option, but they only consider features that are deemed sufficiently reliable. Specifically, they ignore features whose subjective or objective validity falls below a certain threshold. For the features that pass this threshold, individuals simply tally the positive cues (equal weighting) and choose the option with the higher count. This bounded-rationality heuristic combines the simplicity of Tallying with a minimal form of validity sensitivity, filtering out noisy or low-information cues before applying a non-compensatory counting strategy.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Counting expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Only count features whose validity meets or exceeds the threshold
    mask = validities >= threshold
    
    # If the threshold is so high that no features qualify, fall back to guessing (all zeros)
    if not np.any(mask):
        scores = np.zeros(stim.shape[0])
    else:
        scores = stim[:, mask].sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: Since the experimental design ensures that all validities are >= 0.5 and the threshold parameter is bounded at 0.5, Thresholded Counting is mathematically identical to Tallying in this environment. This metric simply calculates the concordance with the Tallying predictions to evaluate baseline behavior, though the two theories will produce identical predictions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def tallying_choice(row):
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        if sum_a > sum_b:
            return 0
        elif sum_b > sum_a:
            return 1
        else:
            return 0.5
            
    data['tally_pred'] = data.apply(tallying_choice, axis=1)
    # Count how often the subject's response matches the tallying prediction
    # (excluding ties where tally_pred is 0.5)
    valid_trials = data[data['tally_pred'] != 0.5]
    if len(valid_trials) == 0:
        return 0.0
    
    match = (valid_trials['response'] == valid_trials['tally_pred']).mean()
    return float(match)

outcome: self_sim=0.8353 (var=0.0139) adversary_sim=0.8605 (var=0.0076) welch_t=-0.862 p=0.3936 (N=25, alpha=0.01) -> reject

[1] rationale: Measures the difference in the rate of choosing A between trials where the lowest-validity feature (index 0) opposes A (trials 4, 5) versus trials where it supports A (trials 1, 2). For Tallying, both sets are 3-vs-2 in favor of A, so the difference should be 0. For Thresholded Counting, if the threshold parameter occasionally overflows 0.5 during fitting/simulation, feature 0 is masked. This makes trials 4 and 5 stronger for A (3-vs-1) and trials 1 and 2 weaker (2-vs-2 ties), causing a positive difference. This difference-in-differences approach eliminates baseline determinism noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def get_group(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if a[0] == 0 and sum(a) > sum(b):
            return 'A_0'
        elif a[0] == 1 and sum(a) > sum(b):
            return 'A_1'
        return 'Other'
        
    data['group'] = data.apply(get_group, axis=1)
    
    chose_a_0 = data[data['group'] == 'A_0']['response'] == 0
    chose_a_1 = data[data['group'] == 'A_1']['response'] == 0
    
    p_0 = chose_a_0.mean() if len(chose_a_0) > 0 else 0.5
    p_1 = chose_a_1.mean() if len(chose_a_1) > 0 else 0.5
    
    return float(p_0 - p_1)
outcome: self_sim=0.0116 (var=0.0052) adversary_sim=-0.0063 (var=0.0060) welch_t=+0.846 p=0.4017 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def is_target_trial(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        # Target trials where the lowest-validity feature opposes the Tallying winner\n        return a[0] == 0 and sum(a) > sum(b)\n        \n    mask = data.apply(is_target_trial, axis=1)\n    if not mask.any():\n        return 0.5\n    return float((data[mask]['response'] == 0).mean())",
  "rationale": "Focuses strictly on the absolute accuracy for trials where the lowest-validity cue (index 0) opposes the Tallying majority (Trials 4 and 5). By isolating only the trials most susceptible to thresholding (where dropping feature 0 increases the net evidence for option A from +1 to +2), this metric maximizes the potential signal of the threshold parameter while avoiding the added binomial variance of the other trials."
}
```

## Usage

```json
{
  "prompt_token_count": 3810,
  "candidates_token_count": 246,
  "total_token_count": 21552
}
```
