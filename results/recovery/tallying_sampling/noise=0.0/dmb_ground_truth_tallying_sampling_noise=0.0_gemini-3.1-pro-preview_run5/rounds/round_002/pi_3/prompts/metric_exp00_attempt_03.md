# metric_exp00_attempt_03

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** This design dissociates Tallying (Equal Weight) from Soft Threshold Tallying by systematically pitting a larger number of low-validity features against a smaller number of high-validity features. Pure Tallying ignores validities entirely and will always choose the option with the highest absolute count of positive features, or remain completely indifferent if the counts are equal. In contrast, Soft Threshold Tallying down-weights or ignores features with validities below a certain subjective threshold. Thus, in trials where Option A has many low-validity cues and Option B has fewer but high-validity cues, Tallying predicts a strong preference for A, while Soft Threshold Tallying predicts a preference for B (or at least a much weaker preference for A, depending on the threshold).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying (Equal Weight) theory posits that decision-makers simply count the number of positive features (or cues) for each option and choose the option with the higher tally, ignoring cue validities completely. This is a compensatory heuristic that treats all pieces of evidence equally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Stimulus is the pair of option feature vectors for the current trial
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Soft Threshold Tallying posits that decision-makers use a compensatory tallying heuristic, but down-weight cues that are less reliable. They evaluate each feature's subjective validity against a personal threshold using a smooth sigmoid function, creating a soft inclusion mask. Features that fall well below the threshold are effectively ignored, while those above are tallied with near-equal weight. This smooth transition provides a bridge between pure equal-weight tallying and selective attention to high-validity information, while allowing for continuous optimization.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- threshold: [-5.0, 1.0]
- steepness: [1.0, 50.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Threshold Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Soft threshold using a logistic sigmoid to create a smooth inclusion mask
    weights = 1.0 / (1.0 + np.exp(-steepness * (validities - threshold)))
    
    # Tally the features weighted by the soft threshold mask
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add uniform lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric calculates the proportion of trials where the subject's choice aligns with the pure Tallying heuristic (i.e., choosing the option with the strictly greater number of positive features). Tallying theory predicts this proportion will be high, as it assumes all features are weighted equally. Soft Threshold Tallying, however, predicts a significantly lower proportion because it down-weights or ignores low-validity features, leading to choices that often contradict the unweighted tally when a few high-validity features are pitted against many low-validity ones.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    responses = data['response'].values
    
    mask = sum_a != sum_b
    if not np.any(mask):
        return 0.5
        
    sum_a_diff = sum_a[mask]
    sum_b_diff = sum_b[mask]
    resp_diff = responses[mask]
    
    matches = ((sum_a_diff > sum_b_diff) & (resp_diff == 0)) | ((sum_b_diff > sum_a_diff) & (resp_diff == 1))
    
    return float(np.mean(matches))
outcome: self_sim=0.8714 (var=0.0094) adversary_sim=0.8159 (var=0.0229) welch_t=+1.545 p=0.13 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of looking at overall tallying alignment, this metric isolates 'conflict' trials where the two theories make diametrically opposed predictions. In these specific trials, one option has a higher overall feature count, while the other option has a higher count of the top 3 most reliable features. Pure Tallying ignores validities and will consistently choose the option with the higher total count (yielding a metric near 0, driven only by noise). Soft Threshold Tallying down-weights the less reliable features and will frequently choose the option with more high-validity features (yielding a significantly higher metric value).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # Calculate total feature counts
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Calculate counts of the top 3 most valid features
    high_a = a_ratings[:, :3].sum(axis=1)
    high_b = b_ratings[:, :3].sum(axis=1)
    
    # Identify 'conflict' trials where the option with MORE total features 
    # actually has FEWER high-validity features.
    conflict_mask = ((sum_a > sum_b) & (high_a < high_b)) | ((sum_a < sum_b) & (high_a > high_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    # For these conflict trials, did the subject choose the option with more high-validity features?
    high_diff = high_a[conflict_mask] - high_b[conflict_mask]
    resp_conflict = responses[conflict_mask]
    
    chose_high_val = ((high_diff > 0) & (resp_conflict == 0)) | ((high_diff < 0) & (resp_conflict == 1))
    
    return float(np.mean(chose_high_val))
outcome: self_sim=0.1285 (var=0.0100) adversary_sim=0.1867 (var=0.0237) welch_t=-1.585 p=0.1207 (N=25, alpha=0.01) -> reject

[2] rationale: This metric explicitly targets trials where pure Tallying sees a perfect tie (equal total number of positive features for both options). On these specific trials, Tallying predicts subjects will guess at random, leading to an expected metric value of exactly 0.5 (with only binomial noise). Soft Threshold Tallying, however, breaks the tie by down-weighting the low-validity features; therefore, it strongly predicts subjects will prefer the option with more high-validity features, leading to a metric value significantly greater than 0.5. By restricting the analysis to Tallying's 'blind spots', we minimize the overlap between the two theories' predictions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # Calculate total feature counts
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Calculate counts of the top 3 most valid features
    high_a = a_ratings[:, :3].sum(axis=1)
    high_b = b_ratings[:, :3].sum(axis=1)
    
    # Isolate trials where pure Tallying sees a tie
    tie_mask = (sum_a == sum_b)
    if not np.any(tie_mask):
        return 0.5
        
    resp_tie = responses[tie_mask]
    high_a_tie = high_a[tie_mask]
    high_b_tie = high_b[tie_mask]
    
    # Further isolate ties where the theories differ on high-validity features
    valid_mask = high_a_tie != high_b_tie
    if not np.any(valid_mask):
        return 0.5
        
    high_a_valid = high_a_tie[valid_mask]
    high_b_valid = high_b_tie[valid_mask]
    resp_valid = resp_tie[valid_mask]
    
    # Did the subject choose the option with more high-validity features?
    chose_high = ((high_a_valid > high_b_valid) & (resp_valid == 0)) | ((high_b_valid > high_a_valid) & (resp_valid == 1))
    
    return float(np.mean(chose_high))
outcome: self_sim=0.5244 (var=0.0143) adversary_sim=0.5233 (var=0.0137) welch_t=+0.033 p=0.9737 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    sum_a = a_ratings.sum(axis=1)\n    sum_b = b_ratings.sum(axis=1)\n    \n    responses = data['response'].values\n    \n    # Isolate trials with the maximum tally difference (diff == 2)\n    # In this design, these are exactly Trial 3 and Trial 7.\n    mask = np.abs(sum_a - sum_b) == 2\n    if not np.any(mask):\n        return 0.5\n        \n    sum_a_diff = sum_a[mask]\n    sum_b_diff = sum_b[mask]\n    resp_diff = responses[mask]\n    \n    # Calculate the proportion of times the subject chose the option with the higher tally\n    matches = ((sum_a_diff > sum_b_diff) & (resp_diff == 0)) | ((sum_b_diff > sum_a_diff) & (resp_diff == 1))\n    \n    return float(np.mean(matches))",
  "rationale": "This metric isolates the trials where the two theories have the strongest possible disagreement. In this specific design, Trials 3 and 7 are the only ones with a tally difference of 2 (the maximum possible). Pure Tallying predicts a very strong, highly consistent preference for the option with more features (due to the large score difference, minimizing softmax noise). Soft Threshold Tallying, however, finds itself in perfect conflict on these exact trials: the option with more total features has mostly low-validity features, while the option with fewer features contains the highest-validity cues. Consequently, Soft Threshold Tallying will frequently choose the option with fewer features, drastically lowering its expected value on this metric compared to Pure Tallying. Filtering solely for 'diff == 2' removes the dilution caused by 'diff == 1' trials, maximizing the mean gap while keeping Tallying's variance extremely tight."
}
```

## Usage

```json
{
  "prompt_token_count": 4973,
  "candidates_token_count": 514,
  "total_token_count": 10171
}
```
