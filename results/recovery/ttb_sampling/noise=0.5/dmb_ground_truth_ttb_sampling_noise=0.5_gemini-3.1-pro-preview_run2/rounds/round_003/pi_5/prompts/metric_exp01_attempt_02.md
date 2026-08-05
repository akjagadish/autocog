# metric_exp01_attempt_02

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Exponentially Weighted Additive Model from Take The Best (TTB) by exploiting TTB's insensitivity to the amount of opposing lower-validity evidence. Across a sequence of trials, Option A is supported by the single highest-validity discriminating feature, while Option B is supported by an increasing number of lower-validity features. Because TTB evaluates features sequentially and stops at the first discriminating feature, it assigns a constant internal score difference (1 vs 0) for all these trials, predicting a completely flat choice probability for Option A across the gradient. In contrast, the Exponential model mathematically integrates all features (even when transformed exponentially), predicting a graded decrease in the preference for Option A as the opposing evidence for Option B accumulates. This graded compensatory response pattern is uniquely predicted by the Exponential model and cannot be captured by TTB's strict lexicographic stopping rule.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Exponentially Weighted Additive Model: Subjects integrate all available features in a compensatory manner, but they apply a steep non-linear (exponential) transformation to the feature validities. This causes the most valid features to heavily dominate the decision, effectively mimicking the non-compensatory 'Take The Best' heuristic while remaining mathematically compensatory. The steepness of this transformation dictates how closely the strategy approximates strict lexicographic choice.

**Parameters:**
- gamma: [0.5, 20.0]
- beta: [0.05, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Steep non-linear transformation of validities to weights
    weights = np.exp(gamma * validities)
    
    # Calculate option scores as weighted sums
    scores = stim @ weights
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
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
[0] rationale: This metric calculates the difference in the rate of choosing Option B between trials where Option B has more positive features than Option A, and trials where they have an equal number of positive features. Under the Take The Best (TTB) heuristic, the highest validity feature always favors Option A in this design, so choosing B is purely a result of random lapses. Thus, the rate of choosing B should be equal across both trial types, yielding a difference near 0. Under the Exponentially Weighted Additive Model, subjects with a lower gamma parameter will act in a compensatory manner, leading them to frequently choose Option B when it has a numerical advantage in positive features, but not when the feature counts are tied (since A's features have higher validities). This results in a significantly positive difference, distinguishing the two theories while controlling for baseline lapse rates.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    b_more_mask = b_sum > a_sum
    b_equal_mask = b_sum == a_sum
    
    b_more_rate = data.loc[b_more_mask, 'response'].mean()
    b_equal_rate = data.loc[b_equal_mask, 'response'].mean()
    
    if pd.isna(b_more_rate): b_more_rate = 0.0
    if pd.isna(b_equal_rate): b_equal_rate = 0.0
    
    return float(b_more_rate - b_equal_rate)
outcome: self_sim=0.0642 (var=0.0540) adversary_sim=0.0136 (var=0.0115) welch_t=+0.988 p=0.3302 (N=25, alpha=0.01) -> reject

[1] rationale: Under the Take The Best (TTB) heuristic, Option A is favored by the highest-validity discriminating feature on every single trial in the design. Consequently, any choices for Option B are purely the result of random lapses, meaning the rate of choosing B should be uniform across all trials. The Exponentially Weighted Additive Model, however, can behave in a compensatory manner. Trial 1 provides the strongest possible compensatory evidence for Option B (4 positive features vs 1 for Option A), making it the trial where the Exponential model is most likely to overcome the highest-validity feature and choose B. By contrasting the B-choice rate on Trial 1 specifically against all other trials, we isolate the maximum compensatory signal while perfectly controlling for the subject's baseline lapse rate, yielding a metric that is exactly 0 for TTB but significantly positive for the Exponential model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 1 uniquely has 4 positive features for Option B
    is_trial_1 = data['option_b_ratings'].apply(sum) == 4
    
    rate_trial_1 = data.loc[is_trial_1, 'response'].mean()
    rate_other = data.loc[~is_trial_1, 'response'].mean()
    
    if pd.isna(rate_trial_1): rate_trial_1 = 0.0
    if pd.isna(rate_other): rate_other = 0.0
    
    return float(rate_trial_1 - rate_other)
outcome: self_sim=0.0812 (var=0.0446) adversary_sim=0.0169 (var=0.0136) welch_t=+1.332 p=0.1909 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the covariance between the compensatory evidence (the difference in the total number of positive features between Option B and Option A) and the subject's binary choice (1 for B, 0 for A). Under the Take The Best (TTB) heuristic, the highest-validity feature points to Option A on every trial, so any choices for Option B are purely random lapses. Because these lapses are independent of the lower-validity features, the covariance is expected to be exactly 0, with very low variance. Under the Exponentially Weighted Additive Model, subjects act in a compensatory manner, meaning the probability of choosing Option B increases as its numerical advantage in features grows. This systematic relationship yields a strictly positive covariance, cleanly and reliably distinguishing the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    x = b_sum - a_sum
    mean_x = x.mean()
    cov = ((x - mean_x) * data['response']).mean()
    return float(cov)
outcome: self_sim=0.0347 (var=0.0096) adversary_sim=0.0054 (var=0.0016) welch_t=+1.390 p=0.1743 (N=25, alpha=0.01) -> reject

[3] rationale: Under the Take The Best (TTB) heuristic, the highest-validity discriminating feature always favors Option A across all 8 trial types in this design. Therefore, any choices for Option B are purely due to random lapses, meaning the true probability of choosing B is constant across all trials. The difference between the maximum and minimum B-choice rates across trial types will simply reflect binomial sampling noise and be relatively small. Under the Exponentially Weighted Additive Model, the compensatory evidence for Option B varies widely across the 8 trial types (e.g., Trial 1 heavily favors B in a compensatory sense, while others do not). This causes the true probability of choosing B to fluctuate significantly between trial types. By taking the range (max - min) of the choice rates across the unique trial types, we capture this systematic variance. The Exponential model will yield a much larger range than TTB, robustly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['trial_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    trial_means = data.groupby('trial_str')['response'].mean()
    return float(trial_means.max() - trial_means.min())
outcome: self_sim=0.1200 (var=0.0544) adversary_sim=0.0433 (var=0.0150) welch_t=+1.456 p=0.1541 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the Area Under the ROC Curve (AUC) for predicting the subject's choice of Option B using the number of positive features Option B possesses (the compensatory evidence). Under the Take The Best (TTB) heuristic, Option A is favored by the highest-validity feature on every trial, meaning choices for Option B are purely due to random lapses. Because these lapses are independent of the lower-validity feature counts, the AUC will be tightly centered around 0.5 (chance level). Under the Exponentially Weighted Additive Model, the probability of choosing Option B systematically increases as its number of positive features grows, yielding an AUC significantly greater than 0.5. By casting the problem as a rank-based AUC, we normalize out the subjects' baseline lapse rates and bound the metric between 0.5 and 1.0, which heavily suppresses the between-subject variance that caused previous raw-difference metrics to fail.
metric_source:
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    responses = data['response']
    n_pos = (responses == 1).sum()
    n_neg = (responses == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    ranks = b_sum.rank()
    pos_ranks = ranks[responses == 1].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
outcome: self_sim=0.6045 (var=0.0235) adversary_sim=0.5102 (var=0.0087) welch_t=+2.629 p=0.01214 (N=25, alpha=0.01) -> reject

[5] rationale: Under the Take The Best (TTB) heuristic, Option A is supported by the highest-validity feature on every single trial, so any choices for Option B are purely random lapses. Because these lapses are independent of how many lower-validity features Option B has, the rate of choosing B should be identical whether Option B has 1 positive feature or multiple (2+). Under the Exponentially Weighted Additive Model, subjects act in a compensatory manner, meaning the probability of choosing Option B increases systematically when it has a numerical advantage in features (2 or more) compared to when it only has 1. By splitting the trials into two large, roughly equal groups (B_sum >= 2 vs B_sum == 1) and taking the difference in choice rates, we maximize the compensatory signal while using enough trials per subject to tightly suppress baseline sampling variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    rate_high = data.loc[b_sum >= 2, 'response'].mean()
    rate_low = data.loc[b_sum == 1, 'response'].mean()
    if pd.isna(rate_high): rate_high = 0.0
    if pd.isna(rate_low): rate_low = 0.0
    return float(rate_high - rate_low)
outcome: self_sim=0.1619 (var=0.0704) adversary_sim=0.0159 (var=0.0094) welch_t=+2.581 p=0.01491 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    b_sum = data['option_b_ratings'].apply(sum)\n    mask_high = b_sum >= 3\n    mask_low = b_sum == 1\n    \n    rate_high = data.loc[mask_high, 'response'].mean()\n    rate_low = data.loc[mask_low, 'response'].mean()\n    \n    if pd.isna(rate_high): rate_high = 0.0\n    if pd.isna(rate_low): rate_low = 0.0\n    \n    return float(rate_high - rate_low)",
  "rationale": "Under the Take The Best (TTB) heuristic, Option A is supported by the highest-validity feature on every single trial. Any choices for Option B are purely random lapses, meaning the rate of choosing B should be identical regardless of how many lower-validity features Option B possesses. Under the Exponentially Weighted Additive Model, subjects act in a compensatory manner, so the probability of choosing Option B increases significantly when it has a strong numerical advantage in features. By contrasting the choice rate on trials where Option B has 3 or 4 positive features against trials where it has only 1, we isolate the strongest compensatory signal. This produces a much larger gap between the two theories than comparing against trials with 2 positive features, while still using enough trials per subject to maintain low baseline sampling variance."
}
```

## Usage

```json
{
  "prompt_token_count": 5590,
  "candidates_token_count": 334,
  "total_token_count": 8178
}
```
