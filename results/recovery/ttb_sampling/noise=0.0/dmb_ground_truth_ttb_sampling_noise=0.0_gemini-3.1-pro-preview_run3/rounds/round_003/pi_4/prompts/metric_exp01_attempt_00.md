# metric_exp01_attempt_00

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
**Validities (n_features=5):** [0.9, 0.88, 0.86, 0.84, 0.82]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To strongly dissociate Probabilistic Take-The-Best (PTTB) from deterministic Take-The-Best (TTB), we use a 5-feature design with closely spaced validities. TTB strictly chooses the option favored by the single most valid discriminating feature, completely ignoring all other features. In contrast, PTTB stochastically selects a discriminating feature with a probability proportional to an exponential function of its validity. By pitting the single best feature against a coalition of several slightly less valid features, TTB predicts a deterministic choice for the option with the best feature, while PTTB's choice probabilities will be much more graded and can even favor the opposing option (for low to moderate gamma values) because the sum of the exponential weights of the opposing features exceeds that of the single best feature.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Take-The-Best (PTTB) with exponential weighting: Decision-makers search through features sequentially but stochastically. The probability of checking a feature is proportional to an exponential function of its validity (exp(gamma * validity)), allowing high-validity features to strongly dominate the search order. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the decision-maker guesses. This yields a Luce choice rule over the exponentially weighted discriminating features.

**Parameters:**
- gamma: [0.1, 200.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating features for each option
    d_a = (a > b).astype(float)
    d_b = (b > a).astype(float)
    d_any = d_a + d_b
    
    if np.sum(d_any) == 0:
        p_a = 0.5
    else:
        # To prevent numerical overflow, subtract max validity among discriminating features
        max_v = np.max(validities[d_any > 0])
        weights = np.exp(gamma * (validities - max_v))
        
        # Sum of weights of features favoring A and B
        w_a = np.sum(d_a * weights)
        w_b = np.sum(d_b * weights)
        
        # The probability that the first discriminating feature found favors A
        p_a = w_a / (w_a + w_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: This metric calculates the proportion of trials where the subject's choice aligns with the deterministic Take-The-Best (TTB) heuristic. TTB strongly predicts a high match rate (deviating only due to uniform lapse rate epsilon). Probabilistic Take-The-Best (PTTB), however, allows for stochastic feature selection. Because the experimental design explicitly pits the single highest-validity discriminating feature against a coalition of multiple slightly lower-validity features, PTTB will frequently sample a lower-validity feature and choose the opposing option. Thus, PTTB will produce a significantly lower match rate than TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # Features are ordered by validity descending.
    # Find the first feature where A and B differ.
    diff = A != B
    first_diff_idx = np.argmax(diff, axis=1)
    
    row_indices = np.arange(len(data))
    a_vals = A[row_indices, first_diff_idx]
    b_vals = B[row_indices, first_diff_idx]
    
    # TTB strictly chooses the option that is superior on this first discriminating feature.
    ttb_choices = (b_vals > a_vals).astype(int)
    
    responses = data['response'].values
    match_rate = np.mean(responses == ttb_choices)
    
    return float(match_rate)
outcome: self_sim=0.9073 (var=0.0075) adversary_sim=0.8669 (var=0.0069) welch_t=+1.686 p=0.09832 (N=25, alpha=0.01) -> reject

[1] rationale: Under the deterministic Take-The-Best (TTB) model, the probability of making a TTB-consistent choice is exactly 1 - epsilon/2 for every single trial type, meaning the true underlying TTB match rate is constant. Any variance in the sample match rates across the 12 trial types is purely due to binomial sampling noise (which is relatively small). In contrast, Probabilistic Take-The-Best (PTTB) evaluates features stochastically proportional to their validity weights. Because the experimental design pits the best feature against varying coalitions of lower-validity features, PTTB's true probability of making a TTB-consistent choice will fluctuate significantly across different trial types depending on the strength of the opposing coalition. Therefore, the variance of the TTB match rate across trial types will be systematically higher for PTTB than for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # TTB prediction: find the first feature where A and B differ
    diff = A != B
    first_diff_idx = np.argmax(diff, axis=1)
    
    row_indices = np.arange(len(data))
    a_vals = A[row_indices, first_diff_idx]
    b_vals = B[row_indices, first_diff_idx]
    
    # TTB strictly chooses the option that is superior on this first discriminating feature
    ttb_choices = (b_vals > a_vals).astype(int)
    
    # Calculate whether the subject's response matched the TTB prediction
    ttb_match = (data['response'].values == ttb_choices).astype(float)
    
    # Create a unique key for each trial type (the combination of A and B features)
    trial_keys = [tuple(a) + tuple(b) for a, b in zip(A, B)]
    
    df_temp = pd.DataFrame({'trial_key': trial_keys, 'ttb_match': ttb_match})
    
    # Calculate the TTB match rate for each of the 12 unique trial types
    match_rates = df_temp.groupby('trial_key')['ttb_match'].mean()
    
    # Return the variance of these match rates across trial types
    return float(np.var(match_rates, ddof=0))

outcome: self_sim=0.0005 (var=0.0001) adversary_sim=0.0004 (var=0.0001) welch_t=+0.023 p=0.9818 (N=25, alpha=0.01) -> reject

[2] rationale: Under the deterministic Take-The-Best (TTB) model, the probability of making a TTB-consistent choice is constant across all trial types (1 - epsilon/2), irrespective of how many lower-validity features oppose the best feature. Thus, the difference in TTB match rates between trials where the TTB choice is supported by 2 features (Group H) and trials where it is supported by only 1 feature (Group L) should be exactly 0 on average. Under Probabilistic Take-The-Best (PTTB), features are sampled stochastically proportional to their exponential validities. In Group L, the opposing coalition has 3 or 4 features, presenting a larger combined probability mass to be sampled first compared to Group H, where the opposing coalition has only 2 or 3 features. Therefore, PTTB predicts a significantly higher TTB match rate in Group H than in Group L, yielding a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # Find the first feature where A and B differ (TTB's deciding feature)
    diff = A != B
    first_diff_idx = np.argmax(diff, axis=1)
    
    row_indices = np.arange(len(data))
    a_vals = A[row_indices, first_diff_idx]
    b_vals = B[row_indices, first_diff_idx]
    
    # TTB strictly chooses the option that is superior on this first discriminating feature
    ttb_choices = (b_vals > a_vals).astype(int)
    ttb_match = (data['response'].values == ttb_choices).astype(float)
    
    # Count how many discriminating features favor the TTB choice
    # and how many favor the opposing coalition.
    # For each trial, sum the discriminating features for A and B
    d_a = (A > B).astype(int)
    d_b = (B > A).astype(int)
    
    # Number of features favoring the TTB choice
    ttb_features_count = np.where(ttb_choices == 0, d_a.sum(axis=1), d_b.sum(axis=1))
    
    # Group L: TTB choice is supported by only 1 feature (the rest form a large opposing coalition)
    # Group H: TTB choice is supported by 2 features (smaller opposing coalition)
    group_l_mask = (ttb_features_count == 1)
    group_h_mask = (ttb_features_count == 2)
    
    if np.sum(group_h_mask) == 0 or np.sum(group_l_mask) == 0:
        return 0.0
        
    match_rate_h = np.mean(ttb_match[group_h_mask])
    match_rate_l = np.mean(ttb_match[group_l_mask])
    
    return float(match_rate_h - match_rate_l)
outcome: self_sim=0.0287 (var=0.0041) adversary_sim=0.0062 (var=0.0054) welch_t=+1.157 p=0.2529 (N=25, alpha=0.01) -> reject

[3] rationale: Under deterministic Take-The-Best (TTB), the probability of choosing the option favored by the most valid discriminating feature is constant across all trial types (1 - epsilon/2), regardless of the validities of the opposing features. Thus, the TTB match rate should be identical for trials where the validity gap between the best feature and the best opposing feature is 0.05 versus 0.10. In contrast, Probabilistic Take-The-Best (PTTB) samples features with probabilities proportional to exp(gamma * validity). A smaller validity gap (0.05) means the opposing features have exponentially higher relative weights compared to a larger gap (0.10), leading to a significantly higher rate of 'errors' (non-TTB choices) on 0.05 gap trials. This metric computes the difference in TTB match rates between large-gap and small-gap trials, which will be near zero for TTB but strictly positive for PTTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    validities = np.array([0.95, 0.85, 0.8, 0.75, 0.7])
    
    d_a = (A > B).astype(float)
    d_b = (B > A).astype(float)
    
    max_v_a = np.max(d_a * validities, axis=1)
    max_v_b = np.max(d_b * validities, axis=1)
    
    ttb_choices = (max_v_b > max_v_a).astype(int)
    gap = np.abs(max_v_a - max_v_b)
    
    responses = data['response'].values
    ttb_match = (responses == ttb_choices).astype(float)
    
    gap_05_mask = np.isclose(gap, 0.05)
    gap_10_mask = gap >= 0.09
    
    if np.sum(gap_05_mask) == 0 or np.sum(gap_10_mask) == 0:
        return 0.0
        
    match_05 = np.mean(ttb_match[gap_05_mask])
    match_10 = np.mean(ttb_match[gap_10_mask])
    
    return float(match_10 - match_05)
outcome: self_sim=0.0400 (var=0.0067) adversary_sim=0.0206 (var=0.0037) welch_t=+0.951 p=0.347 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    A = np.vstack(data['option_a_ratings'].values)\n    B = np.vstack(data['option_b_ratings'].values)\n    \n    # TTB decides based on the first feature where A and B differ\n    diff = A != B\n    first_diff_idx = np.argmax(diff, axis=1)\n    \n    row_indices = np.arange(len(data))\n    a_vals = A[row_indices, first_diff_idx]\n    b_vals = B[row_indices, first_diff_idx]\n    \n    # TTB choice: 0 for A, 1 for B\n    ttb_choices = (b_vals > a_vals).astype(int)\n    \n    # Match between subject response and TTB choice\n    responses = data['response'].values\n    ttb_match = (responses == ttb_choices).astype(float)\n    \n    # Count how many features favor the TTB choice\n    d_a = (A > B).astype(int)\n    d_b = (B > A).astype(int)\n    \n    ttb_features_count = np.where(ttb_choices == 0, d_a.sum(axis=1), d_b.sum(axis=1))\n    \n    # Compare trials where TTB is supported by 3 features vs 1 feature\n    mask_3 = (ttb_features_count == 3)\n    mask_1 = (ttb_features_count == 1)\n    \n    if np.sum(mask_3) == 0 or np.sum(mask_1) == 0:\n        return 0.0\n        \n    return float(np.mean(ttb_match[mask_3]) - np.mean(ttb_match[mask_1]))",
  "rationale": "Under the deterministic Take-The-Best (TTB) model, the probability of making a TTB-consistent choice is constant across all trial types (1 - epsilon/2). Thus, the difference in TTB match rates between trials where the TTB choice is supported by 3 features and trials where it is supported by only 1 feature should be exactly 0 on average. Under Probabilistic Take-The-Best (PTTB), features are sampled stochastically proportional to their exponential validities. When the TTB choice is supported by only 1 feature, the opposing coalition has 3 or 4 features, presenting a much larger combined probability mass to be sampled first compared to trials where the TTB choice is supported by 3 features. Therefore, PTTB predicts a significantly higher TTB match rate when 3 features favor the TTB choice than when only 1 feature does, yielding a strongly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 5875,
  "candidates_token_count": 666,
  "total_token_count": 9523
}
```
