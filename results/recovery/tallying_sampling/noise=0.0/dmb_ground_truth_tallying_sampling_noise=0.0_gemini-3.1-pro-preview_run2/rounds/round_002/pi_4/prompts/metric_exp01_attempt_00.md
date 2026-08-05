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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 9: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 10: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate Thresholded Tallying from Pure Tallying (Equal Weight), we use a 6-feature design with a clear bimodal distribution of validities: three high-validity features (0.85-0.95) and three low-validity features (0.50-0.60). Pure Tallying ignores validities and counts all positive features equally. Thresholded Tallying applies a soft threshold that filters out low-validity features, effectively tallying only the high-validity ones. By pitting options with fewer high-validity features against options with a greater total number of features (which are predominantly low-validity), we create stark dissociations. For example, an option with 2 high-validity and 0 low-validity features will be chosen by Thresholded Tallying over an option with 0 high-validity and 3 low-validity features, while Pure Tallying predicts the opposite. We also include tie-breaking trials where Pure Tallying predicts equal preference but Thresholded Tallying strongly favors the option with high-validity cues.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [-1.0, 1.0]
- k: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
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
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
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
[0] rationale: This metric isolates trials where standard Tallying and Thresholded Tallying make opposing predictions. It computes the proportion of decisions on these conflict trials that align with Thresholded Tallying. For subjects using standard Tallying, this metric will be significantly below 0.5, whereas for subjects using Thresholded Tallying, it will be significantly above 0.5. This provides maximum discrimination between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # Thresholded Tallying focuses heavily on the first 3 (high-validity) features
    tt_a = a_ratings[:, :3].sum(axis=1)
    tt_b = b_ratings[:, :3].sum(axis=1)
    diff_tt = tt_a - tt_b
    
    # Standard Tallying equally weights all 5 features
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    diff_tally = tally_a - tally_b
    
    # Identify conflict trials where the two heuristics prefer different options
    conflict_mask = (diff_tt * diff_tally) < 0
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Check if the subject's response aligns with the Thresholded Tallying prediction
    tt_predicts_a = diff_tt > 0
    tt_predicts_b = diff_tt < 0
    
    aligned = (tt_predicts_a & (responses == 0)) | (tt_predicts_b & (responses == 1))
    
    return float(np.mean(aligned[conflict_mask]))
outcome: self_sim=0.2693 (var=0.0706) adversary_sim=0.1623 (var=0.0065) welch_t=+1.927 p=0.06409 (N=25, alpha=0.01) -> reject

[1] rationale: This metric exploits a structural symmetry in the experimental design to perfectly control for the Tallying model's predictions. In both the congruent trials (7 and 8) and the conflict trials (1, 2, 3, 5, 6), one option has exactly 3 positive features and the other has exactly 2. Because pure Tallying relies exclusively on the unweighted sum of features, its preference for the sum=3 option is mathematically identical across both sets of trials, regardless of the subject's noise (beta) or lapse (epsilon) parameters. Consequently, for subjects using pure Tallying, the difference in the proportion of times they choose the sum=3 option between these two sets of trials will be zero in expectation, with very low variance. Conversely, Thresholded Tallying filters out low-validity features. In trials 7 and 8, the sum=3 option contains mostly high-validity features and is strongly preferred. In the conflict trials, the sum=3 option is dominated by low-validity features, causing Thresholded Tallying to prefer the sum=2 option instead. Thus, for Thresholded Tallying, the difference in choice proportions will be strongly positive, cleanly dissociating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Identify trials where one option has sum=3 and the other has sum=2
    mask_3_2 = (sum_a == 3) & (sum_b == 2)
    mask_2_3 = (sum_a == 2) & (sum_b == 3)
    valid_trials = mask_3_2 | mask_2_3
    
    # For these trials, did the subject choose the option with sum=3?
    responses = data['response'].values
    chose_3 = np.zeros(len(data), dtype=bool)
    chose_3[mask_3_2 & (responses == 0)] = True
    chose_3[mask_2_3 & (responses == 1)] = True
    
    # Differentiate between trials based on the number of high-validity features 
    # (the first 3 features) in the option with sum=3.
    high_a = a_ratings[:, :3].sum(axis=1)
    high_b = b_ratings[:, :3].sum(axis=1)
    
    high_of_3 = np.zeros(len(data))
    high_of_3[mask_3_2] = high_a[mask_3_2]
    high_of_3[mask_2_3] = high_b[mask_2_3]
    
    # Trial 7/8: the sum=3 option has 3 high-validity features
    mask_78 = valid_trials & (high_of_3 == 3)
    
    # Conflict trials (1, 2, 3, 5, 6): the sum=3 option has only 1 high-validity feature
    mask_conflict = valid_trials & (high_of_3 == 1)
    
    if np.sum(mask_78) == 0 or np.sum(mask_conflict) == 0:
        return 0.0
        
    p_78 = np.mean(chose_3[mask_78])
    p_conflict = np.mean(chose_3[mask_conflict])
    
    # Difference in probability of choosing the sum=3 option
    return float(p_78 - p_conflict)
outcome: self_sim=0.1360 (var=0.0739) adversary_sim=-0.0168 (var=0.0065) welch_t=+2.695 p=0.01173 (N=25, alpha=0.01) -> reject

[2] rationale: We estimate the implicit weight the subject places on each feature by computing the covariance between the feature differences (Option A - Option B) and the subject's choices (+1 for A, -1 for B). Because the experiment is adversarially designed such that options with higher total sums often have fewer high-validity features, a pure Tallying strategy (which relies only on the unweighted sum) will produce choices that paradoxically correlate more strongly with the low-validity features. For Tallying, the expected difference between the implicit weights of high-validity and low-validity features is negative. In contrast, Thresholded Tallying filters out low-validity features, causing choices to be driven predominantly by the high-validity features. For Thresholded Tallying, this difference in implicit weights will be strongly positive. This metric provides a robust, linear measure of feature reliance that cleanly separates the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    X = a_ratings - b_ratings
    y = 1 - 2 * data['response'].values
    
    w = np.mean(X * y[:, None], axis=0)
    
    return float((w[0] + w[1] + w[2]) - (w[3] + w[4]))
outcome: self_sim=0.0517 (var=0.9899) adversary_sim=-0.4642 (var=0.0560) welch_t=+2.522 p=0.01795 (N=25, alpha=0.01) -> reject

[3] rationale: Standard Tallying relies exclusively on the unweighted sum of features. In 7 out of the 8 trial types in this design, the options have total sums of 3 and 2. Because the unweighted score difference is exactly identical (+1) across all 7 of these trial types, a pure Tallying strategy predicts the exact same probability of choosing the sum=3 option for every single one of them. Consequently, for Tallying, the variance of the choice proportions across these 7 trial types should theoretically be zero, driven only by finite-sample binomial noise (which vanishes entirely when evaluated on the pooled dataframe).

Thresholded Tallying, however, filters out low-validity features. In two of these trial types (the congruent trials), the sum=3 option is rich in high-validity features, leading to a very high choice probability. In the other five (the conflict trials), the sum=3 option is dominated by low-validity features, leading to a much lower choice probability. This structural difference drastically inflates the variance of the choice proportions across the 7 trial types. By measuring this variance, we create a metric that is mathematically invariant (expected value near 0) for Tallying, while capturing the large structural divergence predicted by Thresholded Tallying, yielding a highly significant and stable dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for each trial type by concatenating the feature tuples
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    data = data.assign(trial_key=a_tuples + b_tuples)
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Isolate the 7 trial types where one option has exactly 3 positive features and the other has 2
    mask_3_2 = (sum_a == 3) & (sum_b == 2)
    mask_2_3 = (sum_a == 2) & (sum_b == 3)
    valid_mask = mask_3_2 | mask_2_3
    
    if not np.any(valid_mask):
        return 0.0
        
    df_valid = data[valid_mask].copy()
    
    # Determine if the subject chose the option with sum=3
    responses = df_valid['response'].values
    is_3_2 = mask_3_2[valid_mask]
    is_2_3 = mask_2_3[valid_mask]
    
    chose_3 = np.zeros(len(df_valid), dtype=float)
    chose_3[is_3_2 & (responses == 0)] = 1.0
    chose_3[is_2_3 & (responses == 1)] = 1.0
    
    df_valid['chose_3'] = chose_3
    
    # Calculate the empirical probability of choosing the sum=3 option for each of the 7 trial types
    means = df_valid.groupby('trial_key')['chose_3'].mean().values
    
    if len(means) == 0:
        return 0.0
        
    # The metric is the variance of these choice probabilities across the 7 trial types
    return float(np.var(means))
outcome: self_sim=0.0039 (var=0.0017) adversary_sim=0.0002 (var=0.0000) welch_t=+0.436 p=0.6666 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    # Pure Tallying relies only on the unweighted sum of features.\n    sum_a = a_ratings.sum(axis=1)\n    sum_b = b_ratings.sum(axis=1)\n    \n    # Identify \"tie\" trials where both options have the exact same number of total positive features.\n    # In the provided design, these are trials 5, 6, 7, and 9.\n    tie_mask = (sum_a == sum_b)\n    \n    if not np.any(tie_mask):\n        return 0.5\n        \n    responses = data['response'].values\n    \n    # In all of these tie trials, Option A was adversarially designed to contain strictly more \n    # high-validity features than Option B. \n    # We calculate the proportion of times the subject chose Option A on these trials.\n    p_a = np.mean(responses[tie_mask] == 0)\n    \n    return float(p_a)",
  "rationale": "For the Pure Tallying model, options A and B in the tie trials (trials 5, 6, 7, and 9) have the exact same unweighted sum of features. Consequently, Pure Tallying predicts a choice probability of exactly 0.5 for Option A on every single one of these trials, regardless of the subject's noise (beta) or lapse (epsilon) parameters. The variance of this metric for Pure Tallying subjects will be extremely low, driven purely by binomial sampling noise over the 36 tie trials per subject. Conversely, Thresholded Tallying softly filters out the low-validity features. Because Option A is heavily loaded with high-validity features in all these tie trials, Thresholded Tallying will strongly prefer Option A whenever its threshold parameter is sufficiently high to engage the filtering. This creates a large, directional mean shift for Thresholded Tallying (> 0.5) against a mathematically locked baseline (0.5) with minimal variance for Pure Tallying, guaranteeing a highly significant t-statistic."
}
```

## Usage

```json
{
  "prompt_token_count": 6236,
  "candidates_token_count": 545,
  "total_token_count": 11768
}
```
