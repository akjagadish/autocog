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
**Validities (n_features=5):** [0.9, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates Random Subset Tallying from Softmax Tallying by testing the linearity of log-odds. Under Softmax Tallying, the log-odds of choosing an option are strictly proportional to the tally difference (e.g., the log-odds for a difference of 4 must be exactly twice the log-odds for a difference of 2). Random Subset Tallying, however, generates choice probabilities through a combinatorial sampling process, which fundamentally violates this linear log-odds property. By systematically varying the tally difference from 1 to 5 using options with entirely non-overlapping features (e.g., 1v0, 2v0, 3v0, 4v0, 5v0), we can evaluate whether the log-odds scale linearly (supporting Softmax Tallying) or non-linearly (supporting Random Subset Tallying).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Random Subset Tallying: Decision-makers use an equal-weight heuristic but are bounded by working memory, preventing them from processing all features simultaneously. Instead of calculating a complete tally and applying post-decision softmax noise, they stochastically sample a subset of the available features on each trial (each feature included independently with some probability) and perform pure tallying strictly on that subset. This provides a mechanistic, cognitive origin for choice variability while preserving the validity-agnostic, compensatory nature of the Tallying heuristic.

**Parameters:**
- sample_prob: [0.7, 1.0]
- epsilon: [0.0, 0.1]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import itertools
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    n_features = stim.shape[1]
    p = float(parameters["sample_prob"])
    epsilon = float(parameters["epsilon"])
    
    prob_A = 0.0
    
    # Iterate over all possible subsets of features (2^n_features)
    for seq in itertools.product([0, 1], repeat=n_features):
        mask = np.array(seq)
        # Probability of sampling this specific subset
        subset_prob = np.prod(np.where(mask == 1, p, 1.0 - p))
        
        if subset_prob == 0:
            continue
            
        score_A = np.sum(stim[0] * mask)
        score_B = np.sum(stim[1] * mask)
        
        # Pure tallying on the sampled subset
        if score_A > score_B:
            prob_A += subset_prob
        elif score_A == score_B:
            prob_A += 0.5 * subset_prob
            
    prob_B = 1.0 - prob_A
    p_core = np.array([prob_A, prob_B])
    
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: Under Softmax Tallying, the probability of choosing Option A depends exclusively on the difference in tallies between A and B. Therefore, the probability of choosing A when the tallies are (k vs 0) should be identical to when the tallies are (k+1 vs 1), because the difference is k in both cases. The proposed metric computes the difference in the choice rate of A between these two conditions. For Softmax Tallying, this metric will be exactly 0 in expectation. Under Random Subset Tallying, however, the presence of an additional feature in Option B (even when Option A also gets an additional feature) increases the combinatorial chance that Option B will tie or win within a sampled subset, systematically reducing the probability of choosing A. Thus, Random Subset Tallying predicts a strictly positive value for this metric, cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_A = data['option_a_ratings'].apply(sum)
    tally_B = data['option_b_ratings'].apply(sum)
    diff = tally_A - tally_B
    
    mask_k_0 = (tally_B == 0) & (diff.isin([1, 2, 3]))
    mask_k_1 = (tally_B == 1) & (diff.isin([1, 2, 3]))
    
    p_A_k_0 = (data.loc[mask_k_0, 'response'] == 0).mean()
    p_A_k_1 = (data.loc[mask_k_1, 'response'] == 0).mean()
    
    return float(p_A_k_0 - p_A_k_1)
outcome: self_sim=0.0228 (var=0.0030) adversary_sim=-0.0167 (var=0.0057) welch_t=+2.111 p=0.04045 (N=25, alpha=0.01) -> reject

[1] rationale: Under Softmax Tallying, choice probability is a logistic function of the tally difference. Because the inverse temperature `beta` is typically > 1 (prior is Uniform[0.1, 20]), the probability of choosing A saturates almost completely even at a difference of 1. Thus, Softmax Tallying predicts almost no difference in accuracy between `diff=1` trials and `diff>=3` trials (the gap is expected to be near zero). Under Random Subset Tallying, decision-makers sample a subset of features. For a difference of 1, there is a substantial combinatorial chance (related to `1-p`) that the deciding feature is missed, leading to a tie and significantly reducing accuracy. For a difference of 3 or more, it is extremely unlikely to miss all deciding features, so accuracy is much higher. Therefore, Random Subset Tallying predicts a significantly larger, robust gap in choice probability between high-difference and low-difference trials compared to Softmax Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the unweighted tally for each option
    tally_A = data['option_a_ratings'].apply(sum)
    tally_B = data['option_b_ratings'].apply(sum)
    diff = tally_A - tally_B
    
    # Group trials into 'high difference' (>= 3) and 'low difference' (== 1)
    mask_high = diff >= 3
    mask_low = diff == 1
    
    # Calculate the choice probability for Option A in both conditions
    p_high = (data.loc[mask_high, 'response'] == 0).mean()
    p_low = (data.loc[mask_low, 'response'] == 0).mean()
    
    # Fallback for empty slices (e.g., if a subject somehow missed all trials of a type)
    if pd.isna(p_high): p_high = 0.5
    if pd.isna(p_low): p_low = 0.5
    
    # Return the slope/gap in accuracy between high and low tally differences
    return float(p_high - p_low)
outcome: self_sim=0.0879 (var=0.0040) adversary_sim=0.0275 (var=0.0118) welch_t=+2.402 p=0.0212 (N=25, alpha=0.01) -> reject

[2] rationale: Under Softmax Tallying, choice probabilities depend strictly on the difference in tallies. Because the difference is identically k for both the (k vs 0) and (k+1 vs 1) trials, the difference in choice probabilities between these pairs is exactly zero in expectation. Under Random Subset Tallying, however, the presence of an extra feature in Option B (even when Option A also gets one) increases the combinatorial chance that Option B ties or wins within a sampled subset, making the choice probability of A systematically lower in (k+1 vs 1) than in (k vs 0). By summing these gaps and dividing by a stabilized estimate of the subject's baseline error rate, we create a metric that is tightly clustered around a positive value for Random Subset Tallying (normalizing out parameter-driven variance) while remaining firmly centered at zero for Softmax Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate unweighted tallies for each option
    tally_A = data['option_a_ratings'].apply(sum)
    tally_B = data['option_b_ratings'].apply(sum)
    
    # Define masks for the relevant trial pairs
    m_1v0 = (tally_A == 1) & (tally_B == 0)
    m_2v1 = (tally_A == 2) & (tally_B == 1)
    m_2v0 = (tally_A == 2) & (tally_B == 0)
    m_3v1 = (tally_A == 3) & (tally_B == 1)
    m_3v0 = (tally_A == 3) & (tally_B == 0)
    m_4v1 = (tally_A == 4) & (tally_B == 1)
    
    # Helper to get the choice probability for Option A
    def get_pA(mask):
        if not mask.any(): 
            return 0.5
        return (data.loc[mask, 'response'] == 0).mean()
        
    # Calculate the sum of differences between (k vs 0) and (k+1 vs 1)
    num = (get_pA(m_1v0) - get_pA(m_2v1)) + \
          (get_pA(m_2v0) - get_pA(m_3v1)) + \
          (get_pA(m_3v0) - get_pA(m_4v1))
          
    # Self-normalizing denominator based on base error rates to reduce between-subject variance
    den = (1.0 - get_pA(m_1v0)) + (1.0 - get_pA(m_2v0)) + 0.2
    
    return float(num / den)
outcome: self_sim=0.2010 (var=0.3712) adversary_sim=-0.0943 (var=0.1751) welch_t=+1.998 p=0.05216 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Calculate unweighted tallies for each option\n    tally_A = data['option_a_ratings'].apply(sum)\n    tally_B = data['option_b_ratings'].apply(sum)\n    \n    # Group trials into two pools with matched tally differences (diff = 1, 2, 3)\n    # Pool 0: Option B has 0 features (1v0, 2v0, 3v0)\n    mask_v0 = (tally_B == 0) & (tally_A.isin([1, 2, 3]))\n    \n    # Pool 1: Option B has 1 feature (2v1, 3v1, 4v1)\n    mask_v1 = (tally_B == 1) & (tally_A.isin([2, 3, 4]))\n    \n    # Count the number of times Option B (the inferior option) was chosen\n    B0 = (data.loc[mask_v0, 'response'] == 1).sum()\n    B1 = (data.loc[mask_v1, 'response'] == 1).sum()\n    \n    # Return the smoothed log-ratio of B choices\n    # Laplace smoothing (+2) stabilizes the variance for subjects with few errors\n    return float(np.log((B1 + 2.0) / (B0 + 2.0)))",
  "rationale": "Under Softmax Tallying, choice probabilities depend strictly on the difference in tallies between Option A and Option B. Therefore, the probability of choosing B in (k vs 0) trials is mathematically identical to the probability in (k+1 vs 1) trials. Because the B-choice counts for these two matched sets are drawn from the exact same binomial distribution, their log-ratio has an expected value of exactly 0 for every subject, completely eliminating parameter-driven variance (beta/epsilon).\n\nUnder Random Subset Tallying, subjects sample a subset of features. The presence of an extra feature in Option B increases the combinatorial probability of a tie within the sampled subset (which is broken randomly). This systematically increases the probability of choosing B in (k+1 vs 1) trials compared to (k vs 0) trials, guaranteeing a strictly positive expected value.\n\nBy computing the log-ratio of B choices with a +2 Laplace smoothing, we obtain a metric that is mathematically anchored at 0 for Softmax Tallying, but robustly positive for Random Subset Tallying. The log-ratio normalizes out individual differences in baseline error rates, tightly constraining the between-subject variance and ensuring high discriminability."
}
```

## Usage

```json
{
  "prompt_token_count": 5019,
  "candidates_token_count": 628,
  "total_token_count": 34005
}
```
