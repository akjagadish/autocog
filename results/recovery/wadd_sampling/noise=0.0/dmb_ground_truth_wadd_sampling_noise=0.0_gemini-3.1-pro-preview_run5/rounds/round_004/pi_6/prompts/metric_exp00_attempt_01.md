# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** This design quantitatively dissociates pure WADD from Evidence-Dependent Noise (EDN) by pitting net evidence against total feature conflict. We construct a sequence of trials where the net evidence favoring Option A strictly increases (e.g., +0.05, +0.10, +0.15), but the total conflict (the sum of validities of all differing features) increases at a much faster rate. Pure WADD dictates that choice probability is monotonically related to net weighted evidence, so it must predict a strictly increasing preference for Option A across this sequence, regardless of its gamma and beta parameters. In contrast, EDN posits that decision noise scales with total conflict. Because conflict grows substantially across these trials, EDN predicts a preference reversal: the choice probability for Option A will regress toward 0.5 on the high-conflict trials despite the larger net evidence in its favor. This structural divergence provides a definitive, parameter-free test between the two theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.

**Parameters:**
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- theta: [0.0, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective feature weighting
    weights = val ** gamma
    
    # Calculate weighted sums of evidence for each option
    scores = np.dot(stim, weights)
    
    # Calculate conflict: total weight of features where the options differ
    diff = np.abs(stim[0] - stim[1])
    conflict = np.dot(diff, weights)
    
    # Effective beta scales inversely with conflict (higher conflict = more noise)
    beta_eff = beta / (1.0 + theta * conflict)
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta_eff * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
[0] rationale: Evaluates the difference in the probability of choosing the option with higher net evidence when the total conflict is high versus when the total conflict is low. Under pure WADD, the net evidence is larger in the high conflict trials, so the probability should increase (positive difference). Under EDN, the decision noise scales with conflict, regressing choice probabilities toward 0.5, thus the difference should be negative or significantly lower.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Low conflict trials (net evidence 0.05)
    t1_mask = (data['A_tuple'] == (1,0,0,0,0,0)) & (data['B_tuple'] == (0,1,0,0,0,0))
    t6_mask = (data['A_tuple'] == (0,1,0,0,0,0)) & (data['B_tuple'] == (1,0,0,0,0,0))
    
    # High conflict trials (net evidence 0.15)
    t3_mask = (data['A_tuple'] == (1,0,1,0,1,0)) & (data['B_tuple'] == (0,1,0,1,0,1))
    t8_mask = (data['A_tuple'] == (0,1,0,1,0,1)) & (data['B_tuple'] == (1,0,1,0,1,0))
    
    p_t1 = (data[t1_mask]['response'] == 0).mean()
    p_t3 = (data[t3_mask]['response'] == 0).mean()
    
    p_t6 = (data[t6_mask]['response'] == 1).mean()
    p_t8 = (data[t8_mask]['response'] == 1).mean()
    
    # Average target probabilities
    p_high_conflict = np.nanmean([p_t3, p_t8])
    p_low_conflict = np.nanmean([p_t1, p_t6])
    
    if np.isnan(p_high_conflict) or np.isnan(p_low_conflict):
        return 0.0
        
    return float(p_high_conflict - p_low_conflict)
outcome: self_sim=-0.0017 (var=0.0137) adversary_sim=0.0758 (var=0.0150) welch_t=-2.284 p=0.02687 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    data = data.copy()\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    data['B_tuple'] = data['option_b_ratings'].apply(tuple)\n    \n    # Low conflict trials (T1, T6)\n    m_t1 = (data['A_tuple'] == (1,0,0,0,0,0)) & (data['B_tuple'] == (0,1,0,0,0,0))\n    m_t6 = (data['A_tuple'] == (0,1,0,0,0,0)) & (data['B_tuple'] == (1,0,0,0,0,0))\n    acc_t1 = (data[m_t1]['response'] == 0).mean()\n    acc_t6 = (data[m_t6]['response'] == 1).mean()\n    acc_low = np.nanmean([acc_t1, acc_t6])\n    \n    # Medium conflict trials (T2, T7)\n    m_t2 = (data['A_tuple'] == (1,0,1,0,0,0)) & (data['B_tuple'] == (0,1,0,1,0,0))\n    m_t7 = (data['A_tuple'] == (0,1,0,1,0,0)) & (data['B_tuple'] == (1,0,1,0,0,0))\n    acc_t2 = (data[m_t2]['response'] == 0).mean()\n    acc_t7 = (data[m_t7]['response'] == 1).mean()\n    acc_med = np.nanmean([acc_t2, acc_t7])\n    \n    # High conflict trials (T3, T8)\n    m_t3 = (data['A_tuple'] == (1,0,1,0,1,0)) & (data['B_tuple'] == (0,1,0,1,0,1))\n    m_t8 = (data['A_tuple'] == (0,1,0,1,0,1)) & (data['B_tuple'] == (1,0,1,0,1,0))\n    acc_t3 = (data[m_t3]['response'] == 0).mean()\n    acc_t8 = (data[m_t8]['response'] == 1).mean()\n    acc_high = np.nanmean([acc_t3, acc_t8])\n    \n    if np.isnan(acc_low) or np.isnan(acc_med) or np.isnan(acc_high):\n        return 0.0\n        \n    # Compare pooled lower-conflict accuracy vs highest-conflict accuracy\n    return float((acc_low + acc_med) / 2.0 - acc_high)",
  "rationale": "By pooling the low and medium conflict trials to form a robust baseline and subtracting the accuracy on the highest conflict trials, we reduce per-subject variance while capturing the core divergence. Under pure WADD, accuracy monotonically increases with net evidence, so the high conflict (but highest net evidence) trials will be more accurate than the baseline, yielding a negative metric. Under EDN, the noise scales faster than the net evidence, causing a preference reversal where the highest conflict trials regress to chance, yielding a positive metric."
}
```

## Usage

```json
{
  "prompt_token_count": 4071,
  "candidates_token_count": 823,
  "total_token_count": 9875
}
```
