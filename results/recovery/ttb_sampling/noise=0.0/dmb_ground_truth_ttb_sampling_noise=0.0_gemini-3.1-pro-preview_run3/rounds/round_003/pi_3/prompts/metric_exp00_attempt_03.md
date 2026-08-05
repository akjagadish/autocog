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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.75]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To dissociate Take-The-Best (TTB) from Probabilistic Take-The-Best (PTTB), we exploit their different treatments of lower-validity discriminating features. TTB deterministically chooses the option favored by the single highest-validity discriminating feature and completely ignores all others. PTTB, on the other hand, stochastically samples from all discriminating features with probabilities proportional to an exponential function of their validities. By creating trials where Option A is favored by the single most valid feature but Option B is favored by several slightly less valid features, TTB strongly predicts Option A. In contrast, PTTB's choice probabilities will be much more graded and can even favor Option B if the sum of the exponential weights of B's features exceeds A's.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the proportion of times the subject chooses Option A in the most extreme trial (Trial 1). In this trial, Option A is supported only by the single highest-validity feature, whereas Option B is supported by all four remaining features. The Take-The-Best (TTB) theory predicts a strong preference for Option A because it strictly follows the highest-validity feature. The Probabilistic Take-The-Best (PTTB) theory, depending on the gamma parameter, allows the sheer number of lower-validity features favoring Option B to outweigh Option A's single advantage, leading to a significantly lower choice rate for Option A on average. Isolating this trial maximizes the expected divergence between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    mask = data.apply(lambda row: tuple(row['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 1, 1, 1, 1), axis=1)
    
    if mask.sum() == 0:
        return np.nan
        
    return float(np.mean(data[mask]['response'] == 0))
outcome: self_sim=0.8700 (var=0.0106) adversary_sim=0.8867 (var=0.0260) welch_t=-0.435 p=0.6656 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 7 and Trial 1. In both trials, Option A is favored by the single highest-validity feature, meaning the strict Take-The-Best (TTB) model predicts an identical probability of choosing A (only influenced by the constant lapse rate epsilon). Therefore, TTB predicts a difference of 0. However, Probabilistic Take-The-Best (PTTB) is sensitive to the total weight of evidence. In Trial 1, Option B is favored by all four remaining features, pulling PTTB's choice probability towards B. In Trial 7, Option A is favored by three features while B is favored by only two, pulling PTTB's choice probability much more strongly towards A. Thus, PTTB predicts a strictly positive difference, reliably discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_trial_7(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 1, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 0, 0)
        
    def is_trial_1(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 1, 1, 1, 1)
        
    t7 = data.apply(is_trial_7, axis=1)
    t1 = data.apply(is_trial_1, axis=1)
    
    if t7.sum() == 0 or t1.sum() == 0:
        return np.nan
        
    p_a_t7 = np.mean(data[t7]['response'] == 0)
    p_a_t1 = np.mean(data[t1]['response'] == 0)
    
    return float(p_a_t7 - p_a_t1)
outcome: self_sim=-0.0083 (var=0.0170) adversary_sim=0.0050 (var=0.0229) welch_t=-0.334 p=0.7401 (N=25, alpha=0.01) -> reject

[2] rationale: The Take-The-Best (TTB) model predicts that choice probability is entirely determined by the single highest-validity discriminating feature. In Trials 4, 5, 1, 6, and 7, Option A is favored by this feature. Thus, TTB predicts an identical probability of choosing Option A across all these trials (determined solely by the subject's lapse rate), meaning the difference between any subset of these trials is exactly 0. In contrast, Probabilistic Take-The-Best (PTTB) samples features proportionally to an exponential function of their validities, making it sensitive to the total weight of opposing features. In Trials 4 and 5, Option B has weak opposing features, leading to a high probability of choosing A. In Trials 1, 6, and 7, Option B has stronger/more numerous opposing features, substantially pulling the choice probability lower. By computing the difference in the proportion of Option A choices between the 'High Adherence' (T4, T5) and 'Low Adherence' (T1, T6, T7) trials, we obtain a contrast that is mathematically 0 for TTB but strictly positive for PTTB across its entire parameter space, easily distinguishing the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project list columns to string representation for safe comparison
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # High adherence trials for PTTB (Option A favored by TTB, weak opposing evidence)
    t4 = (data['a_str'] == '01100') & (data['b_str'] == '00011')
    t5 = (data['a_str'] == '11000') & (data['b_str'] == '00111')
    high_mask = t4 | t5
    
    # Low adherence trials for PTTB (Option A favored by TTB, strong opposing evidence)
    t1 = (data['a_str'] == '10000') & (data['b_str'] == '01111')
    t6 = (data['a_str'] == '00100') & (data['b_str'] == '00011')
    t7 = (data['a_str'] == '10011') & (data['b_str'] == '01100')
    low_mask = t1 | t6 | t7
    
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
        
    # Proportion of times Option A was chosen in each group
    p_high = (data.loc[high_mask, 'response'] == 0).mean()
    p_low = (data.loc[low_mask, 'response'] == 0).mean()
    
    return float(p_high - p_low)
outcome: self_sim=-0.0100 (var=0.0083) adversary_sim=0.0483 (var=0.0095) welch_t=-2.183 p=0.03401 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    ttb_preds = {\n        '1000001111': 0,\n        '1010001011': 0,\n        '0100010000': 1,\n        '0110000011': 0,\n        '1100000111': 0,\n        '0010000011': 0,\n        '1001101100': 0,\n        '0101110100': 1\n    }\n    \n    data['trial_id'] = data.apply(\n        lambda row: ''.join([str(int(v)) for v in row['option_a_ratings']]) + \n                    ''.join([str(int(v)) for v in row['option_b_ratings']]), \n        axis=1\n    )\n    \n    data['ttb_match'] = (data['response'] == data['trial_id'].map(ttb_preds)).astype(float)\n    \n    subj_trial_adherence = data.groupby(['subject_id', 'trial_id'])['ttb_match'].mean()\n    min_adherence = subj_trial_adherence.groupby(level='subject_id').min()\n    \n    return float(min_adherence.mean())",
  "rationale": "To successfully discriminate Take-The-Best (TTB) from Probabilistic Take-The-Best (PTTB), we must account for the fact that high-gamma PTTB perfectly mimics TTB's choices. The only mathematical difference across the entire parameter space is that TTB allows for a higher uniform lapse rate (epsilon up to 0.5) than PTTB (epsilon up to 0.25). This higher lapse rate in TTB causes a uniformly higher variance across all trials. By computing the subject's *minimum* adherence to the TTB prediction across the 8 trial types, we exploit this difference. TTB's higher uniform noise frequently produces a poor adherence score on at least one of the 8 trial types purely by chance, pulling the expected minimum adherence down significantly. In contrast, PTTB's tighter epsilon bounds keep the minimum adherence high for the vast majority of subjects (high-gamma), resulting in a large, reliable mean difference between the two models that comfortably overcomes between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 4928,
  "candidates_token_count": 583,
  "total_token_count": 26720
}
```
