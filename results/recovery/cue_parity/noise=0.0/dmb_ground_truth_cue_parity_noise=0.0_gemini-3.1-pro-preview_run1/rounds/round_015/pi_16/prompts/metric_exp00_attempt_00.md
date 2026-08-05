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
**Validities (n_features=5):** [0.95, 0.94, 0.93, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Normalized Weighted Sequential Evidence Accumulation) from the Competing Theory (Strategy Selection: Mixture of TTB and Tallying), we exploit trials where TTB and Tallying agree, but a compensatory weighted model with a non-linear scaling of validities disagrees. By using validities [0.95, 0.94, 0.93, 0.60, 0.55], we construct a 'trap' trial where Option A wins the 1st, 4th, and 5th features. The Competing Theory must predict A, as both its TTB component (wins F1) and Tallying component (3 vs 2) favor A. However, the Advocated Theory, which normalizes validities raised to a parameter gamma, can heavily weight the 2nd and 3rd features such that their combined evidence overcomes the 1st, 4th, and 5th, predicting a robust preference for Option B if the stopping threshold is not met immediately.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a Normalized Weighted Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity. They accumulate the differences between the options weighted by the features' validities, which are non-linearly scaled and normalized to sum to 1. This normalization ensures the evidence scale is consistent across experiments with varying numbers of features. If the absolute accumulated evidence reaches or exceeds a specific threshold (bounded between 0 and 1), they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally.

**Parameters:**
- threshold: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Normalize validities raised to gamma so they sum to 1
    weights = (val ** gamma) / np.sum(val ** gamma)
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx] * weights[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Selection based on Environment Classification

**Parameters:**
- dispersion_threshold: [0.0, 0.3]
- slope: [1.0, 100.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Environment classification based on validity dispersion
    dispersion = np.std(val)
    threshold = float(parameters["dispersion_threshold"])
    slope = float(parameters["slope"])
    
    # Probability of selecting Take-The-Best over Tallying
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    diff = a - b
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            ttb_a = 1.0
            break
        elif diff[idx] < 0:
            ttb_b = 1.0
            break
            
    # Tallying prediction
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))\n    if not is_trial_1.any():\n        return 0.0\n    return float(data.loc[is_trial_1, 'response'].mean())",
  "rationale": "This metric calculates the proportion of times subjects choose Option B on the 'trap' trial (Trial 1). On this trial, Option A wins the 1st, 4th, and 5th features, while Option B wins the 2nd and 3rd. The Competing Theory strongly predicts Option A because both its Take-The-Best (TTB) component (which relies on the 1st feature) and its Tallying component (3 vs 2 features) point to A. Thus, the Competing Theory's rate of choosing B will be near zero (driven only by noise/epsilon). In contrast, the Advocated Theory uses a non-linearly scaled weighted evidence accumulation. For certain parameter combinations (e.g., high threshold and high gamma), the combined weight of the 2nd and 3rd features can overpower the 1st, 4th, and 5th features, leading to a substantial probability of choosing Option B. Therefore, the mean response on this specific trial will clearly discriminate the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3437,
  "candidates_token_count": 329,
  "total_token_count": 5544
}
```
