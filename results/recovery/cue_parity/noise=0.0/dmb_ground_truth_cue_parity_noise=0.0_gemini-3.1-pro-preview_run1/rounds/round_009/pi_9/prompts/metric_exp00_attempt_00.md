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

**Trial pairs (n=4):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Unweighted Sequential Evidence Accumulation) from the Competing Theory (Validity-Weighted Sequential Accumulation), we exploit the difference between unweighted (+1/-1) and weighted evidence accumulation. By setting validities with a tight high cluster (0.95, 0.94, 0.93) and a drop-off (0.60, 0.55), we can construct trials where the unweighted running tally fluctuates but structurally forces the Advocated Theory to predict one option regardless of its threshold, whereas the Competing Theory's weighted sum overcomes the early deficit and predicts the opposite option for any non-minimal threshold.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
- beta: [0.1, 15.0]
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
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
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
**Description:** Decision-makers use a Validity-Weighted Sequential Accumulation strategy. They inspect features one by one in descending order of validity, accumulating evidence proportional to a non-linear transformation of each feature's chance-centered validity. If the absolute value of this weighted accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice. Otherwise, they evaluate all features and decide based on the final weighted tally. The weight transformation uses the absolute chance-centered validity raised to a power, preserving the sign, to avoid numerical errors while allowing interpolation between Tallying and Take-The-Best.

**Parameters:**
- threshold: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
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
    
    # Parameters
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into weights safely
    centered_val = val - 0.5
    weights = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx] * weights[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    # Softmax with max-subtraction for numerical stability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_vals = data['option_a_ratings'].apply(tuple)\n    t1 = a_vals == (0, 1, 1, 0, 0)\n    t2 = a_vals == (1, 0, 0, 1, 1)\n    t3 = a_vals == (0, 1, 0, 1, 1)\n    t4 = a_vals == (1, 0, 1, 0, 0)\n    \n    m1 = data.loc[t1, 'response'].mean() if t1.any() else 0.5\n    m2 = data.loc[t2, 'response'].mean() if t2.any() else 0.5\n    m3 = data.loc[t3, 'response'].mean() if t3.any() else 0.5\n    m4 = data.loc[t4, 'response'].mean() if t4.any() else 0.5\n    \n    return float((m3 - m4) - (m1 - m2))",
  "rationale": "The metric computes a double difference in the choice proportions of option B across the four trial types. For the Advocated Theory (Unweighted), Trials 1 and 2 structurally force choices of B and A respectively regardless of threshold, making (m1 - m2) strongly positive (~1). Meanwhile, its choices on Trials 3 and 4 depend heavily on the threshold, averaging out across subjects. For the Competing Theory (Weighted), the opposite is true: its choices on Trials 3 and 4 are structurally forced to B and A respectively because the heavy validities align, making (m3 - m4) strongly positive (~1), while Trials 1 and 2 are threshold-dependent. The double difference thus yields opposite signs for the two theories, ensuring strong discrimination."
}
```

## Usage

```json
{
  "prompt_token_count": 3172,
  "candidates_token_count": 443,
  "total_token_count": 7671
}
```
