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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Weighted Sequential Evidence Accumulation with Leaky Integration) from the Competing Theory (Unweighted Sequential Evidence Accumulation), we exploit the Advocated Theory's 'leak' parameter, which allows it to decay the influence of previously evaluated features. By using a set of validities with a tight cluster at the top (0.95, 0.90, 0.85), we construct 'trap' trials where Option B wins the 1st, 4th, and 5th features, while Option A wins the 2nd and 3rd. Under the Competing Theory's unweighted accumulation, the running tally for A starts at -1, goes to 0, +1, 0, and ends at -1. Because the tally never reaches +2, any threshold > 1 will result in the model evaluating to the end and choosing B (tally = -1). If the threshold is 1, it stops at the first feature and also chooses B. Thus, the Competing Theory structurally MUST predict Option B. In contrast, the Advocated Theory with a positive leak (decay) will heavily discount the 1st feature by the time it evaluates the 2nd and 3rd features. The combined validity-weighted evidence of the 2nd and 3rd features will overpower the decayed 1st feature, allowing the accumulator to hit a positive threshold and robustly predict Option A. This creates a stark double dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Sequential Evidence Accumulation with Self-Excitation/Decay (Leaky Accumulator). Decision-makers process information sequentially in descending order of feature validity. Each feature's evidence is weighted by its validity and added to a running accumulator. Crucially, previously accumulated evidence can either 'leak' (decay) or self-excite (amplify) as new features are evaluated. A choice is made either when the accumulator hits a predefined threshold or when all features are exhausted. Self-excitation allows the model to capture strong primacy effects even when the decision threshold is not reached.

**Parameters:**
- threshold: [0.0, 5.0]
- leak: [-1.0, 1.0]
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
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
    
    threshold = float(parameters["threshold"])
    leak = float(parameters["leak"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Evaluate features in descending order of validity
    order = np.argsort(val)[::-1]
    
    A = 0.0
    for idx in order:
        diff = a[idx] - b[idx]
        weight = val[idx] ** gamma
        
        # Leaky accumulation (or self-excitation if leak < 0)
        A = A * (1.0 - leak) + weight * diff
        
        # Stopping rule
        if abs(A) >= threshold and abs(A) > 1e-9:
            break
            
    scores = np.array([A, -A])
    
    # Softmax conversion to probabilities
    z = beta * scores
    z -= np.max(z)  # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Identify Trial 1: Option B wins 1st, 4th, 5th; Option A wins 2nd, 3rd\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1))\n              \n    # Identify Trial 2: Option A wins 1st, 4th, 5th; Option B wins 2nd, 3rd\n    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))\n              \n    # Advocated theory predicts choosing the option that wins the 2nd and 3rd features\n    # because the 1st feature's impact leaks (decays) away.\n    # Competing theory predicts choosing the option that wins the 1st feature.\n    \n    # Trial 1: Advocated predicts Option A (response == 0)\n    t1_adv_rate = (data.loc[t1_mask, 'response'] == 0).mean()\n    \n    # Trial 2: Advocated predicts Option B (response == 1)\n    t2_adv_rate = (data.loc[t2_mask, 'response'] == 1).mean()\n    \n    rates = []\n    if not np.isnan(t1_adv_rate):\n        rates.append(t1_adv_rate)\n    if not np.isnan(t2_adv_rate):\n        rates.append(t2_adv_rate)\n        \n    if not rates:\n        return 0.5\n        \n    return float(np.mean(rates))",
  "rationale": "In Trial 1 and Trial 2, one option wins the 1st, 4th, and 5th features, while the other wins the 2nd and 3rd. Under the Competing Theory's unweighted sequential accumulation, the tally for the option winning the 2nd and 3rd features never exceeds +1 and ends at -1. Thus, it structurally must predict the option that wins the 1st feature, regardless of threshold (as long as it evaluates the 1st). In contrast, the Advocated Theory with a positive leak discounts the 1st feature by the time the 2nd and 3rd are evaluated, allowing the combined weight of the 2nd and 3rd features to dominate and cross a positive threshold. The metric computes the rate at which choices align with the Advocated Theory's unique prediction."
}
```

## Usage

```json
{
  "prompt_token_count": 3420,
  "candidates_token_count": 699,
  "total_token_count": 6487
}
```
