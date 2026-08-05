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
**Validities (n_features=5):** [0.9, 0.89, 0.88, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Strategy Mixture Theory blending TTB, Tallying, and WADD), we exploit the structural limitations of unweighted sequential accumulation. The Advocated Theory maintains a running unweighted tally (+1/-1) and stops if a threshold is reached, or defaults to the final tally if it isn't. By using a steep drop-off in validities (e.g., 0.90, 0.89, 0.88, 0.55, 0.50), we can construct 'trap' trials where Option A wins the 1st, 4th, and 5th features. In this scenario, the unweighted tally starts at +1 (where a low threshold would stop and choose A), drops to 0, then -1, back to 0, and ends at +1 (where a high threshold would finish and choose A). Thus, the Advocated Theory structurally MUST choose A across all possible thresholds. However, Option B wins the 2nd and 3rd features, which combined carry significantly more compensatory weight in the Competing Theory's WADD sub-model than A's features. This creates a sharp dissociation where the Competing Theory can robustly predict B (if its mixture leans towards WADD), a choice mathematically impossible under the Advocated Theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** Strategy Mixture Theory (Environment-Adaptive Toolbox with Tallying Blend)

**Parameters:**
- theta: [0.0, 30.0]
- disp_threshold: [0.0, 0.5]
- gamma: [0.0, 10.0]
- alpha: [0.0, 1.0]
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
    
    # Strategy 1: Take-The-Best (TTB)
    # Inspect features in descending order of validity; stop at the first discriminating feature.
    order = np.argsort(val)[::-1]
    ttb_prob = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_prob = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_prob = np.array([0.0, 1.0])
            break
            
    # Strategy 2: Compensatory (Blend of WADD and Tallying)
    # Evaluate all features, weighting them by a blend of pure Tallying (equal weights) and WADD.
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    centered_val = np.clip(val - 0.5, 1e-6, 0.5)
    
    # Scale validities to [0, 1] based on absolute magnitude, rather than normalizing by max
    scaled_val = centered_val * 2.0
        
    w_wadd = scaled_val ** gamma
    w = alpha * w_wadd + (1.0 - alpha) * 1.0
    
    comp_scores = np.array([np.sum(a * w), np.sum(b * w)])
    
    beta = float(parameters["beta"])
    z = beta * comp_scores
    z -= np.max(z)  # numerical stability
    e = np.exp(z)
    comp_prob = e / np.sum(e)
    
    # Strategy Mixture: Probabilistic selection between TTB and Compensatory based on environment
    # The probability of selecting TTB depends on the dispersion (std dev) of the validities.
    dispersion = float(np.std(val))
    theta = float(parameters["theta"])
    disp_threshold = float(parameters["disp_threshold"])
    
    z_disp = theta * (dispersion - disp_threshold)
    p_ttb = 1.0 / (1.0 + np.exp(-np.clip(z_disp, -20.0, 20.0)))
    
    p_core = p_ttb * ttb_prob + (1.0 - p_ttb) * comp_prob
    
    # Trembling hand error
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # Identify the 'trap' trials\n    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))\n    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))\n    \n    # Measure proportion of choices aligning with the Compensatory strategy\n    # On T1, Compensatory favors B (response 1)\n    # On T2, Compensatory favors A (response 0)\n    t1_choices = data.loc[t1_mask, 'response'] == 1\n    t2_choices = data.loc[t2_mask, 'response'] == 0\n    \n    valid_choices = pd.concat([t1_choices, t2_choices])\n    \n    if len(valid_choices) == 0:\n        return 0.0\n        \n    return float(valid_choices.mean())\n",
  "rationale": "This metric calculates the proportion of choices that align with the Compensatory strategy (WADD) on the specific 'trap' trials identified in the experimental design. On these trials (e.g., A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 0, 0]), the Advocated Theory (Sequential Evidence Accumulation) is structurally constrained to always choose A, because the unweighted tally never drops below 0 and finishes at +1. In contrast, the Competing Theory's compensatory component strongly favors B because the 2nd and 3rd features carry significantly more weight than the 4th and 5th. By measuring the rate of choosing the compensatory option on these specific trials, we isolate the behavior that is mathematically impossible under the Advocated Theory but highly probable under the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3744,
  "candidates_token_count": 523,
  "total_token_count": 7719
}
```
