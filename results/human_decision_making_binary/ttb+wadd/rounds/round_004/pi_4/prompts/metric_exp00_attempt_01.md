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

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.05). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=5):** [0.95, 0.87, 0.79, 0.71, 0.63]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Non-linear Subjective Weighting Model (Advocated) from the Threshold-Gated Compensatory Model (Competing) by exploiting the difference between continuous power-function scaling and discrete gap-triggered discounting. We use a 5-feature design with perfectly uniform validity gaps ([0.95, 0.87, 0.79, 0.71, 0.63], gaps of 0.08). Because the gaps are identical, the Threshold-Gated model must either apply no discounts (acting as WADD) or apply a uniform discount factor recursively at every step. Under WADD, Trial 2 is a perfect tie (1.50 vs 1.50) and Trial 1 favors Option B. If the Threshold-Gated model applies its recursive discount, the mathematical structure of the uniform gaps forces a specific trade-off curve where predicting Option A in Trial 1 requires a severe discount, which structurally contradicts the conditions required to predict Option D in Trial 2. By contrast, the Non-linear Subjective Weighting model uses a continuous exponent. For moderate values (e.g., gamma = 4), the non-linear amplification of the highest validities naturally overcomes the sum of lower validities, uniquely allowing the Advocated model to predict a robust preference for Option A and Option C simultaneously—a pattern mathematically impossible under the rigid step-wise discount of the Competing model.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

**Parameters:**
- gamma: [0.0, 10.0]
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
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Threshold-Gated Compensatory Model: Subjects sort cues by their validities and evaluate the numerical gaps between adjacent cues in this sorted order. If the validities are closely clustered, subjects integrate them additively, functioning as a weighted additive model. However, if a gap exceeds a subjective threshold, all subsequent lower-ranked cues are heavily discounted or completely ignored, triggering a shift toward a non-compensatory, semi-lexicographic heuristic. The threshold is sensitive enough to trigger discrete strategy shifts frequently.

**Parameters:**
- gap_threshold: [0.0, 0.3]
- discount_factor: [0.0, 1.0]
- beta: [0.1, 25.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gap_threshold = float(parameters["gap_threshold"])
    discount_factor = float(parameters["discount_factor"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort validities descending to evaluate gaps
    order = np.argsort(-validities)
    sorted_vals = validities[order]
    
    w_sorted = np.zeros_like(sorted_vals)
    if len(sorted_vals) > 0:
        w_sorted[0] = sorted_vals[0]
        current_discount = 1.0
        
        for i in range(1, len(sorted_vals)):
            # If the gap between adjacent sorted validities exceeds the threshold, apply discount
            if (sorted_vals[i-1] - sorted_vals[i]) > gap_threshold:
                current_discount *= discount_factor
            w_sorted[i] = sorted_vals[i] * current_discount
            
    # Map subjective weights back to original feature order
    w = np.zeros_like(validities)
    w[order] = w_sorted
    
    # Option scores are the sum of feature values weighted by the threshold-gated subjective weights
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: In Trial 2, Option A has features 2 and 5 (sum of validities = 0.87 + 0.63 = 1.50) and Option B has features 3 and 4 (sum = 0.79 + 0.71 = 1.50). Under a purely linear additive model, this is a perfect tie. The Advocated model (Non-linear Subjective Weighting) uses a power function on the validities. Because gamma is uniformly distributed in [0, 10], 90% of subjects will have gamma > 1, which strictly convexifies the weights and systematically breaks the tie in favor of Option A. Thus, the Advocated model predicts a very high rate of choosing Option A. The Competing model (Threshold-Gated) only applies discounts if the gap between validities exceeds a threshold. Since all gaps are exactly 0.08 and the threshold is uniformly drawn from [0.0, 0.3], ~73% of subjects will not trigger any discount, reducing the model to a linear additive one that predicts a 50/50 tie. This creates a massive, reliable gap in the expected proportion of Option A choices between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 2: A=[0, 1, 0, 0, 1], B=[0, 0, 1, 1, 0]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 0, 0, 1))
    t2_data = data[is_t2]
    if len(t2_data) == 0:
        return 0.5
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float((t2_data['response'] == 0).mean())
outcome: self_sim=0.6150 (var=0.0170) adversary_sim=0.5617 (var=0.0204) welch_t=+1.379 p=0.1745 (N=25, alpha=0.05) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 1: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0]\n    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 0, 0, 0, 1))\n    t1_data = data[is_t1]\n    if len(t1_data) == 0:\n        return 0.5\n    \n    # Return the proportion of times Option A was chosen (response == 0)\n    return float((t1_data['response'] == 0).mean())",
  "rationale": "In Trial 1, Option A has features 1 and 5 (sum = 0.95 + 0.63 = 1.58) and Option B has features 2 and 3 (sum = 0.87 + 0.79 = 1.66). Under a linear additive model, Option B is strictly preferred. The Advocated model (Non-linear Subjective Weighting) uses a power function; for gamma > ~3.9 (which occurs ~61% of the time since gamma is uniform in [0, 10]), the weight of the highest validity cue (0.95) is amplified enough to make Option A preferred. Thus, the Advocated model predicts a high rate (~60%) of choosing Option A. The Competing model (Threshold-Gated) only discounts if the gap between validities exceeds a threshold. Since all gaps are 0.08 and the threshold is in [0.0, 0.3], ~73% of subjects do not trigger a discount and linearly prefer Option B. Even when a discount is triggered, Option B can still win if the discount factor is weak. Consequently, the Competing model predicts a low rate (~20%) of choosing Option A. This creates a massive, stable gap in the expected proportion of Option A choices."
}
```

## Usage

```json
{
  "prompt_token_count": 3907,
  "candidates_token_count": 474,
  "total_token_count": 6889
}
```
