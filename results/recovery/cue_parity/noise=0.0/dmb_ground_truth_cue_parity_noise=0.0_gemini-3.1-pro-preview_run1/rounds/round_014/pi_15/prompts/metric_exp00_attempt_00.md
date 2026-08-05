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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Top-K Majority Heuristic) from the Competing Theory (Rank-Weighted Additive with Baseline Feature Decay), we exploit the structural differences between unweighted majority voting within a fixed subset and a compensatory weighted sum over all features. By presenting pairs where one option holds a majority among the top 2 or 3 features, while the other option wins the highest validity feature and accumulates lower-tier features, we create diametrically opposed predictions. The Top-K model enforces a strict cutoff and evaluates unweighted tallies within that window, whereas the Rank-Weighted Additive model applies exponentially decaying weights and a baseline, allowing combinations of the 1st and lower-ranked features to outweigh the 2nd and 3rd features, or vice versa depending on the decay parameter.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Top-K Majority Heuristic with Proportional Confidence: Decision-makers evaluate options by considering a subset of the most valid features. They identify the top K most valid features and perform an unweighted tally within this set. If tied, they may expand the set. Their confidence in the choice scales with the proportional majority margin (vote difference divided by K), meaning a given vote margin yields higher confidence in smaller consideration sets.

**Parameters:**
- k_initial: [1.0, 10.0]
- expand_on_tie: [0.0, 1.0]
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
    
    # Determine initial consideration set size K
    k_init = int(round(float(parameters["k_initial"]))) # e.g., 3 or 5
    k = min(k_init, len(val))
    k = max(1, k)
    
    expand_on_tie = float(parameters["expand_on_tie"]) > 0.5
    
    while True:
        top_k_idx = order[:k]
        a_wins = np.sum(a[top_k_idx] > b[top_k_idx])
        b_wins = np.sum(b[top_k_idx] > a[top_k_idx])
        
        # Stop if there's a strict majority winner within top K
        if a_wins != b_wins:
            break
            
        # If tied, either expand K or accept the tie (and guess)
        if expand_on_tie and k < len(val):
            k += 1
        else:
            break
            
    diff = float(a_wins - b_wins) / k
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the vote difference
    scores = np.array([diff, -diff])
    z = beta * scores
    z -= np.max(z)  # numerical stability
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
**Description:** Rank-Weighted Additive with Baseline Feature Decay: Decision-makers evaluate options by considering all features simultaneously, applying a non-linear, rank-dependent exponential decay to the feature weights. To prevent lower-ranked features from losing all influence, a baseline weight is added to the decayed value. This ensures that while top-ranked features dominate, a sufficient number of aligned lower-ranked features can still compensate and influence the final choice.

**Parameters:**
- lambda_decay: [0.0, 1.0]
- baseline_weight: [0.0, 1.0]
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
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    lambda_decay = float(parameters["lambda_decay"])
    baseline_weight = float(parameters["baseline_weight"])
    
    # Assign exponentially decaying weights based on validity rank plus a baseline
    weights = np.zeros_like(val)
    for i, idx in enumerate(order):
        weights[idx] = (lambda_decay ** i) + baseline_weight
        
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    diff = score_a - score_b
    scores = np.array([diff, -diff])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the score difference
    z = beta * scores
    z -= np.max(z)  # Max-subtraction for numerical stability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_has_1 = data['option_a_ratings'].apply(lambda x: x[0] == 1)\n    chose_1st = ((data['response'] == 0) & a_has_1) | ((data['response'] == 1) & (~a_has_1))\n    return float(chose_1st.mean())",
  "rationale": "Calculates the proportion of trials where the subject chooses the option possessing the most valid feature (feature 1). The Rank-Weighted Additive model with exponential decay strongly privileges the highest-ranked feature, often making it dominant over all others combined. In contrast, the Top-K Majority model tallies unweighted votes within the top K features, so as K increases (which it frequently does given the parameter space), the top feature's influence is heavily diluted and easily outvoted by lower-ranked features. This creates a stark divergence in how often the 1st feature dictates the final choice."
}
```

## Usage

```json
{
  "prompt_token_count": 3450,
  "candidates_token_count": 234,
  "total_token_count": 8238
}
```
