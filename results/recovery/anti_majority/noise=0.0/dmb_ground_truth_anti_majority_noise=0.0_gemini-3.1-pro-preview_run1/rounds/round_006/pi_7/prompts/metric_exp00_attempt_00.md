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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Anchor and Adjust model from the Variance Aversion model, we systematically manipulate the single highest-validity feature (the anchor) while keeping the remaining features identical across the two options. The Anchor and Adjust model evaluates an option by taking its maximum validity feature and adjusting it based on the others; thus, if Option A and Option B share the exact same set of lower-validity features but Option A has a higher anchor, Anchor and Adjust will universally prefer Option A, as improving the anchor strictly increases the evaluation. In contrast, the Variance Aversion model computes the mean of the validities and applies a penalty proportional to their variance. By increasing the highest validity feature while holding the lower ones constant, the variance of the feature set increases. If the variance penalty parameter is sufficiently high, the Variance Aversion model will paradoxically prefer Option B (the option with the lower maximum validity) because its features are more tightly clustered, leading to a smaller configural penalty. This provides a sharp qualitative divergence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Anchor and Adjust: Decision-makers evaluate options by first identifying the highest-validity positive feature, which serves as an anchor. Subsequent positive features adjust this anchored evaluation downwards if their validities are lower than the anchor. This explains the 'less is more' dilution effect, as adding low-validity features drags the overall evaluation down, while naturally predicting that an option with only high-validity features will be preferred over one that adds additional low-validity features (strict subset preference).

**Parameters:**
- gamma: [0.1, 10.0]
- alpha: [0.0, 50.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # The highest validity positive feature is the anchor
        anchor = np.max(active_w)
        idx = np.argmax(active_w)
        
        # Remaining features adjust the evaluation
        other_w = np.delete(active_w, idx)
        if len(other_w) == 0:
            return anchor
            
        # Weighted average of anchor and adjusting features
        return (anchor + alpha * np.sum(other_w)) / (1.0 + alpha * len(other_w))
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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


## COMPETING THEORY
**Description:** Variance Aversion (Configural Weighting): Decision-makers evaluate options by averaging the subjective validities of their present features, but they apply a configural penalty proportional to the variance of those validities. This mechanism reflects a preference for 'holistic consistency'—options with tightly clustered, moderately high features are preferred over options with a mix of very high and very low features, as the latter are penalized for their inconsistency.

**Parameters:**
- gamma: [0.1, 10.0]
- lambda_var: [0.0, 20.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        var_w = np.var(active_w) if len(active_w) > 1 else 0.0
        
        # Average evidence penalized by the variance of the active validities
        return mean_w - lambda_var * var_w
        
    score_a = get_score(a)
    score_b = get_score(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Filter for trials where both options have the same number of positive features\n    # In the design, these are trials 1-6. In all of these, Option A has a higher \n    # maximum validity (anchor) but exactly the same remaining features as Option B.\n    # Therefore, Option B's features are more clustered (lower variance).\n    is_matched = data['option_a_ratings'].apply(sum) == data['option_b_ratings'].apply(sum)\n    matched_data = data[is_matched]\n    if len(matched_data) == 0:\n        return 0.0\n    # Return the proportion of times Option B was chosen\n    return float(matched_data['response'].mean())",
  "rationale": "By isolating trials where both options have the same number of features (trials 1-6), we zero in on the core dissociation between the models. In these trials, Option A always possesses the highest-validity feature (the best anchor), but Option B's features are more tightly clustered (lower variance). The Anchor and Adjust model will strongly and universally prefer Option A because improving the anchor strictly increases the evaluation, driving the metric (proportion of Option B choices) close to 0. In contrast, the Variance Aversion model penalizes Option A for its higher feature variance, driving it to frequently prefer Option B and resulting in a significantly higher metric value."
}
```

## Usage

```json
{
  "prompt_token_count": 3581,
  "candidates_token_count": 330,
  "total_token_count": 6091
}
```
