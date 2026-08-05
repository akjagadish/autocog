# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Raw Validity-Weighted Additive Strategy: Decision-makers do not rely on fast-and-frugal heuristics or non-linear rank transformations. Instead, they compute a simple compensatory weighted sum of the features, where the weights are directly proportional to the raw validities provided in the instructions. This naturally explains the ~0.50 split in conflict trials because the sum of a few high-validity cues roughly equals the sum of several lower-validity cues, bridging the gap between heuristic and rank-based models. A constrained temperature parameter ensures that small differences in these sums translate to intermediate choice probabilities rather than deterministic outcomes.

**Rationale:** Following the arbiter's feedback, this model implements a pure Raw Validity-Weighted Additive Strategy. By using the raw validities directly as weights rather than ordinal ranks or lexicographic rules, the model naturally equates the evidence from a few high-validity cues with the evidence from several lower-validity cues. We constrain the inverse temperature ('beta') to a smaller upper bound (5.0) compared to previous WADD instantiations to ensure that the small score differences on conflict trials correctly produce the observed ~0.50 choice splits, avoiding the over-determinism that caused prior weighted-additive models to fail.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute simple weighted sums using raw validities as weights
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([score_a, score_b])
    
    # Convert scores to probabilities using softmax
    z = beta * scores
    z = z - np.max(z)
    p_core = np.exp(z)
    p_core = p_core / np.sum(p_core)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

**Rationale:** Following the critic's diagnosis, the linear rank-weighting model still leaned too heavily toward WADD in conflict trials because the sum of the top ranks strictly dominated the bottom ranks. To address this, we introduce a non-linear scaling parameter `gamma` applied to the ranks (`weights = rankdata(val) ** gamma`). This minimal change allows the model to smoothly interpolate between Tallying (gamma=0) and steeper rank-weighting, providing the necessary flexibility to exactly match the ~0.50 split in conflict trials while remaining within the prescribed Rank-Weighted Additive family.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Log-Odds Weighted Additive Strategy (Naive Bayes) with Softened Determinism: Decision-makers integrate all features but weight them by the log-odds of their validities (log(v / (1-v))). This non-linear transformation naturally amplifies the importance of highly valid cues over moderate ones, closely approximating Bayesian evidence accumulation. A low-temperature softmax ensures that the large weight differences produced by the log-odds transformation do not lead to hyper-deterministic choices, better matching human probabilistic behavior.

**Rationale:** Following the critic's advice, the maximum value of the softmax inverse temperature parameter 'beta' was further reduced to 0.5, and the maximum lapse rate 'epsilon' was increased to 0.4. Since the log-odds transformation can produce very large weight differences, even a beta of 2.0 resulted in overconfidence on conflict trials. Further constraining beta and allowing a higher noise floor (epsilon) helps the model better capture the ~0.50 choice splits observed in human data on conflict trials.

**Parameters:**
  - `beta`: `[0.01, 0.5]`
  - `epsilon`: `[0.0, 0.4]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Clip validities to prevent division by zero or log(0)
    val = np.clip(val, 0.501, 0.999)
    
    # Compute log-odds weights
    weights = np.log(val / (1.0 - val))
    
    # Calculate weighted sums
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([score_a, score_b])
    
    # Convert scores to probabilities using stable softmax
    z = beta * scores
    z = z - np.max(z)
    p_core = np.exp(z)
    p_core = p_core / np.sum(p_core)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
