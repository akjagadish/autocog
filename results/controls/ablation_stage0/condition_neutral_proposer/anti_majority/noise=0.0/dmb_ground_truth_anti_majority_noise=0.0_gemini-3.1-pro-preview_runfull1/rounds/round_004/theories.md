# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Variance Penalization (Risk Aversion in Cue Integration): Subjects evaluate options by integrating the validities of present features, but they actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. The subjective value of an option is its non-linearly weighted sum of validities minus a penalty proportional to the standard deviation of its active validities, which scales appropriately to strongly influence decisions.

**Rationale:** Following the critic's advice, the penalty term has been changed from variance to standard deviation (`np.std`). Because validities range from 0.5 to 1.0, variance is very small (~0.01), limiting the impact of the penalty. Standard deviation provides a linearly-scaled dispersion measure that can more effectively compete with the weighted sum. Additionally, the upper bound of `lambda_pen` has been increased to 100.0 to allow the risk-aversion component to strongly dominate the choice when necessary, helping to explain the strong deviations observed in Experiments 5, 7, and 8.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `lambda_pen`: `[0.0, 100.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    lambda_pen = float(parameters["lambda_pen"])
    
    def get_score(x):
        active_mask = (x > 0.5)
        if not np.any(active_mask):
            return 0.0
        
        # Weighted sum using exponentiated validities for compensatoriness flexibility
        sum_v = np.sum(val[active_mask] ** gamma)
        
        # Standard deviation penalty using original validities to capture dispersion
        std_v = np.std(val[active_mask]) if np.sum(active_mask) > 1 else 0.0
        
        return sum_v - lambda_pen * std_v

    score_a = get_score(a)
    score_b = get_score(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

**Rationale:** Increased the upper bounds of `gamma` (from 10 to 50) and `beta` (from 20 to 200). Because validities are < 1, exponentiating them by a large gamma creates very small absolute weights. To translate these small score differences into deterministic choices that match the strong empirical preference for TTB on critical trials, the softmax temperature `beta` must be allowed to reach much higher values.

**Parameters:**
  - `beta`: `[0.1, 200.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 50.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Averaging of Active Validities: Subjects evaluate options by computing a weighted combination of the sum and the mean of the validities of the active features, with a strong bias towards the mean. This naturally accounts for strong dilution effects, where adding a low-validity cue to an option with high-validity cues decreases its overall attractiveness because it lowers the average cue quality. The model interpolates between pure additive integration (WADD) and pure averaging, flexibly capturing human behavior across different contexts without requiring an explicit variance penalty.

**Rationale:** To address the underestimation of the dilution effect in Experiments 9 and 10, the range for `w_sum` is restricted to [0.0, 0.2]. This ensures the averaging mechanism dominates the evaluation over strict summation, aligning with the strong empirical preference for subset options with higher average validities. The upper bound of `beta` is also increased to [0.1, 30.0] to allow for sharper, more deterministic choices when average validities differ, improving fit on consensus trials.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `w_sum`: `[0.0, 0.2]`
  - `beta`: `[0.1, 30.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    w_sum = float(parameters["w_sum"])
    
    def get_score(x):
        active_mask = (x > 0.5)
        if not np.any(active_mask):
            return 0.0
        
        v_active = val[active_mask] ** gamma
        return w_sum * np.sum(v_active) + (1.0 - w_sum) * np.mean(v_active)

    score_a = get_score(a)
    score_b = get_score(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
