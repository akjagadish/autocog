# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Attention-Sharing (Fractional Discounting): Subjects evaluate options by linearly summing the validities of present features, but the overall value is discounted by the number of active cues. This reflects a cognitive bottleneck where attention must be divided among multiple features. The subjective value of an option is its sum of validities divided by `1.0 + kappa * (n_cues - 1)`. This mechanism allows a single highly valid cue to maintain its full weight and potentially dominate an option with multiple weaker cues that dilute each other's impact, explaining violations of tallying dominance while remaining mathematically stable.

**Rationale:** Following the critic's advice, we replaced the unstable quadratic configural penalty with a simpler and more robust 'Attention-Sharing' (Fractional Discounting) mechanism. The subjective score of an option is now its linear sum of present validities divided by a discounting factor that grows linearly with the number of present cues (`1.0 + kappa * (n_cues - 1)`). This ensures that the first cue retains its full weight, but additional cues dilute the overall value. This elegantly explains why a single strong cue can beat multiple weaker cues (Exp 1) while avoiding the erratic choice probabilities caused by the previous quadratic interaction term. We also constrained `epsilon` to [0.0, 0.1] to prevent the predictions from washing out to 0.5.

**Parameters:**
  - `kappa`: `[0.0, 2.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
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
    kappa = float(parameters["kappa"])
    
    def get_score(x):
        val_x = val * x
        n_cues = np.sum(x)
        if n_cues == 0:
            return 0.0
        linear = np.sum(val_x)
        return linear / (1.0 + kappa * (n_cues - 1.0))

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


### slot 2 — `pi_3` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
