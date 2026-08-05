# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

**Rationale:** The Take The Best (TTB) heuristic is a non-compensatory, lexicographic decision strategy. We maintain the exact same mechanism as the previous candidate, but adjust the parameter ranges for `beta` (shifted to lower values: [0.01, 5.0]) and `epsilon` (upper bound increased to 1.0). This allows the optimization process to inject enough stochasticity to bridge the gap between pure TTB's extreme predictions (~0.15) and human behavior (~0.30-0.36).

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Bounded Sequential Feature Sampling: Subjects evaluate features in decreasing order of validity, accumulating unweighted evidence (the net difference in feature values) until a low internal threshold is reached. If the threshold is met, search terminates and a choice is made; if all features are exhausted, the choice is based on whatever net evidence was accumulated. This generalizes 'Take The Best' by allowing for a 'Take The Two' or similar heuristic where more than one discriminating feature may be required to reach a decision.

**Rationale:** Following the critic's advice, we revert to the unweighted evidence accumulation mechanism since validity-weighting worsened the fit. To improve the balance between TTB and tallying behavior, we expand the threshold to {1, 2, 3}, allowing the model to smoothly transition from pure Take-The-Best (threshold=1) to 'Take The Two' or simple tallying. Furthermore, we increase the beta range to [5.0, 20.0] to sharpen softmax probabilities and ensure deterministic choices when the threshold is met.

**Parameters:**
  - `threshold`: `{1, 2, 3}`
  - `beta`: `[5.0, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    threshold = int(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    E = 0.0
    # Accumulate evidence until threshold is met
    for idx in order:
        E += (a[idx] - b[idx])
        if abs(E) >= threshold:
            break
            
    # Scores based on accumulated evidence
    scores = np.array([E, -E])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Exponentially Weighted Additive Model: Subjects integrate all available features in a compensatory manner, but they apply a steep non-linear (exponential) transformation to the feature validities. This causes the most valid features to heavily dominate the decision, effectively mimicking the non-compensatory 'Take The Best' heuristic while remaining mathematically compensatory. The steepness of this transformation dictates how closely the strategy approximates strict lexicographic choice.

**Rationale:** Following the critic's advice, we adjust the bounds of `gamma` to [0.5, 20.0] and `beta` to [0.05, 10.0]. This provides a softer lower bound for gamma than 1.0, allowing mildly more compensatory weighting to better fit Tallying-like behavior when needed, while strictly avoiding 0.0 to preserve the core lexicographic approximation that drives the strong performance on TTB-consistent metrics.

**Parameters:**
  - `gamma`: `[0.5, 20.0]`
  - `beta`: `[0.05, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Steep non-linear transformation of validities to weights
    weights = np.exp(gamma * validities)
    
    # Calculate option scores as weighted sums
    scores = stim @ weights
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```
