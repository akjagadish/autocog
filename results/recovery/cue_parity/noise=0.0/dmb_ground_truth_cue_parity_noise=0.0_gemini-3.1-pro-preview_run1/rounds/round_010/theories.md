# Round 10 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_12` — KILLED ✗

**Description:** Two-Phase Heuristic (Take-The-Best with Tallying Fallback): Decision-makers first evaluate the single most valid feature. If it discriminates between options, they make a choice immediately based on it, with the determinism of this choice modulated by a specific weight. If the most valid feature is tied, they abandon sequential lexicographic evaluation and instead perform an unweighted Tally of all remaining features to break the tie.

**Rationale:** Following the critic's advice, we introduce a `ttb_weight` parameter to scale the scores in Phase 1 (the Take-The-Best phase). This decouples the intensity and determinism of the first phase from the Tallying phase, softening the extreme predictions in experiments where TTB points in the opposite direction of human behavior, without altering the core Two-Phase mechanism prescribed by the arbiter.

**Parameters:**
  - `beta`: `[0.1, 15.0]`
  - `ttb_weight`: `[0.1, 10.0]`
  - `tally_scale`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the most valid feature
    best_idx = np.argmax(val)
    diff_best = a[best_idx] - b[best_idx]
    
    if diff_best != 0:
        # Phase 1: The most valid feature discriminates
        ttb_weight = float(parameters["ttb_weight"])
        scores = ttb_weight * np.array([diff_best, -diff_best])
    else:
        # Phase 2: The most valid feature is tied; Tally the remaining features
        mask = np.ones(len(a), dtype=bool)
        mask[best_idx] = False
        tally_a = np.sum(a[mask])
        tally_b = np.sum(b[mask])
        
        tally_scale = float(parameters["tally_scale"])
        scores = tally_scale * np.array([tally_a, tally_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_9` — SURVIVED ✓

**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Rationale:** Following the critic's advice, we implement a Sequential Evidence Accumulation model where features are sampled in descending order of validity. By allowing the stopping threshold to range from 0.0 to 3.0, the model can dynamically recover Take-The-Best behavior (when threshold <= 1.0, stopping after the first differentiating feature) or Tallying behavior (when threshold is high, accumulating across all features). This single unified mechanism avoids the scaling issues of the mixture model and naturally explains intermediate choice proportions observed in the experiments.

**Parameters:**
  - `threshold`: `[0.0, 3.0]`
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_13` → slot 1 (via `new_theory`)

**Description:** Decision-makers use a purely compensatory Weighted Additive Strategy (WADD). Instead of applying early stopping or lexicographic rules, individuals evaluate all available features for both options. Each feature is weighted by a non-linear transformation of its objective validity, capturing varying sensitivities to evidence strength. The overall subjective value of an option is the sum of its weighted features, and choices are made probabilistically based on the difference in these overall values. By allowing for very large scaling parameters, the model can approximate non-compensatory behavior within a purely compensatory framework.

**Rationale:** Following the critic's feedback, we revert to the successful unnormalized chance-centered power-law WADD base from Iteration 1. To improve fit on experiments where human behavior appears lexicographic, we significantly expand the parameter ranges for `gamma` (up to 25.0) and `beta` (up to 100.0). This minimal edit allows the purely compensatory WADD model to naturally approximate non-compensatory behavior by letting the most valid cues dominate the weighted sum when `gamma` is large, without altering the core mechanism.

**Parameters:**
  - `beta`: `[0.1, 100.0]`
  - `gamma`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transformation parameter for validities
    gamma = float(parameters["gamma"])
    
    # Center validities to chance level (0.5), then apply non-linear scaling
    centered_val = np.clip(val - 0.5, 1e-6, 0.5)
    weights = centered_val ** gamma
    
    # Calculate overall subjective values (weighted sums)
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with numerical stability
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
