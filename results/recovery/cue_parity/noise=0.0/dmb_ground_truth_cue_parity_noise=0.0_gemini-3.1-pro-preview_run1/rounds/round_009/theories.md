# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

**Description:** Decision-makers use a Validity-Weighted Sequential Accumulation strategy. They inspect features one by one in descending order of validity, accumulating evidence proportional to a non-linear transformation of each feature's chance-centered validity. If the absolute value of this weighted accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice. Otherwise, they evaluate all features and decide based on the final weighted tally. The weight transformation uses the absolute chance-centered validity raised to a power, preserving the sign, to avoid numerical errors while allowing interpolation between Tallying and Take-The-Best.

**Rationale:** Applying the minimal diff requested by the critic: the weight calculation is updated to `np.sign(centered_val) * (np.abs(centered_val) ** gamma)` to safely apply the power transformation to chance-centered validities without raising negative numbers to float powers (which causes NaNs). This preserves the arbiter's intended Validity-Weighted Sequential Accumulation mechanism while fixing the numerical bug that caused the previous iteration's rejection.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 5.0]`
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
    
    # Parameters
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into weights safely
    centered_val = val - 0.5
    weights = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx] * weights[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
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

### `pi_12` → slot 1 (via `new_theory`)

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
