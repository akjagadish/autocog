# Round 17 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_18` — KILLED ✗

**Description:** Parallel Similarity and Feature-Difference Model: Decision-makers evaluate options in parallel by computing a similarity-weighted score. Instead of a discrete stopping rule, the strategy smoothly transitions between Tallying (equal weighting) and a heavily weighted linear model (WADD) based on the overall similarity of the options. A Jaccard-like similarity metric is used. To ensure stability, the feature weights for both strategies are normalized to sum to 1.0, and the softmax temperature parameter is expanded to accommodate the normalized scale, allowing for highly deterministic choice behavior.

**Rationale:** To address the critic's feedback, we have modified the base (Iteration 2) by explicitly normalizing the Tallying and WADD weight vectors to sum to 1.0 before interpolating between them. This prevents the score variance from artificially shifting due to scale differences between the strategies. To compensate for the resulting compression of the score scale, the upper bound of the `beta` parameter has been expanded from 10.0 to 50.0, restoring the model's ability to fit highly deterministic human choice patterns.

**Parameters:**
  - `sim_threshold`: `[0.0, 1.0]`
  - `slope`: `[-50.0, 50.0]`
  - `gamma`: `[0.1, 10.0]`
  - `beta`: `[0.1, 50.0]`
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
    
    # Calculate similarity using Jaccard index (matching presences)
    a_bool = a > 0.5
    b_bool = b > 0.5
    intersection = np.sum(a_bool & b_bool)
    union = np.sum(a_bool | b_bool)
    sim = float(intersection) / float(union) if union > 0 else 1.0
    
    threshold = float(parameters["sim_threshold"])
    slope = float(parameters["slope"])
    
    # Logistic transition function:
    # With slope in [-50, 50], optimization can determine if high similarity 
    # leads to Tallying (alpha->1) or WADD (alpha->0).
    z_alpha = -slope * (sim - threshold)
    z_alpha = np.clip(z_alpha, -100, 100)
    alpha = 1.0 / (1.0 + np.exp(z_alpha))
    
    gamma = float(parameters["gamma"])
    # Non-linear scaling of validities for the WADD component, normalized to sum to 1
    w_wadd_raw = val ** gamma
    w_wadd = w_wadd_raw / np.sum(w_wadd_raw)
    
    # Tallying weights, normalized to sum to 1
    w_tally = np.ones_like(val) / len(val)
    
    # Interpolate feature weights
    weights = alpha * w_tally + (1.0 - alpha) * w_wadd
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_19` — SURVIVED ✓

**Description:** Decision-makers use a dynamic sequential evidence accumulation process with a fixed absolute threshold. They evaluate features in decreasing order of validity, maintaining a running sum of the validity-weighted differences between options. After each feature, they compare the accumulated absolute evidence to a fixed threshold. If the evidence meets or exceeds this threshold, they stop and make a choice. This dynamically adapts the amount of information processed: highly conflicting early evidence keeps the accumulated difference near zero, prompting the integration of more features (similar to WADD or Tallying), whereas decisive early evidence leads to quick termination (similar to Take-The-Best).

**Rationale:** Following the critic's advice, the dynamic stopping rule was modified to use a fixed absolute threshold (`abs(e) >= threshold`) rather than scaling by the remaining possible evidence. The previous `max_rem` scaling caused the threshold to drop artificially when tied features were encountered, triggering premature stopping even when no new evidence was gathered. Using a fixed threshold ensures that early stopping only occurs when sufficient decisive evidence has actually been accumulated, properly transitioning between TTB-like and WADD-like behavior depending on the level of early conflict. The parameter `threshold` range is expanded to [0.0, 5.0] to cover the maximum possible sum of validities.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
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
    
    e = 0.0
    for idx in order:
        e += val[idx] * (a[idx] - b[idx])
        
        # Dynamic stopping rule based on fixed absolute evidence threshold
        if abs(e) >= threshold and abs(e) > 0:
            break
            
    scores = np.array([e, -e])
    
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

### `pi_20` → slot 1 (via `new_theory`)

**Description:** Non-linear Feature Difference Model with Flexible Validity Centering

**Rationale:** The critic's suggestion in iteration 5 (V_A = net_A if net_A > 0 else lambda_loss * net_A) is mathematically degenerate for a binary choice softmax, as the difference V_A - V_B simplifies to net_A * (1 + lambda_loss), which is exactly equivalent to the linear model with a scaled beta. It provides no new non-linear representational power. To genuinely improve upon the accepted iteration 3 base without introducing mathematical degeneracies, I am retaining the successful `gamma` power function (which properly compresses/expands the evidence difference) but introducing a flexible validity centering parameter `theta` instead of hardcoding it to 0.5. This allows the model to smoothly interpolate between pure Tallying and strict validity weighting, giving it the flexibility needed to fix the miscalibrations in Exps 13, 20, and 22.

**Parameters:**
  - `theta`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 20.0]`
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
    
    # Center validities with a flexible threshold rather than hardcoded 0.5
    theta = float(parameters["theta"])
    w = val - theta
    
    diff = a - b
    # Compute net evidence for option A over B
    net_ev = np.sum(w * diff)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply non-linear value function to the net evidence (Prospect Theory style)
    score = np.sign(net_ev) * (np.abs(net_ev) ** gamma)
    
    # Create scores for A and B
    scores = np.array([score, -score])
    
    # Softmax choice rule
    z = beta * scores
    z -= np.max(z) # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
