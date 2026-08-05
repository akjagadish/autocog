# Round 18 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_20` — SURVIVED ✓

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


### slot 2 — `pi_19` — KILLED ✗

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

### `pi_21` → slot 2 (via `new_theory`)

**Description:** Decision-makers evaluate options using a Leaky Competing Accumulator (LCA) process, where attention shifts sequentially across features in order of their validity. Evidence for each option accumulates continuously over time, subject to information decay (leakage) and lateral inhibition. When lateral inhibition is strong, early evidence from highly valid features quickly suppresses the competing option, locking in a choice and producing non-compensatory 'Take-The-Best' behavior. Conversely, when leakage and inhibition are low, evidence from all features integrates more evenly without suppression, resulting in compensatory 'Tallying' or WADD-like behavior. This provides a unified mechanistic account for the spectrum of decision strategies without requiring explicit rule-switching.

**Rationale:** Following the arbiter's feedback, this theory abandons strict sequential heuristics and simultaneous weighted sums in favor of a Leaky Competing Accumulator (LCA) framework. Features are attended to sequentially in order of their validity, providing inputs to two competing accumulators representing the options. The dynamics of accumulation are governed by leakage (decay of past evidence) and lateral inhibition (suppression of the weaker option). This naturally bridges the gap between compensatory and non-compensatory behavior: strong lateral inhibition causes early, highly valid features to establish a dominant activation that suppresses subsequent evidence (mimicking 'Take-The-Best'), while low inhibition allows all features to be integrated evenly (mimicking 'Tallying' or WADD).

**Parameters:**
  - `leak`: `[0.0, 2.0]`
  - `inhibition`: `[0.0, 5.0]`
  - `steps_per_feature`: `[1, 50]`
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
    
    # Sort features by validity (descending) to simulate sequential attention
    order = np.argsort(val)[::-1]
    
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    steps_per_feature = int(float(parameters["steps_per_feature"]))
    gamma = float(parameters["gamma"])
    
    x_A, x_B = 0.0, 0.0
    dt = 0.1
    
    for idx in order:
        # Non-linear scaling of validities centered at chance
        v = max(0.0, val[idx] - 0.5) ** gamma
        I_A = a[idx] * v
        I_B = b[idx] * v
        
        # Accumulate evidence over time for the current feature
        for _ in range(steps_per_feature):
            dx_A = (I_A - leak * x_A - inhibition * x_B) * dt
            dx_B = (I_B - leak * x_B - inhibition * x_A) * dt
            
            # Rectified linear units (activations cannot be negative)
            x_A = max(0.0, x_A + dx_A)
            x_B = max(0.0, x_B + dx_B)
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([x_A, x_B])
    
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
