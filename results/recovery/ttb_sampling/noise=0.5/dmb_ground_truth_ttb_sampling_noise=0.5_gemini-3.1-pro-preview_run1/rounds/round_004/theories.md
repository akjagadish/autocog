# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) is a lexicographic, non-compensatory heuristic. Decision-makers evaluate options by comparing them sequentially on features, starting with the cue that has the highest subjective validity. The first feature that discriminates between the options determines the choice. If all features tie, the decision-maker guesses. Response noise is modeled as an independent lapse rate that occasionally results in a random choice.

**Rationale:** Following the critic's feedback, the Take The Best (TTB) model successfully captures the qualitative preferences across all four experiments but predicts excessively extreme probabilities. To better match the observed human data, which reflects a higher degree of noise/guessing, the upper bound for the lapse rate parameter `epsilon` has been widened from 0.5 to 1.0. This is a minimal diff that preserves the core TTB mechanism exactly as before.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    p_core = np.array([0.5, 0.5])  # Default to guessing if all features tie
    
    # Sequential comparison
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic TTB choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Dual-Process Strategy Mixture: Decision-makers probabilistically alternate between a strict non-compensatory heuristic (Take The Best) and a simple compensatory strategy (Weighted Additive). A mixture parameter dictates the balance, heavily favoring the non-compensatory heuristic to capture strong non-compensatory invariance in most contexts while still accounting for the partial compensatory pull observed in others.

**Rationale:** Following the critic's advice, we adjust the mixture parameter `alpha` to [0.4, 0.95] to preserve a moderately strong TTB dominance while allowing some compensatory influence, and we reduce the upper bound of the WADD inverse temperature `beta` to [0.1, 5.0]. This makes the compensatory component softer and more graded, enabling a better blend of strict TTB preference with a less extreme WADD signal for Experiments 1-4, without sacrificing the fits for Experiments 5-8.

**Parameters:**
  - `alpha`: `[0.4, 0.95]`
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Weighted Additive (WADD) prediction using validities as weights
    scores = stim @ validities
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # Mixture of TTB and WADD
    alpha = float(parameters["alpha"])
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Blend with uniform lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Thresholded Evidence Accumulation: Decision-makers accumulate evidence sequentially across features, ordered by their validity. Each feature contributes evidence proportional to its validity. If the accumulated evidence difference between the options exceeds an internal threshold, search terminates immediately and a choice is made (capturing non-compensatory, Take-The-Best-like behavior when top cues are strong). If all features are exhausted without crossing the threshold, a choice is made probabilistically based on the final accumulated evidence difference (capturing graded compensatory, WADD-like behavior when top cues are weak or conflicting).

**Rationale:** Following the critic's feedback, the threshold parameter bounds have been reduced from [0.0, 3.0] to [0.0, 1.0], and beta from [0.1, 20.0] to [1.0, 10.0]. Because feature validities typically center around 0.5 to 1.0, a threshold up to 3.0 was rarely reached, causing the model to default to compensatory integration too often. By lowering the threshold, the model will more frequently terminate early on highly valid cues, correctly capturing the strong non-compensatory signals observed in experiments 5, 7, 9, and 10.

**Parameters:**
  - `threshold`: `[0.0, 1.0]`
  - `beta`: `[1.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = 0.0
    terminated = False
    for idx in order:
        diff += validities[idx] * (a[idx] - b[idx])
        if abs(diff) >= threshold:
            terminated = True
            break
            
    if terminated:
        if diff > 0:
            p_a = 1.0
        elif diff < 0:
            p_a = 0.0
        else:
            p_a = 0.5
    else:
        # Probabilistic choice based on final accumulated difference
        z = beta * diff
        z = np.clip(z, -500, 500) # Prevent overflow
        p_a = 1.0 / (1.0 + np.exp(-z))
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```
