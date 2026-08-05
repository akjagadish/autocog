# Round 12 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_14` — SURVIVED ✓

**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Rationale:** Following the critic's advice, we adjust the weight transformation to `weights = np.maximum(val - 0.5, 0.001) ** gamma`. Since validities are typically in the [0.5, 1.0] range, transforming them relative to chance (0.5) before applying the exponent provides much better contrast between high and low validity cues. This allows the model to flexibly flatten or steepen the weights. We also expand the range of `theta` to [0.0, 10.0] to allow the model to avoid early stopping and evaluate all cues when compensatory tallying behavior is required.

**Parameters:**
  - `theta`: `[0.0, 10.0]`
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_11` — KILLED ✗

**Description:** Weighted Additive (WADD) Integration with Zero-Anchored Soft Validity Transformation: Decision-makers compute a subjective value for each option by summing its features, weighted by a zero-anchored exponential transformation of their validities. This transformation (exp(gamma * val) - 1) ensures that non-predictive cues receive no weight, preventing the artificial inflation of tallies by low-validity cues while allowing the highest validity cues to exponentially dominate when necessary. This naturally bridges compensatory and non-compensatory decision-making without heuristic switching.

**Rationale:** Following the critic's advice, we modified the exponential weight transformation to `w = np.exp(gamma * val) - 1.0`. The previous `np.exp(gamma * val)` assigned a baseline weight of 1.0 even to zero-validity cues, which artificially inflated the subjective value of options with many low-validity cues and made the model too compensatory (failing on heavily non-compensatory experiments like 13 and 15). By subtracting 1.0, we anchor the transformation at zero, ensuring that low validities vanish correctly while high validities can still scale exponentially to dominate when needed. This preserves the continuous WADD integration while better capturing TTB-like human behavior.

**Parameters:**
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective transformation of validities
    # Subtracting 1.0 ensures that a zero-validity cue would receive exactly 0 weight,
    # preventing artificial inflation of low-validity cues and allowing the highest
    # validity cues to dominate when necessary.
    w = np.exp(gamma * val) - 1.0
    
    # Compute subjective values (Weighted Additive sum)
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    scores = np.array([v_a, v_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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

### `pi_15` → slot 2 (via `new_theory`)

**Description:** Adaptive Environment-Structured Strategy Selection: Decision-makers adapt their choice strategy based on the global statistical structure of the environment. Specifically, they evaluate the dispersion (e.g., standard deviation) of cue validities. In environments with high dispersion, they probabilistically favor a non-compensatory heuristic (Take-The-Best). In environments with low dispersion, they fall back to a simpler compensatory heuristic (Equal-weights Tallying). The mixture between these strategies is dynamically governed by a logistic function of the environment's validity dispersion, with independent confidence (temperature) parameters for each strategy.

**Rationale:** Following the critic's advice, this edit restores the normalized tallying score from the previously accepted iteration, which scales Tallying differences to the same [-1, 1] bounds as TTB. To address the issue where Tallying predictions were too flat (due to smaller absolute score differences after normalization), we introduce separate inverse temperature parameters (`beta_ttb` and `beta_tally`). This allows the model to independently calibrate the confidence of the TTB and Tallying components without fundamentally altering the scale of the scores themselves, enabling sharp deterministic choices when Tallying dominates.

**Parameters:**
  - `theta`: `[0.0, 0.3]`
  - `k`: `[1.0, 100.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate dispersion of validities in the environment
    dispersion = float(np.std(val))
    
    # Logistic function for strategy weight
    # Higher dispersion -> higher probability of using Take-The-Best
    w_ttb = 1.0 / (1.0 + np.exp(-k * (dispersion - theta)))
    
    # Take-The-Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_diff = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            ttb_diff = 1.0
            break
        elif b[j] > a[j]:
            ttb_diff = -1.0
            break
            
    scores_ttb = np.array([ttb_diff, -ttb_diff])
    
    # Tallying component
    # Normalized by the number of cues to ensure it's on a comparable scale to TTB (-1 to 1)
    tally_diff = (np.sum(a) - np.sum(b)) / len(val)
    scores_tally = np.array([tally_diff, -tally_diff])
    
    # Convert scores to probabilities
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    p_ttb = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Mixture prediction
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
