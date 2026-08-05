# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_10` — KILLED ✗

**Description:** Threshold-Gated Adaptive Strategy: Decision-makers start by evaluating the raw tally difference between the two options. If the absolute difference in tally exceeds a subjective threshold, they rely entirely on this fast, compensatory tallying heuristic to make their choice. However, if the tally difference is small or tied (below the threshold), they shift to a cue-validity-driven strategy (Take-The-Best), heavily overweighting the most valid discriminating cue. Allowing the threshold to take lower values (e.g., 0 or 1) explains why subjects often exhibit strict tallying behavior even when options are structurally close, effectively increasing the overall reliance on tallying to match empirical data.

**Rationale:** Following the critic's feedback, the continuous threshold/WADD fallback from Iteration 2 under-predicted tallying behavior and was rejected. Reverting to the accepted base from Iteration 1, we implement the suggested minimal edit: shifting the discrete threshold range from {1, 2, 3} to {0, 1, 2}. This allows the model to default to tallying more frequently (e.g., when threshold is 0 or 1), directly addressing the under-prediction of tallying in Experiments 9, 15, and 16. We also slightly expanded the upper bound of the beta parameters to 25.0 to ensure deterministic choices can be fully expressed when tallying is engaged.

**Parameters:**
  - `threshold`: `{0, 1, 2}`
  - `beta_tally`: `[0.1, 25.0]`
  - `beta_ttb`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    # Check if the absolute tally difference meets or exceeds the threshold
    if abs(tally_a - tally_b) >= threshold:
        scores = np.array([tally_a, tally_b])
        beta = beta_tally
    else:
        # Shift to Take-The-Best (lexicographic) strategy
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.0, 0.0])
        beta = beta_ttb
            
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


### slot 2 — `pi_11` — SURVIVED ✓

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

### `pi_12` → slot 1 (via `new_theory`)

**Description:** Noisy Take-The-Best (Probabilistic Lexicographic Choice) with Validity Differences

**Rationale:** Following the critic's advice, the probability of successfully executing a cue comparison now scales with the difference between the current cue's validity and the next cue's validity in the sorted order (diff = val[j] - val[j+1]). This mathematically allows the model to skip the primary cue with high probability if its validity is very close to the secondary cue, enabling secondary cues to accumulate enough probability mass to strongly favor compensatory choices (as seen in Exp 9 and Exp 15). The ranges for alpha and gamma have been adjusted to [0.0, 10.0] and [0.1, 5.0] to provide enough scaling flexibility for these validity differences.

**Parameters:**
  - `alpha`: `[0.0, 10.0]`
  - `gamma`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    p_a = 0.0
    p_b = 0.0
    p_skip_accum = 1.0
    
    for i, j in enumerate(cue_order):
        if a[j] != b[j]:
            if i < len(cue_order) - 1:
                next_j = cue_order[i+1]
                diff = val[j] - val[next_j]
            else:
                diff = val[j]
                
            # Probability of successfully using this cue scales with its validity difference
            p_success = alpha * (diff ** gamma)
            p_success = min(max(p_success, 0.0), 1.0)
            
            if a[j] > b[j]:
                p_a += p_skip_accum * p_success
            else:
                p_b += p_skip_accum * p_success
                
            p_skip_accum *= (1.0 - p_success)
            
    # If all discriminating cues are skipped or no cues discriminate, guess randomly
    p_a += p_skip_accum * 0.5
    p_b += p_skip_accum * 0.5
    
    # Apply lapse rate
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b + epsilon * 0.5
    
    return np.array([p_a, p_b])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
