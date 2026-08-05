# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Sequential Evidence Accumulation: Decision-makers inspect cues sequentially in order of validity, accumulating evidence for the favored option. The evidence contributed by each cue is its validity scaled by a non-linear parameter gamma. Search terminates when the absolute evidence difference reaches a threshold theta, or when all cues are exhausted. A choice is then made based on the accumulated evidence with softmax noise. This unified mechanism smoothly interpolates between Take-The-Best (low threshold), Tallying (high threshold, gamma=0), and Weighted Additive (high threshold, gamma>0).

**Rationale:** Following the critic's advice, I further narrowed the parameter ranges for gamma (from [0.0, 2.0] to [0.0, 1.0]) and beta (from [0.1, 10.0] to [0.1, 5.0]). This minimal edit keeps the Sequential Evidence Accumulation mechanism intact while addressing the overestimation in Experiments 3 and 4 by preventing validity differences from dominating the evidence sum too heavily and softening the softmax choices to better capture Tallying-like noisy behavior.

**Parameters:**
  - `theta`: `[0.0, 3.0]`
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += v[j] * diff
            if abs(E) >= theta:
                break
                
    # E > 0 means A is favored; E < 0 means B is favored
    scores = np.array([E, 0.0])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Add lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Parallel Cue Integration with Rank Discounting: Decision-makers process all available cues in parallel rather than strictly sequentially, but they discount the evidence provided by each cue based on its validity rank. The weight of a cue is a function of its validity (scaled non-linearly) and an exponential decay based on its rank order. This mechanism allows for a soft blending of compensatory and non-compensatory decision-making: strong rank discounting mimics Take-The-Best, while weak discounting with varying validity sensitivity smoothly interpolates between Tallying and Weighted Additive strategies, avoiding the need for a rigid probabilistic mixture.

**Rationale:** Following the critic's advice, the upper bounds for the scaling parameters have been further reduced: `beta` is lowered from 2.0 to 1.5, and `gamma` is lowered from 1.0 to 0.8. This edit continues the successful trajectory of softening evidence accumulation and choice determinism to compress the last remaining overshoots in Experiments 2, 4, and 6 closer to the human baseline, while keeping the core rank-discounting logic entirely intact.

**Parameters:**
  - `discount_rate`: `[0.0, 1.0]`
  - `gamma`: `[0.0, 0.8]`
  - `beta`: `[0.1, 1.5]`
  - `epsilon`: `[0.0, 0.8]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    discount_rate = float(parameters["discount_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for rank, j in enumerate(cue_order):
        diff = a[j] - b[j]
        weight = (val[j] ** gamma) * (discount_rate ** rank)
        E += weight * diff
        
    scores = np.array([E, 0.0])
    
    # Softmax for choice probability
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Apply lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Sequential Search with Leaky Evidence Integration

**Rationale:** Following the critic's advice, I constrained the `leak` parameter range to [0.5, 1.0] to prevent excessive forgetting of early valid cues, which preserves the lexicographic nature of the decision process when appropriate. I also expanded `beta_choice` to allow more deterministic choice behavior and introduced a `base_stop` parameter to act as a baseline hazard rate for stopping early, independent of the evidence threshold.

**Parameters:**
  - `leak`: `[0.5, 1.0]`
  - `gamma`: `[0.0, 5.0]`
  - `theta`: `[0.0, 10.0]`
  - `beta_stop`: `[0.01, 20.0]`
  - `beta_choice`: `[0.01, 50.0]`
  - `base_stop`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    theta = float(parameters["theta"])
    beta_stop = float(parameters["beta_stop"])
    beta_choice = float(parameters["beta_choice"])
    base_stop = float(parameters["base_stop"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    p_a_total = 0.0
    p_continue = 1.0
    
    for i, j in enumerate(cue_order):
        diff = a[j] - b[j]
        
        # Accumulated evidence leaks before adding new evidence
        E = E * leak + v[j] * diff
        
        # Soft stopping rule: probability of terminating search
        if i == len(cue_order) - 1:
            p_stop = 1.0
        else:
            # Sigmoid function for stopping probability centered at theta
            p_stop_evidence = 1.0 / (1.0 + np.exp(-beta_stop * (abs(E) - theta)))
            p_stop = base_stop + (1.0 - base_stop) * p_stop_evidence
            
        # Probability of choosing A if search stops at the current cue
        p_a_given_stop = 1.0 / (1.0 + np.exp(-beta_choice * E))
        
        # Accumulate the marginal probability of choosing A at this step
        p_a_total += p_continue * p_stop * p_a_given_stop
        
        # Update the probability that search continues to the next cue
        p_continue *= (1.0 - p_stop)
        
    p_final = np.array([p_a_total, 1.0 - p_a_total])
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_final + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
