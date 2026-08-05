# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.

**Rationale:** Following the critic's advice, the parameter range for `beta` has been slightly widened to [0.1, 3.5]. This intermediate range aims to strike a better balance by restoring some of the compensatory determinism lost in the previous iteration, which should help recover performance in Experiments 2 and 6 while maintaining the gains achieved in Experiments 3 and 4.

**Parameters:**
  - `mixture_p`: `[0.0, 1.0]`
  - `beta`: `[0.1, 3.5]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    mixture_p = float(parameters['mixture_p'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # --- TTB Component ---
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- WADD Component ---
    # WADD uses validities as weights
    scores = stim @ val
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # --- Mixture ---
    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Sequential Evidence Accumulation: Decision-making is driven by a sequential sampling process where features are evaluated in order of their subjective validity. As each feature is processed, the validity-weighted difference between the options is added to a running accumulator. If this accumulated evidence reaches a predefined threshold at any point, search is immediately terminated and a choice is made (mimicking non-compensatory heuristics like Take-The-Best). If all features are exhausted without the evidence hitting the boundary, the subject makes a probabilistic choice based on the final accumulated tally (mimicking compensatory strategies like WADD). This single-process model naturally unifies fast-and-frugal heuristics and exhaustive compensatory integration depending on the height of the evidence threshold.

**Rationale:** Following the critic's advice, we further reduce the upper bound of the `beta` parameter from 5.0 to 1.5. This makes the soft choice component (used when the sequential evidence threshold is not reached) even more probabilistic. This minimal edit aims to further dampen the model's overestimation of compensatory and deterministic responding in Experiments 6, 7, and 8, bringing the simulated metrics closer to the muted empirical effects observed in the data.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 1.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    threshold = float(parameters['threshold'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    A_val = 0.0
    decided = False
    p_a = 0.5
    
    # Sequentially accumulate evidence
    for j in cue_order:
        A_val += val[j] * (a[j] - b[j])
        if A_val >= threshold:
            p_a = 1.0
            decided = True
            break
        elif A_val <= -threshold:
            p_a = 0.0
            decided = True
            break
            
    # If boundary not reached, make a soft choice based on the final tally
    if not decided:
        z = beta * A_val
        # Numerically stable logistic function for P(A)
        if z > 100:
            p_a = 1.0
        elif z < -100:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    p_b = 1.0 - p_a
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Normalized Rank-Dependent Cue Weighting with Base Influence: Decision-making relies on a compensatory integration process where the subjective weight of each cue is a product of its exponentially decaying rank and a dampened version of its objective validity, plus a constant base weight. These weights are then normalized across all available cues to sum to 1. The base weight ensures that even the lowest-ranked cues retain a strict minimum influence, allowing the model to capture the compensatory integration of multiple weak cues without abandoning the core exponential decay that successfully captures heuristic-like dominance of top cues.

**Rationale:** Following the critic's advice, I added a `base_weight` parameter to the exponential decay model from Iteration 3. The Iteration 4 hyperbolic decay was rejected because it distorted the relative weighting too much, worsening fit on Exps 4 and 9. By reverting to the accepted exponential decay and adding a constant `base_weight` before normalization, we provide a strict minimum influence floor for lower-ranked cues. This 'fat tail' allows the model to better capture the compensatory behavior seen in Experiments 2 and 4 while preserving the successful heuristic-like dominance of top cues.

**Parameters:**
  - `decay`: `[0.01, 1.0]`
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 1.0]`
  - `base_weight`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    decay = float(parameters['decay'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    gamma = float(parameters['gamma'])
    base_weight = float(parameters['base_weight'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    weights = np.zeros(len(val))
    for rank, j in enumerate(cue_order):
        weights[j] = (val[j] ** gamma) * (decay ** rank) + base_weight
        
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights /= weights_sum
        
    evidence_diff = np.sum(weights * (a - b))
        
    z = beta * evidence_diff
    
    # Numerically stable logistic function
    if z > 100:
        p_a = 1.0
    elif z < -100:
        p_a = 0.0
    else:
        p_a = 1.0 / (1.0 + np.exp(-z))
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Blend with lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
