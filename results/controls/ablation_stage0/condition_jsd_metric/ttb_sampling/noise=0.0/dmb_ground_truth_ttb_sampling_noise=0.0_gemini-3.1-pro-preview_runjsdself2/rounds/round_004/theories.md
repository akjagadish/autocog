# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) proposes that individuals use a non-compensatory lexicographic heuristic for decision making. They search through features in descending order of their validity and base their choice entirely on the first feature that discriminates between the two options, ignoring all remaining features. If no feature discriminates, they guess. Response noise is incorporated via a softmax function over the resulting binary scores and a lapse rate for random errors.

**Rationale:** Following the critic's advice, we return to the base memoryless Take-The-Best (TTB) formulation but expand the lower bound of the inverse temperature parameter `beta` to [0.001, 10.0]. This allows the optimizer to explore softer decision boundaries and higher intrinsic choice noise without relying entirely on the uniform lapse rate parameter `epsilon` (which is kept at [0.0, 0.5]). The scoring mechanism remains unchanged (0 and 1) to avoid destabilizing the optimization surface.

**Parameters:**
  - `beta`: `[0.001, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Tallying (Equal Weights) heuristic: Decision makers ignore cue validities entirely and simply count the number of positive features for each option. They choose the option with the higher tally (i.e., the greatest number of positive cues). This represents a fast-and-frugal compensatory strategy where all cues are weighted equally, minimizing cognitive effort while still integrating all available features. Response noise is incorporated via a softmax function over the tallies, alongside a uniform lapse rate for random errors.

**Rationale:** Following the arbiter's suggestion, the Weighted Additive model was replaced with the Tallying heuristic. Tallying represents a major alternative in the fast-and-frugal heuristics literature. Instead of weighting cues by their validities (like WADD) or relying on a single best cue (like TTB), Tallying integrates all cues but weights them equally. This is operationalized by simply summing the positive features for each option and applying a softmax choice rule over the totals. It contrasts mechanistically with both TTB and WADD, offering a robust baseline that handles environments where validities are either unknown, ignored, or too computationally costly to integrate.

**Parameters:**
  - `beta`: `[0.001, 10.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    scores = np.array([np.sum(a), np.sum(b)], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Model with Log-Odds Weights: Decision makers integrate all available information by computing a compensatory weighted sum of the features for each option. However, instead of weighting cues by their raw validities, decision makers transform validities into log-odds. This non-linear scaling reflects a Bayesian-like evidence integration process, giving disproportionately higher weight to highly diagnostic cues. The option with the higher weighted sum is favored, with choice probabilities generated via a softmax function and a uniform lapse rate for random errors.

**Rationale:** Following the critic's advice, the raw validities in the WADD model have been replaced with log-odds weights. This minimal edit transforms the linear scaling of cue importance into a non-linear scaling, capturing the tendency of human decision-makers to treat highly valid cues as disproportionately more important, akin to Naive Bayes evidence integration. The validities are clipped to [0.01, 0.99] to ensure numerical stability.

**Parameters:**
  - `beta`: `[0.001, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds for non-linear weighting
    # Clip to avoid log(0) or division by zero
    v_clipped = np.clip(validities, 0.01, 0.99)
    weights = np.log(v_clipped / (1.0 - v_clipped))
    
    # WADD: compute the weighted sum of features for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
