# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Threshold-Gated Compensatory Model: Subjects sort cues by their validities and evaluate the numerical gaps between adjacent cues in this sorted order. If the validities are closely clustered, subjects integrate them additively, functioning as a weighted additive model. However, if a gap exceeds a subjective threshold, all subsequent lower-ranked cues are heavily discounted or completely ignored, triggering a shift toward a non-compensatory, semi-lexicographic heuristic. The threshold is sensitive enough to trigger discrete strategy shifts frequently.

**Rationale:** Applying the minimal edit suggested by the critic: narrowed the `gap_threshold` range to [0.0, 0.3] to ensure the discrete shift triggers more readily for typical validity distributions, and widened `beta` to [0.1, 25.0] to allow for sharper determinism when the non-compensatory heuristic is engaged. The core gap-evaluating mechanism remains unchanged.

**Parameters:**
  - `gap_threshold`: `[0.0, 0.3]`
  - `discount_factor`: `[0.0, 1.0]`
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gap_threshold = float(parameters["gap_threshold"])
    discount_factor = float(parameters["discount_factor"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort validities descending to evaluate gaps
    order = np.argsort(-validities)
    sorted_vals = validities[order]
    
    w_sorted = np.zeros_like(sorted_vals)
    if len(sorted_vals) > 0:
        w_sorted[0] = sorted_vals[0]
        current_discount = 1.0
        
        for i in range(1, len(sorted_vals)):
            # If the gap between adjacent sorted validities exceeds the threshold, apply discount
            if (sorted_vals[i-1] - sorted_vals[i]) > gap_threshold:
                current_discount *= discount_factor
            w_sorted[i] = sorted_vals[i] * current_discount
            
    # Map subjective weights back to original feature order
    w = np.zeros_like(validities)
    w[order] = w_sorted
    
    # Option scores are the sum of feature values weighted by the threshold-gated subjective weights
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

**Rationale:** Following the arbiter's suggestion, this theory unifies compensatory and non-compensatory decision-making without relying on a discrete mixture of heuristics. By applying an individual-specific exponent (gamma) to the cue validities, the model can natively capture Tallying (gamma ~ 0), WADD (gamma ~ 1), and Take The Best (gamma >> 1) behaviors within a single, continuous weighted-additive framework. This provides a more robust and elegant explanation for the intermediate TTB matches and varying strategy adoptions observed in the experiments.

**Parameters:**
  - `gamma`: `[0.0, 10.0]`
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
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

**Description:** Probabilistic Strategy Mixture Model with Flexible Compensatory Component: Subjects maintain a repertoire of distinct cognitive strategies—a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive). On any given trial, they probabilistically select which strategy to deploy based on an individual trait parameter. To accurately capture heavily compensatory behavior observed in certain environments, the Weighted Additive strategy uses a non-linear power transformation of the validities, allowing subjects to tune their compensatory integration rather than strictly using raw validities.

**Rationale:** Following the latest critic feedback, we revert to the successful Iteration 4 base but tighten the upper bound of the lapse noise parameter `epsilon` from 0.5 to 0.2. The previous attempt to widen `gamma` and `beta` to capture deterministic behavior in Experiments 3 and 4 failed. The root cause is likely that a high `epsilon` allows the fitting process to rely on uniform noise to cover up mispredictions, which artificially suppresses the model's confidence on trials where subjects are highly deterministic. By constraining `epsilon`, we force the mixture model to explain the data through strategy selection rather than lapse noise, allowing it to reach the extreme choice probabilities observed in the heavily compensatory experiments.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `beta`: `[0.1, 25.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `gamma`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Strategy 1: Flexible Weighted Additive (WADD)
    wadd_weights = validities ** gamma
    scores_wadd = stim @ wadd_weights
    z = beta * (scores_wadd - np.max(scores_wadd))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Strategy 2: Take-The-Best (TTB)
    order = np.argsort(-validities)
    p_ttb_strat = np.array([0.5, 0.5])
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_ttb_strat = np.array([1.0, 0.0])
            break
        elif stim[1, idx] > stim[0, idx]:
            p_ttb_strat = np.array([0.0, 1.0])
            break
            
    # Mixture of strategies
    p_core = p_ttb * p_ttb_strat + (1.0 - p_ttb) * p_wadd
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
