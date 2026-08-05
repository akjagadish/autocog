# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory strategy. They multiply each feature's value by its corresponding cue validity and sum these products to form an overall subjective value for each option. The option with the higher weighted sum is chosen. This allows multiple lower-validity cues to collectively outweigh a single high-validity cue, capturing behavior that falls between pure Take The Best and pure Tallying. To account for empirical response noise, the decision process incorporates a moderate degree of stochasticity.

**Rationale:** Following the critic's feedback, the WADD mechanism is preserved entirely, but the parameter ranges for decision noise have been adjusted. The empirical metrics lie closer to 0.5 than the previous model's predictions. By restricting `beta` to [0.1, 5.0] and raising the lower bound of `epsilon` to 0.1, the model naturally injects more stochasticity, pulling its predictions closer to the observed human data without altering the core compensatory decision rule.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.1, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

**Description:** Contingent Heuristic Theory (Difference-based Compensatory): Decision-makers conditionally apply decision strategies based on the relative strength of the best available evidence compared to the next best piece of evidence. They identify the most valid cue that discriminates between the two options and the second most valid discriminating cue. If the difference in validity between these two cues exceeds a subjective threshold (indicating the top cue is highly salient relative to the rest), they rely entirely on the top cue (Take-The-Best). If the difference is not sufficiently large, they fall back to a fully compensatory Weighted Additive (WADD) strategy.

**Rationale:** Following the critic's advice, the gating condition for relying on the Take-The-Best (TTB) heuristic was changed from an absolute validity threshold to a relative validity difference (`delta`). The previous model over-predicted TTB choices because absolute validities are often high across the board. By evaluating `v_max - v_next >= delta`, TTB is only triggered when the best discriminating cue is saliently better than the next best available evidence. This minimal edit directly addresses the critic's diagnosis while keeping the model within the prescribed Contingent Heuristic family.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `delta`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    cue_order = np.argsort(-val, kind='stable')
    
    ttb_winner = None
    v_max = 0.0
    v_next = 0.0
    found_first = False
    
    for cue in cue_order:
        if a[cue] != b[cue]:
            if not found_first:
                v_max = val[cue]
                ttb_winner = 0 if a[cue] > b[cue] else 1
                found_first = True
            else:
                v_next = val[cue]
                break
                
    delta = float(parameters["delta"])
    
    if ttb_winner is not None and (v_max - v_next) >= delta:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
    else:
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        scores = np.array([score_a, score_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Rank-Dependent Weighting (Linear Rank-based WADD) Theory: Decision-makers evaluate options using a compensatory strategy, but instead of using exact numerical validities, they rank the cues by validity and assign subjective weights based on a linear drop-off from the top rank. This captures the compensatory behavior observed in the data without assuming perfect mathematical integration of the stated probabilities, offering a cognitively plausible and mathematically stable mechanism for cue integration.

**Rationale:** Following the critic's advice, we narrow the `decay` parameter range to `[0.0, 0.25]` to ensure that combinations of lower-ranked cues can reliably outweigh a single top-ranked cue, as required by empirical data in Exps 9 and 10. We also widen the `beta` parameter range to `[0.1, 20.0]` to allow the softmax function to translate these correct ordinal preferences into more deterministic choice probabilities, pulling the model's predictions away from 0.5.

**Parameters:**
  - `decay`: `[0.0, 0.25]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    # Rank cues by validity (highest to lowest)
    cue_order = np.argsort(-val, kind='stable')
    decay = float(parameters["decay"])
    
    weights = np.zeros_like(val)
    for r, cue_idx in enumerate(cue_order):
        weights[cue_idx] = max(0.0, 1.0 - decay * r)
        
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
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
