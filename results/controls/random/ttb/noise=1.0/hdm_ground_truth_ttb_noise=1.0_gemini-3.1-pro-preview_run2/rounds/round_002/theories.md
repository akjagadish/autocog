# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Take The Best (TTB) Heuristic: Decision makers process information sequentially, searching through features in descending order of their subjective validity. They stop at the first feature that discriminates between the options (i.e., the absolute difference in feature values exceeds a certain threshold) and choose the option favored by that single cue. All other lower-validity features are ignored. If no cue discriminates, they guess. This represents a classic fast-and-frugal one-reason decision-making benchmark.

**Rationale:** Following the critic's feedback, the upper bound for the `threshold` parameter has been expanded from 25.0 to 50.0. This places even more prior probability mass on thresholds that exceed the maximum possible attribute difference, ensuring the model more frequently falls back to the 0.50 guessing rate, closing the gap to the human data.

**Parameters:**
  - `validities`: `validities`
  - `threshold`: `[0.0, 50.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        diff = a[idx] - b[idx]
        if abs(diff) > threshold:
            if diff > 0:
                p_core = np.array([1.0, 0.0])
            else:
                p_core = np.array([0.0, 1.0])
            break
            
    # Incorporate lapse rate (epsilon)
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

**Description:** Random Guessing / High-Lapse Baseline: When confronted with complex multi-attribute choices involving cardinal feature magnitudes that do not easily yield to simple heuristics, decision makers become overwhelmed and resort to random guessing. Their choices reflect a near-uniform probability distribution over the available options, occasionally influenced by a negligible spatial or option-order bias.

**Rationale:** The arbiter noted that the previous WADD theory was degenerate because participants do not exhibit strong compensatory behavior, and aggregate choices are consistently near 0.5. This implies that subjects are likely overwhelmed by the cardinal feature magnitudes and simply guess. A pure Random Guessing model explicitly instantiates this hypothesis, capturing the ~0.5 metric values across all experiments without relying on a mismatched mechanism like TTB with inflated thresholds.

**Parameters:**
  - `bias`: `[0.48, 0.52]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    bias = float(parameters["bias"])
    return np.array([bias, 1.0 - bias])
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** High-Temperature Weighted Additive (WADD) Theory: Subjects attempt to integrate all available features by computing a weighted sum of feature differences, using subjective validities as weights. However, their decision process is subject to extremely high cognitive noise (a very low inverse temperature, beta, in the softmax choice rule). This results in behavior that appears nearly random, capturing the high lapse rates observed across experiments, while still retaining a slight sensitivity to large aggregate differences in option quality.

**Rationale:** Following the critic's feedback, the WADD mechanism is preserved but the upper bound of the inverse temperature (beta) parameter is tightened from 0.015 to 0.005. This extremely low beta range ensures that the model predicts near-zero effects, more accurately reflecting the negligible effect sizes observed in human data in Experiments 4, 5, and 6, while still maintaining the formal structure of multi-attribute integration.

**Parameters:**
  - `beta`: `[0.0, 0.005]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Compute scores as weighted sum of features
    scores = stim @ validities
    
    # Extremely high cognitive noise (low beta)
    beta = float(parameters["beta"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
