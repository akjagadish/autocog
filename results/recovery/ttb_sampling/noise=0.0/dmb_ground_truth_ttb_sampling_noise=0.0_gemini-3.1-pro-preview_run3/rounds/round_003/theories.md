# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

**Rationale:** Following the arbiter's suggestion, this implements the Take-The-Best (TTB) heuristic. TTB is a one-reason, non-compensatory model that searches through features in descending order of validity. It stops at the first discriminating feature and bases the choice entirely on it. This explains the high reliance on the most valid features seen in the data, distinguishing it from both Tallying (which ignores validities) and WADD (which is fully compensatory).

**Parameters:**
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
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Probabilistic Take-The-Best (PTTB) with exponential weighting: Decision-makers search through features sequentially but stochastically. The probability of checking a feature is proportional to an exponential function of its validity (exp(gamma * validity)), allowing high-validity features to strongly dominate the search order. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the decision-maker guesses. This yields a Luce choice rule over the exponentially weighted discriminating features.

**Rationale:** Following the critic's advice, we retain the exact Probabilistic Take-The-Best mechanism with exponential validity weighting, but we tighten the parameter ranges to allow the model to approximate deterministic Take-The-Best even more sharply. We increased the upper bound of gamma to 200.0 to enable steeper drop-offs between closely ranked validities, and reduced the upper bound of epsilon to 0.25 to prevent excessive random guessing that was diluting the deterministic predictions.

**Parameters:**
  - `gamma`: `[0.1, 200.0]`
  - `epsilon`: `[0.0, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating features for each option
    d_a = (a > b).astype(float)
    d_b = (b > a).astype(float)
    d_any = d_a + d_b
    
    if np.sum(d_any) == 0:
        p_a = 0.5
    else:
        # To prevent numerical overflow, subtract max validity among discriminating features
        max_v = np.max(validities[d_any > 0])
        weights = np.exp(gamma * (validities - max_v))
        
        # Sum of weights of features favoring A and B
        w_a = np.sum(d_a * weights)
        w_b = np.sum(d_b * weights)
        
        # The probability that the first discriminating feature found favors A
        p_a = w_a / (w_a + w_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers primarily rely on the deterministic, non-compensatory Take-The-Best heuristic, sequentially checking features ordered by validity and stopping at the first discriminating feature. A small subset of decisions relies on a compensatory Tallying strategy. By restricting the tallying mixture weight to a low range, the model maintains overwhelming TTB dominance, capturing high TTB adherence and the lack of a strong compensatory effect from the number of favoring features, while still allowing structured deviations from pure TTB.

**Rationale:** Following the critic's advice, the mixture weight for Tallying (`w_tally`) has been significantly reduced from [0.0, 0.5] to [0.0, 0.15], and `epsilon` reduced to [0.0, 0.15]. This minimal edit ensures that the model remains overwhelmingly dominated by the non-compensatory TTB heuristic. This addresses the failures in Experiment 7 (where TTB adherence must be >85%) and Experiment 8 (where heavily mixing Tallying spuriously introduced a feature-counting compensatory effect), while retaining the structural benefits of the mixture model.

**Parameters:**
  - `w_tally`: `[0.0, 0.15]`
  - `epsilon`: `[0.0, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # TTB Strategy
    order = np.argsort(validities)[::-1]
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # Tallying Strategy
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tal = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tal = np.array([0.0, 1.0])
    else:
        p_tal = np.array([0.5, 0.5])
        
    # Mixture of TTB and Tallying
    p_core = (1.0 - w_tally) * p_ttb + w_tally * p_tal
    
    # Uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
