# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: Individuals make decisions by sequentially searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the choice is based entirely on that single feature, ignoring all others. If no feature discriminates, a random guess is made.

**Rationale:** Following the arbiter's instructions, this model implements the Take-The-Best (TTB) heuristic. Unlike Tallying (which counts winning features equally) or WADD (which integrates all features compensatorily), TTB is a one-reason decision heuristic. It sorts features by their objective validities in descending order and evaluates them sequentially. The first feature that discriminates between the two options completely determines the decision, and all subsequent features are ignored. If no features discriminate, the decision defaults to a tie (equal scores), resulting in a uniform random guess. Softmax temperature and a uniform lapse rate are included to account for response noise.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            scores = np.array([0.0, 1.0])
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Two-Stage Heuristic (TTB with Tallying Fallback): Decision-makers initially employ a non-compensatory Take-The-Best (TTB) strategy, searching sequentially through features in descending order of validity. However, they only trust features whose validity exceeds a certain subjective threshold. If a discriminating feature is found above this threshold, the choice is based entirely on it. If no such feature exists (either because all high-validity features are tied or none meet the threshold), the decision-maker abandons the sequential search and falls back to a compensatory Tallying strategy, weighing all features equally and choosing the option with the most winning features.

**Rationale:** Following the arbiter's feedback, the pure Weighted Additive (WADD) model performs poorly and is discarded. To provide a strong and relevant alternative to the highly successful pure Take-The-Best (TTB) model, we introduce a 'Two-Stage' heuristic. This model recognizes that while people often rely on single, high-validity cues (TTB), they might not trust this non-compensatory strategy if the available discriminating cues are of low validity. The model searches sequentially but applies a validity threshold; if no high-validity cue discriminates, it falls back to a simple Tallying (Equal Weight) strategy across all features. This hybrid mechanism captures both non-compensatory and compensatory modes of decision-making within a single cohesive framework.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = None
    # Stage 1: Take-The-Best for features above the validity threshold
    for idx in order:
        if validities[idx] >= threshold:
            if a[idx] > b[idx]:
                scores = np.array([1.0, 0.0])
                break
            elif b[idx] > a[idx]:
                scores = np.array([0.0, 1.0])
                break
        else:
            # Stop searching if we hit features below the threshold
            break
            
    # Stage 2: Fallback to Tallying if no high-validity cue discriminated
    if scores is None:
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Model with Log-Odds: Decision-makers evaluate options using a purely compensatory strategy. They compute an overall value for each option by summing the log-odds of the validities for all features where the option has a positive cue. A choice is then made probabilistically based on the difference in these overall values, using a softmax rule combined with a uniform lapse rate to account for decision noise.

**Rationale:** Following the arbiter's recommendation, this proposes a strong, classical compensatory alternative to the non-compensatory Take-The-Best model. By implementing a Weighted Additive (WADD) model that directly derives its weights from the log-odds of the provided validities, we avoid the overfitting associated with learning free weights for each feature. Decision-makers sum these log-odds weights for all positive cues and make a probabilistic choice based on the resulting values.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
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
    
    # Transform validities into log-odds weights
    v = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v / (1.0 - v))
    
    # Compute the overall value (weighted sum) for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
