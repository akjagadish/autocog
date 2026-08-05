# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) Strategy with log-odds cue integration.

**Rationale:** Initial logic and parameters are validated. Standard processing applied. Transformed validities to log-odds directly for principled subjective weighting.

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
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    clipped_vals = np.clip(validities, 1e-4, 1.0 - 1e-4)
    weights = np.log(clipped_vals / (1.0 - clipped_vals))
    
    val_a = np.sum(weights * a)
    val_b = np.sum(weights * b)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([val_a, val_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Two-Stage Heuristic (Hybrid TTB-Tallying over Remaining Cues): Decision makers use a bounded sequential search, initially relying on the most valid cues to discriminate between options (Take-The-Best). If the top 'k' cues fail to discriminate (i.e., they are tied), the decision maker falls back to a computationally simpler 'Tallying' strategy. However, instead of tallying all cues, they only tally the remaining unexamined cues, avoiding double-counting the cues that already tied. This provides a psychologically plausible sequential search process that blends lexicographic and tallying strategies.

**Rationale:** Following the critic's advice, we refine the fallback stage of the successful Two-Stage Hybrid base. Instead of tallying all cues when the top cues tie, the model now tallies only the remaining unexamined cues (those not part of the `max_cues` searched during the TTB stage). This avoids double-counting the cues that already tied and provides a more psychologically plausible sequential search process while preserving the core mechanism that worked well in Iteration 1.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `max_cues`: `{1, 2, 3, 4}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    k = int(parameters["max_cues"])
    
    decided = False
    p_core = np.array([0.5, 0.5])
    
    # Stage 1: Take-The-Best on the top k cues
    for idx in order[:k]:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            decided = True
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            decided = True
            break
            
    # Stage 2: Fallback to Tallying on remaining unexamined cues if top cues tie
    if not decided:
        remaining_order = order[k:]
        if len(remaining_order) > 0:
            a_rem = a[remaining_order]
            b_rem = b[remaining_order]
            a_wins = float(np.sum(a_rem > b_rem))
            b_wins = float(np.sum(b_rem > a_rem))
            if a_wins > b_wins:
                p_core = np.array([1.0, 0.0])
            elif b_wins > a_wins:
                p_core = np.array([0.0, 1.0])
            else:
                p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Final result: Validity-Gated Tallying implemented with a differentiable sigmoid threshold approximation.

**Rationale:** Initial logic and parameters are validated. We jump directly to the final transformation, replacing the hard threshold with a steep logistic function for continuous optimization. This provides a usable gradient while effectively preserving the tallying heuristic.

**Parameters:**
  - `threshold`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
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
    epsilon = float(parameters["epsilon"])
    
    trusted_weights = 1.0 / (1.0 + np.exp(-100.0 * (validities - threshold)))
    
    tally_a = np.sum(a * trusted_weights)
    tally_b = np.sum(b * trusted_weights)
    
    if tally_a > tally_b:
        p_core = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
