# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Rationale:** Following the critic's feedback, the upper bound of the `beta` parameter has been further reduced from 5.0 to 2.0. This ensures the softmax generates even softer probabilities, capturing the highly noisy, near-guessing behavior observed in the human data while maintaining the core Take-The-Best mechanism.

**Parameters:**
  - `beta`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 1.0]`
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
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Dual-Process Hybrid Model (Take-The-Best + Weighted Additive) with soft compensatory noise: Human decision-making is predominantly non-compensatory (Take-The-Best), but subjects sometimes fall back on a compensatory strategy (Weighted Additive). Because WADD operates on sums of validities, its response noise is decoupled and constrained to extremely low values to reflect the highly soft, almost guessing-like nature of human compensatory choice. A high potential lapse rate captures general response noise and further scales down overconfident predictions.

**Rationale:** Following the critic's advice, we increase the upper bound of the uniform lapse rate `epsilon` from 0.6 to 1.0 to address the consistent overprediction (+0.07 to +0.08) in choice probabilities across several experiments. We also further reduce the upper bound of `beta_wadd` to 0.15, continuing the successful trajectory of softening the compensatory fallback component to better match the highly noisy human behavior in these tasks.

**Parameters:**
  - `beta_ttb`: `[0.0, 10.0]`
  - `beta_wadd`: `[0.0, 0.15]`
  - `p_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Hybrid expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) evaluation
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in ranked_features:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    # Weighted Additive (WADD) evaluation
    wadd_a = np.sum(a * validities)
    wadd_b = np.sum(b * validities)
    wadd_scores = np.array([wadd_a, wadd_b])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Softmax probabilities for TTB
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        p_ttb_vec = np.array([0.5, 0.5])
    else:
        z_ttb = beta_ttb * ttb_scores
        z_ttb = z_ttb - np.max(z_ttb)
        e_ttb = np.exp(z_ttb)
        p_ttb_vec = e_ttb / np.sum(e_ttb)
        
    # Softmax probabilities for WADD
    if wadd_a == wadd_b:
        p_wadd_vec = np.array([0.5, 0.5])
    else:
        z_wadd = beta_wadd * wadd_scores
        z_wadd = z_wadd - np.max(z_wadd)
        e_wadd = np.exp(z_wadd)
        p_wadd_vec = e_wadd / np.sum(e_wadd)
        
    # Mix the two processes
    p_ttb_weight = float(parameters["p_ttb"])
    p_core = p_ttb_weight * p_ttb_vec + (1.0 - p_ttb_weight) * p_wadd_vec
    
    # Apply uniform lapse
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Thresholded Tallying: Human decision-making relies on a semi-compensatory heuristic where subjects simplify the decision by tallying (counting) the positive features of each option, but they only consider features that are sufficiently important. Features whose validity falls below a learned threshold are completely ignored. This provides a mechanism that respects feature importance (unlike pure equal-weights tallying) without requiring precise validity-weighted sums or falling back on a dual-process architecture.

**Rationale:** Following the critic's advice, the Dual-Process Hybrid architecture has been entirely replaced with a 'Thresholded Tallying' model. Pure tallying failed because it ignored feature validities, and rank-weighted tallying failed due to arbitrary integer scaling. Thresholded tallying resolves this by introducing a parameter `tau`. Subjects sum the positive features for each option, but only among features whose validity is >= `tau`. This is a pure semi-compensatory strategy that respects the importance of highly valid cues (preventing weak features from overriding strong ones) without reverting to WADD or a hybrid structure.

**Parameters:**
  - `beta`: `[0.0, 10.0]`
  - `tau`: `[0.5, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
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
    
    # Determine which features pass the validity threshold
    tau = float(parameters["tau"])
    valid_mask = validities >= tau
    
    # Thresholded Tallying (Equal Weights) evaluation on valid features only
    tally_scores = np.array([np.sum(a[valid_mask]), np.sum(b[valid_mask])])
    
    beta = float(parameters["beta"])
    
    # Softmax probabilities for Thresholded Tallying
    if tally_scores[0] == tally_scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        z = beta * tally_scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    # Apply uniform lapse
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
