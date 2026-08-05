# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Tallying with Take-The-Best Tiebreaker: Decision makers primarily use a simple Tallying heuristic, counting the number of features that favor each option. If one option has a higher tally, it is chosen. However, if the tallies are tied, they do not simply guess; instead, they fall back to the Take-The-Best (TTB) heuristic, breaking the tie by choosing the option favored by the single most valid differentiating feature. This tie-breaking influence can be parameterized to allow for both positive reinforcement or penalty depending on the specific cue structures.

**Rationale:** Following the critic's advice, we expanded the range of the tie-breaker weight 'tau' to include negative values ([-2.0, 2.0]). Positive values for tau failed to capture the large metric difference in Experiment 6 (0.71), which requires the model to be highly confident in strict tally wins while systematically avoiding the TTB prediction on tied trials. Permitting a negative tau gives the optimizer the flexibility to invert the tie-breaker's direction if that best explains the empirical behavior on tied trials, without altering the primary tallying mechanism.

**Parameters:**
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `tau`: `[-2.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary heuristic: Tallying (counting strict wins)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Tie-breaker: Take-The-Best
    if score_a == score_b:
        tau = float(parameters["tau"])
        val = np.asarray(parameters["validities"], dtype=float)
        # Sort indices by descending validity
        order = np.argsort(-val, kind="stable")
        for idx in order:
            if a[idx] > b[idx]:
                score_a += tau
                break
            elif b[idx] > a[idx]:
                score_b += tau
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Weighted Additive (WADD) with Subjective Weighting: Decision-makers evaluate options using a fully compensatory strategy where each option's value is the weighted sum of its features. Instead of using the raw validities as weights, they apply a subjective transformation modeled as a power law (validity^gamma). This parameterization allows the model to capture a spectrum of strategies: when gamma is 0, it reduces to Equal-Weight (Tallying); when gamma is 1, it is standard WADD. By constraining gamma to be very small, the model predominantly relies on tallying-like behavior while allowing slight compensatory deviations to capture nuances in specific experiments.

**Rationale:** Following the critic's advice, I further reduced the upper bound of the `gamma` parameter from 2.0 to 0.8. This tighter bound forces the subjective weights to remain very close to 1.0 (pure Tallying), which is necessary to capture the overwhelming human preference for equal weighting observed in Experiments 3 and 4. At the same time, it still allows the slight compensatory deviations required to explain the tie-breaking behavior in Experiments 7 and 8.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 0.8]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights via a power law
    weights = val ** gamma
    
    # Calculate the weighted sum for each option
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Top-K Feature Focus with TTB Fallback: Decision-makers evaluate options by focusing their attention only on a subset of the most valid features (the Top-K). They tally the wins and losses strictly within this subset. If this subset tally clearly favors one option, they choose it. However, if the tallies are tied within the Top-K features, they fall back to a strict Take-The-Best (TTB) evaluation on the remaining (lower validity) features to break the tie.

**Rationale:** Following the critic's advice, this edit transitions from a uniform 'Tiered Tallying' to a 'Top-K Feature Focus' mechanism. The decision-maker exclusively tallies the Top-K most valid features. If a tie occurs within this subset, the model falls back to a strict Take-The-Best (TTB) evaluation on the remaining lower-validity cues. This directly addresses the mechanistic failures identified by capturing the strong Top-K preferences seen in Experiment 4 and 6. Additionally, the epsilon parameter range has been tightened to [0.0, 0.15] to prevent predictions from being artificially flattened towards 0.5, allowing the model to match extreme empirical probabilities more closely.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.15]`
  - `k`: `{2, 3, 4, 5, 6}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(parameters["k"])
    
    # Sort features by descending validity
    order = np.argsort(-val, kind="stable")
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    top_k_indices = order[:k]
    a_top = a[top_k_indices]
    b_top = b[top_k_indices]
    
    a_wins = np.sum(a_top > b_top)
    b_wins = np.sum(b_top > a_top)
    
    if a_wins > b_wins:
        score_a = 1.0
    elif b_wins > a_wins:
        score_b = 1.0
    else:
        # Fallback to TTB on remaining features
        remaining_indices = order[k:]
        for idx in remaining_indices:
            if a[idx] > b[idx]:
                score_a = 1.0
                break
            elif b[idx] > a[idx]:
                score_b = 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```
