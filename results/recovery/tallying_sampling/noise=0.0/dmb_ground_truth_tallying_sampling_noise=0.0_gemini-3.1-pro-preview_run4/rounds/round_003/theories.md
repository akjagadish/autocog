# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

**Rationale:** The arbiter feedback indicates that Take The Best (TTB) vastly overpredicts the empirical adherence to the single best cue. Subjects in these experiments systematically choose options backed by a coalition of cues rather than a single highly valid cue. By implementing a Tallying (Equal Weights) theory where subjects simply count the number of positive features for each option, the model naturally favors the option with more positive cues regardless of their continuous validities. This compensatory mechanism is expected to perfectly capture the extremely low TTB adherence observed in the experimental data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
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
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Weighted Additive Strategy (WADD) with Linearly Compressed Weights: Individuals compute a compensatory weighted sum of positive features for each option, but their subjective weights are compressed relative to objective validities. People interpolate between raw validities and equal weights (Tallying), flattening the weight distribution. This allows a large number of less valid cues to compensate for a highly valid cue, matching human behavior in compensatory environments.

**Rationale:** Following the arbiter's most recent feedback, we introduce a linear mixing parameter `alpha` bounded between 0.0 and 1.0. This computes the subjective weights as an interpolation between raw validities (alpha=1) and equal weights (alpha=0). Unlike previous attempts using unbounded power-laws or log-odds (which amplified differences and worsened the fit), this linear compression safely bounded the weights, directly addressing the failures in Experiments 3 and 4 by letting the model smoothly capture the human tendency towards equal-weighting (Tallying) without diverging.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Interpolate between raw validities and equal weights
    weights = alpha * validities + (1.0 - alpha) * 1.0
    
    # Weighted Additive Strategy
    scores = stim @ weights
    
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Bounded Tallying (Top-k): Due to working memory constraints, individuals cannot process all available information. Instead of searching for a single discriminating cue (as in Take-The-Best) or tallying all cues (as in full Tallying), they focus their attention on a limited subset of the most valid features. They select the top 'k' most valid cues and apply an equal-weight tallying strategy only to these cues. This bounded-rationality heuristic combines validity-based attention with non-compensatory counting, serving as a resource-constrained alternative to full-information Tallying.

**Rationale:** Following the critic's advice, I restricted the `epsilon` parameter range to [0.0, 0.2] to prevent excessive random guessing from pulling predictions toward 0.5. I also increased the lower bound of `capacity` to 5.0 (range [5.0, 10.0]) to ensure the bounded tallying mechanism evaluates a robust subset of cues on larger 6-7 feature tasks, preventing underfitting on Experiments 3, 4, and 7 while maintaining the core Top-k mechanism.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `capacity`: `[5.0, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Bounded Tallying expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    capacity = int(round(float(parameters["capacity"]))) 
    
    n_features = stim.shape[1]
    k = min(max(1, capacity), n_features)
    
    # Indices of the top k most valid features
    # argsort is ascending, so we take the last k elements
    top_k_idx = np.argsort(validities)[-k:]
    
    # Tally only the top k features
    scores = stim[:, top_k_idx].sum(axis=1)
    
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
