# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** People use a single-stage Additive Utility evaluation where features are weighted by a power-law transformation of their chance-centered objective validities. By avoiding artificial weight normalization and allowing for sufficiently high softmax temperatures and power parameters, the decision-making process can smoothly and stably interpolate between Tallying (equal weights), proportional weighting, and highly deterministic Take-The-Best behavior (where the most valid cues dominate completely).

**Rationale:** Following the critic's advice, we build directly on the successful Iteration 2 base. The chance-centered power law `weights = (val - 0.5 + 1e-6) ** gamma` is retained without normalization, as normalizing artificially compressed the scores and caused the softmax to become too soft in Iteration 5. To allow the model to make sharper, more deterministic choices when necessary (such as in TTB-dominant experiments like Exp 4), we significantly expand the `beta` range to `[0.1, 50.0]` and the `gamma` range to `[0.0, 10.0]`. This gives the optimizer the freedom to scale the scores naturally and find the strict non-compensatory regimes without hitting artificial numerical ceilings.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities using a simple power law on chance-centered values
    # gamma=0 -> Tallying, gamma=1 -> WADD, gamma>1 -> TTB
    centered_val = val - 0.5
    weights = (centered_val + 1e-6) ** gamma
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_8` — KILLED ✗

**Description:** Rank-Based Weighting with Normalized Power-Law Decay: Decision makers rank features by their objective validities and assign decision weights based on their ordinal rank. By applying a power-law transformation on the ranks (e.g., w_i = rank_i ^ -gamma) and explicitly normalizing these weights to sum to 1, the model ensures that the total accumulated evidence remains bounded on a consistent scale regardless of the steepness parameter gamma. This allows a single temperature parameter to stably govern choice determinism across both strictly compensatory (Tallying) and non-compensatory (Take-The-Best) strategies.

**Rationale:** Following the critic's advice, we revert to the power-law rank weighting from the accepted Iteration 1 base but introduce explicit weight normalization. By normalizing the weights to sum to 1.0 before computing the dot product, the resulting option scores are bounded. This solves the scaling issue where varying gamma drastically changed the absolute magnitude of the scores, which previously prevented a single beta parameter from controlling choice determinism consistently across Tallying and Take-The-Best regimes. The beta range is also expanded to 100.0 to allow for deterministic choices on this normalized scale.

**Parameters:**
  - `beta`: `[0.1, 100.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Rank validities descending (highest validity gets rank 1)
    ranks = rankdata(-val)
    
    # Ordinal weights: w_i = rank_i ^ (-gamma)
    # Features with validity exactly 0.5 are chance and should be ignored
    weights = np.where(val > 0.5, ranks ** (-gamma), 0.0)
    
    # Normalize weights so they sum to 1.0, keeping scores on a consistent scale
    sum_w = np.sum(weights)
    if sum_w > 0:
        weights = weights / sum_w
    
    scores = np.dot(stim, weights)
    
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_9` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Rationale:** Following the critic's advice, we implement a Sequential Evidence Accumulation model where features are sampled in descending order of validity. By allowing the stopping threshold to range from 0.0 to 3.0, the model can dynamically recover Take-The-Best behavior (when threshold <= 1.0, stopping after the first differentiating feature) or Tallying behavior (when threshold is high, accumulating across all features). This single unified mechanism avoids the scaling issues of the mixture model and naturally explains intermediate choice proportions observed in the experiments.

**Parameters:**
  - `threshold`: `[0.0, 3.0]`
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
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
