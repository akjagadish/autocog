# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Tallying (Equal Weighting) posits that decision-makers simply count the total number of positive features for each option, ignoring the differential validities of the cues. The option with the higher count of positive features is preferred. This represents a compensatory but highly frugal heuristic, where evidence is accumulated equally across all available cues. If the counts are equal, the decision-maker guesses. Response noise is modeled via a softmax over these counts with an independent lapse rate. To account for empirical choices that often deviate from pure tallying on conflict trials, the decision process incorporates substantial choice noise.

**Rationale:** Following the critic's advice, we implement a minimal diff to the Tallying model by further restricting the `beta` parameter to `[0.0, 2.0]`. The previous iteration successfully brought the loss down by allowing more choice noise, but the model still slightly underpredicted the empirical rate of TTB-consistent choices on conflict trials. By lowering the upper bound of beta even further, we force the model to be less deterministic on these trials, pushing the predictions closer to the empirical ~0.42-0.45 range.

**Parameters:**
  - `beta`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features)")
        
    # Tallying: count the number of positive features for each option
    scores = np.sum(stim, axis=1)
    
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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Rationale:** Follows the critic's advice to implement a non-compensatory Take-The-Best model with a probabilistic cue search order. The model stops at the first discriminating cue, maintaining a strict non-compensatory mechanism at the single-trial level. However, the order in which cues are searched is sampled without replacement using a softmax over cue validities (via the Gumbel-max trick). The temperature parameter `tau` controls the determinism of the search order. When `tau` is large, the search order becomes uniform, which on average mimics the compensatory tallying-like behavior observed in Experiment 4, bridging the gap between non-compensatory heuristics and compensatory empirical patterns.

**Parameters:**
  - `tau`: `[0.01, 100.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
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

**Description:** Weighted Additive Model with Power-Scaled Log-Odds (WADD-Gamma). Decision-makers compute a weighted sum of features for each option. The weights are derived from the log-odds of the cue validities, raised to a power gamma. This parameterization allows the model to smoothly interpolate between Tallying/Equal-Weighting (gamma = 0) and standard log-odds WADD (gamma = 1). Choices are then made via a softmax over the weighted sums, incorporating an independent lapse rate for noise.

**Rationale:** Following the critic's advice, we adjust the upper bound of `gamma` to 0.75. This provides a more precise middle ground between the accepted gamma=1.0 and the rejected gamma=0.5. It gently pushes the model to be slightly more compensatory, aiming to close the gap in Experiments 3 and 4 without causing the catastrophic failure in Experiment 6.

**Parameters:**
  - `gamma`: `[0.0, 0.75]`
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to ensure log-odds are strictly positive and well-defined
    v_clipped = np.clip(validities, 0.5001, 0.9999)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
