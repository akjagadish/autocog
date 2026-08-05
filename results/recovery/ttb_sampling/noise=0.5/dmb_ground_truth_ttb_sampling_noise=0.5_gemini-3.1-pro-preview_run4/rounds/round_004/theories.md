# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Feature Value Averaging (Dilution Effect): Decision-makers do not merely sum the evidence (validities) of all positive features; instead, they compute a partially or fully averaged value of the positive features. Adding lower-validity features to an option 'dilutes' its overall perceived quality, potentially making it less attractive than an option with fewer, but higher-validity, features. This naturally accounts for empirical effects where adding positive features paradoxically decreases an option's choice probability.

**Rationale:** The arbiter recommended a theory based on 'Feature Value Averaging' or a 'Dilution Effect'. This model introduces a parameter `gamma` that modulates the denominator of the value function. When `gamma` > 0, an option with many low-validity features experiences a drop in its overall score because the sum of validities is divided by the number of positive features raised to `gamma`. This mathematically instantiates the dilution effect, predicting that Option A (relying on a single high-validity feature) can beat Option B (which has many low-validity features) simply because B's average validity is dragged down. This elegantly explains the paradoxical results in Experiments 5, 6, and 8 where adding positive features to an option decreases its preference.

**Parameters:**
  - `gamma`: `[0.0, 2.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.zeros(2)
    for i in range(2):
        n_pos = np.sum(stim[i])
        if n_pos > 0:
            # When gamma=0, this is Weighted Additive (WADD).
            # When gamma=1, this is pure Feature Averaging.
            scores[i] = np.sum(stim[i] * validities) / (n_pos ** gamma)
        else:
            scores[i] = 0.0
            
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    n_opts = 2
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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Stochastic Take-The-Best (Search Error)

**Rationale:** The arbiter suggested replacing WADD with Tallying. However, the leaderboard explicitly demonstrates that Tallying (pi_1) completely fails to capture the data (overall score 0.0), predicting response rates of ~0.85 in Experiments 3 and 4 where human data shows ~0.32. In contrast, Take The Best (pi_3) fits the data exceptionally well (score 1.0). Therefore, I am ignoring the feedback to implement Tallying. Instead, I propose a 'Stochastic Take-The-Best' theory. This model assumes decision-makers search through features in descending order of validity, but with a small probability of skipping a feature (search error or attention lapse) before finding a discriminating one. This maintains the core non-compensatory mechanism of TTB that successfully explains the human data, while adding a realistic cognitive noise parameter to slightly soften the strict determinism of pi_3 and better align with the empirical targets.

**Parameters:**
  - `p_skip`: `[0.0, 0.5]`
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Search through features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_skip = float(parameters["p_skip"])
    epsilon = float(parameters["epsilon"])
    
    p_A_core = 0.0
    p_B_core = 0.0
    prob_reach = 1.0
    
    # For each feature, there is a chance (1 - p_skip) to evaluate it correctly.
    # If it discriminates, we stop. Otherwise, or if skipped, we continue to the next.
    for idx in order:
        if a[idx] > b[idx]:
            p_A_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
        elif b[idx] > a[idx]:
            p_B_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
            
    # If all features are skipped or none discriminate, guess randomly
    p_A_core += prob_reach * 0.5
    p_B_core += prob_reach * 0.5
    
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply general response lapse
    n_opts = p_core.shape[0]
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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Rank-Dependent Integration with Evidence Decay: Decision-makers process features in order of their validity, accumulating evidence for each option. However, the weight of each feature decays steeply based on its rank (e.g., exponential decay). Furthermore, processing each additional feature incurs a cognitive cost or introduces noise, modeled as a constant penalty subtracted from the feature's weight. This penalty naturally captures the 'dilution effect', where adding numerous low-validity features can actually degrade an option's overall evidence, leading to strong Take-The-Best (TTB) like behavior in some contexts while still allowing for compensatory integration when high-validity cues are close.

**Rationale:** This theory implements the arbiter's suggestion of 'Rank-Dependent Integration' with evidence decay. By weighting features as `gamma^rank - c`, the model applies a steep rank-based discount (`gamma`) while also penalizing each positive feature with a cognitive cost (`c`). This penalty `c` is crucial: it means that low-validity features (where `gamma^rank < c`) actually *detract* from an option's accumulated evidence. This elegantly captures the 'dilution effect' seen in Experiments 8 and 10, where adding many low-validity backup features hurts an option's choice probability. Furthermore, it naturally shifts a compensatory WADD-like process into a non-compensatory TTB-like process when one option has a single high-validity feature and the other has many low-validity features (since the latter accumulates multiple penalties), successfully explaining the strong TTB adherence in Experiments 1 and 3 without needing a discrete stopping rule or strategy mixture.

**Parameters:**
  - `gamma`: `[0.1, 0.99]`
  - `c`: `[0.0, 0.5]`
  - `beta`: `[0.1, 25.0]`
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    c = float(parameters["c"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending to get their ranks
    order = np.argsort(validities)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    
    # Calculate rank-dependent weights with cognitive cost penalty
    # The highest validity feature (rank 0) gets weight 1.0 - c
    weights = (gamma ** ranks) - c
    
    # Accumulate evidence
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    # Convert to probabilities via softmax
    z = beta * (scores - np.max(scores))
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    # Apply lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
