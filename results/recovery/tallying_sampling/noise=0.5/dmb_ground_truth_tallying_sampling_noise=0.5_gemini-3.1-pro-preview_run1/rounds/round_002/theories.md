# Round 2 — Theories

**Verdict:** `new_model` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

**Rationale:** Expanded the parameter ranges for beta (lowered minimum to 0.01) and epsilon (raised maximum to 1.0) as per the critic's feedback. This allows the model to produce higher noise levels, enabling it to better fit the ~0.33 TTB match rate observed on compensatory trials without altering the fundamental Tallying mechanism.

**Parameters:**
  - `beta`: `[0.01, 20.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Mixture of Heuristics: Subjects probabilistically switch between a compensatory, equal-weighting strategy (Tallying) and a frugal, lexicographic strategy (Take-The-Best). By mixing these two heuristics with a prior bias towards Tallying, the model captures the dominant tendency to count features while allowing for occasional validity-maximizing behavior, avoiding the unrealistic assumption of full compensatory validity weighting (WADD).

**Rationale:** Following the critic's advice, I expanded and shifted the `p_tally` parameter bounds downwards from [0.5, 1.0] to [0.4, 0.85]. The previous bounds forced the model to be slightly too Tally-dominant, under-predicting the empirical ~35% TTB match rate. This adjustment allows the model to find a slightly more balanced mixture, increasing the proportion of TTB-consistent choices while maintaining the overall preference for Tallying. The core mechanism and other parameters remain unchanged.

**Parameters:**
  - `beta_tally`: `[0.01, 20.0]`
  - `beta_ttb`: `[0.01, 20.0]`
  - `p_tally`: `[0.4, 0.85]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Tallying prediction
    scores_tally = np.sum(stim, axis=1)
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Take-The-Best prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_ttb_core = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Mixture
    p_tally_weight = float(parameters["p_tally"])
    p_combined = p_tally_weight * p_tally_core + (1.0 - p_tally_weight) * p_ttb_core
    
    # Lapse
    epsilon = float(parameters["epsilon"])
    n_opts = p_combined.shape[0]
    
    return (1.0 - epsilon) * p_combined + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_3_1` → slot 1 (via `new_model`)

**Description:** Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

**Rationale:** Maintained the pure Tallying mechanism but restricted the upper bound of the `beta` parameter to [0.0, 3.0] (down from 20.0) and kept `epsilon` up to 1.0. This significantly increases the stochasticity of the model on unequal-sum trials, dampening the previously overconfident predictions (e.g., bringing Experiment 3 closer to the observed 0.6333 from the simulated 0.7700), while perfectly maintaining the 0.50 prediction on equal-sum indifference trials.

**Parameters:**
  - `beta`: `[0.0, 3.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```
