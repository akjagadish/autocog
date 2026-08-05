# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3_1` — SURVIVED ✓

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Cancellation and Focus (Ratio Rule)

**Rationale:** Following the arbiter's guidance, this model implements the 'Cancellation and Focus' heuristic. Subjects first simplify the decision by eliminating (canceling) any features that are shared between the two options. They then focus only on the remaining unique positive features. For binary features, taking the difference of unique features is mathematically identical to taking the difference of total features (Tallying). To ensure this theory is genuinely psychologically and mathematically distinct from Theory 1, choice probabilities are determined by a ratio rule (Luce's choice) over the unique features, scaled by a sensitivity parameter beta. This completely ignores cue validities and perfectly captures the empirical evidence that subjects guess 50/50 when feature counts are tied (since tied total counts imply tied unique counts).

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Cancellation: eliminate features shared by both options.
    # Focus only on unique positive features.
    unique_a = np.sum((stim[0] > stim[1]).astype(float))
    unique_b = np.sum((stim[1] > stim[0]).astype(float))
    
    scores = np.array([unique_a, unique_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if unique_a == 0 and unique_b == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Ratio rule over unique features to provide a distinct alternative to Tallying's softmax.
        s_beta = np.zeros_like(scores)
        for i in range(len(scores)):
            if scores[i] > 0:
                s_beta[i] = scores[i] ** beta
            else:
                s_beta[i] = 0.0 if beta > 0 else 1.0
        p_core = s_beta / np.sum(s_beta)
        
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Normalized Weighted Additive (WADD) Model: Decision-makers integrate all available features, weighting each by its objective cue validity normalized to sum to 1. This ensures that the scale of the integrated scores is invariant to the number of features, stabilizing the choice temperature across different experimental contexts. The option with the highest normalized validity-weighted sum is chosen.

**Rationale:** Following the critic's advice, we revert to the linear WADD framework but introduce weight normalization (`weights = validities / np.sum(validities)`). This keeps the proportional weighting of pure WADD but ensures the maximum possible score is stabilized across all experiments, regardless of the number of cues. This normalization should allow the softmax temperature (`beta`) to apply more consistently across different experimental designs, leading to a better fit.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Normalize validities so they sum to 1
    weights = validities / np.sum(validities)
    
    # WADD: score is the sum of features weighted by their normalized validities
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```
