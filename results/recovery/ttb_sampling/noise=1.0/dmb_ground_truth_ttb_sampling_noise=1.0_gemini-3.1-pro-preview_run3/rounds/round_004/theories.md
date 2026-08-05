# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.

**Rationale:** Following the arbiter's feedback, this theory implements a Noisy Tallying heuristic. Instead of sorting features by validity (TTB) or computing weighted sums (WADD), the model simply counts the total number of positive features for each option. To capture the extreme smoothing seen in the data (where choices hover around 0.5 and differences between conditions are minimal), the model uses a high lapse rate (epsilon in [0.5, 1.0]) and a low inverse temperature (beta in [0.0, 0.5]). This ensures that the predictions are only slightly biased toward the option with more positive features, matching the aggregate empirical patterns.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.5, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying: Sum of active features for each option (ignoring validities)
    scores = np.sum(stim, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

**Rationale:** Following the critic's advice, the parameter ranges for beta and epsilon have been tightened to [0.0, 0.5] and [0.5, 1.0] respectively. This minimal edit further flattens the predictions toward 0.5 to eliminate residual WADD-driven signal, better matching the human data's near-perfect 50/50 guessing behavior across these experiments.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Noisy Take-The-Best (TTB) with High Feature-Ordering Noise: Decision-makers process information sequentially, inspecting features in descending order of their perceived validities. They stop at the first feature that discriminates between the options and favor the option it supports. To account for choice probabilities being close to 0.5 across experiments, perceived validities are subject to high Gaussian noise, allowing the search to degrade toward random-order inspection. Choices are further smoothed by a high uniform lapse rate and a soft-maximization process.

**Rationale:** Following the critic's advice, we revert the `epsilon` parameter range to [0.5, 1.0] to enforce the high lapse rate necessary to match the empirical ~0.5 choice probabilities. We keep the `sigma` range restricted to [0.0, 3.0] to avoid washing out validity differences too aggressively, and we increase `n_samples` from 50 to 200 to reduce simulation variance. This smoother expected-value surface should help the optimizer refine the best baseline loss without falling into overly deterministic local minima.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.5, 1.0]`
  - `sigma`: `[0.0, 3.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma = float(parameters["sigma"])
    
    n_samples = 200
    avg_p_core = np.zeros(2)
    
    for _ in range(n_samples):
        # Probabilistic feature-ordering noise
        noisy_v = validities + np.random.normal(0, sigma, size=validities.shape)
        order = np.argsort(noisy_v)[::-1]
        
        scores = np.zeros(2)
        # Search for the first discriminating feature
        for idx in order:
            if stim[0, idx] > stim[1, idx]:
                scores[0] = 1.0
                break
            elif stim[1, idx] > stim[0, idx]:
                scores[1] = 1.0
                break
                
        if np.sum(scores) == 0:
            p_core = np.array([0.5, 0.5])
        else:
            # Softmax with max-subtraction for numerical stability
            z = beta * scores
            e = np.exp(z - np.max(z))
            p_core = e / np.sum(e)
            
        avg_p_core += p_core
        
    avg_p_core /= n_samples
    
    # Blend with uniform lapse rate
    n_opts = len(avg_p_core)
    return (1.0 - epsilon) * avg_p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
