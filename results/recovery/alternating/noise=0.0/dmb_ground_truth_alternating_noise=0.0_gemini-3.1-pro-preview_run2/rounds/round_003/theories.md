# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3_1` — KILLED ✗

**Description:** Decision makers evaluate options by computing a weighted sum of their feature values, where the weights correspond to a non-linear scaling of the subjective or instructed validities of the cues. This non-linear Weighted Additive (WADD) strategy includes an exponent parameter (gamma) on the validities, allowing the decision process to smoothly interpolate between equal weighting (Tallying, gamma=0), proportional weighting (standard WADD, gamma=1), and lexicographic one-reason decision making (Take The Best, gamma > 1). The option with the higher total weighted score is favored, with response noise modeled via a softmax function over the scores and an independent lapse rate. To capture the fact that human behavior can sometimes be entirely random, the lapse rate and inverse temperature parameters are allowed to span their full theoretical ranges (up to pure guessing).

**Rationale:** Updated the parameter ranges to enforce the 'High Noise' regime requested by the arbiter. Constraining the lapse rate (epsilon) to [0.7, 1.0] and inverse temperature (beta) to [0.0, 0.5] forces the WADD predictions to shrink toward 50/50 guessing. This allows the model to capture the slight deviations from 0.5 observed in the data while maintaining the non-linear WADD mechanism.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.7, 1.0]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    # Apply non-linear scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute the weighted sum (WADD score) for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Rationale:** Following the critic's advice, the deterministic TTB logic is preserved, but the parameter ranges for noise (beta and epsilon) are severely restricted. By constraining epsilon to [0.8, 1.0] and beta to [0.0, 0.5], the average simulated subject is forced to exhibit near-guessing behavior. This minimal edit directly addresses the previous model's overprediction of the TTB signal, bringing the expected metric down to the ~0.50 mark observed across all four experiments.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Contrarian WADD: Decision makers evaluate options by computing a weighted sum of their features, but they may distrust the provided expert ratings or view them as added complexity. Thus, they apply a scaling factor to the validities that can be negative, leading to an 'Anti-Tallying' or contrarian preference for options with lower scores. This weak contrarian signal is obscured by a very high rate of random guessing (lapse rate).

**Rationale:** Following the arbiter's suggestion, we introduce a 'Contrarian WADD' model. This theory posits that subjects may distrust the expert ratings, treating the presence of features as a negative signal (e.g., perceiving them as added complexity or untrustworthy marketing). We implement this by scaling the validities with an `alpha` parameter that spans into negative values, allowing for negative cue utilization. We combine this with the high lapse rate (`epsilon`) and low inverse temperature (`beta`) that successfully captured the baseline noise in previous iterations. This allows the model to capture the negative correlation with WADD scores observed in Experiment 7 while maintaining the ~0.5 baseline in other experiments.

**Parameters:**
  - `alpha`: `[-2.0, 1.0]`
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Contrarian WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute scores with the alpha scaling factor (which can be negative)
    score_a = np.dot(stim[0], val) * alpha
    score_b = np.dot(stim[1], val) * alpha
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
