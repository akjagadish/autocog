# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tallying (Equal Weights) Theory with Very Soft Softmax: Decision-makers evaluate options by assigning equal weight to all features, simply counting the number of positive features for each option. However, their choices are not highly deterministic; they employ a 'very soft' decision rule where the difference in tallies yields only a mild preference (e.g., ~55/45 splits) rather than an overwhelming one, reflecting high levels of noise, uncertainty, or guessing in human decision-making in this domain.

**Rationale:** Following the latest feedback, the core Tallying logic remains exactly the same. The only change is restricting the `beta` parameter range to [0.01, 0.5] and keeping `epsilon` at [0.0, 0.1]. This forces the model into an even softer regime, reducing the extremity of the predictions (from ~0.25/0.75 to ~0.40/0.60) to better match the empirical data.

**Parameters:**
  - `beta`: `[0.01, 0.5]`
  - `epsilon`: `[0.0, 0.1]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Equal Weights expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Linear Weighted Additive Theory: Decision makers evaluate options by summing the features of each option, weighted linearly by their predictive validity (specifically, the validity's edge over chance, validity - 0.5). This represents a moderate integration strategy that avoids the extreme dominance of top cues seen in log-odds (Bayesian) weighting, while still differentiating cue importance unlike Equal-Weight/Tallying.

**Rationale:** Following the arbiter's suggestion, this theory implements a Linear Weighted Additive model. Instead of transforming validities into log-odds (which heavily polarizes weights and over-privileges the highest-validity cues) or ignoring validities entirely (as in Tallying), subjects weight each cue by its raw predictive edge over chance (validity - 0.5). This allows a coalition of lower-validity cues to more easily outvote a single high-validity cue, producing intermediate choice proportions that better match the empirical data across the experiments.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Linear WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to linear weights (edge over chance)
    weights = val - 0.5
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Validity-Biased Tallying: Decision-makers primarily rely on a simple Tallying (Equal Weights) heuristic, counting the number of positive features for each option. However, they retain a residual sensitivity to cue validity, applying a small lexicographic bonus to the option favored by the single most valid discriminating cue (Take-The-Best). This bonus serves to break ties when tallies are equal and slightly shifts preferences when tallies are close, implemented via a soft decision rule.

**Rationale:** Following the arbiter's suggestion, this theory implements 'Validity-Biased Tallying'. It builds upon the highly successful noisy Tallying model (pi_5) by adding a lexicographic bonus (Take-The-Best) scaled by a weight parameter 'w'. This allows the model to capture residual sensitivity to validity—such as breaking ties using the most valid cue—without falling into the trap of over-weighting validity like the fully linear or log-odds WADD models. This hybrid approach tests whether participants use a fast-and-frugal fallback when their primary tallying strategy is uninformative or close.

**Parameters:**
  - `beta`: `[0.01, 1.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `w`: `[0.0, 1.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Validity-Biased Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Determine Take-The-Best winner
    val = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(val)[::-1]
    
    diff = a - b
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if diff[idx] > 0:
            ttb_a = 1.0
            break
        elif diff[idx] < 0:
            ttb_b = 1.0
            break
            
    # Tallying score with TTB bonus
    w = float(parameters["w"])
    score_a = np.sum(a) + w * ttb_a
    score_b = np.sum(b) + w * ttb_b
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the combined scores
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```
