# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: Individuals make decisions by sequentially searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the choice is based entirely on that single feature, ignoring all others. If no feature discriminates, a random guess is made.

**Rationale:** Following the arbiter's instructions, this model implements the Take-The-Best (TTB) heuristic. Unlike Tallying (which counts winning features equally) or WADD (which integrates all features compensatorily), TTB is a one-reason decision heuristic. It sorts features by their objective validities in descending order and evaluates them sequentially. The first feature that discriminates between the two options completely determines the decision, and all subsequent features are ignored. If no features discriminate, the decision defaults to a tie (equal scores), resulting in a uniform random guess. Softmax temperature and a uniform lapse rate are included to account for response noise.

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
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            scores = np.array([0.0, 1.0])
            break
            
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Probabilistic Take-The-Best (PTTB): Decision-makers search through features sequentially to find the first one that discriminates between options. However, rather than searching in a strict deterministic order based on validities, the search order is stochastic. Features are sampled with probabilities proportional to an exponential function of their validities, allowing for noise and slight deviations from strict TTB. The search stops at the first discriminating feature.

**Rationale:** Following the arbiter's suggestion, this theory instantiates a Probabilistic Take-The-Best (PTTB) model. Instead of a strictly deterministic search order, the decision-maker samples features sequentially without replacement, with probabilities proportional to an exponential function of their validities (controlled by a sensitivity parameter `gamma`). The search stops at the first discriminating feature. Mathematically, by the properties of the Plackett-Luce choice model, the probability that a specific discriminating feature is encountered *first* among all discriminating features is exactly proportional to its selection weight. This allows us to compute the exact choice probabilities analytically without simulating the sequential search. The model elegantly interpolates between Tallying on discriminating features (when gamma = 0) and strict TTB (when gamma is large), thus directly addressing the arbiter's recommendations while maintaining mathematical tractability.

**Parameters:**
  - `gamma`: `[0.0, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating features
    diff = a - b
    D = diff != 0
    
    if not np.any(D):
        p_core = np.array([0.5, 0.5])
    else:
        # Under a Plackett-Luce sequential sampling model where items are sampled 
        # proportional to w_i, the probability that a specific item in D is sampled 
        # *first* among all items in D is exactly w_i / sum(w_j for j in D).
        v_D = validities[D]
        
        # Max-subtraction for numerical stability
        w = np.exp(gamma * (v_D - np.max(v_D)))
        w_sum = np.sum(w)
        
        # Probability of choosing A is the sum of weights of features favoring A, divided by w_sum
        p_A = np.sum(w[diff[D] > 0]) / w_sum
        p_B = np.sum(w[diff[D] < 0]) / w_sum
        p_core = np.array([p_A, p_B])
        
    n_opts = 2
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Non-linear Weighted Additive (WADD) Model: Decision-makers evaluate options in a compensatory manner by integrating all available information. However, the weighting of features can be non-linear. Each option's overall value is computed as the sum of its feature values weighted by their exponentiated and normalized validities. The exponentiation parameter allows the model to flexibly capture behavior ranging from equal-weighting to strictly non-compensatory (lexicographic) feature weighting. The probability of choosing an option is then determined by passing these integrated values through a softmax function with a noise parameter.

**Rationale:** Following the critic's advice, we introduce a non-linear scaling parameter `gamma` to exponentiate the validities before normalization. This allows the model to flexibly span the spectrum from equal-weighting (gamma < 1) to linear WADD (gamma = 1) to non-compensatory TTB-like lexicographic weighting (gamma > 1), thereby better capturing the nuances of human decision-making across experiments while remaining within the WADD framework.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Apply non-linear scaling to validities to allow for non-compensatory weighting
    validities = validities ** gamma
    
    # Normalize validities so they sum to 1 to maintain consistent scaling across experiments
    validities = validities / np.sum(validities)
    
    # Compute weighted sum of features for each option
    val_a = np.sum(a * validities)
    val_b = np.sum(b * validities)
    scores = np.array([val_a, val_b])
    
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
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
