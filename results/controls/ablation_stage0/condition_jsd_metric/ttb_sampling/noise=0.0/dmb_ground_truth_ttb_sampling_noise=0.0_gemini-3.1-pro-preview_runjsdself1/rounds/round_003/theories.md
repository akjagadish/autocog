# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Weighted Additive (WADD) Model with Log-Odds: Decision-makers evaluate options using a purely compensatory strategy. They compute an overall value for each option by summing the log-odds of the validities for all features where the option has a positive cue. A choice is then made probabilistically based on the difference in these overall values, using a softmax rule combined with a uniform lapse rate to account for decision noise.

**Rationale:** Following the arbiter's recommendation, this proposes a strong, classical compensatory alternative to the non-compensatory Take-The-Best model. By implementing a Weighted Additive (WADD) model that directly derives its weights from the log-odds of the provided validities, we avoid the overfitting associated with learning free weights for each feature. Decision-makers sum these log-odds weights for all positive cues and make a probabilistic choice based on the resulting values.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities into log-odds weights
    v = np.clip(validities, 1e-5, 1.0 - 1e-5)
    w = np.log(v / (1.0 - v))
    
    # Compute the overall value (weighted sum) for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

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
