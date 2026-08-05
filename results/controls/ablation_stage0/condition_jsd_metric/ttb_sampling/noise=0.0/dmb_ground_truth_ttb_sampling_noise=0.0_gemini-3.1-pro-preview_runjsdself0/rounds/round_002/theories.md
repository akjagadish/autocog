# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: People employ a non-compensatory, lexicographic decision process. They evaluate features sequentially in descending order of their subjective validity. The very first feature that discriminates between the two options (i.e., one option has a positive feature value while the other does not) determines the choice, and all lower-validity features are strictly ignored. If no feature discriminates, they guess. Response noise is modeled via a softmax over the resulting binary preference and an independent random lapse rate.

**Rationale:** Following the arbiter's recommendation, this theory replaces the compensatory and equal-weight mechanisms of WADD and Tallying with the non-compensatory Take-The-Best (TTB) heuristic. It explicitly models the lexicographic evaluation of features sorted by their validities, stopping at the first discriminating feature. This captures human behavior in environments where feature validities are highly skewed and single compelling reasons often drive decisions.

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
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Evaluate features sequentially
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores to allow for noise
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Tallying (Equal Weight) Heuristic: Decision-makers evaluate options by simply counting the total number of positive features for each option, treating all features as equally important and completely ignoring their cue validities. The option with the higher total count is chosen. If the counts are tied, the decision-maker guesses randomly. Choice behavior is subject to response noise modeled via a softmax function over the tally scores, along with an independent random lapse rate.

**Rationale:** Following the arbiter's recommendation, this model implements the Tallying heuristic. Unlike WADD (which weights features by validity) or Take-The-Best (which sequentially considers features by validity), Tallying is a fully compensatory but unweighted strategy. It assumes people simply sum the positive features of each option. This provides a strong baseline to test whether compensatory behavior exists when cue validities are entirely stripped away.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Count the number of positive features for each option
    scores = np.array([np.sum(a), np.sum(b)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores to allow for noise
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) with Validity-Proportional Weights: Subjects make decisions by computing a compensatory weighted sum of features for each option. Unlike a naive tallying strategy, features are weighted by their subjective importance, which is directly derived from their known validities (e.g., raw validity, validity above chance, or log-odds). This allows multiple weak cues to overcome a single strong cue, contrasting with non-compensatory heuristics like Take-The-Best.

**Rationale:** The arbiter noted that Tallying ignores cue validities, while the previous WADD implementation overfit by using entirely free parameters for feature weights. This new Weighted Additive (WADD) theory ties the feature weights directly to the provided validities, as suggested. To account for different plausible psychological transformations of validity into subjective weight, the model includes a `weight_type` parameter allowing weights to be proportional to the raw validity, the validity above chance (v - 0.5), or the Bayesian log-odds. This creates a strong, constrained compensatory baseline to test whether decision-making in this domain is truly non-compensatory (TTB) or if lower-validity cues can collectively outweigh a higher-validity cue.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `weight_type`: `{"log_odds", "validity", "validity_minus_half"}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    validities = np.asarray(parameters["validities"], dtype=float)
    # Clip to avoid division by zero or log(0) for validities of 1.0 or 0.5
    v = np.clip(validities, 0.5001, 0.9999)
    
    wt = parameters["weight_type"]
    if wt == "log_odds":
        w = np.log(v / (1.0 - v))
    elif wt == "validity":
        w = v
    else:  # validity_minus_half
        w = v - 0.5
        
    # Compute weighted sum for each option
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Epsilon-greedy lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
