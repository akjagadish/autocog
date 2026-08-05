# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use the 'Take The Best' (TTB) heuristic to make binary choices. They search through features in descending order of their subjective or stated validities. The first feature that discriminates between the two options (i.e., one option has a positive feature and the other does not) strictly determines the choice, ignoring all other features. If no features discriminate, they guess uniformly. Response noise is modeled via a simple lapse rate where the individual occasionally guesses randomly instead of following the deterministic rule.

**Rationale:** Following the arbiter's recommendation, this model instantiates the Take The Best (TTB) heuristic. It orders features by their validities and makes a strict choice based on the first discriminating feature, acting as a non-compensatory lexicographic rule. This prevents lower-validity cues from overriding the most valid available cue. A lapse rate 'epsilon' is added to account for response noise and unmodeled errors.

**Parameters:**
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
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    epsilon = float(parameters["epsilon"])
    
    # Default to uniform guess if no features discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Equal-Weight (Tallying) Heuristic: Decision-makers simplify choices by ignoring the differential validity of features. Instead, they simply count the total number of positive features (1s) for each option and choose the option with the higher count. If both options have the same number of positive features, they guess randomly. Response variability is modeled via a softmax over these counts and an independent random lapse rate.

**Rationale:** Following the arbiter's suggestion, this theory implements the Equal-Weight (Tallying) heuristic. Unlike WADD, it ignores feature validities and simply counts the number of positive attributes for each option. Unlike Take-The-Best, it considers all features equally rather than stopping at the first discriminating cue. The difference in positive feature counts drives the choice probabilistically via a softmax function, with a lapse rate to account for motor or attentional noise. This provides a formal instantiation of the prescribed boundedly rational heuristic benchmark.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal-Weight expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    # Count the number of positive features for each option
    a_count = float(np.sum(a > 0))
    b_count = float(np.sum(b > 0))
    scores = np.array([a_count, b_count])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax over the total counts
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Incorporate uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) Theory: Decision-makers integrate all available information by weighting each feature according to its validity. Instead of ignoring feature validities (like Equal-Weight) or ignoring all but the best feature (like Take The Best), this compensatory heuristic transforms the stated validities into log-odds weights to represent the amount of evidence each feature provides. The overall subjective value of an option is computed by summing the weights of its positive features. Choice probabilities are then generated by applying a softmax function to these values, meaning multiple weak cues can potentially outweigh a single strong cue. Response variability is captured via the softmax temperature and an independent random lapse rate.

**Rationale:** Following the arbiter's suggestion, this theory implements a compensatory Weighted Additive (WADD) model. It improves upon the Equal-Weight heuristic by incorporating the feature validities, and it contrasts with Take The Best by allowing multiple features to compensate for one another. The validities are transformed into log-odds weights, which is a standard and mathematically principled way to represent evidence in binary decision tasks (equivalent to Naive Bayes evidence accumulation). A softmax function with an inverse temperature parameter transforms the resulting scores into choice probabilities, while a lapse parameter accounts for random errors.

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
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities using log-odds for principled evidence weighting.
    # Clip to avoid log(0) or division by zero for validities of 1.0.
    v_clipped = np.clip(validities, 0.501, 0.999)
    weights = np.log(v_clipped / (1.0 - v_clipped))
    
    # Calculate the weighted additive score for each option
    score_a = float(np.sum(a * weights))
    score_b = float(np.sum(b * weights))
    scores = np.array([score_a, score_b])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax over the total scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Incorporate uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
