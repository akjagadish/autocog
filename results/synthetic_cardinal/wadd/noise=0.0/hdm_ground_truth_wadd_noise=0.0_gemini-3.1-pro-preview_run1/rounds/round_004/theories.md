# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People evaluate options by taking a weighted sum of their feature values, where the weights correspond to the subjective validities of the cues (Weighted Additive rule). This allows for fully compensatory decision making, where many small advantages can outweigh a single large advantage, and is sensitive to the cardinal magnitude of feature differences, unlike simple counting (Tallying) or one-reason heuristics (Take The Best).

**Rationale:** Following the arbiter's guidance, this proposal replaces Take The Best with a Weighted Additive (WADD) model. While TTB ignores all but the single most valid discriminating cue, and Tallying ignores cue validities and feature magnitudes entirely, WADD fully integrates both by computing a weighted sum of the raw cue values. This accommodates compensatory behavior and sensitivity to the cardinal magnitudes of feature differences, which aligns better with the observed metrics showing low TTB adherence and high (but non-perfect) Tallying adherence.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    if len(val) != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Compute the weighted sum of feature values for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return int(np.argmax(probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Equal Weights (EQW) heuristic: People evaluate multi-attribute options by taking an unweighted sum of their cardinal feature values, ignoring the provided cue validities. Unlike Tallying, which ignores magnitudes and only counts strict superiorities, EQW uses the full cardinal information but treats all features as equally important. Options are compared based on their total sum of feature values, and choices are made probabilistically via a softmax rule over these sums.

**Rationale:** Following the arbiter's recommendation, this proposes the Equal Weights (EQW) theory. EQW sums the cardinal values of all features without applying validity weights, unlike WADD (which weights by validity) and Tallying (which ignores magnitudes and only counts wins). This allows the model to capture behavior where subjects favor options with higher total sums even when those sums are driven by lower-validity cues, which is particularly relevant for explaining variance in experiments like Experiment 6 and 8.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"EQW expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Unweighted sum of cardinal feature values
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Subjective Weights and Lexicographic Semiorder: People evaluate multi-attribute options by searching through features in descending order of their subjective importance. The decision process stops at the first feature that discriminates between the options by a magnitude greater than a subjective threshold. If a difference is too small, it is treated as a tie and the search continues. An epsilon lapse rate accounts for decision noise.

**Rationale:** Following the critic's feedback, we introduce a threshold parameter for cue discrimination (Lexicographic Semiorder). This addresses the directional failures in Experiments 3, 7, and 8, where subjects often ignore very small advantages on highly valid cues if the opposing option has a much larger advantage on a slightly less valid cue. By requiring the difference to exceed a fitted threshold before a cue can dictate the decision, the model can capture these magnitude-based reversals while remaining strictly within the prescribed heuristic family. We keep the subjective weights and epsilon lapse rate exactly as they were in the accepted base.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `subj_weights`: `[(0.0, 1.0)] * n_features`
  - `threshold`: `[0.0, 5.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Use subjective weights to determine search order (avoids conflict with objective 'validities')
    val = np.asarray(parameters["subj_weights"], dtype=float)
    
    if len(val) != stim.shape[1]:
        raise ValueError("Length of subj_weights must match n_features.")
    
    # Sort features by subjective weight in descending order
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    
    ttb_pred = None
    for idx in order:
        if a[idx] - b[idx] > threshold:
            ttb_pred = 0
            break
        elif b[idx] - a[idx] > threshold:
            ttb_pred = 1
            break
            
    p = np.array([0.5, 0.5])
    if ttb_pred is not None:
        p[ttb_pred] = 1.0
        p[1 - ttb_pred] = 0.0
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
