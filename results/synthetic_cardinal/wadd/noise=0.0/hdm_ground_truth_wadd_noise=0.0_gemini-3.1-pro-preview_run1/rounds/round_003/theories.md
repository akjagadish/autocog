# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Tallying (Majority of Confirming Dimensions) heuristic: People evaluate multi-attribute options by counting the number of features on which each option is strictly superior to the other. They ignore both the validities of the features and the cardinal magnitudes of the differences, choosing the option that wins on the most features.

**Rationale:** Replaced `np.argmax` with `np.random.choice` in the policy function to enable probabilistic sampling. Initial logic and parameters are validated. Standard processing applied to the Tallying mechanism. Parameter ranges adjusted to beta=[0.0, 5.0] and epsilon=[0.05, 0.5] to ensure graded responses.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.05, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Count strict superiority
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    scores = np.array([tally_a, tally_b], dtype=float)
    
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

### `pi_6` → slot 2 (via `new_theory`)

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
