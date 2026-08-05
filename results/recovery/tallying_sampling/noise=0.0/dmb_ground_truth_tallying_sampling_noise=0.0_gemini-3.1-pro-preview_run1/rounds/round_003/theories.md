# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People make decisions using a 'Tallying' or 'Equal Weight' heuristic. Instead of weighting features by their validity or importance, decision-makers simply count the total number of positive features for each option. They choose the option with the highest tally, effectively ignoring the provided cue validities. This explains why participants consistently choose options with more positive cues even when a single higher-validity cue points in the opposite direction.

**Rationale:** Following the arbiter's feedback, this theory implements Tallying (Equal Weights). By simply summing the number of positive features for each option and ignoring their validities, the model systematically prefers options with more positive cues. This perfectly accounts for the extremely low TTB-consistency metrics observed in the data, as participants routinely choose the option favored by the sheer count of cues rather than the single most valid cue.

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
    
    # Tallying: sum the positive features for each option, ignoring validities.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Take-The-Best (TTB) Heuristic: Decision-makers rely on a non-compensatory lexicographic process. They search through features in descending order of their cue validity, stopping at the very first feature that discriminates between the two options (where one option has a positive feature and the other does not). They then choose the option favored by this single cue, ignoring all less valid features. If no feature discriminates, they guess.

**Rationale:** Following the arbiter's recommendation, this proposes the Take-The-Best (TTB) heuristic. TTB is a non-compensatory lexicographic strategy where decision-makers consider features one at a time in descending order of validity. The search stops at the first feature that discriminates between the two options, and the option favored by that feature is chosen. This provides a mathematically distinct prediction from Tallying or WADD by ignoring all less valid features once a discriminating cue is found. Choice stochasticity is captured via a uniform lapse rate (epsilon).

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity
    order = np.argsort(validities)[::-1]
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

**Description:** Weighted Additive (WADD) Model. Decision-makers compute a weighted sum of the positive features for each option, using the provided cue validities as weights. They then choose the option with the higher weighted sum. This serves as a strong compensatory, rational baseline to rigorously test whether participants are partially or fully integrating validities, as opposed to ignoring them (Tallying) or using a non-compensatory rule (Take The Best).

**Rationale:** The arbiter prescribed the Weighted Additive (WADD) model to serve as a strong compensatory baseline. Unlike Tallying, which gives equal weight to all cues, WADD weights each cue by its validity. Unlike a fitted WADD model (such as pi_2 which fits subjective weights per feature), this strict implementation directly uses the objective cue validities provided in the instructions as weights. This model directly tests the hypothesis that participants integrate the provided validities in a compensatory manner to evaluate options, incorporating standard softmax choice noise and lapses.

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
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: sum of validities for positive features
    scores = stim @ validities
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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
