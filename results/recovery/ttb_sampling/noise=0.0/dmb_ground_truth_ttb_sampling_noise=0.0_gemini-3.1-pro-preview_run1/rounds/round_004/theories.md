# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a non-compensatory, lexicographic heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of their subjective validity and stop at the first feature that discriminates between the two options (i.e., one option has a positive rating and the other does not). The choice is based entirely on this single discriminating cue, completely ignoring all remaining features, regardless of how many lower-validity cues might favor the alternative.

**Rationale:** Following the arbiter's recommendation, this model implements the 'Take The Best' (TTB) heuristic. Unlike Tallying (which ignores validities) and WADD (which integrates all validities in a compensatory manner), TTB is a non-compensatory lexicographic strategy. It searches features in descending order of validity and makes a decision based solely on the first discriminating cue. This directly tests whether subjects are relying on single high-validity cues rather than integrating all available information. Response noise is handled via a uniform lapse rate (epsilon) applied to the deterministic TTB choice.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    # Deterministic choice based on the first discriminating cue
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # If all features tie, guess randomly
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse rate for noise
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Softmax-Weighted Additive Model: Subjects use a fully compensatory integration process where they calculate a weighted sum of all features, but they subjectively scale the stated cue validities using a softmax transformation. By exponentiating and normalizing the validities with a scaling parameter, differences between high and low validity cues are amplified without shrinking the overall magnitudes of the weights. This allows a purely compensatory, additive mechanism to rival and approximate strict non-compensatory behavior (like Take-The-Best) when the scaling parameter is sufficiently large, while still integrating all available information.

**Rationale:** Following the critic's advice, we increased the upper bound of the `gamma` parameter to 1000.0. When validities are numerically very close, a smaller gamma (e.g., 50) fails to sufficiently separate the softmax weights, leaving the model too compensatory. A much larger gamma allows the model to amplify minute differences between validities, pushing the softmax weights toward a one-hot distribution and allowing the Weighted Additive mechanism to truly mimic strict non-compensatory behavior when necessary.

**Parameters:**
  - `gamma`: `[0.0, 1000.0]`
  - `beta`: `[1.0, 50.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Model expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax transformation of validities to amplify importance of top cues
    # without shrinking the overall magnitudes of the weights to zero.
    z_v = gamma * validities
    z_v = z_v - np.max(z_v)
    e_v = np.exp(z_v)
    weights = e_v / np.sum(e_v)
    
    # Compensatory integration (Weighted Additive)
    scores = stim @ weights

    # Softmax decision rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Blended with a uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Probabilistic Take-The-Best (Noisy Cue Search): Decision makers employ a strictly non-compensatory lexicographic strategy, stopping search at the first discriminating cue. However, their subjective representation of cue validities is noisy. On each trial, Gaussian noise perturbs the true validities, which occasionally induces rank-reversals in the cue search order—especially among cues with similar validities. This provides a cognitively plausible mechanism for choice errors, preserving the core lexicographic dynamic while gracefully explaining deviations from strict deterministic Take-The-Best without relying solely on a uniform lapse rate.

**Rationale:** Following the critic's feedback, the core Probabilistic Take-The-Best mechanism is maintained exactly as before, but the parameter ranges for the noise terms have been tightened. Restricting `sigma` to [0.0, 0.15] and `epsilon` to [0.0, 0.1] ensures that rank-reversals in the cue search order primarily occur between cues with very similar validities, significantly reducing the excessive noise seen in the previous iteration and better capturing the high accuracy observed in the ground-truth data.

**Parameters:**
  - `sigma`: `[0.0, 0.15]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Model expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    epsilon = float(parameters["epsilon"])
    
    n_samples = 200
    p_core = np.zeros(2)
    diff = a - b
    
    for _ in range(n_samples):
        # Add Gaussian noise to validities to form a subjective hierarchy for this trial
        noisy_v = validities + np.random.normal(0, sigma, size=len(validities))
        order = np.argsort(noisy_v)[::-1]
        
        chosen = -1
        for idx in order:
            if diff[idx] > 0:
                chosen = 0
                break
            elif diff[idx] < 0:
                chosen = 1
                break
                
        if chosen == 0:
            p_core[0] += 1.0
        elif chosen == 1:
            p_core[1] += 1.0
        else:
            p_core += 0.5
            
    p_core /= n_samples
    
    # Blend with a small uniform lapse rate for pure motor/attention errors
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
