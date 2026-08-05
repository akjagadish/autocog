# Round 4 — Theories

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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Two-Stage Take-The-Best and WADD Heuristic: Decision makers employ a boundedly rational, two-stage process. In the first stage, they act strictly non-compensatory by checking only the single most valid cue (like Take-The-Best). If this primary cue discriminates between the options, the decision is made immediately. However, if the most valid cue is tied, they fall back to a compensatory process, computing a Weighted Additive (WADD) score of the remaining cues to break the tie. This decouples the primary cue's overriding influence from the secondary cues' collective weighting.

**Rationale:** Addresses the critic's feedback by making the two stages truly sequential and fully decoupling their probability calculations. Instead of assigning values to a single scores array and applying softmax once at the end (which risks scale mismatch or accidental mixing of stages), the model computes and returns the softmax probabilities immediately inside the active stage. If the highest-validity cue discriminates, it determines the choice probabilities exclusively. If it ties, the model computes an unnormalized validity-weighted sum (WADD) of the remaining cues and applies the softmax to those fallback scores, allowing secondary cues to confidently resolve ties.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Two-stage expects a (2, n_features) stimulus.")

    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    best_cue = order[0]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def apply_noise(scores):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    
    # Stage 1: Check the single most valid cue
    if stim[0, best_cue] != stim[1, best_cue]:
        scores = np.zeros(2)
        if stim[0, best_cue] > stim[1, best_cue]:
            scores[0] = 1.0
        else:
            scores[1] = 1.0
        return apply_noise(scores)
    
    # Stage 2: Fallback to WADD of the remaining cues if tied
    remaining_cues = order[1:]
    if len(remaining_cues) > 0:
        scores = np.sum(stim[:, remaining_cues] * validities[remaining_cues], axis=1)
    else:
        scores = np.zeros(2)
        
    return apply_noise(scores)
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Tallying (Equal Weights) Heuristic: Decision makers use a simplified compensatory strategy where they ignore cue validities entirely. Instead, they simply count the number of positive features (cues) for each option and choose the one with the higher tally. This represents a highly bounded rationality approach where the cognitive cost of weighting cues by their validity is avoided. Response noise is modeled via a softmax over the tallies and an independent random lapse rate.

**Rationale:** Following the arbiter's suggestion to establish a pure compensatory baseline, this model implements the Tallying (Equal Weights) heuristic. Unlike WADD (which scales features by validity) or TTB (which uses a non-compensatory lexicographic rule), Tallying simply counts the number of positive cues for each option. This provides a clean, zero-parameter-weighting compensatory contrast to the highly successful non-compensatory models, allowing us to evaluate whether a subset of the population relies on this highly simplified integration strategy.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Tallying: sum the positive features for each option, ignoring validities
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Epsilon-greedy lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
