# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

**Rationale:** Following the arbiter's feedback, this theory implements the Take-The-Best (TTB) heuristic, replacing the Tallying model. TTB is a non-compensatory, lexicographic strategy that checks cues in descending order of their validity and stops at the first discriminating cue. This provides a strong contrast to compensatory models like WADD and equal-weighting models like Tallying, allowing us to evaluate whether participants rely on a single highly valid cue rather than integrating all available information when making choices.

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
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Two-Stage Confidence-Threshold Strategy Selection: Decision-makers default to the fast and frugal Take-The-Best (TTB) heuristic, evaluating options based solely on the most valid discriminating cue. However, if the validity of this top discriminating cue falls below a subjective confidence threshold, the decision-maker deems the single-cue evidence insufficient and falls back to a compensatory Weighted Additive (WADD) strategy that integrates all available cues.

**Rationale:** Following the critic's advice, I tightened the range of `confidence_threshold` from `[0.5, 1.0]` to `[0.5, 0.8]`. This ensures the model defaults to TTB more frequently and only falls back to WADD when the top discriminating cue is genuinely weak, thereby reducing over-prediction of compensatory choices and improving TTB match rates.

**Parameters:**
  - `confidence_threshold`: `[0.5, 0.8]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    confidence_threshold = float(parameters["confidence_threshold"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    scores = np.zeros(2)
    if np.any(discrim_mask):
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        if max_v >= confidence_threshold:
            # Strategy 1: Take-The-Best (TTB)
            top_idx = np.where((discrim_mask) & (validities == max_v))[0][0]
            if stim[0, top_idx] > stim[1, top_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
        else:
            # Strategy 2: Weighted Additive (WADD) fallback
            wadd_scores = stim @ validities
            if wadd_scores[0] > wadd_scores[1]:
                scores[0] = 1.0
            elif wadd_scores[1] > wadd_scores[0]:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially, but the search order is probabilistic rather than deterministic. The probability of evaluating a cue is weighted by a softmax over its validity. The first cue encountered that discriminates between the options completely determines the choice. This maintains the fast-and-frugal, non-compensatory nature of Take-The-Best, but allows for systematic deviations that depend on how close the cue validities are to one another.

**Rationale:** Expanded the parameter range for `gamma` to `[0.0, 200.0]` as requested by the critic. This allows the probabilistic cue search to approximate deterministic Take-The-Best (TTB) more closely when required by the data, fitting the high determinism observed in several experiments while maintaining the principled variability of the PCS mechanism.

**Parameters:**
  - `gamma`: `[0.0, 200.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # The probability of a discriminating cue being the *first* discriminating cue encountered
        # in a sampling-without-replacement process is proportional to its weight relative to all
        # other discriminating cues.
        discrim_validities = validities[discrim_mask]
        z = gamma * discrim_validities
        z -= np.max(z) # for numerical stability
        weights = np.exp(z)
        
        full_weights = np.zeros_like(validities)
        full_weights[discrim_mask] = weights
        
        w_A = np.sum(full_weights[diff > 0])
        w_B = np.sum(full_weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
