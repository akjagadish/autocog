# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

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
