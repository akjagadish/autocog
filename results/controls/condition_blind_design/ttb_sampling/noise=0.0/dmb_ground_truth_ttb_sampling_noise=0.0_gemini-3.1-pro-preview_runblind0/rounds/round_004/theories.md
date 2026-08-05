# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

**Rationale:** Following the arbiter's feedback, this model instantiates the Take The Best (TTB) heuristic. Instead of summing features (like Tallying or WADD), TTB sorts features by validity and makes a choice based solely on the highest-validity feature that discriminates between the options. This non-compensatory mechanism naturally accounts for the extreme sensitivity to high-validity cues observed in Experiment 2, where a single predictive feature can strongly dominate the choice over multiple weaker features.

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
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Hybrid TTB-WADD Theory: Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model computes the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blends them using a subject-level mixing parameter.

**Rationale:** Following the critic's advice, we replaced the direct multiplier `beta` with an inverse temperature parameter `tau` in [0.01, 5.0]. Inverse temperature scaling provides a smoother optimization landscape, which should help the fitter find the optimal subject-specific `mix_rate` without getting stuck in sub-optimal compromises. We also constrained `epsilon` to [0.0, 0.05] to prevent random noise from artificially flattening the deterministic TTB predictions.

**Parameters:**
  - `tau`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.05]`
  - `mix_rate`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- TTB Probability ---
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    ttb_p = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_p = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_p = np.array([0.0, 1.0])
            break
            
    # --- WADD Probability ---
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    wadd_scores = np.array([score_a, score_b])
    
    tau = float(parameters["tau"])
    z = (wadd_scores - np.max(wadd_scores)) / tau
    e = np.exp(z)
    wadd_p = e / np.sum(e)
    
    # --- Blend ---
    mix_rate = float(parameters["mix_rate"])
    p_core = mix_rate * ttb_p + (1.0 - mix_rate) * wadd_p
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Probabilistic Stopping Take-The-Best (Probabilistic TTB). Decision-makers search cues sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue, they stop and choose the favored option with probability `p_stop`. With probability `1 - p_stop`, they continue searching and accumulate evidence from subsequent cues. A high p_stop ensures the model behaves primarily in a non-compensatory manner, matching human data.

**Rationale:** Following the critic's feedback, I have tightened the parameter ranges for `p_stop` from [0.1, 1.0] to [0.6, 1.0] and for `epsilon` from [0.0, 0.5] to [0.0, 0.25]. This ensures the model predominantly stops at the first discriminating cue, reducing its compensatory nature to better align with the strong non-compensatory choice patterns observed in the empirical data, while also limiting excessive random choice noise.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.25]`
  - `p_stop`: `[0.6, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_stop = float(parameters["p_stop"])
    
    score_a = 0.0
    score_b = 0.0
    remaining = 1.0
    
    for idx in order:
        if a[idx] > b[idx]:
            score_a += remaining * p_stop
            remaining *= (1.0 - p_stop)
        elif b[idx] > a[idx]:
            score_b += remaining * p_stop
            remaining *= (1.0 - p_stop)
            
    score_a += remaining * 0.5
    score_b += remaining * 0.5
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
