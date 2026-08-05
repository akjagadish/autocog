# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

**Rationale:** The arbiter suggested replacing Take The Best (TTB) with Tallying (Equal Weight heuristic). In Tallying, subjects ignore explicit cue validities and simply count the number of positive features for each option. The option with the higher tally is chosen. This strongly compensatory heuristic easily overrides the single most valid cue if it is outnumbered by other positive cues. This perfectly explains why subjects in Experiment 1 heavily anti-align with TTB (they choose the option with more positive cues, even if it lacks the highest-validity cue) and why subjects in Experiment 2 align closely with WADD when it disagrees with TTB (since WADD and Tallying both favor the option with a greater number of less-valid cues over a single highly-valid cue).

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Majority of Confirming Dimensions (MCD): People make decisions by comparing options dimension by dimension rather than computing holistic values. On each feature, the option with the superior value earns a point (a 'confirming dimension'). The option with the most points across all features is chosen. For binary features, this dimension-wise comparison yields the same relative preference as Tallying, but through a fundamentally different cognitive process (pairwise comparisons rather than independent counting).

**Rationale:** Following the arbiter's suggestion, we introduce the Majority of Confirming Dimensions (MCD) heuristic. Rather than evaluating options holistically (like WADD or Tallying), MCD compares options dimension-by-dimension. It counts how many features strictly favor option A versus option B. In environments with binary features, this boundedly rational, non-compensatory pairwise comparison mathematically mirrors the relative evidence of Tallying, thus perfectly capturing the human data while offering a distinct, cognitively plausible mechanistic explanation based on dimensional comparisons.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("MCD expects a (2, n_features) state.")
        
    a, b = stim[0], stim[1]
    
    # MCD: Count how many features strictly favor each option
    score_a = np.sum(a > b)
    score_b = np.sum(b > a)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over MCD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Weighted Additive Strategy (WADD): Decision makers use a fully compensatory strategy, integrating all available information by computing a weighted sum of feature values for each option, where each feature is weighted by its explicit validity. The option with the highest overall value is chosen.

**Rationale:** Implemented the Weighted Additive Strategy (WADD) as requested. WADD is fully compensatory and integrates all available information by taking the dot product of the cue validities and the feature vectors for each option. This directly addresses the failure of the non-compensatory TTB model to capture the empirical evidence of compensatory behavior across the experiments.

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
        raise ValueError("WADD expects a (2, n_features) state.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: Dot product of feature vectors and validities
    scores = stim @ validities
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
