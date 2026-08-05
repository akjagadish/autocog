# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a Weighted Additive (WADD) strategy to compare options. They compute a subjective value for each option by summing its feature values weighted by their explicit validities. Decisions are made probabilistically using a softmax function over these values, with occasional random lapses.

**Rationale:** Implements the Weighted Additive (WADD) model as prescribed by the arbiter. By weighting each feature by its given validity, WADD integrates all available information in a compensatory manner. This contrasts with Take The Best, which only uses the single most valid discriminating cue, and Tallying, which ignores validity magnitudes.

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
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
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
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Take-The-Best (Lexicographic) Heuristic with Lapse

**Rationale:** Following the arbiter's feedback, this model implements a strict Take-The-Best (lexicographic) decision rule. Subjects process features sequentially in descending order of their explicit validities, stopping at the first feature that discriminates between the two options. The option with the higher value on that feature is chosen. If no features discriminate, the decision defaults to a uniform guess. Unlike previous implementations that used a softmax temperature (beta) over binary scores, this model uses a pure deterministic rule modified only by a lapse rate (epsilon) to account for random response noise, capturing the frugality of the lexicographic heuristic more directly.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    epsilon = float(parameters["epsilon"])
    
    if winner is None:
        # No features discriminate, guess uniformly
        return np.array([0.5, 0.5])
    
    # Deterministic choice with lapse rate
    p = np.zeros(2)
    p[winner] = 1.0
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** People use a Thresholded Tallying heuristic to compare options. To simplify decision making, individuals ignore cues whose validity falls below a certain subjective threshold. For the remaining cues, they disregard the exact validities and simply tally the number of positive features for each option. Decisions are then made probabilistically based on the difference in tallies using a softmax function, with occasional random lapses. The choice rule operates in a softer, probabilistic regime to account for human behavioral variance.

**Rationale:** Following the most recent feedback, we revert to the Thresholded Tallying model that serves as the running-best base (Iteration 1). To increase the simulated variance (dispersion) and better match the human JSD values across experiments, we restrict the range of the softmax inverse-temperature `beta` from [0.1, 20.0] to [0.0, 5.0]. This prevents the choice rule from becoming overly deterministic, forcing a more probabilistic response regime while maintaining the core thresholded tallying mechanism.

**Parameters:**
  - `beta`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Select features with validity at or above the subjective threshold
    mask = val >= threshold
    
    # Tally positive features for each option among selected cues
    score_a = np.sum(a[mask])
    score_b = np.sum(b[mask])
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
