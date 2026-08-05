# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

**Rationale:** The arbiter feedback indicates that Take The Best (TTB) vastly overpredicts the empirical adherence to the single best cue. Subjects in these experiments systematically choose options backed by a coalition of cues rather than a single highly valid cue. By implementing a Tallying (Equal Weights) theory where subjects simply count the number of positive features for each option, the model naturally favors the option with more positive cues regardless of their continuous validities. This compensatory mechanism is expected to perfectly capture the extremely low TTB adherence observed in the experimental data.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Thresholded Counting: People make decisions by counting the number of positive features for each option, but they only consider features that are deemed sufficiently reliable. Specifically, they ignore features whose subjective or objective validity falls below a certain threshold. For the features that pass this threshold, individuals simply tally the positive cues (equal weighting) and choose the option with the higher count. This bounded-rationality heuristic combines the simplicity of Tallying with a minimal form of validity sensitivity, filtering out noisy or low-information cues before applying a non-compensatory counting strategy.

**Rationale:** Following the critic's advice, I tightened the 'threshold' parameter range to [0.0, 0.5]. Since validities in this domain are generally >= 0.5, this ensures the model robustly includes useful cues and closely approximates the highly successful pure Tallying strategy, while maintaining the mechanistic capability to filter out strictly non-informative or deceptive cues (validity <= 0.5).

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Counting expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    threshold = float(parameters["threshold"])
    
    # Only count features whose validity meets or exceeds the threshold
    mask = validities >= threshold
    
    # If the threshold is so high that no features qualify, fall back to guessing (all zeros)
    if not np.any(mask):
        scores = np.zeros(stim.shape[0])
    else:
        scores = stim[:, mask].sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Weighted Additive Strategy (WADD) with Linearly Compressed Weights: Individuals compute a compensatory weighted sum of positive features for each option, but their subjective weights are compressed relative to objective validities. People interpolate between raw validities and equal weights (Tallying), flattening the weight distribution. This allows a large number of less valid cues to compensate for a highly valid cue, matching human behavior in compensatory environments.

**Rationale:** Following the arbiter's most recent feedback, we introduce a linear mixing parameter `alpha` bounded between 0.0 and 1.0. This computes the subjective weights as an interpolation between raw validities (alpha=1) and equal weights (alpha=0). Unlike previous attempts using unbounded power-laws or log-odds (which amplified differences and worsened the fit), this linear compression safely bounded the weights, directly addressing the failures in Experiments 3 and 4 by letting the model smoothly capture the human tendency towards equal-weighting (Tallying) without diverging.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Interpolate between raw validities and equal weights
    weights = alpha * validities + (1.0 - alpha) * 1.0
    
    # Weighted Additive Strategy
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
