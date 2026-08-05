# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Choice Inertia (Autocorrelation) with Tallying: Decision-makers evaluate options by counting the total number of positive features for each option (unweighted sum), but their current choice is also biased by their motor response on the immediately preceding trial. A 'stickiness' parameter shifts the utility toward the previously chosen option index (A or B), reflecting the cognitive ease of repeating a past action regardless of the specific product features.

**Rationale:** Following the arbiter's suggestion, this theory incorporates sequential choice dependencies by modeling 'Choice Inertia'. It builds upon the unweighted Tallying heuristic (which performed reasonably well) but adds a 'stickiness' parameter (`phi`). This parameter directly modifies the utility of the response option (0 or 1) that was selected on the previous trial, capturing the tendency of human subjects to repeat past motor actions due to cognitive ease or response autocorrelation. This aligns perfectly with the metric, which evaluates choice probabilities conditioned on the previous response.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `phi`: `[-3.0, 3.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) state.")
    
    a, b = stim[0], stim[1]
    
    # Base utility: unweighted sum of features (Tallying)
    scores = np.array([np.sum(a), np.sum(b)])
    
    # Choice Inertia: boost the score of the previously chosen action
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        if 0 <= last_resp < 2:
            phi = float(parameters["phi"])
            scores[last_resp] += phi
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the modified utilities with numerical stability
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) Model: People integrate all available information by computing a weighted sum of feature values for each option. The weights are proportional to the log-odds of the cue validities. This allows for compensatory decision-making, where multiple weakly predictive features can jointly override a single highly predictive feature. Choice probabilities are determined by applying a softmax function to the weighted sums, with an independent lapse rate for random guessing.

**Rationale:** Following the arbiter's feedback, this model implements the Weighted Additive (WADD) strategy. Unlike Take The Best (which relies on a single cue) and Tallying (which treats all cues equally), WADD evaluates options by computing a weighted sum of their features. The weights are derived from the log-odds of the cue validities, allowing for compensatory decision-making where multiple weak cues can outweigh a single strong cue. This integration of all available information, appropriately weighted by its reliability, provides a more nuanced and accurate reflection of human choice probabilities in multi-attribute tasks.

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
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Extract validities and clip to avoid log(0) or division by zero
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-4, 1.0 - 1e-4)
    
    # Transform validities to weights using log-odds
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Take-The-Best (TTB) Heuristic: Decision-makers use a fast-and-frugal lexicographic strategy rather than integrating all available information. They rank features in descending order of their validities and inspect them one by one. The search stops at the first feature that discriminates between the two options, and the option with the positive value for that cue is chosen. If no features discriminate, they guess randomly. A lapse rate accounts for attention lapses or execution noise.

**Rationale:** Following the arbiter's feedback, this theory replaces the previous Choice Inertia/Tallying model with the Take-The-Best (TTB) heuristic. TTB is a non-compensatory, lexicographic model where decision-makers rank cues by validity and make a choice based entirely on the first discriminating cue they encounter. This avoids the cognitive burden of summing or weighting all features. A single lapse parameter (epsilon) is included to capture uniform response noise, providing a clean contrast to exhaustive integration models.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) state.")
    
    a, b = stim[0], stim[1]
    
    validities = np.array(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    # We use stable sort to handle ties consistently
    order = np.argsort(-validities, kind='stable')
    
    # Default to random guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Mix with uniform guessing (lapse rate)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure normalization
    return int(np.random.choice(len(probs), p=probs))
```
