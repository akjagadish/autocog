# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Equal Weights (Tallying) heuristic: Decision-makers evaluate options by simply counting the total number of positive features (1s) for each option, completely ignoring the validities of the experts. The option with the higher total number of positive features is chosen. If the tallies are tied, the decision-maker guesses randomly. Response noise is modeled via a softmax over the tallies and an independent lapse rate. This heuristic is compensatory but unweighted, representing a fast-and-frugal approach that integrates all information equally without the cognitive burden of weighting by validity.

**Rationale:** The arbiter requested a Tallying (Equal Weights) theory that ignores varying expert validities. In this model, the decision-maker simply sums the number of positive features (1s) for each option and prefers the one with the higher tally. This is compensatory (multiple cues can outweigh one) but unweighted, distinguishing it from both Take The Best (non-compensatory) and Weighted Additive (compensatory and weighted). It uses a softmax function over the tallies to map the deterministic heuristic into probabilistic choices, capturing human response noise.

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
    
    a, b = stim[0], stim[1]
    
    # Count the total number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies with numerical stability
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

### `pi_5` → slot 1 (via `new_theory`)

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
