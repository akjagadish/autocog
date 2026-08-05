# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Sequential Cue Accumulation (Take-The-Best with Verification): Decision-makers process cues sequentially in order of validity, accumulating evidence for or against each option. They stop searching when the absolute accumulated evidence reaches a threshold. If the evidence reaches the threshold, a decision is made based on the accumulated evidence. However, if all cues are exhausted and the accumulated evidence fails to reach the threshold (e.g., due to highly contradictory cues), the decision-maker defaults to guessing. This captures the chance-like behavior observed in some compensatory settings while maintaining Take-The-Best behavior when early cues provide overwhelming evidence.

**Rationale:** Applying the minimal edit suggested by the critic: we modify the accumulation loop so that if all cues are exhausted and the final absolute accumulated evidence is still below the threshold `theta`, the model defaults to a uniform guess rather than applying the softmax. This directly addresses the failure to capture chance-like behavior in Experiment 2 when evidence falls below the threshold, while preserving the Take-The-Best behavior when the threshold is met.

**Parameters:**
  - `theta`: `[0.0, 1.0]`
  - `gamma`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale and normalize weights
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    threshold_reached = False
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if accumulated evidence reaches the threshold
            if abs(E) >= theta:
                threshold_reached = True
                break
            
    if not threshold_reached:
        p = np.array([0.5, 0.5])
    else:
        scores = np.array([E, -E])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) theory with non-linear scaling posits that decision-makers integrate all available information by weighting each feature by its validity, but subjectively scale these validities to adjust the degree of compensation. By exponentiating validities by a parameter gamma and then normalizing them, the model flexibly interpolates between equal weighting (Tallying), proportional weighting, and highly non-compensatory weighting (Take The Best) while keeping the integrated scores on a stable scale. Choices are made probabilistically using a softmax over the options' scores.

**Rationale:** Addressed the critic's feedback by normalizing the weights after exponentiation (`weights = weights / np.sum(weights)`) and expanding the `gamma` range up to 30.0. This prevents the fractional validities from shrinking to zero when raised to a high power, ensuring the weighted scores remain on a stable scale. This allows the `beta` parameter to properly control the determinism of the choice probabilities and stops them from collapsing to 50/50, letting the model fully express strong non-compensatory (TTB-like) behavior.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 30.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities with normalization to prevent score shrinkage
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Calculate overall scores by weighting features by their scaled validity
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Tallying with Validity Tie-Breaking: Decision-makers primarily rely on a compensatory but simplified strategy where they count the total number of positive cues for each option (Tallying), ignoring the explicit validities. The option with the higher tally is chosen. Only if both options have an equal number of positive cues does the decision-maker fall back to the provided validities to break the tie, typically by consulting the most valid discriminating cue (Take The Best). This boundedly rational approach explains why options supported by a highly valid cue are often rejected if they have fewer total positive cues.

**Rationale:** Following the arbiter's guidance, this theory implements Tallying as the primary decision mechanism, completely ignoring cue validities unless there is a tie in the total number of positive cues. When a tie occurs, the model uses Take The Best to break it. This direct Tallying-first approach is designed to capture the chance-like and deterministic behavior seen in settings where the TTB winner has fewer total cues (e.g., Experiments 7 and 8), explaining the divergence from purely non-compensatory models.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    a_wins = np.sum(a)
    b_wins = np.sum(b)
    
    if a_wins > b_wins:
        scores = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        scores = np.array([0.0, 1.0])
    else:
        # Tie-breaker using Take The Best
        val = np.asarray(parameters["validities"], dtype=float)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
