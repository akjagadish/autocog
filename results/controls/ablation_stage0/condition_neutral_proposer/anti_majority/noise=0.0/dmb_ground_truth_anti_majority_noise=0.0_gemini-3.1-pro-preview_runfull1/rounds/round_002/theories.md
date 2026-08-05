# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Evidence Accumulation with Bounded Threshold: People sample cues sequentially in order of validity. Each discriminating cue adds its validity to the evidence for the favored option. If the absolute difference in evidence between the two options reaches a threshold `theta`, sampling stops and a decision is made based on the accumulated evidence (mimicking Take The Best). If all cues are exhausted without the threshold being reached, the decision is based on the final integrated evidence (mimicking Weighted Additive). This allows for early stopping when initial evidence is strong, but continued sampling when evidence is weak or tied.

**Rationale:** Following the critic's guidance, I have replaced the Dual-Process model with the Evidence Accumulation model from Iteration 6. Cues are sampled sequentially in order of validity, and the evidence (validity) is added to an accumulator. If the absolute difference in evidence reaches the threshold `theta`, sampling stops. Crucially, the threshold `theta` has been restricted to `[0.0, 1.0]` so that early stopping (TTB-like behavior) can actually occur, given that individual cue validities are bounded between 0.5 and 1.0. This fixes the issue in Iteration 6 where a too-high threshold forced the model to default to WADD on every trial, directly addressing the mechanistic failure.

**Parameters:**
  - `theta`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
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
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        if a[j] > b[j]:
            ev_a += val[j]
        elif b[j] > a[j]:
            ev_b += val[j]
            
        # Early stopping if threshold is reached
        if abs(ev_a - ev_b) >= theta:
            break
            
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

**Rationale:** Increased the upper bounds of `gamma` (from 10 to 50) and `beta` (from 20 to 200). Because validities are < 1, exponentiating them by a large gamma creates very small absolute weights. To translate these small score differences into deterministic choices that match the strong empirical preference for TTB on critical trials, the softmax temperature `beta` must be allowed to reach much higher values.

**Parameters:**
  - `beta`: `[0.1, 200.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 50.0]`
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
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Attention-Sharing (Fractional Discounting): Subjects evaluate options by linearly summing the validities of present features, but the overall value is discounted by the number of active cues. This reflects a cognitive bottleneck where attention must be divided among multiple features. The subjective value of an option is its sum of validities divided by `1.0 + kappa * (n_cues - 1)`. This mechanism allows a single highly valid cue to maintain its full weight and potentially dominate an option with multiple weaker cues that dilute each other's impact, explaining violations of tallying dominance while remaining mathematically stable.

**Rationale:** Following the critic's advice, we replaced the unstable quadratic configural penalty with a simpler and more robust 'Attention-Sharing' (Fractional Discounting) mechanism. The subjective score of an option is now its linear sum of present validities divided by a discounting factor that grows linearly with the number of present cues (`1.0 + kappa * (n_cues - 1)`). This ensures that the first cue retains its full weight, but additional cues dilute the overall value. This elegantly explains why a single strong cue can beat multiple weaker cues (Exp 1) while avoiding the erratic choice probabilities caused by the previous quadratic interaction term. We also constrained `epsilon` to [0.0, 0.1] to prevent the predictions from washing out to 0.5.

**Parameters:**
  - `kappa`: `[0.0, 2.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
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
    kappa = float(parameters["kappa"])
    
    def get_score(x):
        val_x = val * x
        n_cues = np.sum(x)
        if n_cues == 0:
            return 0.0
        linear = np.sum(val_x)
        return linear / (1.0 + kappa * (n_cues - 1.0))

    score_a = get_score(a)
    score_b = get_score(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
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
