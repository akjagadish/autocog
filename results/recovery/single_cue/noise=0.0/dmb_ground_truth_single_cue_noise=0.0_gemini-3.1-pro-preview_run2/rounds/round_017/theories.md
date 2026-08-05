# Round 17 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_19` — SURVIVED ✓

**Description:** Primacy-Dominant Anchoring: Decision-makers use the first cue as a powerful anchor that overwhelmingly dominates the evaluation process. While the final cue may receive a secondary recency boost due to short-term memory, the primacy weight is structurally much larger than both the recency weight and the middle cue validities. This explicitly enforces a hierarchy where primacy is the primary driver of choice.

**Rationale:** Based on the arbiter's feedback, this theory models a 'Primacy-Dominant Anchoring' mechanism. Instead of treating primacy and recency as symmetric boundary effects or purely overweighting recency, this model explicitly enforces a hierarchy where the first cue acts as a massive anchor (weight range [10, 30]), while the final cue can have a secondary recency effect (weight range [0, 9]), and middle cues carry their base validities. This captures the strong empirical phenomenon where subjects base choices almost entirely on the initial anchor cue, overriding both compensatory tallying and recency when they conflict.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `primacy_weight`: `[10.0, 30.0]`
  - `recency_weight`: `[0.0, 9.0]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Enforce Primacy-Dominant Anchoring hierarchy
    w[0] = primacy_weight
    if len(w) > 1:
        w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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


### slot 2 — `pi_18` — KILLED ✗

**Description:** Serial Position Dual-Overweighting: Decision-makers evaluate options by integrating features, but due to memory and attention constraints at the sequence boundaries, both the first (primacy) and the last (recency) cues are assigned independent, disproportionately large weights. Unlike models that normalize attention or weights, these boundary weights are unnormalized, allowing them to independently dominate choice when necessary. Middle cues are weighted by their stated validities, scaled non-linearly. This mechanism captures both extreme primacy and extreme recency effects without the dampening effect of weight normalization.

**Rationale:** Following the arbiter's guidance, this theory introduces 'Serial Position Dual-Overweighting'. It assigns independent, unnormalized weights to both the first (primacy) and last (recency) cues, while scaling the middle cues according to their stated validities. By omitting weight normalization, the model prevents the dampening of these extreme boundary weights, enabling it to fully capture the strong primacy and recency effects observed in experiments like Exp 2 and Exp 31.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 5.0]`
  - `primacy_weight`: `[0.0, 10.0]`
  - `recency_weight`: `[0.0, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Overweight the first and final features due to primacy and recency
    if len(w) > 1:
        w[0] = primacy_weight
        w[-1] = recency_weight
    elif len(w) == 1:
        w[0] = primacy_weight + recency_weight
        
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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

### `pi_20` → slot 2 (via `new_theory`)

**Description:** Tallying with Primacy Tie-Breaking: Decision-makers primarily evaluate options by counting the total number of positive features (Tallying), ignoring the stated validities and sequence order. If one option has a strictly greater number of positive features, it is chosen. If the tally scores are tied, decision-makers fall back to a secondary non-compensatory mechanism: using the first cue (primacy) to break the tie.

**Rationale:** This model instantiates 'Tallying with Primacy Tie-Breaking' as requested by the arbiter. It calculates the unweighted sum of the stimulus vectors to perform a pure tally. If the tallies are unequal, the choice is determined by the tally difference. If the tallies are tied, the model falls back to a primacy heuristic, evaluating only the first cue to break the tie. This elegantly captures the strong tallying alignment observed in compensatory experiments while accurately modeling the primacy-driven tie-breaking behavior seen in Experiments 5, 6, 11, and 13, avoiding the mechanistic failures of relying purely on non-linear validity weighting.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_primacy`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    beta_tally = float(parameters["beta_tally"])
    beta_primacy = float(parameters["beta_primacy"])
    epsilon = float(parameters["epsilon"])
    
    # Unweighted sum of positive features
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    # If tally scores are different, choice is driven by tally difference
    if tally_a != tally_b:
        scores = np.array([beta_tally * tally_a, beta_tally * tally_b])
    # If tally scores are tied, choice is driven by primacy (first cue)
    else:
        scores = np.array([beta_primacy * a[0], beta_primacy * b[0]])
        
    # Softmax choice rule with max-subtraction for numerical stability
    z = scores - np.max(scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
