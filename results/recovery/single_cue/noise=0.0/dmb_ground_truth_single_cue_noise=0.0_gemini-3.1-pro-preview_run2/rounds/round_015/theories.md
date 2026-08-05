# Round 15 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_16` — SURVIVED ✓

**Description:** Recency-Biased Cue Overweighting: Decision-makers evaluate options by attempting to integrate all available features, but due to visual recency and short-term memory effects, the final feature in the sequence is disproportionately salient. While the first N-1 features are weighted according to their stated validities (subject to non-linear scaling), the final feature is assigned an independent, often much larger weight. This mechanism explains boundary cases where subjects' choices are driven by the nominally least valid cue, effectively overriding both compensatory tallying and the expected Take-The-Best hierarchy.

**Rationale:** The arbiter provided feedback indicating a need for a model where the final feature in the sequence is highly salient due to visual recency, causing it to disproportionately drive choice. This model instantiates 'Recency-Biased Cue Overweighting' by computing a weighted additive score where the first N-1 features are weighted by their validities (scaled by a parameter gamma), but the final feature is assigned an independent, freely varying 'recency_weight'. This enables the model to capture empirical boundary cases where the nominally least valid feature dominates the decision, explaining variance that pure TTB or Tallying fail to account for.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `recency_weight`: `[0.0, 10.0]`
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
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    # Overweight the final feature due to recency
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


### slot 2 — `pi_17` — KILLED ✗

**Description:** Primacy-Recency Cue Integration (Normalized Weights)

**Rationale:** Following the critic's advice, we normalize the weight vector `w` before computing the scores. This bounds the scores between 0 and 1, decoupling the relative importance of the cues from the overall scale of the evidence. By doing this, we fix the optimization unidentifiability where massive weights pushed the softmax into flat-gradient regions. We restored the wide ranges for `w_primacy` and `w_recency` [0.0, 50.0] to allow extreme boundary behaviors, and set `beta` to [0.1, 20.0] to cleanly control softmax determinism over the normalized scores. `epsilon` is constrained to [0.0, 0.1] to allow for extreme empirical choice probabilities.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `w_primacy`: `[0.0, 50.0]`
  - `w_recency`: `[0.0, 50.0]`
  - `w_middle`: `[0.0, 1.0]`
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
    w_primacy = float(parameters["w_primacy"])
    w_recency = float(parameters["w_recency"])
    w_middle = float(parameters["w_middle"])
    gamma = float(parameters["gamma"])
    
    # Base weights for all cues, incorporating non-linear validity scaling
    w = w_middle * (val ** gamma)
    
    # Overweight the first (primacy) and last (recency) features
    if len(w) > 1:
        w[0] += w_primacy
        w[-1] += w_recency
    elif len(w) == 1:
        w[0] += w_primacy + w_recency
        
    # Normalize weights to sum to 1 to bound scores and improve identifiability
    sum_w = np.sum(w)
    if sum_w > 0:
        w = w / sum_w
    else:
        w = np.ones_like(w) / len(w)
        
    # Compute evidence for each option based on strict wins
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_18` → slot 2 (via `new_theory`)

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
