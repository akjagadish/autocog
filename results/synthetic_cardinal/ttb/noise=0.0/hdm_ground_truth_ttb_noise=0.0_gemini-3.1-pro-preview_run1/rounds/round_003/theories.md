# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Soft Take-The-Best with Power-Law Decay: Decision-makers evaluate options by ranking features according to their subjective validity and integrating evidence across all cues. However, instead of the importance of cues decaying exponentially with their rank, it decays according to a power law. This 'fatter tail' enables a strong accumulation of evidence on multiple lower-ranked cues to compensate for a loss on top-ranked cues, capturing compensatory behavior in scenarios where many weak cues oppose a few strong ones.

**Rationale:** Following the critic's suggestion, we modified the rank-based decay in the Soft TTB model from an exponential decay to a power-law decay. We also expanded the `alpha` parameter range to [0.0, 3.0]. The power-law decay has a 'fatter tail', which prevents the weights of lower-ranked features from vanishing too quickly. This allows a larger number of lower-validity wins to successfully outweigh fewer higher-validity wins, directly addressing the model's mispredictions in Experiments 5 and 6.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.0, 3.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Soft TTB expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(v)[::-1]
    
    alpha = float(parameters["alpha"])
    # Power-law decay based on rank (rank 1, 2, 3...)
    weights = 1.0 / ((np.arange(len(v)) + 1.0) ** alpha)
    
    # Binary wins on ordered features
    a_wins = (a[order] > b[order]).astype(float)
    b_wins = (b[order] > a[order]).astype(float)
    
    score_a = np.sum(a_wins * weights)
    score_b = np.sum(b_wins * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

**Rationale:** Weighted Tallying acts as a bridge between Tallying and Weighted Additive (WADD) models. By discarding cardinal magnitudes and relying solely on strict feature-wise wins, it mimics human robustness to extreme, outlier feature values that would otherwise dominate an additive score. By weighting those wins by the features' validities, it overcomes the main mechanistic failure of pure Tallying: treating all cues equally. This design allows the model to capture behavior where subjects use cue importance to break ties or flip preferences when tally counts are close, matching the arbiter's suggestion.

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
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Pure Tallying Heuristic: Decision-makers evaluate options by completely ignoring the provided validities or weights of the cues. Instead, they simply count the number of features on which one option is strictly better than the other. The option with the higher total count of winning features is chosen. This unweighted counting mechanism explains the robust preference for options with more winning features, even if those features are of lower validity.

**Rationale:** Following the arbiter's feedback, this theory implements a Pure Tallying (unweighted counting) mechanism. In contrast to weighted models (like WADD or Weighted Tallying) that attempt to incorporate cue validities, this model completely ignores validities. It simply tallies the number of strict wins for each option. This naturally captures the observed human behavior in Experiments 1, 7, and 8, where participants showed a strong preference for options with a higher number of winning features, regardless of the features' stated importance.

**Parameters:**
  - `beta`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins; completely ignore validities and magnitudes
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    scores = np.array([a_wins, b_wins], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
