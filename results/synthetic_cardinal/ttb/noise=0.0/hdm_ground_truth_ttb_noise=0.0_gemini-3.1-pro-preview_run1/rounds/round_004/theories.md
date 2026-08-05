# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

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


### slot 2 — `pi_4` — KILLED ✗

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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Heuristic Strategy Mixture with Independent Noise: Decision-makers probabilistically sample between a Pure Tallying strategy and a strict Take-The-Best strategy. Because these heuristics operate on fundamentally different scales (Tallying produces counts of wins, while TTB produces binary indicators), the determinism (inverse temperature) of each strategy is calibrated independently before their choice probabilities are mixed.

**Rationale:** Following the critic's feedback, this minimal edit introduces independent inverse temperatures (`beta_tally` and `beta_ttb`) for the Pure Tallying and strict Take-The-Best components of the mixture model. Because Tallying scores are counts (e.g., differences of 2 or 3) while TTB scores are binary (differences of exactly 1), a shared `beta` forces Tallying to be systematically more deterministic. By decoupling them, the model can independently calibrate the noise level of each heuristic before blending their predictions via `w_tally`.

**Parameters:**
  - `w_tally`: `[0.0, 1.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Pure Tallying Strategy: unweighted count of strict wins
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    scores_tally = np.array([a_wins, b_wins], dtype=float)
    
    # 2. Strict Take-The-Best Strategy: lexicographic choice based on highest-validity differentiating cue
    sorted_idx = np.argsort(-v)
    a_ttb = 0.0
    b_ttb = 0.0
    for idx in sorted_idx:
        if a[idx] > b[idx]:
            a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            b_ttb = 1.0
            break
    scores_ttb = np.array([a_ttb, b_ttb], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    # Tallying probabilities
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # TTB probabilities
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Mixture of the two strategies
    w_tally = float(parameters["w_tally"])
    p_mixed = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Blend with uniform lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mixed)
    p_final = (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)
    
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
