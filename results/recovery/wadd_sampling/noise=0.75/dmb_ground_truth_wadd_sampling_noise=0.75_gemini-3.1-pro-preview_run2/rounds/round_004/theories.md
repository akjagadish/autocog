# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Strategy Mixture of TTB and WADD: Decision-makers draw from a repertoire of strategies, specifically mixing a non-compensatory heuristic (Take The Best) and a fully compensatory one (Weighted Additive). By probabilistically blending TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums of all cues), the model naturally predicts the intermediate choice proportions observed on conflict trials where the two strategies make opposite predictions.

**Rationale:** Following the critic's most recent feedback, the previous attempt to normalize WADD scores degraded the model's fit by squashing evidence differences and making WADD predictions too noisy. Instead of modifying the evidence scale, we revert to the unnormalized log-odds WADD sums from the running-best candidate and simply widen the upper bounds of `beta_ttb` and `beta_wadd` from 10.0 to 100.0. This gives the optimizer the freedom to scale the unnormalized WADD evidence and TTB evidence appropriately, allowing the model to achieve sharper, more deterministic probabilities for both strategies when necessary.

**Parameters:**
  - `beta_ttb`: `[0.01, 100.0]`
  - `beta_wadd`: `[0.01, 100.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `p_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Weighted Additive (WADD) Prediction ---
    # Transform validities to log-odds to represent additive evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores_wadd = np.array([v_a, v_b])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_wadd
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Rank-Weighted Additive Model: Decision-makers assign subjective weights to cues based on their ordinal rank of validity rather than transforming raw validities via log-odds. The weight of each cue decays as an inverse power function of its rank (e.g., 1 / rank^k). This compensatory approach preserves monotonicity and naturally dampens the extreme predictions of pure log-odds WADD, successfully explaining intermediate choice proportions and avoiding the ordinal violations typical of non-compensatory heuristics like Take The Best.

**Rationale:** Following the critic's feedback, the Rank-Weighted Additive Model with inverse power decay (w = 1 / rank^k) is retained exactly as in iteration 2. To fine-tune the aggregate loss without altering the core functional form, the parameter ranges for stochasticity are gently restricted: `beta` is narrowed to [0.1, 3.0] to prevent overly deterministic predictions, and `epsilon` is restricted to [0.0, 0.3] to prevent excessive noise. This minimal adjustment should help balance the model's predictions across all experiments.

**Parameters:**
  - `beta`: `[0.1, 3.0]`
  - `decay_rate`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 0.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    decay_rate = float(parameters["decay_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Determine ranks of validities (highest validity gets rank 1)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros_like(val)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Calculate subjective weights based on rank
    w = 1.0 / (ranks ** decay_rate)
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Sequential Evidence Accumulation Theory with Attention Scaling: Decision-makers evaluate options by sequentially sampling cues with probabilities proportional to a non-linear scaling of their validities. Each sampled cue provides evidence that accumulates over time until it hits a decision boundary. A power parameter (gamma) sharpens or flattens the attention given to the most valid cues, allowing the model to smoothly interpolate between TTB-like focus on the best cue and compensatory tallying of all cues, while the inherent variance of the sampling process provides natural noise.

**Rationale:** Following the critic's advice, a power parameter `gamma` is introduced to scale the cue sampling probabilities non-linearly (`prob_cue = (val ** gamma) / np.sum(val ** gamma)`). The previous iteration overpredicted compensatory choices because sampling strictly proportional to raw validities led to over-sampling of minor cues. By allowing `gamma > 1`, the model can sharpen attention towards the highest validity cue, shifting behavior to be more TTB-like on conflict trials and better capturing the empirical balance without breaking the sequential evidence accumulation mechanism.

**Parameters:**
  - `drift_scale`: `[0.1, 10.0]`
  - `boundary`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    drift_scale = float(parameters["drift_scale"])
    boundary = float(parameters["boundary"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Probability of sampling each cue proportional to its scaled validity
    prob_cue_raw = val ** gamma
    prob_cue = prob_cue_raw / np.sum(prob_cue_raw)
    
    # p = probability of sampling a cue that favors A
    # q = probability of sampling a cue that favors B
    p = np.sum(prob_cue[(a > b)])
    q = np.sum(prob_cue[(b > a)])
    
    if p == 0 and q == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Expected drift and variance per sample
        # Step size is scaled by drift_scale
        v = drift_scale * (p - q)
        var = (drift_scale ** 2) * (p + q - (p - q)**2)
        
        if var <= 1e-6:
            prob_a = 1.0 if v > 0 else (0.0 if v < 0 else 0.5)
        else:
            # DDM probability of hitting upper boundary before lower boundary
            exponent = -2.0 * boundary * v / var
            # Clip exponent for numerical stability
            exponent = np.clip(exponent, -20.0, 20.0)
            prob_a = 1.0 / (1.0 + np.exp(exponent))
            
        p_core = np.array([prob_a, 1.0 - prob_a])
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```
