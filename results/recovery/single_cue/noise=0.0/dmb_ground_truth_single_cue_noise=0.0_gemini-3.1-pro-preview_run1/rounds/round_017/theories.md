# Round 17 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_19` — KILLED ✗

**Description:** Soft Primacy-Biased Sequential Integration

**Rationale:** Following the critic's advice, the previous model's loss improved by restricting the primacy bias, but the fits indicate it still overpredicts TTB/primacy and underpredicts Tallying. To force the model to behave even more like Tallying while staying within the prescribed Primacy family, we further restrict the upper bounds of w_ttb to [0.0, 0.1] and gamma to [0.0, 0.5]. This ensures the WADD component dominates and its weights decay very gently, allowing the total number of positive features to drive the choice, matching human behavior much more closely.

**Parameters:**
  - `w_ttb`: `[0.0, 0.1]`
  - `gamma`: `[0.0, 0.5]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    w_ttb = float(parameters["w_ttb"])
    gamma = float(parameters["gamma"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take-The-Best (TTB)
    # Process from left to right and terminate at the first discriminating feature
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # Strategy 2: Primacy-Boosted WADD
    # Compensatory mechanism with exponentially decaying weights from left to right
    ranks = np.arange(n_features)
    weights = np.exp(-gamma * ranks)
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        weights = np.ones_like(weights) / n_features
    
    score_a_wadd = np.sum(a * weights)
    score_b_wadd = np.sum(b * weights)
    
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    probs_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture Model
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_wadd
    
    # Lapse Rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_16` — SURVIVED ✓

**Description:** Context-Dependent Dual-Process Mixture of TTB and Tallying with Inverse Validity Tie-Breaking: Decision-makers rely on a mixture of Take-The-Best (TTB) and Tallying, but the mixture weight is dynamically determined by the environment. When cue validities are highly dispersed (measured by the standard deviation of the validities), subjects predominantly use TTB; when validities are similar, they rely on Tallying. When Tallying results in a tie, subjects resolve it using an inverse-validity weighting mechanism, heavily favoring options with positive features among the lower-validity (or more recently processed) cues.

**Rationale:** Following the feedback, we revert to the Iteration 4 base (which correctly preserved tie-breaking behavior) and refine the gating mechanism by changing the dispersion metric from the range to the standard deviation of validities (`np.std(validities)`). The standard deviation provides a more holistic and robust measure of whether a cue environment is compensatory or non-compensatory, without introducing the brittleness of an extreme slope bound or overly constrained lapse rate that caused Iteration 5 to fail.

**Parameters:**
  - `validities`: `validities`
  - `disp_slope`: `[0.0, 100.0]`
  - `disp_threshold`: `[0.0, 1.0]`
  - `w_tie`: `[0.0, 0.95]`
  - `gamma`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    disp_slope = float(parameters["disp_slope"])
    disp_threshold = float(parameters["disp_threshold"])
    w_tie = float(parameters["w_tie"])
    gamma = float(parameters["gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_ttb = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying with Inverse Validity Tie-Breaker ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # w_tie < 1.0 ensures the tie-breaker only dictates choice when a_wins == b_wins
    score_a_tally = a_wins + w_tie * tie_score_a
    score_b_tally = b_wins + w_tie * tie_score_b
    
    scores_tally = np.array([score_a_tally, score_b_tally])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    probs_tally = e_tally / np.sum(e_tally)
    
    # --- Strategy 2: Take-The-Best (TTB) ---
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # --- Mixture Model ---
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_tally
    
    # --- Lapse Rate ---
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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

### `pi_20` → slot 1 (via `new_theory`)

**Description:** Validity-Weighted Evidence Accumulation with Linear Interpolation and Extended Threshold

**Rationale:** Following the critic's advice, we expand the range of the linear interpolation parameter `alpha` to `[-5.0, 5.0]` to allow for more extreme inverse-validity and WADD effects. Correspondingly, we expand the upper bound of the threshold parameter `theta` to `[0.0, 20.0]` so that the model can still bypass early stopping when `alpha` is large. The mechanism remains exactly the same as the previous iteration.

**Parameters:**
  - `validities`: `validities`
  - `alpha`: `[-5.0, 5.0]`
  - `theta`: `[0.0, 20.0]`
  - `theta_decay`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    theta = float(parameters["theta"])
    theta_decay = float(parameters["theta_decay"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Interpolate between uniform weights (Tallying) and validities (WADD)
    # A negative alpha natively assigns higher weights to lower-validity features
    v = (1.0 - alpha) + alpha * validities
    
    diff = 0.0
    n = len(a)
    for i in range(n):
        diff += (a[i] - b[i]) * v[i]
        
        # Dynamic threshold that can decay over the course of feature processing
        decay_factor = 1.0 - theta_decay * (i / max(1, n - 1))
        current_theta = theta * decay_factor
        
        # Stop search if evidence difference exceeds threshold (and is non-zero)
        if abs(diff) >= current_theta and abs(diff) > 1e-9:
            break
            
    # Softmax choice based on accumulated evidence at stopping point
    scores = np.array([diff, 0.0])
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
