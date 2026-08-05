# Round 16 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_18` — KILLED ✗

**Description:** Context-Dependent Dual-Process Mixture of Recency-Boosted WADD and Tallying (Score-Mixed, Unnormalized WADD): Decision-makers use a dual-process strategy mixing a Validity-Weighted Additive (WADD) strategy and a Tallying heuristic based on the dispersion of cue validities. The mixture occurs at the level of decision values (scores). To allow WADD to break Tallying ties effectively even when the mixture weight heavily favors Tallying, the WADD cue weights are left unnormalized. This allows the raw WADD scores to scale up naturally with a wider recency parameter, providing a strong enough signal to break ties smoothly without requiring an extreme global softmax temperature.

**Rationale:** To address the critic's diagnosis that the previous iteration became overly deterministic and brittle due to a massive beta and a hard lower bound on the mixture weight, this minimal edit reverts beta to a more moderate upper bound (500.0) and removes the hard weight bound. Instead, it stops normalizing the `w_wadd_cue` weights and expands the `recency` parameter range up to 50.0. This allows the raw WADD scores to scale up naturally, ensuring that even when the WADD mixture weight is very small, the WADD score difference is large enough to decisively break Tallying ties through the softmax, without distorting choice probabilities on non-tie trials.

**Parameters:**
  - `validities`: `validities`
  - `disp_slope`: `[0.0, 200.0]`
  - `disp_threshold`: `[0.0, 0.5]`
  - `recency`: `[0.0, 50.0]`
  - `beta`: `[0.1, 500.0]`
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
    recency = float(parameters["recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_wadd_mix = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    # Normalize to [0, 1] so it is on a comparable scale to WADD scores
    scores_tally_norm = scores_tally / max(1, n_features)
    
    # --- Strategy 2: Recency-Boosted WADD ---
    ranks = np.arange(n_features)
    # Right-most features get a higher recency boost. Weights are left unnormalized 
    # so that WADD differences can naturally scale up to break Tallying ties.
    w_wadd_cue = validities + recency * (ranks / max(1, n_features - 1))
        
    score_a_wadd = np.sum(a * w_wadd_cue)
    score_b_wadd = np.sum(b * w_wadd_cue)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # --- Mixture Model (Mixing Scores) ---
    mixed_scores = (1.0 - w_wadd_mix) * scores_tally_norm + w_wadd_mix * scores_wadd
    
    # Softmax on mixed scores
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # --- Lapse Rate ---
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

### `pi_19` → slot 1 (via `new_theory`)

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
