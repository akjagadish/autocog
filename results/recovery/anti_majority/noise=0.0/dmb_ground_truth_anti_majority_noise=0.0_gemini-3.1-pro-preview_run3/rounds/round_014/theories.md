# Round 14 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_16` — KILLED ✗

**Description:** Context-Dependent Lexicographic Accumulator (CDLA) with Symmetric Dispersion Effect: Decision-makers process information lexicographically, but the weight assigned to the first discriminating cue is contextually modulated by the dispersion of remaining cues in a symmetric manner. Both contradiction and consonance from secondary cues adjust the primary cue's weight. Furthermore, both primary and secondary cues can be weighted negatively, capturing baseline avoidance and perverse context effects.

**Rationale:** Following the critic's advice, I removed the `max(0, ...)` bound on the `contradiction` variable. This allows the contextual modulation of the primary cue to be symmetric, meaning the model can now capture both the amplification of the primary cue when secondary cues support it (consonance) and its attenuation when they oppose it (contradiction). This minor change allows the model to better fit the large preference reversals between agreement and conflict trials observed in Experiments 11 and 12, without changing the functional form or the parameter ranges that have proven successful.

**Parameters:**
  - `w_first`: `[-10.0, 10.0]`
  - `w_remaining`: `[-5.0, 5.0]`
  - `dispersion_effect`: `[-2.0, 2.0]`
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
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    first_disc_idx = -1
    for i in cue_order:
        if a[i] != b[i]:
            first_disc_idx = i
            break
            
    if first_disc_idx == -1:
        return np.array([0.5, 0.5])
        
    w_first = float(parameters["w_first"])
    w_remaining = float(parameters["w_remaining"])
    dispersion_effect = float(parameters["dispersion_effect"])
    
    remaining_a = 0
    remaining_b = 0
    passed_first = False
    for i in cue_order:
        if passed_first:
            remaining_a += a[i]
            remaining_b += b[i]
        if i == first_disc_idx:
            passed_first = True
            
    first_diff = a[first_disc_idx] - b[first_disc_idx]
    
    if first_diff > 0:
        contradiction = remaining_b - remaining_a
    else:
        contradiction = remaining_a - remaining_b
        
    # The weight of the first cue is modulated symmetrically by the contradiction/consonance
    adj_first = w_first * np.exp(dispersion_effect * contradiction)
    
    ev_a = 0.0
    ev_b = 0.0
    
    if first_diff > 0:
        ev_a += adj_first
    else:
        ev_b += adj_first
        
    ev_a += w_remaining * remaining_a
    ev_b += w_remaining * remaining_b
    
    beta = float(parameters["beta"])
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_15` — SURVIVED ✓

**Description:** Probabilistic Strategy Mixture (Soft TTB and Soft Tallying)

**Rationale:** Following the critic's advice, I introduced a separate temperature parameter (`beta_ttb`) for the Take-The-Best (TTB) strategy to soften its deterministic nature. This allows the mixture to better capture trials where subjects lean heavily on the best cue but still exhibit some noise. The soft Tallying component and the mixture probability `p_ttb` are maintained, providing a more flexible and nuanced blending of compensatory and non-compensatory heuristics.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `beta_ttb`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # 1. Take-The-Best (TTB) Prediction (Probabilistic)
    ttb_winner = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 1.0
            break
        elif b[j] > a[j]:
            ttb_winner = 0.0
            break
            
    ttb_scores = np.array([ttb_winner, 1.0 - ttb_winner])
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * ttb_scores
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb_dist = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    # 2. Tallying Prediction
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally = z_tally - np.max(z_tally)
    p_tally_dist = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # 3. Strategy Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_dist + (1.0 - p_ttb) * p_tally_dist
    
    return p_core
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

### `pi_17` → slot 1 (via `new_theory`)

**Description:** Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

**Rationale:** Following the critic's advice, this edit retains the exact same lateral inhibition mechanism but further widens the parameter ranges for self-dilution (`alpha`) up to 5000.0 and decision temperature (`theta`) up to 500.0. This allows the optimizer to find a regime where multi-cue options are entirely squashed by massive self-dilution, reducing their evidence to near exactly zero, while single-cue options remain unsuppressed and deterministically win due to the high theta. This extreme contrast is needed to capture the deep negative reversals seen in Experiments 12, 29, and 30.

**Parameters:**
  - `alpha`: `[0.0, 5000.0]`
  - `beta`: `[0.0, 100.0]`
  - `gamma`: `[0.1, 5.0]`
  - `theta`: `[0.1, 500.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
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
