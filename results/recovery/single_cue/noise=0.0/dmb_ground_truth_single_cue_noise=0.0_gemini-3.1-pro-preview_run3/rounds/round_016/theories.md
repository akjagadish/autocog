# Round 16 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_17` — SURVIVED ✓

**Description:** Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

**Rationale:** Following the critic's feedback, the previous threshold range of [0.0, 2.0] allowed the optimizer to select values greater than 1.0, which inappropriately caused the model to use TTB even when the tally difference was decisive (e.g., delta_tally == 1). By narrowing the threshold range to [0.1, 0.9], the model is forced into a regime where a tied tally (delta_tally == 0) always results in a positive exponent (favoring TTB), while any decisive tally (delta_tally >= 1) results in a negative exponent (favoring Tallying). This perfectly mirrors the successful Tally-then-TTB logic while remaining within the smooth Strategy Selection family.

**Parameters:**
  - `theta`: `[1.0, 20.0]`
  - `threshold`: `[0.1, 0.9]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
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


### slot 2 — `pi_18` — KILLED ✗

**Description:** Proportional Evidence Accumulation on Surviving Features with Exponentiated Tie-Breaker

**Rationale:** Following the critic's instruction, we revert to the Iteration 1 framework which enforces a strict Tally-then-validity hierarchy (gamma < 1) combined with proportional evidence accumulation (dividing by n_surv). We ensure the theta parameter exponentiates the validities before normalization to smoothly interpolate between a WADD-like and a strict TTB-like tie-breaker on the surviving features. This preserves Tallying dominance where appropriate while providing the flexibility needed to capture lexicographic tie-breaking.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 0.99]`
  - `theta`: `[0.1, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    mask = a != b
    n_surv = np.sum(mask)
    
    if n_surv == 0:
        return np.array([0.5, 0.5])
        
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    # Exponentiate validities to smoothly transition between WADD and TTB tie-breaking
    val_transformed = val ** theta
    tie_w = val_transformed / np.sum(val_transformed)
    
    # Score combines Tallying and a strictly bounded validity-based tie-breaker
    score_a = tally_a + gamma * np.sum(tie_w * a_wins)
    score_b = tally_b + gamma * np.sum(tie_w * b_wins)
    
    # The decision variable is the proportion of surviving evidence
    norm_score_a = score_a / n_surv
    norm_score_b = score_b / n_surv
    
    scores = np.array([norm_score_a, norm_score_b])
    
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
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_19` → slot 2 (via `new_theory`)

**Description:** Tallying with Graded Validity Tie-Breaker: Decision-makers primarily use a simple Tallying heuristic (comparing the total number of positive features). When tallies are unequal, choices are highly deterministic. However, when tallies are tied, instead of rigidly applying a non-compensatory Take-The-Best rule (which overpredicts determinism), they fall back to a compensatory Weighted Additive (WADD) evaluation of the features. This tie-breaking process is governed by its own sensitivity parameter, allowing tied decisions to exhibit the softer, empirically observed determinism (~0.60) while maintaining the structural dominance of Tallying.

**Rationale:** Following the arbiter's diagnosis, strict non-compensatory tie-breakers (like Take-The-Best) overpredict determinism on tied trials. This model implements a two-stage lexicographic strategy: it first compares tallies, using `beta_tally` to drive strong determinism on unequal tallies. If tallies are tied, it falls back to a compensatory WADD process over the validities. By assigning a separate sensitivity parameter (`beta_wadd`) to this tie-breaker, the model can naturally capture the softer, more probabilistic choices observed empirically on tied trials, without compromising the dominance of the tallying heuristic on unequal trials.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `theta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    a = np.asarray(stimulus[0], dtype=float)
    b = np.asarray(stimulus[1], dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if tally_a != tally_b:
        # Primary decision based on tally difference
        z = beta_tally * (tally_a - tally_b)
    else:
        # Fallback to WADD-based tie-breaker with exponentiated validities
        val_w = val ** theta
        wadd_a = np.sum(val_w * a)
        wadd_b = np.sum(val_w * b)
        max_wadd = np.sum(val_w)
        if max_wadd == 0:
            max_wadd = 1.0
        # Normalize by max possible WADD for scale invariance
        z = beta_wadd * (wadd_a - wadd_b) / max_wadd
        
    # Numerically stable logistic function
    p_a = 1.0 / (1.0 + np.exp(-z))
    
    # Apply trembling hand epsilon
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    
    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
