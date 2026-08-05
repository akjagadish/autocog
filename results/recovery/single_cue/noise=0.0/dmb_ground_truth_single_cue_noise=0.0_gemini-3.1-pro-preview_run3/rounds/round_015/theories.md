# Round 15 — Theories

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


### slot 2 — `pi_15` — KILLED ✗

**Description:** Feature Cancellation then Tally/TTB with Cancellation-Scaled Determinism

**Rationale:** Following the critic's advice, we built on the successful Feature Cancellation -> Tally -> TTB model and addressed the failures on experiments measuring choice determinism shifts (e.g., Exps 22 and 25). We achieved this by scaling the overall softmax determinism by the number of surviving features, computing the effective beta as `beta / np.sum(mask)`. This elegantly captures the relative weight shift described by the arbiter: when shared cues are canceled, there are fewer surviving distinct features, so the effective beta increases. This produces stronger deterministic preferences on trials with shared cues canceled, without adding any new parameters to the model.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
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
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Feature cancellation: identify surviving distinct features
    mask = a != b
    
    # Tallying on distinct features
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    score_a = tally_a
    score_b = tally_b
    
    # If tallies are tied, break the tie using TTB on surviving features
    if tally_a == tally_b and np.sum(mask) > 0:
        order = np.argsort(val)[::-1]
        for idx in order:
            if mask[idx]:
                if a[idx] > b[idx]:
                    score_a += 1.0
                elif b[idx] > a[idx]:
                    score_b += 1.0
                break
            
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule with cancellation-scaled determinism
    n_surviving = np.sum(mask)
    effective_beta = beta / n_surviving if n_surviving > 0 else beta
    
    z = effective_beta * (scores - np.max(scores))
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

### `pi_18` → slot 2 (via `new_theory`)

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
