# Round 19 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_21` — KILLED ✗

**Description:** Unified Attentional Evidence Accumulation with U-Shaped Primacy-Recency Gradient and Expanded Parameters

**Rationale:** Following the critic's advice, we revert completely to the exact unconstrained Iteration 2 formulation for the primacy-recency gradient. We stop trying to mathematically prevent weight inversion, as previous attempts consistently degraded the loss. Instead, we expand the parameter ranges for `c` to `[0.0, 2.0]` and `beta` to `[0.1, 50.0]`. This maintains the unified attentional framework while giving the optimizer more room to find the empirical balance needed to capture the large inverse-validity tie-breaking effects without our theoretical priors artificially restricting the fit.

**Parameters:**
  - `alpha`: `[0.0, 5.0]`
  - `c`: `[0.0, 2.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    alpha = float(parameters["alpha"])
    c = float(parameters["c"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # delta scales the linear recency boost.
    # By parameterizing delta as a fraction (c) of the initial primacy drop (1 - exp(-alpha)),
    # we guarantee that the highest validity cue ALWAYS receives more attention than the second,
    # preserving basic validity preferences while allowing a U-shaped curve for lower cues.
    delta = c * (1.0 - np.exp(-alpha))
    
    ranks = np.arange(n_features)
    
    # Calculate unified attention weights (Primacy + Recency)
    w = np.exp(-alpha * ranks) + delta * ranks
    w = w / np.sum(w)
    
    # Single integration process (Weighted Additive)
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
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

### `pi_22` → slot 1 (via `new_theory`)

**Description:** Sequential Accumulation with Explicit Validity Weighting, Early Stopping on Discriminating Cues, and Hazard Decay

**Rationale:** Following the critic's advice, I modified the stopping rule so that early stopping is only evaluated if the current cue discriminates between the two options (i.e., a[i] != b[i]). If the cue is tied, the decision-maker gains no new evidence and thus deterministically continues search (p_stop = 0.0), unless it's the very last cue. This prevents the model from stopping prematurely and making random choices on tied cues, properly concentrating probability mass on meaningful stopping points and improving its fit to TTB and Tallying data.

**Parameters:**
  - `validities`: `validities`
  - `gamma`: `[0.0, 1.0]`
  - `delta`: `[0.0, 2.0]`
  - `kappa`: `[-10.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `tau`: `[-2.0, 2.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    E = 0.0
    prob_A = 0.0
    p_reach = 1.0
    
    # Evaluate features sequentially
    for i in range(n_features):
        # Accumulate evidence weighted by validity
        # kappa can be negative to upweight later/lower-validity cues (recency)
        weight = validities[i] ** kappa
        E += weight * (a[i] - b[i])
        
        # Normalize evidence by the number of cues seen so far
        E_scaled = E / ((i + 1) ** tau)
        
        # Probability of choosing A if search stops at this cue
        p_A_given_stop = 1.0 / (1.0 + np.exp(-beta * E_scaled))
        
        # Determine probability of stopping at this step
        if i == n_features - 1:
            p_stop = 1.0
        else:
            if a[i] != b[i]:
                # Hazard rate of stopping with decay, only on discriminating cues
                p_stop = min(1.0, gamma * (delta ** i))
            else:
                # Do not stop on tied cues
                p_stop = 0.0
                
        prob_A += p_reach * p_stop * p_A_given_stop
        p_reach *= (1.0 - p_stop)
        
    final_probs = np.array([prob_A, 1.0 - prob_A])
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * final_probs + epsilon * np.array([0.5, 0.5])
    
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
