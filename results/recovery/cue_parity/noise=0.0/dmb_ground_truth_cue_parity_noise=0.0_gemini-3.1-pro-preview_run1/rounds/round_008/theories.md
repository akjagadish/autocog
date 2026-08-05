# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_10` — KILLED ✗

**Description:** Heuristic Toolbox / Strategy Selection Theory with Graded Sub-models: Decision-makers possess a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive). Instead of assuming pure determinism within each heuristic, each strategy produces graded choice probabilities via a softmax rule applied to its internal evidence (e.g., validity of the discriminating cue for TTB, cue sum for Tallying, log-odds sum for WADD) using a shared choice temperature. Individuals probabilistically select one of these softened heuristics on any given trial.

**Rationale:** Following the critic's advice, I reverted to the accepted Iteration 2 base which uses a linear normalization for the strategy mixture weights. To improve the model's ability to fit trials without requiring extreme parameter tradeoffs, I softened the deterministic edges of the TTB and Tallying sub-models. A single `beta_choice` parameter now applies a softmax function to the internal evidence of each strategy (discriminating cue validity for TTB, cue sum for Tallying, and log-odds sum for WADD) before they are mixed. This provides smoother gradients and graded probabilities when internal evidence is weak, allowing the mixture model to better capture human preference levels.

**Parameters:**
  - `w_ttb`: `[0.0, 50.0]`
  - `w_tally`: `[0.0, 50.0]`
  - `w_wadd`: `[0.0, 50.0]`
  - `beta_choice`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    w_tally = float(parameters["w_tally"])
    w_wadd = float(parameters["w_wadd"])
    beta_choice = float(parameters["beta_choice"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights to get mixture probabilities for the strategies
    total_w = w_ttb + w_tally + w_wadd + 1e-9
    p_ttb = w_ttb / total_w
    p_tally = w_tally / total_w
    p_wadd = w_wadd / total_w
    
    # Strategy 1: Take-The-Best (TTB) Evidence
    order = np.argsort(val)[::-1]
    ttb_score_a, ttb_score_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_score_a, ttb_score_b = val[idx], 0.0
            break
        elif b[idx] > a[idx]:
            ttb_score_a, ttb_score_b = 0.0, val[idx]
            break
            
    z_ttb = beta_choice * np.array([ttb_score_a, ttb_score_b])
    z_ttb -= np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    ttb_probs = e_ttb / np.sum(e_ttb)
    ttb_a, ttb_b = ttb_probs[0], ttb_probs[1]
            
    # Strategy 2: Tallying (Equal Weights) Evidence
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    z_tally = beta_choice * np.array([sum_a, sum_b])
    z_tally -= np.max(z_tally)
    e_tally = np.exp(z_tally)
    tally_probs = e_tally / np.sum(e_tally)
    tally_a, tally_b = tally_probs[0], tally_probs[1]
        
    # Strategy 3: Weighted Additive (WADD) Evidence
    val_clipped = np.clip(val, 0.501, 0.999)
    w = np.log(val_clipped / (1.0 - val_clipped))
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    
    z_wadd = beta_choice * np.array([score_a, score_b])
    z_wadd -= np.max(z_wadd)
    e_wadd = np.exp(z_wadd)
    wadd_probs = e_wadd / np.sum(e_wadd)
    wadd_a, wadd_b = wadd_probs[0], wadd_probs[1]
        
    # Probabilistic mixture of discrete strategy predictions
    p_A_core = p_ttb * ttb_a + p_tally * tally_a + p_wadd * wadd_a
    p_B_core = p_ttb * ttb_b + p_tally * tally_b + p_wadd * wadd_b
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_9` — SURVIVED ✓

**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Rationale:** Following the critic's advice, we implement a Sequential Evidence Accumulation model where features are sampled in descending order of validity. By allowing the stopping threshold to range from 0.0 to 3.0, the model can dynamically recover Take-The-Best behavior (when threshold <= 1.0, stopping after the first differentiating feature) or Tallying behavior (when threshold is high, accumulating across all features). This single unified mechanism avoids the scaling issues of the mixture model and naturally explains intermediate choice proportions observed in the experiments.

**Parameters:**
  - `threshold`: `[0.0, 3.0]`
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_11` → slot 1 (via `new_theory`)

**Description:** Decision-makers use a Validity-Weighted Sequential Accumulation strategy. They inspect features one by one in descending order of validity, accumulating evidence proportional to a non-linear transformation of each feature's chance-centered validity. If the absolute value of this weighted accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice. Otherwise, they evaluate all features and decide based on the final weighted tally. The weight transformation uses the absolute chance-centered validity raised to a power, preserving the sign, to avoid numerical errors while allowing interpolation between Tallying and Take-The-Best.

**Rationale:** Applying the minimal diff requested by the critic: the weight calculation is updated to `np.sign(centered_val) * (np.abs(centered_val) ** gamma)` to safely apply the power transformation to chance-centered validities without raising negative numbers to float powers (which causes NaNs). This preserves the arbiter's intended Validity-Weighted Sequential Accumulation mechanism while fixing the numerical bug that caused the previous iteration's rejection.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Parameters
    gamma = float(parameters["gamma"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into weights safely
    centered_val = val - 0.5
    weights = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx] * weights[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
