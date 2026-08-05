# Round 19 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_20` — KILLED ✗

**Description:** Non-linear Feature Difference Model with Flexible Validity Centering

**Rationale:** The critic's suggestion in iteration 5 (V_A = net_A if net_A > 0 else lambda_loss * net_A) is mathematically degenerate for a binary choice softmax, as the difference V_A - V_B simplifies to net_A * (1 + lambda_loss), which is exactly equivalent to the linear model with a scaled beta. It provides no new non-linear representational power. To genuinely improve upon the accepted iteration 3 base without introducing mathematical degeneracies, I am retaining the successful `gamma` power function (which properly compresses/expands the evidence difference) but introducing a flexible validity centering parameter `theta` instead of hardcoding it to 0.5. This allows the model to smoothly interpolate between pure Tallying and strict validity weighting, giving it the flexibility needed to fix the miscalibrations in Exps 13, 20, and 22.

**Parameters:**
  - `theta`: `[0.0, 1.0]`
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 20.0]`
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
    
    # Center validities with a flexible threshold rather than hardcoded 0.5
    theta = float(parameters["theta"])
    w = val - theta
    
    diff = a - b
    # Compute net evidence for option A over B
    net_ev = np.sum(w * diff)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply non-linear value function to the net evidence (Prospect Theory style)
    score = np.sign(net_ev) * (np.abs(net_ev) ** gamma)
    
    # Create scores for A and B
    scores = np.array([score, -score])
    
    # Softmax choice rule
    z = beta * scores
    z -= np.max(z) # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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


### slot 2 — `pi_21` — SURVIVED ✓

**Description:** Decision-makers evaluate options using a Leaky Competing Accumulator (LCA) process, where attention shifts sequentially across features in order of their validity. Evidence for each option accumulates continuously over time, subject to information decay (leakage) and lateral inhibition. When lateral inhibition is strong, early evidence from highly valid features quickly suppresses the competing option, locking in a choice and producing non-compensatory 'Take-The-Best' behavior. Conversely, when leakage and inhibition are low, evidence from all features integrates more evenly without suppression, resulting in compensatory 'Tallying' or WADD-like behavior. This provides a unified mechanistic account for the spectrum of decision strategies without requiring explicit rule-switching.

**Rationale:** Following the arbiter's feedback, this theory abandons strict sequential heuristics and simultaneous weighted sums in favor of a Leaky Competing Accumulator (LCA) framework. Features are attended to sequentially in order of their validity, providing inputs to two competing accumulators representing the options. The dynamics of accumulation are governed by leakage (decay of past evidence) and lateral inhibition (suppression of the weaker option). This naturally bridges the gap between compensatory and non-compensatory behavior: strong lateral inhibition causes early, highly valid features to establish a dominant activation that suppresses subsequent evidence (mimicking 'Take-The-Best'), while low inhibition allows all features to be integrated evenly (mimicking 'Tallying' or WADD).

**Parameters:**
  - `leak`: `[0.0, 2.0]`
  - `inhibition`: `[0.0, 5.0]`
  - `steps_per_feature`: `[1, 50]`
  - `gamma`: `[0.1, 5.0]`
  - `beta`: `[0.1, 20.0]`
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
    
    # Sort features by validity (descending) to simulate sequential attention
    order = np.argsort(val)[::-1]
    
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    steps_per_feature = int(float(parameters["steps_per_feature"]))
    gamma = float(parameters["gamma"])
    
    x_A, x_B = 0.0, 0.0
    dt = 0.1
    
    for idx in order:
        # Non-linear scaling of validities centered at chance
        v = max(0.0, val[idx] - 0.5) ** gamma
        I_A = a[idx] * v
        I_B = b[idx] * v
        
        # Accumulate evidence over time for the current feature
        for _ in range(steps_per_feature):
            dx_A = (I_A - leak * x_A - inhibition * x_B) * dt
            dx_B = (I_B - leak * x_B - inhibition * x_A) * dt
            
            # Rectified linear units (activations cannot be negative)
            x_A = max(0.0, x_A + dx_A)
            x_B = max(0.0, x_B + dx_B)
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([x_A, x_B])
    
    # Softmax choice rule with numerical stability
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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

### `pi_22` → slot 1 (via `new_theory`)

**Description:** Decision-makers evaluate options by probabilistically selecting among a 'Heuristic Toolbox' (Take-The-Best, Tallying, and Weighted Additive) on a trial-by-trial basis. Strategy selection is driven by environment dispersion and trial-specific difficulty (normalized evidence margins). Tallying can be actively suppressed in high-dispersion environments, naturally shifting probability mass toward TTB and WADD.

**Rationale:** Following the critic's feedback, we revert to the highly successful Iteration 6 base and add a `w_tally_dispersion` parameter to the Tallying logit. This allows the model to actively suppress Tallying in high-dispersion environments (where equal weighting is sub-optimal), naturally shifting the strategy selection probability mass toward TTB and WADD without over-parameterizing the WADD logit or introducing trial-specific TTB margins that were previously rejected.

**Parameters:**
  - `base_ttb`: `[-5.0, 5.0]`
  - `base_tally`: `[-5.0, 5.0]`
  - `w_dispersion`: `[-10.0, 10.0]`
  - `w_margin`: `[-10.0, 10.0]`
  - `w_wadd_margin`: `[-10.0, 10.0]`
  - `w_tally_dispersion`: `[-10.0, 10.0]`
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
    
    # 1. Take-The-Best (TTB)
    order = np.argsort(val)[::-1]
    ttb_a = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_a = 0.0
            break
            
    # 2. Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    tally_margin = float(abs(a_wins - b_wins))
    tally_margin_norm = tally_margin / float(len(val))
    if a_wins > b_wins:
        tally_a = 1.0
    elif b_wins > a_wins:
        tally_a = 0.0
    else:
        tally_a = 0.5
        
    # 3. Weighted Additive (WADD)
    # Use raw validities as weights to prevent entanglement with the Tallying strategy
    w = val
    wadd_a_score = np.sum(a * w)
    wadd_b_score = np.sum(b * w)
    wadd_margin = float(abs(wadd_a_score - wadd_b_score))
    max_wadd_margin = float(np.sum(val))
    wadd_margin_norm = wadd_margin / max_wadd_margin if max_wadd_margin > 0 else 0.0
    
    if wadd_a_score > wadd_b_score:
        wadd_a = 1.0
    elif wadd_b_score > wadd_a_score:
        wadd_a = 0.0
    else:
        wadd_a = 0.5
        
    # Strategy Selection Logits
    base_ttb = float(parameters["base_ttb"])
    base_tally = float(parameters["base_tally"])
    w_dispersion = float(parameters["w_dispersion"])
    w_margin = float(parameters["w_margin"])
    w_wadd_margin = float(parameters["w_wadd_margin"])
    w_tally_dispersion = float(parameters["w_tally_dispersion"])
    
    dispersion = float(np.std(val))
    
    # Logits depend on base rates, environment dispersion, and normalized trial-specific difficulty (margins)
    logit_ttb = base_ttb + w_dispersion * dispersion
    logit_tally = base_tally + w_margin * tally_margin_norm + w_tally_dispersion * dispersion
    logit_wadd = w_wadd_margin * wadd_margin_norm  # WADD base rate is fixed at 0 for identifiability
    
    logits = np.array([logit_ttb, logit_tally, logit_wadd])
    logits -= np.max(logits)  # For numerical stability
    probs = np.exp(logits)
    probs /= np.sum(probs)
    
    p_ttb, p_tally, p_wadd = probs
    
    # Aggregate strategy predictions
    p_a = p_ttb * ttb_a + p_tally * tally_a + p_wadd * wadd_a
    p_b = 1.0 - p_a
    
    # Apply lapse/noise rate
    epsilon = float(parameters["epsilon"])
    p_a_final = (1.0 - epsilon) * p_a + epsilon * 0.5
    p_b_final = (1.0 - epsilon) * p_b + epsilon * 0.5
    
    return np.array([p_a_final, p_b_final])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
