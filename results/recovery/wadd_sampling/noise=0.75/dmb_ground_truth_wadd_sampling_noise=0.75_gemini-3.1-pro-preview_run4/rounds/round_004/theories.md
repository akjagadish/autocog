# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Probabilistic Sequential Search Theory posits that decision-makers evaluate cues sequentially in order of their validity. Instead of adopting a strict stopping rule like Take-The-Best or exhaustively evaluating all cues like WADD, individuals accumulate evidence dynamically. When a cue discriminates between options, there is a constant probability (stop_rate) of stopping the search and deciding based on the accumulated evidence. This creates a flexible, cue-by-cue evidence accumulation process that naturally blends lexicographic and compensatory behaviors, explaining the softer choice probabilities and context-dependent trade-offs seen in human decision-making without forcing an overly aggressive stopping rule.

**Rationale:** Following the critic's advice, the parameter bounds for `epsilon` have been expanded to [0.0, 1.0] and the lower bound for `beta` has been reduced to 0.0. This allows the optimizer to capture the high degree of baseline guessing and softer choice probabilities observed across the human experiments, where metric values consistently hover around 0.45-0.55. These minimal adjustments to the parameter ranges enable the model to flatten its choice probabilities appropriately without altering the core Probabilistic Sequential Search mechanism.

**Parameters:**
  - `stop_rate`: `[0.0, 1.0]`
  - `beta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-val_clipped, kind="stable").tolist()
    
    stop_rate = float(parameters["stop_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    P_A = 0.0
    P_B = 0.0
    P_continue = 1.0
    E = 0.0
    
    n_features = len(val)
    
    for i, idx in enumerate(cue_order):
        diff = a[idx] - b[idx]
        w = np.log(val_clipped[idx] / (1.0 - val_clipped[idx]))
        E += w * diff
        
        # Determine stopping probability at this step
        if i == n_features - 1:
            S_t = 1.0
        else:
            if diff != 0:
                # Cue discriminates: stop probability is constant stop_rate
                S_t = stop_rate
            else:
                # Cue does not discriminate: always continue
                S_t = 0.0
                
        # Probability of choosing A if search stops here (using stable sigmoid)
        z = beta * E
        if z > 20:
            p_A_if_stop = 1.0
        elif z < -20:
            p_A_if_stop = 0.0
        else:
            p_A_if_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_B_if_stop = 1.0 - p_A_if_stop
        
        # Accumulate overall choice probabilities
        P_A += P_continue * S_t * p_A_if_stop
        P_B += P_continue * S_t * p_B_if_stop
        
        # Update probability of continuing to the next cue
        P_continue *= (1.0 - S_t)
        
    # Apply lapse rate (epsilon)
    P_A = (1.0 - epsilon) * P_A + epsilon * 0.5
    P_B = (1.0 - epsilon) * P_B + epsilon * 0.5
    
    return np.array([P_A, P_B])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

**Rationale:** Independent temperature parameters (`beta_wadd`, `beta_tally`, `beta_ttb`) are implemented for each strategy to resolve scaling artifacts. Initial logic and parameters are validated. Standard processing applied to compute raw heuristic scores. The final transformation introduces distinct betas to scale scores independently before the mixture weighted average is calculated, ensuring optimal calibration without manual bounding constraints.

**Parameters:**
  - `beta_wadd`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_wadd`: `[0.0, 1.0]`
  - `w_tally`: `[0.0, 1.0]`
  - `w_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Two-Stage Contingent Processing Theory with Relative Thresholding: Decision-makers evaluate options in two stages. First, they perform a rapid, parallel evaluation using a simple tallying heuristic. If the proportional difference in tally scores (relative to the total number of features) exceeds a subjective threshold, a choice is made immediately based on this simple difference. If the options are too similar (below threshold), the decision-maker transitions to a second, more effortful stage, evaluating the options using a mixture of weighted cues (WADD) and a strict lexicographic rule (Take-The-Best).

**Rationale:** Following the latest feedback, this minimal edit reverts to the accepted base theory from Iteration 1 but modifies the transition rule between Stage 1 and Stage 2. Instead of using the absolute tally difference (`abs_diff`) which scales inconsistently across experiments with different numbers of features, the model now normalizes the difference by the total number of features (`rel_diff = abs_diff / len(a)`). The `threshold` parameter range has been proportionally adjusted to [0.0, 1.0]. This allows the model to maintain a consistent standard for 'decisiveness' across experiments, softening the exaggerated context effect in Experiment 9 without breaking the precise balance achieved in other experiments.

**Parameters:**
  - `threshold`: `[0.0, 1.0]`
  - `beta_thresh`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `beta_wadd`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 10.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    
    # Stage 1: Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    abs_diff = abs(a_wins - b_wins)
    rel_diff = abs_diff / len(a)
    
    threshold = float(parameters["threshold"])
    beta_thresh = float(parameters["beta_thresh"])
    
    # Probability of stopping at Stage 1 (using stable sigmoid)
    z_thresh = beta_thresh * (rel_diff - threshold)
    if z_thresh > 20:
        p_stage1 = 1.0
    elif z_thresh < -20:
        p_stage1 = 0.0
    else:
        p_stage1 = 1.0 / (1.0 + np.exp(-z_thresh))
        
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_tally = get_probs(np.array([a_wins, b_wins]), float(parameters["beta_tally"]))
    
    # Stage 2: Effortful processing (Mixture of WADD and TTB)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.array([np.dot(a, w), np.dot(b, w)])
    p_wadd = get_probs(scores_wadd, float(parameters["beta_wadd"]))
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    p_ttb = get_probs(scores_ttb, float(parameters["beta_ttb"]))
    
    w_wadd = float(parameters["w_wadd"])
    p_stage2 = w_wadd * p_wadd + (1.0 - w_wadd) * p_ttb
    
    # Combine stages
    p_mix = p_stage1 * p_tally + (1.0 - p_stage1) * p_stage2
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
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
