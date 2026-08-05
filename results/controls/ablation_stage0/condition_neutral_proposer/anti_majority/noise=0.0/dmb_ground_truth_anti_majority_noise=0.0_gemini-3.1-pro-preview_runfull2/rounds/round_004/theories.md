# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

**Rationale:** Following the critic's advice, I shifted the `alpha` parameter range from [0.0, 1.0] to [0.5, 1.0] to reflect the strong empirical bias toward non-compensatory choices (TTB-like behavior) observed in the data. I also increased the lower bound of `beta` from 0.1 to 1.0 to reduce baseline noise in the softmax functions and prevent the choice probabilities from flattening too much toward 0.5.

**Parameters:**
  - `alpha`: `[0.5, 1.0]`
  - `beta`: `[1.0, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
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
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
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
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
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


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Rank-Dependent Tallying: Decision-makers primarily evaluate options based on the sheer number of positive features (a tallying-like process), but the weight of each feature is subject to rank-based discounting. Rather than scaling exponentially with log-odds, a feature's weight decays as a power law of its validity rank. This ensures a strong compensatory mechanism where multiple moderate cues can easily overpower a single high-validity cue, and tallying differences dominate choice probabilities unless the validity rank differences are extreme.

**Rationale:** Shifted the `gamma` parameter range even higher to `[2.0, 8.0]` as suggested by the critic. This further steepens the power-law decay of cue weights, penalizing lower-ranked cues more heavily. This reduces the still slightly excessive compensatory behavior observed in the previous iteration and brings the model's predictions closer to the empirical data in Experiments 2 and 4.

**Parameters:**
  - `gamma`: `[2.0, 8.0]`
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
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Assign ranks to features based on validity (1 = highest validity)
    n_features = len(val)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros(n_features)
    ranks[order] = np.arange(1, n_features + 1)
    
    # Rank-based discounting: weight decays as an inverse power of rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted tally for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Sequential Evidence Accumulation with Probabilistic Stopping, Urgency, and Memory Decay: Decision-makers evaluate cues sequentially by validity, accumulating evidence with a non-linear scaling of log-odds. Crucially, evidence from earlier cues decays over time (leakage), allowing later cues to overpower early ones if search continues. A step-dependent urgency signal increases the likelihood of stopping as search progresses, capturing both compensatory behavior and strong negative contrast effects.

**Rationale:** Following the critic's advice, I have reverted to the Iteration 3 base architecture (softmax choice, memory decay, urgency, and non-linear log-odds) and expanded the upper bounds for gamma and alpha. This allows the stopping probability to become hyper-sensitive to the evidence gap, enabling the stark behavioral dichotomy needed to capture the deep negative contrast effects.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.1, 50.0]`
  - `theta`: `[0.0, 10.0]`
  - `delta`: `[-5.0, 10.0]`
  - `alpha`: `[0.1, 10.0]`
  - `phi`: `[0.0, 1.0]`
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
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    delta = float(parameters["delta"])
    alpha = float(parameters["alpha"])
    phi = float(parameters["phi"])
    epsilon = float(parameters["epsilon"])
    
    # Convert validities to log-odds for evidence accumulation
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    w = log_odds ** alpha
    
    # Process cues in descending order of validity
    order = np.argsort(-val, kind="stable")
    
    E = 0.0
    P_A = 0.0
    P_reach = 1.0
    n_features = len(val)
    
    for idx, i in enumerate(order):
        # Accumulate evidence with memory decay (leakage)
        E = E * phi + w[i] * (a[i] - b[i])
        
        if idx == n_features - 1:
            p_stop = 1.0
        else:
            # Probabilistic stopping based on current evidence gap and urgency
            x = gamma * np.abs(E) - theta + delta * idx
            if x > 20:
                p_stop = 1.0
            elif x < -20:
                p_stop = 0.0
            else:
                p_stop = 1.0 / (1.0 + np.exp(-x))
                
        # Probability of choosing A given current evidence E
        scores = np.array([E, 0.0])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_choose_A = e[0] / np.sum(e)
        
        P_A += P_reach * p_stop * p_choose_A
        P_reach *= (1.0 - p_stop)
        
    P_B = 1.0 - P_A
    p_final = (1.0 - epsilon) * np.array([P_A, P_B]) + epsilon * 0.5
    
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
