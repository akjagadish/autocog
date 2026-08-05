# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

**Rationale:** Following the latest feedback, the `beta` parameter range is further constrained from [0.0, 3.0] to [0.0, 1.0]. This minimal adjustment ensures the softmax over feature tallies produces even softer choice probabilities, correctly lowering the model's determinism to better match the observed ~59% human adherence to the tallying rule without changing the core mechanism.

**Parameters:**
  - `beta`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Tallying with Conditional Take-The-Best (TTB) Fallback: Decision makers primarily rely on a compensatory, equal-weight heuristic (Tallying). However, when the options are difficult to distinguish based on tallies alone (i.e., when the tallies are tied or very close), subjects fall back on a non-compensatory strategy, checking the single most valid discriminating cue (Take-The-Best). Importantly, this fallback is weak, meaning that even when tallies are tied, subjects often guess rather than deterministically applying TTB. This predicts localized, minor validity-driven asymmetries specifically when tallies fail to provide a strong signal, avoiding the overprediction of global validity effects characteristic of full WADD models.

**Rationale:** Following the critic's feedback, we increased the upper bound of `w_far` from 0.2 to 0.5. This allows the model to capture the baseline level of TTB usage when tallies are not tied, correcting the underprediction of validity usage on non-tie trials (Experiments 1 and 2) and the overprediction of tallying on Experiment 3. At the same time, it preserves the reduced `w_close` bound that successfully prevents overprediction on tie trials.

**Parameters:**
  - `threshold`: `{0, 1}`
  - `w_close`: `[0.0, 0.3]`
  - `w_far`: `[0.0, 0.5]`
  - `beta_tally`: `[0.0, 5.0]`
  - `beta_ttb`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Strategy 1: Tallying
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    diff = abs(tally_a - tally_b)
    
    # Strategy 2: Take-The-Best (TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    ttb_a = 0.5
    ttb_b = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_a, ttb_b = 1.0, 0.0
            break
        elif b[j] > a[j]:
            ttb_a, ttb_b = 0.0, 1.0
            break
            
    # Conditional reliance on TTB based on tally closeness
    threshold = float(parameters["threshold"])
    if diff <= threshold:
        w_ttb = float(parameters["w_close"])
    else:
        w_ttb = float(parameters["w_far"])
        
    # Softmax for Tallying
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally /= np.sum(p_tally)
    
    # Softmax for TTB
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    z_ttb -= np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb /= np.sum(p_ttb)
    
    # Mixture
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Compressed Weighted Additive (WADD) Model: Decision makers integrate all available features using a compensatory strategy, but the subjective weights applied to the cues are heavily compressed toward equal weighting. This ensures that the simple tally of features dominates the choice probability in most cases, while the objective cue validities exert a small, continuous influence across all trials. This continuous mechanism naturally breaks tally ties and produces subtle validity effects without relying on a discrete fallback to a non-compensatory Take-The-Best rule.

**Rationale:** Following the critic's advice, we revert to the exact mechanism from Iteration 1 (weights = 1.0 + gamma * validities). However, since the baseline already used a beta range of [0.0, 5.0], we further restrict the beta range to [0.0, 3.0]. This softer softmax temperature prevents tally differences from completely saturating the choice probabilities, allowing the subtle, validity-driven score differences to exert a smoother influence on the final choice and avoiding over-determinism.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.0, 3.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective weights compressed toward equal weighting (1.0)
    # The base weight of 1.0 ensures tallying dominates, while gamma scales the validity influence.
    weights = 1.0 + gamma * val
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    
    beta = float(parameters["beta"])
    z = beta * np.array([score_a, score_b])
    z -= np.max(z)
    p = np.exp(z)
    p /= np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```
