# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Threshold-based Sequential Search integrates the fast-and-frugal nature of Take-The-Best with the compensatory evidence accumulation of Weighted Additive (WADD) models. Decision-makers search through cues in descending order of their validity, maintaining a running tally of the evidence (weighted by each cue's validity). Instead of stopping at the very first discriminating cue, search terminates only when the absolute accumulated evidence exceeds an internal confidence threshold. If all cues are exhausted without reaching this threshold, the option with the higher accumulated evidence is chosen. This allows for fast, one-reason decisions when a highly valid cue strongly favors one option, while enabling compensatory behavior when early cues provide weak or conflicting evidence.

**Rationale:** Based on the critic's feedback, tightening the threshold to [0.0, 1.0] overcorrected and underpredicted compensatory behavior, while [0.0, 1.5] slightly overpredicted it. A middle-ground threshold range of [0.0, 1.25] gently balances fast-and-frugal stopping with compensatory accumulation. Additionally, reducing the maximum lapse rate (epsilon) to 0.25 prevents the model from relying on excessive uniform noise to fit the data.

**Parameters:**
  - `threshold`: `[0.0, 1.25]`
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.0, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    threshold = float(parameters["threshold"])
    
    # Sequential search with evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Stop search if the confidence threshold is met or exceeded
        if abs(evidence) >= threshold:
            break
            
    # Convert accumulated evidence into discrete choice scores
    scores = np.zeros(2)
    if evidence > 0:
        scores[0] = 1.0
    elif evidence < 0:
        scores[1] = 1.0
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Strategy Mixture (Take-The-Best and Weighted Additive)

**Rationale:** Pure Take-The-Best (TTB) underpredicts compensatory behavior, while pure Weighted Additive (WADD) drastically overpredicts it. In reality, human decision-making is heterogeneous: some subjects rely on fast-and-frugal heuristics like TTB, while others integrate information compensatorily like WADD, or individuals probabilistically switch between strategies depending on task demands. This Strategy Mixture theory models choice as a probabilistic blend of TTB and WADD, smoothly capturing the intermediate choice proportions observed across experiments.

**Parameters:**
  - `beta_ttb`: `[0.1, 5.0]`
  - `beta_wadd`: `[0.1, 10.0]`
  - `mixture_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    # --- TTB (Take-The-Best) Process ---
    order = np.argsort(validities)[::-1]
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- WADD (Weighted Additive) Process ---
    w = np.asarray(parameters["weights"], dtype=float)
    scores_wadd = stim @ (validities * w)
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Mixture ---
    mix = float(parameters["mixture_ttb"])
    p_mix = mix * p_ttb + (1.0 - mix) * p_wadd
    
    # --- Lapse ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Probabilistic Stopping Sequential Search

**Rationale:** Following the critic's feedback on Iteration 4, we observed that attempting to sharpen the stopping threshold by increasing the slope worsened the model's performance on compensatory trials. To maintain the 'soft' probabilistic stopping that successfully fits Experiments 1, 2, and 5 while improving fits on Experiments 7 and 8 (which require earlier stopping), we shift the 'threshold' parameter range downward from [0.1, 3.0] to [0.0, 1.5]. This allows the model to probabilistically halt search earlier when evidence is modest, without making the stopping rule artificially rigid.

**Parameters:**
  - `threshold`: `[0.0, 1.5]`
  - `slope`: `[0.1, 5.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    
    threshold = float(parameters["threshold"])
    slope = float(parameters["slope"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    p_continue = 1.0
    p_A = 0.0
    
    # Sequential search with probabilistic stopping
    for i, idx in enumerate(order):
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Determine stopping probability at this step
        if i == len(order) - 1:
            p_stop = 1.0
        else:
            # Logistic function for stopping probability
            z = -slope * (abs(evidence) - threshold)
            z = np.clip(z, -50, 50)  # Prevent overflow
            p_stop = 1.0 / (1.0 + np.exp(z))
            
        p_stop_here = p_continue * p_stop
        p_continue *= (1.0 - p_stop)
        
        # Softmax choice probability if search stops at this step
        z_choice = -beta * evidence
        z_choice = np.clip(z_choice, -50, 50)
        p_A_given_stop = 1.0 / (1.0 + np.exp(z_choice))
        
        p_A += p_stop_here * p_A_given_stop
        
    p_B = 1.0 - p_A
    probs = np.array([p_A, p_B])
    
    # Blend in uniform lapse
    return (1.0 - epsilon) * probs + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
