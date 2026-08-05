# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) heuristic posits a lexicographic decision rule where individuals search through features in descending order of validity. They stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. If no feature discriminates, they guess randomly. To account for empirical levels of noise, the choice is mixed with a lapse rate (epsilon) that can span up to 1.0 (pure guessing).

**Rationale:** Following the critic's feedback, the strict Take The Best (TTB) mechanism produced predictions that were too extreme compared to human data. By widening the `epsilon` parameter range from `[0.0, 0.5]` to `[0.0, 1.0]`, the model can accommodate higher levels of response noise and random guessing, naturally regressing the predictions toward the empirical means without altering the core lexicographic mechanism.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Strategy Selection Theory (TTB + WADD): Individuals probabilistically mix between a non-compensatory lexicographic heuristic (Take The Best) and a compensatory heuristic (Weighted Additive, WADD). To account for varying degrees of confidence even when using a lexicographic rule, the TTB component makes probabilistic predictions rather than purely deterministic ones. Depending on individual differences or trial-by-trial strategy selection, a decision-maker relies on the single best discriminating cue a certain fraction of the time, and otherwise considers the validity-weighted sum of all feature differences.

**Rationale:** Following the critic's feedback, we maintained the exact TTB + WADD mixture model from Iteration 4 but decoupled the mixture weight from the TTB confidence. We constrained 'delta' to a strict, tight range [0.0, 0.15] to ensure that whenever the TTB strategy is selected, its predictions remain sharp and confident, preserving the strong TTB match rate observed in Experiment 4. Simultaneously, we widened the strategy selection weight 'w_ttb' to [0.1, 0.85] to allow the model to rely more frequently on the WADD component, capturing the compensatory behavior seen in Experiments 1 and 2 without letting the TTB component degrade into noise.

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `w_ttb`: `[0.1, 0.85]`
  - `delta`: `[0.0, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction with confidence delta
    order = np.argsort(validities)[::-1]
    delta = float(parameters["delta"])
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0 - delta, delta])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([delta, 1.0 - delta])
            break
            
    # WADD prediction
    scores = stim @ validities
    
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # Mix strategies
    w_ttb = float(parameters["w_ttb"])
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # Add lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Downstream Tallying Mixture: Decision makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in descending order of validity. However, to account for corroboration and tie-breaking effects, individuals occasionally mix this strategy with a 'downstream tally'—an equal-weight count of only the remaining, unexamined cues that have lower validity than the discriminating cue. This prevents the discriminating cue from double-contributing to the tally, reducing over-prediction of compensatory WADD-like behavior in environments with highly valid cues, while preserving compensatory variance in environments where downstream cues strongly oppose the best cue.

**Rationale:** Applying the minimal-diff edit suggested by the critic: reverting to the Iteration 1 base (which uses a constant mixture of TTB and Tallying) but modifying the Tallying component to only compute over cues with lower validity than the discriminating cue. This 'downstream tally' prevents the discriminating cue from double-contributing, dampening WADD-like compensatory artifacts in Exps 5-6 while retaining the tie-breaker and corroboration effects needed to fit Exps 1-4.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `w_tally`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    # TTB Prediction
    p_ttb = np.array([0.5, 0.5])
    discrim_idx = len(order)
    for i, idx in enumerate(order):
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_idx = i
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_idx = i
            break
            
    # Downstream Tallying Prediction (only unexamined cues)
    if discrim_idx < len(order) - 1:
        remaining_indices = order[discrim_idx + 1:]
        tally_a = np.sum(a[remaining_indices])
        tally_b = np.sum(b[remaining_indices])
    else:
        tally_a = 0
        tally_b = 0
        
    if tally_a > tally_b:
        p_tally = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Mix TTB and Downstream Tallying
    w_tally = float(parameters["w_tally"])
    p_mix = (1.0 - w_tally) * p_ttb + w_tally * p_tally
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
