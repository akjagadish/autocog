# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Cue-Reading Errors: Decision-makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in strictly descending order of validity. However, they have a non-zero probability of misreading or misremembering individual cue values. This preserves the strict non-compensatory stopping rule while introducing variance that mimics compensatory behavior in certain environments.

**Rationale:** Following the critic's recommendation, we pivot from adding noise to the validities (which caused excessive cue reordering and failed to balance the experiments) to 'Take-The-Best with Cue-Reading Errors'. In this model, the decision-maker strictly orders cues by their true validities but has a small probability (p_error) of misreading or misremembering the value of any individual cue. This maintains the strict non-compensatory stopping rule necessary for Experiments 5-8 while injecting a different kind of noise that can better capture the variance in Experiments 1 and 2.

**Parameters:**
  - `p_error`: `[0.0, 0.2]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_error = float(parameters["p_error"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by true validities descending
    order = np.argsort(validities)[::-1]
    
    n_samples = 200
    p_a = 0.0
    
    for _ in range(n_samples):
        # Independent probability of misreading each cue
        err_a = np.random.rand(len(a)) < p_error
        err_b = np.random.rand(len(b)) < p_error
        
        # Apply errors (flip 0 to 1, and 1 to 0)
        a_noisy = np.abs(a - err_a)
        b_noisy = np.abs(b - err_b)
        
        for idx in order:
            if a_noisy[idx] > b_noisy[idx]:
                p_a += 1.0
                break
            elif b_noisy[idx] > a_noisy[idx]:
                break
        else:
            p_a += 0.5
            
    p_a /= n_samples
    p_core = np.array([p_a, 1.0 - p_a])
    
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
