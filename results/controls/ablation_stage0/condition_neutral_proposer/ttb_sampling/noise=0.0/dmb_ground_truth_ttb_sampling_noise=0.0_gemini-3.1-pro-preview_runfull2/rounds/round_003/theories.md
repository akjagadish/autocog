# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Rationale:** Following the arbiter's instructions, this model implements the Take The Best (TTB) heuristic. Instead of computing a compensatory score (like WADD) or an unweighted sum (like Tallying), TTB sequentially evaluates features in decreasing order of validity. It stops at the first cue that discriminates and bases the choice entirely on it. This explains the high variance in Experiment 1 because trials with identical tally differences can have their decisions driven by different cues depending on the specific feature patterns. It also explains the strong preference in Experiment 2 for the 'fewer but better' option, as the single highest-validity cue dominates the decision regardless of deficits on multiple lower-validity cues.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Strategy Mixture: Decision-makers predominantly use a deterministic non-compensatory heuristic (Take-The-Best) but probabilistically substitute it with a simple compensatory heuristic (Tallying) on a trial-by-trial basis. This mixture captures the overwhelming adherence to TTB while accounting for systematic deviations toward options with a higher quantity of positive cues in extreme conflict scenarios.

**Rationale:** Tightened the parameter ranges for p_ttb to [0.6, 1.0] and epsilon to [0.0, 0.2] as suggested by the critic. This ensures the model predominantly relies on Take-The-Best, yielding the ~83-85% TTB adherence observed in the data, while still allowing the Tallying strategy to explain the systematic deviations seen in extreme conflict trials.

**Parameters:**
  - `p_ttb`: `[0.6, 1.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB) Strategy
    order = np.argsort(validities, kind='stable')[::-1]
    p_ttb_choice = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying Strategy (Equal Weights)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_choice = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    # Mixture of the two strategies
    p_core = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Exponentially Weighted Compensatory Model (Rank-based with Strict Dominance Base): Decision-makers evaluate options using a single compensatory process where the weight of each feature grows exponentially with its subjective validity rank, using a base >= 2.0. This guarantees strict lexicographic dominance, ensuring that a single higher-ranked cue always outweighs all lower-ranked cues combined. By enforcing this strict dominance, the model acts identically to Take-The-Best across all conflict scenarios, relying on an independent lapse rate for probabilistic errors rather than softening the decision temperature.

**Rationale:** Following the critic's advice, I shifted the `base` parameter range to `[2.0, 10.0]`. This enforces strict lexicographic dominance (since base >= 2), guaranteeing that a higher-ranked cue will always outweigh all lower-ranked cues combined. This prevents the fitter from adopting a small `tau` to soften primary-cue conflicts (which inadvertently makes secondary-cue conflicts too noisy), instead forcing the model to rely on `epsilon` for uniform lapses while keeping `tau` large enough to drive decisive choices even on secondary cues.

**Parameters:**
  - `base`: `[2.0, 10.0]`
  - `tau`: `[0.0, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    base = float(parameters["base"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights using exponential growth based on rank
    # A base >= 2.0 ensures strict TTB behavior (lexicographic dominance)
    order = np.argsort(validities, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(validities))
    
    w = base ** ranks
    
    # Compute overall value for each option
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    # Compute choice probabilities using softmax over values
    logits = tau * np.array([v_a, v_b])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
