# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Weighted Additive Model (WADD) with Baseline-Adjusted Validity Weights: Decision-makers evaluate options by computing a weighted sum of their features. The weights are derived from the subjective validities of the cues, adjusted for the chance baseline (0.5), scaled by a non-linear exponent (gamma), and normalized. This centers the evidence accumulation on the actual informative value of the cues, allowing the model to smoothly interpolate between Tallying and highly validity-driven weighting, while remaining compensatory. The final choice is made probabilistically using a softmax rule over the computed values, along with an independent lapse rate for random errors.

**Rationale:** Based on the critic's feedback, the log-odds weighting distorted the subjective evidence accumulation process, leading to a rejection. Reverting to the power-law weighting, but subtracting the chance baseline of 0.5 before exponentiating, ensures that evidence is centered on the actual informative value of the cues. This allows the model to better differentiate between strong and weak cues, aiming to correct the directional errors in Experiments 6 and 10 while maintaining the compensatory nature of the WADD family.

**Parameters:**
  - `gamma`: `[0.0, 20.0]`
  - `tau`: `[0.0, 100.0]`
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
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Compute weights as a power function of baseline-adjusted validities
    # This centers the evidence accumulation on the actual informative value of the cues.
    w = np.maximum(0.0, validities - 0.5) ** gamma
    
    # Normalize weights to prevent vanishing values for large gamma
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones_like(w) / len(w)
    
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
