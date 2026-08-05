# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People make choices by computing a weighted sum of the features for each option (the Weighted Additive, or WADD, heuristic). Unlike Take The Best or Tallying, WADD scales the contribution of each feature by its validity. Furthermore, features are processed as bipolar evidence: the presence of a feature adds to the option's value proportionally to its validity, while its absence actively penalizes the option. The resulting option scores are translated into choice probabilities via a softmax function, with an independent lapse rate accounting for random guessing.

**Rationale:** Following the latest feedback, we revert to the raw validities as weights but change the feature representation from {0, 1} to {-1, 1}. This ensures that the absence of a feature actively penalizes an option, fundamentally altering the score differences and the compensatory dynamics. This structural change aims to capture the human deviation from the reference probabilities in Experiment 2 while preserving the core WADD mechanism that succeeded in Experiment 1.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # WADD (Weighted Additive) heuristic.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    
    # Center the stimulus features to -1 and 1 so absence of a feature penalizes
    stim_centered = stim * 2.0 - 1.0
    
    # Compute the weighted sum of features for each option.
    # The weights are directly proportional to the cue validities.
    scores = np.dot(stim_centered, val)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Weighted Additive with Choice Inertia: Decision-makers evaluate options by taking a weighted sum of their features, where the weights correspond to cue validities. However, choices are not made independently across trials; they are subject to choice inertia. A stickiness parameter biases the current choice toward the option selected in the immediately preceding trial. The final decision is made probabilistically via a softmax function over the adjusted option values, with an additional lapse rate to account for random errors.

**Rationale:** Following the arbiter's feedback, this theory models choice by evaluating options based on their features (using weighted addition with validities as weights) while introducing a 'stickiness' parameter. This parameter adds a constant bonus to the value of the option chosen in the previous trial, capturing the choice inertia and sequence effects evident in the conditional probabilities of the experimental data. The combination of validities, stickiness, softmax temperature (beta), and a lapse rate (epsilon) provides a flexible yet constrained mechanism to explain both feature-driven preferences and sequential dependencies.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `stickiness`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted addition of features
    v_a = np.sum(val * a)
    v_b = np.sum(val * b)
    
    # Apply choice inertia (stickiness) if there is a previous trial
    stickiness = float(parameters['stickiness'])
    if len(history['response']) > 0:
        prev_response = history['response'][-1]
        if prev_response == 0:
            v_a += stickiness
        elif prev_response == 1:
            v_b += stickiness
            
    scores = np.array([v_a, v_b])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Take-The-Best with Choice Inertia and Independent Heuristic Weight: Decision-makers evaluate options using a non-compensatory lexicographic heuristic, stopping at the first discriminating cue. The deterministic TTB preference is scaled by an individual-specific weight and integrated with choice inertia (a bias toward the previously chosen option) via a standard softmax. This parameterization separates the reliance on the heuristic from the reliance on inertia, eliminating redundancy and capturing a wide variance of subject types.

**Rationale:** Initial logic and parameters are validated. We jump directly to resolving the parameter identifiability issue by removing 'beta' and replacing the fixed TTB score with an individual 'ttb_weight'. This cleanly uncouples the heuristic dependence from choice inertia, naturally widening the spectrum of simulated subjects.

**Parameters:**
  - `ttb_weight`: `[0.0, 10.0]`
  - `stickiness`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters['validities'], dtype=float)
    cue_order = np.argsort(-val, kind='stable')
    
    base_scores = np.array([0.0, 0.0])
    ttb_weight = float(parameters['ttb_weight'])
    
    for idx in cue_order:
        if a[idx] > b[idx]:
            base_scores = np.array([ttb_weight, 0.0])
            break
        elif b[idx] > a[idx]:
            base_scores = np.array([0.0, ttb_weight])
            break
            
    stickiness = float(parameters['stickiness'])
    if len(history['response']) > 0:
        prev_response = int(history['response'][-1])
        base_scores[prev_response] += stickiness
        
    epsilon = float(parameters['epsilon'])
    
    z = base_scores - np.max(base_scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
