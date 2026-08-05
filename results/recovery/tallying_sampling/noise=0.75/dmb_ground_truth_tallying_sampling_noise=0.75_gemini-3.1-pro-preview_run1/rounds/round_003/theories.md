# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.

**Rationale:** Following the critic's advice, the parameter range for `beta` has been slightly widened to [0.1, 3.5]. This intermediate range aims to strike a better balance by restoring some of the compensatory determinism lost in the previous iteration, which should help recover performance in Experiments 2 and 6 while maintaining the gains achieved in Experiments 3 and 4.

**Parameters:**
  - `mixture_p`: `[0.0, 1.0]`
  - `beta`: `[0.1, 3.5]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    mixture_p = float(parameters['mixture_p'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # --- TTB Component ---
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- WADD Component ---
    # WADD uses validities as weights
    scores = stim @ val
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # --- Mixture ---
    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    
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


### slot 2 — `pi_2` — KILLED ✗

**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Sequential Evidence Accumulation: Decision-making is driven by a sequential sampling process where features are evaluated in order of their subjective validity. As each feature is processed, the validity-weighted difference between the options is added to a running accumulator. If this accumulated evidence reaches a predefined threshold at any point, search is immediately terminated and a choice is made (mimicking non-compensatory heuristics like Take-The-Best). If all features are exhausted without the evidence hitting the boundary, the subject makes a probabilistic choice based on the final accumulated tally (mimicking compensatory strategies like WADD). This single-process model naturally unifies fast-and-frugal heuristics and exhaustive compensatory integration depending on the height of the evidence threshold.

**Rationale:** Following the critic's advice, we further reduce the upper bound of the `beta` parameter from 5.0 to 1.5. This makes the soft choice component (used when the sequential evidence threshold is not reached) even more probabilistic. This minimal edit aims to further dampen the model's overestimation of compensatory and deterministic responding in Experiments 6, 7, and 8, bringing the simulated metrics closer to the muted empirical effects observed in the data.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 1.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    threshold = float(parameters['threshold'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    A_val = 0.0
    decided = False
    p_a = 0.5
    
    # Sequentially accumulate evidence
    for j in cue_order:
        A_val += val[j] * (a[j] - b[j])
        if A_val >= threshold:
            p_a = 1.0
            decided = True
            break
        elif A_val <= -threshold:
            p_a = 0.0
            decided = True
            break
            
    # If boundary not reached, make a soft choice based on the final tally
    if not decided:
        z = beta * A_val
        # Numerically stable logistic function for P(A)
        if z > 100:
            p_a = 1.0
        elif z < -100:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    p_b = 1.0 - p_a
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
