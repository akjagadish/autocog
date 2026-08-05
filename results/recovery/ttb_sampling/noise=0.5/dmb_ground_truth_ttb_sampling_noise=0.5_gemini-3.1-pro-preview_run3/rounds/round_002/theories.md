# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: People make binary choices by evaluating features sequentially in descending order of their validity. The first feature that discriminates between the two options (i.e., one option has a higher value than the other) entirely determines the choice, and all remaining features are ignored. This strictly non-compensatory strategy allows decision makers to heavily weight highly predictive cues without needing to compute complex compensatory trade-offs.

**Rationale:** Following the arbiter's guidance, this model replaces the compensatory Tallying heuristic with the Take-The-Best (TTB) heuristic. Instead of summing up all winning features, TTB evaluates cues sequentially based on their validity. The first cue that discriminates between the two options completely determines the choice, and all subsequent cues are ignored. This captures the strong preference for high-validity cues observed in the data in a purely non-compensatory (lexicographic) manner. Response noise is handled via a softmax over the binary decision outcome and an independent lapse rate.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    scores = np.array([0.0, 0.0])
    
    # Evaluate features one by one in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse
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


### slot 2 — `pi_2_1` — KILLED ✗

**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** Following the arbiter's instructions, the unconstrained `weights` array has been removed. Instead, the model computes the weighted sum directly using the instructed `validities`. To account for subjective distortions in how people evaluate probabilities or validities, a single scalar parameter `gamma` is introduced, which exponentiates the validities (`subjective_weight = validities ** gamma`). This significantly reduces the degrees of freedom compared to an unconstrained weight vector, preventing overfitting and ensuring the model mechanism leverages the experiment's provided validities.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Apply subjective distortion to validities
    subjective_weights = validities ** gamma
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ subjective_weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

**Rationale:** Following the latest feedback, I reverted to the Iteration 1 base which uses a single beta parameter for both TTB and WADD. To fix the scaling mismatch that forces the optimization into a compromise, I normalized the WADD scores by dividing them by the sum of subjective weights. This strictly bounds the WADD scores to [0, 1], perfectly aligning with the 0/1 scale of the TTB scores. This minimal edit resolves the issue without introducing the degrees of freedom that caused the previous decoupling attempt to fail.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
