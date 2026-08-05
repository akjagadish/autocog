# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) assumes that decision-makers do not integrate all information or simply count features. Instead, they rank features by their subjective or instructed validity and compare options lexicographically. They stop at the first feature that discriminates between the two options and choose the one with the higher value on that feature. If all features tie, they guess.

**Rationale:** Implemented the Take-The-Best (TTB) heuristic as requested by the arbiter. TTB provides a non-compensatory contrast to WADD by evaluating features lexicographically in descending order of validity, stopping at the first discriminating feature. This captures the strong reliance on the most valid cues seen in the data, without requiring full integration of all feature values.

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
    
    # Sort indices by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(-validities, kind='mergesort')
    
    score_a = 0.0
    score_b = 0.0
    
    # Lexicographic comparison
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    epsilon = float(parameters["epsilon"])
    
    if score_a > score_b:
        p_core = np.array([1.0, 0.0])
    elif score_b > score_a:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Logistic Choice Model (Weighted Additive with Non-linear Validity Scaling): Decision-makers integrate all available information by computing a weighted sum of the features for each option. However, the subjective weights placed on these features are a non-linear scaling of their objective validities, governed by an exponent parameter (gamma). This allows the compensatory process to approximate non-compensatory heuristics when gamma is large, as the highest-validity features will dominate the weighted sum. The choice is made probabilistically by passing these weighted sums through a softmax function, modulated by an inverse temperature parameter (beta) to capture choice determinism, along with a baseline lapse rate (epsilon) for random guessing.

**Rationale:** Following the critic's feedback, I kept the core mechanism of the Logistic Choice with Non-linear Validity Scaling intact, but adjusted the parameter ranges to allow for more deterministic behavior. Because validities are less than 1, raising them to a large 'gamma' power shrinks the absolute magnitude of the scores, which weakens the softmax determinism for a given 'beta'. To compensate, I increased the upper bound of 'beta' to 50.0 and restricted the lapse rate 'epsilon' to [0.0, 0.2], preventing excessive random guessing from artificially lowering the predicted choice consistency across the majority of experiments.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `gamma`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities to compute subjective weights
    weights = validities ** gamma
    
    # Compute weighted sum of features using subjective weights
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    p_core = np.exp(z)
    p_core /= p_core.sum()
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

**Description:** Adaptive Toolbox with Confidence-Dependent Strategy Selection: Decision-makers predominantly use the Take-The-Best (TTB) heuristic, but probabilistically fall back to Tallying only when their confidence is low—specifically, when the validity of the first discriminating feature falls below a subjective threshold.

**Rationale:** Following the critic's feedback, we introduce a confidence-dependent fallback mechanism. Instead of unconditionally mixing Tallying on every trial, decision-makers only fall back to Tallying if the validity of the discriminating feature found by Take-The-Best (TTB) is below a certain subjective `threshold`. If TTB terminates on a highly valid feature, the decision is made with high confidence using pure TTB, effectively reducing the Tallying probability to zero. This preserves TTB's strictness on high-validity trials (resolving the artifacts in Experiments 7 and 8 and maintaining accuracy in Experiments 1-3) while still allowing Tallying to explain systematic deviations on lower-confidence trials.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `p_tally`: `[0.0, 0.5]`
  - `threshold`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    epsilon = float(parameters["epsilon"])
    p_tally = float(parameters["p_tally"])
    threshold = float(parameters["threshold"])
    
    # Take-The-Best (TTB) Mechanism
    order = np.argsort(-validities, kind='mergesort')
    p_ttb = np.array([0.5, 0.5])
    discrim_validity = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            discrim_validity = validities[idx]
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            discrim_validity = validities[idx]
            break
            
    # Tallying Mechanism (Equal Weights)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Confidence-dependent fallback to Tallying
    if discrim_validity < threshold:
        actual_p_tally = p_tally
    else:
        actual_p_tally = 0.0
        
    # Probabilistic Strategy Selection
    p_mix = (1.0 - actual_p_tally) * p_ttb + actual_p_tally * p_tally_arr
    
    # Incorporate baseline lapse rate (random guessing)
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
