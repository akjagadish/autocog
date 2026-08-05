# Round 4 — Theories

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


### slot 2 — `pi_6` — KILLED ✗

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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Probabilistic Lexicographic Search (Position-Based): Decision-makers consistently employ a lexicographic stopping rule, but the order in which they search through features is probabilistic. To correctly resolve ties in validity, the probability of checking a feature first is determined by a softmax over its ordinal position in the stable-sorted search order, ensuring deterministic tie-breaking while allowing occasional deviations to lower-priority features.

**Rationale:** Initial logic and parameters are validated. Standard processing applied. The final transformation computes the softmax over strictly decreasing integer scores based on the stable-sorted search order rather than the raw validities, correctly resolving ties by index order and restoring human-like tie-breaking behavior.

**Parameters:**
  - `gamma`: `[1.0, 10.0]`
  - `epsilon`: `[0.0, 0.2]`
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
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between the two options
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, guess randomly
        p_core = np.array([0.5, 0.5])
    else:
        order = np.argsort(-validities, kind='mergesort')
        scores = np.zeros(len(validities))
        for i, idx in enumerate(order):
            scores[idx] = len(validities) - i
            
        discrim_scores = scores[discrim]
        logits = gamma * discrim_scores
        
        # Numerically stable softmax
        logits -= np.max(logits)
        w_discrim = np.exp(logits)
        probs_discrim = w_discrim / np.sum(w_discrim)
        
        p_a = 0.0
        p_b = 0.0
        
        discrim_indices = np.where(discrim)[0]
        for idx, p_feat in zip(discrim_indices, probs_discrim):
            if a[idx] > b[idx]:
                p_a += p_feat
            elif b[idx] > a[idx]:
                p_b += p_feat
                
        p_core = np.array([p_a, p_b])
        
    # Incorporate baseline lapse rate (random guessing)
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
