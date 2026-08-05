# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** High-Lapse Take-The-Best (TTB) Theory: Subjects generally find the multi-attribute choice task tedious without trial-by-trial feedback, leading to a very high rate of random guessing (high lapse rate). However, on the small fraction of trials where they actively engage, they do not integrate all features or tally them. Instead, they use a fast-and-frugal lexicographic heuristic (Take-The-Best): they inspect cues in descending order of their validity and make a deterministic choice based entirely on the single most valid cue that discriminates between the two options.

**Rationale:** Following the arbiter's feedback, this model instantiates a 'High-Lapse Take-The-Best' mechanism. It acknowledges the high noise floor in the experimental data by retaining a large lapse rate (epsilon > 0.6), but replaces the compensatory tallying mechanism of Theory 2 with a purely non-compensatory, lexicographic one. When subjects engage, they simply identify the most valid cue that discriminates the options and choose accordingly, completely ignoring all other cues. This directly tests whether the weak behavioral signal present in the data is driven by a fast-and-frugal lexicographic process rather than unweighted feature summation.

**Parameters:**
  - `epsilon`: `[0.6, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(val)[::-1]
    
    ttb_choice = -1
    for idx in order:
        if a[idx] > b[idx]:
            ttb_choice = 0
            break
        elif b[idx] > a[idx]:
            ttb_choice = 1
            break
            
    if ttb_choice == 0:
        p_core = np.array([1.0, 0.0])
    elif ttb_choice == 1:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    
    # Mix the deterministic TTB choice with a high lapse rate (random guessing)
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


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** High-Lapse Tallying Theory: Subjects find the multi-attribute binary feature task cognitively demanding, leading to a very high rate of random guessing. However, when they do engage with the stimulus, they rely on a simple unweighted tally of positive features (Tallying) rather than integrating complex cue validities. This produces a very weak but non-zero behavioral signal that slightly favors options with a higher simple count of positive features.

**Rationale:** The arbiter noted that subjects exhibit a very weak but non-zero signal that aligns more with unweighted feature counts (Tallying) than with validity-weighted sums. The previous Random Guessing model captured the overall lack of signal but missed the subtle structure. By introducing a Noisy Tallying model with a high lapse rate (epsilon between 0.7 and 1.0), we can explain both the predominantly random behavior and the slight preference for options with more positive features observed in the data.

**Parameters:**
  - `beta`: `[0.1, 5.0]`
  - `epsilon`: `[0.7, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** High-Lapse Weighted Additive (WADD) Theory: Subjects find the multi-attribute binary feature task cognitively demanding without trial-by-trial feedback, leading to a high rate of random guessing. However, on the rare trials where they do actively engage, they do not rely on simple heuristics like Tallying or Take-The-Best. Instead, they integrate all available features by weighting them according to their given validities, producing a compensatory decision process that can account for situations where a few high-validity cues outweigh a larger number of low-validity cues.

**Rationale:** Following the arbiter's feedback, this model replaces the non-compensatory Take-The-Best (TTB) and unweighted Tallying mechanisms with a fully compensatory Weighted Additive (WADD) model, while retaining the high lapse rate. This captures the baseline success of Tallying (since WADD often correlates with Tallying) but better explains the subtle variance in trials where high-validity cues override pure tally counts, which TTB's strict lexicographic rule and Tallying's unweighted rule fail to capture.

**Parameters:**
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.7, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted additive sum of features
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over WADD utilities for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
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
