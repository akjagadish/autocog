# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

**Rationale:** Following the critic's feedback, the upper bound of the `beta` parameter range is further reduced from 5.0 to 2.5. This tighter range forces the model to produce softer choice probabilities on the discriminating features, bringing the overpredicted determinism in Experiment 1 and the overpredicted preference for the single most valid feature in Experiment 2 closer to the empirical levels.

**Parameters:**
  - `beta`: `[0.0, 2.5]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Strict Take-The-Best with Uniform Lapse: Decision-makers rely on a strict lexicographic heuristic, ranking cues by subjective validity and making a deterministic choice based solely on the highest-validity discriminating cue. To account for empirical noise and inattention, decisions are subject to a uniform lapse rate, where the decision-maker simply guesses randomly on a fixed proportion of trials rather than scaling their confidence by the cue's validity.

**Rationale:** Following the arbiter's feedback, this theory implements a strict Take-The-Best mechanism augmented only by a uniform random lapse rate. This tests whether the complexity of scaling choice probability by cue validity (as seen in Theory 1) is truly necessary, or if a parameter-efficient deterministic heuristic with simple inattentional guessing is sufficient to capture the empirical variance.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[f] > a[f]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Model (TTB and WADD): Decision-makers exhibit heterogeneous strategy use, combining non-compensatory and compensatory processes. The population consists of a probabilistic mixture of Take-The-Best (TTB) users and Weighted Additive (WADD) users. On any given trial, a choice is a weighted blend of a lexicographic TTB process (which stops at the first discriminating cue and scales confidence by its validity) and a compensatory WADD process (which computes a global utility for each option by summing all feature values weighted by their subjective validities). To account for the fact that compensatory evaluation often introduces more noise when utility differences are small, the WADD component employs a flatter softmax profile, allowing a subset of subjects to be predominantly WADD without making overly deterministic predictions.

**Rationale:** Following the latest feedback, restricting the mixture weight `w_wadd` to a low interval artificially eliminated compensatory subjects, leading to regressions in several experiments. Therefore, the full range of `w_wadd` [0.0, 1.0] has been restored to allow for individual heterogeneity where some subjects can be fully compensatory. To prevent the WADD component from dominating too aggressively and making overly deterministic predictions on conflict trials (which caused the initial overprediction of compensatory behavior), the upper bound of `beta_wadd` has been lowered from 10.0 to 3.0. This flattens the softmax profile of the WADD component, capturing softer/probabilistic choices when utility differences are small.

**Parameters:**
  - `beta_ttb`: `[0.0, 10.0]`
  - `beta_wadd`: `[0.0, 3.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Prediction ---
    order = np.argsort(validities)[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for f in order:
        if a[f] > b[f]:
            ttb_scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            ttb_scores = np.array([0.0, validities[f]])
            break
            
    if ttb_scores[0] == ttb_scores[1]:
        p_ttb = np.array([0.5, 0.5])
    else:
        z_ttb = beta_ttb * (ttb_scores - ttb_scores.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # --- WADD Prediction ---
    u_a = np.sum(validities * a)
    u_b = np.sum(validities * b)
    wadd_scores = np.array([u_a, u_b])
    
    z_wadd = beta_wadd * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Mixture ---
    p_core = (1.0 - w_wadd) * p_ttb + w_wadd * p_wadd
    
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
