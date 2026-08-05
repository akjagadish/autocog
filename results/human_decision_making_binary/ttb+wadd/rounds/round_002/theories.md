# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Probabilistic Heuristic Selection: Subjects do not rigidly apply a single deterministic decision rule across all trials. Instead, they maintain a repertoire of strategies—specifically, a non-compensatory one-reason heuristic (Take The Best) and a compensatory strategy (Tallying/Equal-Weight). On each trial, a subject probabilistically selects which strategy to deploy based on a subject-specific mixing parameter (p_ttb). This trial-by-trial internal blending natively captures intermediate aggregate metric values while preserving low between-subject variance, as each individual exhibits a mixture of both choice patterns.

**Rationale:** Following the arbiter's guidance, this theory implements a Probabilistic Heuristic Selection model. Prior theories assumed subjects strictly followed one strategy (like TTB or WADD), which either over-predicted (1.0) or under-predicted (~0.1-0.2) the observed TTB adherence metrics (~0.43). By explicitly modeling each subject as probabilistically shifting between TTB and Tallying on a trial-by-trial basis, the model predicts an intermediate choice proportion matching the empirical ~0.43. Sampling the mixing parameter `p_ttb` across a broad range `[0.0, 0.9]` naturally replicates the low between-subject variance (~0.06-0.07) because each subject's individual metric centers around their personal mixture probability.

**Parameters:**
  - `p_ttb`: `[0.0, 0.9]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (Non-compensatory)
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner is None:
        p_ttb_arr = np.array([0.5, 0.5])
    else:
        p_ttb_arr = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Strategy 2: Tallying (Compensatory)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_arr + (1.0 - p_ttb) * p_tally_arr
    
    # Late-stage lapse noise
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

**Rationale:** Following the arbiter's suggestion, this theory unifies compensatory and non-compensatory decision-making without relying on a discrete mixture of heuristics. By applying an individual-specific exponent (gamma) to the cue validities, the model can natively capture Tallying (gamma ~ 0), WADD (gamma ~ 1), and Take The Best (gamma >> 1) behaviors within a single, continuous weighted-additive framework. This provides a more robust and elegant explanation for the intermediate TTB matches and varying strategy adoptions observed in the experiments.

**Parameters:**
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Rank-Based Weighting Model with Exponential Decay (Average Ties): Subjects evaluate options by computing a weighted sum of their features. They rank features based on validities and assign subjective weights using an exponential decay function. Ties in validities are assigned their average rank, providing a smoother spacing of decay weights. A decay parameter lambda smoothly interpolates between Tallying (lambda=1) and Take The Best (lambda=0).

**Rationale:** Reverting to the last accepted base (exponential rank decay) because linear decay degraded the overall fit. Modifying the rankdata method from 'dense' to 'average' as suggested by the critic. This provides a smoother and more statistically standard spacing for the exponential decay function when handling tied validities, without being as extreme as the rejected 'min' method.

**Parameters:**
  - `lambda_param`: `[0.0, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    lambda_param = float(parameters["lambda_param"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity (highest validity = rank 1)
    # Using 'average' method to handle ties smoothly
    ranks = rankdata(-validities, method='average')
    
    # Exponential rank decay: w = lambda_param ^ (rank - 1)
    w = lambda_param ** (ranks - 1.0)
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
