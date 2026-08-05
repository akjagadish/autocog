# Round 17 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_19` — KILLED ✗

**Description:** Feature Coherence and Anchoring (Unique Features Only): Decision-makers simplify choices by first cancelling out shared features, then evaluating the remaining unique features. They expect high-quality options to be coherent. A gap penalty is applied if an option's best unique feature (the anchor) is disconnected from its next best unique feature. However, to avoid unfairly penalizing naturally sparse fallback options, this gap penalty is only applied if the option has a dense profile (>2 unique features) OR if it boasts the absolute highest-validity feature in the environment (index 0) but fails to back it up. A spread penalty is also applied to unique features to penalize internal conflict.

**Rationale:** Applying the critic's exact guidance: we compute the gap penalty strictly using `unique_idx` so that shared features don't interfere with the coherence evaluation, which preserves the cancellation effect (fixing Exp 20). We also updated the threshold condition to `len(unique_idx) > 2 or (len(unique_idx) == 2 and unique_idx[0] == 0)`. This elegantly ensures that sparse fallback options (like Option B in Exp 16) are exempt from the gap penalty, while sparse top-tier options that claim the best feature but fail to back it up (like Option A in Exp 13) are correctly penalized.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `rho`: `[0.0, 1.0]`
  - `lambda_iso`: `[0.0, 5.0]`
  - `lambda_penalty`: `[0.0, 10.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_iso = float(parameters["lambda_iso"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities
    w = (val ** gamma) - rho
    
    # Identify shared and unique features
    shared = (a > 0) & (b > 0)
    unique_a = np.where((a > 0) & ~shared)[0]
    unique_b = np.where((b > 0) & ~shared)[0]
    
    def get_score(unique_idx):
        if len(unique_idx) == 0:
            return 0.0
            
        base_score = np.sum(w[unique_idx])
        
        # Isolation / Gap penalty for the anchor calculated strictly on unique features.
        # Applied if the profile is dense (>2 unique features) OR 
        # if it's a 2-feature profile that claims the absolute best feature (index 0).
        gap_penalty = 0.0
        if len(unique_idx) > 2 or (len(unique_idx) == 2 and unique_idx[0] == 0):
            anchor = unique_idx[0]
            next_best = unique_idx[1]
            gap = next_best - anchor - 1
            if gap > 0:
                gap_penalty = lambda_iso * gap
                
        # Spread penalty on unique features
        conflict = 0.0
        if len(unique_idx) > 1:
            conflict = lambda_penalty * (np.max(w[unique_idx]) - np.min(w[unique_idx]))
            
        return base_score - gap_penalty - conflict
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_14` — SURVIVED ✓

**Description:** Thresholded Unique Features with Spread Penalty: Decision-makers simplify choices by cancelling out shared features, then evaluate the unique features relative to a subjective validity threshold. Features above the threshold provide positive evidence, while those below act as penalties. These values are integrated additively, but options with multiple unique features suffer a conflict penalty proportional to the spread (max - min) of their thresholded validities. This penalizes options with a wide variance in their unique features while strictly preserving shared-feature cancellation.

**Rationale:** Following the critic's advice on the iter 1 base, we redefine the 'conflict' penalty. Instead of using the raw minimum of the thresholded weights (which could be positive and thus act as a boost), we define conflict as the spread (max - min) of the thresholded evidence among the unique active features. We subtract this spread penalty from the base additive score only when there are multiple unique features. This strictly penalizes options with a wide variance in their unique features, ensuring it always acts as a penalty, while perfectly preserving the shared-feature cancellation required to fit Experiments 17, 19, and 20.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `rho`: `[0.0, 1.0]`
  - `lambda_penalty`: `[0.0, 10.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_penalty = float(parameters["lambda_penalty"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities and apply subjective threshold
    v_trans = val ** gamma
    w = v_trans - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # Additive integration of thresholded evidence
        base_score = np.sum(active_w)
        
        # Spread penalty applied if there are multiple unique features
        if len(active_w) > 1:
            conflict_penalty = lambda_penalty * (np.max(active_w) - np.min(active_w))
            return base_score - conflict_penalty
            
        return base_score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

### `pi_20` → slot 1 (via `new_theory`)

**Description:** Feature Contiguity and Density-weighted Integration: Decision-makers simplify choices by cancelling out shared features and evaluating the remaining unique features. Rather than penalizing global variance or arbitrary gaps, they perceive contiguous active features as mutually reinforcing (synergy) and isolated features as less reliable (discounted). Thus, options with clustered positive features receive a subjective bonus, while isolated features are penalized, naturally favoring dense feature profiles.

**Rationale:** Following the arbiter's suggestion, this theory implements a 'Feature Contiguity' mechanism. It discards the brittle index-based gap penalty and global spread penalties that failed in previous iterations. Instead, it integrates features additively after cancelling shared features, but applies a synergy bonus for adjacent positive features and an isolation penalty for features with no active neighbors. This naturally captures the strong preference for clustered profiles seen in Experiment 16 without disrupting the basic compensatory and non-compensatory dynamics in other experiments, as both bonuses and penalties scale smoothly with feature validities.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `rho`: `[0.0, 1.0]`
  - `lambda_iso`: `[0.0, 5.0]`
  - `lambda_syn`: `[0.0, 5.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    rho = float(parameters["rho"])
    lambda_iso = float(parameters["lambda_iso"])
    lambda_syn = float(parameters["lambda_syn"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Base subjective evidence
    w = (val ** gamma) - rho
    
    def get_score(unique_x):
        active_idx = np.where(unique_x > 0)[0]
        if len(active_idx) == 0:
            return 0.0
            
        score = 0.0
        for i in active_idx:
            score += w[i]
            
            # Check isolation: no adjacent active features
            is_isolated = True
            if (i - 1) in active_idx or (i + 1) in active_idx:
                is_isolated = False
                
            if is_isolated:
                # Penalize isolated features proportional to their validity
                score -= lambda_iso * val[i]
                
        # Calculate synergy for adjacent pairs
        for i in range(len(unique_x) - 1):
            if unique_x[i] > 0 and unique_x[i+1] > 0:
                # Reward adjacent pairs proportional to their combined validity
                score += lambda_syn * (val[i] + val[i+1])
                
        return score
        
    score_a = get_score(unique_a)
    score_b = get_score(unique_b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
