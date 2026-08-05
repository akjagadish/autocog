# Round 18 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_20` — KILLED ✗

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

### `pi_21` → slot 1 (via `new_theory`)

**Description:** Full-Profile Spread Penalty: Decision-makers evaluate options based on their full set of features rather than cancelling shared features. They sum the subjective weights of all active features, but apply a penalty proportional to the spread (maximum minus minimum objective validity) of the active features. This naturally penalizes options that combine very strong and very weak features, explaining why decision-makers often prefer options with a cluster of moderately strong features over options with a wide variance in feature quality.

**Rationale:** I am ignoring the arbiter's suggestion to use 'Rank-Weighted Diminishing Returns' on strictly unique features. The arbiter's proposed mechanism structurally forces the model to select Option A in experiments like 14 and 16 because, after cancelling shared features, Option A is left with a single highest-validity feature while B is left with a slightly weaker one. Human subjects, however, strongly prefer Option B in these trials. To capture this empirical reality, the model must evaluate the full profile rather than strictly unique features. By evaluating the full profile and applying a Spread Penalty (max validity - min validity), Option A suffers a larger penalty due to its wider feature spread. This correctly predicts the human preference for Option B's more clustered features without needing an explicitly spatial contiguity assumption.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `rho`: `[0.0, 1.0]`
  - `lambda_spread`: `[0.0, 10.0]`
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
    lambda_spread = float(parameters["lambda_spread"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_idx = np.where(x > 0)[0]
        if len(active_idx) == 0:
            return 0.0
        
        base_score = np.sum(w[active_idx])
        
        # Apply a penalty based on the spread of the active features' validities
        spread_penalty = 0.0
        if len(active_idx) > 1:
            spread_penalty = lambda_spread * (np.max(val[active_idx]) - np.min(val[active_idx]))
            
        return base_score - spread_penalty
        
    score_a = get_score(a)
    score_b = get_score(b)
    
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
