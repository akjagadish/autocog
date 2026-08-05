# Round 19 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_21` — KILLED ✗

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

### `pi_22` → slot 1 (via `new_theory`)

**Description:** Tallying of Valid Unique Features with TTB Tie-Breaker: Decision-makers simplify choices by cancelling out shared features. For the remaining unique features, they tally the number of features that exceed a subjective validity threshold. To resolve ties or capture strong non-compensatory preferences, they also consider the maximum validity among the unique features (a Take-The-Best heuristic component). These two bounded rationality heuristics are integrated to form the final preference.

**Rationale:** Following the critic's advice, we retain the base Tallying mechanism from iteration 1 (which successfully improved the loss) but replace the simple sum-of-validities tie-breaker with a Take-The-Best (TTB) tie-breaker, weighted by a new parameter `lambda_ttb`. This allows the model to flexibly capture both the bounded rationality of counting valid unique features and the strong non-compensatory bias observed in many subjects (e.g., Experiments 1, 3, 9) without discarding the benefits of the thresholded tallying approach.

**Parameters:**
  - `theta`: `[0.5, 1.0]`
  - `lambda_tally`: `[0.0, 10.0]`
  - `lambda_ttb`: `[0.0, 10.0]`
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
    
    theta = float(parameters["theta"])
    lambda_tally = float(parameters["lambda_tally"])
    lambda_ttb = float(parameters["lambda_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = np.where((a > 0) & ~shared)[0]
    unique_b = np.where((b > 0) & ~shared)[0]
    
    def get_score(unique_idx):
        if len(unique_idx) == 0:
            return 0.0
        
        # Tally of unique features that exceed the subjective validity threshold
        valid_tally = np.sum(val[unique_idx] >= theta)
        
        # TTB tie-breaker: maximum validity among unique features
        ttb_score = np.max(val[unique_idx])
        
        return lambda_tally * valid_tally + lambda_ttb * ttb_score
        
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
