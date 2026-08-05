# Round 11 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_11` — SURVIVED ✓

**Description:** Unique Features Weakest-Link: Decision-makers simplify choices by first cancelling out features shared by both options. They then evaluate each option based solely on its unique features, computing the average subjective validity of these features but applying a disproportionate penalty based on the option's 'weakest link' (the gap between its best and worst unique features).

**Rationale:** The theory posits that decision-makers first simplify choices by cancelling out any features shared by both options, a process known as cue-cancellation. After stripping away these commonalities, they evaluate the remaining unique features of each option. This evaluation is based on the average subjective validity of the unique features, but is heavily penalized by the 'weakest link' (the lowest validity unique feature) relative to the best unique feature. This theory elegantly explains why adding shared features has no effect on choice probabilities (solving Experiment 17, where the prior Weakest Link model failed catastrophically) while naturally capturing the variance-aversion and dilution effects observed in other experiments through the weakest-link penalty applied to unique features.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `lambda_weak`: `[0.0, 10.0]`
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
    lambda_weak = float(parameters["lambda_weak"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    def get_score(unique_x):
        active_w = w[unique_x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
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


### slot 2 — `pi_13` — KILLED ✗

**Description:** Unique Features Diminishing Utility with Reference Point: Decision-makers simplify choices by cancelling out shared features, then evaluate options based solely on their unique features. Features are evaluated relative to a subjective reference point (rho), where validities below this point carry negative utility. The total additive utility of these unique features is then subjected to an exponent-based scaling (alpha), which naturally provides diminishing returns for both positive and negative accumulations of features. This captures the 'less is more' effect without mathematically dampening the penalty for negative sums.

**Rationale:** Following the critic's advice on Iteration 4, the upper bound of the lapse rate `epsilon` is restricted from 0.5 to 0.1. The previous parameterization allowed epsilon to absorb too much variance, which artificially flattened the predicted choice probabilities and prevented the model from capturing the extreme ~80-85% empirical preferences seen in Experiments 7, 11, 16, and 21. By tightening the epsilon bounds, the optimizer is forced to rely on the core deterministic mechanism to explain the data variance.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `alpha`: `[0.1, 2.0]`
  - `rho`: `[0.5, 1.0]`
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.1]`
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
    alpha = float(parameters["alpha"])
    rho = float(parameters["rho"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Transform validities into subjective weights relative to a reference point
    # This allows low-validity features to contribute negative utility
    w = np.sign(val - rho) * (np.abs(val - rho) ** gamma)
    
    sum_a = np.sum(w[unique_a > 0]) if np.any(unique_a > 0) else 0.0
    sum_b = np.sum(w[unique_b > 0]) if np.any(unique_b > 0) else 0.0
    
    # Diminishing returns on the total sum of unique features
    score_a = np.sign(sum_a) * (np.abs(sum_a) ** alpha)
    score_b = np.sign(sum_b) * (np.abs(sum_b) ** alpha)
    
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

### `pi_14` → slot 2 (via `new_theory`)

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
