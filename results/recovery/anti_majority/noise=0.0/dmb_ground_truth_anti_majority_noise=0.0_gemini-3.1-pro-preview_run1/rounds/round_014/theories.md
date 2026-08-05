# Round 14 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_16` — KILLED ✗

**Description:** Variance Aversion and Feature Consistency

**Rationale:** Following the critic's feedback, the variance penalty was modified to use standard deviation (`np.std`) scaled by the number of active features (`len(active_w)`). This ensures the penalty competes symmetrically with the additive base score (which is essentially mean * len). The range of `lambda_var` was also expanded to `[0.0, 50.0]` to ensure the penalty can be strong enough to overcome the base utility advantage of a single high-validity feature, directly addressing the failure in Experiment 14 while preserving the overall mechanism.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `rho`: `[0.0, 1.0]`
  - `lambda_var`: `[0.0, 50.0]`
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
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Cancel out shared features
    shared = (a > 0) & (b > 0)
    unique_a = a.copy()
    unique_a[shared] = 0
    unique_b = b.copy()
    unique_b[shared] = 0
    
    # Subjective utility: validities transformed and shifted by a threshold
    w = (val ** gamma) - rho
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        base_score = np.sum(active_w)
        
        # Apply variance penalty for multiple unique features
        if len(active_w) > 1:
            # Use standard deviation scaled by the number of active features
            # so the penalty competes symmetrically with the additive base score.
            std_w = np.std(active_w)
            return base_score - lambda_var * std_w * len(active_w)
            
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

### `pi_17` → slot 1 (via `new_theory`)

**Description:** Diminishing Marginal Utility with Baseline Cost, Independent Discounting, and Loss Aversion: Decision-makers evaluate all features of an option. Each feature's validity is transformed and reduced by a baseline cost 'C'. Features above 'C' provide positive utility, while those below act as negative utility (penalties). To reflect limited attention, positive and negative features are sorted independently by their magnitude and discounted exponentially based on their rank. Furthermore, a loss aversion multiplier amplifies the impact of negative features, explaining why options with multiple weak features are strongly penalized and consistently lose to strict subsets of strong features.

**Rationale:** Following the critic's advice, we introduce a 'weight_neg' multiplier to asymmetrically amplify the negative utility of features that fall below the baseline cost 'C'. By multiplying the sum of the discounted negative features by this parameter, the model can capture the strong empirical aversion to options with multiple weak features, pushing the choice probabilities closer to the extreme values seen in experiments like 11, 14, 16, 21, 28, and 30.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `delta`: `[0.0, 1.0]`
  - `c`: `[0.0, 1.0]`
  - `weight_neg`: `[0.1, 10.0]`
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
    delta = float(parameters["delta"])
    c = float(parameters["c"])
    weight_neg = float(parameters["weight_neg"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # DO NOT cancel shared features; evaluate the full feature set
    
    # Transform validities and subtract baseline cost C
    v_trans = (val ** gamma) - c
    
    def get_score(x):
        active_v = v_trans[x > 0]
        if len(active_v) == 0:
            return 0.0
        
        # Separate positive and negative evidence
        pos_v = active_v[active_v > 0]
        neg_v = active_v[active_v < 0]
        
        score = 0.0
        
        if len(pos_v) > 0:
            # Sort positive features descending by magnitude
            pos_v_sorted = np.sort(pos_v)[::-1]
            ranks = np.arange(len(pos_v_sorted))
            score += np.sum(pos_v_sorted * (delta ** ranks))
            
        if len(neg_v) > 0:
            # Sort negative features ascending (most negative first)
            neg_v_sorted = np.sort(neg_v)
            ranks = np.arange(len(neg_v_sorted))
            score += weight_neg * np.sum(neg_v_sorted * (delta ** ranks))
            
        return score
        
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
