# Round 16 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Rationale:** Following the critic's advice, I reverted to the accepted Iteration 2 base (Evidence Dilution) and widened the upper bounds of the parameters, specifically increasing the upper bound of `gamma` to 10.0 and `lambda_val` to 20.0. This gives the optimizer the flexibility to apply a much stronger dilution penalty to options with many weakly-valid cues, which is necessary to capture the severe negative preference observed in Experiment 9, without fundamentally altering the mathematical formulation that successfully preserved core TTB/Tallying performance in other experiments.

**Parameters:**
  - `lambda_val`: `[1.0, 20.0]`
  - `gamma`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_18` — KILLED ✗

**Description:** Rank-Dependent Averaging Theory

**Rationale:** Following the critic's advice, we restrict the upper bound of the inverse temperature parameter `beta` to [0.1, 10.0] (down from 25.0). Previous attempts to sharpen predictions led to regressions, implying the model was making overly extreme or deterministic predictions that hurt aggregate loss across experiments. Lowering the maximum beta acts as regularization, preventing extreme choice probabilities while maintaining the core rank-dependent averaging mechanism.

**Parameters:**
  - `lambda_val`: `[0.1, 10.0]`
  - `gamma`: `[0.6, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Extract validities of active cues for each option
    val_a = val[a == 1]
    val_b = val[b == 1]
    
    # Sort in descending order to rank the active cues
    val_a = np.sort(val_a)[::-1]
    val_b = np.sort(val_b)[::-1]
    
    def get_value(v_array):
        if len(v_array) == 0:
            return 0.0
        # Non-linear scaling of validities to allow for heightened sensitivity to strong cues
        v_scaled = v_array ** lambda_val
        # Rank-dependent weights: gamma^(rank-1)
        ranks = np.arange(len(v_array))
        weights = gamma ** ranks
        # Weighted average
        return np.sum(weights * v_scaled) / np.sum(weights)
        
    v_a = get_value(val_a)
    v_b = get_value(val_b)
    
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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

### `pi_19` → slot 2 (via `new_theory`)

**Description:** Lexicographic Thresholding with Additive Penalties Theory: Decision-makers first attempt a lexicographic (Take-The-Best) evaluation. If the top cues differ by more than a threshold, the decision is based solely on them. Otherwise, they fall back to an additive tallying mechanism where cues with a validity above a threshold 'theta' add their scaled validity to the option's value, while low-validity cues (below 'theta') subtract a penalty 'gamma'. This heavily penalizes options bloated with weak, uninformative features without discounting strong secondary cues.

**Rationale:** Following the critic's advice, we introduce a validity threshold `theta`. During the fallback tallying phase, cues with validity `>= theta` add positive evidence `(val ** lambda_val)`, whereas cues with validity `< theta` are treated as uninformative 'bloat' and subtract a fixed penalty `gamma`. This allows the model to heavily penalize options burdened with many weak cues (crucial for capturing the extreme negative values in Experiments 9, 22, and 34) without indiscriminately penalizing strong secondary cues that are still useful for breaking ties.

**Parameters:**
  - `tau`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 5.0]`
  - `lambda_val`: `[0.1, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    tau = float(parameters["tau"])
    gamma = float(parameters["gamma"])
    lambda_val = float(parameters["lambda_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    
    top_a = np.max(val[a == 1]) if np.sum(a) > 0 else 0.0
    top_b = np.max(val[b == 1]) if np.sum(b) > 0 else 0.0
    
    if abs(top_a - top_b) > tau:
        v_a = top_a
        v_b = top_b
    else:
        # Additive tallying with penalty for low validity
        w = np.where(val >= theta, val ** lambda_val, -gamma)
        v_a = np.sum(w * a)
        v_b = np.sum(w * b)
        
    scores = np.array([v_a, v_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
