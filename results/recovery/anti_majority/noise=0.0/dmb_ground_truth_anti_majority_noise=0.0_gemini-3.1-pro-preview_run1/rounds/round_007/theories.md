# Round 7 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

**Description:** Variance Aversion (Configural Weighting): Decision-makers evaluate options by averaging the subjective validities of their present features, but they apply a configural penalty proportional to the variance of those validities. This mechanism reflects a preference for 'holistic consistency'—options with tightly clustered, moderately high features are preferred over options with a mix of very high and very low features, as the latter are penalized for their inconsistency.

**Rationale:** Following the arbiter's suggestion, this theory implements a Variance Aversion model. It evaluates options by taking the average subjective validity of their present features and subtracting a penalty proportional to the variance of those validities. This effectively captures subjects' preference for options with consistently moderate-to-high features over those with an extreme mix of high and low validities, explaining the strong deviations from Take-The-Best in Experiments 11 and 12, while also naturally predicting the dilution effect and strict subset preferences seen in Experiment 7.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `lambda_var`: `[0.0, 20.0]`
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
    lambda_var = float(parameters["lambda_var"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        var_w = np.var(active_w) if len(active_w) > 1 else 0.0
        
        # Average evidence penalized by the variance of the active validities
        return mean_w - lambda_var * var_w
        
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


### slot 2 — `pi_9` — SURVIVED ✓

**Description:** Weakest Link (Worst-Case Focus): Decision-makers evaluate an option primarily by averaging the subjective validities of its present features, but they apply a disproportionate penalty based on the option's 'weakest link' (the lowest validity active feature). Specifically, the penalty is proportional to the gap between the option's best feature and its weakest feature. This cognitive mechanism reflects a worst-case focus where attention is drawn to the most inferior attribute, naturally predicting both the dilution effect (adding weak features reduces the mean and increases the weakest-link penalty) and variance aversion (options with tighter feature clusters have smaller weakest-link penalties).

**Rationale:** Following the arbiter's recommendation, this theory implements a 'Weakest Link' mechanism. Instead of computing a global mathematical variance across all active features (which may be cognitively implausible), decision-makers anchor on the average quality of the option but are heavily influenced by the worst-case feature (the lowest validity active feature). The penalty is formalized as the gap between the maximum and minimum active validities. This naturally captures the dilution effect (adding a low-validity feature drastically pulls down the minimum and the mean) and variance aversion (options with high variance have a larger gap between their best and weakest links, leading to a larger penalty), providing a strong mechanistic competitor to holistic variance computation.

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
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        mean_w = np.mean(active_w)
        min_w = np.min(active_w)
        max_w = np.max(active_w)
        
        # Averaging baseline with a penalty based on the weakest link's distance from the best feature
        return mean_w - lambda_weak * (max_w - min_w)
        
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


## Replacement

### `pi_10` → slot 1 (via `new_theory`)

**Description:** Relative Threshold Heuristic with Bounded Regret: Decision-makers evaluate options by directly comparing them on a cue-by-cue basis, focusing only on distinguishing features (unique advantages and disadvantages). To capture the dilution effect without relying on a holistic average, decision-makers apply a threshold (`theta`) to their unique advantages. Low-validity unique advantages that fall below this threshold actually penalize the option. Furthermore, the regret penalty for unique disadvantages is also thresholded (`theta_regret`), ensuring that decision-makers only penalize the absence of truly critical (high-validity) features. This prevents options with multiple medium-validity advantages from being overly punished for missing a single high-validity feature.

**Rationale:** To address the under-prediction of compensatory choices in Experiments 11, 12, 14, and 16, I widened the lower bounds for `gamma` and `delta` to 0.01 to allow for more concave weighting functions. I also introduced a `theta_regret` parameter to threshold the regret penalty. This ensures that only the absence of highly critical features (those with subjective validities above `theta_regret`) incurs a regret penalty, allowing an option with multiple medium-validity advantages to outcompete an option with a single high-validity advantage without being excessively penalized.

**Parameters:**
  - `gamma`: `[0.01, 5.0]`
  - `delta`: `[0.01, 5.0]`
  - `lambda_regret`: `[0.0, 2.0]`
  - `theta`: `[0.0, 1.0]`
  - `theta_regret`: `[0.0, 1.0]`
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
    lambda_regret = float(parameters["lambda_regret"])
    theta = float(parameters["theta"])
    theta_regret = float(parameters["theta_regret"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights for presence and absence
    w_pos = val ** gamma
    w_neg = val ** delta
    
    def get_score(x, other_x):
        unique_adv = (x > 0) & (other_x == 0)
        unique_disadv = (x == 0) & (other_x > 0)
        
        # Sum of unique advantages, penalized by a threshold (theta)
        # This allows low-validity features to have a net negative impact
        adv_score = np.sum(w_pos[unique_adv] - theta) if np.any(unique_adv) else 0.0
        
        # Regret penalty for unique disadvantages, also thresholded
        disadv_score = lambda_regret * np.sum(np.maximum(0.0, w_neg[unique_disadv] - theta_regret)) if np.any(unique_disadv) else 0.0
        
        return adv_score - disadv_score
        
    score_a = get_score(a, b)
    score_b = get_score(b, a)
    
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
