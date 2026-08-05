# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Averaging Model: Decision-makers evaluate options not by summing the evidence (additive) or relying on a single best cue (lexicographic), but by averaging the subjective validities of all positive features present in an option. This mechanism naturally accounts for the 'dilution effect', where adding a positive but low-validity feature to an option can actually decrease its overall subjective value and choice probability.

**Rationale:** Following the latest feedback, we constrain the gamma parameter strictly to values >= 1.0 (range [1.0, 10.0]). This ensures the model continues to respect cue validities and avoids collapsing into purely random behaviour where all features are weighted equally. Concurrently, beta is bounded to a moderate range [0.1, 15.0] to prevent overly deterministic predictions that artificially inflate the TTB-consistency metrics in Exps 1 and 3. The core averaging mechanism remains untouched.

**Parameters:**
  - `gamma`: `[1.0, 10.0]`
  - `beta`: `[0.1, 15.0]`
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
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities to subjective weights
    w = val ** gamma
    
    # Count positive features
    k_a = np.sum(a)
    k_b = np.sum(b)
    
    # Calculate average validity of positive features (0 if no positive features)
    score_a = np.sum(w * a) / k_a if k_a > 0 else 0.0
    score_b = np.sum(w * b) / k_b if k_b > 0 else 0.0
    
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


### slot 2 — `pi_7` — SURVIVED ✓

**Description:** Anchor and Adjust: Decision-makers evaluate options by first identifying the highest-validity positive feature, which serves as an anchor. Subsequent positive features adjust this anchored evaluation downwards if their validities are lower than the anchor. This explains the 'less is more' dilution effect, as adding low-validity features drags the overall evaluation down, while naturally predicting that an option with only high-validity features will be preferred over one that adds additional low-validity features (strict subset preference).

**Rationale:** Following the critic's feedback, the Anchor and Adjust model successfully captures the subset preferences (Experiment 7) but under-predicts the strength of the dilution effect in Experiments 9 and 10. To allow the model to more strongly weight the adjusting (lower validity) features and achieve heavier dilution, I have expanded the upper bound of the `alpha` parameter from 5.0 to 50.0. I also expanded the `beta` range to 50.0 to allow for sharper choices when scores are heavily diluted.

**Parameters:**
  - `gamma`: `[0.1, 10.0]`
  - `alpha`: `[0.0, 50.0]`
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
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities into subjective weights
    w = val ** gamma
    
    def get_score(x):
        active_w = w[x > 0]
        if len(active_w) == 0:
            return 0.0
        
        # The highest validity positive feature is the anchor
        anchor = np.max(active_w)
        idx = np.argmax(active_w)
        
        # Remaining features adjust the evaluation
        other_w = np.delete(active_w, idx)
        if len(other_w) == 0:
            return anchor
            
        # Weighted average of anchor and adjusting features
        return (anchor + alpha * np.sum(other_w)) / (1.0 + alpha * len(other_w))
        
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

### `pi_8` → slot 1 (via `new_theory`)

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
