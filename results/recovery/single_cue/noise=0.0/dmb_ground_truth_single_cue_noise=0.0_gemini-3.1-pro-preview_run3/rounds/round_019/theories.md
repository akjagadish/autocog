# Round 19 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_17` — KILLED ✗

**Description:** Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

**Rationale:** Following the critic's feedback, the previous threshold range of [0.0, 2.0] allowed the optimizer to select values greater than 1.0, which inappropriately caused the model to use TTB even when the tally difference was decisive (e.g., delta_tally == 1). By narrowing the threshold range to [0.1, 0.9], the model is forced into a regime where a tied tally (delta_tally == 0) always results in a positive exponent (favoring TTB), while any decisive tally (delta_tally >= 1) results in a negative exponent (favoring Tallying). This perfectly mirrors the successful Tally-then-TTB logic while remaining within the smooth Strategy Selection family.

**Parameters:**
  - `theta`: `[1.0, 20.0]`
  - `threshold`: `[0.1, 0.9]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
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


### slot 2 — `pi_21` — SURVIVED ✓

**Description:** Tally-plus-Configural WADD Hybrid with Symmetrical Clustering Penalty (Active Suppression): Decision-makers primarily rely on a simple tally of positive features. When tallies are tied or close, they break the tie using a weighted additive process that features configural cue processing. Instead of just discounting clustered features, the model symmetrically penalizes ALL features in a cluster, allowing the penalty factor to be negative. This means clustered cues can be actively detrimental, heavily suppressing options with adjacent top cues (e.g., 11000) and strongly favoring options with spaced-out cues (e.g., 10001), perfectly capturing the extreme preference reversals observed in human data.

**Rationale:** Following the critic's advice, we expand the parameter range of `lambda_adj` from [0.0, 1.0] to [-1.0, 1.0]. This allows the symmetrical clustering penalty to not just zero out clustered features, but to actively suppress them (i.e., make them detrimental). This stronger penalty is necessary to reach the extreme magnitudes of the spacing effects observed in Experiments 37 and 38, where 10001 massively dominates 01100, and 11000 is strongly dispreferred against 00011.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `theta`: `[0.1, 3.0]`
  - `lambda_adj`: `[-1.0, 1.0]`
  - `w_tally`: `[0.0, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    lambda_adj = float(parameters["lambda_adj"])
    w_tally = float(parameters["w_tally"])
    
    # Non-linear weighting of validities
    w = val ** theta
    w = w / np.sum(w)
    
    def calc_value(x):
        val_x = 0.0
        n = len(x)
        for i in range(n):
            if x[i] == 1:
                # Symmetrical adjacency penalty: penalize if adjacent to ANY other 1
                is_clustered = False
                if i > 0 and x[i-1] == 1:
                    is_clustered = True
                if i < n - 1 and x[i+1] == 1:
                    is_clustered = True
                    
                if is_clustered:
                    val_x += w[i] * lambda_adj
                else:
                    val_x += w[i]
        return val_x
        
    # Score is a hybrid of Tallying and Configural WADD
    score_a = w_tally * np.sum(a) + calc_value(a)
    score_b = w_tally * np.sum(b) + calc_value(b)
    
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

**Description:** Pattern-Based Tallying: Decision-makers evaluate options by tallying individual positive features alongside distinct perceptual patterns—specifically, isolated positive features. By treating 'isolated features' as additional positive signals in the tally, the model naturally captures the strong empirical preference for spaced over clustered cues without resorting to complex configural WADD calculations. A weighted additive (WADD) component is integrated as a secondary signal to reflect validity-based tie-breaking.

**Rationale:** Following the feedback, we revert back to the successful Pattern-Based Tallying mechanism from Iteration 2 (which counts isolated positive features) rather than penalizing clustered features, which caused severe regressions. We adjust the `w_pattern` range to a "Goldilocks" interval of `[0.0, 2.5]` to balance the spacing effect without underfitting or overfitting. `w_wadd` and `theta` are kept strictly at their previous ranges to protect the delicate WADD vs Tallying balance in Experiment 3.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_pattern`: `[0.0, 2.5]`
  - `w_wadd`: `[0.0, 2.0]`
  - `theta`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_pattern = float(parameters["w_pattern"])
    w_wadd = float(parameters["w_wadd"])
    theta = float(parameters["theta"])
    
    def count_isolated(x):
        iso = 0
        n = len(x)
        for i in range(n):
            if x[i] == 1:
                left_zero = (i == 0 or x[i-1] == 0)
                right_zero = (i == n - 1 or x[i+1] == 0)
                if left_zero and right_zero:
                    iso += 1
        return iso
        
    # Non-linear weighting of validities for the WADD component
    w = val ** theta
    w = w / np.sum(w)
    
    # Score is a combination of a simple tally, the isolated pattern tally, and WADD
    score_a = np.sum(a) + w_pattern * count_isolated(a) + w_wadd * np.sum(w * a)
    score_b = np.sum(b) + w_pattern * count_isolated(b) + w_wadd * np.sum(w * b)
    
    scores = np.array([score_a, score_b])
    
    # Numerically stable softmax
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
