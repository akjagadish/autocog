# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Tallying with Recency/Right-Most Tie-Breaker: Decision makers primarily evaluate options by tallying the number of features where one option strictly dominates the other, ignoring feature validities (equal weighting). If one option has more winning features, it is chosen. However, if the feature wins are tied, individuals do not simply guess. Instead, they rely on a 'recency' or 'right-most' bias, breaking the tie based solely on the last feature in the array. Responses are subject to softmax noise over the resulting scores and a uniform lapse rate.

**Rationale:** The arbiter noted that subjects in Experiment 1 systematically favored the option that won on the last feature when total feature wins were tied, leading to a sub-0.5 match rate with TTB on those trials. This new theory preserves the excellent fit of Tallying on Experiment 2 (where Tallying makes strict predictions) but replaces the random guessing on tied trials with a deterministic tie-breaker based on the right-most feature. By doing so, it captures the 'recency bias' observed in Experiment 1 without disrupting the primary tallying mechanism.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Tie-breaking mechanism: Recency/Right-most feature bias
    if a_wins == b_wins:
        if a[-1] > b[-1]:
            a_wins += 1.0
        elif b[-1] > a[-1]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Positional Weighted Additive Decision Making

**Rationale:** Following the critic's advice, we adjust the `position_base` parameter range to [1.0, 2.0]. The previous accepted base's range of [1.0, 4.0] allowed right-most features to completely dominate and overpredict Experiments 5 and 6, while the strictly restricted [1.0, 1.4] from Iteration 3 failed to sufficiently capture the right-most bias in Experiments 4, 7, and 8. The [1.0, 2.0] range serves as a middle ground, allowing the model to apply a meaningful right-most bias without completely overriding the primary tallying strategy.

**Parameters:**
  - `position_base`: `[1.0, 2.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    position_base = float(parameters['position_base'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Weights increase exponentially from left to right based on spatial position
    w = position_base ** np.arange(n_features)
    
    # Positional weighted tallying of strict feature-wise wins
    a_wins = float(np.sum(w * (a > b)))
    b_wins = float(np.sum(w * (b > a)))
    
    scores = np.array([a_wins, b_wins])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Validity-Weighted Additive Decision Making with Non-linear Validity Transformation on Strict Wins: Decision makers compute a weighted sum of strict feature wins, but their interpretation of the explicitly provided validities varies. Some treat them as direct probabilities (standard weighting), while others may misinterpret them as error rates or ranks (inverse weighting). This is modeled by exponentiating the validities with a flexible parameter (gamma). Choices are then made probabilistically based on the resulting scores of strict feature wins, subject to a strictly positive softmax temperature to preserve choice determinism, and a baseline lapse rate.

**Rationale:** Following the critic's advice, we revert the rejected normalization step and instead modify the score computation. We change it from validity-weighted raw feature values to validity-weighted strict feature wins (`np.sum(weights * (a > b))`). This restores the sharper, heuristic-like comparisons that humans typically use, while keeping the `gamma` and `beta` parameters to handle directional inversions and choice determinism.

**Parameters:**
  - `validities`: `validities`
  - `gamma`: `[-5.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    gamma = float(parameters['gamma'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Exponentiate validities to allow inverse weighting (gamma < 0), flat (gamma = 0), or standard (gamma > 0)
    weights = validities ** gamma
    
    # Validity-weighted sum of strict feature-wise wins
    a_val = float(np.sum(weights * (a > b)))
    b_val = float(np.sum(weights * (b > a)))
    
    scores = np.array([a_val, b_val])
    
    # Softmax choice with numerical stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
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
