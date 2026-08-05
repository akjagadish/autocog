# Round 2 — Theories

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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Recency-Weighted Tallying: Decision makers evaluate options by computing a weighted tally of feature wins. Due to recency effects in working memory, features presented later (further to the right) are more salient and receive exponentially increasing weights. Because the growth rate of these weights is bounded, the heuristic strictly preserves standard tallying when one option has more feature wins than the other. However, when the number of feature wins is tied, the exponentially increasing weights naturally break the tie in favor of the option that wins on the most recent (right-most) features, providing a continuous and elegant mechanism for tie-breaking without invoking discontinuous rules.

**Rationale:** Following the arbiter's suggestion, this Recency-Weighted Tallying theory replaces the rigid tie-breaker rule of pi_3 with a continuous mechanism. By weighting feature wins with an exponentially increasing weight from left to right (using `recency_base` in [1.01, 1.3]), the model perfectly preserves the primary tallying heuristic when the number of wins is unequal (since the maximum sum of k weights is strictly less than the minimum sum of k+1 weights for these bases). However, when the number of wins is tied, the exponential weights naturally favor the option that wins on the right-most feature (e.g., 1 + rho^3 > rho + rho^2), elegantly explaining the ~0.82 choice probabilities in tie-breaker trials without discontinuous rules.

**Parameters:**
  - `recency_base`: `[1.01, 1.3]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    recency_base = float(parameters['recency_base'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Weights exponentially increasing from left to right to model recency in working memory.
    # For recency_base in [1.01, 1.3], the sum of any k weights is strictly less 
    # than the sum of any k+1 weights, preserving the strict tallying property for unequal wins.
    w = recency_base ** np.arange(n_features)
    
    # Weighted tallying of strict feature-wise wins
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Tallying with Scalable Validity-Based Tie-Breaker: Decision makers primarily evaluate options by tallying the number of features where one option strictly dominates the other, ignoring feature validities. If the feature wins are tied, individuals rely on the feature with the highest objective validity to break the tie. The influence of this tie-breaker is governed by a strategy parameter scaled by the actual validity of the tie-breaking feature, providing a nuanced and continuous score adjustment before response noise is applied.

**Rationale:** Following the critic's advice, I modified the tie-breaking mechanism to scale the `tie_breaker_weight` by the actual validity of the tie-breaking feature (`validities[idx]`). This provides a more continuous and nuanced score adjustment for the softmax function, helping the model capture strong directional preferences rather than defaulting towards 0.50. I retained the parameter ranges from the successful Iteration 2 base.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `tie_breaker_weight`: `[-2.0, 2.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Tie-breaking mechanism: Validity-based fallback (Take-The-Best logic)
    if a_wins == b_wins:
        validities = np.array(parameters['validities'], dtype=float)
        # Sort indices by validity descending
        order = np.argsort(validities)[::-1]
        
        tie_breaker_weight = float(parameters['tie_breaker_weight'])
        
        for idx in order:
            if a[idx] > b[idx]:
                a_wins += tie_breaker_weight * validities[idx]
                break
            elif b[idx] > a[idx]:
                b_wins += tie_breaker_weight * validities[idx]
                break
                
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
