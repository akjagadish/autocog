# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

**Rationale:** The Take The Best (TTB) heuristic is a non-compensatory, lexicographic decision strategy. We maintain the exact same mechanism as the previous candidate, but adjust the parameter ranges for `beta` (shifted to lower values: [0.01, 5.0]) and `epsilon` (upper bound increased to 1.0). This allows the optimization process to inject enough stochasticity to bridge the gap between pure TTB's extreme predictions (~0.15) and human behavior (~0.30-0.36).

**Parameters:**
  - `beta`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Strategy Selection Mixture Model: Subjects possess a repertoire of decision strategies and probabilistically switch between them. The primary strategy is the non-compensatory 'Take The Best' (TTB) heuristic, which evaluates features sequentially by validity. However, on a proportion of trials, subjects employ a simple compensatory 'Tallying' strategy, which counts the total number of positive features for each option regardless of validity. This mixture allows the model to capture both the dominant lexicographic behavior and the occasional compensatory deviations observed in human choices.

**Rationale:** Following the arbiter's suggestion, this model replaces the Exponentially Weighted Additive Model with a Mixture Model. Since TTB explains the majority of the variance but struggles to capture all trial-by-trial deviations, mixing it with a simple compensatory strategy (Tallying) provides a structurally different mechanism. Instead of applying static, extreme weights to all features, the model assumes subjects probabilistically select between a strict non-compensatory heuristic and a simple counting heuristic. This preserves the high accuracy of TTB while injecting cognitively plausible variance.

**Parameters:**
  - `w_ttb`: `[0.5, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (TTB)
    order = np.argsort(validities)[::-1]
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    ttb_scores = np.array([ttb_a, ttb_b])
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Strategy 2: Tallying (Unit-weight linear model)
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    tally_scores = np.array([tally_a, tally_b])
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Naive Bayes Evidence Integration (Log-Odds WADD) with Normalized Strong Amplification: Subjects evaluate options by integrating all available features, weighting each feature by its log-odds of validity. To mimic the strongly non-compensatory 'Take The Best' heuristic, subjects apply a steep non-linear amplification (gamma >= 2) to these log-odds. Crucially, these amplified weights are normalized to sum to 1, ensuring that the subjective evaluation bounded and allowing the softmax temperature (beta) to independently govern decision noise without being overwhelmed by massive unscaled scores.

**Rationale:** Following the critic's diagnosis, the unnormalized exponentiated weights created massive score magnitudes, nullifying the effect of the softmax temperature and forcing deterministic choices. By normalizing the weights to sum to 1 before calculating the scores, the scores are bounded between 0 and 1. This decouples the non-compensatory shape of the weights (controlled by gamma) from the stochasticity of the choice (controlled by beta), allowing the model to better balance the probabilistic matching required by Experiment 3 and the highly non-compensatory choices seen in Experiment 7.

**Parameters:**
  - `gamma`: `[2.0, 10.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    a = np.asarray(state[0], dtype=float)
    b = np.asarray(state[1], dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid division by zero or log(0)
    # Since validities are >= 0.5, log-odds will be strictly positive
    v = np.clip(validities, 0.50001, 0.99999)
    log_odds = np.log(v / (1.0 - v))
    
    # Apply subjective scaling to the log-odds
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Normalize weights to decouple non-compensatory shape from softmax temperature
    weights = weights / np.sum(weights)
    
    # Calculate weighted additive scores
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))
```
