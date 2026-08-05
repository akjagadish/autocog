# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) Theory with Non-Linear Cue Scaling: People evaluate multi-attribute options by computing an overall value for each option. This value is determined by taking a weighted sum of the option's features, where the weights correspond to the subjective validities of the respective cues scaled by a non-linear parameter. This scaling allows decision-makers to flexibly upweight highly valid cues (approaching a non-compensatory strategy) or downweight them (approaching an equal-weight tallying strategy). Decision-makers then choose probabilistically between the options by comparing these total weighted scores via a softmax function.

**Rationale:** Following the critic's advice, I adjusted the upper bound of the `gamma` parameter from 4.0 to 6.0. The previous iteration showed that a maximum gamma of 4.0 caused an undershoot in non-compensatory behavior for Experiment 2 (overpredicting Tallying adherence), whereas an upper bound of 10.0 in an earlier iteration allowed too much non-compensatory behavior. Expanding the range to [0.1, 6.0] should help dial in the exact sweet spot between TTB and Tallying adherence.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `gamma`: `[0.1, 6.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for both options
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Hybrid Heuristic Theory: Decision-makers integrate evidence from multiple strategies before making a choice, rather than probabilistically sampling between distinct strategies. Specifically, individuals compute a combined subjective value for each option by taking a weighted average of normalized compensatory (validity-weighted) and non-compensatory (tallying) evidence. A single stochastic decision process then operates on these integrated values.

**Rationale:** Initial logic and parameters are validated. The final transformation implements a single-softmax Hybrid Heuristic. By normalizing the WADD scores by the sum of validities and the Tallying scores by the total number of features, both evidence sources are projected onto a comparable scale. These normalized scores are then integrated via a weighted average before applying a single softmax temperature, avoiding the dual-softmax calibration issues of previous iterations while capturing the nuanced influence of both strategies.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_mix`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Normalized WADD scores
    sum_val = np.sum(val)
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    # Normalized Tallying scores
    score_tal_a = float(np.sum(a > b)) / n_features
    score_tal_b = float(np.sum(b > a)) / n_features
    
    beta = float(parameters["beta"])
    w_mix = float(parameters["w_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Combined integrated scores
    score_combined_a = w_mix * score_wadd_a + (1.0 - w_mix) * score_tal_a
    score_combined_b = w_mix * score_wadd_b + (1.0 - w_mix) * score_tal_b
    scores = np.array([score_combined_a, score_combined_b])
    
    # Single Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Probabilistic Strategy Selection Theory: Decision-makers maintain a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive/WADD). On each trial, they probabilistically sample one strategy from this repertoire according to base-rate strategy weights, and execute it purely. This discrete mixture of strategies accounts for intermediate aggregate choice proportions while naturally predicting exact ties when the sampled strategy itself encounters a tie.

**Rationale:** Following the arbiter's recommendation, this model implements the Probabilistic Strategy Selection theory. Instead of integrating evidence from different heuristics into a single subjective value (as in the previous Hybrid theory), this model assumes that decision-makers have a repertoire of three distinct strategies: Take-The-Best, Tallying, and Weighted Additive. On any given trial, the decision-maker probabilistically samples one of these strategies and executes it purely. This trial-by-trial mixture of deterministic strategies can produce intermediate aggregate choice proportions while also preserving exact ties when the sampled strategy predicts a tie, addressing the mechanistic failures of the previous integration-based model.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `w_tal`: `[0.0, 1.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB)
    diff = a - b
    order = np.argsort(val)[::-1]
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if diff[idx] > 0:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif diff[idx] < 0:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Tallying
    tal_diff = np.sum(a) - np.sum(b)
    if tal_diff > 0:
        tal_pred = np.array([1.0, 0.0])
    elif tal_diff < 0:
        tal_pred = np.array([0.0, 1.0])
    else:
        tal_pred = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD)
    wadd_diff = np.sum(val * a) - np.sum(val * b)
    if wadd_diff > 0:
        wadd_pred = np.array([1.0, 0.0])
    elif wadd_diff < 0:
        wadd_pred = np.array([0.0, 1.0])
    else:
        wadd_pred = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    w_arr = np.array([w_ttb, w_tal, w_wadd])
    sum_w = np.sum(w_arr)
    if sum_w == 0:
        p_strat = np.array([1/3, 1/3, 1/3])
    else:
        p_strat = w_arr / sum_w
        
    p_core = p_strat[0] * ttb_pred + p_strat[1] * tal_pred + p_strat[2] * wadd_pred
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
