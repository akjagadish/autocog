# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Softened Integrated Strategy Value Theory: Decision-makers evaluate options by integrating evidence from multiple heuristics (Take-The-Best, Tallying, and Weighted Additive) into a single continuous subjective value for each option. To prevent non-compensatory heuristics from disproportionately dominating the integrated value, the TTB signal is softened by scaling it according to the normalized validity of the discriminating cue. These integrated values are then compared via a stochastic decision process (softmax) to produce a choice. This allows the model to gracefully capture indifference in delicately balanced trials while still reflecting heuristic-aligned preferences.

**Rationale:** Following the critic's advice, we modified the TTB score from a hard [1.0, 0.0] assignment to a softened version that assigns the normalized validity of the discriminating cue (val[idx] / sum_val). This prevents the TTB signal from overly dominating the integrated value when the compensatory signals are closely tied, mitigating the large deviation from 0.5 observed in Experiment 6. Additionally, we reduced the upper bound of the softmax temperature parameter `beta` from 20.0 to 10.0 to prevent the model from amplifying small integrated value differences into deterministic choices.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `w_tal`: `[0.0, 1.0]`
  - `w_wadd`: `[0.0, 1.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    sum_val = np.sum(val)
    
    # Take-The-Best (TTB) Score (softened)
    diff = a - b
    order = np.argsort(val)[::-1]
    score_ttb_a, score_ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            score_ttb_a, score_ttb_b = val[idx] / sum_val, 0.0
            break
        elif diff[idx] < 0:
            score_ttb_a, score_ttb_b = 0.0, val[idx] / sum_val
            break
            
    # Tallying Score
    score_tal_a = np.sum(a) / n_features
    score_tal_b = np.sum(b) / n_features
    
    # Weighted Additive (WADD) Score
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights
    w_sum = w_ttb + w_tal + w_wadd
    if w_sum > 0:
        w_ttb /= w_sum
        w_tal /= w_sum
        w_wadd /= w_sum
    else:
        w_ttb = w_tal = w_wadd = 1.0 / 3.0
        
    # Integrated Option Values
    score_a = w_ttb * score_ttb_a + w_tal * score_tal_a + w_wadd * score_wadd_a
    score_b = w_ttb * score_ttb_b + w_tal * score_tal_b + w_wadd * score_wadd_b
    
    scores = np.array([score_a, score_b])
    
    # Single Softmax Decision Process
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Sequential Evidence Accumulation Theory: Decision-makers evaluate features sequentially in decreasing order of validity, maintaining a running sum of validity-weighted evidence. If this accumulated evidence exceeds an internal confidence threshold at any point, evaluation stops and a choice is made based on the current evidence (resembling Take-The-Best when the threshold is low). If all features are exhausted without crossing the threshold, the choice is based on the final accumulated sum (resembling Weighted Additive). This captures the spectrum from non-compensatory to compensatory decision-making through a single mechanistic stopping rule.

**Rationale:** Following the arbiter's suggestion, this theory models decision-making as a sequential evidence accumulation process. Instead of probabilistically selecting between distinct heuristics (as in Theory 1) or statically integrating all features simultaneously (as in Theory 2), this model evaluates features in order of validity and stops as soon as a confidence threshold is reached. By varying the threshold parameter, the model naturally interpolates between purely non-compensatory (Take-The-Best) behavior when the threshold is low, and purely compensatory (Weighted Additive) behavior when the threshold is high. This process-based approach elegantly captures threshold effects and choice indifference in balanced trials without needing to explicitly mix heuristic outputs.

**Parameters:**
  - `threshold`: `[0.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(val)[::-1]
    
    # Accumulate evidence sequentially
    E = 0.0
    for idx in order:
        E += val[idx] * (a[idx] - b[idx])
        if abs(E) >= threshold:
            break
            
    # E represents the final accumulated evidence in favor of Option A (if > 0)
    # or Option B (if < 0). We convert this to choice probabilities via softmax.
    scores = np.array([beta * E, 0.0])
    scores -= np.max(scores)
    p = np.exp(scores) / np.sum(np.exp(scores))
    
    # Apply lapse rate
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
