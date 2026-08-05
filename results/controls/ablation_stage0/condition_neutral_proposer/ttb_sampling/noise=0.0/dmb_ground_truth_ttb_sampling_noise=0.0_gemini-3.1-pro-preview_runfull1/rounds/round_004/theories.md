# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

**Rationale:** Following the arbiter's suggestion, this model implements the 'Take The Best' (TTB) heuristic. Instead of integrating all features compensatorily (like WADD) or equally (like Tallying), TTB searches through features in descending order of their validity and stops at the first feature that discriminates between the two options. The choice is made based entirely on this single cue. This provides a strong non-compensatory heuristic mechanism that tests whether subjects rely on the most valid discriminating cue rather than integrating multiple cues.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** The Weighted Additive (WADD) model posits a fully compensatory decision strategy. Decision-makers evaluate all available cues for both options, weighting each cue by its subjective importance (operationalized as the log-odds of its validity). These weighted cues are integrated into a single compensatory score for each option. The option with the higher score is more likely to be chosen, with choices being probabilistic according to a softmax function over the scores. This provides a strong compensatory baseline to contrast with non-compensatory heuristics like Take The Best.

**Rationale:** Following the arbiter's feedback, I have implemented the Weighted Additive (WADD) model to serve as a classic compensatory baseline. Unlike TTB, which stops at the first discriminating cue, WADD integrates all available information by weighting each feature according to its log-odds validity. This approach captures situations where multiple weak cues might overpower a single strong cue, contrasting sharply with non-compensatory mechanisms and providing a foundational model for the compensatory family.

**Parameters:**
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds to serve as weights
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)
    weights = np.log(validities / (1.0 - validities))
    
    # Calculate compensatory scores for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Convert scores to probabilities via softmax with max-subtraction for stability
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Model (TTB and Tallying): Decision-makers do not rely on a single strategy; instead, they employ a mixture of heuristics. The dominant strategy is Take The Best (TTB), a non-compensatory heuristic where cues are searched in order of validity and the first discriminating cue determines the choice. However, on a subset of trials, decision-makers might use Tallying, a simple compensatory heuristic that ignores cue validities and simply counts the number of cues favoring each option. This mixture accounts for both the strong evidence of non-compensatory processing and the occasional compensatory behavior observed in human data. The mixture overwhelmingly favors TTB.

**Rationale:** Following the critic's feedback, the lower bound of the w_ttb parameter has been increased from 0.5 to 0.8. This ensures that Take The Best (TTB) remains overwhelmingly dominant in the Strategy Mixture Model, reducing the overestimation of Tallying usage and better matching the near-TTB performance observed in human data.

**Parameters:**
  - `w_ttb`: `[0.8, 1.0]`
  - `epsilon`: `[0.0, 0.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) Prediction
    order = np.argsort(validities)[::-1]
    ttb_a_wins = False
    ttb_b_wins = False
    
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            ttb_a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            ttb_b_wins = True
            break
            
    if ttb_a_wins:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_b_wins:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Tallying Prediction (counting winning cues)
    tally_a = np.sum(stim[0] > stim[1])
    tally_b = np.sum(stim[1] > stim[0])
    
    if tally_a > tally_b:
        p_tally = np.array([1.0, 0.0])
    elif tally_b > tally_a:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of TTB and Tallying
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
