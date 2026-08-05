# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Decision-makers use a Compensatory / Weighted Additive (WADD) strategy, where an overall value for each option is computed by weighting each feature by a subjective transformation of its validity (validity raised to a power) and summing these weighted features. On strict conflict trials, the high weight of the single discriminating cue roughly balances the combined weights of multiple opposing lower-validity cues. This produces a near-zero difference in overall value, naturally resulting in a choice probability near 0.5 without requiring a mixture of distinct heuristics.

**Rationale:** Following the critic's advice, we slightly shift the lower bound of the gamma range from 1.0 to 3.0 (resulting in [3.0, 8.0]). This ensures that the subjective weighting always amplifies the highest-validity cue enough to truly balance out multiple secondary cues, eliminating the slight Tallying bias observed in the previous iteration and bringing the simulated TTB match rate squarely to the empirical ~0.50 level. The rest of the mechanism and parameters are kept exactly the same.

**Parameters:**
  - `gamma`: `[3.0, 8.0]`
  - `beta`: `[0.1, 2.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities
    w = val ** gamma
    
    val_a = np.sum(a * w)
    val_b = np.sum(b * w)
    
    beta = float(parameters["beta"])
    scores = np.array([val_a, val_b])
    
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    p = (1.0 - epsilon) * p + epsilon * 0.5
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Top-K Tallying with Restricted Scope: Decision-makers simplify complex multi-attribute choices by focusing only on the top K most valid cues, ignoring all lower-validity information. By restricting K to a narrow band (1, 2, or 3) and allowing for a high degree of random guessing (lapse rate), the model produces consistent behavior across individuals while naturally capturing the pervasive ~0.5 choice probabilities on strict conflict trials where the top cues often tie or balance out.

**Rationale:** Following the critic's advice, we expand the discrete parameter 'k' from {2, 3} to {1, 2, 3}. The previous restriction successfully lowered variance and captured the ~0.5 conflict rates on exps 1, 3, and 4, but caused systematic biases on exps 5 and 6 because k=3 uniquely favors one option while k=2 yields ties. Restoring k=1 introduces a subpopulation that strictly follows the single best cue, perfectly balancing out the k=3 predictions and returning the aggregate metrics on exps 5 and 6 to their empirical values of ~0.50 and ~0.00.

**Parameters:**
  - `k`: `{1, 2, 3}`
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(parameters["k"])
    
    # Ensure k doesn't exceed the number of available features
    k = min(k, len(val))
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally positive features among top K
    a_score = np.sum(a[top_k_cues])
    b_score = np.sum(b[top_k_cues])
    
    if a_score > b_score:
        p_core = np.array([1.0, 0.0])
    elif b_score > a_score:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Conflict-Induced Guessing with Evidence Threshold: Decision-makers evaluate options using simple heuristics (Take The Best and Tallying) but also monitor the overall Weighted Additive (WADD) evidence. When the heuristics make strict opposing predictions, OR when the overall WADD evidence difference between the options is too small to confidently discriminate, the decision-maker experiences uncertainty and resorts to random guessing. This captures the pervasive ~0.5 choice probabilities and low variance across both strict heuristic conflict trials and trials with nominally agreeing heuristics but weak overall evidence.

**Rationale:** Following the critic's advice, I increased the upper bound of the `threshold` parameter from 5.0 to 15.0. This minimal edit allows the parameter search to find a threshold large enough to classify the larger WADD evidence differences in Experiment 8 as 'too close to call', naturally bringing the choice probabilities down closer to the empirical 0.51 while maintaining the successful fits on all other experiments.

**Parameters:**
  - `epsilon`: `[0.0, 0.2]`
  - `threshold`: `[0.0, 15.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Determine Take The Best (TTB) winner
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Determine Tallying winner
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Compute WADD difference
    wadd_a = np.sum(a * val)
    wadd_b = np.sum(b * val)
    wadd_diff = abs(wadd_a - wadd_b)
    threshold = float(parameters["threshold"])
        
    # Check for strict conflict or insufficient WADD evidence
    conflict = (ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner)
    
    if conflict or (wadd_diff <= threshold):
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: rely on the agreed winner (or the one that isn't tied)
        winner = ttb_winner if ttb_winner is not None else tally_winner
        if winner == 0:
            p_core = np.array([1.0, 0.0])
        elif winner == 1:
            p_core = np.array([0.0, 1.0])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
