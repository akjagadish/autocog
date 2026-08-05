# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

**Rationale:** Following the arbiter's feedback, this model instantiates the Take The Best (TTB) heuristic. Instead of summing features (like Tallying or WADD), TTB sorts features by validity and makes a choice based solely on the highest-validity feature that discriminates between the options. This non-compensatory mechanism naturally accounts for the extreme sensitivity to high-validity cues observed in Experiment 2, where a single predictive feature can strongly dominate the choice over multiple weaker features.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Rank-Weighted Additive Theory: Decision-makers evaluate options using a compensatory but steeply decaying weighting scheme. Instead of using raw validities as weights, they rank features by their validity and assign exponentially decaying weights based on their rank (e.g., w_k = decay_rate^{-k}). This creates a 'soft' lexicographic strategy that largely mimics Take The Best by making the most valid cue dominant, but allows for compensation if multiple lower-ranked cues unanimously oppose the top cue.

**Rationale:** Following the arbiter's suggestion, this theory implements a 'soft' lexicographic model. By ranking features by validity and assigning exponentially decaying weights (controlled by a decay_rate parameter), the model bridges the gap between the strict non-compensatory Take The Best (TTB) heuristic and the fully compensatory Weighted Additive (WADD) model. A decay rate close to 2.0 ensures the top cue is usually decisive, but allows a coalition of lower-ranked cues to outvote it, capturing human behavior more robustly across experiments.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `decay_rate`: `[1.5, 4.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Rank-Weighted Additive expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    decay_rate = float(parameters["decay_rate"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Assign exponentially decaying weights based on rank
    weights = np.zeros_like(validities)
    for k, idx in enumerate(order):
        weights[idx] = decay_rate ** (-k)
        
    # Compute weighted sum for each option
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Take-Two with Conditional Fallback: Decision-makers evaluate the top two most valid features. If these two features agree or one favors an option while the other ties, that option is chosen. If they conflict (each option wins one), the decision-maker probabilistically mixes between reverting to the 1st feature and the 3rd feature. If the top two features tie (neither option wins on either), the decision-maker falls back to a simple tally of all features to break the tie.

**Rationale:** Following the critic's feedback, I separated the fallback logic for ties and conflicts. When the top two features tie (neither option wins on either feature), the model falls back to a simple Tallying mechanism to allow lower-validity features to break the tie, correctly capturing the behavior in Experiment 5. When the top two features conflict (each option wins on one feature), the model mixes between reverting to the 1st feature and the 3rd feature, preserving the strong validity-driven preferences required for Experiments 3 and 6. This conditional fallback retains the strengths of both mechanisms where they are empirically supported.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Take-Two expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    # Evaluate the top two features
    f1, f2 = order[0], order[1]
    
    wins_a = (a[f1] > b[f1]) + (a[f2] > b[f2])
    wins_b = (b[f1] > a[f1]) + (b[f2] > a[f2])
    
    if wins_a > wins_b:
        score_a, score_b = 1.0, 0.0
    elif wins_b > wins_a:
        score_a, score_b = 0.0, 1.0
    else:
        if wins_a == 1 and wins_b == 1:
            # Conflict in top 2 features
            gamma = float(parameters["gamma"])
            
            # F1 preference (revert to most valid feature)
            score_a_f1, score_b_f1 = 0.5, 0.5
            if a[f1] > b[f1]:
                score_a_f1, score_b_f1 = 1.0, 0.0
            elif b[f1] > a[f1]:
                score_a_f1, score_b_f1 = 0.0, 1.0
                
            # F3 preference
            score_a_f3, score_b_f3 = 0.5, 0.5
            if len(order) > 2:
                f3 = order[2]
                if a[f3] > b[f3]:
                    score_a_f3, score_b_f3 = 1.0, 0.0
                elif b[f3] > a[f3]:
                    score_a_f3, score_b_f3 = 0.0, 1.0
                    
            score_a = gamma * score_a_f1 + (1.0 - gamma) * score_a_f3
            score_b = gamma * score_b_f1 + (1.0 - gamma) * score_b_f3
        else:
            # Tie in top 2 features (0 wins each)
            tally_a = np.sum(a > b)
            tally_b = np.sum(b > a)
            
            score_a, score_b = 0.5, 0.5
            if tally_a > tally_b:
                score_a, score_b = 1.0, 0.0
            elif tally_b > tally_a:
                score_a, score_b = 0.0, 1.0
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
