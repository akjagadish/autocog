# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

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


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Hybrid TTB-WADD Theory: Decision-makers probabilistically alternate between a strict non-compensatory strategy (Take The Best) and a fully compensatory strategy (Weighted Additive Model / WADD) depending on the trial or internal state. The model computes the choice probabilities for both TTB (relying solely on the first discriminating cue) and WADD (summing the validity-weighted cue values) and blends them using a subject-level mixing parameter.

**Rationale:** Following the critic's advice, we replaced the direct multiplier `beta` with an inverse temperature parameter `tau` in [0.01, 5.0]. Inverse temperature scaling provides a smoother optimization landscape, which should help the fitter find the optimal subject-specific `mix_rate` without getting stuck in sub-optimal compromises. We also constrained `epsilon` to [0.0, 0.05] to prevent random noise from artificially flattening the deterministic TTB predictions.

**Parameters:**
  - `tau`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.05]`
  - `mix_rate`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- TTB Probability ---
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    ttb_p = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_p = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_p = np.array([0.0, 1.0])
            break
            
    # --- WADD Probability ---
    score_a = np.sum(a * validities)
    score_b = np.sum(b * validities)
    wadd_scores = np.array([score_a, score_b])
    
    tau = float(parameters["tau"])
    z = (wadd_scores - np.max(wadd_scores)) / tau
    e = np.exp(z)
    wadd_p = e / np.sum(e)
    
    # --- Blend ---
    mix_rate = float(parameters["mix_rate"])
    p_core = mix_rate * ttb_p + (1.0 - mix_rate) * wadd_p
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
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
