# Round 6 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** Dual-Process Strategy Selection with TTB/WADD Fallback: Decision-makers evaluate the raw tally difference between two options first. If the tally difference is highly discriminative (greater than or equal to a threshold), they rely on the fast, compensatory Tallying heuristic. If the tallies are tied or very close, they switch to a more effortful strategy (WADD or Take-The-Best) and use an independent temperature parameter to scale the distinct evidence magnitudes.

**Rationale:** Initial logic and parameters are validated. Standard processing applied. Added `beta_fallback` parameter to independently scale the determinism of the fallback strategy (TTB/WADD), preventing probability miscalibration from mismatched score magnitudes.

**Parameters:**
  - `threshold`: `{1, 2, 3, 4, 5}`
  - `use_ttb`: `{0, 1}`
  - `beta`: `[0.1, 20.0]`
  - `beta_fallback`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    use_ttb = int(parameters["use_ttb"])
    beta = float(parameters["beta"])
    beta_fallback = float(parameters["beta_fallback"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if abs(tally_a - tally_b) >= threshold:
        scores = np.array([tally_a, tally_b])
        active_beta = beta
    else:
        active_beta = beta_fallback
        if use_ttb == 1:
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = None
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
        else:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            scores = np.array([wadd_a, wadd_b])
        
    z = active_beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_8` — SURVIVED ✓

**Description:** Strict Tallying with Validity Tie-Breaker: Decision-makers primarily rely on the unweighted tally of positive features to evaluate options. They always compute this simple tally first. If one option has a strictly higher tally, they choose it (subject to decision noise), ignoring cue validities entirely. Only when the tallies are exactly equal do they incorporate cue validities as a tie-breaker, utilizing either a lexicographic (Take-The-Best) or Weighted Additive strategy to resolve the decision.

**Rationale:** Following the arbiter's instructions, this theory completely replaces the previous WADD model with a strict lexicographic hierarchy: Tally -> Validity. The decision-maker computes the unweighted tally of positive cues first. If the tallies differ, the choice is driven entirely by the tally difference, naturally predicting the high reliance on tallies (Exp 11) and the ~0 sensitivity to validities when tallies differ (Exp 2). Only when the tallies are tied does the model consider cue validities, using either a Take-The-Best or Weighted Additive tie-breaker (controlled by the `use_wadd` parameter). This maintains validity-based choices in matched-tally trials while capturing the primary dominance of the tallying heuristic.

**Parameters:**
  - `beta_tally`: `[0.1, 20.0]`
  - `beta_tie`: `[0.1, 20.0]`
  - `use_wadd`: `{0, 1}`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_tie = float(parameters["beta_tie"])
    use_wadd = int(parameters["use_wadd"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if tally_a != tally_b:
        scores = np.array([tally_a, tally_b])
        active_beta = beta_tally
    else:
        active_beta = beta_tie
        if use_wadd == 1:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            scores = np.array([wadd_a, wadd_b])
        else:
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = None
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                if b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
                
    z = active_beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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

### `pi_9` → slot 1 (via `new_theory`)

**Description:** Configural Log-Odds Evidence Accumulation with Bounded Non-Linearity

**Rationale:** Following the critic's advice, we restrict the `gamma` parameter range to `[0.0, 4.0]` (down from 10.0) while keeping the configural penalty unchanged. This prevents the top validities from becoming so extremely weighted that they completely overwhelm the compensatory penalty, allowing the model to capture the human preference for the tally winner in Exp 9 and 13 while retaining validity-sensitive behavior in Exp 1 and 2.

**Parameters:**
  - `gamma`: `[0.0, 4.0]`
  - `lambda_pen`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    lambda_pen = float(parameters["lambda_pen"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Clip validities to avoid division by zero or log(1)
    v_clipped = np.clip(val, 0.5001, 0.9999)
    
    # Calculate log-odds (Naive Bayes evidence)
    log_odds = np.log(v_clipped / (1.0 - v_clipped))
    
    # Apply non-linear scaling to capture individual differences in extreme cue weighting
    w = log_odds ** gamma
    
    # Configural penalty: missing cues interact, heavily penalizing options with multiple absent cues
    penalty_a = lambda_pen * (np.sum(w * (1.0 - a))) ** 2
    penalty_b = lambda_pen * (np.sum(w * (1.0 - b))) ** 2
    
    score_a = np.sum(w * a) - penalty_a
    score_b = np.sum(w * b) - penalty_b
    
    scores = np.array([score_a, score_b])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
