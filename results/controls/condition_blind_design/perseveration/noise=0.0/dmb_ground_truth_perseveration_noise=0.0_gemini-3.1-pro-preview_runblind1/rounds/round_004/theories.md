# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Subjects use a Tallying (Equal Weights) heuristic, where they ignore the explicit cue validities and simply count the total number of positive features for each option. They choose the option with the highest total number of positive features, and guess randomly if there is a tie. A highly restricted softmax temperature and lapse rate account for the overall noisy, near-chance behavior observed in the experiments.

**Rationale:** Following the critic's feedback, the Tallying mechanism is retained exactly as in the previous candidate, but the upper bound for the inverse temperature parameter (beta) has been tightened from 1.0 to 0.3. This minimal edit forces the softmax to be flatter, preventing the model from predicting higher accuracy on trials with large differences in total positive features. This adjustment aligns the model's predictions with the empirical data from Experiments 5 and 6, which show near-zero sensitivity to differences in WADD scores.

**Parameters:**
  - `beta`: `[0.0, 0.3]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Count total positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Decision-makers evaluate options using a Weighted Additive (WADD) strategy. Instead of relying on a single best cue (like Take The Best) or ignoring cue importance (like Tallying), individuals integrate all available features by weighting each feature according to its validity. The overall value of an option is the sum of its validity-weighted features. Choices are then made probabilistically by comparing these weighted sums, with response consistency governed by a softmax temperature parameter and a base lapse rate. A highly restricted temperature parameter prevents over-sensitivity to small differences in the weighted sums, matching the near-chance behavior of human subjects.

**Rationale:** Following the arbiter's recommendation, this model implements a Weighted Additive (WADD) heuristic. It computes a weighted sum of positive features using the explicit cue validities as weights. To accurately capture the near-chance empirical performance (e.g., choice proportions hovering around 0.50-0.55), the model incorporates a highly restricted softmax inverse temperature (beta in [0.0, 0.5]) and a lapse rate (epsilon). This prevents the deterministic overconfidence seen in unconstrained WADD models and aligns the choice probabilities with the observed data.

**Parameters:**
  - `beta`: `[0.0, 0.5]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate validity-weighted sums for both options
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Take The Best (TTB) with Extreme Lapse Rate: Decision-makers use a non-compensatory lexicographic heuristic, ranking cues by validity and stopping at the first discriminating feature. However, their execution is highly noisy, resulting in choices that are heavily weighted toward random guessing (extreme lapse rate), matching the empirical observation that subjects' behavior hovers near chance.

**Rationale:** Following the critic's feedback, the epsilon parameter range has been tightened from [0.8, 1.0] to [0.95, 1.0]. This minimal edit reduces the over-prediction of the Take The Best choice, bringing the model's average expected prediction down to roughly 0.51-0.52. This aligns better with several experiments where empirical performance hovers at or slightly below 0.50, without abandoning the core TTB mechanism.

**Parameters:**
  - `epsilon`: `[0.95, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity descending; stable sort preserves original order on ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    epsilon = float(parameters["epsilon"])
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
