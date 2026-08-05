# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic with extreme noise: Decision-makers avoid cognitive overload by not integrating all information. Instead, they search through cues in descending order of their explicitly stated validities. The choice is based entirely on the first cue that discriminates between the two options. However, due to the high cognitive demands of the task or lack of motivation, their behavior is overwhelmingly stochastic, requiring extremely high lapse rates and very low softmax temperatures to capture the near-random empirical choices.

**Rationale:** Following the critic's advice, the TTB mechanism remains exactly the same, but the parameter ranges for `beta` and `epsilon` have been further tightened to enforce near-random behavior. Restricting `beta` to [0.0, 1.0] and raising the lower bound of `epsilon` to [0.8, 1.0] ensures the model is forced into the extreme noise regime demanded by the empirical data, which overwhelmingly hovers around 0.5 across all experiments.

**Parameters:**
  - `beta`: `[0.0, 1.0]`
  - `epsilon`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue found
        p_core = np.array([0.5, 0.5])
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * scores
        z -= np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Biased Random Guessing: Due to cognitive overload from processing multiple conflicting binary cues with explicit validities, subjects abandon systematic cue integration and resort to random guessing. However, their guessing is not perfectly uniform; instead, they exhibit a parameterized spatial or positional bias (e.g., a baseline preference for Option A over Option B due to reading order). This allows the model to capture near-random choice behavior while gracefully absorbing slight empirical deviations from exactly 50%.

**Rationale:** Empirical metrics across multiple experiments hover very closely around 50%, suggesting that participants are largely guessing, likely due to the cognitive overload of integrating multiple explicit validities. However, the exact values often deviate slightly from 0.5. By introducing a single free parameter `p_a` to represent a baseline positional preference for Option A, this model maintains the core insight of the cognitive overload theory while providing the flexibility to fit these slight empirical deviations, improving upon the rigid 0.5/0.5 pure random guessing model.

**Parameters:**
  - `p_a`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess, but with a spatial/positional bias.
    p_a = float(parameters["p_a"])
    
    # Return the biased probabilities for Option A and Option B.
    return np.array([p_a, 1.0 - p_a])
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Tallying with High Noise: Subjects avoid the cognitive overload of weighting cues by their explicitly stated validities. Instead, they use an equal-weighting (Tallying) heuristic, simply counting the number of positive features for each option and favoring the one with the higher count. However, their choices are heavily influenced by random noise and high lapse rates, capturing the near-random guessing behavior observed across many experimental designs without introducing systematic spatial biases.

**Rationale:** Following the critic's advice, the previous iteration successfully improved fits by lowering the upper bound of the softmax temperature and increasing the lower bound of the lapse rate. To squeeze out the remaining over-prediction of the tallying heuristic and better match the near-random guessing observed in Exps 1-6, the beta parameter range is further restricted to [0.0, 0.2], and the epsilon parameter range is increased to [0.8, 1.0].

**Parameters:**
  - `beta`: `[0.0, 0.2]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    beta = float(parameters["beta"])
    scores = np.array([tally_a, tally_b]) * beta
    
    scores -= np.max(scores)
    p = np.exp(scores)
    p_core = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
