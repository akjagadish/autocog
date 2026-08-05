# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Constant-Mixture Strategy Selection with dampened Tallying: Decision-makers evaluate options using a probabilistic mixture of a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying). Individuals have a fixed, trait-like probability of using TTB versus Tallying on any given trial, independent of cue depth. To account for zero covariance between tally score differences and choices in certain environments, the temperature parameter for the Tallying component can reach zero, dampening its linear tracking of score differences.

**Rationale:** Following the critic's feedback, the Constant-Mixture Strategy Selection model is maintained, but the parameter range for `tau` has been widened to `[0.0, 5.0]`. This minor adjustment allows the model to select smaller scaling values for the Tallying softmax, which flattens the Tallying probability distribution. By doing so, it dampens the strict linear tracking of score differences, effectively reducing the inflated covariance observed in Experiment 4 while keeping the core theoretical mechanism intact.

**Parameters:**
  - `alpha`: `[0.0, 1.0]`
  - `tau`: `[0.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine Take-The-Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine Tallying prediction using softmax
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    tau = float(parameters["tau"])
    
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)
        
    # Probabilistic strategy switch (constant mixture, independent of depth)
    alpha = float(parameters["alpha"])
    p_core = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Pure Take-The-Best (Lexicographic) Decision Making: Subjects deterministically choose the option favored by the single most valid cue that discriminates between the two options. They do not use compensatory weighting and do not mix strategies. Choices are governed entirely by the highest-validity differing cue, with a near-zero lapse rate.

**Rationale:** Following the arbiter's feedback, this model implements a Pure Take-The-Best strategy. It simply ranks cues by the provided objective validities, finds the highest-validity cue that differs, and deterministically chooses the option with the higher value on that cue. It avoids the noise introduced by the softmax formulation in earlier theories, utilizing only a near-zero lapse rate (epsilon in [0.0, 0.05]) to capture occasional random errors, and predicts 0.5 when all cues tie.

**Parameters:**
  - `epsilon`: `[0.0, 0.05]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner == 0:
        p = np.array([1.0, 0.0])
    elif winner == 1:
        p = np.array([0.0, 1.0])
    else:
        p = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Random Choice / High-Lapse Theory: Due to task complexity, lack of motivation, or cognitive overload, subjects largely guess between the two options. Any residual strategy use is negligible, making behavior predominantly indistinguishable from a coin flip.

**Rationale:** The arbiter prescribed a Random Choice / High-Lapse theory. The zero covariances and ~0.5 match rates across the experimental suite indicate that subjects largely guess between the two options, likely due to task complexity or cognitive overload. This model implements an overwhelming base lapse rate (epsilon near 1.0), effectively predicting a 50/50 probability for almost all trials. This naturally captures the zero covariances in Exp 4-8 and 10, and the ~0.50 match rates in Exp 1-3.

**Parameters:**
  - `epsilon`: `[0.95, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    epsilon = float(parameters["epsilon"])
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Minimal residual strategy (Take-The-Best) for the very rare non-lapse trials
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner == 0:
        p_core = np.array([1.0, 0.0])
    elif winner == 1:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Overwhelming lapse rate dominates the prediction
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
