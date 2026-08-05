# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Strong Position Bias / Constant Choice: Due to the lack of trial-by-trial feedback and low engagement, subjects adopt a degenerate strategy of always choosing the same option (e.g., always Option A or always Option B) regardless of the cues. This leads to choice probabilities of 1.0 or 0.0 for a given subject across all trials, perfectly explaining the near-zero within-subject variance across trial types and the extreme choice probabilities observed.

**Rationale:** The arbiter pointed out that subjects might be completely disengaged and use a constant choice strategy, resulting in extreme probabilities (0 or 1) for a given subject. This perfectly predicts the 0.0 variance across trial types in Experiments 3 and 4, and the 0.5 extremeness in Experiments 5 and 6, while yielding ~0.5 agreements in Experiments 1 and 2.

**Parameters:**
  - `preferred_option`: `{0, 1}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # The subject has a strict preference for either Option A (0) or Option B (1)
    pref = int(parameters["preferred_option"])
    
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_3` — KILLED ✗

**Description:** People use a Weighted Additive (WADD) strategy when comparing two options, but they subjectively scale the stated cue validities. By applying a non-linear transformation to the validities (exponentiating them by a parameter gamma), decision-makers can amplify the differences between cues (mimicking Take The Best) or compress them (mimicking Tallying). Expanding the range of gamma allows for extreme weight disparities, accommodating individuals who rely heavily on the most valid cue.

**Rationale:** Following the critic's feedback, the upper bound of the 'gamma' parameter is significantly widened from 5.0 to 20.0. This allows the model to achieve the extreme weight disparities needed to more closely mimic Take-The-Best behavior for subjects who heavily prioritize the most valid cues, reducing the over-prediction of Tallying and increasing the TTB agreement to better match empirical targets.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    gamma = float(parameters["gamma"])
    val = np.asarray(parameters["validities"], dtype=float) ** gamma
    
    # Compute the weighted sum of features for each option
    ev_a = np.sum(val * a)
    ev_b = np.sum(val * b)
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

**Description:** Mixture of Constant Choice and Take-The-Best: Subjects primarily exhibit a degenerate strategy of relying on a fixed position preference (always choosing Option A or always Option B) due to low engagement or lack of trial-by-trial feedback. However, on a small fraction of trials, they lapse into using a single-cue heuristic (Take-The-Best), where they compare the options on the most valid cue. This mixture maintains the near-zero variance in choice proportions across most experiments while capturing the slight preference for TTB over Tallying in disagreement trials.

**Rationale:** Following the arbiter's suggestion, this model abandons compensatory cue weighting and builds on the highly successful constant-choice mechanism of Theory 1 (pi_5). By modeling behavior as a mixture of a strict position preference and occasional lapses into a single-cue heuristic (Take-The-Best), the model retains the ability to predict the near-zero variance and extreme choice probabilities seen across most experiments. The small probability of using TTB (governed by epsilon) allows the model to capture the slight deviations from exactly 0.5 observed in experiments that pit heuristics against each other (e.g., pulling the Tallying agreement down to ~0.44 in Experiment 2) without drastically inflating variance.

**Parameters:**
  - `preferred_option`: `{0, 1}`
  - `epsilon`: `[0.0, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    pref = int(parameters["preferred_option"])
    epsilon = float(parameters["epsilon"])
    
    # Constant choice probabilities
    p_const = np.array([1.0, 0.0]) if pref == 0 else np.array([0.0, 1.0])
    
    # Take-The-Best (TTB) prediction
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    ttb_pred = -1
    for idx in order:
        if a[idx] > b[idx]:
            ttb_pred = 0
            break
        elif b[idx] > a[idx]:
            ttb_pred = 1
            break
            
    if ttb_pred == 0:
        p_ttb = np.array([1.0, 0.0])
    elif ttb_pred == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p_const + epsilon * p_ttb
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
