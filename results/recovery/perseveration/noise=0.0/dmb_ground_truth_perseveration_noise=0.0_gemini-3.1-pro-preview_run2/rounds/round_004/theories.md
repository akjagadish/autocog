# Round 4 — Theories

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


### slot 2 — `pi_6` — KILLED ✗

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


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Biased Constant Choice: Due to low engagement or a lack of trial-by-trial feedback, subjects adopt a degenerate strategy of picking one option and sticking to it for the entire experiment. However, the initial selection of this preferred option is not necessarily a symmetric 50/50 split; subjects exhibit a population-level bias (e.g., a baseline preference for Option B). They make a biased probabilistic choice on their very first trial and then deterministically repeat that exact choice on all subsequent trials, resulting in zero within-subject variance while capturing asymmetric population preferences.

**Rationale:** To address the arbiter's feedback, we need a theory that maintains the perfect 0.0000 within-subject variance of Theory 1 (pi_5) while allowing the population-level preference to deviate from a strict 50/50 split. By introducing a continuous parameter 'p_a' that governs the choice probability only on the very first trial, the model can capture population-level biases (e.g., towards Option B in unbalanced designs). By forcing all subsequent trials to deterministically copy the first trial's response via the history object, we guarantee that every simulated subject exhibits exactly zero within-subject variance, perfectly matching the empirical data on variance metrics.

**Parameters:**
  - `p_a`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # If the subject has already made a choice in a previous trial, they stick to it deterministically
    if len(history["response"]) > 0:
        first_choice = history["response"][0]
        if first_choice == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
            
    # On the very first trial, the subject chooses an option based on a population-level bias
    p_a = float(parameters["p_a"])
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
