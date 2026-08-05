# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Random Guessing / High-Noise Tallying: Due to the lack of trial-by-trial feedback and potentially low engagement, subjects largely ignore cue validities and feature values. Instead, they make choices that are nearly indistinguishable from uniform random guessing, which can be modeled as a simple unweighted tally of features heavily diluted by extreme decision noise and high lapse rates. This explains the ~0.5 agreement with Take The Best, ~0.44 agreement with Tallying, and the near-zero variance in choice proportions across different trial types.

**Rationale:** Following the arbiter's recommendation, this theory models subjects as being highly disengaged or unable to form a coherent strategy due to the lack of feedback. They rely on a simple Tallying heuristic but with extremely high decision noise (low beta) and a very high lapse rate (high epsilon), effectively reducing their choices to near-random guessing. This aligns with the observed ~0.50 agreement with TTB (Exp 1), ~0.44 agreement with Tallying (Exp 2), and near-zero variance across trial types in Exps 3 and 4, which is characteristic of a flat, uninformative response profile.

**Parameters:**
  - `beta`: `[0.0, 0.05]`
  - `epsilon`: `[0.8, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of winning features
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    # Blend with uniform guessing based on epsilon
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_3` — SURVIVED ✓

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

### `pi_5` → slot 1 (via `new_theory`)

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
