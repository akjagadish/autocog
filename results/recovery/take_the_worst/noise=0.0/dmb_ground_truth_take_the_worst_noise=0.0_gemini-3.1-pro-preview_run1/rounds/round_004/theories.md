# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Weighted Tallying: People compare options by computing a weighted sum of feature wins, where the weights are a heavily compressed function of the stated cue validities (w_i = validity_i ** gamma, with gamma close to 0). This allows the strategy to behave almost exactly like simple Tallying when one option has a clear numerical advantage in feature wins, but smoothly and compensatorily resolves ties using all available validity information rather than defaulting to a strict lexicographic tie-breaker.

**Rationale:** By tightening the parameter range for gamma from [0.0, 2.0] to [0.0, 0.3], we ensure the weights are heavily compressed and remain very close to 1.0. This prevents the model from acting like a Weighted Additive (WADD) strategy when one option has a clear numerical advantage in feature wins, correctly capturing human reliance on Tallying in Experiments 3, 4, and 6. At the same time, the slight validity differences perfectly preserve the compensatory tie-breaking behavior needed for Experiments 1, 7, and 8.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 0.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Compress validities to create weights that are close to 1 (Tallying-like) 
    # but still retain ordinal validity information for compensatory tie-breaking.
    weights = val ** gamma
    
    # Calculate weighted feature wins (ignoring ties on individual features)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    
    scores = np.array([a_wins, b_wins])
    
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
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_2` — SURVIVED ✓

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Tally-then-Feature-Drop: Decision makers primarily use a Tallying (Equal-Weight) heuristic, counting the number of features on which each option is superior. When this raw count results in a tie, they employ a bottom-up feature elimination heuristic. They iteratively ignore the lowest-validity features and re-tally until the tie is broken, effectively penalizing options that rely on the weakest cues and favoring those with multiple moderately-high validities.

**Rationale:** Based on the arbiter's feedback, the 'Tally-then-TTB' strategy systematically reversed human preferences on tie trials by favoring the single highest-validity cue. To correct this, we implemented the suggested 'feature-drop' heuristic. When the raw tally is tied, the model iteratively ignores the lowest-validity features and re-tallies until the tie is broken. This bottom-up elimination correctly penalizes options that rely on the weakest cues, thereby capturing the human tendency to favor multiple moderately-high validities over a single highest validity when resolving equal-weight ties.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Primary phase: Tallying feature wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins == b_wins:
        # Secondary phase: Bottom-up feature drop tie-breaker
        val = np.asarray(parameters["validities"], dtype=float)
        drop_order = np.argsort(val, kind="stable")
        
        active_features = np.ones(len(val), dtype=bool)
        
        tie_breaker_a = 0.0
        tie_breaker_b = 0.0
        
        for idx in drop_order:
            active_features[idx] = False
            new_a_wins = float(np.sum((a > b) & active_features))
            new_b_wins = float(np.sum((b > a) & active_features))
            
            if new_a_wins > new_b_wins:
                tie_breaker_a = 1.0
                break
            elif new_b_wins > new_a_wins:
                tie_breaker_b = 1.0
                break
                
        scores = np.array([a_wins + tie_breaker_a, b_wins + tie_breaker_b])
    else:
        scores = np.array([a_wins, b_wins])
        
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
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```
