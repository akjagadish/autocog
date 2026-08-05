# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People compare two options by computing a weighted sum of all features for each option, where the weights are directly equal to the feature validities. Because all validities are bounded between 0.5 and 1.0, using them directly restricts the maximum weight ratio between cues, allowing multiple weaker cues to easily outvote a single strong cue. This Weighted Additive (WADD) rule is a fully compensatory strategy that captures the human tendency to integrate all available information while still giving a slight edge to more valid cues. Response noise enters through a softmax over the two options' weighted scores, along with an independent lapse probability.

**Rationale:** Following the critic's feedback, the weight calculation is simplified to use the raw validities directly (weights = validities). Subtracting 0.5 in the previous iteration inadvertently exaggerated the relative differences between cue weights (e.g., 0.9 vs 0.6 became a 4:1 ratio instead of 1.5:1), making the model too non-compensatory. Using the raw validities restricts the maximum weight ratio, naturally allowing multiple weaker cues to outvote a single strong cue. This softer weighting scheme better captures the strongly compensatory (Tallying-like) behavior seen in the human data while remaining within the prescribed WADD family.

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
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Retrieve validities and use them directly as weights
    val = np.asarray(parameters["validities"], dtype=float)
    weights = val
    
    # Compute weighted sum for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
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

### `pi_4` → slot 1 (via `new_theory`)

**Description:** People use a 'Top-K Tallying' strategy. Instead of relying entirely on a single best cue or integrating all available information, individuals save cognitive effort by tallying feature wins only among the top K most valid features. They ignore the least valid cues entirely. This boundedly rational approach captures the robust, compensatory nature of Tallying while acknowledging cognitive limitations in processing many cues.

**Rationale:** Following the critic's advice, I have shifted the parameter range for `k` from [3.0, 6.0] to [4.0, 8.0]. The previous Top-K Tallying model successfully improved the overall loss but still left a gap in capturing the strong human preference for exhaustive Tallying in Experiments 3 and 4. By increasing the average value of `k`, the model will almost always tally at least 4 cues, closely mimicking the overwhelming human preference for exhaustive Tallying in these specific tasks, while still retaining the Top-K cognitive bound.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `k`: `[4.0, 8.0]`
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
    k = int(round(float(parameters["k"]))) 
    
    # Ensure k is at least 1 and at most n_features
    n_features = len(val)
    k = max(1, min(k, n_features))
    
    # Identify the top K most valid cues
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally feature wins only among the top K cues
    a_wins = 0.0
    b_wins = 0.0
    for idx in top_k_cues:
        if a[idx] > b[idx]:
            a_wins += 1.0
        elif b[idx] > a[idx]:
            b_wins += 1.0
            
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
