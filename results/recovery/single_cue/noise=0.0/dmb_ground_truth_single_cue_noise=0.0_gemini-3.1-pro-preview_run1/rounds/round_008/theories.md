# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_10` — KILLED ✗

**Description:** Decision-makers use a 'Take-The-Best (TTB) with Thresholded Compensatory Tallying' strategy. They initially attempt to use a lexicographic TTB approach, favoring the option that wins on the first discriminating feature. However, they evaluate the remaining features: if the number of opposing wins on subsequent features reaches or exceeds a subject-specific threshold, they abandon TTB and switch to a compensatory Tallying strategy. When Tallying results in a tie, a secondary recency (anti-primacy) bias acts as a tie-breaker, which is explicitly constrained to only apply when the primary tally counts are exactly equal, preventing it from overriding strict tallying wins.

**Rationale:** Following the critic's feedback, the recency tie-breaker has been explicitly constrained to only alter the tally scores when `a_wins == b_wins`. This prevents the tie-breaker from overriding a strict 1-win difference in Tallying, which had ruined the model's performance on pure Tallying tasks (Exps 2 and 4) in Iteration 4. With this structural safeguard in place, the `w_recency` range has been safely expanded to [-5.0, 5.0] to ensure it can act as a sufficiently strong deterministic tie-breaker when needed. The `theta` range is kept at [0.0, 3.0] as in the successful Iteration 3 base, maintaining the model's sensitivity to switch to Tallying when opposing evidence exists.

**Parameters:**
  - `theta`: `[0.0, 3.0]`
  - `w_recency`: `[-5.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    theta = float(parameters["theta"])
    w_recency = float(parameters["w_recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # 1. Identify TTB winner and the first discriminating cue
    ttb_winner = None
    k = -1
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_winner = 0
            k = i
            break
        elif b[i] > a[i]:
            ttb_winner = 1
            k = i
            break
            
    if ttb_winner is None:
        return np.array([0.5, 0.5])
        
    # 2. Evaluate remaining features
    if ttb_winner == 0:
        opposing_wins = np.sum(b[k+1:] > a[k+1:])
    else:
        opposing_wins = np.sum(a[k+1:] > b[k+1:])
        
    diff_val = float(opposing_wins)
    
    # 3. Calculate Tallying scores with Recency tie-breaker
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    weights = np.arange(1, n_features + 1, dtype=float)
    weights /= np.sum(weights)
    recency_a = np.sum(a * weights)
    recency_b = np.sum(b * weights)
    
    # Explicitly restrict tie-breaker to only apply when tally counts are equal
    if a_wins == b_wins:
        tally_scores = np.array([
            a_wins + w_recency * recency_a,
            b_wins + w_recency * recency_b
        ])
    else:
        tally_scores = np.array([a_wins, b_wins])
    
    # 4. Apply Threshold Logic
    if diff_val >= theta:
        scores = tally_scores
    else:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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

### `pi_11` → slot 1 (via `new_theory`)

**Description:** Soft Weighted Additive (Soft WADD) Model with Convex Mixture: Decision-makers evaluate options by computing a continuous weighted sum of features. The weight of each feature is a convex combination of a baseline uniform weight (representing a default Tallying tendency) and the explicitly provided expert validities, plus a position-dependent bias (capturing primacy or recency effects). This continuous, bounded integration allows the model to act predominantly like Tallying when validities are similar or the mixture weight is low, while smoothly transitioning to more compensatory or validity-driven behavior without causing the overall scores to explode.

**Rationale:** Following the critic's advice, we replaced the unbounded linear addition of validities with a convex combination between a uniform weight of 1.0 (Tallying) and the feature validities, controlled by a new parameter w_mix in [0.0, 1.0]. This provides a mathematically stable way for the model to smoothly transition from pure Tallying to pure WADD without causing the absolute magnitude of the scores to explode, avoiding the issues seen in previous iterations while preserving the position bias.

**Parameters:**
  - `validities`: `validities`
  - `w_mix`: `[0.0, 1.0]`
  - `w_pos`: `[-5.0, 5.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_mix = float(parameters["w_mix"])
    w_pos = float(parameters["w_pos"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize position to [0, 1] where 0 is the first feature and 1 is the last
    if n_features > 1:
        pos = np.arange(n_features, dtype=float) / (n_features - 1)
    else:
        pos = np.zeros(n_features, dtype=float)
        
    # Feature weights: Convex combination of Tallying (1.0) and validities + position bias
    w = (1.0 - w_mix) * 1.0 + w_mix * validities + w_pos * pos
    
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
