# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

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

### `pi_12` → slot 1 (via `new_theory`)

**Description:** Tallying with Strict Validity Tie-Breaking: Decision-makers primarily evaluate options by counting feature-wise wins (Tallying). To resolve ties in the tally counts, they employ a secondary, compensatory evaluation based on the explicitly provided validities of the winning features. Crucially, this secondary validity component is strictly bounded so that it can never override a strict tallying win, ensuring that the decision is lexicographic: Tallying always dominates, and validities only influence the choice when tally counts are exactly equal.

**Rationale:** Applying the critic's recommendation: To preserve the model's core Tallying dominance (which is critical for fitting Experiments 2 and 4), the validity component must be strictly constrained so it acts ONLY as a tie-breaker. By restricting the `w_val` parameter to strictly less than 1.0 ([0.0, 0.99]) and using normalized validities, the maximum possible difference contributed by the secondary component is strictly less than 1.0. This mathematical guarantee ensures that a tally difference of 1 or more will always determine the choice, while allowing validities to perfectly break exact tally ties, satisfying the Lexicographic Tallying requirement.

**Parameters:**
  - `validities`: `validities`
  - `w_val`: `[0.0, 0.99]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_val = float(parameters["w_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify winning features for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Primary component: Tallying (count of wins)
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    # Secondary component: Validity score of winning features
    # Normalize validities so the secondary component is strictly bounded
    norm_validities = validities / np.sum(validities)
    val_a = np.sum(norm_validities * a_wins)
    val_b = np.sum(norm_validities * b_wins)
    
    # Compute final scores
    # Because w_val < 1.0 and the max difference in val is <= 1.0,
    # the validity component can never exceed 1.0, preserving Tallying dominance.
    score_a = tally_a + w_val * val_a
    score_b = tally_b + w_val * val_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply random lapse rate
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
