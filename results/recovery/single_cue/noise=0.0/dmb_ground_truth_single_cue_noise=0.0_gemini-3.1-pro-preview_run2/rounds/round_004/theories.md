# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Normalized Weighted Additive (WADD) Model: Decision-makers evaluate options by computing a weighted sum of their features. To maintain a consistent scale for choice stochasticity across different environments, the subjective weights are normalized to sum to 1. This compensatory mechanism allows multiple lower-validity cues to correctly outweigh a single high-validity cue. The final choice is made via a softmax function over the normalized expected values, mixed with a uniform lapse rate to account for random errors.

**Rationale:** Following the critic's feedback, the validities are now L1-normalized so they sum to 1. This minor adjustment ensures that the weighted sums are strictly bounded between 0 and 1 across all experiments, regardless of the number of features or the raw sum of validities. Consequently, the `beta` parameter has a consistent scale to operate on, allowing the model to better fit human stochasticity across diverse experimental designs. The `beta` range has also been widened to [0.1, 50.0] to accommodate sharper deterministic choices when needed.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")

    val = np.asarray(parameters["validities"], dtype=float)
    # Normalize validities to sum to 1 to bound scores between 0 and 1
    weights = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option using normalized validities
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
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

**Description:** Decision-makers in binary choice tasks employ a mixture of strategies, primarily relying on the compensatory Tallying heuristic, with a smaller fraction using the non-compensatory Take-The-Best (TTB) heuristic. TTB processes cues lexicographically, searching through features in descending order of their validities and stopping at the first feature that discriminates between the options. Tallying counts the total number of winning features for each option regardless of validity. By skewing the population mixture heavily toward Tallying, the model captures the dominant compensatory behavior observed in human data while retaining enough lexicographic influence to explain subtle choice variances.

**Rationale:** Following the critic's feedback, the range for the mixture weight `w_ttb` was narrowed from [0.0, 1.0] to [0.0, 0.3]. This minimal edit ensures that the population is heavily skewed toward Tallying, which correctly captures the strong compensatory behavior seen in Experiments 2, 3, 4, 9, and 10, while still allowing a minority TTB influence to soften predictions relative to a pure Tallying model.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_ttb`: `[0.0, 0.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    
    # Take-The-Best (TTB) component
    order = np.argsort(-val, kind="stable")
    ttb_score = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    z = beta * (score_tally - np.max(score_tally))
    e = np.exp(z)
    p_tally = e / np.sum(e)
    
    # Mixture of TTB and Tallying
    p_core = w_ttb * ttb_score + (1.0 - w_ttb) * p_tally
    
    n_opts = p_core.shape[0]
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
