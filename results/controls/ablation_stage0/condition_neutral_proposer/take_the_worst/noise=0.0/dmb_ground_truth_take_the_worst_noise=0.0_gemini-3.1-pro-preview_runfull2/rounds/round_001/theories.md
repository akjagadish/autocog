# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People make decisions between options by computing a fully compensatory weighted sum of their features, where each feature is weighted by its subjective or objective validity. This Weighted Additive (WADD) strategy contrasts with non-compensatory rules like Take The Best by allowing multiple weak cues to overcome a single strong cue, and contrasts with Tallying by taking cue validities into account. The resulting scores are transformed into choice probabilities via a softmax function with a lapse rate.

**Rationale:** Following the arbiter's recommendation, we implement the Weighted Additive (WADD) theory. WADD computes the overall score of each option by taking the sum of its feature values weighted by their respective validities. This provides a fully compensatory decision rule that incorporates validity information, determining whether subjects integrate validities in a compensatory manner rather than using a single-cue non-compensatory strategy (Take The Best) or ignoring validities entirely (Tallying).

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")
        
    a, b = stim[0], stim[1]
    
    # WADD computes the sum of feature values weighted by validities
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
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

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Top-N Tallying (Truncated Tallying): People simplify complex decisions by restricting their attention to only the most valid cues and ignoring the rest. Within this considered subset of top cues, they abandon complex compensatory weighting and instead simply tally the number of features favoring each option. If the tallies are tied, they guess. This bridges the gap between fully compensatory WADD and unweighted Tallying by selectively ignoring low-validity features to save cognitive effort, while still avoiding the cognitive cost of cardinal weighting among the considered features.

**Rationale:** Adjusted the parameter ranges based on the critic's latest feedback. Tightened `k_prop` to `[0.8, 1.0]` so the model drops at most one feature in 4- or 5-cue environments, further aligning with the strong human preference for broad feature integration in Experiments 3 and 4. Restricted `epsilon` to `[0.0, 0.1]` to reduce baseline noise and prevent predictions from regressing too much toward 0.5.

**Parameters:**
  - `beta`: `[1.0, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `k_prop`: `[0.8, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Top-N Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Determine k: the number of top features to consider
    k_prop = float(parameters["k_prop"])
    k = max(1, int(round(k_prop * n_features)))
    
    # Get indices of top k validities (descending order, stable tie-breaking)
    cue_order = np.argsort(-val, kind="stable").tolist()
    top_k_indices = cue_order[:k]
    
    a, b = stim[0], stim[1]
    
    # Tally only on the top k features
    a_top = a[top_k_indices]
    b_top = b[top_k_indices]
    
    a_wins = float(np.sum(a_top > b_top))
    b_wins = float(np.sum(b_top > a_top))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```
