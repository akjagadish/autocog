# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** People make decisions by computing a weighted sum of the features for each option, but the weights are non-linear transformations of the provided cue validities. Specifically, subjects exponentiate the validities by a parameter gamma, which allows the strategy to continuously interpolate between Tallying (gamma = 0) and Take-The-Best (gamma -> infinity). The option with the higher total weighted score is preferred, subject to decision noise (softmax) and occasional lapses.

**Rationale:** Following the critic's advice, we adjust the prior range for `gamma` to `[-1.0, 2.0]` to allow the model to more easily access the Tallying regime (where `gamma` is near 0) and prevent the parameter space from being dominated by extreme TTB-like behavior. We also tighten `epsilon` to `[0.0, 0.2]` to reduce excessive baseline noise, enabling the model to better fit the empirical data which often strongly favors Tallying.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `gamma`: `[-1.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
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

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Decision-makers use a discrete mixture of two distinct heuristics: Take-The-Best (TTB) and Tallying. Rather than integrating all information via a continuous weighted sum, a subject either relies entirely on the single most valid discriminating cue (TTB) or ignores cue validities to simply count the number of winning features for each option (Tallying). The probability of using TTB versus Tallying is governed by a mixing parameter. Both heuristic outputs are softened by independent temperature parameters (beta_ttb and beta_tally) to account for decision noise and differing scales of score differences between the two strategies, preventing brittle deterministic predictions.

**Rationale:** Following the critic's advice, I introduced separate temperature parameters (`beta_ttb` and `beta_tally`) for the Take-The-Best and Tallying components. This allows the model to independently scale the certainty of each strategy during parameter fitting, preventing a shared temperature from forcing a compromise that washes out probabilities towards 0.5 due to the differing scales of score differences between the two heuristics.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
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
    
    p_ttb = float(parameters["p_ttb"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    # Sort cues by validity in descending order
    cue_order = np.argsort(val)[::-1]
    
    ttb_scores = np.array([0.0, 0.0])
    for idx in cue_order:
        if a[idx] > b[idx]:
            ttb_scores = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_scores = np.array([0.0, 1.0])
            break
            
    # Softmax over TTB scores
    z_ttb = beta_ttb * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    ttb_probs = e_ttb / np.sum(e_ttb)
            
    # --- Tallying Prediction ---
    # Count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    # Softmax over tally scores
    z = beta_tally * (scores - np.max(scores))
    e = np.exp(z)
    tally_probs = e / np.sum(e)
    
    # --- Mixture and Lapse ---
    # Mix the two strategies
    mixed_probs = p_ttb * ttb_probs + (1.0 - p_ttb) * tally_probs
    
    # Apply random lapse
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```
