# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Tallying with Take-The-Best Tiebreaker: Decision makers primarily use a simple Tallying heuristic, counting the number of features that favor each option. If one option has a higher tally, it is chosen. However, if the tallies are tied, they do not simply guess; instead, they fall back to the Take-The-Best (TTB) heuristic, breaking the tie by choosing the option favored by the single most valid differentiating feature. This tie-breaking influence can be parameterized to allow for both positive reinforcement or penalty depending on the specific cue structures.

**Rationale:** Following the critic's advice, we expanded the range of the tie-breaker weight 'tau' to include negative values ([-2.0, 2.0]). Positive values for tau failed to capture the large metric difference in Experiment 6 (0.71), which requires the model to be highly confident in strict tally wins while systematically avoiding the TTB prediction on tied trials. Permitting a negative tau gives the optimizer the flexibility to invert the tie-breaker's direction if that best explains the empirical behavior on tied trials, without altering the primary tallying mechanism.

**Parameters:**
  - `beta`: `[0.1, 15.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `tau`: `[-2.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary heuristic: Tallying (counting strict wins)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Tie-breaker: Take-The-Best
    if score_a == score_b:
        tau = float(parameters["tau"])
        val = np.asarray(parameters["validities"], dtype=float)
        # Sort indices by descending validity
        order = np.argsort(-val, kind="stable")
        for idx in order:
            if a[idx] > b[idx]:
                score_a += tau
                break
            elif b[idx] > a[idx]:
                score_b += tau
                break
                
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


### slot 2 — `pi_2` — KILLED ✗

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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Weighted Additive (WADD) with Subjective Weighting: Decision-makers evaluate options using a fully compensatory strategy where each option's value is the weighted sum of its features. Instead of using the raw validities as weights, they apply a subjective transformation modeled as a power law (validity^gamma). This parameterization allows the model to capture a spectrum of strategies: when gamma is 0, it reduces to Equal-Weight (Tallying); when gamma is 1, it is standard WADD. By constraining gamma to be very small, the model predominantly relies on tallying-like behavior while allowing slight compensatory deviations to capture nuances in specific experiments.

**Rationale:** Following the critic's advice, I further reduced the upper bound of the `gamma` parameter from 2.0 to 0.8. This tighter bound forces the subjective weights to remain very close to 1.0 (pure Tallying), which is necessary to capture the overwhelming human preference for equal weighting observed in Experiments 3 and 4. At the same time, it still allows the slight compensatory deviations required to explain the tie-breaking behavior in Experiments 7 and 8.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `gamma`: `[0.0, 0.8]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights via a power law
    weights = val ** gamma
    
    # Calculate the weighted sum for each option
    scores = np.sum(stim * weights, axis=1)
    
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```
