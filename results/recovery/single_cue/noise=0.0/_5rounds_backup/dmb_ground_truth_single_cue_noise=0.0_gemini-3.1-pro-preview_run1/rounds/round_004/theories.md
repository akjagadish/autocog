# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Recency-Biased Tallying: People primarily evaluate options by counting the number of features where one option strictly dominates the other (Tallying). However, rather than giving a bonus or penalty based on cue validity, subjects exhibit a structural recency effect. They give a secondary tie-breaking bonus to the option that is superior on the most recently evaluated feature (the last cue). This preserves the dominance of Tallying for clear differences while elegantly explaining why tie-breaking appears at chance-level relative to the most valid (first) cue in most experiments, yet systematically favors the option winning the final cue when specifically tested.

**Rationale:** Following the arbiter's feedback, this model replaces the validity-based penalty with a structural tie-breaker: a recency effect. By assigning a small bonus to the option that wins on the last feature, the model naturally produces chance-level tie-breaking against the most valid (first) cue in Experiments 5, 7, and 8, because the winner of the last cue is orthogonal to the winner of the first cue in those designs. Critically, it accurately captures the strong anti-top-cue deviation observed in Experiment 6, where the experimental design pitted the first cue directly against the last cue during tied tallies.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Recency-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Recency bonus (winning the last feature)
    w = float(parameters["w"])
    a_last_win = float(a[-1] > b[-1])
    b_last_win = float(b[-1] > a[-1])
    
    score_a = a_wins + w * a_last_win
    score_b = b_wins + w * b_last_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Leaky Sequential Tallying: People compare options by processing features sequentially from first to last, maintaining a running accumulator of feature-wise wins. However, memory for earlier features 'leaks' or decays over time. This results in an exponential decay of feature weights based on how long ago they were processed, naturally yielding a recency effect where the most recently evaluated features (the last ones) have the strongest impact on the final decision.

**Rationale:** Based on the arbiter's feedback, we replace the flat tallying model with a 'Leaky Sequential Tallying' theory. Instead of a hardcoded, ad-hoc bonus for the final feature, this model assumes subjects process features sequentially and maintain a running accumulator that 'leaks' over time. The effective weight of each feature difference decays exponentially based on how long ago it was processed (controlled by the retention parameter `gamma`). This naturally yields a recency effect where the last feature has the strongest impact, providing a more principled cognitive mechanism for the recency bias observed in the data.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Leaky Sequential Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate exponential decay weights: gamma^(n - 1 - i)
    # The last feature (i = n - 1) gets weight gamma^0 = 1
    # The first feature (i = 0) gets weight gamma^(n - 1)
    weights = np.array([gamma**(n_features - 1 - i) for i in range(n_features)])
    
    # Feature-wise strict wins
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Accumulate with decaying weights
    score_a = np.sum(weights * a_wins)
    score_b = np.sum(weights * b_wins)
    scores = np.array([score_a, score_b])
    
    # Softmax choice with max-subtraction for numerical stability
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
