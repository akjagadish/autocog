# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People make choices between options by computing a weighted sum of their features (Weighted Additive rule, WADD). To avoid over-relying on the single best cue, decision-makers use compressed or flattened weights derived from cue validities. A scaling exponent allows the weights to be flatter than log-odds, enabling a coalition of lesser cues to correctly outweigh the most valid cue, bridging the gap between exact validity weighting and Tallying.

**Rationale:** Following the critic's advice, the upper bound for the scaling exponent `gamma` was further reduced to 1.0. This guarantees that cue weights are at most linear in validities and often flatter, ensuring that coalitions of lesser cues robustly overpower the most valid cue, pushing the TTB match rate in Experiment 1 closer to the observed 0.16. To counteract the resulting over-prediction of Tallying in Experiment 2 (0.95 vs real 0.89), the upper bound for `epsilon` was increased to 0.4 and `beta`'s upper bound was reduced to 20.0, injecting just enough noise to soften the match rate.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.4]`
  - `gamma`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Convert validities to weights using a scaling exponent gamma
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted additive scores
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
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

**Description:** Skeptical Tallying: Decision-makers primarily rely on a Tallying heuristic, counting the number of features where one option strictly dominates the other. However, when the tallies are tied (or closely matched), they do not simply guess. Instead, they exhibit skepticism toward the highest-validity cues—perhaps viewing them as redundant, overly salient, or manipulated—and systematically break ties by favoring options that possess more lower-validity features. This is modeled by augmenting the tally score with a secondary component that weights features inversely to their stated validity.

**Rationale:** The metrics from the experiments strongly indicate that subjects primarily follow Tallying (Exp 2 accuracy ~89%), but systematically violate WADD and Take-The-Best. Specifically, Exp 3 and Exp 4 show that when Tallying is tied or conflicts with WADD, subjects actively choose the option that WADD predicts against (~12-14% WADD alignment, meaning ~86-88% alignment with the opposite). This suggests subjects are breaking ties by favoring lower-validity features over higher-validity ones. The proposed 'Skeptical Tallying' theory instantiates this by computing the standard Tallying score and adding a secondary score that weights features by (1 - validity). This ensures that in the event of a tally tie, the option with more low-validity features wins, cleanly explaining the systematic anti-WADD behavior in tied scenarios.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Skeptical Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (count of strict feature-wise wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Secondary mechanism: Tie-breaking favoring lower-validity features
    # Features are weighted by (1 - validity) so that lower validity cues provide a larger bonus.
    tie_breaker_a = np.sum(a * (1.0 - val))
    tie_breaker_b = np.sum(b * (1.0 - val))
    
    # Combine tally with the tie-breaker.
    # gamma controls the strength of the tie-breaker relative to a strict tally win.
    score_a = a_wins + gamma * tie_breaker_a
    score_b = b_wins + gamma * tie_breaker_b
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
