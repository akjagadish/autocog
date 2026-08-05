# Round 11 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_13` — KILLED ✗

**Description:** Weighted Additive (WADD) Theory with Normalized Linear Interpolation: Decision-makers evaluate options by computing a compensatory utility for each option. This utility is the dot product of the option's feature vector and a subjectively transformed set of cue validities. To decouple subjective weighting from the choice temperature, the validities and uniform weights (Tallying) are both normalized to sum to 1 before being linearly mixed by an individual-specific parameter (alpha). Choices are then made probabilistically using a softmax function over these bounded utilities.

**Rationale:** Applied the critic's suggestion to normalize both the uniform weights and the provided validities to sum to 1 before interpolating with `alpha`. This ensures that the resulting weights always sum to 1, keeping the compensatory utilities on a stable, bounded scale. This perfectly decouples the subjective weighting parameter from the softmax temperature `beta`, allowing the optimizer to capture exact weighting regimes without confounding the overall magnitude of utilities.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `alpha`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    # Normalize validities and uniform weights to sum to 1
    val_norm = val / np.sum(val)
    uniform_weight = 1.0 / len(val)
    
    # Linear interpolation between normalized uniform weighting and normalized validity weighting
    w = (1.0 - alpha) * uniform_weight + alpha * val_norm
    
    # Compute compensatory utility as the dot product of features and subjective validities
    u_a = np.dot(a, w)
    u_b = np.dot(b, w)
    
    scores = np.array([u_a, u_b])
    
    # Softmax choice rule with max-subtraction for numerical stability
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

### `pi_14` → slot 1 (via `new_theory`)

**Description:** Strict Tallying with Probabilistic Take-The-Best (TTB) Tie-Breaker: Decision-makers primarily rely on a simple tally of winning features to evaluate options, choosing the option with the most winning features via a softmax choice rule. When the tallies are strictly tied, they bypass the tallying confidence mechanism entirely and fall back on a lexicographic Take-The-Best strategy. To capture the empirical finding that subjects are much less deterministic on tie trials, the tie-breaker is applied directly in probability space, allowing for a high rate of random guessing independent of the determinism on clear tally wins.

**Rationale:** Following the critic's advice, we decoupled the confidence of the tie-breaker from the confidence of the Tallying mechanism. Instead of passing the scaled TTB scores through the same softmax function (which caused overconfidence on tie trials due to the high `beta` needed for non-tie trials), we now compute the tie-breaker directly in probability space. We replaced `w_ttb` with a probability parameter `p_ttb` in [0.0, 1.0]. When there is a strict tally tie, the model bypasses the softmax and assigns a probability of `0.5 + p_ttb / 2` to the TTB winner. This allows the model to capture the near-random guessing behavior observed empirically on tie trials without compromising its deterministic Tallying predictions on non-tie trials.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `p_ttb`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(-val, kind="stable")
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    diff = a_wins - b_wins
    
    # If the tally difference is non-zero, strictly use Tallying with Softmax
    if diff != 0:
        score_a = float(np.sign(diff))
        score_b = float(-np.sign(diff))
        scores = np.array([score_a, score_b])
        
        # Softmax choice rule
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
    else:
        # Otherwise, fall back to Take-The-Best tie-breaker directly in probability space
        ttb_winner = 0
        for idx in order:
            if a[idx] > b[idx]:
                ttb_winner = 1
                break
            elif b[idx] > a[idx]:
                ttb_winner = -1
                break
                
        if ttb_winner == 1:
            p_core = np.array([0.5 + p_ttb / 2.0, 0.5 - p_ttb / 2.0])
        elif ttb_winner == -1:
            p_core = np.array([0.5 - p_ttb / 2.0, 0.5 + p_ttb / 2.0])
        else:
            p_core = np.array([0.5, 0.5])
    
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
