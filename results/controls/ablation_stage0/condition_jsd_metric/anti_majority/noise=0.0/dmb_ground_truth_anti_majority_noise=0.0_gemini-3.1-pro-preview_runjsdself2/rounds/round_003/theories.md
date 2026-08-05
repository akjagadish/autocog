# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Decision makers use a Weighted Additive (WADD) strategy to evaluate options, integrating all available features. Instead of raw validities or linear shifts, they weight each feature by its log-odds, which is the mathematically principled way to linearly accumulate independent evidence (equivalent to Naive Bayes). The total score for each option is the sum of these log-odds weights for the features it possesses. The option with the higher total score is chosen probabilistically via a softmax function over the scores, subject to a baseline lapse rate.

**Rationale:** Following the critic's advice, we replace the linear shift (val - 0.5) with the theoretically principled log-odds transformation. This corresponds to the correct Bayesian method for linearly integrating independent cues, scaling the evidence more aggressively for highly predictive cues. This minimal edit keeps the theory firmly within the WADD compensatory family while addressing the under-prediction of variance in several experiments.

**Parameters:**
  - `beta`: `[0.1, 25.0]`
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
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds of validities to represent the true Bayesian weight of evidence.
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Decision makers evaluate options by sequentially processing features in descending order of their validity. They accumulate evidence in the form of log-odds weights for each feature. However, accumulation is not always exhaustive; it stops as soon as the absolute accumulated evidence exceeds a subjective threshold. A low threshold mimics a Take The Best heuristic (stopping at the first discriminating cue), while a high threshold mimics a Weighted Additive strategy (integrating all available cues).

**Rationale:** Applying the minimal-diff edit suggested by the critic: narrowing the threshold range to [0.0, 5.0] to provide higher resolution for the fitter, and widening the beta range to [0.1, 50.0] to allow for more deterministic choices. The mechanism remains exactly the same, as it was already correctly implementing Sequential Evidence Accumulation with Threshold.

**Parameters:**
  - `beta`: `[0.1, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Convert validities to log-odds weights (Bayesian evidence)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    threshold = float(parameters["threshold"])
    
    evidence = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            evidence += weights[j] * diff
            # Stop accumulating if evidence exceeds the subjective threshold
            if abs(evidence) >= threshold:
                break
                
    # The accumulated evidence represents the log-odds favoring option A over B
    scores = np.array([evidence, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
