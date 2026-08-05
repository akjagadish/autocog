# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) heuristic: Decision-makers evaluate options using a non-compensatory, rank-based approach. They first rank the available features by their validity (descending). They then sequentially compare the options on these features, stopping at the first feature that discriminates between them (i.e., one option has a higher value than the other). The option favored by this discriminating feature is chosen, and all remaining features are completely ignored. If no features discriminate between the options, the decision-maker resorts to a random guess.

**Rationale:** Following the arbiter's suggestion, we implement the Take-The-Best (TTB) heuristic. This model sorts features based on their provided validities and evaluates them one by one. The first feature that differentiates the two options determines the choice 100%, ignoring all subsequent features. This is a purely non-compensatory strategy, contrasting strongly with the exhaustive integration of Weighted Additive (WADD) and the equal-weight exhaustive summation of Tallying. Randomness is introduced solely via an epsilon lapse rate, reflecting execution noise or inattention, and uniform guessing is used if no features discriminate.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # Using mergesort for a stable sort in case of ties.
    order = np.argsort(-validities, kind='mergesort')
    
    a, b = stim[0], stim[1]
    
    # Default to guessing if no feature discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequentially check features according to their validity ranking
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend the deterministic choice (or guess) with the lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_2` — KILLED ✗

**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `weights`: `[(0.0, 1.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Probabilistic Lexicographic (ProbLex) Heuristic: Decision-makers evaluate options by sequentially checking features in descending order of their validity, similar to Take-The-Best. However, at each discriminating feature, there is a probability (gamma) that the decision-maker skips the cue or makes an execution error, failing to terminate the search. If a cue is skipped, they move to the next most valid feature. If all discriminating features are skipped or no features discriminate, they resort to guessing. This captures both the rank-based nature of human decision-making and the stochasticity of cue evaluation.

**Rationale:** Following the arbiter's suggestion, we replace WADD with a Probabilistic Lexicographic (ProbLex) model. ProbLex is a non-compensatory heuristic similar to Take-The-Best (TTB), where cues are checked in descending order of validity. However, it introduces a probability `gamma` of skipping a discriminating cue. This provides a more competitive heuristic benchmark against TTB, as it captures the sequential nature of decision-making while allowing for stochasticity in cue evaluation. Unlike a simple lapse rate which falls back to pure random guessing, skipping a cue in ProbLex allows the model to fall back on lower-validity cues, potentially capturing human errors or incomplete cue processing more accurately.

**Parameters:**
  - `gamma`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("ProbLex expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='mergesort')
    a, b = stim[0], stim[1]
    
    p_a = 0.0
    p_b = 0.0
    p_reach = 1.0
    
    # Sequentially check features according to their validity ranking
    for idx in order:
        if a[idx] > b[idx]:
            p_a += p_reach * (1.0 - gamma)
            p_reach *= gamma
        elif b[idx] > a[idx]:
            p_b += p_reach * (1.0 - gamma)
            p_reach *= gamma
            
    # If all cues are skipped or none discriminate, guess.
    p_a += p_reach * 0.5
    p_b += p_reach * 0.5
    
    p_core = np.array([p_a, p_b])
    
    # Blend the deterministic choice (or guess) with the lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
