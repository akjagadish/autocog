# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Rationale:** Applying the minimal diff requested by the critic: narrowing the range of the `gamma` parameter from `[0.0, 10.0]` to `[0.0, 2.0]`. This restricts the model from selecting overly high inverse temperatures, forcing it to distribute its sampling more evenly across cues and bringing the choice proportions closer to the empirical ~0.50 mark on incongruent trials.

**Parameters:**
  - `gamma`: `[0.0, 2.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
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

**Description:** Stochastic Take-The-Best (Lexicographic): Decision-makers search through features sequentially in descending order of their objective validities. The process is stochastic: at each step, the decision-maker may prematurely terminate the search and guess. If they evaluate a cue and it discriminates between the options, they choose the superior option with high probability, but may choose the inferior option due to execution error or misperception. If the cue does not discriminate, the search continues to the next most valid cue. This algorithmic mechanism generates high stochasticity and intermediate choice proportions without relying on compensatory weighted sums.

**Rationale:** Following the arbiter's feedback, this model replaces the compensatory WADD approach with a Stochastic Take-The-Best (lexicographic) heuristic. It searches cues in descending order of validity. To capture the high stochasticity and intermediate choice proportions observed in the data, it introduces two forms of noise: a probability of prematurely stopping the search to guess (`p_stop`), and an execution error when reading a discriminating cue (`p_error`). This provides a non-compensatory, sequential-search alternative to Theory 1 while still mathematically smoothing the predictions to match empirical variances.

**Parameters:**
  - `p_stop`: `[0.0, 1.0]`
  - `p_error`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    p_stop = float(parameters["p_stop"])
    p_error = float(parameters["p_error"])
    
    # Sort features by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(validities, kind='mergesort')[::-1]
    
    p_A = 0.0
    p_B = 0.0
    p_guess = 0.0
    
    reach_prob = 1.0
    
    for idx in order:
        # At each step, there is a probability of prematurely terminating the search and guessing
        prob_stop_here = reach_prob * p_stop
        p_guess += prob_stop_here
        reach_prob -= prob_stop_here
        
        a_val = stim[0, idx]
        b_val = stim[1, idx]
        
        if a_val > b_val:
            # Cue discriminates in favor of A
            p_A += reach_prob * (1.0 - p_error)
            p_B += reach_prob * p_error
            reach_prob = 0.0
            break
        elif b_val > a_val:
            # Cue discriminates in favor of B
            p_B += reach_prob * (1.0 - p_error)
            p_A += reach_prob * p_error
            reach_prob = 0.0
            break
        else:
            # Cue does not discriminate (tie), continue search
            pass
            
    # If search completes and no cue discriminated, guess
    p_guess += reach_prob
    
    # Distribute guessing probability equally
    p_A += p_guess * 0.5
    p_B += p_guess * 0.5
    
    return np.array([p_A, p_B])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
