# Round 10 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_12` — KILLED ✗

**Description:** Decision-makers employ a mixture of two distinct fast-and-frugal heuristics: Tallying and Take-The-Best (TTB). Tallying counts the number of features where an option strictly dominates the other, ignoring cue validities. TTB searches through features sequentially in descending order of validity, stopping at the first feature that discriminates between the options. Rather than integrating these into a single compensatory utility score, individuals apply these non-compensatory rules strictly. The population behavior is modeled by blending the deterministic probability vectors of these two heuristics via a mixing parameter, along with a lapse rate for random guessing.

**Rationale:** Previous models struggled to capture empirical behavior because they applied continuous, compensatory weighting (such as Softmax over tallies or non-linear WADD scores). This approach fails to capture the strict, non-compensatory nature of fast-and-frugal heuristics. Following the arbiter's guidance, this theory treats both Tallying and TTB as strict deterministic rules that output degenerate probability vectors (with uniform guessing on ties). By blending these predictions at the population level with a mixing parameter 'w_tally', the model naturally captures the 50/50 split in conflict trials (Exp 10) and TTB-like tie-breaking behavior (Exp 20) without over-smoothing the choices.

**Parameters:**
  - `w_tally`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 0.5]`
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
    
    w_tally = float(parameters["w_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying component (strict heuristic, no softmax)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_pred = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_pred = np.array([0.0, 1.0])
    else:
        tally_pred = np.array([0.5, 0.5])
        
    # Take-The-Best (TTB) component
    order = np.argsort(-val, kind="stable")
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Blend the deterministic predictions of the two heuristics
    p_core = w_tally * tally_pred + (1.0 - w_tally) * ttb_pred
    
    # Apply lapse rate for random guessing
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_13` → slot 1 (via `new_theory`)

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
