# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Rationale:** Following the arbiter's instructions, this model implements the Take The Best (TTB) heuristic. Instead of computing a compensatory score (like WADD) or an unweighted sum (like Tallying), TTB sequentially evaluates features in decreasing order of validity. It stops at the first cue that discriminates and bases the choice entirely on it. This explains the high variance in Experiment 1 because trials with identical tally differences can have their decisions driven by different cues depending on the specific feature patterns. It also explains the strong preference in Experiment 2 for the 'fewer but better' option, as the single highest-validity cue dominates the decision regardless of deficits on multiple lower-validity cues.

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

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
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

**Description:** Stochastic Take-The-Best (STTB): People use a non-compensatory, one-reason heuristic to compare options, but their search order is probabilistic rather than strictly deterministic. The probability of examining a cue next is determined by a softmax over the subjective validities of the remaining unexamined cues. The search stops at the first feature that discriminates between the two options, and the decision is based solely on that feature. If the selected feature ties, it is ignored and the search continues. If all features are exhausted without a discriminator, the decision maker guesses. This model interpolates between strict Take-The-Best (at high inverse temperature) and the Minimalist heuristic with random cue search (at zero inverse temperature).

**Rationale:** Following the critic's feedback, the upper bound of the softmax inverse temperature (beta) for cue selection is increased from 20.0 to 200.0. In the previous iteration, a beta of 20.0 was insufficient to produce the highly deterministic search orders required to capture the strong Take-The-Best compliance observed in Experiments 3 and 4, since validities often differ by only small amounts (e.g., 0.1). Expanding the upper bound allows the model to closely approximate strict deterministic TTB when fit to subjects who exhibit that behavior, while preserving the ability to model more stochastic search paths via lower beta values.

**Parameters:**
  - `beta`: `[0.0, 200.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("STTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def get_prob(available_cues):
        if len(available_cues) == 0:
            return np.array([0.5, 0.5])
        
        v = validities[available_cues]
        z = beta * v
        z = z - np.max(z)  # numerical stability
        p = np.exp(z)
        p = p / np.sum(p)
        
        ans = np.zeros(2)
        for i, cue_idx in enumerate(available_cues):
            if a[cue_idx] > b[cue_idx]:
                ans[0] += p[i]
            elif b[cue_idx] > a[cue_idx]:
                ans[1] += p[i]
            else:
                new_cues = [c for c in available_cues if c != cue_idx]
                ans += p[i] * get_prob(new_cues)
        return ans

    n_features = len(validities)
    p_core = get_prob(list(range(n_features)))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
