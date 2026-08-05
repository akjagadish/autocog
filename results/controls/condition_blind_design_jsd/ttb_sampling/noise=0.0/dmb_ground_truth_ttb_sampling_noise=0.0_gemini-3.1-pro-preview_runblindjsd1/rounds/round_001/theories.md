# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Take-The-Best (TTB) is a non-compensatory fast-and-frugal heuristic. People rank features by their subjective validity and compare options sequentially. The decision is based entirely on the first feature that discriminates between the options, ignoring all remaining features. If all features tie, the decision maker guesses randomly. Response noise is captured by a uniform lapse rate.

**Rationale:** Implements the Take-The-Best heuristic as requested by the arbiter. It provides a purely non-compensatory benchmark by ranking cues according to their validity and terminating the search at the first cue that strictly discriminates between the options. A lapse rate parameter 'epsilon' is added to account for response noise.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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

**Description:** Two-Stage Heuristic (Hybrid TTB-Tallying over Remaining Cues): Decision makers use a bounded sequential search, initially relying on the most valid cues to discriminate between options (Take-The-Best). If the top 'k' cues fail to discriminate (i.e., they are tied), the decision maker falls back to a computationally simpler 'Tallying' strategy. However, instead of tallying all cues, they only tally the remaining unexamined cues, avoiding double-counting the cues that already tied. This provides a psychologically plausible sequential search process that blends lexicographic and tallying strategies.

**Rationale:** Following the critic's advice, we refine the fallback stage of the successful Two-Stage Hybrid base. Instead of tallying all cues when the top cues tie, the model now tallies only the remaining unexamined cues (those not part of the `max_cues` searched during the TTB stage). This avoids double-counting the cues that already tied and provides a more psychologically plausible sequential search process while preserving the core mechanism that worked well in Iteration 1.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`
  - `max_cues`: `{1, 2, 3, 4}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    k = int(parameters["max_cues"])
    
    decided = False
    p_core = np.array([0.5, 0.5])
    
    # Stage 1: Take-The-Best on the top k cues
    for idx in order[:k]:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            decided = True
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            decided = True
            break
            
    # Stage 2: Fallback to Tallying on remaining unexamined cues if top cues tie
    if not decided:
        remaining_order = order[k:]
        if len(remaining_order) > 0:
            a_rem = a[remaining_order]
            b_rem = b[remaining_order]
            a_wins = float(np.sum(a_rem > b_rem))
            b_wins = float(np.sum(b_rem > a_rem))
            if a_wins > b_wins:
                p_core = np.array([1.0, 0.0])
            elif b_wins > a_wins:
                p_core = np.array([0.0, 1.0])
            else:
                p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
