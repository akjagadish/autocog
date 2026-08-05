# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive Model (WADD): People evaluate options by integrating all available features, weighting each feature by its given validity. Unlike Take The Best, WADD is fully compensatory, meaning multiple lower-validity cues can override a single higher-validity cue. A subjective transformation parameter (gamma) allows the model to scale the validities, seamlessly interpolating between equal-weighting (Tallying, gamma=0) and validity-sensitive compensatory evaluation.

**Rationale:** Following the critic's feedback, the mechanism of the Weighted Additive Model (WADD) is kept exactly the same, but the parameter range for `gamma` has been restricted to `[0.0, 2.0]`. The previous broad range `[0.0, 10.0]` allowed the optimizer to drift into large gamma values, where the model behaves non-compensatorily (like Take The Best). By restricting gamma to `[0.0, 2.0]`, the model is forced to explore the compensatory regime that the human data strongly supports, seamlessly capturing behavior between Tallying (gamma=0) and standard WADD (gamma=1).

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 2.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Subjective transformation of validities
    gamma = float(parameters["gamma"])
    w = val ** gamma
    
    # Compute weighted sums for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
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

**Description:** Take The Best (Lexicographic Heuristic)

**Rationale:** Take The Best (TTB) evaluates options lexicographically by consulting features in decreasing order of validity. The first feature that discriminates between the options determines the choice entirely, and all remaining lower-validity features are ignored. This embodies a fast-and-frugal heuristic that is strictly non-compensatory, contrasting sharply with exhaustive integration models like WADD or Tallying. The model uses a softmax over the binary TTB choice to account for decision noise, coupled with a lapse rate for random guessing.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected state to have shape (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    sorted_indices = np.argsort(-validities, kind="mergesort")
    
    scores = np.zeros(2)
    discriminated = False
    
    # Lexicographic search: stop at the first discriminating feature
    for f_idx in sorted_indices:
        if a[f_idx] != b[f_idx]:
            if a[f_idx] > b[f_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
            discriminated = True
            break
            
    if not discriminated:
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    logits = beta * scores
    logits -= np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # Mixture with uniform lapse
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
