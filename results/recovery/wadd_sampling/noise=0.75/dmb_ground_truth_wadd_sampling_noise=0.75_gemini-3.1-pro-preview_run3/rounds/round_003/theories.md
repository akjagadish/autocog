# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Take-The-Best (TTB) heuristic: People make decisions by considering cues sequentially in descending order of their validity. They stop at the first cue that discriminates between the two options and choose the option favored by that cue, completely ignoring all remaining (less valid) cues. If no cues discriminate, they guess. Response noise is captured through a constant lapse rate.

**Rationale:** Following the arbiter's feedback, this implements the Take-The-Best (TTB) heuristic, which offers a strict non-compensatory baseline. Unlike Tallying (which gives equal weight to all discriminating cues) and WADD (which is fully compensatory), TTB respects cue ordering but completely ignores all cues after the first discriminating one. This captures decision-making strategies where the single most reliable piece of evidence dictates the choice.

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
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity descending
    order = np.argsort(val)[::-1]
    
    chosen = None
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
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


### slot 2 — `pi_2_1` — SURVIVED ✓

**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** Shifted `beta` to a much lower range [0.0, 1.0] and increased `epsilon` to [0.4, 0.9] as requested by the arbiter. This drastically softens the deterministic tallying choices, yielding choice probabilities much closer to 0.5 (e.g., around 0.55 for the tally winner). This naturally lifts the TTB-aligned metrics in experiments where Tallying and TTB disagree from ~0.15 up to the observed ~0.45, while preserving the exact 0.5 predictions on tied-count trials.

**Parameters:**
  - `beta`: `[0.0, 1.0]`
  - `epsilon`: `[0.4, 0.9]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** People make decisions by computing a weighted sum of feature differences, where the weights are proportional to the log-odds of the cue validities. A parameter gamma interpolates between equal weighting (Tallying, gamma=0) and full log-odds weighting (gamma=1). This Compensatory Weighted Additive (WADD) strategy allows multiple lower-validity cues to outweigh a single high-validity cue, while capturing slight validity-driven effects that pure Tallying misses. Response noise is modeled via a softmax function and an independent lapse rate.

**Rationale:** Following the arbiter's feedback, this model replaces strict Take-The-Best with a Weighted Additive (WADD) theory. It computes a weighted sum of strict feature differences, where the weights are derived from the log-odds of the cue validities. A `gamma` parameter allows the model to smoothly interpolate between pure Tallying (gamma=0, where all weights are 1) and full log-odds WADD (gamma=1). This preserves the strong baseline fit of Tallying while allowing for the subtle validity-driven effects observed in human data.

**Parameters:**
  - `beta`: `[0.0, 1.0]`
  - `epsilon`: `[0.4, 0.9]`
  - `gamma`: `[0.0, 1.0]`
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
    gamma = float(parameters["gamma"])
    
    # Calculate log-odds of validities
    clipped_val = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(clipped_val / (1.0 - clipped_val))
    
    # Apply gamma to scale the log-odds (gamma=0 -> equal weights/Tallying)
    weights = np.maximum(log_odds, 1e-9) ** gamma
    
    # Compute weighted sum of feature differences (strict wins)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```
