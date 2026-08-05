# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People make choices by computing a weighted sum of the features for each option (the Weighted Additive, or WADD, heuristic). Unlike Take The Best or Tallying, WADD scales the contribution of each feature by its validity. Furthermore, features are processed as bipolar evidence: the presence of a feature adds to the option's value proportionally to its validity, while its absence actively penalizes the option. The resulting option scores are translated into choice probabilities via a softmax function, with an independent lapse rate accounting for random guessing.

**Rationale:** Following the latest feedback, we revert to the raw validities as weights but change the feature representation from {0, 1} to {-1, 1}. This ensures that the absence of a feature actively penalizes an option, fundamentally altering the score differences and the compensatory dynamics. This structural change aims to capture the human deviation from the reference probabilities in Experiment 2 while preserving the core WADD mechanism that succeeded in Experiment 1.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # WADD (Weighted Additive) heuristic.
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    val = np.asarray(parameters["validities"], dtype=float)
    
    # Center the stimulus features to -1 and 1 so absence of a feature penalizes
    stim_centered = stim * 2.0 - 1.0
    
    # Compute the weighted sum of features for each option.
    # The weights are directly proportional to the cue validities.
    scores = np.dot(stim_centered, val)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Decision-makers use the Take-The-Best (TTB) heuristic, searching through features in descending order of their validities. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the choice is a random guess. To account for behavioral noise without parameter redundancy, deviations from the deterministic TTB rule are modeled using a single lapse rate parameter (epsilon), replacing the redundant softmax temperature found in prior TTB instantiations.

**Rationale:** The arbiter requested replacing Tallying with the Take-The-Best (TTB) heuristic, a non-compensatory strategy that relies on a sequential search ordered by cue validity. While a prior TTB model existed (pi_1), it suffered from parameter redundancy by including both a softmax temperature (beta) and a lapse rate (epsilon) applied to a binary score, causing unidentifiability. This new instantiation implements the pure TTB mechanism as requested but streamlines the noise model to use only a single lapse rate parameter, ensuring more robust parameter estimation and better generalization.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_core = np.array([0.5, 0.5])
    for idx in cue_order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters['epsilon'])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Decision-makers use a Tallying (equal-weight) heuristic, ignoring the specific validities of different experts. They simply count the total number of positive features (endorsements) for each option. Choice probabilities are then generated by applying a softmax function over these unweighted tally scores, capturing the degree to which one option out-tallies the other, with an additional lapse rate to account for random guessing.

**Rationale:** Following the arbiter's feedback, this model implements a pure Tallying heuristic where subjects ignore cue validities and simply count the number of positive features for each option. The choice probabilities are derived via a softmax function over these unweighted tally scores, parameterized by an inverse temperature (beta) and a lapse rate (epsilon). This captures the hypothesis that subjects may resort to equal-weight integration when validities are ignored or too complex to apply.

**Parameters:**
  - `beta`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features
    scores = np.array([np.sum(a), np.sum(b)])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```
