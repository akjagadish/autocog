# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** In complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. Thus, behavior resembles zero-intelligence random guessing.

**Rationale:** The arbiter recommended a Random Guessing baseline theory to capture the observed ~0.50 metrics and low variance across all experiments. The previous models (TTB, WADD, Tallying) assumed systematic integration of features and produced extreme predictions, heavily deviating from the ~0.50 observed performance. This model predicts a 50/50 probability distribution for all trials, ignoring stimulus features entirely, serving as the required baseline.

**Parameters:**
  (none)

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
    return int(np.argmax(probabilities))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Single Random Feature heuristic: On each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all. An epsilon parameter allows blending this strategy with pure random guessing.

**Rationale:** Following the arbiter's recommendation, this theory models subjects as extracting the absolute minimum amount of attribute-level information: picking a single random feature and choosing based on it, ignoring validities. Given that pure random guessing fits the aggregate data perfectly, this model includes an epsilon parameter to interpolate between the Single Random Feature strategy and pure noise, allowing us to capture any residual variance that might be explained by minimal feature extraction.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    epsilon = float(parameters["epsilon"])
    n_features = stim.shape[1]
    
    wins = (stim[0] > stim[1]).sum()
    losses = (stim[0] < stim[1]).sum()
    ties = (stim[0] == stim[1]).sum()
    
    p_A_core = (wins + 0.5 * ties) / n_features
    p_B_core = (losses + 0.5 * ties) / n_features
    
    probs = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * probs + epsilon * 0.5
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
