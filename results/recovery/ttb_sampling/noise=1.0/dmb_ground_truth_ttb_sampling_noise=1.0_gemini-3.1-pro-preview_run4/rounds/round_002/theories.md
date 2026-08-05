# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Random Guessing: Due to task complexity, lack of trial-by-trial correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial.

**Rationale:** Following the arbiter's feedback, the theory posits that subjects are guessing randomly. This is strongly supported by the experimental results: the real metric values across all four experiments are consistently around 0.5 (0.4844, 0.4800, 0.4954, 0.5046). Previous models (like TTB, WADD, and Tallying) predicted much more extreme values and failed to capture this central tendency. A pure random guessing model predicts exactly 0.5 for all metrics, perfectly aligning with the observed data.

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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Random Feature Heuristic (Minimalist): Subjects do not integrate multiple features or weight them by validity. Instead, on each trial, they randomly sample exactly one feature uniformly at random. They choose the option with the higher value on that sampled feature. If the options tie on that feature, they guess randomly. This acts as a highly noisy but boundedly rational alternative to pure guessing, predicting choices that slightly favor the option with more positive features overall. A lapse rate (epsilon) accounts for additional purely random choices.

**Rationale:** Following the arbiter's suggestion, this implements the Random Feature Heuristic. It assumes subjects are minimally engaged, selecting just one feature at random per trial and deciding based on it, ignoring validities entirely. This provides a mechanism that is very close to random guessing (since features often tie or conflict) but systematically deviates slightly in favor of options with more positive features, serving as a boundedly rational alternative to pure noise.

**Parameters:**
  - `epsilon`: `[0.0, 1.0]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # The marginal probability of choosing A when picking one feature at random:
    # P(A) = 1/n * sum(a_i > b_i) + 0.5/n * sum(a_i == b_i)
    # For binary features, this simplifies exactly to 0.5 + 0.5 * (sum(a) - sum(b)) / n
    diff = np.sum(a) - np.sum(b)
    p_a_core = 0.5 + 0.5 * diff / n_features
    p_b_core = 1.0 - p_a_core
    
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b_core + epsilon * 0.5
    
    return np.array([p_a, p_b])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
