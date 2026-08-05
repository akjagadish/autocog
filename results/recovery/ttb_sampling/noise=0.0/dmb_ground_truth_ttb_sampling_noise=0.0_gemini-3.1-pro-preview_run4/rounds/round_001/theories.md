# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

**Rationale:** Following the arbiter's recommendation, this model implements the 'Take-The-Best' (TTB) heuristic. Tallying and WADD failed to capture the empirical data because subjects strongly favor the option that wins on the highest-validity cue, even if the alternative has a greater total number of positive features. By searching cues in descending order of validity and stopping at the first discriminating cue, TTB perfectly captures the non-compensatory nature of human choice in these experiments. Noise is modeled simply as a lapse rate (epsilon) as requested, replacing the softmax mechanism.

**Parameters:**
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
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

**Description:** People's decision making in multi-attribute choice is best described by a Strategy Mixture model. While decision-makers predominantly rely on the fast and frugal 'Take-The-Best' (TTB) heuristic—searching through cues in order of validity and stopping at the first discriminating cue—they occasionally switch to a compensatory 'Tallying' strategy (equal-weight counting of winning features). This mixture allows the model to capture both the high overall success of non-compensatory heuristics and the systematic, compensatory-like deviations observed in specific trial types where multiple lower-validity cues contradict a single high-validity cue. Response noise is incorporated as a uniform lapse rate.

**Rationale:** Following the critic's advice, I reduced the parameter ranges for `p_tally` (from [0.0, 0.5] to [0.0, 0.2]) and `epsilon` (from [0.0, 0.5] to [0.0, 0.1]). This minimal edit ensures that the model reflects Take-The-Best as the overwhelmingly dominant strategy, while still allowing for rare lapses and occasional use of Tallying. This tightens the quantitative fit to human data across experiments by preventing the model from overpredicting compensatory responses.

**Parameters:**
  - `p_tally`: `[0.0, 0.2]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Strategy Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take-The-Best (TTB) Prediction
    order = np.argsort(-validities)
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # 2. Tallying Prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # Blend strategies and add uniform lapse
    p_mix = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    p_core = (1.0 - p_mix) * p_ttb + p_mix * p_tally
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
