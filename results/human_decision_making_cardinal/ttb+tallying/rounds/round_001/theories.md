# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

**Rationale:** To address the model being too deterministic compared to human data, two minimal adjustments were made. First, the lower bound of `beta` was widened to 0.01 to allow for a flatter softmax (more noise). Second, the raw WADD scores were normalized by the sum of the validities before applying the softmax. This ensures that the scale of the score differences remains consistent across experiments with varying numbers of cues, allowing the single `beta` parameter to generalize better and capture the higher level of human noise observed.

**Parameters:**
  - `beta`: `[0.01, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_2` — KILLED ✗

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
    return int(np.argmax(probabilities))
```


## Replacement

### `pi_4` → slot 2 (via `new_theory`)

**Description:** Dynamic Strategy Selection (Threshold Model): Decision-makers dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy on a trial-by-trial basis. The choice of strategy depends on the maximum cardinal difference between the options across all features. If this difference exceeds a certain threshold, the decision-maker is more likely to use WADD to account for the large magnitude; otherwise, they default to the simpler Tallying heuristic. This is modeled as a probabilistic mixture where the weight of WADD is a soft step function (sigmoid) of the maximum feature difference.

**Rationale:** Following the critic's feedback, the fixed probabilistic mixture is replaced with a dynamic, stimulus-driven threshold mechanism. The model computes the maximum cardinal difference between options on any feature. A soft threshold (sigmoid) determines the mixture probability `w_wadd` for that trial based on this max difference. This allows the model to correctly favor Tallying when differences are small (like in Experiment 2) and switch to WADD when there are extreme cardinal differences (like in Experiment 3), addressing the specific failure modes of the previous fixed-mixture theory.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `threshold`: `[0.0, 10.0]`
  - `steepness`: `[0.1, 10.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD score computation
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a_wadd = np.sum(a * val) / sum_val
        score_b_wadd = np.sum(b * val) / sum_val
    else:
        score_a_wadd = np.sum(a)
        score_b_wadd = np.sum(b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # Tallying score computation
    score_a_tally = float(np.sum(a > b))
    score_b_tally = float(np.sum(b > a))
    scores_tally = np.array([score_a_tally, score_b_tally])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Dynamic strategy selection based on max cardinal difference
    max_diff = np.max(np.abs(a - b))
    w_wadd = 1.0 / (1.0 + np.exp(-steepness * (max_diff - threshold)))
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
