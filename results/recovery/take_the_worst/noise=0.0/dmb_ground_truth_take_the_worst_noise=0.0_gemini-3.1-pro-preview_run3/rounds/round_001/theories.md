# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) Decision Making with Option Bias

**Rationale:** Reverting to the pure WADD mechanism from Iteration 3, we introduce an `option_bias` parameter to account for spatial or order biases (e.g., a baseline preference for Option A). This linear shift breaks the trade-off between Experiment 1 and 2 by adjusting baseline choice probabilities independently of the score differences, avoiding the distortion of cue validities that caused previous failures.

**Parameters:**
  - `beta`: `[0.01, 2.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `option_bias`: `[-1.0, 1.0]`
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
    
    # Calculate the weighted sum of features for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    # Apply option bias to Option A's score
    option_bias = float(parameters.get("option_bias", 0.0))
    scores = np.array([score_a + option_bias, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
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

**Description:** Strategy Selection (Mixture of Deterministic Tallying and Probabilistic WADD): Decision-makers probabilistically select between a frugal, unweighted strategy (Tallying) and a fully compensatory, validity-weighted strategy (WADD). Critically, Tallying operates as a deterministic rule (choosing the option with more winning cues, or guessing on ties) rather than a probabilistic score-based process. This breaks the assumption that Tallying consistency scales with the absolute difference in cue counts, allowing the model to capture high consistency in scenarios with small cue count differences (e.g., Exp 4) and lower consistency in scenarios with large cue count differences (e.g., Exp 2). The WADD strategy remains probabilistic and tempers the extremeness of the Tallying predictions.

**Rationale:** Following the latest feedback, applying a softmax to Tallying scores incorrectly forces choice probabilities to scale with the magnitude of the cue count difference, contradicting the empirical data where subjects are more consistent in Exp 4 (small difference) than Exp 2 (large difference). To fix this, Tallying is now modeled as a strictly deterministic rule (1.0 for the winner, 0.5 for ties) rather than a probabilistic score. The `beta` parameter and softmax are applied ONLY to the WADD strategy scores. This minimal diff preserves the mixture architecture and parameter bounds of the accepted Iteration 1 base, while removing the problematic dependence on tally difference magnitudes.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_tally`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # 1. Deterministic Tallying strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # 2. WADD strategy scores
    val = np.asarray(parameters["validities"], dtype=float)
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores_wadd = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_tally = float(parameters["w_tally"])
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of the two strategies
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
