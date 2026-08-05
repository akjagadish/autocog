# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Weighted Additive (WADD) posits that decision-makers integrate all available information by computing a global value for each option. This value is the sum of the option's cue values, each weighted by its respective cue validity. Unlike non-compensatory heuristics like Take The Best (which relies on a single discriminating cue) or Tallying (which weights all cues equally), WADD represents a fully compensatory process where multiple weaker cues can outweigh a single stronger cue. Choice probabilities are generated via a softmax function over the computed values, allowing for noise in the decision process.

**Rationale:** Following the arbiter's feedback, this theory implements the Weighted Additive (WADD) model. It calculates an overall score for each option by taking the sum of its cue values weighted by the corresponding cue validities. This provides a fully compensatory mechanism, which contrasts with the non-compensatory TTB model and the unweighted Tallying model. By incorporating cue validities directly into a global integration process, WADD captures human sensitivity to differential cue validities while allowing multiple weaker cues to compensate for a single strong cue, which TTB systematically fails to do.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD integrates all cues weighted by their validities.
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability.
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
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

**Description:** Take-K Heuristic: Decision-makers sort cues by validity but only tally the wins on the top K most valid cues, ignoring the rest. This blends the frugality of Take-The-Best with the compensatory accumulation of Tallying.

**Rationale:** Following the critic's feedback, we constrain the noise parameters further to capture the extreme choice proportions seen in the human data. `beta` is restricted to `[2.0, 20.0]` to make the softmax more deterministic, and `epsilon` is restricted to `[0.0, 0.1]` to reduce random guessing. Finally, `k_prop` is adjusted to `[0.85, 1.0]` to ensure that the model retains enough cues even in experiments with fewer features, maintaining strong compensatory behavior.

**Parameters:**
  - `beta`: `[2.0, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `k_prop`: `[0.85, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # K is the number of top cues to consider, determined by k_prop
    k_prop = float(parameters["k_prop"])
    k = max(1, int(np.round(k_prop * n_features)))
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    top_cues = cue_order[:k]
    
    a_top = a[top_cues]
    b_top = b[top_cues]
    
    # Tally strict wins on the top K cues
    a_wins = float(np.sum(a_top > b_top))
    b_wins = float(np.sum(b_top > a_top))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
