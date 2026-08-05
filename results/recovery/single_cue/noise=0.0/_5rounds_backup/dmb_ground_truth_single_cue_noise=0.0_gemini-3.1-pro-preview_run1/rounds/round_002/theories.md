# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Validity-Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying), ignoring the magnitudes of the differences and the exact cue validities. However, they are not completely blind to cue validities. Instead of integrating all validities in a compensatory manner (WADD) or strictly following a non-compensatory rule (Take The Best), they give a soft bonus weight to the single most predictive cue. This validity bias acts primarily as a tie-breaker when the tallies are close or equal, slightly skewing choice probabilities toward the option endorsed by the best cue, while still preserving the overall dominance of the Tallying heuristic. This explains why human choices overwhelmingly follow Tallying but show slight deviations toward the most valid cue.

**Rationale:** Following the critic's advice, the upper bound of the validity bonus weight 'w' has been restricted to 0.9 (from 2.0). This ensures that the bonus for winning the most predictive cue acts strictly as a tie-breaker or a very soft bias that cannot override a strict 1-point advantage in the tally count. This minimal edit forces the model to preserve Tallying dominance when there is a clear winner, correctly predicting the low match rates with WADD/TTB in Experiments 3 and 4 while still breaking ties using the best cue.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w`: `[0.0, 0.9]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Validity-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Bonus for winning the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    score_a = a_wins + w * a_best_win
    score_b = b_wins + w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
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

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Moderate-Cue Biased Tallying: People primarily compare options by counting the number of features where one option strictly dominates the other (Tallying). However, rather than giving a bonus to the most valid cue, subjects exhibit an aversion to options that rely too heavily on a single extreme cue at the expense of moderate cues. When tallies are close or tied, this manifests as a penalty to the option endorsed by the most valid cue, reflecting a preference for a broader cluster of supporting evidence over a single strong piece of evidence. This preserves the dominance of Tallying for clear differences but explains the systematic anti-top-cue preference observed in tie-breaking scenarios.

**Rationale:** The arbiter feedback highlights that standard Tallying overestimates the choice probability in Experiment 2 and completely misses the severe anti-top-cue preference in Experiment 6 (where subjects actively choose against the top cue when tallies are tied). To capture this, I introduced a 'Moderate-Cue Bias' via a penalty parameter `w` applied to the option that wins on the most extreme (highest validity) cue. Because `w` is bounded between 0 and 1, Tallying remains the dominant choice rule when one option clearly wins on count (e.g., Exp 1). However, when tallies are tied (as in Exp 6), the penalty forces the model to prefer the option supported by a cluster of moderate cues. Additionally, when the tally difference is small (e.g., 1), the penalty reduces the confidence in the Tallying winner, successfully lowering the predicted match rate in Experiment 2 from ~0.87 down closer to the observed 0.76.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Moderate-Cue Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Penalty for relying on the most valid cue
    val = np.asarray(parameters["validities"], dtype=float)
    best_cue = np.argmax(val)
    
    w = float(parameters["w"])
    a_best_win = float(a[best_cue] > b[best_cue])
    b_best_win = float(b[best_cue] > a[best_cue])
    
    # Subtract the penalty weight from the tally score
    score_a = a_wins - w * a_best_win
    score_b = b_wins - w * b_best_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
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
    return np.random.choice(len(probabilities), p=probabilities)
```
