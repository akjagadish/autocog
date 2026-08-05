# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

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


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Weighted Tallying Heuristic: People make decisions by integrating across multiple features, but rather than treating all cues equally (as in simple Tallying), they weight each feature-wise win by the explicitly provided expert validity. The score for each option is the sum of the validities for the features where that option has a strictly higher rating than the other. The option with the highest validity-weighted tally is chosen. This captures both the tendency to integrate across features and the sensitivity to explicit validities.

**Rationale:** Following the critic's feedback, we replace Take-The-Best with Weighted Tallying. TTB under-predicted the strong tallying behavior seen in Experiments 1 and 3 because it stopped at the first discriminating feature. Weighted Tallying resolves this by integrating across all features (explaining the tallying-like behavior) while still weighting the feature-wise wins by the explicitly provided expert validities, honoring the arbiter's original instruction to incorporate validities into the heuristic without falling back to a fully compensatory magnitude-based model like WADD.

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
        raise ValueError("Weighted Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sum validities for features where an option strictly wins
    score_a = np.sum(validities[a > b])
    score_b = np.sum(validities[b > a])
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
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
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Mixture of Tallying and Weighted Tallying: Decision makers do not universally ignore explicit validities (as in pure Tallying) nor do they perfectly weight every feature by its numerical validity (as in pure Weighted Tallying). Instead, the population consists of a mixture of strategies, or individuals use a blended strategy. Some individuals rely on a simple count of winning features (Tallying), while others incorporate the provided validities to weight those wins. By modeling choice as a convex combination of the Tallying score and the Weighted Tallying score, the model can capture intermediate levels of consensus and account for why human behavior often falls between the predictions of these two pure heuristics.

**Rationale:** Following the arbiter's suggestion, this theory models a mixture between pure Tallying and Weighted Tallying. Tallying completely ignores validities, resulting in overly extreme predictions on experiments where subjects do use validity information (e.g., Exp 6). Conversely, Weighted Tallying relies exclusively on numerical validities, which can underestimate the human tendency to simply count winning features. By introducing a subject-specific blending parameter `w_tally`, the model effectively forms a population-level mixture that smoothly interpolates between these two heuristics. This hybrid score mechanism should yield intermediate consensus levels, bringing predictions much closer to the observed human behavior across the experimental suite.

**Parameters:**
  - `w_tally`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying scores: simple count of strictly winning features
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    
    # Weighted Tallying scores: sum of validities for winning features
    wt_a = np.sum(validities[a > b])
    wt_b = np.sum(validities[b > a])
    
    # Blend the two strategies
    w = float(parameters["w_tally"])
    score_a = w * tally_a + (1.0 - w) * wt_a
    score_b = w * tally_b + (1.0 - w) * wt_b
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the blended scores
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
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
