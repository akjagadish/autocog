# Round 18 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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


### slot 2 — `pi_20` — KILLED ✗

**Description:** Valence-Inverted Compensatory Model: Decision-makers evaluate options using purely compensatory strategies (Weighted Additive or Tallying) but their interpretation of cue valence is subjective. Some individuals treat '1' as the favorable rating, while others treat '0' as the favorable rating. This subjective valence inversion leads to choices that systematically oppose standard models, particularly in environments where compensatory integration naturally conflicts with heuristic processing.

**Rationale:** Following the critic's advice, 'TTB' has been removed from the strategy parameter space, restricting the model to strictly compensatory integration (WADD and TALLY) as recommended by the arbiter. This prevents the fitting algorithm from erroneously attributing valence-inverted choice patterns to noisy TTB heuristics, ensuring the model fully leverages the valence-inverted compensatory mechanisms to capture the extreme negative metrics in Experiments 5, 17, and 35.

**Parameters:**
  - `beta`: `[0.0, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `strategy`: `{"WADD", "TALLY"}`
  - `valence`: `{"1", "-1"}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    strategy = parameters["strategy"]
    valence = float(parameters["valence"])
    
    if strategy == "WADD":
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
    else: # TALLY
        score_a = np.sum(a)
        score_b = np.sum(b)
        
    scores = np.array([score_a, score_b])
    
    z = beta * valence * scores
    z -= np.max(z)  # Numerical stability
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_21` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Model with Normalized Compensatory Evidence: The population consists of individuals who use different decision strategies. A proportion of the population uses a fast-and-frugal Take-The-Best (TTB) heuristic, stopping at the first discriminating cue and making a deterministic choice. The rest of the population uses a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities. To ensure the compensatory strategy is properly calibrated across different experimental environments, the WADD evidence is normalized by the total sum of validities, bounding it to [-1, 1] and allowing a single inverse temperature parameter to apply consistent stochasticity.

**Rationale:** Following the latest critic feedback, we reverted the rejected 3-way mixture to the previously accepted 2-way Strategy Mixture of TTB and WADD. To fix the calibration issues of the WADD component across different experiments, we normalized the WADD evidence by the sum of the validities. This ensures the pre-softmax WADD evidence is always strictly bounded between -1.0 and 1.0, preventing the WADD component from acting like a deterministic step function in experiments with large validity sums and allowing the 'beta' parameter to exert a consistent level of stochasticity.

**Parameters:**
  - `w_ttb`: `[0.0, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # TTB Probability
    ttb_prob = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_prob = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_prob = np.array([0.0, 1.0])
            break
            
    # WADD Probability
    evidence = np.sum(val * (a - b)) / np.sum(val)
    scores = np.array([evidence, -evidence]) / 2.0
    z = beta * scores
    z -= np.max(z)  # Numerical stability
    wadd_prob = np.exp(z) / np.sum(np.exp(z))
    
    # Mixture
    p_core = w_ttb * ttb_prob + (1.0 - w_ttb) * wadd_prob
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
