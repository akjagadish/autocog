# Round 11 — Theories

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


### slot 2 — `pi_13` — KILLED ✗

**Description:** Take The Best with Confirmatory Search: Decision-makers initially anchor their choice on the first discriminating cue (Take The Best). However, rather than stopping search entirely, they perform a confirmatory check of the remaining lower-validity cues. If the subsequent cues present strong contradictory evidence (the net number of opposing cues exceeds a specific threshold), their confidence is slightly undermined, leading to a small reduction in the probability of choosing the TTB winner rather than a complete strategy shift. Otherwise, they stick with the initial TTB choice with high confidence.

**Rationale:** Following the critic's feedback, the penalty for finding contradictory evidence in the confirmatory search phase has been softened. The parameter `shift_prob` was replaced with `contradiction_confidence` ranging from [0.5, 1.0], and `threshold` was shifted higher to {2, 3, 4, 5, 6}. This ensures that when subjects encounter strong opposing lower-validity cues, they merely experience a minor drop in confidence rather than a hard flip or random guessing. This minimal edit allows the model to capture the overwhelming adherence to TTB observed in the empirical data, while still allowing for slight modulations in confidence.

**Parameters:**
  - `epsilon`: `[0.0, 0.2]`
  - `threshold`: `{2, 3, 4, 5, 6}`
  - `contradiction_confidence`: `[0.5, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    epsilon = float(parameters["epsilon"])
    threshold = int(float(parameters["threshold"]))
    contradiction_confidence = float(parameters["contradiction_confidence"])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = -1
    ttb_idx = -1
    for i, j in enumerate(cue_order):
        if a[j] > b[j]:
            ttb_winner = 0
            ttb_idx = i
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            ttb_idx = i
            break
            
    if ttb_winner == -1:
        return np.array([0.5, 0.5])
        
    # Confirmatory search in the remaining cues
    net_contradiction = 0
    for j in cue_order[ttb_idx+1:]:
        if ttb_winner == 0:
            if b[j] > a[j]:
                net_contradiction += 1
            elif a[j] > b[j]:
                net_contradiction -= 1
        else:
            if a[j] > b[j]:
                net_contradiction += 1
            elif b[j] > a[j]:
                net_contradiction -= 1
                
    if net_contradiction >= threshold:
        p_winner = contradiction_confidence
    else:
        p_winner = 1.0 - epsilon
        
    p = np.zeros(2)
    p[ttb_winner] = p_winner
    p[1 - ttb_winner] = 1.0 - p_winner
    
    return p
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

### `pi_14` → slot 2 (via `new_theory`)

**Description:** Decision-makers rely on a pure Tallying strategy, a non-weighted compensatory heuristic. They evaluate options by simply counting the total number of positive cues for each option, completely ignoring the validities of those cues. Choices are made probabilistically based on the difference in these unweighted tally scores, subject to decision noise (inverse temperature) and random lapses.

**Rationale:** Following the arbiter's feedback, this proposes a pure Tallying (unit-weight linear) model. By completely ignoring cue validities and simply summing the positive features for each option, it provides a stark, fundamentally different compensatory baseline to contrast with the lexicographic Take-The-Best model. This will help identify variance from subjects using non-weighted, purely compensatory strategies.

**Parameters:**
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Pure tallying: count total positive cues for each option
    tally_a = float(np.sum(a))
    tally_b = float(np.sum(b))
    
    scores = np.array([tally_a, tally_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * scores
    z -= np.max(z)
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
