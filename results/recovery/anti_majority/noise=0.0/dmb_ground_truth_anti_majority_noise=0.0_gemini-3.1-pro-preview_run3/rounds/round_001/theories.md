# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

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


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** Weighted Additive (WADD) theory with Non-Linear Cue Scaling posits that decision makers compute a total score for each option by summing the validities of the features it possesses, but they may non-linearly amplify or dampen the differences between cue validities. By exponentiating cue validities with a free parameter, the model can naturally interpolate between Tallying (where all cues are weighted equally) and Take The Best (where the highest-validity cue dominates).

**Rationale:** Following the critic's diagnosis, the previous non-linear WADD model still behaved too much like Tallying because the upper bound of the gamma parameter (5.0) was not high enough to allow the highest-validity cue to fully dominate multiple lower-validity cues. By expanding the gamma range to [1.0, 20.0], the model is granted the capacity to heavily amplify the primary cue, shifting predictions closer to the empirical Take The Best-like behavior observed in humans.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[1.0, 20.0]`
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
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly to allow amplification of the best cues
    val_scaled = val ** gamma
    
    # Compute WADD scores: sum of scaled validities for features possessed by the option
    score_a = np.sum(val_scaled * a)
    score_b = np.sum(val_scaled * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Rationale:** Following the critic's advice, the minimal edit adjusts the `p_ttb` range to `[0.35, 0.95]` to perfectly split the difference between the previous two iterations. This should balance the slight over- and under-predictions of TTB consistency across the four experiments while keeping epsilon noise low.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `p_ttb`: `[0.35, 0.95]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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
