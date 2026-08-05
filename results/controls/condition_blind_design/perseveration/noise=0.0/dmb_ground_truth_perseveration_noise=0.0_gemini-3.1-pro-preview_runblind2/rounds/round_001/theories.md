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

**Description:** Weighted Additive (WADD) Theory: People evaluate options by computing a weighted sum of their features, where the weights correspond to the cue validities above chance (validity - 0.5). This linear compensatory strategy allows multiple lower-validity cues to outweigh a single high-validity cue without over-privileging the top cue. Choices are made probabilistically using a softmax over the weighted sums, combined with a lapse rate for random guessing.

**Rationale:** Following the critic's feedback, the weight calculation was changed from log-odds to a linear scaling `val - 0.5`. The log-odds transformation created too large a gap between high and low validity cues, making the model behave too much like Take The Best and over-predicting the dominance of the highest-validity cue. By scaling weights as evidence above chance, the model allows lower-validity cues to better compete, bringing the choice probabilities closer to the observed human indifference levels when cues conflict.

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
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Use linear evidence above chance instead of log-odds to prevent over-dominance of the top cue
    weights = val - 0.5
    
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
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

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Strategy Mixture Theory: The population of decision-makers is heterogeneous, consisting of distinct subgroups that rely on fundamentally different decision rules. Rather than every individual using a noisy version of the same heuristic, about half the population employs a non-compensatory, frugal strategy (Take The Best), while the other half employs a compensatory, unweighted strategy (Tallying). This individual difference accounts for both the ~50% aggregate choice proportions on critical divergence trials and the high (~0.25) between-subject variance, which is characteristic of a Bernoulli distribution of highly deterministic strategies.

**Rationale:** Following the critic's advice, the compensatory strategy is swapped from WADD to Tallying. This directly addresses the mismatch in Experiment 1, which isolates divergence between TTB and Tallying. A 50/50 mixture of TTB and Tallying will naturally predict the ~0.46 match rate observed. Furthermore, the parameter ranges for beta and epsilon are shifted to [5.0, 50.0] and [0.0, 0.1] respectively, making the individual strategies much more deterministic. This reduction in within-subject noise will elevate the between-subject variance on critical divergence trials to match the empirical ~0.25 variance, capturing the signature of a population split between distinct deterministic rules.

**Parameters:**
  - `strategy`: `{0, 1}`
  - `beta`: `[5.0, 50.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    strategy = int(parameters["strategy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if strategy == 0:
        # Take The Best (Non-compensatory)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner is None:
            scores = np.array([0.5, 0.5])
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
            
    else:
        # Tallying (Compensatory)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
