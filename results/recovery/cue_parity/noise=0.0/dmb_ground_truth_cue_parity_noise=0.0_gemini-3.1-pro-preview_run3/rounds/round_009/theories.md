# Round 9 — Theories

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


### slot 2 — `pi_11` — KILLED ✗

**Description:** Decision-makers use a Sequential Evidence Accumulation strategy. They sample cues in descending order of validity and maintain a running sum of evidence (accumulating the differences between options). At each discriminating cue, there is a probabilistic decision to stop sampling and make a choice based on the accumulated evidence so far. A high stopping probability mimics Take The Best (TTB) by usually halting at the first discriminating cue, but the continuous probability of sampling further allows for occasional compensatory effects (Tallying-like behavior) when multiple lower-validity cues overturn the initial evidence.

**Rationale:** Following the arbiter's suggestion, this model instantiates a Sequential Evidence Accumulation process. Instead of forcing a discrete bimodal mixture of TTB and Tallying, it unifies them into a single sequential process. Cues are sampled in descending order of validity, and evidence is accumulated. At each discriminating cue, the decision-maker stops with probability `gamma`. If they stop early, the choice is driven by the strongest cues (mimicking TTB). If they continue, evidence accumulates unweighted, allowing multiple weaker cues to overturn the first cue (mimicking Tallying). This probabilistic stopping mechanism naturally accommodates the near-zero differences in flat-line experiments while allowing slight deviations for compensatory behavior.

**Parameters:**
  - `gamma`: `[0.1, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.special import expit
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sample cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    p_A_total = 0.0
    weight_remaining = 1.0
    
    for j in cue_order:
        delta = a[j] - b[j]
        E += delta
        # Only consider stopping if the cue discriminates
        if delta != 0:
            p_stop = gamma
            # Probability of choosing A if we stop here
            p_A_given_stop = expit(beta * E)
            
            p_A_total += weight_remaining * p_stop * p_A_given_stop
            weight_remaining *= (1.0 - p_stop)
            
    # If the decision-maker evaluates all cues without stopping, decide based on final evidence
    p_A_given_stop = expit(beta * E)
    p_A_total += weight_remaining * p_A_given_stop
    
    # Apply lapse rate
    p_A_final = (1.0 - epsilon) * p_A_total + epsilon * 0.5
    
    return np.array([p_A_final, 1.0 - p_A_final])
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

### `pi_12` → slot 2 (via `new_theory`)

**Description:** Strategy Mixture Model: Decision-makers are heterogeneous and probabilistically select between distinct decision strategies on each trial. The population predominantly relies on a non-compensatory heuristic (Take The Best) but occasionally employs a compensatory strategy (Weighted Additive). To ensure the compensatory strategy generalizes across contexts with varying numbers of cues, the cue validities are normalized before integration.

**Rationale:** Following the critic's advice, the WADD scores are normalized by dividing by the sum of validities (`np.sum(val)`). This ensures the input to the softmax is on a consistent scale across experiments with different numbers of cues, allowing `beta_wadd` to find a single optimal range. The `beta_wadd` range is set to `[1.0, 10.0]` to stabilize predictions while preserving the strong TTB majority (`p_ttb` in `[0.8, 1.0]`) that captures the baseline flatness.

**Parameters:**
  - `p_ttb`: `[0.8, 1.0]`
  - `beta_wadd`: `[1.0, 10.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    p_ttb = float(parameters["p_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    
    # Take The Best (TTB) evaluation
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = -1
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner == 0:
        ttb_probs = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        ttb_probs = np.array([0.0, 1.0])
    else:
        ttb_probs = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD) evaluation with normalization
    wadd_a = np.sum(val * a)
    wadd_b = np.sum(val * b)
    wadd_scores = np.array([wadd_a, wadd_b]) / np.sum(val)
    
    z = beta_wadd * wadd_scores
    z -= np.max(z)
    exp_z = np.exp(z)
    wadd_probs = exp_z / np.sum(exp_z)
    
    # Strategy mixture
    p_core = p_ttb * ttb_probs + (1.0 - p_ttb) * wadd_probs
    
    # Attentional lapses
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
