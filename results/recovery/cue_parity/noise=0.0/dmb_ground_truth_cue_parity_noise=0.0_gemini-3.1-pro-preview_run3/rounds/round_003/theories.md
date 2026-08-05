# Round 3 — Theories

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** Decision-makers evaluate options using Probabilistic Cue Integration, computing a weighted additive sum of the features. Instead of relying on a single discriminating cue (as in Take The Best) or ignoring validities entirely (as in Tallying), individuals weight each cue based on a non-linear transformation of its validity, specifically the log-odds (evidence weight). A power parameter on the log-odds allows the decision-maker to tune the integration, smoothly interpolating between near-tallying and highly non-compensatory behavior. This yields a graded, validity-sensitive compensatory mechanism where multiple weak cues can potentially outweigh a strong cue.

**Rationale:** Following the arbiter's feedback, this model replaces the discrete strategy mixture with a continuous Probabilistic Cue Integration approach. The weights are derived from the log-odds of the cue validities, representing the Bayesian evidence each cue provides. A scaling parameter 'gamma' modulates these log-odds, allowing the model to capture both highly compensatory (Tallying-like) and highly non-compensatory (TTB-like) behaviors within a single unified weighting mechanism. This graded sensitivity to validity avoids the rigid boundaries of discrete strategy selection while still predicting nuanced deviations from pure TTB.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.1, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Clip validities to avoid extreme values or division by zero in log-odds
    val_clipped = np.clip(val, 0.5001, 0.9999)
    
    # Transform validities into evidence weights via log-odds
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    # Apply a non-linear scaling parameter to capture individual differences in validity spread
    gamma = float(parameters["gamma"])
    weights = log_odds ** gamma
    
    # Integrate cues
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Convert integrated evidence to choice probabilities via softmax
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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Decision-makers rely on a Strategy Mixture of Take The Best (TTB) and a Weighted Additive (WADD) strategy. The probability of selecting a non-compensatory strategy (TTB) over a compensatory one (WADD) is sensitive to the global cue environment. Specifically, the mixing weight is determined by a logistic function of the 'validity gap'—the difference between the highest validity and the average of the remaining validities. By widening the sensitivity of this logistic transition, decision-makers can exhibit sharp, near-deterministic shifts between strategies depending on whether the best cue stands out significantly from the rest.

**Rationale:** Following the critic's advice, we return to the successful global validity gap mixture approach (Iteration 3 base) but replace Tallying with a Weighted Additive (WADD) strategy to provide a more nuanced compensatory baseline. Furthermore, we widen the range for w_ttb_slope to [-50.0, 50.0] to allow the model to learn a near-deterministic phase transition between TTB and WADD across different environments, helping it hit the extreme behavioral shifts observed in Experiments 3 and 4.

**Parameters:**
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `w_ttb_base`: `[-10.0, 10.0]`
  - `w_ttb_slope`: `[-50.0, 50.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) Strategy
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner == 0:
        ttb_scores = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        ttb_scores = np.array([0.0, 1.0])
    else:
        ttb_scores = np.array([0.5, 0.5])
        
    # WADD Strategy (Compensatory)
    wadd_scores = np.array([np.sum(val * a), np.sum(val * b)])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Environment-sensitive w_ttb based on validity gap
    w_ttb_base = float(parameters["w_ttb_base"])
    w_ttb_slope = float(parameters["w_ttb_slope"])
    
    val_sorted = np.sort(val)[::-1]
    if len(val_sorted) > 1:
        env_metric = float(val_sorted[0] - np.mean(val_sorted[1:]))
    else:
        env_metric = 0.0
        
    # Smooth logistic sigmoid for mixing weight
    z_w = w_ttb_base + w_ttb_slope * env_metric
    z_w = np.clip(z_w, -20.0, 20.0) # Prevent overflow
    w_ttb = 1.0 / (1.0 + np.exp(-z_w))
    
    epsilon = float(parameters["epsilon"])
    
    # TTB choice probabilities
    z_ttb = beta_ttb * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # WADD choice probabilities
    z_wadd = beta_wadd * (wadd_scores - np.max(wadd_scores))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # Apply lapse rate
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
