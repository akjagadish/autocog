# Round 16 — Theories

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


### slot 2 — `pi_18` — KILLED ✗

**Description:** Decision-makers probabilistically alternate between a non-compensatory strategy (Take The Best) and a compensatory strategy (Weighted Additive). To reflect the strong human tendency to rely on the single best cue in disagreement scenarios, Take The Best remains the dominant strategy (occurring at least 40% of the time). The secondary Weighted Additive strategy captures systematic compensatory shifts in a minority of trials. To ensure consistent sensitivity across experiments with varying numbers of cues and validity magnitudes, the WADD scores are normalized by the sum of validities, allowing the softmax temperature to operate on a consistent [0, 1] scale.

**Rationale:** Following the critic's advice, we normalize the WADD scores by the sum of the validities before applying the softmax. This ensures that the beta parameter operates on a standardized [0, 1] scale, preventing the WADD component from overreacting to raw sum differences in experiments with many cues (like Exps 11, 12, 16, 19, 21). We also widen the w_ttb range slightly to [0.4, 1.0] to allow a stronger compensatory influence where needed (e.g., Exp 30).

**Parameters:**
  - `w_ttb`: `[0.4, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # 1. Take The Best (TTB) prediction
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
        p_ttb = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # 2. Weighted Additive (WADD) prediction
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a = np.sum(val * a) / sum_val
        score_b = np.sum(val * b) / sum_val
    else:
        score_a = 0.0
        score_b = 0.0
        
    beta = float(parameters["beta"])
    
    z = beta * np.array([score_a, score_b])
    z -= np.max(z)  # Numerical stability
    p_wadd = np.exp(z) / np.sum(np.exp(z))
    
    # 3. Strategy mixture
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_wadd
    
    # 4. Add general lapse rate (epsilon)
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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

### `pi_19` → slot 2 (via `new_theory`)

**Description:** Validity-Threshold Model with WADD Fallback: Decision-makers process cues sequentially in order of validity, but their choice of strategy depends on the absolute strength of the best available evidence. If the first discriminating cue has a high validity (exceeding an internal threshold), the decision-maker relies purely on Take The Best (TTB), ignoring all other cues. If the top discriminating cue falls below this threshold, the decision-maker falls back to a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities.

**Rationale:** Following the latest critic feedback, we revert to the accepted iteration 2 mechanism (absolute validity threshold with WADD fallback) but widen the `validity_threshold` parameter range to the full `[0.0, 1.0]` interval. This prevents manually restricting the optimization space, allowing the model to naturally find the best balance between TTB and WADD across experiments. We also expand `beta_comp` to `[0.1, 10.0]` to ensure the compensatory branch can make sufficiently deterministic choices when triggered.

**Parameters:**
  - `validity_threshold`: `[0.0, 1.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_comp`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["validity_threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_comp = float(parameters["beta_comp"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    disc_cue_idx = -1
    for j in cue_order:
        if a[j] != b[j]:
            disc_cue_idx = j
            break
            
    if disc_cue_idx == -1:
        return np.array([0.5, 0.5])
        
    v_disc = val[disc_cue_idx]
    
    if v_disc >= threshold:
        # Take The Best: rely solely on the first discriminating cue
        scores = np.array([1.0, 0.0]) if a[disc_cue_idx] > b[disc_cue_idx] else np.array([0.0, 1.0])
        z = beta_ttb * scores
    else:
        # WADD: weighted sum of cues as a compensatory fallback
        wadd_a = float(np.sum(val * a))
        wadd_b = float(np.sum(val * b))
        scores = np.array([wadd_a, wadd_b])
        z = beta_comp * scores
        
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
