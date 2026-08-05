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

**Description:** People evaluate options by computing a weighted sum of their feature values, where the weights correspond to the subjective or objective validities of the cues (Weighted Additive rule, WADD). Rather than relying on a single discriminating cue (like Take The Best) or ignoring validities entirely (like Tallying), this compensatory strategy allows multiple lower-validity cues to jointly outweigh a single high-validity cue. Choices are made probabilistically via a softmax over the options' weighted sums, along with a lapse rate to account for random errors.

**Rationale:** Following the arbiter's instructions, this model implements the Weighted Additive (WADD) heuristic. Unlike Take The Best (which relies on a single cue) or Tallying (which weights all cues equally), WADD multiplies each feature by its validity and sums them to evaluate each option. This compensatory mechanism is capable of explaining behavior where participants trade off multiple weak cues against a single strong cue, which is often necessary to explain intermediate choice patterns observed across different experiments.

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
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_4` → slot 1 (via `new_theory`)

**Description:** Decision makers adaptively switch between a non-compensatory one-reason heuristic (Take The Best) and a compensatory approach (Weighted Additive rule) across trials. Instead of a soft probabilistic mixture, individuals use a deterministic threshold on cue dispersion (the absolute difference in weighted cue values). If the dispersion exceeds a threshold, they fully commit to the compensatory rule; otherwise, they rely entirely on the one-reason heuristic. This hard-switching mechanism, combined with independent temperature scales for each strategy, captures the high choice determinism and divergence observed in human behavior.

**Rationale:** Following the critic's feedback from Iteration 6, the model replaces the soft linear probabilistic strategy mixture with a deterministic threshold-based switch. The parameters `wadd_prob` and `dispersion_weight` have been replaced by a single `dispersion_threshold` parameter. On any given trial, if the cue dispersion exceeds this threshold, the model fully commits to the WADD strategy; otherwise, it fully commits to TTB. This structural change ensures that simulated choices remain highly deterministic, which will increase the divergence metric to better match the more extreme values observed in the human data.

**Parameters:**
  - `beta_wadd`: `[0.1, 20.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `dispersion_threshold`: `[0.0, 10.0]`
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

    beta_wadd = float(parameters["beta_wadd"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    dispersion_threshold = float(parameters["dispersion_threshold"])

    # --- WADD Component ---
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)

    # --- TTB Component ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)

    # --- Adaptive Mixture based on Cue Dispersion ---
    dispersion = abs(score_a_wadd - score_b_wadd)
    trial_wadd_prob = 1.0 if dispersion >= dispersion_threshold else 0.0
    
    p_core = trial_wadd_prob * p_wadd + (1.0 - trial_wadd_prob) * p_ttb
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
