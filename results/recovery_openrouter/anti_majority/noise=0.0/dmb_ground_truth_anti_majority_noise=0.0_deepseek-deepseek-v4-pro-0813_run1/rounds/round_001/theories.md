# Round 1 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** People execute a display-order Take The Best rule: they scan the physical feature vector left-to-right and stop at the first feature that discriminates between the two options. That first displayed discriminator determines the choice, and later features are ignored. If no feature discriminates, the person guesses. Stated validities do not reorder the scan, but they weakly modulate salience: a first discriminator with lower stated validity is used less decisively than one with higher stated validity. Response noise enters through softmax with inverse temperature beta plus a small lapse probability epsilon.

**Rationale:** The previous validity-based TTB theory predicts the wrong sign because it consults the high-validity cue before the displayed leftmost discriminator. The new theory keeps one-cue stopping but replaces validity-based cue order with display-based cue order. This produces a negative contrast in Experiment 1 because feature 0 is tied in the relevant pairs and feature 1, the first displayed discriminator, favors the option opposite the high-validity cue. It likewise produces a negative first-validity-cue contrast in Experiment 2 before response noise. The small beta/lambda ranges center the model near the observed noise levels, while validity is used only as a weak salience modulator rather than as a cue-ordering principle.

**Parameters:**
  - `beta`: `[0.75, 0.85]`
  - `lambda`: `[6.0, 8.0]`
  - `epsilon`: `[0.0, 0.03]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Display-order TTB expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    n_features = stim.shape[1]
    a = stim[0]
    b = stim[1]

    # Optional validities; they affect decisiveness, not cue order.
    if 'validities' in parameters and parameters['validities'] is not None:
        val = np.asarray(parameters['validities'], dtype=float)
        if val.shape[0] != n_features:
            raise ValueError(
                f'validities length {val.shape[0]} != n_features {n_features}.'
            )
    else:
        val = np.ones(n_features, dtype=float)

    # Scan features in physical display order: 0, 1, 2, ..., n-1.
    winner = None
    discriminator = None
    for j in range(n_features):
        if a[j] > b[j]:
            winner = 0
            discriminator = j
            break
        if b[j] > a[j]:
            winner = 1
            discriminator = j
            break

    # Complete tie: no discriminator, so guess.
    if winner is None:
        return np.ones(2, dtype=float) / 2.0

    beta = float(parameters['beta'])
    lam = float(parameters['lambda'])
    epsilon = float(parameters['epsilon'])

    # The first displayed discriminator is the only cue used for the choice.
    # Its stated validity weakly scales the effective inverse temperature.
    v_j = float(val[discriminator])
    salience = 1.0 + lam * max(0.0, v_j - 0.5)
    beta_eff = beta * salience

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax.
    z = beta_eff * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts, dtype=float) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_2` — KILLED ✗

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

### `pi_4` → slot 2 (via `new_theory`)

**Description:** People scan the displayed features from left to right and accumulate a signed preference signal, without ever committing to a hard lexicographic stop. Each discriminating feature contributes an increment whose weight is a continuous function of serial position, stated validity, and list length. For short lists the position weight is a steep exponential primacy gradient, so the first discriminator dominates probabilistically rather than absolutely. As list length exceeds working-memory capacity the gradient collapses toward a flat weighting function, making the rule behave like a tallying heuristic on long lists. The accumulated score is passed through a softmax with inverse temperature beta, and an independent lapse probability permits occasional intention-independent errors.

**Rationale:** The model replaces pi_2's equal-weight tallying with a capacity-sensitive order-weighted accumulator. On five-feature problems the exponential position gradient is sharp, so the first displayed discriminator exerts partial rather than absolute control: this produces the reversed high-validity-cue effect in Experiment 1, the negative validity-order first-cue effect in Experiment 2, and the intermediate first-displayed-cue agreement in Experiment 3. Because there is no hard stop, later cues still leak into the score, which is why Experiment 3 agreement is around 0.62 rather than the near-ceiling value produced by the hard display-order rule pi_3. On eight-feature problems the capacity-sensitive factor flattens both the position gradient and validity modulation, so tally ties have nearly canceling evidence and choice is near chance, matching the near-zero directional effect in Experiment 4 far better than pi_3's strict first-cue rule.

**Parameters:**
  - `lambda_5`: `[0.692, 0.694]`
  - `capacity_power`: `[14.0, 18.0]`
  - `validity_power`: `[0.049, 0.051]`
  - `beta`: `[4.51, 4.53]`
  - `epsilon`: `[0.095, 0.105]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Accept either a dict with option ratings or a (2, n_features) array.
    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Capacity-sensitive evidence accumulation expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Stated validities weakly modulate feature weight.
    if 'validities' in parameters and parameters['validities'] is not None:
        validities = np.asarray(parameters['validities'], dtype=float)
        if validities.shape[0] != n_features:
            validities = np.ones(n_features, dtype=float)
    else:
        validities = np.ones(n_features, dtype=float)

    lambda_5 = float(parameters['lambda_5'])
    capacity_power = float(parameters['capacity_power'])
    validity_power = float(parameters['validity_power'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Capacity-sensitive flattening. For n_features = 5 this factor is 1,
    # preserving a strong exponential primacy gradient. For longer lists it
    # shrinks rapidly, approximating equal tallying weights.
    capacity_factor = (5.0 / n_features) ** capacity_power

    lam = lambda_5 * capacity_factor
    nu = validity_power * capacity_factor

    positions = np.arange(n_features, dtype=float)
    weights = np.exp(-lam * positions) * (validities ** nu)

    diff = a - b
    score_a = float(np.sum(weights[diff > 0]))
    score_b = float(np.sum(weights[diff < 0]))
    relative_score = score_a - score_b

    # Stable softmax over the accumulated relative evidence.
    logits = np.array([beta * relative_score, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - epsilon) * core_probs + epsilon * np.full(2, 0.5, dtype=float)
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
```
