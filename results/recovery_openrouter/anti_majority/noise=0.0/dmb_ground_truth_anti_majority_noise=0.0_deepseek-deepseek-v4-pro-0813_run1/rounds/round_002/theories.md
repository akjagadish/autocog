# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

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


### slot 2 — `pi_4` — SURVIVED ✓

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


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** People make binary choices by stochastically sampling a bounded, capacity-scaled subset of discriminating cues and accumulating signed evidence. Cue sampling is governed jointly by a moderate serial-position primacy gradient that flattens on longer lists and by stated cue validity, whose influence on evidence is compressive rather than all-or-none. The first discriminating cue receives a serial-position-sensitive bonus: early positions carry a calibrated primacy boost, while later first-cues receive only a modest residual bonus, so behavior is neither deterministic first-cue commitment nor a pure equal-weight tally. On longer lists, some subjects carry partially anti-cue weights, producing subject-level reversals of first-cue effects. Accumulated evidence passes through a noisy softmax with lapse, preserving between-subject heterogeneity.

**Rationale:** This is a minimal-diff edit of the accepted iter-4 bounded-sampling accumulator. Three targeted changes address the remaining failures without revisiting the rejected iter-5/6/7 paths. First, the first-cue bonus now carries a mild serial-position decay after position 1, exp(-first_position_decay * max(0, first_disc - 1)). On five-item lists this fixes the persistent Experiment 5 sign error by making position-0 first cues more consistent than later position-2/3 cues, while leaving the well-calibrated position-0/1 primacy in Experiments 1-3 roughly intact. The base first-cue bonus is raised to [1.65, 1.95], so later-position effective bonuses remain near the critic's recommended 1.0-1.5 range while early-position primacy strengthens enough to move Experiments 1 and 2 toward their negative serial-position effects. Second, long-list cue weights are translated downward in a variance-preserving way: for n_features > 5 the existing clip is followed by subtracting a per-subject long_jitter_shift of about 0.78-0.98, giving a slightly negative mean cue weight around -0.1 to -0.2. This creates anti-cue subjects on eight- and twelve-item lists and should push Experiment 4 from +0.020 toward the observed -0.117 while preserving the already accurate Experiment 6 mean and variance. Third, stated validity remains compressive but is modestly softened in evidence_slope to [0.52, 0.66]. The human Experiments 1 and 2 indicate serial-position/tie-break dominance over the nominal high-validity cue, so softening the late high-validity evidence slightly is required; the validity effect is still graded, non-zero, and influential rather than all-or-none. The architecture remains stochastic bounded sampling with no hard stop and no pure equal-weight tally.

**Parameters:**
  - `capacity_gamma`: `[0.45, 0.55]`
  - `position_decay`: `[0.20, 0.40]`
  - `att_validity_intercept`: `[0.48, 0.52]`
  - `att_validity_slope`: `[0.75, 0.90]`
  - `evidence_intercept`: `[0.50, 0.54]`
  - `evidence_slope`: `[0.52, 0.66]`
  - `first_cue_bonus`: `[1.65, 1.95]`
  - `first_cue_validity_power`: `[0.45, 0.75]`
  - `first_cue_capacity_power`: `[1.50, 2.50]`
  - `first_position_decay`: `[0.12, 0.22]`
  - `subject_scale`: `[0.40, 1.60]`
  - `subject_bias`: `[-0.30, 0.30]`
  - `beta`: `[1.10, 2.30]`
  - `epsilon`: `[0.12, 0.18]`
  - `long_jitter_shift`: `[0.78, 0.98]`
  - `validities`: `validities`
  - `evidence_jitter`: `[(-1.5, 3.0)] * n_features`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    if isinstance(state, dict):
        a = np.asarray(state['option_a_ratings'], dtype=float)
        b = np.asarray(state['option_b_ratings'], dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f'Bounded-sampling accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'
        )

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n_features, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.shape[0] != n_features:
            validities = np.full(n_features, 0.75, dtype=float)

    jitter = parameters.get('evidence_jitter')
    if jitter is None:
        jitter = np.ones(n_features, dtype=float)
    else:
        jitter = np.asarray(jitter, dtype=float)
        if jitter.ndim != 1 or jitter.shape[0] != n_features:
            jitter = np.ones(n_features, dtype=float)
        else:
            if n_features > 5:
                jitter = np.clip(jitter, -2.5, 3.0)
                jitter = jitter - float(parameters['long_jitter_shift'])
            else:
                jitter = np.clip(jitter, 0.1, 3.0)

    capacity_gamma = float(parameters['capacity_gamma'])
    position_decay = float(parameters['position_decay'])
    att_validity_intercept = float(parameters['att_validity_intercept'])
    att_validity_slope = float(parameters['att_validity_slope'])
    evidence_intercept = float(parameters['evidence_intercept'])
    evidence_slope = float(parameters['evidence_slope'])
    first_cue_bonus = float(parameters['first_cue_bonus'])
    first_cue_validity_power = float(parameters['first_cue_validity_power'])
    first_cue_capacity_power = float(parameters['first_cue_capacity_power'])
    first_position_decay = float(parameters['first_position_decay'])
    subject_scale = float(parameters['subject_scale'])
    subject_bias = float(parameters['subject_bias'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    capacity = (5.0 / n_features) ** capacity_gamma

    positions = np.arange(n_features, dtype=float)
    att_validity = np.clip(
        att_validity_intercept + att_validity_slope * (validities - 0.5), 0.05, 1.0
    )

    sampling_weights = (
        capacity
        * np.exp(-position_decay * positions * (capacity ** 2))
        * att_validity
    )

    diff = a - b
    discrim = diff != 0
    signs = np.where(discrim, np.sign(diff), 0.0)
    discrim_idx = np.flatnonzero(discrim)

    if discrim_idx.size == 0:
        return np.full(2, 0.5, dtype=float)

    sample_size = max(2, int(np.ceil(capacity * discrim_idx.size)))

    if discrim_idx.size <= sample_size:
        sampled = discrim_idx.copy()
    else:
        w = sampling_weights[discrim_idx].astype(float)
        if w.sum() <= 0.0:
            w = np.ones(discrim_idx.size, dtype=float)
        w = w / w.sum()
        sampled = np.random.choice(
            discrim_idx, size=sample_size, replace=False, p=w
        )

    first_disc = int(np.argmax(discrim))

    max_validity = float(np.max(validities))
    first_bonus = (
        first_cue_bonus
        * (validities[first_disc] / max_validity) ** first_cue_validity_power
        * np.exp(-first_cue_capacity_power * max(0.0, n_features - 5.0))
        * np.exp(-first_position_decay * max(0.0, first_disc - 1.0))
    )

    evidence_base = subject_scale * np.clip(
        evidence_intercept + evidence_slope * (validities - 0.5), 0.1, 1.2
    )
    evidence = evidence_base * jitter

    score = 0.0
    for j in sampled:
        j = int(j)
        score += signs[j] * evidence[j]
        if j == first_disc:
            score += signs[j] * first_bonus

    logits = np.array([beta * score + subject_bias, 0.0], dtype=float)
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
