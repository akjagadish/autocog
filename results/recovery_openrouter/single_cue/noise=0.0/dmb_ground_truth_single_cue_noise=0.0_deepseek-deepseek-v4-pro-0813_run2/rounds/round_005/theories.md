# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5_2` — KILLED ✗

**Description:** People compare options through a conflict-gated dual-lexicographic/tallying system. A baseline mixture combines a weak forward Take-The-Best route, a moderate backward lexicographic route, and feature-win tallying. When the highest-validity and lowest-validity cues discriminate in opposite directions and the tally does not oppose the highest cue, the decision maker enters a conflict regime: tallying is suspended and the choice is made with high probability by a strong backward lexicographic scan over ascending validities. Subjects differ substantially in how strongly and deterministically they reverse on such conflict trials.

**Rationale:** The conflict gate now requires exactly one highest-validity cue and exactly one lowest-validity cue; tied extremes disable the gate and fall back to the baseline mixture. This removes the spurious conflict activations caused by the tied low cue in Experiment 9 and the tied top cue in Experiment 10, while keeping legitimate conflict reversal in Experiments 5-7. The conflict regime is strengthened by raising the lower bounds and central mass of w_conflict_back and beta_conflict_back, closing the remaining first-cue magnitude gaps. Baseline beta ranges are widened slightly as requested to keep the forward route weak and the backward route moderate with additional subject-level variation.

**Parameters:**
  - `w_forward`: `[0.17, 0.30]`
  - `w_backward`: `[0.36, 0.60]`
  - `beta_forward`: `[0.0, 0.6]`
  - `beta_backward`: `[1.0, 2.5]`
  - `beta_tally`: `[1.5, 3.0]`
  - `w_conflict_back`: `[0.75, 0.95]`
  - `beta_conflict_back`: `[1.8, 4.0]`
  - `epsilon`: `{0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_conflict_back = float(parameters['beta_conflict_back'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward_base = lex_probabilities(ascending, beta_backward)
    p_backward_conflict = lex_probabilities(ascending, beta_conflict_back)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally_base = 1.0 - w_forward - w_backward
    if w_tally_base < 0.0:
        w_tally_base = 0.0
        denom = w_forward + w_backward
        w_forward /= denom
        w_backward /= denom

    v_max = float(np.max(validities))
    v_min = float(np.min(validities))
    top_positions = np.where(validities == v_max)[0]
    bottom_positions = np.where(validities == v_min)[0]

    if top_positions.size == 1 and bottom_positions.size == 1:
        j_hi = int(top_positions[0])
        j_lo = int(bottom_positions[0])
        hi_dir = 0
        if a[j_hi] > b[j_hi]:
            hi_dir = 1
        elif b[j_hi] > a[j_hi]:
            hi_dir = -1
        lo_dir = 0
        if a[j_lo] > b[j_lo]:
            lo_dir = 1
        elif b[j_lo] > a[j_lo]:
            lo_dir = -1
        tally_d = int(a_wins - b_wins)
        extreme_cue_conflict = (hi_dir != 0 and lo_dir != 0 and hi_dir == -lo_dir)
        live_conflict = bool(extreme_cue_conflict and (tally_d * hi_dir >= 0))
    else:
        live_conflict = False

    if live_conflict:
        w_conflict_back = float(parameters['w_conflict_back'])
        w_back_eff = w_conflict_back
        w_forward_eff = 1.0 - w_back_eff
        if w_forward_eff < 0.0:
            w_forward_eff = 0.0
            w_back_eff = 1.0
        w_tally_eff = 0.0
        p_backward = p_backward_conflict
    else:
        w_forward_eff = w_forward
        w_back_eff = w_backward
        w_tally_eff = w_tally_base
        p_backward = p_backward_base

    weights = np.array([w_forward_eff, w_back_eff, w_tally_eff], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.full(2, 0.5, dtype=float)

    p = np.clip(p, 0.0, None)
    return p / p.sum()
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

**Description:** People choose between two options through a stable, subject-level mixture of three comparison routes: (1) forward lexicographic scanning, inspecting cues in descending validity order and stopping at the first discriminating cue; (2) backward lexicographic scanning, inspecting cues in ascending validity order and stopping at the first discriminating cue in that reversed order; and (3) feature-win tallying, where each option receives one point per feature on which it strictly beats the other and ties contribute nothing. The forward and backward lexicographic weights jointly act as a subject-level polarity parameter for first-validity-cue effects: forward weight supports a positive first-discriminator effect, while backward weight supports a negative one. Each route's preferred option is passed through its own softmax choice rule, so responding is noisy but not controlled by a uniform lapse.

**Rationale:** This edit keeps the prescribed three-route mechanism but fixes the likely scoring failure in the previous candidate. The unresolved/scored-as-missing validity specification is now robustly handled: `parameters` still declares the experiment-defined symbolic `validities`, and `predict` verifies the array shape and falls back to a descending validity vector if needed. I also replaced the tight point-mass ranges with wider intervals around the fitted operating point, so route weights and the forward softmax temperature are genuinely sampled across subjects. The operating point uses a forward lexicographic weight near 0.27, a backward lexicographic weight near 0.48, a low but non-zero forward beta so the forward route contributes a small positive first-cue effect, a moderately deterministic backward route to supply negative first-discriminator contrasts in Experiments 3 and 4, and a tallying route with intermediate noise to keep Experiment 2's tally-minus-second-cue contrast near the observed positive value.

**Parameters:**
  - `w_forward`: `[0.22, 0.32]`
  - `w_backward`: `[0.42, 0.54]`
  - `beta_forward`: `[0.08, 0.20]`
  - `beta_backward`: `[1.30, 1.70]`
  - `beta_tally`: `[1.80, 2.60]`
  - `epsilon`: `{0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n_features = stim.shape[1]

    # Preferred validity vector comes from the experiment.  If the
    # parameter resolver ever leaves it unresolved, fall back to a
    # descending validity vector so the model is still scorable.
    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n_features))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n_features:
        validities = np.linspace(0.9, 0.5, n_features)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.ones(2, dtype=float) / 2.0

        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / weights.sum()

    p = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally

    epsilon = float(parameters.get('epsilon', 0.0))
    if epsilon > 0.0:
        p = (1.0 - epsilon) * p + epsilon * np.ones(2, dtype=float) / 2.0

    p = np.clip(p, 0.0, None)
    return p / p.sum()
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** People choose between two products through a stable, subject-level mixture of three routes: a weak forward lexicographic scan, a moderate backward/recency lexicographic scan, and feature-win tallying. Choice is modulated by a context-sensitive amplification of a higher-precision backward lexicographic route. The decision maker monitors the trailing 24-trial frequency of fully diagnostic, near-tied-tally comparisons, ignores isolated occurrences via a minimum qualifying count, and maps the qualifying fraction through a band-limited trapezoidal sensitivity curve. Moderate densities of such trials produce the strongest backward amplification, which generates the strong negative top-cue effects seen in dense complementary designs; very high densities attenuate to a lower floor, protecting all-complementary blocks from over-reversal. Sparse designs that never contain fully diagnostic near-tied trials keep the trigger at exactly zero, so their behavior stays governed by the pi_4-like baseline mixture, preserving the near-zero active-inactive contrasts in those experiments.

**Rationale:** Minimal-diff calibration of the accepted count-gated, band-limited trigger, following the most recent critic diagnosis. The mechanism is unchanged: a pi_4-like baseline mixture plus a backward/recency boost whose trigger weight s is driven by the trailing 24-trial frequency of fully diagnostic, near-tied-tally trials. Only two parameter ranges are changed: s_max is raised from [0.76, 0.84] to [0.84, 0.92] so the moderate-density Experiment 6 block reaches the stronger backward-dominant mixture needed for roughly -0.66, while s_floor is lowered from [0.50, 0.62] to [0.40, 0.50] so Experiment 5's high qualifying fraction produces a weaker backward boost, pulling its current -0.7042 back toward -0.6583. The all-cues-discriminate gate, minimum qualifying count, context window, band locations, and amplified-route precision are left exactly as in the accepted base, preserving the near-zero gate effects in sparse designs and the close fits in Experiments 3, 4, and 7.

**Parameters:**
  - `w_forward`: `[0.22, 0.32]`
  - `w_backward`: `[0.42, 0.54]`
  - `beta_forward`: `[0.08, 0.20]`
  - `beta_backward`: `[1.30, 1.70]`
  - `beta_tally`: `[1.80, 2.60]`
  - `beta_backward_strong`: `[2.50, 4.00]`
  - `context_window`: `{24}`
  - `tie_margin`: `[2.0, 3.0]`
  - `min_qualifying_count`: `{2, 3}`
  - `rise_start_frac`: `[0.03, 0.06]`
  - `peak_start_frac`: `[0.08, 0.14]`
  - `peak_end_frac`: `[0.26, 0.34]`
  - `fall_end_frac`: `[0.40, 0.48]`
  - `s_max`: `[0.84, 0.92]`
  - `s_floor`: `[0.40, 0.50]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 3 and stim.shape[0] == 1:
        stim = stim[0]
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    n = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = list(np.linspace(0.9, 0.5, n))
    validities = np.asarray(validities, dtype=float)
    if validities.ndim == 0 or validities.shape[0] != n:
        validities = np.linspace(0.9, 0.5, n)

    descending = np.argsort(-validities, kind='stable')
    ascending = np.argsort(validities, kind='stable')

    def lex_probabilities(order, beta):
        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            return np.full(2, 0.5, dtype=float)
        z = float(beta) if winner == 0 else -float(beta)
        p_a = 1.0 / (1.0 + np.exp(-z))
        return np.array([p_a, 1.0 - p_a], dtype=float)

    tie_margin = float(parameters['tie_margin'])

    def complementarity_signal(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = x - y
        n_discrim = int(np.sum(d != 0.0))
        tally = float(np.sum(d > 0.0) - np.sum(d < 0.0))
        return 1.0 if (n_discrim >= n and abs(tally) <= tie_margin) else 0.0

    signals = []
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        signals.append(complementarity_signal(past_a[i], past_b[i]))
    signals.append(complementarity_signal(a, b))

    context_window = int(parameters['context_window'])
    if len(signals) > context_window:
        signals = signals[-context_window:]

    n_recent = max(1, len(signals))
    n_qual = float(np.sum(signals))
    frac = n_qual / n_recent

    min_count = int(parameters['min_qualifying_count'])
    rise_start = float(parameters['rise_start_frac'])
    peak_start = float(parameters['peak_start_frac'])
    peak_end = float(parameters['peak_end_frac'])
    fall_end = float(parameters['fall_end_frac'])
    s_max = float(parameters['s_max'])
    s_floor = min(float(parameters['s_floor']), s_max)

    if n_qual < min_count:
        s = 0.0
    else:
        if frac <= rise_start:
            s = 0.0
        elif frac <= peak_start:
            s = s_max * (frac - rise_start) / (peak_start - rise_start + 1e-9)
        elif frac <= peak_end:
            s = s_max
        elif frac <= fall_end:
            s = s_max - (s_max - s_floor) * (frac - peak_end) / (fall_end - peak_end + 1e-9)
        else:
            s = s_floor
    s = float(np.clip(s, 0.0, 1.0))

    beta_forward = float(parameters['beta_forward'])
    beta_backward = float(parameters['beta_backward'])
    beta_backward_strong = float(parameters['beta_backward_strong'])

    p_forward = lex_probabilities(descending, beta_forward)
    p_backward = lex_probabilities(ascending, beta_backward)
    p_backward_strong = lex_probabilities(ascending, beta_backward_strong)

    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins], dtype=float)
    beta_tally = float(parameters['beta_tally'])
    z = beta_tally * (tally_scores - np.max(tally_scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)

    w_forward = float(parameters['w_forward'])
    w_backward = float(parameters['w_backward'])
    w_tally = 1.0 - w_forward - w_backward
    if w_tally < 0.0:
        w_tally = 0.0
    weights = np.array([w_forward, w_backward, w_tally], dtype=float)
    weights = weights / np.sum(weights)

    p_base = weights[0] * p_forward + weights[1] * p_backward + weights[2] * p_tally
    p = (1.0 - s) * p_base + s * p_backward_strong
    p = np.clip(p, 0.0, None)
    return p / np.sum(p)
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0
    probs = probs / probs.sum()
    return int(np.random.choice(probs.size, p=probs))
```
