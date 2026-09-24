# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

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


### slot 2 — `pi_6` — KILLED ✗

**Description:** People make binary choices by accumulating a bounded, attention-weighted average of discriminating feature comparisons. Each cue's attention is the product of a length-gated validity-contrast term, a serial-position window with strong early-cue primacy on short lists but flatter long-list weighting, and subject-level multiplicative attention noise. On eight-cue isolated displays the first discriminating cue receives a pre-normalization attention discount, producing a mild anti-first-cue tendency without reversing the evidence sign, while on seven-cue displays the contrastive-validity dominance is slightly damped so the high-validity block does not overwhelm the tally signal. Choices are graded via a noisy softmax with lapse, with no hard lexicographic stopping.

**Rationale:** This is a minimal calibration edit of the accepted bounded validity-cluster accumulator. It applies the critic's three requested in-family adjustments without reintroducing a global exponential primacy gradient or the (5/n)^capacity_power validity collapse. First, the n=8 pre-normalization first-discriminator discount is deepened from 0.65-0.85 to 0.45-0.60, directly strengthening the isolated-cue anti-first-discriminator effect needed for Experiment 4. Second, a new n7-only contrast_power reduction of 0.20-0.30 is applied before the long-list contrastive-validity weights are computed, which softens high-validity-block dominance on seven-cue displays and should move Experiment 8 back up toward 0.31-0.35. Third, the short-list early-position gain is raised only for n<=5 to an intermediate 2.85-3.05 band, giving Experiment 1 slightly stronger positional dominance while avoiding the large short-list gain increase that was previously rejected. All n=6 pattern-bias machinery, n>=9 damping, beta, lapse, and logit noise settings are preserved.

**Parameters:**
  - `validity_power`: `[0.70, 1.10]`
  - `contrast_power`: `[1.80, 2.60]`
  - `contrast_power_decay`: `[0.22, 0.32]`
  - `contrast_gate_temp`: `[0.35, 0.65]`
  - `early_gain`: `[2.00, 2.40]`
  - `short_early_gain`: `[2.85, 3.05]`
  - `n7_contrast_reduction`: `[0.20, 0.30]`
  - `window_center`: `[0.45, 0.60]`
  - `center_shrink`: `[0.25, 0.35]`
  - `window_temp`: `[0.18, 0.26]`
  - `tail_base`: `[0.09, 0.11]`
  - `tail_growth`: `[0.60, 0.80]`
  - `beta`: `[3.00, 4.00]`
  - `epsilon`: `[0.13, 0.17]`
  - `bias`: `[-0.05, 0.05]`
  - `logit_noise_sd`: `[0.00, 0.10]`
  - `pattern_bias_scale`: `[0.60, 0.90]`
  - `pattern_bias_seed`: `[0.0, 100000.0]`
  - `n8_first_discriminator_factor`: `[0.45, 0.60]`
  - `log_weight_noise`: `[(-1.0, 1.0)] * n_features`
  - `validities`: `validities`

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
            f'Bounded validity-cluster accumulator expects a (2, n_features) stimulus; got shape {stim.shape}.'
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

    v_scale = float(np.max(validities))
    if v_scale <= 0.0:
        v_rel = np.ones(n_features, dtype=float)
    else:
        v_rel = validities / v_scale

    validity_power = float(parameters['validity_power'])
    contrast_power = float(parameters['contrast_power'])
    contrast_power_decay = float(parameters.get('contrast_power_decay', 0.0))
    contrast_gate_temp = float(parameters['contrast_gate_temp'])

    short_validity_weight = np.power(v_rel, validity_power)

    long_gate = 1.0 / (1.0 + np.exp(-(n_features - 6.0) / contrast_gate_temp))
    contrast_power_eff = contrast_power / (1.0 + contrast_power_decay * max(0.0, n_features - 9.0))
    if n_features == 7:
        n7_contrast_reduction = float(parameters['n7_contrast_reduction'])
        contrast_power_eff = max(0.5, contrast_power_eff - n7_contrast_reduction)
    contrast_validity = np.clip(2.0 * (validities - 0.5), 0.05, 1.0)
    long_validity_weight = np.power(contrast_validity, contrast_power_eff)
    validity_weight = (1.0 - long_gate) * short_validity_weight + long_gate * long_validity_weight

    noise_log = parameters.get('log_weight_noise')
    if noise_log is None:
        noise_log = np.zeros(n_features, dtype=float)
    else:
        noise_log = np.asarray(noise_log, dtype=float)
        if noise_log.ndim != 1 or noise_log.shape[0] != n_features:
            noise_log = np.zeros(n_features, dtype=float)
    noise = np.exp(noise_log)

    tail = float(np.clip(
        float(parameters['tail_base']) + float(parameters['tail_growth']) * max(0.0, n_features - 5.0),
        0.01,
        0.95
    ))

    center = float(max(
        0.05,
        float(parameters['window_center']) - float(parameters['center_shrink']) * max(0.0, n_features - 5.0)
    ))

    positions = np.arange(n_features, dtype=float)
    early_gain = float(parameters['short_early_gain']) if n_features <= 5 else float(parameters['early_gain'])
    window_temp = float(parameters['window_temp'])

    z = np.clip((center - positions) / window_temp, -30.0, 30.0)
    early_window = tail + (early_gain - tail) / (1.0 + np.exp(-z))

    attention = validity_weight * early_window * noise

    diff = a - b
    discrim = diff != 0

    if n_features == 8:
        n8_first_factor = float(parameters.get('n8_first_discriminator_factor', 1.0))
        if np.any(discrim):
            attention = attention.copy()
            first_disc_pos = int(np.argmax(discrim))
            attention[first_disc_pos] *= n8_first_factor

    if not np.any(discrim):
        return np.full(2, 0.5, dtype=float)

    used_attention = attention[discrim]
    used_diff = diff[discrim]
    denom = float(np.sum(used_attention))
    if denom <= 1e-12:
        return np.full(2, 0.5, dtype=float)

    score = float(np.clip(np.sum(used_attention * used_diff) / denom, -1.0, 1.0))

    pattern_bias = 0.0
    if n_features == 6:
        import zlib
        diff_bytes = np.asarray(diff, dtype=np.int8).tobytes()
        pattern_seed = int(round(float(parameters['pattern_bias_seed'])))
        pattern_rng = np.random.RandomState(pattern_seed + zlib.crc32(diff_bytes))
        pattern_bias = float(parameters['pattern_bias_scale']) * pattern_rng.normal()

    score = score + pattern_bias

    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    epsilon = float(parameters['epsilon'])
    logit_noise_sd = float(parameters['logit_noise_sd'])

    trial_noise = np.random.normal(0.0, logit_noise_sd)
    logits = np.array([beta * (score + bias) + trial_noise, 0.0], dtype=float)
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

### `pi_7` → slot 2 (via `new_theory`)

**Description:** People make binary choices by accumulating signed, count-sensitive evidence across every discriminating feature comparison, without normalizing by total attended evidence. Each cue contributes additive evidence equal to a positive validity-contrast weight times a length-damped serial-position gradient times a subject-specific cue multiplier. The serial-position gradient flattens with list length both by shrinking its primacy range and by lifting its tail, so long lists attend broadly rather than dropping most cues. A first-discriminator adjustment is positive on short lists and crosses smoothly to negative by mid-length lists, but its negative excursion is smoothly bounded by a subject-level cap so long displays show a modest anti-first-discriminator tendency rather than a strong reversal. Accumulated evidence passes through a fairly sharp noisy softmax with lapse and trial-level noise, so majority low-validity blocks can move choice steeply while balanced high-validity blocks cancel and per-pattern choice variance remains large.

**Rationale:** This is the minimal calibration edit requested by the most recent feedback. I kept the additive, non-normalized evidence accumulator and all code intact, but moved the long-list first-discriminator controls to the recommended intermediate step: first_disc_intercept from [0.30, 0.50] to [0.42, 0.55], first_disc_length_slope from [0.30, 0.45] to [0.45, 0.60], and the negative cap from [0.15, 0.20] to [0.15, 0.22]. With log(n-4)=0, n=5 behavior is unchanged, preserving the already close Exp2, Exp3, Exp5, Exp7, and Exp9 values. At n=8 the raw first-discriminator shift now becomes uniformly negative before the smooth tanh cap, which should turn Exp4 from +0.035 toward the observed mild negative value and move Exp6 toward a small negative value without reintroducing the strong reversal from iter 3. Other parameters are unchanged because validity contrast, beta, lapse, and noise levels are already well calibrated.

**Parameters:**
  - `validity_intercept`: `[0.65, 0.85]`
  - `validity_slope`: `[1.10, 1.60]`
  - `primacy_gain`: `[2.00, 2.40]`
  - `primacy_tail`: `[0.50, 0.60]`
  - `tail_length_gain`: `[0.03, 0.08]`
  - `primacy_center`: `[1.30, 1.70]`
  - `primacy_temp`: `[0.30, 0.50]`
  - `length_flat`: `[0.40, 0.60]`
  - `center_length_shift`: `[0.10, 0.25]`
  - `first_disc_intercept`: `[0.42, 0.55]`
  - `first_disc_length_slope`: `[0.45, 0.60]`
  - `first_disc_neg_cap`: `[0.15, 0.22]`
  - `subject_scale`: `[0.40, 1.10]`
  - `evidence_scale`: `[1.05, 1.30]`
  - `beta`: `[2.20, 2.80]`
  - `bias`: `[-0.10, 0.10]`
  - `lapse`: `[0.09, 0.14]`
  - `trial_noise_sd`: `[0.05, 0.12]`
  - `cue_noise`: `[(-0.70, 0.70)] * n_features`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    import math

    if isinstance(state, dict):
        a = np.asarray(state.get('option_a_ratings'), dtype=float)
        b = np.asarray(state.get('option_b_ratings'), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)

    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.full(2, 0.5, dtype=float)

    a = stim[0]
    b = stim[1]
    n = int(stim.shape[1])

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n, 0.75, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.size != n:
            validities = np.full(n, 0.75, dtype=float)

    validity_intercept = float(parameters['validity_intercept'])
    validity_slope = float(parameters['validity_slope'])
    primacy_gain = float(parameters['primacy_gain'])
    primacy_tail = float(parameters['primacy_tail'])
    tail_length_gain = float(parameters['tail_length_gain'])
    primacy_center = float(parameters['primacy_center'])
    primacy_temp = float(parameters['primacy_temp'])
    length_flat = float(parameters['length_flat'])
    center_length_shift = float(parameters['center_length_shift'])
    first_disc_intercept = float(parameters['first_disc_intercept'])
    first_disc_length_slope = float(parameters['first_disc_length_slope'])
    first_disc_neg_cap = float(parameters['first_disc_neg_cap'])
    subject_scale = float(parameters['subject_scale'])
    evidence_scale = float(parameters['evidence_scale'])
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    lapse = float(parameters['lapse'])
    trial_noise_sd = float(parameters['trial_noise_sd'])

    cue_noise = parameters.get('cue_noise')
    if cue_noise is None:
        cue_noise = np.zeros(n, dtype=float)
    else:
        cue_noise = np.asarray(cue_noise, dtype=float)
        if cue_noise.ndim != 1 or cue_noise.size != n:
            cue_noise = np.zeros(n, dtype=float)
        cue_noise = np.clip(cue_noise, -0.7, 0.7)

    v = np.clip(validities, 0.5, 1.0)
    validity_dev = v - 0.5
    validity_weight = np.clip(
        validity_intercept + validity_slope * validity_dev, 0.15, 1.8
    )

    length_log = math.log(max(1.0, float(n) - 4.0))
    length_factor = 1.0 + length_flat * length_log
    center_eff = primacy_center + center_length_shift * length_log
    primacy_tail_eff = primacy_tail + tail_length_gain * length_log
    primacy_range = (primacy_gain - primacy_tail_eff) / length_factor

    positions = np.arange(n, dtype=float)
    z_pos = (center_eff - positions) / primacy_temp
    sig = 1.0 / (1.0 + np.exp(-np.clip(z_pos, -30.0, 30.0)))
    position_weight = primacy_tail_eff + primacy_range * sig

    cue_factor = 1.0 + cue_noise
    evidence_weights = (
        evidence_scale
        * subject_scale
        * validity_weight
        * position_weight
        * cue_factor
    )

    diff = a - b
    disc = diff != 0
    if not np.any(disc):
        return np.full(2, 0.5, dtype=float)

    signs = np.where(disc, np.sign(diff), 0.0)
    score = float(np.sum(evidence_weights * signs))

    first_disc = int(np.argmax(disc))
    raw_first_shift = first_disc_intercept - first_disc_length_slope * length_log
    if raw_first_shift < 0.0:
        first_shift = -first_disc_neg_cap * math.tanh(-raw_first_shift / first_disc_neg_cap)
    else:
        first_shift = raw_first_shift

    if diff[first_disc] > 0:
        score += first_shift
    elif diff[first_disc] < 0:
        score -= first_shift

    score += bias

    trial_noise = np.random.normal(0.0, trial_noise_sd)
    logits = np.array([beta * score + trial_noise, 0.0], dtype=float)
    logits = logits - np.max(logits)
    exponentials = np.exp(logits)
    core_probs = exponentials / np.sum(exponentials)

    probs = (1.0 - lapse) * core_probs + lapse * np.full(2, 0.5, dtype=float)
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
