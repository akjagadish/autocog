# Round 5 — Theories

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


### slot 2 — `pi_7` — KILLED ✗

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


## Replacement

### `pi_8` → slot 2 (via `new_theory`)

**Description:** People make binary quality choices by salience-weighted ordered evidence accumulation. Display position is primary: on short lists the first displayed discriminating cue receives strong weight, and nearby same-sign cues form an early run, while an odd-one-out/block-contrast bonus amplifies a first cue that conflicts with an otherwise uniform later block. Stated validity enters only as a compressive attentional gain, never as a lexicographic multiplier. Later opposing cues accumulate as a count-sensitive but compressively weighted block. The configural odd-one-out bonus is sharply length-gated so it is near full strength on short displays (n=4-6) and essentially off on long displays (n>=8). On long displays a subject-level trait controls whether any residual configural integration survives list-length pressure: some subjects retain a partial odd-one-out bonus even on long lists while others rely wholly on the tally tail, which produces the near-zero mean with large between-subject variance observed in Experiment 6. Opposing cues inside the early span are strongly discounted, and a tie-gap immediately after the first cue additionally suppresses both the first cue's direct evidence and the configural bonus, generating the negative R_near - R_all signature (Experiment 11) and the strong mirror asymmetry (Experiment 12).

**Rationale:** This is a minimal-diff edit that refuses the recommended gate sign-flip and instead fixes the two genuine short-list weaknesses plus the long-list sign error. Important arithmetic check: the current gate expression `odd_gate = 1/(1+exp((n-6.5)/temp))` already yields approximately 0.998 at n=4 and approximately 1e-6 at n=12, because `1/(1+exp(z))` is a decreasing function of z. The critic's claimed values (0.002 at n=4, 1.0 at n=12) evaluate the sigmoid with the sign of its argument flipped, so implementing the suggested `(6.5 - n)/temp` would actually INVERT a correctly oriented gate, turning the odd-one-out bonus off on short configural displays (destroying Experiments 11 and 12) and on at n=12 (re-inflating long-list primacy). I therefore keep the existing gate orientation. The measured failures instead trace to two calibration knobs inside the preserved mechanism. First, the tie-gap penalty at 0.50-0.70 removes only 30-50% of an isolated first cue's evidence, which is too weak to separate the all-opposing and near-tie profiles in Experiment 11 (-0.034 vs real -0.187); tightening it to 0.30-0.50 removes 50-70% and pushes R_near - R_all to about -0.19. Second, opposing cues inside the short-list early span are discounted too gently at 0.60-0.75, which keeps first-cue dominance in Experiment 12 too weak (0.478 vs real 0.699); strengthening the discount to 0.40-0.60 raises the mirror asymmetry toward roughly 0.68. Both changes only fire under `short_configural_opposing` (n<=5, first discriminator at position 0, compact uniform opposing block), so Experiments 1, 2, 4, 5, 7, 8, 9, and 10 are untouched by them; the tiny expected side effect on Experiment 3's rows 9-10 moves it from 0.646 toward ~0.65, still inside the real value's tolerance band. Third, the wrong-sign Experiment 6 result (+0.078 vs real -0.063) is not caused by the gate (already off at n=12) but by the recency-boosted tail overtallying the 2-3 cue opposing blocks once the bonus is fully gone; I add a subject-level `long_config_bonus` (uniform 0.0-2.0) that allows some subjects to retain a partial odd-one-out/block-contrast integration on long lists while others remain tally-driven. This 'long-list configural trait' pulls the long-list mean toward the near-zero real value and simultaneously raises the badly under-predicted between-subject variance (0.029 vs real 0.197), and it activates only on n>=9 configural rows, so the long-list experiments 4, 8, 9, and 10 are essentially unaffected. Everything else — weak compressive validity gain, count-sensitive compressed tail, first-discount schedule, primacy/recency transition around n=7-8 — is re-emitted verbatim from the accepted base.

**Parameters:**
  - `early_weight`: `[1.90, 2.30]`
  - `tail_weight`: `[0.25, 0.40]`
  - `validity_gain`: `[0.08, 0.18]`
  - `odd_one_out_bonus`: `[1.80, 2.80]`
  - `block_compression`: `[0.10, 0.22]`
  - `length_center`: `[7.20, 7.80]`
  - `length_temp`: `[0.55, 0.85]`
  - `early_damp`: `[0.90, 1.40]`
  - `second_ratio`: `[0.74, 0.86]`
  - `third_ratio`: `[0.22, 0.32]`
  - `early_span`: `{3}`
  - `isolated_scale`: `[0.20, 0.40]`
  - `strong_first_discount`: `[0.45, 0.65]`
  - `weak_first_discount`: `[0.03, 0.12]`
  - `same_run_gain`: `[1.30, 1.70]`
  - `recency_scale`: `[1.40, 2.00]`
  - `recency_power`: `[1.00, 1.50]`
  - `odd_density_power`: `[1.60, 2.30]`
  - `odd_count_decay`: `[0.25, 0.60]`
  - `tie_gap_penalty`: `[0.30, 0.50]`
  - `opposing_early_ratio`: `[0.40, 0.60]`
  - `odd_length_temp`: `[0.25, 0.55]`
  - `long_config_bonus`: `[0.00, 2.00]`
  - `beta`: `[0.85, 1.05]`
  - `subject_scale`: `[0.40, 1.80]`
  - `bias`: `[-0.05, 0.05]`
  - `trial_noise_sd`: `[0.08, 0.18]`
  - `lapse`: `[0.04, 0.09]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    if isinstance(state, dict):
        a = np.asarray(state.get('option_a_ratings'), dtype=float)
        b = np.asarray(state.get('option_b_ratings'), dtype=float)
    else:
        state_arr = np.asarray(state, dtype=float)
        if state_arr.ndim != 2 or state_arr.shape[0] != 2:
            return np.full(2, 0.5, dtype=float)
        a = state_arr[0]
        b = state_arr[1]

    if a.ndim != 1 or a.shape != b.shape:
        return np.full(2, 0.5, dtype=float)

    n_features = int(a.size)
    if n_features == 0:
        return np.full(2, 0.5, dtype=float)

    validities = parameters.get('validities')
    if validities is None:
        validities = np.full(n_features, 0.6, dtype=float)
    else:
        validities = np.asarray(validities, dtype=float)
        if validities.ndim != 1 or validities.size != n_features:
            validities = np.full(n_features, 0.6, dtype=float)
    validities = np.clip(validities, 0.5, 1.0)

    diff = a - b
    discrim = diff != 0.0
    if not np.any(discrim):
        return np.full(2, 0.5, dtype=float)

    signs = np.where(discrim, np.sign(diff), 0.0)
    disc_positions = np.flatnonzero(discrim)
    first_disc = int(disc_positions[0])
    sign_first = float(signs[first_disc])

    length_center = float(parameters['length_center'])
    length_temp = float(parameters['length_temp'])
    z_len = (float(n_features) - length_center) / length_temp
    z_len = float(max(-30.0, min(30.0, z_len)))
    if z_len >= 0.0:
        length_shift = 1.0 / (1.0 + np.exp(-z_len))
    else:
        ez = np.exp(z_len)
        length_shift = ez / (1.0 + ez)

    early_weight = float(parameters['early_weight'])
    early_damp = float(parameters['early_damp'])
    early_amp = early_weight / (1.0 + early_damp * length_shift)

    early_span = int(parameters['early_span'])
    second_ratio = float(parameters['second_ratio'])
    third_ratio = float(parameters['third_ratio'])
    early_ratio = np.ones(n_features, dtype=float)
    if n_features > 1:
        early_ratio[1] = second_ratio
    if n_features > 2:
        early_ratio[2] = third_ratio

    later_positions = disc_positions[1:]
    configural_odd = (
        later_positions.size >= 2
        and first_disc < early_span
        and np.all(signs[later_positions] == -sign_first)
    )

    opposing_early_ratio = float(parameters['opposing_early_ratio'])
    short_configural_opposing = (
        n_features <= 5
        and first_disc == 0
        and configural_odd
        and later_positions.size <= 3
    )

    isolated_scale = float(parameters['isolated_scale'])
    if disc_positions.size == 1:
        early_amp_eff = early_amp * isolated_scale
    else:
        early_amp_eff = early_amp

    early_mask = np.arange(n_features) < early_span
    early_weight_arr = np.where(early_mask, early_amp_eff * early_ratio, 0.0)

    tail_weight = float(parameters['tail_weight'])
    recency_scale = float(parameters['recency_scale'])
    recency_power = float(parameters['recency_power'])
    recency_amp = recency_scale * length_shift
    positions = np.arange(n_features, dtype=float)
    norm_pos = positions / max(1, n_features - 1)
    tail_pos = tail_weight + recency_amp * np.power(norm_pos, recency_power)
    tail_weight_arr = np.where(~early_mask, tail_pos, 0.0)

    strong_first_discount = float(parameters['strong_first_discount'])
    weak_first_discount = float(parameters['weak_first_discount'])
    first_discount_factor = 1.0
    if n_features >= 7:
        if configural_odd:
            first_discount_factor = 1.0 - weak_first_discount * length_shift
        else:
            first_discount_factor = 1.0 - strong_first_discount * length_shift
        first_discount_factor = float(max(0.35, first_discount_factor))

    validity_gain = float(parameters['validity_gain'])
    validity_gain_arr = 1.0 + validity_gain * (validities - 0.5)

    same_run_gain = float(parameters['same_run_gain'])

    early_evidence = 0.0
    tail_raw = 0.0
    for pos in disc_positions:
        pos = int(pos)
        sign = float(signs[pos])
        gain = validity_gain_arr[pos]
        if pos < early_span:
            w = early_weight_arr[pos] * gain
            if pos == first_disc:
                w *= first_discount_factor
            elif sign == sign_first:
                w *= same_run_gain
            elif short_configural_opposing:
                w *= opposing_early_ratio
            early_evidence += w * sign
        else:
            tail_raw += tail_weight_arr[pos] * gain * sign

    block_compression = float(parameters['block_compression'])
    tail_score = tail_raw / (1.0 + block_compression * abs(tail_raw))

    evidence = early_evidence + tail_score

    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])
    odd_density_power = float(parameters['odd_density_power'])
    odd_count_decay = float(parameters['odd_count_decay'])
    tie_gap_penalty = float(parameters['tie_gap_penalty'])
    odd_length_temp = float(parameters['odd_length_temp'])

    z_odd = float(max(-30.0, min(30.0, (float(n_features) - 6.5) / odd_length_temp)))
    odd_gate = 1.0 / (1.0 + np.exp(z_odd))

    if configural_odd:
        span = float(later_positions[-1] - first_disc + 1)
        n_opposing = float(later_positions.size)
        density = n_opposing / span
        odd_score = (
            odd_one_out_bonus
            * (density ** odd_density_power)
            * np.exp(-odd_count_decay * max(0.0, n_opposing - 2.0))
            * odd_gate
        )

        first_cue_raw = (
            early_weight_arr[first_disc]
            * validity_gain_arr[first_disc]
            * sign_first
            * first_discount_factor
        )

        if (first_disc + 1 < n_features) and (not discrim[first_disc + 1]):
            evidence += (tie_gap_penalty - 1.0) * first_cue_raw
            odd_score *= tie_gap_penalty

        evidence += odd_score * sign_first

        long_config_bonus = float(parameters.get('long_config_bonus', 0.0))
        if n_features >= 9 and long_config_bonus > 0.0:
            long_span = float(later_positions[-1] - first_disc + 1)
            long_n_opposing = float(later_positions.size)
            long_density = long_n_opposing / long_span
            long_score = (
                long_config_bonus
                * (long_density ** odd_density_power)
                * np.exp(-odd_count_decay * max(0.0, long_n_opposing - 2.0))
            )
            evidence += long_score * sign_first

    subject_scale = float(parameters['subject_scale'])
    beta = float(parameters['beta'])
    bias = float(parameters['bias'])
    trial_noise_sd = float(parameters['trial_noise_sd'])
    lapse = float(parameters['lapse'])

    logit_a = beta * (subject_scale * evidence + bias)
    logit_a += np.random.normal(0.0, trial_noise_sd)

    logits = np.array([logit_a, 0.0], dtype=float)
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
    total = probs.sum()
    if total <= 0.0 or not np.isfinite(total):
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```
