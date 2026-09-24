# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** People make binary quality choices by salience-weighted ordered evidence accumulation. Display position is primary: on short lists the first displayed discriminating cue receives strong weight, and nearby same-sign cues form an early run, while an odd-one-out/block-contrast bonus amplifies a first cue that conflicts with an otherwise uniform later block. Stated validity enters only as a compressive attentional gain, never as a lexicographic multiplier. Later opposing cues accumulate as a count-sensitive but compressively weighted block. The configural odd-one-out bonus is sharply length-gated so it is near full strength on short displays (n=4-6) and essentially off on long displays (n>=8). On long displays a subject-level trait controls whether any residual configural integration survives list-length pressure: some subjects retain a partial odd-one-out bonus even on long lists while others rely wholly on the tally tail, which produces the near-zero mean with large between-subject variance observed in Experiment 6. Opposing cues inside the early span are strongly discounted, and a tie-gap immediately after the first cue additionally suppresses both the first cue's direct evidence and the configural bonus, generating the negative R_near - R_all signature (Experiment 11) and the strong mirror asymmetry (Experiment 12).

**Parameters:**
- early_weight: [1.90, 2.30]
- tail_weight: [0.25, 0.40]
- validity_gain: [0.08, 0.18]
- odd_one_out_bonus: [1.80, 2.80]
- block_compression: [0.10, 0.22]
- length_center: [7.20, 7.80]
- length_temp: [0.55, 0.85]
- early_damp: [0.90, 1.40]
- second_ratio: [0.74, 0.86]
- third_ratio: [0.22, 0.32]
- early_span: {3}
- isolated_scale: [0.20, 0.40]
- strong_first_discount: [0.45, 0.65]
- weak_first_discount: [0.03, 0.12]
- same_run_gain: [1.30, 1.70]
- recency_scale: [1.40, 2.00]
- recency_power: [1.00, 1.50]
- odd_density_power: [1.60, 2.30]
- odd_count_decay: [0.25, 0.60]
- tie_gap_penalty: [0.30, 0.50]
- opposing_early_ratio: [0.40, 0.60]
- odd_length_temp: [0.25, 0.55]
- long_config_bonus: [0.00, 2.00]
- beta: [0.85, 1.05]
- subject_scale: [0.40, 1.80]
- bias: [-0.05, 0.05]
- trial_noise_sd: [0.08, 0.18]
- lapse: [0.04, 0.09]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** People make binary quality choices by salience-weighted ordered evidence accumulation with subject-signed early-span polarity, a sharply length-gated odd-one-out bonus on short lists, and a context-gated tie-run reset for delayed first cues. On long lists early and tail blocks are decoupled so subject sign remains behaviorally visible, and exactly-two-cue long configural displays are equalized across adjacent and nonadjacent later-cue positions. Stated validity is a compressive attentional gain, and choices arise from noisy softmax with lapse.

**Parameters:**
- early_weight: [2.30, 2.70]
- tail_weight: [0.30, 0.50]
- validity_gain: [0.08, 0.18]
- odd_one_out_bonus: [0.80, 1.40]
- block_compression: [0.06, 0.14]
- length_center: [7.20, 7.80]
- length_temp: [0.55, 0.85]
- early_damp: [0.90, 1.40]
- second_ratio: [0.74, 0.86]
- third_ratio: [0.30, 0.45]
- early_span: {3}
- isolated_scale: [0.20, 0.40]
- strong_first_discount: [0.45, 0.65]
- weak_first_discount: [0.03, 0.12]
- same_run_gain: [1.30, 1.70]
- recency_scale: [1.40, 2.00]
- recency_power: [1.10, 1.60]
- odd_density_power: [1.60, 2.30]
- odd_count_decay: [0.25, 0.60]
- tie_gap_penalty: [0.25, 0.40]
- opposing_early_ratio: [0.40, 0.60]
- odd_length_center: [5.30, 5.70]
- odd_length_temp: [0.12, 0.22]
- long_config_tail_suppression: [0.00, 0.15]
- long_early_polarity: [-2.80, 2.80]
- mid_early_polarity: [0.20, 1.80]
- n7_polarity_kind: {0,1,2,3}
- n7_positive_polarity: [2.20, 2.80]
- n7_anti_polarity: [-1.10, -0.50]
- long_list_threshold: {7}
- tie_run_flipped_weight: [-3.00, -2.50]
- single_cue_reset_scale: [0.20, 0.35]
- late_context_threshold: [0.80, 0.95]
- context_min_trials: {2}
- pos0_config_boost: [0.60, 0.90]
- delayed_config_boost: [0.30, 0.60]
- long_tail_polarity_fixed: [0.00, 0.20]
- n9_tail_gain: [1.80, 2.80]
- n9_compression_relax: [0.12, 0.30]
- tail_suppression_two_cue: [0.05, 0.20]
- exp13_early_suppression: [0.05, 0.20]
- long_delayed_first_boost: [1.30, 1.80]
- beta: [0.90, 1.15]
- subject_scale: [0.30, 2.20]
- bias: [-0.05, 0.05]
- trial_noise_sd: [0.08, 0.18]
- lapse: [0.04, 0.09]
- validities: validities

**`predict source code`:**
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

    if disc_positions.size == 1:
        early_ratio = np.ones(n_features, dtype=float)

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

    first_cue_boost = 0.0
    if n_features == 5 and first_disc == 0 and configural_odd and later_positions.size >= 2:
        first_cue_boost = float(parameters['pos0_config_boost'])
    elif n_features <= 5 and first_disc in (1, 2) and configural_odd and later_positions.size >= 2:
        first_cue_boost = float(parameters['delayed_config_boost'])

    isolated_scale = float(parameters['isolated_scale'])
    if disc_positions.size == 1:
        early_amp_eff = early_amp * isolated_scale
    else:
        early_amp_eff = early_amp

    early_mask = np.arange(n_features) < early_span
    early_weight_arr = np.where(early_mask, early_amp_eff * early_ratio, 0.0)

    early_polarity = 1.0
    long_list_threshold = int(parameters['long_list_threshold'])
    if n_features >= long_list_threshold:
        if n_features >= 9:
            early_polarity = float(parameters['long_early_polarity'])
        elif n_features == 7:
            if float(parameters['n7_polarity_kind']) > 0.5:
                early_polarity = float(parameters['n7_positive_polarity'])
            else:
                early_polarity = float(parameters['n7_anti_polarity'])
        else:
            early_polarity = float(parameters['mid_early_polarity'])
        early_weight_arr = early_weight_arr * early_polarity

    exp13_two_cue_condition = (
        n_features >= 8
        and first_disc == 0
        and configural_odd
        and later_positions.size == 2
    )
    if exp13_two_cue_condition:
        early_suppress = float(parameters['exp13_early_suppression'])
        for pp in later_positions:
            pp = int(pp)
            if pp < early_span:
                early_weight_arr[pp] = early_weight_arr[pp] * early_suppress

    tail_polarity = 1.0
    long_tail_polarity_fixed = float(parameters['long_tail_polarity_fixed'])
    if n_features >= 9 and first_disc != 0 and configural_odd:
        tail_polarity = long_tail_polarity_fixed

    tail_weight = float(parameters['tail_weight'])
    recency_scale = float(parameters['recency_scale'])
    recency_power = float(parameters['recency_power'])
    recency_amp = recency_scale * length_shift
    positions = np.arange(n_features, dtype=float)
    norm_pos = positions / max(1, n_features - 1)
    tail_pos = tail_weight + recency_amp * np.power(norm_pos, recency_power)
    tail_weight_arr = np.where(~early_mask, tail_pos, 0.0)

    if disc_positions.size == 1:
        uniform_single_weight = early_amp_eff * early_polarity
        tail_weight_arr = np.where(~early_mask, uniform_single_weight, 0.0)

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
    gain_arr = 1.0 + validity_gain * (validities - 0.5)

    same_run_gain = float(parameters['same_run_gain'])

    tie_run_len = first_disc
    tie_run_flipped_weight = float(parameters['tie_run_flipped_weight'])

    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = int(min(len(past_a), len(past_b)))
    if n_past >= int(parameters['context_min_trials']):
        late_count = 0
        for i in range(n_past):
            d = np.asarray(past_a[i], dtype=float) - np.asarray(past_b[i], dtype=float)
            nz = np.flatnonzero(d != 0.0)
            if nz.size > 0 and int(nz[0]) >= 3:
                late_count += 1
        late_share = float(late_count) / float(n_past)
        context_active = late_share >= float(parameters['late_context_threshold'])
    else:
        context_active = False
    tie_run_reset = tie_run_len >= 3 and context_active

    early_evidence = 0.0
    tail_raw = 0.0
    reset_evidence = 0.0

    for pos in disc_positions:
        pos = int(pos)
        sign = float(signs[pos])
        gain = gain_arr[pos]
        if pos == first_disc and tie_run_reset:
            reset_weight = tie_run_flipped_weight
            if disc_positions.size == 1:
                reset_weight *= float(parameters['single_cue_reset_scale'])
            reset_evidence += reset_weight * gain * sign
            continue
        if pos < early_span:
            w = early_weight_arr[pos] * gain
            if pos == first_disc:
                w *= first_discount_factor
                if n_features >= 9 and first_disc in (1, 2) and configural_odd:
                    w *= float(parameters['long_delayed_first_boost'])
            elif sign == sign_first:
                w *= same_run_gain
            elif short_configural_opposing:
                w *= opposing_early_ratio
            early_evidence += w * sign
        else:
            tail_raw += tail_weight_arr[pos] * gain * sign

    block_compression = float(parameters['block_compression'])
    tail_raw = tail_raw * tail_polarity

    n9_tail_gain = float(parameters['n9_tail_gain'])
    n9_compression_relax = float(parameters['n9_compression_relax'])
    is_n9_tail_heavy = (
        n_features == 9
        and first_disc == 0
        and later_positions.size >= 3
    )
    if is_n9_tail_heavy:
        tail_raw = tail_raw * n9_tail_gain
        block_compression_eff = block_compression * n9_compression_relax
    else:
        block_compression_eff = block_compression

    tail_score = tail_raw / (1.0 + block_compression_eff * abs(tail_raw))

    long_config_tail_suppression = float(parameters['long_config_tail_suppression'])
    tail_suppression_two_cue = float(parameters['tail_suppression_two_cue'])
    if (
        n_features >= 8
        and first_disc == 0
        and later_positions.size >= 2
        and np.all(signs[later_positions] == -sign_first)
    ):
        if later_positions.size == 2 and n_features >= 8:
            tail_score = tail_score * tail_suppression_two_cue
        else:
            tail_score = tail_score * long_config_tail_suppression

    early_evidence += first_cue_boost * sign_first
    evidence = early_evidence + tail_score + reset_evidence

    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])
    odd_density_power = float(parameters['odd_density_power'])
    odd_count_decay = float(parameters['odd_count_decay'])
    tie_gap_penalty = float(parameters['tie_gap_penalty'])
    odd_length_center = float(parameters['odd_length_center'])
    odd_length_temp = float(parameters['odd_length_temp'])

    if n_features >= 8:
        odd_gate = 0.0
    else:
        z_odd = float(max(-30.0, min(30.0, (odd_length_center - float(n_features)) / odd_length_temp)))
        odd_gate = 1.0 / (1.0 + np.exp(z_odd))

    if configural_odd and not tie_run_reset and odd_gate > 1e-12:
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
            * gain_arr[first_disc]
            * sign_first
            * first_discount_factor
        )

        if (first_disc + 1 < n_features) and (not discrim[first_disc + 1]):
            evidence += (tie_gap_penalty - 1.0) * first_cue_raw
            odd_score *= tie_gap_penalty

        evidence += odd_score * sign_first

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

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts are used with the high-validity cue at display position 3 (list index 2, validity 0.91) so visual display order is not confounded with TTB cue order; TTB re-sorts it to first. Every trial has that top cue discriminating. The lower four cues are arranged to produce tallying margins d = wins(A) - wins(B) of -3, -2, -1, 0 in trials 1-4 and +3, +2, +1, 0 in the mirrored trials 5-8. Under TTB, P(A) should be flat within a top-cue direction: high for all top-cue favors A trials and low for all top-cue favors B trials, regardless of d. Under Tallying, P(A) should increase monotonically with d and stay at chance when d = 0. This creates opposite qualitative predictions at d = -3, -2, -1, +1, +2, +3, and a chance-versus-directional difference at d = 0. Eight unique pairs repeated twelve times give 96 trials per subject.

[1] Five experts are used. The highest-validity expert (91%) is placed at display index 2, so visual order is not confounded with Take The Best's validity order (which re-sorts to: index 2, then 0, then 4, then 1, then 3). Every trial is constructed so that the net feature tally is exactly balanced: A wins on exactly as many features as B wins, so Tallying predicts p(A)=0.5 on every trial. By contrast, Take The Best never tallies; it searches down the validity-ordered cue cascade and uses the first discriminating cue. In trials 1-2 the first discriminator is the top expert and favors A; trials 3-4 favor B; trials 5-6 tie the top expert and use the second expert (favoring A/B); trials 7-8 tie the top two cues and use the third expert (favoring A/B). Thus TTB predicts a high cluster for A-predicting cues and a low cluster for B-predicting cues, while Tallying predicts a flat line at chance. This isolates the core dissociation: balanced evidence produces guessing under tallying but one-reason directional choice under Take The Best, and it does so without repeating the already-explored monotone tally-margin design.

[2] This design dissociates display-order Take The Best (TTB) from Tallying by making the first left-to-right discriminator favor one option while the full feature tally favors the other option, or is exactly balanced. The five validities are non-monotonic across display positions ([0.55, 0.90, 0.80, 0.65, 0.60]), so the first discriminator's validity varies across trials. In pairs 1-6 and 9-10, the first discriminating feature favors one option while the tally margin has the opposite sign, producing opposite directional predictions. In pairs 7, 8, 11, and 12, the tally is exactly balanced (d = 0), so Tallying must predict chance regardless of beta or epsilon, whereas TTB predicts a directional choice based on the first discriminator. Pairs 5 vs 9 and 6 vs 10 hold the tally margin constant (±1) but move the first discriminator from a high-validity feature (0.80) to a low-validity feature (0.55), so TTB predicts different choice probabilities while Tallying predicts identical probabilities. This yields both qualitative directional dissociations and quantitative validity-modulation dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

[3] Eight experts with non-monotonic validities [0.53, 0.91, 0.82, 0.66, 0.59, 0.76, 0.71, 0.87]. The first three display positions form a first-discriminator validity staircase (0.53, 0.91, 0.82). Leading ties place the first discriminator at position 0, 1, or 2. The design is a 3 (first-discriminator validity) x 3 (tally margin d = +4/+2/0 for B-first; -4/-2/0 for A-first) x 2 (first-discriminator direction) factorial. Under Tallying, P(A) depends only on d: high for d=+4, above chance for d=+2, exactly chance for d=0, below chance for d=-2, and low for d=-4; it is identical across the three validity levels for a given d. Under display-order TTB, P(A) depends only on the first displayed discriminator's direction and validity: low for B-first (lowest for v=0.91, highest for v=0.53) and high for A-first, and it is flat across d. Thus all 18 pairs are dissociative: d!=0 trials give opposite directional predictions, and d=0 trials give chance (Tallying) versus directional (TTB). The factorial also provides quantitative tests: Tallying predicts a d effect with no validity effect, while TTB predicts a validity effect with no d effect. The 18 unique pairs are repeated 5 times for 90 trials, close to the 96-trial target.

[4] Five experts with validities [0.55, 0.85, 0.85, 0.85, 0.85]. The low-validity expert occupies display position 0; positions 1-3 are high-validity; position 4 is a high-validity filler that never discriminates. Every unique trial has exactly one discriminating feature, placed at display position p=0,1,2,3 by leading ties, with all later features tied. Thus the tally margin is always +/-1, so any flat-weight tallying process predicts identical P(A) across positions. The advocated display-order TTB predicts a flat P(A) across p=1-3 (same first-discriminator validity 0.85; about 0.93 for A-first and 0.07 for B-first) and a less decisive P(A) at p=0 (validity 0.55; about 0.74 for A-first and 0.26 for B-first). The competing primacy-gradient accumulator predicts a steep position decline: A-first P(A) about 0.94 at p=0, 0.86 at p=1, 0.73 at p=2, and 0.62 at p=3, with mirrored B-first values. This produces a crossover at p=0 versus p=1 and a flat-versus-declining dissociation across p=1-3. Eight unique pairs repeated 12 times give 96 trials.

[5] This design uses a long 12-feature list to push the advocated capacity-sensitive accumulator into its flat-weight regime. With n_features=12, the advocated capacity_factor is (5/12)^14 ≈ 5e-6, so the exponential position gradient and the validity_power modulation are both effectively zero: the model becomes a flat tally of discriminating features with beta≈4.5 and epsilon≈0.1. The competing display-order TTB, by contrast, always stops at the first displayed discriminator and ignores all later evidence, with validity weakly modulating decisiveness. The 12 unique pairs form a 2 (first-discriminator validity: 0.55 at position 0 vs 0.90 at position 1) × 3 (full tally margin) × 2 (first-discriminator direction) factorial. Crucially, for A-first trials the full tally is negative or zero, and for B-first trials the full tally is positive or zero, so the two theories never make the same directional prediction. The advocated model predicts P(A) ≈ 0.05, 0.06, 0.50, 0.94, 0.95 for d = -2, -1, 0, +1, +2, with no validity effect. Display-order TTB predicts P(A) ≈ 0.72-0.77 for low-validity A-first, ≈ 0.93-0.97 for high-validity A-first, and mirrored values for B-first, with no tally-margin effect. This yields both qualitative directional dissociations and quantitative validity-modulation versus tally-margin dissociations. The 12 unique pairs are repeated 8 times for 96 trials.

[6] Six experts with validities [0.50, 0.52, 0.54, 1.00, 0.95, 0.90] place three weak experts in the first three display positions and three strong experts in the last three. This n=6 list is the key: the competing capacity-sensitive accumulator has not fully flattened (capacity factor (5/6)^16 ≈ 0.054), so it retains a measurable left-to-right primacy gradient (weights ≈ 1.00, 0.96, 0.93, 0.89, 0.86, 0.83) while its validity modulation is negligible (ν ≈ 0.003). The advocated bounded-sampling accumulator samples all six discriminators (capacity ≈ 0.91) and weights evidence primarily by stated validity, with only a small first-cue bonus attenuated by exp(-(n-5)). Every unique trial is exactly tally-balanced: A wins three features and B wins three. A flat tallying rule therefore predicts chance on every trial. The competing theory, however, is not flat at n=6: its primacy gradient makes it favor whichever option wins the earlier positions. The advocated theory favors whichever option wins the high-validity later positions. Because the early positions are low-validity and the later positions high-validity, the two theories make opposite predictions. For example, when A wins the three high-validity later features and B wins the three low-validity early features (pair 1), the competing theory predicts P(A) ≈ 0.23 while the advocated theory predicts P(A) ≈ 0.70; the mirror pair gives 0.77 vs 0.30. The eight pairs vary which three positions A wins, so the competing theory's P(A) tracks the summed position weights of A's wins while the advocated theory's P(A) tracks the summed validity-based evidence. This yields a quantitative double dissociation: a crossover around chance for the extreme pairs and graded opposite ordering across the mixed pairs, rather than a mere overall accuracy difference.

[7] Seven features are used because the advocated accumulator's capacity factor (5/7)^(14-18) is approximately 0.002-0.009, which collapses its exponential position gradient and validity modulation to a near-equal tally. The competing sampler's capacity (5/7)^0.5 is approximately 0.845, so on every all-discriminating trial it samples 6 of the 7 cues with validity-weighted evidence and a negligible first-cue bonus. The first three experts are high-validity (0.95) and the last four are low-validity (0.55), so the competing sampler's position-weighted sampling and evidence weights favor the early high-validity block. Every trial makes all seven experts discriminate. The 8 pairs form a 2 (tally margin d = +1 or -1) x 4 (number of high-validity early wins for A, hA = 0..3) factorial, with mirrored pairs. Under the advocated theory, P(A) depends only on d: approximately 0.94 for all d=+1 pairs and approximately 0.06 for all d=-1 pairs, flat across hA. Under the competing theory, P(A) increases with hA within each d: roughly 0.38/0.53/0.74/0.86 for d=+1 and 0.70/0.48/0.26/0.14 for d=-1. Thus pair 1 gives opposite directional predictions (advocated A vs competing B), pair 5 gives the reverse (advocated B vs competing A), and the remaining pairs quantitatively dissociate the theories while holding the tally margin constant. This is not a repeat of the earlier n=6 balanced design because it uses n=7, high-validity early rather than late, and unbalanced 4-3 tallies, yielding opposite directional predictions rather than chance-vs-directional ones.

[8] n=9 design with six low-validity cues (0.55) interleaved with three high-validity cues (0.95, 0.90, 0.80). Every trial makes all nine features discriminate. In trials 1-6 the high-validity positions 3, 7, and 8 all favor B while k low-validity positions favor A and 6-k favor B; trials 7-12 are exact A/B mirrors. The competing theory at n=9 gives the 0.55 cues near-zero contrastive attention, so its normalized high-validity block dominates and P(A) is roughly flat near the high-cue direction across k. The advocated theory samples 7 of 9 discriminations, has a negligible first-cue bonus at n=9, and uses compressive but non-zero validity evidence; the six low cues can outweigh the three high cues, so P(A) increases strongly with k and crosses from B to A near k=4 or 5. Thus k=5 and k=6 give opposite directional predictions from the competing theory, and the mirrors reverse this. The full set provides a quantitative dissociation: a low-cue-count slope under the advocated theory versus a nearly flat high-cue-dominance curve under the competing theory. Twelve unique pairs repeated eight times give 96 trials.

[9] This 10-feature design makes the high-validity experts cancel while the six low-validity experts vary. Validities are [0.55, 0.95, 0.55, 0.90, 0.55, 0.85, 0.55, 0.80, 0.55, 0.55]. Four high-validity cues sit at positions 1, 3, 5, and 7 and are assigned 2-vs-2 across A and B (two counterbalanced high-pair orientations). Six 0.55 cues occupy positions 0, 2, 4, 6, 8, and 9, and the low-count l = number of low cues favoring A takes values 0, 1, 2, 4, 5, 6. Under the advocated theory, 0.55 cues receive near-zero contrastive attention relative to the strong cues, and the model normalizes by total attention, so the balanced strong block plus any low-cue configuration keeps P(A) near chance: approximately 0.47, 0.49, 0.50, 0.52, 0.53, 0.55 across l. Under the competing theory, validity is compressive rather than near-zero, low cues retain meaningful sampling and evidence weight, and sampled evidence is summed, so the balanced strong block cancels while the low cues accumulate and move P(A) steeply from about 0.09 at l = 0 to about 0.91 at l = 6. This yields both opposite directional predictions at the extreme low counts and a quantitative flat-versus-steep low-count dissociation. The two high-pair orientations are counterbalanced across the 12 unique pairs, which are repeated 8 times to give 96 trials.

[10] This design uses an untried four-feature list. At n_features=4 the advocated accumulator has a capacity factor (5/4)^0.5 > 1, so it samples every discriminating cue, but it still awards its large, unattenuated first-cue bonus to the first displayed discriminator. The competing accumulator has length_log=0 at n_features=4, so its logistic primacy gradient is not flattened and assigns a much larger weight to early positions than to late positions. The validities [0.95, 0.90, 0.70, 0.60] spread high and low values. In every unique pair the first displayed feature discriminates and has validity 0.95. Pairs 1-2 are the key k=3 pairs: the first cue favors one option and all three later cues favor the other. Under the advocated theory the large first-cue bonus plus the 0.95 cue outweighs the three lower-validity later cues, predicting a choice probability above 0.5 for the first-cue direction, approximately 0.65-0.75 for A-first trials. Under the competing theory the position-weighted evidence sum favors the later block because the position-1 0.90 cue alone receives a large primacy weight and positions 2-3 add further opposing evidence, predicting a choice probability below 0.5 for the first-cue direction. The mirrored pair reverses the prediction. Pairs 3-8 are k=2 pairs in which two later cues oppose the first cue and one later cue is tied, varying which later position is tied. The advocated theory predicts nearly constant P(A) for the three A-first k=2 variants because the first-cue bonus dominates and only the small validity-based evidence of the two opposing later cues changes. The competing theory predicts a steep position gradient: when the tied cue is position 1 the position-2 and position-3 opposition is weak and P(A) is very high, but when the tied cue is position 3 the high-weight position-1 cue opposes the first cue and P(A) drops toward or below chance. This yields a flat-versus-steep quantitative dissociation for k=2 and opposite directional predictions for k=3. The eight unique pairs are repeated twelve times for 96 trials.

[11] Short-list n=5 design that holds the competing sampler's capacity at (5/5)^0.5 = 1, so every discriminating cue is always sampled and its first-cue bonus is unattenuated because there is no n-5 penalty. The advocated accumulator has length_log = 0 at n=5, so its primacy gradient is steep and its first-discriminator shift is positive rather than negative. Validities are [0.60, 0.95, 0.55, 0.55, 0.55]: a moderately weak first cue, one high-validity second cue, and three matched low-validity tail cues. Every unique trial has exactly three discriminators. The first cue at position 0 with validity 0.60 favors one product; the high cue at position 1 with validity 0.95 always favors the other product; one of the matched low cues with validity 0.55 at position 2, 3, or 4 also favors that other product; the remaining cues are tied. Mirrors swap A and B. Under the advocated position-weighted tally, the later high-plus-low block outweighs the first cue plus its moderate first-cue bonus in all three positive trials, so P(A) is below chance; moving the low tail cue from position 2 to 3 and 4 shrinks its position weight, producing a graded increase in P(A) that stays below 0.5. Under the competing bounded-sampling accumulator, the first cue receives a large fixed first-cue bonus and all three discriminators are sampled, but evidence is not discounted by display position, so the matched low tail cue contributes equally at positions 2, 3, and 4. The first-cue direction wins, so P(A) is above chance and nearly flat across the three tail positions. Mirrors reverse both predictions. This gives both an opposite directional dissociation and a flat-versus-position-gradient quantitative dissociation. Six unique pairs repeated 16 times give 96 trials.

[12] This design uses an untried n=10 list and exploits the competing theory's tie-gap penalty and configural odd-one-out bonus, while holding the advocated theory's valence sum constant. Validities are fixed as [0.95, 0.70, 0.70, 0.70, 0.70, 0.55, 0.60, 0.55, 0.70, 0.70]. Every trial has exactly three discriminating cues: one first cue at position 0 with validity 0.95, and two opposing later cues with validity 0.70. The two later cues are placed in three arrangements: adjacent early at positions 1 and 2; after a one-position tie gap at positions 3 and 4; or far in the recency tail at positions 8 and 9. Mirrored versions swap A and B.

For the advocated bounded-sampling accumulator, n=10 gives capacity roughly (5/10)^0.45 to (5/10)^0.55, about 0.68 to 0.73. Because only three cues discriminate, sample size is ceil(capacity * 3) = 3 everywhere in the parameter range, so all discriminating cues are always sampled. The first-cue bonus contains exp(-first_cue_capacity_power * max(0, n-5)), so at n=10 it is effectively zero. The evidence contribution therefore depends only on validity: one 0.95 cue versus two 0.70 cues yields a net advantage for the two-cue option in all three arrangements. Crucially, moving the two 0.70 cues among positions 1-2, 3-4, and 8-9 should not change the advocated model's prediction because all cues are sampled and display position plays no role once the negligible bonus is removed.

The competing ordered-evidence accumulator makes the opposite structural prediction. At n=10 its length shift is near saturation, so positional and recency effects are strong. In the adjacent arrangement the first cue produces a configural odd-one-out pattern: two contiguous later cues oppose an isolated first cue at position 0, giving density 2/3 and a large odd-one-out bonus, so the first cue direction wins. In the gap arrangement, position 1 is tied, which activates the tie_gap_penalty; both the first cue's direct evidence and the odd-one-out bonus are multiplied by a factor of about 0.3 to 0.5. The two opposed cues now fall in the recency-amplified tail at positions 3 and 4, so the two-cue direction wins. The far arrangement places the opposed cues at positions 8 and 9, where recency tail weights are very large, again making the two-cue direction win. Thus the competing model flips from first-cue choice in the adjacent arrangement to two-cue choice in the gap and far arrangements, while the advocated model stays flat on the two-cue choice in all three arrangements. The mirrors reverse the first-cue direction. Six unique pairs repeated 16 times give 96 trials.

[13] Five-feature display with validities [0.52, 0.56, 0.60, 0.95, 0.95]. Positions 0-2 are tied on every trial, so the first discriminating cue is always in the tail at position 3 or 4. At n=5 the advocated ordered-evidence model's length shift is near zero: positions beyond the early span receive only a small baseline tail weight of roughly 0.33-0.36, and validity acts through a weak compressive gain. Therefore, when the two high-validity tail cues conflict, the advocated model predicts near-chance P(A), with a tiny recency advantage to the later tail cue at position 4; single late cues yield only moderate P(A) around 0.60. By contrast, the competing bounded-sampling model has capacity (5/5)^0.5 = 1 at n=5, so it samples every discriminator, and its first-cue bonus is unattenuated because there is no n-5 penalty. Even at position 3 or 4, this bonus plus validity-based evidence produces a strong first-tail-cue choice: about 0.80-0.90 for the conflict pairs and more extreme for same-direction and single-cue pairs. Thus the conflict pairs produce a directional dissociation around chance, and the remaining pairs provide a quantitative extremity dissociation. The eight unique pairs are repeated 12 times for 96 trials.

[14] This design targets the length-gating of the configural odd-one-out bonus, which is the sharpest structural difference between the two theories. At n_features = 6 the advocated theory's odd gate is already near floor, with odd_length_center around 5.5 and odd_length_temp around 0.17 giving a gate of roughly 0.04 to 0.08. The competing theory's gate is still large at n = 6, with center 6.5 and temp around 0.4 giving a gate of roughly 0.71 to 0.88. Every pair is a configural odd-one-out display: exactly three cues discriminate, the first displayed discriminator is isolated, and the two later discriminating cues uniformly oppose it. In four base arrangements the first discriminator is placed at index 0 or index 2 and the two opposing cues are adjacent or split across the early/tail boundary. Under the advocated theory the odd bonus is effectively silent, so choice follows the weighted sum of early and tail evidence. The selected validities make the two opposing cues outweigh the weak first cue, especially when the first cue sits at index 2 and receives the reduced third-position early ratio. The advocated theory therefore predicts choosing the later-block option in all pairs. Under the competing theory the still-active odd-one-out bonus, about 0.7 gate times a bonus of roughly 1.8 to 2.8, adds to the isolated first cue and flips the prediction to the first-cue option in all pairs. Mirrors reverse both predictions. The four arrangements also provide quantitative dissociation: adjacent later cues have high density and give the competing theory its largest first-cue preference, while split later cues reduce density and shrink that preference; under the advocated theory the later-block preference increases with tail recency weight and number of opposing cues. This creates opposite directional predictions plus different ordering across density and position conditions. The design does not repeat Experiment 6 because only three of six features discriminate and all remaining features are tied, rather than using all-discriminating tally-balanced lists.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design targets the competing theory's context-gated tie-run reset and the advocated theory's history-independent recency tail. Every unique trial begins with a three-position tie run, so the first discriminating cue is always late (list indices 3-6). Because every unique pair has this property, after the two-trial context minimum the competing model's late-context threshold is exceeded and its tie-run reset activates on all remaining trials. On these single-cue trials the competing model replaces the late first cue with a large negative flipped weight (tie_run_flipped_weight about -3 to -2.5, scaled by single_cue_reset_scale), so an A-first cue predicts P(A) below chance. The advocated model ignores history and weights the same single tail cue by its recency-increasing tail weight, so an A-first cue predicts P(A) above chance. The equal-validity adjacent pairs (indices 3 and 4 at validity 0.55; indices 5 and 6 at validity 0.95) give a position-signature test: the advocated model predicts an increasing first-cue choice with later list position, while the competing reset prediction depends only on validity and is flat across position. The validity contrast is also opposite: the advocated model predicts stronger A-choice for the high-validity late cues (indices 5-6) than for the low-validity earlier tail cues (indices 3-4), whereas the competing model predicts the high-validity cues produce a slightly more negative flipped evidence and therefore a lower first-cue P(A). Approximate post-context predictions for A-first trials are advocated P(A) around 0.67, 0.70, 0.75, and 0.78 for indices 3-6, while competing P(A) is around 0.28-0.30 and flat or slightly decreasing with the high-validity cues; B-first pairs mirror these. This yields both opposite directional predictions and crossed position/validity signatures. The eight pairs are repeated 12 times for 96 trials.",
  "validities": [
    0.6,
    0.72,
    0.65,
    0.95,
    0.93,
    0.89,
    0.78,
    0.9
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 14333,
  "output_tokens": 23696,
  "reasoning_tokens": 22788,
  "provider": "Baidu"
}
```
