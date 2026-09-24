# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=6):** [0.5, 0.95, 0.5, 0.95, 0.9, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 8: A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]

**Rationale:** This design targets the length-gating of the configural odd-one-out bonus, which is the sharpest structural difference between the two theories. At n_features = 6 the advocated theory's odd gate is already near floor, with odd_length_center around 5.5 and odd_length_temp around 0.17 giving a gate of roughly 0.04 to 0.08. The competing theory's gate is still large at n = 6, with center 6.5 and temp around 0.4 giving a gate of roughly 0.71 to 0.88. Every pair is a configural odd-one-out display: exactly three cues discriminate, the first displayed discriminator is isolated, and the two later discriminating cues uniformly oppose it. In four base arrangements the first discriminator is placed at index 0 or index 2 and the two opposing cues are adjacent or split across the early/tail boundary. Under the advocated theory the odd bonus is effectively silent, so choice follows the weighted sum of early and tail evidence. The selected validities make the two opposing cues outweigh the weak first cue, especially when the first cue sits at index 2 and receives the reduced third-position early ratio. The advocated theory therefore predicts choosing the later-block option in all pairs. Under the competing theory the still-active odd-one-out bonus, about 0.7 gate times a bonus of roughly 1.8 to 2.8, adds to the isolated first cue and flips the prediction to the first-cue option in all pairs. Mirrors reverse both predictions. The four arrangements also provide quantitative dissociation: adjacent later cues have high density and give the competing theory its largest first-cue preference, while split later cues reduce density and shrink that preference; under the advocated theory the later-block preference increases with tail recency weight and number of opposing cues. This creates opposite directional predictions plus different ordering across density and position conditions. The design does not repeat Experiment 6 because only three of six features discriminate and all remaining features are tied, rather than using all-discriminating tally-balanced lists.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    a = np.stack(data['option_a_ratings'].to_numpy()).astype(float)\n    b = np.stack(data['option_b_ratings'].to_numpy()).astype(float)\n    diff = a - b\n    discrim = diff != 0.0\n    first_pos = np.argmax(discrim, axis=1)\n    rows = np.arange(len(data))\n    first_sign = np.sign(diff[rows, first_pos])\n    expected_first_cue = np.where(first_sign > 0.0, 0, 1)\n    first_cue_match = (data['response'].to_numpy() == expected_first_cue).astype(float)\n    n_disc = discrim.sum(axis=1)\n    focal = ((first_pos == 0)\n             & (n_disc == 3)\n             & discrim[:, 1]\n             & discrim[:, 2]\n             & (np.sign(diff[:, 1]) == -first_sign)\n             & (np.sign(diff[:, 2]) == -first_sign))\n    if focal.sum() == 0 or (~focal).sum() == 0:\n        return 0.0\n    return float(first_cue_match[focal].mean() - first_cue_match[~focal].mean())",
  "rationale": "The metric isolates the high-density adjacent configural odd-one-out trial type: first discriminating cue at position 0, with two opposing cues immediately at positions 1 and 2. In this focal type the competing theory's large odd-one-out bonus strongly favors the first cue, whereas the advocated theory gives little or no first-cue advantage. Subtracting the average first-cue matching rate in the remaining configural trials removes each subject's general bias and scale effects, producing a tight between-subject contrast while retaining a large between-theory difference."
}
```

## Usage

```json
{
  "input_tokens": 9735,
  "output_tokens": 37062,
  "reasoning_tokens": 36632,
  "provider": "Baidu"
}
```
