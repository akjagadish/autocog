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
**Validities (n_features=5):** [0.74, 0.7, 0.95, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0]

**Rationale:** This n=5 design isolates the advocated theory's tie-gap penalty on a non-configural single opposing cue. Positions 0-2 form the early span in both theories, and position 4 is a low-validity tail. Every trial has exactly two discriminating cues: the first discriminating cue at early position p = 0, 1, or 2, an immediately following tied position producing a tie gap, and one opposing low-validity cue at position 4. Because the opposing cue validity 0.55 is lower than the first-cue validities 0.74, 0.70, and 0.95, the advocated theory's narrow tie-gap release does not apply, so the first cue is multiplied by tie_gap_penalty around 0.30-0.45. The early-position gradient then drops from position 0 to position 2, so advocated P(A) for A-first trials should fall from slightly above chance at p=0 to near chance at p=1 and below chance at p=2, reversing the first-cue direction at p=2. The competing theory has no tie-gap penalty for this non-configural single-opposing-cue pattern, so its first-cue evidence stays positive and P(A) remains first-cue directional at all p values, around 0.87 at p=0, 0.82 at p=1, and 0.63 at p=2. Thus p=2 yields opposite directional predictions, while p=0 and p=1 provide a quantitative crossing-versus-above-chance dissociation. Mirrored trials reverse the first-cue direction. The first-cue run lengths are 0, 1, and 2, so the competing context-gated reset never activates on these trials and the advocated delayed-first-cue mechanism is not involved.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make binary quality choices through a bounded ordered evidence-accumulation process with three interacting components. Early discriminating cues receive strong primacy, but that primacy is length-band-specific: on very short lists (n<=5) a narrow validity-order/tie-gap gate suppresses a single opposing late cue inside the early span, producing the strong anti-first-validity signature; on n=7 lists the early block is a subject-level mixture of a zero-early branch and a moderately positive branch; on n=8 lists the early block is subject-signed negative; and on long lists (n>=9) the position-1 early cue is discounted outside of two-cue configural displays, while configural late blocks are suppressed on very long displays. A weakly weighted late tally block accumulates ordinary later cues, while delayed first discriminating cues receive a context-graded negative weight, making isolated late cues fall near chance after delayed-first-cue experience. The configural odd-one-out bonus is moderate on short lists and nearly off by n=6, and long two-cue configural displays are explicitly equalized across adjacent and nonadjacent later-cue positions. Stated validity enters only as a compressive attentional gain. A tie-gap immediately after the first cue suppresses both the first cue's direct evidence and the configural bonus, except in the narrow validity-order/tie-gap case where it is released. A subject-level mirror asymmetry generates the positive mirror effect. Choices arise from a noisy softmax with subject scale, bias, trial noise, and lapse.

**Parameters:**
- early_weight: [2.20, 2.70]
- tail_weight: [0.45, 0.65]
- validity_gain: [0.08, 0.18]
- odd_one_out_bonus: [1.60, 2.40]
- block_compression: [0.03, 0.08]
- length_center: [7.20, 7.80]
- length_temp: [0.55, 0.85]
- early_damp: [0.90, 1.40]
- second_ratio: [0.74, 0.86]
- third_ratio: [0.28, 0.42]
- early_span: {3}
- isolated_scale: [0.15, 0.30]
- strong_first_discount: [0.45, 0.65]
- weak_first_discount: [0.03, 0.12]
- same_run_gain: [1.30, 1.70]
- recency_scale: [0.80, 1.20]
- recency_power: [0.60, 1.00]
- odd_density_power: [1.60, 2.30]
- odd_count_decay: [0.25, 0.60]
- tie_gap_penalty: [0.30, 0.45]
- opposing_early_ratio: [0.40, 0.60]
- odd_length_center: [5.00, 5.40]
- odd_length_temp: [0.30, 0.50]
- long_two_cue_early_suppression: [0.30, 0.70]
- tail_suppression_two_cue: [0.02, 0.10]
- long_early_polarity: [0.50, 1.00]
- mid_early_polarity: [-1.50, -0.30]
- n7_branch_selector: [0, 1]
- n7_early_zero_prob: [0.15, 0.35]
- n7_positive_polarity: [0.90, 1.30]
- long_list_threshold: {7}
- late_threshold: {3}
- late_anti_base: [0.80, 1.20]
- late_context_gain: [0.60, 1.30]
- late_length_scale: [0.20, 0.50]
- late_anti_unconditional_threshold: {4}
- late_context_position_threshold: {3}
- late_context_activation: [0.20, 0.35]
- context_min_trials: {2}
- context_window: {24}
- tail_context_gain: [0.00, 0.10]
- short_config_boost: [0.40, 0.80]
- short_single_opposing_ratio: [0.00, 0.06]
- long_pos1_ratio: [0.35, 0.60]
- mirror_asym: [1.02, 1.18]
- long_config_heavy_tail_suppression: [0.00, 0.15]
- beta: [0.85, 1.10]
- subject_scale: [0.70, 1.30]
- bias: [-0.05, 0.05]
- trial_noise_sd: [0.05, 0.15]
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

    validity_gain = float(parameters['validity_gain'])
    gain_arr = 1.0 + validity_gain * (validities - 0.5)

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

    early_polarity = 1.0
    long_list_threshold = int(parameters['long_list_threshold'])
    if n_features >= long_list_threshold:
        if n_features >= 9:
            early_polarity = float(parameters['long_early_polarity'])
        elif n_features == 7:
            zero_prob = float(parameters['n7_early_zero_prob'])
            branch = float(parameters['n7_branch_selector'])
            if branch < zero_prob:
                early_polarity = 0.0
            else:
                early_polarity = float(parameters['n7_positive_polarity'])
        else:
            early_polarity = float(parameters['mid_early_polarity'])
        early_weight_arr = early_weight_arr * early_polarity

    long_two_cue_config = (
        n_features >= 8
        and first_disc == 0
        and configural_odd
        and later_positions.size == 2
    )
    long_two_cue_early_suppression = float(parameters['long_two_cue_early_suppression'])
    if long_two_cue_config:
        for pp in later_positions:
            pp = int(pp)
            if pp < early_span:
                early_weight_arr[pp] = early_weight_arr[pp] * long_two_cue_early_suppression

    tail_weight = float(parameters['tail_weight'])
    recency_scale = float(parameters['recency_scale'])
    recency_power = float(parameters['recency_power'])
    recency_amp = recency_scale * length_shift
    positions = np.arange(n_features, dtype=float)
    norm_pos = positions / max(1, n_features - 1)
    tail_weight_arr = np.where(
        ~early_mask,
        tail_weight + recency_amp * np.power(norm_pos, recency_power),
        0.0
    )

    late_threshold = int(parameters['late_threshold'])
    context_min_trials = int(parameters['context_min_trials'])
    context_window = int(parameters['context_window'])

    context_strength = 0.0
    past_a = history.get('option_a_ratings', [])
    past_b = history.get('option_b_ratings', [])
    n_past = int(min(len(past_a), len(past_b)))
    if n_past >= context_min_trials:
        start = max(0, n_past - context_window)
        delayed_count = 0
        total_count = 0
        for i in range(start, n_past):
            d = np.asarray(past_a[i], dtype=float) - np.asarray(past_b[i], dtype=float)
            nz = np.flatnonzero(d != 0.0)
            if nz.size > 0:
                if int(nz[0]) >= late_threshold:
                    delayed_count += 1
                total_count += 1
        if total_count > 0:
            context_strength = delayed_count / float(total_count)

    late_anti_base = float(parameters['late_anti_base'])
    late_context_gain = float(parameters['late_context_gain'])
    late_length_scale = float(parameters['late_length_scale'])
    late_anti_unconditional = int(parameters['late_anti_unconditional_threshold'])
    late_context_position = int(parameters['late_context_position_threshold'])
    late_context_activation = float(parameters['late_context_activation'])

    late_first_anti = (
        late_anti_base
        * (1.0 + late_length_scale * length_shift)
        * (1.0 + late_context_gain * context_strength)
    )

    tail_context_gain = float(parameters['tail_context_gain'])
    tail_context_mult = 1.0 - tail_context_gain * context_strength
    tail_context_mult = float(max(0.4, tail_context_mult))
    tail_weight_arr = tail_weight_arr * tail_context_mult

    strong_first_discount = float(parameters['strong_first_discount'])
    weak_first_discount = float(parameters['weak_first_discount'])
    first_discount_factor = 1.0
    if n_features >= 7:
        if configural_odd:
            first_discount_factor = 1.0 - weak_first_discount * length_shift
        else:
            first_discount_factor = 1.0 - strong_first_discount * length_shift
        first_discount_factor = float(max(0.35, first_discount_factor))

    mirror_asym = float(parameters['mirror_asym'])
    if sign_first > 0.0:
        first_sign_mult = 1.0 / mirror_asym
    else:
        first_sign_mult = mirror_asym

    same_run_gain = float(parameters['same_run_gain'])
    tie_gap_penalty = float(parameters['tie_gap_penalty'])
    tie_gap = (first_disc + 1 < n_features) and (not discrim[first_disc + 1])

    isolated_high_validity_opposing = (
        n_features <= 5
        and later_positions.size == 1
        and first_disc < early_span
        and signs[int(later_positions[0])] != sign_first
        and validities[int(later_positions[0])] >= validities[first_disc]
        and tie_gap
    )
    short_single_opposing_ratio = float(parameters['short_single_opposing_ratio'])

    long_pos1_ratio = float(parameters['long_pos1_ratio'])

    early_evidence = 0.0
    tail_raw = 0.0

    for pos in disc_positions:
        pos = int(pos)
        sign = float(signs[pos])
        gain = gain_arr[pos]

        if pos == first_disc:
            if first_disc < late_context_position:
                w = early_weight_arr[pos] * gain * first_discount_factor * first_sign_mult
                if tie_gap and not isolated_high_validity_opposing:
                    w = w * tie_gap_penalty
                early_evidence += w * sign
            elif first_disc == late_context_position and context_strength < late_context_activation:
                tail_raw += tail_weight_arr[pos] * gain * sign
                continue
            else:
                w = -late_first_anti * gain * first_sign_mult
                early_evidence += w * sign
        else:
            if pos < early_span:
                w = early_weight_arr[pos] * gain
                if pos == int(later_positions[0]) and isolated_high_validity_opposing:
                    w = w * short_single_opposing_ratio
                if n_features >= 9 and not long_two_cue_config and pos == 1 and pos != first_disc:
                    w = w * long_pos1_ratio
                if sign == sign_first:
                    w = w * same_run_gain
                elif short_configural_opposing:
                    w = w * opposing_early_ratio
                early_evidence += w * sign
            else:
                tail_contrib = tail_weight_arr[pos] * gain * sign
                if pos == int(later_positions[0]) and isolated_high_validity_opposing:
                    tail_contrib = tail_contrib * short_single_opposing_ratio
                tail_raw += tail_contrib

    short_config_boost = float(parameters['short_config_boost'])
    if n_features == 5 and first_disc == 0 and configural_odd and later_positions.size >= 2:
        early_evidence += short_config_boost * sign_first

    block_compression = float(parameters['block_compression'])
    tail_score = tail_raw / (1.0 + block_compression * abs(tail_raw))

    tail_suppression_two_cue = float(parameters['tail_suppression_two_cue'])
    if long_two_cue_config:
        tail_score = tail_score * tail_suppression_two_cue

    long_config_heavy_tail_suppression = float(parameters['long_config_heavy_tail_suppression'])
    if n_features >= 11 and configural_odd:
        tail_score = tail_score * long_config_heavy_tail_suppression

    evidence = early_evidence + tail_score

    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])
    odd_density_power = float(parameters['odd_density_power'])
    odd_count_decay = float(parameters['odd_count_decay'])
    odd_length_center = float(parameters['odd_length_center'])
    odd_length_temp = float(parameters['odd_length_temp'])

    if n_features >= 8:
        odd_gate = 0.0
    else:
        z_odd = float(max(-30.0, min(30.0, (odd_length_center - float(n_features)) / odd_length_temp)))
        if z_odd >= 0.0:
            odd_gate = 1.0 / (1.0 + np.exp(-z_odd))
        else:
            ez_odd = np.exp(z_odd)
            odd_gate = ez_odd / (1.0 + ez_odd)

    if configural_odd and odd_gate > 1e-12:
        span = float(later_positions[-1] - first_disc + 1)
        n_opposing = float(later_positions.size)
        density = n_opposing / span
        odd_score = (
            odd_one_out_bonus
            * (density ** odd_density_power)
            * np.exp(-odd_count_decay * max(0.0, n_opposing - 2.0))
            * odd_gate
        )
        if tie_gap:
            odd_score = odd_score * tie_gap_penalty
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    a = np.asarray([list(x) for x in data['option_a_ratings']], dtype=np.float64)\n    b = np.asarray([list(x) for x in data['option_b_ratings']], dtype=np.float64)\n    diff = a - b\n    disc = diff != 0.0\n    first_idx = np.argmax(disc, axis=1)\n    rows = np.arange(diff.shape[0])\n    first_diff = diff[rows, first_idx]\n    first_cue_response = np.where(first_diff > 0.0, 0, 1)\n    chosen_first = (np.asarray(data['response'], dtype=int) == first_cue_response)\n    return float(np.mean(chosen_first) - 0.5)",
  "rationale": "Every trial in this design has exactly two discriminating cues: an early first cue at position 0, 1, or 2 followed by a tie gap, and one opposing low-validity tail cue at position 4. The advocated theory applies a tie-gap penalty to the first cue on these non-configural single-opposing-cue displays, shrinking it so much that at position 2 the tail cue can reverse the first-cue direction and overall first-cue following stays near chance. The competing theory does not apply a tie-gap penalty here, so its strong early block keeps first-cue following clearly above chance at every position. The metric therefore pools all 96 trials per subject, balanced across A-first and B-first mirror pairs, and returns the excess rate of choosing the option favored by the first discriminating cue over chance. This should be near zero or slightly negative under the advocated theory and strongly positive under the competing theory, with low per-subject variance because it averages many binary responses."
}
```

## Usage

```json
{
  "input_tokens": 11180,
  "output_tokens": 18283,
  "reasoning_tokens": 17869,
  "provider": "Baidu"
}
```
