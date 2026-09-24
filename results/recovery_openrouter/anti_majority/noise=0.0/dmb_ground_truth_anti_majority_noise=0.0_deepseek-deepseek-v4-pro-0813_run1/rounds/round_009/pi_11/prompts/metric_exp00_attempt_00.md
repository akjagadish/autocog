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
**Validities (n_features=10):** [0.95, 0.9, 0.9, 0.8, 0.7, 0.6, 0.7, 0.9, 0.6, 0.95]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 2: A=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1]

**Rationale:** n=10 non-configural displays target a parameter split that is specific to the advocated model: its n10 first-position discount, n10 high-cue discount, n10 same-run gain, and n10 tail damping are absent from the competing model. Every pair has the first two displayed cues discriminating in the same direction and exactly one later discriminating cue in the opposite direction; all remaining cues are tied. Thus the first discriminator is always position 0, there is no tie gap because position 1 discriminates immediately, no configural odd-one-out pattern because position 1 agrees with position 0, and no delayed-first-cue context builds because all first discriminators are early. Under the advocated model the first cue is multiplied by n10_first_pos0_discount and the same-direction position-1 cue is nearly erased by n10_high_cue_discount times n10_pos1_same_run_gain, so the early block contributes very little; the single opposing cue dominates, giving a later/opposing-option choice. Under the competing model the same early block is strong: position 1 receives long_pos1_ratio times same_run_gain, so the two early cues outweigh a single later opposing cue, giving an early-pair choice. Moving the opposing cue across position 2, position 3, and position 4 provides a quantitative dissociation: the competing model's early-direction preference should be largest at position 2 and shrink toward chance at position 4, while the advocated model's opposing-direction preference should grow as the late cue moves into the recency-weighted tail. Mirrored A/B pairs cancel color bias.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make binary quality choices through run-gated ordered evidence accumulation in which the first discriminating cue's weight is a credibility signal shaped by local cue-run structure, list length, and recent delayed-cue context. A tie-gap immediately after the first cue suppresses that cue unless a validity-order gate releases it. In n=5 single-tail-opponent displays, a recent context of repeated tie-gap/lone-tail-opponent trials re-applies the suppression and adds a position-graded anti-following reversal. On n=10 non-configural displays, late evidence is damped so variance explained by low-cue counts shrinks. Short-list configural odd-one-out bonus, n=8 negative early polarity, long two-cue equalization, compressive validity gain, mirror asymmetry, and noisy softmax with lapse remain intact.

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
- same_run_gain: [1.50, 2.00]
- recency_scale: [0.80, 1.20]
- recency_power: [0.60, 1.00]
- odd_density_power: [1.60, 2.30]
- odd_count_decay: [0.25, 0.60]
- tie_gap_penalty: [0.10, 0.25]
- short_multi_tie_relief: [0.15, 0.30]
- exp18_reversal_weight: [2.20, 2.80]
- exp18_pos_ratio: [0.40, 0.70]
- opposing_early_ratio: [0.40, 0.60]
- n6_opposing_ratio: [0.82, 0.98]
- odd_length_center: [4.80, 5.20]
- odd_length_temp: [0.30, 0.50]
- long_two_cue_early_suppression: [0.30, 0.70]
- long_two_cue_later_weight: [0.20, 0.50]
- tail_suppression_two_cue: [0.02, 0.10]
- long_early_polarity: [0.45, 0.80]
- mid_early_polarity: [-1.20, -0.50]
- n7_branch_selector: [0, 1]
- n7_early_zero_prob: [0.00, 0.06]
- n7_positive_polarity: [1.30, 1.80]
- long_list_threshold: {7}
- late_threshold: {3}
- late_anti_base: [0.75, 1.10]
- late_context_gain: [0.60, 1.05]
- late_length_scale: [0.20, 0.50]
- late_context_position_threshold: {3}
- late_context_activation: [0.30, 0.45]
- context_min_trials: {2}
- context_window: {24}
- reversal_context_window: {5}
- reversal_context_threshold: {3}
- tail_context_gain: [0.00, 0.10]
- short_config_boost: [0.40, 0.80]
- short_single_opposing_ratio: [0.50, 0.80]
- long_pos1_ratio: [0.20, 0.45]
- first_pos1_discount: [0.40, 0.65]
- n10_first_pos1_discount: [0.00, 0.10]
- n10_high_cue_discount: [0.00, 0.05]
- n10_pos1_same_run_gain: [0.10, 0.40]
- n12_first_pos1_discount: [0.45, 0.70]
- n10_first_pos0_discount: [0.35, 0.60]
- n10_tail_damp: [0.55, 0.85]
- single_pos0_boost: [1.20, 1.50]
- mirror_asym: [1.08, 1.26]
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

    if disc_positions.size == 1:
        early_ratio = np.ones(n_features, dtype=float)

    later_positions = disc_positions[1:]
    configural_odd = (
        later_positions.size >= 2
        and first_disc < early_span
        and np.all(signs[later_positions] == -sign_first)
    )

    opposing_early_ratio = float(parameters['opposing_early_ratio'])
    n6_opposing_ratio = float(parameters['n6_opposing_ratio'])
    short_configural_opposing = (
        n_features <= 6
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
        elif n_features == 8:
            early_polarity = float(parameters['mid_early_polarity'])
        else:
            early_polarity = 1.0
        early_weight_arr = early_weight_arr * early_polarity

    long_two_cue_config = (
        n_features >= 8
        and first_disc == 0
        and configural_odd
        and later_positions.size == 2
    )
    long_two_cue_equal = long_two_cue_config and n_features >= 9
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

    n10_tail_damp = float(parameters['n10_tail_damp'])
    if n_features == 10 and not long_two_cue_config:
        tail_weight_arr = tail_weight_arr * n10_tail_damp

    if disc_positions.size == 1:
        uniform_single_weight = early_amp_eff * early_polarity
        tail_weight_arr = np.where(~early_mask, uniform_single_weight, 0.0)

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

    short_multi_tie_relief = float(parameters['short_multi_tie_relief'])
    tie_gap_penalty_eff = tie_gap_penalty
    if long_two_cue_config:
        tie_gap_penalty_eff = 1.0
    elif (
        tie_gap
        and n_features <= 5
        and first_disc == 0
        and configural_odd
        and later_positions.size >= 2
    ):
        tie_gap_penalty_eff = min(0.95, tie_gap_penalty + short_multi_tie_relief)

    release_penalty = False
    isolated_high_validity_opposing = False
    if tie_gap and first_disc < early_span and later_positions.size == 1 and n_features <= 5:
        lp = int(later_positions[0])
        if signs[lp] != sign_first and validities[lp] >= validities[first_disc]:
            isolated_high_validity_opposing = True
    release_penalty = isolated_high_validity_opposing

    exp18_candidate = (
        n_features == 5
        and first_disc < late_context_position
        and tie_gap
        and later_positions.size == 1
        and int(later_positions[0]) >= early_span
        and not configural_odd
    )

    reversal_context_window = int(parameters['reversal_context_window'])
    reversal_context_threshold = int(parameters['reversal_context_threshold'])
    reversal_context_active = False
    if exp18_candidate and n_past >= reversal_context_window:
        start = max(0, n_past - reversal_context_window)
        signature_count = 0
        for i in range(start, n_past):
            d = np.asarray(past_a[i], dtype=float) - np.asarray(past_b[i], dtype=float)
            nz = np.flatnonzero(d != 0.0)
            if nz.size != 2:
                continue
            f = int(nz[0])
            l = int(nz[1])
            if (
                f < late_context_position
                and f + 1 < int(d.size)
                and d[f + 1] == 0.0
                and l >= early_span
                and np.sign(d[f]) != np.sign(d[l])
            ):
                signature_count += 1
        if signature_count >= reversal_context_threshold:
            reversal_context_active = True

    exp18_penalty = exp18_candidate and reversal_context_active
    exp18_reversal = exp18_candidate and reversal_context_active and not release_penalty

    short_single_opposing_ratio = float(parameters['short_single_opposing_ratio'])
    long_pos1_ratio = float(parameters['long_pos1_ratio'])
    first_pos1_discount = float(parameters['first_pos1_discount'])
    n10_first_pos1_discount = float(parameters['n10_first_pos1_discount'])
    n10_high_cue_discount = float(parameters['n10_high_cue_discount'])
    n10_pos1_same_run_gain = float(parameters['n10_pos1_same_run_gain'])
    n12_first_pos1_discount = float(parameters['n12_first_pos1_discount'])
    n10_first_pos0_discount = float(parameters['n10_first_pos0_discount'])
    single_pos0_boost = float(parameters['single_pos0_boost'])
    long_two_cue_later_weight = float(parameters['long_two_cue_later_weight'])

    early_evidence = 0.0
    tail_raw = 0.0

    for pos in disc_positions:
        pos = int(pos)
        sign = float(signs[pos])
        gain = gain_arr[pos]

        if long_two_cue_equal and pos != first_disc and pos in later_positions:
            w = long_two_cue_later_weight
            early_evidence += w * sign
            continue

        if pos == first_disc:
            if first_disc < late_context_position:
                w = early_weight_arr[pos] * gain * first_discount_factor * first_sign_mult
                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:
                    if n_features == 10:
                        w = w * n10_first_pos1_discount
                    elif n_features >= 12:
                        w = w * n12_first_pos1_discount
                    else:
                        w = w * first_pos1_discount
                if first_disc == 0 and n_features == 10 and not long_two_cue_config:
                    w = w * n10_first_pos0_discount
                if disc_positions.size == 1 and first_disc == 0 and n_features == 5:
                    w = w * single_pos0_boost
                if tie_gap and (exp18_penalty or not release_penalty):
                    w = w * tie_gap_penalty_eff
                early_evidence += w * sign
            elif first_disc == late_context_position and context_strength < late_context_activation:
                tail_raw += tail_weight_arr[pos] * gain * sign
                continue
            else:
                w = -late_first_anti * gain * first_sign_mult
                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:
                    if n_features == 10:
                        w = w * n10_first_pos1_discount
                    elif n_features >= 12:
                        w = w * n12_first_pos1_discount
                    else:
                        w = w * first_pos1_discount
                early_evidence += w * sign
        else:
            if pos < early_span:
                w = early_weight_arr[pos] * gain
                if pos == 1 and n_features >= 9 and not long_two_cue_config and pos != first_disc:
                    w = w * long_pos1_ratio
                if n_features == 10 and not long_two_cue_config and pos == 1 and pos != first_disc:
                    w = w * n10_high_cue_discount
                if pos == int(later_positions[0]) and isolated_high_validity_opposing:
                    w = w * short_single_opposing_ratio
                if sign == sign_first:
                    if n_features == 10 and pos == 1 and not long_two_cue_config:
                        w = w * n10_pos1_same_run_gain
                    else:
                        w = w * same_run_gain
                elif short_configural_opposing:
                    if n_features == 6:
                        w = w * n6_opposing_ratio
                    else:
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
            odd_score = odd_score * tie_gap_penalty_eff
        evidence += odd_score * sign_first

    exp18_reversal_weight = float(parameters['exp18_reversal_weight'])
    exp18_pos_ratio = float(parameters['exp18_pos_ratio'])
    if exp18_reversal:
        pos_scale = 1.0 if first_disc == 0 else exp18_pos_ratio
        evidence -= exp18_reversal_weight * pos_scale * sign_first

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
  "metric_source": "import numpy as np\n\n\ndef metric(data):\n    outcomes = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=float)\n        b = np.asarray(row['option_b_ratings'], dtype=float)\n        if a.size == 0 or a.shape != b.shape:\n            continue\n\n        diff = a - b\n        disc = np.flatnonzero(diff != 0.0)\n        if disc.size == 0:\n            continue\n\n        first = int(disc[0])\n        first_direction = np.sign(diff[first])\n\n        response = int(row['response'])\n        chose_early = 1 if ((first_direction > 0 and response == 0) or\n                            (first_direction < 0 and response == 1)) else 0\n        outcomes.append(chose_early)\n\n    n = len(outcomes)\n    if n == 0:\n        return 0.0\n\n    k = int(np.sum(outcomes))\n    p_adj = (k + 0.5) / (n + 1.0)\n    return float(np.log(p_adj / (1.0 - p_adj)))",
  "rationale": "On every trial the first two discriminating cues point to the same option and exactly one later cue points the opposite way. The advocated model heavily discounts the n=10 first-position cue and nearly erases the same-run position-1 cue, so the lone later opposing cue drives choice; the competing model gives the first two same-direction cues a strong same-run boost and therefore produces more early-consistent choices. Computing the empirical log-odds of choosing the option favored by the first two discriminating cues converts this dissociation into a sign contrast: advocated data should sit below zero and competing data near or above zero. Each subject contributes 96 randomized trials, so the per-subject estimate is stable and the metric remains tight across subjects."
}
```

## Usage

```json
{
  "input_tokens": 13089,
  "output_tokens": 26476,
  "reasoning_tokens": 26031,
  "provider": "Baidu"
}
```
