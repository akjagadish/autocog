# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_8") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_8" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
People make binary choices by stochastically sampling a bounded, capacity-scaled subset of discriminating cues and accumulating signed evidence. Cue sampling is governed jointly by a moderate serial-position primacy gradient that flattens on longer lists and by stated cue validity, whose influence on evidence is compressive rather than all-or-none. The first discriminating cue receives a serial-position-sensitive bonus: early positions carry a calibrated primacy boost, while later first-cues receive only a modest residual bonus, so behavior is neither deterministic first-cue commitment nor a pure equal-weight tally. On longer lists, some subjects carry partially anti-cue weights, producing subject-level reversals of first-cue effects. Accumulated evidence passes through a noisy softmax with lapse, preserving between-subject heterogeneity.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_8
People make binary quality choices by salience-weighted ordered evidence accumulation. Display position is primary: on short lists the first displayed discriminating cue receives strong weight, and nearby same-sign cues form an early run, while an odd-one-out/block-contrast bonus amplifies a first cue that conflicts with an otherwise uniform later block. Stated validity enters only as a compressive attentional gain, never as a lexicographic multiplier. Later opposing cues accumulate as a count-sensitive but compressively weighted block. The configural odd-one-out bonus is sharply length-gated so it is near full strength on short displays (n=4-6) and essentially off on long displays (n>=8). On long displays a subject-level trait controls whether any residual configural integration survives list-length pressure: some subjects retain a partial odd-one-out bonus even on long lists while others rely wholly on the tally tail, which produces the near-zero mean with large between-subject variance observed in Experiment 6. Opposing cues inside the early span are strongly discounted, and a tie-gap immediately after the first cue additionally suppresses both the first cue's direct evidence and the configural bonus, generating the negative R_near - R_all signature (Experiment 11) and the strong mirror asymmetry (Experiment 12).

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0.0 or not np.isfinite(total):
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=10):** [0.95, 0.7, 0.7, 0.7, 0.7, 0.75, 0.6, 0.75, 0.7, 0.7]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design uses an untried n=10 list and exploits the competing theory's tie-gap penalty and configural odd-one-out bonus, while holding the advocated theory's valence sum constant. Validities are fixed as [0.95, 0.70, 0.70, 0.70, 0.70, 0.55, 0.60, 0.55, 0.70, 0.70]. Every trial has exactly three discriminating cues: one first cue at position 0 with validity 0.95, and two opposing later cues with validity 0.70. The two later cues are placed in three arrangements: adjacent early at positions 1 and 2; after a one-position tie gap at positions 3 and 4; or far in the recency tail at positions 8 and 9. Mirrored versions swap A and B.

For the advocated bounded-sampling accumulator, n=10 gives capacity roughly (5/10)^0.45 to (5/10)^0.55, about 0.68 to 0.73. Because only three cues discriminate, sample size is ceil(capacity * 3) = 3 everywhere in the parameter range, so all discriminating cues are always sampled. The first-cue bonus contains exp(-first_cue_capacity_power * max(0, n-5)), so at n=10 it is effectively zero. The evidence contribution therefore depends only on validity: one 0.95 cue versus two 0.70 cues yields a net advantage for the two-cue option in all three arrangements. Crucially, moving the two 0.70 cues among positions 1-2, 3-4, and 8-9 should not change the advocated model's prediction because all cues are sampled and display position plays no role once the negligible bonus is removed.

The competing ordered-evidence accumulator makes the opposite structural prediction. At n=10 its length shift is near saturation, so positional and recency effects are strong. In the adjacent arrangement the first cue produces a configural odd-one-out pattern: two contiguous later cues oppose an isolated first cue at position 0, giving density 2/3 and a large odd-one-out bonus, so the first cue direction wins. In the gap arrangement, position 1 is tied, which activates the tie_gap_penalty; both the first cue's direct evidence and the odd-one-out bonus are multiplied by a factor of about 0.3 to 0.5. The two opposed cues now fall in the recency-amplified tail at positions 3 and 4, so the two-cue direction wins. The far arrangement places the opposed cues at positions 8 and 9, where recency tail weights are very large, again making the two-cue direction win. Thus the competing model flips from first-cue choice in the adjacent arrangement to two-cue choice in the gap and far arrangements, while the advocated model stays flat on the two-cue choice in all three arrangements. The mirrors reverse the first-cue direction. Six unique pairs repeated 16 times give 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Targets the only structural disagreement in this design: the adjacent arrangement (discriminating positions 0,1,2) makes the competing model choose the first cue via its odd-one-out bonus, while the advocated model still chooses the two-cue side; in the gap and far arrangements both theories choose the two-cue side. For each trial I locate the leftmost discriminating cue, code whether the response followed that cue, and contrast the adjacent first-cue rate against the pooled gap/far first-cue rate. Under the advocated model the first-cue rate is the same across arrangements because all three discriminating cues are sampled and positional effects are negligible, so this contrast is near zero for every subject (tight variance). Under the competing model the adjacent arrangement has a large first-cue advantage, so the contrast is strongly positive. The metric uses 32 adjacent and 64 baseline trials per subject, keeping per-subject binomial noise small.

Source:
import numpy as np

def metric(data):
    if data.empty:
        return float('nan')
    a = np.stack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.stack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b
    choose_first = np.zeros(len(data), dtype=float)
    is_adj = np.zeros(len(data), dtype=bool)
    is_nonadj = np.zeros(len(data), dtype=bool)
    y = data['response'].to_numpy(dtype=float)
    for i in range(len(data)):
        pos = np.flatnonzero(diff[i] != 0.0)
        if pos.size == 0:
            continue
        first = int(pos[0])
        side_first = 0 if diff[i, first] > 0.0 else 1
        choose_first[i] = 1.0 if y[i] == side_first else 0.0
        p = tuple(pos.tolist())
        if p == (0, 1, 2):
            is_adj[i] = True
        elif p == (0, 3, 4) or p == (0, 8, 9):
            is_nonadj[i] = True
    if not is_adj.any() or not is_nonadj.any():
        return float('nan')
    return float(choose_first[is_adj].mean() - choose_first[is_nonadj].mean())


### RESULTS
- Predicted under pi_5 (simulated): -0.0009 (var=0.1257)
- Predicted under pi_8 (simulated): 0.3244 (var=0.0219)
- Observed on real data: 0.0231 (var=0.0947)

## EXPERIMENT 2 (proposed by pi_8)

### DESIGN
**Validities (n_features=5):** [0.6, 0.65, 0.7, 0.95, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** Five-feature display with validities [0.52, 0.56, 0.60, 0.95, 0.95]. Positions 0-2 are tied on every trial, so the first discriminating cue is always in the tail at position 3 or 4. At n=5 the advocated ordered-evidence model's length shift is near zero: positions beyond the early span receive only a small baseline tail weight of roughly 0.33-0.36, and validity acts through a weak compressive gain. Therefore, when the two high-validity tail cues conflict, the advocated model predicts near-chance P(A), with a tiny recency advantage to the later tail cue at position 4; single late cues yield only moderate P(A) around 0.60. By contrast, the competing bounded-sampling model has capacity (5/5)^0.5 = 1 at n=5, so it samples every discriminator, and its first-cue bonus is unattenuated because there is no n-5 penalty. Even at position 3 or 4, this bonus plus validity-based evidence produces a strong first-tail-cue choice: about 0.80-0.90 for the conflict pairs and more extreme for same-direction and single-cue pairs. Thus the conflict pairs produce a directional dissociation around chance, and the remaining pairs provide a quantitative extremity dissociation. The eight unique pairs are repeated 12 times for 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Leftmost-discriminating-cue adherence directly probes the first-cue bonus that separates the two accounts. In this fixed design the first three display positions are tied on every trial, so the first discriminating cue is always the high-validity tail cue at position 3 except in the two single-position-4 pairs. The advocated model places positions 3 and 4 in a weak tail with only a tiny recency edge at n=5, giving near-chance conflict trials and only moderate single/same-direction choices, so pooled adherence is expected near 0.59. The bounded-sampling model has capacity (5/5)^0.5 = 1 at n=5, samples every discriminating cue, and adds an unattenuated first-cue bonus, yielding adherence near 0.84-0.92 and a pooled value near 0.89. Averaging over all 96 trials per subject yields a large mean gap with small within-subject binomial variance, making the metric reliably discriminable for 25 subjects.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    if len(data) == 0:
        return 0.5
    n_match = 0
    n_valid = 0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        if a.size == 0 or a.shape != b.shape:
            continue
        diff = a != b
        if not np.any(diff):
            continue
        k = int(np.flatnonzero(diff)[0])
        r = int(row['response'])
        a_has_cue = int(a[k] == 1)
        if (a_has_cue == 1 and r == 0) or (a_has_cue == 0 and r == 1):
            n_match += 1
        n_valid += 1
    if n_valid == 0:
        return 0.5
    return float(n_match) / float(n_valid)

### RESULTS
- Predicted under pi_5 (simulated): 0.8456 (var=0.0038)
- Predicted under pi_8 (simulated): 0.5848 (var=0.0039)
- Observed on real data: 0.1771 (var=0.0120)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=int)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=int)
    r = data['response'].to_numpy(dtype=float)

    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    tie = a_wins == b_wins

    # Among tally ties, the high-validity cue at index 2 discriminates.
    cue_favors_b = tie & (b[:, 2] > a[:, 2])
    cue_favors_a = tie & (a[:, 2] > b[:, 2])

    if not np.any(cue_favors_b) or not np.any(cue_favors_a):
        return 0.0

    p_b_given_b_cue = float(r[cue_favors_b].mean())
    p_b_given_a_cue = float(r[cue_favors_a].mean())

    return p_b_given_b_cue - p_b_given_a_cue

```

**Observed (real) value:** -0.4400 (var=0.1503)
**Predicted under pi_5:** -0.5700 (var=0.1570)
**Predicted under pi_8:** -0.4883 (var=0.0319)

### Experiment 4
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
import numpy as np

def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    validities = np.array([0.78, 0.55, 0.91, 0.50, 0.65], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    a = np.vstack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.vstack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b

    ordered_diff = diff[:, cue_order]
    nonzeros = ordered_diff != 0
    has_discriminator = nonzeros.any(axis=1)
    first_pos = nonzeros.argmax(axis=1)
    first_values = ordered_diff[np.arange(ordered_diff.shape[0]), first_pos]

    choose_a = (data['response'].to_numpy() == 0).astype(float)
    first_cue_favors_a = has_discriminator & (first_values > 0)
    first_cue_favors_b = has_discriminator & (first_values < 0)

    if not (first_cue_favors_a.any() and first_cue_favors_b.any()):
        return 0.0

    p_a_given_cue_a = choose_a[first_cue_favors_a].mean()
    p_a_given_cue_b = choose_a[first_cue_favors_b].mean()
    return float(p_a_given_cue_a - p_a_given_cue_b)
```

**Observed (real) value:** -0.3750 (var=0.0321)
**Predicted under pi_5:** -0.2871 (var=0.0275)
**Predicted under pi_8:** -0.3658 (var=0.0115)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    matches = []
    for a_ratings, b_ratings, response in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        winner = None
        for a_val, b_val in zip(a_ratings, b_ratings):
            if a_val > b_val:
                winner = 0
                break
            if a_val < b_val:
                winner = 1
                break
        if winner is not None:
            matches.append(1.0 if response == winner else 0.0)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6246 (var=0.0093)
**Predicted under pi_5:** 0.6331 (var=0.0177)
**Predicted under pi_8:** 0.6619 (var=0.0059)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b_vals = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    a_wins = np.sum(a_vals > b_vals, axis=1)
    b_wins = np.sum(b_vals > a_vals, axis=1)
    tally_tie = a_wins == b_wins

    differ = a_vals != b_vals
    has_disc = differ.any(axis=1)
    first_disc = np.argmax(differ.astype(int), axis=1)

    row_idx = np.arange(a_vals.shape[0])
    first_a_wins = a_vals[row_idx, first_disc] > b_vals[row_idx, first_disc]

    response = data['response'].to_numpy(dtype=float)

    selected = tally_tie & has_disc
    if not np.any(selected):
        return 0.0

    disc_positions = np.unique(first_disc[selected])
    per_pos_diff = []
    for pos in disc_positions:
        in_pos = selected & (first_disc == pos)
        b_first = in_pos & ~first_a_wins
        a_first = in_pos & first_a_wins
        if np.sum(b_first) == 0 or np.sum(a_first) == 0:
            continue
        per_pos_diff.append(
            float(np.mean(response[b_first]) - np.mean(response[a_first]))
        )

    if not per_pos_diff:
        return 0.0
    return float(np.mean(per_pos_diff))
```

**Observed (real) value:** -0.1173 (var=0.0257)
**Predicted under pi_5:** 0.0253 (var=0.0490)
**Predicted under pi_8:** -0.0400 (var=0.0197)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    low = []
    high = []
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        nz = np.flatnonzero(diff)
        if nz.size == 0:
            continue
        pos = int(nz[0])
        favored = 0 if diff[pos] > 0 else 1
        consistent = 1.0 if int(row.response) == favored else 0.0
        if pos == 0:
            low.append(consistent)
        elif pos in (1, 2, 3):
            high.append(consistent)
    if not low or not high:
        return float('nan')
    return float(np.mean(high) - np.mean(low))
```

**Observed (real) value:** -0.0222 (var=0.0067)
**Predicted under pi_5:** 0.0267 (var=0.0064)
**Predicted under pi_8:** -0.0925 (var=0.0097)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hi_a = []
    hi_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'])
        b = np.asarray(row['option_b_ratings'])
        diff = a != b
        if not np.any(diff):
            continue
        first_diff = int(np.argmax(diff))
        if first_diff != 1:
            continue
        if a[1] > b[1]:
            hi_a.append(float(row['response']))
        else:
            hi_b.append(float(row['response']))
    if not hi_a or not hi_b:
        return 0.0
    return float(np.mean(hi_a) - np.mean(hi_b))
```

**Observed (real) value:** -0.0633 (var=0.1971)
**Predicted under pi_5:** -0.0617 (var=0.2674)
**Predicted under pi_8:** -0.0108 (var=0.0280)

### Experiment 9
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    keys = [(0,0,0,1,1,1),(1,1,1,0,0,0),(1,0,0,0,1,1),(0,1,1,1,0,0),(1,1,0,0,0,1),(0,0,1,1,1,0),(1,0,1,0,0,1),(0,1,0,1,1,0)]
    d = data.copy()
    d['a_key'] = d['option_a_ratings'].apply(tuple)
    vals = []
    for _, s in d.groupby('subject_id'):
        props = []
        for k in keys:
            mask = s['a_key'] == k
            if mask.sum() == 0:
                props.append(np.nan)
            else:
                props.append(float((s.loc[mask, 'response'] == 0).mean()))
        props = np.array(props)
        if np.isnan(props).any():
            props = np.where(np.isnan(props), 0.5, props)
        vals.append(float(np.var(props)))
    return float(np.mean(vals)) if vals else 0.0
```

**Observed (real) value:** 0.1099 (var=0.0037)
**Predicted under pi_5:** 0.1186 (var=0.0016)
**Predicted under pi_8:** 0.1079 (var=0.0012)

### Experiment 10
**Design**
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    import pandas as pd
    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)
    diff = a - b
    tally_a_wins = (diff > 0).sum(axis=1)
    tally_b_wins = (diff < 0).sum(axis=1)
    tally_for_a = tally_a_wins > tally_b_wins
    tally_for_b = tally_b_wins > tally_a_wins
    high_a_wins = (diff[:, :3] > 0).sum(axis=1)
    high_b_wins = (diff[:, :3] < 0).sum(axis=1)
    high_for_a = high_a_wins > high_b_wins
    high_for_b = high_b_wins > high_a_wins
    conflict = (high_for_a & tally_for_b) | (high_for_b & tally_for_a)
    if not conflict.any():
        return 0.0
    chose_a = data['response'].to_numpy() == 0
    tally_agreement = (tally_for_a & chose_a) | (tally_for_b & ~chose_a)
    return float(tally_agreement[conflict].mean())
```

**Observed (real) value:** 0.3475 (var=0.0426)
**Predicted under pi_5:** 0.4883 (var=0.0613)
**Predicted under pi_8:** 0.4029 (var=0.0044)

### Experiment 11
**Design**
  A=[1, 1, 1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data):
    low_idx = [0, 1, 2, 4, 5, 6]
    high_idx = [3, 7, 8]

    a_all = np.asarray(data['option_a_ratings'].tolist(), dtype=int)
    b_all = np.asarray(data['option_b_ratings'].tolist(), dtype=int)
    subj = data['subject_id'].to_numpy()
    resp = data['response'].to_numpy(dtype=float)

    subject_abs_slopes = []

    for s in np.unique(subj):
        mask = subj == s
        a = a_all[mask]
        b = b_all[mask]
        r = resp[mask]

        high_a = a[:, high_idx].sum(axis=1)
        high_b = b[:, high_idx].sum(axis=1)
        high_choice = np.where(high_b > high_a, 1, 0)

        low_a = a[:, low_idx].sum(axis=1)
        low_b = b[:, low_idx].sum(axis=1)
        k_opposing = np.where(high_b > high_a, low_a, low_b)

        chose_high = (r == high_choice).astype(float)

        keep = (k_opposing >= 3) & (k_opposing <= 6)
        if keep.sum() < 2 or np.ptp(k_opposing[keep]) == 0:
            subject_abs_slopes.append(0.0)
        else:
            slope = float(np.polyfit(k_opposing[keep], chose_high[keep], 1)[0])
            subject_abs_slopes.append(abs(slope))

    return float(np.mean(subject_abs_slopes))
```

**Observed (real) value:** 0.2025 (var=0.0134)
**Predicted under pi_5:** 0.0824 (var=0.0041)
**Predicted under pi_8:** 0.1761 (var=0.0046)

### Experiment 12
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

LOW_POS = [0, 2, 4, 6, 8, 9]


def _subject_score(sub):
    n = len(sub)
    if n < 12:
        return 0.0

    y = np.asarray(
        [1.0 if int(r) == 0 else 0.0 for r in sub['response'].tolist()],
        dtype=float
    )

    a_list = sub['option_a_ratings'].tolist()
    b_list = sub['option_b_ratings'].tolist()

    l = np.asarray([
        sum([1 for p in LOW_POS if int(a[p]) == 1 and int(b[p]) == 0])
        for a, b in zip(a_list, b_list)
    ], dtype=float)

    h = np.asarray([
        1.0 if (int(a[1]) == 1 and int(b[1]) == 0) else 0.0
        for a, b in zip(a_list, b_list)
    ], dtype=float)

    ybar = float(y.mean())
    sst = float(np.sum((y - ybar) ** 2))
    if sst <= 1e-12:
        return 0.0

    X = np.column_stack([np.ones(n), l, h])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    sse_full = float(np.sum((y - pred) ** 2))

    Xr = np.column_stack([np.ones(n), h])
    beta_r, _, _, _ = np.linalg.lstsq(Xr, y, rcond=None)
    pred_r = Xr @ beta_r
    sse_red = float(np.sum((y - pred_r) ** 2))

    h_effect = float(abs(beta[2]))

    if sse_full <= 1e-12:
        partial_l = 1.0
    else:
        partial_l = (sse_red - sse_full) / sse_red
        partial_l = float(max(0.0, min(partial_l, 1.0)))

    return h_effect + 4.0 * partial_l


def metric(data: pd.DataFrame) -> float:
    if data.empty or 'subject_id' not in data.columns:
        return 0.0

    scores = []
    for _, sub in data.groupby('subject_id'):
        scores.append(_subject_score(sub))

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))

```

**Observed (real) value:** 1.8744 (var=0.7034)
**Predicted under pi_5:** 0.7166 (var=0.1141)
**Predicted under pi_8:** 1.9747 (var=0.3019)

### Experiment 13
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    all_key = {(1, -1, -1, -1), (-1, 1, 1, 1)}
    near_key = {(1, 0, -1, -1), (-1, 0, 1, 1)}

    subject_scores = []
    for sid in pd.unique(data['subject_id']):
        sub = data[data['subject_id'] == sid]
        all_cf = []
        near_cf = []

        for a, b, resp in zip(
            sub['option_a_ratings'],
            sub['option_b_ratings'],
            sub['response'],
        ):
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            d = a_arr - b_arr
            key = tuple([int(np.sign(v)) for v in d])

            nz = np.flatnonzero(d)
            if nz.size == 0:
                continue

            first = int(nz[0])
            if d[first] > 0:
                first_choice = 1.0 if resp == 0 else 0.0
            elif d[first] < 0:
                first_choice = 1.0 if resp == 1 else 0.0
            else:
                first_choice = 0.5

            if key in all_key:
                all_cf.append(first_choice)
            elif key in near_key:
                near_cf.append(first_choice)

        if len(all_cf) == 0 or len(near_cf) == 0:
            continue

        r_all = float(np.mean(all_cf))
        r_near = float(np.mean(near_cf))
        subject_scores.append(r_near - r_all)

    if not subject_scores:
        return float('nan')
    return float(np.mean(subject_scores))

```

**Observed (real) value:** -0.1867 (var=0.1342)
**Predicted under pi_5:** 0.0992 (var=0.0345)
**Predicted under pi_8:** -0.1300 (var=0.0159)

### Experiment 14
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    first_b = []
    mirror_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        disc = np.flatnonzero(diff != 0.0)
        if disc.size < 2:
            continue
        signs = np.sign(diff[disc])
        if signs[0] == 1.0 and np.all(signs[1:] == -1.0):
            first_b.append(float(row['response']))
        elif signs[0] == -1.0 and np.all(signs[1:] == 1.0):
            mirror_b.append(float(row['response']))
    if not first_b or not mirror_b:
        return 0.0
    return float(np.mean(mirror_b) - np.mean(first_b))
```

**Observed (real) value:** 0.6992 (var=0.0351)
**Predicted under pi_5:** 0.1979 (var=0.2805)
**Predicted under pi_8:** 0.5371 (var=0.0273)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 14 experiments, pi_8 is the stronger theory: it is closer on 10/14 (Exps 2,3,4,6,9,10,11,12,13,14), often near exact (Exp4 -0.366 vs -0.375; Exp11 0.176 vs 0.203; Exp12 1.975 vs 1.874; Exp13 -0.130 vs -0.187). pi_5 only wins where the observed effect is flat or near zero (Exp1 0.023 vs pi_8 0.324; Exp7 -0.022 vs pi_8 -0.093; Exp8 -0.063 vs pi_8 -0.011) or where both are close (Exp5). pi_5 is qualitatively wrong on Exp13 (+0.099 vs -0.187), Exp14 (0.198 vs 0.699), Exp12 (0.717 vs 1.874), and Exp2 (0.846 vs 0.177). However pi_8 is not universal either: Exp1 shows its long-list odd-one-out/long-config bonus is too strong (predicted +0.324 vs observed +0.023), and Exp2 is a severe miss for both theories (pi_8 +0.585, pi_5 +0.846 vs observed +0.177). The Exp2 result points to a missing mechanism: when the first discriminating cue appears only after a run of tied positions, subjects do not follow it; they appear to oppose or ignore it and favor the later/tail cue. Neither theory contains a pre-first-cue tie-run reset. Thus neither current theory is task-invariant. pi_8 should be kept as the incumbent, and pi_5 should be replaced by a new theory that retains pi_8's configural machinery while fixing the two systematic failures.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory, e.g. 'tied-prefix reset with length-gated configural contrast'. It should retain pi_8's early-span primacy, same-sign run gain, short-list odd-one-out bonus, tie-gap penalty, compressive validity gain, tail/recency accumulation, and subject heterogeneity. It must add/modify: (1) a pre-cue tie-run reset \u2014 if the first discriminating cue is preceded by >=2 tied positions, its weight is strongly reduced or sign-flipped subject-wise, and the last discriminating cue/tail block dominates, producing leftmost-cue adherence near 0.18 in Exp2; (2) strict length gating of the odd-one-out bonus so it is essentially zero for n>=8, eliminating the Exp1 adjacent-arrangement contrast (target ~0.02); (3) subject-level anti-cue weights on long lists to preserve near-zero means with large between-subject variance in Exp6/8; (4) retain the tie-gap penalty after the first cue to keep the Exp13/14 mirror asymmetry. Parameter ranges should be set so the new theory can produce Exp2 ~0.18, Exp1 ~0.0, Exp11 ~0.20, Exp12 ~1.87, Exp13 ~-0.19, and Exp14 ~0.70, without creating a recency effect for single cues (Exp7 ~-0.02)."
}
```

## Usage

```json
{
  "input_tokens": 18720,
  "output_tokens": 9923,
  "reasoning_tokens": 8853,
  "provider": "CoreWeave"
}
```
