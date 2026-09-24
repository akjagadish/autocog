# generation_iter_06_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Propose a new theory, e.g. 'tied-prefix reset with length-gated configural contrast'. It should retain pi_8's early-span primacy, same-sign run gain, short-list odd-one-out bonus, tie-gap penalty, compressive validity gain, tail/recency accumulation, and subject heterogeneity. It must add/modify: (1) a pre-cue tie-run reset — if the first discriminating cue is preceded by >=2 tied positions, its weight is strongly reduced or sign-flipped subject-wise, and the last discriminating cue/tail block dominates, producing leftmost-cue adherence near 0.18 in Exp2; (2) strict length gating of the odd-one-out bonus so it is essentially zero for n>=8, eliminating the Exp1 adjacent-arrangement contrast (target ~0.02); (3) subject-level anti-cue weights on long lists to preserve near-zero means with large between-subject variance in Exp6/8; (4) retain the tie-gap penalty after the first cue to keep the Exp13/14 mirror asymmetry. Parameter ranges should be set so the new theory can produce Exp2 ~0.18, Exp1 ~0.0, Exp11 ~0.20, Exp12 ~1.87, Exp13 ~-0.19, and Exp14 ~0.70, without creating a recency effect for single cues (Exp7 ~-0.02).

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.594)

**Description**
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


`parameters`:
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.4883 (var=0.0319)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3658 (var=0.0115)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6619 (var=0.0059)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=-0.0400 (var=0.0197)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.0925 (var=0.0097)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=-0.0108 (var=0.0280)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.1079 (var=0.0012)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.4029 (var=0.0044)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.1761 (var=0.0046)
- Experiment 10: real=1.8744 (var=0.7034) vs this=1.9747 (var=0.3019)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=-0.1300 (var=0.0159)
- Experiment 12: real=0.6992 (var=0.0351) vs this=0.5371 (var=0.0273)
- Experiment 13: real=0.0231 (var=0.0947) vs this=0.3244 (var=0.0219)
- Experiment 14: real=0.1771 (var=0.0120) vs this=0.5848 (var=0.0039)


---

### `pi_5` (overall score: 0.477)

**Description**
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


`parameters`:
- capacity_gamma: [0.45, 0.55]
- position_decay: [0.20, 0.40]
- att_validity_intercept: [0.48, 0.52]
- att_validity_slope: [0.75, 0.90]
- evidence_intercept: [0.50, 0.54]
- evidence_slope: [0.52, 0.66]
- first_cue_bonus: [1.65, 1.95]
- first_cue_validity_power: [0.45, 0.75]
- first_cue_capacity_power: [1.50, 2.50]
- first_position_decay: [0.12, 0.22]
- subject_scale: [0.40, 1.60]
- subject_bias: [-0.30, 0.30]
- beta: [1.10, 2.30]
- epsilon: [0.12, 0.18]
- long_jitter_shift: [0.78, 0.98]
- validities: validities
- evidence_jitter: [(-1.5, 3.0)] * n_features

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.5700 (var=0.1570)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.2871 (var=0.0275)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6331 (var=0.0177)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=0.0253 (var=0.0490)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=0.0267 (var=0.0064)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=-0.0617 (var=0.2674)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.1186 (var=0.0016)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.4883 (var=0.0613)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.0824 (var=0.0041)
- Experiment 10: real=1.8744 (var=0.7034) vs this=0.7166 (var=0.1141)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=0.0992 (var=0.0345)
- Experiment 12: real=0.6992 (var=0.0351) vs this=0.1979 (var=0.2805)
- Experiment 13: real=0.0231 (var=0.0947) vs this=-0.0009 (var=0.1257)
- Experiment 14: real=0.1771 (var=0.0120) vs this=0.8456 (var=0.0038)


---

### `pi_6` (overall score: 0.414)

**Description**
People make binary choices by accumulating a bounded, attention-weighted average of discriminating feature comparisons. Each cue's attention is the product of a length-gated validity-contrast term, a serial-position window with strong early-cue primacy on short lists but flatter long-list weighting, and subject-level multiplicative attention noise. On eight-cue isolated displays the first discriminating cue receives a pre-normalization attention discount, producing a mild anti-first-cue tendency without reversing the evidence sign, while on seven-cue displays the contrastive-validity dominance is slightly damped so the high-validity block does not overwhelm the tally signal. Choices are graded via a noisy softmax with lapse, with no hard lexicographic stopping.

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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validity_power: [0.70, 1.10]
- contrast_power: [1.80, 2.60]
- contrast_power_decay: [0.22, 0.32]
- contrast_gate_temp: [0.35, 0.65]
- early_gain: [2.00, 2.40]
- short_early_gain: [2.85, 3.05]
- n7_contrast_reduction: [0.20, 0.30]
- window_center: [0.45, 0.60]
- center_shrink: [0.25, 0.35]
- window_temp: [0.18, 0.26]
- tail_base: [0.09, 0.11]
- tail_growth: [0.60, 0.80]
- beta: [3.00, 4.00]
- epsilon: [0.13, 0.17]
- bias: [-0.05, 0.05]
- logit_noise_sd: [0.00, 0.10]
- pattern_bias_scale: [0.60, 0.90]
- pattern_bias_seed: [0.0, 100000.0]
- n8_first_discriminator_factor: [0.45, 0.60]
- log_weight_noise: [(-1.0, 1.0)] * n_features
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.3200 (var=0.1409)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3213 (var=0.0070)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6160 (var=0.0094)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=-0.1307 (var=0.0469)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.0006 (var=0.0043)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=-0.0858 (var=0.1788)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.0923 (var=0.0010)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.2933 (var=0.0149)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.0260 (var=0.0004)
- Experiment 10: real=1.8744 (var=0.7034) vs this=0.4099 (var=0.0378)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=0.0358 (var=0.0131)
- Experiment 12: real=0.6992 (var=0.0351) vs this=0.5308 (var=0.0471)
- Experiment 13: real=0.0231 (var=0.0947) vs this=0.0178 (var=0.0185)
- Experiment 14: real=0.1771 (var=0.0120) vs this=0.8081 (var=0.0040)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1456 -> ACCEPTED
- iter 2: loss=0.1449 -> ACCEPTED
- iter 3: loss=0.1296 -> ACCEPTED
- iter 4: loss=0.1460 -> REJECTED
- iter 5: loss=0.0924 -> ACCEPTED
- iter 6: loss=0.1121 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.0924 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
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
**Previous candidate values (this loop):**
  - iter 1: -0.4833 (var=0.0453) (Δ vs real -0.0433)
  - iter 2: -0.6133 (var=0.0583) (Δ vs real -0.1733)
  - iter 3: -0.6133 (var=0.0519) (Δ vs real -0.1733)
  - iter 4: -0.4733 (var=0.0415) (Δ vs real -0.0333)
  - iter 5: -0.4467 (var=0.0755) (Δ vs real -0.0067)
  - iter 6 (most recent): -0.7550 (var=0.0584) (Δ vs real -0.3150)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6650 (var=0.0546)
- pi_2: 0.0383 (var=0.0359)
- pi_3: -0.6500 (var=0.0231)
- pi_4: -0.4933 (var=0.0313)
- pi_5: -0.5700 (var=0.1570)
- pi_6: -0.3200 (var=0.1409)
- pi_7: -0.5133 (var=0.1857)
- pi_8: -0.4883 (var=0.0319)

### Experiment 2
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
**Previous candidate values (this loop):**
  - iter 1: -0.3400 (var=0.0117) (Δ vs real +0.0350)
  - iter 2: -0.3708 (var=0.0140) (Δ vs real +0.0042)
  - iter 3: -0.4000 (var=0.0127) (Δ vs real -0.0250)
  - iter 4: -0.3804 (var=0.0153) (Δ vs real -0.0054)
  - iter 5: -0.3575 (var=0.0120) (Δ vs real +0.0175)
  - iter 6 (most recent): -0.3912 (var=0.0157) (Δ vs real -0.0162)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0029 (var=0.0086)
- pi_1: 0.6875 (var=0.0362)
- pi_3: -0.3296 (var=0.0048)
- pi_4: -0.3458 (var=0.0038)
- pi_5: -0.2871 (var=0.0275)
- pi_6: -0.3213 (var=0.0070)
- pi_7: -0.3442 (var=0.0247)
- pi_8: -0.3658 (var=0.0115)

### Experiment 3
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
**Previous candidate values (this loop):**
  - iter 1: 0.6331 (var=0.0045) (Δ vs real +0.0085)
  - iter 2: 0.6165 (var=0.0040) (Δ vs real -0.0081)
  - iter 3: 0.4973 (var=0.0018) (Δ vs real -0.1273)
  - iter 4: 0.5029 (var=0.0031) (Δ vs real -0.1217)
  - iter 5: 0.5621 (var=0.0027) (Δ vs real -0.0625)
  - iter 6 (most recent): 0.6458 (var=0.0041) (Δ vs real +0.0212)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8567 (var=0.0014)
- pi_2: 0.2706 (var=0.0057)
- pi_1: 0.6206 (var=0.0018)
- pi_4: 0.6208 (var=0.0020)
- pi_5: 0.6331 (var=0.0177)
- pi_6: 0.6160 (var=0.0094)
- pi_7: 0.5504 (var=0.0048)
- pi_8: 0.6619 (var=0.0059)

### Experiment 4
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
**Previous candidate values (this loop):**
  - iter 1: -0.0107 (var=0.0255) (Δ vs real +0.1067)
  - iter 2: -0.1080 (var=0.0405) (Δ vs real +0.0093)
  - iter 3: -0.0680 (var=0.0387) (Δ vs real +0.0493)
  - iter 4: -0.1253 (var=0.0198) (Δ vs real -0.0080)
  - iter 5: -0.1013 (var=0.0351) (Δ vs real +0.0160)
  - iter 6 (most recent): 0.0520 (var=0.0354) (Δ vs real +0.1693)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0147 (var=0.0280)
- pi_3: 0.7160 (var=0.0149)
- pi_1: 0.2253 (var=0.0167)
- pi_4: 0.0280 (var=0.0391)
- pi_5: 0.0253 (var=0.0490)
- pi_6: -0.1307 (var=0.0469)
- pi_7: 0.0827 (var=0.0522)
- pi_8: -0.0400 (var=0.0197)

### Experiment 5
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
**Previous candidate values (this loop):**
  - iter 1: -0.0486 (var=0.0111) (Δ vs real -0.0264)
  - iter 2: -0.2264 (var=0.0239) (Δ vs real -0.2042)
  - iter 3: -0.1272 (var=0.0171) (Δ vs real -0.1050)
  - iter 4: -0.0328 (var=0.0109) (Δ vs real -0.0106)
  - iter 5: 0.0142 (var=0.0120) (Δ vs real +0.0364)
  - iter 6 (most recent): 0.0431 (var=0.0073) (Δ vs real +0.0653)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1878 (var=0.0101)
- pi_4: -0.2019 (var=0.0042)
- pi_1: 0.0092 (var=0.0061)
- pi_2: 0.0058 (var=0.0086)
- pi_5: 0.0267 (var=0.0064)
- pi_6: -0.0006 (var=0.0043)
- pi_7: -0.0075 (var=0.0053)
- pi_8: -0.0925 (var=0.0097)

### Experiment 6
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
**Previous candidate values (this loop):**
  - iter 1: 0.1975 (var=0.0236) (Δ vs real +0.2608)
  - iter 2: 0.3725 (var=0.0526) (Δ vs real +0.4358)
  - iter 3: 0.3417 (var=0.0608) (Δ vs real +0.4050)
  - iter 4: -0.0492 (var=0.1468) (Δ vs real +0.0142)
  - iter 5: -0.0317 (var=0.1049) (Δ vs real +0.0317)
  - iter 6 (most recent): 0.0133 (var=0.0574) (Δ vs real +0.0767)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6033 (var=0.0124)
- pi_3: -0.8958 (var=0.0049)
- pi_1: -0.6992 (var=0.0453)
- pi_2: 0.4608 (var=0.0209)
- pi_5: -0.0617 (var=0.2674)
- pi_6: -0.0858 (var=0.1788)
- pi_7: 0.0942 (var=0.2179)
- pi_8: -0.0108 (var=0.0280)

### Experiment 7
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
**Previous candidate values (this loop):**
  - iter 1: 0.1180 (var=0.0007) (Δ vs real +0.0081)
  - iter 2: 0.1230 (var=0.0011) (Δ vs real +0.0131)
  - iter 3: 0.1359 (var=0.0011) (Δ vs real +0.0260)
  - iter 4: 0.1415 (var=0.0014) (Δ vs real +0.0316)
  - iter 5: 0.1268 (var=0.0011) (Δ vs real +0.0169)
  - iter 6 (most recent): 0.1149 (var=0.0011) (Δ vs real +0.0050)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1186 (var=0.0016)
- pi_4: 0.0406 (var=0.0002)
- pi_1: 0.1334 (var=0.0041)
- pi_2: 0.0187 (var=0.0001)
- pi_3: 0.0541 (var=0.0003)
- pi_6: 0.0923 (var=0.0010)
- pi_7: 0.1140 (var=0.0020)
- pi_8: 0.1079 (var=0.0012)

### Experiment 8
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
**Previous candidate values (this loop):**
  - iter 1: 0.3767 (var=0.0049) (Δ vs real +0.0292)
  - iter 2: 0.4921 (var=0.0290) (Δ vs real +0.1446)
  - iter 3: 0.3346 (var=0.0041) (Δ vs real -0.0129)
  - iter 4: 0.4179 (var=0.0040) (Δ vs real +0.0704)
  - iter 5: 0.4267 (var=0.0783) (Δ vs real +0.0792)
  - iter 6 (most recent): 0.4904 (var=0.0154) (Δ vs real +0.1429)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9329 (var=0.0017)
- pi_5: 0.4883 (var=0.0613)
- pi_1: 0.5050 (var=0.0022)
- pi_2: 0.8471 (var=0.0113)
- pi_3: 0.4933 (var=0.0007)
- pi_6: 0.2933 (var=0.0149)
- pi_7: 0.2300 (var=0.0296)
- pi_8: 0.4029 (var=0.0044)

### Experiment 9
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
**Previous candidate values (this loop):**
  - iter 1: 0.1208 (var=0.0043) (Δ vs real -0.0817)
  - iter 2: 0.1194 (var=0.0081) (Δ vs real -0.0831)
  - iter 3: 0.1014 (var=0.0101) (Δ vs real -0.1011)
  - iter 4: 0.1083 (var=0.0144) (Δ vs real -0.0943)
  - iter 5: 0.0946 (var=0.0076) (Δ vs real -0.1079)
  - iter 6 (most recent): 0.1056 (var=0.0145) (Δ vs real -0.0969)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0824 (var=0.0041)
- pi_6: 0.0260 (var=0.0004)
- pi_1: 0.0349 (var=0.0006)
- pi_2: 0.2730 (var=0.0071)
- pi_3: 0.0405 (var=0.0008)
- pi_4: 0.3544 (var=0.0008)
- pi_7: 0.1950 (var=0.0096)
- pi_8: 0.1761 (var=0.0046)

### Experiment 10
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
**Previous candidate values (this loop):**
  - iter 1: 1.9335 (var=0.2983) (Δ vs real +0.0591)
  - iter 2: 1.8885 (var=0.4321) (Δ vs real +0.0141)
  - iter 3: 1.9265 (var=0.4132) (Δ vs real +0.0521)
  - iter 4: 2.0155 (var=0.4124) (Δ vs real +0.1411)
  - iter 5: 1.9509 (var=0.6451) (Δ vs real +0.0765)
  - iter 6 (most recent): 1.9103 (var=0.5134) (Δ vs real +0.0359)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4099 (var=0.0378)
- pi_5: 0.7166 (var=0.1141)
- pi_1: 0.8464 (var=0.3893)
- pi_2: 2.0302 (var=0.7154)
- pi_3: 0.2507 (var=0.0279)
- pi_4: 2.8162 (var=0.0908)
- pi_7: 2.2173 (var=0.1964)
- pi_8: 1.9747 (var=0.3019)

### Experiment 11
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
**Previous candidate values (this loop):**
  - iter 1: -0.2283 (var=0.0215) (Δ vs real -0.0417)
  - iter 2: -0.1583 (var=0.0209) (Δ vs real +0.0283)
  - iter 3: -0.1300 (var=0.0361) (Δ vs real +0.0567)
  - iter 4: -0.3058 (var=0.0269) (Δ vs real -0.1192)
  - iter 5: -0.2233 (var=0.0315) (Δ vs real -0.0367)
  - iter 6 (most recent): -0.1967 (var=0.0310) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0992 (var=0.0345)
- pi_7: 0.3383 (var=0.1102)
- pi_1: 0.0133 (var=0.0088)
- pi_2: 0.0150 (var=0.0106)
- pi_3: 0.0067 (var=0.0021)
- pi_4: -0.0050 (var=0.0059)
- pi_6: 0.0358 (var=0.0131)
- pi_8: -0.1300 (var=0.0159)

### Experiment 12
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
**Previous candidate values (this loop):**
  - iter 1: 0.6025 (var=0.0269) (Δ vs real -0.0967)
  - iter 2: 0.6787 (var=0.0347) (Δ vs real -0.0204)
  - iter 3: 0.4967 (var=0.0492) (Δ vs real -0.2025)
  - iter 4: 0.6642 (var=0.0437) (Δ vs real -0.0350)
  - iter 5: 0.4625 (var=0.0377) (Δ vs real -0.2367)
  - iter 6 (most recent): 0.5417 (var=0.0452) (Δ vs real -0.1575)
**Other theories' values on this metric (for reference):**
- pi_7: -0.3475 (var=0.3307)
- pi_5: 0.1979 (var=0.2805)
- pi_1: -0.6408 (var=0.0276)
- pi_2: -0.6800 (var=0.0406)
- pi_3: 0.5900 (var=0.0052)
- pi_4: 0.5504 (var=0.0067)
- pi_6: 0.5308 (var=0.0471)
- pi_8: 0.5371 (var=0.0273)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0231 (var=0.0947)
**Previous candidate values (this loop):**
  - iter 1: 0.2994 (var=0.0144) (Δ vs real +0.2762)
  - iter 2: 0.0647 (var=0.0385) (Δ vs real +0.0416)
  - iter 3: 0.0141 (var=0.1235) (Δ vs real -0.0091)
  - iter 4: -0.0975 (var=0.1057) (Δ vs real -0.1206)
  - iter 5: -0.0522 (var=0.1935) (Δ vs real -0.0753)
  - iter 6 (most recent): 0.1081 (var=0.1603) (Δ vs real +0.0850)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0009 (var=0.1257)
- pi_8: 0.3244 (var=0.0219)
- pi_1: 0.0041 (var=0.0075)
- pi_2: 0.0078 (var=0.0051)
- pi_3: 0.0056 (var=0.0015)
- pi_4: 0.0028 (var=0.0025)
- pi_6: 0.0178 (var=0.0185)
- pi_7: -0.1825 (var=0.0584)

### Experiment 14
**Design**
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.1771 (var=0.0120)
**Previous candidate values (this loop):**
  - iter 1: 0.5817 (var=0.0029) (Δ vs real +0.4046)
  - iter 2: 0.4546 (var=0.0511) (Δ vs real +0.2775)
  - iter 3: 0.1796 (var=0.0047) (Δ vs real +0.0025)
  - iter 4: 0.6560 (var=0.0052) (Δ vs real +0.4790)
  - iter 5: 0.1994 (var=0.0056) (Δ vs real +0.0223)
  - iter 6 (most recent): 0.1935 (var=0.0062) (Δ vs real +0.0165)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5848 (var=0.0039)
- pi_5: 0.8456 (var=0.0038)
- pi_1: 0.8619 (var=0.0114)
- pi_2: 0.7617 (var=0.0062)
- pi_3: 0.9594 (var=0.0005)
- pi_4: 0.5996 (var=0.0031)
- pi_6: 0.8081 (var=0.0040)
- pi_7: 0.8469 (var=0.0027)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
People make binary quality choices by salience-weighted ordered evidence accumulation. Early display positions receive strong but subject-signed weight, a sharply length-gated odd-one-out bonus amplifies a conflicting first cue on short lists, and a tied-prefix reset suppresses late first cues only when the ambient trial stream is dominated by late discriminators. On long lists, early and tail blocks share a mirror-symmetric subject polarity for non-initial first discriminators. Stated validity enters only as a compressive gain, and evidence passes through a noisy softmax with lapse.

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
            early_polarity = float(parameters['n7_early_polarity'])
        else:
            early_polarity = float(parameters['mid_early_polarity'])
        early_weight_arr = early_weight_arr * early_polarity

    if n_features >= 9 and first_disc != 0:
        tail_polarity = early_polarity
    else:
        tail_polarity = 1.0

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
            elif sign == sign_first:
                w *= same_run_gain
            elif short_configural_opposing:
                w *= opposing_early_ratio
            early_evidence += w * sign
        else:
            tail_raw += tail_weight_arr[pos] * gain * sign

    block_compression = float(parameters['block_compression'])
    tail_raw = tail_raw * tail_polarity
    tail_score = tail_raw / (1.0 + block_compression * abs(tail_raw))

    long_config_tail_suppression = float(parameters['long_config_tail_suppression'])
    if (
        n_features >= 8
        and first_disc == 0
        and later_positions.size >= 2
        and np.all(signs[later_positions] == -sign_first)
    ):
        tail_score = tail_score * long_config_tail_suppression

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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0.0 or not np.isfinite(total):
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- early_weight: [2.20, 2.60]
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
- long_early_polarity: [-2.60, 2.60]
- mid_early_polarity: [0.20, 1.80]
- n7_early_polarity: [-1.20, 3.60]
- long_list_threshold: {7}
- tie_run_flipped_weight: [-3.00, -2.50]
- single_cue_reset_scale: [0.20, 0.35]
- late_context_threshold: [0.80, 0.95]
- context_min_trials: {2}
- beta: [0.90, 1.15]
- subject_scale: [0.30, 2.20]
- bias: [-0.05, 0.05]
- trial_noise_sd: [0.08, 0.18]
- lapse: [0.04, 0.09]
- validities: validities

`rationale`: Minimal-diff correction of the accepted tied-prefix reset base. The unconditional tied-prefix reset is kept but now context-gated on the past trial stream: if the subject's history is dominated by late first-discriminating cues, late tied-prefix cues are strongly flipped, preserving Exp14's ~0.18 adherence, while Exp5's late single cues are embedded in a mixed positional stream so the reset stays off and its positional metric returns toward zero. Odd-one-out strength is increased and its length gate is sharpened between n=5 and n=6 so short-list first-cue primacy recovers for Exp3 and Exp12 without leaking into Exp7; third_ratio is also raised modestly for delayed but still early cues. Recency is lowered toward the pi_8 range to steepen the Exp9 opposing-block slope and tighten Exp7. On long lists with a non-initial first discriminator, the tail block is multiplied by the same mirror-symmetric subject polarity as the early block, producing the near-zero Exp6 mean with large between-subject variance, while first-position configural lists such as Exp13 remain protected. The n=7 early polarity is replaced by an asymmetric subject-level distribution with a positive center and a negative tail to keep Exp8's mean near 0.35 while adding between-subject variance. Finally, tie_gap_penalty is moderately strengthened to move Exp11 toward -0.19 without the iter4 overshoot.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a faithful instantiation of the prescribed mechanism family and it preserves many of pi_8's good fits: Exp1 (-0.483 vs -0.440), Exp2 (-0.340 vs -0.375), Exp3 (0.633 vs 0.625), Exp5 (-0.049 vs -0.022), Exp7 (0.118 vs 0.110), Exp8 mean (0.377 vs 0.348), Exp10 (1.934 vs 1.874), and Exp11 direction (-0.228 vs -0.187) are all broadly reasonable. However, the candidate fails on the very repairs it was designed to implement. Exp14's leftmost-cue adherence is 0.582 against the observed 0.177, meaning the pre-cue tie-run reset is effectively not triggering: in Exp14 every first discriminating cue is preceded by three tied positions, so the reset's additional low-stated-validity gate is the likely blocker. Exp13's adjacent-vs-nonadjacent contrast is 0.299 against the observed 0.023; the strict length gate on the odd-one-out bonus alone was insufficient because the tail/recency accumulation still differs between adjacent and nonadjacent later-cue blocks on long lists. Exp6 is wrong in both mean and heterogeneity (0.198 with var 0.024 vs -0.063 with var 0.197), showing the long-list polarity distribution is miscalibrated: its range [-0.40, 1.40] has a strongly positive mean, so subject-level anti-cue weights do not produce the required near-zero group mean with large between-subject variance. Exp8 has the right mean but almost no between-subject variance (0.005 vs 0.043), so the subject-heterogeneity mechanism is absent or too weak at n=7. Exp9 is too flat (0.121 vs 0.203) and Exp12's mirror asymmetry remains too weak (0.603 vs 0.699). Exp4 is also essentially zero (-0.011 vs -0.117).
Rationale: Keep the overall pi_8-derived architecture and the prescribed repairs, but recalibrate the triggering and distributional knobs. (1) Remove or greatly relax the low-validity gate on the pre-cue tie-run reset; the reset should depend primarily on tie_run_len >= 2. If single-cue experiments need protection, require either multiple discriminators or a high tie-run length, rather than gating on stated validity, and make tie_run_flipped_weight strongly negative subject-wise so the tail block dominates in Exp14. (2) Add a long-list configural tail/recency suppression: when n>=8 and the first discriminating cue is at position 0 with all later cues opposing it, scale tail evidence down toward 0.0-0.25. This removes the residual adjacent/nonadjacent contrast in Exp13 without eliminating short-list configural effects or damaging Exp10, whose arrangement is not a pure configural-odd long list in the same way. (3) Recalibrate long_early_polarity to a zero-mean symmetric broad distribution, e.g. roughly [-2.0, 2.0] or a discrete sign distribution around 0, instead of the positively biased [-0.40, 1.40]; this should fix the Exp6 mean and variance. Extend the subject-heterogeneity mechanism to cover Exp8 (n=7), for example by lowering long_list_threshold to 7-8 or widening subject_scale to about [0.30, 2.30], so Exp8's between-subject variance approaches the observed 0.043. (4) Slightly increase tail/recency weight to recover the Exp9 opposing-cue slope toward 0.20, with the long-list configural tail suppression in place to prevent a new Exp13 contrast. (5) For Exp12, modestly sharpen early cue dominance (slightly higher early_weight or lower trial_noise_sd), while moving tie_gap_penalty away from the current extreme (toward roughly [0.35, 0.55]) so Exp11 does not overshoot beyond the observed -0.187. These are in-family parameter and gating recalibrations, not a change of mechanism family.

**Outcome of this advice:** iter 1 candidate loss=0.1456 -> iter 2 candidate loss=0.1449 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate preserves and improves several strong fits from the accepted base: Exp2 is nearly exact (-0.371 vs -0.375), Exp3 is close (0.616 vs 0.625), Exp4 improved to -0.108 against -0.117, Exp7 remains close (0.123 vs 0.110), Exp10 is accurate (1.889 vs 1.874), Exp11 is close (-0.158 vs -0.187), Exp12 is now near target (0.679 vs 0.699), and Exp13 improved substantially (0.065 vs 0.023). However, the candidate still misses several central empirical patterns. Exp14 remains far above target: 0.455 against an observed leftmost-cue adherence of 0.177. The tie-run reset is therefore still not producing the required strong anti-cue dominance in late first-cue trials. Exp6 is wrong in sign and magnitude: 0.373 against -0.063, with between-subject variance still much too low (0.053 against 0.197). Exp5 became a large negative outlier: -0.226 against -0.022, meaning the single-cue positional gradient is much too steep. Exp8 mean is also too high: 0.492 against 0.348, despite a reasonable variance. Exp9 remains too flat at 0.119 against 0.203, and Exp1 is overly negative at -0.613 against -0.440. The aggregate loss improved only slightly, and the accepted-gate improvement does not cover these qualitative misses.
Rationale: Regenerate within the same tied-prefix reset and configural suppression family, but fix four calibrated components. First, the odd-one-out length gate is implemented with the logistic sign reversed: z_odd increases with list length, so sigmoid(z_odd) is near 1 for long lists and near 0 for short lists, the opposite of the intended short-list-only bonus. This leaves the configural odd bonus active on the n=12 items of Exp6 and helps drive the wrong-sign positive group mean. Change the gate to sigmoid((odd_length_center - n_features) / odd_length_temp) so it is essentially zero for n>=8 and active for short lists, and recalibrate odd_one_out_bonus downward as needed to protect Exp7 and Exp12. Second, the Exp14 reset is still too weak because it retains a validity-threshold qualifier and because reset evidence competes with ordinary tail evidence for the first late cue. Remove the validity qualifier from tie_run_reset when tie_run_len is at least 3, strengthen tie_run_flipped_weight to roughly -2.5 to -3.0, and suppress the first cue's positive tail contribution when the reset fires, so the mirrored tail block dominates and Exp14 moves from 0.455 toward 0.18. Third, add a single-discriminator positional equalization: when disc_positions.size is 1, set the early positional ratios to a uniform value across the early span and align late single-cue weight with that value, so Exp5's high-minus-low contrast returns toward the observed -0.02 without creating recency effects in Exp7. Fourth, recalibrate the n=7 early dominance separately from n>=9: the positive mean of mid_early_polarity is not enough to push Exp8's conflict tally agreement down toward the observed 0.35, so raise the n=7 early polarity center or early weight. Finally, increase tail and recency influence for non-configural long lists, for example by raising recency_scale and recency_power moderately, to steepen the Exp9 opposing-cue slope toward 0.20 while keeping long_config_tail_suppression intact to protect Exp13. Do not switch mechanism families; these are all gating, distributional, and weighting recalibrations inside the prescribed tied-prefix reset family.

**Outcome of this advice:** iter 2 candidate loss=0.1449 -> iter 3 candidate loss=0.1296 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is a real aggregate improvement and nails several targets, especially Exp2 (-0.400 vs -0.375), Exp8 mean (0.335 vs 0.348), Exp10 (1.927 vs 1.874), Exp13 (0.014 vs 0.023), and Exp14 (0.180 vs 0.177). Those gains show the length-gated odd-one-out fix and the strengthened reset are moving in the right direction. However, the unconditional tie-run reset is now overactive: it sign-flips any first discriminating cue preceded by three or more ties, even in short multi-cue displays where human subjects still follow the first cue. This directly damages Exp3 (0.497 vs 0.625, a large regression from the prior iteration's 0.616) and Exp5 (-0.127 vs -0.022), because late first cues in those short lists are anti-cued when they should remain near baseline. Exp12 also regressed sharply (0.497 vs 0.699), and Exp3/Exp12 together show short-list first-cue primacy is now too weak. Exp6 remains wrong in mean and heterogeneity (0.342 vs -0.063, var 0.061 vs 0.197), Exp9 is still too flat (0.101 vs 0.203), Exp1 remains too anti-cue (-0.613 vs -0.440), and Exp11 is somewhat too weak (-0.130 vs -0.187). Exp4 also remains too weak (-0.068 vs -0.117).
Rationale: Revise within the tied-prefix reset and length-gated configural family rather than switching mechanisms. First, make the tie-run reset conditional instead of firing on every tie_run_len >= 3: require either a compact same-sign late tail block or a delayed cue in a low-validity late region, and use a graded reset strength that declines with the delayed cue's validity and with the presence of an opposing later cue. This should preserve Exp14 near 0.18 while restoring positive first-cue adherence in Exp5's single late cues and Exp3's delayed-cue trials. Second, restore short-list first-cue primacy: increase early_weight or the n<=5 configural odd-one-out bonus and reduce opposing_early_ratio, targeting Exp3 near 0.62 and Exp12 near 0.70, while checking Exp7 and Exp1 for compensation. Third, fix the long-list anti-cue mixture: force long_early_polarity to be an explicitly zero-mean subject-level sign distribution and apply the same polarity scaling to the tail block on long lists, so pro-cue and anti-cue subjects are mirror-symmetric and the Exp6 group mean moves toward -0.06 with between-subject variance approaching 0.20. Fourth, steepen Exp9 by lowering block_compression for non-configural long lists rather than increasing recency further, since the previous recency increase flattened rather than steepened the slope. Finally, after the primacy and reset recalibrations, re-check Exp11 and lower tie_gap_penalty slightly toward a stronger gap if needed.

**Outcome of this advice:** iter 3 candidate loss=0.1296 -> iter 4 candidate loss=0.1460 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The iter4 candidate gets several peripheral targets approximately right: Exp1 improves to -0.473 vs -0.440, Exp2 is near exact at -0.380 vs -0.375, Exp4 is close at -0.125 vs -0.117, Exp5 returns to -0.033 vs -0.022, Exp6's mean is nearly right at -0.049 vs -0.063, and Exp12 is close at 0.664 vs 0.699. But the candidate fails decisively on the two central repairs and on several second-order patterns. Exp14 collapsed from the iter3 value of 0.180 to 0.656 against an observed 0.177, meaning the newly added validity-gated tied-run reset is effectively disabled for the delayed first cues that must be anti-cued. Exp13 reversed direction from the good iter3 value of 0.014 to -0.098 against 0.023, so the new signed tail-block polarity and n9 compression adjustments created a spurious adjacent versus nonadjacent contrast. Exp9 remains too flat, 0.108 against 0.203, so the opposing-block slope is still not steep enough. Exp8's mean moved up to 0.418 against 0.348 and its between-subject variance is still essentially zero, 0.004 against 0.043, showing the n=7 subject heterogeneity is not implemented well. Exp3 remains too low at 0.503 against 0.625, so short-list delayed first-cue primacy is still too weak. Exp11 overshoots badly at -0.306 against -0.187. The aggregate loss is 0.1460, worse than the accepted iter3 base at 0.1296, and the gate therefore rejected this candidate.
Rationale: Regenerate, building on the accepted iter3 base and mining only the successful patches from iter4. First, do not reapply the validity-gated reset that followed my previous advice and was rejected: it killed Exp14's anti-cue adherence. Keep an unconditional or near-unconditional graded tie-run reset for tied prefixes, but protect Exp3/Exp5 with structural or narrowly calibrated protections rather than a broad validity gate. If a validity exception is kept, it must be tuned against the actual Exp14 cue validities so the reset fires for those delayed cues, not for informative short-list delayed cues. Second, decouple the subject-level signed polarity from the tail block on long lists. Keep the mirror-symmetric signed early-span polarity that brought Exp6's mean into line, but leave or strongly damp the tail-block polarity for n>=9 configural lists so Exp13 returns toward zero. Widen long_polarity_mag to roughly 1.0 to 2.8 to push Exp6 variance from 0.147 closer to the observed 0.197. Third, replace the constant n7_early_polarity range with a subject-level sign-by-magnitude distribution, centered to preserve Exp8's mean near 0.35 while adding enough spread to get between-subject variance toward 0.04. Fourth, restore short-list first-cue primacy by increasing the odd-one-out gate for n=5, for example lowering odd_length_center to about 4.2 to 4.8 while keeping the hard zero gate for n>=8, and modestly raising second_ratio if needed to help Exp3 without disturbing Exp12. Fifth, make the n=9 non-configural tail accumulate more steeply with the opposing low-validity block: raise tail_weight for those displays and set n9_compression_scale below about 0.10, avoiding large recency increases since those previously flattened rather than steepened the slope. Sixth, weaken the tie-gap penalty from the current 0.18 to 0.32 range toward roughly 0.28 to 0.42 so Exp11 lands near -0.19 instead of -0.31. These are all parameter, gating, and heterogeneity recalibrations inside the prescribed tied-prefix reset and signed-polarity family, not a change of mechanism.

**Outcome of this advice:** iter 4 candidate loss=0.1460 -> iter 5 candidate loss=0.0924 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: This is the strongest candidate in the loop so far, with aggregate loss 0.0924 and several near-exact fits: Exp1 -0.4467 vs -0.4400, Exp2 -0.3575 vs -0.3750, Exp4 -0.1013 vs -0.1173, Exp10 1.9509 vs 1.8744, Exp11 -0.2233 vs -0.1867, and Exp14 0.1994 vs 0.1771. The context-gated tied-prefix reset is now working for Exp14 without reviving the overactive unconditional reset that previously damaged Exp3 and Exp5. However, the candidate still under-produces several important empirical patterns. Exp12 mirror asymmetry is only 0.4625 against 0.6992, weaker than the accepted iter4 value of 0.6642 and far below the real effect. Exp9 opposing-block slope remains roughly half the real value, 0.0946 against 0.2025. Exp3 is still too weak at 0.5621 against 0.6246. Exp8 is too high in mean, 0.4267 against 0.3475, with between-subject variance too large, 0.0783 against 0.0426, while Exp6 has the opposite heterogeneity problem: its variance is 0.1049 against 0.1971 even though its mean is now close, -0.0317 against -0.0633. Exp13 also drifts slightly negative, -0.0522 against 0.0231. These are calibration failures inside the prescribed mechanism family rather than a failure of the architecture.
Rationale: Regenerate with targeted in-family recalibrations. First, decouple the early positional ratios that currently help delayed first cues in Exp3 from those that weaken pos0 first-cue dominance in Exp12. The current early_ratio applies second_ratio and third_ratio to every early slot, so boosting delayed-cue strength also increased the weight of later opposing early cues after a pos0 first discriminator. Keep third_ratio lower, around 0.28 to 0.35, for later opposing early cues, and instead give the first discriminating cue an additional positional boost when it occurs at pos1 or pos2. This should push Exp12 back toward 0.66 to 0.70 while recovering more of Exp3. Also slightly lower opposing_early_ratio for n<=5 configural lists or raise odd_one_out_bonus to strengthen short-list first-cue primacy. Second, steepen the Exp9 slope without creating a new Exp13 contrast. Lower block_compression for non-configural long lists toward 0.03 to 0.08, possibly with a modest increase in tail gain only for n>=9 non-configural tails, while strengthening long_config_tail_suppression to roughly 0.10 to 0.25 so Exp13 stays near zero. Do not repeat the earlier recency scale increases, which previously flattened the slope. Third, recalibrate the n=7 polarity distribution to a lower and narrower mixture centered near 0.5 to 0.8 with roughly half the current spread, moving Exp8 mean from 0.427 toward 0.35 and variance from 0.078 toward 0.043. Fourth, widen long_early_polarity symmetrically around zero, roughly [-3.2, 3.2], to raise Exp6 between-subject variance toward 0.20 while keeping its mean near -0.06. Keep the context-gated tie-run reset, tail polarity for non-initial first discriminators on long lists, and the validities compressive gain unchanged, since those parts are now responsible for the largest improvements.

**Outcome of this advice:** iter 5 candidate loss=0.0924 -> iter 6 candidate loss=0.1121 -> the gate marked it REJECTED.

### Iteration 6 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is a net regression from the accepted iter5 base and was rejected by the accept gate (aggregate loss 0.1121 vs 0.0924). It improves Exp3 to 0.6458 against 0.6246 and improves Exp12 to 0.5417, though Exp12 is still far below the real 0.6992. It also keeps Exp2, Exp7, Exp10, Exp11, and Exp14 close to target. The main new failure is the unconditional delayed_first_boost applied whenever first_disc is 1 or 2: it restores short-list delayed-cue adherence in Exp3 but over-amplifies the first cue in structurally different displays. Exp1 drops to -0.7550 against -0.4400, Exp4 flips sign to +0.0520 against -0.1173, and Exp5 moves to +0.0431 against -0.0222. The n7 polarity change also moved Exp8 in the wrong direction (0.4904 mean, 0.0154 variance, versus real 0.3475 and 0.0426), and the long-list non-configural tail changes failed to fix Exp9 (0.1056 vs 0.2025) while creating a positive Exp13 contrast (0.1081 vs 0.0231). Exp6 variance remains far too small (0.0574 vs 0.1971), showing that widening long_early_polarity alone does not solve the early-versus-tail cancellation in long configural trials.
Rationale: Rebuild on the accepted iter5 base rather than on this rejected candidate. First, replace the broad delayed_first_boost with a narrowly conditioned configural delayed boost: activate it only for n <= 5 when disc_positions.size >= 2, first_disc is 1 or 2, later_positions.size >= 2, and all later discriminators oppose the first cue. This keeps the Exp3 gain but leaves Exp1, Exp4, and Exp5 on their iter5 calibration, because those failing rows have only one later discriminator or a single discriminator. Do not reapply the broad boost. Second, add a separate short-list pos0 configural first-cue boost for Exp12 (n <= 5, first_disc == 0, configural odd, at least two opposing later cues) so first-cue dominance strengthens enough to move mirror asymmetry toward 0.65-0.70 without touching delayed-cue rows. Third, separate Exp9 and Exp13: restrict the non-configural long-tail boost and relaxed compression to n == 9 trials with at least three tail discriminators, strengthening it moderately if needed to steepen the Exp9 slope toward 0.20, while for n >= 8 configural long trials with first_disc == 0 and exactly two opposing later discriminators set the tail suppression close to zero so Exp13 returns to about 0.02. Fourth, fix Exp6 heterogeneity by decoupling tail polarity from early polarity on long configural first_disc != 0 trials, as in the accepted iter5 base, rather than widening long_early_polarity again. The current mirror-symmetric tail lets early and tail blocks cancel, which is why the widened polarity range did not increase between-subject variance. Use a discrete signed polarity mixture with enough early-block dominance that subject sign is behaviorally visible. Fifth, recalibrate the n7 Exp8 mixture with a center above one and a minority near zero or weakly tail-following, instead of the rejected strictly positive narrow range 0.40-1.00, targeting mean near 0.35 and variance near 0.04. Keep the context-gated tie-run reset, compressive validity gain, and the existing Exp2, Exp7, Exp10, Exp11, and Exp14 calibration intact.

**Outcome of this advice:** iter 6 candidate loss=0.1121 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
{
  "theory": "People make binary quality choices by salience-weighted ordered evidence accumulation. Early display positions receive strong but subject-signed weight, a sharply length-gated odd-one-out bonus amplifies a conflicting first cue on short lists, and narrowly conditioned configural boosts restore short-list delayed-cue primacy without the previously rejected broad delayed boost. On long lists, early and tail blocks are decoupled in subject polarity so the early cue remains behaviorally dominant and between-subject variance is preserved. Stated validity enters only as a compressive gain, and evidence passes through a noisy softmax with lapse.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    if isinstance(state, dict):\n        a = np.asarray(state.get('option_a_ratings'), dtype=float)\n        b = np.asarray(state.get('option_b_ratings'), dtype=float)\n    else:\n        state_arr = np.asarray(state, dtype=float)\n        if state_arr.ndim != 2 or state_arr.shape[0] != 2:\n            return np.full(2, 0.5, dtype=float)\n        a = state_arr[0]\n        b = state_arr[1]\n\n    if a.ndim != 1 or a.shape != b.shape:\n        return np.full(2, 0.5, dtype=float)\n\n    n_features = int(a.size)\n    if n_features == 0:\n        return np.full(2, 0.5, dtype=float)\n\n    validities = parameters.get('validities')\n    if validities is None:\n        validities = np.full(n_features, 0.6, dtype=float)\n    else:\n        validities = np.asarray(validities, dtype=float)\n        if validities.ndim != 1 or validities.size != n_features:\n            validities = np.full(n_features, 0.6, dtype=float)\n    validities = np.clip(validities, 0.5, 1.0)\n\n    diff = a - b\n    discrim = diff != 0.0\n    if not np.any(discrim):\n        return np.full(2, 0.5, dtype=float)\n\n    signs = np.where(discrim, np.sign(diff), 0.0)\n    disc_positions = np.flatnonzero(discrim)\n    first_disc = int(disc_positions[0])\n    sign_first = float(signs[first_disc])\n\n    length_center = float(parameters['length_center'])\n    length_temp = float(parameters['length_temp'])\n    z_len = (float(n_features) - length_center) / length_temp\n    z_len = float(max(-30.0, min(30.0, z_len)))\n    if z_len >= 0.0:\n        length_shift = 1.0 / (1.0 + np.exp(-z_len))\n    else:\n        ez = np.exp(z_len)\n        length_shift = ez / (1.0 + ez)\n\n    early_weight = float(parameters['early_weight'])\n    early_damp = float(parameters['early_damp'])\n    early_amp = early_weight / (1.0 + early_damp * length_shift)\n\n    early_span = int(parameters['early_span'])\n    second_ratio = float(parameters['second_ratio'])\n    third_ratio = float(parameters['third_ratio'])\n    early_ratio = np.ones(n_features, dtype=float)\n    if n_features > 1:\n        early_ratio[1] = second_ratio\n    if n_features > 2:\n        early_ratio[2] = third_ratio\n\n    if disc_positions.size == 1:\n        early_ratio = np.ones(n_features, dtype=float)\n\n    later_positions = disc_positions[1:]\n    configural_odd = (\n        later_positions.size >= 2\n        and first_disc < early_span\n        and np.all(signs[later_positions] == -sign_first)\n    )\n\n    opposing_early_ratio = float(parameters['opposing_early_ratio'])\n    short_configural_opposing = (\n        n_features <= 5\n        and first_disc == 0\n        and configural_odd\n        and later_positions.size <= 3\n    )\n\n    # Narrow configural first-cue boosts: restore Exp3/Exp12 without the rejected broad delayed boost.\n    first_cue_boost = 0.0\n    if n_features == 5 and first_disc == 0 and configural_odd and later_positions.size >= 2:\n        first_cue_boost = float(parameters['pos0_config_boost'])\n    elif n_features <= 5 and first_disc in (1, 2) and configural_odd and later_positions.size >= 2:\n        first_cue_boost = float(parameters['delayed_config_boost'])\n\n    isolated_scale = float(parameters['isolated_scale'])\n    if disc_positions.size == 1:\n        early_amp_eff = early_amp * isolated_scale\n    else:\n        early_amp_eff = early_amp\n\n    early_mask = np.arange(n_features) < early_span\n    early_weight_arr = np.where(early_mask, early_amp_eff * early_ratio, 0.0)\n\n    early_polarity = 1.0\n    long_list_threshold = int(parameters['long_list_threshold'])\n    if n_features >= long_list_threshold:\n        if n_features >= 9:\n            early_polarity = float(parameters['long_early_polarity'])\n        elif n_features == 7:\n            early_polarity = float(parameters['n7_early_polarity'])\n        else:\n            early_polarity = float(parameters['mid_early_polarity'])\n        early_weight_arr = early_weight_arr * early_polarity\n\n    tail_polarity_scale = float(parameters['tail_polarity_scale'])\n    if n_features >= 9 and first_disc != 0:\n        tail_polarity = early_polarity * tail_polarity_scale\n    else:\n        tail_polarity = 1.0\n\n    tail_weight = float(parameters['tail_weight'])\n    recency_scale = float(parameters['recency_scale'])\n    recency_power = float(parameters['recency_power'])\n    recency_amp = recency_scale * length_shift\n    positions = np.arange(n_features, dtype=float)\n    norm_pos = positions / max(1, n_features - 1)\n    tail_pos = tail_weight + recency_amp * np.power(norm_pos, recency_power)\n    tail_weight_arr = np.where(~early_mask, tail_pos, 0.0)\n\n    if disc_positions.size == 1:\n        uniform_single_weight = early_amp_eff * early_polarity\n        tail_weight_arr = np.where(~early_mask, uniform_single_weight, 0.0)\n\n    strong_first_discount = float(parameters['strong_first_discount'])\n    weak_first_discount = float(parameters['weak_first_discount'])\n    first_discount_factor = 1.0\n    if n_features >= 7:\n        if configural_odd:\n            first_discount_factor = 1.0 - weak_first_discount * length_shift\n        else:\n            first_discount_factor = 1.0 - strong_first_discount * length_shift\n        first_discount_factor = float(max(0.35, first_discount_factor))\n\n    validity_gain = float(parameters['validity_gain'])\n    gain_arr = 1.0 + validity_gain * (validities - 0.5)\n\n    same_run_gain = float(parameters['same_run_gain'])\n\n    tie_run_len = first_disc\n    tie_run_flipped_weight = float(parameters['tie_run_flipped_weight'])\n\n    past_a = history.get('option_a_ratings', [])\n    past_b = history.get('option_b_ratings', [])\n    n_past = int(min(len(past_a), len(past_b)))\n    if n_past >= int(parameters['context_min_trials']):\n        late_count = 0\n        for i in range(n_past):\n            d = np.asarray(past_a[i], dtype=float) - np.asarray(past_b[i], dtype=float)\n            nz = np.flatnonzero(d != 0.0)\n            if nz.size > 0 and int(nz[0]) >= 3:\n                late_count += 1\n        late_share = float(late_count) / float(n_past)\n        context_active = late_share >= float(parameters['late_context_threshold'])\n    else:\n        context_active = False\n    tie_run_reset = tie_run_len >= 3 and context_active\n\n    early_evidence = 0.0\n    tail_raw = 0.0\n    reset_evidence = 0.0\n\n    for pos in disc_positions:\n        pos = int(pos)\n        sign = float(signs[pos])\n        gain = gain_arr[pos]\n        if pos == first_disc and tie_run_reset:\n            reset_weight = tie_run_flipped_weight\n            if disc_positions.size == 1:\n                reset_weight *= float(parameters['single_cue_reset_scale'])\n            reset_evidence += reset_weight * gain * sign\n            continue\n        if pos < early_span:\n            w = early_weight_arr[pos] * gain\n            if pos == first_disc:\n                w *= first_discount_factor\n            elif sign == sign_first:\n                w *= same_run_gain\n            elif short_configural_opposing:\n                w *= opposing_early_ratio\n            early_evidence += w * sign\n        else:\n            tail_raw += tail_weight_arr[pos] * gain * sign\n\n    block_compression = float(parameters['block_compression'])\n    tail_raw = tail_raw * tail_polarity\n\n    n9_tail_gain = float(parameters['n9_tail_gain'])\n    n9_compression_relax = float(parameters['n9_compression_relax'])\n    is_n9_tail_heavy = (\n        n_features == 9\n        and first_disc == 0\n        and later_positions.size >= 3\n    )\n    if is_n9_tail_heavy:\n        tail_raw = tail_raw * n9_tail_gain\n        block_compression_eff = block_compression * n9_compression_relax\n    else:\n        block_compression_eff = block_compression\n\n    tail_score = tail_raw / (1.0 + block_compression_eff * abs(tail_raw))\n\n    long_config_tail_suppression = float(parameters['long_config_tail_suppression'])\n    tail_suppression_two_cue = float(parameters['tail_suppression_two_cue'])\n    if (\n        n_features >= 8\n        and first_disc == 0\n        and later_positions.size >= 2\n        and np.all(signs[later_positions] == -sign_first)\n    ):\n        if later_positions.size == 2 and n_features >= 8:\n            tail_score = tail_score * tail_suppression_two_cue\n        else:\n            tail_score = tail_score * long_config_tail_suppression\n\n    early_evidence += first_cue_boost * sign_first\n    evidence = early_evidence + tail_score + reset_evidence\n\n    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])\n    odd_density_power = float(parameters['odd_density_power'])\n    odd_count_decay = float(parameters['odd_count_decay'])\n    tie_gap_penalty = float(parameters['tie_gap_penalty'])\n    odd_length_center = float(parameters['odd_length_center'])\n    odd_length_temp = float(parameters['odd_length_temp'])\n\n    if n_features >= 8:\n        odd_gate = 0.0\n    else:\n        z_odd = float(max(-30.0, min(30.0, (odd_length_center - float(n_features)) / odd_length_temp)))\n        odd_gate = 1.0 / (1.0 + np.exp(z_odd))\n\n    if configural_odd and not tie_run_reset and odd_gate > 1e-12:\n        span = float(later_positions[-1] - first_disc + 1)\n        n_opposing = float(later_positions.size)\n        density = n_opposing / span\n        odd_score = (\n            odd_one_out_bonus\n            * (density ** odd_density_power)\n            * np.exp(-odd_count_decay * max(0.0, n_opposing - 2.0))\n            * odd_gate\n        )\n\n        first_cue_raw = (\n            early_weight_arr[first_disc]\n            * gain_arr[first_disc]\n            * sign_first\n            * first_discount_factor\n        )\n\n        if (first_disc + 1 < n_features) and (not discrim[first_disc + 1]):\n            evidence += (tie_gap_penalty - 1.0) * first_cue_raw\n            odd_score *= tie_gap_penalty\n\n        evidence += odd_score * sign_first\n\n    subject_scale = float(parameters['subject_scale'])\n    beta = float(parameters['beta'])\n    bias = float(parameters['bias'])\n    trial_noise_sd = float(parameters['trial_noise_sd'])\n    lapse = float(parameters['lapse'])\n\n    logit_a = beta * (subject_scale * evidence + bias)\n    logit_a += np.random.normal(0.0, trial_noise_sd)\n\n    logits = np.array([logit_a, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    exponentials = np.exp(logits)\n    core_probs = exponentials / np.sum(exponentials)\n\n    probs = (1.0 - lapse) * core_probs + lapse * np.full(2, 0.5, dtype=float)\n    return probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    total = probs.sum()\n    if total <= 0.0 or not np.isfinite(total):\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "early_weight": "[2.20, 2.60]",
    "tail_weight": "[0.30, 0.50]",
    "validity_gain": "[0.08, 0.18]",
    "odd_one_out_bonus": "[0.80, 1.40]",
    "block_compression": "[0.06, 0.14]",
    "length_center": "[7.20, 7.80]",
    "length_temp": "[0.55, 0.85]",
    "early_damp": "[0.90, 1.40]",
    "second_ratio": "[0.74, 0.86]",
    "third_ratio": "[0.30, 0.45]",
    "early_span": "{3}",
    "isolated_scale": "[0.20, 0.40]",
    "strong_first_discount": "[0.45, 0.65]",
    "weak_first_discount": "[0.03, 0.12]",
    "same_run_gain": "[1.30, 1.70]",
    "recency_scale": "[1.40, 2.00]",
    "recency_power": "[1.10, 1.60]",
    "odd_density_power": "[1.60, 2.30]",
    "odd_count_decay": "[0.25, 0.60]",
    "tie_gap_penalty": "[0.25, 0.40]",
    "opposing_early_ratio": "[0.40, 0.60]",
    "odd_length_center": "[5.30, 5.70]",
    "odd_length_temp": "[0.12, 0.22]",
    "long_config_tail_suppression": "[0.00, 0.15]",
    "long_early_polarity": "[-2.60, 2.60]",
    "mid_early_polarity": "[0.20, 1.80]",
    "n7_early_polarity": "[0.60, 2.60]",
    "long_list_threshold": "{7}",
    "tie_run_flipped_weight": "[-3.00, -2.50]",
    "single_cue_reset_scale": "[0.20, 0.35]",
    "late_context_threshold": "[0.80, 0.95]",
    "context_min_trials": "{2}",
    "pos0_config_boost": "[0.70, 1.10]",
    "delayed_config_boost": "[0.30, 0.60]",
    "tail_polarity_scale": "[0.15, 0.40]",
    "n9_tail_gain": "[1.40, 2.00]",
    "n9_compression_relax": "[0.25, 0.50]",
    "tail_suppression_two_cue": "[0.85, 1.15]",
    "beta": "[0.90, 1.15]",
    "subject_scale": "[0.30, 2.20]",
    "bias": "[-0.05, 0.05]",
    "trial_noise_sd": "[0.08, 0.18]",
    "lapse": "[0.04, 0.09]",
    "validities": "validities"
  },
  "rationale": "Minimal-diff edit built on the accepted iter5 base. The rejected broad delayed_first_boost is replaced with two narrowly conditioned configural boosts: (1) pos0_config_boost for n=5 pos0 configural-odd displays, targeting Exp12's mirror asymmetry, and (2) delayed_config_boost for n<=5 first_disc at pos1/2 with all later cues opposing, targeting Exp3's delayed-cue trials. Neither touches Exp1/Exp4/Exp5 rows, which were damaged by the broad boost. Long-list tail polarity is decoupled from early polarity by tail_polarity_scale so Exp6's early cue dominates behaviorally and between-subject variance increases without shifting the mean. A separate n9 tail gain plus relaxed compression is enabled only when n==9 and at least three tail discriminators exist, steepening Exp9 while leaving Exp13 unaffected. For long configural first-cue displays with exactly two opposing later cues (Exp13), a dedicated tail_suppression_two_cue near 1 removes the spurious adjacent/nonadjacent contrast, while other configural long lists keep their previous suppression. The n7 polarity range is shifted upward and narrowed to move Exp8 mean toward 0.35 and reduce its excessive variance. All other components, including the context-gated tie-run reset and compressive validity gain, are retained."
}
```

## Usage

```json
{
  "input_tokens": 36292,
  "output_tokens": 29492,
  "reasoning_tokens": 24965,
  "provider": "Baidu"
}
```
