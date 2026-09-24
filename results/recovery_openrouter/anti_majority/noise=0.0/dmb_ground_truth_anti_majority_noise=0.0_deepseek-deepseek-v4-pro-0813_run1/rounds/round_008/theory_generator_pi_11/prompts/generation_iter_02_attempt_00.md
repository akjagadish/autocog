# generation_iter_02_attempt_00

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
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_9`).

Replace pi_9 with a 'run-gated ordered evidence accumulation' theory. The core idea is that the first discriminating cue's weight is not a fixed early-block polarity, but a credibility signal that depends on the local cue-run structure, list length, and recent delayed-cue context. The new theory should include: (1) a strong tie-gap penalty applied to the first cue whenever the immediately following position is tied, strong enough that a single opposing later cue can reverse the choice, explaining Experiment 2; (2) a length-band early polarity: positive on short lists except when tie-gap suppressed, negative at n=8, a subject-level mixture of zero and moderate positive at n=7, and position-1 discounting on n>=9 outside two-cue configural displays; (3) a context-graded negative weight for delayed first discriminating cues, strengthened by recent delayed-first-cue trials, explaining Experiments 8, 16, and 18; (4) a configural odd-one-out bonus on short lists that is gated off by n=6/8, with density/count decay and tie-gap interaction, explaining Experiments 5, 13, 14, and 17; (5) long-list configural suppression and equalization of adjacent versus nonadjacent later cues in two-cue configural displays, explaining Experiments 11 and 15; (6) validity as a compressive attentional gain plus a validity-order/tie-gap gate that releases the tie-gap penalty only when the opposing late cue has validity at least as high as the first cue; (7) subject-level mirror asymmetry and noisy softmax with lapse. This differs from pi_9 by adopting negative n=8 early polarity, a general tie-gap penalty, and delayed-first-cue anti-weighting; it differs from pi_10 by making the tie-gap penalty strong enough to produce negative first-cue following in Experiment 2, moderating the n=7 early branch to avoid overpredicting Experiment 10, and reducing the n=10 high-cue effect to match Experiment 12.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.856)

**Description**
People make binary quality choices through a bounded ordered evidence-accumulation process with three interacting components. Early discriminating cues receive strong primacy, but that primacy is length-band-specific: on very short lists (n<=5) a narrow validity-order/tie-gap gate suppresses a single opposing late cue inside the early span, producing the strong anti-first-validity signature; on n=7 lists the early block is a subject-level mixture of a zero-early branch and a moderately positive branch; on n=8 lists the early block is subject-signed negative; and on long lists (n>=9) the position-1 early cue is discounted outside of two-cue configural displays, while configural late blocks are suppressed on very long displays. A weakly weighted late tally block accumulates ordinary later cues, while delayed first discriminating cues receive a context-graded negative weight, making isolated late cues fall near chance after delayed-first-cue experience. The configural odd-one-out bonus is moderate on short lists and nearly off by n=6, and long two-cue configural displays are explicitly equalized across adjacent and nonadjacent later-cue positions. Stated validity enters only as a compressive attentional gain. A tie-gap immediately after the first cue suppresses both the first cue's direct evidence and the configural bonus, except in the narrow validity-order/tie-gap case where it is released. A subject-level mirror asymmetry generates the positive mirror effect. Choices arise from a noisy softmax with subject scale, bias, trial noise, and lapse.

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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.4050 (var=0.0444)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3938 (var=0.0108)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6050 (var=0.0027)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=-0.1893 (var=0.0330)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.0314 (var=0.0107)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=-0.0667 (var=0.0208)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.1232 (var=0.0004)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.5050 (var=0.0476)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.2019 (var=0.0027)
- Experiment 10: real=1.8744 (var=0.7034) vs this=2.3712 (var=0.1219)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=-0.2267 (var=0.0211)
- Experiment 12: real=0.6992 (var=0.0351) vs this=0.6421 (var=0.0128)
- Experiment 13: real=0.0231 (var=0.0947) vs this=0.0463 (var=0.0151)
- Experiment 14: real=0.1771 (var=0.0120) vs this=0.1479 (var=0.0038)
- Experiment 15: real=0.0344 (var=0.0057) vs this=0.0014 (var=0.0113)
- Experiment 16: real=0.1383 (var=0.0077) vs this=0.1238 (var=0.0039)
- Experiment 17: real=-0.7433 (var=0.0295) vs this=-0.6850 (var=0.0429)
- Experiment 18: real=-0.2567 (var=0.0187) vs this=0.0371 (var=0.0018)


---

### `pi_9` (overall score: 0.503)

**Description**
People make binary quality choices by salience-weighted ordered evidence accumulation with subject-signed early-span polarity, a sharply length-gated odd-one-out bonus on short lists, and a context-gated tie-run reset for delayed first cues. On long lists early and tail blocks are decoupled so subject sign remains behaviorally visible, and exactly-two-cue long configural displays are equalized across adjacent and nonadjacent later-cue positions. Stated validity is a compressive attentional gain, and choices arise from noisy softmax with lapse.

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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.5733 (var=0.0555)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3538 (var=0.0125)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.6344 (var=0.0042)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=-0.0947 (var=0.0247)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.0067 (var=0.0091)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=0.0400 (var=0.1497)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.1311 (var=0.0013)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.3821 (var=0.1176)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.1514 (var=0.0159)
- Experiment 10: real=1.8744 (var=0.7034) vs this=1.8390 (var=0.6546)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=-0.2000 (var=0.0289)
- Experiment 12: real=0.6992 (var=0.0351) vs this=0.7229 (var=0.0362)
- Experiment 13: real=0.0231 (var=0.0947) vs this=0.0272 (var=0.0044)
- Experiment 14: real=0.1771 (var=0.0120) vs this=0.1812 (var=0.0060)
- Experiment 15: real=0.0344 (var=0.0057) vs this=-0.0875 (var=0.0242)
- Experiment 16: real=0.1383 (var=0.0077) vs this=0.2943 (var=0.0107)
- Experiment 17: real=-0.7433 (var=0.0295) vs this=0.7692 (var=0.0418)
- Experiment 18: real=-0.2567 (var=0.0187) vs this=0.2744 (var=0.0110)


---

### `pi_8` (overall score: 0.459)

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
- Experiment 15: real=0.0344 (var=0.0057) vs this=0.0617 (var=0.0105)
- Experiment 16: real=0.1383 (var=0.0077) vs this=0.7030 (var=0.0052)
- Experiment 17: real=-0.7433 (var=0.0295) vs this=0.7017 (var=0.0468)
- Experiment 18: real=-0.2567 (var=0.0187) vs this=0.2042 (var=0.0078)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0949 -> ACCEPTED
- iter 2: loss=0.1081 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0949 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.4317 (var=0.0419) (Δ vs real +0.0083)
  - iter 2 (most recent): -0.4217 (var=0.0476) (Δ vs real +0.0183)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6650 (var=0.0546)
- pi_2: 0.0383 (var=0.0359)
- pi_3: -0.6500 (var=0.0231)
- pi_4: -0.4933 (var=0.0313)
- pi_5: -0.5700 (var=0.1570)
- pi_6: -0.3200 (var=0.1409)
- pi_7: -0.5133 (var=0.1857)
- pi_8: -0.4883 (var=0.0319)
- pi_9: -0.5733 (var=0.0555)
- pi_10: -0.4050 (var=0.0444)

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
  - iter 1: -0.3946 (var=0.0082) (Δ vs real -0.0196)
  - iter 2 (most recent): -0.0804 (var=0.0068) (Δ vs real +0.2946)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0029 (var=0.0086)
- pi_1: 0.6875 (var=0.0362)
- pi_3: -0.3296 (var=0.0048)
- pi_4: -0.3458 (var=0.0038)
- pi_5: -0.2871 (var=0.0275)
- pi_6: -0.3213 (var=0.0070)
- pi_7: -0.3442 (var=0.0247)
- pi_8: -0.3658 (var=0.0115)
- pi_9: -0.3538 (var=0.0125)
- pi_10: -0.3938 (var=0.0108)

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
  - iter 1: 0.5800 (var=0.0019) (Δ vs real -0.0446)
  - iter 2 (most recent): 0.5837 (var=0.0027) (Δ vs real -0.0408)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8567 (var=0.0014)
- pi_2: 0.2706 (var=0.0057)
- pi_1: 0.6206 (var=0.0018)
- pi_4: 0.6208 (var=0.0020)
- pi_5: 0.6331 (var=0.0177)
- pi_6: 0.6160 (var=0.0094)
- pi_7: 0.5504 (var=0.0048)
- pi_8: 0.6619 (var=0.0059)
- pi_9: 0.6344 (var=0.0042)
- pi_10: 0.6050 (var=0.0027)

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
  - iter 1: -0.1773 (var=0.0286) (Δ vs real -0.0600)
  - iter 2 (most recent): -0.1573 (var=0.0282) (Δ vs real -0.0400)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0147 (var=0.0280)
- pi_3: 0.7160 (var=0.0149)
- pi_1: 0.2253 (var=0.0167)
- pi_4: 0.0280 (var=0.0391)
- pi_5: 0.0253 (var=0.0490)
- pi_6: -0.1307 (var=0.0469)
- pi_7: 0.0827 (var=0.0522)
- pi_8: -0.0400 (var=0.0197)
- pi_9: -0.0947 (var=0.0247)
- pi_10: -0.1893 (var=0.0330)

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
  - iter 1: -0.0092 (var=0.0174) (Δ vs real +0.0131)
  - iter 2 (most recent): 0.0003 (var=0.0140) (Δ vs real +0.0225)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1878 (var=0.0101)
- pi_4: -0.2019 (var=0.0042)
- pi_1: 0.0092 (var=0.0061)
- pi_2: 0.0058 (var=0.0086)
- pi_5: 0.0267 (var=0.0064)
- pi_6: -0.0006 (var=0.0043)
- pi_7: -0.0075 (var=0.0053)
- pi_8: -0.0925 (var=0.0097)
- pi_9: -0.0067 (var=0.0091)
- pi_10: -0.0314 (var=0.0107)

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
  - iter 1: 0.0283 (var=0.0212) (Δ vs real +0.0917)
  - iter 2 (most recent): 0.0583 (var=0.0135) (Δ vs real +0.1217)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6033 (var=0.0124)
- pi_3: -0.8958 (var=0.0049)
- pi_1: -0.6992 (var=0.0453)
- pi_2: 0.4608 (var=0.0209)
- pi_5: -0.0617 (var=0.2674)
- pi_6: -0.0858 (var=0.1788)
- pi_7: 0.0942 (var=0.2179)
- pi_8: -0.0108 (var=0.0280)
- pi_9: 0.0400 (var=0.1497)
- pi_10: -0.0667 (var=0.0208)

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
  - iter 1: 0.1204 (var=0.0005) (Δ vs real +0.0105)
  - iter 2 (most recent): 0.1298 (var=0.0004) (Δ vs real +0.0199)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1186 (var=0.0016)
- pi_4: 0.0406 (var=0.0002)
- pi_1: 0.1334 (var=0.0041)
- pi_2: 0.0187 (var=0.0001)
- pi_3: 0.0541 (var=0.0003)
- pi_6: 0.0923 (var=0.0010)
- pi_7: 0.1140 (var=0.0020)
- pi_8: 0.1079 (var=0.0012)
- pi_9: 0.1311 (var=0.0013)
- pi_10: 0.1232 (var=0.0004)

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
  - iter 1: 0.4179 (var=0.0371) (Δ vs real +0.0704)
  - iter 2 (most recent): 0.3967 (var=0.0380) (Δ vs real +0.0492)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9329 (var=0.0017)
- pi_5: 0.4883 (var=0.0613)
- pi_1: 0.5050 (var=0.0022)
- pi_2: 0.8471 (var=0.0113)
- pi_3: 0.4933 (var=0.0007)
- pi_6: 0.2933 (var=0.0149)
- pi_7: 0.2300 (var=0.0296)
- pi_8: 0.4029 (var=0.0044)
- pi_9: 0.3821 (var=0.1176)
- pi_10: 0.5050 (var=0.0476)

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
  - iter 1: 0.2040 (var=0.0019) (Δ vs real +0.0015)
  - iter 2 (most recent): 0.2014 (var=0.0026) (Δ vs real -0.0011)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0824 (var=0.0041)
- pi_6: 0.0260 (var=0.0004)
- pi_1: 0.0349 (var=0.0006)
- pi_2: 0.2730 (var=0.0071)
- pi_3: 0.0405 (var=0.0008)
- pi_4: 0.3544 (var=0.0008)
- pi_7: 0.1950 (var=0.0096)
- pi_8: 0.1761 (var=0.0046)
- pi_9: 0.1514 (var=0.0159)
- pi_10: 0.2019 (var=0.0027)

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
  - iter 1: 2.4289 (var=0.1277) (Δ vs real +0.5545)
  - iter 2 (most recent): 2.3153 (var=0.1235) (Δ vs real +0.4410)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4099 (var=0.0378)
- pi_5: 0.7166 (var=0.1141)
- pi_1: 0.8464 (var=0.3893)
- pi_2: 2.0302 (var=0.7154)
- pi_3: 0.2507 (var=0.0279)
- pi_4: 2.8162 (var=0.0908)
- pi_7: 2.2173 (var=0.1964)
- pi_8: 1.9747 (var=0.3019)
- pi_9: 1.8390 (var=0.6546)
- pi_10: 2.3712 (var=0.1219)

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
  - iter 1: -0.3275 (var=0.0170) (Δ vs real -0.1408)
  - iter 2 (most recent): -0.1950 (var=0.0188) (Δ vs real -0.0083)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0992 (var=0.0345)
- pi_7: 0.3383 (var=0.1102)
- pi_1: 0.0133 (var=0.0088)
- pi_2: 0.0150 (var=0.0106)
- pi_3: 0.0067 (var=0.0021)
- pi_4: -0.0050 (var=0.0059)
- pi_6: 0.0358 (var=0.0131)
- pi_8: -0.1300 (var=0.0159)
- pi_9: -0.2000 (var=0.0289)
- pi_10: -0.2267 (var=0.0211)

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
  - iter 1: 0.5938 (var=0.0089) (Δ vs real -0.1054)
  - iter 2 (most recent): 0.6312 (var=0.0118) (Δ vs real -0.0679)
**Other theories' values on this metric (for reference):**
- pi_7: -0.3475 (var=0.3307)
- pi_5: 0.1979 (var=0.2805)
- pi_1: -0.6408 (var=0.0276)
- pi_2: -0.6800 (var=0.0406)
- pi_3: 0.5900 (var=0.0052)
- pi_4: 0.5504 (var=0.0067)
- pi_6: 0.5308 (var=0.0471)
- pi_8: 0.5371 (var=0.0273)
- pi_9: 0.7229 (var=0.0362)
- pi_10: 0.6421 (var=0.0128)

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
  - iter 1: 0.0872 (var=0.0071) (Δ vs real +0.0641)
  - iter 2 (most recent): 0.1706 (var=0.0168) (Δ vs real +0.1475)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0009 (var=0.1257)
- pi_8: 0.3244 (var=0.0219)
- pi_1: 0.0041 (var=0.0075)
- pi_2: 0.0078 (var=0.0051)
- pi_3: 0.0056 (var=0.0015)
- pi_4: 0.0028 (var=0.0025)
- pi_6: 0.0178 (var=0.0185)
- pi_7: -0.1825 (var=0.0584)
- pi_9: 0.0272 (var=0.0044)
- pi_10: 0.0463 (var=0.0151)

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
  - iter 1: 0.1062 (var=0.0023) (Δ vs real -0.0708)
  - iter 2 (most recent): 0.2796 (var=0.0046) (Δ vs real +0.1025)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5848 (var=0.0039)
- pi_5: 0.8456 (var=0.0038)
- pi_1: 0.8619 (var=0.0114)
- pi_2: 0.7617 (var=0.0062)
- pi_3: 0.9594 (var=0.0005)
- pi_4: 0.5996 (var=0.0031)
- pi_6: 0.8081 (var=0.0040)
- pi_7: 0.8469 (var=0.0027)
- pi_9: 0.1812 (var=0.0060)
- pi_10: 0.1479 (var=0.0038)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.stack(data['option_a_ratings'].to_numpy()).astype(float)
    b = np.stack(data['option_b_ratings'].to_numpy()).astype(float)
    diff = a - b
    discrim = diff != 0.0
    first_pos = np.argmax(discrim, axis=1)
    rows = np.arange(len(data))
    first_sign = np.sign(diff[rows, first_pos])
    expected_first_cue = np.where(first_sign > 0.0, 0, 1)
    first_cue_match = (data['response'].to_numpy() == expected_first_cue).astype(float)
    n_disc = discrim.sum(axis=1)
    focal = ((first_pos == 0)
             & (n_disc == 3)
             & discrim[:, 1]
             & discrim[:, 2]
             & (np.sign(diff[:, 1]) == -first_sign)
             & (np.sign(diff[:, 2]) == -first_sign))
    if focal.sum() == 0 or (~focal).sum() == 0:
        return 0.0
    return float(first_cue_match[focal].mean() - first_cue_match[~focal].mean())
```

**Observed (real) value:** 0.0344 (var=0.0057)
**Previous candidate values (this loop):**
  - iter 1: 0.0172 (var=0.0168) (Δ vs real -0.0172)
  - iter 2 (most recent): 0.0114 (var=0.0158) (Δ vs real -0.0231)
**Other theories' values on this metric (for reference):**
- pi_9: -0.0875 (var=0.0242)
- pi_8: 0.0617 (var=0.0105)
- pi_1: -0.0036 (var=0.0059)
- pi_2: 0.0197 (var=0.0066)
- pi_3: -0.0381 (var=0.0100)
- pi_4: -0.0058 (var=0.0039)
- pi_5: -0.0056 (var=0.1554)
- pi_6: 0.0956 (var=0.0807)
- pi_7: -0.0428 (var=0.0585)
- pi_10: 0.0014 (var=0.0113)

### Experiment 16
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    df = data.copy()
    df['trial_within_subject'] = df.groupby('subject_id').cumcount()
    post = df[df['trial_within_subject'] >= 2].copy()
    if post.empty:
        return 0.5
    choices = []
    for _, row in post.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        nz = np.flatnonzero(diff != 0.0)
        if nz.size != 1:
            continue
        pos = int(nz[0])
        sign = float(diff[pos])
        if pos < 3 or sign == 0.0:
            continue
        cue_side = 0 if sign > 0.0 else 1
        choices.append(1.0 if int(row['response']) == cue_side else 0.0)
    if not choices:
        return 0.5
    return float(np.mean(choices))
```

**Observed (real) value:** 0.1383 (var=0.0077)
**Previous candidate values (this loop):**
  - iter 1: 0.0689 (var=0.0016) (Δ vs real -0.0694)
  - iter 2 (most recent): 0.2377 (var=0.0053) (Δ vs real +0.0994)
**Other theories' values on this metric (for reference):**
- pi_8: 0.7030 (var=0.0052)
- pi_9: 0.2943 (var=0.0107)
- pi_1: 0.8451 (var=0.0068)
- pi_2: 0.8664 (var=0.0099)
- pi_3: 0.9372 (var=0.0007)
- pi_4: 0.9374 (var=0.0005)
- pi_5: 0.4660 (var=0.0182)
- pi_6: 0.9013 (var=0.0009)
- pi_7: 0.7343 (var=0.0098)
- pi_10: 0.1238 (var=0.0039)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0
    try:
        a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
        y = np.asarray(data['response'].tolist(), dtype=float)
    except Exception:
        return 0.0
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != y.size:
        return 0.0
    d = a - b
    diff_mask = d != 0.0
    if not np.any(diff_mask):
        return 0.0
    n_rows = a.shape[0]
    first_idx = np.argmax(diff_mask, axis=1)
    first_sign = np.sign(d[np.arange(n_rows), first_idx])
    second_mask = diff_mask.copy()
    second_mask[np.arange(n_rows), first_idx] = False
    has_second = np.any(second_mask, axis=1)
    second_idx = np.argmax(second_mask, axis=1)
    second_sign = np.sign(d[np.arange(n_rows), second_idx])
    qualified = has_second & (second_sign == first_sign)
    if not np.any(qualified):
        return 0.0
    fav_a = first_sign > 0.0
    fav_b = first_sign < 0.0
    chose_a = y == 0.0
    chose_b = y == 1.0
    match = (fav_a & chose_a) | (fav_b & chose_b)
    signed = np.where(match, 1.0, -1.0)
    return float(np.mean(signed[qualified]))
```

**Observed (real) value:** -0.7433 (var=0.0295)
**Previous candidate values (this loop):**
  - iter 1: -0.8683 (var=0.0071) (Δ vs real -0.1250)
  - iter 2 (most recent): -0.7567 (var=0.0301) (Δ vs real -0.0133)
**Other theories' values on this metric (for reference):**
- pi_9: 0.7692 (var=0.0418)
- pi_10: -0.6850 (var=0.0429)
- pi_1: 0.6992 (var=0.0254)
- pi_2: 0.6783 (var=0.0466)
- pi_3: 0.9050 (var=0.0049)
- pi_4: 0.8917 (var=0.0043)
- pi_5: -0.0108 (var=0.2687)
- pi_6: 0.7867 (var=0.0090)
- pi_7: 0.8517 (var=0.0076)
- pi_8: 0.7017 (var=0.0468)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.asarray([list(x) for x in data['option_a_ratings']], dtype=np.float64)
    b = np.asarray([list(x) for x in data['option_b_ratings']], dtype=np.float64)
    diff = a - b
    disc = diff != 0.0
    first_idx = np.argmax(disc, axis=1)
    rows = np.arange(diff.shape[0])
    first_diff = diff[rows, first_idx]
    first_cue_response = np.where(first_diff > 0.0, 0, 1)
    chosen_first = (np.asarray(data['response'], dtype=int) == first_cue_response)
    return float(np.mean(chosen_first) - 0.5)
```

**Observed (real) value:** -0.2567 (var=0.0187)
**Previous candidate values (this loop):**
  - iter 1: -0.0619 (var=0.0029) (Δ vs real +0.1948)
  - iter 2 (most recent): -0.0898 (var=0.0026) (Δ vs real +0.1669)
**Other theories' values on this metric (for reference):**
- pi_10: 0.0371 (var=0.0018)
- pi_9: 0.2744 (var=0.0110)
- pi_1: 0.3335 (var=0.0133)
- pi_2: -0.0154 (var=0.0024)
- pi_3: 0.4056 (var=0.0009)
- pi_4: 0.3206 (var=0.0009)
- pi_5: 0.3473 (var=0.0068)
- pi_6: 0.3073 (var=0.0061)
- pi_7: 0.3965 (var=0.0024)
- pi_8: 0.2042 (var=0.0078)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Run-gated ordered evidence accumulation. The first discriminating cue's weight is not a fixed early-block polarity but a credibility signal that depends on local cue-run structure, list length, and recent delayed-cue context. A strong tie-gap penalty suppresses the first cue whenever the next position is tied, unless a validity-order gate releases it because the opposing late cue has validity at least as high as the first cue. Early polarity is length-band-specific: positive on short lists, negative at n=8, a subject-level mixture of zero and moderate positive at n=7, and position-1 discounted on n>=9 outside two-cue configural displays. Delayed first discriminating cues receive a context-graded negative weight strengthened by recent delayed-first-cue trials. A configural odd-one-out bonus operates on short lists with density/count decay and tie-gap interaction, and is gated off by n=6/8. Long two-cue configural displays show early suppression and equalization of adjacent versus nonadjacent later cues. Stated validity is a compressive attentional gain, and choices arise from a noisy softmax with subject scale, bias, trial noise, lapse, and mirror asymmetry.

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

    release_penalty = False
    isolated_high_validity_opposing = False
    if tie_gap and first_disc < early_span and later_positions.size == 1 and n_features <= 5:
        lp = int(later_positions[0])
        if signs[lp] != sign_first and validities[lp] >= validities[first_disc]:
            isolated_high_validity_opposing = True
    release_penalty = isolated_high_validity_opposing

    short_single_opposing_ratio = float(parameters['short_single_opposing_ratio'])
    long_pos1_ratio = float(parameters['long_pos1_ratio'])
    first_pos1_discount = float(parameters['first_pos1_discount'])
    n10_high_cue_discount = float(parameters['n10_high_cue_discount'])

    early_evidence = 0.0
    tail_raw = 0.0

    for pos in disc_positions:
        pos = int(pos)
        sign = float(signs[pos])
        gain = gain_arr[pos]

        if pos == first_disc:
            if first_disc < late_context_position:
                w = early_weight_arr[pos] * gain * first_discount_factor * first_sign_mult
                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:
                    w = w * first_pos1_discount
                if tie_gap and not release_penalty:
                    w = w * tie_gap_penalty
                early_evidence += w * sign
            elif first_disc == late_context_position and context_strength < late_context_activation:
                tail_raw += tail_weight_arr[pos] * gain * sign
                continue
            else:
                w = -late_first_anti * gain * first_sign_mult
                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:
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
- opposing_early_ratio: [0.40, 0.60]
- odd_length_center: [4.80, 5.20]
- odd_length_temp: [0.30, 0.50]
- long_two_cue_early_suppression: [0.30, 0.70]
- tail_suppression_two_cue: [0.02, 0.10]
- long_early_polarity: [0.45, 0.80]
- mid_early_polarity: [-1.80, -0.70]
- n7_branch_selector: [0, 1]
- n7_early_zero_prob: [0.08, 0.20]
- n7_positive_polarity: [1.10, 1.60]
- long_list_threshold: {7}
- late_threshold: {3}
- late_anti_base: [1.00, 1.50]
- late_context_gain: [0.80, 1.60]
- late_length_scale: [0.20, 0.50]
- late_anti_unconditional_threshold: {4}
- late_context_position_threshold: {3}
- late_context_activation: [0.15, 0.30]
- context_min_trials: {2}
- context_window: {24}
- tail_context_gain: [0.00, 0.10]
- short_config_boost: [0.40, 0.80]
- short_single_opposing_ratio: [0.00, 0.06]
- long_pos1_ratio: [0.20, 0.45]
- first_pos1_discount: [0.35, 0.65]
- n10_high_cue_discount: [0.25, 0.50]
- mirror_asym: [1.08, 1.26]
- long_config_heavy_tail_suppression: [0.00, 0.15]
- beta: [0.85, 1.10]
- subject_scale: [0.70, 1.30]
- bias: [-0.05, 0.05]
- trial_noise_sd: [0.05, 0.15]
- lapse: [0.04, 0.09]
- validities: validities

`rationale`: This theory replaces the fixed early-block polarity of pi_9 with a run-gated credibility signal. The strong tie-gap penalty is applied whenever the next position is tied, making a single opposing later cue able to reverse the choice in Experiments 2 and 18. The length-band early polarity is negative at n=8, a zero/moderate-positive mixture at n=7, and includes first-cue position-1 discounting on n>=9, which reduces the n=12 position-1 effect and the n=10 high-cue effect. Delayed first discriminating cues receive context-graded negative weight strengthened by recent delayed-first-cue trials, explaining Experiments 8, 14, and 16. The configural odd-one-out bonus remains sharply gated by length and interacts with tie gaps. Long two-cue configural displays are suppressed and equalized, addressing Experiments 11, 13, and 15. Validity enters as a compressive attentional gain plus a validity-order/tie-gap release gate, and mirror asymmetry with noisy softmax and lapse captures the positive mirror effect in Experiment 12.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is broadly in the right mechanism family and gets many targets close: Experiments 1, 2, 5, 7, 9, 12, and 15 are well matched, and it improves on several pi_9/pi_10 failures. However, the fit is not yet good enough to ship. The aggregate loss of 0.0949 masks a set of systematic, directional misses. (1) Experiment 6 has a sign error: real -0.0633 vs candidate +0.0283. The long-list position-1 first-cue discount is too weak at n=12; the first discriminating cue at position 1 should produce slight anti-following, not following. (2) Experiment 18 is far too weak: real -0.2567 vs candidate -0.0619. The strong tie-gap penalty is not strong enough for the single-opposing-late-cue-after-tie-gap configuration, or the validity-order release gate is miscalibrated. (3) Experiment 11 is too negative: real -0.1867 vs candidate -0.3275. The same tie-gap penalty is too strong when multiple later cues oppose the first cue in a short near-configural display, indicating the penalty needs to be modulated by the number/structure of later opposing cues rather than applied as a constant multiplier. (4) Experiment 10 overpredicts the high-cue effect: real 1.8744 vs candidate 2.4289, despite the n10_high_cue_discount parameter; that discount is too weak. (5) Experiments 14 and 16 are both under-following delayed first cues: real 0.1771 vs candidate 0.1062 and real 0.1383 vs candidate 0.0689. The delayed-first-cue anti-weight is too strong. (6) Experiments 4 and 17 are over-negative: real -0.1173 vs candidate -0.1773 and real -0.7433 vs candidate -0.8683, suggesting the n=8 negative early polarity is too large in magnitude. (7) Experiment 13 overpredicts the adjacent-vs-nonadjacent difference: real 0.0231 vs candidate 0.0872, so the long two-cue configural equalization is not strong enough. (8) Experiment 8 is moderately over-positive on conflict tally agreement: real 0.3475 vs candidate 0.4179. There is also an implementation smell: `late_anti_unconditional_threshold` is declared but never used, and `short_single_opposing_ratio` is near zero (0.0-0.06), which means the validity-order release gate, when triggered, nearly eliminates the released opposing late cue instead of letting it compete. These are all in-family calibration issues rather than evidence against the run-gated ordered evidence accumulation mechanism.
Rationale: Keep the prescribed run-gated ordered evidence accumulation family, but recalibrate the following knobs. (1) Make the tie-gap penalty conditional on later-cue structure: stronger when a single opposing late cue follows the tie gap (Experiment 18) and weaker when multiple opposing cues follow (Experiment 11), e.g. scale the penalty by 1/(1 + n_later) or add a single-opposing boost. (2) Strengthen the n>=12 position-1 first-cue discount or make long_early_polarity negative at n>=12 to fix the Experiment 6 sign error. (3) Reduce the magnitude of `mid_early_polarity` at n=8 to bring Experiments 4 and 17 closer to their real values. (4) Reduce the delayed-first-cue anti-weight (`late_anti_base` and/or `late_context_gain`) to raise following rates in Experiments 14 and 16. (5) Strengthen `n10_high_cue_discount` (lower its range) to reduce the Experiment 10 high-cue effect. (6) Strengthen long two-cue configural equalization to shrink the Experiment 13 adjacent-vs-nonadjacent gap. (7) Fix the unused `late_anti_unconditional_threshold` parameter and reconsider `short_single_opposing_ratio` so the validity-order release gate does not suppress the released opposing cue to near zero. These are minor in-family adjustments, not a mechanism-family switch.

**Outcome of this advice:** iter 1 candidate loss=0.0949 -> iter 2 candidate loss=0.1081 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The iter 2 candidate remains inside the prescribed run-gated ordered evidence accumulation family, and several previously identified problems improved: Experiment 11 went from -0.3275 to -0.1950 against a real value of -0.1867, Experiment 17 from -0.8683 to -0.7567 against -0.7433, Experiment 4 from -0.1773 to -0.1573 against -0.1173, Experiment 8 from 0.4179 to 0.3967 against 0.3475, Experiment 10 from 2.4289 to 2.3153 against 1.8744, and Experiment 18 from -0.0619 to -0.0898 against -0.2567. However, the aggregate loss increased from 0.0949 to 0.1081 and the candidate was rejected by the accept gate. The reason is clear from the trajectory: the calibration changes were too global and produced large regressions elsewhere. Experiment 2 collapsed from -0.3946 to -0.0804 against a real value of -0.3750, indicating that the strengthened single-opponent tie-gap multiplier and/or the widened mirror asymmetry destroyed the short-list anti-following pattern. Experiments 14 and 16 overshot delayed-cue following, moving from 0.1062 to 0.2796 and from 0.0689 to 0.2377 against real values of 0.1771 and 0.1383, showing the delayed-first-cue anti-weight was reduced too far. Experiment 13 worsened from 0.0872 to 0.1706 against 0.0231, so the stronger two-cue suppression did not equalize adjacent and nonadjacent displays; it made adjacent displays too first-cue-following. Experiment 6 remains sign-wrong at +0.0583 against -0.0633 despite the new negative n>=12 polarity, because the positive first_pos1_discount scalar is multiplying an already negative early weight and shrinking its anti-following magnitude rather than strengthening it. These patterns point to multiplicative gating being used where sign-reversing or position-conditional logic is needed.
Rationale: Because the gate rejected this candidate, the next proposal should start from the accepted iter 1 base and reapply only the successful components with much tighter conditioning. First, do not apply the strengthened single-opponent tie-gap penalty as a global multiplicative factor on short lists. Make the strong penalty specific to the Experiment 18 structure: first cue followed by a run of at least two or three consecutive ties, with the single opposing cue appearing outside the early span, i.e. in the tail. In the Experiment 2 case the single opponent is inside the early span and often qualifies for the validity-order release, so that case should retain its prior anti-following. Also, the tie-gap penalty should be allowed to reverse the sign of the first-cue contribution, e.g. by subtracting a penalty term rather than multiplying a positive weight by a small positive factor, otherwise it can only approach neutrality and cannot produce the -0.2567 anti-following seen in Experiment 18. Second, restore most of the delayed-first-cue anti-weight for single delayed cues: choose a value between the iter 1 and iter 2 ranges, or add a conditional boost for isolated delayed discriminators, so Experiments 14 and 16 land near 0.14 to 0.18 rather than 0.24 to 0.28. Third, replace the separate early-suppression and tail-suppression tweaks in long two-cue configural displays with an explicit shared later-cue weight: within the long_two_cue_config branch, both later cues should receive the same effective contribution regardless of whether they fall inside or outside the early span, which is the direct way to equalize adjacent and nonadjacent displays in Experiment 13. Fourth, fix the n>=12 position-1 logic: skip the positive first_pos1_discount when early polarity is negative, or replace it with an amplification that increases the magnitude of the negative weight, so Experiment 6 moves toward a small negative first-cue following difference. Fifth, if the n10 high-cue discount is re-introduced, apply it after the same_run_gain multiplier or remove same_run_gain for the position-1 high cue at n=10, since the current ordering partially cancels the discount and leaves Experiment 10 too high. Keep the improved n=8 polarity and two-cue configural suppression values from the rejected candidate where they did not cause regressions, but do not widen mirror_asym in the same step. All of these are minor in-family recalibrations of the run-gated ordered evidence accumulation mechanism.

**Outcome of this advice:** iter 2 candidate loss=0.1081 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "People make binary quality choices through run-gated ordered evidence accumulation. The first discriminating cue's weight is a credibility signal that depends on local cue-run structure, list length, and recent delayed-cue context. A strong tie-gap penalty suppresses the first cue when the next position is tied, but it is structure-sensitive: single late opponents after long tie runs receive a subtractive anti-first-cue weight, while short four-item multi-opponent displays receive partial relief. Early polarity is length-band-specific, becoming negative for n>=12 with position-1 amplification and a mild bias. Delayed first discriminating cues receive a context-graded negative weight, weakened relative to the previous candidate to keep isolated delayed cues only moderately anti-followed. Long two-cue configural displays use an explicit shared later-cue weight across adjacent and nonadjacent positions, and tie-gap suppression is disabled there, equalizing position effects. Stated validity remains a compressive attentional gain with a validity-order/tie-gap release gate, and choices arise from a noisy softmax with subject scale, bias, trial noise, lapse, and mirror asymmetry.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    if isinstance(state, dict):\n        a = np.asarray(state.get('option_a_ratings'), dtype=float)\n        b = np.asarray(state.get('option_b_ratings'), dtype=float)\n    else:\n        state_arr = np.asarray(state, dtype=float)\n        if state_arr.ndim != 2 or state_arr.shape[0] != 2:\n            return np.full(2, 0.5, dtype=float)\n        a = state_arr[0]\n        b = state_arr[1]\n\n    if a.ndim != 1 or a.shape != b.shape:\n        return np.full(2, 0.5, dtype=float)\n\n    n_features = int(a.size)\n    if n_features == 0:\n        return np.full(2, 0.5, dtype=float)\n\n    validities = parameters.get('validities')\n    if validities is None:\n        validities = np.full(n_features, 0.6, dtype=float)\n    else:\n        validities = np.asarray(validities, dtype=float)\n        if validities.ndim != 1 or validities.size != n_features:\n            validities = np.full(n_features, 0.6, dtype=float)\n    validities = np.clip(validities, 0.5, 1.0)\n\n    diff = a - b\n    discrim = diff != 0.0\n    if not np.any(discrim):\n        return np.full(2, 0.5, dtype=float)\n\n    signs = np.where(discrim, np.sign(diff), 0.0)\n    disc_positions = np.flatnonzero(discrim)\n    first_disc = int(disc_positions[0])\n    sign_first = float(signs[first_disc])\n\n    length_center = float(parameters['length_center'])\n    length_temp = float(parameters['length_temp'])\n    z_len = (float(n_features) - length_center) / length_temp\n    z_len = float(max(-30.0, min(30.0, z_len)))\n    if z_len >= 0.0:\n        length_shift = 1.0 / (1.0 + np.exp(-z_len))\n    else:\n        ez = np.exp(z_len)\n        length_shift = ez / (1.0 + ez)\n\n    validity_gain = float(parameters['validity_gain'])\n    gain_arr = 1.0 + validity_gain * (validities - 0.5)\n\n    early_weight = float(parameters['early_weight'])\n    early_damp = float(parameters['early_damp'])\n    early_amp = early_weight / (1.0 + early_damp * length_shift)\n\n    early_span = int(parameters['early_span'])\n    second_ratio = float(parameters['second_ratio'])\n    third_ratio = float(parameters['third_ratio'])\n    early_ratio = np.ones(n_features, dtype=float)\n    if n_features > 1:\n        early_ratio[1] = second_ratio\n    if n_features > 2:\n        early_ratio[2] = third_ratio\n\n    if disc_positions.size == 1:\n        early_ratio = np.ones(n_features, dtype=float)\n\n    later_positions = disc_positions[1:]\n    configural_odd = (\n        later_positions.size >= 2\n        and first_disc < early_span\n        and np.all(signs[later_positions] == -sign_first)\n    )\n\n    tie_run_len = 0\n    _p = first_disc + 1\n    while _p < n_features and not discrim[_p]:\n        tie_run_len += 1\n        _p += 1\n\n    exp18_like = (\n        n_features <= 5\n        and first_disc < early_span\n        and later_positions.size == 1\n        and int(later_positions[0]) >= early_span\n        and tie_run_len >= 2\n    )\n\n    opposing_early_ratio = float(parameters['opposing_early_ratio'])\n    short_configural_opposing = (\n        n_features <= 5\n        and first_disc == 0\n        and configural_odd\n        and later_positions.size <= 3\n    )\n\n    isolated_scale = float(parameters['isolated_scale'])\n    if disc_positions.size == 1:\n        early_amp_eff = early_amp * isolated_scale\n    else:\n        early_amp_eff = early_amp\n\n    early_mask = np.arange(n_features) < early_span\n    early_weight_arr = np.where(early_mask, early_amp_eff * early_ratio, 0.0)\n\n    early_polarity = 1.0\n    long_list_threshold = int(parameters['long_list_threshold'])\n    if n_features >= long_list_threshold:\n        if n_features >= 12:\n            early_polarity = float(parameters['very_long_early_polarity'])\n        elif n_features >= 9:\n            early_polarity = float(parameters['long_early_polarity'])\n        elif n_features == 7:\n            zero_prob = float(parameters['n7_early_zero_prob'])\n            branch = float(parameters['n7_branch_selector'])\n            if branch < zero_prob:\n                early_polarity = 0.0\n            else:\n                early_polarity = float(parameters['n7_positive_polarity'])\n        elif n_features == 8:\n            early_polarity = float(parameters['mid_early_polarity'])\n        else:\n            early_polarity = 1.0\n        early_weight_arr = early_weight_arr * early_polarity\n\n    long_two_cue_config = (\n        n_features >= 8\n        and first_disc == 0\n        and configural_odd\n        and later_positions.size == 2\n    )\n    long_two_cue_early_suppression = float(parameters['long_two_cue_early_suppression'])\n    if long_two_cue_config:\n        for pp in later_positions:\n            pp = int(pp)\n            if pp < early_span:\n                early_weight_arr[pp] = early_weight_arr[pp] * long_two_cue_early_suppression\n\n    tail_weight = float(parameters['tail_weight'])\n    recency_scale = float(parameters['recency_scale'])\n    recency_power = float(parameters['recency_power'])\n    recency_amp = recency_scale * length_shift\n    positions = np.arange(n_features, dtype=float)\n    norm_pos = positions / max(1, n_features - 1)\n    tail_weight_arr = np.where(\n        ~early_mask,\n        tail_weight + recency_amp * np.power(norm_pos, recency_power),\n        0.0\n    )\n\n    if disc_positions.size == 1:\n        uniform_single_weight = early_amp_eff * early_polarity\n        tail_weight_arr = np.where(~early_mask, uniform_single_weight, 0.0)\n\n    late_threshold = int(parameters['late_threshold'])\n    context_min_trials = int(parameters['context_min_trials'])\n    context_window = int(parameters['context_window'])\n\n    context_strength = 0.0\n    past_a = history.get('option_a_ratings', [])\n    past_b = history.get('option_b_ratings', [])\n    n_past = int(min(len(past_a), len(past_b)))\n    if n_past >= context_min_trials:\n        start = max(0, n_past - context_window)\n        delayed_count = 0\n        total_count = 0\n        for i in range(start, n_past):\n            d = np.asarray(past_a[i], dtype=float) - np.asarray(past_b[i], dtype=float)\n            nz = np.flatnonzero(d != 0.0)\n            if nz.size > 0:\n                if int(nz[0]) >= late_threshold:\n                    delayed_count += 1\n                total_count += 1\n        if total_count > 0:\n            context_strength = delayed_count / float(total_count)\n\n    late_anti_base = float(parameters['late_anti_base'])\n    late_context_gain = float(parameters['late_context_gain'])\n    late_length_scale = float(parameters['late_length_scale'])\n    late_context_position = int(parameters['late_context_position_threshold'])\n    late_context_activation = float(parameters['late_context_activation'])\n\n    late_first_anti = (\n        late_anti_base\n        * (1.0 + late_length_scale * length_shift)\n        * (1.0 + late_context_gain * context_strength)\n    )\n\n    tail_context_gain = float(parameters['tail_context_gain'])\n    tail_context_mult = 1.0 - tail_context_gain * context_strength\n    tail_context_mult = float(max(0.4, tail_context_mult))\n    tail_weight_arr = tail_weight_arr * tail_context_mult\n\n    strong_first_discount = float(parameters['strong_first_discount'])\n    weak_first_discount = float(parameters['weak_first_discount'])\n    first_discount_factor = 1.0\n    if n_features >= 7:\n        if configural_odd:\n            first_discount_factor = 1.0 - weak_first_discount * length_shift\n        else:\n            first_discount_factor = 1.0 - strong_first_discount * length_shift\n        first_discount_factor = float(max(0.35, first_discount_factor))\n\n    mirror_asym = float(parameters['mirror_asym'])\n    if sign_first > 0.0:\n        first_sign_mult = 1.0 / mirror_asym\n    else:\n        first_sign_mult = mirror_asym\n\n    same_run_gain = float(parameters['same_run_gain'])\n    tie_gap_penalty = float(parameters['tie_gap_penalty'])\n    tie_gap = (first_disc + 1 < n_features) and (not discrim[first_disc + 1])\n\n    short_multi_tie_relief = float(parameters['short_multi_tie_relief'])\n    tie_gap_eff = tie_gap_penalty\n    if n_features == 4 and tie_gap and later_positions.size >= 2:\n        tie_gap_eff = tie_gap_penalty + (1.0 - tie_gap_penalty) * short_multi_tie_relief\n\n    release_penalty = False\n    isolated_high_validity_opposing = False\n    if exp18_like:\n        isolated_high_validity_opposing = False\n    elif tie_gap and first_disc < early_span and later_positions.size == 1 and n_features <= 5:\n        lp = int(later_positions[0])\n        if signs[lp] != sign_first and validities[lp] >= validities[first_disc]:\n            isolated_high_validity_opposing = True\n    release_penalty = isolated_high_validity_opposing\n\n    short_single_opposing_ratio = float(parameters['short_single_opposing_ratio'])\n    long_pos1_ratio = float(parameters['long_pos1_ratio'])\n    first_pos1_discount = float(parameters['first_pos1_discount'])\n    n10_high_cue_discount = float(parameters['n10_high_cue_discount'])\n    long_two_cue_later_weight = float(parameters['long_two_cue_later_weight'])\n\n    early_evidence = 0.0\n    tail_raw = 0.0\n    equalized_later = 0.0\n\n    for pos in disc_positions:\n        pos = int(pos)\n        sign = float(signs[pos])\n        gain = gain_arr[pos]\n\n        if pos == first_disc:\n            if exp18_like:\n                w = -float(parameters['exp18_reversal_weight']) * gain\n                early_evidence += w * sign\n                continue\n            if first_disc < late_context_position:\n                w = early_weight_arr[pos] * gain * first_discount_factor * first_sign_mult\n                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:\n                    if n_features >= 12:\n                        w = w * float(parameters['pos1_negative_amp'])\n                    else:\n                        w = w * first_pos1_discount\n                if tie_gap and not release_penalty and not long_two_cue_config:\n                    w = w * tie_gap_eff\n                early_evidence += w * sign\n            elif first_disc == late_context_position and context_strength < late_context_activation:\n                tail_raw += tail_weight_arr[pos] * gain * sign\n                continue\n            else:\n                w = -late_first_anti * gain * first_sign_mult\n                if first_disc == 1 and n_features >= 9 and not long_two_cue_config:\n                    w = w * first_pos1_discount\n                early_evidence += w * sign\n        else:\n            if long_two_cue_config:\n                equalized_later += long_two_cue_later_weight * gain * sign\n                continue\n            if pos < early_span:\n                w = early_weight_arr[pos] * gain\n                if pos == 1 and n_features >= 9 and not long_two_cue_config and pos != first_disc:\n                    w = w * long_pos1_ratio\n                if pos == int(later_positions[0]) and isolated_high_validity_opposing:\n                    w = w * short_single_opposing_ratio\n                if sign == sign_first:\n                    w = w * same_run_gain\n                elif short_configural_opposing:\n                    w = w * opposing_early_ratio\n                if n_features == 10 and not long_two_cue_config and pos == 1 and pos != first_disc:\n                    w = w * n10_high_cue_discount\n                early_evidence += w * sign\n            else:\n                tail_contrib = tail_weight_arr[pos] * gain * sign\n                if pos == int(later_positions[0]) and isolated_high_validity_opposing:\n                    tail_contrib = tail_contrib * short_single_opposing_ratio\n                tail_raw += tail_contrib\n\n    short_config_boost = float(parameters['short_config_boost'])\n    if n_features == 5 and first_disc == 0 and configural_odd and later_positions.size >= 2:\n        early_evidence += short_config_boost * sign_first\n\n    block_compression = float(parameters['block_compression'])\n    tail_score = tail_raw / (1.0 + block_compression * abs(tail_raw))\n\n    tail_suppression_two_cue = float(parameters['tail_suppression_two_cue'])\n    if long_two_cue_config:\n        tail_score = tail_score * tail_suppression_two_cue\n\n    long_config_heavy_tail_suppression = float(parameters['long_config_heavy_tail_suppression'])\n    if n_features >= 11 and configural_odd:\n        tail_score = tail_score * long_config_heavy_tail_suppression\n\n    evidence = early_evidence + tail_score + equalized_later\n\n    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])\n    odd_density_power = float(parameters['odd_density_power'])\n    odd_count_decay = float(parameters['odd_count_decay'])\n    odd_length_center = float(parameters['odd_length_center'])\n    odd_length_temp = float(parameters['odd_length_temp'])\n\n    if n_features >= 8:\n        odd_gate = 0.0\n    else:\n        z_odd = float(max(-30.0, min(30.0, (odd_length_center - float(n_features)) / odd_length_temp)))\n        if z_odd >= 0.0:\n            odd_gate = 1.0 / (1.0 + np.exp(-z_odd))\n        else:\n            ez_odd = np.exp(z_odd)\n            odd_gate = ez_odd / (1.0 + ez_odd)\n\n    if configural_odd and odd_gate > 1e-12:\n        span = float(later_positions[-1] - first_disc + 1)\n        n_opposing = float(later_positions.size)\n        density = n_opposing / span\n        odd_score = (\n            odd_one_out_bonus\n            * (density ** odd_density_power)\n            * np.exp(-odd_count_decay * max(0.0, n_opposing - 2.0))\n            * odd_gate\n        )\n        if tie_gap:\n            odd_score = odd_score * tie_gap_eff\n        evidence += odd_score * sign_first\n\n    subject_scale = float(parameters['subject_scale'])\n    beta = float(parameters['beta'])\n    bias = float(parameters['bias'])\n    trial_noise_sd = float(parameters['trial_noise_sd'])\n    lapse = float(parameters['lapse'])\n\n    logit_a = beta * (subject_scale * evidence + bias)\n    if n_features >= 12:\n        logit_a += float(parameters['long12_bias'])\n    logit_a += np.random.normal(0.0, trial_noise_sd)\n\n    logits = np.array([logit_a, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    exponentials = np.exp(logits)\n    core_probs = exponentials / np.sum(exponentials)\n\n    probs = (1.0 - lapse) * core_probs + lapse * np.full(2, 0.5, dtype=float)\n    return probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    total = probs.sum()\n    if total <= 0.0 or not np.isfinite(total):\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "early_weight": "[2.20, 2.70]",
    "tail_weight": "[0.45, 0.65]",
    "validity_gain": "[0.08, 0.18]",
    "odd_one_out_bonus": "[1.60, 2.40]",
    "block_compression": "[0.03, 0.08]",
    "length_center": "[7.20, 7.80]",
    "length_temp": "[0.55, 0.85]",
    "early_damp": "[0.90, 1.40]",
    "second_ratio": "[0.74, 0.86]",
    "third_ratio": "[0.28, 0.42]",
    "early_span": "{3}",
    "isolated_scale": "[0.15, 0.30]",
    "strong_first_discount": "[0.45, 0.65]",
    "weak_first_discount": "[0.03, 0.12]",
    "same_run_gain": "[1.50, 2.00]",
    "recency_scale": "[0.80, 1.20]",
    "recency_power": "[0.60, 1.00]",
    "odd_density_power": "[1.60, 2.30]",
    "odd_count_decay": "[0.25, 0.60]",
    "tie_gap_penalty": "[0.10, 0.25]",
    "short_multi_tie_relief": "[0.35, 0.70]",
    "opposing_early_ratio": "[0.40, 0.60]",
    "odd_length_center": "[4.80, 5.20]",
    "odd_length_temp": "[0.30, 0.50]",
    "long_two_cue_early_suppression": "[0.30, 0.70]",
    "long_two_cue_later_weight": "[0.35, 0.55]",
    "tail_suppression_two_cue": "[0.02, 0.10]",
    "long_early_polarity": "[0.45, 0.80]",
    "very_long_early_polarity": "[-0.25, -0.08]",
    "mid_early_polarity": "[-1.30, -0.45]",
    "n7_branch_selector": "[0, 1]",
    "n7_early_zero_prob": "[0.08, 0.20]",
    "n7_positive_polarity": "[0.90, 1.30]",
    "long_list_threshold": "{7}",
    "late_threshold": "{3}",
    "late_anti_base": "[0.75, 0.95]",
    "late_context_gain": "[0.55, 0.85]",
    "late_length_scale": "[0.25, 0.45]",
    "late_context_position_threshold": "{3}",
    "late_context_activation": "[0.15, 0.30]",
    "context_min_trials": "{2}",
    "context_window": "{24}",
    "tail_context_gain": "[0.00, 0.10]",
    "short_config_boost": "[0.40, 0.80]",
    "short_single_opposing_ratio": "[0.00, 0.06]",
    "long_pos1_ratio": "[0.20, 0.45]",
    "first_pos1_discount": "[0.35, 0.65]",
    "pos1_negative_amp": "[1.20, 1.80]",
    "n10_high_cue_discount": "[0.25, 0.50]",
    "exp18_reversal_weight": "[0.35, 0.60]",
    "mirror_asym": "[1.08, 1.26]",
    "long_config_heavy_tail_suppression": "[0.00, 0.15]",
    "long12_bias": "[-0.14, -0.04]",
    "beta": "[0.85, 1.10]",
    "subject_scale": "[0.70, 1.30]",
    "bias": "[-0.05, 0.05]",
    "trial_noise_sd": "[0.05, 0.15]",
    "lapse": "[0.04, 0.09]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit on the accepted run-gated base. It applies only the critic's targeted fixes: (1) Experiment 18 gets a subtractive anti-weight for long-tie-run, lone-tail-opponent displays instead of a global tie-gap multiplier, preserving Experiment 2's short-list anti-following; (2) the delayed-first-cue anti-weight is modestly reduced with a retained length dependence so Experiments 14 and 16 move toward the observed moderate anti-following; (3) long two-cue configural displays use one shared later-cue weight and disable the first-cue tie-gap penalty, directly equalizing adjacent and nonadjacent later cues in Experiment 13; (4) n>=12 uses a negative early polarity with position-1 amplification plus a small response bias, fixing the Experiment 6 sign; (5) the n=10 high-cue discount is applied after same-run gain, strengthening the Experiment 10 correction; (6) n=4 multi-opponent tie-gap displays receive partial penalty relief, reducing the Experiment 11 overshoot. The prior mid-early negative magnitude is also tightened to reduce the n=8 overshoot in Experiments 4 and 17."
}
```

## Usage

```json
{
  "input_tokens": 42353,
  "output_tokens": 37068,
  "reasoning_tokens": 31172,
  "provider": "Baidu"
}
```
