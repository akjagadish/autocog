# generation_iter_00_attempt_00

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
- THEORY 2 = `pi_7`
- The recommendation below acts on THEORY 2 (= `pi_7`).

Propose a salience-weighted ordered evidence accumulation theory. All discriminating cues are considered, but display position is primary: short lists show strong primacy with a large first-displayed-cue weight, while longer lists flatten or shift toward recency. Stated validity enters only as a compressive attentional gain, never as a strong additive or lexicographic multiplier. A key new mechanism is an odd-one-out/block-contrast bonus: a cue that conflicts with an otherwise uniform later block receives extra salience, so first-cue dominance can be strong even when the first cue has lower stated validity (Experiment 2), and the contrast between a fully opposing block and a partially tied profile can produce the negative R_near - R_all observed in Experiment 1. Later opposing cues still accumulate as a count-sensitive but compressively weighted block, preserving the sensitivity seen in Experiments 10, 11, and 12. Suggested parameter ranges: early-display weight 1.6-3.5, tail weight 0.3-0.8, validity gain 0.1-0.5, odd-one-out/block contrast bonus 1.0-3.0, block compression 0.3-0.8, length transition around n=5-7, trial noise sd 0.05-0.25, lapse 0.02-0.15, and moderate between-subject scale heterogeneity. This differs from pi_7 because display order and odd-one-out salience dominate validity; it differs from pi_5 because it has no random capacity-limited sampling and explicitly includes configural block/contrast integration.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.542)

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


---

### `pi_7` (overall score: 0.452)

**Description**
People make binary choices by accumulating signed, count-sensitive evidence across every discriminating feature comparison, without normalizing by total attended evidence. Each cue contributes additive evidence equal to a positive validity-contrast weight times a length-damped serial-position gradient times a subject-specific cue multiplier. The serial-position gradient flattens with list length both by shrinking its primacy range and by lifting its tail, so long lists attend broadly rather than dropping most cues. A first-discriminator adjustment is positive on short lists and crosses smoothly to negative by mid-length lists, but its negative excursion is smoothly bounded by a subject-level cap so long displays show a modest anti-first-discriminator tendency rather than a strong reversal. Accumulated evidence passes through a fairly sharp noisy softmax with lapse and trial-level noise, so majority low-validity blocks can move choice steeply while balanced high-validity blocks cancel and per-pattern choice variance remains large.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validity_intercept: [0.65, 0.85]
- validity_slope: [1.10, 1.60]
- primacy_gain: [2.00, 2.40]
- primacy_tail: [0.50, 0.60]
- tail_length_gain: [0.03, 0.08]
- primacy_center: [1.30, 1.70]
- primacy_temp: [0.30, 0.50]
- length_flat: [0.40, 0.60]
- center_length_shift: [0.10, 0.25]
- first_disc_intercept: [0.42, 0.55]
- first_disc_length_slope: [0.45, 0.60]
- first_disc_neg_cap: [0.15, 0.22]
- subject_scale: [0.40, 1.10]
- evidence_scale: [1.05, 1.30]
- beta: [2.20, 2.80]
- bias: [-0.10, 0.10]
- lapse: [0.09, 0.14]
- trial_noise_sd: [0.05, 0.12]
- cue_noise: [(-0.70, 0.70)] * n_features
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4400 (var=0.1503) vs this=-0.5133 (var=0.1857)
- Experiment 2: real=-0.3750 (var=0.0321) vs this=-0.3442 (var=0.0247)
- Experiment 3: real=0.6246 (var=0.0093) vs this=0.5504 (var=0.0048)
- Experiment 4: real=-0.1173 (var=0.0257) vs this=0.0827 (var=0.0522)
- Experiment 5: real=-0.0222 (var=0.0067) vs this=-0.0075 (var=0.0053)
- Experiment 6: real=-0.0633 (var=0.1971) vs this=0.0942 (var=0.2179)
- Experiment 7: real=0.1099 (var=0.0037) vs this=0.1140 (var=0.0020)
- Experiment 8: real=0.3475 (var=0.0426) vs this=0.2300 (var=0.0296)
- Experiment 9: real=0.2025 (var=0.0134) vs this=0.1950 (var=0.0096)
- Experiment 10: real=1.8744 (var=0.7034) vs this=2.2173 (var=0.1964)
- Experiment 11: real=-0.1867 (var=0.1342) vs this=0.3383 (var=0.1102)
- Experiment 12: real=0.6992 (var=0.0351) vs this=-0.3475 (var=0.3307)


---

### `pi_6` (overall score: 0.430)

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
**Other theories' values on this metric (for reference):**
- pi_1: 0.6650 (var=0.0546)
- pi_2: 0.0383 (var=0.0359)
- pi_3: -0.6500 (var=0.0231)
- pi_4: -0.4933 (var=0.0313)
- pi_5: -0.5700 (var=0.1570)
- pi_6: -0.3200 (var=0.1409)
- pi_7: -0.5133 (var=0.1857)

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
**Other theories' values on this metric (for reference):**
- pi_2: -0.0029 (var=0.0086)
- pi_1: 0.6875 (var=0.0362)
- pi_3: -0.3296 (var=0.0048)
- pi_4: -0.3458 (var=0.0038)
- pi_5: -0.2871 (var=0.0275)
- pi_6: -0.3213 (var=0.0070)
- pi_7: -0.3442 (var=0.0247)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.8567 (var=0.0014)
- pi_2: 0.2706 (var=0.0057)
- pi_1: 0.6206 (var=0.0018)
- pi_4: 0.6208 (var=0.0020)
- pi_5: 0.6331 (var=0.0177)
- pi_6: 0.6160 (var=0.0094)
- pi_7: 0.5504 (var=0.0048)

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
**Other theories' values on this metric (for reference):**
- pi_2: -0.0147 (var=0.0280)
- pi_3: 0.7160 (var=0.0149)
- pi_1: 0.2253 (var=0.0167)
- pi_4: 0.0280 (var=0.0391)
- pi_5: 0.0253 (var=0.0490)
- pi_6: -0.1307 (var=0.0469)
- pi_7: 0.0827 (var=0.0522)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.1878 (var=0.0101)
- pi_4: -0.2019 (var=0.0042)
- pi_1: 0.0092 (var=0.0061)
- pi_2: 0.0058 (var=0.0086)
- pi_5: 0.0267 (var=0.0064)
- pi_6: -0.0006 (var=0.0043)
- pi_7: -0.0075 (var=0.0053)

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
**Other theories' values on this metric (for reference):**
- pi_4: 0.6033 (var=0.0124)
- pi_3: -0.8958 (var=0.0049)
- pi_1: -0.6992 (var=0.0453)
- pi_2: 0.4608 (var=0.0209)
- pi_5: -0.0617 (var=0.2674)
- pi_6: -0.0858 (var=0.1788)
- pi_7: 0.0942 (var=0.2179)

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
**Other theories' values on this metric (for reference):**
- pi_5: 0.1186 (var=0.0016)
- pi_4: 0.0406 (var=0.0002)
- pi_1: 0.1334 (var=0.0041)
- pi_2: 0.0187 (var=0.0001)
- pi_3: 0.0541 (var=0.0003)
- pi_6: 0.0923 (var=0.0010)
- pi_7: 0.1140 (var=0.0020)

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
**Other theories' values on this metric (for reference):**
- pi_4: 0.9329 (var=0.0017)
- pi_5: 0.4883 (var=0.0613)
- pi_1: 0.5050 (var=0.0022)
- pi_2: 0.8471 (var=0.0113)
- pi_3: 0.4933 (var=0.0007)
- pi_6: 0.2933 (var=0.0149)
- pi_7: 0.2300 (var=0.0296)

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
**Other theories' values on this metric (for reference):**
- pi_5: 0.0824 (var=0.0041)
- pi_6: 0.0260 (var=0.0004)
- pi_1: 0.0349 (var=0.0006)
- pi_2: 0.2730 (var=0.0071)
- pi_3: 0.0405 (var=0.0008)
- pi_4: 0.3544 (var=0.0008)
- pi_7: 0.1950 (var=0.0096)

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
**Other theories' values on this metric (for reference):**
- pi_6: 0.4099 (var=0.0378)
- pi_5: 0.7166 (var=0.1141)
- pi_1: 0.8464 (var=0.3893)
- pi_2: 2.0302 (var=0.7154)
- pi_3: 0.2507 (var=0.0279)
- pi_4: 2.8162 (var=0.0908)
- pi_7: 2.2173 (var=0.1964)

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
**Other theories' values on this metric (for reference):**
- pi_5: 0.0992 (var=0.0345)
- pi_7: 0.3383 (var=0.1102)
- pi_1: 0.0133 (var=0.0088)
- pi_2: 0.0150 (var=0.0106)
- pi_3: 0.0067 (var=0.0021)
- pi_4: -0.0050 (var=0.0059)
- pi_6: 0.0358 (var=0.0131)

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
**Other theories' values on this metric (for reference):**
- pi_7: -0.3475 (var=0.3307)
- pi_5: 0.1979 (var=0.2805)
- pi_1: -0.6408 (var=0.0276)
- pi_2: -0.6800 (var=0.0406)
- pi_3: 0.5900 (var=0.0052)
- pi_4: 0.5504 (var=0.0067)
- pi_6: 0.5308 (var=0.0471)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "People make binary quality choices through salience-weighted ordered evidence accumulation. Display order is the primary salience signal: the first discriminating position receives a large early-display weight on short lists, while longer lists damp this early weight and add a smooth recency gradient across later positions. Stated cue validity enters only as a compressive attention gain, never as a lexicographic or strongly additive multiplier. All remaining discriminating cues accumulate as a count-sensitive but compressively weighted tail block. A configural odd-one-out/block-contrast bonus adds salience when a cue is isolated against a compact, uniformly opposing later block; however, a tied cue immediately after the first discriminator breaks the block contrast and reduces this salience. Accumulation is graded through a moderately noisy softmax with lapse and between-subject scale heterogeneity.",
  "predict": "def predict(parameters, state, history):\n    import numpy as np\n\n    # Parse the (2, n_features) stimulus.\n    if isinstance(state, dict):\n        a = np.asarray(state.get('option_a_ratings'), dtype=float)\n        b = np.asarray(state.get('option_b_ratings'), dtype=float)\n    else:\n        state_arr = np.asarray(state, dtype=float)\n        if state_arr.ndim != 2 or state_arr.shape[0] != 2:\n            return np.full(2, 0.5, dtype=float)\n        a = state_arr[0]\n        b = state_arr[1]\n\n    if a.ndim != 1 or a.shape != b.shape:\n        return np.full(2, 0.5, dtype=float)\n\n    n_features = int(a.size)\n    if n_features == 0:\n        return np.full(2, 0.5, dtype=float)\n\n    validities = parameters.get('validities')\n    if validities is None:\n        validities = np.full(n_features, 0.75, dtype=float)\n    else:\n        validities = np.asarray(validities, dtype=float)\n        if validities.ndim != 1 or validities.size != n_features:\n            validities = np.full(n_features, 0.75, dtype=float)\n    validities = np.clip(validities, 0.5, 1.0)\n\n    diff = a - b\n    discrim = diff != 0.0\n    if not np.any(discrim):\n        return np.full(2, 0.5, dtype=float)\n\n    signs = np.where(discrim, np.sign(diff), 0.0)\n    disc_positions = np.flatnonzero(discrim)\n    first_disc = int(disc_positions[0])\n    sign_first = float(signs[first_disc])\n\n    # Length-dependent transition from short-list primacy to longer-list\n    # primacy damping plus recency weighting.\n    length_center = float(parameters['length_center'])\n    length_temp = float(parameters['length_temp'])\n    z_len = (float(n_features) - length_center) / length_temp\n    z_len = float(max(-30.0, min(30.0, z_len)))\n    if z_len >= 0.0:\n        length_shift = 1.0 / (1.0 + np.exp(-z_len))\n    else:\n        ez = np.exp(z_len)\n        length_shift = ez / (1.0 + ez)\n\n    early_weight = float(parameters['early_weight'])\n    early_damp = float(parameters['early_damp'])\n    first_decay = float(parameters['first_decay'])\n\n    early_amp = early_weight / (1.0 + early_damp * length_shift)\n    first_weight = early_amp * np.exp(-float(first_disc) / first_decay)\n\n    validity_gain = float(parameters['validity_gain'])\n    validity_gain_arr = 1.0 + validity_gain * (validities - 0.5)\n\n    # Later-position tail weights, with recency growth on longer lists.\n    tail_weight = float(parameters['tail_weight'])\n    recency_scale = float(parameters['recency_scale'])\n    recency_power = float(parameters['recency_power'])\n    recency_amp = recency_scale * length_shift\n    positions = np.arange(n_features, dtype=float)\n    norm_pos = positions / max(1, n_features - 1)\n    tail_pos = tail_weight + recency_amp * np.power(norm_pos, recency_power)\n\n    first_score = first_weight * validity_gain_arr[first_disc] * sign_first\n\n    # Odd-one-out / block-contrast bonus.\n    later_positions = disc_positions[1:]\n    odd_one_out_bonus = float(parameters['odd_one_out_bonus'])\n    odd_density_power = float(parameters['odd_density_power'])\n    odd_count_decay = float(parameters['odd_count_decay'])\n    tie_gap_penalty = float(parameters['tie_gap_penalty'])\n\n    if later_positions.size >= 2:\n        later_signs = signs[later_positions]\n        if np.all(later_signs == (-sign_first)):\n            span = float(later_positions[-1] - first_disc + 1)\n            n_opposing = float(later_positions.size)\n            density = n_opposing / span\n            odd_score = (\n                odd_one_out_bonus\n                * (density ** odd_density_power)\n                * np.exp(-odd_count_decay * max(0.0, n_opposing - 2.0))\n            )\n\n            # A tied cue immediately after the first discriminator breaks the\n            # compact opposing-block configuration.\n            if (first_disc + 1 < n_features) and (not discrim[first_disc + 1]):\n                first_score *= tie_gap_penalty\n                odd_score *= tie_gap_penalty\n\n            first_score += odd_score * sign_first\n\n    # Compressively accumulate the remaining discriminating cues.\n    if later_positions.size > 0:\n        raw_tail = np.sum(\n            tail_pos[later_positions]\n            * validity_gain_arr[later_positions]\n            * signs[later_positions]\n        )\n        block_compression = float(parameters['block_compression'])\n        tail_score = raw_tail / (1.0 + block_compression * np.abs(raw_tail))\n    else:\n        tail_score = 0.0\n\n    evidence = first_score + tail_score\n\n    subject_scale = float(parameters['subject_scale'])\n    beta = float(parameters['beta'])\n    bias = float(parameters['bias'])\n    trial_noise_sd = float(parameters['trial_noise_sd'])\n    lapse = float(parameters['lapse'])\n\n    logit_a = beta * (subject_scale * evidence + bias)\n    logit_a += np.random.normal(0.0, trial_noise_sd)\n\n    logits = np.array([logit_a, 0.0], dtype=float)\n    logits = logits - np.max(logits)\n    exponentials = np.exp(logits)\n    core_probs = exponentials / np.sum(exponentials)\n\n    probs = (1.0 - lapse) * core_probs + lapse * np.full(2, 0.5, dtype=float)\n    return probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    probs = np.asarray(probs, dtype=float)\n    total = probs.sum()\n    if total <= 0.0 or not np.isfinite(total):\n        return 0\n    probs = probs / total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "early_weight": "[2.40, 3.50]",
    "tail_weight": "[0.45, 0.80]",
    "validity_gain": "[0.15, 0.50]",
    "odd_one_out_bonus": "[2.00, 3.00]",
    "block_compression": "[0.30, 0.70]",
    "length_center": "[6.00, 7.00]",
    "length_temp": "[0.60, 1.20]",
    "first_decay": "[1.50, 2.30]",
    "early_damp": "[1.60, 2.80]",
    "recency_scale": "[1.20, 2.00]",
    "recency_power": "[0.80, 1.40]",
    "odd_density_power": "[1.80, 2.50]",
    "odd_count_decay": "[0.25, 0.70]",
    "tie_gap_penalty": "[0.30, 0.65]",
    "beta": "[0.75, 1.25]",
    "subject_scale": "[0.70, 1.30]",
    "bias": "[-0.05, 0.05]",
    "trial_noise_sd": "[0.05, 0.20]",
    "lapse": "[0.02, 0.12]",
    "validities": "validities"
  },
  "rationale": "This theory directly implements the arbiter's salience-weighted ordered-accumulation family. Display position, not validity, is the dominant signal: a strong first-discriminator weight handles the negative high-validity-cue effects in Experiments 1 and 2 and the first-cue dominance in Experiment 12, while length-dependent damping and later recency weighting flatten or reverse early-cue effects on long lists such as Experiments 4 and 6. Validity is only a small compressive gain, so it cannot overpower position or become lexicographic. The odd-one-out bonus explains why a cue isolated against a compact uniform opposing block is especially salient: it strengthens first-cue commitment in Experiment 12, makes full opposition in Experiment 11 more first-cue-consistent than a partial tied profile, and can generate the observed negative R_near - R_all. The tie-gap penalty captures partially tied profiles as configurally fragmented. Later cues are accumulated across all discriminating positions with count-sensitive but compressed weighting, preserving the block sensitivity required by Experiments 8, 9, 10, and 11 without making low-validity blocks dominate choices."
}
```

## Usage

```json
{
  "input_tokens": 21079,
  "output_tokens": 42878,
  "reasoning_tokens": 40450,
  "provider": "Baidu"
}
```
