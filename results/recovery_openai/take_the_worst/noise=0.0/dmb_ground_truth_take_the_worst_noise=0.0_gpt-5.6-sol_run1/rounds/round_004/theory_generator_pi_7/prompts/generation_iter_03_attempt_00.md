# generation_iter_03_attempt_00

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with a Resource-Rational Selective Retrieval theory. The new theory should posit one capacity-limited retrieval-and-accumulation process rather than pi_6's special coherent-coalition reopening or pi_5's three explicitly gated formats. Cue retrieval priority should combine communicated validity, display recency, and unresolved conflict. Capacity and stopping should adapt continuously to diagnostic-cue load and to the distinctiveness of the validity structure: diffuse structures encourage approximate pooling, whereas a clearly dominant validity tier encourages compression onto that tier under capacity pressure. These should be emergent regimes of a common process, not hard strategy switches.

Crucially, late dissent should receive strong attentional priority, but adjacency should not intrinsically amplify dissent. A dispersed set containing a very late opposing cue can therefore be more disruptive than an equally centered adjacent block, accounting for Experiments 1 and 10. Absolute validity and recency sensitivity should vary continuously and partly independently across subjects, allowing the balanced directional result and high variance in Experiment 2. The theory should also include stable subject variation in whether tied cues are pooled or represented by an early retrieved member, preserving the heterogeneity seen in Experiment 7. It must retain pi_5's successful emergent signatures: majority pooling in Experiments 3-5, validity-led behavior under suitable overload structures in Experiment 6, and strong positional retrieval effects in Experiments 8-9. The resulting competitor should specifically test whether apparent load-indexed formats are discrete arbitration states, as pi_5 claims, or continuous consequences of capacity, retrieval competition, validity distinctiveness, and adaptive stopping.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.749)

**Description**
Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # A partially tied leading tier selectively lowers confidence in diffuse
    # pooling at loads 3-4, without changing strictly ordered low-load displays.
    low_gain_multiplier = 1.0
    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:
        low_gain_multiplier = float(parameters["partial_tie_low_attenuation"])

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools graded evidence over the complete active tier.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        high_evidence = float(np.mean(signs[top_tier]))

    # At overload, coherent retrieval of an entire partially tied leading tier
    # gives whole-tier poolers a bounded confidence benefit. The underlying
    # evidence remains its graded mean rather than being collapsed to its sign.
    high_gain = float(parameters["high_evidence_gain"])
    if tie_rule == 1 and partial_top_tie > 0.0 and load >= 6.0:
        high_gain *= float(parameters["partial_tier_pooling_gain"])

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence,
            low_gain_multiplier * float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(high_evidence, high_gain)
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- low_center: [4.3, 4.8]
- high_center: [7.0, 7.35]
- transition_width: [0.9, 1.3]
- high_transition_width: [0.3, 0.5]
- partial_tie_acceleration: [0.9, 1.4]
- all_equal_delay: [0.9, 1.3]
- recency_compression_coupling: [0.25, 0.45]
- recency_strength: [4.8, 6.8]
- validity_strength: [0.2, 0.65]
- tie_rule: {0, 1}
- partial_tie_low_attenuation: [0.35, 0.6]
- low_evidence_gain: [0.6, 0.85]
- high_evidence_gain: [0.4, 0.65]
- partial_tier_pooling_gain: [1.45, 1.8]
- beta: [4.8, 6.2]
- epsilon: [0.0, 0.06]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=0.0367 (var=0.0093)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.6467 (var=0.0132)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.5350 (var=0.0071)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8884 (var=0.0019)
- Experiment 5: real=0.5000 (var=0.2000) vs this=0.5800 (var=0.2436)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=-0.1079 (var=0.0009)
- Experiment 7: real=0.1175 (var=0.0081) vs this=0.0850 (var=0.0021)
- Experiment 8: real=0.7750 (var=0.0279) vs this=0.3750 (var=0.0124)
- Experiment 9: real=1.4567 (var=0.1585) vs this=0.1083 (var=0.0232)
- Experiment 10: real=0.0000 (var=0.7584) vs this=1.0000 (var=0.8656)


---

### `pi_6` (overall score: 0.437)

**Description**
Conflict-Reopening Attentional Race theory: diagnostic cues become available during a noisy, approximately display-ordered attentional race. Accessibility reflects communicated validity and recency, while sampled evidence accumulates toward a subject-specific stopping threshold. Reaching the threshold normally terminates comparison, especially when encountered cues agree. Peripheral detection of unresolved dissent can reopen comparison and transiently prioritize opposing evidence. Reopening after late coherent dissent causes bounded leakage of the provisional accumulator toward zero, undoing premature commitment without repeatedly multiplying opposing evidence. Spatially adjacent same-sign samples establish a coherent coalition, but each run receives only a finite one-shot reinforcement. Equal-validity treatment varies continuously across people: after a tier is opened, pooling-oriented subjects preferentially inspect its unresolved members, whereas first-member-oriented subjects can move on or stop early. Load has no strategy gate and matters only through sampling opportunities, conflict, coalition formation, stopping, and probabilistic reopening.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Conflict-Reopening Attentional Race expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    validity_access_temperature = float(parameters["validity_access_temperature"])
    validity_evidence_temperature = float(parameters["validity_evidence_temperature"])
    validity_modulation = float(parameters["validity_modulation"])
    recency_bias = float(parameters["recency_bias"])
    stopping_threshold = float(parameters["stopping_threshold"])
    conflict_trigger = float(parameters["conflict_trigger"])
    coalition_reinforcement = float(parameters["coalition_reinforcement"])
    tier_pooling = float(parameters["tier_pooling"])
    sampling_noise = float(parameters["sampling_noise"])
    decision_gain = float(parameters["decision_gain"])

    # Communicated validity magnitudes are retained across experiments. A
    # bounded log-odds transform avoids experiment-wise min-max rescaling while
    # allowing accessibility and evidence strength to have modestly different
    # validity temperatures.
    clipped_validities = np.clip(validities, 0.500001, 0.999999)
    validity_log_odds = np.log(clipped_validities / (1.0 - clipped_validities))
    v_access = np.tanh(validity_log_odds / validity_access_temperature)
    v_evidence = np.tanh(validity_log_odds / validity_evidence_temperature)

    if n_features > 1:
        position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        position = np.zeros(n_features, dtype=float)

    # Conflict lateness is relative to the diagnostic sequence rather than the
    # full display. Physical positions remain available for onset and coalition
    # proximity.
    active_lateness = np.zeros(n_features, dtype=float)
    if active.size > 1:
        for active_rank, cue in enumerate(active):
            active_lateness[int(cue)] = active_rank / float(active.size - 1)

    # A steep but continuous disposition controls both the evidential use of
    # later tied cues and the probability of inspecting unresolved tier members.
    effective_pooling = 1.0 / (
        1.0 + np.exp(-12.0 * (tier_pooling - 0.50))
    )

    # A fixed deterministic particle set approximates the stochastic race.
    # Different particles instantiate attentional fluctuations, while model
    # probabilities remain reproducible and do not add Monte Carlo instability.
    n_particles = 64
    particle_p_a = []

    def pseudo_uniform(particle, cue, stream):
        x = np.sin(
            (particle + 1.0) * 12.9898
            + (cue + 1.0) * 78.233
            + (stream + 1.0) * 37.719
        ) * 43758.5453123
        u = x - np.floor(x)
        return float(np.clip(u, 1e-9, 1.0 - 1e-9))

    for particle in range(n_particles):
        remaining = [int(j) for j in active]
        accumulator = 0.0
        last_sign = 0.0
        last_index = -1
        encountered_signs = []
        sampled_by_tier = {}
        alert_sign = 0.0
        reopened = False
        coalition_trace = 0.0
        coalition_impulse_paid = False
        step = 0

        while remaining:
            race_scores = []
            for cue in remaining:
                # Cues enter in approximate display order, but communicated
                # validity, recency, noise, and a conflict alert can change
                # which available cue wins attention next.
                temporal_onset = -4.0 * position[cue]
                accessibility = (
                    temporal_onset
                    + validity_sensitivity * v_access[cue]
                    + recency_bias * position[cue]
                )
                if alert_sign != 0.0 and signs[cue] == alert_sign:
                    accessibility += conflict_trigger

                # Once a validity tier has been opened, pooling-oriented
                # subjects preferentially inspect its unresolved members.
                # This is a graded accessibility effect, not hard completion.
                candidate_tier = int(np.sum(validities > validities[cue] + 1e-12))
                if sampled_by_tier.get(candidate_tier, 0) > 0:
                    accessibility += 2.0 * effective_pooling

                u = pseudo_uniform(particle, cue, step + 3)
                gumbel = -np.log(-np.log(u))
                race_scores.append(accessibility + sampling_noise * gumbel)

            winner = int(np.argmax(np.asarray(race_scores, dtype=float)))
            cue = remaining.pop(winner)
            cue_sign = float(signs[cue])

            # Evidence retains the accepted bounded validity representation.
            # A small capped multiplicative modulation increases communicated-
            # validity differentiation without allowing one cue to outweigh an
            # arbitrarily large opposing coalition.
            increment_size = (
                1.0
                + 0.42 * validity_sensitivity * v_evidence[cue]
                + 0.32 * recency_bias * position[cue]
            )
            capped_log_odds = min(float(validity_log_odds[cue]), 4.0)
            increment_size *= 1.0 + validity_modulation * capped_log_odds

            # Generic conflict receives little amplification. A dissenting cue
            # becomes consequential primarily when it arrives late in the
            # active diagnostic sequence.
            if accumulator * cue_sign < 0.0:
                late_conflict_scale = 0.10 + 1.35 * active_lateness[cue]
                increment_size *= 1.0 + conflict_trigger * late_conflict_scale

            # Nearby same-sign samples establish a bounded coalition trace, but
            # only the first local continuation in a coherent run receives an
            # evidential impulse. Extra run members therefore establish
            # coherence without repeatedly multiplying tally evidence.
            if last_sign == cue_sign and last_index >= 0:
                separation = abs(cue - last_index)
                proximity = np.exp(-float(separation - 1) / 1.5)
                coalition_trace = min(
                    1.0, 0.35 * coalition_trace + float(proximity)
                )
                if not coalition_impulse_paid:
                    increment_size *= 1.0 + coalition_reinforcement * proximity * (
                        0.65 + 0.35 * coalition_trace
                    )
                    if proximity >= 0.50:
                        coalition_impulse_paid = True
            else:
                coalition_trace = 0.0
                coalition_impulse_paid = False

            # Equal-validity treatment remains continuous but is steep enough
            # to generate stable first-member-like and whole-tier-like subjects.
            tier_key = int(np.sum(validities > validities[cue] + 1e-12))
            previous_in_tier = sampled_by_tier.get(tier_key, 0)
            if previous_in_tier > 0:
                increment_size *= 0.08 + 0.92 * effective_pooling
            sampled_by_tier[tier_key] = previous_in_tier + 1

            accumulator += cue_sign * increment_size
            encountered_signs.append(cue_sign)
            last_sign = cue_sign
            last_index = cue
            alert_sign = 0.0
            step += 1

            if abs(accumulator) >= stopping_threshold:
                # Pooling-oriented subjects probabilistically defer stopping
                # when members of the currently opened validity tier remain.
                # This changes inspection rather than imposing hard completion.
                unresolved_same_tier = [
                    j for j in remaining
                    if int(np.sum(validities > validities[j] + 1e-12)) == tier_key
                ]
                if len(unresolved_same_tier) > 0:
                    tier_continue_prob = 0.05 + 0.90 * effective_pooling
                    tier_u = pseudo_uniform(
                        particle, unresolved_same_tier[0], 180 + step
                    )
                    if tier_u < tier_continue_prob:
                        continue

                provisional_sign = 1.0 if accumulator > 0.0 else -1.0
                dissenters = [j for j in remaining if signs[j] == -provisional_sign]

                if len(dissenters) == 0:
                    # Unanimity licenses rapid, confident termination.
                    break

                # Monitoring emphasizes late conflict while retaining the
                # bounded coalition trace as evidence that a coherent run has
                # formed. Early generic conflict is less likely to reopen the
                # process than genuinely late dissent.
                latest_dissent = max(active_lateness[j] for j in dissenters)
                encounter_lateness = min(1.0, step / float(active.size))
                conflict_lateness = 0.75 * latest_dissent + 0.25 * encounter_lateness
                surplus = max(0.0, abs(accumulator) - stopping_threshold)
                monitor_logit = (
                    -1.45
                    + 0.35 * conflict_trigger
                    + 2.10 * recency_bias * conflict_lateness
                    + 0.55 * coalition_trace
                    - 0.45 * surplus
                    + (0.35 if not reopened else -0.35)
                )
                monitor_prob = 1.0 / (1.0 + np.exp(-np.clip(monitor_logit, -40.0, 40.0)))
                monitor_u = pseudo_uniform(particle, dissenters[-1], 100 + step)

                if monitor_u < monitor_prob:
                    alert_sign = -provisional_sign

                    # Late coherent dissent leaks provisional commitment toward
                    # zero. Coherence is based on adjacency among unresolved
                    # dissenters and saturates after a local coalition exists.
                    # Repeated reopening is habituated rather than repeatedly
                    # erasing the accumulator.
                    dissent_coherence = 0.0
                    if len(dissenters) > 1:
                        ordered_dissenters = sorted(dissenters)
                        dissent_coherence = max(
                            np.exp(-float(gap - 1) / 1.5)
                            for gap in np.diff(ordered_dissenters)
                        )
                    habituation = 0.55 if reopened else 1.0
                    leak_fraction = habituation * (
                        0.10
                        + 0.24
                        * conflict_trigger
                        * conflict_lateness
                        * (0.55 + 0.45 * dissent_coherence)
                    )
                    leak_fraction = float(np.clip(leak_fraction, 0.10, 0.82))
                    accumulator *= 1.0 - leak_fraction
                    reopened = True
                else:
                    break

        logit_a = np.clip(decision_gain * accumulator, -60.0, 60.0)
        if logit_a >= 0.0:
            z = np.exp(-logit_a)
            p_a = 1.0 / (1.0 + z)
        else:
            z = np.exp(logit_a)
            p_a = z / (1.0 + z)
        particle_p_a.append(p_a)

    p_a_core = float(np.mean(particle_p_a))
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + 0.5 * epsilon
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- validity_sensitivity: [1.4, 3.2]
- validity_access_temperature: [0.9, 1.5]
- validity_evidence_temperature: [0.7, 1.3]
- validity_modulation: [0.04, 0.10]
- recency_bias: [0.8, 2.2]
- stopping_threshold: [1.8, 3.4]
- conflict_trigger: [1.2, 3.2]
- coalition_reinforcement: [0.55, 1.65]
- tier_pooling: [0, 1]
- sampling_noise: [0.65, 1.15]
- decision_gain: [2.6, 5.2]
- epsilon: [0, 0.06]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=0.6333 (var=0.0110)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.9250 (var=0.0079)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.2469 (var=0.0084)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.6575 (var=0.0122)
- Experiment 5: real=0.5000 (var=0.2000) vs this=0.9200 (var=0.0636)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=-0.0042 (var=0.0027)
- Experiment 7: real=0.1175 (var=0.0081) vs this=0.1588 (var=0.0166)
- Experiment 8: real=0.7750 (var=0.0279) vs this=0.4888 (var=0.0240)
- Experiment 9: real=1.4567 (var=0.1585) vs this=0.2575 (var=0.0373)
- Experiment 10: real=0.0000 (var=0.7584) vs this=-1.0000 (var=0.8800)


---

### `pi_3` (overall score: 0.141)

**Description**
Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- capacity: [4.5, 5.5]
- gate_slope: [2.0, 5.0]
- beta: [1.55, 2.05]
- epsilon: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.0300 (var=0.0056) vs this=-0.0028 (var=0.0068)
- Experiment 2: real=0.6683 (var=0.0367) vs this=0.6600 (var=0.0188)
- Experiment 3: real=0.3612 (var=0.0106) vs this=0.6512 (var=0.0114)
- Experiment 4: real=0.8756 (var=0.0078) vs this=0.8291 (var=0.0019)
- Experiment 5: real=0.5000 (var=0.2000) vs this=1.0000 (var=0.0000)
- Experiment 6: real=-0.1132 (var=0.0031) vs this=0.0009 (var=0.0007)
- Experiment 7: real=0.1175 (var=0.0081) vs this=0.7987 (var=0.0067)
- Experiment 8: real=0.7750 (var=0.0279) vs this=-0.0169 (var=0.0083)
- Experiment 9: real=1.4567 (var=0.1585) vs this=-0.0333 (var=0.0272)
- Experiment 10: real=0.0000 (var=0.7584) vs this=-1.0000 (var=0.7764)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3446 -> ACCEPTED
- iter 2: loss=0.2701 -> ACCEPTED
- iter 3: loss=0.2590 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.2590 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Previous candidate values (this loop):**
  - iter 1: 0.0306 (var=0.0797) (Δ vs real +0.0606)
  - iter 2: -0.2133 (var=0.0225) (Δ vs real -0.1833)
  - iter 3 (most recent): -0.2833 (var=0.0171) (Δ vs real -0.2533)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0167 (var=0.0052)
- pi_2: 0.8033 (var=0.0243)
- pi_3: -0.0028 (var=0.0068)
- pi_4: 0.0328 (var=0.0015)
- pi_5: 0.0367 (var=0.0093)
- pi_6: 0.6333 (var=0.0110)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Previous candidate values (this loop):**
  - iter 1: 0.4192 (var=0.0534) (Δ vs real -0.2492)
  - iter 2: 0.7517 (var=0.0352) (Δ vs real +0.0833)
  - iter 3 (most recent): 0.6150 (var=0.0555) (Δ vs real -0.0533)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7008 (var=0.0514)
- pi_1: -0.6542 (var=0.0564)
- pi_3: 0.6600 (var=0.0188)
- pi_4: 0.5833 (var=0.0114)
- pi_5: 0.6467 (var=0.0132)
- pi_6: 0.9250 (var=0.0079)

### Experiment 3
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))

```

**Observed (real) value:** 0.3612 (var=0.0106)
**Previous candidate values (this loop):**
  - iter 1: 0.1797 (var=0.0115) (Δ vs real -0.1816)
  - iter 2: 0.6006 (var=0.0104) (Δ vs real +0.2394)
  - iter 3 (most recent): 0.5803 (var=0.0051) (Δ vs real +0.2191)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6512 (var=0.0114)
- pi_2: 0.0066 (var=0.0037)
- pi_1: 0.0041 (var=0.0094)
- pi_4: 0.6541 (var=0.0103)
- pi_5: 0.5350 (var=0.0071)
- pi_6: 0.2469 (var=0.0084)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8756 (var=0.0078)
**Previous candidate values (this loop):**
  - iter 1: 0.6659 (var=0.0058) (Δ vs real -0.2097)
  - iter 2: 0.7597 (var=0.0061) (Δ vs real -0.1159)
  - iter 3 (most recent): 0.7369 (var=0.0047) (Δ vs real -0.1388)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5031 (var=0.0037)
- pi_3: 0.8291 (var=0.0019)
- pi_1: 0.8391 (var=0.0151)
- pi_4: 0.8331 (var=0.0025)
- pi_5: 0.8884 (var=0.0019)
- pi_6: 0.6575 (var=0.0122)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_classification(df):
        aligned = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = np.sign(a - b)

            # Select trials where Experts 2 and 3 agree with each other but
            # oppose Expert 1: a 2-to-1 conflict within the top-validity tier.
            if d.size >= 3 and d[0] != 0 and d[1] == d[2] and d[1] == -d[0]:
                response = int(row['response'])
                chose_a = 1.0 if response == 0 else -1.0
                aligned.append(float(chose_a * d[0] > 0))

        if len(aligned) == 0:
            return np.nan
        rate = float(np.mean(aligned))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    values = []
    for _, subj_df in data.groupby('subject_id', sort=False):
        value = subject_classification(subj_df)
        if np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))
```

**Observed (real) value:** 0.5000 (var=0.2000)
**Previous candidate values (this loop):**
  - iter 1: 0.1400 (var=0.1004) (Δ vs real -0.3600)
  - iter 2: 0.0700 (var=0.0501) (Δ vs real -0.4300)
  - iter 3 (most recent): 0.1000 (var=0.0500) (Δ vs real -0.4000)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.5800 (var=0.2436)
- pi_6: 0.9200 (var=0.0636)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast majority-choice rates across the four positions of the
    # second cue opposing the majority. Coefficients sum to zero, so any
    # profile that is flat across positions has expected metric zero.
    weights = {1: -0.24, 3: -0.05, 5: 0.10, 7: 0.19}
    outcomes = {1: [], 3: [], 5: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0)
        if active.size == 0:
            continue

        majority_sign = np.sign(np.sum(signs[active]))
        if majority_sign == 0:
            continue

        # Expert 1 (array index 0) is the highest-validity cue. The target
        # position is the other active cue pointing in the same direction.
        top_sign = signs[0]
        opposing = [int(j) for j in active if j != 0 and signs[j] == top_sign]
        if len(opposing) != 1 or opposing[0] not in outcomes:
            continue

        response = int(row["response"])
        chose_majority = ((response == 0 and majority_sign > 0) or
                          (response == 1 and majority_sign < 0))
        outcomes[opposing[0]].append(float(chose_majority))

    if any(len(outcomes[j]) == 0 for j in outcomes):
        return float("nan")

    rates = {j: float(np.mean(outcomes[j])) for j in outcomes}
    return float(sum(weights[j] * rates[j] for j in weights))

```

**Observed (real) value:** -0.1132 (var=0.0031)
**Previous candidate values (this loop):**
  - iter 1: -0.0750 (var=0.0029) (Δ vs real +0.0381)
  - iter 2: -0.1844 (var=0.0014) (Δ vs real -0.0712)
  - iter 3 (most recent): -0.1707 (var=0.0016) (Δ vs real -0.0576)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0970 (var=0.0005)
- pi_3: 0.0009 (var=0.0007)
- pi_1: 0.0003 (var=0.0005)
- pi_2: 0.0004 (var=0.0004)
- pi_5: -0.1079 (var=0.0009)
- pi_6: -0.0042 (var=0.0027)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        active = np.flatnonzero(diff != 0.0)

        # Restrict to the six-discriminating-cue trials, where the theories'
        # predictions are maximally separated and the competitor has not yet
        # entered its person-variable high-load transition.
        if active.size != 6:
            continue

        # Validities strictly decrease with display position, so the first
        # active feature is the validity-ranked lexicographic cue.
        first = int(active[0])
        lex_response = 0 if diff[first] > 0.0 else 1
        agreements.append(float(int(row["response"]) == lex_response))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1175 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.3837 (var=0.0238) (Δ vs real +0.2662)
  - iter 2: 0.0675 (var=0.0051) (Δ vs real -0.0500)
  - iter 3 (most recent): 0.1200 (var=0.0108) (Δ vs real +0.0025)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7987 (var=0.0067)
- pi_5: 0.0850 (var=0.0021)
- pi_1: 0.8706 (var=0.0097)
- pi_2: 0.1344 (var=0.0103)
- pi_4: 0.9775 (var=0.0008)
- pi_6: 0.1588 (var=0.0166)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    unanimous = []
    late_loser = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0.0)
        if active.size == 0:
            continue

        margin = float(np.sum(signs[active]))
        if margin == 0.0:
            continue

        # Reorient every trial toward its equal-weight-majority winner, thereby
        # combining A/B-reversed pairs and eliminating response-side direction.
        winner_sign = 1.0 if margin > 0.0 else -1.0
        relative_signs = signs[active] * winner_sign
        winner_response = 0 if winner_sign > 0.0 else 1
        followed_winner = float(int(row.response) == winner_response)

        if np.all(relative_signs > 0.0):
            unanimous.append(followed_winner)
        elif np.sum(relative_signs < 0.0) == 1:
            losing_active_position = int(active[np.flatnonzero(relative_signs < 0.0)[0]])
            if losing_active_position == int(np.max(active)):
                late_loser.append(followed_winner)

    if len(unanimous) == 0 or len(late_loser) == 0:
        return 0.0

    return float(np.mean(unanimous) - np.mean(late_loser))
```

**Observed (real) value:** 0.7750 (var=0.0279)
**Previous candidate values (this loop):**
  - iter 1: 0.6387 (var=0.0279) (Δ vs real -0.1363)
  - iter 2: 0.9225 (var=0.0106) (Δ vs real +0.1475)
  - iter 3 (most recent): 0.7900 (var=0.0291) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3750 (var=0.0124)
- pi_3: -0.0169 (var=0.0083)
- pi_1: -0.0088 (var=0.0039)
- pi_2: 0.0231 (var=0.0063)
- pi_4: 0.0719 (var=0.0047)
- pi_6: 0.4888 (var=0.0240)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    coherent = []
    aligned_choice = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)

        # Expert 1 always favors the majority-supported option in this design.
        majority_sign = float(signs[0])
        if majority_sign == 0.0:
            continue

        minority_positions = np.flatnonzero(signs == -majority_sign)
        if minority_positions.size != 3:
            continue

        # The coherent condition has three consecutive minority cues; the
        # dispersed condition has no adjacent minority pair.
        is_coherent = bool(np.all(np.diff(minority_positions) == 1))

        # +1 means choosing the majority/Expert-1-supported option, regardless
        # of whether that option was displayed as A or B.
        response_sign = 1.0 if int(row["response"]) == 0 else -1.0
        aligned = majority_sign * response_sign

        coherent.append(is_coherent)
        aligned_choice.append(aligned)

    coherent = np.asarray(coherent, dtype=bool)
    aligned_choice = np.asarray(aligned_choice, dtype=float)
    if aligned_choice.size == 0 or not np.any(coherent) or not np.any(~coherent):
        return float("nan")

    return float(np.mean(aligned_choice[coherent]) - np.mean(aligned_choice[~coherent]))

```

**Observed (real) value:** 1.4567 (var=0.1585)
**Previous candidate values (this loop):**
  - iter 1: 0.1458 (var=0.0727) (Δ vs real -1.3108)
  - iter 2: 0.8558 (var=0.2474) (Δ vs real -0.6008)
  - iter 3 (most recent): 0.7275 (var=0.2195) (Δ vs real -0.7292)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2575 (var=0.0373)
- pi_5: 0.1083 (var=0.0232)
- pi_1: -0.0158 (var=0.0201)
- pi_2: -0.0183 (var=0.0279)
- pi_3: -0.0333 (var=0.0272)
- pi_4: -0.0017 (var=0.0032)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify whether Expert 5 (index 4) or Expert 7 (index 6) is the
    # diagnostic candidate cue, and which response selects its favored option.
    cells = {4: {1: [], -1: []}, 6: {1: [], -1: []}}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        d = a - b

        active_candidates = [j for j in (4, 6) if d[j] != 0.0]
        if len(active_candidates) != 1:
            continue

        cue = active_candidates[0]
        orientation = 1 if d[cue] > 0.0 else -1
        response = int(row['response'])
        focal_choice = float((response == 0) if orientation == 1 else (response == 1))
        cells[cue][orientation].append(focal_choice)

    def side_balanced_rate(cue):
        rates = []
        for orientation in (1, -1):
            values = cells[cue][orientation]
            if len(values) > 0:
                rates.append(float(np.mean(values)))
        if len(rates) == 0:
            return np.nan
        return float(np.mean(rates))

    early = side_balanced_rate(4)
    late = side_balanced_rate(6)
    if not np.isfinite(early) or not np.isfinite(late):
        return 0.0

    # Direction of the Expert-5 to Expert-7 focal-choice gradient.
    return float(np.sign(late - early))

```

**Observed (real) value:** 0.0000 (var=0.7584)
**Previous candidate values (this loop):**
  - iter 1: -1.0000 (var=0.1856) (Δ vs real -1.0000)
  - iter 2: -1.0000 (var=0.5184) (Δ vs real -1.0000)
  - iter 3 (most recent): 1.0000 (var=0.6004) (Δ vs real +1.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 1.0000 (var=0.8656)
- pi_6: -1.0000 (var=0.8800)
- pi_1: 1.0000 (var=0.7796)
- pi_2: -1.0000 (var=0.8516)
- pi_3: -1.0000 (var=0.7764)
- pi_4: -1.0000 (var=0.6100)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Adaptive Retrieval Budget theory: choice is produced by a single resource-rational retrieval-and-accumulation process. Every diagnostic cue competes for a limited retrieval budget, with priority jointly determined by communicated validity, display recency, and unresolved opposition to the provisional majority. Endpoint-sensitive conflict priority is continuously scaled by the amount of unresolved conflict, diagnostic load, and the certainty of the provisional majority. Thus, one late dissenter produces limited disruption, several late dissenters under high load sharpen competition, and exact-tie displays leave more room for individual validity-recency differences. Under capacity pressure, an exceptional leading validity tier compresses retrieval toward that tier, but the pressure transition remains gradual. Subjects vary independently in validity and recency sensitivity and continuously in tied-tier pooling. Low pooling conserves a tied tier's retrieval mass but reallocates it to its first displayed member; high pooling distributes that mass across the tier. These are emergent regimes of one retrieval process rather than discrete strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Adaptive Retrieval Budget theory expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    if n_features > 1:
        position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        position = np.zeros(n_features, dtype=float)

    clipped_v = np.clip(validities, 0.500001, 0.999999)
    validity_log_odds = np.log(clipped_v / (1.0 - clipped_v))
    active_v = validities[active]
    active_log_odds = validity_log_odds[active]

    # Distinctiveness requires an exceptional leading-tier gap. In addition
    # to the global structure, assess whether the currently diagnostic cues
    # expose a leading gap that is exceptional relative to their other gaps.
    ordered_global = np.sort(np.unique(np.round(validities, 12)))[::-1]
    if ordered_global.size <= 1:
        global_distinctiveness = 0.0
        ordinary_global_gap = 0.0
    else:
        global_gaps = ordered_global[:-1] - ordered_global[1:]
        leading_gap = float(global_gaps[0])
        ordinary_global_gap = float(np.median(global_gaps))
        dominance_ratio = leading_gap / (
            leading_gap + ordinary_global_gap + 1e-12
        )
        global_distinctiveness = float(np.clip(
            (dominance_ratio - 0.50) / 0.35,
            0.0,
            1.0,
        ))

    ordered_active = np.sort(np.unique(np.round(active_v, 12)))[::-1]
    if ordered_active.size <= 1:
        active_distinctiveness = 0.0
    else:
        active_gaps = ordered_active[:-1] - ordered_active[1:]
        active_leading_gap = float(active_gaps[0])
        active_reference_gap = float(np.median(active_gaps))
        if ordinary_global_gap > 0.0:
            active_reference_gap = max(
                ordinary_global_gap,
                min(active_reference_gap, 2.0 * ordinary_global_gap),
            )
        active_ratio = active_leading_gap / (
            active_leading_gap + active_reference_gap + 1e-12
        )
        active_distinctiveness = float(np.clip(
            (active_ratio - 0.55) / 0.35,
            0.0,
            1.0,
        ))

    distinctiveness = max(
        global_distinctiveness,
        0.75 * active_distinctiveness,
    )

    positive_fraction = float(np.mean(signs[active] > 0.0))
    conflict_entropy = 4.0 * positive_fraction * (1.0 - positive_fraction)

    # Resource availability grows sublinearly with diagnostic load and grows
    # further when conflict makes additional retrieval worthwhile.
    capacity = (
        float(parameters["base_capacity"])
        + float(parameters["load_capacity_adaptation"]) * np.log1p(load)
        + 0.35 * float(parameters["load_capacity_adaptation"])
        * conflict_entropy * np.sqrt(load)
    )
    stopping_requirement = float(parameters["stopping_threshold"]) * (
        1.0
        + float(parameters["stopping_adaptation"])
        * conflict_entropy
        * (1.0 - distinctiveness)
    )
    retrieval_dose = max(0.05, capacity * stopping_requirement)

    # A broader pressure transition prevents an abrupt load-indexed crossover.
    pressure_x = (load - capacity) / 1.0
    pressure_x = float(np.clip(pressure_x, -40.0, 40.0))
    pressure = 1.0 / (1.0 + np.exp(-pressure_x))
    compression = pressure * distinctiveness

    if np.std(active_log_odds) > 1e-12:
        relative_validity = (
            active_log_odds - float(np.mean(active_log_odds))
        ) / float(np.std(active_log_odds))
    else:
        relative_validity = np.zeros(active.size, dtype=float)

    # A cheap provisional sign estimate identifies unresolved opposition. It
    # is only an orienting signal: final evidence comes from retrieval.
    tally_sum = float(np.sum(signs[active]))
    provisional_certainty = abs(tally_sum) / max(load, 1.0)
    provisional_sign = float(np.sign(tally_sum))
    if provisional_sign == 0.0:
        provisional_sign = float(np.sign(np.dot(
            signs[active], 1.0 + 0.15 * active_log_odds
        )))

    unresolved = np.zeros(active.size, dtype=float)
    if provisional_sign != 0.0:
        unresolved = (signs[active] == -provisional_sign).astype(float)

    late_conflict = unresolved * np.power(
        position[active], float(parameters["conflict_lateness_power"])
    )

    # Endpoint sensitivity is modulated continuously by unresolved-conflict
    # mass, load, and provisional certainty. Single dissenters are attenuated;
    # several dissenters under high load sharpen competition. Exact tally ties
    # receive only a weak alert because there is no secure commitment to undo.
    unresolved_positions = position[active][unresolved > 0.0]
    unresolved_count = float(np.sum(unresolved))
    if unresolved_positions.size > 0:
        latest_unresolved = float(np.max(unresolved_positions))
        alert_x = np.clip((latest_unresolved - 0.80) / 0.035, -40.0, 40.0)
        endpoint_alert = 1.0 / (1.0 + np.exp(-alert_x))

        mass_x = np.clip((unresolved_count - 2.2) / 0.35, -40.0, 40.0)
        conflict_mass_gate = 1.0 / (1.0 + np.exp(-mass_x))
        load_x = np.clip((load - 5.5) / 1.0, -40.0, 40.0)
        load_gate = 1.0 / (1.0 + np.exp(-load_x))
        certainty_gate = 0.25 + 0.75 * min(
            1.0, 4.0 * provisional_certainty
        )
        alert_multiplier = (
            0.20
            + 1.50 * conflict_mass_gate * (0.70 + 0.30 * load_gate)
        ) * certainty_gate
        late_alert = endpoint_alert * alert_multiplier
    else:
        late_alert = 0.0

    centered_absolute_validity = active_log_odds - np.log(3.0)
    log_priority = (
        float(parameters["validity_access"]) * centered_absolute_validity
        + float(parameters["distinctiveness_gain"])
        * compression * relative_validity
        + float(parameters["recency_access"]) * position[active]
        + float(parameters["conflict_priority"]) * late_conflict
    )

    # Stable person variation in tied-tier representation remains continuous.
    # Allocation is performed after exponentiation so retrieval mass removed
    # from later tied members is transferred to that tier's first member rather
    # than leaking to unrelated lower-validity tiers.
    tie_pooling = float(parameters["tie_pooling"])
    continuation_x = np.clip(14.0 * (tie_pooling - 0.50), -40.0, 40.0)
    continuation = 1.0 / (1.0 + np.exp(-continuation_x))
    continuation = 0.002 + 0.996 * continuation

    # Extreme late dissent sharpens the same finite-budget competition and
    # contracts retrieval breadth; no reopening or adjacency process is added.
    priority_center = float(np.mean(log_priority))
    priority_sharpening = (
        1.0 + float(parameters["late_competition_sharpening"]) * late_alert
    )
    log_priority = priority_center + priority_sharpening * (
        log_priority - priority_center
    )
    retrieval_dose *= max(
        0.08,
        1.0 - float(parameters["late_budget_contraction"]) * late_alert,
    )

    log_priority -= np.max(log_priority)
    priority = np.exp(log_priority)

    # Conserve each tied tier's total competitive mass while continuously
    # reallocating it between first-member representation and whole-tier use.
    for tier_value in np.unique(np.round(active_v, 12)):
        tier_local = np.flatnonzero(np.isclose(
            active_v, tier_value, rtol=1e-9, atol=1e-12
        ))
        if tier_local.size <= 1:
            continue
        first_local = int(tier_local[np.argmin(active[tier_local])])
        tier_mass = float(np.sum(priority[tier_local]))
        priority[tier_local] *= continuation
        priority[first_local] += (1.0 - continuation) * tier_mass

    priority_sum = float(priority.sum())
    if not np.isfinite(priority_sum) or priority_sum <= 0.0:
        priority = np.ones(active.size, dtype=float) / float(active.size)
    else:
        priority /= priority_sum

    retrieval_probability = 1.0 - np.exp(
        -retrieval_dose * float(active.size) * priority
    )
    retrieval_probability = np.clip(retrieval_probability, 0.0, 1.0)

    bounded_absolute = np.tanh(active_log_odds / 2.5)
    bounded_relative = np.tanh(relative_validity / 1.5)
    evidence_strength = (
        1.0
        + float(parameters["absolute_validity_weight"]) * bounded_absolute
        + 0.65 * float(parameters["distinctiveness_gain"])
        * compression * bounded_relative
    )
    evidence_strength = np.clip(evidence_strength, 0.08, 4.0)

    contributions = retrieval_probability * evidence_strength
    contribution_mass = float(np.sum(contributions))
    if not np.isfinite(contribution_mass) or contribution_mass <= 1e-12:
        return np.array([0.5, 0.5], dtype=float)

    accumulated_evidence = float(
        np.dot(contributions, signs[active]) / contribution_mass
    )

    decision_logit = float(parameters["decision_gain"]) * accumulated_evidence
    decision_logit = float(np.clip(decision_logit, -60.0, 60.0))
    if decision_logit >= 0.0:
        z = np.exp(-decision_logit)
        p_a_core = 1.0 / (1.0 + z)
    else:
        z = np.exp(decision_logit)
        p_a_core = z / (1.0 + z)

    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + 0.5 * epsilon
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- base_capacity: [1.8, 2.8]
- load_capacity_adaptation: [0.35, 0.8]
- validity_access: [0.0, 0.9]
- recency_access: [0.0, 9.0]
- conflict_priority: [5.0, 10.0]
- conflict_lateness_power: [6.0, 11.0]
- distinctiveness_gain: [2.5, 5.0]
- stopping_threshold: [0.75, 1.25]
- stopping_adaptation: [0.15, 0.65]
- tie_pooling: [0, 1]
- late_competition_sharpening: [2.0, 4.5]
- late_budget_contraction: [0.70, 0.90]
- absolute_validity_weight: [0.0, 0.25]
- decision_gain: [4.0, 7.0]
- epsilon: [0, 0.06]

`rationale`: This is a minimal interaction-and-normalization edit to the accepted model. First, the endpoint alert is no longer equally strong for every late dissenter. Its effect increases smoothly with unresolved-cue count and load, but is attenuated for a single dissenter and for exact tally ties. This should reduce the overeffects in Experiments 1, 6, and 8, preserve or enlarge the three-dissenter effect in Experiment 9, and allow independent validity-recency variation to balance Experiment 10. Second, the pressure width changes from 0.65 to 1.0, softening the excessive Experiment 3 crossover without introducing a strategy gate. Active-diagnostic distinctiveness is added as a bounded supplement to global distinctiveness, preserving validity-led behavior in Experiment 4 when the active cues expose an exceptional leading gap. Third, tied-tier continuation is moved from independent log-priority attenuation to mass-conserving within-tier allocation. Low-continuation subjects now transfer later tied members' retrieval mass to the first displayed tier member, while high-continuation subjects retain whole-tier pooling. This directly targets Experiment 5's missing split and variance without changing the common retrieval-budget mechanism.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed resource-rational selective-retrieval family and improves the late-dissent signature in Experiment 8, with the correct direction and matched variance (0.639 versus 0.775), as well as the positional effect in Experiment 6 (-0.075 versus -0.113). It also avoids pi_6's extreme failure in Experiment 1. However, the aggregate fit is not yet experiment-invariant. The largest failure is Experiment 9: the model predicts only a small coherent-minus-dispersed contrast (0.146) compared with the observed 1.457. Thus, although late dissent has priority, retrieval competition is not sharp enough for the dispersed condition's latest opposing cue to displace the other evidence. Experiment 10 also fails categorically: the pooled gradient is -1 rather than 0, and simulated between-subject variance is much too low, so validity dominates recency too uniformly across subjects. The model underproduces majority behavior in Experiment 2 (0.419 versus 0.668), the load crossover in Experiment 3 (0.180 versus 0.361), and validity-led overload behavior in Experiment 4 (0.666 versus 0.876). Conversely, it overpredicts lexicographic agreement in Experiment 7 (0.384 versus 0.118), suggesting that the current distinctiveness calculation labels diffuse descending validity structures as too compressible. Finally, tied-tier heterogeneity is inadequate in Experiment 5: only 0.14 of subjects are classified as following the first top-tier member, versus 0.50 observed, with too little variance. Experiment 1's mean is reasonably close but has greatly excessive subject variance.
Rationale: Retain the single retrieval-budget mechanism, but recalibrate its competition and distinctiveness components. First, make late-conflict competition substantially more selective under limited capacity—such as a lower retrieval dose and/or a sharper priority temperature specifically when a very late unresolved cue is present—rather than merely raising all dissent probabilities. This should enlarge Experiments 8 and especially 9 without introducing adjacency. Second, reduce unconditional recency weighting and concentrate positional influence in the late-conflict interaction; broad low-load retrieval should then recover the stronger tally result in Experiment 2, while the conflict term preserves Experiments 6, 8, and 9. Third, redefine validity distinctiveness around evidence for a genuinely dominant tier, rather than giving substantial weight to global validity standard deviation. A diffuse strictly descending structure should remain poolable, reducing the excessive lexicographic prediction in Experiment 7, whereas a separated leading tier under overload should receive stronger compression, improving Experiments 3 and 4. A steeper continuous pressure function may also strengthen the low-to-high load crossover without creating a discrete strategy gate. Fourth, strengthen tied-tier person variation: the current continuation mapping leaves too many later tied cues competitive after late-conflict priority is added. Use a more nonlinear continuation mapping with a near-zero lower tail, so a substantial subset effectively represents the early retrieved tier member while another subset pools the tier, targeting Experiment 5's mean and variance. Finally, widen or rebalance independent recency-versus-validity sensitivity around their crossover so Experiment 10 contains substantial subject mass on both gradient directions and pools near zero; avoid globally weakening validity, since Experiment 4 already requires more validity-led compression. These are parameterization and normalization changes within the recommended common-process family, not a change of mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.3446 -> iter 2 candidate loss=0.2701 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision materially improves the overall fit and remains faithful to the prescribed single resource-rational retrieval process. In particular, Experiment 2 is now close in mean and variance (0.752 versus 0.668), Experiment 4 moves toward validity-led overload behavior, Experiment 7 is close (0.068 versus 0.118), and the previously missing coherent-versus-dispersed effect in Experiment 9 has become substantial (0.856 versus 1.457). Thus the prior advice to sharpen endpoint-sensitive retrieval competition and revise distinctiveness was productive. However, the new endpoint mechanism is too strong when only one late dissenting cue is present: it overshoots Experiment 8 (0.923 versus 0.775), reverses and exaggerates Experiment 1 (-0.213 versus -0.030), and makes Experiment 6 too negative (-0.184 versus -0.113). At the same time, it is still not strong enough for the multiple-dissent, high-load contrast in Experiment 9. Experiment 3 also overshoots its load crossover (0.601 versus 0.361), indicating excessively abrupt load-dependent compression. The largest qualitative failure is Experiment 5: the intended tied-tier heterogeneity did not emerge, with only 0.07 classified as first-member followers versus 0.50 observed and far too little variance. Experiment 10 remains categorically negative rather than balanced, although its variance improved from 0.186 to 0.518; the structural late-conflict effect is evidently still dominating the independent recency-validity variation.
Rationale: Build on this accepted revision rather than undoing its successful direction, but make the late-alert effect depend continuously on unresolved-conflict mass and diagnostic load. Reduce budget contraction and sharpening for a single late dissenter, while preserving or strengthening them when several unresolved opposing cues compete under high load. This should move Experiments 1, 6, and 8 back toward their targets and allow recency-versus-validity heterogeneity to determine Experiment 10, while retaining or enlarging Experiment 9 without adding adjacency or coalition machinery. The modulation can remain continuous, for example by multiplying the endpoint alert by a smooth function of the number or fraction of unresolved cues and load pressure. For Experiment 3, use a less steep pressure transition or an intermediate compression gain rather than further increasing compression; the previous increase helped Experiment 4 but clearly overshot the crossover. Improve Experiment 4 separately through the validity normalization, such as calculating leading-tier distinctiveness among the active diagnostic cues rather than only from the global validity vector, so genuinely diagnostic active gaps receive priority without making all high-load trials abruptly lexicographic. Finally, fix tied-tier retrieval allocation rather than merely making the continuation sigmoid steeper. When continuation is low, attenuated later members of a tied tier should transfer competitive retrieval mass primarily to that tier's first retrieved member, not to unrelated lower tiers; when continuation is high, the whole tier should remain pooled. Applying this within-tier allocation at the final priority-normalization stage should produce the required split and variance in Experiment 5. These are normalization and parameter-interaction adjustments within the prescribed common retrieval-budget family.

**Outcome of this advice:** iter 2 candidate loss=0.2701 -> iter 3 candidate loss=0.2590 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed single, capacity-limited selective-retrieval family, and the latest accepted revision improves aggregate loss. Several signatures are now convincing: Experiment 7 is essentially exact (0.120 versus 0.118), Experiment 8 is close in both mean and variance (0.790 versus 0.775), Experiment 2 is reasonably close with appropriate heterogeneity (0.615 versus 0.668), and Experiment 6 retains the correct positional direction. Thus, the prior recommendations concerning continuous conflict modulation, active distinctiveness, and tied-tier allocation helped overall. Nevertheless, the model is not yet experiment-invariant. Experiment 1 has deteriorated again: its contrast is strongly negative (-0.283) when humans are approximately invariant to congruence (-0.030), indicating that late-conflict competition disrupts majority/top-cue agreement too strongly in this dominant-validity structure. Experiment 9 remains far too small (0.728 versus 1.457), so the same mechanism still does not adequately separate a dispersed set containing a very late dissenter from the centered coherent set. Experiment 3 continues to show excessive load crossover (0.580 versus 0.361), while Experiment 4 remains insufficiently validity-led under overload (0.737 versus 0.876). Together these imply that compression is insufficiently selective: too much crossover is generated in relatively diffuse structures, yet genuinely dominant structures are not protected enough. The mass-conserving tie edit also failed its principal target: Experiment 5 remains near 0.10 rather than 0.50 and has only one quarter of the observed variance. Finally, Experiment 10 has flipped from -1 to +1 rather than pooling at zero; its increased variance is useful, but the balance between validity and recency remains miscentered.
Rationale: Continue within the accepted resource-rational retrieval family, but change different in-family details rather than simply strengthening the previous conflict-mass gate, which improved aggregate loss while worsening Experiment 1 and reducing Experiment 9 from its iteration-2 value. First, let dominant-tier compression continuously suppress the endpoint-induced sharpening and budget contraction, while leaving ordinary positional retrieval intact. Experiment 1 appears to combine very late dissent with a strong validity hierarchy, whereas Experiment 9 requires endpoint disruption in a more diffuse structure. Multiplying only the extreme endpoint modulation by a decreasing function of distinctiveness/compression should therefore protect Experiment 1 without sacrificing the basic Experiment 6 positional effect, while permitting stronger endpoint selectivity in Experiment 9. Second, make distinctiveness more selective rather than globally increasing compression: near-zero compression for diffuse or evenly graded structures, but stronger compression once the leading active/global gap is genuinely exceptional. This should reduce Experiment 3's excessive crossover while improving Experiment 4 and further shielding Experiment 1. A modestly wider pressure transition may help, but the main correction should be the shape of the distinctiveness mapping, not another broad gain increase. Third, fix the remaining tied-tier normalization problem in Experiment 5. Reallocating priority mass to the first tied member does not preserve accumulated evidence because the subsequent saturating retrieval-probability transform converts several potential retrievals into one bounded contribution. Under low pooling, the first-member representation should inherit the tier's effective contribution mass or multiplicity, not merely its pre-saturation priority; under high pooling, separate members should retain their contributions. This is still continuous representational pooling within one retrieval process and should produce both first-member followers and pooled-majority followers. Finally, Experiment 10 has oscillated from negative to positive across the last two accepted candidates. Do not push recency or validity to the opposite extreme again. Tune the exact-tie conflict-alert floor or the validity-recency balance to an intermediate setting between iterations 2 and 3, and preserve broad independent subject variation so the pooled direction approaches zero and variance rises toward 0.758. These targeted adjustments should preserve the now-good fits in Experiments 2, 7, and 8 while addressing the largest remaining structural failures.

**Outcome of this advice:** iter 3 candidate loss=0.2590 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Adaptive Retrieval Budget theory: choice is produced by a single resource-rational retrieval-and-accumulation process. Every diagnostic cue competes for a limited retrieval budget, with priority jointly determined by communicated validity, display recency, and unresolved opposition to the provisional majority. Endpoint-sensitive conflict priority is continuously scaled by unresolved conflict, diagnostic load, provisional certainty, and validity-structure distinctiveness. Extreme endpoint competition is suppressed when a genuinely dominant validity tier supports efficient compression, but remains strong in diffuse structures. Thus, late dispersed dissent can be highly disruptive without any intrinsic adjacency bonus. Under capacity pressure, exceptional leading tiers compress retrieval toward that tier, whereas diffuse validity structures support approximate pooling. Subjects vary independently in validity and recency sensitivity and continuously in tied-tier pooling. Low pooling represents a tied tier through its first displayed member, which inherits the tier's effective contribution capacity; high pooling distributes contributions across its members. These are emergent regimes of one retrieval process rather than discrete strategies.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Adaptive Retrieval Budget theory expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    signs = np.sign(a - b)\n    active = np.flatnonzero(signs != 0.0)\n    load = float(active.size)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=float)\n\n    if n_features > 1:\n        position = np.arange(n_features, dtype=float) / float(n_features - 1)\n    else:\n        position = np.zeros(n_features, dtype=float)\n\n    clipped_v = np.clip(validities, 0.500001, 0.999999)\n    validity_log_odds = np.log(clipped_v / (1.0 - clipped_v))\n    active_v = validities[active]\n    active_log_odds = validity_log_odds[active]\n\n    # Distinctiveness requires an exceptional leading-tier gap. In addition\n    # to the global structure, assess whether the currently diagnostic cues\n    # expose a leading gap that is exceptional relative to their other gaps.\n    ordered_global = np.sort(np.unique(np.round(validities, 12)))[::-1]\n    if ordered_global.size <= 1:\n        global_distinctiveness = 0.0\n        ordinary_global_gap = 0.0\n    else:\n        global_gaps = ordered_global[:-1] - ordered_global[1:]\n        leading_gap = float(global_gaps[0])\n        ordinary_global_gap = float(np.median(global_gaps))\n        dominance_ratio = leading_gap / (\n            leading_gap + ordinary_global_gap + 1e-12\n        )\n        global_distinctiveness = float(np.clip(\n            (dominance_ratio - 0.50) / 0.35,\n            0.0,\n            1.0,\n        ))\n\n    ordered_active = np.sort(np.unique(np.round(active_v, 12)))[::-1]\n    if ordered_active.size <= 1:\n        active_distinctiveness = 0.0\n    else:\n        active_gaps = ordered_active[:-1] - ordered_active[1:]\n        active_leading_gap = float(active_gaps[0])\n        active_reference_gap = float(np.median(active_gaps))\n        if ordinary_global_gap > 0.0:\n            active_reference_gap = max(\n                ordinary_global_gap,\n                min(active_reference_gap, 2.0 * ordinary_global_gap),\n            )\n        active_ratio = active_leading_gap / (\n            active_leading_gap + active_reference_gap + 1e-12\n        )\n        active_distinctiveness = float(np.clip(\n            (active_ratio - 0.55) / 0.35,\n            0.0,\n            1.0,\n        ))\n\n    raw_distinctiveness = max(\n        global_distinctiveness,\n        0.75 * active_distinctiveness,\n    )\n    # Squaring selectively suppresses merely graded structures while retaining\n    # genuinely exceptional leading tiers near the top of the scale.\n    distinctiveness = raw_distinctiveness ** 2\n\n    positive_fraction = float(np.mean(signs[active] > 0.0))\n    conflict_entropy = 4.0 * positive_fraction * (1.0 - positive_fraction)\n\n    # Resource availability grows sublinearly with diagnostic load and grows\n    # further when conflict makes additional retrieval worthwhile.\n    capacity = (\n        float(parameters[\"base_capacity\"])\n        + float(parameters[\"load_capacity_adaptation\"]) * np.log1p(load)\n        + 0.35 * float(parameters[\"load_capacity_adaptation\"])\n        * conflict_entropy * np.sqrt(load)\n    )\n    stopping_requirement = float(parameters[\"stopping_threshold\"]) * (\n        1.0\n        + float(parameters[\"stopping_adaptation\"])\n        * conflict_entropy\n        * (1.0 - distinctiveness)\n    )\n    retrieval_dose = max(0.05, capacity * stopping_requirement)\n\n    # A broad pressure transition prevents an abrupt load-indexed crossover.\n    pressure_x = (load - capacity) / 1.25\n    pressure_x = float(np.clip(pressure_x, -40.0, 40.0))\n    pressure = 1.0 / (1.0 + np.exp(-pressure_x))\n    compression = pressure * distinctiveness\n\n    if np.std(active_log_odds) > 1e-12:\n        relative_validity = (\n            active_log_odds - float(np.mean(active_log_odds))\n        ) / float(np.std(active_log_odds))\n    else:\n        relative_validity = np.zeros(active.size, dtype=float)\n\n    # A cheap provisional sign estimate identifies unresolved opposition. It\n    # is only an orienting signal: final evidence comes from retrieval.\n    tally_sum = float(np.sum(signs[active]))\n    provisional_certainty = abs(tally_sum) / max(load, 1.0)\n    provisional_sign = float(np.sign(tally_sum))\n    if provisional_sign == 0.0:\n        provisional_sign = float(np.sign(np.dot(\n            signs[active], 1.0 + 0.15 * active_log_odds\n        )))\n\n    unresolved = np.zeros(active.size, dtype=float)\n    if provisional_sign != 0.0:\n        unresolved = (signs[active] == -provisional_sign).astype(float)\n\n    late_conflict = unresolved * np.power(\n        position[active], float(parameters[\"conflict_lateness_power\"])\n    )\n\n    # Endpoint sensitivity is modulated continuously by unresolved-conflict\n    # mass, load, and provisional certainty. Single dissenters are attenuated;\n    # several dissenters under high load sharpen competition. Exact tally ties\n    # receive a weak alert because there is no secure commitment to undo.\n    unresolved_positions = position[active][unresolved > 0.0]\n    unresolved_count = float(np.sum(unresolved))\n    if unresolved_positions.size > 0:\n        latest_unresolved = float(np.max(unresolved_positions))\n        alert_x = np.clip((latest_unresolved - 0.80) / 0.035, -40.0, 40.0)\n        endpoint_alert = 1.0 / (1.0 + np.exp(-alert_x))\n\n        mass_x = np.clip((unresolved_count - 2.2) / 0.35, -40.0, 40.0)\n        conflict_mass_gate = 1.0 / (1.0 + np.exp(-mass_x))\n        load_x = np.clip((load - 5.5) / 1.0, -40.0, 40.0)\n        load_gate = 1.0 / (1.0 + np.exp(-load_x))\n        certainty_gate = 0.18 + 0.82 * min(\n            1.0, 4.0 * provisional_certainty\n        )\n        alert_multiplier = (\n            0.20\n            + 1.50 * conflict_mass_gate * (0.70 + 0.30 * load_gate)\n        ) * certainty_gate\n        late_alert = endpoint_alert * alert_multiplier\n    else:\n        late_alert = 0.0\n\n    centered_absolute_validity = active_log_odds - np.log(3.0)\n    log_priority = (\n        float(parameters[\"validity_access\"]) * centered_absolute_validity\n        + float(parameters[\"distinctiveness_gain\"])\n        * compression * relative_validity\n        + float(parameters[\"recency_access\"]) * position[active]\n        + float(parameters[\"conflict_priority\"]) * late_conflict\n    )\n\n    # Stable person variation in tied-tier representation remains continuous.\n    # Allocation is performed after exponentiation so retrieval mass removed\n    # from later tied members is transferred to that tier's first member rather\n    # than leaking to unrelated lower-validity tiers.\n    tie_pooling = float(parameters[\"tie_pooling\"])\n    continuation_x = np.clip(14.0 * (tie_pooling - 0.50), -40.0, 40.0)\n    continuation = 1.0 / (1.0 + np.exp(-continuation_x))\n    continuation = 0.002 + 0.996 * continuation\n\n    # Extreme endpoint competition is useful in diffuse structures, but a\n    # dominant tier supports efficient compression and protects commitment.\n    protected_late_alert = late_alert * (1.0 - 0.90 * compression)\n    priority_center = float(np.mean(log_priority))\n    priority_sharpening = (\n        1.0\n        + float(parameters[\"late_competition_sharpening\"])\n        * protected_late_alert\n    )\n    log_priority = priority_center + priority_sharpening * (\n        log_priority - priority_center\n    )\n    retrieval_dose *= max(\n        0.08,\n        1.0\n        - float(parameters[\"late_budget_contraction\"])\n        * protected_late_alert,\n    )\n\n    log_priority -= np.max(log_priority)\n    priority = np.exp(log_priority)\n\n    # Conserve each tied tier's total competitive mass while continuously\n    # reallocating it between first-member representation and whole-tier use.\n    for tier_value in np.unique(np.round(active_v, 12)):\n        tier_local = np.flatnonzero(np.isclose(\n            active_v, tier_value, rtol=1e-9, atol=1e-12\n        ))\n        if tier_local.size <= 1:\n            continue\n        first_local = int(tier_local[np.argmin(active[tier_local])])\n        tier_mass = float(np.sum(priority[tier_local]))\n        priority[tier_local] *= continuation\n        priority[first_local] += (1.0 - continuation) * tier_mass\n\n    priority_sum = float(priority.sum())\n    if not np.isfinite(priority_sum) or priority_sum <= 0.0:\n        priority = np.ones(active.size, dtype=float) / float(active.size)\n    else:\n        priority /= priority_sum\n\n    retrieval_probability = 1.0 - np.exp(\n        -retrieval_dose * float(active.size) * priority\n    )\n    retrieval_probability = np.clip(retrieval_probability, 0.0, 1.0)\n\n    # A first-member representation inherits the tied tier's effective\n    # contribution multiplicity. This correction is applied after the\n    # saturating retrieval transform, where priority-mass conservation alone\n    # cannot preserve the influence of multiple represented tier members.\n    representation_multiplicity = np.ones(active.size, dtype=float)\n    for tier_value in np.unique(np.round(active_v, 12)):\n        tier_local = np.flatnonzero(np.isclose(\n            active_v, tier_value, rtol=1e-9, atol=1e-12\n        ))\n        if tier_local.size <= 1:\n            continue\n        first_local = int(tier_local[np.argmin(active[tier_local])])\n        representation_multiplicity[first_local] = (\n            1.0 + float(tier_local.size - 1) * (1.0 - continuation)\n        )\n\n    bounded_absolute = np.tanh(active_log_odds / 2.5)\n    bounded_relative = np.tanh(relative_validity / 1.5)\n    evidence_strength = (\n        1.0\n        + float(parameters[\"absolute_validity_weight\"]) * bounded_absolute\n        + 0.65 * float(parameters[\"distinctiveness_gain\"])\n        * compression * bounded_relative\n    )\n    evidence_strength = np.clip(evidence_strength, 0.08, 4.0)\n\n    contributions = (\n        retrieval_probability\n        * representation_multiplicity\n        * evidence_strength\n    )\n    contribution_mass = float(np.sum(contributions))\n    if not np.isfinite(contribution_mass) or contribution_mass <= 1e-12:\n        return np.array([0.5, 0.5], dtype=float)\n\n    accumulated_evidence = float(\n        np.dot(contributions, signs[active]) / contribution_mass\n    )\n\n    decision_logit = float(parameters[\"decision_gain\"]) * accumulated_evidence\n    decision_logit = float(np.clip(decision_logit, -60.0, 60.0))\n    if decision_logit >= 0.0:\n        z = np.exp(-decision_logit)\n        p_a_core = 1.0 / (1.0 + z)\n    else:\n        z = np.exp(decision_logit)\n        p_a_core = z / (1.0 + z)\n\n    epsilon = float(parameters[\"epsilon\"])\n    p_a = (1.0 - epsilon) * p_a_core + 0.5 * epsilon\n    probs = np.array([p_a, 1.0 - p_a], dtype=float)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=float)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "base_capacity": "[1.8, 2.8]",
    "load_capacity_adaptation": "[0.35, 0.8]",
    "validity_access": "[0.0, 0.9]",
    "recency_access": "[0.0, 9.0]",
    "conflict_priority": "[5.0, 10.0]",
    "conflict_lateness_power": "[6.0, 11.0]",
    "distinctiveness_gain": "[3.5, 6.0]",
    "stopping_threshold": "[0.75, 1.25]",
    "stopping_adaptation": "[0.15, 0.65]",
    "tie_pooling": "[0, 1]",
    "late_competition_sharpening": "[2.8, 5.5]",
    "late_budget_contraction": "[0.78, 0.93]",
    "absolute_validity_weight": "[0.0, 0.25]",
    "decision_gain": "[4.0, 7.0]",
    "epsilon": "[0, 0.06]"
  },
  "rationale": "This is a targeted edit to the accepted common-process model. First, distinctiveness is squared and the pressure transition is widened, suppressing compression in merely graded structures to reduce Experiment 3's excessive crossover. The distinctiveness gain is increased only after this selective transformation, preserving stronger validity-led compression for genuinely exceptional tiers in Experiment 4. Second, only endpoint-induced sharpening and budget contraction are attenuated by dominant-tier compression. This should protect Experiment 1 from late-dissent overdisruption while allowing stronger competition in diffuse Experiment 9 displays; the sharpening and contraction ranges are modestly raised to enlarge that dispersed-versus-centered contrast. Ordinary recency and conflict priority are unchanged, preserving Experiments 6 and 8. Third, low-pooling tied tiers now transfer effective contribution multiplicity—not just pre-saturation priority—to their first displayed member. This directly addresses Experiment 5's missing first-member subgroup while retaining continuous pooling variation. Finally, the exact-tie certainty floor is moved modestly from 0.25 to 0.18 rather than reversing validity or recency strength, aiming to center Experiment 10 between its prior negative and positive outcomes while preserving subject heterogeneity."
}
```

## Usage

```json
{
  "input_tokens": 28977,
  "output_tokens": 5302
}
```
