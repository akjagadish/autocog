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
**Description:** Conflict-Reopening Attentional Race theory: diagnostic cues become available during a noisy, approximately display-ordered attentional race. Accessibility reflects communicated validity and recency, while sampled evidence accumulates toward a subject-specific stopping threshold. Reaching the threshold normally terminates comparison, especially when encountered cues agree. Peripheral detection of unresolved dissent can reopen comparison and transiently prioritize opposing evidence. Reopening after late coherent dissent causes bounded leakage of the provisional accumulator toward zero, undoing premature commitment without repeatedly multiplying opposing evidence. Spatially adjacent same-sign samples establish a coherent coalition, but each run receives only a finite one-shot reinforcement. Equal-validity treatment varies continuously across people: after a tier is opened, pooling-oriented subjects preferentially inspect its unresolved members, whereas first-member-oriented subjects can move on or stop early. Load has no strategy gate and matters only through sampling opportunities, conflict, coalition formation, stopping, and probabilistic reopening.

**Parameters:**
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

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

**Parameters:**
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

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight validity-ranked cues create a controlled tally-margin series while holding Take The Best's decision basis constant. In the first eight pairs, the highest-validity cue always favors A, so TTB predicts the same A-choice probability on every pair, regardless of the other seven cues. Across those pairs, however, A wins m = 1,...,8 cues, producing tally differences of -6, -4, -2, 0, 2, 4, 6, and 8. Tallying therefore predicts strong B choices for m=1 to 3, exact guessing at m=4, and increasingly confident A choices for m=5 to 8. In particular, the m=4 pair gives a sharp dissociation: TTB favors A while Tallying predicts exactly 0.5 for any beta and epsilon. The final eight pairs swap A and B, preserving the theoretical contrasts while controlling response-side bias. Thus TTB predicts a flat choice function conditional on the top cue's direction, whereas Tallying predicts a steep monotonic function of the number of cue wins. With 16 unique pairs, each is repeated six times for 96 total trials.

[1] This design orthogonally contrasts the direction of the overall tally with the direction of the first validity-ranked discriminating cue. Every pair contains exactly three discriminating cues. In trials 1-6, A wins two cues and B wins one, so Tallying predicts the same A-choice probability on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta). However, the first discriminating cue occurs successively at validity ranks 1 through 6 and alternates between favoring A and favoring B. Take The Best therefore predicts alternating high and low A-choice probabilities even though the tally is unchanged. Trials 7-12 swap A and B, balancing response side and reversing both predictions. Across all 12 pairs, tally direction and TTB direction are balanced: the theories agree on six pairs and directly conflict on six. Thus evidence for Tallying is a two-level choice pattern determined solely by the 2-versus-1 cue majority, invariant to stopping rank and the top discriminating cue; TTB instead predicts choices grouped by that first discriminating cue. With 12 unique pairs, each pair is repeated eight times for 96 trials.

[2] This design targets the advocated theory's capacity gate by varying the number of discriminating cues from 3 through 8, spanning the capacity range of 4.5-5.5. In the first six pairs, equal-weight tallying favors A while the highest-validity discriminating cue always favors B; the remaining six pairs are exact A/B reversals. Tally margins are held constant within parity: A wins by one cue at loads 3, 5, and 7, and by two cues at loads 4, 6, and 8. Consequently, the competing Tallying theory predicts load-invariant A-choice probabilities within each parity series—sigmoid(beta) for odd loads and sigmoid(2*beta) for even loads—and always favors the cue-majority option. In contrast, Capacity-Gated Comparison predicts a systematic crossover: strong majority-consistent choices at loads 3-4, an intermediate transition near load 5, and strong highest-validity-cue-consistent choices at loads 6-8. Its post-strategy noise does not depend on tally margin, so the predicted load gradient appears in both parity series. The mirrored pairs balance response side and should exhibit the same crossover toward B rather than A. With 12 unique pairs repeated eight times, the experiment contains 96 trials.

[3] This design tests a sharp prediction produced by exact tally ties at and above the proposed capacity boundary. Every pair has an equal number of cues favoring A and B, so Tallying predicts P(A)=0.50 exactly on every trial, independently of beta and epsilon. Discriminating-cue load is varied across 4, 6, and 8 cues. Within each load there are two structurally distinct exemplars, followed by exact A/B reversals. In each unreversed pair, the highest-validity discriminating cue favors A; in each reversal it favors B. Capacity-Gated Comparison therefore predicts near-chance choice at load 4 when integration is usually recruited, but a marked lexicographic preference at loads 6 and 8 as overload recruits the highest-validity cue. Given its parameter ranges, the model predicts roughly P(lex-consistent choice)=0.72-0.89 at load 6 and 0.80-0.89 at load 8, whereas Tallying remains exactly at 0.50. Thus the critical signature is a load-dependent, response-side-symmetric departure from chance toward the highest-validity discriminating cue. Twelve unique pairs repeated eight times yield 96 trials.

[4] This design exploits equal-validity ties to create a qualitative and quantitative dissociation between stable lexicographic tie-breaking and rank integration. Experts 1-3 share the highest validity. All seven cues discriminate, every unreversed pair has the same 4-versus-3 majority favoring A, and Expert 1 always favors A. Thus both branches of Capacity-Gated Comparison—equal-weight tallying and the stable validity-ranked lexicographic search—select A on every unreversed pair. It therefore predicts a flat P(A)=(1-epsilon)*sigmoid(beta)+epsilon/2, approximately 0.80-0.89, regardless of how evidence is distributed among the other tied top experts. Precision-Annealed Rank Integration instead gives Experts 1-3 exactly equal and overwhelmingly dominant attention because they share rank zero, while the next active expert has rank three. The first two pairs have all three top experts favoring A and should produce near-certain A choices; the next two have a 2-to-1 top-tier majority for A and should produce moderately strong A choices; the final two have Experts 2 and 3 opposing Expert 1, so their 2-to-1 top-tier majority favors B and should reverse choice despite the overall 4-to-3 majority for A. Across the stated parameter ranges, the competitor should produce an approximately descending profile from above 0.95, through roughly 0.77-0.87, to roughly 0.14-0.25 P(A), whereas the advocated theory remains flat near 0.80-0.89. Two tail arrangements within each top-tier composition test the competitor's smaller residual rank-weighting effects. The final six pairs are exact A/B reversals, balancing response side. Twelve unique pairs repeated eight times yield 96 trials.

[5] All trials have exactly five discriminating cues: Experts 1, 2, 4, 6, and 8 (validity ranks 0, 1, 3, 5, and 7). In every unreversed pair, three cues favor A, the highest-validity cue favors B, and exactly one additional cue favors B. That second B-favoring cue moves progressively from Expert 2 to Expert 4, Expert 6, and Expert 8. Prospective Capacity-Gated Comparison therefore predicts exactly the same A-choice probability for all four unreversed pairs: load, tally direction and margin, and the first validity-ranked discriminating cue are identical. Its integration branch selects A, its lexicographic branch selects B, and only the fixed load-5 gate determines their mixture. Precision-Annealed Rank Integration instead predicts a strictly increasing A-choice gradient as the second B cue moves down ranks 1, 3, 5, and 7, because the opposing evidence loses attention mass while all other properties remain fixed. At representative parameter midpoints, the approximate unreversed profile is P(A)=0.12, 0.31, 0.46, and 0.55, rather than the competitor's flat profile. Exact A/B reversals should produce the complementary decreasing profile while controlling response bias. The non-discriminating cues vary in their shared displayed values across exemplars but cannot affect either model. Eight unique pairs repeated 12 times yield 96 trials.

[6] This design isolates the competing theory's late high-load format transition while placing all trials above the advocated theory's capacity boundary. There are two exemplars each at discriminating-cue loads 6, 7, and 8, followed by exact A/B reversals. In every unreversed pair, the highest-validity active expert favors A, whereas the most recent active experts favor B. At loads 6 and 7 the equal-weight tally also favors B; at load 8 it is tied. Prospective Capacity-Gated Comparison should already be predominantly lexicographic: it may show some residual B-majority influence at load 6, but should strongly favor A by load 7 and then change very little from load 7 to load 8. Across its parameter ranges, its approximate unreversed prediction is P(A)=0.65-0.89 at load 6, 0.77-0.89 at load 7, and 0.80-0.89 at load 8. Load-Indexed Format Arbitration instead assigns substantial weight to recency-dominated sequential retrieval at loads 6-7, yielding B-directed evidence because the late diagnostic cues all favor B. Its narrow high-load transition then increasingly compresses the display to the uniquely highest-validity active cue, which favors A. It therefore predicts a substantially steeper increase in A choice from load 6 through load 8, with especially diagnostic disagreement at load 6 and a larger load-7-to-load-8 change than the advocated theory. The two exemplars at each load alter active positions while preserving this early-validity-versus-late-recency conflict. Exact reversals require complementary choice profiles and control response-side bias. Twelve unique pairs repeated eight times yield 96 trials.

[7] This design creates matched low-load confidence and recency dissociations while making Prospective Capacity-Gated Comparison exactly invariant. Experts 1 and 2 form a distinctive partially tied leading-validity tier. Each unreversed trial has either three or four discriminating cues, the highest-validity active cue favors A, and the equal-weight majority also favors A. Consequently, both branches of the competing theory select A on every unreversed trial, predicting the same P(A) throughout (approximately 0.80-0.89 over its parameter ranges), regardless of load, unanimity, or the losing cue's display position. Within each fixed active-cue set, Load-Indexed Format Arbitration instead predicts a graded profile. The unanimous condition has evidence +1 in every format and should produce the strongest A preference. Moving to a narrow A majority reduces normalized diffuse evidence to 1/3 at load 3 or 1/2 at load 4; moreover, the partially tied leading tier selectively attenuates low-format confidence at these loads. Among narrow-majority trials, placing the sole B-favoring cue late rather than at an intermediate position should further reduce A choice because sequential retrieval has a strong recency gradient. Thus the critical predicted ordering is unanimous > intermediate-position loser > latest-position loser, especially at load 4 where retrieval recruitment is greater. The competing theory predicts no such within-set ordering because tally direction and lexicographic direction are unchanged. Exact A/B reversals provide response-side balance. There are 24 unique pairs, yielding four repetitions each for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design targets the advocated theory's bounded coalition reinforcement and conflict-triggered reopening rather than another load transition. Every trial has eight diagnostic cues, a 5-to-3 majority, a uniquely highest-validity diagnostic cue, and no validity ties. In each unreversed trial, both the majority and Expert 1 favor A. The critical comparison changes only the arrangement of the three B cues. In the coherent condition they form the late adjacent block {Experts 5,6,7}, after four consecutive A cues have supported premature commitment. In the dispersed condition they occupy {Experts 4,6,8}, yielding an alternating tail with the same number and mean position of B cues but no adjacent B pair. Conflict-Reopening Attentional Race predicts that the late coherent block will be especially disruptive: unresolved adjacent dissent increases reopening leakage, and its first local continuation receives the bounded coalition impulse. It should therefore predict lower P(A) for the coherent block than for the dispersed arrangement. Load-Indexed Format Arbitration predicts the opposite ordering. Its low-load representation is identical because tally evidence is +0.25 in both conditions, and its high-load representation is identical because the uniquely best active expert favors A. In its remaining sequential-retrieval component, accessibility rises approximately exponentially with display position. Consequently, the dispersed set {4,6,8} has greater B retrieval weight than the equally centered block {5,6,7}, principally because it includes the latest diagnostic position; it therefore predicts lower P(A) for the dispersed condition. Thus the sign of the coherent-minus-dispersed choice contrast is diagnostic: negative under the advocated theory but positive under the competitor, with complementary signs after A/B reversal. Expert 9 is nondiagnostic and is shown once as 0/0 and once as 1/1, providing perceptually distinct replications that both theories should treat identically. Eight unique pairs repeated 12 times produce 96 trials.",
  "validities": [
    0.91,
    0.86,
    0.81,
    0.76,
    0.71,
    0.67,
    0.63,
    0.59,
    0.54
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 9359,
  "output_tokens": 4976
}
```
