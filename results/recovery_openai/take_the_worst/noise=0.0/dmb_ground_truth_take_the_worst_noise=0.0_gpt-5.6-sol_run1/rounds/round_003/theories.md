# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

**Rationale:** The prior fixed heuristics fail jointly because each assumes the same evidence-processing regime regardless of comparison load. The proposed gate makes the two regimes consequences of a common, experiment-invariant capacity constraint. Experiment 2 contains only three discriminating cues, placing it below the capacity range and producing predominantly tally-based choices; the resulting noisy majority adherence is expected to be near the observed positive value. Experiment 1 contains eight discriminating cues, placing it well above capacity and producing almost exclusively lexicographic choices. Because lexicographic adherence does not systematically change with tally congruence, its matched contrast remains near zero, as observed. The smooth gate avoids an experiment-specific switch and predicts a graded transition: with feature count, validity profile, and tally margin controlled, majority choice should dominate at roughly two to four diagnostic cues, mixed behavior should emerge near five cues, and top-validity-cue dominance should increase around six to eight cues. History is intentionally unused because the task provides no outcome feedback from which strategy utility or cue validity could be learned.

**Parameters:**
  - `validities`: `validities`
  - `capacity`: `[4.5, 5.5]`
  - `gate_slope`: `[2.0, 5.0]`
  - `beta`: `[1.55, 2.05]`
  - `epsilon`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

**Rationale:** This is a single localized edit to the accepted iteration-6 model. For whole-tier poolers only, high-format gain is multiplied by a modest subject-specific factor when the observable display has a partially tied leading tier and the discriminating load is at least six. The first-member subgroup, format-recruitment gates, recency profile, low-load attenuation, tie-rule prevalence, and response noise are unchanged. In Experiment 3, the pooled leading tier has graded two-to-one evidence in the broader majority direction; strengthening that evidence should increase high-load majority consistency and reduce the excessive crossover. The evidence remains the tier mean, so a weak majority is still less compelling than unanimity. The condition excludes all-equal Experiment 1 displays and strictly ordered Experiments 2, 4, and 6, while Experiment 5's stable classification split should remain intact because the edit reinforces poolers' existing direction rather than changing their tie rule.

**Parameters:**
  - `validities`: `validities`
  - `low_center`: `[4.3, 4.8]`
  - `high_center`: `[7.0, 7.35]`
  - `transition_width`: `[0.9, 1.3]`
  - `high_transition_width`: `[0.3, 0.5]`
  - `partial_tie_acceleration`: `[0.9, 1.4]`
  - `all_equal_delay`: `[0.9, 1.3]`
  - `recency_compression_coupling`: `[0.25, 0.45]`
  - `recency_strength`: `[4.8, 6.8]`
  - `validity_strength`: `[0.2, 0.65]`
  - `tie_rule`: `{0, 1}`
  - `partial_tie_low_attenuation`: `[0.35, 0.6]`
  - `low_evidence_gain`: `[0.6, 0.85]`
  - `high_evidence_gain`: `[0.4, 0.65]`
  - `partial_tier_pooling_gain`: `[1.45, 1.8]`
  - `beta`: `[4.8, 6.2]`
  - `epsilon`: `[0.0, 0.06]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Conflict-Reopening Attentional Race theory: diagnostic cues become available during a noisy, approximately display-ordered attentional race. Accessibility reflects communicated validity and recency, while sampled evidence accumulates toward a subject-specific stopping threshold. Reaching the threshold normally terminates comparison, especially when encountered cues agree. Peripheral detection of unresolved dissent can reopen comparison and transiently prioritize opposing evidence. Reopening after late coherent dissent causes bounded leakage of the provisional accumulator toward zero, undoing premature commitment without repeatedly multiplying opposing evidence. Spatially adjacent same-sign samples establish a coherent coalition, but each run receives only a finite one-shot reinforcement. Equal-validity treatment varies continuously across people: after a tier is opened, pooling-oriented subjects preferentially inspect its unresolved members, whereas first-member-oriented subjects can move on or stop early. Load has no strategy gate and matters only through sampling opportunities, conflict, coalition formation, stopping, and probabilistic reopening.

**Rationale:** This is a minimal edit to the accepted iteration-8 model. All accepted validity, race, stopping, conflict-amplification, and one-shot coalition equations are retained. The fixed 0.82 reopening discount is replaced by bounded accumulator leakage determined by active-sequence lateness and adjacency among unresolved dissenters. Leakage is substantial after late coherent conflict, weak after early or scattered conflict, and habituated after an earlier reopening. It therefore weakens premature commitment without paying another multiplicative evidence bonus to every opponent, targeting Experiments 1, 3, 6, and 8 while preserving validity-led behavior in scattered displays. The equal-tier implementation explicitly uses the already-defined effective_pooling quantity, avoiding the rejected candidate's undefined-variable error. A single accessibility term now prioritizes unresolved members of an opened tier for high-pooling subjects; low-pooling subjects remain first-member-like. This targeted correction should make the top-tier 2-to-1 conflict in Experiment 5 produce the required split population without affecting strictly ordered-validity experiments.

**Parameters:**
  - `validities`: `validities`
  - `validity_sensitivity`: `[1.4, 3.2]`
  - `validity_access_temperature`: `[0.9, 1.5]`
  - `validity_evidence_temperature`: `[0.7, 1.3]`
  - `validity_modulation`: `[0.04, 0.10]`
  - `recency_bias`: `[0.8, 2.2]`
  - `stopping_threshold`: `[1.8, 3.4]`
  - `conflict_trigger`: `[1.2, 3.2]`
  - `coalition_reinforcement`: `[0.55, 1.65]`
  - `tier_pooling`: `[0, 1]`
  - `sampling_noise`: `[0.65, 1.15]`
  - `decision_gain`: `[2.6, 5.2]`
  - `epsilon`: `[0, 0.06]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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
