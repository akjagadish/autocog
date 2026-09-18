# Round 10 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Rationale:** This is a minimal extension of the accepted recency-overwrite model. The cue scan and overwrite equations are unchanged. The only mechanistic addition is a discriminating-cue-count fatigue gate that begins after the second reason, together with a broader and slightly lower lapse range. Two-reason trials therefore retain the strong overwrite needed for Experiment 2, whereas long reason sequences are pulled modestly toward uncertainty, reducing the excessive Experiment-1 contrast. Lower average lapse simultaneously strengthens anti-TTB behavior in Experiment 2. Broad subject-level ranges for fatigue and lapse should also increase the previously underestimated between-subject variability. The gate is task-invariant and depends only on processing load, not experiment identity or stimulus-specific switches.

**Parameters:**
  - `validities`: `validities`
  - `cue_attention`: `[0.95, 1.0]`
  - `overwrite_rate`: `[0.95, 1.0]`
  - `update_fatigue`: `[0.06, 0.17]`
  - `lapse_rate`: `[0.15, 0.41]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
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


### slot 2 — `pi_12` — KILLED ✗

**Description:** Surprise-Bound Coalition Relay. During left-to-right inspection, discriminating cues are compressed into directional coalitions: consecutive cues favoring the same option strengthen one bounded commitment rather than leaving independently summable traces. An opposing coalition initiates a graded contest with the incumbent. Switching is probabilistic and depends jointly on incumbent strength, challenger coherence, relative instructed reliability, terminal singleton salience, and the challenger's absolute display accessibility. Instructed validity therefore modulates encoding and switching thresholds but is not converted into signed additive evidence. Absolute display position is represented separately from ordinal position in the discriminating sequence. The final response averages over unresolved commitment trajectories and is confidence-gated: balanced evidence, ambiguous switches, and fragmented sequences approach chance, whereas a decisive coalition switch can be expressed confidently. Load impairs precision primarily when conflict remains unresolved; unanimous evidence receives only a small load-dependent decline. At a coalition boundary, a modest, regularized drop from the incumbent's exit reliability to the challenger's entry reliability contributes relative surprise, preserving local boundary information without allowing validity differences to dominate the contest.

**Rationale:** This is a narrowly isolated interpolation from the accepted candidate. All attention, coalition integration, terminal salience, erosion, confidence, load, and subject-level parameter ranges are unchanged. The only mechanistic edit is to retain each coalition's entry and exit reliability and add a modest boundary-surprise term based on the actual incumbent-exit/challenger-entry discontinuity. The term uses weight 0.42 and a 0.50 regularizer, substantially weaker than the rejected full-strength boundary implementation. It should preserve part of that implementation's gains in Experiments 3, 4, 11–13, and 20 while reducing its collateral switching in Experiments 2, 5–8 and avoiding an excessive Experiment 20 gradient. Because only downward boundary discontinuities add surprise, ordinary validity differences continue to operate mainly through the accepted run-average threshold and attention mechanisms rather than becoming signed additive evidence.

**Parameters:**
  - `validities`: `validities`
  - `base_attention`: `[0.82, 0.95]`
  - `validity_use`: `[0.22, 0.52]`
  - `position_accessibility`: `[0.28, 0.58]`
  - `coalition_capacity`: `[1.75, 2.45]`
  - `coherence_gain`: `[0.10, 0.30]`
  - `switch_threshold`: `[0.62, 0.88]`
  - `switch_precision`: `[3.8, 6.2]`
  - `relative_surprise`: `[0.45, 0.90]`
  - `terminal_salience`: `[0.42, 0.78]`
  - `opposition_erosion`: `[0.20, 0.42]`
  - `response_gain`: `[2.5, 4.4]`
  - `conflict_collapse`: `[0.48, 0.76]`
  - `load_noise`: `[0.025, 0.085]`
  - `unanimous_decay`: `[0.010, 0.032]`
  - `lapse_rate`: `[0.04, 0.14]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Surprise-Bound Coalition Relay expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    base_attention = float(parameters["base_attention"])
    validity_use = float(parameters["validity_use"])
    position_accessibility = float(parameters["position_accessibility"])
    coalition_capacity = float(parameters["coalition_capacity"])
    coherence_gain = float(parameters["coherence_gain"])
    switch_threshold = float(parameters["switch_threshold"])
    switch_precision = float(parameters["switch_precision"])
    relative_surprise = float(parameters["relative_surprise"])
    terminal_salience = float(parameters["terminal_salience"])
    opposition_erosion = float(parameters["opposition_erosion"])
    response_gain = float(parameters["response_gain"])
    conflict_collapse = float(parameters["conflict_collapse"])
    load_noise = float(parameters["load_noise"])
    unanimous_decay = float(parameters["unanimous_decay"])
    lapse = float(parameters["lapse_rate"])

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    m = int(discriminating.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability affects attentional accessibility and switch thresholds,
    # never the sign of an additive evidence trace.
    reliability = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    base_logit = np.log(
        np.clip(base_attention, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - base_attention, 1e-5, 1.0)
    )

    ordinal_directions = np.sign(diff[discriminating]).astype(float)
    cue_access = []
    for j in discriminating:
        absolute_position = float(j) / float(max(1, n_features - 1))
        attention = sigmoid(
            base_logit
            + validity_use * (float(reliability[j]) - 0.5)
            + position_accessibility * (absolute_position - 0.5)
        )
        # A modest independent accessibility advantage for later display
        # locations is retained after attentional gating.
        access = attention * (
            1.0 + position_accessibility * 0.45 * absolute_position
        )
        cue_access.append(float(access))
    cue_access = np.asarray(cue_access, dtype=float)

    # Form ordinal coalitions. Physical locations remain attached to each
    # coalition, so discriminating ordinal position and display position are
    # not conflated.
    runs = []
    start = 0
    while start < m:
        direction = float(ordinal_directions[start])
        end = start + 1
        while end < m and ordinal_directions[end] == direction:
            end += 1

        loc = slice(start, end)
        run_indices = discriminating[loc]
        raw_access = float(np.sum(cue_access[loc]))
        run_length = int(end - start)

        # Bounded within-coalition integration. Coherence increases strength
        # sublinearly and cannot create an unbounded linear sum.
        bounded = coalition_capacity * (
            1.0 - np.exp(-raw_access / max(coalition_capacity, 1e-8))
        )
        coherence = 1.0 + coherence_gain * (
            1.0 - 1.0 / float(run_length)
        )
        strength = min(coalition_capacity, bounded * coherence)

        weights = cue_access[loc]
        if float(np.sum(weights)) > 0.0:
            mean_reliability = float(
                np.average(reliability[run_indices], weights=weights)
            )
        else:
            mean_reliability = float(np.mean(reliability[run_indices]))

        final_physical_position = float(run_indices[-1]) / float(
            max(1, n_features - 1)
        )
        runs.append({
            "direction": direction,
            "strength": float(strength),
            "length": run_length,
            "reliability": mean_reliability,
            "entry_reliability": float(reliability[run_indices[0]]),
            "exit_reliability": float(reliability[run_indices[-1]]),
            "final_position": final_physical_position,
            "ends_sequence": bool(end == m)
        })
        start = end

    # Each entry is a possible commitment trajectory. Branching implements a
    # genuinely graded switch rather than deterministic overwrite.
    states = [{
        "weight": 1.0,
        "direction": runs[0]["direction"],
        "strength": runs[0]["strength"],
        "reliability": runs[0]["reliability"],
        "exit_reliability": runs[0]["exit_reliability"]
    }]
    switch_ambiguities = []

    for run in runs[1:]:
        next_states = []
        weighted_ambiguity = 0.0
        total_state_weight = 0.0

        for incumbent in states:
            w = float(incumbent["weight"])
            incumbent_strength = float(incumbent["strength"])
            challenger_strength = float(run["strength"])

            reliability_difference = (
                float(incumbent["reliability"])
                - float(run["reliability"])
            )
            threshold_multiplier = 1.0 + validity_use * reliability_difference
            threshold_multiplier = float(np.clip(
                threshold_multiplier, 0.55, 1.55
            ))

            relative_advantage = (
                challenger_strength - incumbent_strength
            ) / max(challenger_strength + incumbent_strength, 1e-8)

            is_terminal_singleton = float(
                run["ends_sequence"] and run["length"] == 1
            )
            singleton_access = (
                is_terminal_singleton
                * terminal_salience
                * (0.55 + 0.45 * float(run["final_position"]))
            )

            # Switching is initiated at the actual coalition boundary. A
            # downward reliability discontinuity can be surprising, but its
            # influence is bounded and strongly regularized. The coefficient
            # is deliberately intermediate rather than the rejected 0.75
            # boundary weight.
            boundary_drop = max(
                0.0,
                float(incumbent["exit_reliability"])
                - float(run["entry_reliability"])
            )
            boundary_surprise = (
                0.42 * boundary_drop / (0.50 + boundary_drop)
            )

            contest = (
                challenger_strength
                - switch_threshold
                * threshold_multiplier
                * incumbent_strength
                + relative_surprise * relative_advantage
                + singleton_access
                + boundary_surprise
            )
            q_switch = sigmoid(switch_precision * contest)

            weighted_ambiguity += w * 4.0 * q_switch * (1.0 - q_switch)
            total_state_weight += w

            # If the challenger is resisted, conflict still erodes incumbent
            # commitment. If it wins, its bounded coalition becomes incumbent.
            retained_strength = max(
                0.08,
                incumbent_strength
                - opposition_erosion * challenger_strength
            )
            switched_strength = min(
                coalition_capacity,
                challenger_strength
                + 0.10 * q_switch * incumbent_strength
            )

            if q_switch < 1.0:
                next_states.append({
                    "weight": w * (1.0 - q_switch),
                    "direction": float(incumbent["direction"]),
                    "strength": float(retained_strength),
                    "reliability": float(incumbent["reliability"]),
                    "exit_reliability": float(incumbent["exit_reliability"])
                })
            if q_switch > 0.0:
                next_states.append({
                    "weight": w * q_switch,
                    "direction": float(run["direction"]),
                    "strength": float(switched_strength),
                    "reliability": float(run["reliability"]),
                    "exit_reliability": float(run["exit_reliability"])
                })

        states = next_states
        if total_state_weight > 0.0:
            switch_ambiguities.append(
                weighted_ambiguity / total_state_weight
            )

    # Accessible support is used only to diagnose unresolved conflict. It is
    # not read out as a linear evidence total.
    positive_mass = float(np.sum(
        cue_access[ordinal_directions > 0.0]
    ))
    negative_mass = float(np.sum(
        cue_access[ordinal_directions < 0.0]
    ))
    total_mass = positive_mass + negative_mass
    if positive_mass <= 0.0 or negative_mass <= 0.0 or total_mass <= 0.0:
        balance = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))

    ambiguity = (
        float(np.mean(switch_ambiguities))
        if switch_ambiguities else 0.0
    )
    fragmentation = max(0.0, float(len(runs) - 2)) / float(max(1, m - 1))
    unresolved = balance * (
        0.20 + 0.55 * ambiguity + 0.45 * fragmentation
    )
    unresolved = float(np.clip(unresolved, 0.0, 1.0))

    confidence = max(0.08, 1.0 - conflict_collapse * unresolved)
    overload = max(0.0, float(m) - 4.0)
    conflict_divisor = 1.0 + load_noise * overload * unresolved * unresolved

    # Unanimous evidence avoids conflict noise but shows a slight empirical
    # decline at high loads rather than ever-increasing certainty.
    if balance == 0.0:
        unanimous_divisor = 1.0 + unanimous_decay * max(0.0, float(m) - 2.0)
    else:
        unanimous_divisor = 1.0

    trajectory_weight = float(np.sum([s["weight"] for s in states]))
    if trajectory_weight <= 0.0 or not np.isfinite(trajectory_weight):
        return np.array([0.5, 0.5], dtype=np.float64)

    p_a_core = 0.0
    for commitment in states:
        decision_variable = (
            response_gain
            * confidence
            * float(commitment["direction"])
            * float(commitment["strength"])
            / (conflict_divisor * unanimous_divisor)
        )
        p_a_state = sigmoid(decision_variable)
        p_a_core += float(commitment["weight"]) * p_a_state
    p_a_core /= trajectory_weight
    p_a_core = float(np.clip(p_a_core, 0.0, 1.0))

    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_13` → slot 2 (via `new_theory`)

**Description:** Spatially Indexed Reliability-Gated Revision. People inspect product features in their displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment. Revision is leaky, so recent cues ordinarily have greater influence, while repeated same-direction cues build a saturating, reliability-sensitive stability trace rather than an all-or-none coalition. Congruent trace consolidation is selectively nonlinear in relative reliability: repeated reliable cues stabilize an incumbent more strongly than equally numerous weak cues, while ordinary cue revision remains unchanged. This trace remains associated with an established incumbent during a reversal and is eroded gradually by cumulative challenger evidence; consequently, every cue in an opposing run remains partially gated until the challenger has continuously displaced the incumbent. Instructed validity jointly affects cue accessibility, revision magnitude, and the rate at which evidence builds or erodes stability. A weak cue that locally opposes a more reliable preceding cue evokes surprise-driven orienting, temporarily increasing its accessibility and capacity to revise the incumbent commitment. Accessibility also rises gradually toward the right side of the display, although the positional gradient is compressed on very long displays, with a small additional benefit for an isolated discriminating cue at the physical right edge. Increasing display load gradually reduces response precision, including a small decline for long unanimous sequences. Choices are generated from this single evolving commitment state, with no trial-level mixture of strategies.

**Rationale:** This is a minimal edit to the accepted iteration-6 recurrence. All parameter ranges, ordinary cue-revision equations, persistent trace gating, surprise orienting, spatial accessibility, long-display retention, load noise, and response mapping are unchanged. The only mechanistic change is a bounded nonlinear reliability multiplier applied when a cue consolidates an already direction-congruent stability trace. Relatively reliable repeated incumbent cues can therefore form a stronger trace in Experiments 1, 6, and 8, while weak repeated cues consolidate less. The first cue receives no special protection, and opposing cues continue to erode the trace under the accepted smooth rule; therefore the edit does not add the rejected reliability-squared gate release or multiplicative gap release and should preserve rapid terminal revision from weak incumbents in Experiments 4, 10, 11, and 13. The multiplier is deliberately modest and bounded between approximately 0.78 and 1.22, preventing rigid coalition formation.

**Parameters:**
  - `validities`: `validities`
  - `spatial_attention`: `[0.55, 1.15]`
  - `validity_gain`: `[0.12, 0.38]`
  - `leak_rate`: `[0.48, 0.67]`
  - `overwrite_strength`: `[1.75, 2.45]`
  - `surprise_orienting`: `[1.25, 2.30]`
  - `commitment_saturation`: `[0.48, 0.88]`
  - `load_noise`: `[0.020, 0.055]`
  - `lapse_rate`: `[0.04, 0.13]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Spatially Indexed Reliability-Gated Revision expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    spatial_attention = float(parameters["spatial_attention"])
    validity_gain = float(parameters["validity_gain"])
    leak_rate = float(parameters["leak_rate"])
    overwrite_strength = float(parameters["overwrite_strength"])
    surprise_orienting = float(parameters["surprise_orienting"])
    commitment_saturation = float(parameters["commitment_saturation"])
    load_noise = float(parameters["load_noise"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is represented as reliability, but it does not
    # determine scan order. Standardization makes the same gain parameter
    # usable across experiments with different validity ranges.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        standardized_reliability = np.zeros(n_features, dtype=float)
    else:
        standardized_reliability = np.clip(
            (reliability - rel_mean) / rel_sd, -2.5, 2.5
        )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # commitment is the bounded current decision state. stability_trace is a
    # continuous signed record of commitment-congruent support. Unlike a run
    # label, it is not reassigned after the first opposing update: cumulative
    # challenger evidence must gradually erode and reverse it.
    commitment = 0.0
    stability_trace = 0.0
    previous_direction = 0.0
    previous_reliability = None
    previous_index = None

    # Compress only the positional component of accessibility on unusually
    # long displays. This prevents normalized scan position from creating an
    # unrealistically large endpoint effect while retaining a rightward bias.
    position_scale = float(min(
        1.0, np.sqrt(8.0 / float(max(8, n_features)))
    ))

    # On unusually long displays, intervening locations increase retention
    # slightly. Ordinary displays retain the original recurrence unchanged.
    long_display_fraction = float(np.clip(
        (float(n_features) - 16.0) / 4.0, 0.0, 1.0
    ))
    effective_leak_rate = float(np.clip(
        leak_rate + 0.10 * long_display_fraction, 0.0, 0.90
    ))

    for j in discriminating:
        j = int(j)
        direction = float(np.sign(diff[j]))
        position = float(j) / float(max(1, n_features - 1))

        # Ordinary accessibility follows a smooth left-to-right gradient.
        # Validity changes accessibility without reordering the cues.
        access_logit = (
            1.15
            + spatial_attention * position_scale * (2.0 * position - 1.0)
            + validity_gain * standardized_reliability[j]
        )

        # A physically isolated discriminating item at the right boundary is
        # modestly foregrounded. The conjunction prevents a generic terminal
        # bonus from guaranteeing reversal on every display.
        if previous_index is None:
            gap_fraction = 0.0
        else:
            gap_fraction = float(max(0, j - previous_index - 1)) / float(
                max(1, n_features - 1)
            )
        right_edge = float(j == n_features - 1)
        isolation_bonus = (
            0.32
            * spatial_attention
            * position_scale
            * right_edge
            * np.sqrt(gap_fraction)
        )
        access_logit += isolation_bonus
        accessibility = sigmoid(access_logit)

        # Reliability also scales encoded revision strength, but only
        # compressively, so weak experts remain psychologically available.
        reliability_strength = float(np.exp(
            validity_gain * 0.55 * standardized_reliability[j]
        ))
        reliability_strength = float(np.clip(reliability_strength, 0.45, 2.20))

        # Surprise is specifically a local reliability drop at an opposing
        # boundary. It boosts attention to a weak challenger rather than
        # installing a separate strategy or reversing the validity ordering.
        orienting = 1.0
        if previous_reliability is not None and previous_direction != direction:
            raw_drop = max(0.0, float(previous_reliability - validities[j]))
            incumbent_alignment = abs(commitment)
            orienting += (
                surprise_orienting
                * raw_drop
                * (0.45 + 0.55 * incumbent_alignment)
            )

        # Leakage creates recency. Every cue remains gated while its direction
        # opposes the persistent stability trace. The gate depends on the trace
        # itself, not instantaneous commitment, so it does not disappear merely
        # because an initial challenger has moved commitment toward zero.
        retained = float(np.clip(
            effective_leak_rate * commitment, -0.999999, 0.999999
        ))
        opposing_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) != direction
        )
        if opposing_stability:
            trace_gate_mass = min(2.5, abs(stability_trace) ** 0.75)
            stability_gate = 1.0 + (
                commitment_saturation * trace_gate_mass
            )
        else:
            stability_gate = 1.0

        revision = (
            overwrite_strength
            * accessibility
            * reliability_strength
            * orienting
            / stability_gate
        )

        # Updating in bounded-logit coordinates gives gradual saturation for
        # repeated agreement while preserving smooth directional revision.
        bounded_coordinate = np.arctanh(retained)
        commitment = float(np.tanh(
            bounded_coordinate + direction * revision
        ))

        # The signed stability trace persists across a reversal. Congruent
        # evidence consolidates it efficiently, whereas opposing evidence
        # erodes it continuously and reliability-dependently before eventually
        # installing a new incumbent. No switch threshold or run state is used.
        trace_alignment = direction * float(np.tanh(stability_trace))
        assimilation = (
            0.50 + 0.50 * sigmoid(2.5 * trace_alignment)
        )

        # Only commitment-congruent consolidation receives this bounded
        # nonlinear reliability modulation. Reliable repeated cues strengthen
        # an established trace, whereas weak repetition consolidates less.
        # Opposing cues retain the accepted linear erosion rule.
        congruent_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) == direction
        )
        if congruent_stability:
            consolidation_gain = (
                0.78
                + 0.44 * sigmoid(2.2 * standardized_reliability[j])
            )
        else:
            consolidation_gain = 1.0

        trace_retention = float(np.sqrt(effective_leak_rate))
        stability_trace = (
            trace_retention * stability_trace
            + direction
            * accessibility
            * reliability_strength
            * assimilation
            * consolidation_gain
        )
        stability_trace = float(np.clip(stability_trace, -6.0, 6.0))

        previous_direction = direction
        previous_reliability = float(validities[j])
        previous_index = j

    m = int(discriminating.size)

    # More reasons and more displayed locations introduce gradual precision
    # loss. There is no fixed fatigue boundary and no conflict-only collapse.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = 3.35 / (
        1.0
        + load_noise * display_load
        + 0.55 * load_noise * evidence_load
    )
    precision = float(np.clip(precision, 0.25, 6.0))

    p_a_core = sigmoid(precision * commitment)
    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
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
