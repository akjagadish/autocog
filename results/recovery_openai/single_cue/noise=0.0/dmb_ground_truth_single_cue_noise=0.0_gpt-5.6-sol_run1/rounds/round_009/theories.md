# Round 9 — Theories

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


### slot 2 — `pi_11` — KILLED ✗

**Description:** Reliability-Compressed Recurrent Evidence Field. Decision makers inspect discriminating experts in descending instructed-validity order and encode each attended rating as signed, compressed log-odds evidence. Evidence remains distributed across cue traces, but accessibility declines smoothly with ordinal distance from response, producing graded recency rather than categorical replacement. Choice depends on the signed sum of accessible traces. Simultaneously active support for both products generates divisive interference, with an additional bounded increase when the accessible recurrent state is genuinely near balance. Increasing evidence load raises readout noise, especially under conflict. Cue-attention uncertainty contributes additional encoding noise, while a small vigilance decrement allows only a modest decline for long unanimous displays. Thus terminal control emerges continuously from terminal reliability, the accessibility gradient, opposing reliability-weighted coalition strength, and total interference. The mechanism has no sequence-class rules, physical-gap effects, fixed capacity boundary, or cross-trial learning.

**Rationale:** This is a minimal edit to the accepted iteration-2 model. The reliability transform, retention gradient, attention, gain, load noise, vigilance, and lapse ranges are unchanged. The only mechanistic addition is a narrow, bounded amplification of the existing divisive normalization based on conflict balance after serial accessibility has been applied. Raising accessible-state balance to the sixth power confines the adjustment to genuinely near-balanced recurrent states; asymmetric states receive almost no additional penalty. This should reduce excessive polarization in balanced conflicts without repeating the rejected retrieval-neutral interference term, injecting large extra readout noise, or weakening terminal control when recency has already made the accessible state strongly asymmetric. Because the adjustment depends only on signed accessible mass, it does not encode coherence, switch count, physical gaps, singleton status, fixed windows, or trial history.

**Parameters:**
  - `validities`: `validities`
  - `validity_compression`: `[0.0, 0.10]`
  - `memory_retention`: `[0.38, 0.54]`
  - `cue_attention`: `[0.86, 0.99]`
  - `conflict_normalization`: `[0.15, 0.55]`
  - `near_balance_normalization`: `[1.5, 2.5]`
  - `load_noise`: `[0.03, 0.16]`
  - `vigilance_decline`: `[0.008, 0.025]`
  - `decision_gain`: `[3.5, 6.5]`
  - `lapse_rate`: `[0.04, 0.16]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recurrent evidence field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    compression = float(parameters["validity_compression"])
    retention = float(parameters["memory_retention"])
    attention = float(parameters["cue_attention"])
    conflict_normalization = float(parameters["conflict_normalization"])
    near_balance_normalization = float(parameters["near_balance_normalization"])
    load_noise = float(parameters["load_noise"])
    vigilance_decline = float(parameters["vigilance_decline"])
    decision_gain = float(parameters["decision_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is converted to normative log-odds and then
    # strongly compressed. Normalizing by the experiment's typical positive
    # log-odds preserves its reliability profile without allowing arbitrary
    # differences in scale to determine response gain.
    v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    positive = log_odds[log_odds > 0.0]
    reliability_scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0
    relative_reliability = np.clip(
        log_odds / reliability_scale, 1e-4, 20.0
    )
    cue_strength = np.power(relative_reliability, compression)
    cue_strength = np.clip(cue_strength, 0.05, 8.0)

    # Every reason retains a separate trace. Accessibility follows a smooth
    # exponential serial-position gradient: no reason is exactly overwritten,
    # and no capacity or sequence-length boundary is imposed.
    signed_traces = []
    unsigned_traces = []
    omission_variance = 0.0
    for serial_position, j in enumerate(discriminating):
        lag = (m - 1) - serial_position
        accessibility = retention ** float(lag)
        strength = float(cue_strength[j])
        direction = float(np.sign(diff[j]))

        expected_trace = attention * accessibility * strength
        signed_traces.append(direction * expected_trace)
        unsigned_traces.append(expected_trace)

        # Bernoulli attention uncertainty is represented at readout as the
        # variance of a potentially omitted trace.
        omission_variance += (
            attention * (1.0 - attention)
            * (accessibility * strength) ** 2
        )

    signed_traces = np.asarray(signed_traces, dtype=float)
    unsigned_traces = np.asarray(unsigned_traces, dtype=float)
    integrated_evidence = float(np.sum(signed_traces))

    positive_mass = float(np.sum(signed_traces[signed_traces > 0.0]))
    negative_mass = float(-np.sum(signed_traces[signed_traces < 0.0]))
    total_mass = positive_mass + negative_mass

    # Simultaneous opposing coalitions produce divisive interference. The
    # smooth balance index is zero for unanimous evidence and approaches one
    # when accessible support is evenly divided.
    if total_mass > 0.0 and np.isfinite(total_mass):
        conflict_balance = (
            4.0 * positive_mass * negative_mass
            / max(total_mass * total_mass, 1e-12)
        )
        conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))
        opposing_mass = 2.0 * min(positive_mass, negative_mass)
        interference_load = opposing_mass / (1.0 + total_mass)
    else:
        conflict_balance = 0.0
        interference_load = 0.0

    # Only an already accessible, genuinely near-balanced state receives an
    # extra bounded normalization. The sixth power leaves asymmetric states
    # essentially unchanged and introduces no sequence or stimulus class.
    near_balance_multiplier = (
        1.0 + near_balance_normalization * conflict_balance ** 6
    )
    conflict_divisor = 1.0 + (
        conflict_normalization
        * interference_load
        * np.sqrt(max(1.0, float(m - 1)))
        * near_balance_multiplier
    )

    # Load mainly increases readout uncertainty under conflict. A small
    # residual component remains for coherent displays, avoiding perfect
    # performance while not producing excessive unanimous-evidence fatigue.
    load_component = np.log1p(max(0.0, float(m - 1)))
    load_multiplier = 0.12 + 0.88 * conflict_balance
    readout_sd = np.sqrt(
        1.0
        + omission_variance
        + load_noise * load_component * load_multiplier
    )

    # A weak, continuous vigilance decline is the only load penalty that also
    # applies fully to unanimous evidence.
    vigilance = np.exp(-vigilance_decline * max(0.0, float(m - 2)))

    decision_variable = (
        decision_gain
        * vigilance
        * integrated_evidence
        / max(conflict_divisor * readout_sd, 1e-12)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
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

### `pi_12` → slot 2 (via `new_theory`)

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
