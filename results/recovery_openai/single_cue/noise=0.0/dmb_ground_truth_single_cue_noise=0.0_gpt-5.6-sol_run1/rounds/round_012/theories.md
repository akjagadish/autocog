# Round 12 — Theories

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


### slot 2 — `pi_14` — KILLED ✗

**Description:** Parallel Contrastive Coalition Field with Novelty-Protected Competition. All discriminating cues are encoded concurrently into a capacity-limited comparison field rather than inspected in a mandatory validity or spatial sequence. Cue activation combines instructed reliability, U-shaped boundary distinctiveness, an estimable spatial tilt, and local contrast. Same-option activations pool into saturating coalitions before divisive competition. A coalition containing a sharply contrastive challenger receives a small, bounded release from cross-coalition inhibition after saturation, allowing prediction-error signals to survive competition without letting raw cue activation bypass capacity limits. Choice follows the normalized coalition difference with load-dependent precision loss and a task-invariant lapse. The model has no trial-history learning, strategy mixture, fixed scan order, or sequential commitment state.

**Rationale:** This is a narrowly bounded edit to the accepted iteration-5 model. Whole-coalition exponential saturation, pooling-before-competition, spatial weighting, reliability normalization, pre-saturation novelty, load loss, and lapse are unchanged. The only mechanistic addition stores each cue's already-computed novelty signal and uses the largest signal within each option to provide a small post-saturation release from that coalition's divisive denominator. The high threshold and narrow 4–10% release range make the pathway nearly inactive for ordinary boundaries, including smooth terminal opposition, while preserving sharply contrastive challengers during competition. Unlike the rejected anchor edits, it neither adds raw activation nor allows an exponentiated cue maximum to bypass saturation; unlike the rejected novelty budget, it does not redistribute salience among cues. This targets below-chance highest-validity adherence and validity-dependent challenger effects while minimizing disruption to the accepted fits in Experiments 5, 22, and 24.

**Parameters:**
  - `validities`: `validities`
  - `reliability_gain`: `[0.25, 0.50]`
  - `boundary_gain`: `[0.25, 0.55]`
  - `spatial_tilt`: `[-0.90, -0.55]`
  - `coalition_saturation`: `[0.55, 0.95]`
  - `novelty_gain`: `[1.50, 2.40]`
  - `novelty_release`: `[0.04, 0.10]`
  - `protection_threshold`: `[1.10, 1.30]`
  - `capacity_loss`: `[0.08, 0.18]`
  - `competition_temperature`: `[1.05, 1.35]`
  - `choice_precision`: `[3.00, 4.30]`
  - `lapse_rate`: `[0.05, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Parallel Contrastive Coalition Field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    reliability_gain = float(parameters["reliability_gain"])
    boundary_gain = float(parameters["boundary_gain"])
    spatial_tilt = float(parameters["spatial_tilt"])
    coalition_saturation = float(parameters["coalition_saturation"])
    novelty_gain = float(parameters["novelty_gain"])
    novelty_release = float(parameters["novelty_release"])
    protection_threshold = float(parameters["protection_threshold"])
    capacity_loss = float(parameters["capacity_loss"])
    competition_temperature = float(parameters["competition_temperature"])
    choice_precision = float(parameters["choice_precision"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[discriminating]).astype(float)
    m = int(discriminating.size)

    # Reliability is represented on the normative log-odds scale. Centering
    # preserves within-display reliability contrasts while making the shared
    # gain portable across experiments with different validity schedules.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        rel_z = np.zeros(n_features, dtype=float)
    else:
        rel_z = np.clip((reliability - rel_mean) / rel_sd, -2.5, 2.5)

    raw_activation = np.zeros(m, dtype=float)
    novelty_signals = np.zeros(m, dtype=float)

    # These activations are computed from the complete display. References to
    # left and right neighbors describe simultaneous spatial relations, not an
    # update order or a serial scan.
    for k, j_cell in enumerate(discriminating):
        j = int(j_cell)
        direction = float(directions[k])
        position = float(j) / float(max(1, n_features - 1))
        signed_position = 2.0 * position - 1.0

        # Both endpoints can be distinctive. The separately estimated tilt
        # permits endpoint asymmetry without assuming universal right recency.
        boundary = signed_position * signed_position
        spatial_log_activation = (
            boundary_gain * boundary + spatial_tilt * signed_position
        )

        surprise = 0.0
        symmetric_contrast = 0.0

        if k > 0:
            left_j = int(discriminating[k - 1])
            left_opposes = float(directions[k - 1] != direction)
            if left_opposes > 0.0:
                # A reliability drop at an opposing spatial boundary is a
                # prediction-error signal, not evidence that cues were scanned.
                drop = max(0.0, float(validities[left_j] - validities[j]))
                surprise += min(1.25, drop / 0.40)
                symmetric_contrast += min(
                    1.0, abs(float(validities[left_j] - validities[j])) / 0.40
                )

        if k + 1 < m:
            right_j = int(discriminating[k + 1])
            right_opposes = float(directions[k + 1] != direction)
            if right_opposes > 0.0:
                symmetric_contrast += min(
                    1.0, abs(float(validities[right_j] - validities[j])) / 0.40
                )

        # Most contrast is a sharp-drop novelty response. A smaller symmetric
        # term captures ordinary local pop-out at either side of a boundary.
        novelty_signal = surprise + 0.18 * symmetric_contrast
        novelty_signals[k] = novelty_signal
        log_activation = (
            reliability_gain * rel_z[j]
            + spatial_log_activation
            + novelty_gain * novelty_signal
        )
        raw_activation[k] = float(np.exp(np.clip(log_activation, -8.0, 8.0)))

    # Concurrent same-option cues pool before competition. Thus they exhibit
    # diminishing returns but do not consume capacity by suppressing one
    # another at the cue level.
    support_a = float(np.sum(raw_activation[directions > 0.0]))
    support_b = float(np.sum(raw_activation[directions < 0.0]))

    sat = max(coalition_saturation, 1e-8)
    coalition_a = sat * (1.0 - np.exp(-support_a / sat))
    coalition_b = sat * (1.0 - np.exp(-support_b / sat))

    # Sharp contrast receives only a bounded post-saturation release from
    # cross-coalition inhibition. This preserves whole-coalition saturation:
    # novelty cannot add activation or pass an unsaturated cue maximum into
    # competition. Smooth, ordinary boundaries remain effectively unprotected.
    novelty_a = float(np.max(novelty_signals[directions > 0.0])) if np.any(directions > 0.0) else 0.0
    novelty_b = float(np.max(novelty_signals[directions < 0.0])) if np.any(directions < 0.0) else 0.0
    protection_a = 1.0 / (
        1.0 + np.exp(-np.clip((novelty_a - protection_threshold) / 0.12, -60.0, 60.0))
    )
    protection_b = 1.0 / (
        1.0 + np.exp(-np.clip((novelty_b - protection_threshold) / 0.12, -60.0, 60.0))
    )
    release_a = float(np.clip(1.0 - novelty_release * protection_a, 0.80, 1.0))
    release_b = float(np.clip(1.0 - novelty_release * protection_b, 0.80, 1.0))

    # Capacity competition occurs between option-level coalitions. The mild
    # temperature sharpens genuine coalition differences after repeated cues
    # have already been compressed by saturation.
    competitive_a = coalition_a / ((1.0 + capacity_loss * coalition_b) * release_a)
    competitive_b = coalition_b / ((1.0 + capacity_loss * coalition_a) * release_b)
    competitive_a = float(np.power(max(competitive_a, 0.0), competition_temperature))
    competitive_b = float(np.power(max(competitive_b, 0.0), competition_temperature))

    # Divisive comparison expresses evidence relative to the total activated
    # field, preventing large displays from creating unbounded decision values.
    coalition_total = competitive_a + competitive_b
    if coalition_total <= 1e-12 or not np.isfinite(coalition_total):
        normalized_difference = 0.0
    else:
        normalized_difference = (competitive_a - competitive_b) / coalition_total

    # Display and evidence load modestly lower readout precision. Saturating
    # pooling and this precision loss can produce a small decline even for long
    # unanimous displays, without a discrete fatigue threshold.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = choice_precision / (
        1.0
        + 0.55 * capacity_loss * display_load
        + 0.35 * capacity_loss * evidence_load
    )
    precision = float(np.clip(precision, 0.20, 8.0))

    logit_a = float(np.clip(precision * normalized_difference, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)

    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
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

### `pi_15` → slot 2 (via `new_theory`)

**Description:** Priority-Latency Coalition Accumulation. Before deliberation, the display is encoded in parallel into a priority-latency map. Absolute instructed reliability gives diagnostically strong cues early access, a stable but heterogeneous directional scan tendency delays cues along the preferred scan path, scan-congruent endpoints attract selective reinspection, and an opposing cue following a sharp absolute validity drop receives surprise priority. Priority is sampled with modest trial-level noise rather than imposing a deterministic order. Evidence then enters a bounded leaky accumulator, so later or refreshed evidence is more available at choice. Updates remain graded: reliability affects update magnitude only weakly and compressively, local surprise increases revision, capacity load attenuates late encoding, and a coherent incumbent coalition continuously gates opposition. Coalition resistance is reduced for spatially isolated challengers, allowing isolated terminal reasons to reverse a coalition without making all terminal cues dominant. Subject-level heterogeneity in validity sensitivity, scan direction, endpoint salience, contrast gain, retention, anchoring, capacity, priority noise, and lapse produces nonuniform recency, crossover, and endpoint effects without changing mechanisms across experiments.

**Rationale:** This is a single, minimal calibration edit to the accepted iteration-2 model. The absolute log-odds transformation and its strong role in access latency are preserved because they generated the accepted reliability crossovers in Experiments 23 and 25. Only the direct reliability-dependent update multiplier changes: its exponential slope is reduced from 0.11 to 0.035 and its clipping interval is narrowed from [0.82, 1.42] to [0.94, 1.14]. Thus instructed validity still determines which evidence tends to arrive early, but highly valid evidence no longer also receives a large second advantage in accumulator magnitude. Later weak cues should consequently reverse more often, reducing excessive early/high-validity control in Experiments 1, 2, 6, and 9 and increasing recency or weak-cue choice in Experiments 3, 4, 13, and 25. No retention, scan, endpoint, contrast, isolation, refresh, capacity, or parameter-range changes are bundled with this intervention, avoiding the mechanisms that worsened loss in rejected iterations 3–7. The coalition equations are also left unchanged so compact multi-cue support remains protected primarily by coherence count and anchoring rather than being redefined by a new strategy.

**Parameters:**
  - `validities`: `validities`
  - `validity_sensitivity`: `[0.45, 0.85]`
  - `scan_direction`: `[-0.25, 1.45]`
  - `endpoint_salience`: `[0.45, 1.65]`
  - `contrast_gain`: `[0.28, 0.72]`
  - `accumulator_retention`: `[0.40, 0.62]`
  - `anchoring_strength`: `[0.22, 0.58]`
  - `capacity`: `[4.2, 7.4]`
  - `priority_temperature`: `[0.02, 0.18]`
  - `lapse_rate`: `[0.05, 0.16]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Priority-Latency Coalition Accumulation expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    scan_direction = float(parameters["scan_direction"])
    endpoint_salience = float(parameters["endpoint_salience"])
    contrast_gain = float(parameters["contrast_gain"])
    accumulator_retention = float(parameters["accumulator_retention"])
    anchoring_strength = float(parameters["anchoring_strength"])
    capacity = float(parameters["capacity"])
    priority_temperature = float(parameters["priority_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))

    # A fixed, task-invariant reliability scale preserves absolute gaps. It
    # therefore distinguishes nearly equal weak cues from a .99-versus-.52
    # conflict instead of mapping both two-cue displays to the same +/-1 pair.
    absolute_reliability = np.clip(
        (reliability - 0.20) / 1.55, -0.15, 3.0
    )

    # The complete map is computed before any sequential accumulation.
    # Local contrast is defined relative to the nearest discriminating item
    # on the left, reflecting perceptual grouping in the displayed layout.
    local_drop = np.zeros(n_features, dtype=float)
    directional_boundary = np.zeros(n_features, dtype=float)
    isolation = np.zeros(n_features, dtype=float)
    previous_j = None
    for j_raw in discriminating:
        j = int(j_raw)
        if previous_j is not None:
            gap_fraction = float(j - previous_j - 1) / float(
                max(1, n_features - 1)
            )
            isolation[j] = max(isolation[j], gap_fraction)
            isolation[previous_j] = max(isolation[previous_j], gap_fraction)

            previous_direction = float(np.sign(diff[previous_j]))
            current_direction = float(np.sign(diff[j]))
            if previous_direction != current_direction:
                directional_boundary[j] = 1.0
                drop = max(0.0, float(validities[previous_j] - validities[j]))
                # Fixed scaling retains the absolute distinction between a
                # shallow and a conspicuous instructed-validity drop.
                local_drop[j] = min(3.0, drop / 0.20)
        previous_j = j

    positions = np.arange(n_features, dtype=float) / float(max(1, n_features - 1))
    signed_position = 2.0 * positions - 1.0
    left_edge = (np.arange(n_features) == 0).astype(float)
    right_edge = (np.arange(n_features) == n_features - 1).astype(float)

    # Lower latency means earlier sampling. Reliability has the strongest
    # effect on access. Endpoint delay follows the subject's scan direction,
    # is amplified by isolation, and is attenuated for exceptionally reliable
    # cues, which can consequently enter before a spatially earlier weak cue.
    scan_sign = 1.0 if scan_direction >= 0.0 else -1.0
    endpoint_delay = (
        endpoint_salience
        * scan_sign
        * (right_edge - left_edge)
        * (0.35 + 0.90 * isolation)
        / (1.0 + np.maximum(0.0, absolute_reliability))
    )
    contrast_delay = contrast_gain * local_drop * directional_boundary
    latency = (
        -validity_sensitivity * absolute_reliability
        + scan_direction * signed_position
        + endpoint_delay
        + contrast_delay
    )

    # Modest Gumbel perturbations instantiate priority-dependent sampling
    # rather than a perfectly deterministic ranking on every presentation.
    sampled_latency = latency + np.random.gumbel(
        loc=0.0, scale=priority_temperature, size=n_features
    )

    # Stable index tie-breaking preserves the physical layout when sampled map
    # values are equal. Only discriminating cues consume integration capacity.
    cue_order = sorted(
        [int(j) for j in discriminating],
        key=lambda j: (float(sampled_latency[j]), j)
    )

    accumulator = 0.0
    incumbent_direction = 0.0
    coherent_count = 0.0
    coalition_strength = 0.0

    def apply_update(j, refresh_fraction, step_index,
                     accumulator, incumbent_direction,
                     coherent_count, coalition_strength):
        direction = float(np.sign(diff[j]))

        # Reliability primarily controls access priority. Its direct effect on
        # evidence magnitude is deliberately shallow, so a highly valid cue
        # sampled early does not remain disproportionately strong after later
        # weak evidence arrives.
        reliability_strength = float(np.exp(
            0.035 * validity_sensitivity * absolute_reliability[j]
        ))
        reliability_strength = float(np.clip(
            reliability_strength, 0.94, 1.14
        ))

        # A reliability drop is surprising only at a directional boundary.
        # Surprise boosts the challenger smoothly, with no reversal switch.
        surprise = 1.0 + (
            contrast_gain
            * local_drop[j]
            * directional_boundary[j]
            * (0.30 + 0.35 * min(1.0, abs(accumulator)))
        )
        surprise = float(np.clip(surprise, 1.0, 3.2))

        # Capacity loss is gradual and applies most strongly after the
        # subject-specific integration span has been exceeded.
        overload = max(0.0, float(step_index + 1) - capacity)
        capacity_factor = 1.0 / np.sqrt(1.0 + 0.28 * overload)

        opposing = bool(
            incumbent_direction != 0.0 and direction != incumbent_direction
        )
        if opposing:
            # Coherent agreement anchors the incumbent continuously. Spatial
            # isolation weakens the gate, allowing a conspicuous separated
            # challenger to revise more than a contiguous singleton.
            coherence = max(0.0, coherent_count - 1.0)
            anchor_mass = (
                coherence
                * min(2.0, coalition_strength)
                * (1.0 - 0.72 * min(1.0, isolation[j]))
            )
            anchor_gate = 1.0 + anchoring_strength * anchor_mass
        else:
            anchor_gate = 1.0

        update = (
            refresh_fraction
            * reliability_strength
            * surprise
            * capacity_factor
            / anchor_gate
        )

        accumulator = (
            accumulator_retention * accumulator + direction * update
        )
        accumulator = float(np.clip(accumulator, -2.75, 2.75))

        # Coalition state is graded. Opposition erodes the incumbent before
        # a new coherent coalition is installed; there is no categorical run
        # policy or thresholded strategy change.
        if incumbent_direction == 0.0:
            incumbent_direction = direction
            coherent_count = refresh_fraction
            coalition_strength = min(2.5, update)
        elif direction == incumbent_direction:
            coherent_count += refresh_fraction
            coalition_strength = min(
                2.5,
                0.78 * coalition_strength + 0.55 * update
            )
        else:
            erosion = refresh_fraction * update
            coalition_strength -= 0.72 * erosion
            coherent_count -= 0.62 * refresh_fraction
            if coalition_strength <= 0.0 or coherent_count <= 0.0:
                incumbent_direction = direction
                coherent_count = max(0.5, refresh_fraction)
                coalition_strength = min(2.5, 0.65 * update)
            else:
                coalition_strength = max(0.0, coalition_strength)
                coherent_count = max(0.0, coherent_count)

        return (
            accumulator,
            incumbent_direction,
            coherent_count,
            coalition_strength
        )

    for step_index, j in enumerate(cue_order):
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            j, 1.0, step_index, accumulator, incumbent_direction,
            coherent_count, coalition_strength
        )

    # Refresh is selective rather than a universal terminal bonus. It depends
    # on endpoint-by-scan congruence, endpoint isolation, and absolute local
    # contrast, preventing contiguous weak terminals from controlling every
    # display while preserving conspicuous endpoint effects.
    refresh_scores = {}
    for j in cue_order:
        scan_endpoint = right_edge[j] if scan_direction >= 0.0 else left_edge[j]
        opposite_endpoint = left_edge[j] if scan_direction >= 0.0 else right_edge[j]
        endpoint_refresh = endpoint_salience * (
            scan_endpoint * (0.20 + 1.35 * isolation[j])
            + 0.12 * opposite_endpoint * isolation[j]
        )
        refresh_scores[j] = (
            endpoint_refresh
            + contrast_gain * local_drop[j] * directional_boundary[j]
            + 0.04 * max(0.0, absolute_reliability[j])
        )

    refresh_j = max(
        cue_order,
        key=lambda j: (refresh_scores[j], sampled_latency[j])
    )
    raw_refresh = float(refresh_scores[refresh_j])
    refresh_fraction = 0.32 * raw_refresh / (1.0 + raw_refresh)
    if refresh_fraction > 1e-8:
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            refresh_j, refresh_fraction, len(cue_order), accumulator,
            incumbent_direction, coherent_count, coalition_strength
        )

    # Display load reduces readout precision even for unanimous evidence,
    # yielding a small rather than catastrophic decline with additional cues.
    nominal_overload = max(0.0, float(n_features) - capacity)
    evidence_overload = max(0.0, float(len(cue_order)) - capacity)
    precision = 2.65 / (
        1.0 + 0.045 * nominal_overload + 0.075 * evidence_overload
    )
    precision = float(np.clip(precision, 0.65, 2.65))

    logit_a = float(np.clip(precision * accumulator, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
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
