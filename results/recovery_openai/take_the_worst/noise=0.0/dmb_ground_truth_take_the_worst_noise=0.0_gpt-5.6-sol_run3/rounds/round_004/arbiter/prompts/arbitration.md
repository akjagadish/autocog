# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Mixed support
    # among the residual cues creates interference with stopping and modestly
    # increases continuation; unanimous residual support leaves it unchanged.
    # Attention then continues through the residual hierarchy. Conditional on
    # a miss, residual accessibility decays shallowly by ordinal rank. A
    # separate terminal-gap penalty limits access to cues lying very far from
    # the leading cue without reinstating a steep gradient over early/middle
    # cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    if len(discriminating_order) > 2:
        residual_signs = np.sign(diff[np.asarray(discriminating_order[1:], dtype=int)])
        residual_mixedness = 1.0 - abs(float(np.mean(residual_signs)))
        mixed_residual_boost = float(parameters["mixed_residual_boost"])
        serial_second_look += mixed_residual_boost * residual_mixedness
        serial_second_look = float(np.clip(serial_second_look, 0.0, 1.0))

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
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


## THEORY 2 — pi_6
Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Relational Chunk-Anchor Arbitration.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def first_coalition(sign_vector):
        idx = np.flatnonzero(sign_vector != 0.0)
        if idx.size == 0:
            return np.asarray([], dtype=int)
        first = int(idx[0])
        direction = float(sign_vector[first])
        members = [first]
        j = first + 1
        # Spatial contiguity matters: a tie or reversal terminates the chunk.
        while j < sign_vector.size and sign_vector[j] == direction:
            members.append(j)
            j += 1
        return np.asarray(members, dtype=int)

    anchor = first_coalition(signs)
    anchor_direction = float(signs[int(anchor[0])])
    anchor_len = int(anchor.size)

    # Count contiguous, same-direction discriminating runs. Ties separate runs.
    run_count = 0
    previous_sign = 0.0
    previous_index = -2
    for j in discriminating:
        j = int(j)
        s = float(signs[j])
        if j != previous_index + 1 or s != previous_sign:
            run_count += 1
        previous_index = j
        previous_sign = s

    if k > 1:
        fragmentation = float(run_count - 1) / float(k - 1)
    else:
        fragmentation = 0.0
    saturation = float(k) / float(max(n_features, 1))

    # A good anchor schema has a compact initial chunk, little same-direction
    # evidence outside that chunk, and a spatially coherent opposing coalition.
    same_total = int(np.sum(signs == anchor_direction))
    opposite_positions = np.flatnonzero(signs == -anchor_direction)
    if opposite_positions.size == 0:
        opposite_concentration = 0.0
    else:
        longest = 1
        current = 1
        for m in range(1, opposite_positions.size):
            if int(opposite_positions[m]) == int(opposite_positions[m - 1]) + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        opposite_concentration = float(longest) / float(opposite_positions.size)

    focal_exclusivity = float(anchor_len) / float(max(same_total, 1))
    compactness = float(np.exp(-0.70 * float(anchor_len - 1)))
    current_schema = compactness * focal_exclusivity * opposite_concentration

    # Repeated displays stabilize a relational anchor without learning cue
    # correctness. We measure whether the same earliest location and coalition
    # have recurred in the subject's preceding displays.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    focal_matches = []
    coalition_matches = []
    for old_a, old_b in zip(past_a, past_b):
        old_a = np.asarray(old_a, dtype=np.float64)
        old_b = np.asarray(old_b, dtype=np.float64)
        if old_a.shape != (n_features,) or old_b.shape != (n_features,):
            continue
        old_signs = np.sign(old_a - old_b)
        old_anchor = first_coalition(old_signs)
        if old_anchor.size == 0:
            continue
        focal_matches.append(float(int(old_anchor[0]) == int(anchor[0])))
        coalition_matches.append(float(np.array_equal(old_anchor, anchor)))

    if focal_matches:
        recurrence = 0.5 * float(np.mean(focal_matches)) + 0.5 * float(np.mean(coalition_matches))
    else:
        recurrence = 0.0

    stability_weight = float(parameters["stability_weight"])
    schema_strength = current_schema * (0.65 + stability_weight * recurrence)
    schema_strength = float(np.clip(schema_strength, 0.0, 1.0))

    # Anchor route. Validity cannot select a nonfocal cue; it only produces a
    # small reliability increment if the already focal coalition is relatively
    # valid. This precludes global validity-rank search and late-cue promotion.
    if np.ptp(validities) > 0.0:
        v_scaled = (validities - np.mean(validities)) / np.ptp(validities)
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    focal_validity = max(0.0, float(np.mean(v_scaled[anchor])))
    anchor_accuracy = float(parameters["anchor_accuracy"])
    validity_reinforcement = float(parameters["validity_reinforcement"])
    anchor_accuracy += validity_reinforcement * focal_validity
    anchor_accuracy = float(np.clip(anchor_accuracy, 0.5, 0.97))

    anchor_winner = 0 if anchor_direction > 0.0 else 1
    p_anchor = np.empty(2, dtype=np.float64)
    p_anchor[anchor_winner] = anchor_accuracy
    p_anchor[1 - anchor_winner] = 1.0 - anchor_accuracy

    # Comparison route: a majority-sensitive tally with only shallow position
    # and validity modulation. Neither modulation can create lexicographic use
    # of a late high-validity cue.
    if n_features > 1:
        normalized_position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    comparison_position_discount = float(parameters["comparison_position_discount"])
    comparison_validity_weight = float(parameters["comparison_validity_weight"])
    comparison_weights = np.exp(-comparison_position_discount * normalized_position)
    comparison_weights *= 1.0 + comparison_validity_weight * v_scaled
    comparison_weights = np.clip(comparison_weights, 1e-8, None)

    signed_comparison = float(np.dot(comparison_weights, signs))
    comparison_precision = float(parameters["comparison_precision"])
    comparison_logits = np.array(
        [comparison_precision * signed_comparison, 0.0], dtype=np.float64
    )
    comparison_logits -= np.max(comparison_logits)
    p_compare = np.exp(comparison_logits)
    p_compare /= p_compare.sum()

    # Arbitration depends jointly on saturation, coalition fragmentation,
    # relational anchor quality, and its recurrence. There is no fixed k
    # threshold, so similarly organized k=7 and k=9 displays remain alike.
    comparison_bias = float(parameters["comparison_bias"])
    saturation_weight = float(parameters["saturation_weight"])
    fragmentation_weight = float(parameters["fragmentation_weight"])
    anchor_schema_weight = float(parameters["anchor_schema_weight"])

    gate_argument = (
        comparison_bias
        + saturation_weight * saturation
        + fragmentation_weight * fragmentation
        - anchor_schema_weight * schema_strength
    )
    gate_argument = float(np.clip(gate_argument, -60.0, 60.0))
    comparison_gate = 1.0 / (1.0 + np.exp(-gate_argument))

    p_core = (1.0 - comparison_gate) * p_anchor + comparison_gate * p_compare

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.74, 0.95, 0.58, 0.83, 0.52, 0.89]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  trial 8: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]

**Rationale:** This fixed-conflict-count design isolates relational organization while keeping all six experts discriminating on every trial (k=6). The compact pattern has one leading cue favoring the designated option followed by a single contiguous five-cue opposing coalition (+-----). The three fragmented patterns retain the same leading-cue winner but distribute a second supporting cue at positions 3, 4, or 5, producing four alternating coalitions and a 2-to-4 tally disadvantage. Conflict-Density Adaptive Serial–Parallel Control remains essentially entirely in sparse serial mode at k=6, well below its 7.25–7.35 transition. It should therefore continue to favor the first-cue option in both compact and fragmented displays, generally around .55–.62, with only a modest positional gradient as the residual supporting cue moves later and encounters the terminal-gap penalty. Relational Chunk–Anchor Arbitration instead predicts a large structural discontinuity. The compact display has a maximally exclusive singleton anchor and a perfectly concentrated opposing coalition, so it should choose the anchor option well above chance, approximately .67–.90 after recurrence stabilizes the common focal location. Fragmentation rises from .20 to .60 in the other displays, their anchor schemas are substantially weaker, and saturation is maximal; this strongly recruits comparison. Because comparison sees a 2-to-4 majority against the anchor, the predicted probability of choosing the first-cue option should collapse to roughly .05–.30. Thus the critical dissociation is near-invariance with continued first-cue preference under the advocated theory versus a large compact-to-fragmented reversal under the competing theory, despite identical k, saturation, earliest discriminating location, and absence of ties. All patterns and their A/B mirrors are included. The eight unique pairs repeat 12 times each, yielding 96 trials; because every display has the same singleton coalition at feature 1, random trial order does not undermine the competing theory's recurrence mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric measures the structural compactness effect on adherence to the leading cue, while coding choices relative to that cue so that A/B mirrors cancel response-side bias. Conflict-Density Adaptive Serial–Parallel Control places every display in the same sparse serial regime and therefore predicts a value near zero, potentially slightly negative because fragmented displays contain an additional cue supporting the leading-cue option. Relational Chunk–Anchor Arbitration predicts a large positive value: compact displays preserve the singleton anchor, whereas fragmented displays recruit majority comparison and reverse choices away from that anchor. Averaging 24 compact and 72 fragmented observations per subject makes the contrast estimable for each subject with relatively low sampling variance.

Source:
def metric(data: pd.DataFrame) -> float:
    """Compact-minus-fragmented change in first-cue-consistent choice."""
    compact_scores = []
    fragmented_scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        signs = np.sign(a - b)
        if signs[0] == 0 or np.any(signs == 0):
            continue

        # Code whether the response follows the option favored by feature 1.
        first_cue_response = 0 if signs[0] > 0 else 1
        follows_first = float(int(row["response"]) == first_cue_response)

        # The compact pattern has a singleton initial coalition followed by
        # one unanimous opposing coalition: +----- or its mirror -+++++.
        is_compact = bool(np.all(signs[1:] == -signs[0]))
        if is_compact:
            compact_scores.append(follows_first)
        else:
            fragmented_scores.append(follows_first)

    if len(compact_scores) == 0 or len(fragmented_scores) == 0:
        return float("nan")

    return float(np.mean(compact_scores) - np.mean(fragmented_scores))

### RESULTS
- Predicted under pi_5 (simulated): -0.0636 (var=0.0139)
- Predicted under pi_6 (simulated): 0.7386 (var=0.0067)
- Observed on real data: -0.2167 (var=0.0112)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=9):** [0.55, 0.95, 0.55, 0.7, 0.8, 0.6, 0.9, 0.65, 0.75]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 1]
  trial 6: A=[0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design isolates history-based relational-anchor stabilization while matching the diagnostic displays on conflict count, tally, fragmentation, coalition lengths, and focal-coalition validity. The two target patterns are translations of the same k=8 structure: a two-cue anchor supporting the designated option followed by a coherent six-cue opposing coalition, with one exterior tie. In the common target the anchor occupies features 1-2 (++------0); in the rare target it occupies features 2-3 (0++------). Their anchor cues have the same mean advertised validity (.75). Eight additional exposure displays also begin with the exact feature-1/feature-2 anchor coalition. Including mirrors, 10 of 12 unique pairs therefore instantiate anchor indices [1,2], whereas only 2 instantiate [2,3]. After an initial burn-in, Relational Chunk–Anchor Arbitration predicts substantially greater schema recurrence for the common target (approximately .83 versus .17). Because the anchor favors the 2-to-6 tally minority, recurrence suppresses majority comparison and should increase designated-option choices for the common target by roughly 8-20 percentage points relative to the rare translated target, with a midpoint prediction near .55 versus .40. The effect should emerge as relevant displays accumulate, despite no correctness feedback. Conflict-Density Adaptive Serial–Parallel Control ignores history and treats both targets as k=8 dense displays. It therefore predicts an almost identical, stable and very low probability of choosing their 2-to-6 minority option; translating the pattern by one feature produces only negligible shallow positional and validity-weight changes. The critical tests are the common-versus-rare target contrast after burn-in and its interaction with trial phase. A/B mirrors preserve coalition locations while controlling product and response-key biases. The 12 unique pairs repeat eight times each for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Initial-coalition choice rate on the diagnostic 2-versus-6 conflicts. The measure counts how often a subject follows the compact two-cue coalition at the earliest discriminating location even though six discriminating cues favor the other option. It pools all 96 diagnostic presentations, including A/B mirrors, so response-key biases cancel and per-subject sampling variance is low. Relational Chunk–Anchor Arbitration predicts frequent anchor-consistent choices, strengthened by recurring coalition locations. Conflict-Density Adaptive Serial–Parallel Control places these k=8 displays in its dense tally regime and therefore predicts almost exclusive choice of the six-cue majority. The metric is an observable strategy signature rather than a direct encoding of either model's probabilities or parameters.

Source:
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        discriminating = np.flatnonzero(signs != 0)
        if discriminating.size == 0:
            continue

        first = int(discriminating[0])
        anchor_direction = signs[first]

        # Identify the initial contiguous coalition, stopping at a tie or reversal.
        anchor_length = 1
        j = first + 1
        while j < signs.size and signs[j] == anchor_direction:
            anchor_length += 1
            j += 1

        same_count = int(np.sum(signs == anchor_direction))
        opposite_count = int(np.sum(signs == -anchor_direction))

        # Restrict the score to the diagnostic 2-versus-6 displays in which
        # following the initial coalition conflicts with the cue majority.
        if anchor_length != 2 or same_count != 2 or opposite_count != 6:
            continue

        anchor_choice = 0 if anchor_direction > 0 else 1
        scores.append(float(int(row["response"]) == anchor_choice))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_5 (simulated): 0.0240 (var=0.0002)
- Predicted under pi_6 (simulated): 0.3240 (var=0.0045)
- Observed on real data: 0.8775 (var=0.0059)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code each response as adherence (1) versus non-adherence (0) to the
    # highest-validity cue, and recover the number of lower cues opposing it.
    adherence = []
    conflict_q = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        if a[0] > b[0]:
            top_cue_choice = 0
            q = int(np.sum(b[1:] > a[1:]))
        elif b[0] > a[0]:
            top_cue_choice = 1
            q = int(np.sum(a[1:] > b[1:]))
        else:
            continue

        adherence.append(float(response == top_cue_choice))
        conflict_q.append(q)

    adherence = np.asarray(adherence, dtype=float)
    conflict_q = np.asarray(conflict_q, dtype=int)

    low_conflict = conflict_q == 1
    high_conflict = conflict_q >= 4
    if not np.any(low_conflict) or not np.any(high_conflict):
        return float("nan")

    return float(np.mean(adherence[low_conflict]) -
                 np.mean(adherence[high_conflict]))

```

**Observed (real) value:** 0.0283 (var=0.0097)
**Predicted under pi_5:** -0.0008 (var=0.0242)
**Predicted under pi_6:** 0.0050 (var=0.0088)

### Experiment 4
**Design**
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    validities = np.asarray([0.67, 0.95, 0.58, 0.82, 0.53, 0.74, 0.61, 0.88], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    signed_agreement = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break

        if winner is not None and pd.notna(row["response"]):
            response = int(row["response"])
            signed_agreement.append(1.0 if response == winner else -1.0)

    return float(np.mean(signed_agreement)) if signed_agreement else float("nan")
```

**Observed (real) value:** -0.0050 (var=0.0051)
**Predicted under pi_5:** -0.0925 (var=0.0100)
**Predicted under pi_6:** 0.0338 (var=0.0097)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)

        # Feature 8 (index 7, equivalently the final feature here) is the
        # globally most valid cue in this experimental design.
        j = 7
        if a.size <= j or b.size <= j or a[j] == b[j]:
            continue

        cue8_winner = 0 if a[j] > b[j] else 1
        agreements.append(float(int(row['response']) == cue8_winner))

    if len(agreements) == 0:
        return float('nan')

    cue8_choice_rate = float(np.mean(agreements))

    # Soft classification around a preregistered separating rate. This is
    # smooth (unlike a hard threshold), remains defined per subject, and
    # compresses rates well inside either theory's predicted region.
    z = float(np.clip(20.0 * (cue8_choice_rate - 0.40), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0089 (var=0.0080)
**Predicted under pi_5:** 0.0196 (var=0.0013)
**Predicted under pi_6:** 0.0020 (var=0.0000)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast choices supported by the higher-validity cue on adjacent
    # conflicts versus conflicts separated by four or seven validity ranks.
    high_choice = {1: [], 4: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = np.flatnonzero(a != b)
        if diff.size != 2:
            continue

        high_idx = int(np.min(diff))  # Validities decrease with feature index.
        low_idx = int(np.max(diff))
        distance = low_idx - high_idx
        if distance not in high_choice:
            continue

        high_option = 0 if a[high_idx] > b[high_idx] else 1
        high_choice[distance].append(
            1.0 if int(row["response"]) == high_option else 0.0
        )

    if any(len(high_choice[d]) == 0 for d in (1, 4, 7)):
        return float("nan")

    adjacent = float(np.mean(high_choice[1]))
    well_separated = 0.5 * (
        float(np.mean(high_choice[4])) + float(np.mean(high_choice[7]))
    )
    return float(well_separated - adjacent)

```

**Observed (real) value:** 0.0042 (var=0.0061)
**Predicted under pi_5:** -0.0012 (var=0.0143)
**Predicted under pi_6:** 0.0012 (var=0.0071)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Soft prevalence of subjects reliably choosing the later conflicting cue."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        later_cue_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            differing = np.flatnonzero(a != b)
            if differing.size == 0:
                continue

            # In this design, the later of the two discriminating cues is the
            # locally high-validity cue.
            later = int(np.max(differing))
            later_winner = 0 if a[later] > b[later] else 1
            later_cue_choices.append(float(int(row["response"]) == later_winner))

        if len(later_cue_choices) == 0:
            continue

        rate = float(np.mean(later_cue_choices))
        # Smoothly score whether the subject shows a reliable (>65%) rather
        # than near-chance preference for the validity-promoted later cue.
        z = float(np.clip(20.0 * (rate - 0.65), -60.0, 60.0))
        scores.append(1.0 / (1.0 + np.exp(-z)))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0033 (var=0.0001)
**Predicted under pi_5:** 0.0185 (var=0.0006)
**Predicted under pi_6:** 0.0001 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    target_choices = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 2 or b.size < 2:
            continue

        # Features 1 and 2 are aligned in this design. Define the target as
        # the option jointly recommended by these two high-validity cues.
        direction = float(np.sum(a[:2] - b[:2]))
        if direction > 0:
            target_response = 0
        elif direction < 0:
            target_response = 1
        else:
            continue

        response = int(row["response"])
        target_choices.append(float(response == target_response))

    if len(target_choices) == 0:
        return float("nan")

    target_rate = float(np.mean(target_choices))

    # Smoothly classify rates around the otherwise-empty region separating
    # the advocated prediction (roughly .38-.48) from the competitor's
    # prediction (roughly .80-.86). Saturation also limits binomial and
    # parameter-driven between-subject variation away from the boundary.
    z = float(np.clip(20.0 * (target_rate - 0.64), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0005 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Difference in first-discriminating-cue choice rates at k=7 versus k=9."""
    sparse_hits = []
    dense_hits = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        k = int(discriminating.size)
        if k not in (7, 9):
            continue

        first = int(discriminating[0])
        first_cue_choice = 0 if diff[first] > 0 else 1
        hit = float(int(row["response"]) == first_cue_choice)

        if k == 7:
            sparse_hits.append(hit)
        else:
            dense_hits.append(hit)

    if len(sparse_hits) == 0 or len(dense_hits) == 0:
        return float("nan")

    return float(np.mean(sparse_hits) - np.mean(dense_hits))

```

**Observed (real) value:** 0.0067 (var=0.0029)
**Predicted under pi_5:** 0.3450 (var=0.0092)
**Predicted under pi_6:** 0.0271 (var=0.0045)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the earliest discriminating cue."""
    agreements = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0 or pd.isna(response):
            continue

        first = int(discriminating[0])
        leading_choice = 0 if diff[first] > 0 else 1
        agreements.append(float(int(response) == leading_choice))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.8446 (var=0.0066)
**Predicted under pi_5:** 0.6575 (var=0.0018)
**Predicted under pi_6:** 0.8446 (var=0.0020)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, pi_6 is the substantially stronger theory, but it is not an adequate task-invariant account. It is close to the observations in Experiments 3\u20139, exactly matches Experiment 10's point estimate (0.8446), and is much closer than pi_5 in the critical Experiment 9 contrast (0.0271 versus observed 0.0067, while pi_5 predicts 0.3450). Experiment 9 directly undermines pi_5's proposed conflict-count transition: behavior is essentially invariant between k=7 and k=9 rather than showing the predicted architectural switch. Pi_5 also severely underpredicts initial-coalition choices in Experiments 2 and 10. Thus, its defining count-triggered serial-to-parallel mechanism is contradicted, not merely mistuned. Pi_6 correctly anticipates the strong primacy in Experiment 10 and is directionally closer in Experiment 2, but it still underpredicts the observed 2-cue-coalition adherence by 0.5535. More importantly, pi_6 catastrophically fails Experiment 1: it predicts a compact-minus-fragmented effect of +0.7386, whereas the observed effect is -0.2167. This is not a small parameter error. The result rejects the model's assumption that a coherent singleton anchor followed by a compact opposing coalition is especially powerful. Instead, the negative contrast is consistent with a singleton being insufficient to establish an anchor: without a robust initial chunk, participants are more likely to follow the leading option when it receives a second supporting cue in fragmented displays than when it loses the tally 1-to-5 in the compact display. Taken together with Experiments 2 and 10, the evidence suggests a nonlinear distinction between unsupported singleton onsets and genuinely redundant initial coalitions, especially coherent two-cue coalitions. Experiments 4\u20137 also indicate that instructed validity has little ability to promote a later cue, while Experiments 3 and 6 show little effect of conflict magnitude or cue separation. The observed variances generally indicate stable subject-level effects and do not rescue either failed point prediction; pi_6 also appears to underestimate some heterogeneity, notably in Experiment 10. Overall, pi_6 should remain the benchmark, but neither current theory explains the full pattern. Pi_5 should be replaced because its central conflict-density transition is specifically falsified across tasks.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_5 with a Redundant-Onset Coalition theory. Its central claim should be that primacy arises only when the earliest evidence forms a sufficiently redundant and perceptually segregated coalition, rather than from the first discriminating cue or from an absolute conflict-count threshold. A singleton onset should have low anchoring strength regardless of how coherent the opposing coalition is; choices then reflect a noisy, weakly position-weighted comparison, predicting lower leading-option choice for a 1-to-5 display than for a 2-to-4 display and hence the negative Experiment 1 contrast. An exclusive contiguous two-cue onset opposed by one coherent coalition should cross a nonlinear chunk-formation threshold and support highly reliable anchor choices, accommodating Experiments 2 and 10 even against a numerical majority. Extra same-direction cues outside the onset, alternation, and fragmentation should reduce coalition exclusivity and return control toward comparison, accommodating Experiment 8. Arbitration should depend on initial-coalition redundancy, exclusivity, opponent concentration, and fragmentation\u2014not k itself\u2014so k=7 and k=9 can behave similarly as in Experiment 9. Validity should remain weak and mainly reinforce an already formed coalition, consistent with Experiments 4\u20137. Recurrence may modestly strengthen an existing coalition, but it should not create an anchor from an otherwise inadequate singleton; moreover, the current Experiment 2 metric establishes high anchor adherence but does not isolate the proposed common-versus-rare history effect, so strong learning or recurrence parameters are not yet justified. Include subject-level variation in the coalition-formation threshold and anchor reliability to match the observed variances. This theory is a stronger competitor to pi_6 because it retains relational organization while making the empirically crucial, falsifiable claim that onset redundancy\u2014not compactness alone\u2014is required for anchoring."
}
```

## Usage

```json
{
  "input_tokens": 17611,
  "output_tokens": 2398
}
```
