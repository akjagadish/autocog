# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

**Rationale:** This is a minimal edit to the accepted iteration-7 model. The rank-normalized residual rule, terminal-gap penalty, high density-transition midpoint, weak validity modifier, tally process, and lapse mechanism are retained unchanged. The only new mechanism is a narrowly constrained mixed_residual_boost: residual mixedness is measured as one minus the absolute mean support direction among cues after the leading discriminating cue, and this smoothly increases the existing continuation probability. It is exactly zero for unanimous residual coalitions. Consequently, Experiment 1 is protected because all cues following cue 1 support the same option, and the two-cue displays in Experiments 4 and 5 are unchanged because their sole residual cue cannot be mixed. Experiment 2's mixed-support displays receive more opportunity for the accepted residual-accessibility process to counter rigid first-cue anti-alignment. In Experiment 3, mixedness can activate the adjustment, but the terminal-gap penalty continues to suppress the remote cue 8 while directing most additional continuation mass toward the earlier coalition cues. The narrow boost range limits subject heterogeneity and avoids the previously rejected changes to validity weighting, density gating, or residual-accessibility shape.

**Parameters:**
  - `validities`: `validities`
  - `transition_count`: `[7.25, 7.35]`
  - `transition_steepness`: `[7.0, 8.5]`
  - `position_bias`: `[0.90, 1.05]`
  - `validity_modifier`: `[0.01, 0.04]`
  - `serial_second_look`: `[0.36, 0.40]`
  - `mixed_residual_boost`: `[0.10, 0.14]`
  - `residual_rank_decay`: `[0.08, 0.12]`
  - `terminal_gap_penalty`: `[4.0, 5.0]`
  - `serial_precision`: `[1.65, 1.90]`
  - `positional_discount`: `[0.06, 0.14]`
  - `tally_precision`: `[1.05, 1.35]`
  - `epsilon`: `[0.01, 0.04]`

**`predict(parameters, stimulus, history)`:**
```python
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


### slot 2 — `pi_3` — KILLED ✗

**Description:** Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

**Rationale:** This is a minimal in-family edit to the accepted candidate. The mapping-simplicity gate, priority equation, validity contribution, positional contribution, and lapse process are retained. Only the deterministic sorting block is replaced by the marginal choice probabilities implied by a Plackett-Luce/Gumbel cue ordering: among currently discriminating cues, softmax priority determines which one is encountered first. Accessibility-weight dispersion is also narrowed from [0,1] to [0.48,0.52]. These changes directly address the excessive Experiment 2 between-subject variance by preventing each subject's arbitrary accessibility draw from fixing a highly polarized cue order. They do not turn the model into globally random responding: visual position and stable latent priorities continue to structure each choice. In Experiment 1, the validity-position alignment gives cue 1 a large priority advantage, so stochastic search remains nearly lexicographic; the selected temperature permits only a small increase in lower-cue influence as conflict grows, moving the contrast toward the observed small positive value. In Experiment 2, the weak mapping gate and balanced mirrored schedule make stochastic accessibility-based search approximately orthogonal to the validity-ranked winner within subjects, reducing reliance on cross-subject cancellation.

**Parameters:**
  - `validities`: `validities`
  - `accessibility_weights`: `[(0.48, 0.52)] * n_features`
  - `position_strength`: `[0.02, 0.15]`
  - `validity_strength`: `[3.0, 5.0]`
  - `mapping_fidelity`: `[0.8, 1.0]`
  - `complexity_power`: `[3.0, 8.0]`
  - `attention_temperature`: `[0.30, 0.40]`
  - `epsilon`: `[0.02, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.

**Rationale:** The model replaces pi_3's global validity-position coherence gate with a representation-based arbitration mechanism. Conflict count alone never selects a strategy. Compact early coalitions and recurring focal roles support anchoring, while saturation and fragmentation jointly recruit comparison. This preserves nearly constant top-cue adherence across Experiment 1's conflict levels, gives the readily chunked five-cue displays in Experiment 8 strong early-coalition choices near the observed 0.845, and sends the fully saturated displays of Experiment 6 toward the opposing majority. The similarly fragmented k=7 and k=9 displays in Experiment 7 both recruit comparison, avoiding pi_5's artificial discontinuity. Experiment 2's scattered coalitions also receive substantial comparative processing, making responses approximately orthogonal to validity-ranked search. Validity only adjusts the reliability of an already selected anchor and has very small weight in comparison, so it cannot promote the late cue in Experiments 3 and 5 or create a validity-distance effect in Experiment 4. Broad subject variation in anchor accuracy and the comparison gate supplies the requested heterogeneity. Unlike a fixed conflict threshold, the theory predicts separable effects of k/n, spatial contiguity, coalition recurrence, and invariant focal placement when k is held constant.

**Parameters:**
  - `validities`: `validities`
  - `anchor_accuracy`: `[0.84, 0.94]`
  - `validity_reinforcement`: `[0.00, 0.03]`
  - `stability_weight`: `[0.25, 0.35]`
  - `comparison_bias`: `[-4.5, -3.5]`
  - `saturation_weight`: `[6.5, 7.5]`
  - `fragmentation_weight`: `[3.5, 4.5]`
  - `anchor_schema_weight`: `[7.0, 8.0]`
  - `comparison_position_discount`: `[0.00, 0.04]`
  - `comparison_validity_weight`: `[0.00, 0.04]`
  - `comparison_precision`: `[1.5, 2.2]`
  - `epsilon`: `[0.01, 0.04]`

**`predict(parameters, stimulus, history)`:**
```python
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
