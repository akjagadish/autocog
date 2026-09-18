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
**Description:** Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

**Parameters:**
- validities: validities
- transition_count: [7.25, 7.35]
- transition_steepness: [7.0, 8.5]
- position_bias: [0.90, 1.05]
- validity_modifier: [0.01, 0.04]
- serial_second_look: [0.36, 0.40]
- mixed_residual_boost: [0.10, 0.14]
- residual_rank_decay: [0.08, 0.12]
- terminal_gap_penalty: [4.0, 5.0]
- serial_precision: [1.65, 1.90]
- positional_discount: [0.06, 0.14]
- tally_precision: [1.05, 1.35]
- epsilon: [0.01, 0.04]

**`predict source code`:**
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
**Description:** Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.

**Parameters:**
- validities: validities
- anchor_accuracy: [0.84, 0.94]
- validity_reinforcement: [0.00, 0.03]
- stability_weight: [0.25, 0.35]
- comparison_bias: [-4.5, -3.5]
- saturation_weight: [6.5, 7.5]
- fragmentation_weight: [3.5, 4.5]
- anchor_schema_weight: [7.0, 8.0]
- comparison_position_discount: [0.00, 0.04]
- comparison_validity_weight: [0.00, 0.04]
- comparison_precision: [1.5, 2.2]
- epsilon: [0.01, 0.04]

**`predict source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The highest-validity expert (feature 1, validity .95) always favors one product, while q = 1 through 6 lower-validity experts favor the other product; all remaining features tie. Each configuration is paired with its A/B mirror to control response and display-side biases. Take The Best therefore follows feature 1 on every trial and predicts the same choice probability at every value of q: apart from orientation, its probability is constant at (1-epsilon)*sigmoid(beta)+epsilon/2. Tallying predicts an exact 50/50 choice when q=1 (one cue win per option), then increasingly strong preference for the lower-cue majority as q rises, with tally margins q-1 and probabilities governed by sigmoid(beta*(q-1)). Thus the key quantitative signature is a flat, high-validity-cue preference under TTB versus a direction reversal and monotonic evidence-strength gradient under Tallying. The 12 unique mirrored pairs repeat eight times each, yielding 96 trials.

[1] This balanced-evidence design targets a parameter-robust dissociation. Every option pair contains exactly three cues favoring A, three favoring B, and two ties. Tallying therefore predicts P(A)=0.5 on every trial, regardless of beta or epsilon. In each of the first six pairs, however, the highest-validity cue that discriminates favors A; in the corresponding A/B mirror it favors B. Take The Best consequently predicts its ranked-cue winner with probability (1-epsilon)*sigmoid(beta)+epsilon/2. The first discriminating cue occurs at validity rank 1, 2, or 3 across conditions, while the tally remains fixed at 3-3. Thus systematic choice of the validity-ranked winner, invariant across these ranks, supports Take The Best, whereas chance-level aggregate and within-pair choices support Tallying. Validity rank is deliberately nonaligned with display position, reducing the possibility that a simple left-to-right feature-order strategy mimics Take The Best. All 12 unique pairs are diagnostic and repeat eight times for 96 trials.

[2] This design creates a parameter-robust directional dissociation while also testing a quantitative position gradient. The validity-to-display mapping is deliberately nonmonotonic and has a positive rank-position correlation, so the competing model's mapping-simplicity term is clipped to zero and communicated validity cannot alter its attentional priorities. On every trial, the globally most valid cue (feature 8, validity .95) favors one option, while exactly three lower-validity cues favor the other. Take The Best therefore always selects the feature-8 winner and predicts the same oriented choice probability in every condition: (1-epsilon)*sigmoid(beta)+epsilon/2, which remains above .50 throughout its parameter range. In the competing stochastic-search model, validity is gated out and the first discriminating cue is selected primarily by accessibility and display position. Because three cues support the opposing option and only one supports the validity-ranked winner, it predicts the validity-ranked winner on only about 20-30% of trials across its parameter range. Moreover, as the three opposing cues shift from early positions (features 1-3) toward later positions (features 5-7), their positional advantage decreases, producing a modest increase in choices of the feature-8 winner; TTB predicts complete invariance across this manipulation. A/B mirrors remove product-key bias. The 12 unique pairs repeat eight times for 96 trials.

[3] This design tests a rank-distance signature that Take The Best cannot produce. The ten validities decrease monotonically with display position, making mapping simplicity exactly maximal for Accessibility-Gated Stochastic Lexicographic Search. Each trial has exactly two discriminating experts: the higher-validity expert favors one product and the lower-validity expert favors the other, while all remaining experts tie. The rank distance between those two experts is manipulated at four levels (1, 2, 4, or 7 ranks), with three different cue pairs at each distance. Take The Best always follows the higher-validity discriminating expert and therefore predicts exactly the same oriented choice probability at every rank distance: (1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, stochastic lexicographic search can inspect either discriminating expert. Because aligned validity information creates a priority difference proportional to their rank separation, it predicts a monotonic increase in choice of the higher-ranked expert's product as rank distance grows: adjacent conflicts should produce appreciable stochastic choice, whereas seven-rank conflicts should be nearly deterministic. Replicating each distance at different absolute positions separates rank-distance effects from idiosyncratic accessibility of a particular expert. Every pair has an A/B mirror to control product-side and response-key biases. The 24 unique pairs repeat four times, giving 96 trials.

[4] The validity sequence is globally anti-aligned with display position: validity ranks increase strongly toward later positions, so the competing theory's rank-position correlation is positive and its mapping-simplicity term is clipped to zero. Consequently, Accessibility-Gated Stochastic Lexicographic Search must ignore communicated validity in this experiment. Each diagnostic trial pits an earlier, low-validity cue against the immediately following, much higher-validity cue. Under the advocated theory, the 0.45-0.47 validity increase is large enough to reverse the local positional ordering for every allowed parameter combination, including the accessibility offsets. It therefore follows the later, high-validity cue and predicts approximately 0.78-0.86 choice probability for its product. The competing theory instead gives the later cue no validity advantage; with only a small one-position accessibility difference, it predicts choices near chance and, on average, a slight preference for the earlier low-validity cue (roughly 0.45-0.53 for the high-validity-cue winner across parameter extremes). Thus the critical signature is systematic selection of locally validity-promoted later cues rather than near-chance choices. Four adjacent-cue conflicts establish that the effect generalizes across the display rather than depending on one expert. Each conflict has an A/B mirror, controlling product-side and response-key biases. The eight unique pairs repeat 12 times each for 96 trials.

[5] This design exploits the competing theories' sharply different treatment of cues below the first two positions. The validity sequence has rank-score order [7,6,0,1,2,3,4,5], giving rank-position correlation -1/6 and mapping simplicity 1/6. Consequently, the advocated theory's validity gate is at most (1/6)^3 ≈ .0046 and attention is governed mainly by weak positional accessibility plus stochastic cue-order sampling. All eight cues discriminate on every trial. Features 1 and 2 always support the same option, five cues support its opponent, and one additional supporting cue is moved systematically from feature 3 through feature 8. The advocated theory therefore integrates the Plackett-Luce weights of three supporting versus five opposing cues and predicts that the option supported by features 1 and 2 will generally be chosen less than 50% of the time, approximately .38-.48 across conditions and parameter extremes. It additionally predicts a graded decline as the third supporting cue moves later in the display. In the fixed-search theory, features 1 and 2 are robustly the top two cues: their large validity and positional advantages cannot be reversed by the allowed accessibility offsets. Because both the first cue and the cue consulted after a search slip support the same option, all six conditions receive exactly the same prediction, approximately .80-.86 choice probability for that option, regardless of where the third supporting cue appears. Thus the theories predict opposite majority choices as well as a quantitative position gradient versus invariance. A/B mirrors control product and response-key biases. The 12 unique pairs repeat eight times for 96 trials.

[6] This design targets the advocated theory's stimulus-contingent transition at approximately 7.3 discriminating cues. Four matched families compare a sparse display with k=7 against a dense display with k=9. Every sparse display has three cues favoring the leading-cue option and four favoring its opponent; the corresponding dense display adds one cue for each option, yielding four versus five. Thus the raw tally margin remains exactly -1 and the proportion of cues supporting the leading-cue option changes only from 3/7=.429 to 4/9=.444. The first discriminating cue always supports the tally minority. At k=7, the advocated theory remains predominantly serial and should choose that first-cue option above chance despite its 3-to-4 tally disadvantage. At k=9, its dense gate is essentially fully active, so it should instead integrate the 4-to-5 tally and choose the same option only about .25-.35 of the time. It therefore predicts a large within-family downward discontinuity and a choice reversal across the transition. The validity sequence increases with display position, producing mapping simplicity zero in the competing theory; communicated validity is consequently gated out. That theory uses nearly uniform accessibility weights with only a shallow positional gradient, so its probability for the designated option is governed mainly by its cue share and should remain near .43-.47 across k=7 and k=9, with no sharp transition or reversal generated by conflict count. The four positional arrangements test generality and vary which late cues are added. Every stimulus has an A/B mirror, controlling product and response-key biases. There are 16 unique pairs, each repeated six times, for 96 trials.

[7] This sparse-conflict design isolates the competing theory's terminal-gap mechanism without crossing its serial–parallel transition. Validities increase monotonically with display position, yielding rank-position correlation +1 and therefore mapping simplicity 0 in Accessibility-Gated Stochastic Lexicographic Search. Its validity gate is exactly closed: choice is governed by nearly uniform accessibility plus a shallow positional gradient. Each base trial has five discriminating cues. Features 1 and 2 favor the designated option, while three cues favor its opponent. Those three opposing cues move together from features 3–5 through features 8–10. Accessibility-gated search should therefore initially favor the 3-to-2 cue majority and show only a gradual increase in designated-option choice as the opposing cues move later (roughly from about .37–.45 toward .45–.55, depending on attentional parameters), with no special discontinuity at a particular gap. Conflict-Density Adaptive Serial–Parallel Control remains almost purely serial because k=5 is far below its 7.25–7.35 transition. Its leading discriminating cue always favors the designated option. More importantly, opposing residual cues begin receiving a strong terminal-gap penalty once they lie more than 55% of the display beyond feature 1. It consequently predicts substantially higher designated-option choice throughout (approximately .60–.65 for opposing cues at features 3–5) and a comparatively sharp rise to approximately .70–.76 when they reach features 8–10. The critical quantitative dissociation is thus a low, shallow cue-location gradient under accessibility-gated stochastic search versus a high curve with accelerated growth as residual cues cross the terminal-gap boundary under adaptive serial control. Every configuration has an A/B mirror, giving 12 unique pairs repeated eight times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This fixed-conflict-count design isolates relational organization while keeping all six experts discriminating on every trial (k=6). The compact pattern has one leading cue favoring the designated option followed by a single contiguous five-cue opposing coalition (+-----). The three fragmented patterns retain the same leading-cue winner but distribute a second supporting cue at positions 3, 4, or 5, producing four alternating coalitions and a 2-to-4 tally disadvantage. Conflict-Density Adaptive Serial\u2013Parallel Control remains essentially entirely in sparse serial mode at k=6, well below its 7.25\u20137.35 transition. It should therefore continue to favor the first-cue option in both compact and fragmented displays, generally around .55\u2013.62, with only a modest positional gradient as the residual supporting cue moves later and encounters the terminal-gap penalty. Relational Chunk\u2013Anchor Arbitration instead predicts a large structural discontinuity. The compact display has a maximally exclusive singleton anchor and a perfectly concentrated opposing coalition, so it should choose the anchor option well above chance, approximately .67\u2013.90 after recurrence stabilizes the common focal location. Fragmentation rises from .20 to .60 in the other displays, their anchor schemas are substantially weaker, and saturation is maximal; this strongly recruits comparison. Because comparison sees a 2-to-4 majority against the anchor, the predicted probability of choosing the first-cue option should collapse to roughly .05\u2013.30. Thus the critical dissociation is near-invariance with continued first-cue preference under the advocated theory versus a large compact-to-fragmented reversal under the competing theory, despite identical k, saturation, earliest discriminating location, and absence of ties. All patterns and their A/B mirrors are included. The eight unique pairs repeat 12 times each, yielding 96 trials; because every display has the same singleton coalition at feature 1, random trial order does not undermine the competing theory's recurrence mechanism.",
  "validities": [
    0.74,
    0.95,
    0.58,
    0.83,
    0.52,
    0.89
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 8327,
  "output_tokens": 4362
}
```
