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
**Description:** Continuous Reliability-Gated Reason Competition with Bounded Embedded Leverage, Credible Participation, and Selective Dissent Decay (CRGRC-BEL-CP-SDD) claims that communicated validities are continuously compressed into reliability evidence and accumulated within option-specific coalitions under divisive redundancy. Credible sparse opposition receives a graded diagnosticity gain, whereas weak opposition is filtered by a smooth credibility threshold. Diagnosticity remains strong for compact credible opposition but declines more sharply once effective opponent multiplicity exceeds the sparse-competition region. Restoration of an isolated, high-validity anchor depends separately on credible participation: each opposing reason contributes continuously according to its weight relative to the coalition's strongest opponent. Moderately reliable contributors now enter participation more readily, while a higher and steeper restoration threshold keeps two- and three-reason coalitions below restoration and allows only broadly participating coalitions to restore the anchor. For uniquely strongest anchors embedded in a coalition, the effect of coalition composition on the final normalized margin remains bounded according to anchor identity share. Serial accessibility remains restricted to actual or psychologically indistinguishable top-validity ties and is selectively strengthened by balanced coalition competition. The model is strictly history-invariant.

**Parameters:**
- validities: validities
- validity_compression: [0.16, 0.48]
- redundancy: [0.10, 0.38]
- sparse_dissent: [3.2, 6.8]
- dissent_saturation: [1.65, 2.20]
- dissent_exponent: [3.0, 4.6]
- credibility_threshold: [0.62, 0.75]
- credibility_slope: [11.0, 18.0]
- embedded_margin_floor: [0.30, 0.55]
- restoration_threshold: [3.55, 3.95]
- restoration_participation_cutoff: [0.30, 0.48]
- restoration_slope: [6.0, 10.0]
- restoration_strength: [0.74, 0.97]
- tie_width: [0.004, 0.022]
- tie_order_weight: [0.58, 0.86]
- beta: [2.0, 4.8]
- lapse: [0.0, 0.10]
- trait_correlation: [-1.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CRGRC expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # CRGRC deliberately performs no learning in this feedback-free task.
    _ = history

    validity_compression = float(parameters["validity_compression"])
    redundancy = float(parameters["redundancy"])
    sparse_dissent = float(parameters["sparse_dissent"])
    dissent_saturation = float(parameters["dissent_saturation"])
    dissent_exponent = float(parameters["dissent_exponent"])
    credibility_threshold = float(parameters["credibility_threshold"])
    credibility_slope = float(parameters["credibility_slope"])
    embedded_margin_floor = float(parameters["embedded_margin_floor"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_participation_cutoff = float(
        parameters["restoration_participation_cutoff"]
    )
    restoration_slope = float(parameters["restoration_slope"])
    restoration_strength = float(parameters["restoration_strength"])
    tie_width = float(parameters["tie_width"])
    tie_order_weight = float(parameters["tie_order_weight"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    trait_correlation = float(parameters["trait_correlation"])

    # A single stable orientation induces correlated random effects. Positive
    # values preserve communicated-validity differences, increase redundancy,
    # restoration, and precision, and reduce dissent and serial reliance.
    z = trait_correlation
    validity_compression *= np.exp(0.32 * z)
    redundancy *= np.exp(0.22 * z)
    sparse_dissent *= np.exp(-0.42 * z)
    restoration_strength *= np.exp(0.28 * z)
    tie_order_weight *= np.exp(-0.24 * z)
    beta *= np.exp(0.18 * z)

    restoration_strength = float(np.clip(restoration_strength, 0.0, 0.98))
    tie_order_weight = float(np.clip(tie_order_weight, 0.0, 0.95))

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is converted to reliability evidence and compressed
    # continuously relative to the strongest currently relevant expert.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(
        validity_compression * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)

    def effective_multiplicity(indices):
        if indices.size == 0:
            return 0.0
        w = np.asarray(cue_weight[indices], dtype=float)
        total = float(np.sum(w))
        return (total * total) / max(float(np.sum(w * w)), 1e-12)

    def coalition_support(indices):
        if indices.size == 0:
            return 0.0
        w = np.asarray(cue_weight[indices], dtype=float)
        raw = float(np.sum(w))
        effective_n = effective_multiplicity(indices)

        # Divisive redundancy is based on effective evidence multiplicity.
        # It therefore responds to coalition validity composition without using
        # cue indices, spatial dispersion, or exact nominal display size.
        excess = max(effective_n - 1.0, 0.0)
        divisor = 1.0 + redundancy * (excess ** 1.20)
        return raw / max(divisor, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    active_v = validities[active]
    descending = np.sort(active_v)[::-1]
    best_v = float(descending[0])
    second_v = float(descending[1]) if descending.size > 1 else 0.5
    top_gap = max(best_v - second_v, 0.0)

    # Ambiguity and uniqueness are complementary continuous gates. Exact ties
    # have ambiguity one; a clearly unique maximum has ambiguity near zero.
    scaled_gap = top_gap / max(tie_width, 1e-8)
    top_ambiguity = float(np.exp(-0.5 * scaled_gap * scaled_gap))
    unique_strength = 1.0 - top_ambiguity

    top_cues = active[np.isclose(active_v, best_v, atol=1e-12)]
    anchor_cue = int(np.min(top_cues))
    anchor_direction = float(directions[anchor_cue])

    if anchor_direction > 0:
        anchor_indices = b_cues
        opponent_indices = a_cues
        anchor_support = support_b
        opponent_support = support_a
    else:
        anchor_indices = a_cues
        opponent_indices = b_cues
        anchor_support = support_a
        opponent_support = support_b

    anchor_effective_n = effective_multiplicity(anchor_indices)
    opponent_effective_n = effective_multiplicity(opponent_indices)

    if anchor_indices.size > 0:
        anchor_weights = np.asarray(cue_weight[anchor_indices], dtype=float)
        anchor_identity_share = float(np.max(anchor_weights)) / max(
            float(np.sum(anchor_weights)), 1e-12
        )
    else:
        anchor_identity_share = 0.0

    credible_participation = 0.0
    if opponent_indices.size > 0:
        opponent_weights = np.asarray(cue_weight[opponent_indices], dtype=float)
        strongest_opponent = float(np.max(opponent_weights))
        mean_opponent = float(np.mean(opponent_weights))

        # Restoration uses a smooth participation count rather than Kish
        # multiplicity. Every opponent contributes in proportion to whether
        # its weight is credible relative to the strongest opposing reason.
        relative_opponent_weights = opponent_weights / max(
            strongest_opponent, 1e-12
        )
        participation_gates = 1.0 / (
            1.0 + np.exp(-12.0 * (
                relative_opponent_weights - restoration_participation_cutoff
            ))
        )
        credible_participation = float(np.sum(participation_gates))

        # Credibility combines the strongest opposing reason and the typical
        # opposing reason. A smooth threshold protects anchors from genuinely
        # weak opposition without suppressing credible intermediate coalitions.
        opposition_credibility = np.sqrt(
            np.clip(strongest_opponent * mean_opponent, 0.0, 1.0)
        )
        credibility_gate = sigmoid(
            credibility_slope
            * (opposition_credibility - credibility_threshold)
        )

        # Sparse diagnosticity is sustained through compact effective
        # multiplicities but decays sharply outside that region. Isolation is
        # continuous: embedded anchors receive less challenge amplification.
        sparse_profile = 1.0 / (
            1.0
            + (opponent_effective_n / max(dissent_saturation, 1e-8))
            ** dissent_exponent
        )
        isolation = anchor_identity_share ** 1.5
        challenge_gain = (
            sparse_dissent
            * unique_strength
            * credibility_gate
            * sparse_profile
            * isolation
        )
        opponent_support *= 1.0 + challenge_gain

    total_support = anchor_support + opponent_support
    if total_support <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total_support
        core_evidence = anchor_direction * anchor_margin

    # Under a unique maximum, embedding bounds the leverage of coalition
    # composition on the normalized margin. Isolated anchors are unchanged.
    embedded_leverage = (
        embedded_margin_floor
        + (1.0 - embedded_margin_floor) * anchor_identity_share
    )
    leverage_gate = 1.0 - unique_strength * (1.0 - embedded_leverage)
    core_evidence *= leverage_gate

    # Restoration grows only when many opponents credibly participate. Unlike
    # Kish multiplicity, this statistic does not collapse a broad coalition
    # merely because some contributing reasons are moderately less reliable.
    anchor_quality = np.clip((best_v - 0.5) / 0.5, 0.0, 1.0)
    count_restoration = sigmoid(
        restoration_slope * (
            credible_participation - restoration_threshold
        )
    )
    restoration_gate = (
        restoration_strength
        * unique_strength
        * (anchor_identity_share ** 1.7)
        * (anchor_quality ** 1.3)
        * count_restoration
    )
    restoration_gate = float(np.clip(restoration_gate, 0.0, 0.97))
    restored_evidence = (
        (1.0 - restoration_gate) * core_evidence
        + restoration_gate * anchor_direction
    )

    # Serial accessibility is calculated only through the continuous top-tie
    # gate. Serial rank among active cues is meaningful presentation order, not
    # spatial span. Reliability closeness prevents clearly inferior cues from
    # acting as serial anchors even when they occur early.
    serial_numerator = 0.0
    serial_denominator = 0.0
    for rank, cue in enumerate(active):
        closeness = np.exp(
            -(best_v - validities[cue]) / max(tie_width, 1e-8)
        )
        accessibility = closeness * np.exp(-float(rank) / 0.70)
        serial_numerator += accessibility * float(directions[cue])
        serial_denominator += accessibility

    if serial_denominator <= 1e-12:
        serial_evidence = 0.0
    else:
        serial_evidence = serial_numerator / serial_denominator

    # Balanced reason competition selectively strengthens tied-top order
    # resolution while retaining the previous imbalanced-profile baseline.
    prechallenge_total = support_a + support_b
    if prechallenge_total <= 1e-12:
        coalition_balance = 1.0
    else:
        coalition_balance = (
            2.0 * min(support_a, support_b) / prechallenge_total
        )
    balance_gate = 0.45 + 0.85 * coalition_balance
    serial_gate = float(np.clip(
        tie_order_weight * top_ambiguity * balance_gate, 0.0, 0.95
    ))

    choice_evidence = (
        (1.0 - serial_gate) * restored_evidence
        + serial_gate * serial_evidence
    )

    logits = beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= total
    return probs.astype(np.float64)
```

**`policy source code`:**
```python
def policy(probs):
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
**Description:** Static Configural Coalition Competition (S3C) proposes that decisions are reconstructed from the current cue display without learning from trial history. Each discriminating expert retains an identity-specific weight derived from communicated validity, but validity evidence is bounded to prevent a uniquely most-valid cue from becoming arbitrarily dominant. Serial accessibility is a distinct regime that is exactly absent under a clear unique maximum and operates only among genuinely tied or extremely near-tied active cues. Option-specific evidence accumulates with smooth within-coalition redundancy. Compact one- and two-cue opposition receives a display-based diagnosticity bonus, while the credibility of a coalition is computed through a bounded aggregation of its members' validity gaps so that moving one cue cannot induce a quasi-discrete change in competition. The effects of credibility and anchor isolation are likewise softly bounded. At larger opposition multiplicities, compact diagnosticity decays and a saturating restoration process can return choice toward a strong isolated anchor. This restoration receives a smooth pre-sigmoid increment when an effectively isolated anchor faces more than two opponents, while embedded anchors remain protected from that increment. Exact balance versus one-step imbalance can modulate response precision without changing evidence direction, allowing activation of an additional cue to sharpen an already configured coalition preference. Stable subject differences jointly vary validity compression, tied-cue accessibility, compact-dissent sensitivity, restoration, balance sensitivity, and response precision. In the absence of outcome feedback, no aspect of the model updates from trial history.

**Parameters:**
- validities: validities
- validity_sensitivity: [0.08, 0.70]
- validity_floor: [0.58, 0.76]
- order_accessibility: [1.20, 3.80]
- accessibility_scale: [0.10, 0.75]
- order_span: [0.55, 2.50]
- balance_accessibility: [0.55, 1.0]
- balance_scale: [0.45, 1.80]
- redundancy_strength: [0.35, 1.40]
- redundancy_curvature: [1.05, 2.20]
- similarity_sensitivity: [2.0, 16.0]
- compact_dissent: [1.5, 6.5]
- compact_peak: [1.10, 1.60]
- compact_width: [0.45, 0.90]
- compactness_gain: [0.20, 1.50]
- credibility_scale: [0.025, 0.16]
- isolation_gain: [0.25, 1.50]
- restoration_strength: [0.65, 0.97]
- restoration_threshold: [2.25, 3.60]
- restoration_slope: [3.5, 11.0]
- balance_restoration: [0.45, 1.0]
- restoration_credibility: [0.30, 0.70]
- isolation_restoration: [0.15, 0.55]
- anchor_threshold: [0.66, 0.82]
- anchor_slope: [10.0, 35.0]
- balance_precision_gain: [0.75, 1.60]
- balance_precision_width: [0.45, 0.75]
- beta: [1.3, 5.0]
- lapse: [0.0, 0.13]
- configuration_type: {0, 1, 2}

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"S3C expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # S3C is deliberately history-invariant in this feedback-free task.
    _ = history

    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    validity_floor = float(parameters["validity_floor"])
    order_accessibility = float(parameters["order_accessibility"])
    accessibility_scale = float(parameters["accessibility_scale"])
    order_span = float(parameters["order_span"])
    balance_accessibility = float(parameters["balance_accessibility"])
    balance_scale = float(parameters["balance_scale"])
    redundancy_strength = float(parameters["redundancy_strength"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    similarity_sensitivity = float(parameters["similarity_sensitivity"])
    compact_dissent = float(parameters["compact_dissent"])
    compact_peak = float(parameters["compact_peak"])
    compact_width = float(parameters["compact_width"])
    compactness_gain = float(parameters["compactness_gain"])
    credibility_scale = float(parameters["credibility_scale"])
    isolation_gain = float(parameters["isolation_gain"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    balance_restoration = float(parameters["balance_restoration"])
    restoration_credibility = float(parameters["restoration_credibility"])
    isolation_restoration = float(parameters["isolation_restoration"])
    anchor_threshold = float(parameters["anchor_threshold"])
    anchor_slope = float(parameters["anchor_slope"])
    balance_precision_gain = float(parameters["balance_precision_gain"])
    balance_precision_width = float(parameters["balance_precision_width"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated stable types alter several mechanisms together rather than
    # independently injecting trial-wise variability.
    if configuration_type == 0:       # coalition-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.12)
        order_accessibility *= 0.75
        restoration_strength *= 0.65
    elif configuration_type == 1:     # tied-order-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.00)
        order_accessibility *= 1.35
        restoration_strength *= 0.85
    else:                              # intermediate configural integrator
        validity_floor = min(0.90, validity_floor + 0.05)
        order_accessibility *= 0.90
        restoration_strength *= 1.00

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -60.0, 60.0))))

    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    best_active_reliability = float(np.max(reliability[active]))

    active_v = validities[active]
    best_v = float(np.max(active_v))
    # Only genuine or extremely close top ties enter the order regime.
    best_candidates = active[np.isclose(active_v, best_v, atol=0.005)]
    tied_order_regime = bool(best_candidates.size > 1)

    # Identity-specific instruction weights with a bounded strongest-to-weakest
    # ratio. Validity ordering is preserved, but a single anchor cannot become
    # arbitrarily dominant under the exponential reliability transform.
    unbounded_weight = np.exp(
        validity_sensitivity * (reliability - best_active_reliability)
    )
    instruction_weight = (
        validity_floor + (1.0 - validity_floor) * unbounded_weight
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    count_imbalance = abs(n_b - n_a)

    # Serial accessibility is exactly absent under a clear unique maximum.
    # This cleanly separates validity-based choice from tied-order choice.
    near_balance = np.exp(
        -float(count_imbalance) / max(balance_scale, 1e-8)
    )
    access = np.ones(n_features, dtype=float)
    if tied_order_regime:
        for serial_rank, cue in enumerate(active):
            validity_closeness = np.exp(
                -(best_active_reliability - reliability[cue])
                / max(accessibility_scale, 1e-8)
            )
            serial_salience = np.exp(
                -float(serial_rank) / max(order_span, 1e-8)
            )
            activation_gate = (
                (1.0 - balance_accessibility)
                + balance_accessibility * near_balance
            )
            access[cue] += (
                order_accessibility
                * validity_closeness
                * serial_salience
                * activation_gate
            )

    accessible_weight = instruction_weight * access

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        weights = np.asarray(accessible_weight[indices], dtype=float)
        raw = float(np.sum(weights))
        dominant = max(float(np.max(weights)), 1e-12)
        effective_multiplicity = raw / dominant
        excess = max(effective_multiplicity - 1.0, 0.0)

        coalition_validities = validities[indices]
        spread = (
            float(np.std(coalition_validities)) if n > 1 else 0.0
        )
        similarity = np.exp(-similarity_sensitivity * spread)
        redundancy = 1.0 + redundancy_strength * similarity * (
            excess ** redundancy_curvature
        )
        return raw / max(redundancy, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # A communicated-validity tie is resolved by bounded current accessibility,
    # not by an unconditional first-cue rule.
    candidate_access = accessible_weight[best_candidates]
    anchor_cue = int(best_candidates[int(np.argmax(candidate_access))])
    anchor_direction = float(directions[anchor_cue])

    if anchor_direction > 0:
        anchor_indices = b_cues
        opponent_indices = a_cues
        anchor_support = support_b
        opponent_support = support_a
    else:
        anchor_indices = a_cues
        opponent_indices = b_cues
        anchor_support = support_a
        opponent_support = support_b

    anchor_count = int(anchor_indices.size)
    opponent_count = int(opponent_indices.size)

    def effective_multiplicity(indices):
        if indices.size == 0:
            return 0.0
        weights = np.asarray(accessible_weight[indices], dtype=float)
        return float(np.sum(weights)) / max(float(np.max(weights)), 1e-12)

    anchor_effective_n = effective_multiplicity(anchor_indices)
    opponent_effective_n = effective_multiplicity(opponent_indices)

    if opponent_count > 0:
        opponent_v = validities[opponent_indices]

        # Each member's validity gap is smoothly bounded before aggregation.
        # Consequently, adding or moving one relatively strong cue cannot
        # multiply coalition credibility by an arbitrarily large amount.
        raw_validity_gaps = np.maximum(best_v - opponent_v, 0.0)
        bounded_gap = float(np.mean(np.tanh(
            raw_validity_gaps / max(credibility_scale, 1e-8)
        )))
        opponent_credibility_gate = np.exp(-bounded_gap)

        # Absolute serial position can affect compactness only inside the tied
        # order regime. It is exactly disabled for clear unique maxima.
        serial_span = int(np.max(opponent_indices) - np.min(opponent_indices) + 1)
        compactness = float(opponent_count) / float(max(serial_span, 1))
        effective_compactness_gain = (
            compactness_gain if tied_order_regime else 0.0
        )

        count_distance = (
            opponent_effective_n - compact_peak
        ) / max(compact_width, 1e-8)
        compact_count_profile = np.exp(-0.5 * count_distance ** 2)

        # Isolation remains strongest for a lone anchor, but its contribution
        # is bounded rather than multiplying challenge without limit.
        isolation = 1.0 / float(max(anchor_count, 1))
        raw_isolation_bonus = isolation_gain * isolation
        bounded_isolation_bonus = raw_isolation_bonus / (
            1.0 + raw_isolation_bonus
        )
        challenge_multiplier = 1.0 + (
            compact_dissent
            * compact_count_profile
            * opponent_credibility_gate
            * (1.0 + effective_compactness_gain * compactness)
            * (1.0 + bounded_isolation_bonus)
        )
        opponent_support *= challenge_multiplier

    total = anchor_support + opponent_support
    if total <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total
        core_evidence = anchor_direction * anchor_margin

    if anchor_count > 0:
        anchor_identity_share = float(accessible_weight[anchor_cue]) / max(
            float(np.sum(accessible_weight[anchor_indices])), 1e-12
        )
    else:
        anchor_identity_share = 0.0

    # The isolated-anchor increment now enters the pre-sigmoid restoration
    # scale. It turns on smoothly above two effective opponents and saturates,
    # while the cubed identity share protects embedded anchors.
    isolated_transition = sigmoid(
        restoration_slope * (opponent_effective_n - 2.35)
    )
    restoration_latent = (
        restoration_slope
        * (opponent_effective_n - restoration_threshold)
        + 4.0
        * isolation_restoration
        * (np.clip(anchor_identity_share, 0.0, 1.0) ** 3)
        * isolated_transition
    )
    high_count_gate = sigmoid(restoration_latent)
    balance_crossing = sigmoid(
        restoration_slope
        * ((opponent_effective_n - anchor_effective_n) - 0.5)
    )
    count_gate = max(
        high_count_gate,
        balance_restoration * balance_crossing
    )

    absolute_anchor_gate = sigmoid(
        anchor_slope * (best_v - anchor_threshold)
    )
    credibility_gate = (
        restoration_credibility
        + (1.0 - restoration_credibility) * absolute_anchor_gate
    )

    # Embedded anchors can be restored, but isolated anchors express the
    # effect most strongly.
    embedding_factor = 0.35 + 0.65 * np.sqrt(
        np.clip(anchor_identity_share, 0.0, 1.0)
    )
    restoration_gate = (
        restoration_strength
        * count_gate
        * credibility_gate
        * embedding_factor
    )
    restoration_gate = float(np.clip(restoration_gate, 0.0, 0.97))

    choice_evidence = (
        (1.0 - restoration_gate) * core_evidence
        + restoration_gate * anchor_direction
    )

    # Moving one member away from exact balance can sharpen an already formed
    # directional preference. This display-only modulation changes precision,
    # not evidence direction, and is disabled in the tied-order regime.
    if tied_order_regime:
        effective_beta = beta
    else:
        width = max(balance_precision_width, 1e-8)
        one_step_profile = np.exp(
            -0.5 * ((float(count_imbalance) - 1.0) / width) ** 2
        )
        balanced_profile = np.exp(
            -0.5 * (float(count_imbalance) / width) ** 2
        )
        precision_shift = balance_precision_gain * (
            one_step_profile - balanced_profile
        )
        effective_beta = beta * np.exp(
            float(np.clip(precision_shift, -1.8, 1.8))
        )

    logits = effective_beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    normalizer = float(probs.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= normalizer
    return probs.astype(np.float64)
```

**`policy source code`:**
```python
def policy(probs):
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
[0] Eight experts have strictly ordered, widely separated validities. Every pair creates a conflict between Take The Best and Tallying: the highest-validity discriminating expert favors one option, while the majority of discriminating experts favors the other. The design orthogonally balances whether A or B is favored by the decisive expert and manipulates the opposing tally margin. In trials where Expert 1 discriminates, the tally margins are 2, 4, and 6; where Expert 1 ties and Expert 2 is the first discriminating expert, the margins are 1, 3, and 5. Take The Best therefore predicts the same choice probability for the option favored by the first discriminating expert on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), invariant to both cue rank and tally margin. Tallying predicts that this option is chosen below chance, with probability epsilon/2 + (1-epsilon)*sigmoid(-beta*m), and predicts a systematic decrease as the opposing margin m increases. Thus, the critical dissociation is both directional and quantitative: invariant support for the top-cue winner under TTB versus increasingly strong rejection of it under Tallying. Swapping A and B for each evidence pattern controls response-side preferences. The 12 unique pairs yield exactly 96 trials with eight repetitions per pair.

[1] This design targets the parameter-free prediction of Tallying on exactly balanced evidence. On every trial, A and B win on the same number of discriminating features, so Tallying predicts P(A)=P(B)=0.5 for every beta and epsilon. Nevertheless, each pair has a uniquely defined first discriminating expert in descending-validity order. Take The Best therefore predicts choice of that expert's favored option with probability epsilon/2 + (1-epsilon)*sigmoid(beta), which is strictly above chance throughout the stated parameter ranges. The six underlying evidence patterns vary both the rank of the first discriminating expert (1, 3, or 5) and the total number of discriminating experts (2, 4, 6, or 8), while preserving an exactly tied tally. Take The Best predicts the same above-chance preference across these manipulations, whereas Tallying remains exactly at chance. Each pattern is paired with an A/B reversal, controlling response-side bias. Thus the critical statistic is the probability of choosing the option favored by the first discriminating expert: exactly 0.5 under Tallying versus a common value above 0.5 under Take The Best. The 12 unique pairs are repeated eight times for 96 trials.

[2] The design exploits a sharp conflict-gated crossover unique to CGRDA while holding Take The Best's decisive cue constant. Expert 5 has the highest validity (0.95) and always discriminates in favor of the designated TTB winner. Each family contains an exactly balanced 2-versus-2 trial and a matched 3-versus-1 trial differing only in Expert 1's direction. On balanced trials, the final discriminating expert (Expert 6) opposes Expert 5. TTB therefore chooses Expert 5's option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) on every trial. CGRDA instead predicts the Expert-6 option on every balanced trial throughout its parameter ranges: tally balance fully opens the recency gate, and recency_strength is at least 1.15 while normalized core evidence is bounded in magnitude by 1. On matched imbalanced trials, reversing Expert 1 creates three endorsements for the TTB winner versus one final endorsement for its opponent. The imbalance of two suppresses recency by at least exp(-5), while the early Expert-1 endorsement and two additional concordant endorsements make CGRDA favor the TTB winner. Thus, the critical statistic is the probability of choosing the Expert-5 winner: TTB predicts invariance between balanced and imbalanced conditions, whereas CGRDA predicts a crossover from below chance on balanced trials to above chance on imbalanced trials. Moving the second pro-TTB endorsement across Experts 2, 3, and 4 provides an additional serial-position manipulation: TTB remains invariant, while CGRDA's integrated evidence changes with primacy. Exact A/B reversals control response-side bias. The 12 unique pairs are each repeated eight times, yielding 96 trials.

[3] This design isolates CGRDA's serial-position weighting and redundancy discounting while holding constant every determinant of Take The Best. All six cues discriminate, Expert 5 (index 4) is uniquely most valid at 0.95, and it always favors the designated TTB winner. Every trial also has the same 4-versus-2 endorsement tally, and the final cue always favors the TTB winner. TTB therefore predicts one common probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) of selecting the designated winner across all patterns. The two cues opposing that winner are moved among the first four serial positions. CGRDA predicts a sharp position-dependent reversal: when the compact opposing pair includes the highly reliable first-position cue (sets {1,2}, {1,3}, or {1,4} in one-based numbering), primacy and redundancy discounting allow those two early endorsements to defeat four later, partly redundant endorsements, producing below-chance selection of the TTB winner. When the opposing pair excludes the first cue ({2,3}, {2,4}, or {3,4}), the TTB winner receives the dominant first-position endorsement and should be selected above chance. This is not a recency-gate effect: the tally imbalance is always two, limiting the gate to exp(-5) or less, and the last cue's direction is fixed. Thus the critical quantitative statistic is the change in TTB-winner choice probability depending on whether Expert 1 opposes or supports that winner: TTB predicts exactly zero change, whereas CGRDA predicts a large crossover. Exact A/B reversals control response-side preference. The 12 unique pairs are repeated eight times for 96 trials.

[4] The validity profile contains four validity-matched cue pairs but remains broad enough to force DCSC into its compensatory regime. Using the most dominance-favoring allowed profile-shape weight, the profile's dominance score is approximately 0.212, still below the minimum dominance threshold of 0.285; hence the dominance gate is at most about 0.006. On every trial, the two highest-validity experts discriminate in opposite directions. Because their validities are tied, Take The Best uses its stable ordering rule and follows Expert 1, predicting the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) for Expert 1's option on every trial. DCSC instead averages the opposing tied anchors, making anchor evidence exactly zero. One or more lower-validity experts then unanimously support Expert 2's option, so DCSC's coalition evidence always points against the TTB winner and predicts selection of the TTB winner below chance. Nested conditions increase the size of this opposing coalition from one to six lower experts, providing a graded DCSC signature while TTB remains exactly invariant. The two single-supporter conditions also contrast a relatively reliable lower expert (82%) with a weaker one (65%), testing DCSC's compressed reliability weighting without affecting TTB. Each pattern has an exact A/B reversal, controlling response-side preference. The 12 unique pairs are repeated eight times, yielding 96 trials.

[5] The broad validity profile forces DCSC into its compensatory regime while preserving a unique highest-validity cue. With diagnosticities [0.70, 0.64, 0.58, 0.52, 0.46, 0.40, 0.34, 0.20], the maximum DCSC dominance score is approximately 0.215, below the minimum dominance threshold of 0.285; even at the least decisive allowed gap sensitivity, the dominance gate is below 0.008. In every underlying pattern, Expert 1 discriminates for a designated top-cue winner, so Take The Best always selects that option with the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of all lower cues. DCSC predicts a distinctive crossover. When only one lower expert opposes Expert 1, the top-cue winner remains weakly favored because its reliability weight is larger. This preference should be stronger when the lone opponent is the weakest expert (Expert 8) rather than Expert 2, directly testing compressed reliability weighting. Once two or more lower experts form the opposing coalition, DCSC predicts the coalition's option above chance, with increasingly strong but diminishing support as the coalition grows from two to three, five, and seven experts. Thus TTB predicts complete invariance, whereas DCSC predicts both a directional crossover between one- and multi-opponent trials and a nonlinear coalition-size gradient. Unlike the previously explored tied-anchor design, the highest-validity anchor is unique here and DCSC initially agrees with TTB before lower-cue accumulation overturns it. Exact A/B reversals control response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

[6] This design targets ADCG's history-dependent learning of selective anti-concurrence, a mechanism entirely absent from Take The Best. Expert 1 is the unique validity anchor (95%) and favors the designated TTB winner in every pair, so TTB predicts one constant probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) of choosing that option, invariant across stimulus type, repetition, and trial position. Five diagnostic-opposition patterns share directions [anchor, anchor, anchor, tie, dissent, tie, tie]: the late Expert 5 repeatedly opposes Experts 1–3 whenever it discriminates. A control pattern has directions [anchor, dissent, anchor, tie, tie, tie, tie]: Expert 2 opposes the anchor there but concurs with it throughout the five diagnostic patterns. Across the randomized block, Expert 5 therefore acquires strong, well-observed anti-concurrence with the anchor coalition, whereas Expert 2's occasional opposition remains below ADCG's anti-concurrence threshold because it usually concurs with that coalition. Experts 2 and 5 have identical advertised validity (72%), and the diagnostic dissenter is presented later, making its predicted learned advantage conservative rather than a simple validity or order effect. ADCG predicts an experience-dependent reduction in choices of the Expert-1 winner on diagnostic-opposition trials: learned diagnostic gain amplifies Expert 5, attenuates the unique early anchor, and suppresses residual anchoring. The corresponding change should be much smaller on control trials and can produce a growing diagnostic-versus-control gap, potentially including below-chance selection of the TTB winner late in the block. TTB predicts exactly zero condition difference and zero early-to-late change. The five diagnostic pairs differ only in values shared by A and B on nondiscriminating cues, so ADCG treats their current directional evidence identically while their multiplicity establishes the intended 5:1 history distribution. Exact A/B reversals control response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials; the critical analysis is the probability of choosing the Expert-1 winner by condition and experience bin.

[7] This design targets ADCG's learned distributional-rarity weighting while holding learned anti-concurrence near chance, separating the proposed rarity mechanism from the anti-concurrence manipulation explored previously. Expert 1 is uniquely most valid (95%) and discriminates on every trial, so Take The Best always follows Expert 1 and predicts the same probability q = epsilon/2 + (1-epsilon)*sigmoid(beta) for its favored option across stimulus types and experience. The critical matched targets are patterns 1–2 and their A/B reversals. In both, Experts 1 and 3 support the TTB winner and one 72%-valid dissenter opposes them; the dissenter is Expert 2 in the common-dissenter target and Expert 5 in the rare-dissenter target. Across the complete block, Expert 2 discriminates in four of six underlying patterns, whereas Expert 5 discriminates in only two. Whenever either dissenter and the anchor coalition are jointly active, it concurs and opposes equally often, keeping pairwise anti-concurrence around 0.5 and below ADCG's threshold. Early in the block, before distributional frequencies are learned, ADCG predicts stronger opposition from Expert 2 because it is presented earlier than the equally valid Expert 5; hence fewer Expert-1-winner choices in the common-dissenter target. With experience, Expert 5 becomes approximately twice as rare as Expert 2. Even at the minimum distribution-attention setting, its rarity advantage outweighs Expert 2's maximum order-prominence advantage; rarity also selectively suppresses the profile gate that otherwise preserves the anchor. ADCG therefore predicts a reversal of the target contrast: late in the block, Expert-1-winner choices should be lower with the rare Expert-5 dissenter than with the common Expert-2 dissenter. The critical quantitative interaction is thus (rare minus common) by experience bin: TTB predicts exactly zero throughout, whereas ADCG predicts a change from positive early to negative late. The remaining patterns establish the intended discrimination frequencies and balance concurrence without feedback. Exact A/B reversals control response-side preference. Twelve unique pairs repeated eight times yield 96 trials.

[8] This design isolates ADCG's learned within-coalition dependence while SCCI remains history-invariant. Expert 5 is the unique 95%-valid anchor. Experts 1–4 all have validity 72%, arranged as an outer pair (Experts 1 and 4) and a middle pair (Experts 2 and 3). Across the four history-building patterns and their A/B reversals, Experts 1 and 4 always discriminate concordantly, whereas Experts 2 and 3 always discriminate in opposite directions. Each lower expert nevertheless concurs with the anchor on half of these patterns and opposes it on half, equating lower-cue validity, discrimination frequency, and relationships with the anchor. The two critical targets are configuration-matched: the anchor supports one product while either the outer pair or the middle pair unanimously supports the other; the unused pair is nondiscriminating. SCCI assigns identical evidence to these targets because both contain one isolated 95% anchor against two 72% opponents. Its large validity gap activates isolated-anchor restoration, and it predicts the same stable, above-chance preference for the anchor's product in both targets at every point in the experiment. ADCG instead learns that the outer pair is highly concurrent and therefore redundant, while the middle pair is usually anti-concurrent and receives little historical redundancy discount when it happens to agree on its target. Relationships between each target cue and the anchor remain approximately 60% opposing—below ADCG's anti-concurrence thresholds—and are matched across pairs, so the key difference is coalition dependence rather than diagnostic dissent. Distributional rarity is also matched: every lower expert discriminates on five of the six underlying patterns. Presentation positions are deliberately balanced in summed prominence (outer positions 1 and 4 versus middle positions 2 and 3); convex order weighting may make the outer coalition slightly stronger initially, but learned redundancy should decisively reverse that contrast. Thus ADCG predicts an experience-dependent increase in P(anchor choice | outer-pair target) relative to P(anchor choice | middle-pair target), potentially with the middle coalition defeating the anchor late in the block. SCCI predicts an exactly zero target contrast and zero experience interaction. Because the first active cue on either target belongs to the two-cue opposing coalition rather than the late anchor, ADCG's minority-first profile gate is closed, exposing the coalition-learning effect in its core comparison. Exact A/B reversals control response-side preference. The 12 unique pairs are repeated eight times, yielding 96 trials.

[9] This design isolates ADCG's selective use of the strongest reliably learned opposing relationship in a compact two-cue coalition, together with diagnostic dissent gain and attenuation of an early unique anchor. Expert 1 is the unique 95%-valid anchor and discriminates on every trial. Experts 2, 3, 5, and 6 all have validity 72% and each discriminates in exactly five of the six underlying patterns, equating their advertised reliability and asymptotic rarity. The first two patterns are the critical targets. In the diagnostic target, Experts 3 and 5 oppose the anchor; in the control target, Experts 2 and 6 oppose it. Both targets therefore present exactly one 95% anchor against two 72% opponents. SCCI assigns them identical evidence and predicts an exactly zero target difference at every experience level because it ignores presentation position and history, and its isolated-anchor restoration depends only on the shared validity gap and coalition counts. Across the four history-building patterns, Expert 3 always opposes Expert 1, so their anti-concurrence approaches its reliable, above-threshold regime. Each of Experts 2, 5, and 6 instead opposes Expert 1 on only two of four builders; including its target occurrence, each has three oppositions and two concurrences with the anchor, leaving its smoothed anti-concurrence below ADCG's threshold. On the diagnostic target, ADCG's reliability-weighted selective maximum can consequently detect the strong Expert-3 relationship without diluting it by Expert 5's weak relationship. This suppresses the profile gate protecting the isolated early anchor, amplifies Expert 3's dissent, and attenuates Expert 1 in the coalition core. Neither control opponent receives these learned effects. Historical concurrence within the two target coalitions is matched: Experts 3 and 5 concur on three of their five joint observations, as do Experts 2 and 6. Thus learned within-coalition redundancy cannot explain the critical contrast. The main quantitative prediction is an experience-dependent decrease in anchor choices on the diagnostic target relative to the control target: the late-minus-early change in [P(anchor choice | diagnostic) - P(anchor choice | control)] should be negative under ADCG but exactly zero under SCCI. Any initial serial-position difference can be estimated in the early trials rather than mistaken for learning; the opposing pairs also have matched summed prominence. Exact A/B reversals preserve every cue relationship while controlling response-side bias. The 12 unique pairs are repeated eight times for 96 trials.

[10] This design targets PDRR's sharp redundancy-restoration threshold for an isolated validity anchor. Expert 8 is the unique, weak-absolute anchor (74%) and favors the designated anchor option; between one and six earlier experts favor the other option. The opponents are individually close to the anchor in validity, maximizing PDRR's compact-challenge mechanism while keeping the validity and credibility gaps below its selective-restoration thresholds. PDRR therefore predicts low anchor-choice probability with one or two opponents, followed by a large discontinuous reversal at exactly three opponents: its count-dominant restoration gate is nearly off at two but approximately 0.98 or higher at three throughout the stated restoration-slope range. It should continue to protect the anchor against four to six opponents. The two identity variants at counts two and three permit the threshold contrast to be estimated without relying on one particular cue subset. ADCG has no corresponding two-to-three restoration branch. Because the validity anchor is presented last and the earliest active cue belongs to the opposing majority, ADCG's minority-first profile gate is closed. Its coalition core instead gives the earlier opposition distinctiveness and dissent bonuses; accumulated anti-concurrence with the anchor and rarity weighting should, if anything, strengthen that opposition over experience. ADCG thus predicts no abrupt upward jump in anchor choices at three opponents and generally continued preference for the opposing coalition. The critical statistic is P(anchor option | three opponents) minus P(anchor option | two opponents): PDRR predicts a large positive discontinuity, whereas ADCG predicts a much smaller, absent, or negative change. Counts one and four through six establish the broader step-function signature rather than a smooth coalition-size trend. Exact A/B reversals control response-side preference. Sixteen unique pairs repeated six times yield 96 trials.

[11] This design exploits a qualitative disagreement about an inactive co-top-validity expert. Experts 4 and 6 are tied at 90%, but Expert 4 is presented earlier. Expert 4 always favors the designated anchor option. In each off/on matched pair, the active evidence changes only by Expert 6: in the off condition Expert 6 is nondiscriminating, whereas in the on condition Expert 6 favors the opposing option. The remaining evidence is three-versus-three when Expert 6 is off and three-versus-four when it is on; Expert 1 always belongs to the opposing coalition, so ADCG's minority-first profile gate is not activated. PDRR treats the off condition as having a unique active anchor and therefore supplies no order gate. Because its ordinary validity compression makes the equal-sized coalitions nearly exchangeable, it predicts only a small Expert-4 advantage. Activating Expert 6 creates an exact active top-validity tie, causing PDRR's bounded order gate to favor the earlier Expert 4 despite adding strong evidence to the opposing coalition. PDRR therefore predicts a positive change in anchor-option choice from off to on. ADCG instead computes its global-top-tie gate from the full instructed validity profile, including nondiscriminating experts. Its tie gate is consequently already present in the off condition and does not newly appear when Expert 6 becomes active. Adding the equally valid Expert 6 to the opposition should reduce anchor choices; repeated perfect opposition between Experts 6 and 4 can further amplify this reduction through learned anti-concurrence. Thus the critical quantitative statistic is P(anchor option | Expert 6 on) minus P(anchor option | Expert 6 off): PDRR predicts a positive contrast, whereas ADCG predicts a negative contrast, potentially becoming more negative with experience. Three lower-cue allocations test generality across serial composition, and exact A/B reversals control response-side preferences. The 12 unique pairs are repeated eight times for 96 trials.

[12] This design targets ACAC's special symmetric normalization for unique-anchor, multi-cue-versus-multi-cue profiles. Expert 6 is the unique 95% validity anchor and always supports the designated anchor option. Expert 1 (85%) always supports the opposing option and is therefore the strongest opponent in every trial. The critical manipulation swaps Expert 3 (82%) and Expert 4 (56%) between the two coalitions while holding the active cue set, coalition counts, anchor identity, strongest-opponent identity, and anchor-to-opponent validity gap constant. In the high-partner condition, the anchor coalition contains Experts 6 and 3, opposed by Experts 1 and 4. In the weak-partner condition, it contains Experts 6 and 4, opposed by Experts 1 and 3. Both configurations are exactly two-versus-two. ACAC consequently replaces both coalitions' supports with the same count-normalized quantities; its credibility protection and challenge terms are also identical because the 95% anchor and 85% strongest opponent are unchanged. ACAC therefore predicts an exactly zero contrast in probability of choosing the anchor option between the high-partner and weak-partner conditions, notwithstanding its potentially nonzero overall anchor preference. ADCG preserves cue-specific validity and serial weights. Expert 3 is both much more valid and earlier than Expert 4, so assigning Expert 3 to the anchor coalition simultaneously strengthens that coalition and removes a strong cue from the dissent coalition. Conversely, assigning Expert 3 to the opposition gives this already stronger cue ADCG's dissent amplification. ADCG therefore predicts substantially more anchor choices in the high-partner than in the weak-partner condition, potentially producing a choice crossover. Historical discrimination frequency is matched because Experts 1, 3, 4, and 6 discriminate on every trial. The schedule also symmetrically balances concurrence relationships involving Experts 3 and 4, while Expert 1's diagnostic opposition to the anchor is common to both conditions. Experts 2 and 5 are nondiscriminating fillers whose shared values generate three perceptually distinct versions of each profile without changing either theory's directional evidence. Exact A/B reversals control response-side preferences. The critical statistic is P(anchor choice | Expert 3 partners with anchor) minus P(anchor choice | Expert 4 partners with anchor): ACAC predicts exactly zero, whereas ADCG predicts a positive contrast. The 12 unique pairs are repeated eight times for 96 trials.

[13] This design isolates ADCG's reliability-sensitive learning of anti-concurrence from discrimination-frequency rarity. Expert 1 is the unique 95%-valid anchor. Experts 2 and 5 have identical 74% validity and discriminate equally often across the block, but Expert 5 is deliberately presented later. The two critical targets are otherwise matched one-versus-one conflicts: Expert 1 supports the anchor option while either Expert 5 (diagnostic target) or Expert 2 (control target) supports the other option. ACAC assigns these targets exactly the same evidence: both have the same unique anchor, opponent validity, validity gap, absolute credibility, coalition counts, and singleton-challenge geometry. Because the maximum validity is unique, ACAC does not use presentation order, and because it excludes history learning, it predicts an exactly zero diagnostic-minus-control target contrast at every experience level. The history-building trials equate Experts 2 and 5 in marginal discrimination frequency. Expert 5, however, repeatedly opposes Expert 1 whenever they discriminate together, providing many observations of their perfectly anti-concurrent relationship. Expert 2 also opposes Expert 1 on its target, but its five builders occur while Expert 1 is nondiscriminating; hence its relationship with the anchor remains much less well observed. This distinguishes confidence in a learned relationship from merely estimating a high opposition proportion. ADCG initially favors the control dissenter because Expert 2 appears much earlier than the equally valid Expert 5. With experience, the well-observed Expert-1/Expert-5 opposition crosses ADCG's reliability-sensitive anti-concurrence gate more strongly, amplifies Expert 5's dissent, suppresses the anchor-protecting profile gate, and attenuates the unique early anchor. Thus ADCG predicts that [P(anchor choice | diagnostic target) - P(anchor choice | control target)] decreases with cumulative experience and may reverse sign, whereas ACAC predicts this contrast and its experience interaction to be exactly zero. Diagnostic and control builders both contain three active cues, while varied shared ratings create distinct displays without changing directional evidence. Every profile has an exact A/B reversal. The 24 unique pairs are repeated four times, yielding 96 trials.

[14] This design exploits the theories' different definitions of a validity tie. Expert 8 is the unique exact maximum at 95%, but the early Expert 1 is only 0.005 percentage points lower (94.995%). S3C treats these experts as near-tied because their validity difference is below its 0.005 tolerance. On each of the three critical profiles, Expert 1 favors A and is the earliest active near-top cue; Expert 8 and three lower experts favor B, while one weak expert joins Expert 1. S3C's tied-order accessibility therefore makes Expert 1 the configured anchor. The four-cue opposition is beyond the compact-dissent peak and activates S3C's high-multiplicity restoration toward that early anchor, predicting above-chance choice of A despite its 2-versus-4 minority. ADCG does not treat the two top experts as exactly tied when selecting its lexicographic cue: Expert 8 is the unique anchor and favors B. At the same time, their extremely small validity gap makes ADCG's global near-tie gate almost maximal, strongly interpolating toward Expert 8's B direction; the 4-versus-2 tally also supports B. ADCG therefore predicts below-chance choice of A on the critical profiles. The primary quantitative statistic is mean P(A) over the first six listed trials after coding reversals back to the Expert-1 option: S3C predicts a positive log-odds preference, whereas ADCG predicts a negative one. The three companion profiles make Experts 1 and 8 concur and balance their historical relationship: across the six underlying patterns they oppose three times and concur three times. Each weak cue that joins Expert 1 on a critical profile also concurs with Expert 8 on exactly one companion profile and is otherwise inactive there. Thus ADCG should not acquire reliable above-threshold anti-concurrence that could erase the static dissociation. The three critical variants rotate the weak ally and the lower members of the opposing coalition, testing generality across cue identity and validity. Every profile has an exact A/B reversal, controlling response-side bias. Twelve unique pairs repeated eight times yield 96 trials.

[15] This design isolates ADCG's serial-prominence weighting under a clear unique validity maximum, where S3C specifies that serial accessibility is exactly absent. Expert 4 is the unique 95%-valid anchor. The critical conflict is always one 95% anchor against one 72% opponent. In the early-opponent condition, Expert 1 opposes the anchor; in the late-opponent condition, Expert 7 opposes it. Experts 1 and 7 have identical communicated validity, and all other experts are nondiscriminating. Consequently, S3C assigns the two conditions exactly identical instruction weights, singleton coalition supports, compact-dissent bonuses, credibility gates, and restoration gates. Because the active maximum is clearly unique, its tied-order regime is disabled, yielding the parameter-wise prediction P(anchor choice | late opponent) - P(anchor choice | early opponent) = 0. ADCG instead applies order prominence even under a unique maximum. Expert 1 therefore receives greater accessibility than the otherwise matched Expert 7, so ADCG predicts fewer anchor choices against the early opponent: P(anchor choice | late opponent) - P(anchor choice | early opponent) > 0. A choice crossover is possible but is not required for the test. The block equates the two opponents' discrimination frequencies and gives each the same perfectly opposing relationship with the anchor, matching learned rarity and anti-concurrence. Three perceptually distinct versions of each condition vary only ratings shared by A and B. Exact A/B reversals control response-side bias. The 12 unique pairs are repeated eight times for 96 trials; the primary analysis recodes choices relative to the Expert-4 option and compares the two opponent-position conditions, with experience bin included to verify that the contrast remains after matched learning.

[16] This design exploits RBIS's saturated six-cue override while testing S3C's continuous coalition competition. Expert 6 is the unique validity maximum (96%), and all six experts discriminate on every trial. Expert 6 defines the coded anchor option, with two to five other experts opposing it. In RBIS, every profile therefore satisfies the terminal rule for a fully discriminating six-cue display with a unique exact maximum and nonsingleton opposition. That rule overwrites earlier state computations with protection_strength times Expert 6's direction. Consequently, RBIS predicts exactly the same anchor-choice probability on all six profiles, irrespective of opponent number or identity: after recoding A/B reversals, P(anchor)=lapse/2+(1-lapse)*sigmoid(beta*protection_strength). S3C has no corresponding saturated-display override. The first fixed-count contrasts exchange highly valid opponents for weak opponents: Experts 1-2 versus Experts 4-5 at opposition count two, and Experts 1-3 versus Experts 3-5 at count three. S3C predicts fewer anchor choices against the more valid coalition because it has greater instruction weight and credibility and simultaneously removes stronger cues from the anchor coalition. The count series from two through five opponents additionally targets S3C's nonmonotonic configuration response: compact two-cue dissent receives its largest diagnostic bonus, whereas larger coalitions lose that compact bonus and progressively activate smooth restoration toward the uniquely valid anchor. Thus S3C predicts substantial profile-dependent variation, including validity-sensitive fixed-count contrasts and recovery toward Expert 6 at high multiplicity, while RBIS predicts a perfectly flat profile. Exact A/B reversals control response-side bias. The 12 unique pairs are repeated eight times for 96 trials.

[17] This design isolates RBIS's exact-balanced coalition-geometry rule under a clear unique validity maximum. Expert 4 (95%) and the weak Expert 3 (55%) always support the coded anchor option, while exactly two 86%-valid experts support the opposing option. Thus every profile contains four active reasons in an exact 2-versus-2 balance and has the same validity multiset on each coalition. The critical manipulation changes only the serial span of the opposing pair. In the broad condition the opponent pairs are Experts {1,7}, {1,8}, and {2,8}; their index spans are at least 80% of the seven-step display span. In the nonbroad condition the pairs are {1,5}, {2,6}, and {5,7}, which fall below that threshold. RBIS therefore applies its full exact-balance resolution tendency to the broad pairs but only 54% of that tendency to the nonbroad pairs, shifting broad displays more strongly toward the unique Expert-4 anchor. All computations before this geometry rule are matched because opponent count, coalition validities, band ranks, anchor identity, and anchor-partner identity are identical. S3C predicts no broad-versus-nonbroad contrast: the 9-point validity gap disables its tied-order regime, and serial span and compactness are exactly ignored under a clear unique maximum. Its coalition support, diagnostic challenge, credibility, restoration, and precision calculations are therefore identical across all six profiles. The primary statistic is P(Expert-4 option | broad opposition) minus P(Expert-4 option | nonbroad opposition): RBIS predicts a geometry-dependent contrast, ordinarily positive because stronger balance resolution pulls choices toward Expert 4, whereas S3C predicts exactly zero for every parameter setting. Subject-level geometry contrasts are also informative because RBIS permits stable heterogeneity in the magnitude, and occasionally the sign, of this effect, while S3C permits no stimulus-geometry effect at all. Exact A/B reversals control response-side preference. The 12 unique pairs are repeated eight times for 96 trials.

[18] This design isolates S3C's tied-regime spatial-compactness mechanism while making CRGRC exactly invariant. On every trial, four 88%-valid experts discriminate. The first two active experts (Experts 1 and 3) support the coded anchor option, and the final two active experts support its opponent. Thus the ordered active sequence, communicated validities, coalition sizes, cue weights, serial ranks, tally balance, and direction sequence are identical across conditions. Only the absolute span of the opposing pair changes: Experts 9-10 form a compact pair (compactness 2/2 = 1), Experts 7-10 form an intermediate pair (2/4 = 0.5), and Experts 5-10 form a broad pair (2/6 = 0.333). Because all active experts tie for maximum validity, S3C enters its tied-order regime, selects the earliest accessible top cue as the anchor, and allows absolute opponent span to amplify compact dissent. All other S3C quantities are matched, so it predicts a strict coded-anchor gradient: P(anchor | compact) < P(anchor | intermediate) < P(anchor | broad). CRGRC uses active serial rank but not absolute cue-index span. Each condition presents it with the identical active validity sequence [0.88, 0.88, 0.88, 0.88] and direction sequence [anchor, anchor, opponent, opponent], so it predicts exactly equal coded-anchor probabilities across all three conditions for every parameter setting. Exact A/B reversals control response-side preferences. The six unique pairs are repeated 16 times each, yielding 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design targets CRGRC's continuous near-tie serial-accessibility gate under a validity gap that is just outside S3C's categorical tied-order regime. Expert 8 is the unique maximum at 0.9500. Experts 1 and 7 are validity-matched at 0.9448, only 0.0052 below the maximum: this exceeds S3C's 0.005 near-tie tolerance (including NumPy's negligible relative tolerance), so S3C treats Expert 8 as a clear unique maximum and sets serial accessibility exactly to zero. Each matched contrast swaps the directions of Experts 1 and 7 while keeping both active. Because they have identical validity, this swap preserves coalition counts, each coalition's validity multiset, redundancy, credibility, restoration, count imbalance, and all other S3C calculations. S3C therefore predicts exactly zero difference between conditions. CRGRC instead has a continuous Gaussian top-ambiguity gate; a 0.0052 gap remains appreciably ambiguous throughout its tie-width range. Expert 1 is the first active cue and Expert 7 is near the end of the active sequence, so CRGRC's exponentially declining serial accessibility gives Expert 1 much greater leverage. It therefore predicts more choices of Expert 8's coded option when Expert 1 supports that option and Expert 7 opposes it than when their directions are exchanged. Importantly, CRGRC's coalition support, challenge, and restoration terms are also exactly matched across each swap; only serial evidence changes. Three six-active-cue families vary the lower-validity coalition composition while preserving exact 3-versus-3 balance, which strengthens CRGRC's serial gate and tests generality. Exact A/B reversals control response-side preference. The critical statistic is P(coded Expert-8 option | Expert 1 supports it) minus P(coded Expert-8 option | Expert 7 supports it): CRGRC predicts a positive contrast, whereas S3C predicts exactly zero for every parameter setting. The 12 unique pairs are repeated eight times for 96 trials.",
  "validities": [
    0.9448,
    0.84,
    0.76,
    0.68,
    0.6,
    0.84,
    0.9448,
    0.95
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
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
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
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
      1,
      1,
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
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
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
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 15947,
  "output_tokens": 4327
}
```
