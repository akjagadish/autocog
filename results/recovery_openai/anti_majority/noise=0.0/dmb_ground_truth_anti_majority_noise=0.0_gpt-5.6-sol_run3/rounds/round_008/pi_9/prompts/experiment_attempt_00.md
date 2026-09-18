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


## COMPETING THEORY
**Description:** Reliability-Band Interpretation Switching (RBIS) proposes that communicated cue validities are encoded as ordinal reliability bands rather than exact numerical weights. On each display, the decision maker identifies the highest occupied band and an initial anchor reason. If several active cues occupy that band, they are functionally tied and serial order selects the provisional anchor; if only one cue occupies it, serial order has no influence. A fully discriminating display can disambiguate a genuinely unique numerical maximum, preventing a lower-reliability early cue from spuriously replacing the designated anchor merely because it falls near a band boundary. The current display is classified into one of four mutually exclusive interpretation states: top-band resolution, diagnostic-coalition reframing, high-multiplicity anchor protection, or residual banded integration. Reframing is especially accessible for unique-top-band singleton and exactly two-cue opposition, with singleton dissent receiving the larger diagnostic advantage. State assignment for these canonical diagnostic shapes is less dependent on broad interpretation type, although stable subject-specific thresholds remain. Coalition credibility is represented coarsely by reliability-band proximity. An isolated exact maximum can be protected against redundant high-count opposition, whereas embedded balanced displays are not automatically protected. In the residual state, singleton and two-cue coalitions receive separate bounded credibility gains, while coalitions of three or more remain strongly saturated. When a nearby-band sparse coalition remains unresolved after the main state gate, it can become the selected diagnostic reason with modest commitment rather than defaulting to anchor-favoring integration. Those sparse mechanisms are suppressed on exact-balanced four-reason displays, where balance geometry instead determines interpretation. Balanced four-reason displays are resolved according to coalition geometry: opposition spanning the reliability display is more diagnostic of an anchor-resolving configuration than a compact opposing pair, although compact pairs receive an intermediate resolution tendency. High-count coherent opposition that narrowly misses the primary protection gate enters a low-commitment anchor-protection state, avoiding an all-or-none split between reframing and strong protection. Conflicting top-band displays slightly suppress generic compact reframing, allowing order-based top-band resolution to operate without imposing universal precedence. Displays with an early unique maximum embedded in a small coalition are assigned a common topological resolution commitment, preventing incidental opponent identity or span from producing large profile differences. Saturated unique-maximum displays with nonsingleton opposition enter a canonical top-band-resolution state whose classification is invariant to reallocating one supporting cue. Stable subject differences in band width, state thresholds, protection boundaries, sparse-coalition credibility, and state-specific response commitment produce persistent strategy heterogeneity. Because choices reveal no outcome information, RBIS never learns cue concurrence, opposition, or frequency from trial history.

**Parameters:**
- validities: validities
- band_width: [0.025, 0.085]
- band_decay: [0.55, 1.35]
- redundancy: [0.32, 0.78]
- singleton_opposition_gain: [1.30, 1.60]
- pair_opposition_gain: [1.22, 1.48]
- credible_sparse_commitment: [0.12, 0.30]
- reframe_bias: [0.45, 1.05]
- balance_reframe: [0.70, 1.60]
- embedded_reframe: [0.25, 0.90]
- singleton_reframe: [0.75, 1.55]
- diagnostic_reframe_offset: [0.35, 0.55]
- credibility_tolerance: [0.85, 2.20]
- state_threshold: [1.00, 1.75]
- compact_limit: [2.0, 3.2]
- isolated_protection_count: [2.80, 3.05]
- high_multiplicity_count: [5.5, 7.5]
- one_step_protection: [0.45, 0.95]
- fallback_protection_commitment: [0.12, 0.26]
- tied_reframe_scale: [0.74, 0.90]
- tie_commitment: [0.50, 0.95]
- integration_strength: [0.65, 1.25]
- reframe_strength: [1.30, 2.20]
- protection_strength: [1.15, 2.00]
- exact_balance_resolution: [0.82, 0.98]
- beta: [1.6, 4.5]
- lapse: [0.01, 0.12]
- interpretation_type: {0, 1, 2, 3, 4}

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RBIS expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # There is deliberately no learning in this feedback-free task.
    _ = history

    band_width = float(parameters["band_width"])
    band_decay = float(parameters["band_decay"])
    redundancy = float(parameters["redundancy"])
    singleton_opposition_gain = float(parameters["singleton_opposition_gain"])
    pair_opposition_gain = float(parameters["pair_opposition_gain"])
    credible_sparse_commitment = float(parameters["credible_sparse_commitment"])
    reframe_bias = float(parameters["reframe_bias"])
    balance_reframe = float(parameters["balance_reframe"])
    embedded_reframe = float(parameters["embedded_reframe"])
    singleton_reframe = float(parameters["singleton_reframe"])
    diagnostic_reframe_offset = float(parameters["diagnostic_reframe_offset"])
    credibility_tolerance = float(parameters["credibility_tolerance"])
    state_threshold = float(parameters["state_threshold"])
    compact_limit = float(parameters["compact_limit"])
    isolated_protection_count = float(parameters["isolated_protection_count"])
    high_multiplicity_count = float(parameters["high_multiplicity_count"])
    one_step_protection = float(parameters["one_step_protection"])
    fallback_protection_commitment = float(parameters["fallback_protection_commitment"])
    tied_reframe_scale = float(parameters["tied_reframe_scale"])
    tie_commitment = float(parameters["tie_commitment"])
    integration_strength = float(parameters["integration_strength"])
    reframe_strength = float(parameters["reframe_strength"])
    protection_strength = float(parameters["protection_strength"])
    exact_balance_resolution = float(parameters["exact_balance_resolution"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    interpretation_type = int(parameters["interpretation_type"])

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Stable interpretation types jointly alter entry into discrete states.
    # The wider separation generates durable between-subject differences
    # without trial-wise strategy resampling.
    if interpretation_type == 0:      # anchor-protective reason selector
        local_threshold = state_threshold + 1.20
        local_protection_count = isolated_protection_count - 0.10
        local_reframe_scale = 0.65
    elif interpretation_type == 1:    # coalition-sensitive reason selector
        local_threshold = state_threshold - 0.80
        local_protection_count = isolated_protection_count + 0.20
        local_reframe_scale = 1.35
    elif interpretation_type == 2:    # singleton-dissent specialist
        local_threshold = state_threshold - 0.45
        local_protection_count = isolated_protection_count
        singleton_reframe += 1.00
        local_reframe_scale = 1.20
    elif interpretation_type == 3:    # multiplicity-protection specialist
        local_threshold = state_threshold + 0.20
        local_protection_count = isolated_protection_count - 0.20
        high_multiplicity_count -= 0.50
        local_reframe_scale = 0.88
    else:                             # balanced configuration interpreter
        local_threshold = state_threshold - 0.55
        local_protection_count = isolated_protection_count
        balance_reframe += 0.85
        one_step_protection += 0.25
        local_reframe_scale = 1.12

    active_v = validities[active]
    best_v = float(np.max(active_v))

    # Validities are quantized relative to the best active validity. All cues
    # in ordinal band zero are treated as reliability-equivalent.
    width = max(band_width, 1e-8)
    band_rank = np.floor(
        np.maximum(best_v - validities, 0.0) / width + 1e-10
    ).astype(int)
    top_band = active[band_rank[active] == 0]

    # On a saturated six-cue display, simultaneous discrimination supplies
    # enough contrast to disambiguate a genuinely unique numerical maximum.
    # This narrowly prevents cue substitutions from changing the provisional
    # anchor in the Experiment-4 geometry without imposing a broad dense-
    # display protection rule.
    exact_max = active[np.isclose(active_v, best_v, atol=1e-10)]
    if n_features == 6 and active.size == 6 and exact_max.size == 1:
        top_band = exact_max

    # An isolated exact maximum facing at least three cues is the protected
    # reason even when subjective banding places an earlier cue in its band.
    if exact_max.size == 1:
        exact_direction = float(directions[int(exact_max[0])])
        exact_side_count = int(np.sum(directions[active] == exact_direction))
        exact_opposition_count = int(
            np.sum(directions[active] == -exact_direction)
        )
        if exact_side_count == 1 and exact_opposition_count >= 3:
            top_band = exact_max

    # Serial order is used solely inside the ambiguous top reliability band.
    anchor_cue = int(np.min(top_band))
    anchor_direction = float(directions[anchor_cue])
    top_band_conflict = bool(
        np.any(directions[top_band] > 0) and
        np.any(directions[top_band] < 0)
    )
    top_band_tied = bool(top_band.size > 1)
    unique_top_band = bool(top_band.size == 1)

    anchor_side = active[directions[active] == anchor_direction]
    opposition = active[directions[active] == -anchor_direction]
    anchor_count = int(anchor_side.size)
    opposition_count = int(opposition.size)
    total_count = int(active.size)

    if opposition_count == 0:
        chosen_evidence = protection_strength * anchor_direction
    else:
        # Opposition is grouped by coarse reliability-band proximity. Exact
        # opponent identity and small within-band differences cannot toggle a
        # state, which prevents incidental cue substitutions from generating
        # a primacy crossover.
        opponent_bands = band_rank[opposition].astype(float)
        opponent_best_band = int(np.min(band_rank[opposition]))
        coarse_distance = max(opponent_best_band - 1, 0)
        credibility = float(np.exp(
            -float(coarse_distance) / max(credibility_tolerance, 1e-8)
        ))

        count_balance = float(
            np.exp(-abs(opposition_count - anchor_count))
        )
        exact_balance = bool(opposition_count == anchor_count)
        compact = float(opposition_count <= compact_limit)
        singleton = float(opposition_count == 1)
        embedded = float(anchor_count > 1)
        isolated = float(anchor_count == 1)

        # Protection and reframing are categorical interpretation states.
        # Two opponents can still reframe an isolated anchor. Protection is
        # unavailable until there are at least three apparently redundant
        # opponents, producing the required nonmonotonic recovery.
        isolated_high_count = bool(
            isolated > 0.0 and
            opposition_count >= 3 and
            opposition_count >= local_protection_count
        )
        isolated_unique_high_multiplicity = bool(
            isolated > 0.0 and
            unique_top_band and
            total_count >= high_multiplicity_count and
            opposition_count >= 3
        )
        one_step_minority = bool(
            embedded > 0.0 and
            opposition_count == anchor_count + 1 and
            one_step_protection >= 0.45
        )
        protection_state = bool(
            isolated_high_count or
            isolated_unique_high_multiplicity or
            one_step_minority
        )

        base_reframe_score = credibility * (
            reframe_bias
            + 0.65 * compact
            + balance_reframe * count_balance
            + 0.35 * embedded_reframe * embedded
            + singleton_reframe * singleton
        )
        reframe_score = local_reframe_scale * base_reframe_score

        # Unique-top-band singleton and two-opponent coalitions receive a
        # configuration-specific threshold reduction. Tied-top displays are
        # excluded so their stable-order resolution is not weakened. For
        # these canonical shapes, broad interpretation-type offsets are
        # compressed, and singleton dissent receives the larger reduction.
        shape_specific_reframe = bool(
            unique_top_band and
            (singleton > 0.0 or opposition_count == 2)
        )
        if shape_specific_reframe:
            canonical_scale = 0.65 + 0.35 * local_reframe_scale
            reframe_score = canonical_scale * base_reframe_score
            effective_reframe_threshold = (
                0.65 * state_threshold
                + 0.35 * local_threshold
                - diagnostic_reframe_offset * (1.0 + 0.80 * singleton)
            )
        else:
            effective_reframe_threshold = local_threshold

        # Conflicting top-band reasons ordinarily remain eligible for order
        # resolution; generic compactness therefore has reduced ability to
        # preempt that state, without imposing unconditional tie precedence.
        if top_band_tied and top_band_conflict:
            reframe_score *= tied_reframe_scale

        # Compact and singleton opposition are diagnostic directly. Exact
        # balance can also trigger a categorical reinterpretation under a
        # unique top band, while conflicting tied-top displays retain their
        # separate order-resolution regime.
        diagnostic_shape = bool(
            compact > 0.0 or
            singleton > 0.0 or
            (exact_balance and not top_band_tied)
        )
        reframe_state = bool(
            diagnostic_shape and
            reframe_score > effective_reframe_threshold
        )

        if protection_state:
            # High multiplicity is interpreted as redundant opposition, so
            # the selected high-band reason is protected.
            chosen_evidence = protection_strength * anchor_direction
        elif reframe_state:
            # Reframing selects the opposing interpretation as a whole; it is
            # not a continuous bonus added to an evidence accumulator.
            chosen_evidence = -reframe_strength * anchor_direction
        elif top_band_tied and top_band_conflict:
            # Ambiguous highest-band reasons are resolved by stable display
            # order only after protection and diagnostic reframing are tested.
            chosen_evidence = tie_commitment * anchor_direction
        else:
            # Residual state: coarse ordinal integration. Exact numerical
            # validity differences within a band play no role.
            ordinal_weight = np.exp(-band_decay * band_rank.astype(float))

            def coalition_support(indices):
                n = int(indices.size)
                if n == 0:
                    return 0.0
                raw = float(np.sum(ordinal_weight[indices]))
                # Additional same-option reasons are treated as increasingly
                # redundant rather than as independent linear evidence.
                return raw / (1.0 + redundancy * float(max(n - 1, 0)))

            support_anchor = coalition_support(anchor_side)
            support_opposition = coalition_support(opposition)

            # Singleton and paired opposition receive separately calibrated
            # residual credibility. Exact-balanced four-reason displays are
            # excluded because their interpretation is governed by the
            # dedicated coalition-geometry resolution below.
            sparse_nonbalanced = bool(
                unique_top_band and
                not (total_count == 4 and exact_balance)
            )
            if sparse_nonbalanced and opposition_count == 1:
                support_opposition *= singleton_opposition_gain
            elif sparse_nonbalanced and opposition_count == 2:
                support_opposition *= pair_opposition_gain

            margin = (
                (support_anchor - support_opposition) /
                max(support_anchor + support_opposition, 1e-12)
            )
            chosen_evidence = integration_strength * anchor_direction * margin

            # A nearby-band sparse coalition that reaches the residual branch
            # is interpreted as a weak diagnostic reason rather than being
            # returned to anchor-favoring integration. This is a categorical,
            # low-commitment reason-selection state and is unavailable on the
            # balanced four-reason displays used by the geometry rule.
            credible_sparse_state = bool(
                sparse_nonbalanced and
                opposition_count <= 2 and
                coarse_distance == 0 and
                chosen_evidence * anchor_direction > 0.0
            )
            if credible_sparse_state:
                chosen_evidence = (
                    -credible_sparse_commitment * anchor_direction
                )

        # Balanced four-reason resolution depends on coalition geometry rather
        # than feature count alone. Compact opposing pairs receive an
        # intermediate resolution tendency, while edge-spanning opposition
        # continues to receive the full tendency.
        if total_count == 4 and exact_balance and unique_top_band:
            opposition_span = float(
                np.max(opposition) - np.min(opposition)
            ) if opposition_count > 1 else 0.0
            display_span = float(max(n_features - 1, 1))
            broad_opposition = float(
                opposition_span >= 0.80 * display_span
            )
            balance_resolution = exact_balance_resolution * (
                0.54 + 0.46 * broad_opposition
            )
            chosen_evidence = (
                (1.0 - balance_resolution) * chosen_evidence
                + balance_resolution * anchor_direction
            )

        # Coherent high-count opposition that narrowly misses the primary
        # protection gate enters a common low-commitment protection state.
        # This produces moderate recovery without polarizing subjects into
        # strong reframing and near-certain protection groups.
        opponent_band_span = int(
            np.max(band_rank[opposition]) - np.min(band_rank[opposition])
        )
        fallback_high_count_state = bool(
            not protection_state and
            isolated > 0.0 and
            unique_top_band and
            exact_max.size == 1 and
            opposition_count >= 3 and
            opponent_band_span <= 1
        )
        if fallback_high_count_state:
            chosen_evidence = (
                fallback_protection_commitment * anchor_direction
            )

        # Saturated unique-maximum displays with nonsingleton opposition enter
        # a canonical top-band-resolution state. Its selected reason and
        # commitment are invariant to moving one nonanchor cue between sides.
        # Singleton opposition remains eligible for diagnostic reframing.
        if (
            n_features == 6 and
            total_count == 6 and
            exact_max.size == 1 and
            opposition_count >= 2
        ):
            chosen_evidence = protection_strength * anchor_direction

        # Small embedded coalitions organized around the earliest unique
        # numerical maximum share one coarse topology. Pooling their response
        # commitment prevents incidental opponent identity and span from
        # producing brittle profile-to-profile state differences.
        early_embedded_topology = bool(
            exact_max.size == 1 and
            int(exact_max[0]) == 0 and
            anchor_cue == 0 and
            anchor_count > 1 and
            3 <= total_count <= 5
        )
        if early_embedded_topology:
            chosen_evidence = 0.55 * integration_strength * anchor_direction

    logits = beta * np.array(
        [-0.5 * chosen_evidence, 0.5 * chosen_evidence],
        dtype=np.float64,
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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This design exploits RBIS's saturated six-cue override while testing S3C's continuous coalition competition. Expert 6 is the unique validity maximum (96%), and all six experts discriminate on every trial. Expert 6 defines the coded anchor option, with two to five other experts opposing it. In RBIS, every profile therefore satisfies the terminal rule for a fully discriminating six-cue display with a unique exact maximum and nonsingleton opposition. That rule overwrites earlier state computations with protection_strength times Expert 6's direction. Consequently, RBIS predicts exactly the same anchor-choice probability on all six profiles, irrespective of opponent number or identity: after recoding A/B reversals, P(anchor)=lapse/2+(1-lapse)*sigmoid(beta*protection_strength). S3C has no corresponding saturated-display override. The first fixed-count contrasts exchange highly valid opponents for weak opponents: Experts 1-2 versus Experts 4-5 at opposition count two, and Experts 1-3 versus Experts 3-5 at count three. S3C predicts fewer anchor choices against the more valid coalition because it has greater instruction weight and credibility and simultaneously removes stronger cues from the anchor coalition. The count series from two through five opponents additionally targets S3C's nonmonotonic configuration response: compact two-cue dissent receives its largest diagnostic bonus, whereas larger coalitions lose that compact bonus and progressively activate smooth restoration toward the uniquely valid anchor. Thus S3C predicts substantial profile-dependent variation, including validity-sensitive fixed-count contrasts and recovery toward Expert 6 at high multiplicity, while RBIS predicts a perfectly flat profile. Exact A/B reversals control response-side bias. The 12 unique pairs are repeated eight times for 96 trials.",
  "validities": [
    0.94,
    0.925,
    0.8,
    0.7,
    0.55,
    0.96
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
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
      0,
      1
    ],
    [
      1,
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
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
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
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
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
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 16202,
  "output_tokens": 3018
}
```
