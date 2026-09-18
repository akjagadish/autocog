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
**Description:** Coherence-Gated Relative Coalition Comparison theory proposes that people compress endorsements into option-level coalition representations before making a choice. Each representation combines smoothly encoded mean assertiveness, sublinear consensus, and a coherence gate. The coherence gate is threshold-shaped: ordinary variation among coalition members has little effect, whereas sufficiently extreme dispersion sharply weakens the coalition. Consequently, a maximally dispersed opposing coalition makes a homogeneous coalition relatively prominent and thus a stronger target of skeptical interpretation, without making routine coalition expansion intrinsically reactance-producing. Sublinear consensus is offset by breadth and active-evidence normalization, yielding approximately breadth-invariant comparison of equal-sized homogeneous coalitions. A narrow-conflict focus rule further distinguishes maximal relative contrasts from ordinary singleton disagreements: a smooth 100%-versus-50% contrast remains strongly skepticism-provoking, while less extreme singleton contrasts are attenuated without an endpoint-specific bonus. A separate validity-blind tally channel grows concavely with vote margin. Its gain is reduced while its curvature is increased, pivoting tally evidence away from one-vote margins and toward moderate and large margins. Continuous correlation between tally orientation and response noise prevents high-tally subjects from becoming deterministically majority-following. Skeptical orientation, tally sensitivity, and decision noise vary continuously across people, with composition-dependent heterogeneity concentrated in genuinely dispersed conflicts. With no correctness feedback, processing remains history independent and exactly option-swap symmetric.

**Parameters:**
- validity_curvature: [1.6, 3.0]
- consensus_exponent: [0.58, 0.76]
- coherence_scale: [0.035, 0.075]
- coherence_floor: [0.05, 0.25]
- coherence_power: [1.8, 3.0]
- coherence_trait_coupling: [-0.70, 0.70]
- attention_floor: [0.28, 0.46]
- evidence_normalization: [0.48, 0.76]
- contrast_accumulation: [1.15, 1.75]
- singleton_focus_floor: [0.45, 0.70]
- skeptical_strength: [1.85, 2.85]
- skeptical_orientation: [0, 1]
- composition_trait_gain: [0.30, 0.90]
- skeptic_margin_coupling: [-0.30, 0.45]
- tally_sensitivity: [0.25, 0.43]
- tally_trait: [0, 1]
- tally_exponent: [0.78, 0.96]
- choice_sensitivity: [0.82, 1.18]
- noise_trait: [0, 1]
- tally_noise_correlation: [0.15, 0.35]
- conflict_noise: [0.10, 0.55]
- lapse_rate: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Coherence-Gated Relative Coalition Comparison. Choice history is ignored
    # because previous choices are not accompanied by correctness feedback.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive signs endorse A and negative signs endorse B. Features on which
    # the options agree carry neither a categorical vote nor a directional claim.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    total_breadth = int(np.count_nonzero(active))

    # A single smooth, monotone, saturating validity code. The normalization
    # maps 50% to zero and 100% to one without assigning either 95% or 100% a
    # qualitatively distinct representation.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["validity_curvature"])
    endpoint = np.tanh(0.5 * curvature)
    assertiveness = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)

    skeptic_trait = float(parameters["skeptical_orientation"])

    def summarize(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_strength = float(np.mean(members))
        dispersion = float(np.mean((members - mean_strength) ** 2))

        # The same continuous skeptical trait changes sensitivity to internal
        # inconsistency. The superlinear dispersion ratio leaves ordinary
        # coalition composition largely intact but sharply gates sufficiently
        # dispersed coalitions.
        scale = float(parameters["coherence_scale"]) * np.exp(
            float(parameters["coherence_trait_coupling"])
            * (skeptic_trait - 0.5)
        )
        floor = float(parameters["coherence_floor"])
        dispersion_ratio = dispersion / max(scale, 1e-12)
        coherence = floor + (1.0 - floor) * np.exp(
            -(dispersion_ratio ** float(parameters["coherence_power"]))
        )

        consensus = float(breadth) ** float(parameters["consensus_exponent"])
        representation = mean_strength * consensus * coherence
        return float(representation), dispersion, breadth

    rep_a, dispersion_a, breadth_a = summarize(1.0)
    rep_b, dispersion_b, breadth_b = summarize(-1.0)

    if total_breadth == 0:
        contextual_contrast = 0.0
        pooled_dispersion = 0.0
    else:
        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)

        # For equal breadth n on both sides, homogeneous representations grow
        # as n**alpha while this breadth unit grows by the same factor. Thus
        # repetition alone produces almost no change in validity contrast.
        alpha = float(parameters["consensus_exponent"])
        breadth_unit = (float(total_breadth) / 2.0) ** alpha

        # Total active assertiveness consumes limited comparison capacity. This
        # is a relative, composition-sensitive normalization and therefore
        # preserves a maximal 100%-versus-50% singleton contrast.
        active_mean = float(np.mean(assertiveness[active]))
        denominator = breadth_unit * (
            float(parameters["attention_floor"])
            + float(parameters["evidence_normalization"]) * active_mean
        )
        raw_relative_contrast = (rep_a - rep_b) / max(denominator, 1e-12)
        contextual_contrast = float(np.tanh(
            float(parameters["contrast_accumulation"])
            * raw_relative_contrast
        ))

    # Stronger relative coalition claims provoke skepticism. Trait effects are
    # selectively coupled to composition and margin, so individual differences
    # grow where construal is genuinely ambiguous rather than on every trial.
    dispersion_index = pooled_dispersion / (pooled_dispersion + 0.04)
    margin_index = abs(tally_margin) / (abs(tally_margin) + 2.0)
    skeptical_multiplier = 0.70 + 0.60 * skeptic_trait
    skeptical_multiplier *= (
        1.0
        + float(parameters["composition_trait_gain"])
        * (skeptic_trait - 0.5)
        * dispersion_index
    )
    skeptical_multiplier *= (
        1.0
        + float(parameters["skeptic_margin_coupling"])
        * (skeptic_trait - 0.5)
        * margin_index
    )
    skeptical_multiplier = max(float(skeptical_multiplier), 0.05)

    # In a narrow bilateral conflict, weak relative contrasts receive less
    # pragmatic focus than maximal contrasts. The adjustment vanishes smoothly
    # with breadth and therefore does not create a general repetition benefit.
    narrow_weight = np.exp(-max(float(total_breadth) - 2.0, 0.0) / 2.0)
    contrast_focus = 1.0 - narrow_weight * (
        1.0 - float(parameters["singleton_focus_floor"])
    ) * (1.0 - abs(contextual_contrast))
    skeptical_evidence = -float(parameters["skeptical_strength"]) * (
        skeptical_multiplier * contrast_focus * contextual_contrast
    )

    # The independent tally channel is validity blind. Its concave nonlinear
    # growth differentiates margins and protects large majorities, but is less
    # steep and less confident than a superlinear margin rule.
    tally_trait = float(parameters["tally_trait"])
    tally_gain = float(parameters["tally_sensitivity"]) * (
        0.20 + 1.60 * tally_trait
    )
    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        magnitude = abs(tally_margin)
        tally_curve = (1.0 + magnitude) ** float(parameters["tally_exponent"]) - 1.0
        tally_evidence = tally_gain * np.sign(tally_margin) * tally_curve

    total_evidence = tally_evidence + skeptical_evidence

    # Noise varies continuously, but conflict-dependent noise is concentrated
    # in dispersed, low-margin comparisons instead of being a blanket variance
    # adjustment. High tally orientation is also coupled to lower response
    # precision, preventing high-gain subjects from becoming deterministic.
    noise_trait = float(parameters["noise_trait"])
    base_gain = float(parameters["choice_sensitivity"]) * (
        0.72 + 0.56 * noise_trait
    )
    base_gain /= (
        1.0
        + float(parameters["tally_noise_correlation"])
        * tally_trait
    )
    low_margin_weight = 1.0 / (1.0 + abs(tally_margin) / 2.0)
    conflict_load = dispersion_index * low_margin_weight
    effective_gain = base_gain / (
        1.0
        + float(parameters["conflict_noise"])
        * (1.20 - 0.40 * noise_trait)
        * conflict_load
    )
    decision_variable = float(effective_gain * total_evidence)

    # Positive evidence favors A. Symmetric logits guarantee exact invariance
    # under swapping the two displayed options.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## COMPETING THEORY
**Description:** Uncertainty-Arbitrated Overclaim Construal theory proposes that people form two competing, stimulus-dependent interpretations of binary expert advice. The reliability construal compresses each option's endorsers into a coalition claim defined by mean communicated reliability and an inferred trustworthiness estimate. Coalition breadth does not directly amplify this claim, so repeating an otherwise homogeneous coalition is approximately representationally redundant. Trustworthiness remains high under ordinary heterogeneity but collapses after dispersion crosses a subject-specific uncertainty threshold. This makes an extremely split coalition a poor bearer of a unified claim and can render the opposing homogeneous coalition the uniquely salient overclaim. A skeptical pragmatic response then favors the option opposite that salient claim. In parallel, a social-count construal ignores validity and represents only the raw endorsement margin. Its evidence follows a sigmoid power curve with convex initial growth and a confidence-limited asymptote. A localized confidence-control window attenuates moderate tally margins while leaving decisive margins nearly unchanged. A metacognitive gate arbitrates between the two construals according to claim certainty, coalition dispersion, tally decisiveness, and representational disagreement. Moderate tied claims receive additional scrutiny only inside a narrow reliability-contrast band, preventing the intervention from spreading to endpoint-like equal-coalition conflicts. Precision is selectively reduced for one-vote conflicts and for endpoint-sized bilateral claims supported by two trustworthy coalitions. Consequently, ambiguous conflicts move toward chance, moderate overclaims can evoke skepticism, and extremely dispersed claims remain discounted. With no correctness feedback, the process is history independent. Every computation uses signed option contrasts, guaranteeing exact option-swap symmetry.

**Parameters:**
- reliability_curvature: [2.0, 3.2]
- dispersion_threshold: [0.050, 0.082]
- dispersion_steepness: [32.0, 52.0]
- trust_tail: [0.0, 0.07]
- sampling_scale: [0.70, 1.30]
- sampling_floor: [0.72, 0.90]
- claim_floor: [0.16, 0.30]
- skeptical_orientation: [0, 1]
- skeptic_trait_spread: [0.50, 0.90]
- skeptical_gain: [2.6, 3.6]
- claim_accumulation: [1.25, 1.75]
- ambiguity_margin_scale: [2.2, 3.8]
- tally_power: [1.75, 2.35]
- tally_half_margin: [2.5, 3.4]
- tally_gain_trait: [0, 1]
- tally_trait_spread: [0.22, 0.48]
- tally_gain: [1.85, 2.35]
- mid_margin_center: [3.6, 4.4]
- mid_margin_width: [0.9, 1.3]
- mid_margin_tally_damping: [0.14, 0.24]
- arbitration_threshold: [0.28, 0.62]
- certainty_weight: [1.45, 2.10]
- disagreement_weight: [0.55, 1.05]
- dispersion_uncertainty_weight: [0.12, 0.38]
- margin_resolution_weight: [1.10, 1.65]
- dispersion_reference: [0.035, 0.065]
- gate_temperature: [0.16, 0.28]
- moderate_tie_gate_boost: [1.8, 2.8]
- moderate_contrast_center: [0.20, 0.27]
- moderate_contrast_width: [0.035, 0.070]
- precision_trait: [0, 1]
- precision_trait_spread: [0.28, 0.55]
- base_precision: [1.05, 1.35]
- small_margin_precision_damping: [0.50, 0.66]
- small_margin_decay: [0.75, 1.15]
- trusted_extreme_precision_damping: [0.35, 0.52]
- extreme_contrast_center: [0.62, 0.72]
- extreme_contrast_width: [0.045, 0.080]
- lapse_rate: [0.0, 0.055]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Uncertainty-Arbitrated Overclaim Construal model. History is intentionally
    # ignored because choices are never followed by correctness feedback.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive signs endorse A and negative signs endorse B. Agreements are
    # non-discriminating and enter neither representation.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)
    active_count = int(np.count_nonzero(active))

    # A smooth bounded reliability code avoids endpoint categories while
    # retaining sensitivity throughout the communicated validity scale.
    validity = np.clip(validities, 0.5, 1.0)
    reliability_curvature = float(parameters["reliability_curvature"])
    endpoint = np.tanh(0.5 * reliability_curvature)
    reliability = np.tanh(
        reliability_curvature * (validity - 0.5)
    ) / max(float(endpoint), 1e-12)

    # The arbitration threshold is a stable continuous individual difference.
    arbitration_threshold = float(parameters["arbitration_threshold"])
    dispersion_threshold = float(parameters["dispersion_threshold"]) * (
        0.82 + 0.36 * arbitration_threshold
    )

    def coalition(side):
        members = reliability[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0

        # Mean reliability is the coalition's location estimate. Breadth is not
        # an unconditional multiplier: homogeneous repetitions communicate the
        # same option-level claim rather than successively stronger claims.
        location = float(np.mean(members))
        dispersion = float(np.mean((members - location) ** 2))

        # Inferred coalition trustworthiness has a threshold-shaped collapse.
        # Ordinary heterogeneity is tolerated, whereas a maximally split
        # coalition is treated as unable to sustain a unified assertion.
        steepness = float(parameters["dispersion_steepness"])
        z = np.clip(steepness * (dispersion - dispersion_threshold), -60.0, 60.0)
        core_trust = 1.0 / (1.0 + np.exp(z))
        tail = float(parameters["trust_tail"])
        trust = tail + (1.0 - tail) * core_trust

        # Very small coalitions carry somewhat less metacognitive certainty,
        # but this saturates rapidly and cannot generate repeated-consensus
        # reactance in broad homogeneous coalitions.
        sampling_confidence = 1.0 - np.exp(
            -float(breadth) / float(parameters["sampling_scale"])
        )
        metacognitive_trust = trust * (
            float(parameters["sampling_floor"])
            + (1.0 - float(parameters["sampling_floor"])) * sampling_confidence
        )
        claim = location * trust
        return float(claim), float(metacognitive_trust), dispersion, breadth

    claim_a, trust_a, dispersion_a, breadth_a = coalition(1.0)
    claim_b, trust_b, dispersion_b, breadth_b = coalition(-1.0)

    if active_count == 0:
        reliability_contrast = 0.0
        pooled_dispersion = 0.0
        claim_certainty = 0.0
    else:
        # Relative normalization makes claim comparison breadth invariant for
        # equal homogeneous coalitions and bounds it under arbitrary designs.
        claim_mass = abs(claim_a) + abs(claim_b)
        reliability_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + claim_mass
        )
        reliability_contrast = float(np.clip(reliability_contrast, -1.0, 1.0))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(active_count)

        # A contrast can be scrutinized when at least one side supports a
        # trustworthy claim. Thus discounting a split coalition can expose the
        # opposing homogeneous coalition as the salient overclaim.
        strongest_trust = max(trust_a, trust_b)
        claim_certainty = strongest_trust * abs(reliability_contrast)

    # Skeptical interpretation reverses a salient reliability-weighted claim.
    # Individual differences are amplified in low-margin, representation-level
    # conflicts rather than uniformly across all trials.
    skeptical_trait = float(parameters["skeptical_orientation"])
    ambiguity_index = 1.0 / (
        1.0 + (margin_magnitude / float(parameters["ambiguity_margin_scale"])) ** 2
    )
    skeptical_multiplier = 1.0 + float(parameters["skeptic_trait_spread"]) * (
        skeptical_trait - 0.5
    ) * (0.35 + 0.65 * ambiguity_index)
    skeptical_gain = float(parameters["skeptical_gain"]) * max(
        skeptical_multiplier, 0.10
    )
    bounded_claim = np.tanh(
        float(parameters["claim_accumulation"]) * reliability_contrast
    )
    skeptical_evidence = -skeptical_gain * float(bounded_claim)

    # The independent social-count representation has convex initial growth,
    # an intermediate steep region, and a confidence-limited asymptote.
    if margin_magnitude < 1e-12:
        tally_shape = 0.0
    else:
        tally_power = float(parameters["tally_power"])
        half_margin = float(parameters["tally_half_margin"])
        numerator = margin_magnitude ** tally_power
        tally_shape = numerator / (half_margin ** tally_power + numerator)

    # Tally-gain heterogeneity is attenuated as count evidence becomes decisive,
    # helping preserve the low between-subject variance of clear margin series.
    tally_trait = float(parameters["tally_gain_trait"])
    tally_uncertainty = 1.0 - tally_shape
    tally_multiplier = 1.0 + float(parameters["tally_trait_spread"]) * (
        tally_trait - 0.5
    ) * (0.25 + 0.75 * tally_uncertainty)
    tally_gain = float(parameters["tally_gain"]) * max(tally_multiplier, 0.10)
    tally_evidence = (
        tally_gain * np.sign(tally_margin) * tally_shape
        if margin_magnitude > 0.0 else 0.0
    )

    # Confidence control selectively attenuates moderate margins, where raw
    # counting is informative but not yet decisive. The Gaussian window has
    # little effect on one-vote trials or margins seven and above.
    mid_margin_window = np.exp(
        -0.5 * (
            (margin_magnitude - float(parameters["mid_margin_center"]))
            / float(parameters["mid_margin_width"])
        ) ** 2
    )
    tally_evidence *= 1.0 - float(
        parameters["mid_margin_tally_damping"]
    ) * mid_margin_window

    # Arbitration depends jointly on claim reliability, coalition dispersion,
    # tally decisiveness, and disagreement between the two construals.
    normalized_claim_direction = float(np.tanh(skeptical_evidence))
    normalized_tally_direction = float(np.tanh(tally_evidence))
    disagreement = abs(
        normalized_claim_direction - normalized_tally_direction
    ) / 2.0
    dispersion_index = pooled_dispersion / (
        pooled_dispersion + float(parameters["dispersion_reference"])
    ) if active_count > 0 else 0.0

    # A localized boost admits skeptical scrutiny of moderate reliability
    # contrasts only when the tally is unresolved and both coalitions remain
    # trustworthy. It therefore does not revive a maximally split coalition.
    both_trustworthy = min(trust_a, trust_b)
    contrast_magnitude = abs(reliability_contrast)
    moderate_center = float(parameters["moderate_contrast_center"])
    moderate_width = float(parameters["moderate_contrast_width"])
    moderate_band = np.exp(
        -0.5 * ((contrast_magnitude - moderate_center) / moderate_width) ** 2
    )
    tie_index = np.exp(-(margin_magnitude / 0.35) ** 2)
    moderate_tie_boost = (
        float(parameters["moderate_tie_gate_boost"])
        * tie_index
        * both_trustworthy
        * moderate_band
    )

    gate_drive = (
        float(parameters["certainty_weight"]) * claim_certainty
        + float(parameters["disagreement_weight"]) * disagreement
        - float(parameters["dispersion_uncertainty_weight"]) * dispersion_index
        - float(parameters["margin_resolution_weight"]) * tally_shape
        - arbitration_threshold
        + moderate_tie_boost
    )
    gate_temperature = float(parameters["gate_temperature"])
    gate_logit = np.clip(gate_drive / gate_temperature, -60.0, 60.0)
    claim_weight = 1.0 / (1.0 + np.exp(-gate_logit))

    # Arbitration is a reliability-weighted competition between construals,
    # not an additive fixed coalition-plus-tally score.
    total_evidence = (
        claim_weight * skeptical_evidence
        + (1.0 - claim_weight) * tally_evidence
    )

    # Precision differences matter chiefly when the stimulus remains ambiguous;
    # their expression contracts on decisive tallies.
    precision_trait = float(parameters["precision_trait"])
    unresolved = 1.0 - tally_shape * (1.0 - claim_weight)
    precision_multiplier = 1.0 + float(parameters["precision_trait_spread"]) * (
        precision_trait - 0.5
    ) * (0.20 + 0.80 * unresolved)
    precision = float(parameters["base_precision"]) * max(
        precision_multiplier, 0.10
    )

    # Selective confidence limiting pulls one-vote conflicts toward chance
    # without changing the moderate-to-large tally asymptote. A second limiter
    # applies to endpoint-sized bilateral claims only when both coalitions are
    # trustworthy, preserving the extreme-dispersion crossover.
    nonzero_margin_index = 1.0 - np.exp(-(margin_magnitude / 0.35) ** 2)
    small_margin_load = nonzero_margin_index * np.exp(
        -max(margin_magnitude - 1.0, 0.0)
        / float(parameters["small_margin_decay"])
    )
    precision *= 1.0 - float(
        parameters["small_margin_precision_damping"]
    ) * small_margin_load

    extreme_center = float(parameters["extreme_contrast_center"])
    extreme_width = float(parameters["extreme_contrast_width"])
    extreme_z = np.clip(
        (contrast_magnitude - extreme_center) / extreme_width, -60.0, 60.0
    )
    extreme_contrast_index = 1.0 / (1.0 + np.exp(-extreme_z))
    trusted_extreme_load = tie_index * both_trustworthy * extreme_contrast_index
    precision *= 1.0 - float(
        parameters["trusted_extreme_precision_damping"]
    ) * trusted_extreme_load

    decision_variable = float(precision * total_evidence)

    # Symmetric logits ensure exact invariance when A and B are exchanged.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 16-pair, 96-trial design creates both directional and quantitative dissociations. In each of the first eight pairs, the highest-validity discriminating cue favors A; the remaining eight pairs are exact A/B reversals. Therefore Take The Best predicts the same choice probability for its lexicographic winner on every trial: epsilon/2 + (1-epsilon)*sigmoid(beta), regardless of which cue is first discriminating or how many lower-validity cues oppose it. Pairs 1-4 manipulate the rank of the first discriminating cue while producing equal tallies (one cue for each option). Tallying must predict exactly 0.5 on these pairs, whereas Take The Best predicts a systematic preference. In pairs 5-8, the lexicographic winner loses the tally by respective score gaps of 1, 2, 3, and 5. Take The Best continues to predict an invariant preference for that winner, while Tallying predicts the opposite option with monotonically increasing confidence as the gap grows. Exact reversals balance response side and preserve every theoretical contrast. With 16 unique pairs, each is repeated six times.

[1] This design isolates a quantitative confidence signature without relying on opposite directional choices. The highest-validity expert (feature 5, validity 0.95) discriminates on every trial, so Take The Best always stops immediately and predicts the same probability for its winner on all eight pairs: epsilon/2 + (1-epsilon)*sigmoid(beta). The tally winner is always the same option as the Take-The-Best winner, but its feature-win margin is systematically manipulated across four conditions: 4-3 (gap 1), 5-2 (gap 3), 6-1 (gap 5), and 7-0 (gap 7), with two tied features in every condition. Tallying therefore predicts winner probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5,7}, producing a strictly increasing choice-consistency gradient except at practical saturation. Take The Best cannot produce this within-subject margin gradient because its binary score difference is always one. Each condition has an exact A/B reversal, balancing response side while preserving the theoretical contrast. There are eight unique pairs, so each is repeated 12 times for approximately 96 trials.

[2] Each base pair gives A one feature win and B two feature wins; the final eight pairs are exact A/B reversals. Tallying therefore predicts exactly the same B-choice probability on all eight base pairs because its tally margin is always 2-1, and the same mirrored A-choice probability on all reversals. Communicated-Reliability Accumulation instead distinguishes the identities of the winning experts. At representative parameter values (gamma about 1.035 and reliability_floor about 0.12), net evidence for A across the eight base pairs is approximately 2.71, 1.80, 1.00, 0.43, -0.10, -0.34, -0.71, and -1.40. Thus it predicts a strong, ordered choice gradient and a crossover from choosing the one-cue minority A to choosing the two-cue majority B. This ordering and crossover are robust over the advocated parameter ranges, whereas no setting of Tallying's beta or epsilon can produce stimulus-specific differences among pairs having the same tally margin. Exact reversals balance response side and should mirror the gradient. With 16 unique pairs, each pair is repeated six times for 96 trials.

[3] This six-pair design makes tally margin increase while communicated-reliability evidence decreases and ultimately reverses direction. In the first three pairs, A's tally margins are respectively 1, 3, and 5: A wins 1-0, 3-0, and 6-1 features. Tallying therefore predicts a strictly increasing A-choice gradient, with probabilities epsilon/2 + (1-epsilon)*sigmoid(beta*d) for d in {1,3,5}. Reliability accumulation predicts the opposite quantitative pattern because the larger tallies are built from progressively weaker experts. At representative parameters (gamma=1.035 and reliability_floor=0.12), its evidence for A is approximately 3.18, 2.44, and -1.04. These evidence values remain ordered across the specified parameter ranges, and the third remains negative: there, Tallying favors A by a five-feature margin, whereas reliability accumulation favors B because the 95%-valid expert outweighs six weak experts. The final three pairs are exact A/B reversals, balancing response side and mirroring both theories' predictions. Thus the critical signatures are an increasing choice-consistency gradient under Tallying versus a decreasing gradient and directional crossover under reliability accumulation. With six unique pairs, each is repeated 16 times for 96 trials.

[4] This eight-pair design holds the signed tally at exactly zero while varying coalition size and composition. In the four base pairs, A is endorsed by one 95%-valid expert plus respectively 0, 1, 2, or 3 near-chance (51%) experts; B is endorsed by respectively 1, 2, 3, or 4 experts of 73.5% validity. Because both sides have equal numbers of endorsers within every pair, the advocated model's common reliability floor cancels. Across its parameter range, its evidence for A is approximately 1.99–2.09, 1.01–1.10, 0.02–0.11, and -0.96–-0.88. It therefore predicts a steep decline in A choices and a robust directional crossover on the fourth condition, where four moderately reliable experts cumulatively outweigh the high expert and three weak experts. Compressed-Reliability Consensus instead assigns every condition a tied tally and compares bounded group summaries. Softmax attention keeps A's group summary dominated by the 95% expert: its reliability contrast remains positive at approximately 0.295, 0.273, 0.252, and 0.232. It consequently predicts A above chance in all four conditions, with only a shallow decline and no crossover. Thus the critical dissociation is a large accumulation-driven gradient ending below chance versus a near-flat, uniformly above-chance consensus pattern. The final four pairs are exact A/B reversals, balancing response side and requiring both patterns to mirror. Eight unique pairs are each repeated 12 times for 96 trials.

[5] This design targets both a robust confidence dissociation and a secondary composition-sensitive gradient. Every base pair gives A seven discriminating endorsements and B two, producing the same tally margin of five. Five of A's endorsements always come from 50%-valid experts. Compressed-Reliability Consensus treats these as full categorical votes and applies its saturated margin-five consensus multiplier; it therefore predicts substantially above-chance A choice. Even at its least confident parameter settings, its predicted A probability is approximately 0.67 or greater. Communicated-Reliability Accumulation gives each 50%-valid expert only the reliability floor, so these five endorsements contribute total evidence of only 5c = 0.50–0.70. The remaining two-versus-two coalitions were selected to have approximately equal summed transformed log-odds weights throughout the accumulation model's narrow gamma range. Consequently, accumulation predicts a nearly invariant and much weaker A preference, approximately 0.53–0.60 across base pairs. The four conditions also increase the dispersion of A's two non-chance reliabilities while preserving their accumulated weight: from 69%/80% to 51%/90%, opposed by two experts near 75%. Softmax compression increasingly emphasizes A's high-reliability member, moving the advocated model's reliability contrast from slightly negative to positive and producing an increasing choice-confidence gradient. Accumulation predicts essentially no such condition gradient because the transformed-weight sums are matched. Exact reversals balance response side and require mirrored predictions. There are eight unique pairs, each repeated 12 times for 96 trials.

[6] This 12-pair design isolates expert-by-expert residual accumulation from bounded group summarization while holding the categorical tally fixed. In each of the six base pairs, A is endorsed by m high-validity (95%) experts and B by m-1 near-chance (51%) experts, for m = 2,...,7. Thus A's tally margin is always exactly one, and both options always have at least one endorser. Compressed-Reliability Consensus predicts the same response probability in every base condition: because each coalition is homogeneous, its softmax-weighted summaries are always the 95% and 51% reliability codes, respectively; the reliability contrast is therefore constant, the tally is always +1, and consensus amplification is inactive. Across its parameter ranges, it predicts an approximately 0.55-0.64 A-choice probability with no coalition-size gradient. Uncertain-Credibility Normalized Voting instead preserves each expert's identity and accumulates residual deviations cue by cue. Every added 95% endorsement for A contributes a positive residual, while every added 51% endorsement for B also shifts residual evidence toward A because that weak B vote is discounted relative to the shared prototype. Divisive normalization slows but does not eliminate this accumulation. Its total evidence for A rises monotonically from approximately 1.54-2.17 at m=2 to 2.58-3.52 at m=7, yielding a substantial within-subject increase in A-choice consistency. The critical signature is therefore a strong monotonic coalition-size gradient under the advocated theory versus an exactly flat profile under the competing theory, despite an identical one-vote tally and identical within-side reliability summaries. The final six pairs are exact A/B reversals, balancing response side and requiring the gradient to mirror. With 12 unique pairs, each is repeated eight times for 96 trials.

[7] This 12-pair design targets the competing theory's exact-tally reliability gate. Every trial has exactly one expert endorsing A and one endorsing B, so the categorical tally is zero. In the six base pairs, A's endorser has validity 0.60, 0.80, or 1.00, whereas B's endorser always has validity 0.50. Each reliability contrast appears in two nuisance contexts: all nondiscriminating experts rate both products 0 or all rate both products 1. Compressed-Reliability Consensus leaves reliability contrasts active in exact ties and therefore predicts an ordered increase in A choices as A's endorser rises from 60% to 80% to 100%. Its softmax group summaries reduce to the two individual endorsers, making the two nuisance contexts exactly equivalent. Uncertain-Credibility Normalized Voting multiplies reliability evidence by only 0.06 in an exact tally tie and then further compresses the resulting tiny evidence through its convex response transformation. It consequently predicts choices extremely close to chance at all three reliability levels and an almost negligible gradient; at 60% versus 50%, lower-tail winsorization makes its signed reliability adjustment exactly zero. Thus the critical quantitative signature is a visible reliability-contrast gradient, reaching a meaningful above-chance preference in the 100%-versus-50% condition, under the advocated theory versus an essentially flat chance-level profile under the competing theory. The final six pairs are exact A/B reversals, balancing response side and requiring the gradient to mirror. With 12 unique pairs, each is repeated eight times for 96 trials.

[8] This eight-pair design targets the theories' opposite use of communicated validity while holding A's categorical tally advantage fixed at 7 endorsements versus 6. The first seven experts have validity .95 and the last seven validity .50. Across the four base pairs, the high-validity endorsement imbalance favoring A is +7, +3, -3, and -6. Under Uncertain-Credibility Normalized Voting, the corresponding signed residual-deviation sums are approximately +4.07, +1.60, -2.10, and -3.96. Its total evidence for A therefore decreases from approximately 2.58–3.53 to 1.62–1.99, then to approximately -0.30–0.19, and finally to -1.50–-0.55 over the reliability-shrinkage range. It predicts a steep conventional-validity gradient: strong A choice in the first conditions, near-chance choice in the third, and a B preference in the fourth despite A's one-vote tally advantage. The competing theory's dominant latent construal reverses validity direction. Its bounded reliability signal is nearly saturated whenever the high-validity imbalance is nonzero; hence it predicts the opposite gradient—B choices for the +7 and +3 conditions because reversed reliability overturns A's weak tally, but confident A choices for the -3 and -6 conditions because reversed reliability aligns with A's tally. The conflicts involve 13 discriminating experts, so the competing theory's compact-coalition attenuation is essentially inactive. Thus the theories predict oppositely signed condition gradients and opposite majority choices at both ends, not merely different overall accuracy. The final four pairs are exact A/B reversals, balancing response side and requiring mirrored effects. Eight unique pairs are each repeated 12 times for 96 trials.

[9] This eight-pair design isolates the advocated theory's exact-tally relative-salience gate while holding both the categorical tally and the signed validity imbalance constant. In each of the four base pairs, A and B receive equal numbers of endorsements. A always has one unmatched 100%-valid endorser and B one unmatched 50%-valid endorser, producing the same signed validity sum favoring A. Conditions 2-4 add one, two, or three canceling pairs of 100%-valid endorsers—one for A and one for B per pair. These additions leave the tally at zero and leave the signed validity sum unchanged, but increase total encoded validity strength. Consequently, relative salience under the advocated model falls exactly from 1 to 1/3, 1/5, and 1/7. For the dominant reversed-validity construal, the endpoint condition should therefore produce a substantial B preference despite A owning the 100%-valid endorsement, approximately 0.72-0.79 B choice across typical parameter combinations. Adding one balanced high-validity pair sharply gates this reliability evidence, and the 1/5 and 1/7 conditions should be close to chance. This predicts a pronounced nonlinear endpoint-versus-diffuse gradient even though raw reliability imbalance is identical. Uncertain-Credibility Normalized Voting instead applies its 0.06 exact-tie gate in every condition and then convexly compresses the already tiny residual evidence. It should remain extremely close to chance throughout, with at most a minute conventional preference for A and no sizable endpoint effect. Thus the critical dissociation is a strong reversed-validity choice in the concentrated 1-versus-1 endpoint condition followed by collapse toward chance under diffuse balanced coalitions, versus an essentially flat chance-level profile under the competing theory. The final four pairs are exact A/B reversals, balancing response side. Eight unique pairs are each repeated 12 times for 96 trials.

[10] This eight-pair design holds the categorical tally at a one-vote advantage for A while manipulating coalition size and proportional coherence. In the four base pairs, A receives m endorsements from 70%-valid experts and B receives m-1 endorsements from 50%-valid experts, for m = 2, 3, 4, and 5. Under Credibility-Claim Skepticism, a 70% claim has positive net weight throughout the full skepticism range: its weight is 0.4 - 0.128*claim_skepticism, or approximately 0.093 to 0.362. The 50% claims have zero claim weight but still contribute categorical votes. Thus both the fixed +1 tally and normalized claim evidence favor A in every base condition. As m increases, the normalization floor becomes less consequential, claim evidence and its share of total evidence increase, and the advocated theory predicts a monotonic increase in already substantial A-choice consistency. The competing theory's dominant reversed construal makes the accumulated 70% validity signal oppose A's tally. Critically, its special one-vote conflict gate protects the compact 2-versus-1 coalition because its relative consensus is 1/3, but progressively removes that protection for the diffuse 3-versus-2, 4-versus-3, and 5-versus-4 coalitions. For m >= 3, even the competing theory's weakest reversed reliability signal robustly overturns the +1 tally, invokes construal commitment, and predicts a B preference. It therefore predicts a sharply decreasing A-choice profile and a crossover toward B as coalition size grows, whereas the advocated theory predicts an increasing A-choice profile and no crossover. The final four pairs are exact A/B reversals, balancing response side and requiring both quantitative profiles to mirror. With eight unique pairs, each is repeated 12 times for approximately 96 trials.

[11] This 16-pair design creates a robust directional dissociation and a coalition-size gradient while keeping the categorical tally exactly zero. In the first six base pairs, A receives m endorsements from 70%-valid experts and B receives m endorsements from 50%-valid experts, for m = 1,...,6. For the competing theory, a 70% claim has net weight 0.4 - 0.128*claim_skepticism, which remains strictly positive (approximately 0.093 to 0.362) across its entire parameter range. With no tally evidence, it therefore predicts A above chance in every moderate-validity condition; divisive normalization also predicts increasing A-choice consistency as m grows and the normalization floor becomes less consequential. The advocated theory's dominant reversed construal gives the opposite prediction: the positive signed validity sum favoring A is interpreted as evidence for B. Relative salience equals one because the opposing 50% experts have zero encoded validity strength, so the exact-tie gate does not attenuate this signal. It predicts B above chance, increasing toward a bounded plateau with coalition size. Conditions 7 and 8 replace the moderate A coalition with one or three 95%-valid experts. They provide a secondary signature: the advocated theory remains monotonic and predicts at least as much B choice for 95% as for size-matched 70% coalitions, whereas the competing theory's extreme-claim correction changes the 95% net weight to 0.9 - 1.458*claim_skepticism and can reverse its direction when skepticism exceeds about 0.617. Thus the primary result is an oppositely directed coalition-size profile for guaranteed-positive 70% claims, supplemented by a theory-specific moderate-versus-extreme validity contrast. The final eight pairs are exact A/B reversals, balancing response side. With 16 unique pairs, each is repeated six times for 96 trials.

[12] This six-pair design holds the competing theory's complete decision-relevant summary constant while changing the advocated theory's pragmatic claim signal. In every base pair, A receives two endorsements and B receives one, so the tally margin is +1 and the discriminating count is three. The opponent validities were chosen to make the competing theory's signed validity sum identical across conditions: tanh(1.2) = tanh(1.35) - tanh(3*(0.51347-0.5)) = tanh(1.5) - tanh(3*(0.52387-0.5)), up to rounding. Thus its bounded reliability evidence, compact-coalition attenuation, reversal commitment, and predicted A-choice probability are effectively identical for the 90%, 95%, and 100% conditions, for every latent construal orientation. Contextual Anti-Reliability Arbitration instead predicts a pronounced 95% notch. The ordinary distrust claim is strong at 90% but sharply compressed at 95%; the small opposing 51.347% claim further cancels part of the already compressed 95% contrast. Literal certainty then activates endpoint distrust in the 100% condition, while its very weak 52.387% opponent produces little endpoint attenuation. Consequently, the advocated theory predicts substantially more A choices at 95% than at 90%, followed by a renewed decrease in A choices at 100% relative to 95%. The critical result is this nonmonotonic, endpoint-rebounding profile versus the competing theory's quantitatively flat profile. The final three pairs are exact A/B reversals, balancing response side. Six unique pairs are each repeated 16 times for 96 trials.

[13] This 12-pair design isolates dilution by coalition breadth and compositional heterogeneity while holding the advocated theory's complete reliability calculation constant. In each of the six base pairs, A has one 100%-valid endorser plus m-1 endorsers with 50% validity, while B has m endorsers with 50% validity, for m=1,...,6. Thus the categorical tally is exactly tied in every condition. For Latent Validity-Construal Tally, 50%-valid experts retain categorical votes but contribute exactly zero validity strength. Consequently, every base pair has the same signed validity sum favoring A, the same total encoded validity strength, relative salience exactly one, and the same bounded reliability evidence. The dominant reversed construal therefore predicts a substantial B preference that is quantitatively flat across coalition sizes; the rare conventional and validity-ignoring construals are likewise flat. Contextual Anti-Reliability Arbitration instead assigns the certainty claim a large endpoint-distrust signal but increasingly embeds that single nonzero claim among active neutral claims. As m rises, its breadth normalizer grows and the A coalition becomes increasingly heterogeneous because it mixes one endpoint claim with more zero-strength claims. Endpoint-versus-neutral protection leaves the endpoint signal intact locally, so the predicted reduction specifically arises during coalition arbitration: the B preference should decline monotonically toward chance from m=1 to m=6. The critical dissociation is therefore a flat, sizable reversed-validity preference under the advocated theory versus a strong coalition-size attenuation gradient under the competing theory, despite identical tally, signed validity imbalance, and exact-tie salience for the advocated model. The final six pairs are exact A/B reversals, balancing response side and requiring mirrored profiles. With 12 unique pairs, each is repeated eight times for 96 trials.

[14] This eight-pair design tests the advocated theory's condition-specific attenuation of endpoint distrust. In each base pair, the 100%-valid expert endorses A and exactly one opposing expert endorses B; the opponent's validity increases across 50%, 70%, 80%, and 90%. The categorical tally is always tied 1-1. Contextual Anti-Reliability Arbitration predicts a qualitative crossover. Against the 50% cue, the endpoint-versus-neutral exemption preserves the 100% expert's full ordinary and endpoint-specific distrust, producing a strong preference for B. A 70% opponent should still leave the endpoint claim more distrusted, whereas the 80% condition should approach the transition. Against the substantive 90% opponent, attentional focusing attenuates the endpoint cue's total distrust enough that the 90% claim becomes more distrusted than the 100% claim; the model therefore predicts a preference for A. Thus, paradoxically, strengthening the expert opposed to the endpoint cue should move choices from rejecting A to favoring A, crossing chance between the low- and high-validity conditions. Finite-Pool Pragmatic Claim Competition cannot produce this within-person crossover. With one endorser per side, both claims receive the same attention because their pairwise competition loads are identical, so allocated claim contrast is attention times [1-(2v_B-1)], which remains positive whenever v_B<1. A skeptical construal consequently favors B in every condition, merely declining toward chance as the opponent approaches 100%; a weak conventional construal favors A throughout and also cannot switch signs. The critical planned contrast is therefore endpoint-endorsed choice below chance for the 50%/70% opponents but above chance for the 90% opponent, rather than only a monotonic confidence reduction. The last four pairs are exact A/B reversals, balancing response side while preserving this interaction. Eight unique pairs are each repeated 12 times for approximately 96 trials.

[15] This six-pair design holds the categorical tally exactly tied at 4-4 while manipulating only the dispersion of B's claim validities. In every base pair, A is endorsed by four homogeneous 75%-valid experts (assertiveness 0.50 each). B's four endorsements have the same mean assertiveness of 0.70 in all three conditions: four 85% experts; 80%/82.5%/87.5%/90% experts; or 77.5%/82.5%/87.5%/92.5% experts. Finite-Pool Pragmatic Claim Competition predicts increasing A choice as B's coalition becomes more dispersed. Homogeneous B claims strongly compete with one another, whereas dispersion releases especially assertive B claims from similarity-based competition, making the allocated contrast increasingly favor B; the predominant skeptical construal reverses that contrast into stronger evidence for A. Contextual Anti-Reliability Arbitration predicts the opposite gradient. B has greater ordinary claim mass than A throughout, so anti-reliability initially favors A, but increasing within-B dispersion divisively dilutes that signal; the most dispersed condition also includes a smoothly compressed 92.5% claim, further reducing A-favoring anti-reliability evidence. It therefore predicts decreasing A choice toward chance. The critical test is the sign of the within-subject dispersion slope under an invariant 4-4 tally and invariant mean linear validity contrast. The last three pairs are exact option reversals. Six unique pairs are each repeated 16 times for 96 trials.

[16] This six-pair design holds the categorical tally, coalition breadth, and aggregate anti-reliability contrast approximately constant while manipulating only the internal dispersion of B's coalition. In every base pair, A is endorsed by four homogeneous 75%-valid experts. B is also endorsed by four experts, but its composition is either homogeneous (70%, 70%, 70%, 70%), moderately dispersed (60%, 65%, 75%, 81.3%), or strongly dispersed (50%, 70%, 80%, 85.6%). These compositions were chosen so that B's summed ordinary claim strength under Contextual Anti-Reliability Arbitration is approximately constant: each coalition sums to about four times 0.4^0.82. A's claim sum, the signed claim contrast, the 4-4 tally, and the eight-cue active breadth therefore remain fixed. The advocated theory predicts that increasing B-side claim dispersion enlarges its compositional normalizer and progressively dilutes anti-reliability evidence against the stronger A coalition. It consequently predicts an increasing A-choice profile toward chance from homogeneous to strongly dispersed B. Coalition-Level Skeptical Consensus predicts the opposite slope. Its B coalitions have nearly matched mean linear assertiveness, but their coherence falls sharply with dispersion. B's coalition summary therefore decreases, enlarging A's coalition-level advantage; skeptical reactance against A becomes stronger, so A choice should decrease as B becomes more dispersed. Thus the critical result is an oppositely signed within-subject dispersion gradient under an invariant tied tally and nearly invariant aggregate claim mass. The final three pairs are exact A/B reversals, balancing response side and requiring mirrored gradients. Six unique pairs are each repeated 16 times for 96 trials.

[17] This design tests the theories' quantitatively different breadth laws while holding the categorical tally, validity composition, and mean assertiveness of each coalition fixed. In each base pair, A receives m endorsements from 80%-valid experts and B receives m endorsements from 55%-valid experts, for m = 1, 2, 4, or 6. Thus every trial has an exactly tied m-versus-m tally, and both coalitions are internally homogeneous at every breadth. Coalition-Level Skeptical Consensus predicts a pronounced increase in B choices with m: the stronger A coalition grows as m^consensus_exponent, the breadth-reinforcement gain rises above its singleton floor, and the special narrow-conflict normalization becomes less influential, while broad-conflict compression is only logarithmic. Skeptical reactance therefore strengthens substantially from the isolated 1-versus-1 conflict to repeated consensus. Contextual Anti-Reliability Arbitration also distrusts A, but predicts a much shallower, rapidly saturating breadth profile. Its linearly growing claim contrast is divided by active breadth raised to coalition_dilution_exponent and then passed through a relatively steep tanh accumulator; consequently, substantial anti-reliability is already present for the singleton conflict and additional identical endorsements yield limited incremental change. The critical planned contrast is the increase in choice of the lower-validity coalition from m=1 to m=6: large under Coalition-Level Skeptical Consensus but comparatively small and plateau-like under Contextual Anti-Reliability Arbitration. Exact option reversals balance response side and require mirrored breadth functions. Eight unique pairs are repeated 12 times each for 96 trials.

[18] This eight-pair design holds coalition breadth and the categorical tally fixed at 4-versus-4 while varying only the dispersion of B's endorsing coalition. In every base pair, A is endorsed by four homogeneous 60%-valid experts. B is endorsed by four experts whose mean linear assertiveness is always 0.50: 75/75/75/75, 70/70/80/80, 60/60/90/90, or 50/50/100/100. Coalition-Level Skeptical Consensus therefore gives A a homogeneous summary proportional to 0.20 and B a summary proportional to 0.50 times its coherence. Because that theory's coherence floor is at least 0.42, B's summary is always at least 0.21 and thus remains larger than A's 0.20 in every condition. Its skeptical sign consequently predicts A above chance throughout, although progressively closer to chance as B becomes dispersed. The advocated Coherence-Gated Relative Coalition Comparison theory predicts a qualitatively different profile. Homogeneous and mildly dispersed B coalitions remain stronger than the homogeneous 60% A coalition, initially favoring A through skeptical reversal. However, the maximally dispersed 50/50/100/100 coalition lies far beyond the advocated theory's threshold-shaped coherence scale. Across its parameter range, B's gated mean representation in that condition is at most approximately 0.13 before the common breadth factor, whereas A's homogeneous representation is at least approximately 0.24. The coalition ordering therefore reverses, and the advocated theory predicts a crossover to choosing B. The intermediate 60/60/90/90 condition localizes the predicted coherence cliff. Thus the critical signature is a nonlinear A-to-B choice crossover under the advocated theory versus an attenuating but uniformly A-favoring profile under the competitor. Exact option reversals balance response side. Eight unique pairs are repeated 12 times each for 96 trials.

[19] This 12-pair design isolates the theories' different tally-margin curvature while making coalition-level validity contrast exactly zero. The 100%- and 80%-valid experts are nondiscriminating on every trial. All discriminating endorsements come from 50%-valid experts, whose assertiveness is encoded as exactly zero by both theories. Consequently, every endorsing coalition has representation zero regardless of its breadth, coherence, or composition, and skeptical evidence is exactly absent. The six base pairs give A unilateral tally margins of 1, 2, 3, 5, 7, and 9. Coalition-Level Skeptical Consensus therefore predicts log-odds proportional to m^q with q in [1.15,1.40], a convex, superlinear rise. Coherence-Gated Relative Coalition Comparison instead predicts log-odds proportional to (1+m)^q-1 with q in [0.78,0.96], a concave rise. This produces a parameter-robust quantitative dissociation: before response compression, the margin-9 to margin-1 evidence ratio is approximately 12.5-21.7 under the advocated theory but only about 7.0-8.6 under the competing theory. Subject-specific tally gain and response sensitivity rescale each curve without changing this curvature contrast, and the competing theory's conflict-dependent noise is inactive because dispersion is zero. The final six pairs are exact A/B reversals, balancing response side and requiring mirrored curves. With 12 unique pairs, each is repeated eight times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 16-pair design targets the competing theory's localized moderate-tie arbitration boost while removing tally evidence and holding coalition breadth and coherence constant. In every base pair, A is endorsed by two homogeneous experts of a common validity and B by two 50%-valid experts; hence the tally is exactly 2-2, both coalitions are homogeneous, and all conditions have four active cues. A's validity increases through a dense low-validity grid (50.5%, 51.25%, 52%, 52.75%, 53.5%, 55%, 60%, and 100%). Under Uncertainty-Arbitrated Overclaim Construal, the normalized reliability contrast passes through its narrow moderate-contrast band, centered at approximately 0.20-0.27, around the 52%-53.5% conditions. Because the tally is exactly tied and both homogeneous coalitions remain trustworthy, its moderate-tie gate boost should sharply increase claim arbitration there, producing a localized shoulder or unusually steep increase in choosing B, the option opposite A's stronger claim. Conditions immediately below and above the band identify this local curvature rather than merely an overall validity effect. Coherence-Gated Relative Coalition Comparison has no corresponding moderate-contrast window: with fixed equal breadth, zero tally, and zero dispersion, it predicts a smooth monotonic skeptical gradient governed by continuous validity encoding, normalization, and tanh compression. Its narrow-conflict adjustment is also substantially attenuated because four cues are active. The critical test is therefore the local second-difference/steepness around 52%-53.5% relative to the surrounding validity increments: a pronounced band-limited acceleration under the competing theory versus a smooth saturating function under the advocated theory. The 60% and 100% conditions anchor the broader curve and help distinguish the competing theory's trusted-extreme precision limiter from ordinary saturation. The final eight pairs are exact A/B reversals, balancing response side and requiring mirrored effects. Sixteen unique pairs are each repeated six times for 96 trials.",
  "validities": [
    0.5,
    0.5,
    0.505,
    0.505,
    0.5125,
    0.5125,
    0.52,
    0.52,
    0.5275,
    0.5275,
    0.535,
    0.535,
    0.55,
    0.55,
    0.6,
    0.6,
    1.0,
    1.0
  ],
  "trial_a_ratings": [
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 14489,
  "output_tokens": 4042
}
```
