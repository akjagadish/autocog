# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_13" and "pi_14") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_13" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_14" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_13
Conflict-Geometry Recruitment theory proposes that people probabilistically recruit one of three representations according to current conflict geometry. Raw counting dominates increasingly decisive vote imbalances. Coalition-claim construal compresses homogeneous repetition into redundant option-level claims and produces a rapidly saturating skeptical response to communicated-authority contrasts, but it is recruited only after isolated endorsements have smoothly coalesced into an active coalition. Moreover, when coalition construal is selected before both opposing sides have formed stable multi-expert coalitions, its pragmatic reversal is proportionally weak rather than fully expressed. Exemplar construal preserves diagnostic experts within heterogeneous coalitions through sidewise upper-tail concentration, but is recruited chiefly when tally conflict remains unresolved. Its suppression grows convexly with tally decisiveness, allowing diagnostic cues to influence weak conflicts without overriding decisive majorities. Stable correlated subject propensities influence recruitment, while trial-level stochastic construal is marginalized into predicted choice probabilities. The process is history independent and exactly invariant under exchanging options A and B.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Conflict-Geometry Recruitment model. History is intentionally ignored:
    # choices are not followed by information that could update expert quality.
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

    # Positive votes favor A and negative votes favor B. Agreements are not
    # evidence for either displayed option.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smooth authority coding on the full communicated-validity continuum.
    validity = np.clip(validities, 0.5, 1.0)
    raw_authority = 2.0 * (validity - 0.5)
    curvature = float(parameters["validity_curvature"])
    authority = np.tanh(curvature * raw_authority) / max(
        float(np.tanh(curvature)), 1e-12
    )

    exemplar_temperature = float(parameters["exemplar_temperature"])

    def represent(side):
        members = authority[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Coalition claims encode mean authority. Consequently, homogeneous
        # repetition is representationally redundant rather than cumulative.
        claim = float(np.mean(members))
        variance = float(np.mean((members - claim) ** 2))
        heterogeneity = float(np.clip(4.0 * variance, 0.0, 1.0))

        # A normalized soft maximum retains diagnostic members without using a
        # hard winner or an endpoint category. It equals the common authority
        # in a homogeneous coalition and approaches its upper tail when mixed.
        centered = exemplar_temperature * (members - np.max(members))
        weights = np.exp(centered)
        weights /= np.sum(weights)
        exemplar = float(np.dot(weights, members))

        # Upper-tail concentration is computed separately within each side, so
        # diagnostic substructure cannot cancel across opposing coalitions.
        upper_tail = max(exemplar - claim, 0.0)
        return claim, exemplar, heterogeneity, upper_tail, breadth

    claim_a, exemplar_a, hetero_a, tail_a, breadth_a = represent(1.0)
    claim_b, exemplar_b, hetero_b, tail_b, breadth_b = represent(-1.0)

    if active_count == 0:
        claim_contrast = 0.0
        exemplar_contrast = 0.0
        pooled_heterogeneity = 0.0
        pooled_upper_tail = 0.0
        sparsity = 1.0
    else:
        claim_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + abs(claim_a) + abs(claim_b)
        )
        claim_contrast = float(np.clip(claim_contrast, -1.0, 1.0))

        exemplar_contrast = (exemplar_a - exemplar_b) / (
            float(parameters["exemplar_floor"])
            + abs(exemplar_a)
            + abs(exemplar_b)
        )
        exemplar_contrast = float(np.clip(exemplar_contrast, -1.0, 1.0))

        pooled_heterogeneity = (
            breadth_a * hetero_a + breadth_b * hetero_b
        ) / float(active_count)
        pooled_upper_tail = (
            breadth_a * tail_a + breadth_b * tail_b
        ) / float(active_count)
        sparsity = 1.0 - float(active_count) / float(max(n_features, 1))

    # Decisiveness requires both a large proportional imbalance and enough
    # absolute votes. This distinguishes a sparse one-vote conflict from a
    # substantial majority without introducing windows for particular margins.
    if active_count == 0:
        normalized_margin = 0.0
    else:
        normalized_margin = margin_magnitude / float(active_count)
    size_certainty = 1.0 - np.exp(
        -margin_magnitude / float(parameters["margin_scale"])
    )
    decisiveness = float(normalized_margin * size_certainty)

    # Signed validity geometry relative to the tally is invariant to swapping
    # A and B: both factors reverse under a swap. Positive values mean that
    # authority and the raw majority point in the same direction.
    if margin_magnitude < 1e-12:
        validity_tally_alignment = 0.0
    else:
        validity_tally_alignment = float(
            np.sign(tally_margin) * claim_contrast
        )

    diagnostic_concentration = abs(exemplar_contrast - claim_contrast)
    diagnostic_scale = float(parameters["diagnostic_scale"])
    diagnostic_extremity = (
        diagnostic_concentration ** 2
        / (diagnostic_scale ** 2 + diagnostic_concentration ** 2 + 1e-12)
    )

    # Sidewise upper-tail concentration only exists when a coalition contains
    # diagnostic internal structure. Gating by heterogeneity prevents a merely
    # homogeneous authority contrast from duplicating coalition-claim evidence.
    upper_tail_salience = (
        pooled_upper_tail ** 2
        / (diagnostic_scale ** 2 + pooled_upper_tail ** 2 + 1e-12)
    ) * np.sqrt(max(pooled_heterogeneity, 0.0))

    # Correlated hierarchical propensities. One shared style variable loads in
    # opposite directions on counting versus interpretive construals, while
    # representation-specific deviations retain individual flexibility.
    shared_style = float(parameters["shared_style"]) - 0.5
    loading = float(parameters["trait_loading"])
    count_trait = (
        -loading * shared_style + float(parameters["count_offset"])
    )
    skeptic_trait = (
        loading * shared_style + float(parameters["skeptic_offset"])
    )
    exemplar_trait = (
        0.55 * loading * shared_style + float(parameters["exemplar_offset"])
    )

    # Claim recruitment responds rapidly to authority contrast independently
    # of coalition breadth. Only an additional, smaller smooth term depends on
    # sparsity, and it is strongest for highly authoritative claims.
    absolute_claim = abs(claim_contrast)
    saturated_claim = 1.0 - np.exp(
        -float(parameters["claim_saturation"]) * absolute_claim
    )
    authority_extremity = max(abs(claim_a), abs(claim_b)) ** 4

    # Isolated endorsements do not automatically form a coalition-level claim.
    # The gate rises smoothly with the absolute number of discriminating
    # experts and is already close to saturation for two multi-expert sides.
    formation_scale = float(parameters["coalition_formation_scale"])
    formation_power = float(parameters["coalition_formation_power"])
    coalition_formation = 1.0 - np.exp(
        -(float(active_count) / formation_scale) ** formation_power
    )
    coalition_formation = float(np.clip(coalition_formation, 1e-8, 1.0))

    # Smooth recruitment utilities. Convex decisiveness suppression leaves
    # weak conflicts accessible to exemplars while strongly protecting large
    # majorities. Upper-tail recruitment is restricted to unresolved conflict.
    count_utility = (
        float(parameters["count_bias"])
        + count_trait
        + float(parameters["margin_recruitment"]) * decisiveness
        + float(parameters["alignment_recruitment"])
        * validity_tally_alignment
    )
    claim_utility = (
        float(parameters["claim_bias"])
        + skeptic_trait
        + float(parameters["claim_salience"]) * saturated_claim
        + float(parameters["sparsity_claim_gain"])
        * sparsity * authority_extremity
        + float(parameters["margin_recruitment"]) * (1.0 - decisiveness)
        - float(parameters["heterogeneity_escape"])
        * pooled_heterogeneity
        + float(parameters["coalition_formation_gain"])
        * np.log(coalition_formation)
    )
    exemplar_utility = (
        float(parameters["exemplar_bias"])
        + exemplar_trait
        + float(parameters["exemplar_heterogeneity"])
        * pooled_heterogeneity
        + (1.0 - decisiveness)
        * (
            float(parameters["exemplar_diagnostic"])
            * diagnostic_extremity
            + float(parameters["exemplar_extremity_gain"])
            * upper_tail_salience
        )
        - float(parameters["exemplar_margin_suppression"])
        * decisiveness ** 2
    )

    utilities = np.asarray(
        [count_utility, claim_utility, exemplar_utility], dtype=np.float64
    )
    utilities /= float(parameters["recruitment_temperature"])
    utilities -= np.max(utilities)
    recruitment = np.exp(utilities)
    recruitment /= np.sum(recruitment)

    # Each representation supplies a complete conditional choice distribution.
    # The count channel grows with margin, the coalition channel reverses a
    # rapidly saturating authority contrast, and the exemplar channel follows
    # the diagnostic upper-tail contrast.
    if margin_magnitude < 1e-12:
        count_evidence = 0.0
    else:
        count_evidence = (
            float(parameters["count_strength"])
            * np.sign(tally_margin)
            * margin_magnitude ** float(parameters["count_power"])
        )

    # Full pragmatic reversal requires both sides to be construed as formed
    # coalitions. Separate smooth side-formation terms make singleton conflicts
    # weaker while rapidly approaching one for established multi-expert sides.
    bilateral_scale = float(parameters["bilateral_formation_scale"])
    bilateral_power = float(parameters["bilateral_formation_power"])
    side_formation_a = 1.0 - np.exp(
        -(float(breadth_a) / bilateral_scale) ** bilateral_power
    )
    side_formation_b = 1.0 - np.exp(
        -(float(breadth_b) / bilateral_scale) ** bilateral_power
    )
    bilateral_core = np.sqrt(max(side_formation_a * side_formation_b, 0.0))
    bilateral_floor = float(parameters["bilateral_formation_floor"])
    bilateral_formation = bilateral_floor + (
        1.0 - bilateral_floor
    ) * bilateral_core

    skeptical_multiplier = max(0.20, 1.0 + 0.55 * skeptic_trait)
    claim_evidence = -float(parameters["skeptic_strength"]) * (
        bilateral_formation
        * skeptical_multiplier
        * np.tanh(float(parameters["claim_slope"]) * claim_contrast)
    )

    exemplar_multiplier = max(0.20, 1.0 + 0.45 * exemplar_trait)
    exemplar_evidence = float(parameters["exemplar_strength"]) * (
        exemplar_multiplier
        * np.tanh(
            float(parameters["exemplar_slope"]) * exemplar_contrast
        )
    )

    precision = float(parameters["choice_precision"])
    evidences = precision * np.asarray(
        [count_evidence, claim_evidence, exemplar_evidence],
        dtype=np.float64,
    )

    # Marginalize stochastic representation recruitment. Computing a mixture
    # of channel-specific choices, rather than averaging evidence, preserves
    # psychologically distinct construals and their trial-level competition.
    channel_p_a = np.empty(3, dtype=np.float64)
    for i, evidence in enumerate(evidences):
        channel_logits = np.asarray(
            [0.5 * evidence, -0.5 * evidence], dtype=np.float64
        )
        channel_logits -= np.max(channel_logits)
        channel_probs = np.exp(channel_logits)
        channel_probs /= np.sum(channel_probs)
        channel_p_a[i] = channel_probs[0]

    p_a = float(np.dot(recruitment, channel_p_a))
    probabilities = np.asarray([p_a, 1.0 - p_a], dtype=np.float64)

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.sum(probabilities)
    return probabilities


`policy(probs) -> int`:
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


## THEORY 2 — pi_14
Bilateral Advocacy-Attribution theory proposes that communicated expert validity has two context-sensitive meanings. By default, endorsements are treated as ordinary evidence and choices follow a nonlinear raw tally whose influence rises smoothly with both absolute and proportional vote margin before saturating. This default governs unilateral endorsement and decisive majorities. A second interpretation becomes available only when endorsers have formed coalitions on both sides and the tally remains unresolved: coalition validity then conveys the intensity of an option-level advocacy claim rather than literal evidential weight. Each claim is represented by bounded mean communicated validity, so homogeneous repetition is largely redundant. Coalition coherence varies smoothly with within-side dispersion and modulates claim salience without any threshold or trust collapse. In balanced conflict, a more reliable and coherent coalition constitutes the more salient advocacy claim and is therefore the target of pragmatic skepticism. When one coalition is highly dispersed, it is a weak unified claim; the opposing homogeneous coalition becomes the salient overclaim, causing skepticism to favor the dispersed side. Arbitration into this skeptical interpretation increases continuously with bilateral formation, reliability contrast, and coherence asymmetry, while absolute and proportional tally decisiveness continuously suppress it. Thus trustworthy unilateral coalitions are followed rather than reversed, large majorities dominate, and anti-validity choice is concentrated in tied or weak-margin bilateral conflicts. Stable but constrained individual differences affect bilateral-skepticism propensity and choice precision.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Bilateral Advocacy-Attribution model. History is intentionally ignored:
    # without correctness feedback, past choices provide no basis for learning
    # the experts' diagnosticities.
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

    # Positive votes favor A and negative votes favor B. Features on which the
    # options agree are nondiscriminating and belong to neither coalition.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smoothly bounded coding of communicated validity. This has no special
    # endpoint category and retains sensitivity over the complete range.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["reliability_curvature"])
    raw_reliability = 2.0 * (validity - 0.5)
    reliability = np.tanh(curvature * raw_reliability) / max(
        float(np.tanh(curvature)), 1e-12
    )

    def coalition(side):
        members = reliability[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0

        # Advocacy intensity is an option-level bounded mean. Consequently,
        # repeating identical endorsers does not multiply the claim.
        mean_reliability = float(np.mean(members))
        dispersion = float(np.mean((members - mean_reliability) ** 2))

        # Dispersion weakens unified advocacy smoothly. The positive floor
        # prevents the hard trust collapse of threshold-based accounts.
        coherence_floor = float(parameters["coherence_floor"])
        coherence_scale = float(parameters["coherence_scale"])
        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(
            -dispersion / max(coherence_scale, 1e-12)
        )

        # A coherent reliable coalition is a salient advocacy claim. A highly
        # dispersed coalition remains represented, but as a weaker unified act.
        salience = mean_reliability * coherence
        return mean_reliability, float(coherence), float(salience), breadth

    mean_a, coherence_a, salience_a, breadth_a = coalition(1.0)
    mean_b, coherence_b, salience_b, breadth_b = coalition(-1.0)

    if active_count == 0:
        proportional_margin = 0.0
    else:
        proportional_margin = margin_magnitude / float(active_count)

    # Default nonlinear tally: absolute margin supplies sample strength, while
    # proportional margin supplies agreement. Both effects are continuous and
    # the resulting evidence saturates for decisive majorities.
    if margin_magnitude < 1e-12:
        absolute_resolution = 0.0
        tally_evidence = 0.0
    else:
        absolute_resolution = 1.0 - np.exp(
            -(margin_magnitude / float(parameters["margin_scale"]))
            ** float(parameters["margin_power"])
        )
        proportional_resolution = (
            float(parameters["proportion_floor"])
            + (1.0 - float(parameters["proportion_floor"]))
            * proportional_margin ** float(parameters["proportion_power"])
        )
        tally_shape = absolute_resolution * proportional_resolution
        tally_evidence = (
            float(parameters["tally_strength"])
            * np.sign(tally_margin)
            * tally_shape
        )

    # Reliability and coherence contrasts are relative option-level contrasts.
    # Relative normalization allows a small informative claim opposing a 50%
    # coalition to be salient without making validity magnitude generically
    # produce ever-stronger skepticism.
    salience_contrast = (salience_a - salience_b) / (
        float(parameters["claim_floor"])
        + abs(salience_a)
        + abs(salience_b)
    )
    salience_contrast = float(np.clip(salience_contrast, -1.0, 1.0))

    mean_contrast = (mean_a - mean_b) / (
        float(parameters["ordinary_claim_floor"])
        + abs(mean_a)
        + abs(mean_b)
    )
    mean_contrast = float(np.clip(mean_contrast, -1.0, 1.0))

    # Each side must form before validity can be attributed to opposed
    # advocacy. The geometric combination is exactly zero for unilateral
    # endorsement and rises smoothly from singleton to broad coalitions.
    formation_scale = float(parameters["formation_scale"])
    formation_power = float(parameters["formation_power"])
    formation_a = 1.0 - np.exp(
        -(float(breadth_a) / formation_scale) ** formation_power
    ) if breadth_a > 0 else 0.0
    formation_b = 1.0 - np.exp(
        -(float(breadth_b) / formation_scale) ** formation_power
    ) if breadth_b > 0 else 0.0
    bilateral_formation = float(np.sqrt(max(formation_a * formation_b, 0.0)))

    # Reliability contrast and coherence asymmetry make advocacy attribution
    # more diagnostic. Coherence asymmetry does not select an upper-tail member:
    # it only identifies which option-level coalition is the clearer claim.
    contrast_readiness = float(np.tanh(
        float(parameters["salience_slope"]) * abs(salience_contrast)
    ))
    coherence_asymmetry = abs(coherence_a - coherence_b)
    coherence_readiness = 1.0 + float(
        parameters["coherence_asymmetry_gain"]
    ) * coherence_asymmetry
    claim_readiness = float(np.clip(
        contrast_readiness * coherence_readiness, 0.0, 1.0
    ))

    # Decisive counting continuously resolves the ambiguity that licenses an
    # advocacy interpretation. There are no margin windows or endpoint rules.
    decisiveness = absolute_resolution * (
        float(parameters["decisiveness_floor"])
        + (1.0 - float(parameters["decisiveness_floor"]))
        * proportional_margin
    )
    unresolved = (1.0 - float(np.clip(decisiveness, 0.0, 1.0))) ** float(
        parameters["resolution_power"]
    )

    arbitration_weight = float(np.clip(
        bilateral_formation * claim_readiness * unresolved,
        0.0,
        1.0,
    ))

    # Outside bilateral unresolved conflict, mean validity is ordinary evidence.
    # This contribution is deliberately modest because the raw tally remains
    # the default, and it vanishes as advocacy attribution takes control.
    ordinary_evidence = (
        float(parameters["ordinary_strength"])
        * (1.0 - arbitration_weight)
        * mean_contrast
    )

    # In bilateral conflict, the more salient claim is pragmatically reversed.
    # If one side is dispersed, its salience falls and the coherent opponent is
    # treated as the overclaim, thereby favoring the dispersed coalition.
    skeptic_trait = float(parameters["bilateral_skepticism"])
    skeptic_multiplier = 1.0 + float(parameters["skeptic_trait_spread"]) * (
        skeptic_trait - 0.5
    )
    skeptic_multiplier = max(float(skeptic_multiplier), 0.25)
    advocacy_direction = float(np.tanh(
        float(parameters["advocacy_accumulation"]) * salience_contrast
    ))
    skeptical_evidence = -float(parameters["reversal_strength"]) * (
        skeptic_multiplier * arbitration_weight * advocacy_direction
    )

    total_evidence = tally_evidence + ordinary_evidence + skeptical_evidence

    # Choice precision is a stable, narrowly constrained individual property.
    # Ambiguous bilateral conflict modestly reduces effective precision without
    # generating large variance on clear tally trials.
    precision = float(parameters["choice_precision"])
    conflict_load = arbitration_weight * (1.0 - abs(advocacy_direction))
    precision /= 1.0 + float(parameters["conflict_noise"]) * conflict_load
    decision_variable = float(precision * total_evidence)

    # Positive evidence favors A. Symmetric stable softmax guarantees exact
    # invariance under exchanging the two displayed options.
    logits = np.asarray(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.sum(probabilities)
    return probabilities


`policy(probs) -> int`:
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


## EXPERIMENT 1 (proposed by pi_13)

### DESIGN
**Validities (n_features=12):** [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.85, 0.85, 0.85, 0.85, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** This design targets a representation-recruitment transition that Bilateral Advocacy-Attribution lacks. Every base trial is an unresolved 4-versus-4 conflict. B is always endorsed by four homogeneous 85%-valid experts, whereas A always contains the same single 70%-valid upper-tail expert and three fillers that are progressively changed from 50% to 60%: 50/50/50/70, 50/50/60/70, 50/60/60/70, and 60/60/60/70. Thus A's breadth and upper endpoint remain fixed while its mean authority rises and its heterogeneity falls. Conflict-Geometry Recruitment predicts a strongly increasing A-choice profile, potentially crossing chance. In the first condition, A's low mean but diagnostic 70% exemplar creates a large claim-versus-exemplar discrepancy; high heterogeneity recruits the exemplar channel, whose sidewise soft maximum still favors B's 85% coalition, so B should be preferred. Replacing neutral fillers with 60% fillers reduces heterogeneity and diagnostic concentration, suppressing exemplar recruitment. The coalition-claim channel then dominates and pragmatically reverses B's stronger mean-authority claim, producing increasing preference for A. Bilateral Advocacy-Attribution instead predicts A above chance throughout because B is always the more salient and coherent advocacy claim and is therefore skeptically reversed. Critically, it predicts the opposite quantitative slope: as A's fillers strengthen and A becomes more coherent, its salience approaches B's, reducing the salience contrast, arbitration weight, and skeptical preference for A. Its much weaker ordinary-validity term should not offset this decline. The key signature is therefore a B-to-A crossover or steep increase under Conflict-Geometry Recruitment versus a uniformly A-favoring but decreasing profile under Bilateral Advocacy-Attribution. Exact option reversals balance response side. Eight unique pairs are each repeated 12 times for approximately 96 trials. The nondiscriminating 95% expert broadens the communicated-validity range without contributing evidence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is the side-balanced linear trend in choosing the focal coalition containing the 70%-valid expert as its three fillers are upgraded from 50% to 60%. Conflict-Geometry Recruitment predicts a sizable positive trend because exemplar recruitment initially favors the homogeneous 85% coalition, whereas reduced heterogeneity later suppresses that channel and allows skeptical claim reversal to favor the focal coalition. Bilateral Advocacy-Attribution predicts the opposite trend: strengthening and increasing the coherence of the focal coalition reduces its skeptical advantage. Estimating the trend from all four levels and 24 observations per level improves precision, while computing it separately for the two exact option reversals removes stable A-versus-B response bias and subject-level differences in overall focal-coalition preference.

Source:
def metric(data: pd.DataFrame) -> float:
    # Estimate the linear change in preference for the 70%-expert coalition as
    # its three fillers are progressively upgraded from 50% to 60% validity.
    records = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        # Feature 6 is the fixed 70%-valid expert and uniquely identifies the
        # focal coalition. Its display side is reversed in half the trials.
        if a[6] > b[6]:
            focal_is_a = True
            focal = a
        elif b[6] > a[6]:
            focal_is_a = False
            focal = b
        else:
            continue

        # Features 3--5 are the three possible 60%-valid fillers. Their sum is
        # the designed progression level k in {0, 1, 2, 3}.
        k = int(np.rint(np.sum(focal[3:6])))
        if k < 0 or k > 3:
            continue

        response = int(row["response"])
        chose_focal = float((response == 0) if focal_is_a else (response == 1))
        records.append((focal_is_a, k, chose_focal))

    if len(records) == 0:
        return 0.0

    # Compute the trend separately for the two display orientations, then
    # average. This cancels stable A/B response biases while giving all four
    # experimental conditions equal weight.
    orientation_slopes = []
    for focal_is_a in (False, True):
        condition_means = []
        levels = []
        for k in range(4):
            vals = [y for side, level, y in records
                    if side == focal_is_a and level == k]
            if len(vals) > 0:
                levels.append(float(k))
                condition_means.append(float(np.mean(vals)))

        if len(levels) >= 2:
            x = np.asarray(levels, dtype=float)
            y = np.asarray(condition_means, dtype=float)
            xc = x - np.mean(x)
            denom = float(np.sum(xc ** 2))
            if denom > 0.0:
                orientation_slopes.append(float(np.sum(xc * y) / denom))

    if len(orientation_slopes) == 0:
        return 0.0

    # Multiplication by three expresses the fitted trend as the predicted
    # preference change from k=0 to k=3.
    return float(3.0 * np.mean(orientation_slopes))

### RESULTS
- Predicted under pi_13 (simulated): 0.4533 (var=0.0337)
- Predicted under pi_14 (simulated): -0.0902 (var=0.0067)
- Observed on real data: -0.3065 (var=0.0176)

## EXPERIMENT 2 (proposed by pi_14)

### DESIGN
**Validities (n_features=12):** [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.51, 0.51, 0.51, 0.51]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This eight-pair design isolates the competing theory's upper-tail exemplar channel from Bilateral Advocacy-Attribution's coherence-weighted coalition salience. Every base trial is an unresolved 4-versus-4 conflict. B is always endorsed by four homogeneous 51%-valid experts. A is endorsed by four experts drawn from a fixed pool of four 100%-valid and four 50%-valid experts; across conditions, the number of 100% endorsers is 1, 2, 3, or 4, with 50% endorsers filling the remaining positions. Bilateral Advocacy-Attribution predicts B choices throughout. Even the highly dispersed 1-high/3-neutral A coalition has coherence-bounded salience exceeding the nearly uninformative B coalition, so A is construed as the more salient advocacy claim and skeptically reversed. As additional 100% endorsers replace neutral fillers, A's mean reliability and coherence increase, strengthening rather than reversing this rejection of A. Conflict-Geometry Recruitment predicts a qualitatively different profile. With one 100% expert among three neutral fillers, A has a low coalition mean but an extreme upper-tail exemplar, creating large claim-versus-exemplar diagnostic concentration and strong heterogeneity-based exemplar recruitment; its exemplar channel therefore favors A. The two-high condition retains substantial upper-tail structure. As the coalition becomes dominated by 100% experts, exemplar and coalition summaries converge, diagnostic concentration disappears, and recruitment shifts toward the coalition-claim channel, which skeptically reverses A's stronger claim and favors B. Thus the critical signature is an A-to-B transition as the number of extreme endorsers rises under Conflict-Geometry Recruitment, versus a uniformly B-favoring and generally strengthening rejection profile under Bilateral Advocacy-Attribution. This differs from changing filler validity with a fixed upper-tail expert: here filler validity is fixed at 50%, and the manipulated variable is the multiplicity of endpoint exemplars. The final four pairs are exact A/B reversals, balancing response side. Eight unique pairs are each repeated 12 times for approximately 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is the side-invariant choice rate for the maximally heterogeneous coalition containing one 100%-valid and three 50%-valid endorsers, against the homogeneous four-member 51%-valid coalition. Conflict-Geometry Recruitment predicts frequent capture by the heterogeneous coalition because its isolated perfect expert creates a strong upper-tail exemplar and a large exemplar-versus-claim discrepancy. Bilateral Advocacy-Attribution instead represents the coalition through its coherence-bounded option-level salience and predicts skeptical rejection of that coalition, hence substantially fewer such choices. Restricting the metric to the condition where the exemplar channel is maximally diagnostic should yield a larger gap than averaging over later conditions where both theories favor the homogeneous coalition. Each subject contributes 24 repetitions across exactly balanced A/B orientations, which suppresses stable response-side bias and keeps the per-subject estimate relatively precise.

Source:
def metric(data: pd.DataFrame) -> float:
    captures = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features 0:8 define the heterogeneous coalition; features 8:12
        # define the homogeneous 51%-valid coalition. Determine which option
        # is supported by the former without relying on display side.
        mixed_margin = float(np.sum(a[:8]) - np.sum(b[:8]))
        if mixed_margin == 0.0:
            continue

        mixed_is_a = mixed_margin > 0.0
        mixed_ratings = a if mixed_is_a else b
        n_perfect_endorsers = int(np.sum(mixed_ratings[:4]))

        # The maximally heterogeneous condition has one 100%-valid endorser
        # and three 50%-valid endorsers.
        if n_perfect_endorsers != 1:
            continue

        response = int(row["response"])
        chose_mixed = (mixed_is_a and response == 0) or (
            (not mixed_is_a) and response == 1
        )
        captures.append(float(chose_mixed))

    if len(captures) == 0:
        return float("nan")
    return float(np.mean(captures))

### RESULTS
- Predicted under pi_13 (simulated): 0.7600 (var=0.0287)
- Predicted under pi_14 (simulated): 0.2225 (var=0.0071)
- Observed on real data: 0.8683 (var=0.0070)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Predicted under pi_13:** 0.2571 (var=0.0056)
**Predicted under pi_14:** 0.3292 (var=0.0051)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Predicted under pi_13:** 101.4886 (var=2.8840)
**Predicted under pi_14:** 187.4611 (var=2.5841)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with fewer positive ratings."""
    if data is None or len(data) == 0:
        return float("nan")

    minority_choices = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        tally_margin_a = float(np.sum(a > b) - np.sum(b > a))
        r = int(response)

        if tally_margin_a < 0:       # A has fewer cue wins
            minority_choices.append(float(r == 0))
        elif tally_margin_a > 0:     # B has fewer cue wins
            minority_choices.append(float(r == 1))
        # Tied tallies have no minority option and are omitted.

    if len(minority_choices) == 0:
        return float("nan")
    return float(np.mean(minority_choices))
```

**Observed (real) value:** 0.4508 (var=0.0031)
**Predicted under pi_13:** 0.2502 (var=0.0050)
**Predicted under pi_14:** 0.3169 (var=0.0017)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    a = np.vstack([np.asarray(x, dtype=float) for x in data["option_a_ratings"]])
    b = np.vstack([np.asarray(x, dtype=float) for x in data["option_b_ratings"]])
    responses = np.asarray(data["response"], dtype=int)

    # Positive margin means that A wins more binary feature comparisons;
    # negative margin means that B does.
    delta = a - b
    tally_margin = np.sum(delta > 0, axis=1) - np.sum(delta < 0, axis=1)

    # Select the two orientations of the diagnostic 6-versus-1 conflict:
    # the tally winner has a five-cue margin but is opposed by expert 1.
    strongest_cue = delta[:, 0]
    critical = (np.abs(tally_margin) == 5) & (strongest_cue * tally_margin < 0)
    if not np.any(critical):
        return float("nan")

    chose_a = responses == 0
    tally_winner_is_a = tally_margin > 0

    # Rate at which the numerous weak experts override the strongest expert.
    return float(np.mean(chose_a[critical] == tally_winner_is_a[critical]))
```

**Observed (real) value:** 0.8213 (var=0.0169)
**Predicted under pi_13:** 0.8313 (var=0.0084)
**Predicted under pi_14:** 0.8056 (var=0.0058)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as choosing the option endorsed by the 95%-valid
    # expert (feature 0), irrespective of whether that option is A or B.
    by_size = {1: [], 2: [], 4: []}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        # In this design, coalition size is the number of experts endorsing A
        # over B (equal to the number endorsing B over A).
        coalition_size = int(np.sum(difference > 0.0))
        if coalition_size not in by_size:
            continue

        high_expert_option = 0 if difference[0] > 0.0 else 1
        chose_high_expert = float(int(row['response']) == high_expert_option)
        by_size[coalition_size].append(chose_high_expert)

    if any(len(by_size[k]) == 0 for k in (1, 2, 4)):
        return float('nan')

    p1 = float(np.mean(by_size[1]))
    p2 = float(np.mean(by_size[2]))
    p4 = float(np.mean(by_size[4]))

    # Approximate inverse-noise weighting of the conditions with the largest
    # predicted between-theory probability differences. The size-3 condition
    # is omitted because both theories predict it close to chance.
    return float(1.8 * (p1 - 0.5) + 0.9 * (p2 - 0.5) - (p4 - 0.5))
```

**Observed (real) value:** -0.0382 (var=0.0352)
**Predicted under pi_13:** -0.3333 (var=0.0486)
**Predicted under pi_14:** -0.1537 (var=0.0452)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with the larger raw vote tally."""
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=np.float64)
        b = np.asarray(b_cell, dtype=np.float64)
        margin = float(np.sum(a - b))
        r = int(response)

        if margin > 0.0:
            scores.append(1.0 if r == 0 else 0.0)
        elif margin < 0.0:
            scores.append(1.0 if r == 1 else 0.0)
        else:
            scores.append(0.5)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8462 (var=0.0122)
**Predicted under pi_13:** 0.8612 (var=0.0028)
**Predicted under pi_14:** 0.8171 (var=0.0013)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response selected the option endorsed by the
    # high-validity experts (features 0 through 6), irrespective of screen side.
    weighted_success = 0.0
    total_weight = 0.0

    # Increasing weights emphasize the larger-coalition conditions, where the
    # theories' predicted choice probabilities diverge most, while retaining
    # lower-m conditions to keep the subject-level estimate stable.
    weights = np.asarray([0.30, 0.55, 0.72, 0.85, 0.94, 1.00], dtype=float)

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        high_margin = float(np.sum(a[:7] - b[:7]))
        m = int(round(abs(high_margin)))
        if high_margin == 0.0 or m < 2 or m > 7:
            continue

        chose_a = int(response) == 0
        high_option_is_a = high_margin > 0.0
        chose_high_coalition = float(chose_a == high_option_is_a)

        weight = float(weights[m - 2])
        weighted_success += weight * chose_high_coalition
        total_weight += weight

    if total_weight == 0.0:
        return float("nan")
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.1161 (var=0.0079)
**Predicted under pi_13:** 0.0835 (var=0.0026)
**Predicted under pi_14:** 0.0942 (var=0.0009)

### Experiment 10
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    # Bounded reliability code specified by Compressed-Reliability Consensus.
    validities = np.array([0.5, 0.6, 0.8, 1.0], dtype=np.float64)
    reliability_code = np.tanh(3.0 * (validities - 0.5)) / np.tanh(1.5)

    numerators = []
    weights = []
    subjects = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        contrast = float(np.dot(a - b, reliability_code))
        weight = abs(contrast)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        # +1 means the response followed the more reliable endorser, and -1
        # means it followed the less reliable endorser. This is invariant to
        # whether that endorser appeared on option A or option B.
        choice_sign = 1.0 - 2.0 * float(row["response"])
        aligned_choice = np.sign(contrast) * choice_sign
        numerators.append(weight * aligned_choice)
        weights.append(weight)
        subjects.append(row["subject_id"])

    if len(weights) == 0:
        return 0.0

    work = pd.DataFrame({
        "subject_id": subjects,
        "numerator": numerators,
        "weight": weights,
    })
    grouped = work.groupby("subject_id", sort=False)[["numerator", "weight"]].sum()
    valid = grouped["weight"] > 0.0
    if not bool(valid.any()):
        return 0.0

    subject_scores = (
        grouped.loc[valid, "numerator"] / grouped.loc[valid, "weight"]
    ).to_numpy(dtype=np.float64)

    # Normalized cumulative directional evidence: for one subject this is the
    # subject's weighted alignment score; pooled evidence grows with sqrt(N).
    return float(np.sqrt(subject_scores.size) * np.mean(subject_scores))

```

**Observed (real) value:** -3.5372 (var=0.0438)
**Predicted under pi_13:** -4.4524 (var=0.0353)
**Predicted under pi_14:** -4.6563 (var=0.0108)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    weighted_success = 0.0
    total_weight = 0.0

    # Conditions are represented in a canonical orientation using the sign of
    # the overall tally. This makes exact A/B-reversed pairs contribute to the
    # same planned condition.
    condition_weights = {
        7: 2.00,
        3: 1.45,
        -3: 0.80,
        -6: 1.10,
    }

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        if a.size < 7 or b.size != a.size:
            continue

        signed_votes = np.sign(a - b)
        high_validity_margin = float(np.sum(signed_votes[:7]))
        tally_margin = float(np.sum(signed_votes))
        if high_validity_margin == 0.0 or tally_margin == 0.0:
            continue

        canonical_margin = int(round(high_validity_margin * np.sign(tally_margin)))
        weight = condition_weights.get(canonical_margin, 1.0)

        response = int(row['response'])
        chose_high_validity_side = (
            (high_validity_margin > 0.0 and response == 0) or
            (high_validity_margin < 0.0 and response == 1)
        )
        weighted_success += weight * float(chose_high_validity_side)
        total_weight += weight

    if total_weight == 0.0:
        return float('nan')
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.3248 (var=0.0018)
**Predicted under pi_13:** 0.2713 (var=0.0034)
**Predicted under pi_14:** 0.1516 (var=0.0017)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # The concentrated endpoint condition has exactly two discriminating
        # experts: one 100%-valid expert and the final 50%-valid expert.
        if np.count_nonzero(diff) != 2 or diff.size < 8 or diff[7] == 0:
            continue

        response = int(row["response"])
        low_validity_side = 0 if diff[7] > 0 else 1
        scores.append(float(response == low_validity_side))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0104)
**Predicted under pi_13:** 0.8358 (var=0.0117)
**Predicted under pi_14:** 0.8267 (var=0.0060)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # The first five experts have 70% validity. Restrict attention to the
        # diffuse conflicts (3-vs-2, 4-vs-3, and 5-vs-4), where at least five
        # experts discriminate between the products.
        discriminating_count = int(np.count_nonzero(diff))
        if discriminating_count < 5:
            continue

        high_validity_margin = float(np.sum(diff[:5]))
        if high_validity_margin > 0:
            chose_high_validity_coalition = int(row['response']) == 0
        elif high_validity_margin < 0:
            chose_high_validity_coalition = int(row['response']) == 1
        else:
            continue

        scores.append(float(chose_high_validity_coalition))

    if not scores:
        return float('nan')
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1572 (var=0.0077)
**Predicted under pi_13:** 0.0964 (var=0.0043)
**Predicted under pi_14:** 0.0842 (var=0.0012)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    conventional_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Select the 12 mirrored conditions comparing equal-sized coalitions
        # of 70%-valid experts (indices 0:6) and 50%-valid experts (6:12),
        # excluding all conditions involving 95%-valid experts.
        d70 = d[:6]
        d50 = d[6:12]
        d95 = d[12:15]
        n70 = int(np.count_nonzero(d70))
        n50 = int(np.count_nonzero(d50))
        s70 = float(np.sum(d70))
        s50 = float(np.sum(d50))

        is_target = (
            n70 > 0
            and n70 == n50
            and abs(s70) == n70
            and abs(s50) == n50
            and np.sign(s70) == -np.sign(s50)
            and np.count_nonzero(d95) == 0
        )
        if not is_target:
            continue

        response = int(row["response"])
        # Score 1 when the chosen product is endorsed by the 70%-valid
        # coalition, regardless of whether that product appears as A or B.
        chose_70_option = ((s70 > 0 and response == 0) or
                           (s70 < 0 and response == 1))
        conventional_choices.append(float(chose_70_option))

    if len(conventional_choices) == 0:
        return float("nan")
    return float(np.mean(conventional_choices))
```

**Observed (real) value:** 0.1261 (var=0.0089)
**Predicted under pi_13:** 0.1078 (var=0.0040)
**Predicted under pi_14:** 0.0944 (var=0.0011)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        successes = np.zeros(3, dtype=np.float64)
        totals = np.zeros(3, dtype=np.float64)

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=np.float64)
            b = np.asarray(row["option_b_ratings"], dtype=np.float64)
            signed = np.sign(a - b)

            # Exactly one of the first three (90%, 95%, 100%) experts is
            # discriminating in each designed trial.
            focal = np.flatnonzero(signed[:3] != 0.0)
            if focal.size != 1:
                continue
            condition = int(focal[0])
            focal_sign = float(signed[condition])
            response = int(row["response"])

            # Score choosing the option endorsed by the focal high-validity
            # expert, making exact A/B reversals contribute on the same scale.
            chose_focal_option = (
                (focal_sign > 0.0 and response == 0)
                or (focal_sign < 0.0 and response == 1)
            )
            successes[condition] += float(chose_focal_option)
            totals[condition] += 1.0

        if np.any(totals == 0.0):
            return np.nan
        rates = successes / totals
        return float(rates[1] - 0.5 * (rates[0] + rates[2]))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_contrast(subj_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return subject_contrast(data)

```

**Observed (real) value:** 0.0094 (var=0.0050)
**Predicted under pi_13:** 0.0481 (var=0.0062)
**Predicted under pi_14:** -0.0197 (var=0.0084)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Preference for the option not endorsed by the perfectly valid expert,
    # scored separately at each coalition breadth and display orientation.
    cell_values = {(m, side): [] for m in range(1, 7) for side in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size == 0 or b.size != a.size:
            continue

        endpoint_side = int(np.sign(a[0] - b[0]))
        if endpoint_side == 0:
            continue

        active_count = int(np.count_nonzero(a - b))
        m = active_count // 2
        if m < 1 or m > 6 or active_count != 2 * m:
            continue

        response = int(row["response"])
        # endpoint_side=+1 means the certainty expert endorses A, so choosing
        # B is the anti-endpoint choice. For endpoint_side=-1, choosing A is.
        anti_endpoint_choice = float(
            (endpoint_side == 1 and response == 1)
            or (endpoint_side == -1 and response == 0)
        )
        cell_values[(m, endpoint_side)].append(anti_endpoint_choice)

    breadth_means = []
    for m in range(1, 7):
        orientation_means = []
        for side in (-1, 1):
            values = cell_values[(m, side)]
            if len(values) == 0:
                return float("nan")
            orientation_means.append(float(np.mean(values)))
        breadth_means.append(float(np.mean(orientation_means)))

    # A normalized linear low-to-high coalition-breadth contrast. Positive
    # values indicate that anti-endpoint preference attenuates as m increases.
    weights = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5], dtype=float)
    return float(np.dot(weights, np.asarray(breadth_means)) / 4.5)

```

**Observed (real) value:** 0.1781 (var=0.0129)
**Predicted under pi_13:** 0.3596 (var=0.0395)
**Predicted under pi_14:** -0.0294 (var=0.0060)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_endpoint_choices = []
    high_endpoint_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = a - b
        active = np.flatnonzero(diff != 0)
        if active.size != 2 or 0 not in active:
            continue

        opponents = active[active != 0]
        if opponents.size != 1:
            continue
        opponent = int(opponents[0])
        if diff[0] * diff[opponent] >= 0:
            continue

        response = int(row["response"])
        endpoint_chosen = (
            (diff[0] > 0 and response == 0)
            or (diff[0] < 0 and response == 1)
        )

        if opponent in (1, 2):
            low_endpoint_choices.append(float(endpoint_chosen))
        elif opponent == 4:
            high_endpoint_choices.append(float(endpoint_chosen))

    if len(low_endpoint_choices) == 0 or len(high_endpoint_choices) == 0:
        return float("nan")

    low_rate = float(np.mean(low_endpoint_choices))
    high_rate = float(np.mean(high_endpoint_choices))

    return float((low_rate < 0.5) and (high_rate > 0.5))

```

**Observed (real) value:** 0.0000 (var=0.2400)
**Predicted under pi_13:** 0.0000 (var=0.2176)
**Predicted under pi_14:** 1.0000 (var=0.2400)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    selected_choices = []

    # Coalitions opposed to the four 75%-valid experts. The maximally
    # dispersed coalition is intentionally excluded because the competing
    # theory predicts that its anti-reliability signal is strongly diluted.
    homogeneous = (4, 5, 6, 7)
    moderate_dispersion = (8, 9, 10, 11)
    selected = {homogeneous, moderate_dispersion}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        response = int(row["response"])

        # Locate the option endorsed by the four 75%-valid experts.
        a_has_low = int(np.sum(a[:4])) > int(np.sum(b[:4]))
        low_option = a if a_has_low else b
        opposing_option = b if a_has_low else a

        # Require the expected four-expert low-validity coalition and select
        # the two conditions where the theories' absolute predictions diverge.
        low_set = tuple(np.flatnonzero(low_option).tolist())
        opposing_set = tuple(np.flatnonzero(opposing_option).tolist())
        if low_set != (0, 1, 2, 3) or opposing_set not in selected:
            continue

        chose_low = (response == 0) if a_has_low else (response == 1)
        selected_choices.append(float(chose_low))

    if len(selected_choices) == 0:
        return 0.0
    return float(np.mean(selected_choices))

```

**Observed (real) value:** 0.8819 (var=0.0059)
**Predicted under pi_13:** 0.6663 (var=0.0061)
**Predicted under pi_14:** 0.5331 (var=0.0030)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the dispersion gradient after orienting every trial toward the
    # homogeneous coalition of four 75%-valid experts (features 0--3).
    cells = {("low", "A"): [], ("low", "B"): [],
             ("high", "A"): [], ("high", "B"): []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.int64)
        b = np.asarray(row["option_b_ratings"], dtype=np.int64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_reference = bool(np.all(a[:4] == 1) and np.all(a[4:] == 0))
        b_is_reference = bool(np.all(b[:4] == 1) and np.all(b[4:] == 0))

        if a_is_reference == b_is_reference:
            continue

        if a_is_reference:
            reference_side = "A"
            competitor = b
            chose_reference = float(int(row["response"]) == 0)
        else:
            reference_side = "B"
            competitor = a
            chose_reference = float(int(row["response"]) == 1)

        if np.all(competitor[4:8] == 1) and int(np.sum(competitor)) == 4:
            level = "low"
        elif np.all(competitor[12:16] == 1) and int(np.sum(competitor)) == 4:
            level = "high"
        else:
            # The moderately dispersed coalition is not needed for the
            # endpoint contrast.
            continue

        cells[(level, reference_side)].append(chose_reference)

    means = {}
    for key, values in cells.items():
        if len(values) == 0:
            return float("nan")
        means[key] = float(np.mean(np.asarray(values, dtype=np.float64)))

    # Average mirrored, side-specific contrasts so fixed A/B response bias
    # cannot masquerade as a dispersion effect.
    gradient_a = means[("high", "A")] - means[("low", "A")]
    gradient_b = means[("high", "B")] - means[("low", "B")]
    return float(0.5 * (gradient_a + gradient_b))
```

**Observed (real) value:** -0.3250 (var=0.0394)
**Predicted under pi_13:** -0.1138 (var=0.0119)
**Predicted under pi_14:** -0.3506 (var=0.0116)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    singleton_choices = []
    broad_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # In this design, the first six experts have validity .80 and the
        # final six have validity .55. Determine which displayed option is
        # endorsed by the higher-validity coalition.
        high_direction = float(np.sum(a[:6] - b[:6]))
        m = int(np.sum(np.abs(a[:6] - b[:6])))
        if high_direction == 0.0:
            continue

        # Code 1 when the lower-validity coalition's option was selected,
        # invariant to whether that option was displayed as A or B.
        if high_direction > 0.0:
            lower_validity_choice = float(response == 1)
        else:
            lower_validity_choice = float(response == 0)

        if m == 1:
            singleton_choices.append(lower_validity_choice)
        elif m in (4, 6):
            broad_choices.append(lower_validity_choice)

    if len(singleton_choices) == 0 or len(broad_choices) == 0:
        return float("nan")

    return float(np.mean(broad_choices) - np.mean(singleton_choices))
```

**Observed (real) value:** 0.0058 (var=0.0053)
**Predicted under pi_13:** 0.0700 (var=0.0088)
**Predicted under pi_14:** 0.0896 (var=0.0063)

### Experiment 21
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size < 20:
            continue

        active = np.abs(a - b) > 0.5
        # Select the maximally dispersed 50/50/100/100 condition and its
        # option-reversed counterpart: four conflicts at indices 0:4 and four
        # at indices 16:20, with no other active features.
        is_extreme = (
            int(np.sum(active[:4])) == 4
            and int(np.sum(active[16:20])) == 4
            and int(np.sum(active)) == 8
        )
        if not is_extreme:
            continue

        # The heterogeneous coalition is the option endorsed at indices 16:20.
        hetero_is_a = float(np.sum(a[16:20])) > float(np.sum(b[16:20]))
        response = int(row["response"])
        chose_heterogeneous = (response == 0) if hetero_is_a else (response == 1)
        scores.append(float(chose_heterogeneous))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0116)
**Predicted under pi_13:** 0.8958 (var=0.0071)
**Predicted under pi_14:** 0.8000 (var=0.0069)

### Experiment 22
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Majority-consistent choice rate on the three largest tally margins."""
    if len(data) == 0:
        return float("nan")

    margins = np.asarray([
        float(np.sum(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
        for a, b in zip(data["option_a_ratings"], data["option_b_ratings"])
    ])
    responses = data["response"].to_numpy(dtype=int)

    # In this design, this selects margins 5, 7, and 9 and their reversals.
    selected = np.abs(margins) >= 5.0
    if not np.any(selected):
        return float("nan")

    # A is majority-supported for positive margins; B for negative margins.
    majority_choice = np.where(margins > 0.0, responses == 0, responses == 1)
    return float(np.mean(majority_choice[selected]))
```

**Observed (real) value:** 0.8567 (var=0.0052)
**Predicted under pi_13:** 0.8075 (var=0.0051)
**Predicted under pi_14:** 0.8633 (var=0.0030)

### Experiment 23
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the probability of choosing the lower-validity (50%) coalition
    # in the five intermediate conditions: 51.25%, 52%, 52.75%, 53.5%, 55%.
    lower_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        difference = a - b

        # The nonbaseline coalition occupies one adjacent feature pair. The
        # selected range corresponds to feature indices 4 through 13.
        active_later = np.flatnonzero(np.abs(difference[2:]) > 0.5) + 2
        if active_later.size == 0:
            continue
        pair_start = int(np.min(active_later))
        if pair_start < 4 or pair_start > 12:
            continue

        # Positive means the more-valid coalition endorses A; negative means
        # it endorses B. Align responses so 1 always denotes selection of the
        # opposing, 50%-valid coalition.
        high_side = float(np.sum(difference[2:]))
        response = int(row["response"])
        if high_side > 0:
            lower_choices.append(float(response == 1))
        elif high_side < 0:
            lower_choices.append(float(response == 0))

    if len(lower_choices) == 0:
        return float("nan")
    return float(np.mean(lower_choices))
```

**Observed (real) value:** 0.8667 (var=0.0085)
**Predicted under pi_13:** 0.8703 (var=0.0098)
**Predicted under pi_14:** 0.8800 (var=0.0026)

### Experiment 24
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response is skeptical: choosing the option opposite
    # the coalition containing the informative (> .50-valid) experts.
    ordinary_scores = []
    endpoint_scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])
        signed = a - b

        ordinary_direction = float(np.sum(signed[:6]))   # 55%-valid experts
        endpoint_direction = float(np.sum(signed[6:12])) # 100%-valid experts

        if abs(ordinary_direction) > 1e-12:
            # Positive means the stronger coalition supports A, so choosing B
            # is the skeptical response. Negative reverses that coding.
            skeptical = response if ordinary_direction > 0 else 1 - response
            ordinary_scores.append(float(skeptical))
        elif abs(endpoint_direction) > 1e-12:
            skeptical = response if endpoint_direction > 0 else 1 - response
            endpoint_scores.append(float(skeptical))

    if len(ordinary_scores) == 0 or len(endpoint_scores) == 0:
        return float("nan")

    return float(np.mean(ordinary_scores) - np.mean(endpoint_scores))
```

**Observed (real) value:** -0.0058 (var=0.0033)
**Predicted under pi_13:** -0.0554 (var=0.0066)
**Predicted under pi_14:** -0.0254 (var=0.0050)

### Experiment 25
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Reversal-balanced choice rate for the high-spread skewed coalition."""
    y_choice_by_orientation = {True: [], False: []}
    target_active = np.arange(18, 26, dtype=int)

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        if a.size < 26 or b.size < 26:
            continue

        active = np.flatnonzero(a != b)
        if not np.array_equal(active, target_active):
            continue

        # In the largest-spread pair, expert 25 is the unique upper-tail
        # member of the positively skewed coalition Y.
        y_is_a = bool(a[25] > b[25])
        response = int(row["response"])
        chose_y = float(response == (0 if y_is_a else 1))
        y_choice_by_orientation[y_is_a].append(chose_y)

    orientation_means = []
    for y_is_a in (True, False):
        values = y_choice_by_orientation[y_is_a]
        if len(values) > 0:
            orientation_means.append(float(np.mean(values)))

    if len(orientation_means) == 0:
        return 0.5
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5000 (var=0.0088)
**Predicted under pi_13:** 0.5337 (var=0.0069)
**Predicted under pi_14:** 0.4625 (var=0.0068)

### Experiment 26
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of choosing the option endorsed by the unilateral coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        margin = float(np.sum(a - b))
        if not np.isfinite(margin) or abs(margin) < 1e-12:
            continue

        response = int(row["response"])
        chose_endorsed = ((margin > 0.0 and response == 0) or
                          (margin < 0.0 and response == 1))
        scores.append(float(chose_endorsed))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5763 (var=0.0023)
**Predicted under pi_13:** 0.5363 (var=0.0187)
**Predicted under pi_14:** 0.5962 (var=0.0031)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither theory is task-invariant, although pi_13 is the stronger overall account. The two focal experiments reveal a genuine cross-dissociation. In Experiment 1, the observed trend is strongly negative (-0.3065): improving and homogenizing the focal coalition reduces its choice share. Pi_14 predicts the correct direction (-0.0902), whereas pi_13 predicts a large positive trend (+0.4533), directly contradicting its proposed transition from exemplar to skeptical-claim recruitment. In Experiment 2, however, participants strongly choose the heterogeneous coalition containing one perfect expert (0.8683). Pi_13 captures this qualitatively and comes reasonably close (0.7600), while pi_14 predicts rejection of that coalition (0.2225). These discrepancies are not merely noisy pooled effects: the observed between-subject variances are modest, especially in Experiment 2 (0.0070), making pi_14's failure there particularly decisive.\n\nAcross Experiments 3-26, pi_13 more often provides the closer prediction. It performs well on decisive-majority behavior (Experiments 6 and 8), anti-validity choices in several balanced conflicts (Experiments 9, 12-14, and 23), the heterogeneous-coalition result in Experiment 21, and the categorical outcome of Experiment 17. It also substantially outperforms pi_14 in Experiments 11 and 18. Pi_14 has important local successes\u2014especially the dispersion gradient in Experiment 19, the large-margin result in Experiment 22, unilateral endorsement in Experiment 26, and the sign of Experiment 1\u2014but it fails sharply whenever a heterogeneous coalition's exceptional member attracts rather than repels choice, most notably Experiments 2 and 17. Its low simulated variances often accompany these confidently wrong predictions.\n\nPi_13 nevertheless should not simply be reparameterized and declared adequate. Its defining mechanism predicts the wrong sign in Experiment 1, underpredicts weak-cue or minority choices in Experiments 3 and 5, exaggerates the breadth contrast in Experiment 16, and understates the preference observed in Experiment 18. The combined evidence suggests a regularity not cleanly represented by either theory: within-coalition heterogeneity can confer positive evidential legitimacy or independence, rather than merely recruiting an upper-tail exemplar as in pi_13 or merely weakening a unified advocacy claim as in pi_14. This can explain why upgrading neutral fillers in Experiment 1 reduces preference by eliminating diversity, while a perfect expert embedded among neutral endorsers in Experiment 2 produces very strong capture. Because pi_14's core claim says the latter coalition should remain the salient target of skepticism, its failure is theoretical rather than a narrow parameter-range problem.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_14 with a Diversity-Legitimacy Attribution theory. The new theory should propose that people infer not only endorsement strength but also the likely informational independence or rhetorical coordination of each coalition. Homogeneous, high-validity coalitions are compressed into coordinated option-level claims and can become targets of skepticism in unresolved bilateral conflicts. In contrast, a coalition combining weak or neutral endorsers with an exceptional expert can be construed as diverse corroboration: heterogeneity makes the exceptional endorsement look independently diagnostic rather than like a stronger advocacy claim.\n\nThe theory should use a continuous, option-symmetric arbitration among: (1) a default nonlinear raw tally that dominates large absolute and proportional margins; (2) coordinated-claim discounting or pragmatic reversal for coherent coalitions in unresolved bilateral conflicts; and (3) a diversity-legitimacy signal combining breadth, dispersion, and upper-tail surprise. The third signal must be an interaction rather than a generic soft maximum: an extreme expert should gain positive influence when embedded in a sufficiently broad heterogeneous coalition, while an equally extreme but isolated or homogeneous endorsement may still be discounted as an overclaim. Coalition heterogeneity should therefore be capable of increasing evidential credibility, not only reducing advocacy salience.\n\nThis account should predict a negative trend in Experiment 1 because replacing neutral fillers with moderately valid fillers removes diversity and upper-tail distinctiveness; high capture in Experiment 2 because the perfect expert is legitimized by a broad heterogeneous coalition; the negative dispersion gradient in Experiment 19; strong heterogeneous-coalition choices in Experiment 21; modest following of unilateral coalitions in Experiment 26; and majority dominance in Experiments 6, 8, and 22. All gates should vary smoothly with breadth, dispersion, validity contrast, and tally decisiveness, without endpoint-specific rules. Stable subject differences should primarily govern sensitivity to coordination versus diversity, with constrained choice precision and lapse parameters. Calibration should be joint across all experiments and should reproduce both pooled metrics and the generally modest subject-level variances."
}
```

## Usage

```json
{
  "input_tokens": 38725,
  "output_tokens": 2538
}
```
