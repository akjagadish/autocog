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
**Description:** Bilateral Advocacy-Attribution theory proposes that communicated expert validity has two context-sensitive meanings. By default, endorsements are treated as ordinary evidence and choices follow a nonlinear raw tally whose influence rises smoothly with both absolute and proportional vote margin before saturating. This default governs unilateral endorsement and decisive majorities. A second interpretation becomes available only when endorsers have formed coalitions on both sides and the tally remains unresolved: coalition validity then conveys the intensity of an option-level advocacy claim rather than literal evidential weight. Each claim is represented by bounded mean communicated validity, so homogeneous repetition is largely redundant. Coalition coherence varies smoothly with within-side dispersion and modulates claim salience without any threshold or trust collapse. In balanced conflict, a more reliable and coherent coalition constitutes the more salient advocacy claim and is therefore the target of pragmatic skepticism. When one coalition is highly dispersed, it is a weak unified claim; the opposing homogeneous coalition becomes the salient overclaim, causing skepticism to favor the dispersed side. Arbitration into this skeptical interpretation increases continuously with bilateral formation, reliability contrast, and coherence asymmetry, while absolute and proportional tally decisiveness continuously suppress it. Thus trustworthy unilateral coalitions are followed rather than reversed, large majorities dominate, and anti-validity choice is concentrated in tied or weak-margin bilateral conflicts. Stable but constrained individual differences affect bilateral-skepticism propensity and choice precision.

**Parameters:**
- reliability_curvature: [1.45, 1.85]
- coherence_floor: [0.24, 0.34]
- coherence_scale: [0.055, 0.075]
- claim_floor: [0.025, 0.045]
- ordinary_claim_floor: [0.20, 0.30]
- formation_scale: [0.82, 1.02]
- formation_power: [1.45, 1.80]
- salience_slope: [2.55, 3.15]
- coherence_asymmetry_gain: [0.75, 1.05]
- margin_scale: [3.55, 3.90]
- margin_power: [3.0, 3.4]
- proportion_floor: [0.42, 0.52]
- proportion_power: [0.70, 0.95]
- tally_strength: [1.85, 2.05]
- decisiveness_floor: [0.60, 0.72]
- resolution_power: [1.65, 2.10]
- ordinary_strength: [0.15, 0.25]
- advocacy_accumulation: [1.45, 1.85]
- reversal_strength: [2.70, 3.15]
- bilateral_skepticism: [0, 1]
- skeptic_trait_spread: [0.12, 0.22]
- choice_precision: [0.94, 1.08]
- conflict_noise: [0.08, 0.22]
- lapse_rate: [0.015, 0.055]
- validities: validities

**`predict source code`:**
```python
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
**Description:** Conflict-Geometry Recruitment theory proposes that people probabilistically recruit one of three representations according to current conflict geometry. Raw counting dominates increasingly decisive vote imbalances. Coalition-claim construal compresses homogeneous repetition into redundant option-level claims and produces a rapidly saturating skeptical response to communicated-authority contrasts, but it is recruited only after isolated endorsements have smoothly coalesced into an active coalition. Moreover, when coalition construal is selected before both opposing sides have formed stable multi-expert coalitions, its pragmatic reversal is proportionally weak rather than fully expressed. Exemplar construal preserves diagnostic experts within heterogeneous coalitions through sidewise upper-tail concentration, but is recruited chiefly when tally conflict remains unresolved. Its suppression grows convexly with tally decisiveness, allowing diagnostic cues to influence weak conflicts without overriding decisive majorities. Stable correlated subject propensities influence recruitment, while trial-level stochastic construal is marginalized into predicted choice probabilities. The process is history independent and exactly invariant under exchanging options A and B.

**Parameters:**
- validity_curvature: [1.2, 2.4]
- claim_floor: [0.055, 0.14]
- claim_slope: [2.2, 4.0]
- claim_saturation: [3.5, 6.0]
- exemplar_temperature: [7.0, 12.0]
- exemplar_floor: [0.08, 0.18]
- exemplar_slope: [2.0, 3.6]
- diagnostic_scale: [0.12, 0.22]
- margin_scale: [1.8, 3.2]
- count_power: [0.42, 0.60]
- count_strength: [0.52, 0.72]
- skeptic_strength: [2.5, 3.8]
- exemplar_strength: [5.0, 8.0]
- shared_style: [0, 1]
- count_offset: [-0.55, 0.55]
- skeptic_offset: [-0.55, 0.55]
- exemplar_offset: [-0.25, 0.25]
- trait_loading: [0.75, 1.30]
- count_bias: [-0.15, 0.30]
- claim_bias: [0.05, 0.45]
- exemplar_bias: [-1.25, -0.65]
- margin_recruitment: [2.2, 3.5]
- alignment_recruitment: [-0.35, 0.35]
- claim_salience: [0.30, 0.65]
- sparsity_claim_gain: [0.02, 0.15]
- heterogeneity_escape: [1.4, 2.8]
- coalition_formation_scale: [2.3, 2.8]
- coalition_formation_power: [2.8, 3.8]
- coalition_formation_gain: [0.75, 1.10]
- bilateral_formation_floor: [0.35, 0.55]
- bilateral_formation_scale: [1.15, 1.50]
- bilateral_formation_power: [1.8, 2.8]
- exemplar_heterogeneity: [1.2, 2.5]
- exemplar_diagnostic: [3.5, 5.5]
- exemplar_extremity_gain: [1.4, 2.8]
- exemplar_margin_suppression: [10.0, 15.0]
- recruitment_temperature: [0.65, 1.15]
- choice_precision: [0.90, 1.20]
- lapse_rate: [0.0, 0.055]
- validities: validities

**`predict source code`:**
```python
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

[20] This 16-pair design targets the competing theory's localized moderate-tie arbitration boost while removing tally evidence and holding coalition breadth and coherence constant. In every base pair, A is endorsed by two homogeneous experts of a common validity and B by two 50%-valid experts; hence the tally is exactly 2-2, both coalitions are homogeneous, and all conditions have four active cues. A's validity increases through a dense low-validity grid (50.5%, 51.25%, 52%, 52.75%, 53.5%, 55%, 60%, and 100%). Under Uncertainty-Arbitrated Overclaim Construal, the normalized reliability contrast passes through its narrow moderate-contrast band, centered at approximately 0.20-0.27, around the 52%-53.5% conditions. Because the tally is exactly tied and both homogeneous coalitions remain trustworthy, its moderate-tie gate boost should sharply increase claim arbitration there, producing a localized shoulder or unusually steep increase in choosing B, the option opposite A's stronger claim. Conditions immediately below and above the band identify this local curvature rather than merely an overall validity effect. Coherence-Gated Relative Coalition Comparison has no corresponding moderate-contrast window: with fixed equal breadth, zero tally, and zero dispersion, it predicts a smooth monotonic skeptical gradient governed by continuous validity encoding, normalization, and tanh compression. Its narrow-conflict adjustment is also substantially attenuated because four cues are active. The critical test is therefore the local second-difference/steepness around 52%-53.5% relative to the surrounding validity increments: a pronounced band-limited acceleration under the competing theory versus a smooth saturating function under the advocated theory. The 60% and 100% conditions anchor the broader curve and help distinguish the competing theory's trusted-extreme precision limiter from ordinary saturation. The final eight pairs are exact A/B reversals, balancing response side and requiring mirrored effects. Sixteen unique pairs are each repeated six times for 96 trials.

[21] This 16-pair design targets the competing theory's narrow-conflict focus rule while holding tally margin, coalition coherence, and within-condition mean reliability contrast fixed. In the first four base pairs, A is endorsed by m homogeneous 55%-valid experts and B by m homogeneous 50%-valid experts, for m = 1, 2, 4, or 6. The next four use homogeneous 100%-valid A endorsers against the same number of 50%-valid B endorsers. Every trial is an exact m-versus-m tally tie, and all coalitions have zero dispersion. Under Uncertainty-Arbitrated Overclaim Construal, homogeneous repetition does not amplify either coalition's claim: its reliability contrast is exactly invariant across m. Sampling-based metacognitive trust saturates rapidly, so especially beyond the smallest coalition it predicts an approximately flat breadth profile. The 55%-versus-50% contrast is also above the narrow moderate-tie band for most of the advocated parameter range, limiting contamination by that separate mechanism. Coherence-Gated Relative Coalition Comparison likewise has a breadth-normalized contextual contrast, but its explicit narrow-conflict focus selectively attenuates ordinary singleton contrasts. Consequently, it predicts substantially fewer choices of B, the skeptical response to A's stronger claim, at m=1 than at m=4 or m=6 in the 55%-versus-50% series. This breadth effect should be much smaller in the 100%-versus-50% series because its contextual contrast is near maximal and therefore escapes singleton attenuation. The critical quantitative signature is thus a selective ordinary-contrast breadth increase under the competing theory versus near breadth invariance under the advocated theory; the endpoint series serves as a within-subject control showing that any effect is not generic repetition sensitivity. The final eight pairs are exact A/B reversals, balancing response side. Each unique pair is repeated six times for 96 trials.

[22] This design isolates sensitivity to coalition skew and upper-tail structure while matching lower-order coalition statistics. In each base trial, A is endorsed by a positively skewed four-expert coalition Y={c-a/sqrt(3), c-a/sqrt(3), c-a/sqrt(3), c+sqrt(3)a}, whereas B is endorsed by a symmetric coalition X={c-a,c-a,c+a,c+a}, with c=0.65 and a in {0.05,0.08,0.11}. The two coalitions have exactly the same raw mean validity c, variance a^2, breadth four, and 4-versus-4 tally. Only their higher-order composition differs: Y has the more diagnostic upper tail, while X places more mass at both symmetric extremes. Conflict-Geometry Recruitment predicts increasing choice of A as a grows. Its coalition means remain nearly matched, but its sidewise soft maximum increasingly preserves Y's single high-validity expert; growing heterogeneity recruits the exemplar channel, while the tied tally cannot favor either option. Uncertainty-Arbitrated Overclaim Construal has no upper-tail representation. After its nonlinear reliability transform, X generally has greater reliability dispersion than Y. As dispersion approaches its threshold, X loses coalition trust more rapidly, making the relatively coherent Y coalition the salient overclaim; skeptical reversal therefore favors B. Thus the largest-spread condition is designed to produce an especially diagnostic directional dissociation—A under the advocated theory versus B under the competitor—along with oppositely signed dispersion gradients. The smaller spreads locate the onset of the effect and distinguish the advocated theory's smooth upper-tail recruitment from the competitor's threshold-shaped trust transition. The 50% and 100% experts provide communicated-validity anchors but are nondiscriminating throughout. The final three trials are exact A/B reversals. Six unique pairs are each repeated 16 times for 96 trials.

[23] This eight-pair design targets the theories' opposite treatment of a unilateral two-expert coalition. In each of the four base pairs, exactly two equally valid experts endorse A and no expert endorses B; all other experts are nondiscriminating. The endorsers' common validity increases from 70% through 80% and 90% to 100%, while the active count and A's tally margin remain fixed at two. Uncertainty-Arbitrated Overclaim Construal treats the homogeneous pair as a trustworthy, salient option-level claim. Because dispersion is zero and a two-member coalition already has high sampling confidence, its skeptical construal should reverse that claim and favor B. Increasing validity strengthens the normalized claim while leaving tally evidence unchanged, predicting an increasing rejection of A, followed by saturation. Conflict-Geometry Recruitment predicts the opposite choice direction. With no opposing coalition, its bilateral-formation rule restricts the strength of pragmatic reversal to the bilateral floor; the sparse active set also limits coalition recruitment. Its count and exemplar channels both favor A, and the fixed two-vote unilateral margin substantially recruits counting. It should therefore choose A above chance throughout, with a much weaker validity-dependent decline at most. The critical signature is thus endorsement-consistent A choice under Conflict-Geometry Recruitment versus endorsement-reversing B choice under Uncertainty-Arbitrated Overclaim Construal, accompanied by a substantially steeper reversed-validity gradient in the latter. The final four pairs are exact A/B reversals, balancing response side. Eight unique pairs are each repeated 12 times for approximately 96 trials.

[24] This design targets a representation-recruitment transition that Bilateral Advocacy-Attribution lacks. Every base trial is an unresolved 4-versus-4 conflict. B is always endorsed by four homogeneous 85%-valid experts, whereas A always contains the same single 70%-valid upper-tail expert and three fillers that are progressively changed from 50% to 60%: 50/50/50/70, 50/50/60/70, 50/60/60/70, and 60/60/60/70. Thus A's breadth and upper endpoint remain fixed while its mean authority rises and its heterogeneity falls. Conflict-Geometry Recruitment predicts a strongly increasing A-choice profile, potentially crossing chance. In the first condition, A's low mean but diagnostic 70% exemplar creates a large claim-versus-exemplar discrepancy; high heterogeneity recruits the exemplar channel, whose sidewise soft maximum still favors B's 85% coalition, so B should be preferred. Replacing neutral fillers with 60% fillers reduces heterogeneity and diagnostic concentration, suppressing exemplar recruitment. The coalition-claim channel then dominates and pragmatically reverses B's stronger mean-authority claim, producing increasing preference for A. Bilateral Advocacy-Attribution instead predicts A above chance throughout because B is always the more salient and coherent advocacy claim and is therefore skeptically reversed. Critically, it predicts the opposite quantitative slope: as A's fillers strengthen and A becomes more coherent, its salience approaches B's, reducing the salience contrast, arbitration weight, and skeptical preference for A. Its much weaker ordinary-validity term should not offset this decline. The key signature is therefore a B-to-A crossover or steep increase under Conflict-Geometry Recruitment versus a uniformly A-favoring but decreasing profile under Bilateral Advocacy-Attribution. Exact option reversals balance response side. Eight unique pairs are each repeated 12 times for approximately 96 trials. The nondiscriminating 95% expert broadens the communicated-validity range without contributing evidence.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This eight-pair design isolates the competing theory's upper-tail exemplar channel from Bilateral Advocacy-Attribution's coherence-weighted coalition salience. Every base trial is an unresolved 4-versus-4 conflict. B is always endorsed by four homogeneous 51%-valid experts. A is endorsed by four experts drawn from a fixed pool of four 100%-valid and four 50%-valid experts; across conditions, the number of 100% endorsers is 1, 2, 3, or 4, with 50% endorsers filling the remaining positions. Bilateral Advocacy-Attribution predicts B choices throughout. Even the highly dispersed 1-high/3-neutral A coalition has coherence-bounded salience exceeding the nearly uninformative B coalition, so A is construed as the more salient advocacy claim and skeptically reversed. As additional 100% endorsers replace neutral fillers, A's mean reliability and coherence increase, strengthening rather than reversing this rejection of A. Conflict-Geometry Recruitment predicts a qualitatively different profile. With one 100% expert among three neutral fillers, A has a low coalition mean but an extreme upper-tail exemplar, creating large claim-versus-exemplar diagnostic concentration and strong heterogeneity-based exemplar recruitment; its exemplar channel therefore favors A. The two-high condition retains substantial upper-tail structure. As the coalition becomes dominated by 100% experts, exemplar and coalition summaries converge, diagnostic concentration disappears, and recruitment shifts toward the coalition-claim channel, which skeptically reverses A's stronger claim and favors B. Thus the critical signature is an A-to-B transition as the number of extreme endorsers rises under Conflict-Geometry Recruitment, versus a uniformly B-favoring and generally strengthening rejection profile under Bilateral Advocacy-Attribution. This differs from changing filler validity with a fixed upper-tail expert: here filler validity is fixed at 50%, and the manipulated variable is the multiplicity of endpoint exemplars. The final four pairs are exact A/B reversals, balancing response side. Eight unique pairs are each repeated 12 times for approximately 96 trials.",
  "validities": [
    1.0,
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.5,
    0.5,
    0.51,
    0.51,
    0.51,
    0.51
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      1,
      1,
      1,
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
      1,
      1,
      1,
      0,
      1,
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
      1,
      1,
      1,
      1
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
      1,
      1
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
      1,
      1
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
      1,
      1
    ]
  ],
  "trial_b_ratings": [
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
      1,
      1
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
      1,
      1
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
      1,
      1
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
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1,
      1,
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
      1,
      1,
      1,
      0,
      1,
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
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 16593,
  "output_tokens": 3725
}
```
