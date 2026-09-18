# Round 12 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_13` — SURVIVED ✓

**Description:** Conflict-Geometry Recruitment theory proposes that people probabilistically recruit one of three representations according to current conflict geometry. Raw counting dominates increasingly decisive vote imbalances. Coalition-claim construal compresses homogeneous repetition into redundant option-level claims and produces a rapidly saturating skeptical response to communicated-authority contrasts, but it is recruited only after isolated endorsements have smoothly coalesced into an active coalition. Moreover, when coalition construal is selected before both opposing sides have formed stable multi-expert coalitions, its pragmatic reversal is proportionally weak rather than fully expressed. Exemplar construal preserves diagnostic experts within heterogeneous coalitions through sidewise upper-tail concentration, but is recruited chiefly when tally conflict remains unresolved. Its suppression grows convexly with tally decisiveness, allowing diagnostic cues to influence weak conflicts without overriding decisive majorities. Stable correlated subject propensities influence recruitment, while trial-level stochastic construal is marginalized into predicted choice probabilities. The process is history independent and exactly invariant under exchanging options A and B.

**Rationale:** This is a localized edit to the accepted iteration-7 model. All recruitment utilities, representation summaries, count calibration, convex exemplar suppression, and the successful heterogeneous upper-tail mechanism are unchanged. The sole mechanistic addition is a bilateral coalition-formation multiplier on the claim channel's conditional skeptical evidence. Separate saturating functions of the two coalition breadths are combined geometrically, so a strong pragmatic reversal is expressed only when both sides are represented as established coalitions. A nonzero floor makes the intervention modest rather than eliminating claim responses in singleton or one-sided conflicts. This directly addresses the latest diagnosis: partially formed claims can still be recruited, but they no longer generate the same near-maximal reversal as multi-expert coalitions. It should soften sparse anti-authority behavior in Experiments 1, 3, 5, 8, 10, and the low-breadth cells of Experiment 14 while leaving broad established-coalition responses in Experiments 16, 18, and 21 nearly unchanged. Because the multiplier depends only on sidewise breadth and is smooth, it introduces no validity endpoint, feature-count sparsity rule, or hand-coded margin window. History independence, trial-level stochastic recruitment, and exact option-swap symmetry are preserved.

**Parameters:**
  - `validity_curvature`: `[1.2, 2.4]`
  - `claim_floor`: `[0.055, 0.14]`
  - `claim_slope`: `[2.2, 4.0]`
  - `claim_saturation`: `[3.5, 6.0]`
  - `exemplar_temperature`: `[7.0, 12.0]`
  - `exemplar_floor`: `[0.08, 0.18]`
  - `exemplar_slope`: `[2.0, 3.6]`
  - `diagnostic_scale`: `[0.12, 0.22]`
  - `margin_scale`: `[1.8, 3.2]`
  - `count_power`: `[0.42, 0.60]`
  - `count_strength`: `[0.52, 0.72]`
  - `skeptic_strength`: `[2.5, 3.8]`
  - `exemplar_strength`: `[5.0, 8.0]`
  - `shared_style`: `[0, 1]`
  - `count_offset`: `[-0.55, 0.55]`
  - `skeptic_offset`: `[-0.55, 0.55]`
  - `exemplar_offset`: `[-0.25, 0.25]`
  - `trait_loading`: `[0.75, 1.30]`
  - `count_bias`: `[-0.15, 0.30]`
  - `claim_bias`: `[0.05, 0.45]`
  - `exemplar_bias`: `[-1.25, -0.65]`
  - `margin_recruitment`: `[2.2, 3.5]`
  - `alignment_recruitment`: `[-0.35, 0.35]`
  - `claim_salience`: `[0.30, 0.65]`
  - `sparsity_claim_gain`: `[0.02, 0.15]`
  - `heterogeneity_escape`: `[1.4, 2.8]`
  - `coalition_formation_scale`: `[2.3, 2.8]`
  - `coalition_formation_power`: `[2.8, 3.8]`
  - `coalition_formation_gain`: `[0.75, 1.10]`
  - `bilateral_formation_floor`: `[0.35, 0.55]`
  - `bilateral_formation_scale`: `[1.15, 1.50]`
  - `bilateral_formation_power`: `[1.8, 2.8]`
  - `exemplar_heterogeneity`: `[1.2, 2.5]`
  - `exemplar_diagnostic`: `[3.5, 5.5]`
  - `exemplar_extremity_gain`: `[1.4, 2.8]`
  - `exemplar_margin_suppression`: `[10.0, 15.0]`
  - `recruitment_temperature`: `[0.65, 1.15]`
  - `choice_precision`: `[0.90, 1.20]`
  - `lapse_rate`: `[0.0, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


### slot 2 — `pi_14` — KILLED ✗

**Description:** Bilateral Advocacy-Attribution theory proposes that communicated expert validity has two context-sensitive meanings. By default, endorsements are treated as ordinary evidence and choices follow a nonlinear raw tally whose influence rises smoothly with both absolute and proportional vote margin before saturating. This default governs unilateral endorsement and decisive majorities. A second interpretation becomes available only when endorsers have formed coalitions on both sides and the tally remains unresolved: coalition validity then conveys the intensity of an option-level advocacy claim rather than literal evidential weight. Each claim is represented by bounded mean communicated validity, so homogeneous repetition is largely redundant. Coalition coherence varies smoothly with within-side dispersion and modulates claim salience without any threshold or trust collapse. In balanced conflict, a more reliable and coherent coalition constitutes the more salient advocacy claim and is therefore the target of pragmatic skepticism. When one coalition is highly dispersed, it is a weak unified claim; the opposing homogeneous coalition becomes the salient overclaim, causing skepticism to favor the dispersed side. Arbitration into this skeptical interpretation increases continuously with bilateral formation, reliability contrast, and coherence asymmetry, while absolute and proportional tally decisiveness continuously suppress it. Thus trustworthy unilateral coalitions are followed rather than reversed, large majorities dominate, and anti-validity choice is concentrated in tied or weak-margin bilateral conflicts. Stable but constrained individual differences affect bilateral-skepticism propensity and choice precision.

**Rationale:** This is a one-parameter-range recalibration of the accepted iteration-8 candidate. The only change is increasing ordinary_strength from [0.12, 0.22] to [0.15, 0.25]. All equations, arbitration dynamics, salience normalization, coherence processing, bilateral formation, nonlinear tallying, skeptical reversal, precision, and subject-variation mechanisms remain unchanged. The slightly stronger default validity channel operates only in proportion to (1 - arbitration_weight), so it does not introduce the previously rejected residual conventional channel inside fully recruited advocacy states. It should modestly restore conventional validity influence in partially arbitrated weak conflicts, moving Experiments 1, 3, 5, 8, 9, and 11 toward their observed values and reducing the overly pure margin-sensitive signature in Experiment 2. It may also reduce the accepted candidate's excessive anti-validity rate in Experiment 21. Because the increase is small and the ordinary channel remains suppressed when advocacy attribution is strong, the established effects in Experiments 7, 10, 17, and 19 should be largely preserved. Unilateral Experiment 24 may increase slightly, but the narrow 0.03 range shift limits that risk. This edit directly follows the most recent feedback while retaining the only arbitration rebalancing that passed the accept gate.

**Parameters:**
  - `reliability_curvature`: `[1.45, 1.85]`
  - `coherence_floor`: `[0.24, 0.34]`
  - `coherence_scale`: `[0.055, 0.075]`
  - `claim_floor`: `[0.025, 0.045]`
  - `ordinary_claim_floor`: `[0.20, 0.30]`
  - `formation_scale`: `[0.82, 1.02]`
  - `formation_power`: `[1.45, 1.80]`
  - `salience_slope`: `[2.55, 3.15]`
  - `coherence_asymmetry_gain`: `[0.75, 1.05]`
  - `margin_scale`: `[3.55, 3.90]`
  - `margin_power`: `[3.0, 3.4]`
  - `proportion_floor`: `[0.42, 0.52]`
  - `proportion_power`: `[0.70, 0.95]`
  - `tally_strength`: `[1.85, 2.05]`
  - `decisiveness_floor`: `[0.60, 0.72]`
  - `resolution_power`: `[1.65, 2.10]`
  - `ordinary_strength`: `[0.15, 0.25]`
  - `advocacy_accumulation`: `[1.45, 1.85]`
  - `reversal_strength`: `[2.70, 3.15]`
  - `bilateral_skepticism`: `[0, 1]`
  - `skeptic_trait_spread`: `[0.12, 0.22]`
  - `choice_precision`: `[0.94, 1.08]`
  - `conflict_noise`: `[0.08, 0.22]`
  - `lapse_rate`: `[0.015, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_15` → slot 2 (via `new_theory`)

**Description:** Diversity-Legitimacy Attribution theory proposes that people infer both evidential strength and the social origin of expert agreement. Choice continuously arbitrates among a nonlinear raw tally, skeptical reversal of coordinated option-level claims, and a diversity-legitimacy signal in which an exceptional expert becomes credible only through broad, heterogeneous, weak-background corroboration. Homogeneous repetition is compressed, while a breadth-normalized residual coordination contrast preserves meaningful differences between coherent and dispersed coalitions. Coordination construal is recruited by bounded between-option conflict rather than by the absolute strength of either coalition, preventing shared coalition magnitude from crowding out counting and diversity. Pragmatic reversal additionally requires sufficient background breadth and is sharply concentrated around unresolved tally conflicts, so isolated or directionally resolved endorsements evoke less skepticism than established, balanced coalitions. Decisive tallies recruit counting strongly, while unresolved conflicts admit coordination and diversity construals. Stable attribution style governs sensitivity to coordination versus diversity; precision and lapse differences remain narrow.

**Rationale:** This is a single-parameter-range edit of the accepted iteration-8 model. The only change raises coordination_raw_residual from [0.35, 0.55] to [0.40, 0.60]. This modestly increases the contribution of the formation-normalized, unsaturated coordination contrast, which preserves discrimination between coherent and dispersed coalitions after the compressed sidewise claims approach saturation. It should strengthen skeptical reversal in exact-tally conflicts such as Experiments 10, 16, and 21 without globally increasing coordination recruitment, coordination strength, dispersion sensitivity, or breadth dependence. The accepted narrow conflict_margin_scale is retained, so the adjustment is strongly attenuated in nonzero-margin majority conflicts such as Experiments 4, 6, and 9. The shift is deliberately small because Experiment 8 is already somewhat too skeptical and Experiment 17's dispersion gradient is already accurate. All tally and diversity mechanisms remain unchanged, avoiding the previously rejected coupled recalibrations and postponing the distinct Experiment-25 arbitration problem to a later isolated edit.

**Parameters:**
  - `validity_curvature`: `[1.6, 2.6]`
  - `breadth_scale`: `[0.85, 1.25]`
  - `breadth_power`: `[0.80, 1.20]`
  - `coordination_dispersion_scale`: `[0.16, 0.25]`
  - `coordination_extremity`: `[1.15, 1.70]`
  - `coordination_authority_scale`: `[0.035, 0.070]`
  - `coordination_raw_residual`: `[0.40, 0.60]`
  - `diversity_dispersion_scale`: `[0.15, 0.24]`
  - `surprise_scale`: `[0.14, 0.24]`
  - `weak_anchor_scale`: `[0.28, 0.45]`
  - `diversity_breadth_power`: `[1.15, 1.75]`
  - `margin_scale`: `[2.2, 3.2]`
  - `margin_power`: `[1.25, 1.75]`
  - `proportion_power`: `[0.65, 1.05]`
  - `count_bias`: `[-0.10, 0.15]`
  - `count_gate_gain`: `[3.8, 5.2]`
  - `coordination_bias`: `[-0.15, 0.20]`
  - `diversity_bias`: `[-1.00, -0.55]`
  - `unresolved_gain`: `[0.75, 1.20]`
  - `coordination_salience`: `[1.35, 2.10]`
  - `diversity_salience`: `[6.0, 9.0]`
  - `attribution_style`: `[0, 1]`
  - `style_span`: `[0.35, 0.65]`
  - `arbitration_temperature`: `[0.72, 1.00]`
  - `count_strength`: `[1.75, 2.25]`
  - `count_exponent`: `[1.8, 2.4]`
  - `count_half_margin`: `[2.0, 2.8]`
  - `upper_count_compression`: `[0.08, 0.20]`
  - `coordination_strength`: `[4.6, 6.2]`
  - `coordination_slope`: `[2.4, 3.6]`
  - `bilateral_scale`: `[0.20, 0.34]`
  - `conflict_margin_scale`: `[1.25, 1.75]`
  - `reversal_breadth_scale`: `[1.20, 1.65]`
  - `reversal_breadth_power`: `[1.40, 2.10]`
  - `diversity_strength`: `[4.8, 6.4]`
  - `diversity_slope`: `[2.2, 3.4]`
  - `choice_precision`: `[0.92, 1.12]`
  - `precision_trait`: `[0, 1]`
  - `precision_span`: `[0.08, 0.20]`
  - `lapse_rate`: `[0.01, 0.055]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Diversity-Legitimacy Attribution model. With no outcome feedback, past
    # choices provide no basis for learning expert validity and are ignored.
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

    # Positive votes favor A and negative votes favor B. Equal ratings are
    # nondiagnostic and do not enter either option's inferred coalition.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smooth coding of communicated validity. No validity receives a special
    # categorical status, including the endpoints of the scale.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["validity_curvature"])
    endpoint = float(np.tanh(0.5 * curvature))
    authority = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)

    def summarize(side):
        members = authority[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return {
                "breadth": 0,
                "formation": 0.0,
                "coordination": 0.0,
                "legitimacy": 0.0,
                "dispersion": 0.0,
            }

        mean_strength = float(np.mean(members))
        dispersion = float(np.sqrt(np.mean((members - mean_strength) ** 2)))
        maximum_index = int(np.argmax(members))
        maximum = float(members[maximum_index])

        # Coalition formation saturates smoothly: repetition establishes that
        # there is a coalition but does not add independent evidence linearly.
        breadth_scale = float(parameters["breadth_scale"])
        breadth_power = float(parameters["breadth_power"])
        formation = 1.0 - np.exp(
            -(float(breadth) / breadth_scale) ** breadth_power
        )

        # Homogeneous, authoritative endorsers are compressed into a coherent
        # option-level claim. Dispersion weakens coordination attribution.
        coordination_homogeneity = np.exp(
            -(dispersion / float(parameters["coordination_dispersion_scale"])) ** 2
        )
        coordination = (
            formation
            * coordination_homogeneity
            * mean_strength ** float(parameters["coordination_extremity"])
        )

        # Upper-tail surprise is assessed relative to the remaining coalition,
        # not relative to a mean containing the exceptional expert itself.
        # The exceptional endorsement is legitimate only when a nontrivial,
        # weak background exists to provide heterogeneous corroboration.
        if breadth <= 1:
            background_breadth = 0
            background_mean = maximum
            weak_anchor = 0.0
        else:
            background = np.delete(members, maximum_index)
            background_breadth = int(background.size)
            background_mean = float(np.mean(background))
            weak_anchor = float(np.mean(np.exp(
                -background / float(parameters["weak_anchor_scale"])
            )))
        upper_surprise = max(maximum - background_mean, 0.0)

        # Diversity legitimacy is a genuine interaction. It is exactly weak
        # when any constituent is absent: background breadth, dispersion,
        # upper-tail surprise, exceptional authority, or weak-background
        # corroboration. Thus this is not a generic soft maximum.
        dispersion_gate = dispersion / (
            dispersion + float(parameters["diversity_dispersion_scale"])
        )
        surprise_gate = upper_surprise / (
            upper_surprise + float(parameters["surprise_scale"])
        )
        background_formation = 1.0 - np.exp(
            -(float(background_breadth) / breadth_scale) ** breadth_power
        ) if background_breadth > 0 else 0.0
        breadth_gate = background_formation ** float(
            parameters["diversity_breadth_power"]
        )
        legitimacy = (
            breadth_gate
            * dispersion_gate
            * surprise_gate
            * maximum
            * weak_anchor
        )

        return {
            "breadth": breadth,
            "formation": float(formation),
            "coordination": float(coordination),
            "legitimacy": float(legitimacy),
            "dispersion": dispersion,
        }

    side_a = summarize(1.0)
    side_b = summarize(-1.0)

    # Low-threshold compression gives weak but coherent coalitions pragmatic
    # salience. The residual contrast is divided by side formation so shared
    # coalition breadth cannot masquerade as a directional coordination cue.
    coordination_scale = float(parameters["coordination_authority_scale"])
    coordination_a = 1.0 - np.exp(
        -side_a["coordination"] / coordination_scale
    )
    coordination_b = 1.0 - np.exp(
        -side_b["coordination"] / coordination_scale
    )
    normalized_raw_a = side_a["coordination"] / max(
        side_a["formation"], 1e-12
    ) if side_a["breadth"] > 0 else 0.0
    normalized_raw_b = side_b["coordination"] / max(
        side_b["formation"], 1e-12
    ) if side_b["breadth"] > 0 else 0.0
    raw_coordination_contrast = normalized_raw_a - normalized_raw_b
    coordination_raw_residual = float(
        parameters["coordination_raw_residual"]
    )
    coordination_contrast = (
        (1.0 - coordination_raw_residual) * (coordination_a - coordination_b)
        + coordination_raw_residual * raw_coordination_contrast
    )
    legitimacy_contrast = side_a["legitimacy"] - side_b["legitimacy"]

    # Decisiveness combines absolute margin and proportional margin. A sparse
    # one-vote advantage is therefore not equated with a broad majority.
    if active_count == 0:
        proportional_margin = 0.0
        decisiveness = 0.0
    else:
        proportional_margin = margin_magnitude / float(active_count)
        absolute_certainty = 1.0 - np.exp(
            -(margin_magnitude / float(parameters["margin_scale"]))
            ** float(parameters["margin_power"])
        )
        decisiveness = absolute_certainty * (
            proportional_margin ** float(parameters["proportion_power"])
        )
    decisiveness = float(np.clip(decisiveness, 0.0, 1.0))
    unresolved = 1.0 - decisiveness

    # Stable attribution style reallocates interpretive attention between
    # coordination skepticism and diversity legitimacy without altering the
    # communicated validities or creating endpoint-specific rules.
    attribution_style = float(parameters["attribution_style"]) - 0.5
    style_shift = float(parameters["style_span"]) * attribution_style

    # Coordination recruitment depends on bounded directional conflict and
    # bilateral formation, not on the strongest coalition's absolute raw
    # magnitude. Shared breadth therefore remains in conflict establishment
    # without crowding count or diversity out of the arbitration.
    recruitment_bilateral = np.sqrt(max(
        side_a["formation"] * side_b["formation"], 0.0
    ))
    bounded_coordination_conflict = abs(np.tanh(
        float(parameters["coordination_slope"]) * coordination_contrast
    ))
    strongest_legitimacy = max(side_a["legitimacy"], side_b["legitimacy"])

    # Smooth arbitration utilities. The raw tally receives increasing priority
    # with decisiveness; attributional construals receive priority in unresolved
    # conflicts according to their stimulus-specific support.
    count_utility = (
        float(parameters["count_bias"])
        + float(parameters["count_gate_gain"]) * decisiveness
    )
    coordination_utility = (
        float(parameters["coordination_bias"])
        + float(parameters["unresolved_gain"]) * unresolved
        + float(parameters["coordination_salience"])
        * bounded_coordination_conflict
        * (0.35 + 0.65 * recruitment_bilateral)
        + style_shift
    )
    diversity_utility = (
        float(parameters["diversity_bias"])
        + float(parameters["unresolved_gain"]) * unresolved
        + float(parameters["diversity_salience"]) * strongest_legitimacy
        - style_shift
    )

    utilities = np.asarray(
        [count_utility, coordination_utility, diversity_utility],
        dtype=np.float64,
    )
    utilities /= float(parameters["arbitration_temperature"])
    utilities -= np.max(utilities)
    arbitration = np.exp(utilities)
    arbitration /= np.sum(arbitration)

    # Default tally evidence follows a Hill curve: one-vote margins remain
    # weak, evidence rises rapidly at intermediate margins, and large margins
    # plateau instead of becoming increasingly diagnostic of margin counting.
    if margin_magnitude < 1e-12:
        count_evidence = 0.0
    else:
        count_power = float(parameters["count_exponent"])
        count_half_margin = float(parameters["count_half_margin"])
        margin_term = margin_magnitude ** count_power
        count_curve = margin_term / (
            count_half_margin ** count_power + margin_term
        )
        # A mild upper-range compression reduces distinctions among already
        # decisive margins without changing the weak-margin Hill geometry.
        count_curve /= 1.0 + float(
            parameters["upper_count_compression"]
        ) * count_curve
        count_evidence = (
            float(parameters["count_strength"])
            * np.sign(tally_margin)
            * count_curve
            * (0.5 + 0.5 * proportional_margin)
        )

    # Pragmatic reversal requires two represented sides. It is sharply
    # localized around unresolved tally conflict, preventing coordination
    # skepticism from persisting once even a modest raw majority resolves the
    # bilateral dispute.
    bilateral_scale = float(parameters["bilateral_scale"])
    bilateral = np.sqrt(
        max(side_a["formation"] * side_b["formation"], 0.0)
    )
    bilateral = bilateral / (bilateral + bilateral_scale)
    conflict_focus = np.exp(
        -(margin_magnitude / float(parameters["conflict_margin_scale"])) ** 2
    )
    minimum_breadth = float(min(side_a["breadth"], side_b["breadth"]))
    reversal_breadth = 1.0 - np.exp(
        -(minimum_breadth / float(parameters["reversal_breadth_scale"]))
        ** float(parameters["reversal_breadth_power"])
    )
    coordination_evidence = -float(parameters["coordination_strength"]) * (
        bilateral
        * reversal_breadth
        * conflict_focus
        * np.tanh(
            float(parameters["coordination_slope"])
            * coordination_contrast
        )
    )

    # Legitimate diversity supports, rather than reverses, the coalition with
    # broad heterogeneous corroboration and a surprising upper-tail expert.
    diversity_evidence = float(parameters["diversity_strength"]) * (
        unresolved
        * np.tanh(
            float(parameters["diversity_slope"])
            * legitimacy_contrast
        )
    )

    precision_trait = float(parameters["precision_trait"]) - 0.5
    precision = float(parameters["choice_precision"]) * (
        1.0 + float(parameters["precision_span"]) * precision_trait
    )
    precision = max(precision, 0.10)

    # Each construal supplies a complete conditional choice distribution. We
    # marginalize over construal recruitment instead of averaging incompatible
    # evidence representations into a single latent score.
    evidences = precision * np.asarray(
        [count_evidence, coordination_evidence, diversity_evidence],
        dtype=np.float64,
    )
    channel_p_a = np.empty(3, dtype=np.float64)
    for i, evidence in enumerate(evidences):
        logits = np.asarray([0.5 * evidence, -0.5 * evidence], dtype=np.float64)
        logits -= np.max(logits)
        channel_probs = np.exp(logits)
        channel_probs /= np.sum(channel_probs)
        channel_p_a[i] = channel_probs[0]

    p_a = float(np.dot(arbitration, channel_p_a))
    probabilities = np.asarray([p_a, 1.0 - p_a], dtype=np.float64)

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.sum(probabilities)
    return probabilities
```

**`policy(probs)`:**
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
