# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
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



## ADVOCATED THEORY
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


## COMPETING THEORY
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Estimate the linear change in preference for the 70%-expert coalition as\n    # its three fillers are progressively upgraded from 50% to 60% validity.\n    records = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 7 or b.size < 7:\n            continue\n\n        # Feature 6 is the fixed 70%-valid expert and uniquely identifies the\n        # focal coalition. Its display side is reversed in half the trials.\n        if a[6] > b[6]:\n            focal_is_a = True\n            focal = a\n        elif b[6] > a[6]:\n            focal_is_a = False\n            focal = b\n        else:\n            continue\n\n        # Features 3--5 are the three possible 60%-valid fillers. Their sum is\n        # the designed progression level k in {0, 1, 2, 3}.\n        k = int(np.rint(np.sum(focal[3:6])))\n        if k < 0 or k > 3:\n            continue\n\n        response = int(row[\"response\"])\n        chose_focal = float((response == 0) if focal_is_a else (response == 1))\n        records.append((focal_is_a, k, chose_focal))\n\n    if len(records) == 0:\n        return 0.0\n\n    # Compute the trend separately for the two display orientations, then\n    # average. This cancels stable A/B response biases while giving all four\n    # experimental conditions equal weight.\n    orientation_slopes = []\n    for focal_is_a in (False, True):\n        condition_means = []\n        levels = []\n        for k in range(4):\n            vals = [y for side, level, y in records\n                    if side == focal_is_a and level == k]\n            if len(vals) > 0:\n                levels.append(float(k))\n                condition_means.append(float(np.mean(vals)))\n\n        if len(levels) >= 2:\n            x = np.asarray(levels, dtype=float)\n            y = np.asarray(condition_means, dtype=float)\n            xc = x - np.mean(x)\n            denom = float(np.sum(xc ** 2))\n            if denom > 0.0:\n                orientation_slopes.append(float(np.sum(xc * y) / denom))\n\n    if len(orientation_slopes) == 0:\n        return 0.0\n\n    # Multiplication by three expresses the fitted trend as the predicted\n    # preference change from k=0 to k=3.\n    return float(3.0 * np.mean(orientation_slopes))",
  "rationale": "This is the side-balanced linear trend in choosing the focal coalition containing the 70%-valid expert as its three fillers are upgraded from 50% to 60%. Conflict-Geometry Recruitment predicts a sizable positive trend because exemplar recruitment initially favors the homogeneous 85% coalition, whereas reduced heterogeneity later suppresses that channel and allows skeptical claim reversal to favor the focal coalition. Bilateral Advocacy-Attribution predicts the opposite trend: strengthening and increasing the coherence of the focal coalition reduces its skeptical advantage. Estimating the trend from all four levels and 24 observations per level improves precision, while computing it separately for the two exact option reversals removes stable A-versus-B response bias and subject-level differences in overall focal-coalition preference."
}
```

## Usage

```json
{
  "input_tokens": 9194,
  "output_tokens": 2615
}
```
