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
**Validities (n_features=12):** [0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** This eight-pair design isolates the advocated theory's requirement that an exceptional expert receive broad, weak-background corroboration. In each base pair, A is endorsed by one 70%-valid expert plus m-1 neutral 50%-valid experts, while B is endorsed by m homogeneous 95%-valid experts, for m in {1,2,4,6}. Every condition is an exact m-versus-m tally tie. Under Diversity-Legitimacy Attribution, B is the strongly coordinated homogeneous coalition and is therefore skeptically reversed in favor of A. As m increases, A's 70% expert also acquires increasingly broad, heterogeneous, weak-background corroboration, activating the diversity-legitimacy channel; both attributional channels consequently favor A, predicting a maintained or increasing A-choice profile with breadth. Conflict-Geometry Recruitment predicts a contrasting transition. Its coalition-claim channel also skeptically favors A because B has the stronger mean claim, but adding 50%-valid members creates a large discrepancy between A's low coalition mean and its retained upper-tail exemplar. This diagnostic geometry recruits the exemplar channel, whose sidewise exemplar favors B because B's 95%-valid experts exceed A's 70%-valid upper tail. It therefore predicts a sharp reduction in A choices after weak fillers are introduced and, especially in the broader conditions, substantially more B choice than the advocated theory. The critical quantitative contrast is thus a positive or nonnegative A-choice breadth slope under Diversity-Legitimacy Attribution versus an early downward shift toward B under Conflict-Geometry Recruitment. Exact A/B reversals balance response side. Eight unique pairs are each repeated 12 times for 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Diversity-Legitimacy Attribution theory proposes that people infer both evidential strength and the social origin of expert agreement. Choice continuously arbitrates among a nonlinear raw tally, skeptical reversal of coordinated option-level claims, and a diversity-legitimacy signal in which an exceptional expert becomes credible only through broad, heterogeneous, weak-background corroboration. Homogeneous repetition is compressed, while a breadth-normalized residual coordination contrast preserves meaningful differences between coherent and dispersed coalitions. Coordination construal is recruited by bounded between-option conflict rather than by the absolute strength of either coalition, preventing shared coalition magnitude from crowding out counting and diversity. Pragmatic reversal additionally requires sufficient background breadth and is sharply concentrated around unresolved tally conflicts, so isolated or directionally resolved endorsements evoke less skepticism than established, balanced coalitions. Decisive tallies recruit counting strongly, while unresolved conflicts admit coordination and diversity construals. Stable attribution style governs sensitivity to coordination versus diversity; precision and lapse differences remain narrow.

**Parameters:**
- validity_curvature: [1.6, 2.6]
- breadth_scale: [0.85, 1.25]
- breadth_power: [0.80, 1.20]
- coordination_dispersion_scale: [0.16, 0.25]
- coordination_extremity: [1.15, 1.70]
- coordination_authority_scale: [0.035, 0.070]
- coordination_raw_residual: [0.40, 0.60]
- diversity_dispersion_scale: [0.15, 0.24]
- surprise_scale: [0.14, 0.24]
- weak_anchor_scale: [0.28, 0.45]
- diversity_breadth_power: [1.15, 1.75]
- margin_scale: [2.2, 3.2]
- margin_power: [1.25, 1.75]
- proportion_power: [0.65, 1.05]
- count_bias: [-0.10, 0.15]
- count_gate_gain: [3.8, 5.2]
- coordination_bias: [-0.15, 0.20]
- diversity_bias: [-1.00, -0.55]
- unresolved_gain: [0.75, 1.20]
- coordination_salience: [1.35, 2.10]
- diversity_salience: [6.0, 9.0]
- attribution_style: [0, 1]
- style_span: [0.35, 0.65]
- arbitration_temperature: [0.72, 1.00]
- count_strength: [1.75, 2.25]
- count_exponent: [1.8, 2.4]
- count_half_margin: [2.0, 2.8]
- upper_count_compression: [0.08, 0.20]
- coordination_strength: [4.6, 6.2]
- coordination_slope: [2.4, 3.6]
- bilateral_scale: [0.20, 0.34]
- conflict_margin_scale: [1.25, 1.75]
- reversal_breadth_scale: [1.20, 1.65]
- reversal_breadth_power: [1.40, 2.10]
- diversity_strength: [4.8, 6.4]
- diversity_slope: [2.2, 3.4]
- choice_precision: [0.92, 1.12]
- precision_trait: [0, 1]
- precision_span: [0.08, 0.20]
- lapse_rate: [0.01, 0.055]
- validities: validities

**`predict source code`:**
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    weak_choices_narrow = []\n    weak_choices_broad = []\n\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=np.float64)\n        b = np.asarray(row.option_b_ratings, dtype=np.float64)\n        response = int(row.response)\n\n        # The 70%-valid expert is feature 0. Its endorsed option is the\n        # heterogeneous/weak-background coalition in every designed pair.\n        weak_side_is_a = bool(a[0] > b[0])\n        chose_weak_side = float(\n            (response == 0 and weak_side_is_a)\n            or (response == 1 and not weak_side_is_a)\n        )\n\n        # Each designed option has the same number m of discriminating votes,\n        # so total discriminating features equals 2m.\n        m = int(round(np.sum(np.abs(a - b)) / 2.0))\n        if m == 1:\n            weak_choices_narrow.append(chose_weak_side)\n        elif m >= 4:\n            weak_choices_broad.append(chose_weak_side)\n\n    if len(weak_choices_narrow) == 0 or len(weak_choices_broad) == 0:\n        return float('nan')\n\n    return float(\n        np.mean(weak_choices_broad) - np.mean(weak_choices_narrow)\n    )",
  "rationale": "This is the within-subject broad-corroboration shift: the probability of choosing the coalition containing the 70%-valid expert in the broad m=4 and m=6 conditions minus the corresponding probability in the singleton m=1 condition. It recodes choices by coalition identity rather than screen side, pooling exact A/B reversals and thereby suppressing response-side bias. Diversity-Legitimacy Attribution predicts that adding a broad weak background activates legitimacy while strengthening pragmatic reversal, yielding a nonnegative or positive shift. Conflict-Geometry Recruitment predicts that the same fillers create claim\u2013exemplar discrepancy and recruit an exemplar that favors the homogeneous 95% coalition, yielding a negative shift. Pooling 48 broad and 24 singleton trials per subject gives a relatively stable estimate while focusing on the conditions where the theories' recruitment predictions diverge most."
}
```

## Usage

```json
{
  "input_tokens": 10078,
  "output_tokens": 1949
}
```
