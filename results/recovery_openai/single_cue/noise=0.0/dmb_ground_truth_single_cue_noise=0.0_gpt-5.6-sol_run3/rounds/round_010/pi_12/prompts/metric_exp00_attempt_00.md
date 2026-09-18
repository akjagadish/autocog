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
**Validities (n_features=18):** [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** This 16-pair design targets the competing theory's narrow-conflict focus rule while holding tally margin, coalition coherence, and within-condition mean reliability contrast fixed. In the first four base pairs, A is endorsed by m homogeneous 55%-valid experts and B by m homogeneous 50%-valid experts, for m = 1, 2, 4, or 6. The next four use homogeneous 100%-valid A endorsers against the same number of 50%-valid B endorsers. Every trial is an exact m-versus-m tally tie, and all coalitions have zero dispersion. Under Uncertainty-Arbitrated Overclaim Construal, homogeneous repetition does not amplify either coalition's claim: its reliability contrast is exactly invariant across m. Sampling-based metacognitive trust saturates rapidly, so especially beyond the smallest coalition it predicts an approximately flat breadth profile. The 55%-versus-50% contrast is also above the narrow moderate-tie band for most of the advocated parameter range, limiting contamination by that separate mechanism. Coherence-Gated Relative Coalition Comparison likewise has a breadth-normalized contextual contrast, but its explicit narrow-conflict focus selectively attenuates ordinary singleton contrasts. Consequently, it predicts substantially fewer choices of B, the skeptical response to A's stronger claim, at m=1 than at m=4 or m=6 in the 55%-versus-50% series. This breadth effect should be much smaller in the 100%-versus-50% series because its contextual contrast is near maximal and therefore escapes singleton attenuation. The critical quantitative signature is thus a selective ordinary-contrast breadth increase under the competing theory versus near breadth invariance under the advocated theory; the endpoint series serves as a within-subject control showing that any effect is not generic repetition sensitivity. The final eight pairs are exact A/B reversals, balancing response side. Each unique pair is repeated six times for 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Score whether each response is skeptical: choosing the option opposite\n    # the coalition containing the informative (> .50-valid) experts.\n    ordinary_scores = []\n    endpoint_scores = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        response = int(row[\"response\"])\n        signed = a - b\n\n        ordinary_direction = float(np.sum(signed[:6]))   # 55%-valid experts\n        endpoint_direction = float(np.sum(signed[6:12])) # 100%-valid experts\n\n        if abs(ordinary_direction) > 1e-12:\n            # Positive means the stronger coalition supports A, so choosing B\n            # is the skeptical response. Negative reverses that coding.\n            skeptical = response if ordinary_direction > 0 else 1 - response\n            ordinary_scores.append(float(skeptical))\n        elif abs(endpoint_direction) > 1e-12:\n            skeptical = response if endpoint_direction > 0 else 1 - response\n            endpoint_scores.append(float(skeptical))\n\n    if len(ordinary_scores) == 0 or len(endpoint_scores) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(ordinary_scores) - np.mean(endpoint_scores))",
  "rationale": "The metric is the within-subject difference between skeptical-choice rates for 55%-versus-50% conflicts and 100%-versus-50% conflicts, after folding the A/B reversals into a common response direction. It uses all 96 trials, giving 48 observations to each rate and therefore substantially better stability than a contrast based only on singleton versus broad conditions. Uncertainty-Arbitrated Overclaim Construal strongly engages skeptical processing for the ordinary homogeneous conflicts, while its trusted-extreme precision limiter selectively pulls endpoint-sized bilateral conflicts toward chance; it therefore predicts a comparatively positive ordinary-minus-endpoint score. Coherence-Gated Relative Coalition Comparison instead attenuates ordinary conflicts\u2014especially narrow ones\u2014while preserving strong skeptical responding to near-maximal 100%-versus-50% contrasts, predicting a markedly more negative score. This is a behavioral interaction between communicated reliability regime and skeptical choice, rather than a definitional restatement of either theory."
}
```

## Usage

```json
{
  "input_tokens": 10246,
  "output_tokens": 3117
}
```
