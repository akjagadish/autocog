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
**Validities (n_features=8):** [0.9448, 0.84, 0.76, 0.68, 0.6, 0.84, 0.9448, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]

**Rationale:** This design targets CRGRC's continuous near-tie serial-accessibility gate under a validity gap that is just outside S3C's categorical tied-order regime. Expert 8 is the unique maximum at 0.9500. Experts 1 and 7 are validity-matched at 0.9448, only 0.0052 below the maximum: this exceeds S3C's 0.005 near-tie tolerance (including NumPy's negligible relative tolerance), so S3C treats Expert 8 as a clear unique maximum and sets serial accessibility exactly to zero. Each matched contrast swaps the directions of Experts 1 and 7 while keeping both active. Because they have identical validity, this swap preserves coalition counts, each coalition's validity multiset, redundancy, credibility, restoration, count imbalance, and all other S3C calculations. S3C therefore predicts exactly zero difference between conditions. CRGRC instead has a continuous Gaussian top-ambiguity gate; a 0.0052 gap remains appreciably ambiguous throughout its tie-width range. Expert 1 is the first active cue and Expert 7 is near the end of the active sequence, so CRGRC's exponentially declining serial accessibility gives Expert 1 much greater leverage. It therefore predicts more choices of Expert 8's coded option when Expert 1 supports that option and Expert 7 opposes it than when their directions are exchanged. Importantly, CRGRC's coalition support, challenge, and restoration terms are also exactly matched across each swap; only serial evidence changes. Three six-active-cue families vary the lower-validity coalition composition while preserving exact 3-versus-3 balance, which strengthens CRGRC's serial gate and tests generality. Exact A/B reversals control response-side preference. The critical statistic is P(coded Expert-8 option | Expert 1 supports it) minus P(coded Expert-8 option | Expert 7 supports it): CRGRC predicts a positive contrast, whereas S3C predicts exactly zero for every parameter setting. The 12 unique pairs are repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Expert-1 versus Expert-7 serial-leverage contrast, coded toward Expert 8.\"\"\"\n    if len(data) == 0:\n        return 0.0\n\n    a = np.vstack(data[\"option_a_ratings\"].apply(\n        lambda x: np.asarray(x, dtype=float)\n    ).to_numpy())\n    b = np.vstack(data[\"option_b_ratings\"].apply(\n        lambda x: np.asarray(x, dtype=float)\n    ).to_numpy())\n    response = data[\"response\"].to_numpy(dtype=int)\n\n    directions = np.sign(b - a)\n    expert8_direction = directions[:, 7]\n\n    # +1: Expert 1 supports Expert 8 and Expert 7 opposes it.\n    # -1: Expert 7 supports Expert 8 and Expert 1 opposes it.\n    expert1_alignment = directions[:, 0] * expert8_direction\n    expert7_alignment = directions[:, 6] * expert8_direction\n    condition = np.zeros(len(data), dtype=float)\n    condition[(expert1_alignment == 1) & (expert7_alignment == -1)] = 1.0\n    condition[(expert1_alignment == -1) & (expert7_alignment == 1)] = -1.0\n\n    critical = (condition != 0) & (expert8_direction != 0)\n    if not np.any(critical):\n        return 0.0\n\n    # Convert each response to +1 when it follows Expert 8 and -1 otherwise,\n    # eliminating A/B response-side orientation from the contrast.\n    chose_expert8 = np.where(\n        expert8_direction > 0,\n        response == 1,\n        response == 0\n    )\n    coded_choice = 2.0 * chose_expert8.astype(float) - 1.0\n\n    # With the design's balanced conditions, this equals\n    # P(follow Expert 8 | Expert 1 aligned) minus\n    # P(follow Expert 8 | Expert 7 aligned).\n    return float(np.mean(condition[critical] * coded_choice[critical]))\n",
  "rationale": "This is the design's targeted, response-side-corrected serial-leverage interaction. It uses every repetition and both A/B orientations, yielding a low-noise within-subject contrast. Positive values mean subjects follow Expert 8 more when early Expert 1 agrees with it than when later Expert 7 agrees with it. CRGRC predicts a sizable positive value because its continuous top-ambiguity gate remains active at the 0.0052 validity gap and gives the early cue substantially greater serial accessibility. S3C disables serial accessibility outside its near-tie tolerance; exchanging equal-validity Experts 1 and 7 then preserves its coalition computations, so it predicts a contrast near zero. Coding choices relative to Expert 8 and averaging the exact swap conditions cancels generic A/B bias and stable overall willingness to follow the strongest expert."
}
```

## Usage

```json
{
  "input_tokens": 9849,
  "output_tokens": 2704
}
```
