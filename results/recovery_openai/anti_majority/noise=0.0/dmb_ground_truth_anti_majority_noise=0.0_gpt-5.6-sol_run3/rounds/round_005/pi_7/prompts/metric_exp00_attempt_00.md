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
**Validities (n_features=8):** [0.735, 0.73, 0.725, 0.72, 0.715, 0.71, 0.52, 0.74]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]

**Rationale:** This design targets PDRR's sharp redundancy-restoration threshold for an isolated validity anchor. Expert 8 is the unique, weak-absolute anchor (74%) and favors the designated anchor option; between one and six earlier experts favor the other option. The opponents are individually close to the anchor in validity, maximizing PDRR's compact-challenge mechanism while keeping the validity and credibility gaps below its selective-restoration thresholds. PDRR therefore predicts low anchor-choice probability with one or two opponents, followed by a large discontinuous reversal at exactly three opponents: its count-dominant restoration gate is nearly off at two but approximately 0.98 or higher at three throughout the stated restoration-slope range. It should continue to protect the anchor against four to six opponents. The two identity variants at counts two and three permit the threshold contrast to be estimated without relying on one particular cue subset. ADCG has no corresponding two-to-three restoration branch. Because the validity anchor is presented last and the earliest active cue belongs to the opposing majority, ADCG's minority-first profile gate is closed. Its coalition core instead gives the earlier opposition distinctiveness and dissent bonuses; accumulated anti-concurrence with the anchor and rarity weighting should, if anything, strengthen that opposition over experience. ADCG thus predicts no abrupt upward jump in anchor choices at three opponents and generally continued preference for the opposing coalition. The critical statistic is P(anchor option | three opponents) minus P(anchor option | two opponents): PDRR predicts a large positive discontinuity, whereas ADCG predicts a much smaller, absent, or negative change. Counts one and four through six establish the broader step-function signature rather than a smooth coalition-size trend. Exact A/B reversals control response-side preference. Sixteen unique pairs repeated six times yield 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Partitioned Dissent and Redundancy Restoration (PDRR) proposes that people construct a stable, instruction-based evidence representation and classify each choice configuration before integrating its cues. Communicated validities determine both relative cue strength and absolute anchor credibility, while cues supporting the same option accumulate sublinearly because concordant reports are treated as partly redundant. In ordinary embedded coalitions, individual cue weights are strongly compressed before accumulation, limiting the influence of any single non-anchor cue while preserving communicated-validity ordering. Opposition to a validity anchor is statically distinctive even when the anchor has allies, but balanced opposition receives a counter-anchor bonus only when the opposing cues are individually close in communicated validity to the anchor and the coalition is broad enough to support a coalition-level interpretation. Isolated-anchor opposition is processed separately. A single opponent receives especially strong challenge status, with stable subject types differing substantially in their sensitivity to this challenge, while exactly two opponents retain a joint-credibility mechanism with greater type-specific heterogeneity. These challenge and singleton-dissent mechanisms are additionally amplified when the locally strongest active cue has low absolute communicated validity, because being best among weak active cues does not create a compelling anchor. Coalitions of three or more trigger a sharp, nearly type-invariant redundancy-restoration branch. Embedded singleton dissent is strongly amplified only against very broad coalitions or weak absolute anchors; its generic effect against smaller coalitions is attenuated to avoid excessive profile variance. Presentation order affects choice primarily as a bounded tie breaker among currently discriminating cues with equal maximal communicated validity, avoiding a general primacy crossover. All mechanisms are history-invariant in this feedback-free task. Stable subject types and random effects over validity sensitivity, balanced dissent, compact challenge, redundancy, temperature, lapse, and absolute-anchor sensitivity generate persistent individual differences.

**Parameters:**
- validities: validities
- validity_sensitivity: [0.15, 0.85]
- redundancy_exponent: [0.28, 0.68]
- redundancy_saturation: [0.15, 0.65]
- ordinary_validity_compression: [0.03, 0.18]
- minority_amplification: [0.20, 0.70]
- balanced_opposition_bonus: [0.55, 1.25]
- singleton_bonus: [1.8, 4.5]
- compact_challenge: [2.4, 6.0]
- compact_width: [0.75, 1.15]
- credibility_gap_scale: [0.018, 0.055]
- absolute_anchor_threshold: [0.68, 0.78]
- absolute_anchor_slope: [20.0, 45.0]
- weak_anchor_gain: [1.2, 3.0]
- restoration_threshold: [1.65, 2.15]
- restoration_slope: [9.0, 18.0]
- restoration_gap_threshold: [0.035, 0.085]
- restoration_gap_slope: [70.0, 150.0]
- restoration_credibility_threshold: [0.025, 0.065]
- restoration_credibility_slope: [70.0, 150.0]
- restoration_ceiling: [0.82, 0.98]
- tie_order_gate: [0.48, 0.78]
- beta: [1.2, 4.2]
- lapse: [0.0, 0.14]
- configuration_type: {0, 1, 2}

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PDRR expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    redundancy_exponent = float(parameters["redundancy_exponent"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    ordinary_validity_compression = float(parameters["ordinary_validity_compression"])
    minority_amplification = float(parameters["minority_amplification"])
    balanced_opposition_bonus = float(parameters["balanced_opposition_bonus"])
    singleton_bonus = float(parameters["singleton_bonus"])
    compact_challenge = float(parameters["compact_challenge"])
    compact_width = float(parameters["compact_width"])
    credibility_gap_scale = float(parameters["credibility_gap_scale"])
    absolute_anchor_threshold = float(parameters["absolute_anchor_threshold"])
    absolute_anchor_slope = float(parameters["absolute_anchor_slope"])
    weak_anchor_gain = float(parameters["weak_anchor_gain"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap_threshold = float(parameters["restoration_gap_threshold"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    restoration_credibility_threshold = float(parameters["restoration_credibility_threshold"])
    restoration_credibility_slope = float(parameters["restoration_credibility_slope"])
    restoration_ceiling = float(parameters["restoration_ceiling"])
    tie_order_gate = float(parameters["tie_order_gate"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated, stable subject types alter configuration sensitivity without
    # introducing trial-wise strategy switching or exposure-based learning.
    if configuration_type == 0:       # compact-challenge-sensitive
        challenge_type_gain = 2.40
        one_challenge_type_gain = 4.20
        singleton_type_gain = 1.50
        restoration_type_gain = 0.95
    elif configuration_type == 1:     # count-restoring
        challenge_type_gain = 0.50
        one_challenge_type_gain = 0.75
        singleton_type_gain = 0.90
        restoration_type_gain = 1.25
    else:                              # weakly configuration-sensitive
        challenge_type_gain = 0.15
        one_challenge_type_gain = 0.15
        singleton_type_gain = 0.55
        restoration_type_gain = 1.00

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Translate communicated accuracy into reliability evidence, retaining
    # absolute ordering but using the strongest active cue as a scale origin.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(
        validity_sensitivity * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices, compress=False):
        n = int(indices.size)
        if n == 0:
            return 0.0
        weights = cue_weight[indices]
        if compress:
            # Bound cue-composition effects in ordinary embedded coalitions.
            # The power transform preserves validity ordering but prevents a
            # single non-anchor cue from creating a broad primacy crossover.
            weights = np.power(
                np.clip(weights, 1e-12, None), ordinary_validity_compression
            )
        raw = float(np.sum(weights))
        # The power term gives ordinary sublinear accumulation. The bounded
        # saturation term makes additional concordant reports increasingly
        # interpretable as redundant without erasing coalition size entirely.
        discount = (float(n) ** redundancy_exponent) * (
            1.0 + redundancy_saturation * float(max(n - 1, 0))
            / float(n + 1)
        )
        return raw / max(discount, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # Highest communicated validity among discriminating cues defines the
    # anchor. Stable display order is used only to resolve a top-validity tie.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]
    anchor_cue = int(np.min(best_candidates))
    anchor_direction = float(directions[anchor_cue])
    unique_anchor = float(best_candidates.size == 1)

    # Relative weights alone make every locally strongest cue equivalent.
    # This bounded gate preserves that representation while marking anchors
    # that are weak in absolute communicated-validity terms as challengeable.
    weak_absolute_anchor = sigmoid(
        absolute_anchor_slope * (absolute_anchor_threshold - best_v)
    )
    weak_anchor_multiplier = 1.0 + weak_anchor_gain * weak_absolute_anchor

    if active.size > 1:
        sorted_active_v = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active_v[0] - sorted_active_v[1])
    else:
        validity_gap = 0.5

    if anchor_direction > 0:
        anchor_side_count = n_b
        opposition_count = n_a
        anchor_support = support_b
        opposition_support = support_a
        opposition_cues = a_cues
    else:
        anchor_side_count = n_a
        opposition_count = n_b
        anchor_support = support_a
        opposition_support = support_b
        opposition_cues = b_cues

    isolated_anchor = anchor_side_count == 1 and opposition_count > 0

    # Credibility is based on the strongest one or two individual opponents.
    # For two-opponent challenge, the weaker member must also be credible, so
    # one strong report cannot make a heterogeneous pair diagnostic.
    if opposition_count > 0:
        opponent_validities = np.sort(validities[opposition_cues])[::-1]
        credibility_set = opponent_validities[:min(2, opposition_count)]
        opponent_credibility = float(np.mean(credibility_set))
        credibility_gap = max(best_v - opponent_credibility, 0.0)
        joint_credibility_gap = max(
            best_v - float(np.min(credibility_set)), 0.0
        )
    else:
        credibility_gap = 0.5
        joint_credibility_gap = 0.5

    if isolated_anchor:
        # Challenge is sharply asymmetric across counts. The one-opponent
        # branch has a less suppressive credibility gate and substantially
        # separated stable subject types. Weak absolute anchors amplify the
        # existing challenge rather than creating a general coalition bonus.
        if opposition_count == 1:
            compact_profile = 3.00
            challenge_gap = 0.35 * credibility_gap
            active_challenge_type_gain = one_challenge_type_gain
        elif opposition_count == 2:
            compact_profile = 1.35
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        else:
            distance = (float(opposition_count) - 1.5) / max(compact_width, 1e-6)
            compact_profile = np.exp(-(distance ** 4))
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        credibility_gate = np.exp(
            -challenge_gap / max(credibility_gap_scale, 1e-6)
        )
        challenged_opposition = opposition_support * (
            1.0
            + active_challenge_type_gain
            * compact_challenge
            * compact_profile
            * credibility_gate
            * weak_anchor_multiplier
        )
        margin = (anchor_support - challenged_opposition) / max(
            anchor_support + challenged_opposition, 1e-12
        )
        core_evidence = anchor_direction * margin
    else:
        # Compress cue-composition effects only for unique-anchor ordinary
        # coalitions. Tied-top profiles retain the accepted integrator so the
        # calibrated bounded order effect is not disturbed.
        if unique_anchor > 0.0:
            support_a = coalition_support(a_cues, compress=True)
            support_b = coalition_support(b_cues, compress=True)
            if anchor_direction > 0:
                anchor_support = support_b
                opposition_support = support_a
            else:
                anchor_support = support_a
                opposition_support = support_b

        # Balanced opposition is distinctive only when the strongest opposing
        # cue is close to the anchor in communicated validity. A bounded breadth
        # factor further prevents a compact 2-vs-2 profile from receiving the
        # same categorical boost as a broader coalition-level conflict.
        if (
            unique_anchor > 0.0
            and anchor_side_count == opposition_count
            and anchor_side_count > 1
        ):
            strongest_opponent_v = float(np.max(validities[opposition_cues]))
            balanced_validity_gap = max(best_v - strongest_opponent_v, 0.0)
            geometry_gate = np.exp(
                -balanced_validity_gap / max(credibility_gap_scale, 1e-6)
            )
            breadth_gate = min(
                1.0, float(anchor_side_count - 1) / 2.0
            )
            balanced_gain = (
                balanced_opposition_bonus * geometry_gate * breadth_gate
            )
            opposition_support *= 1.0 + balanced_gain
            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        # Embedded distinctiveness is sharply singleton-localized. Its broad-
        # coalition component remains strong, while the generic component is
        # attenuated for smaller coalitions to reduce excessive profile variance.
        # Absolute weakness amplifies this existing channel without altering
        # balanced or other ordinary embedded configurations.
        if opposition_count == 1 and anchor_side_count > 1:
            imbalance = (
                float(anchor_side_count - opposition_count)
                / float(anchor_side_count + opposition_count)
            )
            dissent_multiplier = 1.0 + (
                minority_amplification * imbalance * weak_anchor_multiplier
            )
            breadth_weight = 1.0 if anchor_side_count >= 5 else 0.20
            dissent_multiplier += (
                singleton_type_gain
                * singleton_bonus
                * breadth_weight
                * np.log1p(float(anchor_side_count - 1))
                * weak_anchor_multiplier
            )
            # A very broad concordant coalition makes its sole dissenter
            # especially distinctive, while leaving three-versus-one control
            # profiles close to the accepted static accumulator.
            if anchor_side_count >= 5:
                dissent_multiplier += (
                    singleton_type_gain
                    * singleton_bonus
                    * 0.45
                    * float(anchor_side_count - 4)
                    * weak_anchor_multiplier
                )
            opposition_support *= dissent_multiplier

            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        total_support = support_a + support_b
        core_evidence = (support_b - support_a) / max(total_support, 1e-12)

    # Gap-sensitive restoration remains available, but exactly two opponents
    # receive little automatic restoration. A separate steep count branch is
    # almost off at two and effectively reaches the common gate ceiling at
    # three or more, without requiring the gap gates or subject type to agree.
    restoration_gate = 0.0
    if isolated_anchor and unique_anchor > 0.0:
        count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - restoration_threshold)
        )
        gap_gate = sigmoid(
            restoration_gap_slope
            * (validity_gap - restoration_gap_threshold)
        )
        credibility_restoration = sigmoid(
            restoration_credibility_slope
            * (credibility_gap - restoration_credibility_threshold)
        )
        selective_restoration = (
            restoration_ceiling
            * count_gate
            * gap_gate
            * credibility_restoration
        )

        if opposition_count == 2:
            selective_restoration *= 0.30

        large_count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - 2.50)
        )
        count_dominant_restoration = 0.99 * large_count_gate
        restoration_gate = max(
            selective_restoration,
            count_dominant_restoration,
        )

    # Equal top validities remain exchangeable in integration. Order only
    # supplies a bounded tie anchor rather than a general first-cue bonus.
    tied_top = float(best_candidates.size > 1)
    order_gate = tie_order_gate * tied_top
    gate = 1.0 - (1.0 - restoration_gate) * (1.0 - order_gate)
    gate = float(np.clip(gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * anchor_direction
    )

    logits = beta * np.array(
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
    probs /= probs.sum()
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
**Description:** Adaptive Distinctive Coalition Gating (ADCG) proposes that binary-feature decisions arise from task-adaptive cue weights combining communicated validity, presentation prominence, distributional rarity, and learned cue dependence. Directionally concordant cues are compressed as potentially redundant, while sparse dissent can dominate when its cues have reliably and repeatedly opposed the anchor's cues. Learned anti-concurrence is reliability-sensitive: disagreement near chance is strongly suppressed, whereas well-observed disagreement above a subject-specific threshold receives a bounded nonlinear gain. A profile-driven gate interpolates between coalition comparison and a lexicographic anchor, preserving anchoring against large redundant coalitions while permitting diagnostic opposition to suppress it. For compact opposition, gate suppression is controlled by a selective summary of the strongest reliably learned opposing relationship rather than an indiscriminate average across all cue pairs; larger coalitions retain mean aggregation and strong size decay. In addition, a unique, early-presented validity anchor loses accessibility within the coalition core when it faces compact, reliably anti-concurrent opposition. This attenuation targets residual anchoring without increasing dissent support and vanishes for tied-validity anchors, late anchors, and large opposing coalitions. When top validities are tied, presentation order supplies a bounded anchor but the distinctive-coalition core is attenuated by a stable subject-specific amount. Stable variation in validity attention, order prominence, anti-concurrence threshold and gain, diagnostic selectivity, anchor attenuation, tied-top attenuation, redundancy, gating, and temperature produces structured individual differences without experiment-specific strategies.

**Parameters:**
- validities: validities
- validity_attention: [0.2, 0.9]
- order_prominence: [0.15, 0.7]
- distribution_attention: [0.8, 1.8]
- distinctiveness: [0.35, 0.9]
- dissent_bonus: [1.0, 2.2]
- anti_concurrence_attention: [1.0, 4.8]
- anti_concurrence_threshold: [0.62, 0.76]
- diagnostic_selectivity: [2.0, 8.0]
- anchor_attenuation: [0.8, 3.2]
- redundancy: [0.68, 1.15]
- lex_strength: [0.7, 1.4]
- gate_sharpness: [4.0, 11.0]
- gate_rarity: [1.0, 4.0]
- tie_gate_ceiling: [0.55, 0.85]
- tie_core_attenuation: [0.7, 0.95]
- beta: [0.6, 3.4]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"ADCG expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_attention = float(parameters["validity_attention"])
    order_prominence = float(parameters["order_prominence"])
    distribution_attention = float(parameters["distribution_attention"])
    distinctiveness = float(parameters["distinctiveness"])
    dissent_bonus = float(parameters["dissent_bonus"])
    anti_concurrence_attention = float(parameters["anti_concurrence_attention"])
    anti_concurrence_threshold = float(parameters["anti_concurrence_threshold"])
    diagnostic_selectivity = float(parameters["diagnostic_selectivity"])
    anchor_attenuation = float(parameters["anchor_attenuation"])
    redundancy = float(parameters["redundancy"])
    lex_strength = float(parameters["lex_strength"])
    gate_sharpness = float(parameters["gate_sharpness"])
    gate_rarity = float(parameters["gate_rarity"])
    tie_gate_ceiling = float(parameters["tie_gate_ceiling"])
    tie_core_attenuation = float(parameters["tie_core_attenuation"])
    beta = float(parameters["beta"])

    # Communicated validity is represented on a reliability-evidence scale.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    reliability = (reliability - np.mean(reliability)) / (
        np.std(reliability) + 0.35
    )

    # Estimate how often each cue discriminates and how often cue pairs
    # concur. Symmetric pseudocounts make the model well-defined early in
    # the block while allowing randomized task experience to update weights.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    discr_count = np.ones(n_features, dtype=float) * 2.0
    pair_both = np.ones((n_features, n_features), dtype=float) * 2.0
    pair_same = np.ones((n_features, n_features), dtype=float)

    for i in range(n_past):
        pa = np.asarray(past_a[i], dtype=float)
        pb = np.asarray(past_b[i], dtype=float)
        if pa.shape[0] != n_features or pb.shape[0] != n_features:
            continue
        s = np.sign(pb - pa)
        on = np.flatnonzero(s != 0)
        discr_count[on] += 1.0
        for q, j in enumerate(on):
            for k in on[q + 1:]:
                pair_both[j, k] += 1.0
                pair_both[k, j] += 1.0
                if s[j] == s[k]:
                    pair_same[j, k] += 1.0
                    pair_same[k, j] += 1.0

    discr_rate = discr_count / float(n_past + 4.0)
    reference_rate = float(np.mean(discr_rate))
    rarity = np.clip(reference_rate / discr_rate, 0.35, 3.5)

    # Learned anti-concurrence is high only for cue pairs that repeatedly
    # discriminate in opposite directions. Pair-specific observation counts
    # determine reliability, and a steep subject-specific threshold removes
    # weak or chance-level disagreement while retaining systematic dissent.
    anti_concurrence = np.clip(1.0 - pair_same / pair_both, 0.0, 1.0)
    pair_observations = np.maximum(pair_both - 2.0, 0.0)
    pair_confidence = pair_observations / (pair_observations + 2.0)
    pair_confidence *= np.clip((pair_observations - 1.0) / 3.0, 0.0, 1.0)
    threshold_baseline = 1.0 / (
        1.0 + np.exp(-18.0 * (0.5 - anti_concurrence_threshold))
    )
    thresholded_anti = 1.0 / (
        1.0 + np.exp(-18.0 * (anti_concurrence - anti_concurrence_threshold))
    )
    thresholded_anti = np.clip(
        (thresholded_anti - threshold_baseline)
        / max(1.0 - threshold_baseline, 1e-12),
        0.0,
        1.0,
    )
    learned_anti = pair_confidence * thresholded_anti

    # Baseline cue accessibility combines instructed validity, serial
    # prominence, and learned distributional rarity.
    if n_features > 1:
        position = 1.0 - np.arange(n_features, dtype=float) / (n_features - 1.0)
    else:
        position = np.ones(1, dtype=float)
    cue_weight = np.exp(
        validity_attention * reliability
        + order_prominence * position
        + distribution_attention * np.log(rarity)
    )

    # Cues that historically concur with their current coalition partners
    # are treated as less independent. A cue with no partner is not given
    # this historical redundancy penalty.
    for j in active:
        partners = [
            int(k) for k in active
            if k != j and directions[k] == directions[j]
        ]
        if partners:
            concurrence = [pair_same[j, k] / pair_both[j, k] for k in partners]
            corr_load = float(np.mean(concurrence))
            cue_weight[j] /= 1.0 + redundancy * corr_load

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    # Lexicographic anchor: highest communicated validity among currently
    # discriminating cues, with stable presentation order breaking ties.
    active_validity = validities[active]
    best_validity = float(np.max(active_validity))
    best_candidates = active[np.isclose(active_validity, best_validity, atol=1e-10)]
    lex_cue = int(np.min(best_candidates))
    lex_direction = float(directions[lex_cue])

    # A global top-validity tie attenuates profile-based singleton
    # amplification, preventing the first cue from receiving both a strong
    # order anchor and a full distinctiveness bonus.
    sorted_v = np.sort(validities)[::-1]
    if sorted_v.size >= 2:
        top_gap = float(sorted_v[0] - sorted_v[1])
        global_top_tie = float(np.exp(-80.0 * top_gap))
    else:
        global_top_tie = 1.0
    tie_core_multiplier = 1.0 - tie_core_attenuation * global_top_tie

    # Current-profile distinctiveness: members of the smaller directional
    # coalition stand out, while each coalition's accumulated contribution
    # is divided by a redundancy term.
    if n_a > 0 and n_b > 0:
        if n_a < n_b:
            cue_weight[a_cues] *= (float(n_b) / float(n_a)) ** (
                distinctiveness * tie_core_multiplier
            )
        elif n_b < n_a:
            cue_weight[b_cues] *= (float(n_a) / float(n_b)) ** (
                distinctiveness * tie_core_multiplier
            )

        # Sparse evidence opposing the validity-defined anchor receives its
        # ordinary bounded bonus. Additional influence is restricted to
        # dissenting cues with reliably learned above-threshold opposition.
        dissent_cues = a_cues if lex_direction > 0 else b_cues
        anchor_cues = b_cues if lex_direction > 0 else a_cues
        n_dissent = int(dissent_cues.size)
        if n_dissent > 0:
            for j in dissent_cues:
                learned_diagnosticity = 0.0
                if anchor_cues.size > 0:
                    learned_diagnosticity = float(
                        np.mean(learned_anti[j, anchor_cues])
                    )
                bonus = 1.0 + (dissent_bonus / float(n_dissent)) * (
                    1.0
                    + anti_concurrence_attention * learned_diagnosticity
                )
                cue_weight[j] *= bonus

            # A unique and presentation-prominent validity anchor becomes
            # less accessible in the coalition core when compact opposition
            # has a reliably learned anti-concurrent relationship with it.
            # Active-cue validity uniqueness leaves tied-top profiles intact;
            # squared prominence and size decay protect late anchors and
            # anchors facing large, plausibly redundant coalitions.
            if active_validity.size >= 2:
                sorted_active_v = np.sort(active_validity)[::-1]
                active_gap = float(sorted_active_v[0] - sorted_active_v[1])
                anchor_uniqueness = 1.0 - np.exp(-80.0 * active_gap)
            else:
                anchor_uniqueness = 1.0
            anchor_diagnosticity = float(
                np.max(learned_anti[dissent_cues, lex_cue])
            )
            compact_decay = np.exp(-1.5 * float(n_dissent - 1))
            anchor_signal = (
                anchor_uniqueness
                * float(position[lex_cue] ** 2)
                * anchor_diagnosticity
                * compact_decay
            )
            cue_weight[lex_cue] *= np.exp(
                -anchor_attenuation * anchor_signal
            )

    support_a = 0.0
    support_b = 0.0
    if n_a > 0:
        support_a = float(np.sum(cue_weight[a_cues])) / (float(n_a) ** redundancy)
    if n_b > 0:
        support_b = float(np.sum(cue_weight[b_cues])) / (float(n_b) ** redundancy)

    scale = support_a + support_b
    core_evidence = (support_b - support_a) / max(scale, 1e-12)

    # A near tie among the two globally strongest instructed validities
    # raises the stable-order gate, but anchoring has a subject-specific
    # ceiling rather than becoming obligatorily lexicographic.
    tied_top_gate = tie_gate_ceiling * global_top_tie

    # Profile-driven anchoring occurs when the earliest discriminating cue
    # is on an isolated minority side. It is attenuated when a sparse cue in
    # the opposing coalition is historically rare or reliably
    # anti-concurrent, while a large opposition remains interpretable as
    # redundant evidence.
    first_cue = int(active[0])
    first_side_count = n_b if directions[first_cue] > 0 else n_a
    other_side_count = n_a if directions[first_cue] > 0 else n_b
    if first_side_count > 0 and other_side_count > 0:
        isolation = 1.0 - float(first_side_count) / float(other_side_count)
        minority_first = float(first_side_count < other_side_count)
        opposing_cues = a_cues if directions[first_cue] > 0 else b_cues
        first_side_cues = b_cues if directions[first_cue] > 0 else a_cues
        opposing_rarity = float(np.max(np.maximum(np.log(rarity[opposing_cues]), 0.0)))
        opposing_distinctiveness = opposing_rarity / (
            np.log(3.5) * np.sqrt(float(other_side_count))
        )

        # For compact opposition, a reliability-weighted soft maximum lets
        # one strongly diagnostic relation disengage the anchor without
        # being diluted by weak pairings. Larger coalitions retain mean
        # aggregation, and all configurations retain exponential size decay.
        opposition_relations = learned_anti[
            np.ix_(opposing_cues, first_side_cues)
        ].reshape(-1)
        if other_side_count <= 3 and opposition_relations.size > 1:
            select_logits = diagnostic_selectivity * (
                opposition_relations - np.max(opposition_relations)
            )
            select_weights = np.exp(select_logits)
            select_weights /= np.sum(select_weights)
            learned_opposition = float(
                np.sum(select_weights * opposition_relations)
            )
        else:
            learned_opposition = float(np.mean(opposition_relations))
        learned_opposition *= np.exp(-1.5 * float(other_side_count - 1))
        opposing_distinctiveness += (
            anti_concurrence_attention * learned_opposition
        )
    else:
        isolation = 0.0
        minority_first = 0.0
        opposing_distinctiveness = 0.0

    profile_gate = minority_first / (
        1.0 + np.exp(-gate_sharpness * (isolation - 0.30))
    )
    profile_gate *= np.exp(-gate_rarity * opposing_distinctiveness)
    gate = 1.0 - (1.0 - tied_top_gate) * (1.0 - profile_gate)
    gate = float(np.clip(gate, 0.0, 1.0))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * lex_strength * lex_direction
    )

    logits = beta * np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    high_count_anchor_choices = []\n    two_opponent_anchor_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 8 or b.size < 8:\n            continue\n\n        directions = np.sign(b - a)\n        anchor_direction = int(directions[7])\n        if anchor_direction == 0:\n            continue\n\n        # Retain isolated-expert-8 profiles and count cues opposing it.\n        other_directions = directions[:7]\n        if np.any(other_directions == anchor_direction):\n            continue\n        opponent_count = int(np.sum(other_directions == -anchor_direction))\n\n        anchor_response = 1 if anchor_direction > 0 else 0\n        chose_anchor = float(int(row[\"response\"]) == anchor_response)\n\n        if opponent_count == 2:\n            two_opponent_anchor_choices.append(chose_anchor)\n        elif opponent_count >= 3:\n            high_count_anchor_choices.append(chose_anchor)\n\n    if not high_count_anchor_choices or not two_opponent_anchor_choices:\n        return float(\"nan\")\n\n    return float(\n        np.mean(high_count_anchor_choices)\n        - np.mean(two_opponent_anchor_choices)\n    )",
  "rationale": "This is the isolated-anchor restoration-step contrast: the probability of choosing expert 8's favored option with three to six opponents minus its probability with exactly two opponents. Coding choices relative to the anchor, rather than as A versus B responses, uses the exact reversals to cancel response-side bias. PDRR predicts a sharp and nearly type-invariant restoration once opposition reaches three cues, while its two-opponent compact-challenge branch generally suppresses the anchor. ADCG has no count-threshold restoration mechanism and should instead change more smoothly with coalition size. Pooling all 60 observations at counts three through six and contrasting them with the 24 count-two observations increases within-subject precision; count-one trials are deliberately excluded because PDRR assigns them especially large stable type heterogeneity."
}
```

## Usage

```json
{
  "input_tokens": 10773,
  "output_tokens": 2436
}
```
