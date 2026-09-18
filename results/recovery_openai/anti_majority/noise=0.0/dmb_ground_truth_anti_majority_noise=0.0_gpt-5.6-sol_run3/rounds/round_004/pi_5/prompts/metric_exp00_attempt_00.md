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
**Validities (n_features=6):** [0.95, 0.72, 0.72, 0.55, 0.72, 0.72]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 9: A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]

**Rationale:** This design isolates ADCG's selective use of the strongest reliably learned opposing relationship in a compact two-cue coalition, together with diagnostic dissent gain and attenuation of an early unique anchor. Expert 1 is the unique 95%-valid anchor and discriminates on every trial. Experts 2, 3, 5, and 6 all have validity 72% and each discriminates in exactly five of the six underlying patterns, equating their advertised reliability and asymptotic rarity. The first two patterns are the critical targets. In the diagnostic target, Experts 3 and 5 oppose the anchor; in the control target, Experts 2 and 6 oppose it. Both targets therefore present exactly one 95% anchor against two 72% opponents. SCCI assigns them identical evidence and predicts an exactly zero target difference at every experience level because it ignores presentation position and history, and its isolated-anchor restoration depends only on the shared validity gap and coalition counts. Across the four history-building patterns, Expert 3 always opposes Expert 1, so their anti-concurrence approaches its reliable, above-threshold regime. Each of Experts 2, 5, and 6 instead opposes Expert 1 on only two of four builders; including its target occurrence, each has three oppositions and two concurrences with the anchor, leaving its smoothed anti-concurrence below ADCG's threshold. On the diagnostic target, ADCG's reliability-weighted selective maximum can consequently detect the strong Expert-3 relationship without diluting it by Expert 5's weak relationship. This suppresses the profile gate protecting the isolated early anchor, amplifies Expert 3's dissent, and attenuates Expert 1 in the coalition core. Neither control opponent receives these learned effects. Historical concurrence within the two target coalitions is matched: Experts 3 and 5 concur on three of their five joint observations, as do Experts 2 and 6. Thus learned within-coalition redundancy cannot explain the critical contrast. The main quantitative prediction is an experience-dependent decrease in anchor choices on the diagnostic target relative to the control target: the late-minus-early change in [P(anchor choice | diagnostic) - P(anchor choice | control)] should be negative under ADCG but exactly zero under SCCI. Any initial serial-position difference can be estimated in the early trials rather than mistaken for learning; the opposing pairs also have matched summed prominence. Exact A/B reversals preserve every cue relationship while controlling response-side bias. The 12 unique pairs are repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Static Configuration-Contingent Integration (SCCI) claims that people form a fixed, instruction-based representation of cue reliability and apply it to the directional evidence profile on each trial. Each discriminating cue contributes reliability-weighted evidence, so independently supporting cues accumulate, while concordant cues are compressed as potentially redundant. This compression increases moderately convexly with coalition size, reflecting the declining informational value of each additional concordant cue. A bounded configuration gate supplements integration: strong protection of a uniquely most-valid cue is restricted to profiles in which that cue is isolated, has a sufficiently large communicated-validity advantage, and faces multiple concordant opponents. Cue uniqueness functions primarily as a tie detector, while a separate raw validity-gap criterion determines whether restoration is warranted. The transition to restoration is sharp, allowing an isolated anchor to be weak against one credible challenger but strongly protected against a redundant multi-cue bloc. Compact challenge amplification is strictly limited to isolated anchors and therefore cannot create general primacy effects when the anchor has allies. When the most-valid discriminating cues have equal communicated validity, they remain exchangeable in the integrator, but stable display order supplies a bounded tie-breaking anchor. Choices are history-invariant because concurrence, anti-concurrence, and discrimination frequency are not learned without correctness feedback. Stable profile heterogeneity arises from subject-level variation in validity sensitivity, coalition integration, redundancy, isolated-anchor challenge and restoration, tie-order bias, temperature, and lapse.

**Parameters:**
- validities: validities
- validity_sensitivity: [0.03, 0.35]
- coalition_accumulation: [0.98, 1.16]
- redundancy_discount: [0.30, 0.42]
- redundancy_curvature: [1.15, 1.45]
- profile_balance_gain: [0.0, 0.0]
- anchor_gate_strength: [0.90, 0.995]
- challenge_strength: [0.8, 2.8]
- restoration_strength: [0.95, 1.0]
- restoration_threshold: [1.15, 1.55]
- restoration_slope: [16.0, 30.0]
- restoration_gap_threshold: [0.055, 0.075]
- restoration_gap_slope: [160.0, 240.0]
- tie_order_bias: [0.58, 0.85]
- beta: [0.9, 3.4]
- lapse: [0.0, 0.12]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCCI expects state with shape (2, n_features); got {stim.shape}."
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
    coalition_accumulation = float(parameters["coalition_accumulation"])
    redundancy_discount = float(parameters["redundancy_discount"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    profile_balance_gain = float(parameters["profile_balance_gain"])
    anchor_gate_strength = float(parameters["anchor_gate_strength"])
    challenge_strength = float(parameters["challenge_strength"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap_threshold = float(parameters["restoration_gap_threshold"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    tie_order_bias = float(parameters["tie_order_bias"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    # Communicated validity is transformed into reliability evidence. The
    # subtraction only fixes scale and does not alter validity ordering.
    v = np.clip(validities, 0.500001, 0.999999)
    log_odds = np.log(v / (1.0 - v))
    active_reference = float(np.max(log_odds[active]))
    cue_weight = np.exp(
        validity_sensitivity * (log_odds - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Accumulation represents multiple nominally independent reasons.
        # A moderately convex denominator captures increasing redundancy
        # among additional cues in the same directional coalition.
        mean_reliability = float(np.mean(cue_weight[indices]))
        accumulated = mean_reliability * (float(n) ** coalition_accumulation)
        redundancy = 1.0 + redundancy_discount * (
            float(n - 1) ** redundancy_curvature
        )
        return accumulated / redundancy

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # Directional minorities receive only a very small configuration-based
    # salience adjustment. Most coalition effects arise from accumulation,
    # redundancy, and the isolated-anchor mechanism below.
    if n_a > 0 and n_b > 0 and n_a != n_b:
        if n_a < n_b:
            ratio = float(n_b) / float(n_a)
            support_a *= ratio ** profile_balance_gain
        else:
            ratio = float(n_a) / float(n_b)
            support_b *= ratio ** profile_balance_gain

    total_support = support_a + support_b
    core_evidence = (support_b - support_a) / max(total_support, 1e-12)

    # The validity anchor is the most-valid currently discriminating cue.
    # Equal-validity candidates have identical integrative weights; order is
    # consulted only to select a bounded tie anchor.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]
    anchor_cue = int(np.min(best_candidates))
    anchor_direction = float(directions[anchor_cue])

    # Uniqueness is a tie detector rather than a second graded validity-gap
    # penalty. The separate raw-gap sigmoid below selects strong anchors.
    if active.size > 1:
        sorted_active = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active[0] - sorted_active[1])
    else:
        validity_gap = 0.5
    unique_anchor = float(best_candidates.size == 1)

    if anchor_direction > 0:
        anchor_side_count = n_b
        opposition_count = n_a
        anchor_side_support = support_b
        opposition_support = support_a
    else:
        anchor_side_count = n_a
        opposition_count = n_b
        anchor_side_support = support_a
        opposition_support = support_b

    isolated_anchor = float(anchor_side_count == 1)

    if opposition_count == 0:
        weak_opposition = 1.0
    else:
        relative_opposition = opposition_support / max(
            anchor_side_support + opposition_support, 1e-12
        )
        weak_opposition = np.exp(-4.0 * relative_opposition)

    # A single independent opponent receives extra challenge amplification
    # only when the validity anchor is genuinely isolated. Non-isolated
    # profiles are governed by ordinary static coalition integration.
    if opposition_count > 0:
        compact_challenge = np.exp(
            -4.0 * float(opposition_count - 1) ** 2
        )
        challenge_gain = isolated_anchor * (
            coalition_accumulation + challenge_strength
        )
        challenged_opposition = opposition_support * (
            1.0
            + unique_anchor * challenge_gain * compact_challenge
        )
        anchor_margin = (
            anchor_side_support - challenged_opposition
        ) / max(anchor_side_support + challenged_opposition, 1e-12)
        core_evidence = anchor_direction * anchor_margin

    # Strong restoration is specific to an isolated and clearly superior
    # anchor facing multiple concordant opponents. Count and raw validity-gap
    # thresholds vary across subjects, with a sharp gap transition separating
    # modest-gap profiles from genuinely strong-anchor configurations.
    opposition_restoration = 1.0 / (
        1.0 + np.exp(
            -restoration_slope
            * (float(opposition_count) - restoration_threshold)
        )
    )
    gap_restoration = 1.0 / (
        1.0 + np.exp(
            -restoration_gap_slope
            * (validity_gap - restoration_gap_threshold)
        )
    )
    unique_gate_profile = (
        0.01 * weak_opposition
        + isolated_anchor
        * restoration_strength
        * opposition_restoration
        * gap_restoration
    )
    unique_gate = anchor_gate_strength * unique_anchor * float(
        np.clip(unique_gate_profile, 0.0, 1.0)
    )

    # Stable order matters only when multiple currently discriminating cues
    # share the maximum communicated validity.
    tied_active_top = float(best_candidates.size > 1)
    tie_gate = tie_order_bias * tied_active_top

    gate = 1.0 - (1.0 - unique_gate) * (1.0 - tie_gate)
    gate = float(np.clip(gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * anchor_direction
    )

    logits = beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subject_score(subj):\n        # Store anchor-following responses separately for the diagnostic and\n        # control profiles and for their two A/B orientations.\n        cells = {\n            ('diagnostic', 0): [], ('diagnostic', 1): [],\n            ('control', 0): [], ('control', 1): []\n        }\n\n        for _, row in subj.iterrows():\n            a = np.asarray(row['option_a_ratings'], dtype=int)\n            b = np.asarray(row['option_b_ratings'], dtype=int)\n            active = tuple(np.flatnonzero(a != b).tolist())\n\n            if active == (0, 2, 4):\n                profile = 'diagnostic'\n            elif active == (0, 1, 5):\n                profile = 'control'\n            else:\n                continue\n\n            # Expert 1 (index 0) is the validity anchor. This coding makes 1\n            # mean that the response followed the anchor, independently of\n            # whether the anchor favored A or B.\n            anchor_response = 1 if b[0] > a[0] else 0\n            followed_anchor = float(int(row['response']) == anchor_response)\n            cells[(profile, anchor_response)].append(followed_anchor)\n\n        profile_rates = {}\n        for profile in ('diagnostic', 'control'):\n            orientation_rates = []\n            for orientation in (0, 1):\n                vals = cells[(profile, orientation)]\n                # Each exact orientation occurs eight times. Excluding its\n                # first two presentations targets experience-dependent choice\n                # while retaining six observations per orientation.\n                if len(vals) > 2:\n                    vals = vals[2:]\n                if len(vals) > 0:\n                    orientation_rates.append(float(np.mean(vals)))\n            if len(orientation_rates) == 0:\n                return np.nan\n            profile_rates[profile] = float(np.mean(orientation_rates))\n\n        return profile_rates['control'] - profile_rates['diagnostic']\n\n    scores = []\n    for _, subj in data.groupby('subject_id', sort=False):\n        value = subject_score(subj)\n        if np.isfinite(value):\n            scores.append(value)\n    if len(scores) == 0:\n        return np.nan\n    return float(np.mean(scores))",
  "rationale": "This is the late-experience control-minus-diagnostic difference in probability of following the 95%-valid anchor. It separately balances the two A/B reversals, eliminating response-side effects, and averages equally over subjects. ADCG predicts a positive value because learned anti-concurrence between Experts 1 and 3 selectively reduces anchor following on the diagnostic profile, whereas the equally reliable control coalition does not acquire that relationship. SCCI is history-invariant and assigns the diagnostic and control profiles identical anchor-choice probabilities, so it predicts a value near zero. Dropping only the first two occurrences of each orientation emphasizes the learned effect while retaining 24 binary observations per subject for comparatively low sampling variance."
}
```

## Usage

```json
{
  "input_tokens": 8938,
  "output_tokens": 3586
}
```
