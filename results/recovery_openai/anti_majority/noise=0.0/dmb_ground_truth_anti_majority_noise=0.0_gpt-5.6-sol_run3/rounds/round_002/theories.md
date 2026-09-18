# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Dominance-Calibrated Saturating Coalitions (DCSC) proposes that people classify the communicated validity profile before evaluating individual options. A cue activates a dominance regime when it accounts for a sufficiently large share of total diagnosticity and the profile as a whole is concentrated rather than broadly distributed across similarly informative cues. In that regime, the highest-validity discriminating cue anchors the decision, while the remaining cues provide only a small, order-invariant adjustment. Without sufficient profile-level dominance, people enter a compensatory regime and accumulate signed reliability evidence from every discriminating cue. Reliability differences are compressed, and repeated support for the same option has diminishing marginal impact, representing perceived redundancy among concordant experts. Thus, several lower-validity cues can collectively defeat the best discriminating cue, but neither early nor late serial position is intrinsically favored. Evidential balance can induce additional deliberation, modeled as a modest increase in choice consistency rather than a directional recency rule. Stable individual differences concern the dominance threshold, sensitivity to profile concentration, compression and saturation of reliability evidence, leakage of coalition evidence into dominance decisions, deliberation under conflict, and response sensitivity.

**Rationale:** This is a minimal extension of the accepted DCSC gate. The strongest-cue share is retained, but a normalized Herfindahl concentration term is added so profiles with similar top shares are no longer treated as equivalent. A profile with one dominant cue and a generally weak remainder receives a high dominance score, whereas a profile retaining substantial distributed lower-cue diagnosticity remains compensatory. The revised threshold is calibrated to the expanded score rather than increasing dominance globally. This should preserve Experiment 1's stable dominance while moving Experiment 2 into the existing compensatory branch, whose compressed, weakly saturated coalition evidence permits lower cues to overturn the highest discriminating cue. All order-invariant evidence, anchoring, leakage, saturation, conflict-deliberation, and response equations are otherwise unchanged, protecting the exact absence of an Experiment-4 positional crossover. Moving diffuse profiles into compensation also gives the existing balance-dependent deliberation mechanism behavioral leverage in Experiment 3 without adding recency.

**Parameters:**
  - `validities`: `validities`
  - `dominance_threshold`: `[0.285, 0.305]`
  - `gap_sensitivity`: `[70.0, 110.0]`
  - `profile_shape_weight`: `[1.8, 2.2]`
  - `validity_compression`: `[0.02, 0.18]`
  - `redundancy_saturation`: `[0.08, 0.25]`
  - `anchor_strength`: `[0.8, 1.2]`
  - `dominance_leak`: `[0.02, 0.10]`
  - `conflict_deliberation`: `[0.30, 0.60]`
  - `beta`: `[2.8, 4.8]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCSC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive signs favor B and negative signs favor A. Presentation
    # position never enters the evidence weights.
    signs = np.sign(stim[1] - stim[0])
    discriminating = np.flatnonzero(signs != 0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    dominance_threshold = float(parameters["dominance_threshold"])
    gap_sensitivity = float(parameters["gap_sensitivity"])
    profile_shape_weight = float(parameters["profile_shape_weight"])
    validity_compression = float(parameters["validity_compression"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    anchor_strength = float(parameters["anchor_strength"])
    dominance_leak = float(parameters["dominance_leak"])
    conflict_deliberation = float(parameters["conflict_deliberation"])
    beta = float(parameters["beta"])

    # Communicated validities are converted to nonnegative diagnostic
    # strengths. The small exponent compresses differences in the
    # compensatory regime without reversing the communicated ordering.
    diagnosticity = np.clip(2.0 * validities - 1.0, 1e-6, 1.0)
    cue_weights = diagnosticity ** validity_compression

    # Dominance depends both on the strongest cue's diagnosticity share
    # and on concentration across the complete profile. The normalized
    # Herfindahl term distinguishes a genuinely concentrated profile from
    # one in which the strongest cue has a similar share but substantial
    # diagnosticity remains broadly distributed among lower cues.
    profile_shares = diagnosticity / max(float(np.sum(diagnosticity)), 1e-12)
    dominance_concentration = float(np.max(profile_shares))
    profile_hhi = float(np.sum(profile_shares ** 2))
    uniform_hhi = 1.0 / float(n_features)
    shape_concentration = (profile_hhi - uniform_hhi) / max(
        1.0 - uniform_hhi, 1e-12
    )
    shape_concentration = float(np.clip(shape_concentration, 0.0, 1.0))
    dominance_score = (
        dominance_concentration
        + profile_shape_weight * shape_concentration
    )
    gate_logit = gap_sensitivity * (
        dominance_score - dominance_threshold
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    dominance_gate = 1.0 / (1.0 + np.exp(-gate_logit))

    # Aggregate reliability separately for the two coalitions. The
    # exponential transform gives diminishing marginal impact to repeated
    # concordant evidence and is invariant to the order in which cues are
    # displayed.
    support_b = float(np.sum(cue_weights[(signs > 0)]))
    support_a = float(np.sum(cue_weights[(signs < 0)]))
    rho = max(redundancy_saturation, 1e-8)
    bounded_b = -np.expm1(-rho * support_b) / rho
    bounded_a = -np.expm1(-rho * support_a) / rho

    # Express coalition evidence in approximately cue-vote units so the
    # response scale remains comparable across validity profiles.
    mean_discriminating_weight = float(np.mean(cue_weights[discriminating]))
    coalition_evidence = (bounded_b - bounded_a) / max(
        mean_discriminating_weight, 1e-12
    )

    # The anchor is the most valid cue that actually discriminates. Ties
    # in communicated validity are resolved without using serial position:
    # their signed evidence is averaged, so tied opposing anchors cancel.
    disc_weights = diagnosticity[discriminating]
    best_weight = float(np.max(disc_weights))
    best_mask = np.isclose(disc_weights, best_weight, rtol=1e-10, atol=1e-12)
    anchor_direction = float(np.mean(signs[discriminating][best_mask]))
    anchor_evidence = anchor_strength * anchor_direction

    # In dominance mode coalition evidence only leaks weakly into the
    # anchored judgment. In integration mode it is fully compensatory.
    dominance_evidence = anchor_evidence + dominance_leak * coalition_evidence
    choice_evidence = (
        dominance_gate * dominance_evidence
        + (1.0 - dominance_gate) * coalition_evidence
    )

    # Balanced conflict can prompt more careful implementation of the
    # selected rule, but it supplies no directional or last-cue evidence.
    count_b = int(np.sum(signs > 0))
    count_a = int(np.sum(signs < 0))
    balance = np.exp(-abs(count_b - count_a))
    beta_eff = beta * (1.0 + conflict_deliberation * balance)

    utilities = np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = beta_eff * utilities
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Adaptive Distinctive Coalition Gating (ADCG) proposes that binary-feature decisions arise from task-adaptive cue weights combining communicated validity, presentation prominence, distributional rarity, and learned cue dependence. Directionally concordant cues are compressed as potentially redundant, while sparse dissent can dominate when its cues have reliably and repeatedly opposed the anchor's cues. Learned anti-concurrence is reliability-sensitive: disagreement near chance is strongly suppressed, whereas well-observed disagreement above a subject-specific threshold receives a bounded nonlinear gain. A profile-driven gate interpolates between coalition comparison and a lexicographic anchor, preserving anchoring against large redundant coalitions while permitting diagnostic opposition to suppress it. For compact opposition, gate suppression is controlled by a selective summary of the strongest reliably learned opposing relationship rather than an indiscriminate average across all cue pairs; larger coalitions retain mean aggregation and strong size decay. In addition, a unique, early-presented validity anchor loses accessibility within the coalition core when it faces compact, reliably anti-concurrent opposition. This attenuation targets residual anchoring without increasing dissent support and vanishes for tied-validity anchors, late anchors, and large opposing coalitions. When top validities are tied, presentation order supplies a bounded anchor but the distinctive-coalition core is attenuated by a stable subject-specific amount. Stable variation in validity attention, order prominence, anti-concurrence threshold and gain, diagnostic selectivity, anchor attenuation, tied-top attenuation, redundancy, gating, and temperature produces structured individual differences without experiment-specific strategies.

**Rationale:** This is a two-range calibration of the accepted iteration-9 ADCG source; no equations or pathways are added. First, the upper bound of anchor_attenuation increases from 2.4 to 3.2 while its lower bound remains fixed. Because attenuation is already conditional on a unique validity anchor, early presentation, high-confidence anti-concurrence, and compact opposition, the added strength and subject heterogeneity are localized to the residual failures in Experiments 2 and 6. Tied-top profiles, late anchors, and large opposing coalitions remain protected by the existing multiplicative exclusions and size decay. Second, redundancy shifts modestly from [0.75, 1.25] to [0.68, 1.15]. This slightly softens generic coalition compression, reducing the excessive balanced-versus-imbalanced effect in Experiment 3. It should also modestly reduce excess top-cue adherence in Experiments 2 and 5 and strengthen the negative multi-opponent contrast in Experiment 6, while the unchanged large-coalition profile gate preserves Experiment 1. The successful gate-only soft maximum, mean direct diagnostic bonus, tied-top mechanisms, and all confidence computations are otherwise retained exactly.

**Parameters:**
  - `validities`: `validities`
  - `validity_attention`: `[0.2, 0.9]`
  - `order_prominence`: `[0.15, 0.7]`
  - `distribution_attention`: `[0.8, 1.8]`
  - `distinctiveness`: `[0.35, 0.9]`
  - `dissent_bonus`: `[1.0, 2.2]`
  - `anti_concurrence_attention`: `[1.0, 4.8]`
  - `anti_concurrence_threshold`: `[0.62, 0.76]`
  - `diagnostic_selectivity`: `[2.0, 8.0]`
  - `anchor_attenuation`: `[0.8, 3.2]`
  - `redundancy`: `[0.68, 1.15]`
  - `lex_strength`: `[0.7, 1.4]`
  - `gate_sharpness`: `[4.0, 11.0]`
  - `gate_rarity`: `[1.0, 4.0]`
  - `tie_gate_ceiling`: `[0.55, 0.85]`
  - `tie_core_attenuation`: `[0.7, 0.95]`
  - `beta`: `[0.6, 3.4]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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
