# Round 6 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

**Description:** Active-Cue Accessibility Competition (ACAC) proposes that people use a stable representation of communicated cue validity and evaluate only cues that discriminate on the current trial. Active cues first form option-specific coalitions. Coalition support accumulates smoothly but is compressed according to coalition size, validity similarity, and effective cue multiplicity, so additional concordant cues become increasingly redundant without triggering a categorical restoration rule. In unique-anchor configurations where both options have multiple active supporters, coalition evidence is additionally normalized toward effective coalition counts. This prevents the allocation of one relatively valid lower cue from producing an implicit presentation-order crossover. Decisions in this symmetrically normalized geometry are made with moderately greater response precision, reflecting reduced conflict after cue identities have been compressed; this suppresses sampling-induced primacy classifications and strengthens balanced-profile anchor evidence without introducing an order weight. The strongest currently active cue serves as a validity anchor. Opposition receives a sharply count-selective challenge enhancement centered on exactly one opponent. This enhancement is strongest for an isolated anchor and attenuated for embedded-anchor singleton dissent; larger opposing coalitions retain only a small, smooth inverse-count challenge tail. In the exact unique-anchor, isolated, one-opponent configuration, stable competition types form a heterogeneous mixture containing a small genuinely resistant group and a larger set of strongly challenge-accessible integrators. Challenge strength depends continuously on the anchor's absolute validity and validity advantage over the strongest opponent. Presentation order is consulted only when multiple currently active cues share maximal communicated validity. A bounded active-tie gate favors the earliest member of that tied set, with an additional conflict-geometry increment when lower-validity active cues precede tied maxima supporting opposing options. Inactive co-top cues cannot activate either component. Tie accessibility retains the same marginal distribution across subject types as in the previous model. Because the task provides no correctness feedback, ACAC explicitly excludes trial-history learning of cue concurrence, anti-concurrence, or discrimination frequency.

**Rationale:** This is a narrow extension of the accepted iteration-7 model. First, it replaces the increasingly homogeneous three-type singleton loading with six equally sampled correlated mixture cells. Only one-sixth of subjects remains genuinely resistant in the exact unique-anchor, isolated one-opponent branch, while the other five-sixths receive strong challenge through either a larger baseline sensitivity or integrative accessibility. This should further lower Experiment-2 anchor adherence and deepen Experiment 6 while restoring the missing between-subject variance. The tie loading is assigned independently across the six cells in two-cell groups, exactly preserving the previous one-third frequencies of high, low, and intermediate tie accessibility; therefore the already accurate Experiment-12 mean and the Experiment-5 tie effect are protected. Embedded challenge, two-opponent competition, credibility modulation, the high-count tail, and the tie equations are unchanged. Second, the existing hard symmetric normalization is left intact, but response precision is multiplied by 1.6 only in unique-anchor multi-cue-versus-multi-cue configurations. Experiment 4 has identical normalized coalition counts across its Expert-1 support conditions, so this reduces threshold-classification false positives without creating a conditional order effect. The same local precision increase should raise anchor adherence in normalized balanced 2-v-2 profiles relative to unnormalized 3-v-1 profiles, directly targeting Experiment 3's wrong-signed interaction without reviving the rejected convex validity-composition blend. Isolated profiles in Experiments 6, 9, 10, and 11 and all active-tie profiles remain outside this precision edit.

**Parameters:**
  - `validities`: `validities`
  - `validity_gain`: `[0.05, 0.22]`
  - `redundancy_strength`: `[0.62, 1.08]`
  - `redundancy_curvature`: `[1.18, 1.58]`
  - `similarity_sensitivity`: `[5.0, 14.0]`
  - `compact_challenge`: `[5.0, 10.0]`
  - `compact_peak`: `[0.98, 1.02]`
  - `compact_width`: `[0.12, 0.25]`
  - `isolated_singleton_access`: `[1.8, 2.8]`
  - `challenge_tail`: `[0.08, 0.28]`
  - `embedded_compact_scale`: `[0.08, 0.22]`
  - `challenge_gap_scale`: `[0.08, 0.18]`
  - `absolute_threshold`: `[0.68, 0.77]`
  - `absolute_slope`: `[14.0, 32.0]`
  - `weak_anchor_gain`: `[0.65, 1.75]`
  - `isolation_gain`: `[0.35, 1.10]`
  - `credibility_protection`: `[0.0, 0.18]`
  - `active_tie_ceiling`: `[0.70, 0.90]`
  - `tie_activation_rate`: `[1.25, 2.60]`
  - `conflict_tie_boost`: `[0.60, 1.00]`
  - `beta`: `[1.45, 3.80]`
  - `lapse`: `[0.0, 0.10]`
  - `competition_type`: `{0, 1, 2, 3, 4, 5}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"ACAC expects state with shape (2, n_features); got {stim.shape}."
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

    validity_gain = float(parameters["validity_gain"])
    redundancy_strength = float(parameters["redundancy_strength"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    similarity_sensitivity = float(parameters["similarity_sensitivity"])
    compact_challenge = float(parameters["compact_challenge"])
    compact_peak = float(parameters["compact_peak"])
    compact_width = float(parameters["compact_width"])
    isolated_singleton_access = float(parameters["isolated_singleton_access"])
    challenge_tail = float(parameters["challenge_tail"])
    embedded_compact_scale = float(parameters["embedded_compact_scale"])
    challenge_gap_scale = float(parameters["challenge_gap_scale"])
    absolute_threshold = float(parameters["absolute_threshold"])
    absolute_slope = float(parameters["absolute_slope"])
    weak_anchor_gain = float(parameters["weak_anchor_gain"])
    isolation_gain = float(parameters["isolation_gain"])
    credibility_protection = float(parameters["credibility_protection"])
    active_tie_ceiling = float(parameters["active_tie_ceiling"])
    tie_activation_rate = float(parameters["tie_activation_rate"])
    conflict_tie_boost = float(parameters["conflict_tie_boost"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    competition_type = int(parameters["competition_type"])

    # Six correlated mixture cells preserve the previous marginal distribution
    # of active-tie accessibility while assigning only one-sixth of subjects to
    # a genuinely singleton-resistant group. The remaining subjects express a
    # strong but heterogeneous exact-singleton challenge response.
    if competition_type == 0:       # genuinely singleton-resistant
        type_challenge = 0.05
        type_singleton_access = 0.00
    elif competition_type in (1, 2):  # strongly challenge-sensitive
        type_challenge = 4.50
        type_singleton_access = 0.00
    else:                            # strongly accessible integrators
        type_challenge = 1.00
        type_singleton_access = 1.50

    # Tie accessibility is crossed with singleton sensitivity so that changing
    # the singleton mixture does not change the marginal tie-type frequencies.
    if competition_type in (0, 1):
        type_tie = 1.65
    elif competition_type in (2, 3):
        type_tie = 0.30
    else:
        type_tie = 1.00

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Communicated accuracy is represented on a log-evidence scale. Centering
    # on the active maximum improves numerical stability without discarding
    # absolute validity, which is retained separately below.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(validity_gain * (reliability - active_reference))

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        weights = np.asarray(cue_weight[indices], dtype=float)
        raw_support = float(np.sum(weights))

        # Effective multiplicity changes smoothly with coalition composition.
        # A coalition containing several similarly accessible cues has a high
        # effective count and is compressed more strongly than a heterogeneous
        # coalition dominated by one cue.
        effective_n = raw_support / max(float(np.max(weights)), 1e-12)
        effective_excess = max(effective_n - 1.0, 0.0)

        coalition_v = validities[indices]
        validity_spread = float(np.std(coalition_v)) if n > 1 else 0.0
        similarity = np.exp(-similarity_sensitivity * validity_spread)
        discount = 1.0 + redundancy_strength * similarity * (
            effective_excess ** redundancy_curvature
        )
        return raw_support / max(discount, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # The validity anchor is defined only over currently discriminating cues.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    top_candidates = active[np.isclose(active_v, best_v, atol=1e-8)]
    anchor_cue = int(np.min(top_candidates))
    anchor_direction = float(directions[anchor_cue])
    unique_anchor = float(top_candidates.size == 1)

    # In unique-anchor multi-cue-versus-multi-cue configurations, normalize
    # coalition composition toward effective count support. This saturation
    # prevents reallocating one relatively valid lower cue from inducing an
    # implicit primacy crossover, while leaving isolated-anchor diagnostics
    # and all tied-top configurations unchanged.
    normalized_multicue_geometry = False
    if unique_anchor > 0.0 and n_a > 1 and n_b > 1:
        pooled_weight = float(np.mean(cue_weight[active]))

        def normalized_multicue_support(n):
            excess = float(n - 1)
            discount = 1.0 + redundancy_strength * (
                excess ** redundancy_curvature
            )
            return pooled_weight * float(n) / max(discount, 1e-12)

        support_a = normalized_multicue_support(n_a)
        support_b = normalized_multicue_support(n_b)
        normalized_multicue_geometry = True

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

    # Absolute credibility and relative advantage are distinct. A cue can be
    # locally strongest yet remain challengeable when all active cues have low
    # communicated validity.
    weak_anchor = sigmoid(absolute_slope * (absolute_threshold - best_v))
    strong_anchor = 1.0 - weak_anchor

    if opponent_count > 0:
        strongest_opponent_v = float(np.max(validities[opponent_indices]))
        opponent_gap = max(best_v - strongest_opponent_v, 0.0)
    else:
        opponent_gap = 0.5

    gap_accessibility = np.exp(
        -opponent_gap / max(challenge_gap_scale, 1e-8)
    )
    gap_protection = 1.0 - gap_accessibility

    # Credibility protection is deliberately bounded so that merely moving a
    # relatively valid lower cue between coalitions cannot create an implicit
    # unique-anchor presentation-order effect.
    anchor_support *= 1.0 + (
        unique_anchor
        * credibility_protection
        * strong_anchor
        * gap_protection
    )

    if opponent_count > 0:
        # The high-amplitude component is extremely localized at one opponent.
        # Embedded singleton dissent receives only a bounded fraction of it,
        # whereas isolated one-versus-one conflict expresses the full stable
        # subject-type difference. In that exact geometry, a bounded access
        # increment selectively raises strongly accessible integrators while a
        # small mixture remains genuinely resistant.
        count_distance = (
            float(opponent_count) - compact_peak
        ) / max(compact_width, 1e-8)
        singleton_profile = np.exp(-0.5 * count_distance ** 2)

        if anchor_count == 1:
            singleton_geometry = 1.0
        else:
            singleton_geometry = embedded_compact_scale

        active_type_challenge = type_challenge
        if (
            unique_anchor > 0.0
            and anchor_count == 1
            and opponent_count == 1
            and top_candidates.size == 1
        ):
            active_type_challenge += (
                isolated_singleton_access * type_singleton_access
            )

        singleton_intensity = (
            active_type_challenge
            * compact_challenge
            * singleton_profile
            * singleton_geometry
        )
        smooth_tail = challenge_tail / float(opponent_count + 1)

        isolation = 1.0 / float(max(anchor_count, 1))
        geometry_gain = 1.0 + isolation_gain * isolation
        absolute_gain = 1.0 + weak_anchor_gain * weak_anchor

        challenge_multiplier = 1.0 + (
            singleton_intensity + smooth_tail
        ) * gap_accessibility * geometry_gain * absolute_gain
        opponent_support *= challenge_multiplier

    total_support = anchor_support + opponent_support
    if total_support <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total_support
        core_evidence = anchor_direction * anchor_margin

    # A bounded order gate is activated exclusively by a tie among currently
    # active maximum-validity cues. Conflict among tied maxima receives an
    # additional activation increment only when lower-validity active cues
    # precede the tied set. This captures activation-sensitive accessibility
    # without increasing first-cue adherence in unique-maximum profiles.
    active_top_count = int(top_candidates.size)
    if active_top_count >= 2:
        tie_gate = (
            type_tie
            * active_tie_ceiling
            * (1.0 - np.exp(
                -tie_activation_rate * float(active_top_count - 1)
            ))
        )
        tied_directions = directions[top_candidates]
        tied_conflict = float(
            np.any(tied_directions > 0) and np.any(tied_directions < 0)
        )
        preceded_tied_set = float(int(active[0]) < anchor_cue)
        tie_gate += (
            type_tie
            * conflict_tie_boost
            * tied_conflict
            * preceded_tied_set
        )
    else:
        tie_gate = 0.0
    tie_gate = float(np.clip(tie_gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - tie_gate) * core_evidence
        + tie_gate * anchor_direction
    )

    # Symmetric multi-cue normalization removes much of the identity-based
    # conflict, so choices in that geometry are moderately more precise. This
    # affects neither isolated singleton competition nor active-top-tie trials.
    effective_beta = beta * (1.60 if normalized_multicue_geometry else 1.0)
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
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= total
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


### slot 2 — `pi_5` — SURVIVED ✓

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


## Replacement

### `pi_9` → slot 1 (via `new_theory`)

**Description:** Static Configural Coalition Competition (S3C) proposes that decisions are reconstructed from the current cue display without learning from trial history. Each discriminating expert retains an identity-specific weight derived from communicated validity, but validity evidence is bounded to prevent a uniquely most-valid cue from becoming arbitrarily dominant. Serial accessibility is a distinct regime that is exactly absent under a clear unique maximum and operates only among genuinely tied or extremely near-tied active cues. Option-specific evidence accumulates with smooth within-coalition redundancy. Compact one- and two-cue opposition receives a display-based diagnosticity bonus, while the credibility of a coalition is computed through a bounded aggregation of its members' validity gaps so that moving one cue cannot induce a quasi-discrete change in competition. The effects of credibility and anchor isolation are likewise softly bounded. At larger opposition multiplicities, compact diagnosticity decays and a saturating restoration process can return choice toward a strong isolated anchor. This restoration receives a smooth pre-sigmoid increment when an effectively isolated anchor faces more than two opponents, while embedded anchors remain protected from that increment. Exact balance versus one-step imbalance can modulate response precision without changing evidence direction, allowing activation of an additional cue to sharpen an already configured coalition preference. Stable subject differences jointly vary validity compression, tied-cue accessibility, compact-dissent sensitivity, restoration, balance sensitivity, and response precision. In the absence of outcome feedback, no aspect of the model updates from trial history.

**Rationale:** This is a localized edit of the accepted iteration-8 architecture. The bounded validity transform, exact suppression of serial accessibility under unique maxima, identity-preserving coalition support, history invariance, and stable correlated subject types are unchanged. Four targeted changes implement the latest accepted diagnosis. First, opponent credibility now averages tanh-bounded individual validity gaps instead of exponentiating an unbounded mean validity gap, and the isolation contribution is softly bounded. This limits the simultaneous credibility and isolation changes caused by moving Expert 1 between coalitions in Experiment 4 while making compact opposition more consistently accessible in Experiments 2 and 10. Second, the compact-count peak is shifted modestly toward one effective opponent without globally increasing challenge strength. Third, isolated high-count restoration is moved from a weak post-gate multiplier to the pre-sigmoid restoration scale, with a smooth transition above two effective opponents and a cubed anchor-identity-share gate. This targets the missing Experiment 11 restoration contrast and the insufficient Experiment 6 effect while protecting embedded-anchor displays such as Experiment 13. Fourth, only the ranges of tied-order accessibility and balance-sensitive precision are widened. Tied accessibility remains impossible under a unique maximum, while the larger balance-precision range specifically strengthens the exact-balance to one-step-imbalance contrast relevant to Experiment 12 without adding a directional or cue-specific bonus.

**Parameters:**
  - `validities`: `validities`
  - `validity_sensitivity`: `[0.08, 0.70]`
  - `validity_floor`: `[0.58, 0.76]`
  - `order_accessibility`: `[1.20, 3.80]`
  - `accessibility_scale`: `[0.10, 0.75]`
  - `order_span`: `[0.55, 2.50]`
  - `balance_accessibility`: `[0.55, 1.0]`
  - `balance_scale`: `[0.45, 1.80]`
  - `redundancy_strength`: `[0.35, 1.40]`
  - `redundancy_curvature`: `[1.05, 2.20]`
  - `similarity_sensitivity`: `[2.0, 16.0]`
  - `compact_dissent`: `[1.5, 6.5]`
  - `compact_peak`: `[1.10, 1.60]`
  - `compact_width`: `[0.45, 0.90]`
  - `compactness_gain`: `[0.20, 1.50]`
  - `credibility_scale`: `[0.025, 0.16]`
  - `isolation_gain`: `[0.25, 1.50]`
  - `restoration_strength`: `[0.65, 0.97]`
  - `restoration_threshold`: `[2.25, 3.60]`
  - `restoration_slope`: `[3.5, 11.0]`
  - `balance_restoration`: `[0.45, 1.0]`
  - `restoration_credibility`: `[0.30, 0.70]`
  - `isolation_restoration`: `[0.15, 0.55]`
  - `anchor_threshold`: `[0.66, 0.82]`
  - `anchor_slope`: `[10.0, 35.0]`
  - `balance_precision_gain`: `[0.75, 1.60]`
  - `balance_precision_width`: `[0.45, 0.75]`
  - `beta`: `[1.3, 5.0]`
  - `lapse`: `[0.0, 0.13]`
  - `configuration_type`: `{0, 1, 2}`

**`predict(parameters, stimulus, history)`:**
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
