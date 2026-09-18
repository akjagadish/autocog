# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_11` — SURVIVED ✓

**Description:** Tagged Partial-Independence Coalition Coding proposes that people evaluate an agreeing coalition with three psychologically separable operations. First, they apply a highly selective duplicate tag to experts whose instructed reliabilities are exactly or nearly identical. Members carrying the same duplicate tag share evidential credit, and credit losses are summed across all duplicate groups rather than propagated through a similarity network. Second, people encode reliability contextually using a small ordinal vocabulary derived from the reliability levels present in the current experiment. A coalition receives a bounded corroboration bonus for representing multiple distinct ordinal codes. This operation counts represented codes directly and never merges experts through pairwise links, so a chain of locally similar reliabilities cannot collapse transitively into one source. Third, the resulting evidence undergoes generic power-law saturation as a function of nominal coalition size, with no special transition between two and three experts. Duplicate multiplicity, ordinal-code diversity, and coalition cardinality consequently make independent graded contributions. Direct reliability weighting is weak and contrastive, allowing a sufficiently large coalition to defeat the uniquely most reliable expert while limiting broad metric reliability-gap effects. Stable cue salience varies idiosyncratically across people but is exchangeable and centered across cue identities, producing heterogeneity without population-level onset, recency, closure, or spacing mechanisms. Because choices receive no correctness feedback, these representations remain stable across trials.

**Rationale:** This is a minimal two-range calibration of the accepted iteration-5 model. No equation, parameter name, or other mechanism is changed. The diversity-saturation range is reduced substantially, so almost all of the bounded diversity reward is delivered by the first additional contextual code and later codes add very little. The asymptotic diversity-bonus range is also narrowed and shifted below the upper mass of the previous range. Together, these edits strengthen the first-code increment relative to later increments without introducing a dyad/triad gate: unequal dyads can separate more clearly from equal dyads in Experiment 10, while triads and larger coalitions receive little additional diversity growth, constraining Experiments 12, 15, and 18. Duplicate tolerance, duplicate credit, rank reliance, coalition saturation, response sensitivity, lapse, and exchangeable salience remain exactly at the running-best settings to protect the accepted non-transitivity result and positional nulls.

**Parameters:**
  - `validities`: `validities`
  - `cue_salience_profile`: `[(-1.0, 1.0)] * n_features`
  - `ordinal_resolution`: `{2, 3}`
  - `rank_reliance`: `[0.002, 0.015]`
  - `rank_contrast_curvature`: `[0.55, 0.85]`
  - `salience_strength`: `[0.22, 0.46]`
  - `duplicate_tolerance`: `[0.006, 0.012]`
  - `duplicate_credit_curvature`: `[-0.12, 0.02]`
  - `diversity_bonus`: `[0.025, 0.045]`
  - `diversity_saturation`: `[0.18, 0.35]`
  - `coalition_saturation`: `[0.62, 0.90]`
  - `beta`: `[1.50, 1.90]`
  - `epsilon`: `[0.03, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tagged Partial-Independence Coalition Coding expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    ordinal_resolution = int(parameters["ordinal_resolution"])
    rank_reliance = float(parameters["rank_reliance"])
    rank_contrast_curvature = float(parameters["rank_contrast_curvature"])
    salience_strength = float(parameters["salience_strength"])
    duplicate_tolerance = float(parameters["duplicate_tolerance"])
    duplicate_credit_curvature = float(
        parameters["duplicate_credit_curvature"]
    )
    diversity_bonus = float(parameters["diversity_bonus"])
    diversity_saturation = float(parameters["diversity_saturation"])
    coalition_saturation = float(parameters["coalition_saturation"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Construct contextual ordinal reliability codes. Equal instructed
    # reliabilities always receive the same rank. Nonidentical levels are
    # assigned codes from their order in the experiment, not from a fixed
    # metric distance or a graph of pairwise similarities.
    unique_v, inverse = np.unique(v, return_inverse=True)
    n_levels = int(unique_v.size)
    if n_levels <= 1:
        level_contrast = np.zeros(1, dtype=np.float64)
        level_codes = np.zeros(1, dtype=np.int64)
    else:
        level_rank = np.arange(n_levels, dtype=np.float64) / float(n_levels - 1)
        signed_rank = 2.0 * level_rank - 1.0
        level_contrast = np.sign(signed_rank) * np.power(
            np.abs(signed_rank), rank_contrast_curvature
        )
        # Equal-frequency ordinal coding is contextual and non-metric.
        level_codes = np.floor(
            np.arange(n_levels, dtype=np.float64)
            * float(ordinal_resolution)
            / float(n_levels)
        ).astype(np.int64)
        level_codes = np.minimum(level_codes, ordinal_resolution - 1)

    cue_contrast = level_contrast[inverse]
    reliability_weight = np.exp(
        np.clip(rank_reliance * cue_contrast, -10.0, 10.0)
    )
    reliability_weight /= max(float(np.mean(reliability_weight)), 1e-12)

    # Stable salience is exchangeable across expert identities and centered
    # within each subject. It therefore allows individual configuration
    # preferences without imposing a common position gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)
    cue_codes = level_codes[inverse]

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0

        # Operation 1: exact/near-exact duplicate tagging. Sorted complete-link
        # groups require every member to lie within the narrow tag tolerance.
        # Thus a sequence of small pairwise gaps cannot merge transitively.
        order = idx[np.argsort(v[idx], kind="mergesort")]
        groups = []
        current = [int(order[0])]
        group_min = float(v[order[0]])

        for feature in order[1:]:
            feature = int(feature)
            if float(v[feature]) - group_min <= duplicate_tolerance:
                current.append(feature)
            else:
                groups.append(current)
                current = [feature]
                group_min = float(v[feature])
        groups.append(current)

        shared_credit_total = 0.0
        for group in groups:
            members = np.asarray(group, dtype=np.int64)
            multiplicity = int(members.size)
            if multiplicity == 1:
                shared_credit_total += float(cue_weights[members[0]])
            else:
                # Each duplicate group has its own concave credit account.
                # Summing these accounts makes penalties accumulate when a
                # coalition contains several separate duplicate groups.
                group_quality = float(np.mean(cue_weights[members]))
                group_credit = group_quality * (
                    float(multiplicity) ** duplicate_credit_curvature
                )
                shared_credit_total += group_credit

        # Operation 2: bounded diversity reward for the number of represented
        # contextual reliability codes. It is a coalition-level count, not a
        # sum of pairwise kernels, and therefore cannot grow quadratically.
        distinct_codes = int(np.unique(cue_codes[idx]).size)
        diversity_steps = float(max(distinct_codes - 1, 0))
        bounded_diversity = 1.0 - np.exp(
            -diversity_steps / max(diversity_saturation, 1e-12)
        )
        diversity_factor = 1.0 + diversity_bonus * bounded_diversity

        # Operation 3: generic nominal-size saturation. The same power law
        # applies at every cardinality, with no dyad/triad gate or release.
        size_normalizer = float(n) ** coalition_saturation
        signal = (
            shared_credit_total * diversity_factor
            / max(size_normalizer, 1e-12)
        )
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_9` — KILLED ✗

**Description:** Reliability-Space Credit Sharing proposes that people interpret agreement among experts through an inferred dependence network defined entirely in represented reliability space. Advertised validities have two related but distinct cognitive representations: a strongly compressed representation determines each cue's direct evidential weight, while a separately calibrated fixed-anchor representation determines inferred dependence. Within any option-supporting coalition, every expert shares informational credit with similarly represented experts according to a smooth, sharply localized similarity kernel. An expert surrounded by reliability-near neighbors therefore contributes less independent evidence, whereas reliability-distant experts contribute nearly undiluted evidence. The steep but continuous kernel concentrates redundancy around extremely similar reliability representations, producing a strong equality penalty without a broad dyadic reliability-gap gradient. Credit sharing applies at every coalition size, without an equality detector, dyadic/triadic switch, spatial grouping rule, or duplicate-pair-in-context operation. Because additional coalition members create multiple possible similarity relations, local dependence can accumulate nonlinearly in larger coalitions even when its effect on a single nonidentical pair is weak. Effective evidence is additionally subject to generic sublinear count growth. Stable subject-level dependence sensitivity, overlap curvature, compressed validity reliance, and exchangeable zero-centered cue salience generate individual differences without population-level serial-position effects. Choices arise from response sensitivity plus an independent lapse process, and representations remain fixed because no outcome feedback is provided.

**Rationale:** This is a parameter-only edit of the accepted iteration-6 model. The rejected superlinear-overlap transform is not retained: pairwise similarities still combine through the accepted additive local-overlap rule. The dependence coordinate is made less compressed by increasing validity_compression, while the single super-Gaussian core is narrowed. Consequently, genuinely nonzero reliability gaps separate more rapidly on the fixed-anchor coordinate, which should suppress the excessive broad dyadic gradient in Experiment 11 and reduce the modest overshoots in Experiments 12–14. Exact matches remain at unit similarity regardless of bandwidth. A small increase in overlap curvature therefore strengthens their smooth dependence penalty for Experiment 10 without increasing global dependence sensitivity. The dependence-sensitivity range is slightly narrowed to limit subject-level variance. Direct validity compression, direct reliance, count saturation, response sensitivity, lapse, and exchangeable salience are otherwise unchanged, protecting the accepted separation of direct weighting from dependence inference and the established positional nulls.

**Parameters:**
  - `validities`: `validities`
  - `cue_salience_profile`: `[(-1.0, 1.0)] * n_features`
  - `validity_compression`: `[0.30, 0.34]`
  - `direct_validity_compression`: `[0.08, 0.14]`
  - `validity_reliance`: `[0.008, 0.025]`
  - `salience_strength`: `[0.12, 0.28]`
  - `coalition_saturation`: `[0.64, 0.72]`
  - `dependence_sensitivity`: `[2.90, 3.30]`
  - `similarity_bandwidth`: `[0.0045, 0.0065]`
  - `overlap_curvature`: `[0.88, 0.96]`
  - `beta`: `[1.12, 1.38]`
  - `epsilon`: `[0.03, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Space Credit Sharing expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_sensitivity = float(parameters["dependence_sensitivity"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    overlap_curvature = float(parameters["overlap_curvature"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Dependence and direct cue weighting use separately compressed versions
    # of the same instructed validity. Both retain the fixed 0.75 reference,
    # so their scales remain invariant across experiments.
    v = np.clip(validities, 0.500001, 0.999999)
    log_diagnosticity = np.log(v / (1.0 - v))
    reference_diagnosticity = np.log(0.75 / 0.25)

    represented = np.power(
        np.maximum(log_diagnosticity, 1e-12), validity_compression
    )
    represented /= max(
        float(reference_diagnosticity ** validity_compression), 1e-12
    )

    direct_represented = np.power(
        np.maximum(log_diagnosticity, 1e-12),
        direct_validity_compression,
    )
    direct_represented /= max(
        float(reference_diagnosticity ** direct_validity_compression),
        1e-12,
    )

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * direct_represented
    )

    # Subject-specific salience is exchangeable over cue identities and is
    # centered within the display, precluding a population onset or recency
    # advantage while allowing stable individual configuration preferences.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coordinates = represented[idx]
        gaps = coordinates[:, None] - coordinates[None, :]

        # A smooth super-Gaussian concentrates inferred dependence around
        # extremely close reliability representations. It remains continuous
        # and cardinality-general, but decays faster than the previous
        # Gaussian outside the local overlap neighborhood.
        scaled_gaps = gaps / max(similarity_bandwidth, 1e-12)
        similarity = np.exp(-0.5 * np.power(scaled_gaps, 4.0))
        np.fill_diagonal(similarity, 0.0)
        local_overlap = np.sum(similarity, axis=1)

        # Divisive credit allocation approximates precision loss under
        # correlated sources. In a pair, each cue has only one possible
        # neighbor. In larger coalitions, multiple pairwise overlaps converge
        # on each member and create nonlinear redundancy without a size gate.
        independence_fraction = np.power(
            1.0 + dependence_sensitivity * local_overlap,
            -overlap_curvature,
        )
        effective_total = float(
            np.sum(cue_weights[idx] * independence_fraction)
        )

        # Generic count normalization makes corroboration sublinear even for
        # reliability-diverse coalitions. Dependence can further reduce the
        # marginal value of clustered additions but never deletes a cue.
        signal = effective_total / (float(n) ** coalition_saturation)
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_12` → slot 2 (via `new_theory`)

**Description:** Opponent-Relative Stratified Sampling Theory proposes that people compress ordered expert reliabilities into a small contextual vocabulary of reliability strata and jointly compare opposing coalitions by sampled evidence, stratum coverage, balance, and relative concentration. Each stratum has a limited number of representational slots, while a second capacity limit constrains total evidence maintained across strata. Coalition mass is partially normalized, allowing heterogeneous dyad, triad, and tetrad effects without a single global coalition-size power law. Repetition is not intrinsically penalized: its effect depends on the rival coalition's concentration, coverage, and size. This opponent-relative contrast is strongest for diagnostic dyads. When equal-sized dyads have matched duplicate status, their individual mean qualities are partially assimilated toward a pooled comparison standard, attenuating arbitrary ordinal-composition contrasts without eliminating reliability information. For equally large coalitions of at least four experts, a one-sided duplicate-versus-chain contrast is pooled through the same capacity-limited representation, preventing generic diversity from producing a chain advantage. Concentration comparisons remain available when both large coalitions contain repetition but differ in its degree. Weak ordinal reliability weighting and exchangeable centered cue salience preserve sensitivity to instructed reliability without metric-distance, chain, onset, recency, closure, or adjacency effects.

**Rationale:** This is an isolated minimal edit to the accepted iteration-4 candidate. The matched-dyad branch previously removed structural coverage differences but retained each coalition's full mean ordinal quality, allowing ordinal composition and exchangeable cue salience to create an excessive broad-gap contrast in Experiment 11. The new matched_dyad_quality_shrinkage parameter partially assimilates the two mean qualities toward their pooled mean before applying the unchanged common occupancy scale. Its range preserves a residual contrast for Experiment 19 rather than forcing exact indifference. Because the branch still requires equal-sized dyads with matched duplicate status, the edit does not alter Experiment 10's unequal-duplicate comparison, Experiment 15's dyad-versus-triad interaction, the successful distributed effects in Experiments 12–14, or the frozen large-coalition pooling that keeps Experiments 18 and 20 near chance. No metric-distance kernel, partition offset, positional process, joint-capacity revision, or additional large-coalition exception is introduced.

**Parameters:**
  - `validities`: `validities`
  - `cue_salience_profile`: `[(-1.0, 1.0)] * n_features`
  - `stratum_resolution`: `{2, 3, 4}`
  - `sampling_capacity`: `{1, 2, 3}`
  - `global_capacity`: `[2.2, 4.8]`
  - `occupancy_normalization`: `[0.48, 0.80]`
  - `ordinal_reliance`: `[0.005, 0.035]`
  - `salience_strength`: `[0.08, 0.24]`
  - `coverage_preference`: `[0.14, 0.38]`
  - `balance_preference`: `[0.08, 0.30]`
  - `redundancy_sensitivity`: `[0.65, 1.45]`
  - `rival_size_weight`: `[0.20, 0.55]`
  - `context_curvature`: `[0.70, 1.30]`
  - `matched_dyad_quality_shrinkage`: `[0.18, 0.38]`
  - `beta`: `[1.25, 1.70]`
  - `epsilon`: `[0.03, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Opponent-Relative Stratified Sampling expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    stratum_resolution = int(parameters["stratum_resolution"])
    sampling_capacity = int(parameters["sampling_capacity"])
    global_capacity = float(parameters["global_capacity"])
    occupancy_normalization = float(parameters["occupancy_normalization"])
    ordinal_reliance = float(parameters["ordinal_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coverage_preference = float(parameters["coverage_preference"])
    balance_preference = float(parameters["balance_preference"])
    redundancy_sensitivity = float(parameters["redundancy_sensitivity"])
    rival_size_weight = float(parameters["rival_size_weight"])
    context_curvature = float(parameters["context_curvature"])
    matched_dyad_quality_shrinkage = float(
        parameters["matched_dyad_quality_shrinkage"]
    )
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Contextual strata are formed from the ordering of distinct instructed
    # reliability levels. No metric bandwidth, pairwise similarity kernel, or
    # transitive similarity graph is used. Equal levels necessarily share a
    # stratum, while small and large nonzero gaps are treated alike whenever
    # they occupy the same ordinal portion of the reliability vocabulary.
    unique_v, inverse_levels = np.unique(v, return_inverse=True)
    n_levels = int(unique_v.size)
    actual_strata = max(1, min(stratum_resolution, n_levels))

    if n_levels == 1:
        level_strata = np.zeros(1, dtype=np.int64)
        ordinal_contrast = np.zeros(1, dtype=np.float64)
    else:
        level_strata = np.floor(
            np.arange(n_levels, dtype=np.float64)
            * float(actual_strata)
            / float(n_levels)
        ).astype(np.int64)
        level_strata = np.minimum(level_strata, actual_strata - 1)
        ordinal_position = (
            np.arange(n_levels, dtype=np.float64) / float(n_levels - 1)
        )
        ordinal_contrast = 2.0 * ordinal_position - 1.0

    cue_strata = level_strata[inverse_levels]

    # Reliability affects cue strength only through weak ordinal contrast.
    # Consequently, changing a gap from .003 to .0058 has no direct effect.
    reliability_weight = np.exp(
        np.clip(ordinal_reliance * ordinal_contrast[inverse_levels], -10.0, 10.0)
    )
    reliability_weight /= max(float(np.mean(reliability_weight)), 1e-12)

    # Stable cue salience is centered and exchangeable over cue identity.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def summarize(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return {
                "n": 0,
                "coverage": 0,
                "balance": 0.0,
                "duplicate_fraction": 0.0,
                "sampled_signal": 0.0,
                "mean_quality": 0.0,
            }

        strata = cue_strata[idx]
        counts = np.bincount(strata, minlength=actual_strata).astype(np.float64)
        occupied = counts > 0.0
        coverage = int(np.sum(occupied))

        # Each reliability stratum has a finite set of representational slots.
        # This is the expected number of occupied slots after placing the
        # stratum's members into those slots. It creates discrete diminishing
        # returns without a global coalition-size power law.
        cap = float(sampling_capacity)
        expected_slots = cap * (
            1.0 - np.power(1.0 - 1.0 / cap, counts)
        )

        stratum_signal = 0.0
        for s in range(actual_strata):
            members = idx[strata == s]
            if members.size == 0:
                continue
            quality = float(np.mean(cue_weights[members]))
            stratum_signal += quality * float(expected_slots[s])

        # Partial normalization lies between summing occupied evidence and
        # fully averaging it. Sampling-capacity heterogeneity can therefore
        # preserve different dyad, triad, and tetrad profiles.
        expected_total = max(float(np.sum(expected_slots)), 1e-12)
        normalized_signal = stratum_signal / (
            expected_total ** occupancy_normalization
        )

        # A second capacity limit reflects how much sampled evidence can be
        # maintained across all strata on a trial.
        sampled_signal = global_capacity * (
            1.0 - np.exp(-normalized_signal / max(global_capacity, 1e-12))
        )

        if coverage <= 1:
            balance = 0.0
        else:
            proportions = counts[occupied] / float(n)
            entropy = -float(np.sum(proportions * np.log(proportions)))
            balance = entropy / max(float(np.log(coverage)), 1e-12)

        # Repetition is represented categorically at exact instructed levels.
        # It is only a vulnerability passed to the joint comparison below and
        # is not deducted from this coalition's evidence intrinsically.
        level_counts = np.bincount(
            inverse_levels[idx], minlength=n_levels
        ).astype(np.float64)
        duplicate_excess = float(np.sum(np.maximum(level_counts - 1.0, 0.0)))
        duplicate_fraction = duplicate_excess / float(max(n - 1, 1))

        return {
            "n": n,
            "coverage": coverage,
            "balance": balance,
            "duplicate_fraction": duplicate_fraction,
            "sampled_signal": sampled_signal,
            "mean_quality": float(np.mean(cue_weights[idx])),
        }

    sa = summarize(a_support)
    sb = summarize(b_support)

    # Coverage is evaluated relative to the contextual strata that are
    # represented by either opposing coalition on this trial.
    union_indices = np.concatenate((a_support, b_support)).astype(np.int64)
    if union_indices.size == 0:
        union_coverage = 1
    else:
        union_coverage = max(
            int(np.unique(cue_strata[union_indices]).size), 1
        )

    def structural_value(summary):
        if summary["n"] == 0:
            return 0.0
        if union_coverage <= 1:
            coverage_fraction = 0.0
        else:
            coverage_fraction = float(summary["coverage"] - 1) / float(
                union_coverage - 1
            )
        multiplier = (
            1.0
            + coverage_preference * coverage_fraction
            + balance_preference * summary["balance"]
        )
        return float(summary["sampled_signal"] * multiplier)

    value_a = structural_value(sa)
    value_b = structural_value(sb)

    # When equal-sized dyads have the same categorical duplicate status,
    # ordinal-bin coverage is not treated as diagnostic. Both receive the
    # same pooled two-member occupancy scale. Their mean qualities are also
    # assimilated toward their joint mean, leaving only a reduced residual
    # ordinal/salience contrast rather than an arbitrary broad-gap effect.
    matched_dyads = (
        sa["n"] == 2
        and sb["n"] == 2
        and ((sa["duplicate_fraction"] > 0.0)
             == (sb["duplicate_fraction"] > 0.0))
    )
    if matched_dyads:
        cap = float(sampling_capacity)
        pooled_slots = cap * (
            1.0 - (1.0 - 1.0 / cap) ** 2.0
        )
        pooled_scale = pooled_slots / (
            max(pooled_slots, 1e-12) ** occupancy_normalization
        )
        pooled_quality = 0.5 * (
            sa["mean_quality"] + sb["mean_quality"]
        )
        quality_a = pooled_quality + matched_dyad_quality_shrinkage * (
            sa["mean_quality"] - pooled_quality
        )
        quality_b = pooled_quality + matched_dyad_quality_shrinkage * (
            sb["mean_quality"] - pooled_quality
        )
        value_a = global_capacity * (
            1.0 - np.exp(
                -quality_a * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )
        value_b = global_capacity * (
            1.0 - np.exp(
                -quality_b * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )

    # A diverse chain and an exact block are not treated as intrinsically
    # different sources when both coalitions exceed working-memory capacity.
    # Pooling only the one-sided duplicate contrast leaves comparisons between
    # two differently concentrated repeated coalitions available.
    large_one_sided_contrast = (
        sa["n"] == sb["n"]
        and sa["n"] >= 4
        and ((sa["duplicate_fraction"] > 0.0)
             != (sb["duplicate_fraction"] > 0.0))
    )
    if large_one_sided_contrast:
        cap = float(sampling_capacity)
        pooled_slots = cap * (
            1.0 - (1.0 - 1.0 / cap) ** float(sa["n"])
        )
        pooled_scale = pooled_slots / (
            max(pooled_slots, 1e-12) ** occupancy_normalization
        )
        value_a = global_capacity * (
            1.0 - np.exp(
                -sa["mean_quality"] * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )
        value_b = global_capacity * (
            1.0 - np.exp(
                -sb["mean_quality"] * pooled_scale
                / max(global_capacity, 1e-12)
            )
        )

    def contextual_vulnerability(focal, rival):
        if focal["n"] == 0 or focal["duplicate_fraction"] <= 0.0:
            return 0.0

        # One-sided duplicate-versus-chain contrasts are pooled at large
        # cardinalities, so they do not receive an additional vulnerability.
        if large_one_sided_contrast:
            return 0.0

        coverage_advantage = float(
            rival["coverage"] - focal["coverage"]
        ) / float(max(union_coverage, 1))
        size_advantage = float(rival["n"] - focal["n"]) / float(
            max(rival["n"], focal["n"], 1)
        )

        # A bounded direct comparison allows equality avoidance when an
        # equally large rival is less concentrated. It is moderately stronger
        # for dyads, where exact repetition is especially diagnostic.
        concentration_advantage = max(
            0.0,
            float(focal["duplicate_fraction"] - rival["duplicate_fraction"]),
        )
        bounded_concentration = 1.0 - np.exp(
            -3.0 * concentration_advantage
        )
        concentration_weight = (
            1.10 if focal["n"] == 2 and rival["n"] == 2 else 0.75
        )
        opponent_context = max(
            0.0,
            coverage_advantage
            + rival_size_weight * size_advantage
            + concentration_weight * bounded_concentration,
        )
        opponent_context = opponent_context ** context_curvature
        return float(focal["duplicate_fraction"] * opponent_context)

    vulnerability_a = contextual_vulnerability(sa, sb)
    vulnerability_b = contextual_vulnerability(sb, sa)

    adjusted_a = value_a * np.exp(
        -redundancy_sensitivity * vulnerability_a
    )
    adjusted_b = value_b * np.exp(
        -redundancy_sensitivity * vulnerability_b
    )
    net_a = float(adjusted_a - adjusted_b)

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
