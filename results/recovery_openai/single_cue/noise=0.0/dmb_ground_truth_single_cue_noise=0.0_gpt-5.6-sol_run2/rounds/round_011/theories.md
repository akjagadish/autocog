# Round 11 — Theories

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


### slot 2 — `pi_13` — KILLED ✗

**Description:** Adaptive Source-Resolution Compression with Allocated Source Quality proposes that observers represent an agreeing coalition as a probabilistic collection of latent information sources rather than as a tally of experts. Exact equality in communicated reliability is a discrete but non-deterministic common-source cue, whereas any nonzero reliability difference supports source individuation. Pairwise dependence posteriors determine both a continuous effective-source count and cue-specific uniqueness credits. These credits allocate the inferred source count across coalition members: a cue strongly dependent on its partners receives less ownership of source evidence, while an individuated cue retains more ownership. Coalition quality is therefore computed from the reliability and salience of the cues that own independent-source credit, rather than from the unadjusted arithmetic mean of all agreeing cues. The first inferred source is represented fully, while additional sources undergo subject-specific, smoothly saturating compression. Reliability-range and ordinal-stratum coverage provide a diversity benefit that decays with coalition load. The same quality-adjusted compressed source representation enters opponent-relative conflict, allowing several independent moderate cues to outweigh a highly reliable but redundant cue without introducing cardinality-specific rules. Stable heterogeneity in equality inference, compression capacity, diversity sensitivity, source-credit allocation, conflict weighting, and exchangeable cue salience produces graded population prevalence and preserves positional exchangeability.

**Rationale:** This is a localized edit to the accepted iteration-2 model. The scalar effective-source calculation, adaptive compression, diversity retention, heterogeneity ranges, exchangeable salience, and choice rule are preserved. The only substantive addition is cue-level source-credit allocation. Each retained pairwise dependence posterior is accumulated into a cue-specific dependence load; a smooth inverse-power transform converts that load into uniqueness, and the resulting credits are normalized to sum exactly to the accepted scalar effective-source count. Coalition quality is then the credit-weighted mean cue quality rather than the unadjusted arithmetic mean. This retains the original count magnitude while preserving information about which reliability labels own independent-source credit. The quality-adjusted compressed source code is also passed through the existing opponent-relative conflict equation, with no new residual or unbounded count branch. Consequently, dependence concentrated on a high-ranked cue can reduce that cue's effective ownership, allowing independent moderate cues to overturn it in equal-cardinality conflicts such as Experiment 2, while normalization by effective-source count limits renewed cardinality leverage in Experiments 1 and 6. Because the edit leaves coverage and load retention unchanged, it is less likely to repeat the rejected global changes that weakened Experiments 12–17 or destabilized the near-chance large-coalition results in Experiments 18 and 20.

**Parameters:**
  - `validities`: `validities`
  - `cue_salience_profile`: `[(-1.0, 1.0)] * n_features`
  - `source_dependence_prior`: `[-2.20, 0.10]`
  - `equality_evidence`: `[1.80, 5.20]`
  - `nonzero_separation`: `[3.20, 5.80]`
  - `separation_floor`: `[0.58, 0.78]`
  - `positive_gap_rank_gain`: `[0.03, 0.14]`
  - `compression_capacity`: `[1.00, 2.80]`
  - `load_individuation`: `[0.18, 0.55]`
  - `source_signal_curvature`: `[0.40, 0.66]`
  - `source_allocation_curvature`: `[0.70, 1.60]`
  - `rank_reliance`: `[0.008, 0.035]`
  - `diversity_sensitivity`: `[0.12, 0.85]`
  - `coverage_mix`: `[0.45, 0.75]`
  - `salience_strength`: `[0.12, 0.34]`
  - `conflict_gain`: `[0.04, 0.16]`
  - `conflict_scale`: `[0.55, 1.10]`
  - `best_cue_tradeoff`: `[0.05, 0.20]`
  - `beta`: `[1.35, 1.90]`
  - `epsilon`: `[0.025, 0.070]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Adaptive Source-Resolution Compression expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=np.float64)
    if v.ndim != 1 or v.size != n_features:
        raise ValueError(
            f"validities length {v.size} != n_features {n_features}."
        )
    v = np.clip(v, 0.500001, 0.999999)

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    source_dependence_prior = float(parameters["source_dependence_prior"])
    equality_evidence = float(parameters["equality_evidence"])
    nonzero_separation = float(parameters["nonzero_separation"])
    separation_floor = float(parameters["separation_floor"])
    positive_gap_rank_gain = float(parameters["positive_gap_rank_gain"])
    compression_capacity = float(parameters["compression_capacity"])
    load_individuation = float(parameters["load_individuation"])
    source_signal_curvature = float(parameters["source_signal_curvature"])
    source_allocation_curvature = float(
        parameters["source_allocation_curvature"]
    )
    rank_reliance = float(parameters["rank_reliance"])
    diversity_sensitivity = float(parameters["diversity_sensitivity"])
    coverage_mix = float(parameters["coverage_mix"])
    salience_strength = float(parameters["salience_strength"])
    conflict_gain = float(parameters["conflict_gain"])
    conflict_scale = float(parameters["conflict_scale"])
    best_cue_tradeoff = float(parameters["best_cue_tradeoff"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Contextual ordinal reliability representation. Ties share a rank, and
    # cue identity does not enter the population-level weighting rule.
    unique_v, inverse, counts = np.unique(
        v, return_inverse=True, return_counts=True
    )
    n_levels = int(unique_v.size)
    if n_levels == 1:
        level_rank = np.full(1, 0.5, dtype=np.float64)
    else:
        cumulative = np.cumsum(counts).astype(np.float64)
        midranks = cumulative - 0.5 * counts
        level_rank = midranks / float(n_features)

    cue_rank = level_rank[inverse]
    rank_contrast = 2.0 * (cue_rank - float(np.mean(cue_rank)))
    reliability_quality = np.exp(
        np.clip(rank_reliance * rank_contrast, -10.0, 10.0)
    )

    # Stable salience is centered and exchangeable across identities. It can
    # create individual configuration preferences without a common onset,
    # spacing, closure, or recency gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_quality = reliability_quality * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    # Build the task-relative distribution of strictly positive reliability
    # differences. Equality remains a separate probabilistic cue.
    positive_gaps = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            gap = abs(float(v[i] - v[j]))
            if gap > 0.0:
                positive_gaps.append(gap)
    positive_gaps = np.sort(np.asarray(positive_gaps, dtype=np.float64))

    task_range = float(np.max(v) - np.min(v))
    task_rank_sd = float(np.std(cue_rank))

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_representation(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0, 0.0, 0.0

        dependence_load = np.zeros(n, dtype=np.float64)

        if n == 1:
            effective_sources = 1.0
        else:
            dependence_sum = 0.0

            for ii in range(n):
                for jj in range(ii + 1, n):
                    gap = abs(float(v[idx[ii]] - v[idx[jj]]))

                    if gap == 0.0:
                        # Equality is strong but nonabsolute, with broad
                        # subject-level variation in its interpretation.
                        logit_dependence = (
                            source_dependence_prior + equality_evidence
                        )
                    else:
                        # Positive gaps primarily share one separate-source
                        # category. Gap rank supplies only a small contextual
                        # modulation, avoiding a broad metric gradient.
                        if positive_gaps.size == 0:
                            gap_percentile = 1.0
                        else:
                            gap_percentile = float(
                                np.searchsorted(
                                    positive_gaps, gap, side="right"
                                )
                            ) / float(positive_gaps.size)
                        separation = np.clip(
                            separation_floor
                            + positive_gap_rank_gain * gap_percentile,
                            0.0,
                            1.0,
                        )
                        logit_dependence = (
                            source_dependence_prior
                            - nonzero_separation * separation
                        )

                    dependence = 1.0 / (
                        1.0 + np.exp(-np.clip(logit_dependence, -50.0, 50.0))
                    )

                    # Dependence information declines gradually with load.
                    # The reduced decline preserves equality effects when a
                    # dyad becomes part of a triad without introducing a gate.
                    load = float(max(n - 2, 0))
                    retained_dependence = dependence * np.exp(
                        -load_individuation
                        * load
                        / max(compression_capacity, 1e-12)
                    )
                    dependence_sum += retained_dependence
                    dependence_load[ii] += retained_dependence
                    dependence_load[jj] += retained_dependence

            mean_dependence_load = 2.0 * dependence_sum / float(n)
            effective_sources = float(n) / (1.0 + mean_dependence_load)
            effective_sources = float(
                np.clip(effective_sources, 1.0, float(n))
            )

        # Allocate the accepted scalar source count across individual cues.
        # Dependence lowers a cue's ownership of independent-source evidence.
        # Normalization preserves the original effective-source magnitude, so
        # allocation changes source quality without adding a second count term.
        uniqueness = np.power(
            1.0 + dependence_load,
            -source_allocation_curvature,
        )
        uniqueness_total = max(float(np.sum(uniqueness)), 1e-12)
        source_credits = uniqueness * effective_sources / uniqueness_total
        allocated_quality = float(
            np.dot(source_credits, cue_quality[idx])
            / max(effective_sources, 1e-12)
        )

        # The first inferred source is fully represented. Additional sources
        # are capacity-limited and then compressed by a common smooth
        # curvature. This sharply reduces raw-count leverage without a
        # cardinality-specific rule.
        additional_sources = compression_capacity * (
            1.0
            - np.exp(
                -(effective_sources - 1.0)
                / max(compression_capacity, 1e-12)
            )
        )
        compressed_sources = 1.0 + additional_sources
        source_code = compressed_sources ** source_signal_curvature
        quality_adjusted_source_code = source_code * allocated_quality

        coalition_v = v[idx]
        coalition_ranks = cue_rank[idx]

        if n <= 1 or task_range <= 1e-12:
            range_coverage = 0.0
        else:
            range_coverage = float(
                (np.max(coalition_v) - np.min(coalition_v)) / task_range
            )

        if n <= 1 or task_rank_sd <= 1e-12:
            stratum_coverage = 0.0
        else:
            stratum_coverage = float(
                np.clip(
                    np.std(coalition_ranks) / task_rank_sd,
                    0.0,
                    1.0,
                )
            )

        coverage = (
            coverage_mix * range_coverage
            + (1.0 - coverage_mix) * stratum_coverage
        )
        coverage = float(np.clip(coverage, 0.0, 1.0))

        # Diversity is available at low load but decays increasingly strongly
        # for tetrads and larger coalitions through one continuous function.
        load_retention = np.exp(
            -float(max(n - 1, 0))
            / max(compression_capacity, 1e-12)
        )
        diversity_factor = 1.0 + (
            diversity_sensitivity * coverage * load_retention
        )

        best_quality = float(np.max(cue_quality[idx]))
        signal = quality_adjusted_source_code * diversity_factor
        return (
            float(signal),
            float(quality_adjusted_source_code),
            best_quality,
        )

    signal_a, sources_a, best_a = coalition_representation(a_support)
    signal_b, sources_b, best_b = coalition_representation(b_support)

    source_difference = sources_a - sources_b
    best_difference = best_a - best_b

    # Conflict arbitration uses the same quality-adjusted, capacity-limited
    # source code as coalition evidence. Thus neither raw cardinality nor an
    # unallocated arithmetic mean can bypass probabilistic source inference.
    conflict_input = (
        source_difference + best_cue_tradeoff * best_difference
    ) / max(conflict_scale, 1e-12)
    relational_conflict = conflict_gain * np.tanh(conflict_input)

    net_a = (signal_a - signal_b) + relational_conflict

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
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

### `pi_14` → slot 2 (via `new_theory`)

**Description:** Coherence-Constrained Reliability-Strata Inference proposes that observers interpret an agreeing coalition by inferring a compact posterior distribution over latent reliability strata. Experts are assigned continuously rather than categorically to one through four possible source strata. Exact or virtually exact experts have a high probability of sharing a source and therefore incur accumulating redundancy. A small but nonzero reliability difference instead provides maximal corroboration because it supports distinguishable sources that remain mutually coherent. Larger separations progressively lose consensus coherence, producing a Goldilocks-shaped relation between reliability gap and coalition value. Coalition structure also matters: a small, balanced collection of compact strata supported by multiple experts is easier to compress and receives greater confidence than either one undifferentiated duplicate mass or many weakly supported singleton levels. Replication protection is granted only when posterior cluster mass is concentrated tightly around a stratum center; diffuse chains cannot obtain this protection merely through broad coassignment. For coalitions beyond triadic capacity, independence credit is pooled at the latent-stratum level and smoothly limited by the strongest compact, balanced, supported relation between distinguishable strata. Thus a chain cannot accumulate corroboration from every locally narrow expert pair. A dyad can progress from redundant at equality, to coherently corroborative at a narrow gap, and back toward weak consensus at a broader gap. A replicated two-stratum tetrad can outperform four unsupported reliability levels, although each compact stratum's excess membership adds a separate redundancy cost. Instructed reliability has a weak but consequential bounded influence after compression. Coalition magnitude saturates smoothly under finite capacity, with a modest corroborative release at the two-to-three transition that decays beyond three. Exchangeable idiosyncratic cue salience introduces individual variation without population-level position, onset, closure, adjacency, or recency effects.

**Rationale:** This is an isolated parameter-range edit to the accepted iteration-6 model. The only change is increasing count_curvature from [0.34, 0.74] to [0.40, 0.82]. This modestly strengthens the smooth marginal evidence supplied by a supported third cue, targeting the underweighted majorities in Experiments 1, 2, 3, and especially the three-versus-two boundary in Experiment 6. The finite-capacity formula itself is unchanged, so evidence still saturates smoothly and the accepted beyond-triad stratum-link constraint remains intact. All dyadic clustering, direct narrow-gap resolution, redundancy, coherence, compact-replication, salience, and choice-noise mechanisms are unchanged, protecting the strong Experiment 22 result and the positional null effects. No rejected partition-wide consensus, coverage-weighted link, global-span gate, post-clustering quality bonus, duplicate-stratum surcharge, or widened triad-transition range is reintroduced.

**Parameters:**
  - `validities`: `validities`
  - `cue_salience_profile`: `[(-1.0, 1.0)] * n_features`
  - `validity_compression`: `[0.08, 0.18]`
  - `validity_reliance`: `[0.020, 0.065]`
  - `salience_strength`: `[0.10, 0.28]`
  - `redundancy_width`: `[0.0015, 0.0045]`
  - `redundancy_cost`: `[0.70, 1.35]`
  - `compact_excess_cost`: `[0.06, 0.20]`
  - `corroboration_peak`: `[0.004, 0.007]`
  - `corroboration_width`: `[0.003, 0.007]`
  - `independence_benefit`: `[0.18, 0.55]`
  - `coherence_width`: `[0.012, 0.040]`
  - `coherence_cost`: `[0.28, 0.72]`
  - `cluster_width`: `[0.004, 0.014]`
  - `cluster_complexity`: `[0.10, 0.28]`
  - `cluster_temperature`: `[0.45, 0.90]`
  - `support_threshold`: `[1.35, 1.75]`
  - `support_slope`: `[4.0, 8.0]`
  - `cluster_support_bonus`: `[0.30, 0.82]`
  - `capacity_saturation`: `[0.90, 3.10]`
  - `count_curvature`: `[0.40, 0.82]`
  - `triad_transition_gain`: `[0.04, 0.24]`
  - `beta`: `[1.25, 1.75]`
  - `epsilon`: `[0.025, 0.070]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Coherence-Constrained Reliability-Strata Inference expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=np.float64)
    if v.ndim != 1 or v.size != n_features:
        raise ValueError(
            f"validities length {v.size} != n_features {n_features}."
        )
    v = np.clip(v, 0.500001, 0.999999)

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    redundancy_width = float(parameters["redundancy_width"])
    redundancy_cost = float(parameters["redundancy_cost"])
    compact_excess_cost = float(parameters["compact_excess_cost"])
    corroboration_peak = float(parameters["corroboration_peak"])
    corroboration_width = float(parameters["corroboration_width"])
    independence_benefit = float(parameters["independence_benefit"])
    coherence_width = float(parameters["coherence_width"])
    coherence_cost = float(parameters["coherence_cost"])
    cluster_width = float(parameters["cluster_width"])
    cluster_complexity = float(parameters["cluster_complexity"])
    cluster_temperature = float(parameters["cluster_temperature"])
    support_threshold = float(parameters["support_threshold"])
    support_slope = float(parameters["support_slope"])
    cluster_support_bonus = float(parameters["cluster_support_bonus"])
    capacity_saturation = float(parameters["capacity_saturation"])
    count_curvature = float(parameters["count_curvature"])
    triad_transition_gain = float(parameters["triad_transition_gain"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated reliability is used, but diagnosticity differences are
    # strongly compressed so that source organization can dominate isolated
    # numerical reliability advantages.
    diagnosticity = np.log(v / (1.0 - v))
    represented = np.power(
        np.maximum(diagnosticity, 1e-12), validity_compression
    )
    represented /= max(float(np.mean(represented)), 1e-12)
    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented
    )

    # The salience profile is exchangeable across expert identities and is
    # centered within subject. It therefore adds stable heterogeneity without
    # imposing a common serial-position or spatial-organization gradient.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)
    cue_quality = reliability_weight * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def soft_partition(x, k):
        """Deterministic soft clustering for one candidate stratum count."""
        n = int(x.size)
        if k == 1:
            assignments = np.ones((n, 1), dtype=np.float64)
            centers = np.array([float(np.mean(x))], dtype=np.float64)
            return assignments, centers

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax - xmin <= 1e-15:
            centers = np.full(k, xmin, dtype=np.float64)
        else:
            quantiles = np.linspace(0.0, 1.0, k)
            centers = np.quantile(x, quantiles).astype(np.float64)

        assignments = np.full((n, k), 1.0 / float(k), dtype=np.float64)
        for _ in range(16):
            squared = (x[:, None] - centers[None, :]) ** 2
            logits = -0.5 * squared / max(cluster_width ** 2, 1e-12)
            logits -= np.max(logits, axis=1, keepdims=True)
            assignments = np.exp(logits)
            assignments /= np.maximum(
                np.sum(assignments, axis=1, keepdims=True), 1e-12
            )
            masses = np.sum(assignments, axis=0)
            updated = np.sum(assignments * x[:, None], axis=0) / np.maximum(
                masses, 1e-12
            )
            centers = 0.65 * centers + 0.35 * updated

        order = np.argsort(centers)
        return assignments[:, order], centers[order]

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_quality[idx[0]])

        x = v[idx]
        qualities = cue_quality[idx]

        # Smooth Bayesian model comparison over a small number of reliability
        # strata. The likelihood rewards fit, while the complexity term makes
        # unsupported singleton strata expensive. When all values coincide,
        # the resolution gate strongly favors a single latent source.
        max_k = int(min(4, n))
        models = []
        model_scores = []
        observed_range = float(np.max(x) - np.min(x))
        resolution_gate = 1.0 - np.exp(
            -0.5 * (observed_range / max(cluster_width, 1e-12)) ** 2
        )

        for k in range(1, max_k + 1):
            assignments, centers = soft_partition(x, k)
            masses = np.sum(assignments, axis=0)
            residual = np.sum(
                assignments * (x[:, None] - centers[None, :]) ** 2
            )
            fit_cost = residual / max(
                float(n) * cluster_width ** 2, 1e-12
            )

            support = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        support_slope * (masses - support_threshold),
                        -50.0,
                        50.0,
                    )
                )
            )
            unsupported_cost = float(np.mean(1.0 - support))
            extra_strata = float(k - 1)
            resolution_penalty = (
                cluster_complexity
                * extra_strata
                / max(0.12 + resolution_gate, 1e-12)
            )
            score = -fit_cost - resolution_penalty - 0.35 * unsupported_cost
            models.append((assignments, centers, masses, support))
            model_scores.append(score)

        model_scores = np.asarray(model_scores, dtype=np.float64)
        posterior_logits = model_scores / max(cluster_temperature, 1e-12)
        posterior_logits -= np.max(posterior_logits)
        posterior = np.exp(posterior_logits)
        posterior /= max(float(np.sum(posterior)), 1e-12)

        # Posterior coassignment is a continuous estimate that two experts
        # derive from the same latent source.
        coassignment = np.zeros((n, n), dtype=np.float64)
        compressibility = 0.0
        expected_strata = 0.0
        compact_excess = 0.0
        stratum_link_credit = 0.0
        for probability, model in zip(posterior, models):
            assignments, centers, masses, support = model
            k = int(assignments.shape[1])
            coassignment += probability * (assignments @ assignments.T)
            expected_strata += probability * float(k)

            # Replication requires both excess posterior mass and low
            # within-stratum variance. Broad assignments over a smooth chain
            # therefore do not count as replicated support.
            within_variance = np.sum(
                assignments * (x[:, None] - centers[None, :]) ** 2,
                axis=0,
            ) / np.maximum(masses, 1e-12)
            concentration = np.exp(
                -0.5
                * within_variance
                / max(redundancy_width ** 2, 1e-12)
            )
            model_compact_excess = float(
                np.sum(
                    np.maximum(masses - 1.0, 0.0)
                    * support
                    * concentration
                )
                / float(n)
            )
            compact_excess += probability * model_compact_excess

            if k > 1:
                proportions = masses / max(float(np.sum(masses)), 1e-12)
                entropy = -float(
                    np.sum(proportions * np.log(np.maximum(proportions, 1e-12)))
                ) / max(np.log(float(k)), 1e-12)
                supported_mass = float(
                    np.sum(proportions * support * concentration)
                )
                compactness = 1.0 - float(k - 1) / float(max(n, 2))
                model_compressibility = (
                    entropy
                    * supported_mass
                    * max(compactness, 0.0)
                    * (float(k - 1) / float(k))
                )
                compressibility += probability * model_compressibility

                # Independence is represented once between inferred strata,
                # rather than once for every expert pair. A link receives
                # credit only when both strata are compact and supported, and
                # when their posterior masses are reasonably balanced. Taking
                # the strongest link is a smooth capacity cap: a reliability
                # chain cannot collect several rewards from adjacent gaps.
                model_links = []
                for ci in range(k):
                    for cj in range(ci + 1, k):
                        center_gap = abs(float(centers[ci] - centers[cj]))
                        center_resolution = 1.0 - np.exp(
                            -0.5
                            * (
                                center_gap
                                / max(redundancy_width, 1e-12)
                            ) ** 2
                        )
                        center_goldilocks = np.exp(
                            -0.5
                            * (
                                (center_gap - corroboration_peak)
                                / max(corroboration_width, 1e-12)
                            ) ** 2
                        )
                        balanced_mass = (
                            2.0 * min(float(masses[ci]), float(masses[cj]))
                            / max(float(n), 1e-12)
                        )
                        compact_support = np.sqrt(
                            max(float(support[ci] * support[cj]), 0.0)
                            * max(
                                float(concentration[ci] * concentration[cj]),
                                0.0,
                            )
                        )
                        model_links.append(
                            center_resolution
                            * center_goldilocks
                            * np.clip(balanced_mass, 0.0, 1.0)
                            * compact_support
                        )
                if model_links:
                    stratum_link_credit += probability * float(
                        np.max(model_links)
                    )

        # Pairwise source evidence has a Goldilocks shape. Equality implies
        # common-source redundancy; a resolvable narrow gap is corroborative;
        # and only the tail beyond that narrow region loses coherence.
        redundant_load = 0.0
        corroborative_load = 0.0
        broad_dispersion = 0.0
        pair_count = 0.0

        tail_scale = max(0.5 * corroboration_width, 1e-12)
        tail_onset = corroboration_peak + corroboration_width
        tail_floor = 1.0 / (
            1.0 + np.exp(np.clip(tail_onset / tail_scale, -50.0, 50.0))
        )

        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(float(x[i] - x[j]))
                exact_similarity = np.exp(
                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2
                )
                redundant_load += coassignment[i, j] * exact_similarity

                goldilocks = np.exp(
                    -0.5
                    * (
                        (gap - corroboration_peak)
                        / max(corroboration_width, 1e-12)
                    ) ** 2
                )
                direct_resolution = 1.0 - np.exp(
                    -0.5 * (gap / max(redundancy_width, 1e-12)) ** 2
                )
                corroborative_load += direct_resolution * goldilocks

                tail = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            (gap - tail_onset) / tail_scale,
                            -50.0,
                            50.0,
                        )
                    )
                )
                tail = np.clip(
                    (tail - tail_floor) / max(1.0 - tail_floor, 1e-12),
                    0.0,
                    1.0,
                )
                broad_dispersion += tail
                pair_count += 1.0

        pair_count = max(pair_count, 1.0)
        mean_corroboration = corroborative_load / pair_count
        mean_dispersion = broad_dispersion / pair_count

        # Dyads and triads retain the accepted pairwise Goldilocks relation.
        # Beyond triadic capacity, corroboration is increasingly constrained
        # by the single best compact and balanced link between latent strata.
        # This local aggregation correction leaves all other coalition terms
        # unchanged and avoids a global span penalty.
        higher_order_load = float(max(n - 3, 0))
        stratum_pooling = 1.0 - np.exp(-higher_order_load)
        link_limited_corroboration = min(
            mean_corroboration, stratum_link_credit
        )
        mean_corroboration = (
            (1.0 - stratum_pooling) * mean_corroboration
            + stratum_pooling * link_limited_corroboration
        )

        # Only compact, genuinely replicated strata protect a coalition from
        # dispersion. Posterior overlap by itself cannot shield a smooth chain.
        replication_guard = 1.0 / (1.0 + 1.5 * compact_excess)
        contextual_advantage = max(
            float(np.mean(represented[idx])) - 1.0, 0.0
        )
        quality_guard = np.exp(-2.0 * contextual_advantage)
        mean_dispersion *= replication_guard * quality_guard

        # Redundancy accumulates with the number of close within-source pairs.
        # Compact excess membership adds a smaller incremental cost, allowing
        # a second supported duplicate group to remain consequential.
        redundancy_per_member = redundant_load / float(n)

        additional = capacity_saturation * (
            1.0
            - np.exp(
                -float(n - 1) / max(capacity_saturation, 1e-12)
            )
        )
        represented_count = (1.0 + additional) ** count_curvature

        # A smooth two-to-three corroborative release supports heterogeneous
        # three-versus-two judgments but decays beyond three under capacity.
        triad_transition = 1.0 / (
            1.0 + np.exp(-np.clip((float(n) - 2.5) / 0.18, -50.0, 50.0))
        )
        beyond_three_decay = np.exp(
            -float(max(n - 3, 0)) / max(capacity_saturation, 1e-12)
        )
        represented_count *= np.exp(
            triad_transition_gain * triad_transition * beyond_three_decay
        )

        # Source-credit weighting gives less influence to cues heavily
        # coassigned with coalition partners without introducing cue position.
        dependence_load = np.sum(coassignment, axis=1) - 1.0
        uniqueness = 1.0 / np.maximum(1.0 + dependence_load, 1e-12)
        quality_weights = 0.55 + 0.45 * uniqueness / max(
            float(np.mean(uniqueness)), 1e-12
        )
        coalition_quality = float(
            np.sum(quality_weights * qualities)
            / max(float(np.sum(quality_weights)), 1e-12)
        )

        structural_log_value = (
            -redundancy_cost * redundancy_per_member
            -compact_excess_cost * compact_excess
            + independence_benefit * mean_corroboration
            - coherence_cost * mean_dispersion
            + cluster_support_bonus * compressibility
        )
        structural_factor = np.exp(
            np.clip(structural_log_value, -8.0, 8.0)
        )

        # Expected strata mildly calibrate confidence but are capacity-limited;
        # they do not create an unbounded second count term.
        source_confidence = 1.0 + 0.06 * (
            1.0
            - np.exp(
                -(expected_strata - 1.0)
                / max(capacity_saturation, 1e-12)
            )
        )

        signal = (
            represented_count
            * coalition_quality
            * structural_factor
            * source_confidence
        )
        return float(max(signal, 0.0))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
    net_a = evidence_a - evidence_b

    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= max(float(np.sum(probs)), 1e-12)

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
