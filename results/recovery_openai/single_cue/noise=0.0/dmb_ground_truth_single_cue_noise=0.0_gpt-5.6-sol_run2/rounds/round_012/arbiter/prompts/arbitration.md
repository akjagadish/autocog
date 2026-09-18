# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_11" and "pi_14") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_11" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_14" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_11
Tagged Partial-Independence Coalition Coding proposes that people evaluate an agreeing coalition with three psychologically separable operations. First, they apply a highly selective duplicate tag to experts whose instructed reliabilities are exactly or nearly identical. Members carrying the same duplicate tag share evidential credit, and credit losses are summed across all duplicate groups rather than propagated through a similarity network. Second, people encode reliability contextually using a small ordinal vocabulary derived from the reliability levels present in the current experiment. A coalition receives a bounded corroboration bonus for representing multiple distinct ordinal codes. This operation counts represented codes directly and never merges experts through pairwise links, so a chain of locally similar reliabilities cannot collapse transitively into one source. Third, the resulting evidence undergoes generic power-law saturation as a function of nominal coalition size, with no special transition between two and three experts. Duplicate multiplicity, ordinal-code diversity, and coalition cardinality consequently make independent graded contributions. Direct reliability weighting is weak and contrastive, allowing a sufficiently large coalition to defeat the uniquely most reliable expert while limiting broad metric reliability-gap effects. Stable cue salience varies idiosyncratically across people but is exchangeable and centered across cue identities, producing heterogeneity without population-level onset, recency, closure, or spacing mechanisms. Because choices receive no correctness feedback, these representations remain stable across trials.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_14
Coherence-Constrained Reliability-Strata Inference proposes that observers interpret an agreeing coalition by inferring a compact posterior distribution over latent reliability strata. Experts are assigned continuously rather than categorically to one through four possible source strata. Exact or virtually exact experts have a high probability of sharing a source and therefore incur accumulating redundancy. A small but nonzero reliability difference instead provides maximal corroboration because it supports distinguishable sources that remain mutually coherent. Larger separations progressively lose consensus coherence, producing a Goldilocks-shaped relation between reliability gap and coalition value. Coalition structure also matters: a small, balanced collection of compact strata supported by multiple experts is easier to compress and receives greater confidence than either one undifferentiated duplicate mass or many weakly supported singleton levels. Replication protection is granted only when posterior cluster mass is concentrated tightly around a stratum center; diffuse chains cannot obtain this protection merely through broad coassignment. For coalitions beyond triadic capacity, independence credit is pooled at the latent-stratum level and smoothly limited by the strongest compact, balanced, supported relation between distinguishable strata. Thus a chain cannot accumulate corroboration from every locally narrow expert pair. A dyad can progress from redundant at equality, to coherently corroborative at a narrow gap, and back toward weak consensus at a broader gap. A replicated two-stratum tetrad can outperform four unsupported reliability levels, although each compact stratum's excess membership adds a separate redundancy cost. Instructed reliability has a weak but consequential bounded influence after compression. Coalition magnitude saturates smoothly under finite capacity, with a modest corroborative release at the two-to-three transition that decays beyond three. Exchangeable idiosyncratic cue salience introduces individual variation without population-level position, onset, closure, adjacency, or recency effects.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_11)

### DESIGN
**Validities (n_features=16):** [0.746, 0.748, 0.752, 0.754, 0.746, 0.748, 0.752, 0.754, 0.55, 0.68, 0.82, 0.95, 0.55, 0.68, 0.82, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This 16-expert design creates a sign-reversing test of independent duplicate-group accounting versus coherence-constrained latent-strata inference. Each compact coalition has validities (0.746, 0.748, 0.752, 0.754): it contains two tightly supported reliability neighborhoods centered near 0.747 and 0.753, with a center gap close to the competing theory's Goldilocks corroboration peak. Each dispersed coalition has validities (0.55, 0.68, 0.82, 0.95): it has the same cardinality and mean validity (0.75), but consists of four widely separated, unsupported levels. Under Tagged Partial-Independence Coalition Coding, the compact coalition is reduced to at most two complete-link duplicate groups and, for larger duplicate tolerances, to one group; the dispersed coalition always contributes four singleton credit accounts. Both coalition types represent the same number of contextual ordinal codes under ordinal resolution 2 or 3, and nominal-size saturation is identical. The advocated theory therefore predicts a strong preference for the dispersed coalition. Coherence-Constrained Reliability-Strata Inference predicts the opposite: the compact coalition supplies a small, balanced, multiply supported set of coherent strata with several gaps near the corroborative region, whereas the dispersed coalition incurs broad-consensus loss and has only weakly supported singleton strata. The critical statistic is mirror-corrected compact-coalition choice: substantially below chance under the advocated theory but above chance under the competitor. Two instances of each topology are fully crossed to average over stable cue salience, and exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This is the prevalence of subjects whose mirror-corrected choices show a majority preference for the dispersed coalition. Compact-coalition choices are averaged separately when compact is A and when compact is B, then those rates are equally averaged, eliminating stable response-side bias. The advocated theory predicts a robust dispersed majority because four singleton credit accounts dominate the duplicate-tagged compact coalition; the competitor predicts the reverse because the compact, replicated reliability strata are more coherent. Thresholding each subject's estimate after aggregating all 96 trials converts this sign-reversing prediction into a stable subject-level classification, reducing sensitivity to binomial choice noise and idiosyncratic cue salience while remaining entirely response-dependent.

Source:
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        compact_sides = []
        compact_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_support = np.flatnonzero(a > b)
            b_support = np.flatnonzero(b > a)

            a_is_compact = a_support.size > 0 and np.all(a_support < 8)
            b_is_compact = b_support.size > 0 and np.all(b_support < 8)
            if a_is_compact == b_is_compact:
                continue

            compact_side = 0 if a_is_compact else 1
            compact_sides.append(compact_side)
            compact_choices.append(int(row["response"]) == compact_side)

        if len(compact_choices) == 0:
            return 0.5

        compact_sides = np.asarray(compact_sides, dtype=int)
        compact_choices = np.asarray(compact_choices, dtype=float)

        side_rates = []
        for side in (0, 1):
            mask = compact_sides == side
            if np.any(mask):
                side_rates.append(float(np.mean(compact_choices[mask])))

        compact_rate = float(np.mean(side_rates))
        dispersed_rate = 1.0 - compact_rate

        if dispersed_rate > 0.5:
            return 1.0
        if dispersed_rate < 0.5:
            return 0.0
        return 0.5

    if len(data) == 0:
        return 0.5

    scores = [subject_score(df) for _, df in data.groupby("subject_id", sort=False)]
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_11 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_14 (simulated): 0.4100 (var=0.2369)
- Observed on real data: 0.9200 (var=0.0736)

## EXPERIMENT 2 (proposed by pi_14)

### DESIGN
**Validities (n_features=22):** [0.747, 0.747, 0.753, 0.753, 0.747, 0.747, 0.753, 0.753, 0.738, 0.746, 0.754, 0.762, 0.729, 0.743, 0.757, 0.771, 0.72, 0.74, 0.76, 0.78, 0.55, 0.95]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This 22-expert design tests replication-protected latent strata against complete-link duplicate-group accounting using a spacing series of four-member reliability chains. Two compact replicated coalitions each have validities (0.747, 0.747, 0.753, 0.753): they form two balanced, multiply supported strata separated by the advocated theory's Goldilocks gap of 0.006. Three chain coalitions all have mean validity 0.75 but adjacent gaps of 0.008, 0.014, or 0.020. Coherence-Constrained Reliability-Strata Inference should favor the compact coalition increasingly as chain spacing grows. The compact tetrad receives replication protection and supported two-stratum compressibility, whereas increasingly diffuse chains incur broad-consensus loss and cannot accumulate corroboration from every local link after the theory's higher-order stratum pooling is applied. Tagged Partial-Independence Coalition Coding predicts the opposite pattern. The compact tetrad always forms one complete-link duplicate group because its total span equals the minimum duplicate tolerance, while the chains contribute at least two groups at gap 0.008 and four singleton groups once adjacent gaps exceed the maximum tolerance at 0.014 and 0.020. Ordinal-code diversity also weakly favors, rather than penalizes, the chains. Thus the tagged theory predicts compact-coalition choice below chance and a nonpositive 0.008-to-0.020 slope, while the advocated theory predicts a positive spacing slope and especially strong compact choice against the widest chain. Every chain is compared with both compact-coalition instances to reduce cue-salience contamination, and exact A/B mirrors control response-side bias. Twelve unique pairs repeated eight times yield exactly 96 trials. Experts with validities 0.55 and 0.95 are tied on every trial and provide the advertised reliability spread.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the side-invariant compact-coalition choice rate on the 64 repeated trials comparing a replicated compact tetrad with the two clearly diffuse chains (adjacent gaps .014 and .020). These conditions target the theories' strongest structural disagreement while pooling two chain identities, two compact identities, exact A/B mirrors, and eight repetitions to limit cue-salience, side-bias, and binomial variance. Coherence-Constrained Reliability-Strata Inference predicts that supported compact strata are protected from dispersion and should outperform these unsupported reliability chains. Tagged Partial-Independence Coalition Coding instead splits both diffuse chains into four singleton duplicate groups, allowing them substantially more accumulated credit than the compact tetrad. Excluding the less diagnostic .008 chain avoids dilution from subjects whose duplicate tolerance or clustering width places that chain near a grouping boundary.

Source:
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # Retain comparisons involving either of the two diffuse chains:
        # features 12:16 have spacing .014 and features 16:20 spacing .020.
        all_support = np.concatenate((a_support, b_support))
        diffuse_trial = np.any((all_support >= 12) & (all_support < 20))
        if not diffuse_trial:
            continue

        # The compact coalitions occupy features 0:8. Determine whether the
        # compact coalition supports A or B, making the score side-invariant.
        a_is_compact = (
            a_support.size > 0
            and np.all((a_support >= 0) & (a_support < 8))
        )
        b_is_compact = (
            b_support.size > 0
            and np.all((b_support >= 0) & (b_support < 8))
        )

        if a_is_compact == b_is_compact:
            continue

        response = int(row["response"])
        chose_compact = (response == 0) if a_is_compact else (response == 1)
        scores.append(float(chose_compact))

    if len(scores) == 0:
        return 0.5
    return float(np.mean(np.asarray(scores, dtype=float)))

### RESULTS
- Predicted under pi_11 (simulated): 0.1512 (var=0.0035)
- Predicted under pi_14 (simulated): 0.4944 (var=0.0099)
- Observed on real data: 0.4956 (var=0.0045)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed endorsement of the highest-validity cue's winner."""
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the uniquely highest-validity cue in this design.
        if a[0] > b[0]:
            top_cue_winner = 0
        elif b[0] > a[0]:
            top_cue_winner = 1
        else:
            continue

        scores.append(1.0 if response == top_cue_winner else -1.0)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1217 (var=0.0117)
**Predicted under pi_11:** -0.2383 (var=0.0360)
**Predicted under pi_14:** -0.1487 (var=0.0343)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    if data is None or len(data) == 0:
        return 0.0

    signed_agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        j = int(differing[0])
        ttb_response = 0 if a[j] > b[j] else 1
        observed_response = int(row["response"])
        signed_agreements.append(1.0 if observed_response == ttb_response else -1.0)

    if not signed_agreements:
        return 0.0
    return float(np.mean(signed_agreements))
```

**Observed (real) value:** -0.2642 (var=0.0087)
**Predicted under pi_11:** 0.0096 (var=0.0132)
**Predicted under pi_14:** 0.0617 (var=0.0113)

### Experiment 5
**Design**
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    # Fixed, preregistered midpoint instantiation of the advocated model.
    validities = np.array([0.52, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95], dtype=float)
    balance = 0.30
    recency = 2.10
    curvature = 0.825
    saturation = 0.725
    beta = 0.90
    epsilon = 0.05

    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity /= np.max(diagnosticity)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= np.mean(diagnosticity)

    position = np.arange(len(validities), dtype=float) / float(len(validities) - 1)
    accessibility = np.exp(recency * position)
    accessibility /= np.mean(accessibility)

    cue_weights = balance * diagnosticity + (1.0 - balance) * accessibility
    cue_weights /= np.mean(cue_weights)

    predicted_majority_probs = []
    chose_majority = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Orient every pair toward the option winning the 3-versus-2 tally.
        tally_margin = float(np.sum(d > 0) - np.sum(d < 0))
        if tally_margin == 0.0 or len(d) != len(cue_weights):
            continue
        majority_is_a = tally_margin > 0.0
        oriented = d if majority_is_a else -d

        positive = oriented > 0.0
        negative = oriented < 0.0
        n_pos = max(int(np.sum(positive)), 1)
        n_neg = max(int(np.sum(negative)), 1)
        min_n = min(n_pos, n_neg)

        pos_evidence = float(np.dot(cue_weights, np.maximum(oriented, 0.0)))
        neg_evidence = float(np.dot(cue_weights, np.maximum(-oriented, 0.0)))
        evidence = (
            pos_evidence / (max(1.0, float(n_pos) / float(min_n)) ** saturation)
            - neg_evidence / (max(1.0, float(n_neg) / float(min_n)) ** saturation)
        )

        core_p = 1.0 / (1.0 + np.exp(-np.clip(beta * evidence, -50.0, 50.0)))
        predicted_p = (1.0 - epsilon) * core_p + epsilon * 0.5
        predicted_majority_probs.append(predicted_p)

        response = int(row["response"])
        chose_majority.append(float((response == 0) if majority_is_a else (response == 1)))

    if len(predicted_majority_probs) == 0:
        return 0.0

    p = np.asarray(predicted_majority_probs, dtype=float)
    y = np.asarray(chose_majority, dtype=float)
    contrast = p - np.mean(p)
    return float(np.mean(contrast * (y - 0.5)))

```

**Observed (real) value:** -0.0410 (var=0.0003)
**Predicted under pi_11:** 0.0039 (var=0.0005)
**Predicted under pi_14:** 0.0059 (var=0.0003)

### Experiment 6
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size < 2:
            continue

        differing = np.flatnonzero(a != b)
        if differing.size != 2:
            continue

        early = int(np.min(differing))
        late = int(np.max(differing))
        late_supports_a = bool(a[late] > b[late])
        later_option_response = 0 if late_supports_a else 1
        chose_later_supported_option = float(int(row["response"]) == later_option_response)

        # Approximate matched-filter weight for the competing theory's
        # exponential positional-accessibility contrast.
        scale = float(a.size - 1)
        weight = np.exp(2.1 * late / scale) - np.exp(2.1 * early / scale)
        if np.isfinite(weight) and weight > 0.0:
            weighted_choices.append(chose_later_supported_option)
            weights.append(weight)

    if len(weights) == 0:
        return float("nan")

    return float(np.average(np.asarray(weighted_choices), weights=np.asarray(weights)))
```

**Observed (real) value:** 0.5087 (var=0.0038)
**Predicted under pi_11:** 0.4771 (var=0.0066)
**Predicted under pi_14:** 0.4938 (var=0.0036)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.5

    a = np.vstack(data["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    responses = pd.to_numeric(data["response"], errors="coerce").to_numpy(dtype=float)

    n_features = a.shape[1]
    if n_features == 1:
        positional_weights = np.ones(1, dtype=float)
    else:
        positions = np.arange(n_features, dtype=float) / float(n_features - 1)
        positional_weights = np.exp(2.1 * positions)

    # Positive evidence means positional accessibility favors A; negative
    # evidence means it favors B.
    accessibility_difference = np.dot(a - b, positional_weights)
    valid = np.isfinite(responses) & (accessibility_difference != 0.0)
    if not np.any(valid):
        return 0.5

    chose_a = responses[valid] == 0.0
    accessibility_favors_a = accessibility_difference[valid] > 0.0
    accessibility_aligned = chose_a == accessibility_favors_a
    return float(np.mean(accessibility_aligned))
```

**Observed (real) value:** 0.5021 (var=0.0024)
**Predicted under pi_11:** 0.5060 (var=0.0029)
**Predicted under pi_14:** 0.4821 (var=0.0030)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    df = data[["subject_id", "option_a_ratings", "option_b_ratings", "response"]].copy()

    def early_coalition_choice(row):
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 3 or b.size < 3:
            return np.nan
        score_a = float(np.sum(a[:3]))
        score_b = float(np.sum(b[:3]))
        if score_a == score_b:
            return np.nan
        early_side = 0 if score_a > score_b else 1
        return float(int(row["response"]) == early_side)

    df["chose_early_coalition"] = df.apply(early_coalition_choice, axis=1)
    rates = df.groupby("subject_id", sort=False)["chose_early_coalition"].mean().dropna()
    if len(rates) == 0:
        return float("nan")

    # Fraction of subjects whose majority choice favors the fixed early
    # three-cue coalition rather than the later two-cue coalition.
    return float(np.mean(rates.to_numpy(dtype=float) > 0.5))
```

**Observed (real) value:** 0.4400 (var=0.2464)
**Predicted under pi_11:** 0.1600 (var=0.1344)
**Predicted under pi_14:** 0.1600 (var=0.1344)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    entry_sum = 0.0
    entry_weight = 0.0
    exit_sum = 0.0
    exit_weight = 0.0

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        pos = np.flatnonzero(diff > 0)
        neg = np.flatnonzero(diff < 0)

        # Identify the three-cue coalition and whether it supports A or B.
        if pos.size == 3 and neg.size == 2:
            coalition = pos
            endorsed_three = 1.0 if int(row.response) == 0 else 0.0
        elif neg.size == 3 and pos.size == 2:
            coalition = neg
            endorsed_three = 1.0 if int(row.response) == 1 else 0.0
        else:
            continue

        gaps = np.diff(coalition).astype(float)
        if gaps.size != 2:
            continue

        first_compact = bool(gaps[0] == 1 and gaps[1] > 1)
        last_compact = bool(gaps[1] == 1 and gaps[0] > 1)
        if not (first_compact or last_compact):
            continue

        # Larger separation makes entry- versus exit-compact organization
        # more distinct. This bounded weight is the difference between the
        # two adjacent-link binding strengths.
        weight = abs(np.exp(-(gaps[0] - 1.0)) - np.exp(-(gaps[1] - 1.0)))

        if first_compact:
            entry_sum += weight * endorsed_three
            entry_weight += weight
        else:
            exit_sum += weight * endorsed_three
            exit_weight += weight

    if entry_weight <= 0.0 or exit_weight <= 0.0:
        return 0.0

    return float(entry_sum / entry_weight - exit_sum / exit_weight)

```

**Observed (real) value:** 0.0219 (var=0.0094)
**Predicted under pi_11:** 0.0143 (var=0.0064)
**Predicted under pi_14:** -0.0001 (var=0.0100)

### Experiment 10
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []
    late_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        a_support = np.flatnonzero(differences > 0)
        b_support = np.flatnonzero(differences < 0)
        response = int(row["response"])

        if a_support.size == 3 and b_support.size == 2:
            onset_difference = int(a_support[0]) - int(b_support[0])
            chose_three = float(response == 0)
        elif a_support.size == 2 and b_support.size == 3:
            onset_difference = int(b_support[0]) - int(a_support[0])
            chose_three = float(response == 1)
        else:
            continue

        if onset_difference < 0:
            early_choices.append(chose_three)
        elif onset_difference > 0:
            late_choices.append(chose_three)

    if len(early_choices) == 0 or len(late_choices) == 0:
        return 0.0

    return float(np.mean(early_choices) - np.mean(late_choices))
```

**Observed (real) value:** 0.0117 (var=0.0101)
**Predicted under pi_11:** -0.0054 (var=0.0118)
**Predicted under pi_14:** 0.0179 (var=0.0084)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    closure_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        difference = a - b
        discriminating = np.flatnonzero(difference != 0.0)

        if discriminating.size == 0:
            continue

        # The closure coalition is the option supported by the final
        # discriminating cue, irrespective of whether it is displayed as A or B.
        last_index = int(discriminating[-1])
        closure_response = 0 if difference[last_index] > 0.0 else 1
        closure_choices.append(float(int(row["response"]) == closure_response))

    if len(closure_choices) == 0:
        return float("nan")

    return float(np.mean(np.asarray(closure_choices, dtype=np.float64)))
```

**Observed (real) value:** 0.4958 (var=0.0041)
**Predicted under pi_11:** 0.5090 (var=0.0024)
**Predicted under pi_14:** 0.5110 (var=0.0027)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.55, 0.70,
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.60, 0.70
    ], dtype=np.float64)
    bounded = np.clip(2.0 * validities - 1.0, 0.0, 1.0)

    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size != validities.size or b.size != validities.size:
            continue

        differences = a - b
        a_support = np.flatnonzero(differences > 0.0)
        b_support = np.flatnonzero(differences < 0.0)
        if a_support.size != 2 or b_support.size != 2:
            continue

        a_equal = bool(np.isclose(validities[a_support[0]], validities[a_support[1]]))
        b_equal = bool(np.isclose(validities[b_support[0]], validities[b_support[1]]))
        if a_equal == b_equal:
            continue

        load_a = float(bounded[a_support[0]] * bounded[a_support[1]])
        load_b = float(bounded[b_support[0]] * bounded[b_support[1]])
        weight = abs(load_a - load_b)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        response = int(row["response"])
        chose_equal = float((response == 0) if a_equal else (response == 1))
        weighted_choices.append(weight * chose_equal)
        weights.append(weight)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.5
    return float(np.sum(weighted_choices) / total_weight)

```

**Observed (real) value:** 0.1723 (var=0.0081)
**Predicted under pi_11:** 0.2817 (var=0.0060)
**Predicted under pi_14:** 0.4606 (var=0.0076)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        high_gap = []
        low_gap = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b

            # Target coalitions occupy cues 0--15; cues 16--17 are the
            # fixed reference coalition. Each target coalition is one of
            # the adjacent cue pairs (0,1), (2,3), ..., (14,15).
            target_idx = np.flatnonzero(d[:16] != 0.0)
            if target_idx.size != 2:
                continue

            pair_id = int(np.min(target_idx) // 2)
            direction = float(np.mean(d[target_idx]))
            if direction == 0.0:
                continue

            target_response = 0 if direction > 0.0 else 1
            chose_target = float(int(row["response"]) == target_response)

            # Low gaps comprise 0.00 and 0.04; high gaps comprise 0.10
            # and 0.18. This uses all 96 observations per subject.
            if pair_id <= 3:
                low_gap.append(chose_target)
            else:
                high_gap.append(chose_target)

        if len(high_gap) == 0 or len(low_gap) == 0:
            return np.nan
        return float(np.mean(high_gap) - np.mean(low_gap))

    if "subject_id" in data.columns:
        values = [subject_contrast(df) for _, df in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_contrast(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))
```

**Observed (real) value:** 0.0317 (var=0.0030)
**Predicted under pi_11:** 0.1204 (var=0.0114)
**Predicted under pi_14:** 0.0250 (var=0.0090)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    distributed = {(3, 4, 5), (9, 10, 11)}

    def subject_score(df):
        records = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = a - b
            a_support = tuple(np.flatnonzero(diff > 0).tolist())
            b_support = tuple(np.flatnonzero(diff < 0).tolist())
            a_is_distributed = a_support in distributed
            b_is_distributed = b_support in distributed
            if a_is_distributed == b_is_distributed:
                continue
            chose_a = int(row['response']) == 0
            chose_distributed = (
                (chose_a and a_is_distributed) or
                ((not chose_a) and b_is_distributed)
            )
            records.append((a_is_distributed, float(chose_distributed)))

        if not records:
            return np.nan

        records = pd.DataFrame(records, columns=['distributed_on_a', 'choice'])
        role_means = records.groupby('distributed_on_a')['choice'].mean()
        if len(role_means) == 2:
            return float(0.5 * (role_means.loc[False] + role_means.loc[True]))
        return float(records['choice'].mean())

    if len(data) == 0:
        return float('nan')

    if 'subject_id' in data.columns:
        scores = [subject_score(df) for _, df in data.groupby('subject_id', sort=False)]
    else:
        scores = [subject_score(data)]
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    return float(np.mean(scores)) if scores.size else float('nan')
```

**Observed (real) value:** 0.6692 (var=0.0068)
**Predicted under pi_11:** 0.6735 (var=0.0042)
**Predicted under pi_14:** 0.5206 (var=0.0047)

### Experiment 15
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    cell_values = {}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        response = int(row['response'])

        # In this design, distributed coalitions occupy cues 6--11 and
        # clustered coalitions occupy cues 0--5. Identify which display side
        # contains the distributed coalition from discriminating support.
        a_support = np.flatnonzero(a > b)
        b_support = np.flatnonzero(b > a)
        a_is_distributed = (
            a_support.size > 0 and np.all(a_support >= 6)
        )
        b_is_distributed = (
            b_support.size > 0 and np.all(b_support >= 6)
        )

        if a_is_distributed == b_is_distributed:
            continue

        chose_distributed = (
            (response == 0 and a_is_distributed) or
            (response == 1 and b_is_distributed)
        )
        signed_choice = 1.0 if chose_distributed else -1.0

        # Equal-weight the eight unique configurations so that the statistic
        # remains mirror-corrected even if a few trials are missing.
        key = (tuple(a.tolist()), tuple(b.tolist()))
        if key not in cell_values:
            cell_values[key] = []
        cell_values[key].append(signed_choice)

    if not cell_values:
        return float('nan')

    cell_means = [float(np.mean(values)) for values in cell_values.values()]
    return float(np.mean(cell_means))

```

**Observed (real) value:** 0.3242 (var=0.0194)
**Predicted under pi_11:** 0.3646 (var=0.0242)
**Predicted under pi_14:** -0.0279 (var=0.0164)

### Experiment 16
**Design**
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Concentrate on the direct endpoint comparison and the three early
    # reliability-gap transitions where region release should be largest.
    diagnostic_pairs = {(0, 1), (1, 2), (2, 3), (0, 5)}
    distributed_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.int8)
        b = np.asarray(row["option_b_ratings"], dtype=np.int8)
        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        if a_support.size == 0 or b_support.size == 0:
            continue

        # Each coalition occupies one of six consecutive three-cue blocks.
        a_block = int(a_support[0] // 3)
        b_block = int(b_support[0] // 3)
        pair = tuple(sorted((a_block, b_block)))
        if pair not in diagnostic_pairs:
            continue

        response = int(row["response"])
        chosen_block = a_block if response == 0 else b_block
        distributed_block = max(a_block, b_block)
        distributed_choices.append(float(chosen_block == distributed_block))

    if not distributed_choices:
        return float("nan")
    return float(np.mean(distributed_choices))
```

**Observed (real) value:** 0.5869 (var=0.0047)
**Predicted under pi_11:** 0.5844 (var=0.0036)
**Predicted under pi_14:** 0.4959 (var=0.0038)

### Experiment 17
**Design**
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Triadic onset in preference for reliability-separated coalitions."""
    dyad_choices = []
    triad_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.int8)
        b = np.asarray(row.option_b_ratings, dtype=np.int8)
        d = a - b

        a_support = np.flatnonzero(d > 0)
        b_support = np.flatnonzero(d < 0)
        if a_support.size != b_support.size or a_support.size not in (2, 3):
            continue

        # The reliability-separated pair uses cues 4:8 (validities .65 and
        # .7666667); the matched pair uses two cues 0:4 (validity .70).
        a_mid = int(np.sum((a_support >= 4) & (a_support < 8)))
        b_mid = int(np.sum((b_support >= 4) & (b_support < 8)))
        a_equal = int(np.sum(a_support < 4))
        b_equal = int(np.sum(b_support < 4))

        if a_mid == 2 and b_equal == 2:
            separated_is_a = True
        elif b_mid == 2 and a_equal == 2:
            separated_is_a = False
        else:
            continue

        response = int(row.response)
        chose_separated = float(
            (separated_is_a and response == 0)
            or ((not separated_is_a) and response == 1)
        )

        if a_support.size == 2:
            dyad_choices.append(chose_separated)
        else:
            triad_choices.append(chose_separated)

    if len(dyad_choices) == 0 or len(triad_choices) == 0:
        return float("nan")

    return float(np.mean(triad_choices) - np.mean(dyad_choices))
```

**Observed (real) value:** 0.0142 (var=0.0045)
**Predicted under pi_11:** -0.0717 (var=0.0091)
**Predicted under pi_14:** -0.0237 (var=0.0095)

### Experiment 18
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Mirror-corrected probability of choosing a single-duplicate coalition
    # (features 8:16) over a double-duplicate coalition (features 0:8).
    chosen_single = []
    single_on_a = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_single = float(np.sum(a[8:16])) > float(np.sum(a[0:8]))
        b_is_single = float(np.sum(b[8:16])) > float(np.sum(b[0:8]))
        if a_is_single == b_is_single:
            continue

        response = int(row["response"])
        chose_single = ((response == 0 and a_is_single) or
                         (response == 1 and b_is_single))
        chosen_single.append(float(chose_single))
        single_on_a.append(bool(a_is_single))

    if not chosen_single:
        return float("nan")

    y = np.asarray(chosen_single, dtype=np.float64)
    side = np.asarray(single_on_a, dtype=bool)

    # Give the two mirror orientations equal weight, canceling stable A/B bias.
    if np.any(side) and np.any(~side):
        return float(0.5 * (np.mean(y[side]) + np.mean(y[~side])))
    return float(np.mean(y))

```

**Observed (real) value:** 0.6779 (var=0.0032)
**Predicted under pi_11:** 0.6365 (var=0.0053)
**Predicted under pi_14:** 0.5260 (var=0.0041)

### Experiment 19
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed, inverse-variance-weighted cardinality profile of choices for the
    # reliability-diverse coalition. The duplicate pair identifying each
    # cardinality is fixed by this experimental design.
    duplicate_pair = {2: (0, 1), 3: (4, 5), 4: (10, 11)}
    weights = {2: -1.00, 3: 0.53, 4: -0.51}

    def subject_score(df):
        chosen_diverse = {2: [], 3: [], 4: []}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b
            cardinality = int(np.sum(d > 0))
            if cardinality not in duplicate_pair:
                continue

            i, j = duplicate_pair[cardinality]
            if d[i] > 0 and d[j] > 0:
                # The duplicate coalition supports A, so B is diverse.
                y = float(int(row["response"]) == 1)
            elif d[i] < 0 and d[j] < 0:
                # The duplicate coalition supports B, so A is diverse.
                y = float(int(row["response"]) == 0)
            else:
                continue
            chosen_diverse[cardinality].append(y)

        if any(len(chosen_diverse[k]) == 0 for k in (2, 3, 4)):
            return np.nan

        rates = {k: float(np.mean(chosen_diverse[k])) for k in (2, 3, 4)}
        return float(sum(weights[k] * (rates[k] - 0.5) for k in (2, 3, 4)))

    scores = []
    for _, subj_df in data.groupby("subject_id", sort=False):
        value = subject_score(subj_df)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.3486 (var=0.0131)
**Predicted under pi_11:** -0.1886 (var=0.0173)
**Predicted under pi_14:** -0.0326 (var=0.0166)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Equal-weighted, mirror-corrected probability of choosing the chain coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size != a.size:
            continue

        d = a - b
        chain_direction = float(np.sum(d[:10]))
        if abs(chain_direction) < 1e-12:
            continue

        # Positive means that A is supported by one of the two chain blocks;
        # negative means that B is supported by a chain block.
        chain_is_a = chain_direction > 0.0
        response = int(row["response"])
        chain_chosen = float((response == 0 and chain_is_a) or
                             (response == 1 and not chain_is_a))

        # Identify the crossed chain/comparison configuration independently
        # of whether it was displayed in its original or mirrored orientation.
        chain_block = int(np.sum(np.abs(d[5:10])) > np.sum(np.abs(d[0:5])))
        comparison_block = int(
            np.sum(np.abs(d[15:20])) > np.sum(np.abs(d[10:15]))
        )
        family = 2 * chain_block + comparison_block
        records.append((row["subject_id"], family, chain_chosen))

    if not records:
        return float("nan")

    scored = pd.DataFrame(records, columns=["subject_id", "family", "score"])
    # Equal weighting of the four crossed configurations prevents stable cue
    # salience or an uneven number of usable repetitions from dominating.
    family_rates = scored.groupby(["subject_id", "family"])["score"].mean()
    subject_rates = family_rates.groupby(level=0).mean()
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.4983 (var=0.0018)
**Predicted under pi_11:** 0.4998 (var=0.0039)
**Predicted under pi_14:** 0.6452 (var=0.0049)

### Experiment 21
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    narrow_choices = []
    wide_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        a_support = np.flatnonzero(a > b)
        b_support = np.flatnonzero(b > a)

        # Features 0--3 form the two narrow-gap target dyads;
        # features 4--7 form the two wide-gap target dyads.
        a_narrow = np.any(a_support < 4)
        b_narrow = np.any(b_support < 4)
        a_wide = np.any((a_support >= 4) & (a_support < 8))
        b_wide = np.any((b_support >= 4) & (b_support < 8))

        if a_narrow and not b_narrow:
            narrow_choices.append(float(response == 0))
        elif b_narrow and not a_narrow:
            narrow_choices.append(float(response == 1))
        elif a_wide and not b_wide:
            wide_choices.append(float(response == 0))
        elif b_wide and not a_wide:
            wide_choices.append(float(response == 1))

    if len(narrow_choices) == 0 or len(wide_choices) == 0:
        return float("nan")

    return float(np.mean(wide_choices) - np.mean(narrow_choices))
```

**Observed (real) value:** -0.0283 (var=0.0129)
**Predicted under pi_11:** -0.0029 (var=0.0090)
**Predicted under pi_14:** 0.1217 (var=0.0113)

### Experiment 22
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score each response as 1 when the chain-validity coalition was chosen
    # and 0 when the exact-duplicate comparison coalition was chosen.
    chain_when_a = []
    chain_when_b = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        d = a - b

        # In this design, chain experts occupy indices 0:8. Exactly one side
        # is supported by four of these experts on every experimental trial.
        a_chain_support = int(np.sum(d[:8] > 0.0))
        b_chain_support = int(np.sum(d[:8] < 0.0))
        response = int(row["response"])

        if a_chain_support > b_chain_support:
            chain_when_a.append(float(response == 0))
        elif b_chain_support > a_chain_support:
            chain_when_b.append(float(response == 1))

    # Average the two response-side orientations separately so the statistic
    # is explicitly mirror-corrected rather than sensitive to side imbalance.
    orientation_means = []
    if len(chain_when_a) > 0:
        orientation_means.append(float(np.mean(chain_when_a)))
    if len(chain_when_b) > 0:
        orientation_means.append(float(np.mean(chain_when_b)))

    if len(orientation_means) == 0:
        return float("nan")
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5042 (var=0.0027)
**Predicted under pi_11:** 0.5040 (var=0.0037)
**Predicted under pi_14:** 0.6625 (var=0.0051)

### Experiment 23
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Fraction of subjects showing a supermajority preference for diverse coalitions."""
    if data is None or len(data) == 0:
        return 0.0

    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size != 16 or b.size != 16:
            continue

        diff = a - b
        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # In this design, indices 0--7 form the two reliability-diverse
        # coalitions, whereas indices 8--15 form the exact-duplicate blocks.
        a_diverse = (a_support.size == 4 and np.all(a_support < 8))
        b_diverse = (b_support.size == 4 and np.all(b_support < 8))
        a_duplicate = (a_support.size == 4 and np.all(a_support >= 8))
        b_duplicate = (b_support.size == 4 and np.all(b_support >= 8))

        if a_diverse and b_duplicate:
            chose_diverse = int(row["response"] == 0)
        elif b_diverse and a_duplicate:
            chose_diverse = int(row["response"] == 1)
        else:
            continue

        records.append((row["subject_id"], chose_diverse))

    if len(records) == 0:
        return 0.0

    scored = pd.DataFrame(records, columns=["subject_id", "chose_diverse"])
    subject_rates = scored.groupby("subject_id")["chose_diverse"].mean()

    # A 2/3 threshold lies well above the competing theory's near-chance
    # prediction but below the advocated theory's predicted range.
    return float(np.mean(subject_rates >= (2.0 / 3.0)))
```

**Observed (real) value:** 0.6400 (var=0.2304)
**Predicted under pi_11:** 1.0000 (var=0.0000)
**Predicted under pi_14:** 0.9000 (var=0.0900)

### Experiment 24
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In this fixed design, features 0--3 form the two near-duplicate
    # coalitions, whereas features 4--7 form the two exact-duplicate
    # coalitions. Every trial pits one type against the other.
    near_choices = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=np.float64)
        b = np.asarray(row.option_b_ratings, dtype=np.float64)

        a_near_mass = float(np.sum(a[:4]))
        a_exact_mass = float(np.sum(a[4:8]))
        b_near_mass = float(np.sum(b[:4]))
        b_exact_mass = float(np.sum(b[4:8]))

        a_is_near = a_near_mass > a_exact_mass
        b_is_near = b_near_mass > b_exact_mass

        # Retain only the intended near-versus-exact diagnostic trials.
        if a_is_near == b_is_near:
            continue

        response = int(row.response)
        chose_near = a_is_near if response == 0 else b_is_near
        near_choices.append(float(chose_near))

    if len(near_choices) == 0:
        return 0.5
    return float(np.mean(near_choices))
```

**Observed (real) value:** 0.6817 (var=0.0026)
**Predicted under pi_11:** 0.4998 (var=0.0029)
**Predicted under pi_14:** 0.6856 (var=0.0112)

### Experiment 25
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    a = np.vstack([np.asarray(x, dtype=np.float64) for x in data['option_a_ratings']])
    b = np.vstack([np.asarray(x, dtype=np.float64) for x in data['option_b_ratings']])
    responses = data['response'].to_numpy(dtype=np.int64)

    # Features 8:16 comprise the two fully separated coalitions. Their signed
    # rating difference identifies whether the separated coalition is on A
    # (+1) or B (-1), independently of the particular coalition identities.
    separated_side = np.sign(np.sum(a[:, 8:16] - b[:, 8:16], axis=1))
    valid = (separated_side != 0) & ((responses == 0) | (responses == 1))
    if not np.any(valid):
        return float('nan')

    # +1 denotes choosing A and -1 choosing B. Multiplication mirror-corrects
    # every response so +1 always means choosing the separated coalition.
    chosen_side = 1.0 - 2.0 * responses[valid].astype(np.float64)
    return float(np.mean(chosen_side * separated_side[valid]))

```

**Observed (real) value:** -0.3542 (var=0.0111)
**Predicted under pi_11:** 0.0071 (var=0.0085)
**Predicted under pi_14:** -0.0804 (var=0.0180)

### Experiment 26
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    validities = np.asarray([
        0.55, 0.554, 0.56, 0.574, 0.58, 0.594, 0.6, 0.604,
        0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79,
        0.81, 0.83, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95
    ], dtype=float)

    broad_when_a = []
    broad_when_b = []

    def coalition_gap(ratings):
        x = np.asarray(ratings, dtype=float)
        idx = np.flatnonzero(x > 0.5)
        if idx.size != 2:
            return np.nan
        return float(abs(validities[idx[1]] - validities[idx[0]]))

    for _, row in data.iterrows():
        gap_a = coalition_gap(row["option_a_ratings"])
        gap_b = coalition_gap(row["option_b_ratings"])
        if not np.isfinite(gap_a) or not np.isfinite(gap_b):
            continue

        # The design contains narrow gaps of .004 and broad gaps of .014;
        # .009 is their fixed midpoint and does not depend on responses.
        a_is_broad = gap_a > 0.009
        b_is_broad = gap_b > 0.009
        if a_is_broad == b_is_broad:
            continue

        response = int(row["response"])
        chose_broad = float(
            (a_is_broad and response == 0) or
            (b_is_broad and response == 1)
        )

        if a_is_broad:
            broad_when_a.append(chose_broad)
        else:
            broad_when_b.append(chose_broad)

    # Average the two mirror orientations separately before combining them.
    # This exactly removes a stable A/B response tendency in the balanced
    # schedule and remains well-defined for a single subject.
    orientation_means = []
    if len(broad_when_a) > 0:
        orientation_means.append(float(np.mean(broad_when_a)))
    if len(broad_when_b) > 0:
        orientation_means.append(float(np.mean(broad_when_b)))

    if len(orientation_means) == 0:
        return 0.0
    return float(np.mean(orientation_means))

```

**Observed (real) value:** 0.3208 (var=0.0038)
**Predicted under pi_11:** 0.7165 (var=0.0043)
**Predicted under pi_14:** 0.3846 (var=0.0102)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the full 26-experiment record, pi_11 is the stronger task-invariant account, although neither theory is adequate. The clearest comparison is the pair of newly reported experiments. In Experiment 1, pi_11 predicts the observed direction and prevalence well: 1.000 versus 0.920 observed, whereas pi_14 predicts only 0.410 and reverses the dominant population pattern. In Experiment 2, pi_14 is essentially exact\u20140.4944 versus 0.4956\u2014while pi_11 strongly overpredicts diffuse-chain preference, yielding 0.1512 compact choice. Thus each theory wins one new experiment, but Experiment 1 directly rejects pi_14's strong compact-strata advantage, while Experiment 2 rejects pi_11's unqualified accumulation of credit across separated singleton groups.\n\nThe broader record favors pi_11. It closely captures the distributed/diverse-coalition effects in Experiments 14\u201316, the duplicate-accounting effect in Experiment 18, and the neutral chain results in Experiments 20 and 22. It is also materially closer in Experiments 1, 7, 9, 11, 12, 14\u201316, 18\u201322, and often in Experiment 19. Several of these are theoretically diagnostic rather than merely small numerical wins: pi_14 predicts substantial chain preference in Experiments 20 and 22 (0.6452 and 0.6625) when behavior is almost exactly at chance (0.4983 and 0.5042), and it misses the strong distributed-coalition preferences in Experiments 14\u201316 and 18. These failures indicate that broad or chain-like reliability structure does not generally lose value through a coalition-wide coherence constraint, nor does compact replicated-stratum structure generally earn the large protection posited by pi_14.\n\nNevertheless, pi_14 identifies a real local phenomenon that pi_11 lacks. Experiment 24 shows a strong preference for near- over exact-duplicate dyads (0.6817), captured almost exactly by pi_14 and missed by pi_11. Experiment 26 then shows broad dyads being disfavored (0.3208), again closer to pi_14, supporting a local equality-to-small-gap release followed by decline. Pi_14 also fits Experiments 2 and 13 and is closer in Experiments 3, 10, 17, 23, and 25. The important inference is therefore not that source organization is irrelevant, but that its effect is local and capacity-dependent rather than a general latent-strata coherence principle.\n\nBoth theories also share consequential failures. Both predict the wrong or much too weak effects in Experiments 4 and 5, underpredict the early three-cue prevalence in Experiment 8, overpredict the supermajority in Experiment 23, and substantially understate the compact-over-separated effect in Experiment 25. The low observed between-subject variances in Experiments 4, 5, and 25 make these discrepancies especially difficult to dismiss as heterogeneous cue salience. Conversely, the near-chance positional, onset, closure, and chain metrics in Experiments 6, 7, 9\u201311, 20, and 22 argue against repairing either theory with a generic serial-position mechanism.\n\nPi_14 should therefore be replaced rather than merely retuned. Its central commitments\u2014compact replicated strata receiving protection, broad separation losing consensus coherence, and higher-order value being controlled by a strongest supported stratum link\u2014produce repeated qualitative errors. Weakening those mechanisms enough to fit Experiments 1, 14\u201316, 18, 20, and 22 would remove the defining content of the theory, while retaining enough strength to fit Experiments 2, 24, and 26. That is a theoretical conflict, not just a poor parameter range. Pi_11 should remain as the current benchmark because it explains more of the cross-task structure, but a new competitor is needed to capture the genuine local Goldilocks effect without inheriting pi_14's global coherence penalty.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_14 with a theory of Capacity-Gated Local Source Individuation. Its core claim should be that source dependence is inferred locally among the closest reliability neighbors, while larger coalitions are evaluated through bounded, nontransitive evidence accounts rather than through a globally coherent latent partition. Exact equality should strongly activate a common-source tag and discount duplicate evidence. A small nonzero gap should release that tag and provide local independence credit, producing the near-over-exact effect in Experiment 24. For dyads, that credit should decline once the gap becomes too broad, capturing Experiment 26. Crucially, this dyadic Goldilocks function must not become a coalition-wide span penalty: in tetrads and larger coalitions, local links should be capacity-limited and nonaccumulating, while distinct reliability regions can still contribute separate bounded accounts. This should allow diffuse coalitions to beat compact duplicate-heavy coalitions in Experiment 1 and Experiments 14\u201316 and 18, while preventing chains from accumulating a large advantage in Experiments 20 and 22.\n\nThe new theory should include an explicit cardinality gate: dyads permit fine metric comparison of reliability gaps; at three or more cues, observers increasingly compress reliability into a few contextual codes and use bounded account coverage plus duplicate penalties. Any two-to-three release should be weak and empirically constrained, because Experiment 17 is near zero, and the model must reproduce the cardinality profile in Experiment 19 rather than imposing a generic triadic bonus. Reliability weighting should remain compressed but consequential, and duplicate penalties should accumulate by locally tagged group as in the successful portion of pi_11. Do not add a generic global coherence, total-span, or strongest-stratum-link term.\n\nThe model should also distinguish choice sensitivity from structural valuation so that near-chance means and subject-threshold metrics are not forced to move together. Stable subject heterogeneity may vary duplicate tolerance, local-gap resolution, reliability compression, and response sensitivity, but these distributions must be shared across experiments rather than selected per design. Retain exchangeable cue salience and avoid population-level recency, onset, adjacency, or closure gradients, given the repeated null results for those mechanisms. During development, jointly calibrate against all experiments, with hard qualitative checks that the model simultaneously produces: dispersed preference in Experiment 1; approximately chance compact choice in Experiment 2; near-over-exact preference in Experiment 24; broad-dyad avoidance in Experiment 26; approximately chance chain choice in Experiments 20 and 22; duplicate avoidance in Experiments 12 and 18; and distributed-coalition preference in Experiments 14\u201316. It should additionally incorporate a coalition-level normalization or contrast process capable of addressing the shared failures in Experiments 4, 5, 8, and 25, without invoking unsupported serial-position effects."
}
```

## Usage

```json
{
  "input_tokens": 45094,
  "output_tokens": 2491
}
```
