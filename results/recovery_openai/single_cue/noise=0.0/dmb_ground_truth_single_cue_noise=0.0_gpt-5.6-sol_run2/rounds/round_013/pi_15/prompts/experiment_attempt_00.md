# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Capacity-Gated Local Source Individuation proposes that observers do not infer a globally coherent partition of experts into latent sources. They inspect a capacity-limited set of nearest reliability relations. Exact equality activates a categorical common-source tag, while dyads permit fine metric comparison through a localized Goldilocks function: narrow positive gaps earn independence credit, moderately broad gaps incur uncertainty, and very large gaps recover toward reliability-weighted baseline. At cardinalities of three or more, ordinary coalition evidence is computed from compressed cue quality and cardinality normalization, independently of contextual-code occupancy. Contextual codes contribute only a bounded structural residual licensed by one or two sampled, resolvable local links; consequently, occupying additional reliability codes cannot itself give a broad chain an evidence advantage. Duplicate penalties remain additive across locally tagged equality groups. Structural valuation is separated from response sensitivity, with a stable subject-specific overload criterion governing unequal three-versus-two choices. Reliability compression, duplicate tolerance, local-gap resolution, overload criterion, and response sensitivity vary across subjects from distributions shared across experiments. Cue salience remains exchangeable, with no population-level recency, onset, adjacency, or closure gradient.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.34]
- validity_reliance: [0.06, 0.15]
- salience_strength: [0.04, 0.18]
- equality_resolution: [0.0000001, 0.00001]
- dyad_gap_resolution: [0.0025, 0.0055]
- dyad_broad_scale: [0.0045, 0.0090]
- dyad_independence_gain: [0.45, 1.05]
- dyad_duplicate_penalty: [0.65, 1.35]
- dyad_trough_strength: [0.18, 0.52]
- dyad_trough_center: [0.011, 0.017]
- dyad_trough_width: [0.0035, 0.0065]
- code_resolution: {2, 3}
- account_capacity: [0.85, 1.65]
- account_normalization: [0.50, 0.72]
- coverage_gain: [0.10, 0.42]
- large_coalition_gate: [0.85, 1.80]
- local_link_gain: [0.08, 0.30]
- local_link_capacity: {1, 2}
- coalition_duplicate_penalty: [0.18, 0.48]
- duplicate_group_saturation: [0.45, 1.05]
- count_curvature: [0.72, 0.90]
- contrast_normalization: [0.35, 0.70]
- contrast_scale: [0.55, 1.15]
- overload_criterion: [-0.05, 0.65]
- beta: [0.85, 2.20]
- epsilon: [0.025, 0.085]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Local Source Individuation expects shape "
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

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    equality_resolution = float(parameters["equality_resolution"])
    dyad_gap_resolution = float(parameters["dyad_gap_resolution"])
    dyad_broad_scale = float(parameters["dyad_broad_scale"])
    dyad_independence_gain = float(parameters["dyad_independence_gain"])
    dyad_duplicate_penalty = float(parameters["dyad_duplicate_penalty"])
    dyad_trough_strength = float(parameters["dyad_trough_strength"])
    dyad_trough_center = float(parameters["dyad_trough_center"])
    dyad_trough_width = float(parameters["dyad_trough_width"])
    code_resolution = int(parameters["code_resolution"])
    account_capacity = float(parameters["account_capacity"])
    account_normalization = float(parameters["account_normalization"])
    coverage_gain = float(parameters["coverage_gain"])
    large_coalition_gate = float(parameters["large_coalition_gate"])
    local_link_gain = float(parameters["local_link_gain"])
    local_link_capacity = int(parameters["local_link_capacity"])
    coalition_duplicate_penalty = float(
        parameters["coalition_duplicate_penalty"]
    )
    duplicate_group_saturation = float(
        parameters["duplicate_group_saturation"]
    )
    count_curvature = float(parameters["count_curvature"])
    contrast_normalization = float(parameters["contrast_normalization"])
    contrast_scale = float(parameters["contrast_scale"])
    overload_criterion = float(parameters["overload_criterion"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated reliability remains compressed but consequential.
    log_odds = np.log(v / (1.0 - v))
    compressed = np.power(np.maximum(log_odds, 1e-12), validity_compression)
    compressed /= max(float(np.mean(compressed)), 1e-12)
    reliability_quality = (
        (1.0 - validity_reliance)
        + validity_reliance * compressed
    )

    # Subject-specific cue salience is centered and exchangeable.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_quality = reliability_quality * attention
    cue_quality /= max(float(np.mean(cue_quality)), 1e-12)

    # Coarse contextual codes are bounded evidence accounts, not latent
    # transitive source partitions.
    unique_v, inverse_levels = np.unique(v, return_inverse=True)
    n_levels = int(unique_v.size)
    actual_codes = max(1, min(code_resolution, n_levels))
    if n_levels == 1:
        level_codes = np.zeros(1, dtype=np.int64)
    else:
        level_codes = np.floor(
            np.arange(n_levels, dtype=np.float64)
            * float(actual_codes)
            / float(n_levels)
        ).astype(np.int64)
        level_codes = np.minimum(level_codes, actual_codes - 1)
    cue_codes = level_codes[inverse_levels]

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def goldilocks_credit(gap):
        gap = float(max(gap, 0.0))
        if gap <= equality_resolution:
            return 0.0

        release = 1.0 - np.exp(
            -gap / max(dyad_gap_resolution, 1e-12)
        )
        broad_decay = np.exp(
            -gap / max(dyad_broad_scale, 1e-12)
        )
        return float(np.clip(release * broad_decay, 0.0, 1.0))

    def duplicate_load(indices):
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size <= 1:
            return 0.0

        levels = inverse_levels[idx]
        counts = np.bincount(levels, minlength=n_levels).astype(np.float64)
        load = 0.0
        for count in counts[counts > 1.0]:
            excess = float(count - 1.0)
            load += excess / (
                1.0 + duplicate_group_saturation * excess
            )
        return float(load)

    def coalition_value(indices):
        idx = np.asarray(indices, dtype=np.int64)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_quality[idx[0]])

        qualities = cue_quality[idx]
        coalition_v = v[idx]

        if n == 2:
            gap = abs(float(coalition_v[0] - coalition_v[1]))
            base = float(np.sum(qualities)) / (
                2.0 ** account_normalization
            )

            if gap <= equality_resolution:
                structural_factor = np.exp(-dyad_duplicate_penalty)
            else:
                credit = goldilocks_credit(gap)
                localized_trough = np.exp(
                    -0.5
                    * ((gap - dyad_trough_center)
                       / max(dyad_trough_width, 1e-12)) ** 2
                )
                structural_factor = (
                    1.0
                    + dyad_independence_gain * credit
                    - dyad_trough_strength * localized_trough
                )
                structural_factor = max(structural_factor, 0.05)

            return float(
                np.power(max(base, 1e-12), count_curvature)
                * structural_factor
            )

        codes = cue_codes[idx]
        occupied_codes = np.unique(codes)
        account_values = []
        for code in occupied_codes:
            members = np.flatnonzero(codes == code)
            m = int(members.size)
            mean_quality = float(np.mean(qualities[members]))
            if m == 1:
                represented_mass = 1.0
            else:
                represented_mass = 1.0 + account_capacity * (
                    1.0
                    - np.exp(
                        -float(m - 1) / max(account_capacity, 1e-12)
                    )
                )
            account_values.append(mean_quality * represented_mass)

        coverage = int(len(account_values))

        # Ordinary evidence is independent of contextual-code occupancy.
        # Codes can affect only the separately gated structural residual below.
        normalized_accounts = float(np.sum(qualities)) / (
            float(n) ** account_normalization
        )

        bounded_coverage = float(min(max(coverage - 1, 0), 1))
        load_gate = 1.0 / (
            1.0 + large_coalition_gate * float(max(n - 3, 0))
        )

        # Only a bounded set of nearest positive links is inspected.
        order = np.argsort(coalition_v)
        sorted_v = coalition_v[order]
        adjacent_gaps = np.diff(sorted_v)
        positive_gaps = adjacent_gaps[adjacent_gaps > equality_resolution]
        if positive_gaps.size == 0:
            local_credit = 0.0
        else:
            nearest = np.sort(positive_gaps)[:local_link_capacity]
            credits = np.asarray(
                [goldilocks_credit(g) for g in nearest],
                dtype=np.float64,
            )
            local_credit = float(np.mean(credits)) if credits.size else 0.0

        # Triads retain bounded account coverage. At four or more cues, a
        # second account earns coverage credit only when the sampled local
        # relation is itself resolvable. Squaring makes weak or broad links
        # contribute little without introducing a global-span penalty.
        if n == 3:
            coverage_activation = 1.0
            link_activation = local_credit
        else:
            coverage_activation = local_credit ** 2
            link_activation = local_credit ** 2

        coverage_factor = 1.0 + (
            coverage_gain
            * bounded_coverage
            * load_gate
            * coverage_activation
        )
        local_factor = 1.0 + (
            local_link_gain * link_activation * load_gate
        )

        dup_load = duplicate_load(idx)
        duplicate_factor = np.exp(
            -coalition_duplicate_penalty * dup_load
        )

        represented = np.power(
            max(normalized_accounts, 1e-12), count_curvature
        )
        return float(
            represented
            * coverage_factor
            * local_factor
            * duplicate_factor
        )

    value_a = coalition_value(a_support)
    value_b = coalition_value(b_support)

    comparison_scale = (
        1.0 + 0.5 * (abs(value_a) + abs(value_b))
    ) ** contrast_normalization
    normalized_difference = (value_a - value_b) / max(
        comparison_scale, 1e-12
    )

    structural_contrast = contrast_scale * np.tanh(
        normalized_difference / max(contrast_scale, 1e-12)
    )

    # A stable response-stage overload criterion applies only to unequal
    # three-versus-two conflicts. Its heterogeneous distribution can change
    # subject-threshold measures without imposing a generic triad valuation
    # bonus or penalty on equal-cardinality comparisons.
    sizes = {int(a_support.size), int(b_support.size)}
    if sizes == {2, 3}:
        larger_side = float(np.sign(a_support.size - b_support.size))
        structural_contrast -= overload_criterion * larger_side

    logits = np.array(
        [0.5 * beta * structural_contrast,
         -0.5 * beta * structural_contrast],
        dtype=np.float64,
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** Tagged Partial-Independence Coalition Coding proposes that people evaluate an agreeing coalition with three psychologically separable operations. First, they apply a highly selective duplicate tag to experts whose instructed reliabilities are exactly or nearly identical. Members carrying the same duplicate tag share evidential credit, and credit losses are summed across all duplicate groups rather than propagated through a similarity network. Second, people encode reliability contextually using a small ordinal vocabulary derived from the reliability levels present in the current experiment. A coalition receives a bounded corroboration bonus for representing multiple distinct ordinal codes. This operation counts represented codes directly and never merges experts through pairwise links, so a chain of locally similar reliabilities cannot collapse transitively into one source. Third, the resulting evidence undergoes generic power-law saturation as a function of nominal coalition size, with no special transition between two and three experts. Duplicate multiplicity, ordinal-code diversity, and coalition cardinality consequently make independent graded contributions. Direct reliability weighting is weak and contrastive, allowing a sufficiently large coalition to defeat the uniquely most reliable expert while limiting broad metric reliability-gap effects. Stable cue salience varies idiosyncratically across people but is exchangeable and centered across cue identities, producing heterogeneity without population-level onset, recency, closure, or spacing mechanisms. Because choices receive no correctness feedback, these representations remain stable across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- ordinal_resolution: {2, 3}
- rank_reliance: [0.002, 0.015]
- rank_contrast_curvature: [0.55, 0.85]
- salience_strength: [0.22, 0.46]
- duplicate_tolerance: [0.006, 0.012]
- duplicate_credit_curvature: [-0.12, 0.02]
- diversity_bonus: [0.025, 0.045]
- diversity_saturation: [0.18, 0.35]
- coalition_saturation: [0.62, 0.90]
- beta: [1.50, 1.90]
- epsilon: [0.03, 0.08]

**`predict source code`:**
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

**`policy source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 7-cue design creates a parameter-robust quantitative dissociation. On every trial, the highest-validity cue discriminates and favors one option, so Take The Best stops immediately and predicts the same choice probability for that option at every evidence level: (1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, the lower-validity cues increasingly favor the other option. The tally margin opposing the top-cue winner ranges from 0 through 5. Tallying therefore predicts exactly 0.5 choice probability at margin 0 and an increasingly strong preference for the opposite option at margins 1-5, with probability (1-epsilon)*sigmoid(-beta*d)+epsilon/2 for the top-cue winner. Thus the critical signature is TTB's flat, above-chance endorsement of the top-cue winner versus Tallying's graded reversal as the opposing cue count grows. Mirrored A/B versions control for response-side bias. There are 12 unique pairs, yielding 8 repetitions each and exactly 96 trials.

[1] This design tests a validity-order invariance unique to Tallying. Every pair contains exactly three A-winning cues, three B-winning cues, and two tied cues. Tallying therefore predicts P(A)=0.5 on every trial, exactly and independently of beta and epsilon. In contrast, Take The Best finds a discriminating cue at validity rank 1, 2, or 3 and predicts the option favored by that cue with probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Thus TTB predicts a stable, above-chance directional preference even though the total cue tally is perfectly balanced, whereas Tallying predicts uniform guessing. The first discriminating cue favors A in half the pairs and B in the corresponding option-swapped mirrors, controlling response-side bias. There are two distinct evidence configurations at each stopping rank and their mirrors, yielding four trials per rank and 12 unique pairs total; each is repeated eight times for 96 trials. The critical advocated-theory signature is chance-level choice across all configurations despite large differences in which high-validity cue favors which option; TTB instead predicts complementary probabilities q and 1-q for the mirrored pairs.

[2] This 7-cue design holds the tally margin constant while systematically relocating the same 3-versus-2 cue coalition from early, low-validity positions to late, high-validity positions. In each of the first six pairs, A wins exactly three cues, B wins exactly two, and two cues tie. Tallying therefore predicts exactly the same A-choice probability on all six: (1-epsilon)*sigmoid(beta)+epsilon/2. The final six pairs are A/B-swapped mirrors, for which Tallying predicts the complementary constant probability. Accessibility-Balanced Evidence Integration instead assigns progressively greater weight to later cues; the ascending validities reinforce this positional gradient throughout the advocated parameter ranges. Consequently, its predicted A-choice probability should increase monotonically across the first six configurations. At the low end, A's three early wins should be overcome by B's two late wins, reversing the tally preference; at the high end, A's late coalition should produce a strong A preference. Coalition saturation further penalizes A's three-cue majority relative to B's two-cue coalition but is constant in form across the progression. The mirrored pairs should exhibit the complementary graded pattern, controlling response-side bias. Thus the decisive signature is a large configuration-dependent slope, including a preference reversal despite an invariant tally margin; Tallying permits neither. There are 12 unique pairs, each repeated eight times, yielding exactly 96 trials.

[3] This design isolates a positional-accessibility gradient while holding both cue tally and diagnosticity exactly constant. In each of the first six pairs, A wins one cue and B wins one cue, so Tallying predicts P(A)=0.5 exactly, independently of beta and epsilon. B's win is always at Expert 2 (index 1), whereas A's win moves successively from Expert 3 through Expert 8 (indices 2-7). All of these discriminating experts have identical advertised validity (0.70), so their diagnosticity components are equal. Accessibility-Balanced Evidence Integration therefore predicts a strictly increasing A-choice probability as A's supporting cue moves later: its evidence difference is proportional to accessibility(target)-accessibility(index 1). Coalition saturation is inactive because the coalitions are always one cue versus one cue. Across the competing theory's parameter ranges, the nearest-position contrast should be only slightly above chance (approximately 0.51-0.52), while the most separated contrast should produce approximately 0.62-0.79 A choices. The final six pairs swap A and B, yielding complementary predictions and controlling response-side bias. The high- and low-validity experts provide the required communicated validity spread but are tied in these trials, ensuring that the graded competing-theory prediction is specifically attributable to positional accessibility rather than validity. There are 12 unique pairs, each repeated eight times for 96 trials.

[4] This 10-cue design pits universal positional accessibility against switch-closure in balanced two-versus-two conflicts. Experts 1 and 4-10 all have validity 0.70; only these equal-validity experts discriminate. Experts 2 and 3 provide the required validity spread (0.95 and 0.55) but are tied, so validity differences cannot explain choices. In each of the first six pairs, A is supported by two cues including the final discriminating cue, whereas B is supported by two cues whose combined positions are later on average. Switch-Closure Coalition Integration enters its balanced-coalition branch: because A supplies the final discriminating cue and all four discriminating cues have identical validity, it predicts the same above-chance A-choice probability for all six configurations (approximately 0.58-0.66 across its parameter ranges), irrespective of spacing. Accessibility-Balanced Evidence Integration instead sums the cues' exponentially increasing positional weights. In every pair, B's two adjacent late cues have greater total accessibility than A's early-plus-terminal pair, so it robustly predicts P(A)<0.5. It additionally predicts graded changes as A's early cue moves later and the accessibility deficit shrinks, whereas the competing theory remains flat because coalition size and closure direction are unchanged. The final six pairs are exact A/B mirrors, producing complementary predictions and controlling response-side bias. Thus the decisive signature is a sign reversal—advocated-theory preference for B versus competing-theory preference for A—together with a graded spacing effect predicted only by the advocated theory. There are 12 unique pairs, each repeated eight times, for 96 trials.

[5] This 10-cue design targets the advocated theory's amplified coalition-primacy gate in unequal but nearly balanced three-versus-two conflicts. In each of the first four pairs, A receives a fixed, adjacent three-cue coalition at Experts 1-3, while B receives an adjacent two-cue coalition that moves progressively later, from Experts 4-5 through Experts 7-8. All discriminating experts have identical validity (0.70); the high- and low-validity Experts 9-10 are always tied and provide the required validity spread without affecting directional evidence. Switch-Closure Coalition Integration predicts A above chance throughout: its sublinear accumulation gives the three-cue coalition a modest numerical advantage, and its near-balance gate strongly favors A's earlier-onset coherent coalition. As B's coalition moves later, its onset-primacy score declines, so the advocated theory predicts a monotonic increase in P(A). Accessibility-Balanced Evidence Integration makes the opposite predictions. Its strong universal recency gradient gives B's two later cues more total weight than A's saturation-penalized three early cues, producing P(A)<0.5 even in the earliest configuration. Moving B later further increases its accessibility, so it predicts a monotonic decrease in P(A). Thus the theories differ both categorically in preferred option and quantitatively in the sign of the positional slope. The final four pairs are exact A/B mirrors, controlling response-side bias and yielding complementary predictions. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[6] This 10-cue design isolates the advocated theory's translation-invariant run-asymmetry mechanism in nearly balanced three-versus-two conflicts. In each of the first six trials, A is supported by three equal-validity cues and B by the fixed adjacent pair at Experts 7-8. A's coalition always begins at Expert 1 and always contains exactly one adjacent link, but its internal ordering alternates between an entry-compact form [1,1,g] and an exit-compact form [g,1], for gaps g=2, 3, and 4. Thus each matched pair has identical coalition sizes, onset positions, total validity, and conventional adjacency coherence. Switch-Closure Coalition Integration consequently predicts exactly the same A-choice probability across all six configurations: its unequal-coalition gate only represents onset primacy and the proportion of adjacent links, both of which are held constant. Reliability-Compressed Parsing Mixtures instead assigns positive centered run coherence to entry-compact coalitions and negative coherence to exit-compact coalitions. It therefore predicts higher P(A) for every entry-compact configuration than for its matched exit-compact configuration. Moreover, the entry-minus-exit contrast should increase from gap 2 to gap 4 because the compact-versus-fragmented distinction becomes more pronounced. This contrast has the same sign for every integration style, although the stable size-calibration split can shift subjects' overall preference for the three-cue coalition. The final six trials are exact A/B mirrors, so the corresponding prediction is higher endorsement of the three-cue coalition when it is entry-compact, while controlling response-side bias. Experts 9-10 provide the required validity spread but are tied; every discriminating expert has validity 0.70. Twelve unique pairs repeated eight times yield exactly 96 trials.

[7] This 16-cue design tests absolute coalition-onset sensitivity against the competing theory's translation-invariant parsing rule in nearly balanced three-versus-two conflicts. In the first six pairs, A always has a three-cue coalition with the identical internal gap pattern [1,2] (one adjacent link), while B always has an adjacent two-cue coalition. Only their locations change: A's coalition begins progressively earlier than B's in the first three configurations and later than B's in the final three. All discriminating experts have validity 0.70; Experts 15 and 16 provide the validity spread but are tied. Switch-Closure Coalition Integration holds coalition size, accumulation, coherence, and validity evidence constant while its amplified near-balance gate increasingly disfavors A as A's coalition onset moves from before to after B's. It therefore predicts a marked monotonic decline in endorsement of the three-cue coalition, with a robust early-versus-late preference reversal. Reliability-Compressed Parsing Mixtures predicts exactly invariant endorsement of the three-cue coalition across these translations for every integration style: the size ratio is fixed, the three-cue coalition's centered run coherence is unchanged, the two-cue coalition's centered run coherence is always zero, and absolute chunk position is not used in its three-versus-two branch. The last six pairs are exact A/B mirrors, permitting the critical analysis to code choice of the three-cue coalition rather than response side. Twelve unique pairs repeated eight times yield 96 trials.

[8] This 8-cue design isolates the advocated theory's population-common completion term in balanced three-versus-three conflicts. Experts 1-6 all have validity 0.70 and discriminate on every trial; Experts 7-8 have validities 0.95 and 0.55 but give tied ratings, providing the required validity spread without creating reliability or composition differences between coalitions. In each of the first eight pairs, A is supported by exactly three cues, including Expert 6, the final discriminating cue, while B is supported by the complementary three cues. The eight configurations vary which two earlier cues join Expert 6, sampling compact, fragmented, and interleaved coalitions. Reliability-Compressed Parsing Mixtures adds a completion signal favoring A in every configuration because A supplies the final discriminating cue. Its completion increment ranges from 0.323 to 0.575 before response sensitivity, implying a common above-chance closure-coalition tendency, with smaller style-dependent parsing deviations around it. Exchangeable Corroboration with Idiosyncratic Salience has no completion mechanism: coalition sizes, validities, and order-free composition loads are exactly equal, so its population prediction for the closure coalition is 0.5. It permits stable individual configuration preferences from random salience profiles, but these preferences are zero-centered and therefore should produce approximately equal numbers of positive and negative subject-level closure effects rather than the advocated theory's common positive displacement. The final eight pairs are exact A/B mirrors, controlling response-side bias while preserving which spatial coalition contains the final discriminating cue. The decisive quantitative statistics are the closure-coded grand mean and the distribution of subject-level closure effects: reliably above 0.5 and predominantly positive under the advocated theory, versus population-centered at 0.5 with sign-symmetric heterogeneity under the competing theory. Sixteen unique pairs repeated six times yield exactly 96 trials.

[9] This 16-cue design isolates the advocated theory's order-free composition-redundancy mechanism in balanced two-versus-two conflicts. In the first six pairs, A is supported by two experts having the same advertised validity, which increases across conditions from 0.68 to 0.97; B is always supported by the low-validity Experts 7 and 15 (0.55 and 0.60). Every supporting pair is separated by eight positions. Consequently, both coalitions have zero adjacency and cannot place both members in the same width-2, width-3, or width-4 parsing chunk under any competing-theory integration style. Coalition size is also exactly balanced, and the two-versus-two format prevents completion. Reliability-Compressed Parsing Mixtures is therefore reduced to additive compressed reliability evidence and predicts a modest A preference that increases monotonically as A's two validities rise. Exchangeable Corroboration with Idiosyncratic Salience instead computes the pairwise composition load. A's load rises from (2*0.68-1)^2 = 0.1296 to (2*0.97-1)^2 = 0.8836, whereas B's load remains (2*0.55-1)(2*0.60-1) = 0.02. Its strong composition discount overwhelms its very weak reliability weighting, predicting preference for the low-validity B coalition and a steep monotonic decrease in P(A) as A's validity increases. Exchangeable salience can create stable individual deviations but supplies no population-level slope tied to these translated configurations. The final six pairs are exact A/B mirrors. The primary statistic should code selection of the equal-validity coalition: the advocated theory predicts a pronounced negative validity slope, while the competing theory predicts a shallow positive slope. Twelve unique pairs repeated eight times yield exactly 96 trials.

[10] This 18-cue design isolates the advocated theory's reliability-gap-dependent latent-source inference in balanced two-versus-two conflicts. In the first eight pairs, A is supported by a target pair whose mean advertised validity is always 0.90, while B is supported by the fixed heterogeneous reference pair (0.82, 0.68). The target pair's within-pair validity gap increases from 0.00 through 0.04 and 0.10 to 0.18; each level has two order-reversed or duplicate realizations to average over idiosyncratic cue salience. Reliability-Schema Source Inference predicts a large positive gap effect: the equal extreme pair (0.90, 0.90) receives both narrow shared-source discounting and the equality-specific extreme-pair penalty, whereas increasing the gap rapidly releases the target cues from inferred overlap. At the 0.18 gap, the (0.99, 0.81) pair is treated as nearly independent, so target-pair choice should approach approximately one half despite the reference coalition. Exchangeable Corroboration with Idiosyncratic Salience predicts a much flatter profile. Its target composition load changes only from 0.64 at equality to 0.6076 at the largest gap because the target mean remains fixed, and all target loads remain far above the reference load of 0.2304. It should therefore continue strongly discounting the target coalition, generally keeping target choice around the lower quarter of the response range with only a small gap-related increase. The decisive statistic is thus the slope or equality-to-0.18 contrast in target-coalition choice: large under the advocated theory but small under the competitor, particularly producing near-chance versus clearly below-chance target choice at the widest gap. The final eight pairs are exact A/B mirrors. Sixteen unique pairs repeated six times yield exactly 96 trials.

[11] This 12-cue design tests whether the internal reliability topology of a three-expert coalition matters after coalition size, mean validity, and the advocated theory's order-free composition load have been matched. Each clustered coalition has validities (0.87, 0.87, 0.96), whereas each distributed coalition has validities approximately (0.84804, 0.90, 0.95196). On the bounded-strength scale x=2v-1, these are respectively (0.74, 0.74, 0.92) and (0.69608, 0.80, 0.90392). Both triples have the same sum, 2.40, and exactly the same mean pairwise product, 0.6364. Thus Exchangeable Corroboration with Idiosyncratic Salience gives the coalitions equal composition discounts, equal size saturation, and identical three-member corroboration calibration. Their summed compressed-reliability weights are also nearly identical throughout the advocated parameter ranges. Its population prediction is therefore approximately P(distributed)=0.50, although stable individual salience profiles can produce subject-specific deviations. Reliability-Schema Source Inference instead predicts substantially greater choice of the distributed coalition. The clustered coalition contains an exactly matched, jointly reliable 0.87 pair, activating strong narrow shared-source overlap for two of its three members. The distributed coalition has no equal pair; its adjacent validity gaps are about 0.052 and its largest gap is about 0.104, yielding markedly less overlap despite some shoulder dependence. The four crossed comparisons use two duplicate instances of each coalition type, balancing how often every expert participates and reducing contamination from cue-specific salience. Four exact A/B mirrors control response-side bias. The decisive statistic is distributed-coalition choice pooled across mirrors: approximately chance under the advocated theory but robustly above chance under the competing theory. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[12] This 12-cue design isolates the advocated theory's contextual reliability-region topology operation in equal-cardinality three-versus-three contests. There are two interchangeable clustered coalitions, each with validities (0.60, 0.60, 0.90), and two interchangeable distributed coalitions, each with validities (0.87320508, 0.70, 0.52679492). On the bounded diagnostic-strength scale x=2v-1, the clustered triple is (0.20, 0.20, 0.80), while the distributed triple is approximately (0.74641016, 0.40, 0.05358984). Both have strength sum 1.20 and sum of pairwise products 0.36, so their mean pair-composition load is exactly 0.12. Coalition cardinality, saturation, third-member calibration, and composition redundancy therefore cancel between options in both theories. The competing theory has no reliability-topology operation and should predict distributed-coalition choice very near 0.50 at the population level; its compressed reliability sums actually give the clustered coalition a tiny advantage, so it cannot explain a substantial distributed preference. In contrast, the advocated theory represents the two 0.60 clustered cues in effectively the same reliability region while the 0.90 cue occupies a distinct region. Its triadic operation consequently compresses the clustered coalition toward two informational units. The distributed triple has three widely separated represented diagnosticities and receives essentially no such compression. Equal cardinality also fully activates the advocated theory's relational topology contrast, producing a robust preference for the distributed coalition whose magnitude should increase with subject-level topology sensitivity. Four crossed comparisons make every coalition instance appear equally often, and four exact A/B mirrors control response-side bias. The primary statistic is mirror-corrected selection of the distributed coalition: clearly above chance under Contextual Reliability-Region Compression but approximately chance, or marginally below chance, under Exchangeable Corroboration. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[13] This 18-cue design maps the competing theory's reliability-region clustering function while holding constant every order-free composition statistic used by Exchangeable Corroboration. The cues form six disjoint three-expert coalitions. Writing bounded diagnostic strength as x=2v-1, every coalition has exactly the same sum of strengths (1.20) and the same sum of pairwise products (0.36), hence the same mean pair-composition load (0.12). Coalition size, saturation, third-member calibration, mean advertised validity, and composition redundancy therefore cancel. The coalitions differ only in how far an initially duplicate low-validity pair is separated: the within-pair validity gaps are 0, 0.015, 0.03, 0.06, 0.10, and 0.14. Exchangeable Corroboration has no reliability-gap or topology operation and thus predicts an approximately flat, chance-level probability of choosing the less-clustered coalition, apart from tiny compressed-reliability residuals and zero-centered subject-specific salience. Contextual Reliability-Region Compression instead predicts a nonlinear release from duplicate-pair compression. Its largest effects should occur across the first few gap increments, where represented separation crosses the 0.018-0.040 cluster-bandwidth range; comparisons between the two widest-gap coalitions should approach chance because both are already distributed. The direct zero-gap versus 0.14-gap comparison should show the strongest preference for the distributed coalition, subject to the theory's concave topology contrast. The six comparisons form a cycle, so every coalition and every expert participates equally often. Exact A/B mirrors control response-side bias. The critical quantitative signature is therefore a localized, nonlinear gap-release profile under Contextual Reliability-Region Compression versus a nearly flat profile under Exchangeable Corroboration. Twelve unique pairs repeated eight times yield exactly 96 trials.

[14] This design tests a cardinality-specific contextual onset of redundancy. Four equal-reliability coalitions contain two 0.70 experts, whereas four reliability-separated coalitions contain experts with validities 0.65 and 0.7666667. On the bounded-strength scale x=2v-1, both pair types have exactly the same product: 0.40*0.40 = 0.30*0.5333334 = 0.16. Thus, in the 2-versus-2 trials, Contextual Reliability-Region Compression assigns them the same pair-composition load and has no triadic topology operation; apart from very small compressed-validity residuals, it predicts approximately chance-level selection of the separated pair. Reliability-Space Credit Sharing instead strongly discounts the exactly matched 0.70 pair, while the 0.65/0.7666667 pair lies far outside its narrow similarity kernel, predicting a large above-chance separated-pair preference. In matched 3-versus-3 trials, one distinct 0.92 outsider is added to each coalition. This activates the advocated theory's duplicate-pair-in-context operation only for the equal 0.70 pair, as well as its equal-cardinality relational topology contrast, so it now predicts a substantial separated-coalition preference. The competing theory continues to penalize the equal pair through the same cardinality-general credit-sharing rule and therefore predicts a strong separated preference in both dyadic and triadic contests, if anything slightly attenuated after adding an independent outsider. The decisive statistic is the coalition-size interaction: Contextual Reliability-Region Compression predicts an onset from approximately 0.50 in dyads to clearly above 0.50 in triads, whereas Reliability-Space Credit Sharing predicts an already-large dyadic effect with little or negative dyad-to-triad change. Crossed coalition instances, balanced outsider assignments, and exact A/B mirrors control cue salience and response-side bias. Sixteen unique pairs repeated six times yield exactly 96 trials.

[15] This 16-cue design tests whether redundancy continues accumulating across multiple duplicate reliability clusters or saturates after detecting a duplicate-pair-in-context relation. Two double-duplicate coalitions have validities (0.60, 0.60, 0.90, 0.90), while two single-duplicate coalitions have validities (0.75, 0.75, 0.9621320344, 0.5378679656). Writing bounded diagnostic strength as x=2v-1, these are respectively (0.20, 0.20, 0.80, 0.80) and (0.50, 0.50, 0.9242640688, 0.0757359312). Both coalition types have strength sum 2.00 and sum of pairwise products 1.32, so Contextual Reliability-Region Compression gives them exactly the same pair-composition load. Their sums of compressed diagnosticities are also nearly identical throughout that theory's validity-compression range. Both contain an exact duplicate pair accompanied by reliability-distinct outsiders, making their contextual topology scores close to saturation. Because the contextual theory's topology operation removes at most approximately one informational unit, it predicts only a small preference for the single-duplicate coalition, generally near chance. Reliability-Space Credit Sharing instead applies divisive sharing separately to every coalition member. In the double-duplicate coalition, all four experts have a reliability-identical neighbor and are discounted. In the single-duplicate coalition, only the two 0.75 experts are discounted, while the extreme outsiders contribute nearly undiluted evidence. It therefore predicts a substantially larger preference for the single-duplicate coalition, roughly in the 0.62-0.69 range before subject heterogeneity is integrated. The decisive statistic is mirror-corrected selection of the single-duplicate coalition: a large displacement above chance supports cardinality-general credit accumulation, whereas a small effect supports contextual one-unit topology compression. Four crossed comparisons balance coalition instances and stable cue salience; four exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[16] This design tests the advocated theory's sharply calibrated release when effective source count crosses from two to three, using duplicate-versus-independent coalitions at cardinalities 2, 3, and 4. At each cardinality, the duplicate coalition and the reliability-diverse coalition have the same mean advertised validity: dyads compare (0.75,0.75) with (0.60,0.90), triads compare (0.75,0.75,0.60) with (0.60,0.70,0.80), and tetrads compare (0.75,0.75,0.60,0.90) with (0.60,0.70,0.80,0.90). Thus the principal manipulation is source topology rather than coalition size or mean reliability. Categorical Source-Budget Inference assigns the duplicate and diverse coalitions respectively 1 versus 2 effective sources in dyads, 2 versus 3 in triads, and 3 versus 4 in tetrads. Its saturating source-count release is concentrated specifically between two and three sources. It therefore predicts a nonmonotonic cardinality profile: preference for the diverse coalition should increase from the dyadic contrast to a maximum in the triadic contrast, where only the diverse coalition crosses the release threshold, and then decrease in the tetradic contrast, where both coalitions are already beyond that threshold. Reliability-Space Credit Sharing has no two-to-three-source gate. It continuously discounts the two identical 0.75 experts at every cardinality, while all reliability-separated experts contribute nearly independently. Generic count normalization makes the duplicate penalty's effect on coalition signal largest in the dyad and progressively smaller after independent outsiders are added. It therefore predicts the opposite dyad-to-triad slope: diverse-coalition preference should decline monotonically, or at minimum decline from dyads to triads, rather than peak at three members. The decisive statistic is the cardinality-by-topology interaction, especially the change in mirror-corrected diverse-coalition choice from dyads to triads: positive under the advocated theory and negative under the competitor. Exact A/B mirrors control response-side bias. Six unique pairs are each repeated 16 times, yielding exactly 96 trials.

[17] This design exploits a sign-reversing distinction between continuous local credit sharing and transitive categorical source assignment. Each chain coalition contains five experts with adjacent validity gaps of 0.0059: (0.7382, 0.7441, 0.7500, 0.7559, 0.7618). This gap is below the competing theory's minimum possible coassignment radius, 0.008*exp(-0.30)≈0.00593, so all five experts are joined transitively into one latent source for every allowed parameterization. Each comparison coalition instead has two exact 0.72 duplicates, one 0.75 singleton, and two exact 0.78 duplicates. Its 0.03 between-cluster gaps exceed the competing theory's maximum radius, 0.018*exp(0.30)≈0.0243, yielding exactly three sources. Categorical Source-Budget Inference should therefore strongly prefer the three-source coalition: its separate budgets add, and it receives the sharply calibrated release at three effective sources, whereas the chain receives only one concave source budget. Reliability-Space Credit Sharing predicts the opposite. On its represented-reliability scale, adjacent 0.0059 gaps lie mostly outside the extremely narrow super-Gaussian overlap band, and nonadjacent chain members have essentially zero similarity. The chain's five cues thus retain much more independent evidence than the comparison coalition, in which four cues have exact-duplicate neighbors and undergo maximal divisive discounting. Both coalitions contain five experts and have mean validity 0.75, so nominal cardinality normalization and first-order validity differences are controlled. The decisive statistic is mirror-corrected choice of the chain coalition: clearly above chance under Reliability-Space Credit Sharing but clearly below chance under Categorical Source-Budget Inference. Two instances of each topology are crossed to balance stable cue salience, and exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each, yielding exactly 96 trials. The tied 0.55 and 0.95 experts provide a broad advertised-validity spread without contributing directional evidence.

[18] This 14-expert design contrasts the advocated theory's narrow complete-link duplicate tag with the competing theory's continuous reliability-space overlap kernel. Target coalitions are near-duplicate dyads centered at validity 0.75, with either a 0.0030 gap (0.7485, 0.7515) or a 0.0058 gap (0.7471, 0.7529). Reference coalitions are exact-duplicate dyads (0.75, 0.75). Both target gaps remain below the advocated theory's minimum duplicate tolerance of 0.006, so every target and reference dyad receives the same multiplicity-two duplicate treatment. The validity levels are arranged so that both kinds of target dyad represent exactly two ordinal codes under either ordinal resolution, whereas an exact dyad represents one. Consequently, the advocated theory predicts the same small, bounded target advantage at both gaps, with essentially no 0.0030-to-0.0058 slope; symmetric ranks around 0.75 also minimize direct reliability differences. Reliability-Space Credit Sharing instead predicts a pronounced nonlinear release with increasing gap. On its compressed reliability scale, the 0.0030 pair retains appreciable local overlap, while the 0.0058 pair is substantially farther through the sharply localized super-Gaussian kernel. It therefore predicts markedly greater selection of the wide-gap target than the narrow-gap target, and a much larger wide-gap advantage over exact duplicates than the advocated theory permits. Two instances of each target type are crossed with two exact-reference instances, and exact A/B mirrors control response-side bias. The primary statistic is the mirror-corrected increase in target-coalition choice from gap 0.0030 to gap 0.0058: approximately zero under Tagged Partial-Independence Coalition Coding but robustly positive under Reliability-Space Credit Sharing. There are 16 unique pairs, each repeated six times, yielding exactly 96 trials.

[19] This design contrasts continuous local credit sharing with complete-link duplicate tagging in four-versus-four conflicts. Each chain coalition has validities (0.7413, 0.7471, 0.7529, 0.7587), with adjacent gaps of 0.0058 and a total span of 0.0174. Under Tagged Partial-Independence Coalition Coding, the chain is always partitioned into exactly two duplicate groups throughout the allowed duplicate-tolerance range: it forms two dyads below a tolerance of 0.0116 and a triad plus singleton at or above 0.0116. Each comparison coalition, (0.72, 0.72, 0.78, 0.78), also always forms exactly two duplicate groups. The two possible chain partitions have nearly identical summed group credit because duplicate-credit curvature is close to zero. Moreover, after including the tied 0.55 and 0.95 experts, both coalition types represent exactly two ordinal codes under either ordinal resolution. Coalition size and diversity bonuses therefore cancel, and the tagged theory predicts chain choice very near 0.50, with only negligible curvature and weak reliability-weight residuals. Reliability-Space Credit Sharing predicts a pronounced chain preference. Exact duplicates in the comparison coalition give every member a maximally similar neighbor and strongly reduce all four contributions. The chain's 0.0058 adjacent gaps correspond to much weaker overlap under the sharply localized super-Gaussian kernel; nonadjacent overlap is essentially absent. Its members consequently retain substantially more independent evidence despite the tagged theory assigning only two credit groups to either coalition. The decisive statistic is mirror-corrected chain-coalition choice: clearly above chance under Reliability-Space Credit Sharing but approximately chance under Tagged Partial-Independence Coalition Coding. Two instances of each topology are fully crossed to average over stable cue salience, and exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[20] This 16-expert design directly targets the competing theory's special pooling rule for equally large, one-sided duplicate-versus-chain contrasts at cardinality four. Two diverse coalitions have validities (0.60, 0.70, 0.80, 0.90), while two exact-duplicate coalitions contain four 0.75 experts. Every contest is therefore four versus four with identical mean advertised validity. Under Tagged Partial-Independence Coalition Coding, each diverse coalition forms four separate duplicate-credit groups, whereas each exact block forms one multiplicity-four group. Nominal-size saturation is identical on the two sides, so the summed independent credit and ordinal-code diversity of the diverse coalition produce a large advantage. Across the advocated parameter ranges, mirror-corrected choice of the diverse coalition should be roughly 0.76-0.90. Opponent-Relative Stratified Sampling Theory explicitly identifies these trials as large one-sided duplicate contrasts and replaces both structural values with the same pooled four-member occupancy scale. Only coalition mean quality remains. Because the diverse validities are symmetric around 0.75 in ordinal rank, their mean reliability weight is virtually identical to that of the duplicate block; hence the competing theory predicts diverse-coalition choice close to 0.50, apart from tiny convexity residuals and zero-centered individual salience. Two instances of each topology are fully crossed, and exact A/B mirrors control response-side bias. The decisive statistic is mirror-corrected selection of the diverse coalition: a large above-chance displacement supports independent accumulation of duplicate-group credit, whereas approximately chance choice supports capacity-driven large-coalition pooling. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[21] This design isolates the theories' different definitions of repetition while eliminating ordinal-code and reliability-composition confounds. Each near-duplicate coalition contains experts with validities 0.598 and 0.602, a gap of 0.004. Tagged Partial-Independence Coalition Coding must assign these experts one duplicate tag because the gap is below its minimum duplicate tolerance of 0.006. Each exact coalition contains two 0.600 experts and likewise forms one tagged duplicate group. The nine distinct validity levels are arranged so that 0.598, 0.600, and 0.602 all occupy the lowest contextual ordinal code under either ordinal resolution; they also occupy the same stratum under every advocated-theory stratum resolution. Thus the competing theory gives near and exact dyads the same duplicate-group multiplicity, nominal-size normalization, and diversity bonus. Their mean reliability is identical, leaving only negligible rank-curvature residuals; it predicts mirror-corrected near-dyad choice close to 0.50. Opponent-Relative Stratified Sampling Theory treats repetition only at exactly equal instructed levels. The exact dyad therefore has duplicate fraction 1, whereas the 0.598/0.602 dyad has duplicate fraction 0. In an equal-sized dyadic contest, the exact coalition's greater concentration produces the theory's especially strong opponent-relative vulnerability, while coverage and mean ordinal quality are matched. It consequently predicts a substantial preference for the near-duplicate coalition despite the tiny metric gap. Two instances of each coalition type are fully crossed to average over exchangeable cue salience, and exact A/B mirrors control response-side bias. The decisive statistic is mirror-corrected selection of the near-duplicate coalition: clearly above chance under Opponent-Relative Stratified Sampling but approximately chance under Tagged Partial-Independence Coalition Coding. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[22] This 20-expert design contrasts complete-link duplicate tagging with probabilistic pairwise source resolution while matching coalition cardinality, mean validity, total reliability span, and range coverage. Two replicated paired-endpoint coalitions have validities (0.5971, 0.5971, 0.6029, 0.6029), whereas two replicated fully separated coalitions have validities (0.5971, 0.5990, 0.6010, 0.6029). Every coalition has four members, mean validity 0.6000, and span 0.0058. Because that span is below the advocated theory's minimum duplicate tolerance of 0.006, complete-link tagging places all four members of either coalition type into one multiplicity-four duplicate group. The eight unique validity levels in the experiment also make ordinal diversity identical: under ordinal resolution 2 all four low levels share one code, while under resolution 3 the common endpoints ensure that both coalition types represent the same two codes. Nominal-size saturation and the diversity bonus therefore cancel, leaving Tagged Partial-Independence Coalition Coding to predict separated-coalition choice very near 0.50, apart from negligible weak-rank curvature residuals. Adaptive Source-Resolution Compression instead distinguishes exact equality from every positive reliability difference. The paired-endpoint coalition contains two exact-equality links, whereas all six pairwise links in the fully separated coalition have nonzero gaps. It consequently assigns the separated coalition a larger effective-source count and a less compressed source signal, predicting mirror-corrected separated-coalition choice above chance. The coalitions share identical endpoints, so metric range coverage is exactly controlled; any stratum-coverage difference favors the endpoint-concentrated coalition and thus makes the predicted separated preference a conservative test of source resolution. Two instances of each topology are fully crossed to average over stable cue salience, and four exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[23] This 24-expert design tests the competing theory's hard upper boundary for complete-link duplicate tagging against the advocated theory's near-invariance across nonzero reliability gaps. Two narrow dyads have validity gaps of 0.004: N1=(0.550,0.554) and N2=(0.600,0.604). Because 0.004 is below the minimum duplicate tolerance of 0.006, Tagged Partial-Independence Coalition Coding always collapses each narrow dyad into one duplicate-credit group. Two broad dyads have gaps of 0.014: B1=(0.560,0.574) and B2=(0.580,0.594). Because 0.014 exceeds the maximum duplicate tolerance of 0.012, the tagged theory always treats both broad-dyad members as separate groups. It therefore predicts a large, parameter-robust preference for the broad dyad: its approximately two units of shared credit are contrasted with approximately one unit for the narrow dyad, while nominal coalition-size saturation is identical. Adaptive Source-Resolution Compression instead treats every nonzero gap as strong evidence for source individuation. Both 0.004 and 0.014 dyads therefore have nearly two effective sources, predicting broad-dyad choice close to chance. Its only systematic broad advantage comes from the very small metric-range difference and weak positive-gap-rank modulation. The 16 higher-validity, trialwise-tied experts make 24 unique reliability levels. Consequently, all eight manipulated low-validity experts occupy the same contextual ordinal code under ordinal resolution 2 or 3, eliminating the tagged theory's code-diversity bonus. Every coalition also consists of adjacent reliability ranks, exactly matching ordinal-stratum coverage. Narrow and broad coalition instances have the same aggregate mean rank: narrow centers are ranks 0.5 and 6.5, while broad centers are 2.5 and 4.5. Fully crossing the two broad and two narrow dyads balances cue-specific salience and rank quality; exact A/B mirrors control response-side bias. The decisive statistic is mirror-corrected broad-dyad choice: strongly above 0.5 under Tagged Partial-Independence Coalition Coding but only slightly above 0.5 under Adaptive Source-Resolution Compression. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[24] This 16-expert design creates a sign-reversing test of independent duplicate-group accounting versus coherence-constrained latent-strata inference. Each compact coalition has validities (0.746, 0.748, 0.752, 0.754): it contains two tightly supported reliability neighborhoods centered near 0.747 and 0.753, with a center gap close to the competing theory's Goldilocks corroboration peak. Each dispersed coalition has validities (0.55, 0.68, 0.82, 0.95): it has the same cardinality and mean validity (0.75), but consists of four widely separated, unsupported levels. Under Tagged Partial-Independence Coalition Coding, the compact coalition is reduced to at most two complete-link duplicate groups and, for larger duplicate tolerances, to one group; the dispersed coalition always contributes four singleton credit accounts. Both coalition types represent the same number of contextual ordinal codes under ordinal resolution 2 or 3, and nominal-size saturation is identical. The advocated theory therefore predicts a strong preference for the dispersed coalition. Coherence-Constrained Reliability-Strata Inference predicts the opposite: the compact coalition supplies a small, balanced, multiply supported set of coherent strata with several gaps near the corroborative region, whereas the dispersed coalition incurs broad-consensus loss and has only weakly supported singleton strata. The critical statistic is mirror-corrected compact-coalition choice: substantially below chance under the advocated theory but above chance under the competitor. Two instances of each topology are fully crossed to average over stable cue salience, and exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[25] This 22-expert design tests replication-protected latent strata against complete-link duplicate-group accounting using a spacing series of four-member reliability chains. Two compact replicated coalitions each have validities (0.747, 0.747, 0.753, 0.753): they form two balanced, multiply supported strata separated by the advocated theory's Goldilocks gap of 0.006. Three chain coalitions all have mean validity 0.75 but adjacent gaps of 0.008, 0.014, or 0.020. Coherence-Constrained Reliability-Strata Inference should favor the compact coalition increasingly as chain spacing grows. The compact tetrad receives replication protection and supported two-stratum compressibility, whereas increasingly diffuse chains incur broad-consensus loss and cannot accumulate corroboration from every local link after the theory's higher-order stratum pooling is applied. Tagged Partial-Independence Coalition Coding predicts the opposite pattern. The compact tetrad always forms one complete-link duplicate group because its total span equals the minimum duplicate tolerance, while the chains contribute at least two groups at gap 0.008 and four singleton groups once adjacent gaps exceed the maximum tolerance at 0.014 and 0.020. Ordinal-code diversity also weakly favors, rather than penalizes, the chains. Thus the tagged theory predicts compact-coalition choice below chance and a nonpositive 0.008-to-0.020 slope, while the advocated theory predicts a positive spacing slope and especially strong compact choice against the widest chain. Every chain is compared with both compact-coalition instances to reduce cue-salience contamination, and exact A/B mirrors control response-side bias. Twelve unique pairs repeated eight times yield exactly 96 trials. Experts with validities 0.55 and 0.95 are tied on every trial and provide the advertised reliability spread.

[26] This 22-expert design targets the competing theory's cardinality-specific dyadic Goldilocks/trough mechanism while holding coalition size, mean validity, contextual-code occupancy, and cue participation tightly controlled. Two medium-gap dyads have validities (0.743, 0.757), giving a gap of 0.014 centered at 0.75; two broad-gap dyads have validities (0.70, 0.80), giving a gap of 0.10 with the same mean. In dyad-versus-dyad trials, Capacity-Gated Local Source Individuation places the 0.014 gap near its localized uncertainty trough, whereas the 0.10 gap has recovered toward baseline. It therefore predicts reduced choice of the medium-gap coalition. In matched triad-versus-triad trials, a 0.90 outsider is added to each coalition. The dyadic trough is no longer applied: the medium coalition retains a resolvable nearest local link, while all links in the broad coalition are too broad to earn appreciable local credit. The competitor consequently predicts a strong cardinality-by-gap interaction, with the medium-gap deficit shrinking substantially and potentially reversing in triads. Tagged Partial-Independence Coalition Coding has no dyad/triad gate. Both gaps exceed its maximum duplicate tolerance, so every manipulated coalition consists entirely of singleton duplicate-credit groups. Medium and broad coalitions have equal cardinality, equal mean validity, and identical contextual-code occupancy under both ordinal resolutions. The additional validity levels place the pair members symmetrically in ordinal rank and make these controls exact: under resolution 2 both pair types occupy codes 0 and 1, while under resolution 3 both occupy codes 0 and 2; adding the 0.90 outsider introduces no unmatched code. Thus the advocated theory predicts medium-gap choice close to chance and, critically, little dyad-to-triad change apart from negligible weak-rank residuals. Two instances of each topology are fully crossed, and exact A/B mirrors control stable cue salience and response-side bias. Sixteen unique pairs repeated six times yield 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 16-expert design maps the advocated theory's dyad-specific Goldilocks function against the competing theory's complete-link duplicate threshold. Four target topologies are centered at validity 0.75: exact duplicates (0.750, 0.750), narrow-gap dyads (0.748, 0.752; gap 0.004), medium-gap dyads (0.743, 0.757; gap 0.014), and broad reference dyads (0.600, 0.900; gap 0.300). Capacity-Gated Local Source Individuation predicts a nonmonotonic profile. Exact equality receives its categorical duplicate penalty; the 0.004 gap escapes equality and lies near the dyadic Goldilocks region, earning localized independence credit; the 0.014 gap lies near the uncertainty trough; and the 0.300 gap recovers to the reliability-weighted baseline. Consequently, against the broad reference it predicts low exact-dyad choice, above-chance narrow-dyad choice, and reduced medium-dyad choice. Tagged Partial-Independence Coalition Coding predicts a qualitatively different threshold profile. Both exact and 0.004 dyads always form one duplicate group because 0.004 is below its minimum duplicate tolerance of 0.006, whereas both 0.014 and 0.300 dyads always comprise two singleton groups because their gaps exceed its maximum tolerance of 0.012. The seven distinct validity levels are arranged so that every positive-gap dyad represents two contextual ordinal codes under either ordinal resolution; hence code diversity is matched. Symmetry around 0.75 also minimizes direct rank-weight differences. The competitor therefore predicts strong broad-reference preferences against exact and narrow targets, but approximately chance choice in medium-versus-broad contests. The decisive dissociations are a sign reversal for narrow-versus-broad trials\u2014narrow preferred by the advocated theory but broad preferred by the competitor\u2014and a medium-gap deficit predicted only by the advocated theory. Two disjoint instances of every topology are fully crossed to average over exchangeable cue salience. Exact A/B mirrors control response-side bias. Twenty-four unique pairs repeated four times yield exactly 96 trials.",
  "validities": [
    0.75,
    0.75,
    0.75,
    0.75,
    0.748,
    0.752,
    0.748,
    0.752,
    0.743,
    0.757,
    0.743,
    0.757,
    0.6,
    0.9,
    0.6,
    0.9
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 17065,
  "output_tokens": 5346
}
```
