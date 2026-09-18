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
**Validities (n_features=16):** [0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This 16-expert design directly targets the competing theory's special pooling rule for equally large, one-sided duplicate-versus-chain contrasts at cardinality four. Two diverse coalitions have validities (0.60, 0.70, 0.80, 0.90), while two exact-duplicate coalitions contain four 0.75 experts. Every contest is therefore four versus four with identical mean advertised validity. Under Tagged Partial-Independence Coalition Coding, each diverse coalition forms four separate duplicate-credit groups, whereas each exact block forms one multiplicity-four group. Nominal-size saturation is identical on the two sides, so the summed independent credit and ordinal-code diversity of the diverse coalition produce a large advantage. Across the advocated parameter ranges, mirror-corrected choice of the diverse coalition should be roughly 0.76-0.90. Opponent-Relative Stratified Sampling Theory explicitly identifies these trials as large one-sided duplicate contrasts and replaces both structural values with the same pooled four-member occupancy scale. Only coalition mean quality remains. Because the diverse validities are symmetric around 0.75 in ordinal rank, their mean reliability weight is virtually identical to that of the duplicate block; hence the competing theory predicts diverse-coalition choice close to 0.50, apart from tiny convexity residuals and zero-centered individual salience. Two instances of each topology are fully crossed, and exact A/B mirrors control response-side bias. The decisive statistic is mirror-corrected selection of the diverse coalition: a large above-chance displacement supports independent accumulation of duplicate-group credit, whereas approximately chance choice supports capacity-driven large-coalition pooling. Eight unique pairs are repeated 12 times each for exactly 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Opponent-Relative Stratified Sampling Theory proposes that people compress ordered expert reliabilities into a small contextual vocabulary of reliability strata and jointly compare opposing coalitions by sampled evidence, stratum coverage, balance, and relative concentration. Each stratum has a limited number of representational slots, while a second capacity limit constrains total evidence maintained across strata. Coalition mass is partially normalized, allowing heterogeneous dyad, triad, and tetrad effects without a single global coalition-size power law. Repetition is not intrinsically penalized: its effect depends on the rival coalition's concentration, coverage, and size. This opponent-relative contrast is strongest for diagnostic dyads. When equal-sized dyads have matched duplicate status, their individual mean qualities are partially assimilated toward a pooled comparison standard, attenuating arbitrary ordinal-composition contrasts without eliminating reliability information. For equally large coalitions of at least four experts, a one-sided duplicate-versus-chain contrast is pooled through the same capacity-limited representation, preventing generic diversity from producing a chain advantage. Concentration comparisons remain available when both large coalitions contain repetition but differ in its degree. Weak ordinal reliability weighting and exchangeable centered cue salience preserve sensitivity to instructed reliability without metric-distance, chain, onset, recency, closure, or adjacency effects.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- stratum_resolution: {2, 3, 4}
- sampling_capacity: {1, 2, 3}
- global_capacity: [2.2, 4.8]
- occupancy_normalization: [0.48, 0.80]
- ordinal_reliance: [0.005, 0.035]
- salience_strength: [0.08, 0.24]
- coverage_preference: [0.14, 0.38]
- balance_preference: [0.08, 0.30]
- redundancy_sensitivity: [0.65, 1.45]
- rival_size_weight: [0.20, 0.55]
- context_curvature: [0.70, 1.30]
- matched_dyad_quality_shrinkage: [0.18, 0.38]
- beta: [1.25, 1.70]
- epsilon: [0.03, 0.08]

**`predict source code`:**
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Fraction of subjects showing a supermajority preference for diverse coalitions.\"\"\"\n    if data is None or len(data) == 0:\n        return 0.0\n\n    records = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size != 16 or b.size != 16:\n            continue\n\n        diff = a - b\n        a_support = np.flatnonzero(diff > 0)\n        b_support = np.flatnonzero(diff < 0)\n\n        # In this design, indices 0--7 form the two reliability-diverse\n        # coalitions, whereas indices 8--15 form the exact-duplicate blocks.\n        a_diverse = (a_support.size == 4 and np.all(a_support < 8))\n        b_diverse = (b_support.size == 4 and np.all(b_support < 8))\n        a_duplicate = (a_support.size == 4 and np.all(a_support >= 8))\n        b_duplicate = (b_support.size == 4 and np.all(b_support >= 8))\n\n        if a_diverse and b_duplicate:\n            chose_diverse = int(row[\"response\"] == 0)\n        elif b_diverse and a_duplicate:\n            chose_diverse = int(row[\"response\"] == 1)\n        else:\n            continue\n\n        records.append((row[\"subject_id\"], chose_diverse))\n\n    if len(records) == 0:\n        return 0.0\n\n    scored = pd.DataFrame(records, columns=[\"subject_id\", \"chose_diverse\"])\n    subject_rates = scored.groupby(\"subject_id\")[\"chose_diverse\"].mean()\n\n    # A 2/3 threshold lies well above the competing theory's near-chance\n    # prediction but below the advocated theory's predicted range.\n    return float(np.mean(subject_rates >= (2.0 / 3.0)))",
  "rationale": "The metric is the prevalence of subjects who choose the reliability-diverse coalition on at least two thirds of the 96 trials. Diverse choice is coded relative to the stimulus, so exact A/B mirrors remove response-side bias, while crossing both diverse and both duplicate blocks averages over coalition identity. Tagged Partial-Independence Coalition Coding predicts diverse-choice rates around 0.76\u20130.90, making the supermajority criterion likely for nearly every subject. Opponent-Relative Stratified Sampling instead pools these large one-sided duplicate contrasts and predicts rates near 0.50, making a 2/3 rate very unlikely over 96 observations. The subject-level threshold converts the predicted separation into a low-variance prevalence contrast while remaining an empirically falsifiable behavioral endpoint rather than a direct restatement of either model's computations."
}
```

## Usage

```json
{
  "input_tokens": 8825,
  "output_tokens": 1854
}
```
