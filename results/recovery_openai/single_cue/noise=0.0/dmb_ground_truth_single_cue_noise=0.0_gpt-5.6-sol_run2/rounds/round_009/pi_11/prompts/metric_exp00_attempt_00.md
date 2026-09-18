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
**Validities (n_features=14):** [0.7485, 0.7515, 0.7485, 0.7515, 0.7471, 0.7529, 0.7471, 0.7529, 0.75, 0.75, 0.75, 0.75, 0.55, 0.95]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** This 14-expert design contrasts the advocated theory's narrow complete-link duplicate tag with the competing theory's continuous reliability-space overlap kernel. Target coalitions are near-duplicate dyads centered at validity 0.75, with either a 0.0030 gap (0.7485, 0.7515) or a 0.0058 gap (0.7471, 0.7529). Reference coalitions are exact-duplicate dyads (0.75, 0.75). Both target gaps remain below the advocated theory's minimum duplicate tolerance of 0.006, so every target and reference dyad receives the same multiplicity-two duplicate treatment. The validity levels are arranged so that both kinds of target dyad represent exactly two ordinal codes under either ordinal resolution, whereas an exact dyad represents one. Consequently, the advocated theory predicts the same small, bounded target advantage at both gaps, with essentially no 0.0030-to-0.0058 slope; symmetric ranks around 0.75 also minimize direct reliability differences. Reliability-Space Credit Sharing instead predicts a pronounced nonlinear release with increasing gap. On its compressed reliability scale, the 0.0030 pair retains appreciable local overlap, while the 0.0058 pair is substantially farther through the sharply localized super-Gaussian kernel. It therefore predicts markedly greater selection of the wide-gap target than the narrow-gap target, and a much larger wide-gap advantage over exact duplicates than the advocated theory permits. Two instances of each target type are crossed with two exact-reference instances, and exact A/B mirrors control response-side bias. The primary statistic is the mirror-corrected increase in target-coalition choice from gap 0.0030 to gap 0.0058: approximately zero under Tagged Partial-Independence Coalition Coding but robustly positive under Reliability-Space Credit Sharing. There are 16 unique pairs, each repeated six times, yielding exactly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



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
**Description:** Reliability-Space Credit Sharing proposes that people interpret agreement among experts through an inferred dependence network defined entirely in represented reliability space. Advertised validities have two related but distinct cognitive representations: a strongly compressed representation determines each cue's direct evidential weight, while a separately calibrated fixed-anchor representation determines inferred dependence. Within any option-supporting coalition, every expert shares informational credit with similarly represented experts according to a smooth, sharply localized similarity kernel. An expert surrounded by reliability-near neighbors therefore contributes less independent evidence, whereas reliability-distant experts contribute nearly undiluted evidence. The steep but continuous kernel concentrates redundancy around extremely similar reliability representations, producing a strong equality penalty without a broad dyadic reliability-gap gradient. Credit sharing applies at every coalition size, without an equality detector, dyadic/triadic switch, spatial grouping rule, or duplicate-pair-in-context operation. Because additional coalition members create multiple possible similarity relations, local dependence can accumulate nonlinearly in larger coalitions even when its effect on a single nonidentical pair is weak. Effective evidence is additionally subject to generic sublinear count growth. Stable subject-level dependence sensitivity, overlap curvature, compressed validity reliance, and exchangeable zero-centered cue salience generate individual differences without population-level serial-position effects. Choices arise from response sensitivity plus an independent lapse process, and representations remain fixed because no outcome feedback is provided.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.30, 0.34]
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.008, 0.025]
- salience_strength: [0.12, 0.28]
- coalition_saturation: [0.64, 0.72]
- dependence_sensitivity: [2.90, 3.30]
- similarity_bandwidth: [0.0045, 0.0065]
- overlap_curvature: [0.88, 0.96]
- beta: [1.12, 1.38]
- epsilon: [0.03, 0.08]

**`predict source code`:**
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    narrow_choices = []\n    wide_choices = []\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        response = int(row[\"response\"])\n\n        a_support = np.flatnonzero(a > b)\n        b_support = np.flatnonzero(b > a)\n\n        # Features 0--3 form the two narrow-gap target dyads;\n        # features 4--7 form the two wide-gap target dyads.\n        a_narrow = np.any(a_support < 4)\n        b_narrow = np.any(b_support < 4)\n        a_wide = np.any((a_support >= 4) & (a_support < 8))\n        b_wide = np.any((b_support >= 4) & (b_support < 8))\n\n        if a_narrow and not b_narrow:\n            narrow_choices.append(float(response == 0))\n        elif b_narrow and not a_narrow:\n            narrow_choices.append(float(response == 1))\n        elif a_wide and not b_wide:\n            wide_choices.append(float(response == 0))\n        elif b_wide and not a_wide:\n            wide_choices.append(float(response == 1))\n\n    if len(narrow_choices) == 0 or len(wide_choices) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(wide_choices) - np.mean(narrow_choices))",
  "rationale": "This is the mirror-corrected wide-minus-narrow target-choice slope. Each response is recoded as choosing the target coalition, irrespective of whether that coalition appeared as A or B, and then averaged over all target identities, reference identities, mirrors, and repetitions. Tagged Partial-Independence Coalition Coding places both 0.0030 and 0.0058 target dyads inside the same duplicate tag and gives them the same ordinal diversity, so it predicts a slope near zero. Reliability-Space Credit Sharing predicts a substantially positive slope because overlap\u2014and hence redundancy\u2014falls sharply between these two separations. Pooling 48 observations in each gap condition per subject keeps trial-level variance low, while mirror recoding removes choice-side effects."
}
```

## Usage

```json
{
  "input_tokens": 7761,
  "output_tokens": 1677
}
```
