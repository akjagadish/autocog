# metric_exp00_attempt_02

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
**Validities (n_features=12):** [0.5, 0.6, 0.7, 0.742, 0.746, 0.748, 0.752, 0.754, 0.758, 0.8, 0.9, 0.99]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Rationale:** This 12-expert design creates a quantitative diversity-coverage gradient while matching coalition cardinality, duplicate status, ordinal-code diversity, and mean ordinal reliability. Four disjoint two-expert coalitions are symmetric in reliability rank but differ sharply in advertised-validity span: C1=(0.50,0.99), C2=(0.60,0.90), C3=(0.70,0.80), and C4=(0.742,0.758). Every within-coalition gap exceeds the competing theory's maximum duplicate tolerance of 0.012, so Tagged Partial-Independence Coalition Coding treats all members as separate singleton groups. Under either ordinal resolution, every coalition represents exactly two ordinal codes. Because the endpoints of each coalition occupy rank-symmetric positions, summed contextual reliability weights are virtually equal; nominal size saturation and diversity bonuses also cancel. The competing theory therefore predicts choices close to 0.50 across all six pairwise coalition comparisons, with only negligible rank-curvature residuals. Adaptive Source-Resolution Compression instead assigns progressively greater range and ordinal-stratum coverage to C4<C3<C2<C1. Its diversity benefit should generate an ordered choice gradient favoring the broader coalition, with effect size increasing with the coverage contrast; small positive-gap effects on inferred independence reinforce this ordering. All six unordered comparisons are included along with exact A/B mirrors. The complete pairwise design gives each coalition equal participation and permits estimation of a monotonic coverage score while averaging exchangeable cue salience. The decisive signature is a sizable graded increase in broad-coalition choice from adjacent through extreme contrasts under the advocated theory versus an essentially flat, near-chance profile under the competitor. Twelve unique pairs are repeated eight times each for exactly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Adaptive Source-Resolution Compression with Allocated Source Quality proposes that observers represent an agreeing coalition as a probabilistic collection of latent information sources rather than as a tally of experts. Exact equality in communicated reliability is a discrete but non-deterministic common-source cue, whereas any nonzero reliability difference supports source individuation. Pairwise dependence posteriors determine both a continuous effective-source count and cue-specific uniqueness credits. These credits allocate the inferred source count across coalition members: a cue strongly dependent on its partners receives less ownership of source evidence, while an individuated cue retains more ownership. Coalition quality is therefore computed from the reliability and salience of the cues that own independent-source credit, rather than from the unadjusted arithmetic mean of all agreeing cues. The first inferred source is represented fully, while additional sources undergo subject-specific, smoothly saturating compression. Reliability-range and ordinal-stratum coverage provide a diversity benefit that decays with coalition load. The same quality-adjusted compressed source representation enters opponent-relative conflict, allowing several independent moderate cues to outweigh a highly reliable but redundant cue without introducing cardinality-specific rules. Stable heterogeneity in equality inference, compression capacity, diversity sensitivity, source-credit allocation, conflict weighting, and exchangeable cue salience produces graded population prevalence and preserves positional exchangeability.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- source_dependence_prior: [-2.20, 0.10]
- equality_evidence: [1.80, 5.20]
- nonzero_separation: [3.20, 5.80]
- separation_floor: [0.58, 0.78]
- positive_gap_rank_gain: [0.03, 0.14]
- compression_capacity: [1.00, 2.80]
- load_individuation: [0.18, 0.55]
- source_signal_curvature: [0.40, 0.66]
- source_allocation_curvature: [0.70, 1.60]
- rank_reliance: [0.008, 0.035]
- diversity_sensitivity: [0.12, 0.85]
- coverage_mix: [0.45, 0.75]
- salience_strength: [0.12, 0.34]
- conflict_gain: [0.04, 0.16]
- conflict_scale: [0.55, 1.10]
- best_cue_tradeoff: [0.05, 0.20]
- beta: [1.35, 1.90]
- epsilon: [0.025, 0.070]

**`predict source code`:**
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
[0] rationale: For each subject, the metric first estimates a balanced choice strength for each of the four coalitions from all of its appearances in the complete pairwise schedule. It then computes a Kendall-like concordance between those strengths and coalition reliability breadth (expert-index spans 5, 7, 9, and 11). Adaptive Source-Resolution Compression predicts a coherent increasing ordering because range and ordinal-stratum coverage raise coalition value. The competing theory predicts no common ordering: its matched ordinal-code counts, cardinality, and nearly matched contextual reliability weights leave rankings mainly determined by exchangeable subject-specific salience. Using ordinal concordance rather than a raw extreme-pair choice rate reduces sensitivity to the magnitude of individual salience and diversity parameters, while pooling all comparisons and repetitions stabilizes each subject's coalition ranking. On pooled data the function explicitly averages subject-level concordances, ensuring that its canonical estimate corresponds to the same bounded statistic used for between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def subject_concordance(df):
        # Coalition breadth is indexed by the separation between its two
        # supporting experts. In this design the four values are 5, 7, 9, 11.
        chosen_breadths = []
        presented_breadths = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            ia = np.flatnonzero(a > 0.5)
            ib = np.flatnonzero(b > 0.5)
            if ia.size < 2 or ib.size < 2:
                continue

            ba = float(np.max(ia) - np.min(ia))
            bb = float(np.max(ib) - np.min(ib))
            presented_breadths.extend([ba, bb])
            chosen_breadths.append(ba if int(row["response"]) == 0 else bb)

        if len(chosen_breadths) == 0:
            return 0.0

        levels = np.sort(np.unique(np.asarray(presented_breadths, dtype=float)))
        if levels.size < 2:
            return 0.0

        # Balanced Borda-style choice score for each coalition: proportion of
        # its appearances on which it was selected.
        appearances = {float(x): 0.0 for x in levels}
        wins = {float(x): 0.0 for x in levels}
        k = 0
        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            ia = np.flatnonzero(a > 0.5)
            ib = np.flatnonzero(b > 0.5)
            if ia.size < 2 or ib.size < 2:
                continue
            ba = float(np.max(ia) - np.min(ia))
            bb = float(np.max(ib) - np.min(ib))
            if ba not in appearances or bb not in appearances:
                continue
            appearances[ba] += 1.0
            appearances[bb] += 1.0
            if int(row["response"]) == 0:
                wins[ba] += 1.0
            else:
                wins[bb] += 1.0
            k += 1

        scores = []
        kept_levels = []
        for x in levels:
            x = float(x)
            if appearances[x] > 0.0:
                kept_levels.append(x)
                scores.append(wins[x] / appearances[x])

        scores = np.asarray(scores, dtype=float)
        kept_levels = np.asarray(kept_levels, dtype=float)
        if scores.size < 2:
            return 0.0

        # Kendall-like concordance between advertised-validity breadth and
        # empirical coalition strength. Exact empirical ties contribute zero.
        vals = []
        for i in range(scores.size):
            for j in range(i + 1, scores.size):
                direction = (kept_levels[j] - kept_levels[i]) * (scores[j] - scores[i])
                if direction > 0.0:
                    vals.append(1.0)
                elif direction < 0.0:
                    vals.append(-1.0)
                else:
                    vals.append(0.0)
        return float(np.mean(vals)) if len(vals) else 0.0

    if len(data) == 0:
        return 0.0
    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = [subject_concordance(g) for _, g in data.groupby("subject_id", sort=False)]
        return float(np.mean(values))
    return subject_concordance(data)

outcome: self_sim=0.3033 (var=0.2174) adversary_sim=0.0333 (var=0.2278) welch_t=+2.023 p=0.04864 (N=25, alpha=0.01) -> reject

[1] rationale: This is a continuous, trial-level regression coefficient for choice direction on the difference in coalition coverage. Coverage combines advertised-validity range and contextual rank-stratum spread using the midpoint of the advocated theory's coverage_mix interval. The advocated theory makes its diversity contribution approximately linear in this coverage quantity, so weighting each response by the signed coverage contrast is close to a matched-filter contrast: extreme coalition comparisons receive more weight because they carry more predicted signal. The competing theory matches coalition size and ordinal-code diversity and therefore predicts a coefficient near zero apart from weak reliability and exchangeable-salience residuals. Unlike the previously failed discontinuous rank-concordance statistic, this metric retains response magnitude information from all 96 trials, treats exact A/B mirrors identically, and averages repeated Bernoulli choices directly. That should substantially reduce per-subject variance while preserving the theoretically diagnostic graded effect. The statistic is not tautological: it measures whether observed choices covary with coverage, rather than merely computing coverage from the stimuli.
metric_source:
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    # Advertised validities and contextual ranks from the fixed design.
    validities = np.asarray(
        [0.5, 0.6, 0.7, 0.742, 0.746, 0.748,
         0.752, 0.754, 0.758, 0.8, 0.9, 0.99],
        dtype=float,
    )
    ranks = (np.arange(validities.size, dtype=float) + 0.5) / float(validities.size)
    task_range = float(np.max(validities) - np.min(validities))
    task_rank_sd = float(np.std(ranks))

    def coverage(ratings):
        x = np.asarray(ratings, dtype=float)
        idx = np.flatnonzero(x > 0.5)
        if idx.size <= 1:
            return 0.0

        range_part = (
            float(np.max(validities[idx]) - np.min(validities[idx]))
            / max(task_range, 1e-12)
        )
        stratum_part = float(
            np.clip(np.std(ranks[idx]) / max(task_rank_sd, 1e-12), 0.0, 1.0)
        )

        # Midpoint of the advocated theory's coverage-mix range.
        return 0.60 * range_part + 0.40 * stratum_part

    numerator = 0.0
    denominator = 0.0

    for _, row in data.iterrows():
        ca = coverage(row["option_a_ratings"])
        cb = coverage(row["option_b_ratings"])
        contrast = ca - cb
        if not np.isfinite(contrast) or abs(contrast) <= 1e-12:
            continue

        # +1 denotes choosing A and -1 choosing B. Thus the product is
        # positive exactly when the higher-coverage coalition is selected.
        choice_sign = 1.0 if int(row["response"]) == 0 else -1.0
        numerator += contrast * choice_sign
        denominator += contrast * contrast

    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)

outcome: self_sim=0.2790 (var=0.1175) adversary_sim=0.0394 (var=0.0967) welch_t=+2.589 p=0.01274 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if data is None or len(data) == 0:\n        return 0.0\n\n    validities = np.asarray(\n        [0.5, 0.6, 0.7, 0.742, 0.746, 0.748,\n         0.752, 0.754, 0.758, 0.8, 0.9, 0.99],\n        dtype=float,\n    )\n    ranks = (np.arange(validities.size, dtype=float) + 0.5) / float(validities.size)\n    task_range = float(np.max(validities) - np.min(validities))\n    task_rank_sd = float(np.std(ranks))\n\n    def coalition_coverage(idx):\n        idx = np.asarray(idx, dtype=int)\n        if idx.size <= 1:\n            return 0.0\n        range_coverage = float(\n            (np.max(validities[idx]) - np.min(validities[idx]))\n            / max(task_range, 1e-12)\n        )\n        stratum_coverage = float(np.clip(\n            np.std(ranks[idx]) / max(task_rank_sd, 1e-12), 0.0, 1.0\n        ))\n        return 0.60 * range_coverage + 0.40 * stratum_coverage\n\n    # The four coalitions in increasing expert-index order.\n    coalition_keys = [(0, 11), (1, 10), (2, 9), (3, 8)]\n    coverages = np.asarray(\n        [coalition_coverage(k) for k in coalition_keys], dtype=float\n    )\n    key_to_index = {k: i for i, k in enumerate(coalition_keys)}\n\n    def one_subject(df):\n        appearances = np.zeros(4, dtype=float)\n        wins = np.zeros(4, dtype=float)\n\n        for _, row in df.iterrows():\n            a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n            b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n            ka = tuple(int(x) for x in np.flatnonzero(a > 0.5))\n            kb = tuple(int(x) for x in np.flatnonzero(b > 0.5))\n            if ka not in key_to_index or kb not in key_to_index:\n                continue\n\n            ia = key_to_index[ka]\n            ib = key_to_index[kb]\n            appearances[ia] += 1.0\n            appearances[ib] += 1.0\n            if int(row[\"response\"]) == 0:\n                wins[ia] += 1.0\n            else:\n                wins[ib] += 1.0\n\n        keep = appearances > 0.0\n        if np.sum(keep) < 3:\n            return 0.0\n\n        strength = wins[keep] / appearances[keep]\n        c = coverages[keep]\n        c_centered = c - float(np.mean(c))\n        denom = float(np.dot(c_centered, c_centered))\n        if denom <= 1e-12:\n            return 0.0\n\n        # Coverage-aligned coalition-strength slope.\n        slope = float(\n            np.dot(c_centered, strength - float(np.mean(strength))) / denom\n        )\n\n        # Estimate the subject's coalition preferences that cannot be\n        # explained by the predicted coverage ordering. These residuals are a\n        # proxy for exchangeable cue-salience contamination. Smooth shrinkage\n        # prevents strongly idiosyncratic subjects from dominating the scalar.\n        fitted = float(np.mean(strength)) + slope * c_centered\n        residual_mse = float(np.mean((strength - fitted) ** 2))\n        shrinkage = 1.0 + 40.0 * max(residual_mse, 0.0)\n        return float(slope / shrinkage)\n\n    if \"subject_id\" in data.columns and data[\"subject_id\"].nunique() > 1:\n        values = [\n            one_subject(g)\n            for _, g in data.groupby(\"subject_id\", sort=False)\n        ]\n        return float(np.mean(values)) if len(values) else 0.0\n\n    return one_subject(data)\n",
  "rationale": "This metric estimates each coalition's balanced win rate and projects the resulting four-coalition strength profile onto the advocated theory's coverage gradient. Unlike the prior raw trial-level coefficient, it also estimates preference structure orthogonal to coverage. Such residual structure is expected when exchangeable cue salience produces arbitrary coalition preferences, especially under the competing theory's stronger salience range. The coverage slope is therefore smoothly shrunk for subjects whose choices exhibit substantial non-coverage coalition structure. This retains a coherent common coverage gradient while reducing between-subject variance from idiosyncratic salience. The pooled statistic is the mean of exactly the same bounded, regularized subject-level statistic used in the variance calculation."
}
```

## Usage

```json
{
  "input_tokens": 10228,
  "output_tokens": 5499
}
```
