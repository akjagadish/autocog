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
**Description:** Coherence-Constrained Reliability-Strata Inference proposes that observers interpret an agreeing coalition by inferring a compact posterior distribution over latent reliability strata. Experts are assigned continuously rather than categorically to one through four possible source strata. Exact or virtually exact experts have a high probability of sharing a source and therefore incur accumulating redundancy. A small but nonzero reliability difference instead provides maximal corroboration because it supports distinguishable sources that remain mutually coherent. Larger separations progressively lose consensus coherence, producing a Goldilocks-shaped relation between reliability gap and coalition value. Coalition structure also matters: a small, balanced collection of compact strata supported by multiple experts is easier to compress and receives greater confidence than either one undifferentiated duplicate mass or many weakly supported singleton levels. Replication protection is granted only when posterior cluster mass is concentrated tightly around a stratum center; diffuse chains cannot obtain this protection merely through broad coassignment. For coalitions beyond triadic capacity, independence credit is pooled at the latent-stratum level and smoothly limited by the strongest compact, balanced, supported relation between distinguishable strata. Thus a chain cannot accumulate corroboration from every locally narrow expert pair. A dyad can progress from redundant at equality, to coherently corroborative at a narrow gap, and back toward weak consensus at a broader gap. A replicated two-stratum tetrad can outperform four unsupported reliability levels, although each compact stratum's excess membership adds a separate redundancy cost. Instructed reliability has a weak but consequential bounded influence after compression. Coalition magnitude saturates smoothly under finite capacity, with a modest corroborative release at the two-to-three transition that decays beyond three. Exchangeable idiosyncratic cue salience introduces individual variation without population-level position, onset, closure, adjacency, or recency effects.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.08, 0.18]
- validity_reliance: [0.020, 0.065]
- salience_strength: [0.10, 0.28]
- redundancy_width: [0.0015, 0.0045]
- redundancy_cost: [0.70, 1.35]
- compact_excess_cost: [0.06, 0.20]
- corroboration_peak: [0.004, 0.007]
- corroboration_width: [0.003, 0.007]
- independence_benefit: [0.18, 0.55]
- coherence_width: [0.012, 0.040]
- coherence_cost: [0.28, 0.72]
- cluster_width: [0.004, 0.014]
- cluster_complexity: [0.10, 0.28]
- cluster_temperature: [0.45, 0.90]
- support_threshold: [1.35, 1.75]
- support_slope: [4.0, 8.0]
- cluster_support_bonus: [0.30, 0.82]
- capacity_saturation: [0.90, 3.10]
- count_curvature: [0.40, 0.82]
- triad_transition_gain: [0.04, 0.24]
- beta: [1.25, 1.75]
- epsilon: [0.025, 0.070]

**`predict source code`:**
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

**`policy source code`:**
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def subject_score(df):\n        compact_sides = []\n        compact_choices = []\n\n        for _, row in df.iterrows():\n            a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n            b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n            a_support = np.flatnonzero(a > b)\n            b_support = np.flatnonzero(b > a)\n\n            a_is_compact = a_support.size > 0 and np.all(a_support < 8)\n            b_is_compact = b_support.size > 0 and np.all(b_support < 8)\n            if a_is_compact == b_is_compact:\n                continue\n\n            compact_side = 0 if a_is_compact else 1\n            compact_sides.append(compact_side)\n            compact_choices.append(int(row[\"response\"]) == compact_side)\n\n        if len(compact_choices) == 0:\n            return 0.5\n\n        compact_sides = np.asarray(compact_sides, dtype=int)\n        compact_choices = np.asarray(compact_choices, dtype=float)\n\n        side_rates = []\n        for side in (0, 1):\n            mask = compact_sides == side\n            if np.any(mask):\n                side_rates.append(float(np.mean(compact_choices[mask])))\n\n        compact_rate = float(np.mean(side_rates))\n        dispersed_rate = 1.0 - compact_rate\n\n        if dispersed_rate > 0.5:\n            return 1.0\n        if dispersed_rate < 0.5:\n            return 0.0\n        return 0.5\n\n    if len(data) == 0:\n        return 0.5\n\n    scores = [subject_score(df) for _, df in data.groupby(\"subject_id\", sort=False)]\n    return float(np.mean(scores))",
  "rationale": "This is the prevalence of subjects whose mirror-corrected choices show a majority preference for the dispersed coalition. Compact-coalition choices are averaged separately when compact is A and when compact is B, then those rates are equally averaged, eliminating stable response-side bias. The advocated theory predicts a robust dispersed majority because four singleton credit accounts dominate the duplicate-tagged compact coalition; the competitor predicts the reverse because the compact, replicated reliability strata are more coherent. Thresholding each subject's estimate after aggregating all 96 trials converts this sign-reversing prediction into a stable subject-level classification, reducing sensitivity to binomial choice noise and idiosyncratic cue salience while remaining entirely response-dependent."
}
```

## Usage

```json
{
  "input_tokens": 10086,
  "output_tokens": 2139
}
```
