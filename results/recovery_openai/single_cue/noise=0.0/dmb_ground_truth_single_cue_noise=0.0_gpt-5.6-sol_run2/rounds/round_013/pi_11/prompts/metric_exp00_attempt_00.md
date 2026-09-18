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
**Validities (n_features=22):** [0.743, 0.757, 0.743, 0.757, 0.7, 0.8, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.55, 0.6, 0.65, 0.746, 0.748, 0.75, 0.752, 0.754, 0.86, 0.98]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This 22-expert design targets the competing theory's cardinality-specific dyadic Goldilocks/trough mechanism while holding coalition size, mean validity, contextual-code occupancy, and cue participation tightly controlled. Two medium-gap dyads have validities (0.743, 0.757), giving a gap of 0.014 centered at 0.75; two broad-gap dyads have validities (0.70, 0.80), giving a gap of 0.10 with the same mean. In dyad-versus-dyad trials, Capacity-Gated Local Source Individuation places the 0.014 gap near its localized uncertainty trough, whereas the 0.10 gap has recovered toward baseline. It therefore predicts reduced choice of the medium-gap coalition. In matched triad-versus-triad trials, a 0.90 outsider is added to each coalition. The dyadic trough is no longer applied: the medium coalition retains a resolvable nearest local link, while all links in the broad coalition are too broad to earn appreciable local credit. The competitor consequently predicts a strong cardinality-by-gap interaction, with the medium-gap deficit shrinking substantially and potentially reversing in triads. Tagged Partial-Independence Coalition Coding has no dyad/triad gate. Both gaps exceed its maximum duplicate tolerance, so every manipulated coalition consists entirely of singleton duplicate-credit groups. Medium and broad coalitions have equal cardinality, equal mean validity, and identical contextual-code occupancy under both ordinal resolutions. The additional validity levels place the pair members symmetrically in ordinal rank and make these controls exact: under resolution 2 both pair types occupy codes 0 and 1, while under resolution 3 both occupy codes 0 and 2; adding the 0.90 outsider introduces no unmatched code. Thus the advocated theory predicts medium-gap choice close to chance and, critically, little dyad-to-triad change apart from negligible weak-rank residuals. Two instances of each topology are fully crossed, and exact A/B mirrors control stable cue salience and response-side bias. Sixteen unique pairs repeated six times yield 96 trials.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.asarray([\n        0.743, 0.757, 0.743, 0.757, 0.700, 0.800, 0.700, 0.800,\n        0.900, 0.900, 0.900, 0.900, 0.550, 0.600, 0.650, 0.746,\n        0.748, 0.750, 0.752, 0.754, 0.860, 0.980\n    ], dtype=np.float64)\n\n    dyad_medium_choices = []\n    triad_medium_choices = []\n\n    def minimum_gap(indices):\n        vals = np.sort(validities[np.asarray(indices, dtype=np.int64)])\n        if vals.size < 2:\n            return np.inf\n        return float(np.min(np.diff(vals)))\n\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=np.float64)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=np.float64)\n        difference = a - b\n        a_support = np.flatnonzero(difference > 0.0)\n        b_support = np.flatnonzero(difference < 0.0)\n\n        if a_support.size != b_support.size or a_support.size not in (2, 3):\n            continue\n\n        gap_a = minimum_gap(a_support)\n        gap_b = minimum_gap(b_support)\n        if not np.isfinite(gap_a) or not np.isfinite(gap_b) or gap_a == gap_b:\n            continue\n\n        medium_is_a = gap_a < gap_b\n        chose_a = int(row[\"response\"]) == 0\n        chose_medium = float(chose_a == medium_is_a)\n\n        if a_support.size == 2:\n            dyad_medium_choices.append(chose_medium)\n        else:\n            triad_medium_choices.append(chose_medium)\n\n    if len(dyad_medium_choices) == 0 or len(triad_medium_choices) == 0:\n        return 0.0\n\n    return float(\n        np.mean(triad_medium_choices) - np.mean(dyad_medium_choices)\n    )",
  "rationale": "This is the within-subject cardinality-by-reliability-gap interaction: the rate of choosing the medium-gap coalition in triad conflicts minus its rate in matched dyad conflicts. Coalition type is recovered from the minimum within-coalition validity gap, making the measure invariant to whether that coalition appears as A or B. Capacity-Gated Local Source Individuation predicts a positive interaction because its localized uncertainty trough selectively depresses the 0.014-gap dyads, whereas the trough is released for triads. Tagged Partial-Independence Coalition Coding has no dyad/triad gate and treats both manipulated gaps as beyond its duplicate tolerance, predicting little systematic change. The exact A/B mirrors and fully crossed coalition instances make this difference score resistant to response-side bias and stable cue-specific preferences."
}
```

## Usage

```json
{
  "input_tokens": 9953,
  "output_tokens": 2453
}
```
