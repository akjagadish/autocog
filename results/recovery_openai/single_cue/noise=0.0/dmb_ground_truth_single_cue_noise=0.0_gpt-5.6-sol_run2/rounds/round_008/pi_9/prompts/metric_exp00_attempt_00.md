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
**Validities (n_features=22):** [0.7382, 0.7441, 0.75, 0.7559, 0.7618, 0.7382, 0.7441, 0.75, 0.7559, 0.7618, 0.72, 0.72, 0.75, 0.78, 0.78, 0.72, 0.72, 0.75, 0.78, 0.78, 0.55, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design exploits a sign-reversing distinction between continuous local credit sharing and transitive categorical source assignment. Each chain coalition contains five experts with adjacent validity gaps of 0.0059: (0.7382, 0.7441, 0.7500, 0.7559, 0.7618). This gap is below the competing theory's minimum possible coassignment radius, 0.008*exp(-0.30)≈0.00593, so all five experts are joined transitively into one latent source for every allowed parameterization. Each comparison coalition instead has two exact 0.72 duplicates, one 0.75 singleton, and two exact 0.78 duplicates. Its 0.03 between-cluster gaps exceed the competing theory's maximum radius, 0.018*exp(0.30)≈0.0243, yielding exactly three sources. Categorical Source-Budget Inference should therefore strongly prefer the three-source coalition: its separate budgets add, and it receives the sharply calibrated release at three effective sources, whereas the chain receives only one concave source budget. Reliability-Space Credit Sharing predicts the opposite. On its represented-reliability scale, adjacent 0.0059 gaps lie mostly outside the extremely narrow super-Gaussian overlap band, and nonadjacent chain members have essentially zero similarity. The chain's five cues thus retain much more independent evidence than the comparison coalition, in which four cues have exact-duplicate neighbors and undergo maximal divisive discounting. Both coalitions contain five experts and have mean validity 0.75, so nominal cardinality normalization and first-order validity differences are controlled. The decisive statistic is mirror-corrected choice of the chain coalition: clearly above chance under Reliability-Space Credit Sharing but clearly below chance under Categorical Source-Budget Inference. Two instances of each topology are crossed to balance stable cue salience, and exact A/B mirrors control response-side bias. Eight unique pairs are repeated 12 times each, yielding exactly 96 trials. The tied 0.55 and 0.95 experts provide a broad advertised-validity spread without contributing directional evidence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Categorical Source-Budget Inference proposes that people recode instructed expert reliabilities into a small number of latent information-source categories. Category membership is inferred locally: experts with exactly or nearly matching represented reliabilities are likely to be treated as exchangeable observations from one source, whereas this coassignment declines sharply outside a narrow subject-specific neighborhood. Within each option-supporting coalition, members assigned to one source draw from a shared evidence budget whose value grows as a strongly concave power of cluster membership. Exact reliability matches additionally invoke a categorical equality atom, producing an almost flat shared budget without changing how nonzero reliability gaps are partitioned. Budgets belonging to distinct inferred sources are added. A duplicate pair is therefore redundant in a dyad, an unrelated outsider adds a separate budget without changing that duplicate cost, and two duplicate clusters consume two separate budgets. The total is then subjected to generic nominal-coalition normalization and a separately calibrated saturating release that is concentrated between two and three effective sources. Instructed validity differences have little direct influence after strong compression. Stable, exchangeable, zero-centered cue salience permits individual configuration preferences but creates no population-level serial-position mechanism. With no correctness feedback, source assignments and cue salience remain fixed across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- direct_validity_compression: [0.08, 0.14]
- validity_reliance: [0.006, 0.024]
- salience_strength: [0.18, 0.40]
- category_width: [0.008, 0.018]
- category_boundary_shift: [-0.30, 0.30]
- within_source_curvature: [0.08, 0.22]
- equality_source_curvature: [0.00, 0.05]
- coalition_normalization: [0.88, 0.98]
- source_count_curvature: [0.18, 0.26]
- source_count_midpoint: [2.45, 2.65]
- source_count_slope: [7.0, 11.0]
- beta: [1.45, 1.85]
- epsilon: [0.03, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Categorical Source-Budget Inference expects shape "
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

    direct_validity_compression = float(
        parameters["direct_validity_compression"]
    )
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    category_width = float(parameters["category_width"])
    category_boundary_shift = float(parameters["category_boundary_shift"])
    within_source_curvature = float(parameters["within_source_curvature"])
    equality_source_curvature = float(
        parameters["equality_source_curvature"]
    )
    coalition_normalization = float(parameters["coalition_normalization"])
    source_count_curvature = float(parameters["source_count_curvature"])
    source_count_midpoint = float(parameters["source_count_midpoint"])
    source_count_slope = float(parameters["source_count_slope"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    v = np.clip(validities, 0.500001, 0.999999)

    # Reliability has only a weak direct effect. Diagnosticity differences
    # are strongly compressed around a fixed .75 reference so the scale is
    # comparable across experiments.
    diagnosticity = np.log(v / (1.0 - v))
    reference = np.log(0.75 / 0.25)
    represented_direct = np.power(
        np.maximum(diagnosticity, 1e-12), direct_validity_compression
    )
    represented_direct /= max(
        float(reference ** direct_validity_compression), 1e-12
    )
    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented_direct
    )

    # Idiosyncratic salience is exchangeable across identities and explicitly
    # centered, so it cannot generate a common onset, recency, or closure rule.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    # Category uncertainty controls a narrow local coassignment radius.
    # Exact matches are always coassigned; nonidentical reliabilities are
    # coassigned only inside a sharply localized subject-specific region.
    coassignment_radius = category_width * np.exp(category_boundary_shift)

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

        # Infer a local source partition by joining only reliability-near
        # coalition members. Connected components are exchangeable in expert
        # identity and produce no repeated global boundary artifacts.
        parent = np.arange(n, dtype=np.int64)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        coalition_v = v[idx]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(coalition_v[i] - coalition_v[j])) <= coassignment_radius:
                    union(i, j)

        labels = np.asarray([find(i) for i in range(n)], dtype=np.int64)
        unique_labels = np.unique(labels)
        source_budget_total = 0.0

        # Experts assigned to one latent source share a cluster-level budget.
        # Exact reliability matches invoke a categorical equality atom with
        # near-flat membership growth. Nonidentical local clusters retain the
        # accepted model's original pooling curvature.
        for label in unique_labels:
            local_members = np.flatnonzero(labels == label)
            members = idx[local_members]
            membership = float(members.size)
            source_quality = float(np.mean(cue_weights[members]))
            local_validities = coalition_v[local_members]
            exact_match = (
                members.size > 1
                and float(np.max(local_validities) - np.min(local_validities))
                <= 1e-12
            )
            cluster_curvature = (
                equality_source_curvature
                if exact_match
                else within_source_curvature
            )
            source_budget_total += source_quality * (
                membership ** cluster_curvature
            )

        effective_source_count = float(unique_labels.size)

        # Source budgets already add independent-source evidence. The extra
        # count calibration is therefore bounded and releases primarily when
        # effective source count crosses from two to three, rather than
        # multiplying evidence by an unbounded second power of source count.
        release = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    source_count_slope
                    * (effective_source_count - source_count_midpoint),
                    -50.0,
                    50.0,
                )
            )
        )
        count_factor = np.exp(source_count_curvature * release)
        nominal_normalizer = float(n) ** coalition_normalization
        signal = source_budget_total * count_factor / max(
            nominal_normalizer, 1e-12
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Equal-weighted, mirror-corrected probability of choosing the chain coalition.\"\"\"\n    if data is None or len(data) == 0:\n        return float(\"nan\")\n\n    records = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        if a.size < 20 or b.size != a.size:\n            continue\n\n        d = a - b\n        chain_direction = float(np.sum(d[:10]))\n        if abs(chain_direction) < 1e-12:\n            continue\n\n        # Positive means that A is supported by one of the two chain blocks;\n        # negative means that B is supported by a chain block.\n        chain_is_a = chain_direction > 0.0\n        response = int(row[\"response\"])\n        chain_chosen = float((response == 0 and chain_is_a) or\n                             (response == 1 and not chain_is_a))\n\n        # Identify the crossed chain/comparison configuration independently\n        # of whether it was displayed in its original or mirrored orientation.\n        chain_block = int(np.sum(np.abs(d[5:10])) > np.sum(np.abs(d[0:5])))\n        comparison_block = int(\n            np.sum(np.abs(d[15:20])) > np.sum(np.abs(d[10:15]))\n        )\n        family = 2 * chain_block + comparison_block\n        records.append((row[\"subject_id\"], family, chain_chosen))\n\n    if not records:\n        return float(\"nan\")\n\n    scored = pd.DataFrame(records, columns=[\"subject_id\", \"family\", \"score\"])\n    # Equal weighting of the four crossed configurations prevents stable cue\n    # salience or an uneven number of usable repetitions from dominating.\n    family_rates = scored.groupby([\"subject_id\", \"family\"])[\"score\"].mean()\n    subject_rates = family_rates.groupby(level=0).mean()\n    return float(subject_rates.mean())",
  "rationale": "This is the mirror-corrected chain-coalition choice rate, with equal weight assigned to each of the four crossed chain/comparison configurations and then to each subject. It targets the design's sign-reversing prediction: Reliability-Space Credit Sharing predicts selection of the five-member reliability chain above chance because its mostly nonoverlapping members retain independent credit, whereas Categorical Source-Budget Inference predicts selection below chance because the chain collapses transitively to one source while the comparison coalition receives three source budgets. Coding choices relative to the chain rather than as raw A/B responses removes response-side bias. Averaging all 96 trials and balancing the four configurations should also yield substantially lower between-subject noise than a statistic based on individual trial types."
}
```

## Usage

```json
{
  "input_tokens": 7515,
  "output_tokens": 1991
}
```
