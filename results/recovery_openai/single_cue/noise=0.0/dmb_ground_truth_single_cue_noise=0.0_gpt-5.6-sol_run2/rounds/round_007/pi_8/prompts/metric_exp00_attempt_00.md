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
**Validities (n_features=12):** [0.7, 0.7, 0.7, 0.7, 0.65, 0.7666667, 0.7666667, 0.65, 0.92, 0.92, 0.92, 0.92]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
  trial 15: A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

**Rationale:** This design tests a cardinality-specific contextual onset of redundancy. Four equal-reliability coalitions contain two 0.70 experts, whereas four reliability-separated coalitions contain experts with validities 0.65 and 0.7666667. On the bounded-strength scale x=2v-1, both pair types have exactly the same product: 0.40*0.40 = 0.30*0.5333334 = 0.16. Thus, in the 2-versus-2 trials, Contextual Reliability-Region Compression assigns them the same pair-composition load and has no triadic topology operation; apart from very small compressed-validity residuals, it predicts approximately chance-level selection of the separated pair. Reliability-Space Credit Sharing instead strongly discounts the exactly matched 0.70 pair, while the 0.65/0.7666667 pair lies far outside its narrow similarity kernel, predicting a large above-chance separated-pair preference. In matched 3-versus-3 trials, one distinct 0.92 outsider is added to each coalition. This activates the advocated theory's duplicate-pair-in-context operation only for the equal 0.70 pair, as well as its equal-cardinality relational topology contrast, so it now predicts a substantial separated-coalition preference. The competing theory continues to penalize the equal pair through the same cardinality-general credit-sharing rule and therefore predicts a strong separated preference in both dyadic and triadic contests, if anything slightly attenuated after adding an independent outsider. The decisive statistic is the coalition-size interaction: Contextual Reliability-Region Compression predicts an onset from approximately 0.50 in dyads to clearly above 0.50 in triads, whereas Reliability-Space Credit Sharing predicts an already-large dyadic effect with little or negative dyad-to-triad change. Crossed coalition instances, balanced outsider assignments, and exact A/B mirrors control cue salience and response-side bias. Sixteen unique pairs repeated six times yield exactly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Contextual Reliability-Region Compression proposes that people integrate every discriminating expert rating into an exchangeable option-level coalition while representing advertised validities on a strongly compressed reliability-region scale. Coalition evidence grows sublinearly, and stable count calibration can make a third cue either corroborative or partially redundant. Redundancy is order-specific in cardinality but not spatial position: two-cue coalitions receive only a smooth joint-diagnostic composition discount, whereas coalitions of three or more undergo a contextual topology operation. In the latter operation, two cues occupying nearly the same represented reliability region are compressed toward one informational unit when another cue occupies a distinct region. In equal-cardinality contests between multi-cue coalitions, people additionally compare the coalitions' topology relationally. This relational comparison is moderately concave: modest clustered-versus-distributed differences are transmitted approximately linearly, whereas larger differences receive stronger diminishing amplification. The comparison fades smoothly when coalition cardinalities differ, preserving independently calibrated count judgments. Topology sensitivity is stable and heterogeneous across subjects, while pair-composition sensitivity is comparatively narrow. Stable, zero-centered idiosyncratic cue salience permits individual configuration preferences without population-level onset, closure, recency, adjacency, entry, or exit effects. Because subjects receive no outcome feedback, these representations remain fixed across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.32]
- validity_reliance: [0.025, 0.070]
- salience_strength: [0.12, 0.27]
- coalition_saturation: [0.74, 0.84]
- count_calibration: [-0.75, 0.25]
- count_calibration_strength: [0.36, 0.48]
- pair_composition_sensitivity: [2.10, 2.65]
- topology_sensitivity: [0.40, 8.00]
- topology_contrast_gain: [1.50, 2.75]
- cluster_bandwidth: [0.018, 0.040]
- context_scale: [0.025, 0.060]
- beta: [1.02, 1.30]
- epsilon: [0.03, 0.08]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contextual Reliability-Region Compression expects shape "
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

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    count_calibration = float(parameters["count_calibration"])
    count_calibration_strength = float(parameters["count_calibration_strength"])
    pair_composition_sensitivity = float(
        parameters["pair_composition_sensitivity"]
    )
    topology_sensitivity = float(parameters["topology_sensitivity"])
    topology_contrast_gain = float(parameters["topology_contrast_gain"])
    cluster_bandwidth = float(parameters["cluster_bandwidth"])
    context_scale = float(parameters["context_scale"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Reliability is encoded on a diagnosticity scale, but differences are
    # strongly compressed. Mean normalization prevents an experiment's
    # overall validity level from changing the generic evidence scale.
    v = np.clip(validities, 0.500001, 0.999999)
    log_odds = np.log(v / (1.0 - v))
    represented_diagnosticity = np.power(
        np.maximum(log_odds, 1e-12), validity_compression
    )
    represented_diagnosticity /= max(
        float(np.mean(represented_diagnosticity)), 1e-12
    )

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented_diagnosticity
    )

    # Components are exchangeably sampled and centered within each subject.
    # Thus salience is stable and can generate individual differences, but
    # no display position has a positive population-level privilege.
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
    n_a = int(a_support.size)
    n_b = int(b_support.size)

    if n_a == 0 and n_b == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    bounded_diagnosticity = np.clip(2.0 * v - 1.0, 0.0, 1.0)

    def triadic_topology(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 3:
            return 0.0

        scores = []
        # Topological distance is computed in the same compressed,
        # mean-normalized reliability-region coordinate used by cognition,
        # rather than in raw advertised-validity units.
        for x in range(n - 2):
            for y in range(x + 1, n - 1):
                for z in range(y + 1, n):
                    triad = np.sort(
                        represented_diagnosticity[idx[[x, y, z]]]
                    )
                    lower_gap = float(triad[1] - triad[0])
                    upper_gap = float(triad[2] - triad[1])

                    if lower_gap <= upper_gap:
                        cluster_gap = lower_gap
                        cluster_center = 0.5 * float(triad[0] + triad[1])
                        outsider = float(triad[2])
                    else:
                        cluster_gap = upper_gap
                        cluster_center = 0.5 * float(triad[1] + triad[2])
                        outsider = float(triad[0])

                    separation = abs(outsider - cluster_center)
                    duplicate_affinity = np.exp(
                        -0.5 * (cluster_gap / cluster_bandwidth) ** 2
                    )
                    contextual_distinctness = separation / (
                        separation + context_scale
                    )
                    scores.append(
                        float(duplicate_affinity * contextual_distinctness)
                    )

        if not scores:
            return 0.0

        # A smooth probabilistic union detects whether the coalition contains
        # at least one duplicate-pair-in-context relation without privileging
        # a cue identity, ordering, or spatial adjacency.
        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 0.999999)
        return float(1.0 - np.prod(1.0 - scores))

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0

        total = float(np.sum(cue_weights[idx]))

        if n >= 3:
            topology = triadic_topology(idx)
            # Sensitivity maps smoothly to the fraction of one informational
            # unit removed. At its upper limit a near-duplicate pair is worth
            # approximately one unit rather than two; all cues nevertheless
            # retain a positive contribution.
            compression_fraction = (
                1.0 - np.exp(-topology_sensitivity * topology)
            )
            coalition_unit = float(np.mean(cue_weights[idx]))
            total -= compression_fraction * coalition_unit
            total = max(total, 1e-12)

        signal = total / (float(n) ** coalition_saturation)
        return float(signal)

    def pair_composition_load(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 2:
            return 0.0

        products = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                products.append(
                    float(
                        bounded_diagnosticity[idx[i]]
                        * bounded_diagnosticity[idx[j]]
                    )
                )
        return float(np.mean(products)) if products else 0.0

    topology_a = triadic_topology(a_support)
    topology_b = triadic_topology(b_support)
    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)

    # The status of third and later members is flexibly but boundedly
    # calibrated. This is a count-level corroboration judgment, not a cue
    # position mechanism, and is stable across the subject's trials.
    if n_a >= 2 and n_b >= 2:
        extra_a = float(max(n_a - 2, 0))
        extra_b = float(max(n_b - 2, 0))
        evidence_a *= np.exp(
            np.clip(
                count_calibration_strength * count_calibration * extra_a,
                -4.0,
                4.0,
            )
        )
        evidence_b *= np.exp(
            np.clip(
                count_calibration_strength * count_calibration * extra_b,
                -4.0,
                4.0,
            )
        )

        # Dyadic redundancy is deliberately gap-free and equality-free: it
        # depends only on smooth joint diagnostic composition. Centering the
        # adjustment between options avoids introducing an overall bias.
        load_difference = (
            pair_composition_load(a_support)
            - pair_composition_load(b_support)
        )
        size_scale = 1.0 if n_a == n_b else 0.60
        shift = (
            pair_composition_sensitivity * size_scale * load_difference
        )
        evidence_a *= np.exp(np.clip(-0.5 * shift, -4.0, 4.0))
        evidence_b *= np.exp(np.clip(0.5 * shift, -4.0, 4.0))

    # A relational topology comparison amplifies a clustered-versus-
    # distributed distinction when both alternatives support genuinely
    # multi-cue interpretations. Logistic cardinality gates and a smooth
    # balance factor make this additional contrast negligible in 3-versus-2
    # and one-versus-many judgments, leaving count calibration in control.
    # Moderate concavity preserves small contrasts approximately linearly
    # while selectively attenuating larger topology differences.
    multi_a = 1.0 / (1.0 + np.exp(-6.0 * (float(n_a) - 2.5)))
    multi_b = 1.0 / (1.0 + np.exp(-6.0 * (float(n_b) - 2.5)))
    if n_a > 0 and n_b > 0:
        log_size_ratio = np.log(float(n_a) / float(n_b))
        cardinality_balance = np.exp(-6.0 * log_size_ratio * log_size_ratio)
    else:
        cardinality_balance = 0.0
    raw_topology_contrast = topology_a - topology_b
    mild_topology_contrast = np.tanh(1.8 * raw_topology_contrast) / 1.8
    topology_shift = (
        topology_contrast_gain
        * multi_a
        * multi_b
        * cardinality_balance
        * topology_sensitivity
        / (1.0 + topology_sensitivity)
        * mild_topology_contrast
    )
    evidence_a *= np.exp(np.clip(-0.5 * topology_shift, -4.0, 4.0))
    evidence_b *= np.exp(np.clip(0.5 * topology_shift, -4.0, 4.0))

    net_a = evidence_a - evidence_b
    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total_probability = float(probs.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total_probability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Triadic onset in preference for reliability-separated coalitions.\"\"\"\n    dyad_choices = []\n    triad_choices = []\n\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=np.int8)\n        b = np.asarray(row.option_b_ratings, dtype=np.int8)\n        d = a - b\n\n        a_support = np.flatnonzero(d > 0)\n        b_support = np.flatnonzero(d < 0)\n        if a_support.size != b_support.size or a_support.size not in (2, 3):\n            continue\n\n        # The reliability-separated pair uses cues 4:8 (validities .65 and\n        # .7666667); the matched pair uses two cues 0:4 (validity .70).\n        a_mid = int(np.sum((a_support >= 4) & (a_support < 8)))\n        b_mid = int(np.sum((b_support >= 4) & (b_support < 8)))\n        a_equal = int(np.sum(a_support < 4))\n        b_equal = int(np.sum(b_support < 4))\n\n        if a_mid == 2 and b_equal == 2:\n            separated_is_a = True\n        elif b_mid == 2 and a_equal == 2:\n            separated_is_a = False\n        else:\n            continue\n\n        response = int(row.response)\n        chose_separated = float(\n            (separated_is_a and response == 0)\n            or ((not separated_is_a) and response == 1)\n        )\n\n        if a_support.size == 2:\n            dyad_choices.append(chose_separated)\n        else:\n            triad_choices.append(chose_separated)\n\n    if len(dyad_choices) == 0 or len(triad_choices) == 0:\n        return float(\"nan\")\n\n    return float(np.mean(triad_choices) - np.mean(dyad_choices))",
  "rationale": "This is the preregistered coalition-size interaction: separated-coalition choice probability in 3-versus-3 contests minus the same probability in 2-versus-2 contests. It codes choices relative to coalition type rather than A/B side, so the exact mirrors cancel response-side bias and stable cue-specific salience. Contextual Reliability-Region Compression predicts a positive onset because its duplicate-pair topology operation is activated only once an outsider creates a triad. Reliability-Space Credit Sharing penalizes the equal-validity pair already in dyads and therefore predicts little or negative change after the outsider is added. Each subject contributes 48 observations to each term, making the contrast relatively stable while directly targeting the theories' cardinality-specific disagreement."
}
```

## Usage

```json
{
  "input_tokens": 8398,
  "output_tokens": 1430
}
```
