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
**Validities (n_features=12):** [0.6, 0.6, 0.9, 0.9, 0.6, 0.6, 0.87320508, 0.7, 0.52679492, 0.52679492, 0.87320508, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** This 12-cue design isolates the advocated theory's contextual reliability-region topology operation in equal-cardinality three-versus-three contests. There are two interchangeable clustered coalitions, each with validities (0.60, 0.60, 0.90), and two interchangeable distributed coalitions, each with validities (0.87320508, 0.70, 0.52679492). On the bounded diagnostic-strength scale x=2v-1, the clustered triple is (0.20, 0.20, 0.80), while the distributed triple is approximately (0.74641016, 0.40, 0.05358984). Both have strength sum 1.20 and sum of pairwise products 0.36, so their mean pair-composition load is exactly 0.12. Coalition cardinality, saturation, third-member calibration, and composition redundancy therefore cancel between options in both theories. The competing theory has no reliability-topology operation and should predict distributed-coalition choice very near 0.50 at the population level; its compressed reliability sums actually give the clustered coalition a tiny advantage, so it cannot explain a substantial distributed preference. In contrast, the advocated theory represents the two 0.60 clustered cues in effectively the same reliability region while the 0.90 cue occupies a distinct region. Its triadic operation consequently compresses the clustered coalition toward two informational units. The distributed triple has three widely separated represented diagnosticities and receives essentially no such compression. Equal cardinality also fully activates the advocated theory's relational topology contrast, producing a robust preference for the distributed coalition whose magnitude should increase with subject-level topology sensitivity. Four crossed comparisons make every coalition instance appear equally often, and four exact A/B mirrors control response-side bias. The primary statistic is mirror-corrected selection of the distributed coalition: clearly above chance under Contextual Reliability-Region Compression but approximately chance, or marginally below chance, under Exchangeable Corroboration. Eight unique pairs are repeated 12 times each for exactly 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** Exchangeable Corroboration with Idiosyncratic Salience proposes that people retain the direction of every discriminating expert rating but strongly compress instructed differences in expert validity. Supporting cues are integrated into option-level coalitions whose evidence increases sublinearly, so additional agreement provides corroboration with diminishing marginal impact. Subjects differ in how they calibrate coalition members beyond the first two: a signed, stable calibration parameter makes an additional cue corroborative for some people and partially redundant for others. Separately, jointly diagnostic cues can be perceived as partially redundant: an order-free composition term discounts coalitions containing several simultaneously strong cues, with its effect centered between the competing coalitions. This composition discount is strongest for equal-sized coalitions but remains moderately active when coalition sizes differ. Cue order has no population-level directional role. Instead, each subject has a stable, broad cue-salience profile drawn from a zero-centered exchangeable distribution and centered again within the display. Consequently, individual subjects can show substantial configuration preferences, but there is no common early, late, onset, closure, or entry-versus-exit advantage. Choices result from the difference between the two saturated coalition signals, transformed by response sensitivity and an independent lapse process. Because no outcome feedback is supplied, neither validity beliefs nor salience profiles are learned across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.34]
- validity_reliance: [0.02, 0.07]
- salience_strength: [0.18, 0.42]
- coalition_saturation: [0.76, 0.86]
- corroboration_calibration: [-1.30, 0.70]
- corroboration_strength: [0.40, 0.50]
- composition_redundancy: [2.00, 2.80]
- beta: [0.82, 1.12]
- epsilon: [0.03, 0.09]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Exchangeable Corroboration with Idiosyncratic Salience expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(parameters["cue_salience_profile"], dtype=np.float64)
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != n_features "
            f"{n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    corroboration_calibration = float(parameters["corroboration_calibration"])
    corroboration_strength = float(parameters["corroboration_strength"])
    composition_redundancy = float(parameters["composition_redundancy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Instructed reliability is represented on a diagnosticity scale, but
    # differences among experts are strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Salience is a stable subject-level profile, not a fixed position rule.
    # Centering removes a subject's global offset. Because profile components
    # have identical sampling distributions, no serial position has a
    # population-positive expected advantage.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attentional_weight = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attentional_weight /= max(float(np.mean(attentional_weight)), 1e-12)

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * diagnosticity
    )
    cue_weights = reliability_weight * attentional_weight
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)
    n_a = int(a_support.size)
    n_b = int(b_support.size)

    if n_a == 0 and n_b == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Every discriminating cue contributes. Division by n**saturation
        # makes total evidence grow approximately as n**(1-saturation),
        # implementing diminishing marginal corroboration without stopping.
        total = float(np.sum(cue_weights[indices]))
        return total / (float(n) ** coalition_saturation)

    def composition_load(indices):
        # Mean pairwise joint diagnostic strength is a bounded, order-free
        # measure of potential redundancy. It is high when several members
        # are all strongly diagnostic, not when a coalition merely appears
        # early or late in the display.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 2:
            return 0.0
        bounded_strength = np.clip(2.0 * v[idx] - 1.0, 0.0, 1.0)
        pair_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_sum += float(bounded_strength[i] * bounded_strength[j])
                pair_count += 1
        return pair_sum / float(max(pair_count, 1))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)

    # Calibration of excess coalition membership is invoked only when both
    # options already have multi-cue interpretations. Thus it captures the
    # psychological status of a third or later coalition member rather than
    # introducing a general position gate. Positive values treat extra cues
    # as corroboration; negative values treat them as redundancy.
    if n_a >= 2 and n_b >= 2:
        extra_a = float(max(n_a - 2, 0))
        extra_b = float(max(n_b - 2, 0))
        log_adjustment_a = (
            corroboration_strength * corroboration_calibration * extra_a
        )
        log_adjustment_b = (
            corroboration_strength * corroboration_calibration * extra_b
        )
        evidence_a *= np.exp(np.clip(log_adjustment_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_adjustment_b, -10.0, 10.0))

        # Decouple composition from the heterogeneous size calibration.
        # Coalitions containing several jointly strong cues receive a mild
        # redundancy discount. Centering the adjustment between options
        # prevents a global response bias. Composition is most diagnostic
        # when coalition sizes are equal and remains moderately active in
        # size conflicts without becoming a spatial or directional gate.
        load_difference = composition_load(a_support) - composition_load(b_support)
        composition_scale = 1.0 if n_a == n_b else 0.60
        composition_shift = (
            composition_redundancy * composition_scale * load_difference
        )
        evidence_a *= np.exp(np.clip(-0.5 * composition_shift, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(0.5 * composition_shift, -10.0, 10.0))

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
def policy(probs):
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if data is None or len(data) == 0:\n        return float('nan')\n\n    cell_values = {}\n    for _, row in data.iterrows():\n        a = np.asarray(row['option_a_ratings'], dtype=int)\n        b = np.asarray(row['option_b_ratings'], dtype=int)\n        response = int(row['response'])\n\n        # In this design, distributed coalitions occupy cues 6--11 and\n        # clustered coalitions occupy cues 0--5. Identify which display side\n        # contains the distributed coalition from discriminating support.\n        a_support = np.flatnonzero(a > b)\n        b_support = np.flatnonzero(b > a)\n        a_is_distributed = (\n            a_support.size > 0 and np.all(a_support >= 6)\n        )\n        b_is_distributed = (\n            b_support.size > 0 and np.all(b_support >= 6)\n        )\n\n        if a_is_distributed == b_is_distributed:\n            continue\n\n        chose_distributed = (\n            (response == 0 and a_is_distributed) or\n            (response == 1 and b_is_distributed)\n        )\n        signed_choice = 1.0 if chose_distributed else -1.0\n\n        # Equal-weight the eight unique configurations so that the statistic\n        # remains mirror-corrected even if a few trials are missing.\n        key = (tuple(a.tolist()), tuple(b.tolist()))\n        if key not in cell_values:\n            cell_values[key] = []\n        cell_values[key].append(signed_choice)\n\n    if not cell_values:\n        return float('nan')\n\n    cell_means = [float(np.mean(values)) for values in cell_values.values()]\n    return float(np.mean(cell_means))\n",
  "rationale": "Mirror-corrected distributed-coalition preference, scaled from -1 (always chooses the clustered triple) to +1 (always chooses the distributed triple), with 0 denoting indifference. Equal weighting of the eight configurations removes response-side and cell-frequency effects. The advocated theory predicts a consistently positive contrast because both its within-coalition compression and relational topology operation favor each distributed triple. The competing theory lacks that common topology term; the matched composition loads and crossed coalition identities leave its population contrast near zero, while repeated trials and averaging across all four crossings suppress choice noise and idiosyncratic cue-salience effects."
}
```

## Usage

```json
{
  "input_tokens": 7964,
  "output_tokens": 2060
}
```
