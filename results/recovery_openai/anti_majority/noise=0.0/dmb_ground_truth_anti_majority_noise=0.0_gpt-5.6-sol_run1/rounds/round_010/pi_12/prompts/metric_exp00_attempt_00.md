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
**Validities (n_features=32):** [0.8, 0.83, 0.84, 0.99, 0.52, 0.6, 0.72, 0.84, 0.55, 0.64, 0.76, 0.58, 0.7, 0.84, 0.53, 0.62, 0.74, 0.57, 0.68, 0.84, 0.54, 0.66, 0.78, 0.59, 0.71, 0.84, 0.51, 0.63, 0.75, 0.56, 0.69, 0.84]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Rationale:** This 32-expert design isolates the advocated theory's load-dependent late-override threshold while disabling the competing theory's only spatial mechanism. In every base pair, the first expert (80%) favors B and therefore establishes the order anchor. A two-source recommendation favors A: a fixed 83% supporter at Expert 2 and a stronger 84% representative appearing at Experts 3, 8, 14, 20, 26, or 32. All remaining experts tie. Source count, recommendation validity, compressed support, anchor validity, and the complete evidence multiset are constant; only the display position of the strongest opposing source changes. Load-Adaptive Anchor–Validity Arbitration predicts a monotonic decline in A choices as the 84% representative moves later. Its validity-directed route is position-invariant, but its order-anchor route makes an opposing recommendation progressively less able to overturn the initial B anchor because the required override drive rises with the representative's relative position. The modest 4-point validity advantage remains below the conspicuous-gap region, preserving substantial anchor-route influence. Validity-Gated Intrinsic Recommendation Capacity predicts an exactly flat location profile: the A recommendation always has two sources, so its isolated-recommendation position gradient is obligatorily absent, while its priority, count, capacity cost, uncertainty, and competition normalization are unchanged. Thus the critical quantitative signature is a negative early-to-late slope in following the 84%-indexed recommendation under the advocated theory versus no slope under the competitor. The final six pairs are exact A/B reversals, balancing response side while preserving the anchor-relative effect. Twelve unique pairs repeated eight times yield 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Load-Adaptive Anchor–Validity Arbitration. Choice is produced by competition between a validity-directed search route and an order-anchor route. Validity-directed search compresses all currently agreeing experts into option-level recommendations, ranks those recommendations by their strongest communicated validity, and usually stops at the first accessible recommendation. Its accessibility includes localized source–validity binding uncertainty and an intrinsic count-concentration cost. The count cost is independent of spatial arrangement, opponent count, and within-bundle validity dispersion; it remains weak for ordinary bundles but rises sharply when at least five or six extremely high-validity sources must be maintained as one recommendation. The order-anchor route is gated by display load. In long displays, the first discriminating recommendation establishes a provisional choice. A later opposing recommendation overturns that anchor only when its validity advantage and compressed diagnostic support jointly exceed a load-dependent threshold. The threshold rises with the display position at which the strongest opposing source is encountered, producing primacy rather than recency. Conspicuous validity differences switch processing back toward validity-directed search, whereas small differences such as 80% versus 82% leave the early anchor influential. Stable subject-level variation in order-anchor reliance produces heterogeneous long-display behavior without trial feedback or experiment-specific rules.

**Parameters:**
- n_features: n_features
- validities: validities
- retrieval_baseline: [2.9, 3.4]
- validity_compression: [0.85, 1.15]
- binding_uncertainty_strength: [1.5, 2.1]
- binding_uncertainty_center: [0.60, 0.67]
- binding_uncertainty_width: [0.08, 0.13]
- compressed_support_gain: [0.08, 0.18]
- high_validity_overload_strength: [8.0, 10.0]
- overload_count_threshold: [5.75, 5.9]
- overload_count_width: [0.08, 0.16]
- overload_validity_center: [0.90, 0.92]
- overload_validity_width: [0.014, 0.025]
- order_anchor_reliance: [0.68, 1.0]
- conspicuous_gap_center: [0.055, 0.075]
- conspicuous_gap_width: [0.012, 0.022]
- conspicuous_route_switch: [0.35, 0.55]
- override_gap_scale: [0.025, 0.04]
- override_support_gain: [0.06, 0.14]
- override_threshold: [0.75, 1.05]
- load_override_threshold: [0.65, 0.95]
- late_override_threshold: [0.65, 0.95]
- override_temperature: [0.20, 0.34]
- response_lapse: [0.09, 0.15]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Load-Adaptive Anchor-Validity Arbitration expects state shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"State has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )

    retrieval_baseline = float(parameters["retrieval_baseline"])
    validity_compression = float(parameters["validity_compression"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(float(parameters["binding_uncertainty_width"]), 1e-9)
    reliability_gain = float(parameters["compressed_support_gain"])
    overload_strength = float(parameters["high_validity_overload_strength"])
    overload_threshold = float(parameters["overload_count_threshold"])
    overload_width = max(float(parameters["overload_count_width"]), 1e-9)
    high_validity_center = float(parameters["overload_validity_center"])
    high_validity_width = max(float(parameters["overload_validity_width"]), 1e-9)
    anchor_reliance = float(parameters["order_anchor_reliance"])
    conspicuous_center = float(parameters["conspicuous_gap_center"])
    conspicuous_width = max(float(parameters["conspicuous_gap_width"]), 1e-9)
    conspicuous_switch = float(parameters["conspicuous_route_switch"])
    override_gap_scale = max(float(parameters["override_gap_scale"]), 1e-9)
    override_support_gain = float(parameters["override_support_gain"])
    override_threshold = float(parameters["override_threshold"])
    load_threshold = float(parameters["load_override_threshold"])
    position_threshold = float(parameters["late_override_threshold"])
    override_temperature = max(float(parameters["override_temperature"]), 1e-9)
    lapse = float(parameters["response_lapse"])

    difference = stim[0] - stim[1]
    directions = np.sign(difference)
    active = np.flatnonzero(directions != 0.0).astype(int)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def sigmoid_scalar(x):
        x = float(x)
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    # Subjective diagnosticity affects binding and accumulated support, while
    # communicated validity itself determines recommendation priority.
    objective_diagnosticity = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_diagnosticity = np.power(
        objective_diagnosticity, validity_compression
    )
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_diagnosticity - uncertainty_center) / uncertainty_width
        )
    )

    units = {}
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size == 0:
            continue

        member_validities = validities[members]
        best_local = int(np.argmax(member_validities))
        representative = int(members[best_local])
        priority = float(validities[representative])
        count = int(members.size)

        # Same-direction sources form one recommendation. Their positive
        # reliability contribution is deliberately small and saturating.
        support_bonus = reliability_gain * (
            1.0 - np.exp(-float(max(count - 1, 0)))
        )

        # Intrinsic concentration overload depends only on the recommendation's
        # count and strongest diagnosticity. It contains no dispersion,
        # adjacency, run, position, or opponent-count term. The count gate is
        # normalized to zero for a singleton and becomes steep near 5-6 cues.
        raw_count_gate = sigmoid_scalar(
            (float(count) - overload_threshold) / overload_width
        )
        singleton_gate = sigmoid_scalar(
            (1.0 - overload_threshold) / overload_width
        )
        count_gate = (raw_count_gate - singleton_gate) / max(
            1.0 - singleton_gate, 1e-9
        )
        count_gate = float(np.clip(count_gate, 0.0, 1.0))
        validity_gate = sigmoid_scalar(
            (priority - high_validity_center) / high_validity_width
        )
        overload_cost = overload_strength * count_gate * validity_gate

        retrieval_logit = (
            retrieval_baseline
            - uncertainty_strength * uncertainty_profile[representative]
            + support_bonus
            - overload_cost
        )
        retrieval = sigmoid_scalar(retrieval_logit)

        units[direction] = {
            "direction": int(direction),
            "members": members,
            "representative": representative,
            "priority": priority,
            "count": count,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
            "overload_cost": float(overload_cost),
        }

    # Validity-directed route: option-level recommendations enter a
    # lexicographic accessibility cascade.
    validity_order = sorted(
        units.values(),
        key=lambda u: (-u["priority"], u["representative"]),
    )

    reach = 1.0
    validity_a = 0.0
    validity_b = 0.0
    for unit in validity_order:
        stop = reach * unit["retrieval"]
        if unit["direction"] > 0:
            validity_a += stop
        else:
            validity_b += stop
        reach *= 1.0 - unit["retrieval"]
    validity_a += 0.5 * reach
    validity_b += 0.5 * reach
    validity_total = validity_a + validity_b
    if not np.isfinite(validity_total) or validity_total <= 0.0:
        p_a_validity = 0.5
    else:
        p_a_validity = float(validity_a / validity_total)

    # The display-load gate is exactly zero through 24 features and reaches
    # one at 32. Thus ordinary displays are governed by validity search, while
    # long displays permit stable individual differences in anchoring.
    display_load = float(np.clip(
        (float(n_features) - 24.0) / 8.0, 0.0, 1.0
    )) ** 2

    first_index = int(active[0])
    anchor_direction = int(np.sign(directions[first_index]))
    anchor_validity = float(validities[first_index])
    opposing_unit = units.get(-anchor_direction)

    if opposing_unit is None:
        p_override = 0.0
        priority_gap = 0.0
    else:
        priority_gap = float(
            opposing_unit["priority"] - anchor_validity
        )

        # Additional agreeing sources are compressed: they provide diminishing
        # diagnostic support rather than independent votes. The contribution
        # depends on count and diagnosticity but not spatial organization.
        opposing_members = opposing_unit["members"]
        mean_diagnosticity = float(np.mean(
            subjective_diagnosticity[opposing_members]
        ))
        compressed_support = (
            np.log1p(float(max(opposing_unit["count"] - 1, 0)))
            * mean_diagnosticity
        )

        # Later evidence has a harder time overturning the anchor. Position is
        # defined by the strongest opposing source, so rearranging weaker
        # supporters, changing adjacency, or changing run structure has no
        # direct effect.
        if n_features > 1:
            relative_position = (
                opposing_unit["representative"] / float(n_features - 1)
            )
        else:
            relative_position = 0.0

        override_drive = (
            priority_gap / override_gap_scale
            + override_support_gain * compressed_support
        )
        required_drive = (
            override_threshold
            + display_load * load_threshold
            + display_load * position_threshold * relative_position
        )
        p_override = sigmoid_scalar(
            (override_drive - required_drive) / override_temperature
        )

    # The anchor route chooses the first recommendation unless the opposing
    # recommendation clears the override threshold.
    if anchor_direction > 0:
        p_a_anchor = 1.0 - p_override
    else:
        p_a_anchor = p_override

    # Conspicuous priority gaps recruit validity-directed search even in a long
    # display. Small gaps such as .80 versus .82 leave the anchor route active.
    if len(units) == 2:
        priorities = [u["priority"] for u in units.values()]
        absolute_gap = abs(float(priorities[0] - priorities[1]))
    else:
        absolute_gap = 0.0
    conspicuousness = sigmoid_scalar(
        (absolute_gap - conspicuous_center) / conspicuous_width
    )
    anchor_weight = (
        anchor_reliance
        * display_load
        * (1.0 - conspicuous_switch * conspicuousness)
    )
    anchor_weight = float(np.clip(anchor_weight, 0.0, 1.0))

    p_a_intended = (
        anchor_weight * p_a_anchor
        + (1.0 - anchor_weight) * p_a_validity
    )

    p_a = (1.0 - lapse) * p_a_intended + lapse * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Validity-Gated Intrinsic Recommendation Capacity. On each trial, all active experts favoring the same option are obligatorily compressed into one recommendation representation, irrespective of adjacency, spacing, pairing, or the location of the strongest source. The representation is indexed and ranked by its strongest communicated validity. Additional supporters yield only a negligible, rapidly saturating reliability increment rather than independent retrieval opportunities. Retrieval is limited by the intrinsic representational concentration of each recommendation: jointly binding many same-direction sources becomes costly only at very high diagnosticity, with load increasing smoothly with its own source count. This capacity cost belongs to that recommendation and is never gated by how many sources support its opponent. Thus an opponent acquiring separated supporters cannot suddenly impair an unchanged run. Retrieval also contains a narrow localized nonlinear source-validity binding uncertainty. Retrieved recommendations enter a predominantly lexicographic race ordered by their strongest communicated validity, with accessibility normalized against the competing recommendation. Spatial organization ordinarily has no role. Only an isolated recommendation in a very long display receives a monotonic late-position accessibility gradient, gated so that it is appreciable around 32 features but negligible around 24. If neither recommendation is retrieved, the decision maker guesses; a small symmetric response lapse follows.

**Parameters:**
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.2]
- retrieval_baseline: [2.8, 3.5]
- binding_uncertainty_strength: [0.9, 1.5]
- binding_uncertainty_center: [0.58, 0.68]
- binding_uncertainty_width: [0.1, 0.16]
- intrinsic_capacity_strength: [0.7, 1.2]
- saturating_reliability_gain: [0.0, 0.003]
- long_display_position_loss: [0.8, 1.4]
- response_lapse: [0.1, 0.16]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Validity-Gated Intrinsic Recommendation Capacity expects "
            f"state shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"State has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["retrieval_baseline"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(float(parameters["binding_uncertainty_width"]), 1e-9)
    capacity_strength = float(parameters["intrinsic_capacity_strength"])
    saturation_gain = float(parameters["saturating_reliability_gain"])
    position_loss = float(parameters["long_display_position_loss"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    if not np.any(directions != 0.0):
        return np.array([0.5, 0.5], dtype=np.float64)

    # Binding uncertainty is localized on a compressed internal validity scale.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    def sigmoid_scalar(x):
        x = float(x)
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    units = {}
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size == 0:
            continue

        member_validities = validities[members]
        representative = int(members[int(np.argmax(member_validities))])
        count = int(members.size)
        priority = float(validities[representative])

        # Redundant sources do not create independent retrieval attempts.
        # Their positive reliability contribution saturates almost immediately.
        reliability_gain = saturation_gain * (
            1.0 - np.exp(-2.5 * float(max(count - 1, 0)))
        )

        # Intrinsic recommendation capacity. The softplus difference is zero
        # for a singleton and rises smoothly as more sources must be jointly
        # bound. A narrow high-diagnosticity gate confines substantial overload
        # to recommendations near the top of the communicated-validity scale.
        # It depends only on this recommendation, never on opponent count or
        # spatial arrangement.
        concentration = (
            np.logaddexp(0.0, float(count) - 3.0)
            - np.logaddexp(0.0, 1.0 - 3.0)
        )
        diagnosticity_gate = sigmoid_scalar((priority - 0.90) / 0.022)
        intrinsic_cost = (
            capacity_strength * diagnosticity_gate * concentration
        )

        raw_logit = (
            baseline
            - uncertainty_strength * uncertainty_profile[representative]
            + reliability_gain
            - intrinsic_cost
        )

        # Only isolated recommendations acquire a monotonic position gradient,
        # and only under substantial display load. The quadratic gate is zero
        # through 24 features and reaches full strength at 32 features. The
        # positive orientation matches the observed increase in target-cue
        # following across the late positions of the 32-feature display.
        if count == 1 and n_features > 1:
            display_gate = float(np.clip(
                (float(n_features) - 24.0) / 8.0, 0.0, 1.0
            )) ** 2
            relative_position = representative / float(n_features - 1)
            raw_logit += position_loss * display_gate * relative_position

        units[direction] = {
            "direction": int(direction),
            "representative": representative,
            "priority": priority,
            "raw_logit": float(raw_logit),
        }

    # Recommendation-level competition normalization. Opponent activation can
    # normalize accessibility, but opponent source count has no direct gate on
    # the focal recommendation's intrinsic capacity cost.
    for direction, unit in units.items():
        opponent = units.get(-direction)
        if opponent is None:
            competition = 0.0
        else:
            competition = float(np.logaddexp(
                0.0, opponent["raw_logit"] - baseline
            ))
        normalized_logit = unit["raw_logit"] - competition
        unit["retrieval"] = float(np.clip(
            sigmoid_scalar(normalized_logit), 1e-9, 1.0 - 1e-9
        ))

    # Predominantly lexicographic retrieval: the highest-validity available
    # recommendation terminates search. Exact ties use source position only as
    # a deterministic sorting convention, not as an accessibility advantage.
    ordered = sorted(
        units.values(),
        key=lambda u: (-u["priority"], u["representative"]),
    )

    reach = 1.0
    intended_a = 0.0
    intended_b = 0.0
    for unit in ordered:
        stop_probability = reach * unit["retrieval"]
        if unit["direction"] > 0:
            intended_a += stop_probability
        else:
            intended_b += stop_probability
        reach *= 1.0 - unit["retrieval"]

    # Complete recommendation retrieval failure produces guessing.
    intended_a += 0.5 * reach
    intended_b += 0.5 * reach
    total = intended_a + intended_b
    if not np.isfinite(total) or total <= 0.0:
        p_a_intended = 0.5
    else:
        p_a_intended = intended_a / total

    # Small symmetric lapse mixes the intended response with a uniform choice.
    p_a = (1.0 - lapse) * p_a_intended + lapse * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if len(data) == 0:\n        return float(\"nan\")\n\n    target_is_a = []\n    followed_target = []\n\n    for a_cell, b_cell, response in zip(\n        data[\"option_a_ratings\"],\n        data[\"option_b_ratings\"],\n        data[\"response\"],\n    ):\n        a = np.asarray(a_cell, dtype=float)\n        b = np.asarray(b_cell, dtype=float)\n        direction = a[1] - b[1]\n        if direction == 0 or pd.isna(response):\n            continue\n\n        is_a = direction > 0\n        chose_target = (int(response) == (0 if is_a else 1))\n        target_is_a.append(is_a)\n        followed_target.append(float(chose_target))\n\n    if len(followed_target) == 0:\n        return float(\"nan\")\n\n    target_is_a = np.asarray(target_is_a, dtype=bool)\n    followed_target = np.asarray(followed_target, dtype=float)\n\n    # Average the two mirrored response-side conditions separately so that\n    # stable A/B response bias cannot masquerade as recommendation following.\n    side_means = []\n    for mask in (target_is_a, ~target_is_a):\n        if np.any(mask):\n            side_means.append(float(np.mean(followed_target[mask])))\n\n    return float(np.mean(side_means))",
  "rationale": "Side-balanced probability of choosing the option supported by the two-source recommendation (the fixed 83% expert at index 1 and the moving 84% expert), rather than the option supported by the initial 80% anchor. The competing theory's lexicographic retrieval race should strongly favor this recommendation because its 84% representative outranks the singleton 80% recommendation, and its two-source status disables the competitor's isolated-cue position mechanism. The advocated theory instead assigns substantial weight to the first discriminating expert in this 32-feature display, so the later opposing bundle often fails to override the anchor. This creates a large predicted level difference while using all 96 observations per subject, yielding substantially lower sampling variance than an early-versus-late slope alone. Separately averaging trials where the target is A and where it is B removes stable response-side bias and exploits the exact mirror construction."
}
```

## Usage

```json
{
  "input_tokens": 9526,
  "output_tokens": 2534
}
```
