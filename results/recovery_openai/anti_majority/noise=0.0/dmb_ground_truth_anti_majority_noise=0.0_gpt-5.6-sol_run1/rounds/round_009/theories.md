# Round 9 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_11` — KILLED ✗

**Description:** Causal-Copy Discounted Lexicographic Choice with Local Priority Uncertainty. Decision makers compress all discriminating experts supporting the same option into an option-level recommendation and infer whether agreement reflects copied reports, shared evidence, or a common source. Common-cause suspicion rises nonlinearly with supporter count and is amplified by conspicuously high, homogeneous validities, actively discounting the recommendation's epistemic accessibility. Recommendations are ordinarily considered lexicographically by their strongest communicated validity. However, within a very narrow priority-gap neighborhood, validity ranks are not perfectly resolved: a small proportion of comparisons probabilistically interchange the two recommendation priorities. This local uncertainty permits an equal-validity or nearly equal-validity opposing recommendation to govern some choices without weakening lexicographic dominance for clearly separated validities. The causal discount remains independent of opponent count, adjacency, spacing, run structure, strongest-source placement, and terminal clustering. Separately, source-validity binding has a localized nonlinear uncertainty region, and isolated recommendations suffer a weak late-position accessibility loss only in genuinely long displays. Failure to access either recommendation leads to guessing, followed by a small response lapse.

**Rationale:** This is a minimal two-range edit to the accepted iteration-7 candidate. The binding-uncertainty strength is left unchanged because both attempted global increases were rejected. Instead, `binding_uncertainty_center` is narrowed from [0.59, 0.67] to [0.625, 0.65], and `binding_uncertainty_width` is reduced from [0.11, 0.16] to [0.075, 0.105]. On the model's compressed diagnosticity scale, this concentrates the accessibility trough near the approximately 81.8%-valid singleton responsible for the negative validity-gap contrast in Experiment 6. The narrower profile should recover part of that contrast without broadly changing moderate-validity cues or the 94%-95% bundles central to Experiments 8 and 18. Equal-validity 80%-82% bundles may have lower absolute accessibility, but their within-experiment count and layout contrasts remain protected because the adjustment depends only on validity, not supporter count, opponent count, adjacency, spacing, run structure, or placement. All accepted near-tie priority, causal-copy, redundancy, lapse, and long-display mechanisms are otherwise unchanged.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `lexicographic_reliability`: `[3.0, 3.5]`
  - `validity_compression`: `[0.85, 1.15]`
  - `binding_uncertainty_strength`: `[1.7, 2.2]`
  - `binding_uncertainty_center`: `[0.625, 0.65]`
  - `binding_uncertainty_width`: `[0.075, 0.105]`
  - `common_cause_count_threshold`: `[4.0, 4.4]`
  - `common_cause_count_slope`: `[2.0, 2.7]`
  - `common_cause_homogeneity_scale`: `[0.028, 0.038]`
  - `redundancy_validity_threshold`: `[0.895, 0.91]`
  - `redundancy_validity_width`: `[0.016, 0.024]`
  - `redundancy_discount_strength`: `[7.2, 8.0]`
  - `long_display_position_loss`: `[0.0, 0.15]`
  - `near_tie_priority_mixing`: `[0.12, 0.18]`
  - `response_lapse`: `[0.1, 0.16]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Causal-Copy Discounted Lexicographic Choice expects state "
            f"shape (2, n_features); got {stim.shape}."
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

    baseline = float(parameters["lexicographic_reliability"])
    compression = float(parameters["validity_compression"])
    uncertainty_strength = float(parameters["binding_uncertainty_strength"])
    uncertainty_center = float(parameters["binding_uncertainty_center"])
    uncertainty_width = max(
        float(parameters["binding_uncertainty_width"]), 1e-9
    )
    count_threshold = float(parameters["common_cause_count_threshold"])
    count_slope = float(parameters["common_cause_count_slope"])
    homogeneity_scale = max(
        float(parameters["common_cause_homogeneity_scale"]), 1e-9
    )
    validity_threshold = float(parameters["redundancy_validity_threshold"])
    validity_width = max(
        float(parameters["redundancy_validity_width"]), 1e-9
    )
    discount_strength = float(parameters["redundancy_discount_strength"])
    position_loss = float(parameters["long_display_position_loss"])
    priority_mix = float(parameters["near_tie_priority_mixing"])
    lapse = float(parameters["response_lapse"])

    difference = stim[0] - stim[1]
    directions = np.sign(difference)
    if not np.any(directions != 0.0):
        return np.array([0.5, 0.5], dtype=np.float64)

    def sigmoid_scalar(x):
        x = float(x)
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    # Communicated validity is retained for lexicographic ordering, while a
    # compressed diagnosticity representation controls source-binding error.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)
    uncertainty_profile = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    units = []
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size == 0:
            continue

        member_validities = validities[members]
        best_local = int(np.argmax(member_validities))
        representative = int(members[best_local])
        priority = float(validities[representative])
        count = int(members.size)

        # Infer whether agreement reflects a common source. The normalized
        # count signal is exactly zero for a singleton and approaches one only
        # after the nonlinear count threshold. It uses this bundle alone.
        count_at_one = sigmoid_scalar(
            count_slope * (1.0 - count_threshold)
        )
        count_raw = sigmoid_scalar(
            count_slope * (float(count) - count_threshold)
        )
        count_signal = (count_raw - count_at_one) / max(
            1.0 - count_at_one, 1e-9
        )
        count_signal = float(np.clip(count_signal, 0.0, 1.0))

        # Equal or nearly equal communicated validities are treated as a clue
        # that reports may derive from the same underlying analysis. Physical
        # positions and contiguity do not enter this inference.
        if count <= 1:
            homogeneity = 1.0
        else:
            spread = float(np.std(member_validities))
            homogeneity = float(np.exp(
                -np.square(spread / homogeneity_scale)
            ))

        # A conspicuously strong member can make the provenance of the whole
        # compressed bundle salient. This avoids requiring every supporter to
        # clear the high-validity threshold while retaining validity gating.
        conspicuous_high_validity = sigmoid_scalar(
            (priority - validity_threshold) / validity_width
        )
        common_cause_probability = float(np.clip(
            count_signal * homogeneity * conspicuous_high_validity,
            0.0,
            1.0,
        ))

        # Epistemic discounting is additionally validity-gated. Thus large
        # homogeneous 94% bundles can be strongly discredited, whereas equal
        # 80%-82% bundles acquire little cost even at comparable counts.
        validity_gate = sigmoid_scalar(
            (priority - validity_threshold) / validity_width
        )
        redundancy_cost = (
            discount_strength
            * common_cause_probability
            * validity_gate
        )

        retrieval_logit = (
            baseline
            - uncertainty_strength * uncertainty_profile[representative]
            - redundancy_cost
        )

        # Only isolated recommendations receive a display-position loss. The
        # quadratic display gate is zero through 24 features and reaches full
        # strength at 32, preventing ordinary layouts from producing effects.
        if count == 1 and n_features > 1:
            display_gate = float(np.clip(
                (float(n_features) - 24.0) / 8.0, 0.0, 1.0
            )) ** 2
            relative_position = representative / float(n_features - 1)
            retrieval_logit -= (
                position_loss * display_gate * relative_position
            )

        retrieval = sigmoid_scalar(retrieval_logit)
        units.append({
            "direction": int(direction),
            "priority": priority,
            "representative": representative,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
        })

    # Search remains validity ordered after causal credibility determines
    # whether each compressed recommendation is accessible. Position supplies
    # the default exact-tie convention but confers no accessibility advantage.
    units.sort(key=lambda u: (-u["priority"], u["representative"]))

    def cascade_probability_a(ordered_units):
        reach = 1.0
        intended_a = 0.0
        intended_b = 0.0
        for unit in ordered_units:
            stop = reach * unit["retrieval"]
            if unit["direction"] > 0:
                intended_a += stop
            else:
                intended_b += stop
            reach *= 1.0 - unit["retrieval"]

        intended_a += 0.5 * reach
        intended_b += 0.5 * reach
        total = intended_a + intended_b
        if not np.isfinite(total) or total <= 0.0:
            return 0.5
        return float(intended_a / total)

    p_a_lexicographic = cascade_probability_a(units)

    # A small fraction of comparisons have uncertain priority only within a
    # narrow validity-gap neighborhood. The fixed 0.012 scale makes the soft
    # order effectively deterministic for clearly separated recommendations.
    # Both possible orders are marginalized, rather than sampled in predict.
    if len(units) == 2:
        priority_gap = max(
            float(units[0]["priority"] - units[1]["priority"]), 0.0
        )
        first_order_probability = sigmoid_scalar(priority_gap / 0.012)
        p_a_soft_order = (
            first_order_probability * cascade_probability_a(units)
            + (1.0 - first_order_probability)
            * cascade_probability_a([units[1], units[0]])
        )
        p_a_intended = (
            (1.0 - priority_mix) * p_a_lexicographic
            + priority_mix * p_a_soft_order
        )
    else:
        p_a_intended = p_a_lexicographic

    p_a = (1.0 - lapse) * p_a_intended + lapse * 0.5
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
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


### slot 2 — `pi_10` — SURVIVED ✓

**Description:** Validity-Gated Intrinsic Recommendation Capacity. On each trial, all active experts favoring the same option are obligatorily compressed into one recommendation representation, irrespective of adjacency, spacing, pairing, or the location of the strongest source. The representation is indexed and ranked by its strongest communicated validity. Additional supporters yield only a negligible, rapidly saturating reliability increment rather than independent retrieval opportunities. Retrieval is limited by the intrinsic representational concentration of each recommendation: jointly binding many same-direction sources becomes costly only at very high diagnosticity, with load increasing smoothly with its own source count. This capacity cost belongs to that recommendation and is never gated by how many sources support its opponent. Thus an opponent acquiring separated supporters cannot suddenly impair an unchanged run. Retrieval also contains a narrow localized nonlinear source-validity binding uncertainty. Retrieved recommendations enter a predominantly lexicographic race ordered by their strongest communicated validity, with accessibility normalized against the competing recommendation. Spatial organization ordinarily has no role. Only an isolated recommendation in a very long display receives a monotonic late-position accessibility gradient, gated so that it is appreciable around 32 features but negligible around 24. If neither recommendation is retrieved, the decision maker guesses; a small symmetric response lapse follows.

**Rationale:** This is a minimal calibration of the accepted model. The 32-feature singleton position term changes sign, converting the previous late-position accessibility subtraction into a monotonic accessibility increase; this targets the reversed Experiment 7 contrast while the unchanged quadratic display gate keeps the term exactly absent at 24 features. The intrinsic diagnosticity gate moves from 0.88/0.025 to 0.90/0.022, sharply reducing overload for 82%-valid recommendations in Experiment 11 while retaining nearly full overload for the 95%-valid recommendation in Experiment 8. The binding-uncertainty strength and width are modestly reduced to temper the excessive inverse validity-gap effect in Experiment 6. Finally, a slightly larger symmetric lapse raises the overly low override rates in Experiments 4 and 5 and mildly attenuates the excessive Experiment 8 decrement. No adjacency, spacing, strongest-member-location, or opponent-count mechanism is introduced.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `validity_compression`: `[0.85, 1.2]`
  - `retrieval_baseline`: `[2.8, 3.5]`
  - `binding_uncertainty_strength`: `[0.9, 1.5]`
  - `binding_uncertainty_center`: `[0.58, 0.68]`
  - `binding_uncertainty_width`: `[0.1, 0.16]`
  - `intrinsic_capacity_strength`: `[0.7, 1.2]`
  - `saturating_reliability_gain`: `[0.0, 0.003]`
  - `long_display_position_loss`: `[0.8, 1.4]`
  - `response_lapse`: `[0.1, 0.16]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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


## Replacement

### `pi_12` → slot 1 (via `new_theory`)

**Description:** Load-Adaptive Anchor–Validity Arbitration. Choice is produced by competition between a validity-directed search route and an order-anchor route. Validity-directed search compresses all currently agreeing experts into option-level recommendations, ranks those recommendations by their strongest communicated validity, and usually stops at the first accessible recommendation. Its accessibility includes localized source–validity binding uncertainty and an intrinsic count-concentration cost. The count cost is independent of spatial arrangement, opponent count, and within-bundle validity dispersion; it remains weak for ordinary bundles but rises sharply when at least five or six extremely high-validity sources must be maintained as one recommendation. The order-anchor route is gated by display load. In long displays, the first discriminating recommendation establishes a provisional choice. A later opposing recommendation overturns that anchor only when its validity advantage and compressed diagnostic support jointly exceed a load-dependent threshold. The threshold rises with the display position at which the strongest opposing source is encountered, producing primacy rather than recency. Conspicuous validity differences switch processing back toward validity-directed search, whereas small differences such as 80% versus 82% leave the early anchor influential. Stable subject-level variation in order-anchor reliance produces heterogeneous long-display behavior without trial feedback or experiment-specific rules.

**Rationale:** This is a parameter-only minimal-diff edit. Reducing conspicuous_route_switch preserves a substantial position-sensitive anchor route even when the later cue has a noticeable validity advantage, while the moderately larger late_override_threshold makes later overrides harder; together these changes should turn Experiment 7's nearly flat gradient negative without changing the mechanism. The overload threshold is moved close to six and its width narrowed sharply, so five-source bundles receive little penalty while six-source bundles cross a steep capacity cliff. Increasing overload strength then lowers the six-94% absolute choice rate in Experiment 18 while avoiding the excessive average five/six penalty in Experiment 8. A modestly larger compressed_support_gain improves ordinary-bundle integration in Experiments 4 and 5, whereas the high-validity overload cost still dominates extreme bundles. Finally, order_anchor_reliance is widened around approximately the same central tendency, increasing stable between-subject heterogeneity in Experiment 20 without using response noise or trial-specific adaptation.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `retrieval_baseline`: `[2.9, 3.4]`
  - `validity_compression`: `[0.85, 1.15]`
  - `binding_uncertainty_strength`: `[1.5, 2.1]`
  - `binding_uncertainty_center`: `[0.60, 0.67]`
  - `binding_uncertainty_width`: `[0.08, 0.13]`
  - `compressed_support_gain`: `[0.08, 0.18]`
  - `high_validity_overload_strength`: `[8.0, 10.0]`
  - `overload_count_threshold`: `[5.75, 5.9]`
  - `overload_count_width`: `[0.08, 0.16]`
  - `overload_validity_center`: `[0.90, 0.92]`
  - `overload_validity_width`: `[0.014, 0.025]`
  - `order_anchor_reliance`: `[0.68, 1.0]`
  - `conspicuous_gap_center`: `[0.055, 0.075]`
  - `conspicuous_gap_width`: `[0.012, 0.022]`
  - `conspicuous_route_switch`: `[0.35, 0.55]`
  - `override_gap_scale`: `[0.025, 0.04]`
  - `override_support_gain`: `[0.06, 0.14]`
  - `override_threshold`: `[0.75, 1.05]`
  - `load_override_threshold`: `[0.65, 0.95]`
  - `late_override_threshold`: `[0.65, 0.95]`
  - `override_temperature`: `[0.20, 0.34]`
  - `response_lapse`: `[0.09, 0.15]`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
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
