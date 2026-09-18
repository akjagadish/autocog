# Round 8 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_9` — KILLED ✗

**Description:** Competition-Normalized Recommendation Race. People first compress all currently active experts favoring the same option into a single recommendation-level representation. Consequently, redundant experts are retrieved correlatively rather than as independent opportunities for rescue, and reliability gains from additional experts saturate almost immediately. Each recommendation is indexed by its strongest communicated validity and the two opposing recommendation representations compete for retrieval before a validity-ordered, predominantly lexicographic decision. Source-to-validity binding has a localized nonlinear uncertainty region, allowing moderate-to-high validity cues to be less accessible than cues outside that region. Spatial arrangement ordinarily has no effect. The exception is density overload: a recommendation loses binding fidelity when most of its support forms an extended contiguous run and the opposing recommendation itself has multiple active supporters. This competition gate prevents long runs from being penalized merely for containing many sources and leaves adjacent pairs, separated redundancy, and long runs facing a singleton largely unaffected. The strongest member's location within a run provides no protection. In long displays only, an isolated recommendation receives a weak monotonic accessibility loss with later position; there is no terminal-recency process. Failure to retrieve either recommendation produces guessing, followed by a small symmetric response lapse.

**Rationale:** This is a minimal parameter-range recalibration of the accepted model; no mechanism or equation is changed. Raising retrieval_baseline and modestly lowering response_lapse increase availability and execution fidelity for the decisive high-validity recommendation, reducing the excessive lower-validity overrides in Experiments 4 and 5. Narrowing binding_uncertainty_strength preserves the required inverse validity-gap direction while reducing its excessive magnitude in Experiment 6. Contextual_position_loss is shrunk close to zero because its previous strong monotonic penalty generated a wrong-signed contrast in Experiment 7; this removes that artifact without introducing recency. Contiguous_density_overload is strengthened modestly to increase the decrement for extended dense runs in Experiment 8 while retaining the existing competition and run-length gates that protect the null effects elsewhere. Finally, recommendation_saturation_gain is narrowed to nearly zero to suppress the residual positive redundancy effect in Experiment 11. Recommendation-level compression, correlated retrieval, lexicographic priority, and the absence of global source interference remain unchanged.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `validity_compression`: `[0.85, 1.2]`
  - `retrieval_baseline`: `[2.8, 3.5]`
  - `binding_uncertainty_strength`: `[0.8, 1.5]`
  - `binding_uncertainty_center`: `[0.58, 0.68]`
  - `binding_uncertainty_width`: `[0.12, 0.2]`
  - `contiguous_density_overload`: `[1.0, 1.6]`
  - `recommendation_saturation_gain`: `[0.0, 0.003]`
  - `contextual_position_loss`: `[0.0, 0.15]`
  - `response_lapse`: `[0.07, 0.11]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Competition-Normalized Recommendation Race expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
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
    uncertainty_width = max(
        float(parameters["binding_uncertainty_width"]), 1e-9
    )
    overload_strength = float(parameters["contiguous_density_overload"])
    saturation_gain = float(parameters["recommendation_saturation_gain"])
    position_loss = float(parameters["contextual_position_loss"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    objective_strength = np.clip(
        (validities - 0.5) / 0.5, 0.0, 1.0
    )
    subjective_strength = np.power(objective_strength, compression)
    binding_uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    groups = {}
    for direction in (-1, 1):
        members = np.flatnonzero(directions == float(direction)).astype(int)
        if members.size:
            groups[direction] = members

    def longest_contiguous_run(members):
        if members.size == 0:
            return 0
        longest = 1
        current = 1
        for k in range(1, members.size):
            if int(members[k]) == int(members[k - 1]) + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        return int(longest)

    units = {}
    for direction, members in groups.items():
        member_validities = validities[members]
        rep = int(members[int(np.argmax(member_validities))])
        count = int(members.size)

        # All same-direction sources share one recommendation representation.
        # Extra sources give only a tiny and rapidly saturating reliability gain.
        count_gain = saturation_gain * (
            1.0 - np.exp(-2.0 * float(max(count - 1, 0)))
        )

        retrieval_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
            + count_gain
        )

        # Density overload requires an extended contiguous run, concentration
        # of the recommendation's support inside that run, and at least two
        # active sources supporting the competing recommendation.
        run_length = longest_contiguous_run(members)
        opponent = groups.get(-direction, np.asarray([], dtype=int))
        opponent_excess = max(int(opponent.size) - 1, 0)
        competition_gate = 1.0 - np.exp(-float(opponent_excess))
        run_excess = max(run_length - 3, 0)
        if run_excess > 0 and competition_gate > 0.0:
            density = float(run_length) / float(count)
            overload_cost = (
                overload_strength
                * float(run_excess)
                * density * density
                * competition_gate
            )
            retrieval_logit -= overload_cost

        # Only isolated recommendations in substantially loaded displays have
        # a monotonic late-position accessibility loss. There is no recency.
        if count == 1 and n_features > 1:
            display_load = float(np.clip(
                (n_features - 24.0) / 8.0, 0.0, 1.0
            ))
            relative_position = rep / float(n_features - 1)
            retrieval_logit -= (
                position_loss * display_load * relative_position
            )

        units[direction] = {
            "direction": int(direction),
            "representative": rep,
            "priority": float(validities[rep]),
            "raw_logit": float(retrieval_logit),
        }

    # Normalize each recommendation's accessibility against activation of its
    # competitor. The fixed soft competition operation introduces no spatial
    # or source-count interference parameter.
    for direction, unit in units.items():
        opponent = units.get(-direction)
        if opponent is None:
            competition = 0.0
        else:
            competition = float(np.logaddexp(
                0.0, opponent["raw_logit"] - baseline
            ))
        normalized_logit = unit["raw_logit"] - competition
        if normalized_logit >= 0.0:
            retrieval = 1.0 / (1.0 + np.exp(-normalized_logit))
        else:
            ex = np.exp(normalized_logit)
            retrieval = ex / (1.0 + ex)
        unit["retrieval"] = float(np.clip(
            retrieval, 1e-9, 1.0 - 1e-9
        ))

    ordered_units = sorted(
        units.values(),
        key=lambda unit: (-unit["priority"], unit["representative"]),
    )

    # A retrieved higher-validity recommendation terminates the search. Search
    # continues only when that recommendation is unavailable.
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
        p_a_intended = 0.5
    else:
        p_a_intended = intended_a / total

    p_a = (
        (1.0 - lapse) * p_a_intended
        + lapse * (1.0 - p_a_intended)
    )
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

### `pi_11` → slot 1 (via `new_theory`)

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
