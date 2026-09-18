# Round 7 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_9` — SURVIVED ✓

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


### slot 2 — `pi_8` — KILLED ✗

**Description:** Structure-Gated Ensemble Lexicographic Retrieval. Decision makers preserve the communicated validity hierarchy and search encoded recommendation units from highest to lowest validity, but the units available to search depend on the grouping structure of the display. An isolated expert is represented as a source-specific item. Two adjacent experts making the same recommendation are encoded as a compact pair chunk with high, position-invariant fidelity. Contiguous same-direction runs of three or more experts are instead compressed into a single ensemble whose source-to-validity binding becomes progressively less reliable with run length. A uniquely strongest expert embedded inside such a run remains structurally identifiable, whereas a strongest expert at a run boundary receives no such protection from overload. Same-direction experts separated by inactive or opposing experts are represented by a recommendation-level ensemble: additional separated sources provide only a negligible, rapidly saturating retrieval benefit and impose no distance-based crowding cost. Independently, validity labels have a localized nonlinear binding-uncertainty region, so a larger communicated validity gap need not imply more reliable retrieval. Serial position has no terminal-recency bonus. Only isolated sources in sufficiently long displays exhibit a weak subject-specific monotonic accessibility gradient centered near zero, allowing heterogeneous serial-position effects without a systematic pooled recency advantage. Once the encoded units are formed, choice is primarily lexicographic: the first successfully retrieved unit in descending communicated-validity order determines the intended response. Complete retrieval failure produces guessing, and a small execution lapse can reverse the intended response.

**Rationale:** This is a minimal edit to the accepted iteration-2 model. Only the contiguous-run branch changes. Run overload now grows linearly rather than with exponent 0.85, modestly strengthening impairment for five- and six-member runs without changing the accepted overload parameter range. Crucially, 90% of that overload is removed only when there is exactly one highest-validity member and it lies strictly inside the run. This within-chunk structural anchor should preserve retrieval of Experiment 9's embedded uniquely most-valid cue, eliminating the model's generic dense-arrangement disadvantage, while leaving Experiment 8's endpoint-anchored terminal runs fully overloaded. Equal-validity runs and boundary anchors also remain unprotected, retaining the desired reduction of majority overrides in Experiment 5. The rejected average-similarity gate and negative noncontiguous gain are not repeated. Binding uncertainty, pair fidelity, the near-zero heterogeneous serial-position term, and all other accepted mechanisms and parameter ranges remain unchanged.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `validity_compression`: `[0.85, 1.2]`
  - `retrieval_baseline`: `[2.4, 3.2]`
  - `binding_uncertainty_strength`: `[1.0, 2.0]`
  - `binding_uncertainty_center`: `[0.58, 0.68]`
  - `binding_uncertainty_width`: `[0.12, 0.2]`
  - `pair_chunk_fidelity`: `[0.9, 0.99]`
  - `contiguous_run_overload`: `[0.45, 0.65]`
  - `noncontiguous_redundancy_gain`: `[0.0, 0.02]`
  - `long_display_position_slope`: `[-0.35, 0.35]`
  - `response_lapse`: `[0.08, 0.14]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Structure-Gated Ensemble Lexicographic Retrieval expects "
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
    pair_fidelity = float(parameters["pair_chunk_fidelity"])
    run_overload = float(parameters["contiguous_run_overload"])
    separated_gain = float(
        parameters["noncontiguous_redundancy_gain"]
    )
    position_slope = float(parameters["long_display_position_slope"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Validity labels are transformed only for binding fidelity. Search rank
    # below continues to use communicated validity itself.
    objective_strength = np.clip(
        (validities - 0.5) / 0.5, 0.0, 1.0
    )
    subjective_strength = np.power(objective_strength, compression)
    binding_uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / uncertainty_width
        )
    )

    def logistic(x):
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    def representative(members):
        member_array = np.asarray(members, dtype=int)
        vals = validities[member_array]
        # Earliest position resolves exact communicated-validity ties.
        return int(member_array[int(np.argmax(vals))])

    # Identify maximal contiguous runs of active sources giving the same
    # recommendation. Inactive cues and opposite recommendations both break
    # a run; no metric of generic spatial distance is used.
    runs = []
    i = 0
    while i < active.size:
        current = [int(active[i])]
        j = i + 1
        while (
            j < active.size
            and int(active[j]) == int(active[j - 1]) + 1
            and directions[int(active[j])] == directions[int(active[i])]
        ):
            current.append(int(active[j]))
            j += 1
        runs.append(current)
        i = j

    units = []
    isolated_by_direction = {-1: [], 1: []}

    for members in runs:
        direction = int(np.sign(directions[members[0]]))
        rep = representative(members)
        base_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
        )

        if len(members) == 1:
            isolated_by_direction[direction].append(rep)
            continue

        if len(members) == 2:
            # A same-direction adjacent pair is a compact chunk. Pair fidelity
            # supplies a high retrieval floor and there is no position term.
            ordinary = logistic(base_logit)
            retrieval = pair_fidelity + (1.0 - pair_fidelity) * ordinary
            kind = "pair"
        else:
            # Runs of three or more are over-compressed. Internal source
            # attribution deteriorates with run length, independently of the
            # run's absolute location. A uniquely strongest member strictly
            # inside the run remains structurally identifiable; an endpoint
            # anchor or a validity tie receives no protection.
            excess_length = float(len(members) - 2)
            overload_cost = run_overload * np.power(excess_length, 1.0)
            member_array = np.asarray(members, dtype=int)
            member_validities = validities[member_array]
            strongest = np.flatnonzero(
                np.isclose(
                    member_validities,
                    np.max(member_validities),
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            uniquely_embedded = (
                strongest.size == 1
                and 0 < int(strongest[0]) < len(members) - 1
            )
            if uniquely_embedded:
                overload_cost *= 0.10
            retrieval = logistic(base_logit - overload_cost)
            kind = "run"

        units.append({
            "priority": float(validities[rep]),
            "tie_position": int(min(members)),
            "direction": direction,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
            "kind": kind,
        })

    # Separated isolated sources supporting the same option enter one
    # recommendation-level ensemble. Source count has only a small,
    # saturating benefit and never generates a spacing or crowding penalty.
    for direction in (-1, 1):
        members = isolated_by_direction[direction]
        if not members:
            continue

        rep = representative(members)
        retrieval_logit = (
            baseline
            - uncertainty_strength * binding_uncertainty[rep]
        )

        if len(members) == 1:
            # Under substantial display load, isolated sources receive a weak
            # subject-specific monotonic accessibility gradient. It contains
            # no terminal-specific recency bonus.
            load_factor = np.clip((n_features - 24.0) / 8.0, 0.0, 1.0)
            if n_features > 1:
                relative_position = rep / float(n_features - 1)
            else:
                relative_position = 0.0
            retrieval_logit -= (
                position_slope * load_factor * relative_position
            )
            kind = "singleton"
        else:
            extra_sources = float(len(members) - 1)
            saturation = 1.0 - np.exp(-extra_sources)
            retrieval_logit += separated_gain * saturation
            kind = "separated_ensemble"

        retrieval = logistic(retrieval_logit)
        units.append({
            "priority": float(validities[rep]),
            "tie_position": int(min(members)),
            "direction": direction,
            "retrieval": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),
            "kind": kind,
        })

    # Communicated validity supplies the primary lexicographic hierarchy.
    units.sort(key=lambda u: (-u["priority"], u["tie_position"]))

    reach = 1.0
    intended_a = 0.0
    intended_b = 0.0
    for unit in units:
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

    # A symmetric execution lapse reverses the intended binary response.
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_10` → slot 2 (via `new_theory`)

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
