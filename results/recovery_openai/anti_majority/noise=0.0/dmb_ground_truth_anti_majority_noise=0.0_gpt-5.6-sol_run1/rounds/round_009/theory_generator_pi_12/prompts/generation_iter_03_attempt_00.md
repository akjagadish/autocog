# generation_iter_03_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_11`).

Replace pi_11 with a load-adaptive dual-route theory rather than another causal-copy account. The new theory should posit competition between (a) validity-directed search and (b) display-order anchoring. In short displays or when validity differences are conspicuous, validity-directed search predominates and yields the lexicographic signatures seen in Experiments 3-8. As display length and visual search load increase, the first discriminating recommendation becomes an anchor; later evidence can overturn it only when its validity advantage or accumulated diagnostic support exceeds a load-dependent threshold. This route should generate strong early-cue control for the 80%-versus-82% conflict in the 32-feature Experiment 2, a negative rather than positive late-position gradient in Experiment 9, and negligible position effects around 24 features as in Experiments 12 and 14. Same-direction sources may be compressed, but their cost should depend on count, diagnosticity, and working-memory concentration—not on validity homogeneity, adjacency, run structure, or opponent count. Count costs should be weak at moderate validity, consistent with Experiments 1 and 13, but sharply stronger for very large, extremely high-validity bundles, consistent with Experiments 10 and 20. Include stable subject-level variation in reliance on the order-anchor versus validity route to accommodate Experiment 2's larger observed heterogeneity. The model should use a validity-gap-sensitive override function, a display-load-gated primacy term with the empirically supported sign, and a high-validity count-overload term that is invariant to within-bundle validity dispersion. It should make joint predictions for absolute choice rates and positional/count contrasts, rather than adding isolated mechanisms tailored to individual metrics.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.634)

**Description**
Validity-Gated Intrinsic Recommendation Capacity. On each trial, all active experts favoring the same option are obligatorily compressed into one recommendation representation, irrespective of adjacency, spacing, pairing, or the location of the strongest source. The representation is indexed and ranked by its strongest communicated validity. Additional supporters yield only a negligible, rapidly saturating reliability increment rather than independent retrieval opportunities. Retrieval is limited by the intrinsic representational concentration of each recommendation: jointly binding many same-direction sources becomes costly only at very high diagnosticity, with load increasing smoothly with its own source count. This capacity cost belongs to that recommendation and is never gated by how many sources support its opponent. Thus an opponent acquiring separated supporters cannot suddenly impair an unchanged run. Retrieval also contains a narrow localized nonlinear source-validity binding uncertainty. Retrieved recommendations enter a predominantly lexicographic race ordered by their strongest communicated validity, with accessibility normalized against the competing recommendation. Spatial organization ordinarily has no role. Only an isolated recommendation in a very long display receives a monotonic late-position accessibility gradient, gated so that it is appreciable around 32 features but negligible around 24. If neither recommendation is retrieved, the decision maker guesses; a small symmetric response lapse follows.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0564)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.1348 (var=0.0021)
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.1531 (var=0.0047)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=-0.0900 (var=0.0174)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=-0.0234 (var=0.0013)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=-0.2950 (var=0.0113)
- Experiment 9: real=-0.0056 (var=0.0030) vs this=-0.0047 (var=0.0075)
- Experiment 10: real=-0.0045 (var=0.0098) vs this=0.0035 (var=0.0146)
- Experiment 11: real=-0.0225 (var=0.0063) vs this=-0.0225 (var=0.0113)
- Experiment 12: real=0.0150 (var=0.0426) vs this=0.0217 (var=0.0137)
- Experiment 13: real=-0.0030 (var=0.0059) vs this=0.0273 (var=0.0083)
- Experiment 14: real=-0.0183 (var=0.0044) vs this=0.0025 (var=0.0048)
- Experiment 15: real=-0.0205 (var=0.0094) vs this=0.0340 (var=0.0153)
- Experiment 16: real=0.0550 (var=0.0207) vs this=0.0337 (var=0.0218)
- Experiment 17: real=-0.0200 (var=0.0140) vs this=0.0287 (var=0.0211)
- Experiment 18: real=0.1375 (var=0.0106) vs this=0.4850 (var=0.0238)
- Experiment 19: real=0.0212 (var=0.0051) vs this=-0.0038 (var=0.0181)
- Experiment 20: real=0.2125 (var=0.0153) vs this=0.8329 (var=0.0018)


---

### `pi_11` (overall score: 0.526)

**Description**
Causal-Copy Discounted Lexicographic Choice with Local Priority Uncertainty. Decision makers compress all discriminating experts supporting the same option into an option-level recommendation and infer whether agreement reflects copied reports, shared evidence, or a common source. Common-cause suspicion rises nonlinearly with supporter count and is amplified by conspicuously high, homogeneous validities, actively discounting the recommendation's epistemic accessibility. Recommendations are ordinarily considered lexicographically by their strongest communicated validity. However, within a very narrow priority-gap neighborhood, validity ranks are not perfectly resolved: a small proportion of comparisons probabilistically interchange the two recommendation priorities. This local uncertainty permits an equal-validity or nearly equal-validity opposing recommendation to govern some choices without weakening lexicographic dominance for clearly separated validities. The causal discount remains independent of opponent count, adjacency, spacing, run structure, strongest-source placement, and terminal clustering. Separately, source-validity binding has a localized nonlinear uncertainty region, and isolated recommendations suffer a weak late-position accessibility loss only in genuinely long displays. Failure to access either recommendation leads to guessing, followed by a small response lapse.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- n_features: n_features
- validities: validities
- lexicographic_reliability: [3.0, 3.5]
- validity_compression: [0.85, 1.15]
- binding_uncertainty_strength: [1.7, 2.2]
- binding_uncertainty_center: [0.625, 0.65]
- binding_uncertainty_width: [0.075, 0.105]
- common_cause_count_threshold: [4.0, 4.4]
- common_cause_count_slope: [2.0, 2.7]
- common_cause_homogeneity_scale: [0.028, 0.038]
- redundancy_validity_threshold: [0.895, 0.91]
- redundancy_validity_width: [0.016, 0.024]
- redundancy_discount_strength: [7.2, 8.0]
- long_display_position_loss: [0.0, 0.15]
- near_tie_priority_mixing: [0.12, 0.18]
- response_lapse: [0.1, 0.16]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0736)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.2400)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.1615 (var=0.0015)
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.1181 (var=0.0022)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=-0.1088 (var=0.0192)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=-0.0075 (var=0.0012)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=-0.2094 (var=0.0172)
- Experiment 9: real=-0.0056 (var=0.0030) vs this=0.0262 (var=0.0133)
- Experiment 10: real=-0.0045 (var=0.0098) vs this=0.0128 (var=0.0124)
- Experiment 11: real=-0.0225 (var=0.0063) vs this=-0.0050 (var=0.0103)
- Experiment 12: real=0.0150 (var=0.0426) vs this=-0.0133 (var=0.0171)
- Experiment 13: real=-0.0030 (var=0.0059) vs this=-0.0022 (var=0.0095)
- Experiment 14: real=-0.0183 (var=0.0044) vs this=-0.0142 (var=0.0053)
- Experiment 15: real=-0.0205 (var=0.0094) vs this=-0.0032 (var=0.0097)
- Experiment 16: real=0.0550 (var=0.0207) vs this=0.0162 (var=0.0214)
- Experiment 17: real=-0.0200 (var=0.0140) vs this=0.0063 (var=0.0164)
- Experiment 18: real=0.1375 (var=0.0106) vs this=0.1663 (var=0.0110)
- Experiment 19: real=0.0212 (var=0.0051) vs this=0.6781 (var=0.0108)
- Experiment 20: real=0.2125 (var=0.0153) vs this=0.7515 (var=0.0034)


---

### `pi_7` (overall score: 0.497)

**Description**
Chunked Source-Retrieval Lexicographic Search. People intend to inspect expert recommendations in descending order of communicated validity, but a cue can govern choice only if its source identity, validity label, and current recommendation are jointly retrieved. Retrieval interference is source-based rather than spatial: active sources recommending the same option interfere in proportion to their subjective-validity similarity, regardless of their physical distance. Consequently, rearranging a fixed source set does not release interference. Adjacent sources giving the same recommendation can instead be encoded as a single recommendation chunk. A chunk enters the validity-ordered search at the rank of its most valid member and, if retrieved, directly recommends an option; this makes a translated two-source cluster nearly position-invariant. Chunk retrieval nevertheless retains the attribution load generated by its constituent sources, so increasing a redundant set can overload retrieval even when the recommendation is coherent. In addition, sets of at least three similar same-direction sources share a correlated source-attribution gate. If that common binding episode fails, all representations carrying that recommendation are temporarily unavailable; thus redundant cues cannot always rescue one another through independent retrieval attempts. The shared gate depends on source count and validity similarity but not physical spacing. Isolated sources have a very weak monotonic late-position accessibility loss, while chunks receive no positional bonus or penalty; there is no terminal-recency mechanism. Communicated validity is nonlinearly compressed, and source binding has a localized uncertainty peak in the moderate-to-high diagnosticity region. This can make a moderately high-validity source less retrievable than nearly nondiagnostic sources and thereby produce inverse validity-gap effects. Search remains lexicographic after retrieval: the first retrieved discriminating cue or chunk determines the intended response. If nothing is retrieved, the person guesses, and a small independent execution lapse can reverse the intended response.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Chunked Source-Retrieval Lexicographic Search expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but "
            f"n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities must have length {n_features}; got "
            f"shape {validities.shape}."
        )

    compression = float(parameters["validity_compression"])
    baseline = float(parameters["retrieval_baseline"])
    uncertainty_strength = float(parameters["uncertainty_strength"])
    uncertainty_center = float(parameters["uncertainty_center"])
    uncertainty_width = float(parameters["uncertainty_width"])
    load_strength = float(parameters["similar_source_load"])
    chunk_probability = float(parameters["chunk_probability"])
    position_slope = float(parameters["singleton_position_slope"])
    lapse = float(parameters["response_lapse"])

    differences = stim[0] - stim[1]
    directions = np.sign(differences)
    active = np.flatnonzero(directions != 0.0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Nonlinear subjective diagnosticity. Search order still uses the
    # communicated validities, whereas similarity and uncertainty operate on
    # this compressed internal scale.
    objective_strength = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    subjective_strength = np.power(objective_strength, compression)

    width = max(uncertainty_width, 1e-9)
    uncertainty = np.exp(
        -0.5 * np.square(
            (subjective_strength - uncertainty_center) / width
        )
    )

    # Similarity is deliberately independent of display distance. Only active
    # sources giving the same recommendation interfere. The fixed bandwidth
    # avoids introducing a separate spatial or neighborhood parameter.
    similarity_bandwidth = 0.18
    source_load = np.zeros(n_features, dtype=np.float64)
    for j in active:
        same = active[(active != j) & (directions[active] == directions[j])]
        if same.size:
            source_load[j] = float(np.sum(np.exp(
                -np.abs(
                    subjective_strength[same] - subjective_strength[j]
                ) / similarity_bandwidth
            )))

    # Similar same-direction sets also share a correlated attribution gate.
    # The gate is exact for a recommendation direction, so multiple units do
    # not provide independent rescue opportunities when source attribution
    # fails. Pairs are exempt to preserve stable translation of a two-cue
    # chunk. The equation contains no display-distance term.
    shared_gates = {}
    for direction_key in (-1, 1):
        group = active[directions[active] == float(direction_key)]
        if group.size <= 2:
            shared_gates[direction_key] = 1.0
            continue

        group_ambiguity = float(np.mean(source_load[group]))
        gate_logit = (
            baseline
            - 0.45 * load_strength * max(group_ambiguity, 0.0)
        )
        if gate_logit >= 0.0:
            gate_prob = 1.0 / (1.0 + np.exp(-gate_logit))
        else:
            ez = np.exp(gate_logit)
            gate_prob = ez / (1.0 + ez)
        shared_gates[direction_key] = float(np.clip(
            gate_prob, 1e-9, 1.0 - 1e-9
        ))

    def make_units(use_chunks):
        units = []
        active_set = set(int(x) for x in active)
        consumed = set()

        for j_raw in active:
            j = int(j_raw)
            if j in consumed:
                continue

            members = [j]
            if use_chunks:
                k = j + 1
                while (
                    k in active_set
                    and directions[k] == directions[j]
                ):
                    members.append(k)
                    k += 1

            for m in members:
                consumed.add(m)

            # A recommendation chunk is ranked by its most valid source.
            member_array = np.asarray(members, dtype=int)
            member_validities = validities[member_array]
            best_local = int(np.argmax(member_validities))
            representative = int(member_array[best_local])

            # Constituent-source ambiguity remains present after chunking.
            # Averaging source loads prevents mere chunk length from becoming
            # evidence accumulation, while source_load itself increases with
            # the number and similarity of redundant sources.
            attribution_load = float(np.mean(source_load[member_array]))
            load_cost = load_strength * np.power(
                max(attribution_load, 0.0), 1.25
            )

            retrieval_logit = (
                baseline
                - uncertainty_strength * uncertainty[representative]
                - load_cost
            )

            # Only an unchunked singleton receives the weak monotonic
            # late-display accessibility loss. There is no recency bonus.
            if len(members) == 1 and n_features > 1:
                relative_position = representative / float(n_features - 1)
                retrieval_logit -= position_slope * relative_position

            if retrieval_logit >= 0.0:
                retrieval_prob = 1.0 / (1.0 + np.exp(-retrieval_logit))
            else:
                ez = np.exp(retrieval_logit)
                retrieval_prob = ez / (1.0 + ez)

            units.append({
                "priority": float(validities[representative]),
                "tie_position": min(members),
                "direction": float(directions[j]),
                "gate_key": int(np.sign(directions[j])),
                "retrieval": float(np.clip(
                    retrieval_prob, 1e-9, 1.0 - 1e-9
                )),
            })

        # Descending validity, with display position used only for exact ties.
        units.sort(key=lambda u: (-u["priority"], u["tie_position"]))
        return units

    def cascade_probability_a(units, available_directions=None):
        reach = 1.0
        intended_a = 0.0
        intended_b = 0.0

        for unit in units:
            if (
                available_directions is not None
                and unit["gate_key"] not in available_directions
            ):
                continue
            stop = reach * unit["retrieval"]
            if unit["direction"] > 0.0:
                intended_a += stop
            else:
                intended_b += stop
            reach *= 1.0 - unit["retrieval"]

        # Failure to retrieve any discriminating representation yields a guess.
        intended_a += 0.5 * reach
        intended_b += 0.5 * reach
        total = intended_a + intended_b
        if not np.isfinite(total) or total <= 0.0:
            return 0.5
        return float(intended_a / total)

    def gated_cascade_probability_a(units):
        # Exactly marginalize the two recommendation-level gate states rather
        # than drawing a latent gate inside predict.
        p_positive = shared_gates[1]
        p_negative = shared_gates[-1]
        result = 0.0
        for positive_open, p_pos_state in (
            (False, 1.0 - p_positive),
            (True, p_positive),
        ):
            for negative_open, p_neg_state in (
                (False, 1.0 - p_negative),
                (True, p_negative),
            ):
                available = set()
                if positive_open:
                    available.add(1)
                if negative_open:
                    available.add(-1)
                result += (
                    p_pos_state
                    * p_neg_state
                    * cascade_probability_a(units, available)
                )
        return float(result)

    # Marginalize the encoding state rather than randomly choosing it inside
    # predict. This keeps returned probabilities stable and interpretable.
    p_a_unchunked = gated_cascade_probability_a(make_units(False))
    p_a_chunked = gated_cascade_probability_a(make_units(True))
    p_a_intended = (
        chunk_probability * p_a_chunked
        + (1.0 - chunk_probability) * p_a_unchunked
    )

    # Independent execution lapse reverses the intended binary response.
    p_a = (
        (1.0 - lapse) * p_a_intended
        + lapse * (1.0 - p_a_intended)
    )
    p_a = float(np.clip(p_a, 0.0, 1.0))
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- validity_compression: [0.85, 1.2]
- retrieval_baseline: [2.4, 3.2]
- uncertainty_strength: [1.0, 2.0]
- uncertainty_center: [0.58, 0.68]
- uncertainty_width: [0.12, 0.2]
- similar_source_load: [0.45, 0.9]
- chunk_probability: [0.85, 0.98]
- singleton_position_slope: [0.0, 0.06]
- response_lapse: [0.08, 0.16]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0196)
- Experiment 3: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0384)
- Experiment 4: real=0.1620 (var=0.0198) vs this=0.1578 (var=0.0020)
- Experiment 5: real=0.1613 (var=0.0081) vs this=0.1775 (var=0.0051)
- Experiment 6: real=-0.1275 (var=0.0523) vs this=-0.1250 (var=0.0233)
- Experiment 7: real=-0.0543 (var=0.0193) vs this=-0.0026 (var=0.0021)
- Experiment 8: real=-0.2675 (var=0.0382) vs this=-0.2888 (var=0.0210)
- Experiment 9: real=-0.0056 (var=0.0030) vs this=0.0225 (var=0.0115)
- Experiment 10: real=-0.0045 (var=0.0098) vs this=-0.0140 (var=0.0142)
- Experiment 11: real=-0.0225 (var=0.0063) vs this=-0.4881 (var=0.0165)
- Experiment 12: real=0.0150 (var=0.0426) vs this=-0.0079 (var=0.0203)
- Experiment 13: real=-0.0030 (var=0.0059) vs this=-0.0313 (var=0.0119)
- Experiment 14: real=-0.0183 (var=0.0044) vs this=-0.1175 (var=0.0113)
- Experiment 15: real=-0.0205 (var=0.0094) vs this=-0.0058 (var=0.0258)
- Experiment 16: real=0.0550 (var=0.0207) vs this=0.0663 (var=0.0387)
- Experiment 17: real=-0.0200 (var=0.0140) vs this=0.0550 (var=0.0204)
- Experiment 18: real=0.1375 (var=0.0106) vs this=0.3725 (var=0.0355)
- Experiment 19: real=0.0212 (var=0.0051) vs this=0.1681 (var=0.0117)
- Experiment 20: real=0.2125 (var=0.0153) vs this=0.6837 (var=0.0053)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0541 -> ACCEPTED
- iter 2: loss=0.0410 -> ACCEPTED
- iter 3: loss=0.0418 -> REJECTED
Running-best (last ACCEPTED) base: iter 2 at loss=0.0410 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify whether choices are predominantly controlled by the highest-ranked
    # discriminating cue on trials where at least three net lower-cue votes point
    # in the opposite direction.
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        nz = np.flatnonzero(d != 0)
        if nz.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        first = int(nz[0])
        ttb_sign = 1.0 if d[first] > 0 else -1.0  # +1 means A, -1 means B
        tally_margin = float(np.sum(np.sign(d)))  # positive means tally favors A

        # Retain only strong conflicts: tally favors the other option by at
        # least three net cue votes. This selects patterns 2, 3, 5, and 6 and
        # their A/B reversals in the specified design.
        if ttb_sign * tally_margin < 0 and abs(tally_margin) >= 3.0:
            ttb_response = 0 if ttb_sign > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

    if len(agreements) == 0:
        return 0.5

    # A categorical dominance index: 1 if the top discriminating cue controls
    # a strict majority of these high-conflict choices, otherwise 0.
    return float(np.mean(agreements) > 0.5)

```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0196)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 1.0000 (var=0.0384)
- pi_7: 1.0000 (var=0.0000)
- pi_8: 1.0000 (var=0.0000)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)
- pi_11: 1.0000 (var=0.0000)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """AIC model-selection indicator for graded tally-margin sensitivity."""
    if data is None or len(data) == 0:
        return 0.0

    margins = []
    successes = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signed_margin = int(np.sum(a > b) - np.sum(b > a))
        if signed_margin == 0:
            continue

        winner = 0 if signed_margin > 0 else 1
        margins.append(abs(signed_margin))
        successes.append(float(int(row["response"]) == winner))

    if len(successes) == 0:
        return 0.0

    margins = np.asarray(margins, dtype=int)
    successes = np.asarray(successes, dtype=float)
    levels = np.arange(1, 7, dtype=float)
    n_by_margin = np.asarray([np.sum(margins == m) for m in range(1, 7)], dtype=float)
    k_by_margin = np.asarray(
        [np.sum(successes[margins == m]) for m in range(1, 7)], dtype=float
    )
    keep = n_by_margin > 0
    levels = levels[keep]
    n_by_margin = n_by_margin[keep]
    k_by_margin = k_by_margin[keep]

    if len(levels) < 2:
        return 0.0

    # Best margin-invariant winner probability: the observable implication
    # of TTB for this design. Its parameterization permits probabilities
    # arbitrarily close to the full interval [0.5, 1].
    n = float(np.sum(n_by_margin))
    k = float(np.sum(k_by_margin))
    q_const = float(np.clip(k / n, 0.5 + 1e-12, 1.0 - 1e-12))
    ll_ttb = k * np.log(q_const) + (n - k) * np.log1p(-q_const)

    # Maximize the exact Tallying likelihood over its stated beta and lapse
    # ranges. The grid is denser where sigmoid(beta*m) changes most rapidly.
    beta = np.concatenate(
        [np.linspace(0.1, 3.0, 241), np.linspace(3.1, 20.0, 170)]
    )
    epsilon = np.linspace(0.0, 0.5, 251)

    core = 1.0 / (1.0 + np.exp(-beta[:, None] * levels[None, :]))
    q = (
        (1.0 - epsilon[None, :, None]) * core[:, None, :]
        + epsilon[None, :, None] * 0.5
    )
    q = np.clip(q, 1e-12, 1.0 - 1e-12)

    ll = np.sum(
        k_by_margin[None, None, :] * np.log(q)
        + (n_by_margin - k_by_margin)[None, None, :] * np.log1p(-q),
        axis=2,
    )
    ll_tally = float(np.max(ll))

    # Tallying has one additional identifiable probability-shape parameter.
    # Thus AIC favors it exactly when the likelihood-ratio deviance exceeds 2.
    deviance = 2.0 * (ll_tally - ll_ttb)
    return float(deviance > 2.0)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 1.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0736)
- pi_3: 0.0000 (var=0.0900)
- pi_4: 0.0000 (var=0.1056)
- pi_5: 0.0000 (var=0.0564)
- pi_6: 0.0000 (var=0.0564)
- pi_7: 0.0000 (var=0.0196)
- pi_8: 0.0000 (var=0.0000)
- pi_9: 0.0000 (var=0.0384)
- pi_10: 0.0000 (var=0.0000)
- pi_11: 0.0000 (var=0.0736)

### Experiment 3
**Design**
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return 1 when first-discriminating-cue consistency is at least 90%."""
    if len(data) == 0:
        return float("nan")

    follows_first_cue = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        expected_response = None
        for j in range(min(a.size, b.size)):
            if a[j] > b[j]:
                expected_response = 0
                break
            if b[j] > a[j]:
                expected_response = 1
                break

        if expected_response is not None:
            follows_first_cue.append(int(int(response) == expected_response))

    if len(follows_first_cue) == 0:
        return float("nan")

    consistency = float(np.mean(follows_first_cue))
    return float(consistency >= 0.90)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 3 (most recent): 0.0000 (var=0.2100) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.2400)
- pi_3: 1.0000 (var=0.1056)
- pi_2: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0564)
- pi_5: 0.0000 (var=0.1204)
- pi_6: 0.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.0384)
- pi_8: 0.0000 (var=0.0384)
- pi_9: 0.0000 (var=0.0196)
- pi_10: 0.0000 (var=0.0564)
- pi_11: 0.0000 (var=0.2400)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Proportion of choices following the net direction of all weak cues on
    # trials where that direction opposes Expert 2. These are conflict levels
    # M=2,...,6; the M=1 cancellation trials are omitted.
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        directions = np.sign(a - b)

        # Feature 0 always ties. Features 1 onward are the equally valid weak
        # cues that the advocated theory integrates.
        weak_net = float(np.sum(directions[1:]))
        expert2_direction = float(directions[1])

        # Retain only genuine reversal trials: the weak-cue aggregate points
        # opposite to the first discriminating cue used by TTB.
        if weak_net == 0.0 or expert2_direction == 0.0:
            continue
        if np.sign(weak_net) == np.sign(expert2_direction):
            continue

        integrated_winner = 0 if weak_net > 0.0 else 1
        scores.append(float(int(row["response"]) == integrated_winner))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1620 (var=0.0198)
**Previous candidate values (this loop):**
  - iter 1: 0.0912 (var=0.0013) (Δ vs real -0.0708)
  - iter 2: 0.0960 (var=0.0013) (Δ vs real -0.0660)
  - iter 3 (most recent): 0.0993 (var=0.0013) (Δ vs real -0.0628)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5913 (var=0.0137)
- pi_1: 0.1670 (var=0.0108)
- pi_2: 0.8492 (var=0.0103)
- pi_4: 0.1683 (var=0.0038)
- pi_5: 0.1505 (var=0.0020)
- pi_6: 0.1693 (var=0.0024)
- pi_7: 0.1578 (var=0.0020)
- pi_8: 0.1487 (var=0.0015)
- pi_9: 0.1457 (var=0.0026)
- pi_10: 0.1348 (var=0.0021)
- pi_11: 0.1615 (var=0.0015)

### Experiment 5
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which the 70%-cue majority overrides the opposing 72% cue
    on the two strongest conflict levels (five or six active 70% cues)."""
    matches = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 9 or b.size < 9:
            continue

        # Features 3--8 (zero-based indices 2:8) are the six 70% cues.
        d70 = a[2:8] - b[2:8]
        active = d70[d70 != 0]
        if active.size < 5:
            continue

        # Retain only designed conflicts in which the active 70% cues agree
        # and the 72% cue recommends the opposite option.
        direction = np.sign(active[0])
        if not np.all(np.sign(active) == direction):
            continue
        d72 = a[8] - b[8]
        if d72 == 0 or np.sign(d72) == direction:
            continue

        majority_choice = 0 if direction > 0 else 1
        response = int(row["response"])
        if response in (0, 1):
            matches.append(float(response == majority_choice))

    if len(matches) == 0:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1613 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.1237 (var=0.0040) (Δ vs real -0.0375)
  - iter 2: 0.1181 (var=0.0026) (Δ vs real -0.0431)
  - iter 3 (most recent): 0.1062 (var=0.0031) (Δ vs real -0.0550)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1450 (var=0.0101)
- pi_4: 0.8588 (var=0.0039)
- pi_2: 0.8450 (var=0.0101)
- pi_3: 0.0369 (var=0.0012)
- pi_5: 0.1588 (var=0.0035)
- pi_6: 0.1850 (var=0.0060)
- pi_7: 0.1775 (var=0.0051)
- pi_8: 0.1844 (var=0.0040)
- pi_9: 0.1688 (var=0.0073)
- pi_10: 0.1531 (var=0.0047)
- pi_11: 0.1181 (var=0.0022)

### Experiment 6
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.55, 0.95, 0.5525, 0.55, 0.5, 0.5868,
        0.55, 0.95, 0.6343, 0.55, 0.5, 0.6901,
        0.55, 0.95, 0.7504, 0.55, 0.5, 0.8177
    ], dtype=float)
    log_odds = np.log(validities / (1.0 - validities))

    gaps = []
    follows_higher = []

    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 2:
            continue

        j0, j1 = int(discriminating[0]), int(discriminating[1])
        if validities[j0] == validities[j1]:
            continue
        high = j0 if validities[j0] > validities[j1] else j1
        low = j1 if high == j0 else j0

        high_favors_a = diff[high] > 0
        chose_a = int(response) == 0
        follows_higher.append(float(chose_a == high_favors_a))
        gaps.append(float(log_odds[high] - log_odds[low]))

    gaps = np.asarray(gaps, dtype=float)
    follows_higher = np.asarray(follows_higher, dtype=float)
    if gaps.size == 0:
        return float('nan')

    unique_gaps = np.unique(np.round(gaps, 10))
    if unique_gaps.size < 2:
        return float('nan')

    smallest = unique_gaps[0]
    largest = unique_gaps[-1]
    low_mask = np.isclose(gaps, smallest, atol=1e-9, rtol=0.0)
    high_mask = np.isclose(gaps, largest, atol=1e-9, rtol=0.0)

    return float(
        np.mean(follows_higher[high_mask])
        - np.mean(follows_higher[low_mask])
    )
```

**Observed (real) value:** -0.1275 (var=0.0523)
**Previous candidate values (this loop):**
  - iter 1: -0.1625 (var=0.0169) (Δ vs real -0.0350)
  - iter 2: -0.1212 (var=0.0165) (Δ vs real +0.0063)
  - iter 3 (most recent): -0.1225 (var=0.0184) (Δ vs real +0.0050)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7075 (var=0.0151)
- pi_1: 0.0113 (var=0.0129)
- pi_2: -0.0112 (var=0.0229)
- pi_3: 0.1400 (var=0.0120)
- pi_5: -0.1363 (var=0.0200)
- pi_6: -0.0913 (var=0.0324)
- pi_7: -0.1250 (var=0.0233)
- pi_8: -0.1362 (var=0.0170)
- pi_9: -0.0962 (var=0.0183)
- pi_10: -0.0900 (var=0.0174)
- pi_11: -0.1088 (var=0.0192)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure whether choices favoring the decisive high-validity cue decline
    # as that cue moves toward the end of the display.
    positions = np.array([2, 7, 13, 19, 25, 31], dtype=int)
    # Approximate centered accessibility-cascade profile. The weights sum to
    # zero, so any position-invariant choice reliability contributes zero.
    weights = np.array([0.27, 0.24, 0.16, -0.02, -0.25, -0.40], dtype=float)

    weighted_choices = []
    used_weights = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # In this design, exactly one of these six positions contains the
        # higher-validity target cue that opposes the two early cues.
        target_matches = [j for j in positions if j < diff.size and diff[j] != 0]
        if len(target_matches) != 1:
            continue

        target_pos = target_matches[0]
        k = int(np.where(positions == target_pos)[0][0])
        w = weights[k]

        response = int(row['response'])
        target_choice = 1.0 if ((diff[target_pos] > 0 and response == 0) or
                                (diff[target_pos] < 0 and response == 1)) else 0.0
        weighted_choices.append(w * target_choice)
        used_weights.append(abs(w))

    if len(weighted_choices) == 0:
        return float('nan')

    # Absolute-weight normalization gives an interpretable bounded contrast
    # while preserving the matched-filter ordering of the six conditions.
    denom = float(np.sum(used_weights))
    if denom <= 0.0:
        return float('nan')
    return float(np.sum(weighted_choices) / denom)

```

**Observed (real) value:** -0.0543 (var=0.0193)
**Previous candidate values (this loop):**
  - iter 1: 0.0052 (var=0.0015) (Δ vs real +0.0595)
  - iter 2: 0.0258 (var=0.0020) (Δ vs real +0.0801)
  - iter 3 (most recent): -0.0003 (var=0.0010) (Δ vs real +0.0540)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0015 (var=0.0020)
- pi_5: 0.2664 (var=0.0022)
- pi_2: -0.0100 (var=0.0017)
- pi_3: 0.0028 (var=0.0006)
- pi_4: 0.2945 (var=0.0059)
- pi_6: -0.0510 (var=0.0058)
- pi_7: -0.0026 (var=0.0021)
- pi_8: 0.0028 (var=0.0019)
- pi_9: -0.0071 (var=0.0015)
- pi_10: -0.0234 (var=0.0013)
- pi_11: -0.0075 (var=0.0012)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure choice of the option favored by the highest-validity cue,
    # separately for sparse versus highly redundant terminal-cue conditions.
    high_redundancy = []
    low_redundancy = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 24 or b.size < 24:
            continue

        diff = a - b
        # Features 19--24 (zero-based indices 18--23) are the nested set
        # of high-validity cues favoring the same option.
        redundancy = int(np.count_nonzero(diff[18:24]))

        # Feature 24 has validity .95 and is TTB's decisive cue on every
        # experimental trial. Code whether the response follows that cue.
        if diff[23] > 0:
            followed_top_cue = float(int(row["response"]) == 0)
        elif diff[23] < 0:
            followed_top_cue = float(int(row["response"]) == 1)
        else:
            continue

        if redundancy in (1, 2):
            low_redundancy.append(followed_top_cue)
        elif redundancy in (5, 6):
            high_redundancy.append(followed_top_cue)

    if len(low_redundancy) == 0 or len(high_redundancy) == 0:
        return float("nan")

    return float(np.mean(high_redundancy) - np.mean(low_redundancy))

```

**Observed (real) value:** -0.2675 (var=0.0382)
**Previous candidate values (this loop):**
  - iter 1: -0.4206 (var=0.0100) (Δ vs real -0.1531)
  - iter 2: -0.3763 (var=0.0045) (Δ vs real -0.1088)
  - iter 3 (most recent): -0.3706 (var=0.0051) (Δ vs real -0.1031)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3306 (var=0.0175)
- pi_1: 0.0169 (var=0.0052)
- pi_2: 0.5481 (var=0.0232)
- pi_3: -0.0113 (var=0.0033)
- pi_4: 0.0087 (var=0.0075)
- pi_6: -0.2762 (var=0.0511)
- pi_7: -0.2888 (var=0.0210)
- pi_8: -0.2625 (var=0.0123)
- pi_9: -0.2544 (var=0.0160)
- pi_10: -0.2950 (var=0.0113)
- pi_11: -0.2094 (var=0.0172)

### Experiment 9
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Choice congruence with the uniquely most valid cue (Expert 17, index 16),
    # contrasted between the two sparsest and four densest cue arrangements.
    congruent_sparse = []
    congruent_dense = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        if diff.size <= 16 or diff[16] == 0:
            continue

        leading_direction = np.sign(diff[16])
        leading_response = 0 if leading_direction > 0 else 1
        congruent = float(int(row["response"]) == leading_response)

        # Recover the arrangement from the six other cues supporting the same
        # option as Expert 17. Their mean distance from index 16 is exactly one
        # of 2, 3, 4, 5, 7, or 10 in this design.
        indices = np.arange(diff.size)
        supporters = indices[(np.sign(diff) == leading_direction) & (indices != 16)]
        if supporters.size == 0:
            continue
        spread = float(np.mean(np.abs(supporters - 16)))

        if spread > 6.0:
            congruent_sparse.append(congruent)
        else:
            congruent_dense.append(congruent)

    if len(congruent_sparse) == 0 or len(congruent_dense) == 0:
        return 0.0

    return float(np.mean(congruent_sparse) - np.mean(congruent_dense))
```

**Observed (real) value:** -0.0056 (var=0.0030)
**Previous candidate values (this loop):**
  - iter 1: 0.0044 (var=0.0105) (Δ vs real +0.0100)
  - iter 2: -0.0088 (var=0.0094) (Δ vs real -0.0031)
  - iter 3 (most recent): -0.0047 (var=0.0109) (Δ vs real +0.0009)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0003 (var=0.0059)
- pi_6: 0.3762 (var=0.0253)
- pi_2: 0.0038 (var=0.0049)
- pi_3: -0.0047 (var=0.0013)
- pi_4: 0.2303 (var=0.0907)
- pi_5: 0.0066 (var=0.0045)
- pi_7: 0.0225 (var=0.0115)
- pi_8: 0.0038 (var=0.0085)
- pi_9: -0.0047 (var=0.0080)
- pi_10: -0.0047 (var=0.0075)
- pi_11: 0.0262 (var=0.0133)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response follows the adjacent, same-direction cue pair.
    # The six possible zero-based starts are fixed by the experimental design.
    starts = (0, 6, 12, 18, 24, 30)
    follows_cluster = []
    is_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        cluster_start = None
        cluster_direction = 0.0
        for s in starts:
            if d[s] != 0.0 and d[s + 1] == d[s]:
                cluster_start = s
                cluster_direction = d[s]
                break

        if cluster_start is None:
            continue

        # Positive direction means the cluster supports A (response 0), while
        # negative direction means it supports B (response 1).
        cluster_response = 0 if cluster_direction > 0.0 else 1
        follows_cluster.append(float(int(row["response"]) == cluster_response))
        is_terminal.append(cluster_start == 30)

    follows_cluster = np.asarray(follows_cluster, dtype=float)
    is_terminal = np.asarray(is_terminal, dtype=bool)

    if follows_cluster.size == 0 or not np.any(is_terminal) or not np.any(~is_terminal):
        return float("nan")

    terminal_rate = float(np.mean(follows_cluster[is_terminal]))
    earlier_rate = float(np.mean(follows_cluster[~is_terminal]))
    return terminal_rate - earlier_rate

```

**Observed (real) value:** -0.0045 (var=0.0098)
**Previous candidate values (this loop):**
  - iter 1: 0.0127 (var=0.0103) (Δ vs real +0.0172)
  - iter 2: -0.0140 (var=0.0094) (Δ vs real -0.0095)
  - iter 3 (most recent): -0.0057 (var=0.0090) (Δ vs real -0.0012)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3420 (var=0.0223)
- pi_1: -0.0100 (var=0.0083)
- pi_2: 0.0002 (var=0.0078)
- pi_3: 0.0035 (var=0.0028)
- pi_4: -0.4612 (var=0.0902)
- pi_5: -0.4055 (var=0.0218)
- pi_7: -0.0140 (var=0.0142)
- pi_8: 0.0237 (var=0.0050)
- pi_9: -0.0120 (var=0.0102)
- pi_10: 0.0035 (var=0.0146)
- pi_11: 0.0128 (var=0.0124)

### Experiment 11
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Indices of the seven 82%-valid experts manipulated across conditions.
    source_idx = np.array([3, 10, 17, 24, 31, 38, 45], dtype=int)

    counts = []
    chose_redundant_side = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size <= int(source_idx.max()) or b.size <= int(source_idx.max()):
            continue

        d = a[source_idx] - b[source_idx]
        active = d != 0
        n_sources = int(np.sum(active))
        direction = float(np.sum(d[active]))
        if n_sources == 0 or direction == 0.0:
            continue

        response = int(row["response"])
        redundant_side_is_a = direction > 0.0
        chose_support = (response == 0) if redundant_side_is_a else (response == 1)

        counts.append(n_sources)
        chose_redundant_side.append(float(chose_support))

    counts = np.asarray(counts, dtype=int)
    chose_redundant_side = np.asarray(chose_redundant_side, dtype=float)

    low = counts <= 3
    high = counts >= 6
    if not np.any(low) or not np.any(high):
        raise ValueError("Metric requires both low (2-3) and high (6-7) redundancy trials.")

    return float(np.mean(chose_redundant_side[high]) -
                 np.mean(chose_redundant_side[low]))

```

**Observed (real) value:** -0.0225 (var=0.0063)
**Previous candidate values (this loop):**
  - iter 1: -0.0106 (var=0.0081) (Δ vs real +0.0119)
  - iter 2: -0.0100 (var=0.0079) (Δ vs real +0.0125)
  - iter 3 (most recent): -0.0138 (var=0.0098) (Δ vs real +0.0087)
**Other theories' values on this metric (for reference):**
- pi_7: -0.4881 (var=0.0165)
- pi_6: 0.1419 (var=0.0138)
- pi_1: -0.0069 (var=0.0079)
- pi_2: 0.0138 (var=0.0111)
- pi_3: -0.0031 (var=0.0033)
- pi_4: 0.0075 (var=0.0088)
- pi_5: 0.0738 (var=0.0093)
- pi_8: -0.0156 (var=0.0119)
- pi_9: -0.0088 (var=0.0093)
- pi_10: -0.0225 (var=0.0113)
- pi_11: -0.0050 (var=0.0103)

### Experiment 12
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response followed the 82%-valid singleton rather than
    # the fixed 70%-valid cue at feature 1.
    positions = []
    followed_target = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        diff = a - b

        # Every designed trial has cue 1 and exactly one later cue active.
        active_late = np.flatnonzero((diff != 0.0) & (np.arange(diff.size) != 0))
        if active_late.size != 1:
            continue

        j = int(active_late[0])
        response = int(row['response'])
        target_choice = (response == 0) if diff[j] > 0.0 else (response == 1)
        positions.append(j + 1)  # Convert to the design's one-based positions.
        followed_target.append(float(target_choice))

    if len(positions) == 0:
        return float('nan')

    positions = np.asarray(positions, dtype=int)
    followed_target = np.asarray(followed_target, dtype=np.float64)

    terminal = followed_target[positions == 24]
    early = followed_target[np.isin(positions, np.array([14, 16, 18]))]
    if terminal.size == 0 or early.size == 0:
        return float('nan')

    return float(np.mean(terminal) - np.mean(early))
```

**Observed (real) value:** 0.0150 (var=0.0426)
**Previous candidate values (this loop):**
  - iter 1: -0.0125 (var=0.0128) (Δ vs real -0.0275)
  - iter 2: -0.0304 (var=0.0183) (Δ vs real -0.0454)
  - iter 3 (most recent): 0.0354 (var=0.0105) (Δ vs real +0.0204)
**Other theories' values on this metric (for reference):**
- pi_6: 0.1892 (var=0.0198)
- pi_7: -0.0079 (var=0.0203)
- pi_1: 0.0088 (var=0.0055)
- pi_2: -0.0187 (var=0.0139)
- pi_3: -0.0025 (var=0.0024)
- pi_4: -0.0171 (var=0.0249)
- pi_5: -0.2475 (var=0.0268)
- pi_8: -0.0463 (var=0.0163)
- pi_9: 0.0171 (var=0.0149)
- pi_10: 0.0217 (var=0.0137)
- pi_11: -0.0133 (var=0.0171)

### Experiment 13
**Design**
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.97, 0.70, 0.92, 0.90, 0.88, 0.90, 0.92, 0.88,
        0.92, 0.90, 0.88, 0.86, 0.90, 0.92, 0.88, 0.86,
        0.92, 0.90, 0.88, 0.86, 0.84, 0.90, 0.88, 0.92,
        0.86, 0.84
    ], dtype=float)

    cells = {(length, embedded): []
             for length in (3, 4, 5)
             for embedded in (False, True)}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        i = 0
        run = None
        while i < difference.size:
            if difference[i] == 0:
                i += 1
                continue
            direction = difference[i]
            j = i + 1
            while j < difference.size and difference[j] == direction:
                j += 1
            if j - i >= 3:
                run = (i, j, direction)
                break
            i = j

        if run is None:
            continue

        start, stop, direction = run
        length = stop - start
        if length not in (3, 4, 5):
            continue

        run_validities = validities[start:stop]
        strongest = np.flatnonzero(np.isclose(
            run_validities, np.max(run_validities), atol=1e-12, rtol=0.0
        ))
        embedded = bool(
            strongest.size == 1
            and 0 < int(strongest[0]) < length - 1
        )

        run_supported_response = 0 if direction > 0 else 1
        chose_run = float(int(row['response']) == run_supported_response)
        cells[(length, embedded)].append(chose_run)

    # Longer runs receive more weight because the competing account's
    # overload manipulation grows with excess run length.
    weights = {3: 1.0, 4: 2.5, 5: 4.0}
    weighted_contrasts = []
    used_weights = []
    for length in (3, 4, 5):
        boundary = cells[(length, False)]
        embedded = cells[(length, True)]
        if len(boundary) == 0 or len(embedded) == 0:
            continue
        contrast = float(np.mean(embedded) - np.mean(boundary))
        weighted_contrasts.append(weights[length] * contrast)
        used_weights.append(weights[length])

    if len(used_weights) == 0:
        return float('nan')
    return float(np.sum(weighted_contrasts) / np.sum(used_weights))
```

**Observed (real) value:** -0.0030 (var=0.0059)
**Previous candidate values (this loop):**
  - iter 1: 0.0173 (var=0.0098) (Δ vs real +0.0203)
  - iter 2: -0.0050 (var=0.0047) (Δ vs real -0.0020)
  - iter 3 (most recent): 0.0226 (var=0.0048) (Δ vs real +0.0256)
**Other theories' values on this metric (for reference):**
- pi_7: -0.0313 (var=0.0119)
- pi_8: 0.1210 (var=0.0087)
- pi_1: 0.0009 (var=0.0050)
- pi_2: 0.0097 (var=0.0054)
- pi_3: 0.0143 (var=0.0019)
- pi_4: -0.1334 (var=0.0298)
- pi_5: -0.0434 (var=0.0066)
- pi_6: 0.0202 (var=0.0136)
- pi_9: 0.0281 (var=0.0061)
- pi_10: 0.0273 (var=0.0083)
- pi_11: -0.0022 (var=0.0095)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    adjacent_choices = []
    separated_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        a_support = np.flatnonzero(diff > 0)
        b_support = np.flatnonzero(diff < 0)

        # Identify the option supported by the two-expert 82% pair.
        if a_support.size == 2 and b_support.size == 1:
            pair_positions = a_support
            pair_response = 0
        elif b_support.size == 2 and a_support.size == 1:
            pair_positions = b_support
            pair_response = 1
        else:
            continue

        chose_pair = float(int(row["response"]) == pair_response)
        is_adjacent = int(np.abs(pair_positions[1] - pair_positions[0])) == 1

        if is_adjacent:
            adjacent_choices.append(chose_pair)
        else:
            separated_choices.append(chose_pair)

    if len(adjacent_choices) == 0 or len(separated_choices) == 0:
        return 0.0

    return float(np.mean(adjacent_choices) - np.mean(separated_choices))
```

**Observed (real) value:** -0.0183 (var=0.0044)
**Previous candidate values (this loop):**
  - iter 1: -0.0079 (var=0.0061) (Δ vs real +0.0104)
  - iter 2: 0.0021 (var=0.0068) (Δ vs real +0.0204)
  - iter 3 (most recent): -0.0121 (var=0.0077) (Δ vs real +0.0063)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1392 (var=0.0118)
- pi_7: -0.1175 (var=0.0113)
- pi_1: -0.0021 (var=0.0051)
- pi_2: -0.0067 (var=0.0050)
- pi_3: -0.0150 (var=0.0011)
- pi_4: -0.2296 (var=0.0095)
- pi_5: -0.0083 (var=0.0034)
- pi_6: -0.2642 (var=0.0145)
- pi_9: 0.0267 (var=0.0071)
- pi_10: 0.0025 (var=0.0048)
- pi_11: -0.0142 (var=0.0053)

### Experiment 15
**Design**
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_cliff(df):
        counts = []
        run_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b

            # Features 2-7 (zero-based positions 1:7) are the fixed run.
            run_direction = np.sign(np.sum(diff[1:7]))
            if run_direction == 0:
                continue

            # Count active sources recommending against the run-supported option.
            opponent_count = int(np.sum(diff == -run_direction))
            response = int(row["response"])
            chose_run_option = float(
                (run_direction > 0 and response == 0)
                or (run_direction < 0 and response == 1)
            )

            counts.append(opponent_count)
            run_choices.append(chose_run_option)

        counts = np.asarray(counts, dtype=int)
        run_choices = np.asarray(run_choices, dtype=float)
        first = run_choices[counts == 1]
        competitive = run_choices[counts >= 2]

        if first.size == 0 or competitive.size == 0:
            return np.nan
        return float(np.mean(first) - np.mean(competitive))

    if "subject_id" in data.columns:
        values = []
        for _, df in data.groupby("subject_id", sort=False):
            value = subject_cliff(df)
            if np.isfinite(value):
                values.append(value)
        if values:
            return float(np.mean(values))

    return subject_cliff(data)
```

**Observed (real) value:** -0.0205 (var=0.0094)
**Previous candidate values (this loop):**
  - iter 1: -0.0100 (var=0.0131) (Δ vs real +0.0105)
  - iter 2: -0.0135 (var=0.0122) (Δ vs real +0.0070)
  - iter 3 (most recent): -0.0172 (var=0.0096) (Δ vs real +0.0032)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5137 (var=0.0192)
- pi_8: -0.0035 (var=0.0146)
- pi_1: 0.0105 (var=0.0095)
- pi_2: 0.0730 (var=0.0107)
- pi_3: -0.0035 (var=0.0022)
- pi_4: 0.0067 (var=0.0063)
- pi_5: 0.0035 (var=0.0044)
- pi_6: 0.0530 (var=0.0194)
- pi_7: -0.0058 (var=0.0258)
- pi_10: 0.0340 (var=0.0153)
- pi_11: -0.0032 (var=0.0097)

### Experiment 16
**Design**
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    pair_choices = []
    run_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        diff = a - b

        pos = np.flatnonzero(diff > 0)
        neg = np.flatnonzero(diff < 0)

        # Retain the four-supporters-versus-singleton trials only.
        if len(pos) == 4 and len(neg) == 1:
            target_idx = pos
            chose_target = float(int(row["response"]) == 0)
        elif len(neg) == 4 and len(pos) == 1:
            target_idx = neg
            chose_target = float(int(row["response"]) == 1)
        else:
            continue

        target_idx = np.sort(target_idx)
        longest_run = 1
        current_run = 1
        for j in range(1, len(target_idx)):
            if target_idx[j] == target_idx[j - 1] + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        if longest_run == 4:
            run_choices.append(chose_target)
        elif longest_run == 2:
            # In this design, this identifies the two-separated-pairs condition.
            pair_choices.append(chose_target)

    if len(pair_choices) == 0 or len(run_choices) == 0:
        return float("nan")

    return float(np.mean(pair_choices) - np.mean(run_choices))
```

**Observed (real) value:** 0.0550 (var=0.0207)
**Previous candidate values (this loop):**
  - iter 1: 0.0113 (var=0.0150) (Δ vs real -0.0437)
  - iter 2: 0.0413 (var=0.0168) (Δ vs real -0.0137)
  - iter 3 (most recent): 0.0175 (var=0.0189) (Δ vs real -0.0375)
**Other theories' values on this metric (for reference):**
- pi_8: 0.3287 (var=0.0175)
- pi_9: 0.0088 (var=0.0220)
- pi_1: -0.0187 (var=0.0233)
- pi_2: -0.0088 (var=0.0164)
- pi_3: 0.0112 (var=0.0062)
- pi_4: 0.0200 (var=0.0137)
- pi_5: -0.0125 (var=0.0094)
- pi_6: 0.1738 (var=0.0304)
- pi_7: 0.0663 (var=0.0387)
- pi_10: 0.0337 (var=0.0218)
- pi_11: 0.0162 (var=0.0214)

### Experiment 17
**Design**
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Measure the opponent-count effect specifically for six-source runs.
    # Positive values mean greater choice of the run-supported option when it
    # faces one opposing supporter rather than two.
    one_opponent = []
    two_opponents = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = np.sign(a - b)

        n_pos = int(np.sum(d > 0))
        n_neg = int(np.sum(d < 0))

        if n_pos == 6 and n_neg in (1, 2):
            run_supports_a = True
            opponent_count = n_neg
        elif n_neg == 6 and n_pos in (1, 2):
            run_supports_a = False
            opponent_count = n_pos
        else:
            continue

        response = int(row["response"])
        chose_run = float((response == 0) if run_supports_a else (response == 1))

        if opponent_count == 1:
            one_opponent.append(chose_run)
        else:
            two_opponents.append(chose_run)

    if len(one_opponent) == 0 or len(two_opponents) == 0:
        return float("nan")

    return float(np.mean(one_opponent) - np.mean(two_opponents))
```

**Observed (real) value:** -0.0200 (var=0.0140)
**Previous candidate values (this loop):**
  - iter 1: -0.0063 (var=0.0194) (Δ vs real +0.0137)
  - iter 2: -0.0175 (var=0.0173) (Δ vs real +0.0025)
  - iter 3 (most recent): 0.0000 (var=0.0173) (Δ vs real +0.0200)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3450 (var=0.0322)
- pi_10: 0.0287 (var=0.0211)
- pi_1: 0.0200 (var=0.0093)
- pi_2: -0.0075 (var=0.0129)
- pi_3: -0.0012 (var=0.0046)
- pi_4: -0.0088 (var=0.0152)
- pi_5: 0.0400 (var=0.0075)
- pi_6: 0.1050 (var=0.0219)
- pi_7: 0.0550 (var=0.0204)
- pi_8: -0.0125 (var=0.0361)
- pi_11: 0.0063 (var=0.0164)

### Experiment 18
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Experts 2, 5, 8, 11, 14, and 17 (zero-based indices below) are
    # the six 94%-valid sources used in the critical count-six condition.
    high94 = np.asarray([1, 4, 7, 10, 13, 16], dtype=int)
    following = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        active94 = high94[d[high94] != 0]
        if active94.size != 6:
            continue

        directions = np.sign(d[active94])
        if not np.all(directions == directions[0]):
            continue

        # Require the intended opposing singleton at Expert 22.
        if d.size <= 21 or d[21] == 0 or np.sign(d[21]) == directions[0]:
            continue

        focal_response = 0 if directions[0] > 0 else 1
        following.append(float(int(row["response"]) == focal_response))

    if len(following) == 0:
        return float("nan")
    return float(np.mean(following))
```

**Observed (real) value:** 0.1375 (var=0.0106)
**Previous candidate values (this loop):**
  - iter 1: 0.2313 (var=0.0107) (Δ vs real +0.0938)
  - iter 2: 0.1613 (var=0.0127) (Δ vs real +0.0237)
  - iter 3 (most recent): 0.1487 (var=0.0098) (Δ vs real +0.0112)
**Other theories' values on this metric (for reference):**
- pi_10: 0.4850 (var=0.0238)
- pi_9: 0.8525 (var=0.0070)
- pi_1: 0.8850 (var=0.0102)
- pi_2: 0.8838 (var=0.0111)
- pi_3: 0.9487 (var=0.0051)
- pi_4: 0.8363 (var=0.0087)
- pi_5: 0.8900 (var=0.0056)
- pi_6: 0.8650 (var=0.0083)
- pi_7: 0.3725 (var=0.0355)
- pi_8: 0.8425 (var=0.0066)
- pi_11: 0.1663 (var=0.0110)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.89, 0.94, 0.94, 0.938, 0.936, 0.934, 0.932,
        0.90, 0.86, 0.82, 0.78, 0.70, 0.99, 0.52
    ], dtype=float)

    low_homogeneity_choices = []
    high_heterogeneity_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        difference = a - b
        a_supporters = np.flatnonzero(difference > 0)
        b_supporters = np.flatnonzero(difference < 0)

        if len(a_supporters) > len(b_supporters):
            bundle_supporters = a_supporters
            bundle_response = 0
        elif len(b_supporters) > len(a_supporters):
            bundle_supporters = b_supporters
            bundle_response = 1
        else:
            continue

        if len(bundle_supporters) != 6:
            continue

        spread = float(np.std(validities[bundle_supporters]))
        bundle_chosen = float(int(row["response"]) == bundle_response)

        # Conditions 1-2: validity SD approximately .003 and .014.
        if spread <= 0.020:
            low_homogeneity_choices.append(bundle_chosen)
        # Conditions 5-6: validity SD approximately .066 and .095.
        elif spread >= 0.055:
            high_heterogeneity_choices.append(bundle_chosen)

    if not low_homogeneity_choices or not high_heterogeneity_choices:
        return float("nan")

    return float(
        np.mean(high_heterogeneity_choices)
        - np.mean(low_homogeneity_choices)
    )
```

**Observed (real) value:** 0.0212 (var=0.0051)
**Previous candidate values (this loop):**
  - iter 1: 0.0006 (var=0.0132) (Δ vs real -0.0206)
  - iter 2: -0.0025 (var=0.0080) (Δ vs real -0.0237)
  - iter 3 (most recent): -0.0056 (var=0.0092) (Δ vs real -0.0269)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6781 (var=0.0108)
- pi_10: -0.0038 (var=0.0181)
- pi_1: -0.0025 (var=0.0078)
- pi_2: -0.0106 (var=0.0078)
- pi_3: -0.0062 (var=0.0024)
- pi_4: -0.0281 (var=0.0078)
- pi_5: 0.0019 (var=0.0075)
- pi_6: 0.0806 (var=0.0194)
- pi_7: 0.1681 (var=0.0117)
- pi_8: 0.0644 (var=0.0141)
- pi_9: -0.0112 (var=0.0087)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The 82%-valid target recommendation is carried by positions 3/4,
    # 16/17, or 31/32 (zero-based indices below). The opposing
    # recommendation is carried by the 80%-valid expert at position 1.
    target_indices = np.array([2, 3, 15, 16, 30, 31], dtype=int)
    target_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size <= int(target_indices.max()) or b.size <= int(target_indices.max()):
            continue

        target_direction = float(np.sum(a[target_indices] - b[target_indices]))
        response = int(row["response"])

        if target_direction > 0.0:
            target_choices.append(float(response == 0))
        elif target_direction < 0.0:
            target_choices.append(float(response == 1))

    if len(target_choices) == 0:
        return float("nan")
    return float(np.mean(target_choices))
```

**Observed (real) value:** 0.2125 (var=0.0153)
**Previous candidate values (this loop):**
  - iter 1: 0.2202 (var=0.0025) (Δ vs real +0.0077)
  - iter 2: 0.2125 (var=0.0081) (Δ vs real +0.0000)
  - iter 3 (most recent): 0.2046 (var=0.0069) (Δ vs real -0.0079)
**Other theories' values on this metric (for reference):**
- pi_10: 0.8329 (var=0.0018)
- pi_11: 0.7515 (var=0.0034)
- pi_1: 0.8860 (var=0.0053)
- pi_2: 0.6704 (var=0.0048)
- pi_3: 0.9629 (var=0.0008)
- pi_4: 0.2067 (var=0.0069)
- pi_5: 0.5942 (var=0.0022)
- pi_6: 0.6258 (var=0.0026)
- pi_7: 0.6837 (var=0.0053)
- pi_8: 0.7983 (var=0.0025)
- pi_9: 0.7815 (var=0.0030)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Load-Adaptive Anchor–Validity Arbitration. Choice is produced by competition between a validity-directed search route and an order-anchor route. Validity-directed search compresses all currently agreeing experts into option-level recommendations, ranks those recommendations by their strongest communicated validity, and usually stops at the first accessible recommendation. Its accessibility includes localized source–validity binding uncertainty and an intrinsic count-concentration cost. The count cost is independent of spatial arrangement, opponent count, and within-bundle validity dispersion; it remains weak for ordinary bundles but rises sharply when at least five or six extremely high-validity sources must be maintained as one recommendation. The order-anchor route is gated by display load. In long displays, the first discriminating recommendation establishes a provisional choice. A later opposing recommendation overturns that anchor only when its validity advantage and compressed diagnostic support jointly exceed a load-dependent threshold. The threshold rises with the display position at which the strongest opposing source is encountered, producing primacy rather than recency. Conspicuous validity differences switch processing back toward validity-directed search, whereas small differences such as 80% versus 82% leave the early anchor influential. Stable subject-level variation in order-anchor reliance produces heterogeneous long-display behavior without trial feedback or experiment-specific rules.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
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

`rationale`: This is a parameter-only minimal-diff edit. Reducing conspicuous_route_switch preserves a substantial position-sensitive anchor route even when the later cue has a noticeable validity advantage, while the moderately larger late_override_threshold makes later overrides harder; together these changes should turn Experiment 7's nearly flat gradient negative without changing the mechanism. The overload threshold is moved close to six and its width narrowed sharply, so five-source bundles receive little penalty while six-source bundles cross a steep capacity cliff. Increasing overload strength then lowers the six-94% absolute choice rate in Experiment 18 while avoiding the excessive average five/six penalty in Experiment 8. A modestly larger compressed_support_gain improves ordinary-bundle integration in Experiments 4 and 5, whereas the high-validity overload cost still dominates extreme bundles. Finally, order_anchor_reliance is widened around approximately the same central tendency, increasing stable between-subject heterogeneity in Experiment 20 without using response noise or trial-specific adaptation.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed load-adaptive dual-route family and achieves several important cross-experiment successes. It exactly matches the categorical signatures in Experiments 1–3, closely captures the inverse validity-gap effect in Experiment 6, preserves negligible arrangement, adjacency, run-embedding, and opponent-count effects in Experiments 9–17 and 19, and provides an excellent absolute fit to the critical 32-feature 80%-versus-82% conflict in Experiment 20 (0.220 versus 0.213). However, three central quantitative signatures remain miscalibrated. First, the 32-feature positional manipulation in Experiment 7 is essentially flat and slightly positive (0.005) rather than reliably negative (-0.054), so the late-override penalty is not actually expressed where the theory requires primacy. Second, high-validity overload is too strong in the 24-feature redundancy contrast of Experiment 8 (-0.421 versus -0.268), yet not strong enough in the count-six absolute-choice test of Experiment 18 (0.231 versus 0.138). This indicates that the overload function turns on too early across counts five and six rather than producing a sharper count-six cliff. Third, ordinary-bundle integration is somewhat too weak in Experiments 4 and 5: choices following the opposing weak-cue aggregate are underpredicted (0.091 versus 0.162 and 0.124 versus 0.161). Between-subject variability is also generally compressed, notably in Experiment 20 (variance 0.0025 versus 0.0153), despite the intended stable variation in route reliance.
Rationale: Keep the dual-route mechanism, but retune its gates. For the positional failure, preserve more residual anchor-route weight when a validity gap is conspicuous—e.g. reduce `conspicuous_route_switch`—and/or increase `late_override_threshold`, because the present conspicuousness switch appears to suppress the position-sensitive route before its late-position penalty can generate the observed negative gradient. Check that this leaves the already accurate small-gap absolute rate in Experiment 20 intact. For overload, move `overload_count_threshold` closer to six and make `overload_count_width` narrower, while retaining enough `high_validity_overload_strength` to lower the six-94% choice rate in Experiment 18. A sharper five-to-six transition should reduce the excessive average penalty over the five/six conditions in Experiment 8 while strengthening the count-six endpoint. Modestly increase `compressed_support_gain` for ordinary-validity bundles to raise the integrated-choice rates in Experiments 4 and 5; the high-validity overload gate should keep this from rescuing the extreme bundles. Finally, widen effective subject-level variation in anchor reliance or override threshold, rather than merely increasing response noise, so Experiment 20 retains its mean while acquiring more realistic heterogeneity. These are parameterization and gating adjustments within the prescribed family, not a change of mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.0541 -> iter 2 candidate loss=0.0410 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed load-adaptive dual-route family and the accepted revision substantially improves aggregate fit (0.0541 to 0.0410). It now matches Experiment 20's critical 32-feature absolute choice rate exactly, brings Experiment 18's count-six high-validity choice rate close to the data, captures Experiment 6, and preserves the desired near-zero arrangement, adjacency, embedding, and opponent-count contrasts across Experiments 9–17 and 19. The sharper overload transition also moved Experiment 8 in the correct direction. However, the central positional prediction still fails: Experiment 7 is positive (0.0258) rather than negative (-0.0543), and it worsened from the prior iteration despite the previous recommendation to increase the late-override penalty. This reveals a sign-level calibration problem in the implemented position gate. In the code, a later opposing representative raises `required_drive`, necessarily making target-cue choice decline with later position; under the actual metric this generates the positive rather than observed negative contrast. Remaining quantitative misses include excessive high-validity overload in Experiment 8 (-0.376 versus -0.268), insufficient ordinary-bundle integration in Experiments 4 and 5 (0.096 versus 0.162 and 0.118 versus 0.161), and a modest spurious negative position effect at 24 features in Experiment 12 (-0.030 versus 0.015). Between-subject variability also remains compressed in several key experiments, although Experiment 20 improved substantially.
Rationale: Keep the accepted dual-route model, but correct the positional gate rather than further increasing `late_override_threshold`. The previous push in that direction was accepted overall but moved Experiment 7 farther from the target, so it should not be repeated. Reverse or reparameterize the position contribution in `required_drive`—for example, allow a small negative `late_override_threshold`, or define the position term using remaining rather than elapsed display distance—so that the simulated contrast has the empirically required negative sign. Calibrate this narrowly against Experiment 7 while retaining the exact Experiment 20 mean and the near-zero position/arrangement effects in Experiments 9, 10, 12, and 14. For overload, extend the successful sharper-cliff adjustment: move the count threshold slightly closer to six and narrow its width further, then tune strength at count six. This should retain Experiment 18's low absolute rate while reducing the excessive average penalty across Experiment 8's pooled count-five and count-six conditions. Use only a small increase in ordinary-validity bundle accessibility or support, since Experiment 4 improved but Experiment 5 did not; avoid a broad overload reduction that would undo Experiment 18. Finally, modest subject-level variation in override threshold or position sensitivity could raise heterogeneity without disturbing Experiment 20's now-correct pooled mean.

**Outcome of this advice:** iter 2 candidate loss=0.0410 -> iter 3 candidate loss=0.0418 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, but it was rejected by the accept gate (loss 0.0418 versus the accepted 0.0410), so it should not replace the iteration-2 base. The position-sign correction moved Experiment 7 in the intended direction, from +0.0258 to approximately zero, but still falls well short of the observed negative contrast (-0.0543). Moreover, the combined revision produced collateral regressions: Experiment 5 moved farther below the observed aggregate-override rate (0.1062 versus 0.1613), Experiment 13 acquired a spurious arrangement contrast (0.0226 versus -0.0030), Experiment 16 lost much of its previously good pair-versus-run contrast (0.0175 versus 0.0550), Experiment 17 moved away from its near-zero opponent-count effect, and Experiment 20 lost its previously exact absolute choice rate. The sharper overload change did improve Experiments 8 and 18 slightly, but not enough to offset these losses: Experiment 8 remains too negative (-0.3706 versus -0.2675), while Experiment 18 is already close (0.1487 versus 0.1375). Thus the main unresolved issue is not the mechanism family but the implementation of positional sensitivity: the current uncentered remaining-distance term changes both the positional slope and the average override threshold, thereby disturbing absolute rates and unrelated contrasts.
Rationale: Build from the unchanged accepted iteration-2 model, not from this rejected revision. Do not repeat the rejected joint push of further narrowing the overload cliff and increasing compressed support. Instead, make the position correction mean-preserving: replace the elapsed-position term with a centered remaining-distance term, such as position_slope * (mean_relative_position - relative_position), where the centering constant is fixed by the display or relevant target positions. Tune the slope to make Experiment 7 negative while leaving the average override threshold unchanged. This is especially suitable for Experiment 20, whose early, middle, and late target positions are roughly balanced, so a centered term can preserve its exact pooled rate while introducing the required gradient. If the gradient remains attenuated because conspicuous gaps suppress the anchor route, allow a modest residual anchor weight specifically through the existing conspicuousness gate rather than globally increasing anchor reliance. Retain iteration 2's overload and ordinary-support settings initially: the latest sharper-cliff/support changes were rejected and worsened Experiments 5, 13, 16, 17, and 20. After isolating the centered positional fix, only consider a very small adjustment to moderate-validity retrieval uncertainty—not another broad increase in compressed support—to raise Experiments 4 and 5 without weakening the successful high-validity overload fit in Experiment 18 or introducing arrangement dependence.

**Outcome of this advice:** iter 3 candidate loss=0.0418 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
{
  "theory": "Load-Adaptive Anchor–Validity Arbitration. Choice is produced by competition between a validity-directed search route and an order-anchor route. Validity-directed search compresses all currently agreeing experts into option-level recommendations, ranks those recommendations by their strongest communicated validity, and usually stops at the first accessible recommendation. Its accessibility includes localized source–validity binding uncertainty and an intrinsic count-concentration cost. The count cost is independent of spatial arrangement, opponent count, and within-bundle validity dispersion; it remains weak for ordinary bundles but rises sharply when at least five or six extremely high-validity sources must be maintained as one recommendation. The order-anchor route is gated by display load. In long displays, the first discriminating recommendation establishes a provisional choice. A later opposing recommendation overturns that anchor only when its validity advantage and compressed diagnostic support jointly exceed a load-dependent threshold. Positional sensitivity is represented by a mean-preserving, centered remaining-distance signal: it changes the positional gradient without changing the average override threshold across a balanced long display. Conspicuous validity differences switch processing back toward validity-directed search, whereas small differences such as 80% versus 82% leave the early anchor influential. Stable subject-level variation in order-anchor reliance produces heterogeneous long-display behavior without trial feedback or experiment-specific rules.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            \"Load-Adaptive Anchor-Validity Arbitration expects state shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"State has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities must have length {n_features}; got {validities.shape}.\"\n        )\n\n    retrieval_baseline = float(parameters[\"retrieval_baseline\"])\n    validity_compression = float(parameters[\"validity_compression\"])\n    uncertainty_strength = float(parameters[\"binding_uncertainty_strength\"])\n    uncertainty_center = float(parameters[\"binding_uncertainty_center\"])\n    uncertainty_width = max(float(parameters[\"binding_uncertainty_width\"]), 1e-9)\n    reliability_gain = float(parameters[\"compressed_support_gain\"])\n    overload_strength = float(parameters[\"high_validity_overload_strength\"])\n    overload_threshold = float(parameters[\"overload_count_threshold\"])\n    overload_width = max(float(parameters[\"overload_count_width\"]), 1e-9)\n    high_validity_center = float(parameters[\"overload_validity_center\"])\n    high_validity_width = max(float(parameters[\"overload_validity_width\"]), 1e-9)\n    anchor_reliance = float(parameters[\"order_anchor_reliance\"])\n    conspicuous_center = float(parameters[\"conspicuous_gap_center\"])\n    conspicuous_width = max(float(parameters[\"conspicuous_gap_width\"]), 1e-9)\n    conspicuous_switch = float(parameters[\"conspicuous_route_switch\"])\n    override_gap_scale = max(float(parameters[\"override_gap_scale\"]), 1e-9)\n    override_support_gain = float(parameters[\"override_support_gain\"])\n    override_threshold = float(parameters[\"override_threshold\"])\n    load_threshold = float(parameters[\"load_override_threshold\"])\n    position_threshold = float(parameters[\"late_override_threshold\"])\n    override_temperature = max(float(parameters[\"override_temperature\"]), 1e-9)\n    lapse = float(parameters[\"response_lapse\"])\n\n    difference = stim[0] - stim[1]\n    directions = np.sign(difference)\n    active = np.flatnonzero(directions != 0.0).astype(int)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def sigmoid_scalar(x):\n        x = float(x)\n        if x >= 0.0:\n            return 1.0 / (1.0 + np.exp(-x))\n        ex = np.exp(x)\n        return ex / (1.0 + ex)\n\n    # Subjective diagnosticity affects binding and accumulated support, while\n    # communicated validity itself determines recommendation priority.\n    objective_diagnosticity = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)\n    subjective_diagnosticity = np.power(\n        objective_diagnosticity, validity_compression\n    )\n    uncertainty_profile = np.exp(\n        -0.5 * np.square(\n            (subjective_diagnosticity - uncertainty_center) / uncertainty_width\n        )\n    )\n\n    units = {}\n    for direction in (-1, 1):\n        members = np.flatnonzero(directions == float(direction)).astype(int)\n        if members.size == 0:\n            continue\n\n        member_validities = validities[members]\n        best_local = int(np.argmax(member_validities))\n        representative = int(members[best_local])\n        priority = float(validities[representative])\n        count = int(members.size)\n\n        # Same-direction sources form one recommendation. Their positive\n        # reliability contribution is deliberately small and saturating.\n        support_bonus = reliability_gain * (\n            1.0 - np.exp(-float(max(count - 1, 0)))\n        )\n\n        # Intrinsic concentration overload depends only on the recommendation's\n        # count and strongest diagnosticity. It contains no dispersion,\n        # adjacency, run, position, or opponent-count term. The count gate is\n        # normalized to zero for a singleton and becomes steep near 5-6 cues.\n        raw_count_gate = sigmoid_scalar(\n            (float(count) - overload_threshold) / overload_width\n        )\n        singleton_gate = sigmoid_scalar(\n            (1.0 - overload_threshold) / overload_width\n        )\n        count_gate = (raw_count_gate - singleton_gate) / max(\n            1.0 - singleton_gate, 1e-9\n        )\n        count_gate = float(np.clip(count_gate, 0.0, 1.0))\n        validity_gate = sigmoid_scalar(\n            (priority - high_validity_center) / high_validity_width\n        )\n        overload_cost = overload_strength * count_gate * validity_gate\n\n        retrieval_logit = (\n            retrieval_baseline\n            - uncertainty_strength * uncertainty_profile[representative]\n            + support_bonus\n            - overload_cost\n        )\n        retrieval = sigmoid_scalar(retrieval_logit)\n\n        units[direction] = {\n            \"direction\": int(direction),\n            \"members\": members,\n            \"representative\": representative,\n            \"priority\": priority,\n            \"count\": count,\n            \"retrieval\": float(np.clip(retrieval, 1e-9, 1.0 - 1e-9)),\n            \"overload_cost\": float(overload_cost),\n        }\n\n    # Validity-directed route: option-level recommendations enter a\n    # lexicographic accessibility cascade.\n    validity_order = sorted(\n        units.values(),\n        key=lambda u: (-u[\"priority\"], u[\"representative\"]),\n    )\n\n    reach = 1.0\n    validity_a = 0.0\n    validity_b = 0.0\n    for unit in validity_order:\n        stop = reach * unit[\"retrieval\"]\n        if unit[\"direction\"] > 0:\n            validity_a += stop\n        else:\n            validity_b += stop\n        reach *= 1.0 - unit[\"retrieval\"]\n    validity_a += 0.5 * reach\n    validity_b += 0.5 * reach\n    validity_total = validity_a + validity_b\n    if not np.isfinite(validity_total) or validity_total <= 0.0:\n        p_a_validity = 0.5\n    else:\n        p_a_validity = float(validity_a / validity_total)\n\n    # The display-load gate is exactly zero through 24 features and reaches\n    # one at 32. Thus ordinary displays are governed by validity search, while\n    # long displays permit stable individual differences in anchoring.\n    display_load = float(np.clip(\n        (float(n_features) - 24.0) / 8.0, 0.0, 1.0\n    )) ** 2\n\n    first_index = int(active[0])\n    anchor_direction = int(np.sign(directions[first_index]))\n    anchor_validity = float(validities[first_index])\n    opposing_unit = units.get(-anchor_direction)\n\n    if opposing_unit is None:\n        p_override = 0.0\n        priority_gap = 0.0\n    else:\n        priority_gap = float(\n            opposing_unit[\"priority\"] - anchor_validity\n        )\n\n        # Additional agreeing sources are compressed: they provide diminishing\n        # diagnostic support rather than independent votes. The contribution\n        # depends on count and diagnosticity but not spatial organization.\n        opposing_members = opposing_unit[\"members\"]\n        mean_diagnosticity = float(np.mean(\n            subjective_diagnosticity[opposing_members]\n        ))\n        compressed_support = (\n            np.log1p(float(max(opposing_unit[\"count\"] - 1, 0)))\n            * mean_diagnosticity\n        )\n\n        # Position is defined by the strongest opposing source. The centered\n        # component reverses the empirical gradient while the added 0.5 offset\n        # preserves the average threshold of the accepted elapsed-position\n        # term for balanced long displays.\n        if n_features > 1:\n            relative_position = (\n                opposing_unit[\"representative\"] / float(n_features - 1)\n            )\n        else:\n            relative_position = 0.0\n        mean_relative_position = 0.5\n        centered_remaining_distance = (\n            mean_relative_position\n            + (mean_relative_position - relative_position)\n        )\n\n        override_drive = (\n            priority_gap / override_gap_scale\n            + override_support_gain * compressed_support\n        )\n        required_drive = (\n            override_threshold\n            + display_load * load_threshold\n            + display_load * position_threshold * centered_remaining_distance\n        )\n        p_override = sigmoid_scalar(\n            (override_drive - required_drive) / override_temperature\n        )\n\n    # The anchor route chooses the first recommendation unless the opposing\n    # recommendation clears the override threshold.\n    if anchor_direction > 0:\n        p_a_anchor = 1.0 - p_override\n    else:\n        p_a_anchor = p_override\n\n    # Conspicuous priority gaps recruit validity-directed search even in a long\n    # display. Small gaps such as .80 versus .82 leave the anchor route active.\n    if len(units) == 2:\n        priorities = [u[\"priority\"] for u in units.values()]\n        absolute_gap = abs(float(priorities[0] - priorities[1]))\n    else:\n        absolute_gap = 0.0\n    conspicuousness = sigmoid_scalar(\n        (absolute_gap - conspicuous_center) / conspicuous_width\n    )\n    anchor_weight = (\n        anchor_reliance\n        * display_load\n        * (1.0 - conspicuous_switch * conspicuousness)\n    )\n    anchor_weight = float(np.clip(anchor_weight, 0.0, 1.0))\n\n    p_a_intended = (\n        anchor_weight * p_a_anchor\n        + (1.0 - anchor_weight) * p_a_validity\n    )\n\n    p_a = (1.0 - lapse) * p_a_intended + lapse * 0.5\n    p_a = float(np.clip(p_a, 0.0, 1.0))\n    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs /= probs.sum()\n    return probs",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "retrieval_baseline": "[2.9, 3.4]",
    "validity_compression": "[0.85, 1.15]",
    "binding_uncertainty_strength": "[1.5, 2.1]",
    "binding_uncertainty_center": "[0.60, 0.67]",
    "binding_uncertainty_width": "[0.08, 0.13]",
    "compressed_support_gain": "[0.08, 0.18]",
    "high_validity_overload_strength": "[8.0, 10.0]",
    "overload_count_threshold": "[5.75, 5.9]",
    "overload_count_width": "[0.08, 0.16]",
    "overload_validity_center": "[0.90, 0.92]",
    "overload_validity_width": "[0.014, 0.025]",
    "order_anchor_reliance": "[0.68, 1.0]",
    "conspicuous_gap_center": "[0.055, 0.075]",
    "conspicuous_gap_width": "[0.012, 0.022]",
    "conspicuous_route_switch": "[0.35, 0.55]",
    "override_gap_scale": "[0.025, 0.04]",
    "override_support_gain": "[0.06, 0.14]",
    "override_threshold": "[0.75, 1.05]",
    "load_override_threshold": "[0.65, 0.95]",
    "late_override_threshold": "[0.65, 0.95]",
    "override_temperature": "[0.20, 0.34]",
    "response_lapse": "[0.09, 0.15]"
  },
  "rationale": "This is an isolated position-sign correction built directly from the accepted iteration-2 candidate. All parameter ranges, overload settings, ordinary-support settings, retrieval equations, and route-mixture equations are retained. The only substantive equation change replaces the elapsed-position factor with 0.5 + (0.5 - relative_position), which is a centered remaining-distance gradient plus the accepted model's original mean offset. Across balanced early, middle, and late positions its mean remains approximately 0.5, so it reverses the positional slope needed for Experiment 7 without shifting the average override threshold or disturbing Experiment 20's already exact pooled choice rate. Keeping the iteration-2 overload and support parameters avoids repeating the rejected changes that harmed Experiments 5, 13, 16, 17, and 20."
}
```

## Usage

```json
{
  "input_tokens": 63292,
  "output_tokens": 5248
}
```
