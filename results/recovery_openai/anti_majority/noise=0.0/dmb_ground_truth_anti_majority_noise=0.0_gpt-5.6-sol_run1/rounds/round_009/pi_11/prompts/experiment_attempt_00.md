# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Causal-Copy Discounted Lexicographic Choice with Local Priority Uncertainty. Decision makers compress all discriminating experts supporting the same option into an option-level recommendation and infer whether agreement reflects copied reports, shared evidence, or a common source. Common-cause suspicion rises nonlinearly with supporter count and is amplified by conspicuously high, homogeneous validities, actively discounting the recommendation's epistemic accessibility. Recommendations are ordinarily considered lexicographically by their strongest communicated validity. However, within a very narrow priority-gap neighborhood, validity ranks are not perfectly resolved: a small proportion of comparisons probabilistically interchange the two recommendation priorities. This local uncertainty permits an equal-validity or nearly equal-validity opposing recommendation to govern some choices without weakening lexicographic dominance for clearly separated validities. The causal discount remains independent of opponent count, adjacency, spacing, run structure, strongest-source placement, and terminal clustering. Separately, source-validity binding has a localized nonlinear uncertainty region, and isolated recommendations suffer a weak late-position accessibility loss only in genuinely long displays. Failure to access either recommendation leads to guessing, followed by a small response lapse.

**Parameters:**
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

**`predict source code`:**
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The 12 unique pairs form six diagnostic comparison patterns plus exact A/B reversals, yielding 96 trials with eight repetitions per pair. In every pattern, the highest-validity cue that discriminates favors the option designated by Take The Best, while the unweighted tally either favors the other option or is tied. The first three pattern types hold TTB's prediction constant while making the opposing tally margin equal to 1, 3, or 5 cues. Thus TTB predicts the same choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 across these margins, whereas Tallying predicts probability (1-epsilon)*sigmoid(-beta*m)+epsilon/2 for the TTB-designated option, with m in {1,3,5}. A fourth pattern gives an exact tally tie, so Tallying predicts 0.5 while TTB remains directional. The fifth and sixth patterns make the highest-validity cue tie and place the first discriminating cue at validity rank 2 or 3; TTB still follows that cue, whereas the majority of lower-ranked cues favors the opposite option by three. Exact reversals balance response side and should reverse TTB's preference without changing effect magnitudes. The broad, strictly ordered validities make the cue cascade salient and unambiguous.

[1] This design tests a quantitative confidence signature rather than a directional conflict. In every pair, the highest-validity discriminating cue and the unweighted majority favor the same option, but the tally margin varies systematically from 1 through 6. The six base patterns have A-minus-B win-count margins of 1, 2, 3, 4, 5, and 6; each also contains at least one cue favoring B, avoiding trivial unanimous dominance. Tallying predicts a graded increase in choice probability for A: (1-epsilon)*sigmoid(beta*m)+epsilon/2, where m is the tally margin. Take The Best always stops at Expert 1 and therefore predicts the same probability, (1-epsilon)*sigmoid(beta)+epsilon/2, at every margin. Margin 1 provides a useful probability anchor because the two rules have the same functional prediction there, whereas margins 2-6 reveal the nonlinear accumulation unique to Tallying. Exact A/B reversals control response-side preferences while preserving each margin. With 12 unique pairs repeated eight times, the experiment contains exactly 96 trials. Unlike previously explored cue-majority conflict designs, this congruent design asks whether additional agreeing comparative evidence changes choice reliability after the top cue has already selected the same winner.

[2] This design isolates a quantitative confidence dissociation between the two cascade theories. The six base pairs place the first discriminating expert at validity ranks 1 through 6; their exact A/B reversals balance response side. Take The Best assigns the same binary winner score on every pair, so it predicts one rank-invariant probability of following the first discriminating cue: q=(1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, every advertised validity is guaranteed to exceed the competing theory's confidence threshold throughout its parameter range: even the weakest expert has minimum subjective confidence 0.75*log(0.57/0.43)=0.211, above the maximum threshold of 0.20. Consequently, Confidence-Capped Cascade Integration also stops at the first discriminating cue, but its evidence magnitude equals sensitivity times that cue's log odds. It therefore predicts a strictly decreasing choice probability as the first discrimination moves from the 95% expert to the 57% expert: p_r=(1-epsilon)*sigmoid(beta*sensitivity*log(v_r/(1-v_r)))+epsilon/2. Thus, direction of choice is matched across theories while the predicted rank-by-confidence gradient is present only under the competing theory. Lower-ranked ratings alternate in direction where possible, preventing most displays from being simple unanimous dominance while remaining irrelevant after stopping. The 12 unique pairs are each repeated eight times, yielding exactly 96 trials.

[3] The 12 pairs comprise six weak-cue conflict levels and their exact A/B reversals, giving 96 trials with eight repetitions per pair. Expert 1 is highly valid (95%) but always ties, so it cannot determine a choice. Experts 2–8 all have validity 50.8%. Even at the advocated theory's maximum validity sensitivity, their subjective confidence is only 1.5*log(0.508/0.492)≈0.048, below its minimum confidence threshold of 0.05. Consequently, no discriminating cue can trigger cascade stopping, and Confidence-Capped Cascade Integration must combine all discriminating weak cues. Within each base pair, Expert 2 favors A and is the first discriminating cue under stable validity-tie ordering, while M of the remaining experts favor B, for M=1,...,6. Take The Best therefore predicts the same above-chance preference for A at every conflict level. The advocated theory instead predicts exact indifference when M=1 and an increasingly strong preference for B as M rises: its integrated evidence is proportional to (1-M)*[log(0.508/0.492)]^integration_exponent. Thus the critical signature is a reversal plus a graded conflict-level effect, neither of which TTB can generate with its single condition-invariant winner probability. Exact option reversals balance response-side biases while preserving these predictions. The tied 95% expert also supplies the required broad validity spread without accidentally activating the advocated theory's stopping rule.

[4] The six base pairs create a cumulative priority-race manipulation and are followed by exact A/B reversals, yielding 12 unique pairs repeated eight times for 96 trials. Expert 1 (95%) and Expert 2 (55%) always tie. The last-displayed expert has 72% validity and favors A in every base pair, whereas one through six earlier-displayed 70% experts favor B. Take The Best sorts cues strictly by advertised validity, skips the tied 95% cue, and always stops at the 72% cue. It therefore predicts the same above-chance probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A at every base conflict level, irrespective of how many 70% cues oppose it. Noisy Priority Race Search instead gives the earlier 70% cues a serial-position advantage. Conflict level 1 activates only the 70% cue immediately before the 72% cue; each subsequent level adds the next-earlier 70% cue. Adding a B-favoring competitor strictly increases the race probability that B supplies the stopping cue, so the competing theory predicts a monotonic decline in A choices across levels. The highest conflict level also provides a parameter-robust directional reversal: for the earliest active 70% cue, its minimum priority advantage over the 72% cue is 6(0.15)-5[logit(0.72)-logit(0.70)]-0.16, approximately 0.254. Even at retrieval noise 0.08, that cue alone has over 23 times the race weight of the 72% cue, and all other discriminating race cues also favor B. Thus the competing theory predicts a strong B preference at level 6, while TTB must continue to prefer A. Exact option reversals balance response-side biases and reverse directions without changing the predicted conflict-level profile.

[5] The six base pairs isolate conflicts between two discriminating experts separated by exactly two display positions; all other experts tie and therefore cannot terminate either search rule. In every base pair, the earlier expert has validity 0.55 and favors B, whereas the later expert has slightly to substantially higher validity and favors A. The later-minus-earlier validity-log-odds gaps are approximately 0.01, 0.15, 0.35, 0.60, 0.90, and 1.30. Take The Best always ranks the later expert above the 0.55 expert, so it predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 at all six levels, regardless of the size of the validity gap. Noisy Priority Race Search instead predicts a validity-gap-dependent crossover because the earlier B-favoring expert receives a constant two-position serial advantage. Its later-minus-earlier priority difference is sensitivity*D - 2*serial_priority + (bias_later-bias_earlier). At D≈0.01 this difference is at most 5(0.01)-2(0.15)+0.16=-0.09 across the entire parameter range, so the advocated theory robustly predicts that the nominally less-valid, earlier expert wins the race and B is chosen more often. At D≈1.30 the difference is at least 1(1.30)-2(0.45)-0.16=0.24, so it robustly predicts that the more-valid expert wins and A is chosen more often. Intermediate gaps locate each subject's stochastic priority crossover and should produce a rising psychometric profile, potentially with stable cue-specific deviations from perfect monotonicity. Thus the critical signature is both a directional reversal at the smallest gap and a quantitative increase in following the higher-validity expert as the log-odds gap grows; TTB can only fit one condition-invariant response probability. The intervening tied experts preserve the same two-position separation, while tied 95% and 50% experts provide a broad advertised-validity spread without affecting choices. Each base pair has an exact A/B reversal, balancing response-side preferences. The 12 unique pairs are repeated eight times, yielding exactly 96 trials.

[6] The six base pairs hold Take The Best's prediction constant while moving the sole higher-validity A-favoring discriminating cue through display positions 3, 8, 14, 20, 26, and 32 (zero-based positions 2, 7, 13, 19, 25, and 31). Two lower-validity cues at the first two display positions always favor B; every other cue ties. TTB skips the higher-validity tying cues, stops at the designated target cue, and therefore predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in all six conditions. The accessibility-gated cascade instead predicts a steep decline in A choices with target position. If a_t is target accessibility and R=(1-a_0)(1-a_1) is the probability that both early B cues are inaccessible, its intended A probability is a_t+0.5*(1-a_t)*R. The linear and quadratic position costs make a_t decrease strongly across conditions, while the two early fallback cues remain highly accessible. This produces a parameter-robust crossover: at position 2, target accessibility is at least about 0.78, whereas at position 31 it is at most about 0.38. Even using the least-accessible permissible fallback cues, intended A probability in the final condition is at most about 0.40, so execution flips and lapses attenuate but do not reverse the predicted B preference. Thus the competitor predicts a declining psychometric profile from a clear A preference to a B preference, while TTB predicts rank- and position-invariant choice reliability. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

[7] The six base conditions manipulate same-direction redundancy within the validity cascade while holding Take The Best's decisive cue fixed. In every base pair, the 95%-valid Expert 24 favors A, so TTB always stops there and predicts the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Experts 1 and 2 favor B but have lower validities (72% and 70%). Condition 1 contains only the late-displayed 95% A cue; conditions 2 through 6 successively add A-favoring backup cues with validities 93%, 91%, 89%, 87%, and 85%. All remaining cues tie. Under the Accessibility-Gated Validity Cascade, each added backup supplies another opportunity for an A-favoring cue to be accessible before search reaches the B-favoring cues. If Q_m is the probability that all m active A cues are inaccessible and S is the probability that both B cues are inaccessible, intended P(A)=1-Q_m+0.5*Q_m*S. Thus A choice must increase monotonically as backups are added, with a characteristic diminishing-returns profile from the multiplicative accessibility failures. The manipulation is especially diagnostic because all A cues occupy terminal display positions 19–24, where the advocated theory predicts appreciable accessibility failures in a 24-cue list, whereas the two B fallbacks occupy the highly accessible first positions. Execution flips and lapses attenuate but preserve the positive redundancy gradient. TTB cannot produce this condition effect because its 95% cue already discriminates in every condition and lower-validity cues are never consulted. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted redundancy profile. Twelve unique pairs repeated eight times yield 96 trials.

[8] The six base conditions hold both the amount and direction of evidence constant while manipulating only the spatial density of a same-direction cue cluster. Expert 17 (index 16) has the uniquely highest validity, 92%, and always favors A. Six 90%-valid experts also favor A, while an isolated 55%-valid expert favors B. Across conditions, the six redundant A cues move symmetrically away from Expert 17: their offsets are ±{1,2,3}, ±{1,3,5}, ±{2,4,6}, ±{2,5,8}, ±{3,7,11}, and ±{5,10,15}. This symmetry approximately balances early-versus-late positional effects in the 32-cue display, and the number, validities, and recommendations of all discriminating cues remain fixed. Take The Best always stops at the 92% cue and therefore predicts one condition-invariant probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 of choosing A. The Competitive Source-Binding Cascade instead predicts a pronounced spacing gradient. In the densest condition, the seven nearby, similarly valid A sources form a mutually inhibiting cluster; even the leading 92% source receives crowding from both sides, permitting the isolated B source to control many choices. As spacing increases, positional similarity and hence crowding decline exponentially, so successful retrieval of at least one A source should increase and A choices should rise, potentially crossing from below to above chance. This is an adversarial redundancy effect: distributing exactly the same supporting evidence improves its use, whereas TTB is indifferent to both redundancy and spacing. Exact A/B reversals balance response-side preferences and should invert the competing theory's spacing profile without changing its magnitude. The 12 unique pairs are repeated eight times, yielding 96 trials.

[9] This design translates a fixed two-cue competitive cluster across a 32-expert display while holding its size, within-cluster spacing, validity, recommendation, and opposing evidence constant. In each base pair, exactly two adjacent 82%-valid experts favor A, and the isolated 55%-valid Expert 16 favors B. The six A-cluster locations are positions 1–2, 7–8, 13–14, 19–20, 25–26, and 31–32. The 95%-valid Expert 4 always ties. Take The Best therefore skips the tied 95% expert and stops at the first active 82% expert, predicting the same A-choice probability q=(1-epsilon)*sigmoid(beta)+epsilon/2 in every condition. The Competitive Source-Binding Cascade predicts a different quantitative profile. Because the two A cues are adjacent, equally valid, and recommend the same option, their mutual competitive crowding is identical across conditions; because their moderate validity lies near the model's binding-uncertainty region, failed retrieval should frequently permit the fixed, isolated B cue to control choice. Translating the cluster later selectively increases its binding support through the 32-cue long-display positional contrast, with an additional sharply localized recency benefit for the terminal cluster. Thus the model predicts increasing A choices from early to late cluster locations, typically with the largest improvement at positions 31–32 and potentially a crossover from following B to following A. This isolates serial-position recovery from cluster density, unlike a spacing manipulation: all competitive relationships within the cluster remain unchanged. The final six pairs are exact A/B reversals, balancing response-side preferences while preserving the predicted location gradient. Twelve unique pairs repeated eight times yield 96 trials.

[10] The six base conditions vary the number of widely separated, same-direction 82%-valid sources supporting A from 2 through 7. Their positions are selected to remain broadly symmetric around the display midpoint, with at least seven positions between neighboring A sources. A single 70%-valid terminal source supports B, while the 95%-valid source and all remaining sources tie. The advocated Chunked Source-Retrieval Lexicographic Search treats similarity-based interference as source-based and independent of spatial distance. Because the A sources are nonadjacent, they cannot form chunks. Increasing their number therefore raises each source's attribution load; from three sources onward it also invokes the correlated recommendation-level gate. It predicts a pronounced decline in A choices as the A set grows, particularly from approximately 3–4 to 7 sources, despite all added evidence supporting A. The Competitive Source-Binding Cascade predicts the opposite profile. Its inhibition is strongly attenuated by the large spatial separations, so each added 82% source provides another relatively independent opportunity to stop the validity-ordered cascade before the 70% B source. It therefore predicts increasing A choices with source count. Thus the critical quantitative test is the sign of the redundancy slope: negative under the advocated theory but positive under the competitor. The six exact A/B reversals balance response-side preferences and should invert choice direction while preserving the count effect. Twelve unique pairs repeated eight times yield 96 trials.

[11] This 24-expert design isolates the advocated theory's terminal-recency mechanism without activating its long-display positional contrast, which is exactly zero at n_features=24. In each base pair, one 82%-valid singleton favors A and the fixed 70%-valid Expert 1 favors B; the 95%-valid expert and every other expert tie. Across six conditions, the otherwise identical 82% singleton appears at positions 14, 16, 18, 20, 22, and 24. There is no same-direction crowding because only one cue supports each option, and neither adjacency chunking nor recommendation-level shared gates can occur. Competitive Source-Binding Cascade predicts a sharply increasing, convex A-choice profile as the target approaches the display endpoint: its recency contribution rises from approximately exp(-10/2.76) to 1 across these positions, producing a terminal binding-logit gain of at least about 1.3 relative to position 14. When the 82% cue is not bound, search reaches the fixed 70% B cue, making target-binding changes consequential and potentially producing a crossover toward A at the terminal position. Chunked Source-Retrieval Lexicographic Search has no recency benefit. Its target retrieval probability is position-invariant except for the weak monotonic singleton late-position loss; across positions 14 to 24, that penalty changes by at most about 0.026 logits. It therefore predicts an approximately flat or slightly decreasing A-choice profile, directly opposing the advocated theory's pronounced terminal increase. Exact A/B reversals balance response-side preferences. The 12 unique pairs are repeated eight times, yielding 96 trials.

[12] The six base trials form three matched run-length contrasts (lengths 3, 4, and 5), followed by exact A/B reversals. In every base trial, a contiguous run of same-direction experts favors A, its uniquely strongest member has 92% validity, and an isolated 70%-valid expert favors B. For each run length, the validity multiset is held fixed while the 92% expert is either at the run boundary or strictly inside the run. Chunked Source-Retrieval Lexicographic Search predicts approximately no boundary-versus-interior difference: each contiguous run is encoded as one chunk ranked by its 92% member, and its attribution load depends on source count and validity similarity rather than the strongest member's location within the run. It may predict declining A choice with run length because larger redundant sets increase source load and share a correlated gate, but that decline should be essentially parallel for boundary and embedded arrangements. Structure-Gated Ensemble Lexicographic Retrieval makes the opposing quantitative prediction. A boundary-anchored run receives the full overload cost, which increases as 0.45–0.65 times run length minus two, whereas a uniquely strongest interior expert reduces that cost to 10% of its ordinary value. It therefore predicts substantially more A choices for embedded than boundary runs, with the contrast growing sharply from length 3 to length 5. Failure to retrieve the A ensemble exposes the highly retrievable 70% B singleton, making this structural difference consequential for choice. The critical signature is thus a run-length-by-anchor-location interaction: near-zero matched contrasts under the advocated theory versus an increasingly positive embedded-anchor advantage under the competitor. The tied 97% expert broadens the validity range without entering the search. Twelve unique pairs repeated eight times yield 96 trials, and exact reversals balance response side.

[13] This design creates three matched adjacency contrasts, each at a different display midpoint. In every base trial, exactly two equally valid 82% experts favor A and one less-valid 70% expert favors B; the 97% expert and all other experts tie. Within each matched contrast, the two A experts have the same midpoint but are either adjacent or separated by six intervening positions: positions 7–8 versus 4 and 11, positions 10–11 versus 7 and 14, and positions 13–14 versus 10 and 17 (one-based). Thus validity, recommendation count, opposing evidence, and mean serial position are held fixed while only adjacency changes. Structure-Gated Ensemble Lexicographic Retrieval predicts an adjacency advantage. An adjacent same-direction pair becomes a high-fidelity pair chunk, whereas two separated sources become a single recommendation-level ensemble whose extra-source gain is at most negligible; hence A choices should be reliably higher for adjacent than separated pairs. Chunked Source-Retrieval Lexicographic Search predicts the opposite sign. Its source-similarity load is identical regardless of spacing, but adjacency usually collapses the two A sources into one chunk that retains the full attribution load and offers only one retrieval opportunity. When separated, the two loaded source representations remain distinct and can independently rescue one another; because groups of two are exempt from the shared gate, this should increase A choices. The weak singleton position penalty cannot explain the contrast because each adjacent/separated pair is midpoint-matched and its maximum magnitude is tiny relative to the second retrieval opportunity. The critical quantitative signature is therefore a sign-reversing adjacency effect: adjacent greater than separated under the advocated theory, but separated greater than adjacent under the competitor, replicated at three locations. The six exact A/B reversals balance response side, producing 12 unique pairs repeated eight times for 96 trials.

[14] This design directly targets the advocated theory's competition gate while keeping the contiguous recommendation itself fixed. In every base trial, Experts 2–7 form the same six-source contiguous run favoring A, with the uniquely strongest 92%-valid member at the run boundary. A lower-validity recommendation favoring B is supported by one through six mutually separated experts at positions 9, 11, 13, 15, 17, and 19. The first B expert is always the strongest B source (70%); added B experts are weaker and separated by inactive experts, so they never form pairs or runs. Competition-Normalized Recommendation Race predicts a large, sharply saturating decline in A choices. With one B supporter, its opponent-excess term is zero and the six-source A run receives no density-overload penalty. Adding the second B supporter opens the competition gate, imposing an A-retrieval-logit cost of 3*contiguous_density_overload*(1-exp(-1)), approximately 1.90–3.03 logits. Further B supporters raise the gate toward one, yielding a total cost near 2.98–4.77 logits at six supporters. Thus the largest choice change should occur from one to two B sources, followed by saturation. Structure-Gated Ensemble Lexicographic Retrieval predicts an approximately flat count profile: the fixed A run receives the same length-six overload in every condition, while all separated B sources are compressed into one recommendation-level ensemble whose count-dependent gain is at most 0.02*(1-exp(-(m-1))) logits. Hence it cannot produce the advocated theory's large one-to-two-source cliff. The tied 97% expert broadens the validity range without entering either search. The final six pairs are exact A/B reversals, balancing response side while preserving the predicted count profile. Twelve unique pairs repeated eight times yield 96 trials.

[15] The six base trials form two matched structure sets with three or four 82%-valid experts favoring A and one fixed 70%-valid singleton favoring B. Within each source-count set, the A experts are arranged as (i) one contiguous run, (ii) structures containing adjacent pairs (a pair plus a singleton for count three; two separated pairs for count four), or (iii) entirely separated sources. All potentially active A positions have identical 82% validity, so recommendation count, validity, and opposing evidence are held constant within each set; only grouping structure changes. The 18-feature display also eliminates both theories' long-display position terms. Structure-Gated Ensemble Lexicographic Retrieval predicts a pronounced structural ordering. Adjacent pairs become high-fidelity chunks and remain separate from other A units, providing multiple lexicographic retrieval opportunities; therefore pair-plus-singleton and especially two-pair arrangements should produce the most A choices. Entirely separated A sources collapse into one recommendation-level ensemble with negligible redundancy gain, producing an intermediate A-choice rate. A contiguous run is compressed into one overloaded ensemble; because the 82% members are tied, there is no uniquely embedded strongest-member protection, and overload doubles from the length-three to the length-four run. Thus the advocated theory predicts pair-containing structures > separated structures > contiguous runs, with a particularly large two-pairs-versus-length-four-run contrast. Competition-Normalized Recommendation Race instead compresses all A supporters into one recommendation representation. Within a fixed source count, its accessibility is exactly invariant to these spatial arrangements: the length-three run never reaches its run-excess threshold, and although the length-four run does, its singleton opponent makes the competition gate zero. It therefore predicts equal choice probabilities across the three structures at each count, apart from no structural effect at all. The six exact A/B reversals balance response side and invert the advocated structural profile. Twelve unique pairs repeated eight times yield 96 trials.

[16] This 18-expert factorial design crosses the length of a fixed contiguous high-validity recommendation (three, four, or six 94%-valid experts favoring the run-supported option) with whether its lower-validity opponent has one supporter (70%) or two separated supporters (70% and 60%). The tied 98% expert broadens the advertised-validity range without entering either recommendation. Competition-Normalized Recommendation Race predicts a qualitative interaction. Against the singleton opponent, the competition gate is exactly closed, so increasing the run from three to six sources produces essentially no loss; the only count effect is a negligible saturating gain. Against the two-source opponent, the gate opens. The length-three run remains unpenalized, the length-four run loses approximately 0.63–1.01 retrieval logits, and the length-six run loses approximately 1.90–3.03 logits. It therefore predicts a flat run-length profile with one opponent supporter but a steep, nonlinear decline in choices for the run-supported option with two opponent supporters. Validity-Gated Intrinsic Recommendation Capacity predicts a different quantitative pattern: the 94%-indexed recommendation pays its own smoothly increasing concentration cost regardless of opponent count. From length three to length six, that cost increases by approximately 1.42–2.43 logits across its parameter range. Because the added opponent has only 60% validity and the opponent representation is indexed by 70%, its high-diagnosticity capacity gate is essentially zero; adding it changes accessibility only negligibly. Thus the competing theory predicts similarly steep, approximately parallel run-length declines in both opponent-count strata, not the advocated theory's run-length-by-opponent-count interaction. The final six pairs are exact A/B reversals, balancing response side while preserving the probability of choosing the run-supported recommendation. Twelve unique pairs repeated eight times yield 96 trials.

[17] This 24-expert design tests the advocated theory's validity-gated intrinsic capacity cost while eliminating both theories' long-display position terms. The six base trials cross recommendation validity (94% versus 80%) with the number of same-direction supporters (1, 3, or 6). The 94% supporters occupy Experts 2, 5, 8, 11, 14, and 17; the 80% supporters occupy Experts 3, 6, 9, 12, 15, and 18. Thus supporters are always mutually separated, and the opposing recommendation is always the same 70%-valid singleton at Expert 22. Under Validity-Gated Intrinsic Recommendation Capacity, the count-dependent concentration term is strongly enabled at 94% but almost completely disabled at 80%: the diagnosticity gates are approximately sigmoid((0.94-0.90)/0.022)=0.86 and sigmoid((0.80-0.90)/0.022)=0.011. Relative to a singleton, the concentration term is approximately 0.566 at count 3 and 2.922 at count 6. Consequently, the advocated theory predicts a modest-to-moderate decline in following the 94% recommendation from one to three supporters and a very large decline by six supporters, whereas the 80% count profile should remain nearly flat. Competition-Normalized Recommendation Race predicts no corresponding interaction. Every multi-source recommendation has longest contiguous run one, and its opponent is a singleton, so density overload is identically absent. Its only count effect is a positive saturating gain of at most 0.003 logits, yielding essentially flat or minutely increasing profiles at both validities. The critical signature is therefore a large negative source-count slope specifically in the 94% condition, producing a strong validity-by-count interaction. A tied 98% expert and several low-validity tied experts provide a broad advertised-validity spread without entering either recommendation. The final six trials are exact A/B reversals, balancing response side. The 12 unique pairs are repeated eight times for 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 14-expert design isolates the advocated theory's common-cause homogeneity discount while holding recommendation count, strongest validity, opposing evidence, and display position constant. In every base trial, exactly six experts favor A: the fixed 94%-valid Expert 2 plus five additional supporters. A fixed 89%-valid singleton favors B. Across six conditions, the A bundle progresses from extremely homogeneous validities [94, 94, 93.8, 93.6, 93.4, 93.2] to increasingly heterogeneous sets, ending with [94, 94, 93.8, 82, 78, 70]. The bundle always has six supporters and is always represented by the same earliest 94% expert. Causal-Copy Discounted Lexicographic Choice predicts a large nonlinear increase in choices for the bundle-supported option: the homogeneous high-validity bundle crosses its count threshold and receives a severe inferred-common-cause discount, often allowing the 89% opponent to govern choice, whereas increasing validity dispersion rapidly eliminates the homogeneity term and restores strong lexicographic preference for the 94%-indexed bundle. Validity-Gated Intrinsic Recommendation Capacity predicts exactly the same choice probability across all six conditions: its recommendation representation depends on the unchanged strongest expert, source count, and opponent, while the validities of nonrepresentative supporters never enter its capacity or retrieval calculations. Thus the critical signature is a steep, potentially choice-reversing homogeneity gradient under the advocated theory versus a flat profile under the competitor. The tied 99% and 52% experts broaden the advertised-validity range without entering either recommendation. Six exact A/B reversals balance response side, producing 12 unique pairs repeated eight times for 96 trials.",
  "validities": [
    0.89,
    0.94,
    0.94,
    0.938,
    0.936,
    0.934,
    0.932,
    0.9,
    0.86,
    0.82,
    0.78,
    0.7,
    0.99,
    0.52
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 12385,
  "output_tokens": 3079
}
```
