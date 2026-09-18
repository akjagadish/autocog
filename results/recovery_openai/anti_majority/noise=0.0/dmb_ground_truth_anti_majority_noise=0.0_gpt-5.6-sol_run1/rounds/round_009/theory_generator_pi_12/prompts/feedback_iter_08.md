# feedback_iter_08

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_11`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_11`).

Replace pi_11 with a load-adaptive dual-route theory rather than another causal-copy account. The new theory should posit competition between (a) validity-directed search and (b) display-order anchoring. In short displays or when validity differences are conspicuous, validity-directed search predominates and yields the lexicographic signatures seen in Experiments 3-8. As display length and visual search load increase, the first discriminating recommendation becomes an anchor; later evidence can overturn it only when its validity advantage or accumulated diagnostic support exceeds a load-dependent threshold. This route should generate strong early-cue control for the 80%-versus-82% conflict in the 32-feature Experiment 2, a negative rather than positive late-position gradient in Experiment 9, and negligible position effects around 24 features as in Experiments 12 and 14. Same-direction sources may be compressed, but their cost should depend on count, diagnosticity, and working-memory concentration—not on validity homogeneity, adjacency, run structure, or opponent count. Count costs should be weak at moderate validity, consistent with Experiments 1 and 13, but sharply stronger for very large, extremely high-validity bundles, consistent with Experiments 10 and 20. Include stable subject-level variation in reliance on the order-anchor versus validity route to accommodate Experiment 2's larger observed heterogeneity. The model should use a validity-gap-sensitive override function, a display-load-gated primacy term with the empirically supported sign, and a high-validity count-overload term that is invariant to within-bundle validity dispersion. It should make joint predictions for absolute choice rates and positional/count contrasts, rather than adding isolated mechanisms tailored to individual metrics.


## CANDIDATE THEORY
Load-Adaptive Anchor–Validity Arbitration. Choice is produced by competition between a validity-directed search route and an order-anchor route. Validity-directed search compresses all currently agreeing experts into option-level recommendations, ranks those recommendations by their strongest communicated validity, and usually stops at the first accessible recommendation. Its accessibility includes localized source–validity binding uncertainty and an intrinsic count-concentration cost. The count cost is independent of spatial arrangement, opponent count, and within-bundle validity dispersion; it remains weak for ordinary bundles but rises sharply when at least five or six extremely high-validity sources must be maintained as one recommendation. The order-anchor route is gated by display load. In long displays, the first discriminating recommendation establishes a provisional choice. A later opposing recommendation overturns that anchor only when its validity advantage and compressed diagnostic support jointly exceed a load-dependent threshold. In addition, when a validity-superior challenger is conspicuous, its display position modulates competition between the routes: later challengers reduce reliance on the anchor route and shift control toward validity-directed search. This route-arbitration modulation is mean-centered over display position, absent for short displays, and negligible for small validity gaps such as 80% versus 82%. Stable subject-level variation in order-anchor reliance and route-position sensitivity produces heterogeneous long-display behavior without trial feedback or experiment-specific rules.

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
    route_position_slope = float(parameters["route_position_slope"])
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

    # For a conspicuous challenger in a long display, position changes route
    # arbitration rather than the override threshold. The modulation is
    # centered at the display midpoint, so it changes the positional slope
    # while approximately preserving average anchor reliance. Later
    # challengers reduce anchor-route weight and favor validity-directed search.
    if opposing_unit is not None and 0.0 < anchor_weight < 1.0:
        centered_position = 0.5 - relative_position
        anchor_logit = np.log(anchor_weight) - np.log1p(-anchor_weight)
        anchor_logit += (
            route_position_slope
            * display_load
            * conspicuousness
            * centered_position
        )
        anchor_weight = sigmoid_scalar(anchor_logit)

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
- route_position_slope: [0.8, 1.4]
- override_gap_scale: [0.025, 0.04]
- override_support_gain: [0.06, 0.14]
- override_threshold: [0.75, 1.05]
- load_override_threshold: [0.65, 0.95]
- late_override_threshold: [0.65, 0.95]
- override_temperature: [0.20, 0.34]
- response_lapse: [0.09, 0.15]

`rationale`:
This is an isolated minimal-diff edit to the accepted iteration-2 base. All accepted retrieval, uncertainty, support, overload, override, lapse, and route-mixture ranges are retained. The only new mechanism is `route_position_slope`: a mean-centered position term is added to the logit of anchor-route weight, rather than to the override threshold or override probability. For conspicuous challengers in 32-feature displays, early positions increase anchor reliance and later positions decrease it, allowing the validity route to produce the required negative Experiment-7 contrast. Multiplication by display load protects the 24-feature position nulls, while multiplication by conspicuousness makes the modulation negligible for Experiment 20's small 80%-versus-82% gap. Centering at relative position 0.5 approximately preserves pooled anchor reliance and therefore protects Experiment 20's already exact absolute choice rate. Subject-level sampling of the new slope also supplies theoretically relevant heterogeneity without increasing response lapse.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0541 -> ACCEPTED
- iter 2: loss=0.0410 -> ACCEPTED
- iter 3: loss=0.0418 -> REJECTED
- iter 4: loss=0.0536 -> REJECTED
- iter 5: loss=0.0465 -> REJECTED
- iter 6: loss=0.0494 -> REJECTED
- iter 7: loss=0.0449 -> REJECTED
- iter 8: loss=0.0550 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.0516 -> REJECTED
Running-best (last accepted) base: iter 2 at loss=0.0410.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
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
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (current): 1.0000 (var=0.0000) (Δ vs real +0.0000)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.0000) (Δ vs real +0.0000)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.2400) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.2436) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.2356) (Δ vs real +0.0000)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0912 (var=0.0013) (Δ vs real -0.0708)
  - iter 2: 0.0960 (var=0.0013) (Δ vs real -0.0660)
  - iter 3: 0.0993 (var=0.0013) (Δ vs real -0.0628)
  - iter 4: 0.0915 (var=0.0011) (Δ vs real -0.0705)
  - iter 5: 0.0925 (var=0.0013) (Δ vs real -0.0695)
  - iter 6: 0.0905 (var=0.0013) (Δ vs real -0.0715)
  - iter 7: 0.0910 (var=0.0012) (Δ vs real -0.0710)
  - iter 8: 0.1040 (var=0.0011) (Δ vs real -0.0580)
  - iter 9 (current): 0.0955 (var=0.0007) (Δ vs real -0.0665)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.1237 (var=0.0040) (Δ vs real -0.0375)
  - iter 2: 0.1181 (var=0.0026) (Δ vs real -0.0431)
  - iter 3: 0.1062 (var=0.0031) (Δ vs real -0.0550)
  - iter 4: 0.1138 (var=0.0030) (Δ vs real -0.0475)
  - iter 5: 0.1144 (var=0.0033) (Δ vs real -0.0469)
  - iter 6: 0.1106 (var=0.0041) (Δ vs real -0.0506)
  - iter 7: 0.1244 (var=0.0043) (Δ vs real -0.0369)
  - iter 8: 0.1294 (var=0.0038) (Δ vs real -0.0319)
  - iter 9 (current): 0.1031 (var=0.0035) (Δ vs real -0.0581)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.1625 (var=0.0169) (Δ vs real -0.0350)
  - iter 2: -0.1212 (var=0.0165) (Δ vs real +0.0063)
  - iter 3: -0.1225 (var=0.0184) (Δ vs real +0.0050)
  - iter 4: -0.1588 (var=0.0169) (Δ vs real -0.0313)
  - iter 5: -0.1338 (var=0.0252) (Δ vs real -0.0063)
  - iter 6: -0.1237 (var=0.0226) (Δ vs real +0.0038)
  - iter 7: -0.1400 (var=0.0123) (Δ vs real -0.0125)
  - iter 8: -0.1212 (var=0.0127) (Δ vs real +0.0063)
  - iter 9 (current): -0.1725 (var=0.0156) (Δ vs real -0.0450)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0052 (var=0.0015) (Δ vs real +0.0595)
  - iter 2: 0.0258 (var=0.0020) (Δ vs real +0.0801)
  - iter 3: -0.0003 (var=0.0010) (Δ vs real +0.0540)
  - iter 4: 0.0014 (var=0.0010) (Δ vs real +0.0557)
  - iter 5: -0.0432 (var=0.0017) (Δ vs real +0.0112)
  - iter 6: -0.0279 (var=0.0017) (Δ vs real +0.0264)
  - iter 7: 0.0262 (var=0.0023) (Δ vs real +0.0805)
  - iter 8: 0.0204 (var=0.0021) (Δ vs real +0.0747)
  - iter 9 (current): 0.0134 (var=0.0016) (Δ vs real +0.0677)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.4206 (var=0.0100) (Δ vs real -0.1531)
  - iter 2: -0.3763 (var=0.0045) (Δ vs real -0.1088)
  - iter 3: -0.3706 (var=0.0051) (Δ vs real -0.1031)
  - iter 4: -0.3769 (var=0.0044) (Δ vs real -0.1094)
  - iter 5: -0.3862 (var=0.0045) (Δ vs real -0.1187)
  - iter 6: -0.3831 (var=0.0057) (Δ vs real -0.1156)
  - iter 7: -0.3931 (var=0.0052) (Δ vs real -0.1256)
  - iter 8: -0.4031 (var=0.0067) (Δ vs real -0.1356)
  - iter 9 (current): -0.3900 (var=0.0079) (Δ vs real -0.1225)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0044 (var=0.0105) (Δ vs real +0.0100)
  - iter 2: -0.0088 (var=0.0094) (Δ vs real -0.0031)
  - iter 3: -0.0047 (var=0.0109) (Δ vs real +0.0009)
  - iter 4: 0.0075 (var=0.0107) (Δ vs real +0.0131)
  - iter 5: 0.0156 (var=0.0135) (Δ vs real +0.0212)
  - iter 6: -0.0122 (var=0.0091) (Δ vs real -0.0066)
  - iter 7: 0.0088 (var=0.0121) (Δ vs real +0.0144)
  - iter 8: 0.0053 (var=0.0120) (Δ vs real +0.0109)
  - iter 9 (current): -0.0044 (var=0.0124) (Δ vs real +0.0012)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0127 (var=0.0103) (Δ vs real +0.0172)
  - iter 2: -0.0140 (var=0.0094) (Δ vs real -0.0095)
  - iter 3: -0.0057 (var=0.0090) (Δ vs real -0.0012)
  - iter 4: -0.0055 (var=0.0099) (Δ vs real -0.0010)
  - iter 5: 0.0343 (var=0.0077) (Δ vs real +0.0388)
  - iter 6: 0.0048 (var=0.0047) (Δ vs real +0.0093)
  - iter 7: -0.0080 (var=0.0106) (Δ vs real -0.0035)
  - iter 8: 0.0012 (var=0.0061) (Δ vs real +0.0057)
  - iter 9 (current): -0.0297 (var=0.0118) (Δ vs real -0.0252)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0106 (var=0.0081) (Δ vs real +0.0119)
  - iter 2: -0.0100 (var=0.0079) (Δ vs real +0.0125)
  - iter 3: -0.0138 (var=0.0098) (Δ vs real +0.0087)
  - iter 4: -0.0112 (var=0.0091) (Δ vs real +0.0113)
  - iter 5: 0.0050 (var=0.0083) (Δ vs real +0.0275)
  - iter 6: 0.0069 (var=0.0051) (Δ vs real +0.0294)
  - iter 7: 0.0075 (var=0.0078) (Δ vs real +0.0300)
  - iter 8: 0.0069 (var=0.0066) (Δ vs real +0.0294)
  - iter 9 (current): -0.0119 (var=0.0064) (Δ vs real +0.0106)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0125 (var=0.0128) (Δ vs real -0.0275)
  - iter 2: -0.0304 (var=0.0183) (Δ vs real -0.0454)
  - iter 3: 0.0354 (var=0.0105) (Δ vs real +0.0204)
  - iter 4: -0.0079 (var=0.0227) (Δ vs real -0.0229)
  - iter 5: -0.0138 (var=0.0150) (Δ vs real -0.0288)
  - iter 6: -0.0050 (var=0.0205) (Δ vs real -0.0200)
  - iter 7: 0.0054 (var=0.0132) (Δ vs real -0.0096)
  - iter 8: -0.0146 (var=0.0151) (Δ vs real -0.0296)
  - iter 9 (current): -0.0046 (var=0.0194) (Δ vs real -0.0196)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0173 (var=0.0098) (Δ vs real +0.0203)
  - iter 2: -0.0050 (var=0.0047) (Δ vs real -0.0020)
  - iter 3: 0.0226 (var=0.0048) (Δ vs real +0.0256)
  - iter 4: -0.0088 (var=0.0055) (Δ vs real -0.0058)
  - iter 5: 0.0075 (var=0.0049) (Δ vs real +0.0105)
  - iter 6: -0.0221 (var=0.0049) (Δ vs real -0.0191)
  - iter 7: -0.0092 (var=0.0058) (Δ vs real -0.0062)
  - iter 8: 0.0064 (var=0.0036) (Δ vs real +0.0094)
  - iter 9 (current): -0.0036 (var=0.0046) (Δ vs real -0.0006)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0079 (var=0.0061) (Δ vs real +0.0104)
  - iter 2: 0.0021 (var=0.0068) (Δ vs real +0.0204)
  - iter 3: -0.0121 (var=0.0077) (Δ vs real +0.0063)
  - iter 4: 0.0271 (var=0.0085) (Δ vs real +0.0454)
  - iter 5: -0.0042 (var=0.0059) (Δ vs real +0.0142)
  - iter 6: 0.0175 (var=0.0065) (Δ vs real +0.0358)
  - iter 7: -0.0117 (var=0.0059) (Δ vs real +0.0067)
  - iter 8: 0.0212 (var=0.0069) (Δ vs real +0.0396)
  - iter 9 (current): -0.0067 (var=0.0076) (Δ vs real +0.0117)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0100 (var=0.0131) (Δ vs real +0.0105)
  - iter 2: -0.0135 (var=0.0122) (Δ vs real +0.0070)
  - iter 3: -0.0172 (var=0.0096) (Δ vs real +0.0032)
  - iter 4: 0.0015 (var=0.0184) (Δ vs real +0.0220)
  - iter 5: -0.0197 (var=0.0116) (Δ vs real +0.0008)
  - iter 6: 0.0010 (var=0.0122) (Δ vs real +0.0215)
  - iter 7: -0.0160 (var=0.0117) (Δ vs real +0.0045)
  - iter 8: 0.0190 (var=0.0174) (Δ vs real +0.0395)
  - iter 9 (current): 0.0185 (var=0.0103) (Δ vs real +0.0390)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0113 (var=0.0150) (Δ vs real -0.0437)
  - iter 2: 0.0413 (var=0.0168) (Δ vs real -0.0137)
  - iter 3: 0.0175 (var=0.0189) (Δ vs real -0.0375)
  - iter 4: -0.0112 (var=0.0245) (Δ vs real -0.0662)
  - iter 5: 0.0037 (var=0.0207) (Δ vs real -0.0513)
  - iter 6: -0.0188 (var=0.0158) (Δ vs real -0.0738)
  - iter 7: 0.0037 (var=0.0174) (Δ vs real -0.0513)
  - iter 8: 0.0137 (var=0.0129) (Δ vs real -0.0413)
  - iter 9 (current): -0.0150 (var=0.0193) (Δ vs real -0.0700)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0063 (var=0.0194) (Δ vs real +0.0137)
  - iter 2: -0.0175 (var=0.0173) (Δ vs real +0.0025)
  - iter 3: 0.0000 (var=0.0173) (Δ vs real +0.0200)
  - iter 4: 0.0037 (var=0.0119) (Δ vs real +0.0237)
  - iter 5: 0.0175 (var=0.0153) (Δ vs real +0.0375)
  - iter 6: 0.0000 (var=0.0109) (Δ vs real +0.0200)
  - iter 7: -0.0038 (var=0.0144) (Δ vs real +0.0162)
  - iter 8: 0.0225 (var=0.0170) (Δ vs real +0.0425)
  - iter 9 (current): -0.0113 (var=0.0214) (Δ vs real +0.0087)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.2313 (var=0.0107) (Δ vs real +0.0938)
  - iter 2: 0.1613 (var=0.0127) (Δ vs real +0.0237)
  - iter 3: 0.1487 (var=0.0098) (Δ vs real +0.0112)
  - iter 4: 0.1512 (var=0.0081) (Δ vs real +0.0137)
  - iter 5: 0.1650 (var=0.0104) (Δ vs real +0.0275)
  - iter 6: 0.1250 (var=0.0083) (Δ vs real -0.0125)
  - iter 7: 0.1350 (var=0.0130) (Δ vs real -0.0025)
  - iter 8: 0.1425 (var=0.0089) (Δ vs real +0.0050)
  - iter 9 (current): 0.1475 (var=0.0078) (Δ vs real +0.0100)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.0006 (var=0.0132) (Δ vs real -0.0206)
  - iter 2: -0.0025 (var=0.0080) (Δ vs real -0.0237)
  - iter 3: -0.0056 (var=0.0092) (Δ vs real -0.0269)
  - iter 4: -0.0106 (var=0.0067) (Δ vs real -0.0319)
  - iter 5: -0.0050 (var=0.0101) (Δ vs real -0.0262)
  - iter 6: 0.0187 (var=0.0080) (Δ vs real -0.0025)
  - iter 7: 0.0081 (var=0.0057) (Δ vs real -0.0131)
  - iter 8: 0.0331 (var=0.0067) (Δ vs real +0.0119)
  - iter 9 (current): -0.0075 (var=0.0097) (Δ vs real -0.0287)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.2202 (var=0.0025) (Δ vs real +0.0077)
  - iter 2: 0.2125 (var=0.0081) (Δ vs real +0.0000)
  - iter 3: 0.2046 (var=0.0069) (Δ vs real -0.0079)
  - iter 4: 0.2171 (var=0.0050) (Δ vs real +0.0046)
  - iter 5: 0.2104 (var=0.0050) (Δ vs real -0.0021)
  - iter 6: 0.1994 (var=0.0068) (Δ vs real -0.0131)
  - iter 7: 0.2162 (var=0.0058) (Δ vs real +0.0038)
  - iter 8: 0.2106 (var=0.0064) (Δ vs real -0.0019)
  - iter 9 (current): 0.1985 (var=0.0072) (Δ vs real -0.0140)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed load-adaptive dual-route family and achieves several important cross-experiment successes. It exactly matches the categorical signatures in Experiments 1–3, closely captures the inverse validity-gap effect in Experiment 6, preserves negligible arrangement, adjacency, run-embedding, and opponent-count effects in Experiments 9–17 and 19, and provides an excellent absolute fit to the critical 32-feature 80%-versus-82% conflict in Experiment 20 (0.220 versus 0.213). However, three central quantitative signatures remain miscalibrated. First, the 32-feature positional manipulation in Experiment 7 is essentially flat and slightly positive (0.005) rather than reliably negative (-0.054), so the late-override penalty is not actually expressed where the theory requires primacy. Second, high-validity overload is too strong in the 24-feature redundancy contrast of Experiment 8 (-0.421 versus -0.268), yet not strong enough in the count-six absolute-choice test of Experiment 18 (0.231 versus 0.138). This indicates that the overload function turns on too early across counts five and six rather than producing a sharper count-six cliff. Third, ordinary-bundle integration is somewhat too weak in Experiments 4 and 5: choices following the opposing weak-cue aggregate are underpredicted (0.091 versus 0.162 and 0.124 versus 0.161). Between-subject variability is also generally compressed, notably in Experiment 20 (variance 0.0025 versus 0.0153), despite the intended stable variation in route reliance.
Rationale: Keep the dual-route mechanism, but retune its gates. For the positional failure, preserve more residual anchor-route weight when a validity gap is conspicuous—e.g. reduce `conspicuous_route_switch`—and/or increase `late_override_threshold`, because the present conspicuousness switch appears to suppress the position-sensitive route before its late-position penalty can generate the observed negative gradient. Check that this leaves the already accurate small-gap absolute rate in Experiment 20 intact. For overload, move `overload_count_threshold` closer to six and make `overload_count_width` narrower, while retaining enough `high_validity_overload_strength` to lower the six-94% choice rate in Experiment 18. A sharper five-to-six transition should reduce the excessive average penalty over the five/six conditions in Experiment 8 while strengthening the count-six endpoint. Modestly increase `compressed_support_gain` for ordinary-validity bundles to raise the integrated-choice rates in Experiments 4 and 5; the high-validity overload gate should keep this from rescuing the extreme bundles. Finally, widen effective subject-level variation in anchor reliance or override threshold, rather than merely increasing response noise, so Experiment 20 retains its mean while acquiring more realistic heterogeneity. These are parameterization and gating adjustments within the prescribed family, not a change of mechanism.

**Outcome of your advice:** iter 1 candidate loss=0.0541 -> iter 2 candidate loss=0.0410 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed load-adaptive dual-route family and the accepted revision substantially improves aggregate fit (0.0541 to 0.0410). It now matches Experiment 20's critical 32-feature absolute choice rate exactly, brings Experiment 18's count-six high-validity choice rate close to the data, captures Experiment 6, and preserves the desired near-zero arrangement, adjacency, embedding, and opponent-count contrasts across Experiments 9–17 and 19. The sharper overload transition also moved Experiment 8 in the correct direction. However, the central positional prediction still fails: Experiment 7 is positive (0.0258) rather than negative (-0.0543), and it worsened from the prior iteration despite the previous recommendation to increase the late-override penalty. This reveals a sign-level calibration problem in the implemented position gate. In the code, a later opposing representative raises `required_drive`, necessarily making target-cue choice decline with later position; under the actual metric this generates the positive rather than observed negative contrast. Remaining quantitative misses include excessive high-validity overload in Experiment 8 (-0.376 versus -0.268), insufficient ordinary-bundle integration in Experiments 4 and 5 (0.096 versus 0.162 and 0.118 versus 0.161), and a modest spurious negative position effect at 24 features in Experiment 12 (-0.030 versus 0.015). Between-subject variability also remains compressed in several key experiments, although Experiment 20 improved substantially.
Rationale: Keep the accepted dual-route model, but correct the positional gate rather than further increasing `late_override_threshold`. The previous push in that direction was accepted overall but moved Experiment 7 farther from the target, so it should not be repeated. Reverse or reparameterize the position contribution in `required_drive`—for example, allow a small negative `late_override_threshold`, or define the position term using remaining rather than elapsed display distance—so that the simulated contrast has the empirically required negative sign. Calibrate this narrowly against Experiment 7 while retaining the exact Experiment 20 mean and the near-zero position/arrangement effects in Experiments 9, 10, 12, and 14. For overload, extend the successful sharper-cliff adjustment: move the count threshold slightly closer to six and narrow its width further, then tune strength at count six. This should retain Experiment 18's low absolute rate while reducing the excessive average penalty across Experiment 8's pooled count-five and count-six conditions. Use only a small increase in ordinary-validity bundle accessibility or support, since Experiment 4 improved but Experiment 5 did not; avoid a broad overload reduction that would undo Experiment 18. Finally, modest subject-level variation in override threshold or position sensitivity could raise heterogeneity without disturbing Experiment 20's now-correct pooled mean.

**Outcome of your advice:** iter 2 candidate loss=0.0410 -> iter 3 candidate loss=0.0418 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, but it was rejected by the accept gate (loss 0.0418 versus the accepted 0.0410), so it should not replace the iteration-2 base. The position-sign correction moved Experiment 7 in the intended direction, from +0.0258 to approximately zero, but still falls well short of the observed negative contrast (-0.0543). Moreover, the combined revision produced collateral regressions: Experiment 5 moved farther below the observed aggregate-override rate (0.1062 versus 0.1613), Experiment 13 acquired a spurious arrangement contrast (0.0226 versus -0.0030), Experiment 16 lost much of its previously good pair-versus-run contrast (0.0175 versus 0.0550), Experiment 17 moved away from its near-zero opponent-count effect, and Experiment 20 lost its previously exact absolute choice rate. The sharper overload change did improve Experiments 8 and 18 slightly, but not enough to offset these losses: Experiment 8 remains too negative (-0.3706 versus -0.2675), while Experiment 18 is already close (0.1487 versus 0.1375). Thus the main unresolved issue is not the mechanism family but the implementation of positional sensitivity: the current uncentered remaining-distance term changes both the positional slope and the average override threshold, thereby disturbing absolute rates and unrelated contrasts.
Rationale: Build from the unchanged accepted iteration-2 model, not from this rejected revision. Do not repeat the rejected joint push of further narrowing the overload cliff and increasing compressed support. Instead, make the position correction mean-preserving: replace the elapsed-position term with a centered remaining-distance term, such as position_slope * (mean_relative_position - relative_position), where the centering constant is fixed by the display or relevant target positions. Tune the slope to make Experiment 7 negative while leaving the average override threshold unchanged. This is especially suitable for Experiment 20, whose early, middle, and late target positions are roughly balanced, so a centered term can preserve its exact pooled rate while introducing the required gradient. If the gradient remains attenuated because conspicuous gaps suppress the anchor route, allow a modest residual anchor weight specifically through the existing conspicuousness gate rather than globally increasing anchor reliance. Retain iteration 2's overload and ordinary-support settings initially: the latest sharper-cliff/support changes were rejected and worsened Experiments 5, 13, 16, 17, and 20. After isolating the centered positional fix, only consider a very small adjustment to moderate-validity retrieval uncertainty—not another broad increase in compressed support—to raise Experiments 4 and 5 without weakening the successful high-validity overload fit in Experiment 18 or introducing arrangement dependence.

**Outcome of your advice:** iter 3 candidate loss=0.0418 -> iter 4 candidate loss=0.0536 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, but it was decisively rejected by the accept gate: loss rose from the accepted iteration-2 value of 0.0410 to 0.0536. The intended position correction still does not produce the central long-display signature. Experiment 7 remains essentially flat (0.0014) instead of negative (-0.0543), while the accepted base was +0.0258. Thus reversing the threshold gradient moved the prediction only partway and did not make positional sensitivity behaviorally effective. The likely problem is that centering the threshold is not equivalent to centering choice probability: the override sigmoid may be saturated, and the mixture with the validity route further attenuates the positional perturbation. Adding 0.5 to the remaining-distance term preserves an approximate mean threshold, but nonlinear transformation and route mixing need not preserve either the pooled override rate or the contrast. Experiment 20 remains close in mean (0.2171 versus 0.2125), but the revision provides no aggregate benefit and heterogeneity remains too low. Other notable current misses are under-integration in Experiments 4 and 5, excessive overload in Experiment 8, and regressions in Experiments 14–17; several of those occur where this position-only code change should have little or no deterministic effect, suggesting simulation variability rather than evidence for further broad mechanism changes. The accepted iteration-2 settings for overload and ordinary support should therefore remain untouched.
Rationale: Build on the unchanged iteration-2 base. The previous recommendation to use a threshold-centered remaining-distance term was rejected, so do not repeat that implementation or again alter overload and compressed-support parameters. Instead, make the positional correction saturation-aware within the same override mechanism. Apply the correctly signed position perturbation on the override logit and normalize its intercept so that the mean override probability—not merely the mean threshold—matches the position-free override probability across a generic 32-position display. Equivalently, jointly tune the position slope and a compensating override intercept after the sigmoid, using the smallest slope that moves Experiment 7 toward -0.054 while preserving Experiment 20's pooled rate near 0.213. Inspect whether all Experiment-7 conditions lie on a saturated portion of the override sigmoid; if so, shift the high-load override operating point or modestly widen its temperature before increasing the positional slope. Keep this adjustment gated by display load and, if needed, the existing conspicuous-gap signal so it affects the conspicuous long-display conflict in Experiment 7 without creating gradients in the small-gap Experiment 20 or the near-null position tests. Retain iteration-2 overload, retrieval, support, and route-mixture ranges initially: prior attempts to retune those were rejected and damaged Experiments 5, 13, 16, 17, and 20.

**Outcome of your advice:** iter 4 candidate loss=0.0536 -> iter 5 candidate loss=0.0465 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, and the post-sigmoid correction finally expresses the central 32-feature positional signature: Experiment 7 improves markedly from the accepted base's +0.0258 to -0.0432, close to the observed -0.0543, while Experiment 20 remains accurate in absolute choice rate (0.2104 versus 0.2125). Nevertheless, the candidate was rejected by the gate because aggregate loss rose from 0.0410 to 0.0465. The positional term is too broadly applied to every opposing recommendation in high-load displays. It consequently creates unwanted contrasts for multi-source recommendations, especially Experiment 10 (0.0343 versus -0.0045) and Experiment 11 (0.0050 versus -0.0225), and it moves Experiment 9 away from its near-null target. Regressions in Experiments 13, 16, and 17 are likely largely simulation variability because their displays receive no deterministic position modulation; they should not motivate broad parameter changes. Persistent base-level misses remain in Experiments 4 and 5, where ordinary bundles are under-integrated, and Experiment 8, where high-validity overload is excessive. However, prior attempts to retune support and overload were rejected, so those knobs should not be changed again in the same direction. Between-subject variances also remain too compressed, notably in Experiments 4, 7, 8, and 20.
Rationale: Return to the unchanged accepted iteration-2 base and do not retain the current ungated post-sigmoid position perturbation. The previous advice to introduce a generic saturation-aware perturbation was rejected overall, even though it successfully diagnosed the sign and operating scale for Experiment 7. Try a narrower, theoretically motivated in-family implementation: make positional override sensitivity decay continuously with the opposing recommendation's accumulated diagnostic support or source count. A singleton high-validity challenger, as in Experiment 7, should retain nearly the current positional slope, whereas a pair or larger compressed bundle should receive little positional modulation because its accumulated support already determines override. A smooth gate such as an exponential or sigmoid in excess count/support is preferable to an experiment-specific hard condition. This preserves the arbiter's count-and-diagnosticity principle and should keep the Experiment 7 gain while removing spurious terminal-position and redundancy effects in Experiments 10 and 11. Continue gating by display load and validity gap so the 24-feature nulls and the small-gap Experiment 20 mean remain intact. Keep iteration-2 overload, retrieval, support, route-mixture, and lapse ranges fixed for this test; previous joint changes to overload and ordinary support were rejected. Also avoid reacting to single-iteration fluctuations in Experiments 13, 16, and 17. If subject heterogeneity is adjusted, vary the new position-sensitivity gate across subjects rather than increasing lapse, since the latter would damage the already good absolute rates.

**Outcome of your advice:** iter 5 candidate loss=0.0465 -> iter 6 candidate loss=0.0494 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, but it was rejected by the accept gate: loss increased to 0.0494 relative to the iteration-2 running best of 0.0410. The newly support-shielded position correction therefore should not be retained. It produces a moderately negative Experiment-7 gradient (-0.0279), but this is weaker than the preceding rejected attempt and still only about half the observed effect (-0.0543). Meanwhile, aggregate fit deteriorates across several outcomes: ordinary-bundle integration remains too weak in Experiments 4 and 5 (0.0905 versus 0.1620 and 0.1106 versus 0.1613), high-validity overload remains excessive in Experiment 8 (-0.3831 versus -0.2675), and Experiment 20 slips below its critical absolute rate (0.1994 versus 0.2125). The candidate also shows incorrect contrasts in Experiments 11, 14, 15, 16, and 17, although some of these changes likely reflect simulation variability because the added deterministic correction should have little influence there. Important successes remain: Experiments 1 and 2 match categorically, Experiment 6 is excellent, Experiment 18 is close, Experiment 19 is accurate, and most arrangement effects remain small. However, between-subject heterogeneity is still substantially compressed in central cases such as Experiments 4, 7, 8, and 20.
Rationale: The previous recommendation to shield the post-sigmoid positional correction by count and diagnostic support was rejected, so do not repeat that strategy or add another specialized position gate. Build the next test on the unchanged iteration-2 base and leave its overload, anchor, and explicit position settings fixed. Try a different in-family knob that can address the largest shared residuals: modestly broaden the existing binding-uncertainty profile while keeping its center near the accepted value. For example, test a binding_uncertainty_width range around 0.11–0.17 rather than 0.08–0.13, with little or no change to peak uncertainty strength. A broader localized retrieval trough should reduce premature stopping on the 70–72% singleton recommendation and thereby raise aggregate-side choices in Experiments 4 and 5. It may also lower accessibility in Experiment 8's low-redundancy high-validity condition while leaving the already overloaded count-six condition comparatively unchanged, reducing the overly negative high-minus-low contrast without weakening the successful count-six cliff in Experiment 18. This is preferable to another increase in compressed support or another sharpening of overload, both of which were previously rejected. Test this retrieval-width change in isolation and use subject-level variation in that width or uncertainty strength to increase heterogeneity rather than increasing lapse. Preserve the iteration-2 fit to Experiment 20 and the inverse-gap signature in Experiment 6 as constraints; if broadening harms either, use a smaller intermediate width rather than changing mechanism family or returning to the repeatedly rejected positional corrections.

**Outcome of your advice:** iter 6 candidate loss=0.0494 -> iter 7 candidate loss=0.0449 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed load-adaptive dual-route family, but the isolated widening of binding uncertainty was rejected by the accept gate: loss is 0.0449 versus the iteration-2 running best of 0.0410. The change yields a modest improvement in Experiment 5 (0.1244 versus 0.1181 in the accepted base; observed 0.1613), but it does not generalize to Experiment 4, which worsens to 0.0910 versus 0.1620. It also aggravates the overload error in Experiment 8 (-0.3931 versus -0.2675), contrary to the intended rationale, while Experiment 18 remains excellent (0.1350 versus 0.1375). The central long-display positional signature remains wrong in Experiment 7: the contrast is positive at 0.0262 rather than negative at -0.0543. Other important successes are preserved, including Experiments 1–2, the near-null arrangement effects in Experiments 9–15, and the critical Experiment-20 absolute rate (0.2162 versus 0.2125). However, Experiment 16 also substantially underpredicts the pair-versus-run contrast (0.0037 versus 0.0550), and subject-level variance remains too low in Experiments 4, 7, 8, and 20. Overall, broadening the uncertainty trough is not an experiment-invariant solution: it slightly helps one moderate-validity bundle test while harming the other and worsening high-validity redundancy.
Rationale: Because this candidate was rejected, build from the unchanged iteration-2 base and restore its narrower binding-uncertainty width. My previous recommendation to broaden that width did not pass the gate and should not be repeated. Also avoid another explicit positional correction: several threshold-centered, post-sigmoid, and support-shielded variants have already been rejected despite occasionally improving Experiment 7.

Try a small isolated shift of the existing uncertainty profile rather than widening it. The current center of 0.60–0.67 in subjective diagnosticity lies above the diagnosticity of 70–72% cues (approximately 0.40–0.44 before compression). Moving the center modestly downward while retaining the accepted narrow width should selectively reduce premature retrieval of the 70–72% singleton or first recommendation in Experiments 4 and 5, without extending substantial uncertainty into the .94–.95 range and thereby worsening Experiment 8. A conservative intermediate range, rather than a large relocation, is warranted because Experiment 6 is already well fit and depends on the shape of this validity-sensitive retrieval function. Test this center adjustment alone, with Experiment 6, Experiment 18, and Experiment 20 treated as preservation constraints.

The resulting target is higher aggregate-side choice in both Experiments 4 and 5, no further increase in the magnitude of Experiment 8's negative contrast, and unchanged count-six overload in Experiment 18. Do not react to noisy single-round fluctuations in Experiments 13–17 by adding arrangement or opponent-count terms, which would violate the arbiter's invariance requirement. Subject heterogeneity can be revisited only after a pooled-fit improvement; increasing lapse would not solve the systematic directional residuals.

**Outcome of your advice:** iter 7 candidate loss=0.0449 -> iter 8 candidate loss=0.0550 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed load-adaptive dual-route family, but this iteration was rejected decisively: loss increased to 0.0550 from the iteration-2 running best of 0.0410. The downward shift of the binding-uncertainty center therefore is not an experiment-invariant improvement. It raises the underpredicted aggregate-side choices in Experiments 4 and 5 to 0.1040 and 0.1294, but both remain below the observed 0.1620 and 0.1613. More importantly, it worsens the high-validity redundancy contrast in Experiment 8 to -0.4031 versus -0.2675, despite preserving the count-six absolute rate in Experiment 18 (0.1425 versus 0.1375). The central long-display positional signature also remains wrong: Experiment 7 is positive at 0.0204 rather than negative at -0.0543. Several nominally null contrasts fluctuate away from the data, notably Experiments 14, 15, and 17, although these are likely simulation variation rather than evidence for arrangement or opponent-count mechanisms. Important successes remain in Experiments 1–2, 6, 9–13, 18–20, especially the critical Experiment-20 absolute rate of 0.2106 versus 0.2125. Subject-level variability is still compressed in Experiments 4, 7, 8, and 20. Overall, the uncertainty-center relocation trades small gains in moderate-validity integration for larger losses elsewhere and should not replace the accepted base.
Rationale: Build from the unchanged iteration-2 base and restore its accepted binding-uncertainty center. My previous recommendation to shift that center downward was rejected, so do not repeat it, broaden the uncertainty profile again, or make another concurrent support/overload adjustment. Several direct modifications of the override-position threshold have also been rejected despite occasionally improving Experiment 7, so try a different locus within the same dual-route mechanism: let relative position modulate route arbitration rather than the override sigmoid itself. Specifically, apply a small, mean-centered position term to the logit of `anchor_weight`, gated by display load and the existing conspicuous-gap function. For a conspicuous-gap challenger encountered later, anchor-route weight should decrease slightly, allowing the validity route—which favors that challenger—to produce the required negative Experiment-7 contrast. Center the modulation so average anchor weight across display positions is preserved. Because Experiment 20 has only a .02 validity gap, its conspicuous-gap gate should be near zero, protecting its accurately fitted pooled choice rate; the load gate should preserve the 24-feature nulls in Experiments 12 and 14. This differs from the rejected attempts that perturbed the override threshold or post-sigmoid override probability. Test this route-weight adjustment in isolation, with Experiments 7 and 20 as the primary slope-and-intercept constraints and Experiments 9–12 as null-effect checks. Retain iteration-2 retrieval, overload, compressed-support, lapse, and override ranges for this test. Only after a pooled-fit improvement should heterogeneity or the residual Experiment-8 overload error be revisited.

**Outcome of your advice:** iter 8 candidate loss=0.0550 -> CURRENT candidate loss=0.0516 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed load-adaptive dual-route family, but it was rejected by the accept gate: loss 0.0516 is substantially worse than the iteration-2 running best of 0.0410. The route-arbitration position term recommended in the previous critique did not solve the central positional failure. Experiment 7 remains positive (0.0134) rather than negative (-0.0543), while Experiment 20 has slipped from an exact fit in the accepted base to 0.1985 versus 0.2125. Thus the new route-position slope is being outweighed by the existing positive elapsed-position penalty in the override threshold, and it adds parameter variation without delivering the required sign. Other systematic base-level errors remain: ordinary aggregate evidence is underweighted in Experiments 4 and 5 (0.0955 and 0.1031 versus 0.1620 and 0.1613), and high-validity overload is excessive in Experiment 8 (-0.3900 versus -0.2675), even though the count-six absolute overload result in Experiment 18 is close (0.1475 versus 0.1375). The current round also misses Experiments 6, 15, 16, and 19, but their unstable trajectories suggest that several of these are simulation fluctuations rather than grounds for adding arrangement or opponent-count mechanisms. The exact categorical fits in Experiments 1\u20133 and near-null structural contrasts in Experiments 9, 11, 13, 14, and 17 remain important successes.",
  "verdict": "regenerate",
  "rationale": "Build again from the unchanged iteration-2 base. My previous recommendation to place the position effect on the logit of route weight was rejected, so do not repeat it, increase `route_position_slope`, or add another specialized position gate. The sequence of rejected threshold-, override-probability-, support-shielded-, and route-weight corrections indicates that further direct positional elaboration is unlikely to beat the accepted base.\n\nTry a different in-family calibration aimed at the shared Experiments 4, 5, and 8 residual: add a modest validity-route decision temperature or confidence transform after the accessibility cascade, while leaving the anchor route unchanged. The transform should be strongest when neither recommendation has crossed the high-validity count-overload gate and should vanish for the overloaded count-six regime. Softening the overly decisive ordinary validity route would raise the currently losing aggregate side in Experiments 4 and 5. In Experiment 8 it should lower the highly favored low-redundancy baseline while leaving the already overloaded count-five/six conditions largely intact, thereby shrinking the excessively negative redundancy contrast without weakening the count-six suppression needed for Experiment 18. This is a normalization/temperature adjustment within the prescribed dual-route family, not a new evidence-integration mechanism.\n\nTest this change in isolation with the accepted retrieval, uncertainty, support, overload, anchor, and lapse ranges. Keep the temperature close to one and constrain it against Experiments 1 and 3 so the lexicographic signatures survive, against Experiment 6 so its inverse-gap effect is preserved, and against Experiment 18 so its low absolute choice rate does not rise. Preserve iteration 2's exact Experiment-20 mean by making the temperature apply only to the validity route rather than changing global lapse or anchor reliance. Do not add adjacency, run, dispersion, or opponent-count terms, and defer heterogeneity tuning until pooled fit improves."
}
```

## Usage

```json
{
  "input_tokens": 60292,
  "output_tokens": 2187
}
```
