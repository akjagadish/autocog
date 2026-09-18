# feedback_iter_06

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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with a Resource-Rational Selective Retrieval theory. The new theory should posit one capacity-limited retrieval-and-accumulation process rather than pi_6's special coherent-coalition reopening or pi_5's three explicitly gated formats. Cue retrieval priority should combine communicated validity, display recency, and unresolved conflict. Capacity and stopping should adapt continuously to diagnostic-cue load and to the distinctiveness of the validity structure: diffuse structures encourage approximate pooling, whereas a clearly dominant validity tier encourages compression onto that tier under capacity pressure. These should be emergent regimes of a common process, not hard strategy switches.

Crucially, late dissent should receive strong attentional priority, but adjacency should not intrinsically amplify dissent. A dispersed set containing a very late opposing cue can therefore be more disruptive than an equally centered adjacent block, accounting for Experiments 1 and 10. Absolute validity and recency sensitivity should vary continuously and partly independently across subjects, allowing the balanced directional result and high variance in Experiment 2. The theory should also include stable subject variation in whether tied cues are pooled or represented by an early retrieved member, preserving the heterogeneity seen in Experiment 7. It must retain pi_5's successful emergent signatures: majority pooling in Experiments 3-5, validity-led behavior under suitable overload structures in Experiment 6, and strong positional retrieval effects in Experiments 8-9. The resulting competitor should specifically test whether apparent load-indexed formats are discrete arbitration states, as pi_5 claims, or continuous consequences of capacity, retrieval competition, validity distinctiveness, and adaptive stopping.


## CANDIDATE THEORY
Adaptive Retrieval Budget theory: choice is produced by a single resource-rational retrieval-and-accumulation process. Every diagnostic cue competes for a limited retrieval budget, with priority jointly determined by communicated validity, display recency, and unresolved opposition to the provisional majority. Resource growth depends continuously on validity structure: diffuse structures receive more load-related capacity and support approximate pooling, whereas distinctive leading tiers remain capacity-compressed and receive stronger validity-weighted evidence. In genuinely dominant structures, compression also protects the leading representation by attenuating ordinary recency and conflict competition, not merely extreme endpoint alerts. Endpoint conflict priority is continuously scaled by unresolved conflict, diagnostic load, and provisional certainty. When several dissenters remain in a diffuse high-load display, a fixed share of retrieval mass is redistributed toward the latest dissenter rather than globally increasing attention or reducing the retrieval budget. Dominant-tier compression protects validity-led representations from this redistribution. Subjects vary independently in validity and recency sensitivity and continuously in tied-tier pooling. Low pooling conserves a tied tier's retrieval mass, concentrates it on its first displayed member, and modestly increases that member's within-budget competitive concentration; high pooling distributes the mass across the tier. These are emergent regimes of one retrieval process rather than discrete strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Adaptive Retrieval Budget theory expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    if n_features > 1:
        position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        position = np.zeros(n_features, dtype=float)

    clipped_v = np.clip(validities, 0.500001, 0.999999)
    validity_log_odds = np.log(clipped_v / (1.0 - clipped_v))
    active_v = validities[active]
    active_log_odds = validity_log_odds[active]

    # Distinctiveness requires an exceptional leading-tier gap. In addition
    # to the global structure, assess whether the currently diagnostic cues
    # expose a leading gap that is exceptional relative to their other gaps.
    ordered_global = np.sort(np.unique(np.round(validities, 12)))[::-1]
    if ordered_global.size <= 1:
        global_distinctiveness = 0.0
        ordinary_global_gap = 0.0
    else:
        global_gaps = ordered_global[:-1] - ordered_global[1:]
        leading_gap = float(global_gaps[0])
        ordinary_global_gap = float(np.median(global_gaps))
        dominance_ratio = leading_gap / (
            leading_gap + ordinary_global_gap + 1e-12
        )
        global_distinctiveness = float(np.clip(
            (dominance_ratio - 0.50) / 0.35,
            0.0,
            1.0,
        ))

    ordered_active = np.sort(np.unique(np.round(active_v, 12)))[::-1]
    if ordered_active.size <= 1:
        active_distinctiveness = 0.0
    else:
        active_gaps = ordered_active[:-1] - ordered_active[1:]
        active_leading_gap = float(active_gaps[0])
        active_reference_gap = float(np.median(active_gaps))
        if ordinary_global_gap > 0.0:
            active_reference_gap = max(
                ordinary_global_gap,
                min(active_reference_gap, 2.0 * ordinary_global_gap),
            )
        active_ratio = active_leading_gap / (
            active_leading_gap + active_reference_gap + 1e-12
        )
        active_distinctiveness = float(np.clip(
            (active_ratio - 0.55) / 0.35,
            0.0,
            1.0,
        ))

    distinctiveness = max(
        global_distinctiveness,
        0.75 * active_distinctiveness,
    )

    positive_fraction = float(np.mean(signs[active] > 0.0))
    conflict_entropy = 4.0 * positive_fraction * (1.0 - positive_fraction)

    # Diffuse validity structures rationally receive more load-related
    # capacity, while distinctive structures retain compression pressure.
    diffuse_capacity_multiplier = 1.0 + 0.70 * (1.0 - distinctiveness)
    capacity = (
        float(parameters["base_capacity"])
        + float(parameters["load_capacity_adaptation"])
        * diffuse_capacity_multiplier * np.log1p(load)
        + 0.45 * float(parameters["load_capacity_adaptation"])
        * diffuse_capacity_multiplier
        * conflict_entropy * np.sqrt(load)
    )
    stopping_requirement = float(parameters["stopping_threshold"]) * (
        1.0
        + float(parameters["stopping_adaptation"])
        * conflict_entropy
        * (1.0 - distinctiveness)
    )
    retrieval_dose = max(0.05, capacity * stopping_requirement)

    # A broader pressure transition prevents an abrupt load-indexed crossover.
    pressure_x = (load - capacity) / 1.0
    pressure_x = float(np.clip(pressure_x, -40.0, 40.0))
    pressure = 1.0 / (1.0 + np.exp(-pressure_x))
    compression = pressure * distinctiveness

    if np.std(active_log_odds) > 1e-12:
        relative_validity = (
            active_log_odds - float(np.mean(active_log_odds))
        ) / float(np.std(active_log_odds))
    else:
        relative_validity = np.zeros(active.size, dtype=float)

    # A cheap provisional sign estimate identifies unresolved opposition. It
    # is only an orienting signal: final evidence comes from retrieval.
    tally_sum = float(np.sum(signs[active]))
    provisional_certainty = abs(tally_sum) / max(load, 1.0)
    provisional_sign = float(np.sign(tally_sum))
    if provisional_sign == 0.0:
        provisional_sign = float(np.sign(np.dot(
            signs[active], 1.0 + 0.15 * active_log_odds
        )))

    unresolved = np.zeros(active.size, dtype=float)
    if provisional_sign != 0.0:
        unresolved = (signs[active] == -provisional_sign).astype(float)

    late_conflict = unresolved * np.power(
        position[active], float(parameters["conflict_lateness_power"])
    )

    # Endpoint sensitivity is modulated continuously by unresolved-conflict
    # mass, load, and provisional certainty. Single dissenters are attenuated;
    # several dissenters under high load sharpen competition. Exact tally ties
    # receive only a weak alert because there is no secure commitment to undo.
    unresolved_positions = position[active][unresolved > 0.0]
    unresolved_count = float(np.sum(unresolved))
    if unresolved_positions.size > 0:
        latest_unresolved = float(np.max(unresolved_positions))
        alert_x = np.clip((latest_unresolved - 0.80) / 0.035, -40.0, 40.0)
        endpoint_alert = 1.0 / (1.0 + np.exp(-alert_x))

        mass_x = np.clip((unresolved_count - 2.2) / 0.35, -40.0, 40.0)
        conflict_mass_gate = 1.0 / (1.0 + np.exp(-mass_x))
        load_x = np.clip((load - 5.5) / 1.0, -40.0, 40.0)
        load_gate = 1.0 / (1.0 + np.exp(-load_x))
        certainty_gate = 0.25 + 0.75 * min(
            1.0, 4.0 * provisional_certainty
        )
        alert_multiplier = (
            0.20
            + 1.50 * conflict_mass_gate * (0.70 + 0.30 * load_gate)
        ) * certainty_gate
        late_alert = endpoint_alert * alert_multiplier
    else:
        latest_unresolved = 0.0
        endpoint_alert = 0.0
        conflict_mass_gate = 0.0
        load_gate = 0.0
        certainty_gate = 0.25
        late_alert = 0.0

    # A compressed dominant tier protects the provisional representation from
    # extreme endpoint modulation without removing ordinary recency effects.
    late_alert *= max(0.20, 1.0 - 0.75 * compression)

    centered_absolute_validity = active_log_odds - np.log(3.0)
    ordinary_competition_protection = max(0.12, 1.0 - 0.88 * compression)
    log_priority = (
        float(parameters["validity_access"]) * centered_absolute_validity
        + float(parameters["distinctiveness_gain"])
        * compression * relative_validity
        + ordinary_competition_protection
        * float(parameters["recency_access"]) * position[active]
        + ordinary_competition_protection
        * float(parameters["conflict_priority"]) * late_conflict
    )

    # Stable person variation in tied-tier representation remains continuous.
    # Allocation is performed after exponentiation so retrieval mass removed
    # from later tied members is transferred to that tier's first member rather
    # than leaking to unrelated lower-validity tiers.
    tie_pooling = float(parameters["tie_pooling"])
    continuation_x = np.clip(18.0 * (tie_pooling - 0.82), -40.0, 40.0)
    continuation = 1.0 / (1.0 + np.exp(-continuation_x))
    continuation = 0.002 + 0.996 * continuation

    # Extreme late dissent sharpens the same finite-budget competition and
    # contracts retrieval breadth; no reopening or adjacency process is added.
    priority_center = float(np.mean(log_priority))
    priority_sharpening = (
        1.0 + float(parameters["late_competition_sharpening"]) * late_alert
    )
    log_priority = priority_center + priority_sharpening * (
        log_priority - priority_center
    )
    retrieval_dose *= max(
        0.08,
        1.0 - float(parameters["late_budget_contraction"]) * late_alert,
    )

    log_priority -= np.max(log_priority)
    priority = np.exp(log_priority)

    # Conserve each tied tier's competitive mass while continuously
    # reallocating it between first-member representation and whole-tier use.
    for tier_value in np.unique(np.round(active_v, 12)):
        tier_local = np.flatnonzero(np.isclose(
            active_v, tier_value, rtol=1e-9, atol=1e-12
        ))
        if tier_local.size <= 1:
            continue
        first_local = int(tier_local[np.argmin(active[tier_local])])
        tier_mass = float(np.sum(priority[tier_local]))
        priority[tier_local] *= continuation
        priority[first_local] += (1.0 - continuation) * tier_mass
        priority[first_local] *= 1.0 + 0.85 * (1.0 - continuation)

    priority_sum = float(priority.sum())
    if not np.isfinite(priority_sum) or priority_sum <= 0.0:
        priority = np.ones(active.size, dtype=float) / float(active.size)
    else:
        priority /= priority_sum

    # With several unresolved cues, redistribute rather than add a bounded
    # share of the existing retrieval mass toward the latest dissent. This is
    # negligible for a single dissenter and is suppressed by tier compression.
    if unresolved_count > 0.0:
        unresolved_local = np.flatnonzero(unresolved > 0.0)
        conflict_logits = 24.0 * position[active][unresolved_local]
        conflict_logits -= np.max(conflict_logits)
        conflict_weights = np.exp(conflict_logits)
        conflict_weights /= conflict_weights.sum()
        conflict_allocation = np.zeros(active.size, dtype=float)
        conflict_allocation[unresolved_local] = conflict_weights

        redistribution = (
            0.58
            * conflict_mass_gate
            * load_gate
            * endpoint_alert
            * certainty_gate
            * (1.0 - compression)
        )
        redistribution = float(np.clip(redistribution, 0.0, 0.58))
        priority = (
            (1.0 - redistribution) * priority
            + redistribution * conflict_allocation
        )
        priority /= float(priority.sum())

    retrieval_probability = 1.0 - np.exp(
        -retrieval_dose * float(active.size) * priority
    )
    retrieval_probability = np.clip(retrieval_probability, 0.0, 1.0)

    bounded_absolute = np.tanh(active_log_odds / 2.5)
    bounded_relative = np.tanh(relative_validity / 1.5)
    evidence_strength = (
        1.0
        + float(parameters["absolute_validity_weight"]) * bounded_absolute
        + 0.95 * float(parameters["distinctiveness_gain"])
        * compression * bounded_relative
    )
    evidence_strength = np.clip(evidence_strength, 0.08, 4.0)

    contributions = retrieval_probability * evidence_strength
    contribution_mass = float(np.sum(contributions))
    if not np.isfinite(contribution_mass) or contribution_mass <= 1e-12:
        return np.array([0.5, 0.5], dtype=float)

    accumulated_evidence = float(
        np.dot(contributions, signs[active]) / contribution_mass
    )

    decision_logit = float(parameters["decision_gain"]) * accumulated_evidence
    decision_logit = float(np.clip(decision_logit, -60.0, 60.0))
    if decision_logit >= 0.0:
        z = np.exp(-decision_logit)
        p_a_core = 1.0 / (1.0 + z)
    else:
        z = np.exp(decision_logit)
        p_a_core = z / (1.0 + z)

    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a_core + 0.5 * epsilon
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


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
- validities: validities
- base_capacity: [1.8, 2.8]
- load_capacity_adaptation: [0.35, 0.8]
- validity_access: [0.0, 0.9]
- recency_access: [0.0, 9.0]
- conflict_priority: [5.0, 10.0]
- conflict_lateness_power: [6.0, 11.0]
- distinctiveness_gain: [2.5, 5.0]
- stopping_threshold: [0.75, 1.25]
- stopping_adaptation: [0.15, 0.65]
- tie_pooling: [0, 1]
- late_competition_sharpening: [2.0, 4.5]
- late_budget_contraction: [0.70, 0.90]
- absolute_validity_weight: [0.0, 0.25]
- decision_gain: [4.0, 7.0]
- epsilon: [0, 0.06]

`rationale`:
This is a minimal edit to the accepted iteration-6 model. The main change introduces a smooth compression-dependent protection factor on ordinary recency and conflict priority. Thus, when the validity structure supplies a genuinely dominant tier, the same capacity pressure that compresses retrieval also prevents broad positional competition from displacing that tier. This directly targets Experiment 1's excessive negative contrast and Experiment 4's insufficient validity-led overload behavior, while leaving diffuse structures—and therefore the strong Experiment 8 fit—largely unchanged. Diffuse capacity growth and conflict-sensitive capacity receive small increases to broaden high-load retrieval and reduce Experiment 3's excessive crossover without imposing a strategy gate. The already accepted mass-conserving redistribution mechanism is strengthened cautiously from 0.45 to 0.58 and its unresolved-cue allocation is made more endpoint-selective, targeting Experiment 9 while remaining negligible for a single dissenter and being suppressed by dominant-tier compression. Finally, the continuous tie-pooling transition is shifted only slightly, and low-pooling representations receive modest extra first-member concentration before the existing saturating retrieval transform. This continues the accepted improvement in Experiment 5 without reinstating the rejected contribution-multiplicity mechanism. Experiment 10 is not directly optimized because its pooled sign metric is discontinuous and the current model already produces substantial subject heterogeneity.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3446 -> ACCEPTED
- iter 2: loss=0.2701 -> ACCEPTED
- iter 3: loss=0.2590 -> ACCEPTED
- iter 4: loss=0.2647 -> REJECTED
- iter 5: loss=0.2669 -> REJECTED
- iter 6: loss=0.2497 -> ACCEPTED
- iter 7 (current candidate you are grading): loss=0.2517 -> REJECTED
Running-best (last accepted) base: iter 6 at loss=0.2497.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast top-cue adherence between tally-congruent and
    # tally-conflicting trials, matching on absolute tally margin.
    cells = {(d, c): [] for d in (2, 4, 6) for c in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the highest-validity cue in this experiment.
        if a[0] > b[0]:
            top_choice = 0
            top_relative_margin = int(np.sum(a > b) - np.sum(b > a))
        elif b[0] > a[0]:
            top_choice = 1
            top_relative_margin = int(np.sum(b > a) - np.sum(a > b))
        else:
            continue

        d = abs(top_relative_margin)
        if d not in (2, 4, 6) or top_relative_margin == 0:
            continue

        congruence = 1 if top_relative_margin > 0 else -1
        top_adherence = 1.0 if response == top_choice else -1.0
        cells[(d, congruence)].append(top_adherence)

    matched_contrasts = []
    for d in (2, 4, 6):
        congruent = cells[(d, 1)]
        conflicting = cells[(d, -1)]
        if len(congruent) == 0 or len(conflicting) == 0:
            continue
        # Division by two keeps the metric in [-1, 1].
        matched_contrasts.append(
            0.5 * (float(np.mean(congruent)) - float(np.mean(conflicting)))
        )

    if len(matched_contrasts) == 0:
        return float("nan")
    return float(np.mean(matched_contrasts))
```

**Observed (real) value:** -0.0300 (var=0.0056)
**Candidate trajectory (this loop):**
  - iter 1: 0.0306 (var=0.0797) (Δ vs real +0.0606)
  - iter 2: -0.2133 (var=0.0225) (Δ vs real -0.1833)
  - iter 3: -0.2833 (var=0.0171) (Δ vs real -0.2533)
  - iter 4: -0.0922 (var=0.0090) (Δ vs real -0.0622)
  - iter 5: -0.3394 (var=0.0250) (Δ vs real -0.3094)
  - iter 6: -0.3072 (var=0.0234) (Δ vs real -0.2772)
  - iter 7 (current): -0.2583 (var=0.0470) (Δ vs real -0.2283)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0167 (var=0.0052)
- pi_2: 0.8033 (var=0.0243)
- pi_3: -0.0028 (var=0.0068)
- pi_4: 0.0328 (var=0.0015)
- pi_5: 0.0367 (var=0.0093)
- pi_6: 0.6333 (var=0.0110)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # +1 denotes an A prediction and -1 a B prediction.
        tally_margin = np.sum(np.sign(d))
        if tally_margin == 0:
            continue
        tally_direction = 1.0 if tally_margin > 0 else -1.0

        # Validities decrease with feature index in this experiment.
        discriminating = np.flatnonzero(d != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_direction = 1.0 if d[first] > 0 else -1.0

        # Agreement trials cannot distinguish the theories.
        if tally_direction == ttb_direction:
            continue

        observed_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(observed_direction * tally_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.6683 (var=0.0367)
**Candidate trajectory (this loop):**
  - iter 1: 0.4192 (var=0.0534) (Δ vs real -0.2492)
  - iter 2: 0.7517 (var=0.0352) (Δ vs real +0.0833)
  - iter 3: 0.6150 (var=0.0555) (Δ vs real -0.0533)
  - iter 4: 0.7542 (var=0.0330) (Δ vs real +0.0858)
  - iter 5: 0.5733 (var=0.0734) (Δ vs real -0.0950)
  - iter 6: 0.6500 (var=0.0478) (Δ vs real -0.0183)
  - iter 7 (current): 0.5967 (var=0.0470) (Δ vs real -0.0717)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7008 (var=0.0514)
- pi_1: -0.6542 (var=0.0564)
- pi_3: 0.6600 (var=0.0188)
- pi_4: 0.5833 (var=0.0114)
- pi_5: 0.6467 (var=0.0132)
- pi_6: 0.9250 (var=0.0079)

### Experiment 3
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Return a parity-matched low-to-high load crossover index."""
    if data is None or len(data) == 0:
        return float("nan")

    loads = []
    majority_consistent = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        loads.append(int(np.count_nonzero(diff)))

        tally_margin = float(np.sum(np.sign(diff)))
        if tally_margin > 0:
            majority_response = 0
        elif tally_margin < 0:
            majority_response = 1
        else:
            majority_response = -1

        majority_consistent.append(
            float(int(response) == majority_response)
            if majority_response >= 0 else np.nan
        )

    work = pd.DataFrame({
        "load": np.asarray(loads, dtype=int),
        "majority_consistent": np.asarray(majority_consistent, dtype=float),
    })

    p = {}
    for load in (3, 4, 6, 7, 8):
        values = work.loc[
            work["load"] == load, "majority_consistent"
        ].dropna()
        if len(values) == 0:
            return float("nan")
        p[load] = float(values.mean())

    odd_crossover = p[3] - p[7]
    even_crossover = p[4] - 0.5 * (p[6] + p[8])
    return float(0.5 * (odd_crossover + even_crossover))

```

**Observed (real) value:** 0.3612 (var=0.0106)
**Candidate trajectory (this loop):**
  - iter 1: 0.1797 (var=0.0115) (Δ vs real -0.1816)
  - iter 2: 0.6006 (var=0.0104) (Δ vs real +0.2394)
  - iter 3: 0.5803 (var=0.0051) (Δ vs real +0.2191)
  - iter 4: 0.6578 (var=0.0053) (Δ vs real +0.2966)
  - iter 5: 0.5966 (var=0.0064) (Δ vs real +0.2353)
  - iter 6: 0.5809 (var=0.0048) (Δ vs real +0.2197)
  - iter 7 (current): 0.6038 (var=0.0100) (Δ vs real +0.2425)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6512 (var=0.0114)
- pi_2: 0.0066 (var=0.0037)
- pi_1: 0.0041 (var=0.0094)
- pi_4: 0.6541 (var=0.0103)
- pi_5: 0.5350 (var=0.0071)
- pi_6: 0.2469 (var=0.0084)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of validity-leading-cue-consistent choices on overload trials."""
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        discriminating = np.flatnonzero(differences != 0)

        # Focus on the 6- and 8-cue overload conditions.
        if discriminating.size < 6:
            continue

        # Validities are strictly descending by feature index, so the first
        # discriminating index is the highest-validity diagnostic cue.
        leading_cue = int(discriminating[0])
        lex_response = 0 if differences[leading_cue] > 0 else 1
        scores.append(float(int(row["response"]) == lex_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8756 (var=0.0078)
**Candidate trajectory (this loop):**
  - iter 1: 0.6659 (var=0.0058) (Δ vs real -0.2097)
  - iter 2: 0.7597 (var=0.0061) (Δ vs real -0.1159)
  - iter 3: 0.7369 (var=0.0047) (Δ vs real -0.1388)
  - iter 4: 0.7475 (var=0.0040) (Δ vs real -0.1281)
  - iter 5: 0.7428 (var=0.0037) (Δ vs real -0.1328)
  - iter 6: 0.7119 (var=0.0047) (Δ vs real -0.1637)
  - iter 7 (current): 0.7397 (var=0.0036) (Δ vs real -0.1359)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5031 (var=0.0037)
- pi_3: 0.8291 (var=0.0019)
- pi_1: 0.8391 (var=0.0151)
- pi_4: 0.8331 (var=0.0025)
- pi_5: 0.8884 (var=0.0019)
- pi_6: 0.6575 (var=0.0122)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_classification(df):
        aligned = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = np.sign(a - b)

            # Select trials where Experts 2 and 3 agree with each other but
            # oppose Expert 1: a 2-to-1 conflict within the top-validity tier.
            if d.size >= 3 and d[0] != 0 and d[1] == d[2] and d[1] == -d[0]:
                response = int(row['response'])
                chose_a = 1.0 if response == 0 else -1.0
                aligned.append(float(chose_a * d[0] > 0))

        if len(aligned) == 0:
            return np.nan
        rate = float(np.mean(aligned))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    values = []
    for _, subj_df in data.groupby('subject_id', sort=False):
        value = subject_classification(subj_df)
        if np.isfinite(value):
            values.append(value)

    if len(values) == 0:
        return float('nan')
    return float(np.mean(values))
```

**Observed (real) value:** 0.5000 (var=0.2000)
**Candidate trajectory (this loop):**
  - iter 1: 0.1400 (var=0.1004) (Δ vs real -0.3600)
  - iter 2: 0.0700 (var=0.0501) (Δ vs real -0.4300)
  - iter 3: 0.1000 (var=0.0500) (Δ vs real -0.4000)
  - iter 4: 0.1900 (var=0.1089) (Δ vs real -0.3100)
  - iter 5: 0.1200 (var=0.0556) (Δ vs real -0.3800)
  - iter 6: 0.2100 (var=0.1109) (Δ vs real -0.2900)
  - iter 7 (current): 0.2600 (var=0.1124) (Δ vs real -0.2400)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.5800 (var=0.2436)
- pi_6: 0.9200 (var=0.0636)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast majority-choice rates across the four positions of the
    # second cue opposing the majority. Coefficients sum to zero, so any
    # profile that is flat across positions has expected metric zero.
    weights = {1: -0.24, 3: -0.05, 5: 0.10, 7: 0.19}
    outcomes = {1: [], 3: [], 5: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0)
        if active.size == 0:
            continue

        majority_sign = np.sign(np.sum(signs[active]))
        if majority_sign == 0:
            continue

        # Expert 1 (array index 0) is the highest-validity cue. The target
        # position is the other active cue pointing in the same direction.
        top_sign = signs[0]
        opposing = [int(j) for j in active if j != 0 and signs[j] == top_sign]
        if len(opposing) != 1 or opposing[0] not in outcomes:
            continue

        response = int(row["response"])
        chose_majority = ((response == 0 and majority_sign > 0) or
                          (response == 1 and majority_sign < 0))
        outcomes[opposing[0]].append(float(chose_majority))

    if any(len(outcomes[j]) == 0 for j in outcomes):
        return float("nan")

    rates = {j: float(np.mean(outcomes[j])) for j in outcomes}
    return float(sum(weights[j] * rates[j] for j in weights))

```

**Observed (real) value:** -0.1132 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: -0.0750 (var=0.0029) (Δ vs real +0.0381)
  - iter 2: -0.1844 (var=0.0014) (Δ vs real -0.0712)
  - iter 3: -0.1707 (var=0.0016) (Δ vs real -0.0576)
  - iter 4: -0.1751 (var=0.0015) (Δ vs real -0.0619)
  - iter 5: -0.1670 (var=0.0012) (Δ vs real -0.0539)
  - iter 6: -0.1686 (var=0.0018) (Δ vs real -0.0554)
  - iter 7 (current): -0.1662 (var=0.0013) (Δ vs real -0.0530)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0970 (var=0.0005)
- pi_3: 0.0009 (var=0.0007)
- pi_1: 0.0003 (var=0.0005)
- pi_2: 0.0004 (var=0.0004)
- pi_5: -0.1079 (var=0.0009)
- pi_6: -0.0042 (var=0.0027)

### Experiment 7
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agreements = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        active = np.flatnonzero(diff != 0.0)

        # Restrict to the six-discriminating-cue trials, where the theories'
        # predictions are maximally separated and the competitor has not yet
        # entered its person-variable high-load transition.
        if active.size != 6:
            continue

        # Validities strictly decrease with display position, so the first
        # active feature is the validity-ranked lexicographic cue.
        first = int(active[0])
        lex_response = 0 if diff[first] > 0.0 else 1
        agreements.append(float(int(row["response"]) == lex_response))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1175 (var=0.0081)
**Candidate trajectory (this loop):**
  - iter 1: 0.3837 (var=0.0238) (Δ vs real +0.2662)
  - iter 2: 0.0675 (var=0.0051) (Δ vs real -0.0500)
  - iter 3: 0.1200 (var=0.0108) (Δ vs real +0.0025)
  - iter 4: 0.0794 (var=0.0077) (Δ vs real -0.0381)
  - iter 5: 0.1288 (var=0.0139) (Δ vs real +0.0113)
  - iter 6: 0.1212 (var=0.0103) (Δ vs real +0.0038)
  - iter 7 (current): 0.1237 (var=0.0078) (Δ vs real +0.0063)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7987 (var=0.0067)
- pi_5: 0.0850 (var=0.0021)
- pi_1: 0.8706 (var=0.0097)
- pi_2: 0.1344 (var=0.0103)
- pi_4: 0.9775 (var=0.0008)
- pi_6: 0.1588 (var=0.0166)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    unanimous = []
    late_loser = []

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        signs = np.sign(a - b)
        active = np.flatnonzero(signs != 0.0)
        if active.size == 0:
            continue

        margin = float(np.sum(signs[active]))
        if margin == 0.0:
            continue

        # Reorient every trial toward its equal-weight-majority winner, thereby
        # combining A/B-reversed pairs and eliminating response-side direction.
        winner_sign = 1.0 if margin > 0.0 else -1.0
        relative_signs = signs[active] * winner_sign
        winner_response = 0 if winner_sign > 0.0 else 1
        followed_winner = float(int(row.response) == winner_response)

        if np.all(relative_signs > 0.0):
            unanimous.append(followed_winner)
        elif np.sum(relative_signs < 0.0) == 1:
            losing_active_position = int(active[np.flatnonzero(relative_signs < 0.0)[0]])
            if losing_active_position == int(np.max(active)):
                late_loser.append(followed_winner)

    if len(unanimous) == 0 or len(late_loser) == 0:
        return 0.0

    return float(np.mean(unanimous) - np.mean(late_loser))
```

**Observed (real) value:** 0.7750 (var=0.0279)
**Candidate trajectory (this loop):**
  - iter 1: 0.6387 (var=0.0279) (Δ vs real -0.1363)
  - iter 2: 0.9225 (var=0.0106) (Δ vs real +0.1475)
  - iter 3: 0.7900 (var=0.0291) (Δ vs real +0.0150)
  - iter 4: 0.8869 (var=0.0150) (Δ vs real +0.1119)
  - iter 5: 0.7956 (var=0.0352) (Δ vs real +0.0206)
  - iter 6: 0.7800 (var=0.0275) (Δ vs real +0.0050)
  - iter 7 (current): 0.7756 (var=0.0272) (Δ vs real +0.0006)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3750 (var=0.0124)
- pi_3: -0.0169 (var=0.0083)
- pi_1: -0.0088 (var=0.0039)
- pi_2: 0.0231 (var=0.0063)
- pi_4: 0.0719 (var=0.0047)
- pi_6: 0.4888 (var=0.0240)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    coherent = []
    aligned_choice = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)

        # Expert 1 always favors the majority-supported option in this design.
        majority_sign = float(signs[0])
        if majority_sign == 0.0:
            continue

        minority_positions = np.flatnonzero(signs == -majority_sign)
        if minority_positions.size != 3:
            continue

        # The coherent condition has three consecutive minority cues; the
        # dispersed condition has no adjacent minority pair.
        is_coherent = bool(np.all(np.diff(minority_positions) == 1))

        # +1 means choosing the majority/Expert-1-supported option, regardless
        # of whether that option was displayed as A or B.
        response_sign = 1.0 if int(row["response"]) == 0 else -1.0
        aligned = majority_sign * response_sign

        coherent.append(is_coherent)
        aligned_choice.append(aligned)

    coherent = np.asarray(coherent, dtype=bool)
    aligned_choice = np.asarray(aligned_choice, dtype=float)
    if aligned_choice.size == 0 or not np.any(coherent) or not np.any(~coherent):
        return float("nan")

    return float(np.mean(aligned_choice[coherent]) - np.mean(aligned_choice[~coherent]))

```

**Observed (real) value:** 1.4567 (var=0.1585)
**Candidate trajectory (this loop):**
  - iter 1: 0.1458 (var=0.0727) (Δ vs real -1.3108)
  - iter 2: 0.8558 (var=0.2474) (Δ vs real -0.6008)
  - iter 3: 0.7275 (var=0.2195) (Δ vs real -0.7292)
  - iter 4: 0.6617 (var=0.1633) (Δ vs real -0.7950)
  - iter 5: 0.6975 (var=0.1673) (Δ vs real -0.7592)
  - iter 6: 0.7658 (var=0.2969) (Δ vs real -0.6908)
  - iter 7 (current): 0.5908 (var=0.2246) (Δ vs real -0.8658)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2575 (var=0.0373)
- pi_5: 0.1083 (var=0.0232)
- pi_1: -0.0158 (var=0.0201)
- pi_2: -0.0183 (var=0.0279)
- pi_3: -0.0333 (var=0.0272)
- pi_4: -0.0017 (var=0.0032)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify whether Expert 5 (index 4) or Expert 7 (index 6) is the
    # diagnostic candidate cue, and which response selects its favored option.
    cells = {4: {1: [], -1: []}, 6: {1: [], -1: []}}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        d = a - b

        active_candidates = [j for j in (4, 6) if d[j] != 0.0]
        if len(active_candidates) != 1:
            continue

        cue = active_candidates[0]
        orientation = 1 if d[cue] > 0.0 else -1
        response = int(row['response'])
        focal_choice = float((response == 0) if orientation == 1 else (response == 1))
        cells[cue][orientation].append(focal_choice)

    def side_balanced_rate(cue):
        rates = []
        for orientation in (1, -1):
            values = cells[cue][orientation]
            if len(values) > 0:
                rates.append(float(np.mean(values)))
        if len(rates) == 0:
            return np.nan
        return float(np.mean(rates))

    early = side_balanced_rate(4)
    late = side_balanced_rate(6)
    if not np.isfinite(early) or not np.isfinite(late):
        return 0.0

    # Direction of the Expert-5 to Expert-7 focal-choice gradient.
    return float(np.sign(late - early))

```

**Observed (real) value:** 0.0000 (var=0.7584)
**Candidate trajectory (this loop):**
  - iter 1: -1.0000 (var=0.1856) (Δ vs real -1.0000)
  - iter 2: -1.0000 (var=0.5184) (Δ vs real -1.0000)
  - iter 3: 1.0000 (var=0.6004) (Δ vs real +1.0000)
  - iter 4: -1.0000 (var=0.5344) (Δ vs real -1.0000)
  - iter 5: 1.0000 (var=0.6800) (Δ vs real +1.0000)
  - iter 6: 1.0000 (var=0.5056) (Δ vs real +1.0000)
  - iter 7 (current): -1.0000 (var=0.6916) (Δ vs real -1.0000)
**Other theories' values on this metric (for reference):**
- pi_5: 1.0000 (var=0.8656)
- pi_6: -1.0000 (var=0.8800)
- pi_1: 1.0000 (var=0.7796)
- pi_2: -1.0000 (var=0.8516)
- pi_3: -1.0000 (var=0.7764)
- pi_4: -1.0000 (var=0.6100)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed resource-rational selective-retrieval family and improves the late-dissent signature in Experiment 8, with the correct direction and matched variance (0.639 versus 0.775), as well as the positional effect in Experiment 6 (-0.075 versus -0.113). It also avoids pi_6's extreme failure in Experiment 1. However, the aggregate fit is not yet experiment-invariant. The largest failure is Experiment 9: the model predicts only a small coherent-minus-dispersed contrast (0.146) compared with the observed 1.457. Thus, although late dissent has priority, retrieval competition is not sharp enough for the dispersed condition's latest opposing cue to displace the other evidence. Experiment 10 also fails categorically: the pooled gradient is -1 rather than 0, and simulated between-subject variance is much too low, so validity dominates recency too uniformly across subjects. The model underproduces majority behavior in Experiment 2 (0.419 versus 0.668), the load crossover in Experiment 3 (0.180 versus 0.361), and validity-led overload behavior in Experiment 4 (0.666 versus 0.876). Conversely, it overpredicts lexicographic agreement in Experiment 7 (0.384 versus 0.118), suggesting that the current distinctiveness calculation labels diffuse descending validity structures as too compressible. Finally, tied-tier heterogeneity is inadequate in Experiment 5: only 0.14 of subjects are classified as following the first top-tier member, versus 0.50 observed, with too little variance. Experiment 1's mean is reasonably close but has greatly excessive subject variance.
Rationale: Retain the single retrieval-budget mechanism, but recalibrate its competition and distinctiveness components. First, make late-conflict competition substantially more selective under limited capacity—such as a lower retrieval dose and/or a sharper priority temperature specifically when a very late unresolved cue is present—rather than merely raising all dissent probabilities. This should enlarge Experiments 8 and especially 9 without introducing adjacency. Second, reduce unconditional recency weighting and concentrate positional influence in the late-conflict interaction; broad low-load retrieval should then recover the stronger tally result in Experiment 2, while the conflict term preserves Experiments 6, 8, and 9. Third, redefine validity distinctiveness around evidence for a genuinely dominant tier, rather than giving substantial weight to global validity standard deviation. A diffuse strictly descending structure should remain poolable, reducing the excessive lexicographic prediction in Experiment 7, whereas a separated leading tier under overload should receive stronger compression, improving Experiments 3 and 4. A steeper continuous pressure function may also strengthen the low-to-high load crossover without creating a discrete strategy gate. Fourth, strengthen tied-tier person variation: the current continuation mapping leaves too many later tied cues competitive after late-conflict priority is added. Use a more nonlinear continuation mapping with a near-zero lower tail, so a substantial subset effectively represents the early retrieved tier member while another subset pools the tier, targeting Experiment 5's mean and variance. Finally, widen or rebalance independent recency-versus-validity sensitivity around their crossover so Experiment 10 contains substantial subject mass on both gradient directions and pools near zero; avoid globally weakening validity, since Experiment 4 already requires more validity-led compression. These are parameterization and normalization changes within the recommended common-process family, not a change of mechanism.

**Outcome of your advice:** iter 1 candidate loss=0.3446 -> iter 2 candidate loss=0.2701 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision materially improves the overall fit and remains faithful to the prescribed single resource-rational retrieval process. In particular, Experiment 2 is now close in mean and variance (0.752 versus 0.668), Experiment 4 moves toward validity-led overload behavior, Experiment 7 is close (0.068 versus 0.118), and the previously missing coherent-versus-dispersed effect in Experiment 9 has become substantial (0.856 versus 1.457). Thus the prior advice to sharpen endpoint-sensitive retrieval competition and revise distinctiveness was productive. However, the new endpoint mechanism is too strong when only one late dissenting cue is present: it overshoots Experiment 8 (0.923 versus 0.775), reverses and exaggerates Experiment 1 (-0.213 versus -0.030), and makes Experiment 6 too negative (-0.184 versus -0.113). At the same time, it is still not strong enough for the multiple-dissent, high-load contrast in Experiment 9. Experiment 3 also overshoots its load crossover (0.601 versus 0.361), indicating excessively abrupt load-dependent compression. The largest qualitative failure is Experiment 5: the intended tied-tier heterogeneity did not emerge, with only 0.07 classified as first-member followers versus 0.50 observed and far too little variance. Experiment 10 remains categorically negative rather than balanced, although its variance improved from 0.186 to 0.518; the structural late-conflict effect is evidently still dominating the independent recency-validity variation.
Rationale: Build on this accepted revision rather than undoing its successful direction, but make the late-alert effect depend continuously on unresolved-conflict mass and diagnostic load. Reduce budget contraction and sharpening for a single late dissenter, while preserving or strengthening them when several unresolved opposing cues compete under high load. This should move Experiments 1, 6, and 8 back toward their targets and allow recency-versus-validity heterogeneity to determine Experiment 10, while retaining or enlarging Experiment 9 without adding adjacency or coalition machinery. The modulation can remain continuous, for example by multiplying the endpoint alert by a smooth function of the number or fraction of unresolved cues and load pressure. For Experiment 3, use a less steep pressure transition or an intermediate compression gain rather than further increasing compression; the previous increase helped Experiment 4 but clearly overshot the crossover. Improve Experiment 4 separately through the validity normalization, such as calculating leading-tier distinctiveness among the active diagnostic cues rather than only from the global validity vector, so genuinely diagnostic active gaps receive priority without making all high-load trials abruptly lexicographic. Finally, fix tied-tier retrieval allocation rather than merely making the continuation sigmoid steeper. When continuation is low, attenuated later members of a tied tier should transfer competitive retrieval mass primarily to that tier's first retrieved member, not to unrelated lower tiers; when continuation is high, the whole tier should remain pooled. Applying this within-tier allocation at the final priority-normalization stage should produce the required split and variance in Experiment 5. These are normalization and parameter-interaction adjustments within the prescribed common retrieval-budget family.

**Outcome of your advice:** iter 2 candidate loss=0.2701 -> iter 3 candidate loss=0.2590 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed single, capacity-limited selective-retrieval family, and the latest accepted revision improves aggregate loss. Several signatures are now convincing: Experiment 7 is essentially exact (0.120 versus 0.118), Experiment 8 is close in both mean and variance (0.790 versus 0.775), Experiment 2 is reasonably close with appropriate heterogeneity (0.615 versus 0.668), and Experiment 6 retains the correct positional direction. Thus, the prior recommendations concerning continuous conflict modulation, active distinctiveness, and tied-tier allocation helped overall. Nevertheless, the model is not yet experiment-invariant. Experiment 1 has deteriorated again: its contrast is strongly negative (-0.283) when humans are approximately invariant to congruence (-0.030), indicating that late-conflict competition disrupts majority/top-cue agreement too strongly in this dominant-validity structure. Experiment 9 remains far too small (0.728 versus 1.457), so the same mechanism still does not adequately separate a dispersed set containing a very late dissenter from the centered coherent set. Experiment 3 continues to show excessive load crossover (0.580 versus 0.361), while Experiment 4 remains insufficiently validity-led under overload (0.737 versus 0.876). Together these imply that compression is insufficiently selective: too much crossover is generated in relatively diffuse structures, yet genuinely dominant structures are not protected enough. The mass-conserving tie edit also failed its principal target: Experiment 5 remains near 0.10 rather than 0.50 and has only one quarter of the observed variance. Finally, Experiment 10 has flipped from -1 to +1 rather than pooling at zero; its increased variance is useful, but the balance between validity and recency remains miscentered.
Rationale: Continue within the accepted resource-rational retrieval family, but change different in-family details rather than simply strengthening the previous conflict-mass gate, which improved aggregate loss while worsening Experiment 1 and reducing Experiment 9 from its iteration-2 value. First, let dominant-tier compression continuously suppress the endpoint-induced sharpening and budget contraction, while leaving ordinary positional retrieval intact. Experiment 1 appears to combine very late dissent with a strong validity hierarchy, whereas Experiment 9 requires endpoint disruption in a more diffuse structure. Multiplying only the extreme endpoint modulation by a decreasing function of distinctiveness/compression should therefore protect Experiment 1 without sacrificing the basic Experiment 6 positional effect, while permitting stronger endpoint selectivity in Experiment 9. Second, make distinctiveness more selective rather than globally increasing compression: near-zero compression for diffuse or evenly graded structures, but stronger compression once the leading active/global gap is genuinely exceptional. This should reduce Experiment 3's excessive crossover while improving Experiment 4 and further shielding Experiment 1. A modestly wider pressure transition may help, but the main correction should be the shape of the distinctiveness mapping, not another broad gain increase. Third, fix the remaining tied-tier normalization problem in Experiment 5. Reallocating priority mass to the first tied member does not preserve accumulated evidence because the subsequent saturating retrieval-probability transform converts several potential retrievals into one bounded contribution. Under low pooling, the first-member representation should inherit the tier's effective contribution mass or multiplicity, not merely its pre-saturation priority; under high pooling, separate members should retain their contributions. This is still continuous representational pooling within one retrieval process and should produce both first-member followers and pooled-majority followers. Finally, Experiment 10 has oscillated from negative to positive across the last two accepted candidates. Do not push recency or validity to the opposite extreme again. Tune the exact-tie conflict-alert floor or the validity-recency balance to an intermediate setting between iterations 2 and 3, and preserve broad independent subject variation so the pooled direction approaches zero and variance rises toward 0.758. These targeted adjustments should preserve the now-good fits in Experiments 2, 7, and 8 while addressing the largest remaining structural failures.

**Outcome of your advice:** iter 3 candidate loss=0.2590 -> iter 4 candidate loss=0.2647 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed resource-rational selective-retrieval family, but it was rejected by the accept gate and should not replace the iteration-3 base. The rejected package produced useful local improvements in Experiment 1 (-0.092 versus -0.030) and Experiment 5 (0.190 versus 0.500), but these were outweighed by regressions. Most notably, Experiment 3 moved farther from the human crossover (0.658 versus 0.361), Experiment 8 lost its previously near-exact fit (0.887 versus 0.775), Experiment 7 also moved away from target, and Experiment 9 declined further to 0.662 versus 1.457. Experiment 4 remains insufficiently validity-led (0.748 versus 0.876), while Experiment 5 still lacks enough first-member followers and between-subject heterogeneity. Experiment 10 again switches categorical direction rather than balancing at zero. The rejected changes therefore overcouple distinctiveness, global priority sharpening, and budget contraction: they protect Experiment 1 somewhat, but amplify the load crossover and single-late-dissenter effect without generating the required multi-dissenter coherent-versus-dispersed separation.
Rationale: Build on the unchanged iteration-3 accepted base, not on this rejected candidate. Because the previous combined push—squared distinctiveness, stronger global sharpening/contraction, and post-retrieval multiplicity—was rejected, do not repeat that package or increase its gains. Instead make a smaller, more diagnostic in-family revision. Preserve the iteration-3 settings that already fit Experiments 2, 7, and 8. For Experiment 9, replace global endpoint-induced sharpening or dose contraction with a cue-local competition term applied specifically to the latest unresolved cue, scaled smoothly by the presence of multiple opposing cues. This can increase disruption from the dispersed three-dissenter pattern without further exaggerating the one-dissenter effect in Experiment 8 or broadly disturbing all retrieved evidence. For Experiment 3, reduce the excessive crossover through a modest reduction in stopping/load adaptation in diffuse structures rather than another nonlinear remapping of distinctiveness; retain strong leading-tier validity access for Experiment 4 through the evidence-strength or active-gap normalization, not through globally stronger compression. Treat the Experiment 5 edit conservatively: the rejected multiplicity correction moved the mean in the right direction but not enough, so isolate and calibrate tied-tier representation independently—ensure that low-pooling subjects assign the tied tier's expected retrieved contribution to its first member, while leaving high-pooling subjects unchanged—before combining it with any endpoint changes. Finally, Experiment 10 has oscillated from -1 to +1 and back to -1. Use a bracketed intermediate exact-tie alert setting between the iteration-3 and rejected values, rather than pushing validity or recency again, and retain broad independent subject variation to raise variance toward 0.758. This is a targeted recalibration of the common retrieval process rather than a change of mechanism family.

**Outcome of your advice:** iter 4 candidate loss=0.2647 -> iter 5 candidate loss=0.2669 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed resource-rational selective-retrieval family, but it was rejected by the accept gate and is worse than the iteration-3 base. The latest cue-local addition did not solve its intended target: Experiment 9 remains far below the human coherent-minus-dispersed contrast (0.698 versus 1.457) and is slightly worse than iteration 3. Meanwhile, Experiment 1 deteriorates sharply to -0.339 versus -0.030, showing excessive disruption by late conflict in displays where behavior should be nearly invariant to top-cue/tally congruence. Experiment 3 still has an overly large load crossover (0.597 versus 0.361), whereas Experiment 4 remains insufficiently validity-led under overload (0.743 versus 0.876). This combination indicates that adaptive capacity is not differentiating diffuse from genuinely distinctive validity structures strongly enough. Experiment 5 also remains a major failure: only 0.12 of subjects are classified as first-member followers versus 0.50 observed, with much too little variance. The contribution-stage tied-tier correction therefore did not generate the required subject split. Experiment 10 still misses the canonical pooled value categorically (+1 versus 0), although its variance of 0.680 is approaching the observed 0.758. Fits in Experiments 6-8 remain respectable, especially Experiments 7 and 8, but they do not compensate for the large systematic failures.
Rationale: Build again from the unchanged iteration-3 accepted base. The two most recent rejected attempts show that neither stronger global endpoint modulation nor an additive latest-unresolved-cue bonus is the right next adjustment, so do not repeat or enlarge those terms. For Experiment 9, try a mass-conserving normalization within the unresolved-cue competition instead: as unresolved-cue count and load rise, continuously redistribute a fixed conflict-priority mass toward the latest dissenting cue rather than adding priority or contracting the whole retrieval budget. This can increase the dispersed-versus-centered separation while limiting global effects on Experiment 1 and the single-dissenter trials in Experiment 8. The main capacity correction should distinguish validity structures through resource allocation: allow greater load-related capacity growth in diffuse structures, preserving approximate majority pooling and reducing Experiment 3's excessive crossover, while allowing distinctive leading-tier structures to remain capacity-compressed and receive stronger validity-weighted evidence, improving Experiment 4 and making top-cue adherence more uniform across congruent and conflicting trials in Experiment 1. This is preferable to another steep distinctiveness remapping or global gain increase, both implicated in rejected candidates. For Experiment 5, stop changing the contribution algebra: the last two structural tied-tier corrections were rejected. Instead recalibrate the continuous tie-pooling mapping or its parameter distribution so that a substantially larger share of subjects lies in the effectively first-member-represented region while retaining a broad high-pooling tail; target both the 0.50 classification rate and high variance. Finally, do not make another large validity-versus-recency reversal for Experiment 10. Preserve the broad heterogeneity that now gives approximately correct variance and use only a very small centering adjustment based on the underlying continuous late-minus-early choice difference, rather than tuning directly against its discontinuous sign metric. Preserve the iteration-3 settings responsible for the strong Experiment 7 and 8 fits.

**Outcome of your advice:** iter 5 candidate loss=0.2669 -> iter 6 candidate loss=0.2497 -> the gate marked it ACCEPTED.

### Iteration 6 (most recent)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed resource-rational selective-retrieval family and is the best accepted version so far. The most recent advice was productive: diffuse capacity growth, mass-conserving late-dissent redistribution, and recalibrated tie pooling lowered aggregate loss from 0.2590 to 0.2497. Experiments 2, 7, and 8 are now excellent in both point estimate and variance: 0.650 versus 0.668, 0.121 versus 0.118, and 0.780 versus 0.775, respectively. Experiment 6 also retains the correct positional effect, albeit somewhat too strongly. Nevertheless, the theory is not yet experiment-invariant. Experiment 1 is a major failure: the model predicts a strongly negative congruence contrast (-0.307) where humans are nearly invariant (-0.030), with excessive subject variance. Thus dominant-tier compression is not adequately shielding the highest-validity cue from ordinary recency and conflict competition. Experiment 3 still has too much low-to-high-load crossover (0.581 versus 0.361), despite the added diffuse capacity, so high-load diffuse displays remain too selectively or positionally sampled. Conversely, Experiment 4 is insufficiently validity-led under overload (0.712 versus 0.876) and actually regressed, indicating that validity protection is weak precisely where it should be strongest. Experiment 5 improved from 0.10 to 0.21, confirming that the tie-pooling recalibration is useful, but it remains far below the observed 0.50 and has only about half the observed variance. Experiment 9 also improved to 0.766, but still captures only about half of the very large human coherent-minus-dispersed contrast of 1.457. Experiment 10 remains categorically positive rather than pooled at zero, although this discontinuous sign metric should be treated cautiously given the observed high subject variance.
Rationale: Build on this accepted candidate and preserve the changes that produced the strong fits in Experiments 2, 7, and 8. First, alter how compression interacts with priority rather than merely increasing compression globally. In a genuinely dominant validity structure, compression should attenuate ordinary recency_access and unresolved-conflict priority as well as the extreme late-alert terms. At present only the extreme endpoint modulation is strongly protected, leaving broad recency capable of producing Experiment 1's large negative contrast. A smooth compression-dependent reduction of recency and conflict coefficients should simultaneously protect Experiment 1 and increase validity-led choices in Experiment 4, while leaving diffuse structures and the successful single-dissenter effect in Experiment 8 largely unchanged. Second, extend the accepted diffuse-capacity correction through retrieval breadth or stopping, not through another steep distinctiveness remapping: diffuse high-load displays need somewhat more complete pooling to reduce Experiment 3's excessive crossover. Keep this adjustment conditional on low distinctiveness so it does not weaken dominant-tier behavior in Experiments 1 and 4. Third, the mass-conserving redistribution direction was accepted and modestly improved Experiment 9, so it can be strengthened cautiously. Increase its attainable share or sharpen its dependence on very-late position and multi-dissenter load, while using low provisional certainty to limit spillover into the narrow-majority high-load trials that drive Experiment 3. Do not return to global budget contraction, additive cue bonuses, or adjacency, which prior rejected iterations showed were counterproductive or outside the intended account. Fourth, continue the successful tie-pooling calibration in the same direction with a smaller step: place more subject mass in the effectively first-member region and strengthen top-tier retrieval concentration before the saturating retrieval transform, while retaining a substantial pooling tail. Avoid reopening the previously rejected contribution-multiplicity algebra. Finally, do not optimize directly against Experiment 10's discontinuous pooled sign. Make at most a small centering adjustment to the underlying late-minus-early probability difference and widen independent validity-versus-recency heterogeneity if this can be done without disturbing Experiments 2 and 8. The central priority is to obtain selective validity protection in Experiments 1 and 4, broader diffuse pooling in Experiment 3, and stronger multi-dissenter endpoint competition in Experiment 9 within the same continuous retrieval process.

**Outcome of your advice:** iter 6 candidate loss=0.2497 -> CURRENT candidate loss=0.2517 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed resource-rational selective-retrieval family, but the accept gate rejected it because aggregate loss increased from 0.2497 to 0.2517. The latest package produced local gains\u2014Experiment 1 became somewhat less negative, Experiment 4 became more validity-led, Experiment 5 rose from 0.21 to 0.26, and Experiment 8 is essentially exact\u2014but these did not outweigh its regressions. Experiment 1 remains a major failure at -0.258 versus -0.030, with much too much subject variance, so tally-defined late conflict still creates a strong congruence asymmetry absent in humans. Experiment 3's crossover worsened to 0.604 versus 0.361, indicating insufficient pooling in diffuse high-load displays. Experiment 9 deteriorated sharply from 0.766 to 0.591 versus the observed 1.457, suggesting that the new compression protection is spilling into a structure that should permit strong late-dissent disruption. Experiment 2 also moved away from target, from 0.650 to 0.597 versus 0.668. Experiment 4 remains low at 0.740 versus 0.876, and Experiment 5 still underproduces both first-member followers and heterogeneity. Experiments 7 and 8 remain excellent, while Experiment 6 has the correct but slightly excessive positional effect. Experiment 10 again flips categorical sign, but its high simulated variance is reasonably close to the observed heterogeneity, so its discontinuous pooled sign should not drive the next revision.",
  "verdict": "regenerate",
  "rationale": "Build from the unchanged iteration-6 accepted base. The most recent advice to jointly strengthen broad compression protection, diffuse capacity, conflict redistribution, and first-member concentration was rejected, so do not repeat or enlarge that package. Instead isolate the conflict-orientation calculation. The present mechanism defines provisional commitment by raw tally and treats unresolved conflict largely through binary cue counts. This makes numerous low-validity cues capable of triggering strong late-conflict competition in Experiment 1 even when a dominant validity structure should make them weak competitors. Compute provisional commitment and unresolved-conflict mass from the pre-conflict retrieval distribution or a smoothly validity-weighted baseline priority, with the weighting increasing continuously with distinctiveness. Conflict mass should reflect expected accessible opposing evidence rather than the raw number of opposing cues. This can suppress low-validity late opposition in Experiments 1 and 4 without blanket attenuation of recency or conflict, while leaving diffuse structures in Experiments 3, 8, and 9 tally-oriented. Also revise distinctiveness narrowly so compression requires not only an exceptional gap but concentration in a genuinely dominant active tier; tier multiplicity and active diagnostic relevance should continuously discount apparent global distinctiveness. This should prevent Experiment 9 from being inadvertently protected and reduce false compression in Experiment 3, without another steep global remapping. Preserve iteration-6 capacity, redistribution, and tie-pooling settings initially so the effect of this normalization change can be evaluated independently. Only after that isolated change should Experiment 5 receive a small parameter-distribution calibration; do not reintroduce rejected contribution multiplicity or further sharpen the tie transition now. Preserve the successful settings for Experiments 7 and 8, and make no direct categorical optimization of Experiment 10."
}
```

## Usage

```json
{
  "input_tokens": 24471,
  "output_tokens": 2094
}
```
