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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_14`
- The recommendation below acts on THEORY 2 (= `pi_14`).

Replace pi_14 with a new Salience-Guided Sequential Integration theory. The display should first generate a parallel preattentive priority map from instructed validity, endpoint salience, stable directional scan bias, and local validity/directional contrast. Cues are then sampled or refreshed in the priority-dependent order, and their signed evidence enters a leaky bounded accumulator. Recent evidence receives greater readout weight because earlier evidence decays, but updates are graded rather than categorical: update strength should depend on cue reliability, contrast surprise, the coherence and strength of the incumbent coalition, and capacity load. Thus a late weak cue can reverse an earlier decision without doing so invariably; several agreeing cues can establish an anchor that sometimes resists a singleton; and conspicuous endpoints or sharp validity drops can alter which reason is processed or refreshed last. Use task-invariant subject-level parameters for validity sensitivity, spatial scan direction, endpoint salience, contrast gain, accumulator retention, anchoring, capacity, and lapse, with a hierarchical distribution to represent the observed heterogeneity. The regenerated theory should be jointly constrained to produce partial weak-cue reversal in Experiment 1, three-cue dominance in Experiment 2, strong but nonuniform recency in Experiments 4, 6, 12, 13, and 15, moderated terminal influence in Experiments 10, 21, and 26, the small unanimous-load decline in Experiment 17, the large endpoint asymmetry in Experiment 23, the local-drop effect in Experiment 24, and the intermediate crossover in Experiment 25. It should predict trial-level probabilities and graded condition profiles rather than being tuned primarily to thresholded metrics.


## CANDIDATE THEORY
Priority-Latency Coalition Accumulation. Before deliberation, the display is encoded in parallel into a priority-latency map. Absolute instructed reliability gives diagnostically strong cues early access, a stable but heterogeneous directional scan tendency delays cues along the preferred scan path, scan-congruent endpoints attract selective reinspection, and an opposing cue following a sharp absolute validity drop receives surprise priority. Priority is sampled with modest trial-level noise rather than imposing a deterministic order. Evidence then enters a bounded leaky accumulator, so later or refreshed evidence is more available at choice. Updates remain graded: reliability affects accumulator magnitude only very weakly and compressively, local surprise increases revision, capacity load attenuates late encoding, and a coherent incumbent coalition continuously gates opposition. Coalition bookkeeping uses reliability-normalized increments, so coherence represents accumulated agreement rather than duplicating reliability's influence. Coalition resistance is reduced for spatially isolated challengers, allowing isolated terminal reasons to reverse a coalition without making all terminal cues dominant. Subject-level heterogeneity in validity sensitivity, scan direction, endpoint salience, contrast gain, retention, anchoring, capacity, priority noise, and lapse produces nonuniform recency, crossover, and endpoint effects without changing mechanisms across experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Priority-Latency Coalition Accumulation expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_sensitivity = float(parameters["validity_sensitivity"])
    scan_direction = float(parameters["scan_direction"])
    endpoint_salience = float(parameters["endpoint_salience"])
    contrast_gain = float(parameters["contrast_gain"])
    accumulator_retention = float(parameters["accumulator_retention"])
    anchoring_strength = float(parameters["anchoring_strength"])
    capacity = float(parameters["capacity"])
    priority_temperature = float(parameters["priority_temperature"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))

    # A fixed, task-invariant reliability scale preserves absolute gaps. It
    # therefore distinguishes nearly equal weak cues from a .99-versus-.52
    # conflict instead of mapping both two-cue displays to the same +/-1 pair.
    absolute_reliability = np.clip(
        (reliability - 0.20) / 1.55, -0.15, 3.0
    )

    # The complete map is computed before any sequential accumulation.
    # Local contrast is defined relative to the nearest discriminating item
    # on the left, reflecting perceptual grouping in the displayed layout.
    local_drop = np.zeros(n_features, dtype=float)
    directional_boundary = np.zeros(n_features, dtype=float)
    isolation = np.zeros(n_features, dtype=float)
    previous_j = None
    for j_raw in discriminating:
        j = int(j_raw)
        if previous_j is not None:
            gap_fraction = float(j - previous_j - 1) / float(
                max(1, n_features - 1)
            )
            isolation[j] = max(isolation[j], gap_fraction)
            isolation[previous_j] = max(isolation[previous_j], gap_fraction)

            previous_direction = float(np.sign(diff[previous_j]))
            current_direction = float(np.sign(diff[j]))
            if previous_direction != current_direction:
                directional_boundary[j] = 1.0
                drop = max(0.0, float(validities[previous_j] - validities[j]))
                # Fixed scaling retains the absolute distinction between a
                # shallow and a conspicuous instructed-validity drop.
                local_drop[j] = min(3.0, drop / 0.20)
        previous_j = j

    positions = np.arange(n_features, dtype=float) / float(max(1, n_features - 1))
    signed_position = 2.0 * positions - 1.0
    left_edge = (np.arange(n_features) == 0).astype(float)
    right_edge = (np.arange(n_features) == n_features - 1).astype(float)

    # Lower latency means earlier sampling. Reliability has the strongest
    # effect on access. Endpoint delay follows the subject's scan direction,
    # is amplified by isolation, and is attenuated for exceptionally reliable
    # cues, which can consequently enter before a spatially earlier weak cue.
    scan_sign = 1.0 if scan_direction >= 0.0 else -1.0
    endpoint_delay = (
        endpoint_salience
        * scan_sign
        * (right_edge - left_edge)
        * (0.35 + 0.90 * isolation)
        / (1.0 + np.maximum(0.0, absolute_reliability))
    )
    contrast_delay = contrast_gain * local_drop * directional_boundary
    latency = (
        -validity_sensitivity * absolute_reliability
        + scan_direction * signed_position
        + endpoint_delay
        + contrast_delay
    )

    # Modest Gumbel perturbations instantiate priority-dependent sampling
    # rather than a perfectly deterministic ranking on every presentation.
    sampled_latency = latency + np.random.gumbel(
        loc=0.0, scale=priority_temperature, size=n_features
    )

    # Stable index tie-breaking preserves the physical layout when sampled map
    # values are equal. Only discriminating cues consume integration capacity.
    cue_order = sorted(
        [int(j) for j in discriminating],
        key=lambda j: (float(sampled_latency[j]), j)
    )

    accumulator = 0.0
    incumbent_direction = 0.0
    coherent_count = 0.0
    coalition_strength = 0.0

    def apply_update(j, refresh_fraction, step_index,
                     accumulator, incumbent_direction,
                     coherent_count, coalition_strength):
        direction = float(np.sign(diff[j]))

        # Reliability primarily controls access priority. Its direct effect on
        # evidence magnitude is nearly flat, so a highly valid cue sampled
        # early does not retain a second large advantage over later evidence.
        reliability_strength = float(np.exp(
            0.018 * validity_sensitivity * absolute_reliability[j]
        ))
        reliability_strength = float(np.clip(
            reliability_strength, 0.97, 1.07
        ))

        # A reliability drop is surprising only at a directional boundary.
        # Surprise boosts the challenger smoothly, with no reversal switch.
        surprise = 1.0 + (
            contrast_gain
            * local_drop[j]
            * directional_boundary[j]
            * (0.30 + 0.35 * min(1.0, abs(accumulator)))
        )
        surprise = float(np.clip(surprise, 1.0, 3.2))

        # Capacity loss is gradual and applies most strongly after the
        # subject-specific integration span has been exceeded.
        overload = max(0.0, float(step_index + 1) - capacity)
        capacity_factor = 1.0 / np.sqrt(1.0 + 0.28 * overload)

        opposing = bool(
            incumbent_direction != 0.0 and direction != incumbent_direction
        )
        if opposing:
            # Coherent agreement anchors the incumbent continuously. Spatial
            # isolation weakens the gate, allowing a conspicuous separated
            # challenger to revise more than a contiguous singleton.
            coherence = max(0.0, coherent_count - 1.0)
            anchor_mass = (
                coherence
                * min(2.0, coalition_strength)
                * (1.0 - 0.72 * min(1.0, isolation[j]))
            )
            anchor_gate = 1.0 + anchoring_strength * anchor_mass
        else:
            anchor_gate = 1.0

        update = (
            refresh_fraction
            * reliability_strength
            * surprise
            * capacity_factor
            / anchor_gate
        )

        accumulator = (
            accumulator_retention * accumulator + direction * update
        )
        accumulator = float(np.clip(accumulator, -2.75, 2.75))

        # Coalition bookkeeping represents directional agreement without
        # counting reliability twice. Reliability still affects the actual
        # accumulator update and remains fully active in access latency.
        coalition_increment = update / max(reliability_strength, 1e-8)

        # Coalition state is graded. Opposition erodes the incumbent before
        # a new coherent coalition is installed; there is no categorical run
        # policy or thresholded strategy change.
        if incumbent_direction == 0.0:
            incumbent_direction = direction
            coherent_count = refresh_fraction
            coalition_strength = min(2.5, coalition_increment)
        elif direction == incumbent_direction:
            coherent_count += refresh_fraction
            coalition_strength = min(
                2.5,
                0.78 * coalition_strength + 0.55 * coalition_increment
            )
        else:
            erosion = refresh_fraction * coalition_increment
            coalition_strength -= 0.72 * erosion
            coherent_count -= 0.62 * refresh_fraction
            if coalition_strength <= 0.0 or coherent_count <= 0.0:
                incumbent_direction = direction
                coherent_count = max(0.5, refresh_fraction)
                coalition_strength = min(2.5, 0.65 * coalition_increment)
            else:
                coalition_strength = max(0.0, coalition_strength)
                coherent_count = max(0.0, coherent_count)

        return (
            accumulator,
            incumbent_direction,
            coherent_count,
            coalition_strength
        )

    for step_index, j in enumerate(cue_order):
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            j, 1.0, step_index, accumulator, incumbent_direction,
            coherent_count, coalition_strength
        )

    # Refresh is selective rather than a universal terminal bonus. It depends
    # on endpoint-by-scan congruence, endpoint isolation, and absolute local
    # contrast, preventing contiguous weak terminals from controlling every
    # display while preserving conspicuous endpoint effects.
    refresh_scores = {}
    for j in cue_order:
        scan_endpoint = right_edge[j] if scan_direction >= 0.0 else left_edge[j]
        opposite_endpoint = left_edge[j] if scan_direction >= 0.0 else right_edge[j]
        endpoint_refresh = endpoint_salience * (
            scan_endpoint * (0.20 + 1.35 * isolation[j])
            + 0.12 * opposite_endpoint * isolation[j]
        )
        refresh_scores[j] = (
            endpoint_refresh
            + contrast_gain * local_drop[j] * directional_boundary[j]
            + 0.04 * max(0.0, absolute_reliability[j])
        )

    refresh_j = max(
        cue_order,
        key=lambda j: (refresh_scores[j], sampled_latency[j])
    )
    raw_refresh = float(refresh_scores[refresh_j])
    refresh_fraction = 0.32 * raw_refresh / (1.0 + raw_refresh)
    if refresh_fraction > 1e-8:
        accumulator, incumbent_direction, coherent_count, coalition_strength = apply_update(
            refresh_j, refresh_fraction, len(cue_order), accumulator,
            incumbent_direction, coherent_count, coalition_strength
        )

    # Display load reduces readout precision even for unanimous evidence,
    # yielding a small rather than catastrophic decline with additional cues.
    nominal_overload = max(0.0, float(n_features) - capacity)
    evidence_overload = max(0.0, float(len(cue_order)) - capacity)
    precision = 2.65 / (
        1.0 + 0.045 * nominal_overload + 0.075 * evidence_overload
    )
    precision = float(np.clip(precision, 0.65, 2.65))

    logit_a = float(np.clip(precision * accumulator, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
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
- validity_sensitivity: [0.45, 0.85]
- scan_direction: [-0.25, 1.45]
- endpoint_salience: [0.45, 1.65]
- contrast_gain: [0.28, 0.72]
- accumulator_retention: [0.40, 0.62]
- anchoring_strength: [0.22, 0.58]
- capacity: [4.2, 7.4]
- priority_temperature: [0.02, 0.18]
- lapse_rate: [0.05, 0.16]

`rationale`:
This is a narrow extension of the accepted iteration-8 edit. The direct reliability multiplier is flattened from an exponential slope of 0.035 with clipping at [0.94, 1.14] to a slope of 0.018 with clipping at [0.97, 1.07]. Absolute instructed reliability remains unchanged in the priority-latency map, preserving the accepted reliability crossovers in Experiments 23 and 25 while further reducing the second, update-magnitude advantage that sustains excessive high-validity control in Experiments 1, 2, 6, and 9. The only associated implementation change is to divide reliability back out when updating coalition strength. Thus compact coalitions retain approximately the same agreement-based protection even as reliability's direct accumulator effect is flattened, reducing the risk of worsening Experiment 8. Retention, scan direction, endpoint salience, contrast, isolation, refresh, capacity, readout, and every parameter range are left unchanged. In particular, this attempt does not bundle the endpoint-salience adjustment proposed as a subsequent independent test, because previous multi-mechanism endpoint edits were rejected and a universal endpoint increase could worsen Experiments 8 and 24.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2072 -> ACCEPTED
- iter 2: loss=0.1675 -> ACCEPTED
- iter 3: loss=0.1779 -> REJECTED
- iter 4: loss=0.2156 -> REJECTED
- iter 5: loss=0.1964 -> REJECTED
- iter 6: loss=0.1729 -> REJECTED
- iter 7: loss=0.1858 -> REJECTED
- iter 8: loss=0.1604 -> ACCEPTED
- iter 9 (current candidate you are grading): loss=0.1725 -> REJECTED
Running-best (last accepted) base: iter 8 at loss=0.1604.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast TTB-consistent choice rates between trials where the
    # equal-weight tally supports versus opposes the TTB winner.
    support_choices = []
    oppose_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        differing = np.flatnonzero(d != 0)
        if differing.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        ttb_direction = float(np.sign(d[differing[0]]))  # +1: A, -1: B
        ttb_response = 0 if ttb_direction > 0 else 1
        chose_ttb = float(int(row["response"]) == ttb_response)

        # Positive iff the tally margin points toward the TTB winner.
        tally_a_margin = float(np.sign(d).sum())
        oriented_tally_margin = ttb_direction * tally_a_margin

        if oriented_tally_margin > 0:
            support_choices.append(chose_ttb)
        elif oriented_tally_margin < 0:
            oppose_choices.append(chose_ttb)

    if len(support_choices) == 0 or len(oppose_choices) == 0:
        return float("nan")

    return float(np.mean(support_choices) - np.mean(oppose_choices))
```

**Observed (real) value:** 0.4944 (var=0.0204)
**Candidate trajectory (this loop):**
  - iter 1: 0.6921 (var=0.0124) (Δ vs real +0.1976)
  - iter 2: 0.7206 (var=0.0143) (Δ vs real +0.2261)
  - iter 3: 0.6540 (var=0.0158) (Δ vs real +0.1596)
  - iter 4: 0.7604 (var=0.0103) (Δ vs real +0.2660)
  - iter 5: 0.7937 (var=0.0104) (Δ vs real +0.2993)
  - iter 6: 0.6724 (var=0.0136) (Δ vs real +0.1779)
  - iter 7: 0.7047 (var=0.0103) (Δ vs real +0.2103)
  - iter 8: 0.6961 (var=0.0131) (Δ vs real +0.2017)
  - iter 9 (current): 0.7458 (var=0.0072) (Δ vs real +0.2514)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)
- pi_6: 0.6093 (var=0.0085)
- pi_7: 0.5681 (var=0.0095)
- pi_8: 0.5276 (var=0.0147)
- pi_9: 0.8585 (var=0.0048)
- pi_10: 0.7281 (var=0.0159)
- pi_11: 0.8919 (var=0.0037)
- pi_12: 0.8796 (var=0.0040)
- pi_13: 0.7653 (var=0.0059)
- pi_14: 0.3789 (var=0.0089)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which choices follow the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    matches = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            # An all-tie pair provides no directional diagnosticity.
            continue

        first = int(discriminating[0])
        ttb_response = 0 if diff[first] > 0 else 1
        matches.append(float(int(row["response"]) == ttb_response))

    if not matches:
        return float("nan")
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1938 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: 0.2633 (var=0.0030) (Δ vs real +0.0696)
  - iter 2: 0.2671 (var=0.0077) (Δ vs real +0.0733)
  - iter 3: 0.3075 (var=0.0027) (Δ vs real +0.1137)
  - iter 4: 0.2706 (var=0.0044) (Δ vs real +0.0769)
  - iter 5: 0.2335 (var=0.0050) (Δ vs real +0.0398)
  - iter 6: 0.2515 (var=0.0032) (Δ vs real +0.0577)
  - iter 7: 0.2765 (var=0.0052) (Δ vs real +0.0827)
  - iter 8: 0.2569 (var=0.0041) (Δ vs real +0.0631)
  - iter 9 (current): 0.2767 (var=0.0058) (Δ vs real +0.0829)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)
- pi_6: 0.2027 (var=0.0056)
- pi_7: 0.1940 (var=0.0026)
- pi_8: 0.1765 (var=0.0015)
- pi_9: 0.3002 (var=0.0056)
- pi_10: 0.2408 (var=0.0078)
- pi_11: 0.2185 (var=0.0064)
- pi_12: 0.1323 (var=0.0014)
- pi_13: 0.1877 (var=0.0021)
- pi_14: 0.4854 (var=0.0035)

### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the least-valid discriminating cue."""
    if len(data) == 0:
        return float("nan")

    agreements = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        # Features are displayed in descending validity order, so the
        # largest differing index is the final discriminating reason.
        final_idx = int(differing[-1])
        final_cue_choice = 0 if a[final_idx] > b[final_idx] else 1
        agreements.append(float(int(response) == final_cue_choice))

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5750 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.4100 (var=0.0032) (Δ vs real -0.1650)
  - iter 2: 0.4773 (var=0.0054) (Δ vs real -0.0977)
  - iter 3: 0.4792 (var=0.0041) (Δ vs real -0.0958)
  - iter 4: 0.4688 (var=0.0062) (Δ vs real -0.1062)
  - iter 5: 0.3958 (var=0.0060) (Δ vs real -0.1792)
  - iter 6: 0.5281 (var=0.0032) (Δ vs real -0.0469)
  - iter 7: 0.4431 (var=0.0041) (Δ vs real -0.1319)
  - iter 8: 0.4890 (var=0.0058) (Δ vs real -0.0860)
  - iter 9 (current): 0.4913 (var=0.0043) (Δ vs real -0.0837)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)
- pi_6: 0.6577 (var=0.0054)
- pi_7: 0.7098 (var=0.0021)
- pi_8: 0.6569 (var=0.0032)
- pi_9: 0.4806 (var=0.0078)
- pi_10: 0.7527 (var=0.0062)
- pi_11: 0.6517 (var=0.0035)
- pi_12: 0.4804 (var=0.0126)
- pi_13: 0.6510 (var=0.0055)
- pi_14: 0.4919 (var=0.0025)

### Experiment 4
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed association between choice and the lowest-validity (fourth) cue.
    # +1 means the response follows that cue, -1 means it opposes that cue.
    products = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_direction = np.sign(a[-1] - b[-1])  # +1 favors A, -1 favors B
        if terminal_direction == 0:
            continue
        response = int(row['response'])
        choice_direction = 1.0 if response == 0 else -1.0
        products.append(choice_direction * terminal_direction)
    if len(products) == 0:
        return float('nan')
    return float(np.mean(products))
```

**Observed (real) value:** 0.6650 (var=0.0621)
**Candidate trajectory (this loop):**
  - iter 1: 0.4637 (var=0.0059) (Δ vs real -0.2013)
  - iter 2: 0.4379 (var=0.0275) (Δ vs real -0.2271)
  - iter 3: 0.3771 (var=0.0166) (Δ vs real -0.2879)
  - iter 4: 0.4213 (var=0.0226) (Δ vs real -0.2438)
  - iter 5: 0.3767 (var=0.0245) (Δ vs real -0.2883)
  - iter 6: 0.4546 (var=0.0108) (Δ vs real -0.2104)
  - iter 7: 0.4029 (var=0.0193) (Δ vs real -0.2621)
  - iter 8: 0.4354 (var=0.0250) (Δ vs real -0.2296)
  - iter 9 (current): 0.4350 (var=0.0332) (Δ vs real -0.2300)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)
- pi_6: 0.6704 (var=0.0140)
- pi_7: 0.6104 (var=0.0134)
- pi_8: 0.6408 (var=0.0101)
- pi_9: 0.2967 (var=0.0253)
- pi_10: 0.5654 (var=0.0275)
- pi_11: 0.5696 (var=0.0140)
- pi_12: 0.4846 (var=0.0212)
- pi_13: 0.6937 (var=0.0064)
- pi_14: -0.0242 (var=0.0099)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    block_terminal = []
    mixed_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # The block permutation has one option favored by cues 1--5 and
        # the other option favored by cues 6--10.
        is_block = (
            d.size == 10
            and np.all(d[:5] == d[0])
            and np.all(d[5:] == d[5])
            and d[0] == -d[5]
        )

        # In this design the final two cues agree. Identify whether the
        # response selected the option favored by those terminal cues.
        terminal_response = 0 if d[-1] > 0 else 1
        chose_terminal = float(int(row["response"]) == terminal_response)

        if is_block:
            block_terminal.append(chose_terminal)
        else:
            mixed_terminal.append(chose_terminal)

    if len(block_terminal) == 0 or len(mixed_terminal) == 0:
        return float("nan")

    return float(np.mean(mixed_terminal) - np.mean(block_terminal))
```

**Observed (real) value:** 0.0033 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: -0.0108 (var=0.0102) (Δ vs real -0.0142)
  - iter 2: -0.0308 (var=0.0116) (Δ vs real -0.0342)
  - iter 3: 0.0054 (var=0.0149) (Δ vs real +0.0021)
  - iter 4: -0.0229 (var=0.0155) (Δ vs real -0.0262)
  - iter 5: 0.0217 (var=0.0135) (Δ vs real +0.0183)
  - iter 6: -0.0246 (var=0.0120) (Δ vs real -0.0279)
  - iter 7: 0.0212 (var=0.0145) (Δ vs real +0.0179)
  - iter 8: -0.0812 (var=0.0116) (Δ vs real -0.0846)
  - iter 9 (current): -0.0433 (var=0.0114) (Δ vs real -0.0467)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)
- pi_6: -0.0142 (var=0.0089)
- pi_7: 0.0079 (var=0.0079)
- pi_8: 0.0112 (var=0.0099)
- pi_9: -0.0658 (var=0.0097)
- pi_10: 0.1788 (var=0.0194)
- pi_11: -0.0883 (var=0.0064)
- pi_12: -0.5029 (var=0.0440)
- pi_13: -0.0292 (var=0.0056)
- pi_14: -0.0050 (var=0.0087)

### Experiment 6
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)

        # Restrict to the 7- and 8-reason overload conditions.
        if discriminating.size < 7:
            continue

        # The first discriminating cue identifies the option favored by the
        # high-validity, early run, regardless of physical A/B orientation.
        first_direction = diff[discriminating[0]]
        response = int(row["response"])
        chose_early_option = (
            (first_direction > 0 and response == 0) or
            (first_direction < 0 and response == 1)
        )
        early_choices.append(float(chose_early_option))

    if not early_choices:
        return float("nan")
    return float(np.mean(early_choices))
```

**Observed (real) value:** 0.3067 (var=0.0051)
**Candidate trajectory (this loop):**
  - iter 1: 0.3879 (var=0.0076) (Δ vs real +0.0813)
  - iter 2: 0.3817 (var=0.0094) (Δ vs real +0.0750)
  - iter 3: 0.4521 (var=0.0113) (Δ vs real +0.1454)
  - iter 4: 0.3858 (var=0.0063) (Δ vs real +0.0792)
  - iter 5: 0.4088 (var=0.0143) (Δ vs real +0.1021)
  - iter 6: 0.4079 (var=0.0041) (Δ vs real +0.1013)
  - iter 7: 0.4342 (var=0.0101) (Δ vs real +0.1275)
  - iter 8: 0.3696 (var=0.0086) (Δ vs real +0.0629)
  - iter 9 (current): 0.3642 (var=0.0087) (Δ vs real +0.0575)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)
- pi_6: 0.3067 (var=0.0061)
- pi_7: 0.3404 (var=0.0071)
- pi_8: 0.3050 (var=0.0041)
- pi_9: 0.1708 (var=0.0061)
- pi_10: 0.2900 (var=0.0177)
- pi_11: 0.0546 (var=0.0021)
- pi_12: 0.2771 (var=0.0184)
- pi_13: 0.1221 (var=0.0029)
- pi_14: 0.5279 (var=0.0042)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_follow = []
    high_follow = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = diff[diff != 0]

        # The strongest predicted chunking contrast occurs for the 8- and
        # 10-reason displays.
        if discriminating.size < 8:
            continue

        directions = np.sign(discriminating)
        n_switches = int(np.sum(directions[1:] != directions[:-1]))

        # Response coding is 0=A and 1=B. Determine whether the response
        # follows the terminal discriminating cue, independent of side.
        terminal_response = 0 if directions[-1] > 0 else 1
        follows_terminal = float(int(row["response"]) == terminal_response)

        if n_switches <= 1:
            low_follow.append(follows_terminal)
        elif n_switches >= 4:
            high_follow.append(follows_terminal)

    if len(low_follow) == 0 or len(high_follow) == 0:
        return float("nan")

    contrast = float(np.mean(low_follow) - np.mean(high_follow))

    # A preregisterable practical-significance diagnostic: whether coherent
    # low-switch sequences have more than an 8-percentage-point terminal-cue
    # advantage over switch-dense sequences.
    return float(contrast > 0.08)

```

**Observed (real) value:** 0.0000 (var=0.2016)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2496) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2464) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.2484) (Δ vs real +1.0000)
  - iter 5: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.2176) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.2436) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.2400) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)
- pi_6: 0.0000 (var=0.2100)
- pi_7: 0.0000 (var=0.2356)
- pi_8: 0.0000 (var=0.1924)
- pi_9: 1.0000 (var=0.2436)
- pi_10: 0.0000 (var=0.2356)
- pi_11: 0.0000 (var=0.2436)
- pi_12: 0.0000 (var=0.2176)
- pi_13: 0.0000 (var=0.2100)
- pi_14: 0.0000 (var=0.1924)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Direction of the terminal discriminating cue in displayed order.
        terminal_direction = diff[discriminating[-1]]
        terminal_response = 0 if terminal_direction > 0 else 1
        agreements.append(float(int(row['response']) == terminal_response))

    if len(agreements) == 0:
        return float('nan')
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1721 (var=0.0112)
**Candidate trajectory (this loop):**
  - iter 1: 0.6204 (var=0.0027) (Δ vs real +0.4483)
  - iter 2: 0.5363 (var=0.0123) (Δ vs real +0.3642)
  - iter 3: 0.5325 (var=0.0113) (Δ vs real +0.3604)
  - iter 4: 0.4931 (var=0.0170) (Δ vs real +0.3210)
  - iter 5: 0.5179 (var=0.0142) (Δ vs real +0.3458)
  - iter 6: 0.5823 (var=0.0023) (Δ vs real +0.4102)
  - iter 7: 0.5083 (var=0.0120) (Δ vs real +0.3362)
  - iter 8: 0.5400 (var=0.0128) (Δ vs real +0.3679)
  - iter 9 (current): 0.5440 (var=0.0141) (Δ vs real +0.3719)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)
- pi_6: 0.2250 (var=0.0059)
- pi_7: 0.3131 (var=0.0029)
- pi_8: 0.3069 (var=0.0027)
- pi_9: 0.3046 (var=0.0042)
- pi_10: 0.1440 (var=0.0033)
- pi_11: 0.0848 (var=0.0020)
- pi_12: 0.6604 (var=0.0051)
- pi_13: 0.8115 (var=0.0028)
- pi_14: 0.4813 (var=0.0033)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are already arranged in descending instructed validity.
        first_direction = float(np.sign(diff[int(discriminating[0])]))
        choice_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(first_direction * choice_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** -0.0325 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.2021 (var=0.0129) (Δ vs real +0.2346)
  - iter 2: 0.1833 (var=0.0210) (Δ vs real +0.2158)
  - iter 3: 0.1521 (var=0.0146) (Δ vs real +0.1846)
  - iter 4: 0.2013 (var=0.0162) (Δ vs real +0.2338)
  - iter 5: 0.2908 (var=0.0137) (Δ vs real +0.3233)
  - iter 6: 0.1842 (var=0.0141) (Δ vs real +0.2167)
  - iter 7: 0.2087 (var=0.0149) (Δ vs real +0.2412)
  - iter 8: 0.1425 (var=0.0123) (Δ vs real +0.1750)
  - iter 9 (current): 0.1412 (var=0.0149) (Δ vs real +0.1737)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0083 (var=0.0103)
- pi_6: 0.2096 (var=0.0101)
- pi_1: 0.7021 (var=0.0307)
- pi_2: 0.7054 (var=0.0384)
- pi_4: 0.0375 (var=0.0075)
- pi_5: 0.0108 (var=0.0053)
- pi_7: 0.0638 (var=0.0071)
- pi_8: 0.0083 (var=0.0044)
- pi_9: 0.0933 (var=0.0102)
- pi_10: 0.0575 (var=0.0053)
- pi_11: 0.1892 (var=0.0079)
- pi_12: 0.2621 (var=0.0301)
- pi_13: 0.1483 (var=0.0096)
- pi_14: 0.1450 (var=0.0175)

### Experiment 10
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Coherence-weighted agreement with the least-valid terminal cue."""
    if len(data) == 0:
        return float("nan")

    pattern_weights = {
        (1, 1, 1, 1): 1.00,   # fully coherent
        (-1, 1, 1, 1): 0.80,  # earliest cue conflicts
        (1, -1, 1, 1): 0.65,  # second cue conflicts
        (-1, -1, 1, 1): 0.35  # both prefix cues conflict
    }

    weighted_agreement = 0.0
    total_weight = 0.0

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        if diff.size != 4 or diff[3] == 0:
            continue

        # Orient every trial so that the terminal (least-valid) cue is +1.
        terminal_sign = float(np.sign(diff[3]))
        relative_pattern = tuple(
            int(np.sign(x) * terminal_sign) for x in diff
        )
        weight = pattern_weights.get(relative_pattern, 0.0)
        if weight <= 0.0:
            continue

        terminal_favoring_response = 0 if terminal_sign > 0 else 1
        agreement = float(int(response) == terminal_favoring_response)
        weighted_agreement += weight * agreement
        total_weight += weight

    if total_weight <= 0.0:
        return float("nan")
    return float(weighted_agreement / total_weight)
```

**Observed (real) value:** 0.8748 (var=0.0069)
**Candidate trajectory (this loop):**
  - iter 1: 0.8492 (var=0.0022) (Δ vs real -0.0255)
  - iter 2: 0.8639 (var=0.0020) (Δ vs real -0.0108)
  - iter 3: 0.8126 (var=0.0026) (Δ vs real -0.0622)
  - iter 4: 0.8604 (var=0.0021) (Δ vs real -0.0144)
  - iter 5: 0.8953 (var=0.0012) (Δ vs real +0.0205)
  - iter 6: 0.8297 (var=0.0022) (Δ vs real -0.0451)
  - iter 7: 0.8378 (var=0.0021) (Δ vs real -0.0370)
  - iter 8: 0.8486 (var=0.0024) (Δ vs real -0.0261)
  - iter 9 (current): 0.8452 (var=0.0039) (Δ vs real -0.0296)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9281 (var=0.0013)
- pi_3: 0.7843 (var=0.0034)
- pi_1: 0.5635 (var=0.0023)
- pi_2: 0.8101 (var=0.0072)
- pi_4: 0.8645 (var=0.0034)
- pi_5: 0.9256 (var=0.0034)
- pi_7: 0.8717 (var=0.0018)
- pi_8: 0.8753 (var=0.0015)
- pi_9: 0.8852 (var=0.0027)
- pi_10: 0.8569 (var=0.0036)
- pi_11: 0.9369 (var=0.0009)
- pi_12: 0.9271 (var=0.0010)
- pi_13: 0.9058 (var=0.0010)
- pi_14: 0.6261 (var=0.0019)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        m = int(np.count_nonzero(diff))

        # The diagnostic manageable-load conflicts contain 5 or 6
        # discriminating cues. The final (lowest-validity) cue is index 9.
        if m not in (5, 6) or diff[-1] == 0:
            continue

        terminal_response = 0 if diff[-1] > 0 else 1
        scores.append(float(int(row["response"]) == terminal_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8425 (var=0.0123)
**Candidate trajectory (this loop):**
  - iter 1: 0.3750 (var=0.0098) (Δ vs real -0.4675)
  - iter 2: 0.3804 (var=0.0100) (Δ vs real -0.4621)
  - iter 3: 0.5196 (var=0.0084) (Δ vs real -0.3229)
  - iter 4: 0.4192 (var=0.0101) (Δ vs real -0.4233)
  - iter 5: 0.4188 (var=0.0172) (Δ vs real -0.4238)
  - iter 6: 0.4679 (var=0.0070) (Δ vs real -0.3746)
  - iter 7: 0.4683 (var=0.0240) (Δ vs real -0.3742)
  - iter 8: 0.3942 (var=0.0103) (Δ vs real -0.4483)
  - iter 9 (current): 0.4275 (var=0.0119) (Δ vs real -0.4150)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6863 (var=0.0047)
- pi_7: 0.0642 (var=0.0023)
- pi_1: 0.1583 (var=0.0130)
- pi_2: 0.1342 (var=0.0084)
- pi_4: 0.3117 (var=0.0174)
- pi_5: 0.6737 (var=0.0053)
- pi_6: 0.6196 (var=0.0088)
- pi_8: 0.8558 (var=0.0026)
- pi_9: 0.6025 (var=0.0106)
- pi_10: 0.8008 (var=0.0073)
- pi_11: 0.5088 (var=0.0101)
- pi_12: 0.1600 (var=0.0117)
- pi_13: 0.6921 (var=0.0099)
- pi_14: 0.4788 (var=0.0041)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Side-balanced rate of choosing the 3-to-2 majority in strong-coalition trials."""
    majority_a = []
    majority_b = []
    strong_set = {0, 1, 2, 11, 12}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = set(np.flatnonzero(diff != 0).tolist())
        if discriminating != strong_set:
            continue

        response = int(row["response"])
        majority_direction = float(np.sum(np.sign(diff)))
        if majority_direction > 0:
            majority_a.append(float(response == 0))
        elif majority_direction < 0:
            majority_b.append(float(response == 1))

    if majority_a and majority_b:
        return float(0.5 * (np.mean(majority_a) + np.mean(majority_b)))

    pooled = majority_a + majority_b
    if pooled:
        return float(np.mean(pooled))
    return 0.5

```

**Observed (real) value:** 0.1713 (var=0.0095)
**Candidate trajectory (this loop):**
  - iter 1: 0.3725 (var=0.0118) (Δ vs real +0.2012)
  - iter 2: 0.2906 (var=0.0081) (Δ vs real +0.1194)
  - iter 3: 0.3556 (var=0.0059) (Δ vs real +0.1844)
  - iter 4: 0.2956 (var=0.0080) (Δ vs real +0.1244)
  - iter 5: 0.2687 (var=0.0082) (Δ vs real +0.0975)
  - iter 6: 0.3344 (var=0.0082) (Δ vs real +0.1631)
  - iter 7: 0.2275 (var=0.0064) (Δ vs real +0.0562)
  - iter 8: 0.2756 (var=0.0059) (Δ vs real +0.1044)
  - iter 9 (current): 0.2781 (var=0.0073) (Δ vs real +0.1069)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7800 (var=0.0127)
- pi_3: 0.2650 (var=0.0077)
- pi_1: 0.8450 (var=0.0106)
- pi_2: 0.8638 (var=0.0097)
- pi_4: 0.3144 (var=0.0590)
- pi_5: 0.2125 (var=0.0050)
- pi_6: 0.1788 (var=0.0107)
- pi_8: 0.1313 (var=0.0048)
- pi_9: 0.2231 (var=0.0108)
- pi_10: 0.1781 (var=0.0070)
- pi_11: 0.1200 (var=0.0052)
- pi_12: 0.5169 (var=0.0476)
- pi_13: 0.1919 (var=0.0066)
- pi_14: 0.5112 (var=0.0056)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    d = data.copy()
    if len(d) == 0:
        return float('nan')

    # The competing theory's contextual calibration has reached full weight
    # well before this point; the advocated theory has no history dependence.
    d['_trial_index'] = d.groupby('subject_id', sort=False).cumcount()
    d = d.loc[d['_trial_index'] >= 24].copy()
    if len(d) == 0:
        return float('nan')

    def terminal_match(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_diff = a[-1] - b[-1]
        if terminal_diff > 0:
            terminal_response = 0
        elif terminal_diff < 0:
            terminal_response = 1
        else:
            return np.nan
        return float(int(row['response']) == terminal_response)

    d['_terminal_match'] = d.apply(terminal_match, axis=1)
    d['_stimulus_key'] = d.apply(
        lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])),
        axis=1
    )

    # Equal-weight subject-by-stimulus cells, eliminating variance caused by
    # random trial order and unequal late-block counts of the six pairs.
    cells = d.groupby(
        ['subject_id', '_stimulus_key'], sort=False
    )['_terminal_match'].mean()
    return float(cells.mean())
```

**Observed (real) value:** 0.8810 (var=0.0070)
**Candidate trajectory (this loop):**
  - iter 1: 0.6422 (var=0.0031) (Δ vs real -0.2388)
  - iter 2: 0.6511 (var=0.0027) (Δ vs real -0.2299)
  - iter 3: 0.6173 (var=0.0041) (Δ vs real -0.2637)
  - iter 4: 0.6393 (var=0.0040) (Δ vs real -0.2416)
  - iter 5: 0.6278 (var=0.0029) (Δ vs real -0.2532)
  - iter 6: 0.6308 (var=0.0044) (Δ vs real -0.2501)
  - iter 7: 0.6281 (var=0.0038) (Δ vs real -0.2529)
  - iter 8: 0.6294 (var=0.0036) (Δ vs real -0.2516)
  - iter 9 (current): 0.6498 (var=0.0046) (Δ vs real -0.2312)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7201 (var=0.0040)
- pi_8: 0.3399 (var=0.0051)
- pi_1: 0.1647 (var=0.0106)
- pi_2: 0.4955 (var=0.0024)
- pi_4: 0.4758 (var=0.0129)
- pi_5: 0.6957 (var=0.0044)
- pi_6: 0.7470 (var=0.0062)
- pi_7: 0.5510 (var=0.0027)
- pi_9: 0.7048 (var=0.0044)
- pi_10: 0.8495 (var=0.0032)
- pi_11: 0.8124 (var=0.0029)
- pi_12: 0.6105 (var=0.0030)
- pi_13: 0.8103 (var=0.0027)
- pi_14: 0.4336 (var=0.0032)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify each trial structurally and code whether the response followed
    # the physically terminal discriminating reason. In this design that
    # reason is also the final reason in instructed-validity scan order.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        idx = np.flatnonzero(diff != 0)
        if idx.size == 0:
            continue

        terminal_idx = int(idx[-1])
        terminal_is_b = bool(diff[terminal_idx] < 0)
        chose_terminal = float(int(row["response"]) == int(terminal_is_b))

        if idx.size > 1:
            largest_gap = int(np.max(np.diff(idx) - 1))
        else:
            largest_gap = 0
        foregrounded = bool(terminal_idx == (len(diff) - 1) and largest_gap > 0)
        records.append((foregrounded, terminal_is_b, chose_terminal))

    if not records:
        return float("nan")

    frame = pd.DataFrame(
        records, columns=["foregrounded", "terminal_is_b", "chose_terminal"]
    )

    # Compute the foregrounding contrast separately for terminal-A and
    # terminal-B trials, then average. This exactly removes any stable A/B
    # response-side preference rather than allowing it to inflate variance.
    side_contrasts = []
    for terminal_is_b in (False, True):
        side = frame[frame["terminal_is_b"] == terminal_is_b]
        fg = side.loc[side["foregrounded"], "chose_terminal"]
        clustered = side.loc[~side["foregrounded"], "chose_terminal"]
        if len(fg) > 0 and len(clustered) > 0:
            side_contrasts.append(float(fg.mean() - clustered.mean()))

    if not side_contrasts:
        return float("nan")
    return float(np.mean(side_contrasts))
```

**Observed (real) value:** 0.0192 (var=0.0062)
**Candidate trajectory (this loop):**
  - iter 1: 0.0758 (var=0.0084) (Δ vs real +0.0567)
  - iter 2: 0.0112 (var=0.0114) (Δ vs real -0.0079)
  - iter 3: 0.0321 (var=0.0097) (Δ vs real +0.0129)
  - iter 4: 0.0008 (var=0.0060) (Δ vs real -0.0183)
  - iter 5: 0.0417 (var=0.0127) (Δ vs real +0.0225)
  - iter 6: 0.0350 (var=0.0078) (Δ vs real +0.0158)
  - iter 7: 0.0633 (var=0.0070) (Δ vs real +0.0442)
  - iter 8: 0.0129 (var=0.0096) (Δ vs real -0.0063)
  - iter 9 (current): -0.0158 (var=0.0112) (Δ vs real -0.0350)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0863 (var=0.0106)
- pi_3: -0.0217 (var=0.0090)
- pi_1: -0.0004 (var=0.0063)
- pi_2: -0.0008 (var=0.0037)
- pi_4: -0.0146 (var=0.0066)
- pi_5: 0.0054 (var=0.0094)
- pi_6: -0.0137 (var=0.0073)
- pi_7: -0.0042 (var=0.0041)
- pi_9: 0.0017 (var=0.0085)
- pi_10: -0.0117 (var=0.0094)
- pi_11: -0.0104 (var=0.0091)
- pi_12: 0.0742 (var=0.0074)
- pi_13: 0.0525 (var=0.0055)
- pi_14: -0.0733 (var=0.0126)

### Experiment 15
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Winner-choice rate at each unanimous evidence load.
    correct_by_load = {2: [], 4: [], 7: [], 10: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        load = int(np.count_nonzero(diff))
        net_direction = float(np.sum(diff))

        if load not in correct_by_load or net_direction == 0.0:
            continue

        response = int(row["response"])
        favored_response = 0 if net_direction > 0.0 else 1
        correct_by_load[load].append(float(response == favored_response))

    rates = {}
    for load in (2, 4, 7, 10):
        values = correct_by_load[load]
        if len(values) == 0:
            return float("nan")
        rates[load] = float(np.mean(values))

    # Endpoint-emphasized, zero-sum decline contrast. Positive values indicate
    # that unanimous evidence becomes less effective as more reasons are added.
    return float(
        0.75 * rates[2]
        + 0.25 * rates[4]
        - 0.25 * rates[7]
        - 0.75 * rates[10]
    )
```

**Observed (real) value:** 0.0471 (var=0.0135)
**Candidate trajectory (this loop):**
  - iter 1: 0.0471 (var=0.0066) (Δ vs real +0.0000)
  - iter 2: 0.0167 (var=0.0077) (Δ vs real -0.0304)
  - iter 3: 0.0485 (var=0.0144) (Δ vs real +0.0015)
  - iter 4: 0.0371 (var=0.0059) (Δ vs real -0.0100)
  - iter 5: 0.0217 (var=0.0072) (Δ vs real -0.0254)
  - iter 6: 0.0504 (var=0.0083) (Δ vs real +0.0033)
  - iter 7: 0.0275 (var=0.0084) (Δ vs real -0.0196)
  - iter 8: 0.0387 (var=0.0120) (Δ vs real -0.0083)
  - iter 9 (current): 0.0325 (var=0.0097) (Δ vs real -0.0146)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1525 (var=0.0093)
- pi_9: 0.0019 (var=0.0017)
- pi_1: -0.0021 (var=0.0054)
- pi_2: -0.0167 (var=0.0071)
- pi_4: -0.0006 (var=0.0027)
- pi_5: 0.1579 (var=0.0074)
- pi_6: 0.3079 (var=0.0057)
- pi_7: 0.2092 (var=0.0108)
- pi_8: 0.0406 (var=0.0094)
- pi_10: 0.0652 (var=0.0056)
- pi_11: 0.0015 (var=0.0017)
- pi_12: -0.0058 (var=0.0023)
- pi_13: 0.0102 (var=0.0048)
- pi_14: 0.0292 (var=0.0079)

### Experiment 16
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51], dtype=float)
    reliability = np.log(np.clip(validities, 0.5001, 0.999) / (1.0 - np.clip(validities, 0.5001, 0.999)))
    scale = float(np.mean(reliability[reliability > 0.0]))
    strengths = np.power(np.clip(reliability / scale, 0.15, 6.0), 0.30)

    def summarize(a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        idx = np.flatnonzero(d != 0.0)
        if idx.size == 0:
            return None
        dirs = np.sign(d[idx])
        m = int(dirs.size)
        switches = np.flatnonzero(dirs[1:] != dirs[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if dirs[k] == dirs[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        accumulator = 0.0
        susceptibility = []
        k = 0
        while k < m:
            direction = float(dirs[k])
            end = k + 1
            while end < m and dirs[end] == direction:
                end += 1
            incoming = strengths[idx[k:end]]
            run_length = end - k
            if accumulator != 0.0 and np.sign(accumulator) != direction:
                incumbent = abs(accumulator) * (0.55 ** run_length)
                attack = 0.0
                for s in incoming:
                    attack = 0.55 * attack + float(s)
                advantage = (attack - incumbent) / max(attack + incumbent, 1e-12)
                susceptibility.append(1.0 / (1.0 + np.exp(-4.0 * np.clip(advantage, -15.0, 15.0))))
            for s in incoming:
                accumulator = 0.55 * accumulator + direction * float(s)
            k = end

        reversal_susceptibility = float(np.mean(susceptibility)) if susceptibility else 0.35
        fractions = switches.astype(float) / float(m) if switches.size else np.asarray([], dtype=float)
        canonical = tuple(int(x) for x in (dirs * dirs[0]))
        return {
            'terminal_singleton': float(len(run_lengths) > 1 and run_lengths[-1] == 1),
            'two_run': float(len(run_lengths) == 2),
            'switch_density': float(switches.size) / float(max(1, m - 1)),
            'susceptibility': reversal_susceptibility,
            'fractions': fractions,
            'canonical': canonical,
            'terminal_direction': float(dirs[-1])
        }

    def context_value(summaries, learning_rate):
        ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
        w = np.power(1.0 - learning_rate, ages)
        if len(summaries) > 1:
            w[-1] *= 0.35
        w = w / max(float(np.sum(w)), 1e-12)

        terminal = float(np.sum([wi * s['terminal_singleton'] for wi, s in zip(w, summaries)]))
        two_run = float(np.sum([wi * s['two_run'] for wi, s in zip(w, summaries)]))
        switch_density = float(np.sum([wi * s['switch_density'] for wi, s in zip(w, summaries)]))
        susceptibility = float(np.sum([wi * s['susceptibility'] for wi, s in zip(w, summaries)]))

        hist = np.zeros(5, dtype=float)
        mass = 0.0
        for wi, s in zip(w, summaries):
            fr = s['fractions']
            if fr.size == 0:
                continue
            contribution = float(wi) / float(fr.size)
            for f in fr:
                j = min(4, max(0, int(np.floor(5.0 * float(f)))))
                hist[j] += contribution
                mass += contribution
        if mass > 0.0:
            p = hist / mass
            concentration = float(np.sum(p ** 2))
            positive = p[p > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(5.0))
        else:
            concentration = 0.2
            entropy = 1.0

        validity_gradient = float(np.clip(np.std(reliability) / max(np.mean(reliability), 1e-8), 0.0, 2.0))
        score = (1.30
                 + 1.20 * (terminal - 0.30)
                 + 1.20 * (concentration - 0.30)
                 + 0.80 * (two_run - 0.50)
                 - 1.50 * (entropy - 0.70)
                 - 0.70 * (switch_density - 0.25)
                 - 0.20 * validity_gradient
                 + 0.85 * (susceptibility - 0.35))
        return 1.0 / (1.0 + np.exp(-1.90 * np.clip(score - 1.30, -30.0, 30.0)))

    def one_subject(frame):
        summaries = []
        records = []
        for _, row in frame.iterrows():
            s = summarize(row['option_a_ratings'], row['option_b_ratings'])
            if s is None:
                continue
            summaries.append(s)

            # Average across the advocated learning-rate range to reduce
            # sensitivity to unknown individual parameter values.
            x = float(np.mean([context_value(summaries, lr) for lr in (0.05, 0.085, 0.12)]))

            # Retain conflict sequences only. The three canonical geometries
            # are +-----, +++-, and +++++- (including their A/B reversals).
            if len(s['canonical']) > 1 and len(set(s['canonical'])) > 1:
                chosen_direction = 1.0 if int(row['response']) == 0 else -1.0
                y = 1.0 if chosen_direction == s['terminal_direction'] else -1.0
                records.append((s['canonical'], x, y))

        if len(records) < 4:
            return 0.0

        residual_x = np.zeros(len(records), dtype=float)
        keys = [r[0] for r in records]
        xvals = np.asarray([r[1] for r in records], dtype=float)
        yvals = np.asarray([r[2] for r in records], dtype=float)

        for key in set(keys):
            loc = np.asarray([k == key for k in keys], dtype=bool)
            residual_x[loc] = xvals[loc] - float(np.mean(xvals[loc]))

        rms = float(np.sqrt(np.mean(residual_x ** 2)))
        if not np.isfinite(rms) or rms < 1e-10:
            return 0.0
        z = residual_x / rms
        return float(np.mean(z * yvals))

    if 'subject_id' in data.columns:
        values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]
        return float(np.mean(values)) if values else 0.0
    return one_subject(data)

```

**Observed (real) value:** 0.0311 (var=0.0170)
**Candidate trajectory (this loop):**
  - iter 1: 0.0089 (var=0.0095) (Δ vs real -0.0222)
  - iter 2: 0.0056 (var=0.0114) (Δ vs real -0.0255)
  - iter 3: -0.0029 (var=0.0082) (Δ vs real -0.0340)
  - iter 4: -0.0121 (var=0.0133) (Δ vs real -0.0432)
  - iter 5: -0.0102 (var=0.0096) (Δ vs real -0.0413)
  - iter 6: 0.0081 (var=0.0143) (Δ vs real -0.0230)
  - iter 7: 0.0076 (var=0.0069) (Δ vs real -0.0235)
  - iter 8: 0.0098 (var=0.0146) (Δ vs real -0.0213)
  - iter 9 (current): 0.0019 (var=0.0109) (Δ vs real -0.0292)
**Other theories' values on this metric (for reference):**
- pi_9: 0.1406 (var=0.0138)
- pi_3: -0.0111 (var=0.0098)
- pi_1: -0.0000 (var=0.0077)
- pi_2: -0.0071 (var=0.0059)
- pi_4: 0.0090 (var=0.0080)
- pi_5: 0.0102 (var=0.0090)
- pi_6: 0.0001 (var=0.0112)
- pi_7: -0.0004 (var=0.0066)
- pi_8: -0.0155 (var=0.0095)
- pi_10: 0.0122 (var=0.0056)
- pi_11: -0.0004 (var=0.0147)
- pi_12: 0.0069 (var=0.0058)
- pi_13: 0.0022 (var=0.0104)
- pi_14: 0.0170 (var=0.0111)

### Experiment 17
**Design**
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    coherent_terminal = []
    fragmented_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        directions = np.sign(diff[diff != 0])
        m = int(directions.size)

        if m not in (5, 6):
            continue

        # The diagnostic displays have a singleton terminal reversal.
        terminal_singleton = bool(
            m >= 2 and directions[-1] != directions[-2]
        )
        if not terminal_singleton:
            continue

        terminal_direction = float(directions[-1])
        response = int(row["response"])
        chose_terminal_option = float(
            (terminal_direction > 0 and response == 0)
            or (terminal_direction < 0 and response == 1)
        )

        prefix = directions[:-1]
        coherent = bool(
            prefix.size > 0
            and np.all(prefix == prefix[0])
            and prefix[0] != terminal_direction
        )

        if coherent:
            coherent_terminal.append(chose_terminal_option)
        else:
            fragmented_terminal.append(chose_terminal_option)

    if len(coherent_terminal) == 0 or len(fragmented_terminal) == 0:
        return float("nan")

    raw_contrast = (
        float(np.mean(coherent_terminal))
        - float(np.mean(fragmented_terminal))
    )

    # A 10-percentage-point dead band treats smaller differences as negligible;
    # effects of 30 percentage points or more receive the ceiling score.
    return float(np.clip((raw_contrast - 0.10) / 0.20, 0.0, 1.0))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0003) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0110) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0058) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0010) (Δ vs real +0.0000)
  - iter 9 (current): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.0227)
- pi_10: 1.0000 (var=0.0480)
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.0000 (var=0.0011)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0773)
- pi_6: 0.0000 (var=0.0043)
- pi_7: 0.0000 (var=0.0000)
- pi_8: 0.0000 (var=0.0013)
- pi_9: 0.0000 (var=0.0000)
- pi_11: 0.0000 (var=0.0000)
- pi_12: 0.0000 (var=0.0000)
- pi_13: 0.0000 (var=0.0054)
- pi_14: 0.0000 (var=0.0031)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code whether the response selected the option favored by the final
    # discriminating expert in descending-validity order. In this experiment,
    # feature columns are already ordered from highest to lowest validity.
    by_load = {4: [], 5: [], 6: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        load = int(discriminating.size)
        if load not in by_load:
            continue

        final_index = int(discriminating[-1])
        terminal_option = 0 if diff[final_index] > 0 else 1
        terminal_chosen = float(int(row["response"]) == terminal_option)
        by_load[load].append(terminal_chosen)

    if any(len(by_load[m]) == 0 for m in (4, 5, 6, 7)):
        return float("nan")

    p = {m: float(np.mean(by_load[m])) for m in (4, 5, 6, 7)}
    # Average of the upward boundary jump (4->5) and downward boundary
    # jump in terminal choice (6->7).
    return float(0.5 * ((p[5] - p[4]) + (p[6] - p[7])))
```

**Observed (real) value:** 0.0262 (var=0.0075)
**Candidate trajectory (this loop):**
  - iter 1: -0.0163 (var=0.0117) (Δ vs real -0.0425)
  - iter 2: -0.0537 (var=0.0125) (Δ vs real -0.0800)
  - iter 3: -0.0331 (var=0.0129) (Δ vs real -0.0594)
  - iter 4: -0.0350 (var=0.0148) (Δ vs real -0.0612)
  - iter 5: -0.0419 (var=0.0178) (Δ vs real -0.0681)
  - iter 6: -0.0194 (var=0.0116) (Δ vs real -0.0456)
  - iter 7: -0.0588 (var=0.0149) (Δ vs real -0.0850)
  - iter 8: -0.0619 (var=0.0113) (Δ vs real -0.0881)
  - iter 9 (current): -0.0881 (var=0.0142) (Δ vs real -0.1144)
**Other theories' values on this metric (for reference):**
- pi_10: 0.2694 (var=0.0251)
- pi_3: -0.0175 (var=0.0140)
- pi_1: -0.0075 (var=0.0060)
- pi_2: 0.0087 (var=0.0069)
- pi_4: -0.0919 (var=0.0133)
- pi_5: -0.0750 (var=0.0168)
- pi_6: 0.0356 (var=0.0134)
- pi_7: -0.1919 (var=0.0090)
- pi_8: 0.0144 (var=0.0064)
- pi_9: 0.0100 (var=0.0112)
- pi_11: -0.0281 (var=0.0122)
- pi_12: -0.0638 (var=0.0107)
- pi_13: 0.0250 (var=0.0116)
- pi_14: 0.0169 (var=0.0134)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Select sequences whose recency-weighted evidence strongly supports the
    # terminal reason. These are trials 1, 7, 8 and their A/B reversals in the
    # specified design, without relying on trial numbers or presentation order.
    retention_probe = 0.46
    conformity = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        directions = np.sign(diff[diff != 0.0])
        if directions.size == 0:
            continue

        terminal_direction = float(directions[-1])
        m = int(directions.size)
        aligned_field = 0.0
        for pos, direction in enumerate(directions):
            lag = (m - 1) - pos
            aligned_field += (
                terminal_direction
                * float(direction)
                * retention_probe ** float(lag)
            )

        if aligned_field >= 1.0:
            terminal_response = 0 if terminal_direction > 0.0 else 1
            conformity.append(float(int(row["response"]) == terminal_response))

    if not conformity:
        return float("nan")
    return float(np.mean(conformity))
```

**Observed (real) value:** 0.4978 (var=0.0077)
**Candidate trajectory (this loop):**
  - iter 1: 0.6461 (var=0.0068) (Δ vs real +0.1483)
  - iter 2: 0.5033 (var=0.0213) (Δ vs real +0.0056)
  - iter 3: 0.5094 (var=0.0153) (Δ vs real +0.0117)
  - iter 4: 0.5283 (var=0.0187) (Δ vs real +0.0306)
  - iter 5: 0.4561 (var=0.0177) (Δ vs real -0.0417)
  - iter 6: 0.5711 (var=0.0095) (Δ vs real +0.0733)
  - iter 7: 0.4617 (var=0.0085) (Δ vs real -0.0361)
  - iter 8: 0.4950 (var=0.0123) (Δ vs real -0.0028)
  - iter 9 (current): 0.5294 (var=0.0150) (Δ vs real +0.0317)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6767 (var=0.0091)
- pi_11: 0.9267 (var=0.0028)
- pi_1: 0.1572 (var=0.0093)
- pi_2: 0.1372 (var=0.0090)
- pi_4: 0.4606 (var=0.0093)
- pi_5: 0.5828 (var=0.0078)
- pi_6: 0.7239 (var=0.0096)
- pi_7: 0.7239 (var=0.0070)
- pi_8: 0.6833 (var=0.0052)
- pi_9: 0.7178 (var=0.0112)
- pi_10: 0.8167 (var=0.0065)
- pi_12: 0.7150 (var=0.0117)
- pi_13: 0.8639 (var=0.0037)
- pi_14: 0.4928 (var=0.0087)

### Experiment 20
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Weak- versus strong-terminal increase in choosing the three-cue coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    weak_choices = []
    strong_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 4:
            continue

        # In this design, the first three discriminating experts constitute
        # the repeated coalition and the last one is the opposing terminal cue.
        terminal_idx = int(discriminating[-1])
        coalition_supports_a = bool(diff[discriminating[0]] > 0)
        response = int(row["response"])
        chose_coalition = float(
            (response == 0 and coalition_supports_a)
            or (response == 1 and not coalition_supports_a)
        )

        # Strong terminals: validities .86 and .70 (indices 3 and 5).
        # Weak terminals: validities .505 and .50 (indices 10 and 11).
        if terminal_idx in (3, 5):
            strong_choices.append(chose_coalition)
        elif terminal_idx in (10, 11):
            weak_choices.append(chose_coalition)

    if len(weak_choices) == 0 or len(strong_choices) == 0:
        return float("nan")

    return float(np.mean(weak_choices) - np.mean(strong_choices))
```

**Observed (real) value:** -0.1813 (var=0.0102)
**Candidate trajectory (this loop):**
  - iter 1: -0.1337 (var=0.0163) (Δ vs real +0.0475)
  - iter 2: -0.1694 (var=0.0170) (Δ vs real +0.0119)
  - iter 3: -0.1387 (var=0.0147) (Δ vs real +0.0425)
  - iter 4: -0.1450 (var=0.0201) (Δ vs real +0.0363)
  - iter 5: -0.2500 (var=0.0148) (Δ vs real -0.0687)
  - iter 6: -0.1219 (var=0.0164) (Δ vs real +0.0594)
  - iter 7: -0.3437 (var=0.0165) (Δ vs real -0.1625)
  - iter 8: -0.1475 (var=0.0180) (Δ vs real +0.0338)
  - iter 9 (current): -0.1469 (var=0.0121) (Δ vs real +0.0344)
**Other theories' values on this metric (for reference):**
- pi_11: 0.1394 (var=0.0177)
- pi_3: 0.0013 (var=0.0091)
- pi_1: -0.0125 (var=0.0080)
- pi_2: 0.0194 (var=0.0043)
- pi_4: 0.0775 (var=0.0099)
- pi_5: -0.0019 (var=0.0095)
- pi_6: 0.1275 (var=0.0180)
- pi_7: 0.1931 (var=0.0170)
- pi_8: -0.0681 (var=0.0101)
- pi_9: 0.2087 (var=0.0183)
- pi_10: 0.2712 (var=0.0228)
- pi_12: -0.0944 (var=0.0201)
- pi_13: -0.1356 (var=0.0110)
- pi_14: -0.0481 (var=0.0156)

### Experiment 21
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    endpoint_choices = {0: [], 19: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # In this design, one option carries the singleton 96%-valid cue,
        # whereas the other carries four lower-validity cues.
        if np.sum(a) == 1 and np.sum(b) == 4:
            singleton = a
            chose_singleton = float(response == 0)
        elif np.sum(b) == 1 and np.sum(a) == 4:
            singleton = b
            chose_singleton = float(response == 1)
        else:
            continue

        positions = np.flatnonzero(singleton != 0)
        if positions.size == 1:
            pos = int(positions[0])
            if pos in endpoint_choices:
                endpoint_choices[pos].append(chose_singleton)

    if len(endpoint_choices[0]) == 0 or len(endpoint_choices[19]) == 0:
        return float("nan")

    left_rate = float(np.mean(endpoint_choices[0]))
    right_rate = float(np.mean(endpoint_choices[19]))
    return float(abs(right_rate - left_rate))
```

**Observed (real) value:** 0.3900 (var=0.0214)
**Candidate trajectory (this loop):**
  - iter 1: 0.1567 (var=0.0095) (Δ vs real -0.2333)
  - iter 2: 0.0350 (var=0.0079) (Δ vs real -0.3550)
  - iter 3: 0.0850 (var=0.0122) (Δ vs real -0.3050)
  - iter 4: 0.0892 (var=0.0092) (Δ vs real -0.3008)
  - iter 5: 0.0375 (var=0.0038) (Δ vs real -0.3525)
  - iter 6: 0.0733 (var=0.0087) (Δ vs real -0.3167)
  - iter 7: 0.0875 (var=0.0096) (Δ vs real -0.3025)
  - iter 8: 0.0567 (var=0.0061) (Δ vs real -0.3333)
  - iter 9 (current): 0.0433 (var=0.0079) (Δ vs real -0.3467)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0167 (var=0.0062)
- pi_12: 0.4200 (var=0.0491)
- pi_1: 0.0025 (var=0.0042)
- pi_2: 0.0150 (var=0.0038)
- pi_4: 0.6633 (var=0.0518)
- pi_5: 0.5100 (var=0.0281)
- pi_6: 0.0033 (var=0.0029)
- pi_7: 0.0125 (var=0.0042)
- pi_8: 0.0675 (var=0.0094)
- pi_9: 0.0258 (var=0.0051)
- pi_10: 0.0183 (var=0.0024)
- pi_11: 0.0142 (var=0.0025)
- pi_13: 0.4867 (var=0.0277)
- pi_14: 0.0342 (var=0.0077)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.86, 0.98, 0.98, 0.86,
        0.53, 0.86, 0.86, 0.53,
        0.98, 0.53, 0.53, 0.98,
        0.52, 0.52, 0.52, 0.52
    ], dtype=float)

    large_drop_choices = []
    small_drop_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0.0)
        if discriminating.size < 2:
            continue

        # Coalitions are defined in displayed (left-to-right) order. In this
        # design the final discriminating cue is the terminal singleton and
        # the preceding discriminating cue is the incumbent coalition's exit.
        terminal_index = int(discriminating[-1])
        exit_index = int(discriminating[-2])
        boundary_drop = float(validities[exit_index] - validities[terminal_index])

        response = int(row["response"])
        terminal_favors_a = bool(diff[terminal_index] > 0.0)
        chose_terminal = float(
            (terminal_favors_a and response == 0)
            or ((not terminal_favors_a) and response == 1)
        )

        if boundary_drop > 0.20:
            large_drop_choices.append(chose_terminal)
        else:
            small_drop_choices.append(chose_terminal)

    if len(large_drop_choices) == 0 or len(small_drop_choices) == 0:
        return 0.0

    return float(
        np.mean(large_drop_choices) - np.mean(small_drop_choices)
    )
```

**Observed (real) value:** 0.1750 (var=0.0208)
**Candidate trajectory (this loop):**
  - iter 1: 0.1321 (var=0.0108) (Δ vs real -0.0429)
  - iter 2: 0.1871 (var=0.0111) (Δ vs real +0.0121)
  - iter 3: 0.1283 (var=0.0103) (Δ vs real -0.0467)
  - iter 4: 0.1308 (var=0.0096) (Δ vs real -0.0442)
  - iter 5: 0.2312 (var=0.0134) (Δ vs real +0.0562)
  - iter 6: 0.1613 (var=0.0114) (Δ vs real -0.0137)
  - iter 7: 0.1571 (var=0.0146) (Δ vs real -0.0179)
  - iter 8: 0.2004 (var=0.0109) (Δ vs real +0.0254)
  - iter 9 (current): 0.1696 (var=0.0118) (Δ vs real -0.0054)
**Other theories' values on this metric (for reference):**
- pi_12: 0.1642 (var=0.0154)
- pi_3: 0.0117 (var=0.0066)
- pi_1: -0.0187 (var=0.0035)
- pi_2: 0.0138 (var=0.0047)
- pi_4: -0.0846 (var=0.0080)
- pi_5: -0.0050 (var=0.0066)
- pi_6: 0.0062 (var=0.0081)
- pi_7: -0.0021 (var=0.0084)
- pi_8: 0.0458 (var=0.0077)
- pi_9: -0.0042 (var=0.0100)
- pi_10: 0.0050 (var=0.0060)
- pi_11: 0.0175 (var=0.0089)
- pi_13: 0.1788 (var=0.0082)
- pi_14: 0.1617 (var=0.0104)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response follows the physically last discriminating
    # cue, then contrast crossover conditions 1-2 against conditions 3-4.
    condition_scores = {0: [], 1: [], 2: [], 3: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # In this design, the leftmost discriminating location (0, 1, 2, or 3)
        # uniquely identifies the reliability-crossover condition, including
        # the response-side-reversed versions of each pair.
        condition = int(discriminating[0])
        if condition not in condition_scores:
            continue

        last_direction = diff[int(discriminating[-1])]
        terminal_response = 0 if last_direction > 0 else 1
        follows_terminal = 1.0 if int(row["response"]) == terminal_response else -1.0
        condition_scores[condition].append(follows_terminal)

    means = {}
    for condition, values in condition_scores.items():
        if len(values) == 0:
            return float("nan")
        means[condition] = float(np.mean(values))

    early_reliability_regime = 0.5 * (means[0] + means[1])
    late_reliability_regime = 0.5 * (means[2] + means[3])
    return float(early_reliability_regime - late_reliability_regime)

```

**Observed (real) value:** 0.4183 (var=0.0370)
**Candidate trajectory (this loop):**
  - iter 1: 0.0575 (var=0.0299) (Δ vs real -0.3608)
  - iter 2: 0.3483 (var=0.1068) (Δ vs real -0.0700)
  - iter 3: 0.2192 (var=0.0908) (Δ vs real -0.1992)
  - iter 4: 0.2217 (var=0.1223) (Δ vs real -0.1967)
  - iter 5: 0.3808 (var=0.0945) (Δ vs real -0.0375)
  - iter 6: 0.1092 (var=0.0548) (Δ vs real -0.3092)
  - iter 7: 0.3008 (var=0.1084) (Δ vs real -0.1175)
  - iter 8: 0.3758 (var=0.0839) (Δ vs real -0.0425)
  - iter 9 (current): 0.3908 (var=0.0981) (Δ vs real -0.0275)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8758 (var=0.0591)
- pi_13: -0.0642 (var=0.0281)
- pi_1: -1.3967 (var=0.2530)
- pi_2: 0.0075 (var=0.0433)
- pi_4: -0.7858 (var=0.0648)
- pi_5: -0.0275 (var=0.0273)
- pi_6: 1.2750 (var=0.0699)
- pi_7: 1.1083 (var=0.0467)
- pi_8: 0.3808 (var=0.0414)
- pi_9: 1.4067 (var=0.0637)
- pi_10: 1.4275 (var=0.0498)
- pi_11: 1.7375 (var=0.0134)
- pi_12: -0.2100 (var=0.0458)
- pi_14: 0.0075 (var=0.0288)

### Experiment 24
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    scores = []
    for a_ratings, b_ratings, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_ratings)
        b = np.asarray(b_ratings)
        terminal_diff = float(a[-1] - b[-1])
        if terminal_diff == 0.0:
            continue
        terminal_supported_response = 0 if terminal_diff > 0.0 else 1
        scores.append(float(int(response) == terminal_supported_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4192 (var=0.0032)
**Candidate trajectory (this loop):**
  - iter 1: 0.6890 (var=0.0018) (Δ vs real +0.2698)
  - iter 2: 0.6585 (var=0.0148) (Δ vs real +0.2394)
  - iter 3: 0.6069 (var=0.0148) (Δ vs real +0.1877)
  - iter 4: 0.6400 (var=0.0212) (Δ vs real +0.2208)
  - iter 5: 0.6740 (var=0.0187) (Δ vs real +0.2548)
  - iter 6: 0.7033 (var=0.0033) (Δ vs real +0.2842)
  - iter 7: 0.6608 (var=0.0113) (Δ vs real +0.2417)
  - iter 8: 0.6081 (var=0.0203) (Δ vs real +0.1890)
  - iter 9 (current): 0.6327 (var=0.0138) (Δ vs real +0.2135)
**Other theories' values on this metric (for reference):**
- pi_13: 0.8050 (var=0.0021)
- pi_3: 0.1706 (var=0.0019)
- pi_1: 0.8556 (var=0.0080)
- pi_2: 0.4990 (var=0.0021)
- pi_4: 0.9263 (var=0.0011)
- pi_5: 0.8629 (var=0.0031)
- pi_6: 0.1717 (var=0.0039)
- pi_7: 0.1848 (var=0.0026)
- pi_8: 0.1258 (var=0.0022)
- pi_9: 0.2177 (var=0.0041)
- pi_10: 0.1810 (var=0.0035)
- pi_11: 0.2227 (var=0.0048)
- pi_12: 0.8950 (var=0.0021)
- pi_14: 0.4319 (var=0.0023)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    weak_cue_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # In this design, Expert 10 (index 9) is the 99%-valid cue.
        # The other discriminating cue is one of the 52%-valid experts.
        weak_indices = np.flatnonzero(diff[:9] != 0.0)
        if weak_indices.size != 1:
            continue

        weak_direction = diff[int(weak_indices[0])]
        weak_supported_response = 0 if weak_direction > 0.0 else 1
        weak_cue_choices.append(
            float(int(row["response"]) == weak_supported_response)
        )

    if len(weak_cue_choices) == 0:
        return float("nan")
    return float(np.mean(weak_cue_choices))
```

**Observed (real) value:** 0.5908 (var=0.0015)
**Candidate trajectory (this loop):**
  - iter 1: 0.2527 (var=0.0053) (Δ vs real -0.3381)
  - iter 2: 0.4540 (var=0.0106) (Δ vs real -0.1369)
  - iter 3: 0.4085 (var=0.0135) (Δ vs real -0.1823)
  - iter 4: 0.4517 (var=0.0043) (Δ vs real -0.1392)
  - iter 5: 0.4910 (var=0.0090) (Δ vs real -0.0998)
  - iter 6: 0.4462 (var=0.0034) (Δ vs real -0.1446)
  - iter 7: 0.4519 (var=0.0084) (Δ vs real -0.1390)
  - iter 8: 0.4804 (var=0.0076) (Δ vs real -0.1104)
  - iter 9 (current): 0.4754 (var=0.0081) (Δ vs real -0.1154)
**Other theories' values on this metric (for reference):**
- pi_3: 0.8175 (var=0.0031)
- pi_14: 0.4600 (var=0.0025)
- pi_1: 0.1633 (var=0.0090)
- pi_2: 0.4819 (var=0.0031)
- pi_4: 0.0683 (var=0.0008)
- pi_5: 0.1383 (var=0.0027)
- pi_6: 0.7940 (var=0.0074)
- pi_7: 0.8096 (var=0.0027)
- pi_8: 0.8623 (var=0.0026)
- pi_9: 0.6292 (var=0.0138)
- pi_10: 0.6706 (var=0.0227)
- pi_11: 0.7465 (var=0.0056)
- pi_12: 0.0967 (var=0.0021)
- pi_13: 0.1335 (var=0.0016)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    chose_three_cue_option = []
    for a_ratings, b_ratings, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a_count = float(np.sum(np.asarray(a_ratings, dtype=float)))
        b_count = float(np.sum(np.asarray(b_ratings, dtype=float)))
        r = int(response)
        chose_three = ((a_count > b_count) and (r == 0)) or ((b_count > a_count) and (r == 1))
        chose_three_cue_option.append(float(chose_three))

    dominance_rate = float(np.mean(chose_three_cue_option))
    return float(dominance_rate >= 0.625)

```

**Observed (real) value:** 1.0000 (var=0.2496)
**Candidate trajectory (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (current): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_14: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0564)
- pi_4: 1.0000 (var=0.1924)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 1.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.1600)
- pi_8: 1.0000 (var=0.0384)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.1600)
- pi_11: 1.0000 (var=0.0000)
- pi_12: 1.0000 (var=0.0000)
- pi_13: 1.0000 (var=0.0000)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed salience-guided sequential-integration family and improves substantially over pi_14 on several targeted findings. It captures Experiment 10 (0.849 vs 0.875), the unanimous-load decline in Experiment 15 exactly (0.047), the weak-versus-strong terminal contrast in Experiment 20 reasonably well (-0.134 vs -0.181), the local-drop contrast in Experiment 22 (0.132 vs 0.175), and the thresholded three-cue dominance result in Experiment 26. It also gets the null or near-null results in Experiments 5, 7, 16, and 17. Nevertheless, the accepted aggregate loss of 0.2072 still reflects several large, systematic failures. Recency is too weak in Experiments 3, 4, 11, and 13: terminal-following is 0.410 vs 0.575, 0.464 vs 0.665, 0.375 vs 0.843, and 0.642 vs 0.881. Conversely, terminal influence is much too strong in Experiments 8 and 24: 0.620 vs 0.172 and 0.689 vs 0.419. The candidate also retains too much highest-validity/early-cue control in Experiments 1, 2, 6, and 9. Most importantly, it misses the requested endpoint/crossover structure: Experiment 21's endpoint asymmetry is only 0.157 vs 0.390, Experiment 23's reliability crossover is 0.058 vs 0.418, and Experiment 25 predicts only 0.253 weak-cue choices where humans show 0.591. Thus the model produces generic terminal recency but does not condition that recency correctly on absolute reliability, position, and coalition geometry. Between-subject variance is also often too low, especially in Experiments 4, 21, and 26, indicating overly homogeneous deterministic priority orders.
Rationale: Keep the current mechanism family, but revise its calibration and priority computation. The central technical problem is the within-display z-normalization of reliability. In a two-cue display it maps cues to roughly +1 and -1 regardless of whether their validities are nearly equal or 0.99 versus 0.52, erasing the absolute reliability gap needed for the intermediate crossover in Experiment 25 and likely flattening Experiment 23. Replace this with an absolute, task-invariant transformation of instructed log odds using a fixed scale. Let absolute reliability affect latency more strongly than update magnitude: a highly reliable terminal cue can then be sampled early, while a weak spatially earlier challenger can arrive later and sometimes reverse it, rather than the 99%-valid endpoint automatically dominating as it currently does. Preserve graded evidence by compressing reliability-dependent update size and retaining lapse/choice temperature.

The endpoint mechanism also needs rebalancing. Increase the interaction of endpoint salience with scan direction and widen subject-level scan/endpoint heterogeneity, potentially allowing a minority of opposite-direction scanners, to raise the large endpoint asymmetry in Experiment 21 and the associated between-subject variance. Do not simply increase the universal endpoint refresh, because terminal control is already excessive in Experiments 8 and 24. Instead, make refresh probability depend more selectively on endpoint-by-scan congruence and absolute contrast, while strengthening graded coalition resistance for coherent multi-cue prefixes; this should suppress the excessive singleton control in Experiment 8 without eliminating the strong recency required in Experiments 11 and 13. A small priority-sampling temperature or latency noise would also make the advertised sampling process genuinely probabilistic and reduce the current deterministic condition profiles. Jointly retune against Experiments 3, 4, 8, 11, 13, 21, 23, 24, and 25 while preserving the already good fits in 10, 15, 20, 22, and 26.

**Outcome of your advice:** iter 1 candidate loss=0.2072 -> iter 2 candidate loss=0.1675 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed salience-guided sequential-integration family, and the accepted loss improvement from 0.2072 to 0.1675 shows that the previous absolute-reliability calibration was productive. In particular, Experiment 23 improved dramatically from 0.058 to 0.348 against 0.418, Experiment 25 from 0.253 to 0.454 against 0.591, Experiment 22 is now nearly exact (0.187 vs 0.175), and Experiments 19 and 20 are also close. Fits remain good for Experiments 10, 14, 17, 19, 20, 22, and the thresholded result in 26. However, the theory is not yet experiment-invariant. It severely underpredicts selective recency in Experiments 4 (0.438 vs 0.665), 11 (0.380 vs 0.843), and 13 (0.651 vs 0.881), while still overpredicting terminal influence in Experiments 8 (0.536 vs 0.172) and 24 (0.659 vs 0.419). Highest-validity or early-cue control remains excessive in Experiments 1, 2, 6, and 9. Most seriously, Experiment 21's endpoint asymmetry deteriorated from 0.157 to 0.035 against 0.390. Thus the new absolute scaling captures reliability crossover, but the endpoint and coalition pathways do not yet selectively translate salience into late effective evidence. Heterogeneity is also allocated incorrectly: variance is too low in Experiments 4 and 21 but much too high in Experiment 23 (0.107 vs 0.037).
Rationale: Preserve the accepted absolute log-odds scaling rather than undoing it; its large gains in Experiments 23 and 25 validate that direction. The next revision should target three implementation-level issues within the same mechanism. First, the endpoint latency term divides salience by `1 + absolute_reliability`, suppressing positional modulation precisely for the highly reliable singleton used in Experiment 21. Soften or remove that attenuation and replace the discontinuous `scan_sign` interaction with a smooth signed scan-by-endpoint term. Put the added endpoint effect primarily into latency or selective refresh, not a universal terminal update.

Second, the current refresh ceiling is only 0.32 and is insufficient for the isolated terminal singleton to overcome a four- or five-cue incumbent in Experiment 11. Permit a larger refresh specifically when a scan-congruent endpoint challenger is spatially isolated and opposes an already coherent coalition. This condition distinguishes Experiment 11—and potentially the endpoint contrast in 21—from Experiment 24, where only one incumbent cue precedes the terminal cue, and from the contiguous conflicts in Experiment 8. Strengthen the isolation-based release from anchoring under that same condition rather than weakening anchoring globally.

Third, local contrast is currently counted twice: it delays the challenger and then can multiply its update by as much as 3.2. Retain the contrast-dependent ordering/refresh needed for Experiment 22, but reduce or compress the direct surprise multiplier. This should lower the excessive terminal choice in Experiment 24 without deleting the local-drop contrast. A modest reduction in accumulator retention can then raise recency in Experiments 3, 4, and 13, provided coherent spatial coalitions—not merely consecutive cues in the sampled order—receive stronger graded protection, preventing another increase in Experiment 8. Finally, shift heterogeneity toward the endpoint/scan interaction while narrowing crossover-relevant validity/contrast dispersion, aiming to increase variance in Experiment 21 without preserving the excessive variance in Experiment 23. Tune these changes against trial-level probabilities, especially Experiments 4, 8, 11, 13, 21, 23, 24, and 25, while preserving the current gains in 19, 20, and 22.

**Outcome of your advice:** iter 2 candidate loss=0.1675 -> iter 3 candidate loss=0.1779 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed salience-guided sequential-integration family, but it was correctly rejected: aggregate loss rose from the accepted 0.1675 to 0.1779. The new endpoint-release and lower-retention bundle improved Experiment 11 (0.380 to 0.520), Experiment 21 slightly (0.035 to 0.085), and Experiment 24 (0.659 to 0.607), but these gains were outweighed by broad regressions. Recency moved farther from the data in Experiments 4 (0.438 to 0.377 versus 0.665), 10 (0.864 to 0.813 versus 0.875), and 13 (0.651 to 0.617 versus 0.881). Early-coalition control worsened in Experiments 2, 6, and 12. Previously strong calibration results also deteriorated: Experiment 20 moved from -0.169 to -0.139 versus -0.181, Experiment 22 from 0.187 to 0.128 versus 0.175, Experiment 23 from 0.348 to 0.219 versus 0.418, and Experiment 25 from 0.454 to 0.409 versus 0.591. The model still greatly overpredicts terminal control in Experiment 8 (0.533 versus 0.172) and underpredicts the endpoint asymmetry in Experiment 21 (0.085 versus 0.390). Thus the current selective endpoint challenge is somewhat effective for Experiment 11, but its accompanying global retention, contrast, anchoring, and heterogeneity changes disrupt the accepted reliability-crossover structure without generating a sufficiently large endpoint effect.
Rationale: Build the next attempt on the unchanged accepted iteration-2 base. The previous recommendation to combine lower retention, compressed direct contrast, widened scan heterogeneity, spatial anchoring, and a large challenge refresh was rejected by the gate, so do not repeat that bundle. Preserve iteration 2's absolute reliability transformation and its update/contrast calibration, which produced the strong fits in Experiments 19, 20, 22, 23, and 25.

Try a narrower intervention in the priority map. Add a smooth endpoint-by-scan latency boost only for an isolated endpoint cue belonging to a singleton coalition that opposes at least three cues. Scale it continuously by endpoint-to-nearest-opponent separation or comparable spatial isolation. This targets the separated singleton configurations in Experiments 11 and 21, while becoming negligible for the contiguous singleton reversal in Experiment 8 and being inactive for the one-versus-one displays in Experiments 24 and 25 and the balanced coalitions in Experiment 23. Put this correction in sampling latency rather than adding another large refresh or releasing anchoring; ordinary leaky accumulation can then translate the delayed sample into graded influence.

For Experiment 21, increase the population mean of the signed scan effect within this geometry-gated pathway rather than merely widening its range. The rejected widening produced too little aggregate asymmetry while leaving crossover variance excessive. A stronger but narrower directional tendency should increase the left-right mean difference and keep the endpoint-related heterogeneity localized, instead of contaminating Experiment 23.

Finally, inspect the reliability-only refresh term. Selecting a cue for refresh partly from `0.04 * absolute_reliability` can repeatedly reinstate an early high-validity cue even when there is neither endpoint salience nor a local drop. Removing or sharply reducing that baseline is an in-family way to reduce excessive early-cue control in Experiments 1, 2, 6, and 9 and strengthen ordinary recency in Experiments 4 and 13 without globally lowering accumulator retention. Retain refresh for genuine endpoint and contrast signals. Jointly verify that these surgical changes preserve iteration 2's fits in Experiments 19, 20, 22, 23, and 25 while improving 4, 11, 13, and 21 and not increasing the already excessive terminal response in Experiment 8.

**Outcome of your advice:** iter 3 candidate loss=0.1779 -> iter 4 candidate loss=0.2156 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed salience-guided sequential-integration family, but it was correctly rejected: aggregate loss increased sharply from the accepted iteration-2 value of 0.1675 to 0.2156. The geometry-gated endpoint latency correction did not deliver its intended gains. Experiment 21 endpoint asymmetry remains far too small (0.089 versus 0.390), and Experiment 11 terminal influence remains severely underpredicted (0.419 versus 0.843). Meanwhile, removal of reliability-based refresh produced broad regressions rather than reducing early-cue control: Experiment 1 rose to 0.760 versus 0.494, Experiment 9 rose to 0.201 versus -0.033, and ordinary recency remains too weak in Experiments 3, 4, and 13. The candidate also lost iteration 2's important crossover calibration: Experiment 23 fell from 0.348 to 0.222 versus 0.418, while Experiment 25 remains low at 0.452 versus 0.591. Terminal influence is still excessive in Experiments 8 and 24, at 0.493 versus 0.172 and 0.640 versus 0.419. Experiment 7 additionally crossed its threshold in the wrong direction, changing from the correct 0 to 1. Fits remain good for Experiments 10, 14, 15, 17, 19, 20, 22, and 26, but those successes do not offset the systematic failure to distinguish isolated endpoint challenges from ordinary terminal singleton conflicts. Heterogeneity is also misallocated: Experiment 23 variance is much too high (0.122 versus 0.037), whereas several recency effects remain more homogeneous than the human data.
Rationale: Because this attempt was rejected, the next proposal should build on the unchanged accepted iteration-2 base. My previous recommendation to add a large geometry-gated endpoint latency delay and remove reliability-only refresh was unsuccessful, so do not repeat either intervention. Preserve iteration 2's absolute-reliability transformation, direct contrast calibration, and small reliability-refresh component, which supported the substantially better fits in Experiments 19, 20, 22, 23, and 25.

Try a different in-family implementation correction: coalition coherence is currently constructed from the priority-sampled cue order, even though the relevant distinction is spatial organization in the parallel priority map. Compute a graded physical-run coherence statistic before sampling—using directional agreement, adjacency, and gap size—and let that statistic modulate the incumbent gate during accumulation. A contiguous majority should receive strong resistance, suppressing the excessive terminal singleton response in Experiment 8, whereas a challenger separated by a large physical gap should receive a smooth release from that resistance, helping Experiment 11. Make this a moderate gate adjustment on the accepted calibration rather than another simultaneous change to retention, contrast gain, or reliability weighting.

For Experiment 21, replace the failed signed endpoint latency offset with probabilistic priority-dependent reinspection. Sample whether and which cue is refreshed from a softmax or hazard based on endpoint-by-scan congruence, isolation, and local contrast, while keeping refresh update magnitude modest. The current deterministic maximum-score refresh and very large latency displacement can saturate ordering without producing the required graded left-right choice difference. A scan-biased refresh probability should create endpoint asymmetry and subject heterogeneity directly, without perturbing the reliability crossover in non-endpoint displays. Keep the scan distribution relatively narrow and alter its mean only within this refresh calculation, since widening or globally strengthening scan effects previously harmed Experiment 23.

Finally, explicitly regression-test the continuous condition profiles, not only thresholded metrics: preserve Experiment 7 below its threshold and the iteration-2 fits in Experiments 19, 20, 22, 23, and 25 while targeting Experiments 8, 11, and 21. If a physical-coherence gate plus probabilistic endpoint reinspection cannot improve those diagnostics in isolation, do not bundle it with global retention or contrast changes, which were already rejected.

**Outcome of your advice:** iter 4 candidate loss=0.2156 -> iter 5 candidate loss=0.1964 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed salience-guided sequential-integration family, but it was rejected appropriately: loss 0.1964 is substantially worse than the accepted iteration-2 loss of 0.1675. The new spatial-coherence and probabilistic-reinspection modifications produced a few local gains—Experiment 2 improved to 0.234 versus 0.194, Experiment 23 to 0.381 versus 0.418, and Experiment 25 to 0.491 versus 0.591—but caused major regressions elsewhere. Highest-validity control became especially excessive in Experiments 1 (0.794 versus 0.494) and 9 (0.291 versus -0.033). Recency is now markedly too weak in Experiments 3, 4, and 13, and the isolated endpoint challenger in Experiment 11 remains far too ineffective (0.419 versus 0.843). At the same time, terminal influence remains much too strong in Experiment 8 (0.518 versus 0.172) and Experiment 24 (0.674 versus 0.419). Most decisively, the proposed reinspection mechanism does not generate the required endpoint asymmetry in Experiment 21: 0.038 versus 0.390. The model therefore still lacks the required selective profile—strong recency for multi-reason sequences and isolated endpoint challenges, but moderated terminal control in ordinary two-cue and compact-majority conflicts. Variance is also misallocated: Experiment 21 is much too homogeneous, whereas Experiment 23 remains substantially more variable than the human result.
Rationale: Build on the unchanged accepted iteration-2 base. My prior recommendation to add a physical-coherence field and probabilistic endpoint reinspection was rejected by the gate, so do not repeat or elaborate that intervention. Likewise, avoid another bundled change to global retention, contrast gain, or endpoint refresh.

Try a smaller capacity-conditioned recalibration of the accepted accumulator. At present, retention is constant and capacity only attenuates updates after the nominal span is exceeded. This makes terminal weighting insufficient in the four-to-six-reason displays that drive Experiments 3, 4, 11, and 13, even though terminal influence is already excessive in the two-cue Experiment 24. Within the prescribed mechanism, let effective retention decline smoothly with the fraction of integration capacity already consumed, rather than changing the global retention range. This supplies progressively stronger recency in longer sequences while leaving two-cue conflicts comparatively unchanged. Couple this only to the existing graded incumbent coalition state: as same-direction coalition strength grows, reduce the load-induced retention decline or increase resistance smoothly. That should protect the five-versus-one compact majority in Experiment 8 while allowing the existing isolation term to release the four- or five-cue incumbent in Experiment 11. Use a modest interaction, not a categorical run rule.

Separately, inspect the endpoint asymmetry pathway as an implementation/calibration problem. The current use of `scan_sign` converts any near-zero scan parameter into a full-strength endpoint polarity, while the broad distribution includes both signs; this can create individual endpoint effects that cancel in the pooled Experiment 21 metric. On the iteration-2 base, replace only this discontinuity with a bounded continuous signed factor such as `tanh(scan_direction / scale)` and fit a consistently directed but not overly dispersed hierarchical scan distribution. Do not simultaneously alter endpoint magnitude or add another refresh process. The target is increased pooled asymmetry and variance in Experiment 21 without disturbing the reliability crossover in Experiment 23.

Regression-test especially Experiments 1, 3, 4, 8, 9, 11, 13, 21, and 24, while preserving iteration 2's strong fits in Experiments 19, 20, 22, 23, and 25. The next candidate should be accepted only if these narrow changes improve aggregate loss over 0.1675; the current candidate should not be defended or retained.

**Outcome of your advice:** iter 5 candidate loss=0.1964 -> iter 6 candidate loss=0.1729 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed salience-guided sequential-integration family, but the gate correctly rejected it: loss 0.1729 does not improve on the accepted iteration-2 loss of 0.1675. Capacity-adaptive leakage produced useful local gains in Experiment 3 (0.528 versus 0.575) and the unanimous-load decline in Experiment 15 (0.050 versus 0.047), but it did not solve the broader selective-recency problem. Recency remains too weak in Experiments 4 (0.455 versus 0.665), 11 (0.468 versus 0.843), and 13 (0.631 versus 0.881), while terminal influence remains drastically too strong for the compact majority in Experiment 8 (0.582 versus 0.172) and the two-cue display in Experiment 24 (0.703 versus 0.419). Early or highest-validity control is also excessive in Experiments 1, 2, 6, and 9. More importantly, the new changes damaged several of iteration 2's strongest results: Experiment 19 moved from 0.503 to 0.571, Experiment 20 from -0.169 to -0.122, Experiment 23 from 0.348 to 0.109, and Experiment 25 from 0.454 to 0.446. The consistently positive continuous scan distribution only raised Experiment 21's endpoint asymmetry from 0.035 to 0.073, still far below 0.390. Thus the added leakage generates some generic long-sequence recency but does not condition it adequately on coalition compactness and challenger separation, while the scan recalibration perturbs crossover behavior without creating a sufficiently strong endpoint contrast.
Rationale: Build again on the unchanged accepted iteration-2 base. My previous recommendation to add capacity-dependent retention decline and impose a consistently positive continuous scan distribution was rejected by the gate. Do not repeat the current 0.20 retention-drop schedule or the global positive-scan restriction. If load-dependent leakage is retained at all, test a much smaller isolated step—roughly one-quarter to one-half of the present decline—without changing the scan distribution, since the present bundle's gains in Experiment 3 were outweighed by regressions in 8, 12, 19, 20, 23, 24, and 25.

Use a different existing in-family knob to address the central Experiment 8 versus Experiment 11 dissociation: recalibrate the graded anchoring gate and its isolation release. On the accepted base, increase coalition resistance as a smooth function of incumbent coherence and accumulator magnitude, but make the release from that resistance substantially steeper with the physical gap between the incumbent coalition and an opposing challenger. The current linear factor `1 - 0.72 * isolation` leaves too much anchoring for the widely separated endpoint challenger in Experiment 11, while anchoring is still too weak for the contiguous five-versus-one pattern in Experiment 8. A bounded nonlinear isolation function can strongly protect compact coalitions at near-zero gaps and smoothly remove that protection at large gaps, without introducing a categorical strategy. Make this a calibration of the existing gate, not a new coherence field, new refresh process, or global retention change; those bundled interventions were already rejected.

For Experiment 21, the repeated attempts to alter global scan direction have failed and have destabilized Experiment 23. Instead inspect the scaling of the existing endpoint term itself. Its division by `1 + absolute_reliability` suppresses endpoint modulation for the highly reliable singleton used in Experiment 21. Test a modestly softened attenuation only inside the endpoint-by-scan component, while preserving iteration 2's overall scan distribution and reliability-dependent sampling. Gate this endpoint amplification continuously by opposing-coalition load so that it is strong for the singleton-versus-four configuration in Experiment 21 but negligible in the one-versus-one displays of Experiments 24 and 25. This remains the prescribed priority-map mechanism and avoids a universal terminal bonus.

Regression-test these changes independently before combining them. The priority targets are lowering Experiment 8, raising Experiments 11 and 21, and preserving iteration 2's fits in Experiments 19, 20, 22, 23, and 25. The persistent large errors in Experiments 1, 4, 9, 13, and 24 also preclude shipping the current model.

**Outcome of your advice:** iter 6 candidate loss=0.1729 -> iter 7 candidate loss=0.1858 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed salience-guided sequential-integration family, but it was correctly rejected: loss rose to 0.1858 from the accepted iteration-2 loss of 0.1675. The new nonlinear isolation release and endpoint geometry gate yielded only modest gains in Experiment 8 (0.536 to 0.508), Experiment 11 (0.380 to 0.468), and Experiment 21 (0.035 to 0.088), leaving all three far from their observed values of 0.172, 0.843, and 0.390. These gains were outweighed by regressions in Experiment 3, Experiment 6, Experiment 18, and especially Experiment 20, whose previously accurate contrast moved from -0.169 to -0.344 against -0.181. The accepted crossover structure also weakened: Experiment 23 fell from 0.348 to 0.301, while Experiment 25 remains low at 0.452 versus 0.591. More generally, the model still gives excessive control to early or highly valid cues in Experiments 1, 2, 6, and 9, while underpredicting recency in Experiments 3, 4, 11, and 13. Terminal influence remains excessive in Experiments 8 and 24. Fits are satisfactory for Experiments 5, 7, 10, 12, 14–17, 19, 22, and 26, but the central experiment-invariant profile is not captured.
Rationale: Build on the unchanged accepted iteration-2 base. My previous recommendation to strengthen accumulator-dependent anchoring, steepen isolation release, and amplify the endpoint term for singleton-versus-coalition displays was rejected by the gate, so do not repeat or further strengthen that intervention. Repeated endpoint-specific geometry and scan modifications have produced little of the required Experiment 21 asymmetry while destabilizing crossover results.

Try a different, narrower calibration: further decouple validity's effect on access priority from its effect on update magnitude. Preserve iteration 2's absolute log-odds transformation in the latency map, because it generated the strong Experiment 23 crossover and improved Experiment 25. However, flatten only the reliability-dependent update multiplier—e.g., reduce its exponential slope and narrow its effective range—so that highly valid cues still receive early access but do not retain excessive evidential strength after later cues arrive. This should reduce highest-validity control in Experiments 1, 2, 6, and 9 and permit more weak-cue reversal in Experiments 3, 4, 13, and 25 without changing retention, scan direction, contrast gain, endpoint refresh, or isolation. Because coalition strength is currently computed partly from update magnitude, normalize that state update or leave its accepted calibration effectively unchanged; otherwise flattening reliability could inadvertently weaken the compact majority in Experiment 8.

Test this as a single intervention before adding anything else. Preserve the accepted fits in Experiments 19, 20, 22, and 23, especially the sign and magnitude of Experiment 20. If it improves aggregate loss, only then consider a small one-dimensional endpoint-salience range adjustment for Experiment 21 rather than another bespoke geometry gate. The current candidate should not be retained.

**Outcome of your advice:** iter 7 candidate loss=0.1858 -> iter 8 candidate loss=0.1604 -> the gate marked it ACCEPTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed salience-guided sequential-integration family, and iteration 8 is a genuine improvement: flattening reliability-dependent update magnitude reduced aggregate loss from 0.1675 to 0.1604. It improved early/high-validity overcontrol in Experiments 1, 2, 6, and 9, moved Experiment 24 from 0.659 to 0.608, raised weak-cue choice in Experiment 25 from 0.454 to 0.480, and improved the Experiment 23 crossover from 0.348 to 0.376. It also preserves good fits in Experiments 10, 14, 15, 19, 20, and 22 and the thresholded results in 7, 17, and 26. Nevertheless, the model is not sufficiently experiment-invariant. Large errors remain in Experiment 1 (0.696 versus 0.494), Experiment 4 (0.435 versus 0.665), Experiment 8 (0.540 versus 0.172), Experiment 9 (0.143 versus -0.033), Experiment 11 (0.394 versus 0.843), Experiment 13 (0.629 versus 0.881), Experiment 21 (0.057 versus 0.390), and Experiment 24 (0.608 versus 0.419). Experiment 25 also remains materially low at 0.480 versus 0.591. Thus reliability still has too much effective influence in several conflicts, ordinary and isolated-endpoint recency remains too weak, but terminal influence is simultaneously excessive in compact-majority and two-cue configurations. The endpoint pathway especially fails to produce the large positional asymmetry in Experiment 21. Heterogeneity is also misplaced: Experiment 21 variance is too low, whereas Experiment 23 variance remains substantially too high.
Rationale: The previous advice to decouple reliability access from update magnitude was accepted by the gate, so extend that successful direction rather than returning to the rejected retention, anchoring, isolation, or bespoke endpoint-geometry bundles. On the accepted iteration-8 base, test a smaller additional flattening of the direct reliability multiplier—for example, move the exponential slope from 0.035 toward roughly 0.015–0.025 and narrow the clip modestly toward unity. Preserve absolute reliability in latency, since that is supporting Experiments 23 and 25. To avoid worsening the already excessive terminal response in Experiment 8, keep coalition coherence bookkeeping approximately invariant to this reliability rescaling—for example, update coalition strength with a reliability-normalized evidence increment while retaining reliability in the actual accumulator update. This remains a graded coalition accumulator rather than a new strategy family.

Regression-test that isolated change against Experiments 1, 2, 6, 9, 23, 24, and 25. It should further reduce high-validity control without sacrificing the accepted crossover gains or the accurate Experiment 20 contrast. Do not again combine it with global retention decline, nonlinear isolation release, or a new geometry-gated refresh process; those directions were repeatedly rejected.

Because the accepted reliability edit has now succeeded, a subsequent independent one-dimensional test of endpoint salience is warranted, as anticipated in the prior critique. Shift the endpoint-salience distribution modestly upward while leaving scan, attenuation, refresh equations, and all other parameters fixed, and retain enough hierarchical spread to address Experiment 21's underestimated variance. Accept that adjustment only if it substantially raises Experiment 21 while preserving Experiments 8, 23, 24, and 25; a universal endpoint increase that worsens their already excessive terminal effects should be discarded. The remaining large and directionally systematic failures preclude shipping the current candidate despite its new running-best loss.

**Outcome of your advice:** iter 8 candidate loss=0.1604 -> CURRENT candidate loss=0.1725 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed salience-guided sequential-integration family, but the gate correctly rejected it: loss increased from the accepted iteration-8 value of 0.1604 to 0.1725. The additional reliability flattening overshot. It produced small gains in Experiments 6, 9, and 23, but worsened the central high-validity diagnostics it was intended to improve: Experiment 1 rose from 0.696 to 0.746 versus 0.494, and Experiment 2 rose from 0.257 to 0.277 versus 0.194. Experiment 18 also deteriorated substantially, from -0.062 to -0.088 versus 0.026. The persistent experiment-invariance failures remain large: recency is too weak in Experiments 4, 11, and 13; terminal influence is too strong in Experiments 8 and 24; endpoint asymmetry is nearly absent in Experiment 21; and weak-cue reversal remains insufficient in Experiment 25. Fits remain satisfactory in Experiments 10, 14, 15, 19, 20, 22, 23, and 26, but they do not offset these systematic discrepancies. Variance is also misplaced, particularly the very low endpoint-effect variance in Experiment 21 and excessive crossover variance in Experiment 23.",
  "verdict": "regenerate",
  "rationale": "Build on the unchanged accepted iteration-8 base. My previous recommendation to flatten the update multiplier further to a slope near 0.018 and reliability-normalize coalition bookkeeping was rejected, so do not repeat that full step. The accepted move from the earlier calibration to iteration 8 was beneficial, but the current result indicates that the second step was too large or that changing coalition bookkeeping simultaneously defeated its intended effect.\n\nUse a smaller, isolated interpolation rather than oscillating between the accepted and rejected settings: test an update-reliability slope around 0.027\u20130.030, between iteration 8's 0.035 and the rejected 0.018, with a correspondingly modest clip near unity. Crucially, retain iteration 8's original coalition bookkeeping for this test rather than dividing reliability out. That separates the effect of accumulator evidence compression from the new coalition-state rescaling and avoids inadvertently strengthening agreement-based resistance relative to actual evidence. Keep latency reliability, retention, scan, endpoint salience, contrast, isolation, refresh, capacity, and readout unchanged.\n\nRegression-test Experiments 1, 2, 6, 9, 18, 20, 23, 24, and 25. The midpoint should retain the current candidate's modest gains in 6 and 23 without reproducing its regressions in 1, 2, and 18 or sacrificing the accurate Experiment 20 contrast. Do not add an endpoint-salience increase in the same attempt: repeated endpoint/scan interventions have been rejected, and a universal increase risks worsening the already excessive terminal effects in Experiments 8 and 24. If the isolated midpoint does not beat 0.1604, retain iteration 8's reliability calibration and move to a genuinely separate one-dimensional adjustment rather than flattening further."
}
```

## Usage

```json
{
  "input_tokens": 50634,
  "output_tokens": 2516
}
```
