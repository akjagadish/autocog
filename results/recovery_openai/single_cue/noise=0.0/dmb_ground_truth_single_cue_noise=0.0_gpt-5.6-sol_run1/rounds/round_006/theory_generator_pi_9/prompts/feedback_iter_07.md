# feedback_iter_07

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
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 2 (= `pi_8`).

Replace pi_8 with a new theory of context-calibrated sequential competition rather than regenerating its current model. The new theory should retain validity-ordered inspection but reject both the hard coherent-chunk/primacy policy and direct physical-edge foregrounding. Each attended cue should contribute signed evidence, with later evidence competing against and potentially inhibiting the current commitment. The degree of overwrite versus integration should be a continuous latent task-set learned from the experiment-wide distribution of scan-order conflicts, reversal locations, validity gradients, and repeated sequence families. Nondiscriminating physical gaps should have little or no direct effect, consistent with Experiment 2.

Unlike pi_8, the new theory should permit tally and weighted coalitions to alter the selected direction—not merely confidence—because Experiment 3 shows a large tally-dependent change. Unlike pi_3, reliability should not decline mechanically with discriminating-cue count: coherent or salient terminal reversals can remain highly reliable at four to six reasons, as in Experiments 1, 6, 12, and 13. Interference should instead depend on conflict geometry, relative run strength, validity profile, and learned task context. Stable subject-level variation can govern attention, leak/inhibition, context-learning rate, response gain, and lapse, but contextual effects should be continuous rather than a mixture of terminal and primacy policies.

The replacement should be constrained to reproduce the major cross-task signatures simultaneously: approximately 0.88 terminal following in Experiment 1; an approximately zero physical-foregrounding contrast in Experiment 2; the roughly 0.49 tally interaction in Experiment 3; weak terminal following in Experiments 5 and 10; strong terminal control in Experiments 6, 12, and 13; and rejection of the early 3-to-2 majority in Experiment 14. Explicit out-of-sample tests should cross the same ordered cue sequence with different validity profiles and experiment-wide sequence distributions, which would distinguish learned sequential competition from pi_3's task-invariant overwrite/fatigue account.


## CANDIDATE THEORY
Distribution-Calibrated Competitive Accumulation with Terminal-Weighted Reversal Susceptibility. People inspect discriminating cues in descending instructed-validity order and encode each attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the active commitment, while an experiment-level task set learned from ordinal conflict families continuously controls retention and overwrite. The task set estimates how readily opposing runs would reverse a reference accumulator, emphasizing effective attacks that extend toward the end of the validity-ordered scan and recurrent high-susceptibility sequence families. This terminal-weighted upper-tail statistic combines reversal location, relative run strength, and cue reliability without treating physical gaps or weak terminal cues as direct overwrite signals. Reliability-weighted coalitions can reverse the sequential state, but their access to choice is proportional to their normalized evidence margin. Reliability depends on conflict geometry and evidence margin rather than discriminating-cue count.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive accumulation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["attention_rate"])
    integration_retention = float(parameters["integration_retention"])
    overwrite_retention = float(parameters["overwrite_retention"])
    conflict_inhibition = float(parameters["conflict_inhibition"])
    coalition_weight = float(parameters["coalition_weight"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    context_sensitivity = float(parameters["context_sensitivity"])
    context_threshold = float(parameters["context_threshold"])
    learning_rate = float(parameters["context_learning_rate"])
    susceptibility_coupling = float(parameters["susceptibility_coupling"])
    response_gain = float(parameters["response_gain"])
    lapse = float(parameters["lapse_rate"])

    cue_order = np.argsort(-validities, kind="stable")

    def ordinal_summary(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
            return None
        d = a - b
        ordered_idx = [int(j) for j in cue_order if j < d.size and d[j] != 0]
        if not ordered_idx:
            return None
        directions = np.asarray([np.sign(d[j]) for j in ordered_idx], dtype=float)
        m = directions.size
        switches = np.flatnonzero(directions[1:] != directions[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if directions[k] == directions[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        if switches.size:
            reversal_fractions = switches.astype(float) / float(m)
        else:
            reversal_fractions = np.asarray([], dtype=float)

        # Estimate reversal susceptibility with the same signed evidence units
        # used by the decision process. At every run boundary, compare the
        # incoming opposing run with the reference-retained incumbent. The
        # smooth score records both actual reversals and near reversals.
        local_v = np.clip(validities, 0.5001, 0.999)
        local_reliability = np.log(local_v / (1.0 - local_v))
        local_scale = float(np.mean(local_reliability[local_reliability > 0.0]))
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = 1.0
        local_normalized = np.clip(local_reliability / local_scale, 0.15, 6.0)
        local_strength = np.power(local_normalized, validity_sensitivity)
        ordered_strength = np.asarray(
            [attention * float(local_strength[j]) for j in ordered_idx],
            dtype=float
        )

        reference_retention = 0.55
        reference_accumulator = 0.0
        susceptibility_scores = []
        susceptibility_locations = []
        k = 0
        while k < m:
            run_direction = float(directions[k])
            run_end = k + 1
            while run_end < m and directions[run_end] == run_direction:
                run_end += 1

            run_evidence = ordered_strength[k:run_end]
            run_length = int(run_end - k)
            if reference_accumulator != 0.0 and np.sign(reference_accumulator) != run_direction:
                retained_incumbent = abs(reference_accumulator) * (
                    reference_retention ** run_length
                )
                attack = 0.0
                for strength in run_evidence:
                    attack = reference_retention * attack + float(strength)
                normalized_advantage = (
                    attack - retained_incumbent
                ) / max(attack + retained_incumbent, 1e-9)
                smooth_reversal = 1.0 / (
                    1.0 + np.exp(-4.0 * np.clip(normalized_advantage, -15.0, 15.0))
                )
                susceptibility_scores.append(float(smooth_reversal))
                susceptibility_locations.append(float(run_end) / float(m))

            for strength in run_evidence:
                reference_accumulator = (
                    reference_retention * reference_accumulator
                    + run_direction * float(strength)
                )
            k = run_end

        if susceptibility_scores:
            score_array = np.asarray(susceptibility_scores, dtype=float)
            location_array = np.asarray(susceptibility_locations, dtype=float)
            tail_logits = 1.5 * location_array + 2.0 * (score_array - 0.5)
            tail_logits -= float(np.max(tail_logits))
            tail_weights = np.exp(tail_logits)
            tail_weights /= max(float(np.sum(tail_weights)), 1e-12)
            reversal_susceptibility = float(np.sum(tail_weights * score_array))
        else:
            reversal_susceptibility = 0.35

        return {
            "m": float(m),
            "directions": directions,
            "reversal_fractions": reversal_fractions,
            "terminal_singleton": float(run_lengths[-1] == 1 and len(run_lengths) > 1),
            "two_run": float(len(run_lengths) == 2),
            "switch_density": float(switches.size) / float(max(1, m - 1)),
            "reversal_susceptibility": reversal_susceptibility
        }

    # Learn the task set from ordinal scan sequences only. Physical indices,
    # distances between cues, and nondiscriminating gaps never enter these
    # statistics. Exponential weighting implements continuous contextual
    # learning while retaining a weak neutral prior early in the block.
    summaries = []
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        s = ordinal_summary(past_a[i], past_b[i])
        if s is not None:
            summaries.append(s)

    current_summary = ordinal_summary(stim[0], stim[1])
    if current_summary is None:
        return np.array([0.5, 0.5], dtype=np.float64)
    summaries.append(current_summary)

    ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
    weights = np.power(np.clip(1.0 - learning_rate, 1e-6, 1.0), ages)
    # The current stimulus can update calibration, but once context exists it
    # receives a bounded weight and cannot sharply change the experiment-level
    # task set solely because of its own conflict geometry.
    if len(summaries) > 1:
        weights[-1] *= 0.35
    weights /= max(float(weights.sum()), 1e-12)

    terminal_singleton_rate = float(np.sum([
        w * s["terminal_singleton"] for w, s in zip(weights, summaries)
    ]))
    two_run_rate = float(np.sum([
        w * s["two_run"] for w, s in zip(weights, summaries)
    ]))
    switch_density = float(np.sum([
        w * s["switch_density"] for w, s in zip(weights, summaries)
    ]))
    susceptibility_values = np.asarray([
        s["reversal_susceptibility"] for s in summaries
    ], dtype=float)
    susceptibility_logits = 1.5 * (susceptibility_values - 0.35)
    susceptibility_logits -= float(np.max(susceptibility_logits))
    susceptibility_weights = weights * np.exp(susceptibility_logits)
    susceptibility_weights /= max(float(np.sum(susceptibility_weights)), 1e-12)
    reversal_susceptibility = float(np.sum(
        susceptibility_weights * susceptibility_values
    ))

    # A five-bin distribution represents reversal locations in ordinal scan
    # time, not their physical screen positions. Concentrated sequence
    # families promote overwrite; high reversal entropy promotes integration.
    reversal_hist = np.zeros(5, dtype=float)
    reversal_mass = 0.0
    for w, s in zip(weights, summaries):
        fractions = s["reversal_fractions"]
        if fractions.size == 0:
            continue
        contribution = w / float(fractions.size)
        for fraction in fractions:
            bin_index = min(4, max(0, int(np.floor(5.0 * float(fraction)))))
            reversal_hist[bin_index] += contribution
            reversal_mass += contribution

    if reversal_mass > 0.0:
        reversal_distribution = reversal_hist / reversal_mass
        concentration = float(np.sum(reversal_distribution ** 2))
        positive = reversal_distribution[reversal_distribution > 0.0]
        reversal_entropy = float(
            -np.sum(positive * np.log(positive)) / np.log(5.0)
        )
    else:
        concentration = 0.2
        reversal_entropy = 1.0

    # Validity-profile steepness can alter task calibration, but only weakly;
    # this allows the same ordinal sequence to generalize differently under a
    # changed reliability profile without turning validity into a hard rule.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    mean_rel = max(float(np.mean(reliability)), 1e-8)
    validity_gradient = float(np.std(reliability) / mean_rel)
    validity_gradient = float(np.clip(validity_gradient, 0.0, 2.0))

    context_score = (
        3.0 * terminal_singleton_rate
        + 3.0 * concentration
        + 2.0 * two_run_rate
        - 2.5 * reversal_entropy
        - 1.2 * switch_density
        - 0.20 * validity_gradient
        + susceptibility_coupling * (reversal_susceptibility - 0.35)
    )
    context_logit = context_sensitivity * (context_score - context_threshold)
    context_logit = float(np.clip(context_logit, -60.0, 60.0))
    overwrite_task_set = 1.0 / (1.0 + np.exp(-context_logit))

    # The latent task set continuously interpolates between integration and
    # overwrite. It never selects a discrete primacy or terminal policy.
    base_retention = (
        (1.0 - overwrite_task_set) * integration_retention
        + overwrite_task_set * overwrite_retention
    )
    base_retention = float(np.clip(base_retention, 0.01, 0.999))

    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    directions = np.asarray([np.sign(diff[j]) for j in discriminating], dtype=float)

    mean_positive_rel = float(np.mean(reliability[reliability > 0.0]))
    if not np.isfinite(mean_positive_rel) or mean_positive_rel <= 0.0:
        mean_positive_rel = 1.0
    normalized_rel = np.clip(reliability / mean_positive_rel, 0.15, 6.0)
    cue_strength = np.power(normalized_rel, validity_sensitivity)

    accumulator = 0.0
    coalition = 0.0
    current_run_strength = 0.0
    previous_direction = 0.0

    for j, direction in zip(discriminating, directions):
        evidence = attention * float(cue_strength[j]) * float(direction)
        coalition += evidence

        if previous_direction == 0.0 or direction == previous_direction:
            current_run_strength += abs(evidence)
        else:
            current_run_strength = abs(evidence)

        retention = base_retention
        if accumulator != 0.0 and np.sign(accumulator) != direction:
            # Opposition inhibits the existing commitment in proportion to
            # the new reason's strength relative to the active commitment.
            relative_attack = abs(evidence) / (abs(accumulator) + abs(evidence) + 1e-9)
            run_attack = current_run_strength / (abs(accumulator) + current_run_strength + 1e-9)
            inhibition = conflict_inhibition * overwrite_task_set
            retention *= np.exp(-inhibition * (0.65 * relative_attack + 0.35 * run_attack))

        accumulator = retention * accumulator + evidence
        previous_direction = float(direction)

    # A reliability-weighted coalition has strong access only when it has a
    # substantial normalized margin. Near balance, sequential opposition can
    # determine the sign; clear early majorities remain protected by global
    # coalition evidence. This is continuous evidence gating, not policy mixing.
    total_input = float(np.sum([
        attention * float(cue_strength[j]) for j in discriminating
    ]))
    coalition_margin = abs(coalition) / max(total_input, 1e-9)
    coalition_margin = float(np.clip(coalition_margin, 0.0, 1.0))
    margin_access = 0.10 + 0.90 * coalition_margin / (coalition_margin + 0.20)
    effective_coalition_weight = (
        coalition_weight * (1.0 - overwrite_task_set) * margin_access
    )
    decision_evidence = (
        (1.0 - effective_coalition_weight) * accumulator
        + effective_coalition_weight * coalition
    )

    # Confidence follows the final evidence margin. There is deliberately no
    # fatigue or count penalty, allowing coherent long reversals to remain
    # reliable while balanced conflict naturally remains uncertain.
    margin = abs(decision_evidence) / max(total_input, 1e-9)
    geometry_gain = 0.75 + 0.50 * np.sqrt(np.clip(margin, 0.0, 1.0))
    decision_variable = response_gain * geometry_gain * decision_evidence
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5], dtype=np.float64)
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
- attention_rate: [0.88, 1.0]
- integration_retention: [0.84, 0.97]
- overwrite_retention: [0.14, 0.36]
- conflict_inhibition: [0.35, 1.20]
- coalition_weight: [0.30, 0.62]
- validity_sensitivity: [0.15, 0.45]
- context_sensitivity: [1.4, 2.4]
- context_threshold: [1.0, 1.6]
- context_learning_rate: [0.05, 0.12]
- susceptibility_coupling: [0.50, 0.95]
- response_gain: [1.8, 3.8]
- lapse_rate: [0.04, 0.15]

`rationale`:
This is a minimal edit of accepted iteration 7. All evidence weighting, coalition access, local inhibition, context learning, response mapping, and exclusion of physical gaps remain unchanged. The arithmetic mean of reversal susceptibility is replaced in two places. Within a sequence, a stable soft upper-tail average gives smoothly greater weight to effective opposing runs that extend later in the scan. Across sequence families, the existing recency weights are multiplied by a mild susceptibility softmax, so recurrent effective attacks influence calibration more than many weak reversals without introducing a threshold or discrete policy. A single weak terminal cue remains weak because soft weighting cannot increase its susceptibility score; only genuinely effective attacks receive greater influence. The susceptibility-coupling range is narrowed slightly downward to compensate for the upper-tail aggregation and avoid a global overwrite increase that would worsen Experiments 6 and 8. This should selectively strengthen terminal competition in recurrent late-reversal contexts such as Experiments 2, 3, 4, 11, and 13 while preserving coalition protection, low learning-rate calibration, and the accepted fits in Experiments 7, 10, 12, and 14.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.2560 -> ACCEPTED
- iter 2: loss=0.2844 -> REJECTED
- iter 3: loss=0.2185 -> ACCEPTED
- iter 4: loss=0.2882 -> REJECTED
- iter 5: loss=0.2816 -> REJECTED
- iter 6: loss=0.2926 -> REJECTED
- iter 7: loss=0.1986 -> ACCEPTED
- iter 8 (current candidate you are grading): loss=0.2077 -> REJECTED
Running-best (last accepted) base: iter 7 at loss=0.1986.

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
  - iter 1: 0.8978 (var=0.0036) (Δ vs real +0.4033)
  - iter 2: 0.8865 (var=0.0047) (Δ vs real +0.3921)
  - iter 3: 0.8503 (var=0.0038) (Δ vs real +0.3558)
  - iter 4: 0.7601 (var=0.0069) (Δ vs real +0.2657)
  - iter 5: 0.8676 (var=0.0040) (Δ vs real +0.3732)
  - iter 6: 0.8311 (var=0.0056) (Δ vs real +0.3367)
  - iter 7: 0.8336 (var=0.0055) (Δ vs real +0.3392)
  - iter 8 (current): 0.8564 (var=0.0048) (Δ vs real +0.3619)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)
- pi_6: 0.6093 (var=0.0085)
- pi_7: 0.5681 (var=0.0095)
- pi_8: 0.5276 (var=0.0147)

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
  - iter 1: 0.1504 (var=0.0030) (Δ vs real -0.0433)
  - iter 2: 0.2394 (var=0.0036) (Δ vs real +0.0456)
  - iter 3: 0.2919 (var=0.0051) (Δ vs real +0.0981)
  - iter 4: 0.2360 (var=0.0029) (Δ vs real +0.0423)
  - iter 5: 0.2979 (var=0.0073) (Δ vs real +0.1042)
  - iter 6: 0.3029 (var=0.0061) (Δ vs real +0.1092)
  - iter 7: 0.2892 (var=0.0056) (Δ vs real +0.0954)
  - iter 8 (current): 0.2879 (var=0.0069) (Δ vs real +0.0942)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)
- pi_6: 0.2027 (var=0.0056)
- pi_7: 0.1940 (var=0.0026)
- pi_8: 0.1765 (var=0.0015)

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
  - iter 1: 0.3783 (var=0.0074) (Δ vs real -0.1967)
  - iter 2: 0.3979 (var=0.0054) (Δ vs real -0.1771)
  - iter 3: 0.3585 (var=0.0076) (Δ vs real -0.2165)
  - iter 4: 0.2931 (var=0.0059) (Δ vs real -0.2819)
  - iter 5: 0.4188 (var=0.0055) (Δ vs real -0.1562)
  - iter 6: 0.1798 (var=0.0037) (Δ vs real -0.3952)
  - iter 7: 0.3906 (var=0.0076) (Δ vs real -0.1844)
  - iter 8 (current): 0.3862 (var=0.0088) (Δ vs real -0.1887)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)
- pi_6: 0.6577 (var=0.0054)
- pi_7: 0.7098 (var=0.0021)
- pi_8: 0.6569 (var=0.0032)

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
  - iter 1: 0.3742 (var=0.0174) (Δ vs real -0.2908)
  - iter 2: 0.2604 (var=0.0136) (Δ vs real -0.4046)
  - iter 3: 0.1767 (var=0.0315) (Δ vs real -0.4883)
  - iter 4: -0.0896 (var=0.0331) (Δ vs real -0.7546)
  - iter 5: 0.2421 (var=0.0184) (Δ vs real -0.4229)
  - iter 6: -0.1062 (var=0.0288) (Δ vs real -0.7712)
  - iter 7: 0.2479 (var=0.0258) (Δ vs real -0.4171)
  - iter 8 (current): 0.2450 (var=0.0244) (Δ vs real -0.4200)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)
- pi_6: 0.6704 (var=0.0140)
- pi_7: 0.6104 (var=0.0134)
- pi_8: 0.6408 (var=0.0101)

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
  - iter 1: -0.3354 (var=0.0097) (Δ vs real -0.3387)
  - iter 2: -0.2242 (var=0.0079) (Δ vs real -0.2275)
  - iter 3: -0.0938 (var=0.0338) (Δ vs real -0.0971)
  - iter 4: 0.5758 (var=0.0322) (Δ vs real +0.5725)
  - iter 5: -0.2413 (var=0.0250) (Δ vs real -0.2446)
  - iter 6: 0.1100 (var=0.0653) (Δ vs real +0.1067)
  - iter 7: -0.1046 (var=0.0293) (Δ vs real -0.1079)
  - iter 8 (current): -0.1254 (var=0.0351) (Δ vs real -0.1287)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)
- pi_6: -0.0142 (var=0.0089)
- pi_7: 0.0079 (var=0.0079)
- pi_8: 0.0112 (var=0.0099)

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
  - iter 1: 0.1058 (var=0.0039) (Δ vs real -0.2008)
  - iter 2: 0.1225 (var=0.0036) (Δ vs real -0.1842)
  - iter 3: 0.1754 (var=0.0085) (Δ vs real -0.1312)
  - iter 4: 0.1433 (var=0.0049) (Δ vs real -0.1633)
  - iter 5: 0.1642 (var=0.0054) (Δ vs real -0.1425)
  - iter 6: 0.1537 (var=0.0044) (Δ vs real -0.1529)
  - iter 7: 0.1717 (var=0.0076) (Δ vs real -0.1350)
  - iter 8 (current): 0.1883 (var=0.0068) (Δ vs real -0.1183)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)
- pi_6: 0.3067 (var=0.0061)
- pi_7: 0.3404 (var=0.0071)
- pi_8: 0.3050 (var=0.0041)

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
  - iter 1: 1.0000 (var=0.0900) (Δ vs real +1.0000)
  - iter 2: 1.0000 (var=0.2016) (Δ vs real +1.0000)
  - iter 3: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.1924) (Δ vs real +1.0000)
  - iter 6: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 8 (current): 0.0000 (var=0.2464) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)
- pi_6: 0.0000 (var=0.2100)
- pi_7: 0.0000 (var=0.2356)
- pi_8: 0.0000 (var=0.1924)

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
  - iter 1: 0.3221 (var=0.0021) (Δ vs real +0.1500)
  - iter 2: 0.3427 (var=0.0023) (Δ vs real +0.1706)
  - iter 3: 0.3848 (var=0.0018) (Δ vs real +0.2127)
  - iter 4: 0.4248 (var=0.0008) (Δ vs real +0.2527)
  - iter 5: 0.3902 (var=0.0016) (Δ vs real +0.2181)
  - iter 6: 0.4227 (var=0.0019) (Δ vs real +0.2506)
  - iter 7: 0.3835 (var=0.0019) (Δ vs real +0.2115)
  - iter 8 (current): 0.3867 (var=0.0018) (Δ vs real +0.2146)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)
- pi_6: 0.2250 (var=0.0059)
- pi_7: 0.3131 (var=0.0029)
- pi_8: 0.3069 (var=0.0027)

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
  - iter 1: 0.0392 (var=0.0052) (Δ vs real +0.0717)
  - iter 2: 0.1167 (var=0.0081) (Δ vs real +0.1492)
  - iter 3: 0.0638 (var=0.0093) (Δ vs real +0.0963)
  - iter 4: 0.0392 (var=0.0058) (Δ vs real +0.0717)
  - iter 5: 0.0254 (var=0.0064) (Δ vs real +0.0579)
  - iter 6: 0.1017 (var=0.0114) (Δ vs real +0.1342)
  - iter 7: 0.0283 (var=0.0072) (Δ vs real +0.0608)
  - iter 8 (current): 0.0437 (var=0.0046) (Δ vs real +0.0762)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0083 (var=0.0103)
- pi_6: 0.2096 (var=0.0101)
- pi_1: 0.7021 (var=0.0307)
- pi_2: 0.7054 (var=0.0384)
- pi_4: 0.0375 (var=0.0075)
- pi_5: 0.0108 (var=0.0053)
- pi_7: 0.0638 (var=0.0071)
- pi_8: 0.0083 (var=0.0044)

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
  - iter 1: 0.9299 (var=0.0010) (Δ vs real +0.0551)
  - iter 2: 0.9095 (var=0.0012) (Δ vs real +0.0347)
  - iter 3: 0.8784 (var=0.0015) (Δ vs real +0.0036)
  - iter 4: 0.8099 (var=0.0025) (Δ vs real -0.0648)
  - iter 5: 0.8923 (var=0.0020) (Δ vs real +0.0176)
  - iter 6: 0.8527 (var=0.0012) (Δ vs real -0.0220)
  - iter 7: 0.8820 (var=0.0013) (Δ vs real +0.0072)
  - iter 8 (current): 0.8852 (var=0.0015) (Δ vs real +0.0104)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9281 (var=0.0013)
- pi_3: 0.7843 (var=0.0034)
- pi_1: 0.5635 (var=0.0023)
- pi_2: 0.8101 (var=0.0072)
- pi_4: 0.8645 (var=0.0034)
- pi_5: 0.9256 (var=0.0034)
- pi_7: 0.8717 (var=0.0018)
- pi_8: 0.8753 (var=0.0015)

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
  - iter 1: 0.7779 (var=0.0051) (Δ vs real -0.0646)
  - iter 2: 0.6492 (var=0.0066) (Δ vs real -0.1933)
  - iter 3: 0.6488 (var=0.0106) (Δ vs real -0.1937)
  - iter 4: 0.7183 (var=0.0082) (Δ vs real -0.1242)
  - iter 5: 0.6533 (var=0.0090) (Δ vs real -0.1892)
  - iter 6: 0.6283 (var=0.0103) (Δ vs real -0.2142)
  - iter 7: 0.6375 (var=0.0110) (Δ vs real -0.2050)
  - iter 8 (current): 0.6517 (var=0.0110) (Δ vs real -0.1908)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6863 (var=0.0047)
- pi_7: 0.0642 (var=0.0023)
- pi_1: 0.1583 (var=0.0130)
- pi_2: 0.1342 (var=0.0084)
- pi_4: 0.3117 (var=0.0174)
- pi_5: 0.6737 (var=0.0053)
- pi_6: 0.6196 (var=0.0088)
- pi_8: 0.8558 (var=0.0026)

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
  - iter 1: 0.1156 (var=0.0046) (Δ vs real -0.0556)
  - iter 2: 0.2013 (var=0.0038) (Δ vs real +0.0300)
  - iter 3: 0.2181 (var=0.0053) (Δ vs real +0.0469)
  - iter 4: 0.1638 (var=0.0074) (Δ vs real -0.0075)
  - iter 5: 0.2119 (var=0.0088) (Δ vs real +0.0406)
  - iter 6: 0.2181 (var=0.0105) (Δ vs real +0.0469)
  - iter 7: 0.2144 (var=0.0073) (Δ vs real +0.0431)
  - iter 8 (current): 0.2231 (var=0.0091) (Δ vs real +0.0519)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7800 (var=0.0127)
- pi_3: 0.2650 (var=0.0077)
- pi_1: 0.8450 (var=0.0106)
- pi_2: 0.8638 (var=0.0097)
- pi_4: 0.3144 (var=0.0590)
- pi_5: 0.2125 (var=0.0050)
- pi_6: 0.1788 (var=0.0107)
- pi_8: 0.1313 (var=0.0048)

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
  - iter 1: 0.8049 (var=0.0032) (Δ vs real -0.0760)
  - iter 2: 0.7080 (var=0.0045) (Δ vs real -0.1730)
  - iter 3: 0.7349 (var=0.0057) (Δ vs real -0.1461)
  - iter 4: 0.7161 (var=0.0043) (Δ vs real -0.1649)
  - iter 5: 0.7165 (var=0.0031) (Δ vs real -0.1645)
  - iter 6: 0.5537 (var=0.0054) (Δ vs real -0.3273)
  - iter 7: 0.7286 (var=0.0056) (Δ vs real -0.1524)
  - iter 8 (current): 0.7367 (var=0.0052) (Δ vs real -0.1443)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7201 (var=0.0040)
- pi_8: 0.3399 (var=0.0051)
- pi_1: 0.1647 (var=0.0106)
- pi_2: 0.4955 (var=0.0024)
- pi_4: 0.4758 (var=0.0129)
- pi_5: 0.6957 (var=0.0044)
- pi_6: 0.7470 (var=0.0062)
- pi_7: 0.5510 (var=0.0027)

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
  - iter 1: 0.0213 (var=0.0068) (Δ vs real +0.0021)
  - iter 2: 0.0108 (var=0.0073) (Δ vs real -0.0083)
  - iter 3: -0.0050 (var=0.0093) (Δ vs real -0.0242)
  - iter 4: 0.0083 (var=0.0116) (Δ vs real -0.0108)
  - iter 5: 0.0083 (var=0.0095) (Δ vs real -0.0108)
  - iter 6: 0.0046 (var=0.0097) (Δ vs real -0.0146)
  - iter 7: 0.0083 (var=0.0128) (Δ vs real -0.0108)
  - iter 8 (current): -0.0004 (var=0.0111) (Δ vs real -0.0196)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0863 (var=0.0106)
- pi_3: -0.0217 (var=0.0090)
- pi_1: -0.0004 (var=0.0063)
- pi_2: -0.0008 (var=0.0037)
- pi_4: -0.0146 (var=0.0066)
- pi_5: 0.0054 (var=0.0094)
- pi_6: -0.0137 (var=0.0073)
- pi_7: -0.0042 (var=0.0041)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed sequential-competition family and successfully eliminates direct physical foregrounding: Experiment 14 is essentially exact (0.021 versus 0.019), while Experiments 2 and 9 are also reasonably close. It additionally captures the broad direction of strong terminal control in Experiments 10, 11, 12, and 13, although it is somewhat too deterministic in several of them. However, the cross-experiment calibration is not yet adequate. The largest failures are the excessive tally interaction in Experiment 1 (0.898 versus 0.494), insufficient terminal influence in Experiments 3 and 4 (0.378 versus 0.575 and 0.374 versus 0.665), a large spurious block-versus-mixed contrast in Experiment 5 (-0.335 versus approximately zero), excessive terminal replacement in Experiment 6 (early choice 0.106 versus 0.307), the incorrect thresholded contrast in Experiment 7 (1 rather than 0), and excessive terminal following in Experiment 8 (0.322 versus 0.172). These discrepancies indicate that the model is generating overly sharp, stimulus-dependent shifts between integration and overwrite rather than a sufficiently stable experiment-level continuous task set.
Rationale: Retain the distribution-calibrated competitive accumulator, but revise its calibration. The most likely implementation problem is that the current trial is appended to the history before computing the task set. Consequently, each stimulus's own terminal-singleton, two-run, and switch-density statistics directly change the supposedly experiment-wide task set. This can create the large within-experiment geometry effects seen in Experiments 1, 5, and 7. Compute contextual calibration primarily from preceding trials, with a neutral prior and at most a small explicitly bounded contribution from the current sequence. Also soften the large two-run, concentration, and switch-density coefficients or reduce context sensitivity so the continuous task set does not approach opposite extremes across individual stimuli. Do not simply weaken terminal competition globally: that would worsen Experiments 3, 4, and 13. Instead, add a modest baseline opposition-inhibition/recency component that remains active even in integration-calibrated contexts, while reserving the task-set-dependent increment for strong overwrite. This should allow a late cue to tip balanced or near-balanced coalitions in Experiments 3 and 4 without overturning the substantial early majorities in Experiment 8. Refit retention and coalition ranges after removing current-trial leakage, aiming specifically to reduce the Experiment 1 tally contrast, eliminate the Experiment 5 and 7 geometry artifacts, and moderate Experiment 6 while preserving the good near-zero physical-gap effect and strong late-block performance in Experiments 10–14.

**Outcome of your advice:** iter 1 candidate loss=0.2560 -> iter 2 candidate loss=0.2844 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to context-calibrated sequential competition and preserves the important absence of physical-gap foregrounding in Experiment 14 (0.011 versus 0.019). It also fits Experiments 2, 10, and 12 reasonably well. Nevertheless, it was rejected by the accept gate and is worse overall than the accepted iteration-1 base. The history-only calibration change modestly improved Experiments 3, 5, 6, 10, and 12, but substantially damaged terminal control in Experiments 4, 11, and 13, worsened Experiments 8 and 9, and did not fix the decisive failures in Experiments 1 or 7. The model still produces an excessive tally-dependent contrast in Experiment 1 (0.887 versus 0.494), too little terminal influence in balanced conflicts in Experiments 3 and 4, a spurious geometry contrast in Experiment 5, excessive terminal following against early majorities in Experiments 6 and 8, and the wrong thresholded result in Experiment 7. Its low between-subject variances in many experiments also indicate overly homogeneous predictions.
Rationale: Because the previous recommendation to remove current-trial calibration and add baseline inhibition was rejected, do not repeat that full history-only move. Build from the unchanged accepted iteration-1 base and try a smaller, different in-family correction. First, raise and widen validity sensitivity: the current 0.02–0.12 range makes evidence almost an equal-weight tally, plausibly causing the enormous Experiment 1 tally contrast. Stronger reliability weighting can reduce that contrast without requiring primacy or abandoning sequential competition. Second, make the coalition contribution depend smoothly on the validity-weighted coalition margin: retain more coalition influence when there is a substantial early majority, as in Experiments 6 and 8, but reduce it near balance so sequential opposition can let terminal evidence tip Experiments 3 and 4. This is a continuous normalization within the prescribed accumulator, not a policy mixture. Third, regularize rather than eliminate current-sequence calibration—for example, cap its contribution or blend the accepted base's task set with a mostly history-derived estimate. An intermediate calibration is warranted because complete exclusion improved Experiment 5 but sharply weakened the required terminal control in Experiments 11 and 13. Tune against the joint constraints that Experiment 5 approach zero and Experiment 7 fall below its 0.08 threshold while preserving strong terminal performance in Experiments 10–13 and the near-zero Experiment 14 contrast. Avoid another global reduction in response gain or terminal competition; the current iteration shows that this sacrifices the strong-terminal experiments without resolving Experiment 1.

**Outcome of your advice:** iter 2 candidate loss=0.2844 -> iter 3 candidate loss=0.2185 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted iteration is a genuine aggregate improvement and remains faithful to context-calibrated sequential competition. The intermediate current-trial regularization successfully removes the thresholded error in Experiment 7, substantially reduces the spurious Experiment 5 contrast, and gives an excellent fit in Experiment 10. Physical foregrounding remains appropriately absent in Experiment 14. Experiment 6 and the Experiment 1 tally interaction also move in the correct direction. However, the model is not yet experiment-invariant enough to ship. The tally interaction in Experiment 1 remains far too large (0.850 versus 0.494). Terminal evidence remains much too weak in Experiments 3 and 4 (0.359 versus 0.575; 0.177 versus 0.665), and terminal control is also underestimated in Experiments 11 and 13 (0.649 versus 0.843; 0.735 versus 0.881). Conversely, terminal following is excessive in Experiment 8 (0.385 versus 0.172), while highest-validity-cue following is too high in Experiment 2 (0.292 versus 0.194). Thus the static increase in validity sensitivity helped protect early coalitions, but it also broadly strengthened early evidence in contexts where humans exhibit overwrite. The margin gate alone cannot resolve this because reliability weighting is currently fixed across task sets. Several predicted between-subject variances, especially in Experiments 1, 8, and 10, are also too small, indicating overly uniform response regimes.
Rationale: Retain the accepted intermediate history calibration and margin-gated competitive accumulator; the previous advice leading to these changes was accepted and should not be undone. The next adjustment should make validity sensitivity itself continuously task-set dependent rather than increasing it globally again. Preserve relatively strong validity weighting in integration-calibrated or substantial-early-margin contexts, which should let the first cue resist opposing tallies in Experiment 1 and protect early runs in Experiment 8. Attenuate validity weighting as the learned overwrite/terminal-singleton context strengthens, allowing late opposition to regain control in Experiments 2, 3, 4, 11, and 13. A bounded interpolation with a nonzero sensitivity floor is preferable to a discrete rule. In addition, add only a modest coalition-margin-dependent floor to opposition inhibition for nearly balanced conflicts; the current inhibition is multiplied entirely by `overwrite_task_set`, leaving too little late competition in Experiments 3 and 4. Do not make this floor global or strong at large margins, because Experiment 6 already shows excessive overwrite and Experiment 12 is close. Tune these two couplings jointly, targeting a large reduction of Experiment 1's interaction and increased terminal choice in Experiments 3/4/11/13 without sacrificing the newly correct Experiments 5, 7, 10, and 14. Subject variation in the new coupling or response gain can then be widened slightly if necessary to address the remaining variance compression.

**Outcome of your advice:** iter 3 candidate loss=0.2185 -> iter 4 candidate loss=0.2882 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed context-calibrated sequential-competition family, but it was rejected by the accept gate and substantially regresses from the accepted iteration-3 base. The task-dependent validity exponent and balanced-inhibition floor move Experiment 1 somewhat toward the data and improve Experiments 2 and 12, but they produce severe collateral failures. Terminal influence becomes even weaker in Experiments 3 and 4 (0.293 versus 0.575 and -0.090 versus 0.665), the previously improved Experiment 5 contrast explodes from -0.094 to 0.576 rather than approximately zero, Experiment 8 moves farther above its low observed terminal-following rate, and Experiment 10 loses its nearly exact fit. Strong terminal control also remains underestimated in Experiments 11 and 13, while Experiment 6 still shows too much overwrite. Only Experiments 7, 9, 12, and 14 are acceptably close. The especially large reversal in Experiment 5 indicates that task-set-dependent validity compression is coupling too strongly to stimulus geometry and recreating within-experiment regime differences. The balanced-inhibition addition also fails its intended purpose: rather than increasing terminal control in Experiments 3 and 4, it reduces it, likely because preview-weighted balance and validity compression alter coalition signs or task calibration before the local competition unfolds.
Rationale: Because the previous recommendation to add task-dependent validity sensitivity and a balanced-conflict inhibition floor was rejected, do not repeat or enlarge either adjustment. The next candidate should build on the unchanged accepted iteration-3 base and keep its intermediate history calibration, static validity weighting, and successful fits in Experiments 5, 7, 10, and 14. Use a different in-family knob: revise the existing coalition-margin access function and local opposition update rather than changing experiment-level validity expression. In particular, remove or sharply reduce the coalition gate's nonzero access floor near a zero weighted margin and make access rise more convexly with margin. That allows the sequential state—not a residual global coalition—to select the direction in balanced conflicts such as Experiments 3 and 4, while retaining coalition protection for clear margins in Experiments 6, 8, and 12. If additional terminal competition is needed, implement it as a modest cumulative opposing-run attack on the incumbent state, normalized by incumbent strength and total attended evidence, rather than as another preview-balance floor. This should strengthen coherent late runs in Experiments 11 and 13 without globally lowering retention. Separately, use conflict-dependent response gain or temperature: lower gain for unresolved mixed sequences to reduce Experiment 1's excessive tally interaction, but preserve gain for coherent terminal runs so Experiments 10 and 13 are not weakened. Parameter ranges for these changes should be narrow around the accepted base, since the current broad couplings caused large sign-changing regressions and excessive cross-stimulus instability.

**Outcome of your advice:** iter 4 candidate loss=0.2882 -> iter 5 candidate loss=0.2816 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate remains faithful to context-calibrated sequential competition and continues to avoid direct physical-gap foregrounding: Experiment 14 is close to exact, and Experiments 9, 10, and 12 are reasonably fitted. Nevertheless, it was rejected by the accept gate and is substantially worse than the accepted iteration-3 base. The convex coalition gate and revised cumulative-run attack do not solve the central cross-task problem. Experiment 1's tally interaction rebounds to 0.868 versus 0.494; terminal control remains much too weak in Experiments 3, 4, 11, and 13; Experiment 5 again develops a sizable spurious geometry contrast; and Experiment 7 returns to the wrong thresholded outcome. The model also overwrites too strongly in Experiments 6 and 8, as shown by too little early following in Experiment 6 and too much terminal following in Experiment 8. Thus the same local competition rule cannot distinguish tasks requiring strong terminal control from tasks requiring protection of early coalitions. The likely bottleneck is the coarse context representation: it summarizes singleton frequency, switch density, two-run frequency, and reversal-bin concentration, but omits experiment-wide distributions of relative run strength and weighted prefix-versus-terminal strength. Its validity-profile representation is only a weak coefficient on the coefficient of variation of cue reliabilities. Consequently, distinct experiments can receive similar task sets despite sharply different validity profiles and conflict-strength distributions. Between-subject variance also remains compressed in several conditions, especially Experiment 1.
Rationale: Build the next attempt on the unchanged accepted iteration-3 base. The previous recommendation to convexify coalition access and strengthen cumulative opposing-run attack was rejected, so do not repeat or enlarge either change; likewise retain iteration 3's static validity sensitivity, intermediate current-trial regularization, and successful local update. Instead, improve the context calibration within the prescribed mechanism family. Add continuous experiment-level summaries of validity-weighted conflict geometry: terminal-run strength relative to incumbent prefix strength, weighted coalition margin at each reversal, reversal-location-by-run-balance interactions, and profile-specific quantities such as the log-odds validity slope and terminal-to-prefix reliability ratio. Aggregate these over preceding sequence families with the existing bounded current-trial contribution. Map them smoothly into the single overwrite task set with narrow coefficients and regularization, rather than adding a new policy or trial-specific gate. This should let identical ordinal patterns induce stronger overwrite under the profiles and sequence distributions of Experiments 3, 4, 11, and 13, but stronger integration when experiment-wide families contain protected early coalitions as in Experiments 6 and 8. Keep physical indices and nondiscriminating gaps excluded. Tune first against the joint constraints of Experiments 1, 3, 4, 5, 6, 8, 11, and 13 while requiring that the accepted fits in Experiments 7, 10, 12, and 14 do not regress. Only after directional fit improves should subject-level variation in context coefficients or response gain be widened modestly to address variance compression.

**Outcome of your advice:** iter 5 candidate loss=0.2816 -> iter 6 candidate loss=0.2926 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed context-calibrated sequential-competition family and still correctly excludes physical foregrounding: Experiment 14 remains close to zero (0.005 versus 0.019), and Experiment 10 is reasonably fitted. However, it was rejected by the gate and is markedly worse than the accepted iteration-3 base. The newly added validity-profile and weighted-geometry calibration moves several central signatures in exactly the wrong direction. Terminal following collapses in Experiments 3 (0.180 versus 0.575), 4 (-0.106 versus 0.665), and 13 (0.554 versus 0.881), while Experiment 11 also remains too low (0.628 versus 0.843). At the same time, the model still produces excessive tally dependence in Experiment 1 (0.831 versus 0.494), too much highest-validity-cue following in Experiment 2, excessive terminal following in Experiment 8, and excessive overwrite in Experiment 6. Experiment 5 retains a spurious contrast and unusually high subject variance. Thus the added summaries do not separate overwrite-demanding from integration-demanding tasks; they mostly suppress overwrite across descending-validity profiles. In particular, the terminal-to-prefix reliability main effect mechanically classifies weak terminal cues as evidence for integration, even though Experiments 3, 4, 11, and 13 show that coherent or recurrent terminal attacks can dominate despite lower instructed validity.
Rationale: Build again from the unchanged accepted iteration-3 base. My previous recommendation to add a package of terminal-share, coalition-margin, reversal-balance, and validity-profile main effects was rejected, so do not repeat that package or merely widen its coefficients. Remove the direct terminal-to-prefix reliability and log-odds-slope additions: their global sign is empirically wrong and duplicates validity weighting already present in cue evidence. Instead, add at most one narrowly weighted, interaction-based context statistic measuring model-implied reversal susceptibility: under a neutral/reference retention setting, estimate across preceding sequence families how often and by what normalized margin an opposing run would reverse the incumbent accumulator. This combines validity, run length, reversal location, and incumbent strength in the same units as the actual sequential process, rather than summing several correlated ad hoc scores. It should permit recurrent coherent late attacks to promote overwrite in Experiments 3, 4, 11, and 13 without treating every low-validity terminal cue as strong, which is essential for Experiments 6 and 8. Keep iteration 3's static validity sensitivity, local update, coalition gate, and bounded current-trial contribution unchanged. To prevent trial-order volatility, pool the susceptibility statistic with a shrinkage running mean or low-learning-rate estimate rather than the present relatively fast exponential weighting; do not return to the fully history-only calibration that was previously rejected. Tune this single coupling conservatively and require preservation of iteration 3's fits in Experiments 5, 7, 10, 12, and 14. The first acceptance criterion should be directional improvement in Experiments 3/4/11/13 without increasing Experiment 1 or Experiment 8; variance tuning should remain secondary until that cross-task separation is achieved.

**Outcome of your advice:** iter 6 candidate loss=0.2926 -> iter 7 candidate loss=0.1986 -> the gate marked it ACCEPTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: Iteration 7 is the best candidate so far and remains faithful to the prescribed context-calibrated sequential-competition family. The newly introduced reversal-susceptibility statistic was productive: the gate accepted it, aggregate loss fell from 0.2185 to 0.1986, Experiments 3 and 4 moved toward stronger terminal control, Experiment 1 improved slightly, and the good fits in Experiments 7, 10, and 14 were preserved. Nevertheless, the model is not yet experiment-invariant enough to ship. Experiment 1 still shows far too much tally dependence (0.834 versus 0.494). Terminal evidence is substantially too weak in Experiments 3, 4, 11, and 13 (0.391 versus 0.575; 0.248 versus 0.665; 0.638 versus 0.843; 0.729 versus 0.881), and highest-validity-cue following remains too high in Experiment 2 (0.289 versus 0.194). In the opposite direction, terminal control remains excessive in Experiment 8 (0.384 versus 0.172) and in Experiment 6, where early following is only 0.172 versus 0.307. Experiment 5 also retains a modest spurious sequence-geometry contrast. Thus the accepted statistic improves calibration but still does not adequately separate experiments requiring overwrite from those requiring integration. Predicted between-subject variance is also compressed in several important tasks, especially Experiments 1, 8, and 10.
Rationale: Build on the accepted iteration-7 candidate and retain the reversal-susceptibility approach; unlike the earlier multi-statistic validity-profile package, this direction was accepted and should be refined rather than removed. The likely remaining limitation is how susceptibility is summarized: the current arithmetic mean over every run boundary dilutes a coherent terminal attack with earlier, less choice-relevant reversals. Replace that mean with a continuous terminal-weighted or soft-upper-tail aggregation of model-implied reversal susceptibility across sequence families. Later boundaries should receive smoothly greater weight, and recurrent near-reversals may contribute through a softmax or upper-quantile summary rather than a hard threshold. Because the attack is still computed in validity-weighted accumulator units, weak terminal cues in Experiments 6 and 8 need not automatically promote overwrite, whereas recurrent effective terminal attacks can raise the task set in Experiments 2, 3, 4, 11, and 13. Keep the low learning rate, bounded current-trial contribution, static validity sensitivity, coalition-margin gate, local opposition update, and exclusion of physical gaps unchanged. Tune the susceptibility coupling or context threshold only modestly after changing the aggregation; a global increase in overwrite would worsen Experiments 6 and 8. Require improvement in Experiments 3, 4, 11, and 13 without increasing Experiment 8 or the Experiment 1 contrast, while preserving the current fits in Experiments 7, 10, 12, and 14. Subject-level response-gain variation can be widened slightly afterward to address variance compression, but it should not substitute for correcting the directional context calibration.

**Outcome of your advice:** iter 7 candidate loss=0.1986 -> CURRENT candidate loss=0.2077 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate remains faithful to context-calibrated sequential competition and preserves several useful signatures: Experiment 7 has the correct thresholded result, Experiment 10 is close (0.885 versus 0.875), and physical foregrounding remains negligible in Experiment 14. However, the candidate was rejected by the accept gate, with loss increasing from 0.1986 to 0.2077. The terminal-weighted upper-tail modification therefore did not improve experiment invariance. Relative to iteration 7, it modestly improves Experiments 6, 11, and 13, but worsens the already excessive tally interaction in Experiment 1 (0.856 versus 0.494), the spurious contrast in Experiment 5, excessive terminal following in Experiment 8, and Experiment 12. The major failures remain: terminal influence is too weak in Experiments 3, 4, 11, and 13; highest-validity-cue following is too high in Experiment 2; and overwrite is too strong in Experiments 6 and 8. The model consequently still assigns similar task sets to contexts that humans treat very differently. The upper-tail aggregation appears to amplify generally reversible sequence families rather than distinguishing reliably terminal-controlling contexts from protected early-coalition contexts.",
  "verdict": "regenerate",
  "rationale": "Build from the unchanged accepted iteration-7 base. My previous recommendation to replace the arithmetic susceptibility mean with terminal-weighted soft-upper-tail aggregation was rejected by the gate, so do not repeat it or increase its sharpness. Retain the accepted arithmetic model-implied susceptibility statistic, but re-balance the existing context mapping around it. In particular, reduce the large direct coefficients on terminal-singleton rate, two-run rate, and reversal-location concentration. These coarse geometry proxies likely drive excessive overwrite in Experiment 8 and the within-experiment artifact in Experiment 5, regardless of whether the terminal attack is strong enough to reverse the accumulator. Let the accepted susceptibility statistic carry more of the calibration, with only a modest compensating threshold adjustment so average overwrite does not collapse. A useful conservative implementation would shrink those proxy coefficients toward zero while keeping susceptibility coupling near iteration 7, rather than globally increasing overwrite. If the fixed reference retention of 0.55 limits discrimination, replace it with a smooth average over a narrow retention grid centered on the candidate's admissible retention regimes; use reversal robustness across that grid as the same single susceptibility statistic. This remains within the prescribed continuous sequential-competition family and avoids another terminal softmax. Tune against the joint requirement that Experiments 3, 4, 11, and 13 rise while Experiments 6 and 8 move toward greater early protection, with hard preservation checks on Experiments 7, 10, 12, and 14. Do not prioritize variance widening yet; the directional context miscalibration is the primary failure."
}
```

## Usage

```json
{
  "input_tokens": 26931,
  "output_tokens": 1554
}
```
