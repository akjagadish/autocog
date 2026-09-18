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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 with a genuinely different competitor to pi_5: a Conflict-Triggered Attentional Sampling theory. Rather than selecting discrete tally, retrieval, or compression formats at fixed load transitions, subjects should use one continuous stochastic comparison process. Diagnostic cues enter an attentional race whose accessibility jointly reflects communicated validity, display position/recency, and local conflict structure. Sampled signed evidence accumulates toward a stopping bound; unanimous evidence promotes rapid, confident termination, while a newly encountered dissenting cue reopens or prolongs sampling and receives a transient attentional boost. Same-sign cues encountered close together should reinforce one another as a coherent coalition, allowing the late opposing block in Experiment 1 to overwhelm the leading cue, while leading validity can dominate high-load displays without such a coherent recency pattern, as in Experiment 6. Load should affect behavior indirectly by increasing sampling opportunities, conflict, and the probability of reaching the bound before exhaustive inspection—not through hard-coded load-specific formats. Stable individual differences should govern validity sensitivity, recency bias, stopping threshold, conflict-trigger strength, coalition reinforcement, and treatment of equal-validity tiers; the latter can range continuously from first-member selection to whole-tier pooling, accounting for Experiment 7's heterogeneity. This theory should generate the strong unanimous-versus-late-dissenter contrast in Experiment 2, the gradual crossover in Experiment 5, and the positional gradient in Experiment 8 from the same sampling dynamics, while retaining ordinary post-decision lapse noise. It would be a stronger and more parsimonious challenger to pi_5 because it replaces pi_5's numerous load- and tie-specific clauses with a single process and makes novel predictions from cue order, conflict timing, and sign clustering.


## CANDIDATE THEORY
Conflict-Reopening Attentional Race theory: diagnostic cues become available during a noisy, approximately display-ordered attentional race. Their accessibility reflects communicated diagnostic validity and recency, while sampled evidence accumulates toward a subject-specific stopping threshold. Reaching the threshold normally terminates comparison, especially when encountered cues agree. Peripheral detection of an unsampled dissenting cue can reopen the comparison, transiently prioritize opposing evidence, and amplify it. Conflict lateness is represented relative to the active diagnostic sequence and current encounter time, whereas physical cue distance governs coalition formation. Consecutive, spatially close same-sign cues leave a bounded coalition trace that reinforces continued sampling and coherent evidence without changing baseline cue evidence. Equal-validity cues are governed by a steep but continuous pooling disposition, ranging from first-member dominance to effectively complete tier pooling. Load has no strategy gate and matters only through sampling opportunities, conflict, coalition formation, stopping, and reopening.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Conflict-Reopening Attentional Race expects shape (2, n_features); got {stim.shape}."
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
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    validity_access_temperature = float(parameters["validity_access_temperature"])
    validity_evidence_temperature = float(parameters["validity_evidence_temperature"])
    validity_modulation = float(parameters["validity_modulation"])
    recency_bias = float(parameters["recency_bias"])
    stopping_threshold = float(parameters["stopping_threshold"])
    conflict_trigger = float(parameters["conflict_trigger"])
    coalition_reinforcement = float(parameters["coalition_reinforcement"])
    tier_pooling = float(parameters["tier_pooling"])
    sampling_noise = float(parameters["sampling_noise"])
    decision_gain = float(parameters["decision_gain"])

    # Communicated validity magnitudes are retained across experiments. A
    # bounded log-odds transform avoids experiment-wise min-max rescaling while
    # allowing accessibility and evidence strength to have modestly different
    # validity temperatures.
    clipped_validities = np.clip(validities, 0.500001, 0.999999)
    validity_log_odds = np.log(clipped_validities / (1.0 - clipped_validities))
    v_access = np.tanh(validity_log_odds / validity_access_temperature)
    v_evidence = np.tanh(validity_log_odds / validity_evidence_temperature)

    if n_features > 1:
        position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        position = np.zeros(n_features, dtype=float)

    # Conflict lateness is relative to the diagnostic sequence rather than the
    # full display. Physical positions remain available for onset and coalition
    # proximity.
    active_lateness = np.zeros(n_features, dtype=float)
    if active.size > 1:
        for active_rank, cue in enumerate(active):
            active_lateness[int(cue)] = active_rank / float(active.size - 1)

    # A fixed deterministic particle set approximates the stochastic race.
    # Different particles instantiate attentional fluctuations, while model
    # probabilities remain reproducible and do not add Monte Carlo instability.
    n_particles = 64
    particle_p_a = []

    def pseudo_uniform(particle, cue, stream):
        x = np.sin(
            (particle + 1.0) * 12.9898
            + (cue + 1.0) * 78.233
            + (stream + 1.0) * 37.719
        ) * 43758.5453123
        u = x - np.floor(x)
        return float(np.clip(u, 1e-9, 1.0 - 1e-9))

    for particle in range(n_particles):
        remaining = [int(j) for j in active]
        accumulator = 0.0
        last_sign = 0.0
        last_index = -1
        encountered_signs = []
        sampled_by_tier = {}
        alert_sign = 0.0
        reopened = False
        coalition_trace = 0.0
        step = 0

        while remaining:
            race_scores = []
            for cue in remaining:
                # Cues enter in approximate display order, but communicated
                # validity, recency, noise, and a conflict alert can change
                # which available cue wins attention next.
                temporal_onset = -4.0 * position[cue]
                accessibility = (
                    temporal_onset
                    + validity_sensitivity * v_access[cue]
                    + recency_bias * position[cue]
                )
                if alert_sign != 0.0 and signs[cue] == alert_sign:
                    accessibility += conflict_trigger

                u = pseudo_uniform(particle, cue, step + 3)
                gumbel = -np.log(-np.log(u))
                race_scores.append(accessibility + sampling_noise * gumbel)

            winner = int(np.argmax(np.asarray(race_scores, dtype=float)))
            cue = remaining.pop(winner)
            cue_sign = float(signs[cue])

            # Evidence retains the accepted bounded validity representation.
            # A small capped multiplicative modulation increases communicated-
            # validity differentiation without allowing one cue to outweigh an
            # arbitrarily large opposing coalition.
            increment_size = (
                1.0
                + 0.42 * validity_sensitivity * v_evidence[cue]
                + 0.32 * recency_bias * position[cue]
            )
            capped_log_odds = min(float(validity_log_odds[cue]), 4.0)
            increment_size *= 1.0 + validity_modulation * capped_log_odds

            # A newly sampled cue opposing the provisional conclusion receives
            # a transient boost that increases with its lateness in the active
            # diagnostic sequence.
            if accumulator * cue_sign < 0.0:
                late_conflict_scale = 0.55 + 0.75 * active_lateness[cue]
                increment_size *= 1.0 + conflict_trigger * late_conflict_scale

            # Nearby same-sign samples build a bounded coalition trace. A sign
            # change or spatial gap rapidly erases it, preventing scattered
            # evidence from receiving the benefit of a coherent physical run.
            if last_sign == cue_sign and last_index >= 0:
                separation = abs(cue - last_index)
                proximity = np.exp(-float(separation - 1) / 1.5)
                coalition_trace = min(
                    1.0, 0.35 * coalition_trace + float(proximity)
                )
                increment_size *= 1.0 + coalition_reinforcement * proximity * (
                    0.65 + 0.35 * coalition_trace
                )
            else:
                coalition_trace = 0.0

            # Equal-validity treatment remains continuous but is steep enough
            # to generate stable first-member-like and whole-tier-like subjects.
            tier_key = int(np.sum(validities > validities[cue] + 1e-12))
            previous_in_tier = sampled_by_tier.get(tier_key, 0)
            if previous_in_tier > 0:
                effective_pooling = 1.0 / (
                    1.0 + np.exp(-12.0 * (tier_pooling - 0.38))
                )
                increment_size *= 0.08 + 0.92 * effective_pooling
            sampled_by_tier[tier_key] = previous_in_tier + 1

            accumulator += cue_sign * increment_size
            encountered_signs.append(cue_sign)
            last_sign = cue_sign
            last_index = cue
            alert_sign = 0.0
            step += 1

            if abs(accumulator) >= stopping_threshold:
                provisional_sign = 1.0 if accumulator > 0.0 else -1.0
                dissenters = [j for j in remaining if signs[j] == -provisional_sign]

                if len(dissenters) == 0:
                    # Unanimity licenses rapid, confident termination.
                    break

                # Monitoring uses lateness in the active diagnostic sequence
                # together with elapsed encounter time. The bounded trace makes
                # conflict following a coherent sampled run more likely to be
                # noticed, without changing baseline evidence accessibility.
                latest_dissent = max(active_lateness[j] for j in dissenters)
                encounter_lateness = min(1.0, step / float(active.size))
                conflict_lateness = 0.75 * latest_dissent + 0.25 * encounter_lateness
                surplus = max(0.0, abs(accumulator) - stopping_threshold)
                monitor_logit = (
                    -1.45
                    + 0.85 * conflict_trigger
                    + 1.25 * recency_bias * conflict_lateness
                    + 0.55 * coalition_trace
                    - 0.45 * surplus
                    + (0.35 if not reopened else -0.35)
                )
                monitor_prob = 1.0 / (1.0 + np.exp(-np.clip(monitor_logit, -40.0, 40.0)))
                monitor_u = pseudo_uniform(particle, dissenters[-1], 100 + step)

                if monitor_u < monitor_prob:
                    alert_sign = -provisional_sign
                    reopened = True
                    # Reopening partially discounts commitment to the bound,
                    # without erasing the evidence that generated it.
                    accumulator *= 0.82
                else:
                    break

        logit_a = np.clip(decision_gain * accumulator, -60.0, 60.0)
        if logit_a >= 0.0:
            z = np.exp(-logit_a)
            p_a = 1.0 / (1.0 + z)
        else:
            z = np.exp(logit_a)
            p_a = z / (1.0 + z)
        particle_p_a.append(p_a)

    p_a_core = float(np.mean(particle_p_a))
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
- validity_sensitivity: [1.4, 3.2]
- validity_access_temperature: [1.1, 1.8]
- validity_evidence_temperature: [0.7, 1.3]
- validity_modulation: [0.04, 0.10]
- recency_bias: [0.8, 2.2]
- stopping_threshold: [1.8, 3.4]
- conflict_trigger: [1.2, 3.2]
- coalition_reinforcement: [0.55, 1.65]
- tier_pooling: [0, 1]
- sampling_noise: [0.65, 1.15]
- decision_gain: [2.6, 5.2]
- epsilon: [0, 0.06]

`rationale`:
This is a minimal edit to the accepted iteration-5 source. It retains the accepted non-normalized bounded validity transform, active-sequence lateness, threshold, fully accessible race, and lapse process. First, validity receives only a small capped multiplicative modulation, kept much closer to the accepted additive calibration than to the rejected direct-log-odds formulation. This should moderately increase validity leadership in Experiment 4 and reduce excessive tally following in Experiment 2 without recreating Experiment 7's lexicographic excess. Second, the existing local coalition calculation is converted into a bounded, rapidly decaying sampled-history trace. The trace affects coherent same-sign continuation and conflict monitoring but not baseline accessibility, so physical blocks can matter in Experiments 1 and 8 while scattered overload conflict receives little benefit. Conflict amplification is also graded by active-sequence lateness, targeting the negative positional profile in Experiment 6 and the late-loser contrast in Experiment 8 without a hard release window. Finally, the steep tier-pooling midpoint is lowered from 0.50 to 0.38, transplanting the isolated calibration that successfully produced Experiment 5's approximately split population in the rejected iteration while omitting that iteration's damaging release and direct-evidence changes.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.5094 -> ACCEPTED
- iter 2: loss=0.6254 -> REJECTED
- iter 3: loss=0.5556 -> REJECTED
- iter 4: loss=0.5510 -> REJECTED
- iter 5: loss=0.4997 -> ACCEPTED
- iter 6: loss=0.5934 -> REJECTED
- iter 7 (current candidate you are grading): loss=0.4786 -> ACCEPTED
Running-best (last accepted) base: iter 7 at loss=0.4786.

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
  - iter 1: 0.6589 (var=0.0089) (Δ vs real +0.6889)
  - iter 2: 0.7661 (var=0.0086) (Δ vs real +0.7961)
  - iter 3: 0.7444 (var=0.0081) (Δ vs real +0.7744)
  - iter 4: 0.5500 (var=0.0093) (Δ vs real +0.5800)
  - iter 5: 0.6478 (var=0.0146) (Δ vs real +0.6778)
  - iter 6: 0.5144 (var=0.0261) (Δ vs real +0.5444)
  - iter 7 (current): 0.6522 (var=0.0120) (Δ vs real +0.6822)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0167 (var=0.0052)
- pi_2: 0.8033 (var=0.0243)
- pi_3: -0.0028 (var=0.0068)
- pi_4: 0.0328 (var=0.0015)
- pi_5: 0.0367 (var=0.0093)

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
  - iter 1: 0.9100 (var=0.0044) (Δ vs real +0.2417)
  - iter 2: 0.9658 (var=0.0016) (Δ vs real +0.2975)
  - iter 3: 0.9433 (var=0.0024) (Δ vs real +0.2750)
  - iter 4: 0.7267 (var=0.0228) (Δ vs real +0.0583)
  - iter 5: 0.9017 (var=0.0057) (Δ vs real +0.2333)
  - iter 6: 0.0375 (var=0.0207) (Δ vs real -0.6308)
  - iter 7 (current): 0.9142 (var=0.0096) (Δ vs real +0.2458)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7008 (var=0.0514)
- pi_1: -0.6542 (var=0.0564)
- pi_3: 0.6600 (var=0.0188)
- pi_4: 0.5833 (var=0.0114)
- pi_5: 0.6467 (var=0.0132)

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
  - iter 1: 0.0291 (var=0.0100) (Δ vs real -0.3322)
  - iter 2: 0.0075 (var=0.0007) (Δ vs real -0.3537)
  - iter 3: 0.0656 (var=0.0036) (Δ vs real -0.2956)
  - iter 4: 0.0422 (var=0.0031) (Δ vs real -0.3191)
  - iter 5: 0.0919 (var=0.0107) (Δ vs real -0.2694)
  - iter 6: 0.0181 (var=0.0101) (Δ vs real -0.3431)
  - iter 7 (current): 0.1434 (var=0.0083) (Δ vs real -0.2178)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6512 (var=0.0114)
- pi_2: 0.0066 (var=0.0037)
- pi_1: 0.0041 (var=0.0094)
- pi_4: 0.6541 (var=0.0103)
- pi_5: 0.5350 (var=0.0071)

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
  - iter 1: 0.4428 (var=0.0085) (Δ vs real -0.4328)
  - iter 2: 0.4850 (var=0.0116) (Δ vs real -0.3906)
  - iter 3: 0.4834 (var=0.0041) (Δ vs real -0.3922)
  - iter 4: 0.2938 (var=0.0099) (Δ vs real -0.5819)
  - iter 5: 0.4281 (var=0.0069) (Δ vs real -0.4475)
  - iter 6: 0.5419 (var=0.0112) (Δ vs real -0.3337)
  - iter 7 (current): 0.5209 (var=0.0104) (Δ vs real -0.3547)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5031 (var=0.0037)
- pi_3: 0.8291 (var=0.0019)
- pi_1: 0.8391 (var=0.0151)
- pi_4: 0.8331 (var=0.0025)
- pi_5: 0.8884 (var=0.0019)

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
  - iter 1: 0.9600 (var=0.0284) (Δ vs real +0.4600)
  - iter 2: 0.9800 (var=0.0196) (Δ vs real +0.4800)
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.5000)
  - iter 4: 0.5600 (var=0.2264) (Δ vs real +0.0600)
  - iter 5: 0.8400 (var=0.1144) (Δ vs real +0.3400)
  - iter 6: 0.5200 (var=0.2096) (Δ vs real +0.0200)
  - iter 7 (current): 0.9000 (var=0.0800) (Δ vs real +0.4000)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.5800 (var=0.2436)

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
  - iter 1: 0.0104 (var=0.0024) (Δ vs real +0.1235)
  - iter 2: 0.0575 (var=0.0028) (Δ vs real +0.1706)
  - iter 3: 0.0226 (var=0.0044) (Δ vs real +0.1358)
  - iter 4: 0.1147 (var=0.0030) (Δ vs real +0.2279)
  - iter 5: 0.0242 (var=0.0021) (Δ vs real +0.1374)
  - iter 6: 0.0452 (var=0.0026) (Δ vs real +0.1583)
  - iter 7 (current): 0.0230 (var=0.0026) (Δ vs real +0.1361)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0970 (var=0.0005)
- pi_3: 0.0009 (var=0.0007)
- pi_1: 0.0003 (var=0.0005)
- pi_2: 0.0004 (var=0.0004)
- pi_5: -0.1079 (var=0.0009)

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
  - iter 1: 0.1300 (var=0.0148) (Δ vs real +0.0125)
  - iter 2: 0.0375 (var=0.0035) (Δ vs real -0.0800)
  - iter 3: 0.0563 (var=0.0039) (Δ vs real -0.0612)
  - iter 4: 0.4119 (var=0.0403) (Δ vs real +0.2944)
  - iter 5: 0.1906 (var=0.0169) (Δ vs real +0.0731)
  - iter 6: 0.5394 (var=0.0176) (Δ vs real +0.4219)
  - iter 7 (current): 0.1719 (var=0.0273) (Δ vs real +0.0544)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7987 (var=0.0067)
- pi_5: 0.0850 (var=0.0021)
- pi_1: 0.8706 (var=0.0097)
- pi_2: 0.1344 (var=0.0103)
- pi_4: 0.9775 (var=0.0008)

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
  - iter 1: 0.3394 (var=0.0183) (Δ vs real -0.4356)
  - iter 2: 0.0031 (var=0.0013) (Δ vs real -0.7719)
  - iter 3: 0.2150 (var=0.0233) (Δ vs real -0.5600)
  - iter 4: 0.1331 (var=0.0218) (Δ vs real -0.6419)
  - iter 5: 0.3219 (var=0.0279) (Δ vs real -0.4531)
  - iter 6: -0.0075 (var=0.0004) (Δ vs real -0.7825)
  - iter 7 (current): 0.3875 (var=0.0154) (Δ vs real -0.3875)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3750 (var=0.0124)
- pi_3: -0.0169 (var=0.0083)
- pi_1: -0.0088 (var=0.0039)
- pi_2: 0.0231 (var=0.0063)
- pi_4: 0.0719 (var=0.0047)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed conflict-triggered attentional-sampling family, and it succeeds notably on Experiment 7: leading-cue agreement is 0.130 versus 0.118 observed. It also produces the correct positive unanimous-versus-late-dissenter contrast in Experiment 8, but only about half its required magnitude. Overall, however, the process is not experiment-invariant yet. Experiment 1 is badly wrong (0.659 versus -0.030): matched tally conflict has far too large an effect on top-cue adherence, indicating that the late contiguous coalition does not selectively neutralize the early top-cue advantage. Experiment 4 shows the opposite necessary failure: validity-leading choices under incoherent overload are much too rare (0.443 versus 0.876). Thus the model does not sufficiently distinguish coherent late opposition from scattered opposition. Experiment 3 has almost no load crossover (0.029 versus 0.361), and Experiment 6 has an essentially flat/slightly positive positional profile (0.010 versus -0.113). Experiment 2 is excessively tally-consistent (0.910 versus 0.668). Finally, Experiment 5 misses both the mean and heterogeneity: 0.960 with variance 0.028 versus 0.500 with variance 0.200, so the nominal tier-pooling individual difference is not behaviorally effective.
Rationale: Keep the attentional race, stopping, reopening, coalition, and continuous tier-pooling architecture, but recalibrate how these components interact. The fixed temporal-onset term of -4 dominates the allowed recency range, while the same conflict_trigger parameter broadly controls monitoring, accessibility, and evidence multiplication. This creates strong generic primacy followed by indiscriminate conflict reversals rather than the required selective response to conflict timing and sign clustering. Increase validity-based evidence sufficiently to recover leading-cue dominance for the incoherent overload displays in Experiment 4 and reduce the excessive tally preference in Experiment 2, while making reversal depend much more sharply on adjacency/coherence and late position. Strengthen truly local same-sign coalition reinforcement, but attenuate generic boosts for any cue opposing the accumulator; this should let the contiguous late block matter in Experiment 1 without allowing scattered opposition to erase the leading cue in Experiment 4. Put recency primarily into conflict detection and the transient dissent boost rather than adding it uniformly to every cue's evidence. That should also generate the missing negative positional gradient in Experiment 6 and enlarge the Experiment 8 contrast. Recalibrate the threshold relative to single-cue increments so one early cue does not terminate low-load trials, while high-load displays can still reach the bound before exhaustive inspection; this is needed for Experiment 3's load crossover. Finally, do not permit terminal stopping before unresolved members of the currently sampled equal-validity tier have a pooling-dependent opportunity to enter the race. The current early stopping renders tier_pooling nearly inert, causing Experiment 5's homogeneous first-member behavior. Making tier accessibility or stopping continuation vary continuously with tier_pooling should produce both first-member and pooled-majority subjects without adding a tie-specific strategy switch.

**Outcome of your advice:** iter 1 candidate loss=0.5094 -> iter 2 candidate loss=0.6254 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-triggered attentional-sampling family, but the current recalibration is a clear regression: aggregate loss rose from 0.5094 to 0.6254 and the gate rejected it. The only material improvement is Experiment 4, where leading-validity adherence increased from 0.443 to 0.485, but it remains far below 0.876. Most central signatures worsened. Experiment 1 moved farther from the near-zero contrast (0.766 versus -0.030), Experiment 2 became almost deterministically tally-consistent (0.966 versus 0.668), Experiment 3 lost nearly all load crossover (0.008 versus 0.361), Experiment 6 acquired the wrong positive positional gradient (0.058 versus -0.113), and Experiment 8's unanimous-versus-late-dissenter contrast collapsed from 0.339 to 0.003 rather than approaching 0.775. Experiment 5 also remains almost uniformly first-cue aligned (0.980, variance 0.020) instead of showing the observed split population (0.500, variance 0.200). Experiment 7 was previously accurate but is now somewhat too low. The current high bound and near-exhaustive processing appear to wash out stopping-order effects, while late conflict amplification does not work because monitoring uses only the latest dissenter's position and not the coherence of the unresolved dissenting run. Moreover, the equal-tier protection has a bookkeeping flaw: protection is considered only for the tier of the cue that crosses the bound, so an earlier opened top tier can still be abandoned when a lower-tier cue triggers stopping.
Rationale: Because the previous broad push toward a higher threshold, sharply local coalition reinforcement, and conflict-only recency was rejected by the gate, do not repeat or intensify that package. Build the next attempt on the unchanged accepted base and make smaller, separable in-family changes. First, test a moderately lower stopping threshold while keeping the maximum single-cue increment below it. Stopping after a short coalition—but not after one cue—should restore the order-sensitive contrasts in Experiments 3 and 8; the current 3.2–4.8 range promotes exhaustive sampling and produces nearly deterministic, low-variance outcomes. Second, change reopening detection from `latest_dissent` alone to a cluster-aware statistic computed over unresolved cues, combining lateness with adjacent same-sign run length. Once a coherent run is detected, temporarily protect or accelerate that run. This is different from simply increasing the post-sampling adjacency multiplier: it allows the late block in Experiment 1 to enter the race while leaving scattered opposition in Experiment 4 relatively weak. Third, restore a modest graded recency contribution to race accessibility rather than concentrating recency almost entirely in conflict amplification; removing that component moved Experiment 6 in the wrong direction. Finally, fix tier bookkeeping independently: maintain every previously opened equal-validity tier until a pooling-dependent continuation decision is made, rather than checking only the threshold-crossing cue's tier. Later members should receive full rather than repeatedly multiplicative pooling weight when pooling is high. This targeted correction is needed for Experiment 5's mean and between-subject variance and can be evaluated without disturbing strictly ranked experiments.

**Outcome of your advice:** iter 2 candidate loss=0.6254 -> iter 3 candidate loss=0.5556 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-triggered attentional-sampling family, but it was rejected by the accept gate and remains substantially worse than the accepted iteration-1 base. The cluster-aware reopening revision partially recovered Experiments 3 and 8 relative to iteration 2, but neither recovered its target or even the accepted base on Experiment 8. The central failures remain severe: Experiment 1 predicts a very large tally-congruence effect (0.744 versus -0.030), Experiment 4 shows far too little validity-leading behavior under scattered overload (0.483 versus 0.876), and Experiment 2 is nearly deterministically tally-consistent (0.943 versus 0.668). Experiment 3 still lacks the human load crossover (0.066 versus 0.361), while Experiments 6 and 8 lack sufficiently strong effects of late dissent (0.023 versus -0.113 and 0.215 versus 0.775). Experiment 5 actually became completely homogeneous: 1.000 with zero variance versus an observed 0.500 with variance 0.200. Thus tracking opened tiers did not make tier_pooling behaviorally effective. The present mechanism overweights the size of an unresolved dissenting run while failing to distinguish a salient first dissent after a coherent sequence from repeated conflict in an already mixed display.
Rationale: Do not repeat the rejected pushes toward lower thresholds, stronger run-length protection, or further tier bookkeeping changes. Build instead on the unchanged iteration-1 base and revise the conflict gate itself. Make reopening and transient amplification depend primarily on conflict onset history: a first dissent encountered after a long uninterrupted same-sign sequence should receive a large, recency-sensitive boost, whereas subsequent dissent after earlier sign changes should be habituated or discounted. Cap or saturate the contribution of adjacent-run length once local coherence is established, rather than scaling strongly with the complete run length. This should make late dissent powerful in Experiments 1, 6, and 8 without making tally margin determine Experiment 1's contrast, while mixed and repeatedly conflicting displays in Experiment 4 retain validity-led choices. Add validity-weighted resistance to reopening—based on how much of the provisional bound was supplied by high-validity evidence—rather than globally increasing validity increments. That can reduce excessive tally following in Experiment 2 and protect leading-cue decisions in Experiment 4, while the first-dissent boost can still overturn commitment after a genuinely coherent late transition. Finally, resolve equal-validity evidence before treating disagreement within that tier as a conflict event: continuously interpolate between first-member commitment and pooled tier evidence using tier_pooling, and suppress within-tier reopening in proportion to pooling. The current model lets conflict amplification undo pooling even when all tier members are sampled, explaining Experiment 5's unanimous first-member classification. A steep but continuous mapping from tier_pooling to this pre-conflict tier integration should produce both behavioral subpopulations and the required between-subject variance without introducing a discrete strategy gate.

**Outcome of your advice:** iter 3 candidate loss=0.5556 -> iter 4 candidate loss=0.5510 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed attentional-sampling family, but it was rejected by the gate and remains worse than the accepted iteration-1 base. The onset-habituation revision produces useful local gains: Experiment 2 is now close to the observed tally-consistency rate (0.727 versus 0.668), and Experiment 5 now captures both the split mean and heterogeneity (0.560, variance 0.226 versus 0.500, variance 0.200). These gains are outweighed by severe regressions and remaining failures. Experiment 1 still has a large positive matched-congruence effect (0.550 versus -0.030). Experiment 3 still lacks the load crossover (0.042 versus 0.361). Validity-leading behavior in Experiment 4 collapses to 0.294 versus 0.876. Experiment 6 now has a strong effect in the wrong direction (0.115 versus -0.113), Experiment 7 rises from an accurate accepted-base value to 0.412 versus 0.118, and Experiment 8's contrast remains far too small (0.133 versus 0.775). Thus the current conflict gate improves selected conflict and tier cases but does not generalize across experiments.
Rationale: Do not further intensify the conflict-onset/habituation package: that was the previous recommendation, and the resulting candidate was rejected. Start from the unchanged accepted base and try a different in-family correction centered on validity representation. The current experiment-wise min-max scaling erases the communicated magnitude and spacing of validities: every experiment receives a cue with strength 1 and a cue with strength 0 regardless of its actual reliability structure. This makes it difficult to obtain strong validity leadership in Experiment 4 while retaining weak lexicographic agreement in Experiment 7, and it encourages the same generic conflict dynamics across displays that should behave differently. Replace min-max ranks with a bounded diagnostic transform of the communicated values, such as temperature-scaled log odds, without subtracting each experiment's minimum. Allow a modest separate temperature for validity in race accessibility versus evidence increments, while retaining the same continuous race and stopping mechanism. This should preserve genuine between-experiment validity profiles, strengthen high-validity commitment where warranted, and let lower-validity cues retain nonzero diagnostic weight.

As a second targeted correction, compute conflict lateness from a cue's rank among active diagnostic cues and from encounter time, not solely from its absolute feature index. Experiment 8 defines a late loser relative to the active set, whereas the present absolute-position term can treat an active-set-final cue as only moderately late; physical index distance can remain the basis for coalition proximity. This coordinate separation is also a plausible route to correcting Experiment 6's reversed positional gradient without another global increase in conflict strength. Preserve the steep continuous tier-pooling behavior only as an isolated component, since it uniquely repaired Experiment 5, but do not carry over the broader onset gate that damaged Experiments 4, 6, 7, and 8. Evaluate these changes as separable ablations rather than jointly retuning thresholds or coalition strength, directions already rejected in prior rounds.

**Outcome of your advice:** iter 4 candidate loss=0.5510 -> iter 5 candidate loss=0.4997 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed conflict-triggered attentional-sampling family, and the latest changes were directionally useful: aggregate loss improved from 0.5094 to 0.4997, so the accepted validity representation and active-sequence lateness changes should be retained. Experiment 3 improved from 0.029 to 0.092, and Experiment 8 retains the correct positive contrast with exactly the observed between-subject variance. Experiment 7 is also reasonably close. Nevertheless, the model is not experiment-invariant enough to ship. Experiment 1 remains grossly wrong (0.648 versus -0.030), showing ordinary tally-congruence sensitivity rather than selective domination by late coherent opposition. Experiment 4 remains far too weakly validity-led under overload (0.428 versus 0.876), and Experiment 2 is too tally-consistent and too homogeneous (0.902, variance 0.006 versus 0.668, variance 0.037). The load crossover in Experiment 3 is still only about one quarter of the observed effect. Late-conflict position is also insufficient in Experiment 6 and has the wrong sign relative to the target (0.024 versus -0.113), while Experiment 8 reaches only 0.322 versus 0.775. Finally, the steep tier-pooling mapping has not preserved the useful iteration-4 result: Experiment 5 has returned to predominantly first-member behavior (0.840 versus 0.500) with insufficient heterogeneity.
Rationale: Build on the accepted candidate and retain its non-normalized communicated-validity transform and active-sequence lateness calculation; that advice improved the aggregate loss. The next correction should target how the attentional race implements order. At present, every remaining cue is available from the first step and `temporal_onset` is merely a static score penalty. Consequently, noisy winners can scramble physical runs, coalition reinforcement depends on accidental sampling adjacency, and monitoring can inspect every unresolved dissenter immediately. Implement genuine graded release times or a moving attentional window: later cues should become available as encounter time advances, with conflict-triggered reopening accelerating a detected dissenter and its nearby same-sign neighbors. This remains one continuous attentional race, but should make physical sign blocks behaviorally coherent. It offers a common correction for Experiments 1, 6, and 8 without repeating the previously rejected strategy of globally increasing coalition strength or lowering the stopping threshold. Scattered conflict in Experiment 4 should receive much less benefit than an ordered late block.

The accepted validity change also needs a narrower calibration adjustment. The current evidence formula adds a large common baseline of 1.0, while the tanh transform makes all valid cues positive and often compressed; thus communicated validity differences remain too weak to produce Experiment 4's leading-cue dominance. Preserve global, non-experiment-normalized log odds, but let evidence magnitude depend more directly and multiplicatively on globally scaled log odds, with a small floor, rather than adding a modest validity term to an equal-weight increment. Avoid simply increasing `validity_sensitivity`, which could damage Experiment 7. Separate accessibility from evidence so validity mainly controls evidence strength while ordered release controls encounter order. This should also reduce excessive tally following in Experiment 2.

Finally, make the isolated tier-pooling parameter behaviorally effective through a minor calibration rather than another broad tier redesign: lower the sigmoid midpoint or skew the allowed `tier_pooling` range upward until roughly half the simulated subjects pool the top tier strongly enough for its 2-to-1 majority to prevail. The present nominal 50/50 sigmoid yields only about 16% non-first-member classifications. Preserve the steep continuous mapping and verify both Experiment 5's mean and variance. Do not revive the rejected broad conflict-onset gate, generic run-length amplification, or wholesale threshold reductions.

**Outcome of your advice:** iter 5 candidate loss=0.4997 -> iter 6 candidate loss=0.5934 -> the gate marked it REJECTED.

### Iteration 6 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed conflict-triggered attentional-sampling family, but the bundled revision was rejected and substantially worsened aggregate fit (0.5934 versus the accepted base's 0.4997). Its main success is Experiment 5: the lower tier-pooling midpoint produces the correct split mean and variance (0.520, var 0.210 versus 0.500, var 0.200). Experiment 4 also improves to 0.542, though it remains well below the observed 0.876, and Experiment 1 improves to 0.514 but is still grossly above the near-zero target. These gains are overwhelmed by severe regressions. Experiment 2 shifts from excessive tally following to almost no tally following (0.038 versus 0.668), while Experiment 7 becomes far too lexicographic (0.539 versus 0.118). This shows that direct log-odds evidence overshot the required validity calibration. The hard moving release window also eliminates rather than strengthens the late-dissenter signature: Experiment 8 falls from 0.322 to -0.008 versus 0.775, Experiment 6 remains wrong-signed (0.045 versus -0.113), and Experiment 3 loses its limited crossover improvement (0.018 versus 0.361). Evidently, withholding later cues lets the process terminate before the critical dissenting evidence becomes effective; conditional acceleration does not compensate reliably.
Rationale: Because the previous recommendation's graded-release/direct-evidence package was rejected, do not repeat or intensify it. Build on the unchanged accepted iteration-5 base. For validity, take an intermediate calibration rather than oscillating between the accepted additive formula and the current dominant direct-log-odds formula: retain the accepted bounded, non-normalized validity representation and add only a modest multiplicative log-odds modulation, capped so the highest-validity cue cannot dominate several opposing cues by itself. Experiment 2 brackets the target sharply—0.902 in the accepted base and 0.038 here around an observed 0.668—so the next step should be much closer to the accepted scaling than to the current one. This should improve Experiment 4 without recreating Experiment 7's large lexicographic excess.

For order effects, leave all diagnostic cues probabilistically accessible and replace the rejected hard release window with a bounded, decaying coalition-eligibility trace over actually sampled evidence. Consecutive same-sign cues that are also physically nearby should build a saturating trace; intervening opposite signs or large spatial gaps should rapidly erase it. Apply the trace only to conflict-triggered continuation or the next same-sign increment, not to baseline evidence. This is a smaller in-family change than advance cluster protection or hard cue release: it can selectively amplify genuinely encountered late blocks in Experiments 1, 6, and 8 while giving little advantage to scattered conflict in Experiment 4. Avoid changing the global threshold again, since prior threshold-centered revisions were rejected.

Finally, the current tier-pooling calibration is the one clearly successful local component. Transplant that component as an isolated ablation onto the accepted base—without the hard release and direct-evidence changes—and verify that it preserves Experiment 5's mean and between-subject variance. The next candidate should therefore separate three effects rather than rebundle them: accepted validity/lateness as the base, a small capped validity modulation, and a sampled-history coalition trace, with the demonstrated tier calibration evaluated independently.

**Outcome of your advice:** iter 6 candidate loss=0.5934 -> CURRENT candidate loss=0.4786 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is faithful to the prescribed conflict-triggered attentional-sampling family, and the latest accepted revision is a genuine aggregate improvement: loss fell from 0.4997 to 0.4786. The accepted sampled-history trace and modest validity modulation should therefore be retained. They improve Experiment 3's crossover to 0.143 and Experiment 8's contrast to 0.388, while Experiment 4 rises to 0.521 and Experiment 7 remains fairly close at 0.172 versus 0.118. Nevertheless, the model is not experiment-invariant enough to ship. Experiment 1 remains grossly wrong at 0.652 versus -0.030, indicating that coalition reinforcement still behaves like amplified tally accumulation rather than a selective response to an encountered coherent block. Experiment 2 is too tally-consistent and too homogeneous at 0.914 with variance 0.010 versus 0.668 with variance 0.037. Experiment 4 remains far below the strong human validity-leading rate of 0.876. Experiment 3 still produces less than half the observed load crossover. The positional effect in Experiment 6 remains weak and wrong-signed, 0.023 versus -0.113, and Experiment 8 reaches only half the required unanimous-versus-late-dissenter contrast. Finally, transplanting the previously successful tier midpoint did not preserve Experiment 5: the result returned to 0.900 with variance 0.080 rather than the observed split population at 0.500 with variance 0.200.",
  "verdict": "regenerate",
  "rationale": "Build on the accepted iteration-7 candidate and retain its bounded non-normalized validity transform, active-sequence lateness, fully accessible race, and sampled-history coalition trace. The latest accepted direction helped, but the trace is currently paid repeatedly: every adjacent same-sign sample receives a multiplicative coalition bonus. Consequently, longer blocks acquire more total reinforced increments, preserving the large tally-margin effect in Experiment 1. Convert this into a finite, saturating coalition budget or one-shot coalition impulse per coherent run, applied mainly when conflict is first encountered or to the first continuation after conflict. Coalition coherence should matter, but additional members beyond establishing the block should not repeatedly multiply evidence. This should reduce Experiment 1's tally sensitivity while preserving or enlarging the order effects gained in Experiments 3 and 8.\n\nRebalance the existing conflict gate rather than globally increasing conflict strength or changing the threshold. Reduce the generic conflict boost and make its fixed lateness slope substantially steeper, preferably separating the coefficients for monitoring and transient evidence amplification while retaining one subject-level conflict-sensitivity parameter. The current broad baseline boost affects ordinary conflicts, but its lateness modulation is too weak to produce Experiment 6's negative positional gradient. A lower baseline with a stronger late-conflict component should make a genuinely late dissenter consequential in Experiments 6 and 8 without allowing scattered opposition to dominate Experiment 4 or ordinary tally margins to control Experiment 1.\n\nFor validity, do not return to the rejected direct-log-odds evidence formulation. The modest modulation improved Experiment 4 without causing Experiment 7's extreme regression, but further evidence multiplication risks overshooting again. Instead, make a small increase in validity selectivity on race accessibility, such as modestly lowering the accessibility temperature, while leaving the capped evidence modulation near its current range. This can prioritize the validity-leading diagnostic cue in Experiments 2 and 4 without allowing one cue's evidence magnitude to overwhelm an opposing coalition.\n\nFinally, do not merely lower the tier-pooling sigmoid midpoint again: that isolated transplant failed here despite succeeding in the rejected release-window candidate. The parameter currently attenuates later tier members but does not reliably ensure that high-pooling subjects inspect unresolved same-tier evidence before stopping. Let the same continuous tier_pooling value modestly raise monitoring or continuation probability specifically when unresolved members of the currently active equal-validity tier remain. Keep this probabilistic rather than imposing hard tier completion. That should make high-pooling subjects express the 2-to-1 tier majority and restore Experiment 5's mean and variance while leaving strictly ranked experiments unchanged."
}
```

## Usage

```json
{
  "input_tokens": 20812,
  "output_tokens": 2481
}
```
