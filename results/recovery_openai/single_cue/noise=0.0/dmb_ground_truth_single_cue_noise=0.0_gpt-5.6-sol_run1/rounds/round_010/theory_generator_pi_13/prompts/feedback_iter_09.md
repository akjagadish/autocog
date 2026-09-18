# feedback_iter_09

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
- THEORY 2 = `pi_12`
- The recommendation below acts on THEORY 2 (= `pi_12`).

Replace pi_12 with a Spatially Indexed Reliability-Gated Revision theory. Subjects should normally inspect cues in physical left-to-right order, while instructed validity controls cue accessibility and revision strength rather than imposing a separate validity-sorted scan. Each attended discriminating cue updates a bounded directional commitment through leaky, recency-weighted revision. Repeated same-direction cues increase commitment with saturation, but they should not form the rigid coalitions responsible for pi_12's failures in Experiments 7, 10, 13, and 14. A local drop from a reliable incumbent cue to a weak opposing cue should trigger an orienting or surprise response that transiently increases attention to the challenger, accounting for Experiment 2. Absolute right-edge position and terminal isolation should also increase accessibility, accounting for Experiment 1, but these bonuses must be conditional and modest rather than guaranteeing a coalition switch. Commitment stability should depend jointly on recent direction, accumulated coherent support, cue validity, and load: recent cues can dominate short or manageable conflicts, as in Experiments 6, 12, 13, and 15, while strong coherent prefixes can resist a terminal singleton in other configurations, as required by Experiments 10 and 14. Additional reasons should introduce gradual precision loss rather than pi_3's fixed post-two-cue fatigue or pi_12's conflict-specific collapse; unanimous sequences should show only the small decline observed in Experiment 17. Use a single continuous state-update mechanism, not a trial-level mixture of unrelated strategies. Include stable subject-level parameters for spatial attention, validity gain, leak/overwrite strength, surprise orienting, commitment saturation, load noise, and lapse, with regularized ranges shared across experiments. The regenerated theory should be calibrated jointly against all continuous metrics, especially the positive effects in Experiments 1 and 2 and the near-zero contrasts in Experiments 7 and 16, while treating thresholded Experiments 9 and 19 as weak constraints.


## CANDIDATE THEORY
Spatially Indexed Reliability-Gated Revision. People inspect product features in displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment, with leakage favoring recent evidence and repeated agreement building a persistent saturating stability trace. The trace additionally preserves the average relative reliability quality of the evidence that established it. Thus, equally large traces exert different resistance to reversal: a trace formed by relatively reliable cues gates challengers more strongly, whereas one formed by weak cues releases more readily. Reliability quality modulates the trace-to-gate mapping rather than increasing trace magnitude, so it remains effective after trace saturation without producing rigid coalitions. Opposing evidence continuously erodes the signed trace, and a challenger that reverses the trace installs its own reliability quality. A local reliability drop at an opposing boundary evokes surprise-driven orienting. Accessibility rises smoothly toward the right side of the display, with a modest conditional bonus for an isolated right-edge cue. Increasing display and evidence load gradually reduces response precision. Choices arise from this single continuous commitment process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Spatially Indexed Reliability-Gated Revision expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    spatial_attention = float(parameters["spatial_attention"])
    validity_gain = float(parameters["validity_gain"])
    leak_rate = float(parameters["leak_rate"])
    overwrite_strength = float(parameters["overwrite_strength"])
    surprise_orienting = float(parameters["surprise_orienting"])
    commitment_saturation = float(parameters["commitment_saturation"])
    load_noise = float(parameters["load_noise"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is represented as reliability, but it does not
    # determine scan order. Standardization makes the same gain parameter
    # usable across experiments with different validity ranges.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        standardized_reliability = np.zeros(n_features, dtype=float)
    else:
        standardized_reliability = np.clip(
            (reliability - rel_mean) / rel_sd, -2.5, 2.5
        )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # commitment is the bounded current decision state. stability_trace is a
    # continuous signed record of commitment-congruent support. Unlike a run
    # label, it is not reassigned after the first opposing update: cumulative
    # challenger evidence must gradually erode and reverse it.
    commitment = 0.0
    stability_trace = 0.0
    incumbent_quality = 0.0
    previous_direction = 0.0
    previous_reliability = None
    previous_index = None

    # Compress only the positional component of accessibility on unusually
    # long displays. This prevents normalized scan position from creating an
    # unrealistically large endpoint effect while retaining a rightward bias.
    position_scale = float(min(
        1.0, np.sqrt(8.0 / float(max(8, n_features)))
    ))

    # On unusually long displays, intervening locations increase retention
    # slightly. Ordinary displays retain the original recurrence unchanged.
    long_display_fraction = float(np.clip(
        (float(n_features) - 16.0) / 4.0, 0.0, 1.0
    ))
    effective_leak_rate = float(np.clip(
        leak_rate + 0.10 * long_display_fraction, 0.0, 0.90
    ))

    for j in discriminating:
        j = int(j)
        direction = float(np.sign(diff[j]))
        position = float(j) / float(max(1, n_features - 1))

        # Ordinary accessibility follows a smooth left-to-right gradient.
        # Validity changes accessibility without reordering the cues.
        access_logit = (
            1.15
            + spatial_attention * position_scale * (2.0 * position - 1.0)
            + validity_gain * standardized_reliability[j]
        )

        # A physically isolated discriminating item at the right boundary is
        # modestly foregrounded. The conjunction prevents a generic terminal
        # bonus from guaranteeing reversal on every display.
        if previous_index is None:
            gap_fraction = 0.0
        else:
            gap_fraction = float(max(0, j - previous_index - 1)) / float(
                max(1, n_features - 1)
            )
        right_edge = float(j == n_features - 1)
        isolation_bonus = (
            0.32
            * spatial_attention
            * position_scale
            * right_edge
            * np.sqrt(gap_fraction)
        )
        access_logit += isolation_bonus
        accessibility = sigmoid(access_logit)

        # Reliability also scales encoded revision strength, but only
        # compressively, so weak experts remain psychologically available.
        reliability_strength = float(np.exp(
            validity_gain * 0.55 * standardized_reliability[j]
        ))
        reliability_strength = float(np.clip(reliability_strength, 0.45, 2.20))

        # Surprise is specifically a local reliability drop at an opposing
        # boundary. It boosts attention to a weak challenger rather than
        # installing a separate strategy or reversing the validity ordering.
        orienting = 1.0
        if previous_reliability is not None and previous_direction != direction:
            raw_drop = max(0.0, float(previous_reliability - validities[j]))
            incumbent_alignment = abs(commitment)
            orienting += (
                surprise_orienting
                * raw_drop
                * (0.45 + 0.55 * incumbent_alignment)
            )

        # Leakage creates recency. Every cue remains gated while its direction
        # opposes the persistent stability trace. The gate depends on both the
        # trace mass and the centered reliability quality retained by the
        # incumbent, so saturated traces can still differ in reversal strength.
        retained = float(np.clip(
            effective_leak_rate * commitment, -0.999999, 0.999999
        ))
        opposing_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) != direction
        )
        if opposing_stability:
            trace_gate_mass = min(2.5, abs(stability_trace) ** 0.75)
            quality_gate = float(np.exp(
                0.65 * np.clip(incumbent_quality, -1.0, 1.0)
            ))
            stability_gate = 1.0 + (
                commitment_saturation * trace_gate_mass * quality_gate
            )
        else:
            stability_gate = 1.0

        revision = (
            overwrite_strength
            * accessibility
            * reliability_strength
            * orienting
            / stability_gate
        )

        # Updating in bounded-logit coordinates gives gradual saturation for
        # repeated agreement while preserving smooth directional revision.
        bounded_coordinate = np.arctanh(retained)
        commitment = float(np.tanh(
            bounded_coordinate + direction * revision
        ))

        # The signed stability trace persists across a reversal. Congruent
        # evidence consolidates it efficiently, whereas opposing evidence
        # erodes it continuously and reliability-dependently before eventually
        # installing a new incumbent. No switch threshold or run state is used.
        old_trace = stability_trace
        trace_alignment = direction * float(np.tanh(stability_trace))
        assimilation = (
            0.50 + 0.50 * sigmoid(2.5 * trace_alignment)
        )

        # Only commitment-congruent consolidation receives this bounded
        # nonlinear reliability modulation. Reliable repeated cues strengthen
        # an established trace, whereas weak repetition consolidates less.
        # Opposing cues retain the accepted linear erosion rule.
        congruent_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) == direction
        )
        if congruent_stability:
            consolidation_gain = (
                0.78
                + 0.44 * sigmoid(2.2 * standardized_reliability[j])
            )
        else:
            consolidation_gain = 1.0

        trace_retention = float(np.sqrt(effective_leak_rate))
        stability_trace = (
            trace_retention * stability_trace
            + direction
            * accessibility
            * reliability_strength
            * assimilation
            * consolidation_gain
        )
        stability_trace = float(np.clip(stability_trace, -6.0, 6.0))

        # Preserve reliability quality separately from capped trace magnitude.
        # Congruent support updates a smooth evidence-mass-weighted average;
        # challengers retain the incumbent quality while eroding its trace.
        # Once the trace reverses, the challenger installs its own quality.
        cue_quality = float(np.tanh(
            0.80 * standardized_reliability[j]
        ))
        cue_mass = float(accessibility * reliability_strength)
        if old_trace == 0.0:
            incumbent_quality = cue_quality
        elif np.sign(old_trace) == direction:
            incumbent_quality = float(np.clip(
                (
                    abs(old_trace) * incumbent_quality
                    + cue_mass * cue_quality
                ) / max(abs(old_trace) + cue_mass, 1e-12),
                -1.0,
                1.0
            ))
        elif stability_trace != 0.0 and np.sign(stability_trace) == direction:
            incumbent_quality = cue_quality

        previous_direction = direction
        previous_reliability = float(validities[j])
        previous_index = j

    m = int(discriminating.size)

    # More reasons and more displayed locations introduce gradual precision
    # loss. There is no fixed fatigue boundary and no conflict-only collapse.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = 3.35 / (
        1.0
        + load_noise * display_load
        + 0.55 * load_noise * evidence_load
    )
    precision = float(np.clip(precision, 0.25, 6.0))

    p_a_core = sigmoid(precision * commitment)
    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
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
- spatial_attention: [0.55, 1.15]
- validity_gain: [0.12, 0.38]
- leak_rate: [0.48, 0.67]
- overwrite_strength: [1.75, 2.45]
- surprise_orienting: [1.25, 2.30]
- commitment_saturation: [0.48, 0.88]
- load_noise: [0.020, 0.055]
- lapse_rate: [0.04, 0.13]

`rationale`:
This is a targeted edit to the accepted iteration-8 model. Ordinary accessibility, revision, surprise, leakage, spatial foregrounding, trace accumulation, load noise, response mapping, and every parameter range are unchanged. The edit adds one continuous state variable, incumbent_quality, which stores a bounded reliability-weighted average of the cues supporting the current signed stability trace. That quality directly modulates the existing trace-to-gate mapping. Consequently, reliability information remains operative even when stability_trace magnitude reaches its clip or trace_gate_mass reaches its cap—the implementation saturation diagnosed after iteration 9 no longer discards the desired selectivity. Reliable prefixes can therefore resist late weak evidence more strongly in Experiments 1, 6, and 8, while traces formed from relatively weak cues release more readily in Experiments 4, 10, 11, and 13. During opposition, quality remains attached to the incumbent while its signed trace is continuously eroded; it is replaced only when the challenger actually reverses the trace. The exponential quality factor is centered at one and bounded by the bounded quality state, avoiding a universal increase in incumbent protection or a rigid coalition threshold. The rejected reliability-profile-dispersion scaling and skipped-location decay are not retained, protecting the accepted fits in Experiments 14, 21, and 22.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1951 -> ACCEPTED
- iter 2: loss=0.2022 -> REJECTED
- iter 3: loss=0.2160 -> REJECTED
- iter 4: loss=0.2120 -> REJECTED
- iter 5: loss=0.1880 -> ACCEPTED
- iter 6: loss=0.1724 -> ACCEPTED
- iter 7: loss=0.1832 -> REJECTED
- iter 8: loss=0.1549 -> ACCEPTED
- iter 9: loss=0.2121 -> REJECTED
- iter 10 (current candidate you are grading): loss=0.1637 -> REJECTED
Running-best (last accepted) base: iter 8 at loss=0.1549.

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
  - iter 1: 0.8106 (var=0.0045) (Δ vs real +0.3161)
  - iter 2: 0.7037 (var=0.0071) (Δ vs real +0.2093)
  - iter 3: 0.7974 (var=0.0045) (Δ vs real +0.3029)
  - iter 4: 0.8154 (var=0.0044) (Δ vs real +0.3210)
  - iter 5: 0.8089 (var=0.0051) (Δ vs real +0.3144)
  - iter 6: 0.7960 (var=0.0055) (Δ vs real +0.3015)
  - iter 7: 0.7619 (var=0.0050) (Δ vs real +0.2675)
  - iter 8: 0.7888 (var=0.0067) (Δ vs real +0.2943)
  - iter 9: 0.7936 (var=0.0036) (Δ vs real +0.2992)
  - iter 10 (current): 0.7871 (var=0.0052) (Δ vs real +0.2926)
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
  - iter 1: 0.1508 (var=0.0020) (Δ vs real -0.0429)
  - iter 2: 0.2990 (var=0.0078) (Δ vs real +0.1052)
  - iter 3: 0.1285 (var=0.0017) (Δ vs real -0.0652)
  - iter 4: 0.1581 (var=0.0018) (Δ vs real -0.0356)
  - iter 5: 0.1383 (var=0.0026) (Δ vs real -0.0554)
  - iter 6: 0.1954 (var=0.0033) (Δ vs real +0.0017)
  - iter 7: 0.2079 (var=0.0040) (Δ vs real +0.0142)
  - iter 8: 0.1762 (var=0.0020) (Δ vs real -0.0175)
  - iter 9: 0.1671 (var=0.0025) (Δ vs real -0.0267)
  - iter 10 (current): 0.1821 (var=0.0038) (Δ vs real -0.0117)
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
  - iter 1: 0.7490 (var=0.0049) (Δ vs real +0.1740)
  - iter 2: 0.4958 (var=0.0104) (Δ vs real -0.0792)
  - iter 3: 0.7194 (var=0.0065) (Δ vs real +0.1444)
  - iter 4: 0.7202 (var=0.0071) (Δ vs real +0.1452)
  - iter 5: 0.7638 (var=0.0064) (Δ vs real +0.1888)
  - iter 6: 0.6713 (var=0.0051) (Δ vs real +0.0963)
  - iter 7: 0.6140 (var=0.0087) (Δ vs real +0.0390)
  - iter 8: 0.6650 (var=0.0044) (Δ vs real +0.0900)
  - iter 9: 0.6758 (var=0.0048) (Δ vs real +0.1008)
  - iter 10 (current): 0.6460 (var=0.0058) (Δ vs real +0.0710)
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
  - iter 1: 0.7121 (var=0.0066) (Δ vs real +0.0471)
  - iter 2: 0.3558 (var=0.0186) (Δ vs real -0.3092)
  - iter 3: 0.7250 (var=0.0111) (Δ vs real +0.0600)
  - iter 4: 0.6533 (var=0.0134) (Δ vs real -0.0117)
  - iter 5: 0.7542 (var=0.0066) (Δ vs real +0.0892)
  - iter 6: 0.6892 (var=0.0095) (Δ vs real +0.0242)
  - iter 7: 0.5858 (var=0.0228) (Δ vs real -0.0792)
  - iter 8: 0.6996 (var=0.0077) (Δ vs real +0.0346)
  - iter 9: 0.6813 (var=0.0074) (Δ vs real +0.0162)
  - iter 10 (current): 0.7125 (var=0.0058) (Δ vs real +0.0475)
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
  - iter 1: 0.0033 (var=0.0044) (Δ vs real +0.0000)
  - iter 2: -0.0796 (var=0.0086) (Δ vs real -0.0829)
  - iter 3: -0.0063 (var=0.0047) (Δ vs real -0.0096)
  - iter 4: -0.0104 (var=0.0055) (Δ vs real -0.0137)
  - iter 5: 0.0004 (var=0.0043) (Δ vs real -0.0029)
  - iter 6: -0.0292 (var=0.0052) (Δ vs real -0.0325)
  - iter 7: -0.0642 (var=0.0048) (Δ vs real -0.0675)
  - iter 8: -0.0204 (var=0.0032) (Δ vs real -0.0238)
  - iter 9: -0.0138 (var=0.0061) (Δ vs real -0.0171)
  - iter 10 (current): -0.0142 (var=0.0044) (Δ vs real -0.0175)
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
  - iter 1: 0.1054 (var=0.0016) (Δ vs real -0.2012)
  - iter 2: 0.1588 (var=0.0029) (Δ vs real -0.1479)
  - iter 3: 0.1158 (var=0.0018) (Δ vs real -0.1908)
  - iter 4: 0.1037 (var=0.0025) (Δ vs real -0.2029)
  - iter 5: 0.1104 (var=0.0015) (Δ vs real -0.1962)
  - iter 6: 0.1187 (var=0.0027) (Δ vs real -0.1879)
  - iter 7: 0.1354 (var=0.0035) (Δ vs real -0.1712)
  - iter 8: 0.1346 (var=0.0036) (Δ vs real -0.1721)
  - iter 9: 0.1158 (var=0.0020) (Δ vs real -0.1908)
  - iter 10 (current): 0.1263 (var=0.0025) (Δ vs real -0.1804)
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
  - iter 1: 0.0000 (var=0.1056) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 9: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 10 (current): 0.0000 (var=0.1600) (Δ vs real +0.0000)
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
  - iter 1: 0.8633 (var=0.0018) (Δ vs real +0.6912)
  - iter 2: 0.7444 (var=0.0034) (Δ vs real +0.5723)
  - iter 3: 0.8450 (var=0.0019) (Δ vs real +0.6729)
  - iter 4: 0.8408 (var=0.0023) (Δ vs real +0.6687)
  - iter 5: 0.8454 (var=0.0025) (Δ vs real +0.6733)
  - iter 6: 0.8167 (var=0.0029) (Δ vs real +0.6446)
  - iter 7: 0.7990 (var=0.0023) (Δ vs real +0.6269)
  - iter 8: 0.8129 (var=0.0016) (Δ vs real +0.6408)
  - iter 9: 0.8044 (var=0.0031) (Δ vs real +0.6323)
  - iter 10 (current): 0.7921 (var=0.0026) (Δ vs real +0.6200)
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
  - iter 1: 0.0892 (var=0.0075) (Δ vs real +0.1217)
  - iter 2: 0.3525 (var=0.0263) (Δ vs real +0.3850)
  - iter 3: 0.0862 (var=0.0095) (Δ vs real +0.1187)
  - iter 4: 0.1483 (var=0.0107) (Δ vs real +0.1808)
  - iter 5: 0.0867 (var=0.0071) (Δ vs real +0.1192)
  - iter 6: 0.1696 (var=0.0090) (Δ vs real +0.2021)
  - iter 7: 0.2017 (var=0.0101) (Δ vs real +0.2342)
  - iter 8: 0.1492 (var=0.0067) (Δ vs real +0.1817)
  - iter 9: 0.1433 (var=0.0077) (Δ vs real +0.1758)
  - iter 10 (current): 0.1437 (var=0.0089) (Δ vs real +0.1762)
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
  - iter 1: 0.9164 (var=0.0008) (Δ vs real +0.0417)
  - iter 2: 0.8843 (var=0.0014) (Δ vs real +0.0095)
  - iter 3: 0.9045 (var=0.0015) (Δ vs real +0.0298)
  - iter 4: 0.9093 (var=0.0010) (Δ vs real +0.0345)
  - iter 5: 0.9070 (var=0.0015) (Δ vs real +0.0322)
  - iter 6: 0.8994 (var=0.0009) (Δ vs real +0.0247)
  - iter 7: 0.9083 (var=0.0010) (Δ vs real +0.0335)
  - iter 8: 0.9037 (var=0.0012) (Δ vs real +0.0289)
  - iter 9: 0.9054 (var=0.0011) (Δ vs real +0.0306)
  - iter 10 (current): 0.8996 (var=0.0011) (Δ vs real +0.0249)
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
  - iter 1: 0.8308 (var=0.0059) (Δ vs real -0.0117)
  - iter 2: 0.5529 (var=0.0144) (Δ vs real -0.2896)
  - iter 3: 0.7779 (var=0.0067) (Δ vs real -0.0646)
  - iter 4: 0.7746 (var=0.0074) (Δ vs real -0.0679)
  - iter 5: 0.7858 (var=0.0086) (Δ vs real -0.0567)
  - iter 6: 0.6725 (var=0.0105) (Δ vs real -0.1700)
  - iter 7: 0.6933 (var=0.0120) (Δ vs real -0.1492)
  - iter 8: 0.6683 (var=0.0130) (Δ vs real -0.1742)
  - iter 9: 0.7296 (var=0.0122) (Δ vs real -0.1129)
  - iter 10 (current): 0.6021 (var=0.0118) (Δ vs real -0.2404)
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
  - iter 1: 0.1419 (var=0.0047) (Δ vs real -0.0294)
  - iter 2: 0.1919 (var=0.0070) (Δ vs real +0.0206)
  - iter 3: 0.1288 (var=0.0032) (Δ vs real -0.0425)
  - iter 4: 0.1419 (var=0.0045) (Δ vs real -0.0294)
  - iter 5: 0.1556 (var=0.0037) (Δ vs real -0.0156)
  - iter 6: 0.1688 (var=0.0059) (Δ vs real -0.0025)
  - iter 7: 0.1800 (var=0.0069) (Δ vs real +0.0087)
  - iter 8: 0.1881 (var=0.0060) (Δ vs real +0.0169)
  - iter 9: 0.1806 (var=0.0049) (Δ vs real +0.0094)
  - iter 10 (current): 0.2487 (var=0.0100) (Δ vs real +0.0775)
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
  - iter 1: 0.8396 (var=0.0025) (Δ vs real -0.0414)
  - iter 2: 0.7048 (var=0.0032) (Δ vs real -0.1761)
  - iter 3: 0.8489 (var=0.0024) (Δ vs real -0.0321)
  - iter 4: 0.8082 (var=0.0030) (Δ vs real -0.0727)
  - iter 5: 0.8218 (var=0.0025) (Δ vs real -0.0592)
  - iter 6: 0.7858 (var=0.0030) (Δ vs real -0.0952)
  - iter 7: 0.7417 (var=0.0042) (Δ vs real -0.1393)
  - iter 8: 0.7905 (var=0.0023) (Δ vs real -0.0905)
  - iter 9: 0.7939 (var=0.0028) (Δ vs real -0.0870)
  - iter 10 (current): 0.7877 (var=0.0022) (Δ vs real -0.0933)
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
  - iter 1: 0.0796 (var=0.0108) (Δ vs real +0.0604)
  - iter 2: 0.0458 (var=0.0081) (Δ vs real +0.0267)
  - iter 3: 0.0871 (var=0.0094) (Δ vs real +0.0679)
  - iter 4: 0.0908 (var=0.0093) (Δ vs real +0.0717)
  - iter 5: 0.0533 (var=0.0097) (Δ vs real +0.0342)
  - iter 6: 0.0454 (var=0.0090) (Δ vs real +0.0263)
  - iter 7: 0.0608 (var=0.0080) (Δ vs real +0.0417)
  - iter 8: 0.0517 (var=0.0131) (Δ vs real +0.0325)
  - iter 9: 0.1258 (var=0.0072) (Δ vs real +0.1067)
  - iter 10 (current): 0.0296 (var=0.0089) (Δ vs real +0.0104)
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
  - iter 1: -0.0052 (var=0.0051) (Δ vs real -0.0523)
  - iter 2: 0.0292 (var=0.0089) (Δ vs real -0.0179)
  - iter 3: 0.0158 (var=0.0045) (Δ vs real -0.0312)
  - iter 4: -0.0054 (var=0.0033) (Δ vs real -0.0525)
  - iter 5: -0.0031 (var=0.0044) (Δ vs real -0.0502)
  - iter 6: 0.0135 (var=0.0090) (Δ vs real -0.0335)
  - iter 7: 0.0100 (var=0.0065) (Δ vs real -0.0371)
  - iter 8: 0.0406 (var=0.0080) (Δ vs real -0.0065)
  - iter 9: 0.0283 (var=0.0052) (Δ vs real -0.0187)
  - iter 10 (current): 0.0408 (var=0.0072) (Δ vs real -0.0062)
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
  - iter 1: 0.0155 (var=0.0112) (Δ vs real -0.0156)
  - iter 2: -0.0113 (var=0.0118) (Δ vs real -0.0424)
  - iter 3: -0.0177 (var=0.0057) (Δ vs real -0.0488)
  - iter 4: 0.0153 (var=0.0119) (Δ vs real -0.0158)
  - iter 5: 0.0026 (var=0.0137) (Δ vs real -0.0285)
  - iter 6: -0.0153 (var=0.0109) (Δ vs real -0.0464)
  - iter 7: -0.0083 (var=0.0095) (Δ vs real -0.0394)
  - iter 8: 0.0388 (var=0.0120) (Δ vs real +0.0077)
  - iter 9: -0.0150 (var=0.0081) (Δ vs real -0.0461)
  - iter 10 (current): 0.0315 (var=0.0159) (Δ vs real +0.0004)
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
  - iter 1: 0.0000 (var=0.0006) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0067) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0016) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0045) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0022) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0038) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0013) (Δ vs real +0.0000)
  - iter 8: 0.0000 (var=0.0003) (Δ vs real +0.0000)
  - iter 9: 0.0000 (var=0.0031) (Δ vs real +0.0000)
  - iter 10 (current): 0.0000 (var=0.0006) (Δ vs real +0.0000)
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
  - iter 1: 0.0344 (var=0.0063) (Δ vs real +0.0081)
  - iter 2: 0.0137 (var=0.0170) (Δ vs real -0.0125)
  - iter 3: -0.0088 (var=0.0136) (Δ vs real -0.0350)
  - iter 4: 0.0150 (var=0.0129) (Δ vs real -0.0113)
  - iter 5: 0.0350 (var=0.0120) (Δ vs real +0.0087)
  - iter 6: 0.0069 (var=0.0088) (Δ vs real -0.0194)
  - iter 7: 0.0119 (var=0.0139) (Δ vs real -0.0144)
  - iter 8: 0.0394 (var=0.0124) (Δ vs real +0.0131)
  - iter 9: -0.0162 (var=0.0118) (Δ vs real -0.0425)
  - iter 10 (current): -0.0094 (var=0.0099) (Δ vs real -0.0356)
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
  - iter 1: 0.8972 (var=0.0023) (Δ vs real +0.3994)
  - iter 2: 0.7944 (var=0.0055) (Δ vs real +0.2967)
  - iter 3: 0.8956 (var=0.0028) (Δ vs real +0.3978)
  - iter 4: 0.9000 (var=0.0027) (Δ vs real +0.4022)
  - iter 5: 0.8761 (var=0.0031) (Δ vs real +0.3783)
  - iter 6: 0.8350 (var=0.0049) (Δ vs real +0.3372)
  - iter 7: 0.8478 (var=0.0033) (Δ vs real +0.3500)
  - iter 8: 0.8689 (var=0.0052) (Δ vs real +0.3711)
  - iter 9: 0.8772 (var=0.0020) (Δ vs real +0.3794)
  - iter 10 (current): 0.8572 (var=0.0033) (Δ vs real +0.3594)
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
  - iter 1: -0.1062 (var=0.0087) (Δ vs real +0.0750)
  - iter 2: -0.1188 (var=0.0211) (Δ vs real +0.0625)
  - iter 3: -0.1106 (var=0.0157) (Δ vs real +0.0706)
  - iter 4: -0.1050 (var=0.0147) (Δ vs real +0.0763)
  - iter 5: -0.1156 (var=0.0141) (Δ vs real +0.0656)
  - iter 6: -0.1269 (var=0.0134) (Δ vs real +0.0544)
  - iter 7: -0.1175 (var=0.0139) (Δ vs real +0.0638)
  - iter 8: -0.1625 (var=0.0173) (Δ vs real +0.0188)
  - iter 9: -0.2081 (var=0.0123) (Δ vs real -0.0269)
  - iter 10 (current): -0.1088 (var=0.0157) (Δ vs real +0.0725)
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
  - iter 1: 0.6633 (var=0.0134) (Δ vs real +0.2733)
  - iter 2: 0.3242 (var=0.0320) (Δ vs real -0.0658)
  - iter 3: 0.6908 (var=0.0119) (Δ vs real +0.3008)
  - iter 4: 0.6458 (var=0.0186) (Δ vs real +0.2558)
  - iter 5: 0.6367 (var=0.0113) (Δ vs real +0.2467)
  - iter 6: 0.4808 (var=0.0272) (Δ vs real +0.0908)
  - iter 7: 0.4958 (var=0.0299) (Δ vs real +0.1058)
  - iter 8: 0.4775 (var=0.0199) (Δ vs real +0.0875)
  - iter 9: 0.6242 (var=0.0220) (Δ vs real +0.2342)
  - iter 10 (current): 0.5358 (var=0.0163) (Δ vs real +0.1458)
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
  - iter 1: 0.1512 (var=0.0075) (Δ vs real -0.0238)
  - iter 2: 0.1542 (var=0.0127) (Δ vs real -0.0208)
  - iter 3: 0.1062 (var=0.0073) (Δ vs real -0.0688)
  - iter 4: 0.0975 (var=0.0054) (Δ vs real -0.0775)
  - iter 5: 0.1525 (var=0.0081) (Δ vs real -0.0225)
  - iter 6: 0.1921 (var=0.0096) (Δ vs real +0.0171)
  - iter 7: 0.1546 (var=0.0060) (Δ vs real -0.0204)
  - iter 8: 0.1796 (var=0.0108) (Δ vs real +0.0046)
  - iter 9: 0.0917 (var=0.0070) (Δ vs real -0.0833)
  - iter 10 (current): 0.1500 (var=0.0110) (Δ vs real -0.0250)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed continuous spatial revision family and successfully captures several important signatures: the null contrasts in Experiments 5, 7, 16, and 17; terminal influence in Experiments 10, 11, and 13; the recency-majority result in Experiment 12; and the reliability-drop effect in Experiment 22. However, it is not yet experiment-invariant. Its dominant error is excessive, overly general terminal overwrite. Terminal-cue agreement is far too high in Experiment 8 (0.863 versus 0.172), Experiment 19 (0.897 versus 0.498), and Experiment 3 (0.749 versus 0.575), while early-run choice is consequently too low in Experiment 6 (0.105 versus 0.307). Experiment 1's tally-support contrast is also inflated (0.811 versus 0.494). Separately, spatial foregrounding is too strong: Experiment 21's endpoint effect is 0.663 versus 0.390, and Experiment 14's foregrounding contrast is 0.080 versus 0.019. The model also lacks the small unanimous-load decline in Experiment 15 (-0.005 versus 0.047). Many simulated between-subject variances are substantially below the observed variances, suggesting overly narrow effective heterogeneity.
Rationale: Retain the Spatially Indexed Reliability-Gated Revision mechanism, but recalibrate three in-family components. First, reduce generic overwrite by moderately increasing commitment retention and/or strengthening continuous coherent-support gating, particularly under long displays and established same-direction prefixes. The adjustment must be graded rather than categorical so that the strong terminal effects in Experiments 10, 11, and 13 remain. Most importantly, the surprise multiplier is currently applied to every opposing reliability drop, including adjacent block boundaries; make surprise orienting conditional on challenger isolation or a meaningful spatial gap. That should sharply reduce the spurious terminal dominance in Experiment 8 while preserving the separated-singleton effects in Experiments 20 and 22. Second, lower the spatial-attention range and especially the right-edge isolation coefficient to bring Experiments 21 and 14 down without eliminating their positive direction. Third, strengthen gradual load-dependent precision loss enough to yield Experiment 15's small positive decline, avoiding any fixed fatigue threshold. Jointly tune these knobs rather than merely raising lapse, since a global lapse would damage the already good directional contrasts. After correcting mean fit, modestly widen subject-level parameter ranges to better reproduce observed heterogeneity.

**Outcome of your advice:** iter 1 candidate loss=0.1951 -> iter 2 candidate loss=0.2022 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family, but it is worse than the accepted base and therefore should not be retained. The previous recommendation to broadly reduce overwrite and spatial attention was over-applied. It improved Experiment 1, the foregrounding effects in Experiments 14 and 21, the load decline in Experiment 15, and somewhat reduced the excessive terminal choice in Experiments 8 and 19. However, it simultaneously destroyed several previously good terminal-revision results: Experiment 4 fell from 0.712 to 0.356 versus 0.665 observed, Experiment 11 from 0.831 to 0.553 versus 0.843, and Experiment 13 from 0.840 to 0.705 versus 0.881. Experiment 2 also crossed from a small underestimate to a substantial overestimate, and Experiment 9 deteriorated sharply. The central failure nevertheless remains Experiment 8, where terminal agreement is still 0.744 rather than 0.172; Experiment 19 also remains too terminal-driven at 0.794 rather than 0.498, while early-run choice in Experiment 6 remains too low. Thus the model still cannot selectively protect an incumbent in the configurations that require resistance without suppressing revision in Experiments 4, 10, 11, and 13. A likely source is the within-experiment z-standardization of reliability: it discards absolute reliability level and profile information, making superficially similar spatial sequences across experiments receive similar update strengths even when their communicated validities differ materially.
Rationale: Because this iteration was rejected by the gate, build from the unchanged accepted iteration-1 base rather than repeating the rejected broad increase in retention, decrease in overwrite, and widening of heterogeneity. Use a different in-family knob: revise reliability calibration. Replace per-experiment z-scoring of log-odds reliability with a fixed, cross-experiment normalization or a compressive transform anchored at validity 0.5. Reliability should retain absolute magnitude and the absolute drop between incumbent and challenger, while remaining bounded. Then let the existing continuous coherence gate depend on accumulated reliability-weighted support, not merely accessibility summed across cues. This should permit strong reliable prefixes to resist weak terminal challengers in Experiment 8 and reduce the inflated terminal conformity in Experiments 6 and 19, without globally weakening terminal revision in Experiments 4, 11, and 13. Keep surprise orienting continuous and separation-sensitive, but do not remove adjacent revision wholesale; Experiment 2 and the separated-drop effect in Experiment 22 require reliability-drop sensitivity. Preserve the accepted base's generic overwrite, leak, spatial attention, and load settings initially, and tune the new fixed reliability scale jointly against Experiments 2, 4, 6, 8, 11, 13, 20, and 22. Treat Experiments 9 and 19 as weaker constraints as instructed. Do not further widen parameter ranges until the pooled means improve, since the rejected iteration's increased heterogeneity did not compensate for its mean-fit regressions.

**Outcome of your advice:** iter 2 candidate loss=0.2022 -> iter 3 candidate loss=0.2160 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family, but its aggregate loss is worse than both the accepted base and the previous rejected attempt. The fixed absolute-reliability calibration did not solve the key selectivity problem. Relative to the accepted base, Experiment 8 barely improved (terminal agreement 0.845 versus 0.863, still far above 0.172), Experiment 19 was effectively unchanged (0.896 versus 0.897, observed 0.498), and Experiment 6 remained much too recency-driven (early choice 0.116 versus 0.307). It also worsened Experiment 2, Experiment 12, Experiment 16, Experiment 18, and the reliability-drop effect in Experiment 22. The persistent signature is excessive revision by contiguous late challengers: this inflates terminal influence in Experiments 3, 8, and 19, suppresses early-run choice in Experiment 6, and contributes to the excessive tally contrast in Experiment 1. Spatial sensitivity is also still too large in Experiments 14 and 21. At the same time, generic terminal revision is useful and should be preserved: Experiments 4, 10, 11, and 13 remain reasonably close, and the model correctly captures the null/near-null patterns in Experiments 5, 7, and 17. Simulated variances are often below observed variances, but mean-pattern errors remain the priority.
Rationale: The current candidate was rejected, and the prior fixed-reliability recommendation was also rejected, so do not continue tuning the absolute reliability anchor or reliability-weighted coherent-support accumulation. Make a narrower in-family intervention on the unchanged accepted base. Specifically, change the surprise term rather than globally changing leak or overwrite: a reliability drop should receive substantial orienting amplification only when the opposing challenger is spatially separated from the incumbent run, isolated at the right edge, or otherwise has a meaningful gap from the preceding discriminating cue. Adjacent opposing cues should retain ordinary continuous revision but receive little or no surprise multiplier. This directly distinguishes the contiguous reversals driving excessive terminal choice in Experiments 6, 8, and 19 from the separated challengers central to Experiments 2, 11, 20, and 22, while leaving generic recency available for the adjacent terminal effects in Experiments 4, 10, and 13. Implement the modulation smoothly, for example by multiplying surprise by a bounded function of physical gap plus a modest right-edge-isolation term, rather than imposing a categorical strategy or threshold. Separately, test a small attenuation or saturation of the smooth position-to-accessibility gradient while keeping leak and overwrite at the accepted settings; this targets the oversized endpoint and foregrounding effects in Experiments 21 and 14 without repeating the rejected broad suppression of revision. Jointly calibrate these two surgical changes against Experiments 1, 2, 4, 6, 8, 11, 13, 14, 21, and 22. Defer widening subject-level ranges until pooled means improve.

**Outcome of your advice:** iter 3 candidate loss=0.2160 -> iter 4 candidate loss=0.2120 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed Spatially Indexed Reliability-Gated Revision family, but it was rejected by the accept gate and remains substantially worse than the accepted iteration-1 base. The latest separation-gated surprise intervention did not address the dominant errors: terminal agreement remains far too high in Experiment 8 (0.841 versus 0.172) and Experiment 19 (0.900 versus 0.498), early-run choice remains far too low in Experiment 6 (0.104 versus 0.307), and the tally-support contrast remains inflated in Experiment 1 (0.815 versus 0.494). Endpoint sensitivity also remains excessive in Experiment 21 (0.646 versus 0.390), while Experiment 14's foregrounding contrast moved farther from zero. Moreover, attenuating separated surprise harmed results that the accepted base handled better: Experiment 11 fell from 0.831 to 0.775, Experiment 13 from 0.840 to 0.808, and Experiment 22 from 0.151 to 0.098. The candidate still succeeds on Experiments 4, 5, 7, 10, 12, 16, 17, and 18, but that is insufficiently experiment-invariant. The central implementation problem is now clearer: surprise is not the main source of excessive recency. After the first opposing cue, coherent_support is rapidly reassigned to the challenger, and subsequent same-direction challenger cues are classified as non-opposing and therefore bypass the stability gate. A contiguous late run can consequently erase a coherent incumbent through ordinary updates even when surprise is suppressed. Simulated between-subject variances also remain generally too small, but the large pooled-mean errors should be fixed first.
Rationale: Build from the unchanged accepted iteration-1 base. The previous pushes toward broad retention/overwrite reduction, fixed absolute-reliability normalization, and spatially gated surprise were all rejected, so do not repeat them. Instead, make a distinct in-family correction to the recurrence: maintain a persistent, continuous incumbent-stability trace that is not immediately relabeled or reset after the first reversal. Opposing evidence should erode this trace gradually according to cumulative challenger strength, reliability, leak, and load; every cue in an opposing run should remain partially gated until the challenger has continuously displaced the incumbent. This can be implemented with two smooth state variables—commitment and a decaying commitment-congruent support or stability trace—without introducing runs, thresholds, coalitions, or strategy mixtures. Preserve ordinary rapid revision when the incumbent is weak, thereby protecting Experiments 4, 10, 11, and 13, but let established coherent support resist multi-cue late runs enough to improve Experiments 6 and 8. Use the accepted relative reliability transform initially rather than reopening the rejected absolute-normalization change, and tune how reliability scales erosion of the persistent trace. Experiment 19 is a weak thresholded constraint, but its large error should still improve as a consequence rather than driving calibration. For Experiment 21, avoid another uniform reduction of spatial attention. Instead, compress the effective recency generated by scan position on very long displays—for example, make leak over successive discriminating cues mildly load-dependent or saturate positional accessibility with display length—while retaining a modest explicit right-edge accessibility bonus required by Experiment 1. Do not further weaken surprise: the latest change degraded Experiments 11, 13, and 22 without solving Experiment 8. After improving pooled means, subject-level parameter ranges can be widened modestly to address the systematically low variance.

**Outcome of your advice:** iter 4 candidate loss=0.2120 -> iter 5 candidate loss=0.1880 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The accepted persistent-stability revision is a genuine improvement within the prescribed mechanism family: aggregate loss fell from 0.1951 to 0.1880, while the model retains excellent fits for the null contrasts in Experiments 5, 7, and 17, the near-null history result in Experiment 16, the boundary pattern in Experiment 18, the recency-majority result in Experiment 12, and the reliability-drop effect in Experiment 22. Experiment 14 and, more modestly, Experiment 21 also improved in the intended direction. Nevertheless, the model is not yet experiment-invariant. Its dominant empirical failure remains excessive terminal/late-cue control: Experiment 8 is 0.845 versus 0.172, Experiment 3 is 0.764 versus 0.575, and the weakly weighted Experiment 19 is 0.876 versus 0.498. Correspondingly, early-run choice in Experiment 6 is only 0.110 versus 0.307, and Experiment 1's tally-support contrast remains 0.809 versus 0.494. The new trace therefore improved aggregate loss without fixing the central recurrence error it was designed to address; Experiment 8 is essentially unchanged and Experiment 3 worsened. Spatial order effects also remain excessive in Experiment 21 (0.637 versus 0.390), although less so than before. The model additionally misses the small unanimous-load decline in Experiment 15 (-0.003 versus 0.047). Fits in Experiments 10, 12, 18, 20, and 22 show that reliability-gated revision itself is viable, but the simulated behavior is generally too extreme and too homogeneous: many between-subject variances are far below the observed values.
Rationale: Retain the accepted signed persistent-stability trace—the previous advice introducing it was accepted—but correct how it gates reversal. In the current code, the opposing gate is proportional to abs(retained). Precisely when challenger evidence drives commitment toward zero, this factor collapses and later challenger cues bypass the still-persistent incumbent trace. Thus the implementation does not fully realize the stated idea that every cue remains gated until cumulative displacement. Make the opposing gate depend smoothly on incumbent trace magnitude and accumulated challenger erosion, not on the instantaneous magnitude of commitment; allow it to weaken only as the signed trace itself is eroded. Calibrate this as a modest nonlinear or saturating change rather than globally raising leak or lowering overwrite, since those broad changes were previously rejected and harmed Experiments 4, 11, and 13. This targeted correction should protect coherent prefixes in Experiments 1, 6, and especially 8 while preserving rapid revision from weak incumbents in Experiments 4, 10, 11, and 13. Separately, extend the accepted display-length compression with a small long-display-specific reduction in effective recency—such as smoothly moving leak toward greater retention as total display length becomes unusually large—rather than another global reduction in spatial attention. This targets Experiments 14 and 21 without altering ordinary six- to ten-feature displays. Finally, modestly strengthen gradual evidence-load precision loss so unanimous sequences produce the small positive decline in Experiment 15. Treat Experiments 9 and 19 as weak calibration constraints, and defer broad range widening until these pooled means improve; afterward, slightly wider subject-level saturation, load-noise, and lapse ranges may address the systematically underestimated heterogeneity.

**Outcome of your advice:** iter 5 candidate loss=0.1880 -> iter 6 candidate loss=0.1724 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate is a meaningful accepted improvement and remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family. The revised trace gate improved aggregate loss from 0.1880 to 0.1724 and substantially improved Experiment 2 (0.195 versus 0.194), Experiment 3, Experiment 12, Experiment 14, Experiment 20, Experiment 21, and Experiment 22. It also preserves the null results in Experiments 7 and 17 and reasonable fits in Experiments 4, 5, 10, and 18. Thus the prior advice to gate reversal by the persistent trace rather than instantaneous commitment was productive and should be retained. However, the theory is still not experiment-invariant. It remains dramatically too terminal-driven in Experiment 8 (0.817 versus 0.172) and Experiment 19 (0.835 versus 0.498), with corresponding underprediction of early-run choice in Experiment 6 (0.119 versus 0.307). Experiment 1's contrast is still much too large (0.796 versus 0.494). At the same time, the stronger persistent gate now suppresses terminal revision too much in Experiment 11 (0.673 versus 0.843) and Experiment 13 (0.786 versus 0.881). This combination shows that another global increase in incumbent protection would be counterproductive: the missing calibration is selective release of the stability gate according to challenger reliability, coherence, and spatial isolation. The small load decline also remains underestimated in Experiment 15 (0.014 versus 0.047), and simulated between-subject variances are commonly lower than observed.
Rationale: Keep the accepted persistent signed stability trace and its independence from instantaneous commitment; that change was validated by the accept gate. Do not simply strengthen the gate further, because the present change already moved Experiments 11 and 13 in the wrong direction while only modestly improving Experiments 6 and 8. Instead, refine how the existing gate is eroded or released. A spatially isolated challenger, such as the separated terminal cue in Experiment 11, should erode the incumbent trace more effectively or receive a modest reduction of the trace gate. This should be implemented as a smooth gap-dependent modulation of the same recurrence, not as another strategy and not by broadly increasing surprise. For contiguous challengers, make erosion more sensitive to their reliability-weighted cumulative strength: weak contiguous late evidence should remain gated in Experiment 8, whereas sufficiently reliable or coherent challenger runs should progressively unlock revision in Experiment 13. Preserve the current relative reliability representation, since the previous switch to absolute normalization was rejected, but consider a modestly wider validity-gain range or a bounded nonlinear reliability interaction specifically in trace erosion rather than globally scaling every update. Jointly calibrate this selectivity against Experiments 6, 8, 11, and 13, while checking that the excellent Experiment 2 and Experiment 22 reliability-drop effects remain intact. The accepted long-display retention adjustment sharply improved Experiment 21, so a slightly stronger version is warranted, but keep it confined to unusually long displays to avoid disturbing ordinary conditions; Experiment 14 is already close. Continue the accepted direction of gradually increasing load-dependent precision loss by a small amount to approach Experiment 15, rather than introducing a fatigue threshold. Only after these pooled means improve should subject-level saturation, load-noise, and lapse ranges be widened modestly to address underestimated heterogeneity.

**Outcome of your advice:** iter 6 candidate loss=0.1724 -> iter 7 candidate loss=0.1832 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family, but it was rejected by the accept gate: loss increased from the accepted iteration-6 value of 0.1724 to 0.1832. The proposed reliability- and gap-dependent gate release produced limited gains in Experiments 3, 6, and 8, but these were outweighed by regressions in Experiments 4, 5, 9, 13, 14, 21, and 22. In particular, Experiment 13 terminal agreement fell from 0.786 to 0.742 against 0.881 observed, Experiment 4 fell from 0.689 to 0.586 against 0.665, and the previously accurate reliability-drop effect in Experiment 22 moved from 0.192 to 0.155 against 0.175. The central pattern also remains unresolved: terminal control is still far too strong in Experiment 8 (0.799 versus 0.172) and Experiment 19 (0.848 versus 0.498), early-run choice remains too low in Experiment 6 (0.135 versus 0.307), and Experiment 1 remains overly contrastive (0.762 versus 0.494). Conversely, terminal revision is now too weak in Experiments 11 and 13. Thus the nonlinear challenger-release modification did not provide the required selectivity. The model continues to capture Experiments 2, 7, 10, 12, 17, 18, and 22 reasonably well, but the large cross-experiment reversal errors preclude an experiment-invariant account. Simulated variances are also generally too low, although mean fit remains the primary issue.
Rationale: Build from the unchanged accepted iteration-6 base. The previous recommendation to add nonlinear reliability release, gap release, stronger long-display retention, and higher load noise was rejected, so do not repeat that bundled adjustment. Instead, modify a different part of the same recurrence: make formation of the persistent stability trace more selectively reliability-sensitive, while leaving ordinary cue revision and gate release near iteration-6 settings. Specifically, use a bounded nonlinear reliability interaction only when consolidating commitment-congruent support, so a coherent sequence of relatively reliable incumbent cues forms a stronger trace than an equally long weak sequence. Opposing evidence should continue to erode that trace smoothly; do not immediately amplify each challenger through a reliability-squared gate-release term. This targets the excessive reversal of reliable prefixes in Experiments 1, 6, and 8 without globally suppressing terminal revision from weak incumbents in Experiments 4, 10, 11, and 13. If spatial isolation is retained, use only a small gap-dependent erosion increment rather than the rejected multiplicative release, and calibrate it primarily against Experiment 11 while protecting Experiments 14, 21, and 22. Because iteration 7 bundled several changes, keep long-display retention and load-noise at the accepted iteration-6 values initially; Experiment 21 and Experiment 15 both worsened despite those increases. Jointly test the revised trace-consolidation rule against Experiments 1, 4, 6, 8, 11, and 13, with Experiments 2 and 22 as safeguards. Parameter-range widening should remain secondary until pooled means improve.

**Outcome of your advice:** iter 7 candidate loss=0.1832 -> iter 8 candidate loss=0.1549 -> the gate marked it ACCEPTED.

### Iteration 8
Verdict: regenerate
Interpretation: Iteration 8 is the best candidate so far and is faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family. The accepted reliability-sensitive trace-consolidation edit reduced aggregate loss from 0.1724 to 0.1549, so that direction was productive. The candidate now fits Experiments 2, 4, 5, 7, 10, 12, 15, 16, 17, 18, 20, and 22 reasonably well, including the surprise effect in Experiment 22, gradual unanimous-load decline in Experiment 15, and near-zero contrasts in Experiments 7 and 16. Nevertheless, it is not sufficiently experiment-invariant. The dominant failure remains extreme late-cue control in Experiment 8: terminal agreement is 0.813 versus 0.172 observed. Related errors remain in Experiment 6, where early-run choice is only 0.135 versus 0.307, Experiment 1, where the tally-support contrast is 0.789 versus 0.494, and the weakly weighted Experiment 19, where terminal conformity is 0.869 versus 0.498. Experiment 3 is also too terminal-driven at 0.665 versus 0.575. Conversely, terminal revision is insufficient in the separated-singleton configuration of Experiment 11 (0.668 versus 0.843) and somewhat insufficient in Experiment 13 (0.791 versus 0.881). Thus generic recency is still too strong, while the model cannot selectively release a stabilized incumbent for configurations that support terminal revision. The current consolidation multiplier only spans roughly 0.78 to 1.22 and depends on standardized reliability, so it largely discards how steep or flat the actual communicated reliability profile is. Between-subject variability also remains too low in several major failures, especially Experiments 1 and 8, but the pooled means remain the primary problem.
Rationale: Retain the accepted persistent trace and reliability-sensitive congruent consolidation; the latest advice demonstrably improved the running best. Do not revive the rejected global reductions in overwrite, global increases in retention, absolute-reliability replacement, or multiplicative challenger gate release. Instead, extend the accepted consolidation mechanism using a bounded hybrid reliability calibration: preserve standardized reliability for ordinary cue accessibility and revision, but scale the nonlinear trace-consolidation contrast by the within-display dispersion or range of log-odds reliability. A steep reliability profile should permit substantially stronger consolidation by the reliable prefix, whereas a flat profile should keep consolidation near neutral. This targets Experiments 1, 6, and especially 8 without globally protecting every coherent prefix and thereby further damaging Experiment 13. The present 0.78–1.22 multiplier appears too weak; widen it cautiously and regularize it through reliability-profile dispersion rather than simply raising commitment_saturation for all displays. Separately, address Experiment 11 through the recurrence's treatment of physical gaps. The code currently jumps directly between discriminating cues, so intervening nondiscriminating locations do not decay the incumbent trace or commitment. Introduce modest elapsed-position leakage across skipped locations before processing the next discriminating cue. That provides a smooth, mechanistically faithful release for an isolated terminal challenger without reintroducing the rejected explicit multiplicative gap-release rule. Compress this gap decay on very long displays and calibrate it against Experiments 14 and 21, where spatial effects are already somewhat too large. Jointly safeguard Experiments 2, 4, 10, 13, 20, and 22. Only after improving these pooled means should subject-level ranges be widened modestly to address the underestimated variance.

**Outcome of your advice:** iter 8 candidate loss=0.1549 -> iter 9 candidate loss=0.2121 -> the gate marked it REJECTED.

### Iteration 9 (most recent)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed Spatially Indexed Reliability-Gated Revision family, but it was decisively rejected: loss rose from the accepted iteration-8 value of 0.1549 to 0.2121. The two latest edits should not be retained. Reliability-profile dispersion produced essentially no improvement in the central failures: Experiment 8 remains extremely terminal-driven at 0.804 versus 0.172, Experiment 6 still underpredicts early-run choice at 0.116 versus 0.307, Experiment 1 remains too contrastive at 0.794 versus 0.494, and Experiment 19 remains too terminal-driven at 0.877 versus 0.498. Although elapsed-position decay improved Experiment 11 from 0.668 to 0.730, it caused much larger regressions in spatially separated displays: Experiment 14 increased from 0.052 to 0.126 versus 0.019, Experiment 21 increased from 0.478 to 0.624 versus 0.390, and the reliability-drop effect in Experiment 22 fell from 0.180 to 0.092 versus 0.175. Experiments 16 and 18 also deteriorated. The likely reason the hybrid consolidation change was ineffective is implementation saturation: stability_trace is clipped, and trace_gate_mass is capped at 2.5. Once a coherent prefix reaches that cap, stronger reliability-dependent consolidation cannot create stronger incumbent protection, so the profile-dispersion multiplier is largely discarded precisely on the long coherent sequences it was intended to distinguish. The scalar trace also loses incumbent reliability quality after accumulation. The candidate still fits Experiments 2, 4, 5, 7, 10, 12, 15, 17, and 20 reasonably well, but the large and systematic cross-experiment errors preclude acceptance.
Rationale: Build from the unchanged accepted iteration-8 base. The latest recommendation to add profile-dispersion scaling and skipped-location decay was rejected, so do not repeat either change or merely retune their coefficients. Retain the accepted persistent trace and reliability-sensitive congruent consolidation, but make reliability quality survive into the opposing gate rather than trying to increase the already capped trace magnitude. A suitable in-family adjustment is a bounded, centered incumbent-quality modulation of gate sensitivity: maintain a smooth reliability-weighted quality summary for commitment-congruent support and let high-quality traces exert a stronger gate while low-quality traces exert a weaker one. Alternatively, apply the existing consolidation factor directly to the trace-to-gate mapping before its saturation. This should distinguish reliable prefixes in Experiments 1, 6, and 8 without globally strengthening every coherent incumbent or repeating the rejected reliability-dependent challenger-release intervention. Recalibrate or soften the current trace-mass cap so this quality modulation remains operative, but preserve an upper bound to avoid rigid pi_12-like coalitions. For Experiment 11, do not decay commitment and trace across every skipped location. If additional isolated-terminal accessibility is needed, adjust the existing right-edge isolation accessibility term with strong attenuation on long displays, so it can modestly help the 10-feature manageable-load case without amplifying Experiments 14 and 21. Keep surprise orienting, ordinary leak, overwrite, load noise, and parameter ranges near iteration-8 settings while testing this targeted gate-quality correction. Jointly safeguard Experiments 2, 4, 10, 12, 13, 20, and 22. Mean fit remains the priority; widening heterogeneity should wait until the large pooled errors in Experiments 1, 6, and 8 improve.

**Outcome of your advice:** iter 9 candidate loss=0.2121 -> CURRENT candidate loss=0.1637 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate remains faithful to the prescribed Spatially Indexed Reliability-Gated Revision family, but the incumbent-quality addition was rejected by the accept gate: loss increased from the accepted iteration-8 value of 0.1549 to 0.1637. It produced modest improvements in Experiments 3, 14, and 16, but these were outweighed by substantial regressions in Experiment 11 (terminal choice 0.602 versus 0.668 at iteration 8 and 0.843 observed), Experiment 12 (0.249 versus 0.188 and 0.171 observed), Experiment 20 (-0.109 versus -0.163 and -0.181 observed), and Experiment 21 (0.536 versus 0.478 and 0.390 observed). The core failures were barely affected: Experiment 8 remains dramatically too terminal-driven (0.792 versus 0.172), Experiment 6 still gives too little early-run choice (0.126 versus 0.307), Experiment 1 remains too contrastive (0.787 versus 0.494), and Experiment 19 remains too terminal-driven (0.857 versus 0.498). The quality gate therefore adds resistance in the wrong configurations without solving the generic late-cue dominance. Simulated behavior also remains much more homogeneous than human behavior in major failures such as Experiments 1 and 8. The accepted iteration-8 model nevertheless continues to provide a viable foundation, with good fits to Experiments 2, 4, 5, 7, 10, 12, 15, 16, 17, 18, 20, and 22.",
  "verdict": "regenerate",
  "rationale": "Build from the unchanged accepted iteration-8 base. The latest incumbent-quality modulation was rejected, so do not retain it or further tune its exponential quality gate. The earlier absolute-reliability, reliability-dispersion, challenger-release, and skipped-location-decay interventions were also rejected and should not be revisited. Try a different in-family use of the already accepted persistent stability trace: currently the trace affects update gating but is absent from the final decision readout, which depends only on the highly recency-sensitive commitment. Consequently, a late run can move commitment toward its terminal direction even while substantial coherent support remains in the trace. Form the final decision variable as a bounded continuous combination of commitment and a modest signed transform of the residual stability trace, with the trace contribution increasing smoothly with its surviving mass and vanishing as opposing evidence erodes or reverses it. This is still one continuous revision process, not a strategy mixture or coalition threshold. It should let surviving coherent support pull choices away from the terminal cue in Experiments 1, 6, and 8, while long challenger runs that genuinely reverse the trace can continue to dominate in Experiment 13. Calibrate the trace-readout weight conservatively and jointly safeguard Experiments 4, 10, 11, 12, and 13; in particular, do not globally strengthen the update gate, because terminal revision is already too weak in Experiment 11. Keep the accepted iteration-8 spatial, surprise, load, and reliability settings initially. Treat Experiment 19 as a weak constraint, and postpone widening subject-level ranges until the pooled errors improve."
}
```

## Usage

```json
{
  "input_tokens": 43946,
  "output_tokens": 1916
}
```
