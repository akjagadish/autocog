# generation_iter_08_attempt_00

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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_13`
- The recommendation below acts on THEORY 2 (= `pi_13`).

Replace pi_13 with a parallel competitive-encoding theory rather than another fixed scan-order model. The new theory should claim that cues are initially encoded in parallel into a capacity-limited comparison field. Each cue's activation is jointly determined by instructed reliability, serial-position distinctiveness at both boundaries, and local contrast with neighboring cues; there is no mandatory validity sort and no obligatory left-to-right update sequence. Activated cues supporting the same option pool into a coalition, but coalition strength saturates, so repeated cues neither add linearly nor create pi_13's persistent incumbent trace. Conflict triggers divisive competition: a distinctive opposing cue can receive a novelty or prediction-error boost, especially after a sharp reliability change, while a coherent coalition can suppress an ordinary terminal singleton. Choice is generated from the normalized difference between the two coalition activations, with a modest load-dependent precision loss and a task-invariant lapse. This framework should produce graded validity control in Experiment 1, below-chance terminal adherence in sparse dyads such as Experiment 2 when early/validity-based activation dominates, resistance to terminal cues in Experiments 3 and 10, but strong terminal influence when boundary distinctiveness or contrastive surprise is high, as in Experiments 6, 12, 13, 15, 23, and 24. Serial-position weighting should be estimable and potentially U-shaped rather than hard-coded as rightward, because Experiment 23 reports an absolute endpoint difference and does not itself establish its direction. Do not add experiment-specific rules or a broad trial-level mixture of validity-scan and spatial-scan strategies. Use a small set of shared parameters for reliability gain, boundary distinctiveness, coalition saturation, conflict/novelty gain, capacity or precision, and lapse; permit only constrained subject-level variation. History dependence should be absent or weak given Experiment 18. Fit and validate the replacement jointly across all experiments, with particular emphasis on reproducing the intermediate magnitudes in Experiments 1 and 2 while avoiding pi_13's catastrophic terminal overprediction in Experiments 10 and 21.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.584)

**Description**
Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.4743 (var=0.0168)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.1931 (var=0.0026)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.7129 (var=0.0033)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.5262 (var=0.0122)
- Experiment 5: real=0.0033 (var=0.0049) vs this=0.0150 (var=0.0104)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.3379 (var=0.0051)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.1824)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.2754 (var=0.0030)
- Experiment 9: real=-0.0325 (var=0.0099) vs this=-0.0083 (var=0.0103)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.7843 (var=0.0034)
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.6863 (var=0.0047)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.2650 (var=0.0077)
- Experiment 13: real=0.8810 (var=0.0070) vs this=0.7201 (var=0.0040)
- Experiment 14: real=0.0192 (var=0.0062) vs this=-0.0217 (var=0.0090)
- Experiment 15: real=0.0471 (var=0.0135) vs this=0.1525 (var=0.0093)
- Experiment 16: real=0.0311 (var=0.0170) vs this=-0.0111 (var=0.0098)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0227)
- Experiment 18: real=0.0262 (var=0.0075) vs this=-0.0175 (var=0.0140)
- Experiment 19: real=0.4978 (var=0.0077) vs this=0.6767 (var=0.0091)
- Experiment 20: real=-0.1813 (var=0.0102) vs this=0.0013 (var=0.0091)
- Experiment 21: real=0.3900 (var=0.0214) vs this=0.0167 (var=0.0062)
- Experiment 22: real=0.1750 (var=0.0208) vs this=0.0117 (var=0.0066)
- Experiment 23: real=0.4183 (var=0.0370) vs this=0.8758 (var=0.0591)
- Experiment 24: real=0.4192 (var=0.0032) vs this=0.1706 (var=0.0019)


---

### `pi_13` (overall score: 0.550)

**Description**
Spatially Indexed Reliability-Gated Revision. People inspect product features in their displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment. Revision is leaky, so recent cues ordinarily have greater influence, while repeated same-direction cues build a saturating, reliability-sensitive stability trace rather than an all-or-none coalition. Congruent trace consolidation is selectively nonlinear in relative reliability: repeated reliable cues stabilize an incumbent more strongly than equally numerous weak cues, while ordinary cue revision remains unchanged. This trace remains associated with an established incumbent during a reversal and is eroded gradually by cumulative challenger evidence; consequently, every cue in an opposing run remains partially gated until the challenger has continuously displaced the incumbent. Instructed validity jointly affects cue accessibility, revision magnitude, and the rate at which evidence builds or erodes stability. A weak cue that locally opposes a more reliable preceding cue evokes surprise-driven orienting, temporarily increasing its accessibility and capacity to revise the incumbent commitment. Accessibility also rises gradually toward the right side of the display, although the positional gradient is compressed on very long displays, with a small additional benefit for an isolated discriminating cue at the physical right edge. Increasing display load gradually reduces response precision, including a small decline for long unanimous sequences. Choices are generated from this single evolving commitment state, with no trial-level mixture of strategies.

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
        # opposes the persistent stability trace. The gate depends on the trace
        # itself, not instantaneous commitment, so it does not disappear merely
        # because an initial challenger has moved commitment toward zero.
        retained = float(np.clip(
            effective_leak_rate * commitment, -0.999999, 0.999999
        ))
        opposing_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) != direction
        )
        if opposing_stability:
            trace_gate_mass = min(2.5, abs(stability_trace) ** 0.75)
            stability_gate = 1.0 + (
                commitment_saturation * trace_gate_mass
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.7653 (var=0.0059)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.1877 (var=0.0021)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.6510 (var=0.0055)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6937 (var=0.0064)
- Experiment 5: real=0.0033 (var=0.0049) vs this=-0.0292 (var=0.0056)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.1221 (var=0.0029)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.2100)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.8115 (var=0.0028)
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.1483 (var=0.0096)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.9058 (var=0.0010)
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.6921 (var=0.0099)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.1919 (var=0.0066)
- Experiment 13: real=0.8810 (var=0.0070) vs this=0.8103 (var=0.0027)
- Experiment 14: real=0.0192 (var=0.0062) vs this=0.0525 (var=0.0055)
- Experiment 15: real=0.0471 (var=0.0135) vs this=0.0102 (var=0.0048)
- Experiment 16: real=0.0311 (var=0.0170) vs this=0.0022 (var=0.0104)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0054)
- Experiment 18: real=0.0262 (var=0.0075) vs this=0.0250 (var=0.0116)
- Experiment 19: real=0.4978 (var=0.0077) vs this=0.8639 (var=0.0037)
- Experiment 20: real=-0.1813 (var=0.0102) vs this=-0.1356 (var=0.0110)
- Experiment 21: real=0.3900 (var=0.0214) vs this=0.4867 (var=0.0277)
- Experiment 22: real=0.1750 (var=0.0208) vs this=0.1788 (var=0.0082)
- Experiment 23: real=0.4183 (var=0.0370) vs this=-0.0642 (var=0.0281)
- Experiment 24: real=0.4192 (var=0.0032) vs this=0.8050 (var=0.0021)


---

### `pi_8` (overall score: 0.547)

**Description**
Context-Calibrated Ordinal Event Policies. Decision makers inspect experts in instructed-validity order but encode the resulting discriminating sequence as ordinal events: an initial commitment, coherent directional runs, reversals, and a terminal commitment. Choice is generated by a context-sensitive selection between two subject-stable policies. The default terminal policy expresses the most recently attended commitment, whereas a coherent-chunk policy preserves the initial commitment when a moderately long, densely populated display is naturally represented as two opposing blocks. The chunk policy is recruited by conjunctions of density, coherent run structure, manageable load, and boundary regularities learned across trials—not by raw switch count alone. Sparse late suffixes or isolated terminal opposition foreground the terminal event and increase its confidence, even at high nominal feature counts. Equal-weight tally agreement can strengthen or weaken confidence in an already selected commitment but cannot reverse that commitment. Communicated validities determine inspection order and contribute only a strongly compressed confidence adjustment. Stable individual differences govern attention, recency and primacy propensities, chunk-switch threshold, contextual calibration, response gain, and lapse. In non-foregrounded moderate-load sequences, replacement of an existing commitment is slightly less complete, allowing preceding ordinal commitments to persist without changing policy selection.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ordinal event policy expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["attention_rate"])
    recency_propensity = float(parameters["recency_propensity"])
    primacy_propensity = float(parameters["primacy_propensity"])
    chunk_threshold = float(parameters["chunk_switch_threshold"])
    context_sensitivity = float(parameters["context_sensitivity"])
    load_tolerance = float(parameters["load_tolerance"])
    confidence_gain = float(parameters["confidence_gain"])
    tally_confidence = float(parameters["tally_confidence"])
    validity_compression = float(parameters["validity_compression"])
    lapse = float(parameters["lapse_rate"])

    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    def sequence_summary(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size == 0:
            return None
        d = a - b
        idx = np.flatnonzero(d != 0)
        if idx.size == 0:
            return None
        ds = np.sign(d[idx]).astype(float)
        run_lengths = []
        current = 1
        for k in range(1, ds.size):
            if ds[k] == ds[k - 1]:
                current += 1
            else:
                run_lengths.append(current)
                current = 1
        run_lengths.append(current)
        gaps = np.diff(idx) if idx.size > 1 else np.asarray([], dtype=float)
        max_gap = float(np.max(gaps) - 1.0) if gaps.size else 0.0
        max_gap = max(0.0, max_gap)
        return {
            "m": float(idx.size),
            "density": float(idx.size) / float(a.size),
            "n_runs": float(len(run_lengths)),
            "boundary": float(run_lengths[0]) / float(idx.size),
            "opposition": float(ds[0] != ds[-1]),
            "foreground": max_gap / float(max(1, a.size - 1)),
            "terminal_at_edge": float(idx[-1] == a.size - 1)
        }

    summaries = []
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        summary = sequence_summary(past_a[i], past_b[i])
        if summary is not None:
            summaries.append(summary)

    current_summary = sequence_summary(stim[0], stim[1])
    if current_summary is not None:
        summaries.append(current_summary)

    # Context statistics are based on structural distributions, not choices.
    # Shrink early estimates toward neutral priors so the model behaves
    # sensibly before it has sampled the task's display distribution.
    if summaries:
        sample_weight = min(1.0, len(summaries) / 12.0)
        opposition_rate_raw = float(np.mean([s["opposition"] for s in summaries]))
        foreground_rate_raw = float(np.mean([
            s["foreground"] * s["terminal_at_edge"] for s in summaries
        ]))
        coherent_boundaries = [
            s["boundary"] for s in summaries
            if s["density"] >= 0.85 and s["n_runs"] == 2.0
        ]
        if len(coherent_boundaries) >= 2:
            boundary_diversity_raw = min(
                1.0, 4.0 * float(np.std(coherent_boundaries))
            )
        else:
            boundary_diversity_raw = 0.0
        opposition_rate = (
            sample_weight * opposition_rate_raw + (1.0 - sample_weight) * 0.5
        )
        foreground_rate = sample_weight * foreground_rate_raw
        boundary_diversity = sample_weight * boundary_diversity_raw
    else:
        opposition_rate = 0.5
        foreground_rate = 0.0
        boundary_diversity = 0.0

    density = float(m) / float(max(1, n_features))

    run_lengths = []
    current_run = 1
    for k in range(1, m):
        if directions[k] == directions[k - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    n_runs = len(run_lengths)
    coherent_two_block = 1.0 if n_runs == 2 else 0.0

    physical_positions = np.sort(np.asarray(discriminating, dtype=int))
    if physical_positions.size > 1:
        empty_gaps = np.diff(physical_positions) - 1
        largest_gap = max(0.0, float(np.max(empty_gaps)))
    else:
        largest_gap = 0.0
    foreground = largest_gap / float(max(1, n_features - 1))
    terminal_at_edge = float(np.max(physical_positions) == n_features - 1)
    terminal_foreground = foreground * terminal_at_edge

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # A coherent-chunk policy is available only in a moderate-load window.
    # Both sides of the window are smooth and subject-specific. Dense
    # two-block structure is a necessary conjunction, so merely adding
    # switches cannot recruit primacy. Across-trial boundary diversity makes
    # block comparison somewhat more likely without making it mandatory.
    chunk_onset = sigmoid((float(m) - chunk_threshold) / 0.35)
    chunk_offset = sigmoid((chunk_threshold + 1.35 - float(m)) / 0.35)
    density_gate = sigmoid((density - 0.88) / 0.045)
    context_factor = 0.72 + context_sensitivity * boundary_diversity
    primacy_gate = (
        primacy_propensity
        * coherent_two_block
        * chunk_onset
        * chunk_offset
        * density_gate
        * context_factor
    )
    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))

    # Repeated sparse terminal opposition suppresses chunk recruitment and
    # restores terminal capture. This is especially important when a late
    # suffix is separated from the initial block by nondiscriminating cues.
    terminal_context = np.clip(
        terminal_foreground
        + context_sensitivity * foreground_rate
        + 0.20 * opposition_rate * terminal_foreground,
        0.0,
        1.0
    )
    primacy_gate *= (1.0 - 0.85 * terminal_context)
    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))

    # Replacement is modestly less complete in the subject's moderate-load
    # window. Density scales the adjustment smoothly, while foregrounded
    # terminal suffixes retain the baseline near-complete replacement rate.
    replacement_onset = sigmoid(
        (float(m) - (chunk_threshold - 0.50)) / 0.30
    )
    replacement_offset = sigmoid(
        (chunk_threshold + 1.00 - float(m)) / 0.30
    )
    moderate_replacement_window = replacement_onset * replacement_offset
    density_modulation = 0.70 + 0.30 * density
    replacement_reduction = (
        0.10
        * moderate_replacement_window
        * density_modulation
        * (1.0 - terminal_context)
    )
    effective_replacement = float(np.clip(
        attention - replacement_reduction, 0.0, 1.0
    ))

    # Exact expected direction of the most recently attended commitment.
    # A later attended event replaces the previous one; unattended events do
    # not. The no-commitment state contributes zero directional evidence.
    recency_direction = 0.0
    for direction in directions:
        recency_direction = (
            (1.0 - effective_replacement) * recency_direction
            + effective_replacement * direction
        )

    # Exact expected direction of the first attended commitment.
    primacy_direction = 0.0
    probability_uncommitted = 1.0
    for direction in directions:
        encoded_here = probability_uncommitted * attention
        primacy_direction += encoded_here * direction
        probability_uncommitted *= (1.0 - attention)

    recency_direction *= recency_propensity

    # Nominal display load weakens expression, while coherent runs and a
    # foregrounded late suffix restore confidence. Run coherence affects
    # confidence, not policy recruitment, avoiding a raw-switch-count rule.
    longest_run_fraction = float(max(run_lengths)) / float(m)
    display_overload = max(0.0, float(n_features) - load_tolerance)
    overload_divisor = 1.0 + 0.22 * display_overload
    overload_divisor *= 1.0 + 0.10 * max(0.0, float(m) - load_tolerance)

    coherence_support = max(0.0, longest_run_fraction - 0.35)
    confidence_multiplier = 1.0 + 0.90 * coherence_support
    confidence_multiplier += 3.0 * terminal_foreground
    confidence_multiplier += 1.25 * context_sensitivity * foreground_rate

    # Validity magnitudes have only compressed influence. Their main role was
    # already to determine inspection order.
    v = np.clip(validities[discriminating], 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    mean_log_odds = max(float(np.mean(log_odds)), 1e-8)
    terminal_reliability = max(float(log_odds[-1] / mean_log_odds), 1e-6)
    validity_factor = float(
        np.power(np.clip(terminal_reliability, 0.25, 4.0), validity_compression)
    )

    # Tally is a confidence cue only. It can reinforce or attenuate the
    # terminal commitment but cannot change its sign or install a weighted
    # global coalition as the selected response.
    tally_direction = float(np.sign(np.sum(directions)))
    terminal_direction = float(directions[-1])
    if tally_direction == 0.0:
        tally_factor = 1.0
    elif tally_direction == terminal_direction:
        tally_factor = 1.0 + tally_confidence
    else:
        tally_factor = 1.0 - 0.30 * tally_confidence

    gain = (
        confidence_gain
        * confidence_multiplier
        * validity_factor
        * tally_factor
        / max(overload_divisor, 1e-8)
    )
    gain = float(np.clip(gain, 0.05, 12.0))

    p_a_recency = sigmoid(gain * recency_direction)
    p_a_primacy = sigmoid(gain * primacy_direction)
    p_a_core = (
        (1.0 - primacy_gate) * p_a_recency
        + primacy_gate * p_a_primacy
    )

    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- attention_rate: [0.91, 0.99]
- recency_propensity: [0.88, 1.0]
- primacy_propensity: [0.76, 0.94]
- chunk_switch_threshold: [5.05, 5.65]
- context_sensitivity: [0.35, 0.75]
- load_tolerance: [4.4, 5.8]
- confidence_gain: [1.60, 2.50]
- tally_confidence: [0.05, 0.22]
- validity_compression: [0.0, 0.10]
- lapse_rate: [0.05, 0.17]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.5276 (var=0.0147)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.1765 (var=0.0015)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.6569 (var=0.0032)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6408 (var=0.0101)
- Experiment 5: real=0.0033 (var=0.0049) vs this=0.0112 (var=0.0099)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.3050 (var=0.0041)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.1924)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.3069 (var=0.0027)
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.0083 (var=0.0044)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.8753 (var=0.0015)
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.8558 (var=0.0026)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.1313 (var=0.0048)
- Experiment 13: real=0.8810 (var=0.0070) vs this=0.3399 (var=0.0051)
- Experiment 14: real=0.0192 (var=0.0062) vs this=0.0863 (var=0.0106)
- Experiment 15: real=0.0471 (var=0.0135) vs this=0.0406 (var=0.0094)
- Experiment 16: real=0.0311 (var=0.0170) vs this=-0.0155 (var=0.0095)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0013)
- Experiment 18: real=0.0262 (var=0.0075) vs this=0.0144 (var=0.0064)
- Experiment 19: real=0.4978 (var=0.0077) vs this=0.6833 (var=0.0052)
- Experiment 20: real=-0.1813 (var=0.0102) vs this=-0.0681 (var=0.0101)
- Experiment 21: real=0.3900 (var=0.0214) vs this=0.0675 (var=0.0094)
- Experiment 22: real=0.1750 (var=0.0208) vs this=0.0458 (var=0.0077)
- Experiment 23: real=0.4183 (var=0.0370) vs this=0.3808 (var=0.0414)
- Experiment 24: real=0.4192 (var=0.0032) vs this=0.1258 (var=0.0022)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2935 -> ACCEPTED
- iter 2: loss=0.3251 -> REJECTED
- iter 3: loss=0.3075 -> REJECTED
- iter 4: loss=0.3148 -> REJECTED
- iter 5: loss=0.2684 -> ACCEPTED
- iter 6: loss=0.3774 -> REJECTED
- iter 7: loss=0.2723 -> REJECTED
- iter 8: loss=0.2727 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.2684 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
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
**Previous candidate values (this loop):**
  - iter 1: 0.3488 (var=0.0113) (Δ vs real -0.1457)
  - iter 2: 0.4456 (var=0.0073) (Δ vs real -0.0489)
  - iter 3: 0.3438 (var=0.0087) (Δ vs real -0.1507)
  - iter 4: 0.3442 (var=0.0123) (Δ vs real -0.1503)
  - iter 5: 0.3758 (var=0.0058) (Δ vs real -0.1186)
  - iter 6: 0.1851 (var=0.0071) (Δ vs real -0.3093)
  - iter 7: 0.3758 (var=0.0068) (Δ vs real -0.1186)
  - iter 8 (most recent): 0.3994 (var=0.0060) (Δ vs real -0.0950)
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
**Previous candidate values (this loop):**
  - iter 1: 0.4944 (var=0.0034) (Δ vs real +0.3006)
  - iter 2: 0.5256 (var=0.0055) (Δ vs real +0.3319)
  - iter 3: 0.4856 (var=0.0030) (Δ vs real +0.2919)
  - iter 4: 0.4921 (var=0.0038) (Δ vs real +0.2983)
  - iter 5: 0.4802 (var=0.0020) (Δ vs real +0.2865)
  - iter 6: 0.5060 (var=0.0076) (Δ vs real +0.3123)
  - iter 7: 0.4888 (var=0.0031) (Δ vs real +0.2950)
  - iter 8 (most recent): 0.4894 (var=0.0038) (Δ vs real +0.2956)
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
**Previous candidate values (this loop):**
  - iter 1: 0.4481 (var=0.0026) (Δ vs real -0.1269)
  - iter 2: 0.3544 (var=0.0020) (Δ vs real -0.2206)
  - iter 3: 0.4371 (var=0.0021) (Δ vs real -0.1379)
  - iter 4: 0.4688 (var=0.0024) (Δ vs real -0.1062)
  - iter 5: 0.4988 (var=0.0019) (Δ vs real -0.0762)
  - iter 6: 0.2996 (var=0.0030) (Δ vs real -0.2754)
  - iter 7: 0.5067 (var=0.0021) (Δ vs real -0.0683)
  - iter 8 (most recent): 0.4838 (var=0.0033) (Δ vs real -0.0912)
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
**Previous candidate values (this loop):**
  - iter 1: -0.1037 (var=0.0158) (Δ vs real -0.7688)
  - iter 2: 0.0746 (var=0.0086) (Δ vs real -0.5904)
  - iter 3: -0.1142 (var=0.0106) (Δ vs real -0.7792)
  - iter 4: -0.0546 (var=0.0109) (Δ vs real -0.7196)
  - iter 5: -0.0346 (var=0.0107) (Δ vs real -0.6996)
  - iter 6: -0.1979 (var=0.0071) (Δ vs real -0.8629)
  - iter 7: -0.0004 (var=0.0075) (Δ vs real -0.6654)
  - iter 8 (most recent): -0.0312 (var=0.0106) (Δ vs real -0.6963)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1687 (var=0.0097) (Δ vs real +0.1654)
  - iter 2: 0.2013 (var=0.0103) (Δ vs real +0.1979)
  - iter 3: 0.2192 (var=0.0125) (Δ vs real +0.2158)
  - iter 4: 0.1708 (var=0.0114) (Δ vs real +0.1675)
  - iter 5: -0.0292 (var=0.0168) (Δ vs real -0.0325)
  - iter 6: 0.5308 (var=0.0105) (Δ vs real +0.5275)
  - iter 7: 0.0300 (var=0.0090) (Δ vs real +0.0267)
  - iter 8 (most recent): -0.0112 (var=0.0116) (Δ vs real -0.0146)
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
**Previous candidate values (this loop):**
  - iter 1: 0.6850 (var=0.0038) (Δ vs real +0.3783)
  - iter 2: 0.6150 (var=0.0048) (Δ vs real +0.3083)
  - iter 3: 0.6700 (var=0.0038) (Δ vs real +0.3633)
  - iter 4: 0.6250 (var=0.0054) (Δ vs real +0.3183)
  - iter 5: 0.5121 (var=0.0055) (Δ vs real +0.2054)
  - iter 6: 0.8304 (var=0.0039) (Δ vs real +0.5238)
  - iter 7: 0.5171 (var=0.0051) (Δ vs real +0.2104)
  - iter 8 (most recent): 0.5317 (var=0.0051) (Δ vs real +0.2250)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0900) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.1344) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0564) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 8 (most recent): 0.0000 (var=0.2016) (Δ vs real +0.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3915 (var=0.0028) (Δ vs real +0.2194)
  - iter 2: 0.5590 (var=0.0026) (Δ vs real +0.3869)
  - iter 3: 0.3887 (var=0.0020) (Δ vs real +0.2167)
  - iter 4: 0.3377 (var=0.0027) (Δ vs real +0.1656)
  - iter 5: 0.4562 (var=0.0026) (Δ vs real +0.2842)
  - iter 6: 0.3696 (var=0.0039) (Δ vs real +0.1975)
  - iter 7: 0.4748 (var=0.0028) (Δ vs real +0.3027)
  - iter 8 (most recent): 0.4725 (var=0.0022) (Δ vs real +0.3004)
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
**Previous candidate values (this loop):**
  - iter 1: 0.2696 (var=0.0111) (Δ vs real +0.3021)
  - iter 2: 0.4504 (var=0.0084) (Δ vs real +0.4829)
  - iter 3: 0.2754 (var=0.0130) (Δ vs real +0.3079)
  - iter 4: 0.1275 (var=0.0202) (Δ vs real +0.1600)
  - iter 5: 0.1537 (var=0.0088) (Δ vs real +0.1862)
  - iter 6: 0.3400 (var=0.0159) (Δ vs real +0.3725)
  - iter 7: 0.1467 (var=0.0147) (Δ vs real +0.1792)
  - iter 8 (most recent): 0.1537 (var=0.0132) (Δ vs real +0.1862)
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
**Previous candidate values (this loop):**
  - iter 1: 0.6322 (var=0.0024) (Δ vs real -0.2425)
  - iter 2: 0.6908 (var=0.0024) (Δ vs real -0.1840)
  - iter 3: 0.6244 (var=0.0014) (Δ vs real -0.2504)
  - iter 4: 0.6328 (var=0.0016) (Δ vs real -0.2419)
  - iter 5: 0.6305 (var=0.0018) (Δ vs real -0.2443)
  - iter 6: 0.5872 (var=0.0017) (Δ vs real -0.2875)
  - iter 7: 0.6395 (var=0.0018) (Δ vs real -0.2353)
  - iter 8 (most recent): 0.6534 (var=0.0020) (Δ vs real -0.2214)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3133 (var=0.0067) (Δ vs real -0.5292)
  - iter 2: 0.4775 (var=0.0070) (Δ vs real -0.3650)
  - iter 3: 0.3108 (var=0.0079) (Δ vs real -0.5317)
  - iter 4: 0.3617 (var=0.0058) (Δ vs real -0.4808)
  - iter 5: 0.4954 (var=0.0056) (Δ vs real -0.3471)
  - iter 6: 0.2233 (var=0.0070) (Δ vs real -0.6192)
  - iter 7: 0.4804 (var=0.0075) (Δ vs real -0.3621)
  - iter 8 (most recent): 0.4679 (var=0.0054) (Δ vs real -0.3746)
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
**Previous candidate values (this loop):**
  - iter 1: 0.6150 (var=0.0084) (Δ vs real +0.4437)
  - iter 2: 0.5181 (var=0.0097) (Δ vs real +0.3469)
  - iter 3: 0.5869 (var=0.0081) (Δ vs real +0.4156)
  - iter 4: 0.6525 (var=0.0084) (Δ vs real +0.4813)
  - iter 5: 0.4625 (var=0.0086) (Δ vs real +0.2913)
  - iter 6: 0.7125 (var=0.0166) (Δ vs real +0.5413)
  - iter 7: 0.4988 (var=0.0089) (Δ vs real +0.3275)
  - iter 8 (most recent): 0.5100 (var=0.0103) (Δ vs real +0.3387)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3463 (var=0.0035) (Δ vs real -0.5347)
  - iter 2: 0.4134 (var=0.0048) (Δ vs real -0.4676)
  - iter 3: 0.3301 (var=0.0025) (Δ vs real -0.5508)
  - iter 4: 0.3946 (var=0.0043) (Δ vs real -0.4864)
  - iter 5: 0.4335 (var=0.0045) (Δ vs real -0.4475)
  - iter 6: 0.2050 (var=0.0025) (Δ vs real -0.6760)
  - iter 7: 0.4466 (var=0.0037) (Δ vs real -0.4344)
  - iter 8 (most recent): 0.4220 (var=0.0032) (Δ vs real -0.4590)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0267 (var=0.0061) (Δ vs real +0.0075)
  - iter 2: 0.1938 (var=0.0078) (Δ vs real +0.1746)
  - iter 3: -0.0183 (var=0.0070) (Δ vs real -0.0375)
  - iter 4: -0.0308 (var=0.0077) (Δ vs real -0.0500)
  - iter 5: -0.0533 (var=0.0070) (Δ vs real -0.0725)
  - iter 6: 0.0325 (var=0.0082) (Δ vs real +0.0133)
  - iter 7: -0.0779 (var=0.0122) (Δ vs real -0.0971)
  - iter 8 (most recent): -0.0463 (var=0.0082) (Δ vs real -0.0654)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0271 (var=0.0071) (Δ vs real -0.0200)
  - iter 2: 0.0171 (var=0.0057) (Δ vs real -0.0300)
  - iter 3: 0.0200 (var=0.0062) (Δ vs real -0.0271)
  - iter 4: 0.0519 (var=0.0055) (Δ vs real +0.0048)
  - iter 5: 0.0215 (var=0.0065) (Δ vs real -0.0256)
  - iter 6: 0.0254 (var=0.0077) (Δ vs real -0.0217)
  - iter 7: 0.0137 (var=0.0083) (Δ vs real -0.0333)
  - iter 8 (most recent): 0.0029 (var=0.0045) (Δ vs real -0.0442)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0185 (var=0.0117) (Δ vs real -0.0126)
  - iter 2: 0.0237 (var=0.0142) (Δ vs real -0.0074)
  - iter 3: -0.0037 (var=0.0138) (Δ vs real -0.0348)
  - iter 4: -0.0013 (var=0.0138) (Δ vs real -0.0324)
  - iter 5: 0.0019 (var=0.0139) (Δ vs real -0.0292)
  - iter 6: 0.0098 (var=0.0090) (Δ vs real -0.0213)
  - iter 7: 0.0095 (var=0.0132) (Δ vs real -0.0216)
  - iter 8 (most recent): -0.0011 (var=0.0077) (Δ vs real -0.0322)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0010) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0047) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 8 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0231 (var=0.0091) (Δ vs real -0.0031)
  - iter 2: 0.0256 (var=0.0175) (Δ vs real -0.0006)
  - iter 3: -0.0150 (var=0.0168) (Δ vs real -0.0412)
  - iter 4: 0.0244 (var=0.0122) (Δ vs real -0.0019)
  - iter 5: -0.0019 (var=0.0181) (Δ vs real -0.0281)
  - iter 6: -0.0094 (var=0.0083) (Δ vs real -0.0356)
  - iter 7: 0.0031 (var=0.0142) (Δ vs real -0.0231)
  - iter 8 (most recent): 0.0044 (var=0.0123) (Δ vs real -0.0219)
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
**Previous candidate values (this loop):**
  - iter 1: 0.4172 (var=0.0076) (Δ vs real -0.0806)
  - iter 2: 0.5239 (var=0.0085) (Δ vs real +0.0261)
  - iter 3: 0.4106 (var=0.0063) (Δ vs real -0.0872)
  - iter 4: 0.4333 (var=0.0048) (Δ vs real -0.0644)
  - iter 5: 0.4828 (var=0.0054) (Δ vs real -0.0150)
  - iter 6: 0.3739 (var=0.0090) (Δ vs real -0.1239)
  - iter 7: 0.4828 (var=0.0054) (Δ vs real -0.0150)
  - iter 8 (most recent): 0.5100 (var=0.0053) (Δ vs real +0.0122)
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
**Previous candidate values (this loop):**
  - iter 1: -0.0706 (var=0.0172) (Δ vs real +0.1106)
  - iter 2: -0.2662 (var=0.0230) (Δ vs real -0.0850)
  - iter 3: -0.0744 (var=0.0115) (Δ vs real +0.1069)
  - iter 4: 0.0606 (var=0.0170) (Δ vs real +0.2419)
  - iter 5: 0.0125 (var=0.0222) (Δ vs real +0.1938)
  - iter 6: -0.1512 (var=0.0206) (Δ vs real +0.0300)
  - iter 7: -0.0037 (var=0.0189) (Δ vs real +0.1775)
  - iter 8 (most recent): -0.0131 (var=0.0207) (Δ vs real +0.1681)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0792 (var=0.0120) (Δ vs real -0.3108)
  - iter 2: 0.1258 (var=0.0102) (Δ vs real -0.2642)
  - iter 3: 0.2583 (var=0.0202) (Δ vs real -0.1317)
  - iter 4: 0.1667 (var=0.0162) (Δ vs real -0.2233)
  - iter 5: 0.0242 (var=0.0101) (Δ vs real -0.3658)
  - iter 6: 0.1975 (var=0.0179) (Δ vs real -0.1925)
  - iter 7: 0.0392 (var=0.0085) (Δ vs real -0.3508)
  - iter 8 (most recent): 0.0042 (var=0.0118) (Δ vs real -0.3858)
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
**Previous candidate values (this loop):**
  - iter 1: 0.2667 (var=0.0122) (Δ vs real +0.0917)
  - iter 2: 0.1092 (var=0.0075) (Δ vs real -0.0658)
  - iter 3: 0.2538 (var=0.0110) (Δ vs real +0.0787)
  - iter 4: 0.0154 (var=0.0110) (Δ vs real -0.1596)
  - iter 5: 0.1742 (var=0.0107) (Δ vs real -0.0008)
  - iter 6: 0.4925 (var=0.0147) (Δ vs real +0.3175)
  - iter 7: 0.1417 (var=0.0101) (Δ vs real -0.0333)
  - iter 8 (most recent): 0.1679 (var=0.0114) (Δ vs real -0.0071)
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
**Previous candidate values (this loop):**
  - iter 1: -0.0892 (var=0.0458) (Δ vs real -0.5075)
  - iter 2: -0.3058 (var=0.0528) (Δ vs real -0.7242)
  - iter 3: -0.1825 (var=0.0333) (Δ vs real -0.6008)
  - iter 4: -0.0708 (var=0.0497) (Δ vs real -0.4892)
  - iter 5: -0.0250 (var=0.0398) (Δ vs real -0.4433)
  - iter 6: -0.3775 (var=0.0625) (Δ vs real -0.7958)
  - iter 7: 0.0500 (var=0.0434) (Δ vs real -0.3683)
  - iter 8 (most recent): 0.0333 (var=0.0408) (Δ vs real -0.3850)
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
**Previous candidate values (this loop):**
  - iter 1: 0.4202 (var=0.0024) (Δ vs real +0.0010)
  - iter 2: 0.7942 (var=0.0021) (Δ vs real +0.3750)
  - iter 3: 0.4021 (var=0.0038) (Δ vs real -0.0171)
  - iter 4: 0.3348 (var=0.0037) (Δ vs real -0.0844)
  - iter 5: 0.4369 (var=0.0029) (Δ vs real +0.0177)
  - iter 6: 0.3506 (var=0.0044) (Δ vs real -0.0685)
  - iter 7: 0.4465 (var=0.0027) (Δ vs real +0.0273)
  - iter 8 (most recent): 0.4371 (var=0.0027) (Δ vs real +0.0179)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Parallel Contrastive Coalition Field. All discriminating cues are encoded concurrently into a capacity-limited comparison field rather than inspected in a mandatory validity or spatial sequence. A cue's activation combines its instructed reliability with a U-shaped boundary-distinctiveness component, an estimable spatial tilt, and local contrast against the nearest discriminating cues. Opposing cues that occur across a sharp reliability drop receive a contrastive-surprise boost, allowing a distinctive weak challenger to overcome its low instructed reliability without creating a persistent incumbent trace. Simultaneously active cues supporting the same option first pool into a saturating coalition, so repeated support has diminishing returns and same-option cues do not suppress one another. Divisive capacity competition then operates between the two pooled coalitions, preferentially preserving the stronger coalition without restoring linear numerical dominance. Choice follows their temperature-normalized difference, with load-dependent precision loss and a task-invariant lapse. The model has no trial-history learning, strategy mixture, fixed scan order, or sequential commitment state.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Parallel Contrastive Coalition Field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    reliability_gain = float(parameters["reliability_gain"])
    boundary_gain = float(parameters["boundary_gain"])
    spatial_tilt = float(parameters["spatial_tilt"])
    coalition_saturation = float(parameters["coalition_saturation"])
    novelty_gain = float(parameters["novelty_gain"])
    capacity_loss = float(parameters["capacity_loss"])
    competition_temperature = float(parameters["competition_temperature"])
    choice_precision = float(parameters["choice_precision"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[discriminating]).astype(float)
    m = int(discriminating.size)

    # Reliability is represented on the normative log-odds scale. Centering
    # preserves within-display reliability contrasts while making the shared
    # gain portable across experiments with different validity schedules.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        rel_z = np.zeros(n_features, dtype=float)
    else:
        rel_z = np.clip((reliability - rel_mean) / rel_sd, -2.5, 2.5)

    raw_activation = np.zeros(m, dtype=float)

    # These activations are computed from the complete display. References to
    # left and right neighbors describe simultaneous spatial relations, not an
    # update order or a serial scan.
    for k, j_cell in enumerate(discriminating):
        j = int(j_cell)
        direction = float(directions[k])
        position = float(j) / float(max(1, n_features - 1))
        signed_position = 2.0 * position - 1.0

        # Both endpoints can be distinctive. The separately estimated tilt
        # permits endpoint asymmetry without assuming universal right recency.
        boundary = signed_position * signed_position
        spatial_log_activation = (
            boundary_gain * boundary + spatial_tilt * signed_position
        )

        surprise = 0.0
        symmetric_contrast = 0.0

        if k > 0:
            left_j = int(discriminating[k - 1])
            left_opposes = float(directions[k - 1] != direction)
            if left_opposes > 0.0:
                # A reliability drop at an opposing spatial boundary is a
                # prediction-error signal, not evidence that cues were scanned.
                drop = max(0.0, float(validities[left_j] - validities[j]))
                surprise += min(1.25, drop / 0.40)
                symmetric_contrast += min(
                    1.0, abs(float(validities[left_j] - validities[j])) / 0.40
                )

        if k + 1 < m:
            right_j = int(discriminating[k + 1])
            right_opposes = float(directions[k + 1] != direction)
            if right_opposes > 0.0:
                symmetric_contrast += min(
                    1.0, abs(float(validities[right_j] - validities[j])) / 0.40
                )

        # Most contrast is a sharp-drop novelty response. A smaller symmetric
        # term captures ordinary local pop-out at either side of a boundary.
        novelty_signal = surprise + 0.18 * symmetric_contrast
        log_activation = (
            reliability_gain * rel_z[j]
            + spatial_log_activation
            + novelty_gain * novelty_signal
        )
        raw_activation[k] = float(np.exp(np.clip(log_activation, -8.0, 8.0)))

    # Concurrent same-option cues pool before competition. Thus they exhibit
    # diminishing returns but do not consume capacity by suppressing one
    # another at the cue level.
    support_a = float(np.sum(raw_activation[directions > 0.0]))
    support_b = float(np.sum(raw_activation[directions < 0.0]))

    sat = max(coalition_saturation, 1e-8)
    coalition_a = sat * (1.0 - np.exp(-support_a / sat))
    coalition_b = sat * (1.0 - np.exp(-support_b / sat))

    # Capacity competition occurs between option-level coalitions. The mild
    # temperature sharpens genuine coalition differences after repeated cues
    # have already been compressed by saturation.
    competitive_a = coalition_a / (1.0 + capacity_loss * coalition_b)
    competitive_b = coalition_b / (1.0 + capacity_loss * coalition_a)
    competitive_a = float(np.power(max(competitive_a, 0.0), competition_temperature))
    competitive_b = float(np.power(max(competitive_b, 0.0), competition_temperature))

    # Divisive comparison expresses evidence relative to the total activated
    # field, preventing large displays from creating unbounded decision values.
    coalition_total = competitive_a + competitive_b
    if coalition_total <= 1e-12 or not np.isfinite(coalition_total):
        normalized_difference = 0.0
    else:
        normalized_difference = (competitive_a - competitive_b) / coalition_total

    # Display and evidence load modestly lower readout precision. Saturating
    # pooling and this precision loss can produce a small decline even for long
    # unanimous displays, without a discrete fatigue threshold.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = choice_precision / (
        1.0
        + 0.55 * capacity_loss * display_load
        + 0.35 * capacity_loss * evidence_load
    )
    precision = float(np.clip(precision, 0.20, 8.0))

    logit_a = float(np.clip(precision * normalized_difference, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)

    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- reliability_gain: [0.25, 0.50]
- boundary_gain: [0.25, 0.55]
- spatial_tilt: [-0.90, -0.55]
- coalition_saturation: [0.55, 0.95]
- novelty_gain: [1.50, 2.40]
- capacity_loss: [0.08, 0.18]
- competition_temperature: [1.05, 1.35]
- choice_precision: [3.00, 4.30]
- lapse_rate: [0.05, 0.12]

`rationale`: This is a minimal normalization-order edit to the accepted candidate. Cue activation, reliability calibration, spatial weighting, local novelty, load loss, lapse, and the no-history parallel architecture are unchanged. The sole mechanistic correction is that raw cue activations now pool and saturate within each option before divisive competition occurs. The previous all-to-all cue suppression made congruent cues suppress one another while also crushing isolated challengers, yet still left large coalitions numerically dominant. Earlier saturation removes that distortion: repeated support quickly reaches diminishing returns, whereas a salient opposing singleton retains its activation. Competition is then applied only across option coalitions, with a narrowly constrained temperature controlling readout contrast. The stronger saturation range should reduce oversized early or majority-coalition choices in Experiments 2, 6, and 12 and permit contrastively activated challengers to raise terminal influence in Experiments 4, 11, and 13. Because the accepted spatial and novelty equations are retained, this avoids repeating the rejected global endpoint, reliability-normalization, or novelty-budget edits and is intended to preserve the established fits in Experiments 14 and 24 while limiting indiscriminate terminal enhancement in Experiment 8.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed parallel competitive-encoding family and improves dramatically on pi_13 in some important respects: it exactly captures Experiment 24, closely fits Experiments 14, 15, 16, and 18, and gets the null diagnostics in Experiments 7 and 17. It also avoids pi_13's extreme terminal overprediction in Experiment 8. Nevertheless, it is not experiment-invariant. A broad cluster of results shows that salient opposing or terminal cues remain much too weak: Experiment 2 is near chance rather than below chance (0.494 vs 0.194), Experiment 4 reverses the terminal association (-0.104 vs 0.665), Experiment 6 overpredicts early-coalition choice (0.685 vs 0.307), Experiment 11 underpredicts terminal choice (0.313 vs 0.843), Experiment 12 overpredicts the 3-to-2 majority (0.615 vs 0.171), and Experiment 13 severely underpredicts terminal choice (0.346 vs 0.881). Experiment 21 also misses the endpoint asymmetry (0.079 vs 0.390), while Experiment 23 has the wrong crossover sign (-0.089 vs 0.418). Experiment 10 is substantially low as well (0.632 vs 0.875), although avoiding pi_13's overprediction was a stated objective. Conversely, globally increasing terminal weighting would be inadequate because Experiment 8 already overpredicts ordinary terminal adherence (0.392 vs 0.172), Experiment 22 overstates the drop contrast (0.267 vs 0.175), and Experiment 24 is already exact. Experiment 1 is also too weak (0.349 vs 0.494), indicating insufficient graded reliability/tally control, while Experiments 5 and 9 show residual sequence-geometry and highest-validity biases.
Rationale: Although this first candidate was accepted by the gate, the aggregate fit remains inadequate and several misses are qualitative. The central problem is not the parallel-coalition mechanism itself but its activation calibration. The fixed strongly negative spatial-tilt range suppresses right-side terminal cues too generally, while the novelty computation is narrowly directional: only a reliability drop from the immediately preceding opposing discriminating cue receives the main boost. That combination cannot distinguish salient terminal challengers in Experiments 2, 4, 6, 11, 12, 13, 21, and 23 from ordinary terminal singletons that should remain weak in Experiment 8. Revise within the same family by broadening the spatial-tilt range toward zero rather than hard-wiring a large leftward bias, and let endpoint asymmetry be estimated jointly with a stronger boundary term. More importantly, replace the one-sided linear drop boost with a sharply gated, saturating contrast/prediction-error signal based on the absolute local reliability discontinuity and the challenger's opposition to the active coalition. This should make sharp, isolated crossover cues salient without indiscriminately raising every terminal cue or further inflating Experiment 22. Recheck the reliability normalization as well: it is currently standardized over the full validity schedule, including nondiscriminating cues, despite the comment describing within-display contrasts. Normalizing over encoded/discriminating cues, with a retained absolute reliability component, may improve the sparse conflicts in Experiment 2 and the graded effect in Experiment 1. Jointly retune novelty, boundary strength, coalition saturation, and divisive competition while preserving the parallel field, no-history assumption, and task-invariant lapse. Priority constraints are to lower Experiments 2, 6, and 12; raise Experiments 4, 11, and 13; reverse Experiment 23's sign; and do so without increasing Experiments 8, 22, or 24.

**Outcome of this advice:** iter 1 candidate loss=0.2935 -> iter 2 candidate loss=0.3251 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed parallel competitive-encoding family, but the current calibration is worse than the accepted base and was correctly rejected by the gate (loss 0.3251 versus 0.2935). It improves Experiment 1 (0.446 versus 0.494) and moves Experiments 4, 6, 10–13, 19–22 in the desired direction. Those gains are outweighed by severe regressions: Experiment 2 moves farther from the required below-chance TTB rate (0.526 versus 0.194); terminal adherence becomes much too high in Experiment 8 (0.559 versus 0.172) and Experiment 24 (0.794 versus 0.419); highest-validity bias rises in Experiment 9 (0.450 versus -0.033); Experiment 14 develops a spurious foregrounding effect (0.194 versus 0.019); and the Experiment 23 crossover becomes more negative rather than reversing sign (-0.306 versus 0.418). Experiment 3 also deteriorates (0.354 versus 0.575), while Experiments 4, 11, 13, and 21 remain substantially too weak despite their improvements. Thus the edit trades broad, indiscriminate endpoint and reliability effects for a few local gains rather than producing experiment-invariant selectivity. The very low candidate variances in several large-miss experiments, including 4 and 21, indicate that these are systematic mean-prediction errors rather than merely insufficient subject heterogeneity.
Rationale: The previous recommendation to replace the accepted base's calibration with encoded-cue z-scoring, a positive spatial tilt, a stronger symmetric boundary term, and absolute-discontinuity novelty was rejected by the gate, so that package should not be repeated. In particular, encoded-only standardization makes sparse sets overly dependent on their own extrema, while the positive tilt plus strong boundary gain indiscriminately promotes late endpoints. That explains the regressions in Experiments 8, 9, 14, 23, and especially 24. Build the next variant on the unchanged accepted base and make a smaller, more selective in-family intervention. Preserve mostly absolute/full-schedule reliability calibration, adding at most a weak blend of encoded-set normalization rather than replacing it. Keep spatial tilt near zero in a narrow range between the accepted negative setting and the rejected positive setting; do not globally increase boundary gain. Instead, let boundary distinctiveness interact with conflict-specific surprise and display load: ordinary endpoints should receive little extra activation, whereas an endpoint opposing a coherent coalition can be boosted when its reliability is unexpectedly different relative to a locally estimated reliability baseline. Compute locality from actual adjacent display locations rather than treating widely separated discriminating cues as neighbors and then rewarding their isolation; that construction likely creates the false effects in Experiments 14 and 24. Use a signed or two-channel reliability prediction error with separately constrained upward- and downward-change gains, rather than the rejected symmetric absolute discontinuity. This can retain the genuine large-drop contrast in Experiment 22 while allowing Experiment 23's reliability-crossover sign to be fitted. A modest load-dependent amplification of spatial asymmetry is also preferable to a global tilt for raising the 20-feature endpoint difference in Experiment 21 without disrupting shorter displays. Priority tests are to preserve the accepted base's exact Experiment 24 and near-null Experiment 14 fits, lower Experiments 2, 8, and 9, reverse Experiment 23, and then assess whether conflict-gated endpoint activation raises Experiments 4, 11, and 13 without globally increasing terminal adherence.

**Outcome of this advice:** iter 2 candidate loss=0.3251 -> iter 3 candidate loss=0.3075 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed parallel competitive-encoding family, but it is not experiment-invariant and was rejected by the accept gate (loss 0.3075 versus the accepted base’s 0.2935). The new long-sparse asymmetry does improve its intended target, Experiment 21, from 0.079 to 0.258, but the endpoint difference remains below 0.390 and this isolated gain does not compensate for the unchanged core failures. The model still predicts approximately chance adherence to the highest-validity cue in Experiment 2 rather than the observed strong reversal (0.486 versus 0.194), and it gives the wrong effective winner in Experiments 4 (-0.114 versus 0.665), 6 (0.670 early-coalition choice versus 0.307), and 12 (0.587 majority choice versus 0.171). Terminal influence remains far too weak in Experiments 3, 10, 11, and 13, especially Experiments 11 (0.311 versus 0.843) and 13 (0.330 versus 0.881). At the same time, ordinary terminal adherence is still too high in Experiment 8 (0.389 versus 0.172), so a general terminal or endpoint boost is not appropriate. Experiment 23 retains the wrong crossover sign (-0.183 versus 0.418), while Experiments 1, 5, and 9 show insufficient validity-by-tally control, excessive sequence-geometry sensitivity, and excessive highest-validity influence, respectively. The good fits to Experiments 7, 14, 15, 17, and 24 and the near-null history result do support parts of the framework, but they are outweighed by numerous systematic directional errors. The low variances accompanying many misses further indicate incorrect mean activation rankings rather than merely inadequate subject variation.
Rationale: Because this candidate was rejected, the next proposal should build on the unchanged iteration-1 base, not retain the long-sparse asymmetry patch. The previous global endpoint/reliability recalibration was rejected, and the present display-length-specific amplification was also rejected; neither direction should be repeated. The next in-family adjustment should instead target how contrastive activation is allocated during divisive competition. In the current equations, every opposing predecessor produces an additive novelty increment before exponentiation, while suppression depends only on total raw activation. Consequently, the field cannot reliably distinguish one genuinely diagnostic challenger from several mundane switch boundaries, and it lacks a coalition-relative criterion for deciding whether a singleton is surprising enough to survive competition. Replace this additive per-boundary novelty accumulation with a capped, capacity-budgeted contrast gate: compute cue residual salience in parallel relative to the local physical neighborhood and the opposing coalition’s pooled reliability/activation, pass that residual through a thresholded sigmoid or softmax temperature, and allocate a fixed novelty budget to the most distinctive challenger. Ordinary boundaries should receive little or no boost, and multiple switches should compete for rather than independently accumulate novelty. Keep the accepted base’s global spatial tilt, boundary term, and full-schedule reliability normalization initially fixed so this new knob can be evaluated without recreating the rejected broad regressions. Jointly tune the contrast threshold, competition temperature, and coalition saturation. Priority constraints are to reverse the activation ordering in Experiments 2, 4, 6, and 12; raise selective terminal influence in Experiments 11 and 13; and correct Experiment 23’s crossover sign while preserving the low terminal rate in Experiment 8 and the existing fits in Experiments 14 and 24.

**Outcome of this advice:** iter 3 candidate loss=0.3075 -> iter 4 candidate loss=0.3148 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed parallel competitive-encoding family, but the fixed novelty-budget revision was rejected by the gate and worsened aggregate loss from the accepted base’s 0.2935 to 0.3148. It does not solve the central activation-ranking failures: Experiment 2 remains near chance instead of strongly below chance (0.492 versus 0.194); Experiments 6 and 12 still favor the larger/early coalition far too strongly (0.625 versus 0.307 and 0.653 versus 0.171); and terminal influence remains much too weak in Experiments 4, 10, 11, and 13. Experiment 23 still has the wrong crossover sign (-0.071 versus 0.418). Moreover, the budgeted novelty edit creates notable regressions relative to the accepted base in Experiments 12, 20, 22, and 24: the validity-drop effect in Experiment 22 nearly disappears (0.015 versus 0.175), Experiment 20 reverses to the wrong sign, and the previously exact Experiment 24 fit falls to 0.335 versus 0.419. Experiment 1 also remains too weak, and Experiment 5 retains an excessive sequence-geometry contrast. Good fits in Experiments 7, 15, 17, and 18 do not compensate for these systematic failures. The generally low between-subject variances show that the problem is the mean computation rather than excessive heterogeneity.
Rationale: The previous recommendation to replace additive novelty with a globally fixed, softmax-allocated novelty budget was rejected, so do not repeat or merely strengthen that intervention. Build on the unchanged iteration-1 base and try a different in-family adjustment: revise the ordering and normalization of coalition pooling versus divisive competition. The current model applies all-to-all cue-level suppression before pooling, so cues supporting the same option suppress one another according to total display activation; only afterward are their already-distorted contributions saturated. This can preserve numerical-coalition dominance while simultaneously crushing an isolated challenger, exactly the pattern seen in Experiments 2, 6, 11–13. Instead, first pool each option’s concurrently encoded cue activations with saturating returns, then conduct divisive competition between the two coalition-level activations. Same-option cues should contribute diminishing returns without acting as suppressive competitors, while cross-option competition should determine which coalition survives. Jointly tune a somewhat stronger saturation range and coalition-level normalization temperature, keeping the accepted base’s spatial weighting, full-schedule reliability calibration, local novelty rule, lapse, and no-history architecture initially fixed. This is a normalization correction within the prescribed mechanism, not a new strategy family. It should be tested for lowering the oversized-coalition choices in Experiments 2, 6, and 12 and raising selective challengers in Experiments 4, 11, and 13. Constrain the fit explicitly against increasing ordinary terminal adherence in Experiment 8 and preserve the accepted base’s Experiment 14 and 24 results. Only after the coalition ordering is corrected should reliability gain be narrowly retuned to raise Experiment 1; another broad endpoint or novelty recalibration would repeat changes already rejected by the gate.

**Outcome of this advice:** iter 4 candidate loss=0.3148 -> iter 5 candidate loss=0.2684 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed parallel competitive-encoding family, and the accepted pooling-order correction is a genuine improvement: aggregate loss falls from 0.2935 to 0.2684. Saturating within options before coalition-level competition substantially improves Experiments 5, 6, 11–13, 19, and 22 while preserving the null results in Experiments 7 and 17 and the strong fit in Experiment 24. In particular, Experiment 22 is now essentially exact (0.174 versus 0.175), Experiment 5 is close to null, and the oversized-coalition errors in Experiments 6 and 12 are reduced. Nevertheless, the model is not yet experiment-invariant. Experiment 2 remains near chance rather than showing the observed strong reversal of the highest-validity cue (0.480 versus 0.194). Experiments 4, 11, and 13 still severely underpredict terminal influence (-0.035 versus 0.665, 0.495 versus 0.843, and 0.434 versus 0.881), while Experiment 8 overpredicts ordinary terminal adherence (0.456 versus 0.172). Experiment 10 is also too low (0.631 versus 0.875), and Experiment 12 still overpredicts majority choice (0.463 versus 0.171). The model misses Experiment 20's validity-dependent contrast and Experiment 23's crossover direction, and its previously partial endpoint fit in Experiment 21 collapses almost completely (0.024 versus 0.390). Experiment 1 remains too weak as well. These are systematic activation-ranking errors, not merely noise: candidate between-subject variances are often below the observed variances.
Rationale: The accepted advice about pooling before competition should be retained and extended rather than reversed. The likely remaining defect is that the current exponential saturation is applied to the entire coalition, including a coalition containing only one cue. A sufficiently activated singleton is therefore compressed toward the same ceiling as other singletons, erasing reliability, novelty, and endpoint differences before competition. This explains the near-chance singleton conflicts in Experiment 2, the loss of endpoint asymmetry in Experiment 21, the weak validity contrast in Experiment 20, and likely part of the wrong Experiment 23 crossover. At the same time, repeated coalitions remain influential enough to keep Experiments 6 and 12 too high.

Revise the same coalition mechanism so that diminishing returns apply primarily to repeated same-option support rather than to the strongest cue itself. For example, preserve the strongest activated cue as an unsaturated anchor and saturate only the residual support, such as `coalition = max_activation + sat * (1-exp(-(sum_activation-max_activation)/sat))`, followed by the existing coalition-level divisive competition. A smooth generalized-mean or count-adjusted equivalent would also be appropriate. This is a minor normalization refinement within the accepted parallel coalition family. It should preserve the gains from compressing numerical majorities while allowing a distinctive singleton's absolute activation to survive into competition.

Jointly retune the residual-saturation scale and competition temperature, with only a narrow adjustment of choice precision if needed. Do not repeat the rejected global endpoint, encoded-only reliability normalization, or fixed novelty-budget changes. Explicit constraints should be: retain the improved Experiments 5, 6, 12, 22, and 24; lower Experiment 2 below chance and restore Experiment 20's negative contrast and Experiment 21's endpoint difference; raise selective challenger influence in Experiments 4, 11, and 13; and avoid increasing Experiment 8. If preserving singleton activation does not improve Experiment 23's sign, adjust how the existing reliability-drop signal enters the unsaturated anchor rather than introducing another broad novelty or spatial boost.

**Outcome of this advice:** iter 5 candidate loss=0.2684 -> iter 6 candidate loss=0.3774 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed parallel competitive-encoding family, but the unsaturated-anchor refinement is a major regression and was correctly rejected by the gate: loss rises from the accepted 0.2684 to 0.3774. Preserving the strongest raw activation does help the intended singleton diagnostics—Experiment 20 improves from 0.013 to -0.151 against -0.181, and Experiment 21 rises from 0.024 to 0.198 against 0.390—but these gains are overwhelmed by widespread failures. Experiment 1 collapses from 0.376 to 0.185 against 0.494; Experiment 3 falls from 0.499 to 0.300; Experiment 5 develops a very large spurious sequence contrast, 0.531 against 0.003; and early or majority coalitions become much too dominant in Experiments 6 and 12, reaching 0.830 against 0.307 and 0.713 against 0.171. Selective terminal influence also deteriorates sharply in Experiments 4, 10, 11, and 13, with Experiment 13 falling to 0.205 against 0.881. The previously exact reliability-drop result in Experiment 22 explodes to 0.493 against 0.175, Experiment 23 becomes even more negative (-0.378 against 0.418), and Experiment 24 moves away from its good fit. Experiment 2 remains near chance rather than below chance. Thus the anchor has not merely failed to solve the residual errors; it has destabilized the activation ranking across most displays. The low variances on many of these misses indicate systematic mean-computation errors.
Rationale: The previous recommendation to leave the maximum activation fully unsaturated was rejected decisively, so it should not be repeated. The failure is mechanistically clear: raw cue activation is exponentiated after reliability, boundary, and novelty terms, and the new anchor then passes that potentially very large maximum directly into coalition competition. A single extreme cue can therefore bypass the compression that made iteration 5 successful, while competition temperature further magnifies the distortion. This explains both the useful movement in sparse singleton tests and the catastrophic regressions in coalition and contrast tests.

Retain the accepted iteration-5 ordering—whole-coalition saturating pooling followed by coalition-level divisive competition—and try only a small, bounded retention of within-coalition salience. Rather than an additive unsaturated maximum, use a convex or multiplicative modulation of the saturated coalition by a compressed max-to-total statistic, with a small coefficient tightly concentrated near zero. For example, the coalition could remain `sat*(1-exp(-support/sat))` and receive a bounded factor based on `max_activation/support` and a compressed value such as `log1p(max_activation)`; alternatively, blend only 10–25% toward an anchored form. This intermediate construction is warranted because full saturation produced the best aggregate fit, whereas a complete anchor improved Experiments 20 and 21 but damaged nearly everything else. Also compress the raw activation entering any max statistic with `log1p` or a sublinear power so exponentiated novelty cannot evade capacity limits. Do not broadly alter spatial tilt, reliability normalization, or novelty allocation, since those earlier packages were rejected.

The immediate acceptance constraints should be preservation of iteration 5's near-null Experiment 5 and near-exact Experiment 22 and 24 fits, with no renewed increases in Experiments 6 and 12. Any bounded salience-retention term is useful only if it modestly improves Experiments 20 and 21 without degrading those anchors. Experiment 23's sign and the large selective-terminal misses in Experiments 4, 11, and 13 should not be attacked by increasing the bypass coefficient; the current results show that stronger max preservation moves those targets in the wrong direction.

**Outcome of this advice:** iter 6 candidate loss=0.3774 -> iter 7 candidate loss=0.2723 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed parallel competitive-encoding family, but it was rejected by the gate: loss 0.2723 is slightly worse than the accepted iteration-5 loss of 0.2684. The bounded salience-retention term is much safer than the rejected unsaturated anchor, yet it provides almost no useful leverage. Experiment 20 moves only from 0.0125 to -0.0037 against -0.1813, and Experiment 21 only from 0.0242 to 0.0392 against 0.3900. Those small gains are outweighed by regressions in Experiments 5, 8, 12, 14, 18, 22, and 24. The main empirical failures therefore remain: Experiment 2 is near chance rather than strongly below chance (0.4888 versus 0.1938); ordinary terminal adherence is excessive in Experiment 8 (0.4748 versus 0.1721); terminal or late-coalition influence is far too weak in Experiments 4, 10, 11, and 13; the early or numerical coalition remains too strong in Experiments 6 and 12; and Experiments 21 and 23 still miss the endpoint asymmetry and reliability crossover. Experiment 1 also remains too weak. Conversely, the model continues to fit Experiments 5, 7, 15–19, 22, and 24 reasonably well, supporting the accepted pooling-before-competition architecture and the weak-history assumption. The persistent low candidate variances in several badly missed experiments indicate systematic activation-ranking errors rather than insufficient simulation noise.
Rationale: Because the bounded max-retention proposal was rejected, do not repeat it with a larger coefficient or another peak-preservation formula. Return to the unchanged accepted iteration-5 computation and remove or fix salience_retention at zero. The next adjustment should target a different omission already present in the arbiter’s prescribed mechanism: coalition coherence currently has no explicit role. The model pools all same-option activations solely through their sum, so a contiguous, mutually reinforcing run and a spatially fragmented set with the same activation total exert essentially the same competitive force. It therefore cannot reliably let a coherent coalition suppress an ordinary singleton, which helps explain excessive terminal choice in Experiment 8, while still allowing a surprising challenger or terminal coalition to survive in Experiments 2, 10, 12, and 13.

Add a small bounded coalition-coherence term computed in parallel from physical adjacency, such as the proportion of same-direction adjacent discriminating pairs or a compressed largest-run statistic. Use coherence primarily to modulate the inhibition a coalition exerts during coalition-level divisive competition, rather than adding unrestricted activation before saturation. A target cue or coalition with high existing contrastive-surprise activation should be partially exempt from this extra coherence-based suppression; an ordinary singleton should not be. This preserves whole-coalition saturation and avoids another raw-activation bypass. Keep the accepted spatial tilt, reliability normalization, and novelty construction fixed initially, since broad changes to those components were rejected in earlier iterations. If coherence suppression successfully lowers Experiment 8 and preserves Experiments 5, 22, and 24, only then consider a very narrow increase in the existing novelty gain so sparse opposing dyads in Experiment 2 can move below chance without restoring broad terminal overprediction.

Acceptance constraints should be explicit: improve Experiment 8 without worsening the already low terminal influence in Experiments 10 and 13; lower Experiments 2, 6, and 12; preserve the near-null Experiment 5 and the strong Experiment 22 and 24 fits. Experiment 11 and the Experiment 23 crossover may remain difficult, but they should not be attacked through another global endpoint boost or max-retention term, both of which have already failed the gate.

**Outcome of this advice:** iter 7 candidate loss=0.2723 -> iter 8 candidate loss=0.2727 -> the gate marked it REJECTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed parallel competitive-encoding family, but the coherence-gated revision was rejected by the accept gate and is slightly worse than the accepted iteration-5 base (loss 0.2727 versus 0.2684). More importantly, it moves several intended targets in the wrong direction. Relative to iteration 5, ordinary terminal adherence in Experiment 8 increases from 0.4562 to 0.4725 rather than falling toward 0.1721; the majority error in Experiment 12 increases from 0.4625 to 0.5100; and selective terminal influence declines in Experiments 11 and 13, from 0.4954 to 0.4679 and from 0.4335 to 0.4220. Experiment 3 also deteriorates. The edit modestly improves Experiment 1 and preserves good fits in Experiments 5, 7, 17, 19, 22, and 24, but it leaves the principal failures essentially intact: Experiment 2 remains near chance rather than below chance (0.4894 versus 0.1938), Experiment 4 has the wrong terminal association (-0.0312 versus 0.6650), Experiments 10, 11, and 13 substantially underpredict terminal influence, and Experiments 6 and 12 overpredict early or numerical-coalition choice. The endpoint asymmetry in Experiment 21 is virtually absent (0.0042 versus 0.3900), and the Experiment 23 crossover remains much too small (0.0333 versus 0.4183). Low candidate variances in many of these experiments indicate systematic activation-ranking errors rather than insufficient subject heterogeneity.
Rationale: Because the current candidate was rejected, the next proposal should build on the unchanged accepted iteration-5 model. The prior push toward adjacency-based coherence inhibition did not help: in Experiment 8, both the large prefix and the terminal suffix can be spatially coherent, so the statistic does not selectively identify the ordinary singleton that should be suppressed. Do not increase coherence_inhibition, alter the coherence statistic, or retain this extra pathway.

Try a different, narrowly bounded correction within the same divisive-competition mechanism. Whole-coalition saturation appears to erase much of a distinctive challenger's novelty before option-level competition, while the rejected unsaturated-anchor variants allowed raw exponentiated activation to bypass saturation. Preserve whole-coalition saturation, but let the existing contrast signal provide a small bounded release from cross-coalition inhibition after saturation. For example, multiply the target coalition's divisive denominator by a factor such as `1 - rho * sigmoid((novelty_target-threshold)/temperature)`, with `rho` tightly constrained and well below one. This should not add activation, preserve a raw maximum, or allocate a fixed novelty budget. It instead implements the arbiter-prescribed idea that a surprising opposing cue can survive divisive competition. Reduce its pre-saturation novelty contribution correspondingly if necessary to avoid double counting.

This post-saturation protection offers targeted leverage that the coherence edit lacked: sharp-drop challengers can lower highest-validity adherence in Experiment 2, produce the negative validity contrast in Experiment 20, and raise terminal choice in Experiments 4, 11, and 13, while smooth ordinary terminal cues in Experiment 8 should receive little protection. Tune the protection threshold and strength jointly against Experiment 22, which is already nearly exact and must not be inflated. Keep spatial tilt, boundary gain, full-schedule reliability normalization, saturation, and lapse fixed initially; broad endpoint recalibration, max retention, novelty budgeting, and coherence modulation have all already failed the gate. Acceptance should require improvement over iteration 5 in Experiments 2, 8, 11–13, and 20 without losing the near-null Experiment 5 or the strong Experiment 22 and 24 fits. Experiment 21 should not be pursued with another global endpoint boost unless this more selective competition correction first improves aggregate loss.

**Outcome of this advice:** iter 8 candidate loss=0.2727 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Parallel Contrastive Coalition Field with Novelty-Protected Competition. All discriminating cues are encoded concurrently into a capacity-limited comparison field rather than inspected in a mandatory validity or spatial sequence. Cue activation combines instructed reliability, U-shaped boundary distinctiveness, an estimable spatial tilt, and local contrast. Same-option activations pool into saturating coalitions before divisive competition. A coalition containing a sharply contrastive challenger receives a small, bounded release from cross-coalition inhibition after saturation, allowing prediction-error signals to survive competition without letting raw cue activation bypass capacity limits. Choice follows the normalized coalition difference with load-dependent precision loss and a task-invariant lapse. The model has no trial-history learning, strategy mixture, fixed scan order, or sequential commitment state.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Parallel Contrastive Coalition Field expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    reliability_gain = float(parameters[\"reliability_gain\"])\n    boundary_gain = float(parameters[\"boundary_gain\"])\n    spatial_tilt = float(parameters[\"spatial_tilt\"])\n    coalition_saturation = float(parameters[\"coalition_saturation\"])\n    novelty_gain = float(parameters[\"novelty_gain\"])\n    novelty_release = float(parameters[\"novelty_release\"])\n    protection_threshold = float(parameters[\"protection_threshold\"])\n    capacity_loss = float(parameters[\"capacity_loss\"])\n    competition_temperature = float(parameters[\"competition_temperature\"])\n    choice_precision = float(parameters[\"choice_precision\"])\n    lapse_rate = float(parameters[\"lapse_rate\"])\n\n    diff = stim[0] - stim[1]\n    discriminating = np.flatnonzero(diff != 0.0)\n    if discriminating.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    directions = np.sign(diff[discriminating]).astype(float)\n    m = int(discriminating.size)\n\n    # Reliability is represented on the normative log-odds scale. Centering\n    # preserves within-display reliability contrasts while making the shared\n    # gain portable across experiments with different validity schedules.\n    clipped_v = np.clip(validities, 0.5001, 0.999)\n    reliability = np.log(clipped_v / (1.0 - clipped_v))\n    rel_mean = float(np.mean(reliability))\n    rel_sd = float(np.std(reliability))\n    if rel_sd < 1e-8:\n        rel_z = np.zeros(n_features, dtype=float)\n    else:\n        rel_z = np.clip((reliability - rel_mean) / rel_sd, -2.5, 2.5)\n\n    raw_activation = np.zeros(m, dtype=float)\n    novelty_signals = np.zeros(m, dtype=float)\n\n    # These activations are computed from the complete display. References to\n    # left and right neighbors describe simultaneous spatial relations, not an\n    # update order or a serial scan.\n    for k, j_cell in enumerate(discriminating):\n        j = int(j_cell)\n        direction = float(directions[k])\n        position = float(j) / float(max(1, n_features - 1))\n        signed_position = 2.0 * position - 1.0\n\n        # Both endpoints can be distinctive. The separately estimated tilt\n        # permits endpoint asymmetry without assuming universal right recency.\n        boundary = signed_position * signed_position\n        spatial_log_activation = (\n            boundary_gain * boundary + spatial_tilt * signed_position\n        )\n\n        surprise = 0.0\n        symmetric_contrast = 0.0\n\n        if k > 0:\n            left_j = int(discriminating[k - 1])\n            left_opposes = float(directions[k - 1] != direction)\n            if left_opposes > 0.0:\n                # A reliability drop at an opposing spatial boundary is a\n                # prediction-error signal, not evidence that cues were scanned.\n                drop = max(0.0, float(validities[left_j] - validities[j]))\n                surprise += min(1.25, drop / 0.40)\n                symmetric_contrast += min(\n                    1.0, abs(float(validities[left_j] - validities[j])) / 0.40\n                )\n\n        if k + 1 < m:\n            right_j = int(discriminating[k + 1])\n            right_opposes = float(directions[k + 1] != direction)\n            if right_opposes > 0.0:\n                symmetric_contrast += min(\n                    1.0, abs(float(validities[right_j] - validities[j])) / 0.40\n                )\n\n        # Most contrast is a sharp-drop novelty response. A smaller symmetric\n        # term captures ordinary local pop-out at either side of a boundary.\n        novelty_signal = surprise + 0.18 * symmetric_contrast\n        novelty_signals[k] = novelty_signal\n        log_activation = (\n            reliability_gain * rel_z[j]\n            + spatial_log_activation\n            + novelty_gain * novelty_signal\n        )\n        raw_activation[k] = float(np.exp(np.clip(log_activation, -8.0, 8.0)))\n\n    # Concurrent same-option cues pool before competition. Thus they exhibit\n    # diminishing returns but do not consume capacity by suppressing one\n    # another at the cue level.\n    support_a = float(np.sum(raw_activation[directions > 0.0]))\n    support_b = float(np.sum(raw_activation[directions < 0.0]))\n\n    sat = max(coalition_saturation, 1e-8)\n    coalition_a = sat * (1.0 - np.exp(-support_a / sat))\n    coalition_b = sat * (1.0 - np.exp(-support_b / sat))\n\n    # Sharp contrast receives only a bounded post-saturation release from\n    # cross-coalition inhibition. This preserves whole-coalition saturation:\n    # novelty cannot add activation or pass an unsaturated cue maximum into\n    # competition. Smooth, ordinary boundaries remain effectively unprotected.\n    novelty_a = float(np.max(novelty_signals[directions > 0.0])) if np.any(directions > 0.0) else 0.0\n    novelty_b = float(np.max(novelty_signals[directions < 0.0])) if np.any(directions < 0.0) else 0.0\n    protection_a = 1.0 / (\n        1.0 + np.exp(-np.clip((novelty_a - protection_threshold) / 0.12, -60.0, 60.0))\n    )\n    protection_b = 1.0 / (\n        1.0 + np.exp(-np.clip((novelty_b - protection_threshold) / 0.12, -60.0, 60.0))\n    )\n    release_a = float(np.clip(1.0 - novelty_release * protection_a, 0.80, 1.0))\n    release_b = float(np.clip(1.0 - novelty_release * protection_b, 0.80, 1.0))\n\n    # Capacity competition occurs between option-level coalitions. The mild\n    # temperature sharpens genuine coalition differences after repeated cues\n    # have already been compressed by saturation.\n    competitive_a = coalition_a / ((1.0 + capacity_loss * coalition_b) * release_a)\n    competitive_b = coalition_b / ((1.0 + capacity_loss * coalition_a) * release_b)\n    competitive_a = float(np.power(max(competitive_a, 0.0), competition_temperature))\n    competitive_b = float(np.power(max(competitive_b, 0.0), competition_temperature))\n\n    # Divisive comparison expresses evidence relative to the total activated\n    # field, preventing large displays from creating unbounded decision values.\n    coalition_total = competitive_a + competitive_b\n    if coalition_total <= 1e-12 or not np.isfinite(coalition_total):\n        normalized_difference = 0.0\n    else:\n        normalized_difference = (competitive_a - competitive_b) / coalition_total\n\n    # Display and evidence load modestly lower readout precision. Saturating\n    # pooling and this precision loss can produce a small decline even for long\n    # unanimous displays, without a discrete fatigue threshold.\n    display_load = float(max(0, n_features - 1))\n    evidence_load = float(max(0, m - 2))\n    precision = choice_precision / (\n        1.0\n        + 0.55 * capacity_loss * display_load\n        + 0.35 * capacity_loss * evidence_load\n    )\n    precision = float(np.clip(precision, 0.20, 8.0))\n\n    logit_a = float(np.clip(precision * normalized_difference, -60.0, 60.0))\n    p_a = 1.0 / (1.0 + np.exp(-logit_a))\n    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n\n    probs = (\n        (1.0 - lapse_rate) * core\n        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)\n    )\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "reliability_gain": "[0.25, 0.50]",
    "boundary_gain": "[0.25, 0.55]",
    "spatial_tilt": "[-0.90, -0.55]",
    "coalition_saturation": "[0.55, 0.95]",
    "novelty_gain": "[1.50, 2.40]",
    "novelty_release": "[0.04, 0.10]",
    "protection_threshold": "[1.10, 1.30]",
    "capacity_loss": "[0.08, 0.18]",
    "competition_temperature": "[1.05, 1.35]",
    "choice_precision": "[3.00, 4.30]",
    "lapse_rate": "[0.05, 0.12]"
  },
  "rationale": "This is a narrowly bounded edit to the accepted iteration-5 model. Whole-coalition exponential saturation, pooling-before-competition, spatial weighting, reliability normalization, pre-saturation novelty, load loss, and lapse are unchanged. The only mechanistic addition stores each cue's already-computed novelty signal and uses the largest signal within each option to provide a small post-saturation release from that coalition's divisive denominator. The high threshold and narrow 4–10% release range make the pathway nearly inactive for ordinary boundaries, including smooth terminal opposition, while preserving sharply contrastive challengers during competition. Unlike the rejected anchor edits, it neither adds raw activation nor allows an exponentiated cue maximum to bypass saturation; unlike the rejected novelty budget, it does not redistribute salience among cues. This targets below-chance highest-validity adherence and validity-dependent challenger effects while minimizing disruption to the accepted fits in Experiments 5, 22, and 24."
}
```

## Usage

```json
{
  "input_tokens": 55890,
  "output_tokens": 3773
}
```
