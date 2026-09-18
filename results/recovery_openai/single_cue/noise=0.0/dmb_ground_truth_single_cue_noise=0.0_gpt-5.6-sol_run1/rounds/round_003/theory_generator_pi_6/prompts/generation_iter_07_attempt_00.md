# generation_iter_07_attempt_00

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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace pi_5 with a stronger competitor to pi_3: a validity-ranked, capacity-limited evidence-buffer theory. Participants order inspection by instructed validity, not physical display position. Each discriminating cue deposits signed, validity-scaled evidence into a small working-memory buffer; later entries gradually displace or interfere with earlier entries, but do not categorically overwrite them and are not compressed into directional runs. Choice is based on the noisy sum of the evidence still accessible at response time. A limited buffer produces strong late-reason control on short conflicts, tally sensitivity when several reasons remain accessible, and contraction toward chance as longer sequences exceed capacity. It also predicts little intrinsic effect of switch density or run boundaries, matching Experiments 1 and 7. Include stable subject-level variation in buffer capacity, displacement rate, validity sensitivity, and response gain to address the much larger observed variances in Experiments 2 and 6. The theory should predict very strong validity-order recency in Experiment 2, strong terminal influence for four-reason conflicts in Experiment 6, but more moderate pooled final-cue agreement across the longer conditions of Experiment 5. This differs from pi_3 by replacing discrete provisional-choice overwrite with competition among multiple retained reasons, and from pi_5 by rejecting displayed-order scanning, run chunking, suffix bonuses, and switch-specific interference.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.871)

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


---

### `pi_4` (overall score: 0.504)

**Description**
Interference-Limited Retrospective Reason Sampling with Length-Calibrated Accessibility. After viewing all cues, a decision maker retrospectively retrieves one or, subject to stable working-memory capacity, occasionally two discriminating reasons. In two-reason conflicts, retrieval has a strong saturated terminal-recency floor, producing reliable last-reason capture without importing subject heterogeneity from longer conflicts. Around four reasons, stable variation in recency remains expressed. When conflicts exceed four reasons, nonlinear source interference flattens recency while accessibility approaches a validity-biased rather than uniform asymptote. Choice follows the signed, validity-sensitive evidence supplied by the retrieved reasons, with a small response lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Retrospective reason sampling expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = int(parameters["memory_capacity"])
    recency = float(parameters["recency_bias"])
    validity_access = float(parameters["validity_access"])
    interference = float(parameters["interference_rate"])
    continuation = float(parameters["second_reason_rate"])
    evidence_floor = float(parameters["evidence_floor"])
    evidence_gain = float(parameters["evidence_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability is represented as instructed log-odds. Clipping only
    # prevents infinite evidence for a validity communicated as exactly 1.
    v = np.clip(validities[cue_indices], 0.500001, 0.999)
    reliability = np.log(v / (1.0 - v))
    directions = np.sign(diff[cue_indices])  # +1 favors A, -1 favors B

    # Cues are displayed from highest to lowest validity. Ordinal position
    # among discriminating cues therefore indexes retrospective recency.
    ordinal_recency = np.arange(m, dtype=float)

    # Interference is negligible through four reasons but grows cubically
    # thereafter, rapidly flattening accessibility in long conflicts.
    overload = float(max(0, m - 4))
    effective_recency = recency / (1.0 + interference * overload * overload * overload)

    # A two-reason conflict is represented as a compact terminal episode:
    # its recency gradient is strong and saturates across subjects. Stable
    # recency differences remain fully expressed in four-reason conflicts.
    if m == 2:
        effective_recency = 3.4 + 0.15 * np.tanh(recency - 2.0)

    # Under overload, interference removes temporal-source information but
    # does not make all reasons equally accessible. Retrieval instead tends
    # toward a modestly validity-biased asymptote.
    effective_validity_access = (
        validity_access + 0.55 * overload / (1.0 + overload)
    )

    log_access = (
        effective_recency * ordinal_recency
        + effective_validity_access * reliability
    )
    log_access -= np.max(log_access)
    access = np.exp(log_access)
    first_probs = access / access.sum()

    def probability_a(retrieved_positions):
        # Every retrieved reason supplies substantial directional evidence;
        # instructed validity additionally scales that evidence and resolves
        # conflicts when more than one reason is retrieved.
        positions = list(retrieved_positions)
        evidence = float(np.sum(
            directions[positions]
            * (evidence_floor + reliability[positions])
        ))
        x = float(np.clip(evidence_gain * evidence, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Exact marginalization over the first retrospectively retrieved reason.
    p_a_one = 0.0
    for i in range(m):
        p_a_one += float(first_probs[i]) * probability_a((i,))

    # Capacity is an upper bound: even capacity-two subjects often terminate
    # retrieval after one sufficient reason. If retrieval continues, the
    # second reason is sampled without replacement from the same accessibility
    # distribution and both reasons are integrated.
    if capacity >= 2 and m >= 2 and continuation > 0.0:
        p_a_two = 0.0
        for i in range(m):
            remaining_mass = float(access.sum() - access[i])
            if remaining_mass <= 0.0:
                continue
            for j in range(m):
                if j == i:
                    continue
                p_ordered = float(first_probs[i]) * float(access[j] / remaining_mass)
                p_a_two += p_ordered * probability_a((i, j))
        p_a_core = (1.0 - continuation) * p_a_one + continuation * p_a_two
    else:
        p_a_core = p_a_one

    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- memory_capacity: {1, 2}
- recency_bias: [0.2, 4.6]
- validity_access: [0.05, 0.35]
- interference_rate: [0.6, 1.4]
- second_reason_rate: [0.02, 0.12]
- evidence_floor: [0.35, 0.75]
- evidence_gain: [3.0, 7.0]
- lapse_rate: [0.03, 0.17]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.5279 (var=0.0111)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.2060 (var=0.0019)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.5346 (var=0.0080)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6104 (var=0.0504)
- Experiment 5: real=0.0033 (var=0.0049) vs this=0.4617 (var=0.0125)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.7804 (var=0.0054)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.0384)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.4802 (var=0.0042)


---

### `pi_5` (overall score: 0.419)

**Description**
Saturated Directional-Run Chunking with Boundary Interference. People scan discriminating cues in their instructed order and compress consecutive cues favoring the same option into a single directional chunk. Chunk confidence rises rapidly with run length but saturates, so two agreeing terminal cues can be nearly as effective as a much longer terminal run. At each directional boundary, the newly completed chunk suppresses the accumulated representation of earlier chunks. A coherent terminal run receives an additional replacement advantage, causing stimuli with the same strong suffix to converge despite differences in prefix segmentation. Short four-reason sequences preserve run structure at high temporal resolution and express terminal chunks with a heterogeneous gain, producing strong but individually variable terminal control. In longer sequences, switch-dense patterns suffer source interference, while long low-switch conflicts undergo a separate modest loss of response precision without recovering early high-validity cues. Instructed validity only modestly scales chunk encoding and cannot become the retrieval asymptote. Choices arise from recency-weighted competition among run summaries plus a small response lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Directional-run chunking expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    saturation = float(parameters["run_saturation"])
    suppression = float(parameters["boundary_suppression"])
    recency_decay = float(parameters["chunk_recency_decay"])
    short_boost = float(parameters["short_resolution_boost"])
    suffix_replacement = float(parameters["suffix_replacement"])
    four_gain = float(parameters["four_reason_expression_gain"])
    switch_interference = float(parameters["switch_interference"])
    length_interference = float(parameters["length_interference"])
    long_low_switch_loss = float(parameters["long_low_switch_loss"])
    validity_scale = float(parameters["validity_encoding_scale"])
    choice_gain = float(parameters["choice_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[cue_indices]).astype(float)
    cue_validities = np.clip(validities[cue_indices], 0.5, 1.0)

    # Validity changes encoding strength only modestly. Every discriminating
    # cue has a substantial baseline representation, preventing overload from
    # turning retrieval into a validity-biased early-cue process.
    encoded_strengths = 1.0 + validity_scale * (2.0 * cue_validities - 1.0)

    # Compress contiguous cues with a common direction into run summaries.
    runs = []
    start = 0
    for i in range(1, m + 1):
        if i == m or directions[i] != directions[start]:
            direction = float(directions[start])
            encoding_mass = float(np.sum(encoded_strengths[start:i]))
            # Repetition increases chunk confidence, but confidence rapidly
            # saturates rather than growing as an unbounded cue tally.
            confidence = 1.0 - np.exp(-saturation * encoding_mass)
            runs.append((direction, confidence, i - start))
            start = i

    n_switches = max(0, len(runs) - 1)

    # Compete run summaries online. A new coherent chunk both enters memory
    # and suppresses prior chunks. Short sequences retain sharper boundary
    # resolution, yielding especially effective terminal replacement.
    directional_memory = 0.0
    represented_mass = 0.0
    for run_number, (direction, confidence, run_length) in enumerate(runs):
        if run_number == 0:
            directional_memory = direction * confidence
            represented_mass = confidence
            continue

        effective_suppression = suppression
        if m <= 4:
            effective_suppression += short_boost
        # A repeated-direction suffix is encoded as a completed terminal
        # decision chunk and therefore more fully replaces prefix summaries.
        if run_number == len(runs) - 1 and run_length >= 2:
            effective_suppression += suffix_replacement
        effective_suppression = float(np.clip(effective_suppression, 0.0, 0.98))

        retention = recency_decay * (1.0 - effective_suppression * confidence)
        retention = float(np.clip(retention, 0.0, 1.0))
        directional_memory = retention * directional_memory + direction * confidence
        represented_mass = retention * represented_mass + confidence

    if represented_mass <= 0.0 or not np.isfinite(represented_mass):
        normalized_direction = 0.0
    else:
        normalized_direction = float(
            np.clip(directional_memory / represented_mass, -1.0, 1.0)
        )

    # Switch-dense sequences lose directional source resolution. Generic
    # length uncertainty remains mild, while long low-switch conflicts have
    # a separate expression loss that attenuates extremity without restoring
    # privileged access to early cues.
    excess_switches = max(0, n_switches - 2)
    low_switch_excess = max(0, m - 5) if n_switches <= 2 else 0
    precision_denominator = (
        1.0
        + switch_interference * float(excess_switches ** 2)
        + length_interference * float(max(0, m - 4))
        + long_low_switch_loss * float(low_switch_excess)
    )
    effective_gain = choice_gain / precision_denominator

    # Four-reason displays preserve unusually high temporal resolution. The
    # broad subject-level gain range captures heterogeneity in expressing the
    # terminal chunk without changing longer conflicts.
    if m == 4:
        effective_gain *= four_gain

    logit_a = float(np.clip(effective_gain * normalized_direction, -60.0, 60.0))
    p_a_core = 1.0 / (1.0 + np.exp(-logit_a))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
- run_saturation: [1.8, 3.2]
- boundary_suppression: [0.35, 0.55]
- chunk_recency_decay: [0.78, 0.94]
- short_resolution_boost: [0.12, 0.68]
- suffix_replacement: [0.08, 0.25]
- four_reason_expression_gain: [0.35, 2.60]
- switch_interference: [1.8, 2.8]
- length_interference: [0.06, 0.12]
- long_low_switch_loss: [0.08, 0.34]
- validity_encoding_scale: [0.02, 0.20]
- choice_gain: [2.6, 3.8]
- lapse_rate: [0.04, 0.14]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.5949 (var=0.0094)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.1969 (var=0.0043)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.5996 (var=0.0042)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6737 (var=0.0181)
- Experiment 5: real=0.0033 (var=0.0049) vs this=0.0071 (var=0.0075)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.3113 (var=0.0065)
- Experiment 7: real=0.0000 (var=0.2016) vs this=1.0000 (var=0.1924)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.7169 (var=0.0030)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.4823 -> ACCEPTED
- iter 2: loss=0.1620 -> ACCEPTED
- iter 3: loss=0.1384 -> ACCEPTED
- iter 4: loss=0.1512 -> REJECTED
- iter 5: loss=0.0828 -> ACCEPTED
- iter 6: loss=0.0780 -> ACCEPTED
- iter 7: loss=0.0705 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.0705 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.3913 (var=0.0209) (Δ vs real -0.1032)
  - iter 2: 0.5850 (var=0.0157) (Δ vs real +0.0906)
  - iter 3: 0.7929 (var=0.0109) (Δ vs real +0.2985)
  - iter 4: 0.8539 (var=0.0060) (Δ vs real +0.3594)
  - iter 5: 0.6342 (var=0.0163) (Δ vs real +0.1397)
  - iter 6: 0.6431 (var=0.0088) (Δ vs real +0.1486)
  - iter 7 (most recent): 0.6542 (var=0.0116) (Δ vs real +0.1597)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)

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
  - iter 1: 0.7944 (var=0.0029) (Δ vs real +0.6006)
  - iter 2: 0.5396 (var=0.0148) (Δ vs real +0.3458)
  - iter 3: 0.2258 (var=0.0091) (Δ vs real +0.0321)
  - iter 4: 0.2410 (var=0.0098) (Δ vs real +0.0473)
  - iter 5: 0.2594 (var=0.0075) (Δ vs real +0.0656)
  - iter 6: 0.2623 (var=0.0080) (Δ vs real +0.0685)
  - iter 7 (most recent): 0.1877 (var=0.0052) (Δ vs real -0.0060)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)

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
  - iter 1: 0.2383 (var=0.0105) (Δ vs real -0.3367)
  - iter 2: 0.7277 (var=0.0053) (Δ vs real +0.1527)
  - iter 3: 0.7596 (var=0.0062) (Δ vs real +0.1846)
  - iter 4: 0.6810 (var=0.0048) (Δ vs real +0.1060)
  - iter 5: 0.6950 (var=0.0043) (Δ vs real +0.1200)
  - iter 6: 0.6892 (var=0.0034) (Δ vs real +0.1142)
  - iter 7 (most recent): 0.6948 (var=0.0044) (Δ vs real +0.1198)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)

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
  - iter 1: -0.4279 (var=0.0146) (Δ vs real -1.0929)
  - iter 2: 0.4500 (var=0.0344) (Δ vs real -0.2150)
  - iter 3: 0.7350 (var=0.0101) (Δ vs real +0.0700)
  - iter 4: 0.6783 (var=0.0134) (Δ vs real +0.0133)
  - iter 5: 0.6825 (var=0.0116) (Δ vs real +0.0175)
  - iter 6: 0.6787 (var=0.0128) (Δ vs real +0.0137)
  - iter 7 (most recent): 0.7442 (var=0.0114) (Δ vs real +0.0792)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)

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
  - iter 1: 0.0488 (var=0.0140) (Δ vs real +0.0454)
  - iter 2: -0.0033 (var=0.0055) (Δ vs real -0.0067)
  - iter 3: 0.0121 (var=0.0096) (Δ vs real +0.0088)
  - iter 4: -0.1229 (var=0.0045) (Δ vs real -0.1262)
  - iter 5: -0.0317 (var=0.0112) (Δ vs real -0.0350)
  - iter 6: -0.0175 (var=0.0089) (Δ vs real -0.0208)
  - iter 7 (most recent): -0.0083 (var=0.0124) (Δ vs real -0.0117)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)

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
  - iter 1: 0.6542 (var=0.0420) (Δ vs real +0.3475)
  - iter 2: 0.3129 (var=0.0086) (Δ vs real +0.0063)
  - iter 3: 0.1692 (var=0.0076) (Δ vs real -0.1375)
  - iter 4: 0.1275 (var=0.0043) (Δ vs real -0.1792)
  - iter 5: 0.2825 (var=0.0078) (Δ vs real -0.0242)
  - iter 6: 0.3196 (var=0.0062) (Δ vs real +0.0129)
  - iter 7 (most recent): 0.2958 (var=0.0064) (Δ vs real -0.0108)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)

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
  - iter 1: 0.0000 (var=0.1600) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2244) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.2500) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.1600) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.1924) (Δ vs real +0.0000)
  - iter 7 (most recent): 0.0000 (var=0.1600) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)

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
  - iter 1: 0.4623 (var=0.0033) (Δ vs real +0.2902)
  - iter 2: 0.2523 (var=0.0082) (Δ vs real +0.0802)
  - iter 3: 0.1142 (var=0.0027) (Δ vs real -0.0579)
  - iter 4: 0.1065 (var=0.0018) (Δ vs real -0.0656)
  - iter 5: 0.2010 (var=0.0053) (Δ vs real +0.0290)
  - iter 6: 0.2000 (var=0.0044) (Δ vs real +0.0279)
  - iter 7 (most recent): 0.1785 (var=0.0036) (Δ vs real +0.0065)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Validity-Ordered Competitive Evidence Buffer. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces. Readout uncertainty grows quadratically with capacity overload and receives an additional count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, and response gain generate heterogeneous recency and integration behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive evidence buffer expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["buffer_capacity"])
    displacement = float(parameters["displacement_rate"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    response_gain = float(parameters["response_gain"])
    overflow_noise = float(parameters["overflow_noise"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]

    # Inspection follows instructed validity, independently of physical
    # feature position. Stable sorting only resolves exact validity ties.
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Convert instructed reliability into evidence strength. Normalization
    # removes arbitrary differences in the overall validity scale between
    # experiments while preserving all within-experiment validity ratios.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    positive_reliability = reliability[reliability > 0.0]
    reliability_scale = float(np.mean(positive_reliability))
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0

    normalized_reliability = np.clip(reliability / reliability_scale, 1e-4, 20.0)
    cue_strength = np.power(normalized_reliability, validity_sensitivity)
    cue_strength = np.clip(cue_strength, 1e-6, 30.0)

    # The buffer contains separate reason traces. A new discriminating reason
    # interferes continuously with every old trace. Displacement is present
    # even below capacity and increases smoothly under overload; there is no
    # categorical slot deletion and no compression into directional runs.
    traces = []
    for j in discriminating:
        current_load = float(len(traces))
        overload_before_entry = max(0.0, current_load + 1.0 - capacity)
        load_factor = 1.0 + overload_before_entry / max(capacity, 1e-6)
        retention = np.exp(-displacement * load_factor / max(capacity, 1e-6))
        retention = float(np.clip(retention, 0.0, 1.0))

        if traces:
            traces = [retention * trace for trace in traces]

        direction = float(np.sign(diff[j]))  # positive favors A
        traces.append(direction * float(cue_strength[j]))

    accessible_evidence = float(np.sum(traces))

    # Preserve the accepted quadratic capacity-overload uncertainty through
    # six reasons. Beginning explicitly at seven discriminating reasons, add
    # a modest count-based attenuation that is not smeared by capacity.
    overflow = max(0.0, float(m) - capacity)
    long_sequence_excess = max(0.0, float(m) - 6.0)
    convex_multiplier = 1.0 + 0.45 * long_sequence_excess
    readout_sd = np.sqrt(
        1.0 + overflow_noise * overflow * overflow * convex_multiplier
    )
    decision_variable = response_gain * accessible_evidence / readout_sd
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a_core = 1.0 / (1.0 + np.exp(-decision_variable))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
- buffer_capacity: [3.0, 3.8]
- displacement_rate: [2.1, 2.9]
- validity_sensitivity: [0.0, 0.08]
- response_gain: [3.0, 6.5]
- overflow_noise: [0.75, 1.50]
- lapse_rate: [0.03, 0.15]

`rationale`: This makes only the two targeted edits recommended for the accepted iteration-6 base. First, validity-sensitivity heterogeneity is narrowed from [0, 0.18] to [0, 0.08], reducing the residual advantage of the highest-validity cue in Experiment 2 without changing inspection order or displacement. Second, the capacity-smeared max(1, overflow/3) term is replaced by a count-based multiplier that is exactly one through six reasons and rises moderately from seven reasons onward. This protects the successful six-reason anti-terminal result in Experiment 8 while selectively contracting the longer conditions responsible for excessive tally modulation in Experiment 1 and terminal agreement in Experiment 3. The coefficient is conservative because additional contraction in the seven- and eight-reason trials can move Experiment 6 toward chance. All accepted trace-retention dynamics, capacity and displacement ranges, and the absence of run, suffix, switch-density, or display-position mechanisms are preserved.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed validity-ordered competitive-buffer family, and it correctly produces the null arrangement effects in Experiments 5 and 7. However, its central evidence-weighting calibration is badly wrong. It remains strongly controlled by early, high-validity cues: highest-validity agreement is 0.794 versus 0.194 in Experiment 2, terminal agreement is 0.238 versus 0.575 in Experiment 3, the terminal association is actually negative in Experiment 4 (-0.428 versus 0.665), and early-run choice is 0.654 versus 0.307 in Experiment 6. Experiment 1 also shows too little tally modulation (0.391 versus 0.494). These errors jointly indicate that validity weighting overwhelms both graded displacement and accumulation of multiple lower-ranked reasons. Experiment 8 supplies an important constraint: terminal agreement is too high there (0.462 versus 0.172), so the solution should not be a near-overwrite recency rule. Instead, several concordant reasons must remain able to defeat one terminal reason. Variance is also misallocated: Experiment 6 is far too heterogeneous (0.042 versus 0.005), while Experiment 4 is much too homogeneous (0.015 versus 0.062).
Rationale: Retain the prescribed competitive evidence buffer, but recalibrate the balance between validity and multiplicity. The current validity_sensitivity range [1.0, 2.6] is likely too steep; reduce it substantially so that one early cue does not dominate several later reasons. This should simultaneously increase tally sensitivity in Experiment 1, permit the later cue to win two-reason conflicts in Experiment 2, produce terminal control in Experiments 3 and 4, reduce early-run choice in Experiment 6, and let the five-versus-one majorities in Experiment 8 remain anti-terminal. If necessary, combine this with only a moderate increase in age-based displacement rather than categorical or near-categorical overwrite. Capacity heterogeneity should also be narrowed or moved so that all 7–8-reason trials are consistently overloaded; the present [3, 6] range appears to place subjects on opposite sides of the overload boundary and inflates Experiment 6 variance. Short-conflict heterogeneity can instead come from response gain, displacement, and validity sensitivity. Preserve separate traces, smooth displacement, validity-ranked inspection, and the absence of switch/run bonuses—the good fits in Experiments 5 and 7 show that those structural commitments are working.

**Outcome of this advice:** iter 1 candidate loss=0.4823 -> iter 2 candidate loss=0.1620 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted recalibration is a major improvement and remains faithful to the prescribed validity-ordered competitive-buffer family. It now fits Experiments 5, 6, and 7 very well, and Experiment 8 is much closer. Nevertheless, it is not yet experiment-invariant. The largest failure remains Experiment 2: highest-validity agreement is 0.540 rather than 0.194, so two-reason conflicts still give far too much control to the early, more valid cue. Short-conflict recency is also too weak in Experiment 4 (terminal association 0.450 versus 0.665), while terminal control is already too strong in Experiment 3 (0.728 versus 0.575). Experiment 8 remains somewhat too terminal-driven (0.252 versus 0.172), and Experiment 1 now modestly overshoots tally modulation (0.585 versus 0.494). Thus, simply increasing displacement again would be counterproductive: it could help Experiments 2 and 4 but would worsen Experiments 3 and 8. Variance is also misplaced, with excessive heterogeneity in Experiment 2 (0.0148 versus 0.0049) but insufficient heterogeneity in Experiment 4 (0.0344 versus 0.0621).
Rationale: The previous joint reduction in validity sensitivity and increase in displacement was accepted and should be built upon, but the remaining pattern now calls for disentangling those knobs rather than repeating a general recency increase. Flatten validity weighting further—preferably with a narrow range near zero—so the early cue in Experiment 2 cannot retain a reliability advantage large enough to offset graded recency. At the same time, use an intermediate, slightly lower displacement range than the current [2.4, 3.4], rather than returning fully to the old setting. This should let several concordant traces jointly resist an isolated terminal cue in Experiments 3 and 8 while still allowing an approximately equal-strength later cue to defeat one earlier cue in Experiment 2. A plausible next search region is validity_sensitivity around [0.0, 0.25] and displacement_rate around [2.1, 2.9], with the exact values tuned jointly. Narrowing validity-sensitivity heterogeneity should also reduce the excessive Experiment 2 variance. Once Experiment 2 is pushed near its low agreement floor, response-gain or modest displacement heterogeneity can be widened selectively to recover the larger Experiment 4 variance without greatly increasing Experiment 2 variance. Preserve the successful capacity-limited separate-trace architecture, validity-ranked inspection, and absence of run, switch, suffix, or display-order mechanisms; the fits in Experiments 5–7 show those commitments are working.

**Outcome of this advice:** iter 2 candidate loss=0.1620 -> iter 3 candidate loss=0.1384 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed validity-ordered competitive-buffer family and improves the accepted loss again. The flattened validity weighting successfully fixes the principal Experiment 2 failure: highest-validity agreement is now 0.226 versus 0.194. Experiment 4 is also close at 0.735 versus 0.665, while Experiments 5 and 7 preserve the required null arrangement and switch-density effects. However, the candidate is not yet experiment-invariant. Recency is now systematically excessive once several reasons accumulate: Experiment 3 terminal agreement is 0.760 versus 0.575, Experiment 6 early-run choice is too low at 0.169 versus 0.307, and Experiment 8 terminal agreement has crossed below the target at 0.114 versus 0.172. Experiment 1 also strongly overshoots tally modulation, 0.793 versus 0.494. This coordinated pattern points to overly aggressive compounding displacement in longer sequences, rather than a remaining validity-weighting problem. The current load factor increases displacement after capacity is exceeded, on top of overflow readout noise, so overload suppresses early traces through two mechanisms. Variance remains notably too low in Experiment 4, although the Experiment 6 variance is now reasonably close.
Rationale: Keep the successful near-zero validity sensitivity and do not reverse the accepted recalibration: increasing validity sensitivity would likely reopen the large Experiment 2 error. Instead, adjust a new in-family knob—the overload amplification of trace displacement. Preserve substantial baseline displacement for two-reason conflicts, where Experiment 2 now fits, but weaken or cap the extra load-dependent multiplier once capacity is exceeded. For example, replace `1 + overload/capacity` with `1 + k*overload/capacity` using a small fitted coefficient, or use a saturating overload term. Capacity limitation can remain expressed through graded displacement plus the existing overflow readout noise; no run, switch, suffix, or overwrite mechanism is needed. Less accelerated displacement at loads of roughly 5–8 should restore surviving early evidence in Experiments 3, 6, and 8 and reduce the excessive Experiment 1 contrast, while leaving the critical two-cue recency in Experiment 2 essentially unchanged. It should also modestly lower Experiment 4 from its current overshoot and may expose more subject-level heterogeneity by moving choices away from saturation. Tune this overload slope before changing the established capacity or validity-sensitivity ranges, since the previous narrowing of those ranges was twice accepted by the gate.

**Outcome of this advice:** iter 3 candidate loss=0.1384 -> iter 4 candidate loss=0.1512 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed validity-ranked, capacity-limited competitive-buffer family, but it was rejected by the accept gate and should not replace the iteration-3 base. Weakening overload-dependent displacement did improve Experiment 3 (terminal agreement 0.681 rather than 0.760, toward 0.575) and Experiment 4 (0.678 versus 0.665), but the aggregate fit deteriorated because several other diagnostics moved in the wrong direction. Experiment 1's tally contrast increased further to 0.854 versus 0.494; Experiment 6's early-run choice fell to 0.128 versus 0.307; Experiment 8 remained too low at 0.107 versus 0.172; and the previously correct null contrast in Experiment 5 shifted substantially to -0.123 versus 0.003. Experiment 2 also slipped slightly from 0.226 to 0.241 versus 0.194. Across Experiments 1, 3, 6, and 8, the accepted base still exhibits overly decisive long-sequence behavior rather than merely excessive trace destruction: the relevant rates need to contract toward chance. Variance also remains too low in Experiment 4 (0.013 versus 0.062), while the point estimate there is already accurate.
Rationale: The previous recommendation to weaken overload-amplified displacement was rejected by the gate, so do not repeat it or build on the current overload-slope variant. Return computationally to the unchanged iteration-3 accepted base and try a different in-family knob: strengthen overflow-dependent readout uncertainty while leaving trace retention, near-zero validity sensitivity, and baseline displacement intact. A moderate upward shift or widening of `overflow_noise` should contract overloaded choices toward 0.5. That has the jointly correct directional effect: reduce Experiment 3 terminal agreement from 0.760 toward 0.575, raise Experiment 6 early choice from 0.169 toward 0.307, raise Experiment 8 terminal agreement from 0.114 toward 0.172, and reduce Experiment 1's oversized tally contrast. Because the two-reason trials in Experiment 2 do not exceed the 3.0–3.8 capacity range, this adjustment should largely preserve the successful short-conflict recency there. It should also modestly reduce Experiment 4's overshoot, where four reasons produce only slight overflow. Tune the increase conservatively and verify that Experiment 5 remains near zero; unlike changing trace displacement, symmetric readout attenuation should not intrinsically create block-versus-mixed differences. If heterogeneity is adjusted, widen overflow-noise or response-gain variation only modestly to improve Experiment 4 variance without restoring the excessive Experiment 2 variance. Preserve validity-ordered inspection, separate graded traces, and the absence of run, switch, suffix, or display-order mechanisms.

**Outcome of this advice:** iter 4 candidate loss=0.1512 -> iter 5 candidate loss=0.0828 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed validity-ordered competitive-buffer family, and the accepted increase in overflow uncertainty substantially improved experiment invariance. Experiments 4, 5, 6, 7, and 8 are now reasonably close, with the required null switch-density effect exactly reproduced and the arrangement contrast in Experiment 5 remaining small. Variance in Experiments 2 and 6 is also close to the observed scale. Two systematic point-estimate errors remain, however: tally modulation in Experiment 1 is still too large (0.634 versus 0.494), and terminal agreement in Experiment 3 remains too high (0.695 versus 0.575). Experiment 2 also retains somewhat too much highest-validity-cue agreement (0.259 versus 0.194). These indicate residual overdecisiveness and slightly insufficient short-conflict displacement. A uniform further increase in overflow noise is not ideal because Experiment 8 is already too close to chance in the wrong direction (0.201 versus 0.172).
Rationale: Build on this accepted candidate and preserve its trace-retention dynamics, near-zero validity sensitivity, and successful increase in overload uncertainty. The prior overflow-noise recommendation was accepted, but the next adjustment should reshape rather than uniformly increase that noise: make attenuation weaker or intermediate around six reasons, while making it more sharply convex for seven to nine reasons. For example, add a small overload threshold and a fitted exponent above two, then retune the noise scale. This should preserve Experiment 8's six-reason anti-terminal behavior while further contracting the seven-to-nine-reason predictions in Experiments 1, 3, and 6 toward chance. If needed, make only a very small upward adjustment to baseline displacement to reduce Experiment 2's remaining early-cue agreement; compensate its effect on longer sequences through the revised overload readout, rather than reopening validity weighting. Do not revisit the rejected weakening of overload-dependent trace displacement, and retain separate graded traces with no run, switch, suffix, or display-position terms.

**Outcome of this advice:** iter 5 candidate loss=0.0828 -> iter 6 candidate loss=0.0780 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed validity-ranked competitive-buffer family, and the latest accepted adjustment slightly improves aggregate loss. It fits Experiments 4, 5, 6, and 7 well: four-reason terminal control is accurate, arrangement and switch-density effects remain near null, and overloaded early-choice performance is close. Experiment 8 is also reasonably close. Nevertheless, three systematic errors prevent an experiment-invariant fit. Experiment 1 still shows excessive tally modulation (0.643 versus 0.494), Experiment 3 remains too terminal-controlled (0.689 versus 0.575), and Experiment 2 still follows the highest-validity cue too often (0.262 versus 0.194). The new convex-overload term helped aggregate loss and slightly reduced Experiment 3's error, but it did not contract Experiment 1 and slightly worsened Experiments 2 and 6. Its onset is also smeared by capacity: at seven reasons, `overflow/3` is often only just above one, so attenuation is weak and varies unnecessarily across subjects. Variance remains notably too low in Experiment 4 (0.013 versus 0.062), although its pooled estimate is excellent.
Rationale: Build on this accepted candidate, but make the previous accepted direction more targeted rather than uniformly increasing `overflow_noise`. Replace the weak capacity-smeared `max(1, overflow/3)` multiplier with a modest, smooth extra attenuation that begins explicitly around seven discriminating reasons; preserve the current uncertainty profile through six reasons. This should contract the long conditions driving the excessive Experiment 1 contrast and Experiment 3 terminal agreement while protecting Experiment 8's six-reason result. Keep the adjustment conservative because Experiment 6 is already slightly above its target and further contraction would raise its early-choice rate toward 0.5. Separately, narrow the already near-zero validity-sensitivity range, for example from [0, 0.18] toward [0, 0.08], rather than increasing displacement. That should reduce residual highest-validity agreement and heterogeneity in Experiment 2 and should also strengthen the anti-terminal majority pattern in Experiment 8. The long-sequence attenuation can offset the resulting extra tally/terminal influence in Experiments 1 and 3. Do not revisit the previously rejected weakening of overload-dependent trace displacement, and do not add run, suffix, switch, or display-position terms. Preserve the current capacity and displacement ranges unless the targeted changes demonstrably disturb the strong Experiment 4–7 fits.

**Outcome of this advice:** iter 6 candidate loss=0.0780 -> iter 7 candidate loss=0.0705 -> the gate marked it ACCEPTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed validity-ordered competitive-evidence-buffer family, and iteration 7 is another accepted improvement. It now fits the strongest diagnostic constraints very well: Experiment 2 is essentially exact (0.188 versus 0.194), Experiments 5 and 7 retain the required null arrangement and switch-density effects, Experiment 6 is close (0.296 versus 0.307), and Experiment 8 is nearly exact (0.179 versus 0.172). However, the remaining errors are systematic rather than sampling-sized. Evidence accumulation remains too decisive in Experiment 1, where tally modulation is 0.654 rather than 0.494, and terminal control remains too strong in Experiment 3, at 0.695 rather than 0.575. Four-reason terminal control in Experiment 4 also rose to 0.744 versus 0.665. Thus the model captures two-reason recency and six-to-eight-reason anti-terminal behavior, but overexpresses choice-consistent evidence once three or more retained traces combine. Experiment 4 also remains much too homogeneous across subjects (variance 0.011 versus 0.062), despite its reasonably close pooled estimate.
Rationale: Build on this accepted candidate and retain the successful narrow validity-sensitivity range: that change fixed Experiment 2 and improved aggregate loss, so it should not be reversed. The explicit seven-plus attenuation did not itself resolve the targeted Experiment 1 and 3 errors—their deviations slightly increased from iteration 6—so do not simply strengthen that coefficient again. Instead, try a mild load-dependent normalization of readout gain beginning after the two-reason case, such as dividing summed accessible evidence by a smooth function of retained trace count or total absolute accessible trace mass. This remains within the competitive-buffer family and preserves strong two-cue recency, while reducing overdecisive integration in Experiments 1, 3, and 4. A modest contraction should also move Experiment 6 slightly toward its target; it must be conservative because contraction would move Experiment 8 upward from its already accurate anti-terminal rate. Applying the normalization before the existing heterogeneous response gain may also move Experiment 4 away from saturation and expose more of the existing subject-level gain variation, increasing its currently deficient variance without widening validity sensitivity or displacement heterogeneity. Preserve validity-ranked inspection, graded separate traces, current capacity and displacement ranges, and the absence of run, suffix, switch-density, or display-position mechanisms.

**Outcome of this advice:** iter 7 candidate loss=0.0705 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Validity-Ordered Competitive Evidence Buffer with Load-Normalized Readout. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces, while a mild normalization beginning after two reasons prevents accumulated evidence from being expressed with unrealistically high confidence. Readout uncertainty grows quadratically with capacity overload and receives an additional count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, response gain, and integration normalization generate heterogeneous recency and integration behavior.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Competitive evidence buffer expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    capacity = float(parameters[\"buffer_capacity\"])\n    displacement = float(parameters[\"displacement_rate\"])\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    response_gain = float(parameters[\"response_gain\"])\n    integration_normalization = float(parameters[\"integration_normalization\"])\n    overflow_noise = float(parameters[\"overflow_noise\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    diff = stim[0] - stim[1]\n\n    # Inspection follows instructed validity, independently of physical\n    # feature position. Stable sorting only resolves exact validity ties.\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    discriminating = [int(j) for j in cue_order if diff[j] != 0]\n    m = len(discriminating)\n    if m == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Convert instructed reliability into evidence strength. Normalization\n    # removes arbitrary differences in the overall validity scale between\n    # experiments while preserving all within-experiment validity ratios.\n    clipped_v = np.clip(validities, 0.5001, 0.999)\n    reliability = np.log(clipped_v / (1.0 - clipped_v))\n    positive_reliability = reliability[reliability > 0.0]\n    reliability_scale = float(np.mean(positive_reliability))\n    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:\n        reliability_scale = 1.0\n\n    normalized_reliability = np.clip(reliability / reliability_scale, 1e-4, 20.0)\n    cue_strength = np.power(normalized_reliability, validity_sensitivity)\n    cue_strength = np.clip(cue_strength, 1e-6, 30.0)\n\n    # The buffer contains separate reason traces. A new discriminating reason\n    # interferes continuously with every old trace. Displacement is present\n    # even below capacity and increases smoothly under overload; there is no\n    # categorical slot deletion and no compression into directional runs.\n    traces = []\n    for j in discriminating:\n        current_load = float(len(traces))\n        overload_before_entry = max(0.0, current_load + 1.0 - capacity)\n        load_factor = 1.0 + overload_before_entry / max(capacity, 1e-6)\n        retention = np.exp(-displacement * load_factor / max(capacity, 1e-6))\n        retention = float(np.clip(retention, 0.0, 1.0))\n\n        if traces:\n            traces = [retention * trace for trace in traces]\n\n        direction = float(np.sign(diff[j]))  # positive favors A\n        traces.append(direction * float(cue_strength[j]))\n\n    accessible_evidence = float(np.sum(traces))\n\n    # Preserve the accepted quadratic capacity-overload uncertainty through\n    # six reasons. Beginning explicitly at seven discriminating reasons, add\n    # a modest count-based attenuation that is not smeared by capacity.\n    overflow = max(0.0, float(m) - capacity)\n    long_sequence_excess = max(0.0, float(m) - 6.0)\n    convex_multiplier = 1.0 + 0.45 * long_sequence_excess\n    readout_sd = np.sqrt(\n        1.0 + overflow_noise * overflow * overflow * convex_multiplier\n    )\n\n    # Once more than two reasons are retained, summed evidence is mildly\n    # normalized for integration load. The square-root form is deliberately\n    # sublinear so it contracts overconfident multi-reason choices without\n    # erasing tally information or imposing a sharp capacity boundary.\n    integration_load = np.sqrt(max(0.0, float(m) - 2.0))\n    integration_divisor = 1.0 + integration_normalization * integration_load\n\n    decision_variable = (\n        response_gain * accessible_evidence\n        / (readout_sd * integration_divisor)\n    )\n    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))\n\n    p_a_core = 1.0 / (1.0 + np.exp(-decision_variable))\n    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)\n\n    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "buffer_capacity": "[3.0, 3.8]",
    "displacement_rate": "[2.1, 2.9]",
    "validity_sensitivity": "[0.0, 0.08]",
    "response_gain": "[3.0, 6.5]",
    "integration_normalization": "[0.03, 0.12]",
    "overflow_noise": "[0.75, 1.50]",
    "lapse_rate": "[0.03, 0.15]"
  },
  "rationale": "This is a minimal edit to the accepted iteration-7 model. It preserves cue ordering, trace encoding, displacement, validity sensitivity, overload noise, and all existing parameter ranges. The only computational change is a mild sublinear readout divisor, 1 + integration_normalization * sqrt(max(0, m - 2)), applied after the two-reason case. Consequently, the nearly exact two-cue recency result in Experiment 2 is protected, while choices based on three or more combined traces are expressed less decisively. This should reduce excessive tally modulation in Experiment 1 and excessive terminal control in Experiments 3 and 4, while moving Experiment 6 slightly toward its target. The square-root growth is intentionally conservative to limit upward contraction of the already accurate anti-terminal rate in Experiment 8. Stable variation in the new normalization strength can also reveal more individual variation in four-reason readout without reopening validity-weighting heterogeneity or adding run-, suffix-, switch-, or display-position-specific mechanisms."
}
```

## Usage

```json
{
  "input_tokens": 22688,
  "output_tokens": 3227
}
```
