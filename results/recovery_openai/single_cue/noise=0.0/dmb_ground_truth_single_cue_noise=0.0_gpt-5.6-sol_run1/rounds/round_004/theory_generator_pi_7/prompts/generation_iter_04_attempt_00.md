# generation_iter_04_attempt_00

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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a Load-Contingent Dual-Code Arbitration theory. The new theory should posit two representations: a rapidly revised local commitment, controlled by the most recently encoded discriminating reasons, and a validity-weighted global coalition summary. Short sequences ordinarily use local commitment, allowing a two-cue suffix to eliminate first-reason effects and accounting for Experiments 1, 2, 4, and 6. When a conflict reaches an intermediate, manageable level, subjects can recruit the global summary, so coherent high-validity coalitions or tally margins can overcome a terminal cue, as required by Experiments 3 and 10. Once effective capacity is exceeded, the global summary becomes inaccessible or noisy and responding falls back toward attenuated local recency, accommodating Experiments 5 and 8. Arbitration should depend smoothly on effective load, weighted coalition strength, and conflict balance—not on an ad hoc fixed cue count or explicit directional-run chunks—so the theory predicts weak switch-arrangement effects consistent with Experiments 7 and 9. Stable subject differences should govern capacity, the threshold for recruiting global comparison, and the relative gain of local versus global representations; these differences should be modeled jointly across tasks to reproduce the elevated between-subject variances. This differs from pi_3 by permitting genuine cumulative evidence to control manageable longer conflicts, and from pi_6 by allowing early traces to be functionally absent during short local revision rather than obligatorily surviving in a common evidence buffer.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.853)

**Description**
Validity-Ordered Competitive Evidence Buffer with Conflict-Normalized Readout. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces. Readout normalization depends on the balance of accessible evidence: it is strongest when substantial trace mass supports both options and weak when the accessible buffer contains a coherent majority. Readout uncertainty additionally grows quadratically with capacity overload and receives a count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, response gain, and conflict normalization generate heterogeneous recency and integration behavior.

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
    integration_normalization = float(parameters["integration_normalization"])
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

    # Normalize integrated evidence only when accessible traces substantially
    # support both options. Squaring the balance ratio leaves coherent
    # majorities largely intact. The effect remains absent for two reasons.
    positive_mass = float(np.sum([trace for trace in traces if trace > 0.0]))
    negative_mass = float(-np.sum([trace for trace in traces if trace < 0.0]))
    total_mass = positive_mass + negative_mass
    if m <= 2 or total_mass <= 0.0 or not np.isfinite(total_mass):
        conflict_load = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))
        conflict_load = np.sqrt(max(0.0, float(m) - 2.0)) * balance * balance
    integration_divisor = 1.0 + integration_normalization * conflict_load

    decision_variable = (
        response_gain * accessible_evidence
        / (readout_sd * integration_divisor)
    )
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
- response_gain: [2.0, 6.5]
- integration_normalization: [0.02, 0.34]
- overflow_noise: [0.75, 1.50]
- lapse_rate: [0.03, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.6093 (var=0.0085)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.2027 (var=0.0056)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.6577 (var=0.0054)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6704 (var=0.0140)
- Experiment 5: real=0.0033 (var=0.0049) vs this=-0.0142 (var=0.0089)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.3067 (var=0.0061)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.2100)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.2250 (var=0.0059)
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.2096 (var=0.0101)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.9281 (var=0.0013)


---

### `pi_3` (overall score: 0.849)

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


---

### `pi_4` (overall score: 0.562)

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
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.0375 (var=0.0075)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.8645 (var=0.0034)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1356 -> ACCEPTED
- iter 2: loss=0.1921 -> REJECTED
- iter 3: loss=0.1178 -> ACCEPTED
- iter 4: loss=0.1149 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 4 at loss=0.1149 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.6425 (var=0.0082) (Δ vs real +0.1481)
  - iter 2: 0.6651 (var=0.0143) (Δ vs real +0.1707)
  - iter 3: 0.6614 (var=0.0090) (Δ vs real +0.1669)
  - iter 4 (most recent): 0.5908 (var=0.0100) (Δ vs real +0.0964)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)
- pi_6: 0.6093 (var=0.0085)

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
  - iter 1: 0.2360 (var=0.0026) (Δ vs real +0.0423)
  - iter 2: 0.2494 (var=0.0131) (Δ vs real +0.0556)
  - iter 3: 0.1944 (var=0.0030) (Δ vs real +0.0006)
  - iter 4 (most recent): 0.2075 (var=0.0028) (Δ vs real +0.0137)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)
- pi_6: 0.2027 (var=0.0056)

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
  - iter 1: 0.6904 (var=0.0038) (Δ vs real +0.1154)
  - iter 2: 0.6787 (var=0.0063) (Δ vs real +0.1038)
  - iter 3: 0.6973 (var=0.0044) (Δ vs real +0.1223)
  - iter 4 (most recent): 0.6863 (var=0.0036) (Δ vs real +0.1113)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)
- pi_6: 0.6577 (var=0.0054)

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
  - iter 1: 0.4079 (var=0.0182) (Δ vs real -0.2571)
  - iter 2: 0.2000 (var=0.0469) (Δ vs real -0.4650)
  - iter 3: 0.6104 (var=0.0109) (Δ vs real -0.0546)
  - iter 4 (most recent): 0.6217 (var=0.0105) (Δ vs real -0.0433)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)
- pi_6: 0.6704 (var=0.0140)

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
  - iter 1: 0.0146 (var=0.0079) (Δ vs real +0.0113)
  - iter 2: 0.0258 (var=0.0095) (Δ vs real +0.0225)
  - iter 3: -0.0104 (var=0.0087) (Δ vs real -0.0137)
  - iter 4 (most recent): -0.0162 (var=0.0125) (Δ vs real -0.0196)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)
- pi_6: -0.0142 (var=0.0089)

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
  - iter 1: 0.2904 (var=0.0064) (Δ vs real -0.0162)
  - iter 2: 0.3554 (var=0.0217) (Δ vs real +0.0488)
  - iter 3: 0.2675 (var=0.0063) (Δ vs real -0.0392)
  - iter 4 (most recent): 0.3463 (var=0.0062) (Δ vs real +0.0396)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)
- pi_6: 0.3067 (var=0.0061)

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
  - iter 1: 0.0000 (var=0.1716) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2016) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2244) (Δ vs real +0.0000)
  - iter 4 (most recent): 0.0000 (var=0.2244) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)
- pi_6: 0.0000 (var=0.2100)

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
  - iter 1: 0.2494 (var=0.0026) (Δ vs real +0.0773)
  - iter 2: 0.2958 (var=0.0018) (Δ vs real +0.1237)
  - iter 3: 0.2992 (var=0.0015) (Δ vs real +0.1271)
  - iter 4 (most recent): 0.3304 (var=0.0025) (Δ vs real +0.1583)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)
- pi_6: 0.2250 (var=0.0059)

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
  - iter 1: 0.1496 (var=0.0157) (Δ vs real +0.1821)
  - iter 2: 0.2333 (var=0.0690) (Δ vs real +0.2658)
  - iter 3: 0.2117 (var=0.0175) (Δ vs real +0.2442)
  - iter 4 (most recent): 0.2188 (var=0.0132) (Δ vs real +0.2512)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0083 (var=0.0103)
- pi_6: 0.2096 (var=0.0101)
- pi_1: 0.7021 (var=0.0307)
- pi_2: 0.7054 (var=0.0384)
- pi_4: 0.0375 (var=0.0075)
- pi_5: 0.0108 (var=0.0053)

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
  - iter 1: 0.8099 (var=0.0020) (Δ vs real -0.0649)
  - iter 2: 0.7977 (var=0.0029) (Δ vs real -0.0771)
  - iter 3: 0.8657 (var=0.0015) (Δ vs real -0.0091)
  - iter 4 (most recent): 0.8667 (var=0.0017) (Δ vs real -0.0081)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9281 (var=0.0013)
- pi_3: 0.7843 (var=0.0034)
- pi_1: 0.5635 (var=0.0023)
- pi_2: 0.8101 (var=0.0072)
- pi_4: 0.8645 (var=0.0034)
- pi_5: 0.9256 (var=0.0034)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Load-Contingent Dual-Code Arbitration. While inspecting discriminating experts in instructed-validity order, a decision maker constructs two qualitatively different codes. The local code is a rapidly revised categorical commitment: each new discriminating reason largely replaces the previous commitment, so reasons outside the recent suffix become functionally absent rather than merely weakened traces. In parallel, the person can form a validity-weighted global coalition summary that preserves cumulative directional evidence without preserving cue order. Choice is governed by smooth arbitration between these codes. Global comparison is recruited only after conflict is sufficiently elaborate to warrant comparison, is favored when validity-weighted evidence forms a diagnostic coalition supported by an unweighted directional majority, and loses accessibility when subject-specific effective capacity is exceeded. Thus short or count-balanced conflicts are dominated by local revision, manageable majority-coalition conflicts can be decided cumulatively, and overloaded conflicts revert toward an uncertainty-attenuated recent commitment. Capacity, recruitment threshold, coalition selectivity, and local and global gains are stable subject characteristics shared across trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Dual-code arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["effective_capacity"])
    recruitment_threshold = float(parameters["recruitment_threshold"])
    arbitration_temperature = float(parameters["arbitration_temperature"])
    local_retention = float(parameters["local_retention"])
    local_gain = float(parameters["local_gain"])
    global_gain = float(parameters["global_gain"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    coalition_focus = float(parameters["coalition_focus"])
    overload_attenuation = float(parameters["overload_attenuation"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    # LOCAL CODE: a categorical commitment rather than a common evidence
    # buffer. Every newly encoded reason installs its own direction and only
    # a small residue of the immediately preceding commitment remains.
    # Consequently, sufficiently early reasons are functionally absent.
    local_commitment = 0.0
    for direction in directions:
        local_commitment = direction + local_retention * local_commitment
    local_commitment = float(np.clip(local_commitment, -1.5, 1.5))

    # GLOBAL CODE: cumulative validity-weighted coalition evidence. Log odds
    # represent communicated reliability, while normalization makes the code
    # portable across experiments with different absolute validity scales.
    v = np.clip(validities[discriminating], 0.5001, 0.999)
    reliability = np.log(v / (1.0 - v))
    positive = reliability[reliability > 0.0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    weights = np.power(
        np.clip(reliability / scale, 1e-4, 20.0), validity_sensitivity
    )
    weights = np.clip(weights, 1e-5, 30.0)

    signed_global = float(np.sum(weights * directions))
    global_norm = float(np.sqrt(np.sum(weights * weights)))
    if not np.isfinite(global_norm) or global_norm <= 0.0:
        global_norm = 1.0
    global_summary = signed_global / global_norm

    total_weight = float(np.sum(weights))
    coalition_strength = abs(signed_global) / max(total_weight, 1e-9)
    coalition_strength = float(np.clip(coalition_strength, 0.0, 1.0))

    # Conflict balance is high when both options have several reasons. It is
    # arrangement-free: only directional masses enter, never switch counts or
    # directional runs.
    positive_mass = float(np.sum(weights[directions > 0.0]))
    negative_mass = float(np.sum(weights[directions < 0.0]))
    if positive_mass + negative_mass <= 0.0:
        conflict_balance = 0.0
    else:
        conflict_balance = (
            2.0 * min(positive_mass, negative_mass)
            / (positive_mass + negative_mass)
        )
    conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))

    temp = max(arbitration_temperature, 1e-6)

    # Recruitment rises smoothly once a conflict warrants an explicit global
    # comparison, but global accessibility falls smoothly near the person's
    # effective capacity. No fixed cue-count boundary is imposed.
    onset_x = np.clip((float(m) - recruitment_threshold) / temp, -60.0, 60.0)
    onset = 1.0 / (1.0 + np.exp(-onset_x))
    capacity_x = np.clip((capacity - float(m)) / temp, -60.0, 60.0)
    accessibility = 1.0 / (1.0 + np.exp(-capacity_x))

    # Validity weighting determines the content of the global summary, but
    # recruitment also requires an unweighted directional coalition. This
    # prevents validity imbalance alone from turning a count-balanced 2-vs-2
    # display into a globally diagnostic conflict. A smooth weighted-strength
    # criterion additionally prevents weak 2-vs-1 coalitions from recruiting
    # global comparison as strongly as genuinely coherent coalitions.
    count_coalition = abs(float(np.sum(directions))) / float(m)
    count_coalition = float(np.clip(3.0 * count_coalition, 0.0, 1.0))
    weighted_selectivity_x = np.clip(
        (coalition_strength - 0.35) / 0.12, -60.0, 60.0
    )
    weighted_selectivity = 1.0 / (1.0 + np.exp(-weighted_selectivity_x))
    diagnosticity = np.power(
        np.clip(coalition_strength, 0.0, 1.0), coalition_focus
    )
    diagnosticity *= count_coalition * weighted_selectivity
    diagnosticity = float(np.clip(diagnosticity, 0.0, 1.0))

    global_gate = float(np.clip(onset * accessibility * diagnosticity, 0.0, 1.0))

    # Capacity pressure attenuates expression of the local commitment. Its
    # smooth onset slightly precedes nominal capacity, reflecting the cost of
    # maintaining both codes before capacity is completely exceeded.
    overload = temp * np.logaddexp(
        0.0, (float(m) - capacity + 0.75) / temp
    )
    local_divisor = np.sqrt(1.0 + overload_attenuation * overload * overload)
    local_dv = local_gain * local_commitment / local_divisor
    global_dv = global_gain * global_summary

    decision_variable = (
        (1.0 - global_gate) * local_dv + global_gate * global_dv
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
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
- effective_capacity: [5.8, 6.8]
- recruitment_threshold: [2.8, 3.8]
- arbitration_temperature: [0.35, 0.75]
- local_retention: [0.02, 0.16]
- local_gain: [1.35, 2.15]
- global_gain: [2.0, 4.2]
- validity_sensitivity: [0.65, 1.35]
- coalition_focus: [0.75, 1.65]
- overload_attenuation: [0.8, 1.6]
- lapse_rate: [0.03, 0.14]

`rationale`: This preserves the accepted count-coalition mechanism, including its exact zero gate for count-balanced conflicts, and makes only two targeted changes. First, a modest smooth threshold on weighted coalition strength is inserted into diagnosticity. Weak 2-versus-1 coalitions therefore recruit the global code less strongly, which should reduce the excessive first-cue association in Experiment 9 and temper Experiment 1 without undoing local protection in Experiment 4. Strong validity-weighted coalitions can still pass the criterion and control manageable conflicts in Experiment 3. Second, local overload attenuation now begins smoothly about three-quarters of a reason before nominal capacity, and its range is increased modestly. This specifically weakens residual terminal commitment around six to eight reasons, allowing the existing global coalition to lower terminal agreement in Experiment 8, increase early-coalition choices in Experiment 6, and reduce terminal dominance in longer Experiment 3 conditions. Global gain, recruitment sharpness, and the broad subject-level ranges are otherwise unchanged, avoiding the rejected iteration's simultaneous high-gain and heterogeneity expansion.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed dual-code family and captures several qualitative invariances: Experiment 5's arrangement contrast is near zero, Experiment 7 is correctly below the practical-effect threshold, and overload behavior in Experiment 6 is close to observed. However, the fit is not yet experiment-invariant. Terminal/local control is much too weak in Experiment 4 (0.408 vs 0.665) and somewhat too weak in Experiment 10 (0.810 vs 0.875), while it is too strong in Experiment 3 (0.690 vs 0.575). The model also retains excessive high-validity/first-cue influence in Experiments 2 and 9 (0.236 vs 0.194 and 0.150 vs -0.033), and insufficiently reverses terminal control in Experiment 8 (0.249 vs 0.172). Experiment 1's tally-support contrast is also exaggerated (0.643 vs 0.494). Thus the two codes are plausible, but arbitration is insufficiently selective: the global code intrudes on balanced short conflicts such as Experiment 4, yet is not recruited strongly enough for the diagnostic coalitions in Experiments 3 and 8. Between-subject variances are also substantially compressed in Experiments 1, 4, 8, and 10.
Rationale: Keep the dual-code architecture, but recalibrate the gate rather than uniformly increasing either local or global gain. Make global recruitment more selective for genuinely strong validity-weighted coalitions—e.g. increase or sharpen coalition_focus and reduce gate activation for balanced, weak-net-evidence displays. This should protect terminal local commitment in Experiment 4 and reduce the spurious first-cue association in Experiment 9. Conditional on passing that selectivity test, increase global gain/accessibility so strong opposing coalitions can more effectively reduce terminal agreement in Experiment 3 and Experiment 8. A modestly higher and wider effective-capacity range may help Experiment 8 while remaining compatible with the near-correct Experiment 6 overload result, but avoid a blanket reduction of overload attenuation. The recruitment transition should also be sharper around the short-to-manageable boundary so two-cue cases remain local while diagnostic longer conflicts recruit the global code. Finally, widen stable subject-level ranges governing capacity, recruitment, and relative code gains to address the markedly underestimated variances, especially in Experiments 4, 8, and 10. Avoid simply raising local_gain across the board: although that could repair Experiment 4, it would worsen the already excessive terminal agreement in Experiment 3 and likely inflate Experiment 1's contrast further.

**Outcome of this advice:** iter 1 candidate loss=0.1356 -> iter 2 candidate loss=0.1921 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed dual-code family, but it is a clear regression from the accepted base (loss 0.1921 versus 0.1356). The previous attempt to sharpen coalition selectivity while increasing global gain did not produce the intended selective arbitration. It severely weakened terminal/local control in Experiment 4 (0.200 versus 0.665; previously 0.408), and also worsened Experiments 1, 2, 6, 8, 9, and 10. In particular, first/high-validity influence remains excessive in Experiment 9 (0.233 versus -0.033), terminal agreement remains too high in Experiment 8 (0.296 versus 0.172), and Experiment 1's tally-support contrast is further exaggerated (0.665 versus 0.494). The small improvement in Experiment 3 does not compensate for these losses. Experiments 5 and 7 still correctly show weak arrangement effects, so the architectural invariance remains sound; the failure lies in how coalition diagnosticity is computed and translated into arbitration. The weighted conflict-balance statistic can label nominally balanced 2-versus-2 displays as asymmetric merely because early validities carry greater mass. Consequently, even a suppressed gate can inject a large opposing global signal after the increase in global gain, explaining the collapse of terminal control in Experiment 4.
Rationale: Because this iteration was rejected, build from the unchanged iter-1 accepted base and do not repeat the rejected combination of sharper recruitment, higher global gain, and broadly widened ranges. Try a different in-family adjustment: revise the gate's conflict normalization so that coalition structure is not determined solely by validity-weighted mass. Use unweighted directional counts, or a bounded hybrid of count balance and weighted balance, for the conflict/diagnosticity term while retaining validity weighting inside the global summary itself. This should distinguish balanced 2-versus-2 conflicts in Experiment 4, which should remain locally terminal, from 2-versus-1 conflicts in Experiments 3 and 9, where a genuine directional majority can recruit the global code. It may also improve Experiment 9 by allowing the two-cue coalition to overcome the first cue without globally increasing gain. Keep global gain and recruitment temperature near the accepted-base ranges; normalize or cap the effective global contribution so a small gate cannot dominate the local code. Do not further widen subject ranges yet, since that rejected change inflated variance dramatically in Experiment 9 and Experiment 6 while leaving variance too low elsewhere. First repair the pooled means through the count/weight normalization of arbitration, then tune heterogeneity only if the mean pattern improves.

**Outcome of this advice:** iter 2 candidate loss=0.1921 -> iter 3 candidate loss=0.1178 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted count-coalition revision is a genuine improvement and should be retained. It repairs Experiment 4 substantially (0.610 versus 0.665), while Experiment 2 is essentially exact, Experiment 10 is close, Experiments 5 and 7 preserve the required weak arrangement effects, and Experiment 6 remains reasonably close. The candidate is also faithful to the prescribed load-contingent dual-code family. Nevertheless, the fit is not yet experiment-invariant. Large pooled discrepancies remain in Experiment 1, where the tally-support contrast is too strong (0.661 versus 0.494); Experiment 3, where terminal-cue agreement is too high (0.697 versus 0.575); Experiment 8, where overloaded terminal agreement remains much too high (0.299 versus 0.172); and especially Experiment 9, where the model retains a positive first-cue association (0.212) despite the observed near-zero/slightly negative association (-0.033). The model also generally underproduces between-subject heterogeneity, notably in Experiments 1, 4, 8, and 10. Thus the tie-protection fix worked, but arbitration among non-tied coalitions and the attenuation of the local code under overload remain miscalibrated.
Rationale: Build on the accepted candidate and preserve the new zero gate for exactly count-balanced conflicts; that successful adjustment should not be undone. The next change should target overload expression rather than repeat the previously rejected package of sharper recruitment, higher global gain, and broad range widening. In particular, strengthen or slightly advance the smooth overload attenuation of the local code around six to eight discriminating cues. At present, residual local terminal commitment remains too influential in Experiment 8. More attenuation there would let the already-computed global coalition exert greater relative control, lowering terminal agreement; it should also raise early-coalition choices in Experiment 6 from 0.268 toward 0.307 and may reduce the excessive terminal agreement in the longer conditions of Experiment 3. This can be done through a modest upward adjustment of overload_attenuation or a smoother divisor that begins operating nearer capacity, without increasing global_gain. Separately, the current multiplier saturates every 2-versus-1 count majority at full count diagnosticity, allowing mixed three-cue displays in Experiment 9 to retain too much validity-weighted first-cue influence. Add a modest smooth threshold or temperature on weighted coalition strength so only genuinely coherent weighted coalitions recruit strongly; do not merely sharpen all recruitment or suppress every 2-versus-1 case, because Experiment 3 requires cumulative control for strong coalitions. This should also temper Experiment 1's exaggerated tally contrast. Tune pooled means first and defer broad heterogeneity widening; once the mean fit improves, selectively widen stable capacity or relative-gain variation to address the compressed variances without recreating the rejected variance inflation in Experiments 6 and 9.

**Outcome of this advice:** iter 3 candidate loss=0.1178 -> iter 4 candidate loss=0.1149 -> the gate marked it ACCEPTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed load-contingent dual-code family and is a small accepted improvement overall. The successful count-balance protection remains important: Experiment 4 is close (0.622 vs 0.665), Experiment 2 is reasonably close, Experiment 10 is nearly exact, and Experiments 5 and 7 correctly retain weak arrangement effects. The new weighted-selectivity adjustment also reduced Experiment 1's excessive contrast from 0.661 to 0.591, although it remains above 0.494. However, the model is not yet experiment-invariant. Experiment 9 is the clearest failure: first-cue association remains strongly positive (0.219) rather than near zero/slightly negative (-0.033), and the weighted threshold did not improve it. Terminal agreement is also too high in Experiment 3 (0.686 vs 0.575) and especially Experiment 8 (0.330 vs 0.172). Moreover, the attempted earlier overload attenuation moved Experiment 8 in the wrong direction and pushed Experiment 6 from below the target to above it (0.346 vs 0.307). Between-subject variances remain markedly compressed in Experiments 1, 4, 8, and 10.
Rationale: Retain the accepted count-balanced zero gate and the dual-code architecture, but do not further strengthen or advance blanket overload attenuation: that previously helpful-looking direction was accepted only marginally and produced worse local fits in Experiments 6 and 8. The next adjustment should make coalition selectivity explicitly and smoothly load-contingent. Suppress global recruitment more strongly at three discriminating cues, where Experiment 9 shows that a 2-versus-1 majority should not preserve first-cue influence, while relaxing the new weighted-strength threshold within the manageable five-to-six-cue window so genuine cumulative coalitions can oppose the terminal code in Experiments 3 and 8. Global accessibility should still decline beyond that window, protecting the overload pattern and avoiding further inflation of Experiment 6. This can be implemented as a smooth load-dependent selectivity criterion or recruitment window, rather than a fixed cue-count rule, and without raising global_gain. The current fixed coalition-strength cutoff is too blunt: it improved Experiment 1 but apparently suppressed useful global control at six cues without eliminating unwanted three-cue control. After repairing these pooled means, selectively widen stable capacity or arbitration-threshold variation to address compressed heterogeneity; do not broadly widen all parameter ranges, since that earlier package was rejected.

**Outcome of this advice:** iter 4 candidate loss=0.1149 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Load-Contingent Dual-Code Arbitration. While inspecting discriminating experts in instructed-validity order, a decision maker constructs two qualitatively different codes. The local code is a rapidly revised categorical commitment: each new discriminating reason largely replaces the previous commitment, so reasons outside the recent suffix become functionally absent rather than merely weakened traces. In parallel, the person can form a validity-weighted global coalition summary that preserves cumulative directional evidence without preserving cue order. Choice is governed by smooth arbitration between these codes. Global comparison is recruited only after conflict is sufficiently elaborate to warrant comparison, is favored when validity-weighted evidence forms a diagnostic coalition supported by an unweighted directional majority, and loses accessibility when subject-specific effective capacity is exceeded. Coalition selectivity is itself load-contingent: a coalition must be especially strong near the recruitment boundary, whereas weaker cumulative coalitions become usable inside the subject's manageable comparison window before becoming inaccessible under overload. Thus short or count-balanced conflicts are dominated by local revision, manageable majority-coalition conflicts can be decided cumulatively, and overloaded conflicts revert toward an uncertainty-attenuated recent commitment. Capacity, recruitment threshold, coalition selectivity, and local and global gains are stable subject characteristics shared across trials.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Dual-code arbitration expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    capacity = float(parameters[\"effective_capacity\"])\n    recruitment_threshold = float(parameters[\"recruitment_threshold\"])\n    arbitration_temperature = float(parameters[\"arbitration_temperature\"])\n    local_retention = float(parameters[\"local_retention\"])\n    local_gain = float(parameters[\"local_gain\"])\n    global_gain = float(parameters[\"global_gain\"])\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    coalition_focus = float(parameters[\"coalition_focus\"])\n    overload_attenuation = float(parameters[\"overload_attenuation\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    diff = stim[0] - stim[1]\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    discriminating = [int(j) for j in cue_order if diff[j] != 0]\n    m = len(discriminating)\n    if m == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    directions = np.asarray(\n        [float(np.sign(diff[j])) for j in discriminating], dtype=float\n    )\n\n    # LOCAL CODE: a categorical commitment rather than a common evidence\n    # buffer. Every newly encoded reason installs its own direction and only\n    # a small residue of the immediately preceding commitment remains.\n    # Consequently, sufficiently early reasons are functionally absent.\n    local_commitment = 0.0\n    for direction in directions:\n        local_commitment = direction + local_retention * local_commitment\n    local_commitment = float(np.clip(local_commitment, -1.5, 1.5))\n\n    # GLOBAL CODE: cumulative validity-weighted coalition evidence. Log odds\n    # represent communicated reliability, while normalization makes the code\n    # portable across experiments with different absolute validity scales.\n    v = np.clip(validities[discriminating], 0.5001, 0.999)\n    reliability = np.log(v / (1.0 - v))\n    positive = reliability[reliability > 0.0]\n    scale = float(np.mean(positive)) if positive.size else 1.0\n    if not np.isfinite(scale) or scale <= 0.0:\n        scale = 1.0\n    weights = np.power(\n        np.clip(reliability / scale, 1e-4, 20.0), validity_sensitivity\n    )\n    weights = np.clip(weights, 1e-5, 30.0)\n\n    signed_global = float(np.sum(weights * directions))\n    global_norm = float(np.sqrt(np.sum(weights * weights)))\n    if not np.isfinite(global_norm) or global_norm <= 0.0:\n        global_norm = 1.0\n    global_summary = signed_global / global_norm\n\n    total_weight = float(np.sum(weights))\n    coalition_strength = abs(signed_global) / max(total_weight, 1e-9)\n    coalition_strength = float(np.clip(coalition_strength, 0.0, 1.0))\n\n    # Conflict balance is high when both options have several reasons. It is\n    # arrangement-free: only directional masses enter, never switch counts or\n    # directional runs.\n    positive_mass = float(np.sum(weights[directions > 0.0]))\n    negative_mass = float(np.sum(weights[directions < 0.0]))\n    if positive_mass + negative_mass <= 0.0:\n        conflict_balance = 0.0\n    else:\n        conflict_balance = (\n            2.0 * min(positive_mass, negative_mass)\n            / (positive_mass + negative_mass)\n        )\n    conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))\n\n    temp = max(arbitration_temperature, 1e-6)\n\n    # Recruitment rises smoothly once a conflict warrants an explicit global\n    # comparison, but global accessibility falls smoothly near the person's\n    # effective capacity. No fixed cue-count boundary is imposed.\n    onset_x = np.clip((float(m) - recruitment_threshold) / temp, -60.0, 60.0)\n    onset = 1.0 / (1.0 + np.exp(-onset_x))\n    capacity_x = np.clip((capacity - float(m)) / temp, -60.0, 60.0)\n    accessibility = 1.0 / (1.0 + np.exp(-capacity_x))\n\n    # Validity weighting determines the content of the global summary, but\n    # recruitment also requires an unweighted directional coalition. This\n    # prevents validity imbalance alone from turning a count-balanced 2-vs-2\n    # display into a globally diagnostic conflict.\n    count_coalition = abs(float(np.sum(directions))) / float(m)\n    count_coalition = float(np.clip(3.0 * count_coalition, 0.0, 1.0))\n\n    # Coalition selectivity varies smoothly with effective load. Near the\n    # subject's recruitment boundary, only a strong weighted coalition can\n    # engage global comparison. Once load is clearly above that boundary but\n    # still below effective capacity, weaker cumulative coalitions become\n    # usable. Selectivity tightens again as overload approaches. The window is\n    # defined relative to stable subject-specific thresholds rather than by a\n    # fixed number of cues.\n    comparison_x = np.clip(\n        (float(m) - recruitment_threshold - 0.75) / temp, -60.0, 60.0\n    )\n    comparison_ready = 1.0 / (1.0 + np.exp(-comparison_x))\n    pre_overload_x = np.clip(\n        (capacity + 0.50 - float(m)) / temp, -60.0, 60.0\n    )\n    pre_overload_access = 1.0 / (1.0 + np.exp(-pre_overload_x))\n    manageable_window = comparison_ready * pre_overload_access\n    selectivity_cutoff = 0.56 - 0.30 * manageable_window\n\n    weighted_selectivity_x = np.clip(\n        (coalition_strength - selectivity_cutoff) / 0.12, -60.0, 60.0\n    )\n    weighted_selectivity = 1.0 / (1.0 + np.exp(-weighted_selectivity_x))\n    diagnosticity = np.power(\n        np.clip(coalition_strength, 0.0, 1.0), coalition_focus\n    )\n    diagnosticity *= count_coalition * weighted_selectivity\n    diagnosticity = float(np.clip(diagnosticity, 0.0, 1.0))\n\n    global_gate = float(np.clip(onset * accessibility * diagnosticity, 0.0, 1.0))\n\n    # Capacity pressure attenuates expression of the local commitment. Its\n    # smooth onset slightly precedes nominal capacity, reflecting the cost of\n    # maintaining both codes before capacity is completely exceeded.\n    overload = temp * np.logaddexp(\n        0.0, (float(m) - capacity + 0.75) / temp\n    )\n    local_divisor = np.sqrt(1.0 + overload_attenuation * overload * overload)\n    local_dv = local_gain * local_commitment / local_divisor\n    global_dv = global_gain * global_summary\n\n    decision_variable = (\n        (1.0 - global_gate) * local_dv + global_gate * global_dv\n    )\n    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-decision_variable))\n    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])\n\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "effective_capacity": "[5.8, 6.8]",
    "recruitment_threshold": "[2.8, 3.8]",
    "arbitration_temperature": "[0.35, 0.75]",
    "local_retention": "[0.02, 0.16]",
    "local_gain": "[1.35, 2.15]",
    "global_gain": "[2.0, 4.2]",
    "validity_sensitivity": "[0.65, 1.35]",
    "coalition_focus": "[0.75, 1.65]",
    "overload_attenuation": "[0.8, 1.6]",
    "lapse_rate": "[0.03, 0.14]"
  },
  "rationale": "This is a minimal edit to the accepted candidate: the local code, global summary, count-balanced zero gate, gains, overload equation, and all parameter ranges are retained. Only the fixed weighted-coalition cutoff is replaced by a smooth subject-relative comparison window. Near the recruitment boundary, the cutoff is high, suppressing unwanted global influence in three-reason displays and allowing terminal-local responding to remove the positive first-cue association in Experiment 9. Inside the manageable interval between recruitment and capacity, the cutoff falls, allowing cumulative five-to-six-reason coalitions to oppose the terminal code more effectively in Experiments 3 and 8. As capacity is approached, the cutoff rises again while the existing accessibility gate declines, avoiding an indiscriminate increase in global control for the seven-to-eight-reason overload conditions of Experiment 6. The edit depends on each subject's recruitment threshold and effective capacity rather than an explicit cue-count boundary or directional-run structure, preserving the weak arrangement effects in Experiments 5 and 7."
}
```

## Usage

```json
{
  "input_tokens": 22871,
  "output_tokens": 4054
}
```
