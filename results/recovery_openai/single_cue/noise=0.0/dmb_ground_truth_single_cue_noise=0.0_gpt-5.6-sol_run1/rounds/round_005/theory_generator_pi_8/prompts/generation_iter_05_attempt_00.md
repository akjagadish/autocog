# generation_iter_05_attempt_00

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
- THEORY 2 = `pi_7`
- The recommendation below acts on THEORY 2 (= `pi_7`).

Replace pi_7 with a Context-Calibrated Sequence-Policy theory. The new theory should posit that people compress cue sequences into ordinal events—initial commitment, reversals, runs, and terminal commitment—and select among recency, primacy, and limited tally policies using both display structure and the distribution of structures encountered across the task. Communicated validities should primarily determine inspection order and confidence; their magnitudes should have strongly compressed influence and should not automatically recruit a global weighted coalition. Stable subject traits should govern attention, positional bias, policy-switch thresholds, and lapse, while task context determines which policy is engaged. The theory should allow strong terminal capture when a task repeatedly foregrounds a stable terminal opposition or suffix, as in Experiments 1, 2, 6, and 12; weaker terminal control under heterogeneous loads and arrangements, as in Experiment 5; and early/block dominance when varying run boundaries encourage comparison of coherent chunks, as in Experiment 10. Tally agreement may modulate confidence, consistent with Experiment 3, without being able to reverse a recent commitment merely because the weighted coalition is strong. Raw switch count alone must not drive the mechanism, given the null arrangement contrasts in Experiments 7 and 9. A hierarchical implementation should use a small set of psychologically interpretable, subject-stable policy propensities and a parsimonious context gate, with preregistered cross-experiment predictions. This would create a genuinely stronger competitor to pi_3 rather than retaining pi_7 and tuning its global gate nearly to zero.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.859)

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
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.6196 (var=0.0088)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.1788 (var=0.0107)


---

### `pi_3` (overall score: 0.853)

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


---

### `pi_7` (overall score: 0.557)

**Description**
Load-Contingent Dual-Code Arbitration. While inspecting discriminating experts in instructed-validity order, a decision maker constructs two qualitatively different codes. The local code is a rapidly revised categorical commitment: each new discriminating reason almost completely replaces the previous commitment, so reasons outside the recent suffix become functionally absent rather than merely weakened traces. In parallel, the person can form a validity-weighted global coalition summary that preserves cumulative directional evidence without preserving cue order. Choice is governed by smooth arbitration between these codes. Global comparison is recruited only after conflict is sufficiently elaborate to warrant comparison, is favored when validity-weighted evidence forms a diagnostic coalition supported by an unweighted directional majority, and loses accessibility when subject-specific effective capacity is exceeded. Coalition selectivity is itself load-contingent: a coalition must be especially strong near the recruitment boundary, whereas weaker cumulative coalitions become usable inside the subject's manageable comparison window before becoming inaccessible under overload. Within that manageable window, global control receives a selective increment when the global coalition opposes the current local commitment, because only such disagreement warrants effortful comparison. Thus short or count-balanced conflicts are dominated by local revision, manageable majority-coalition conflicts can be decided cumulatively, and overloaded conflicts revert toward an uncertainty-attenuated recent commitment. Capacity, recruitment threshold, coalition selectivity, and local and global gains are stable subject characteristics shared across trials.

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
    # recruitment also requires an unweighted directional coalition. A
    # smooth, nonsaturating transform gives a one-vote majority only partial
    # diagnosticity, while preserving exact protection for count balance and
    # approaching full support only for genuinely lopsided coalitions.
    raw_count_coalition = abs(float(np.sum(directions))) / float(m)
    raw_count_coalition = float(np.clip(raw_count_coalition, 0.0, 1.0))
    count_coalition = (
        (1.0 - np.exp(-2.0 * raw_count_coalition))
        / (1.0 - np.exp(-2.0))
    )
    count_coalition = float(np.clip(count_coalition, 0.0, 1.0))

    # Coalition selectivity varies smoothly with effective load. Near the
    # subject's recruitment boundary, only a strong weighted coalition can
    # engage global comparison. Once load is clearly above that boundary but
    # still below effective capacity, weaker cumulative coalitions become
    # usable. Selectivity tightens again as overload approaches. The window is
    # defined relative to stable subject-specific thresholds rather than by a
    # fixed number of cues.
    comparison_x = np.clip(
        (float(m) - recruitment_threshold - 0.75) / temp, -60.0, 60.0
    )
    comparison_ready = 1.0 / (1.0 + np.exp(-comparison_x))
    pre_overload_x = np.clip(
        (capacity + 0.50 - float(m)) / temp, -60.0, 60.0
    )
    pre_overload_access = 1.0 / (1.0 + np.exp(-pre_overload_x))
    manageable_window = comparison_ready * pre_overload_access
    selectivity_cutoff = 0.56 - 0.30 * manageable_window

    weighted_selectivity_x = np.clip(
        (coalition_strength - selectivity_cutoff) / 0.12, -60.0, 60.0
    )
    weighted_selectivity = 1.0 / (1.0 + np.exp(-weighted_selectivity_x))
    diagnosticity = np.power(
        np.clip(coalition_strength, 0.0, 1.0), coalition_focus
    )
    diagnosticity *= count_coalition * weighted_selectivity
    diagnosticity = float(np.clip(diagnosticity, 0.0, 1.0))

    global_gate = float(np.clip(onset * accessibility * diagnosticity, 0.0, 1.0))

    # Effortful global arbitration is selectively useful when its cumulative
    # conclusion opposes the current local commitment. A bounded additive
    # increment can rescue a weak baseline gate inside the manageable window,
    # while count support, weighted selectivity, and accessibility suppress it
    # near recruitment, in balanced displays, and under overload.
    code_opposition = 0.5 * (
        1.0 - np.tanh(local_commitment * global_summary / 0.20)
    )
    opposition_window = manageable_window * manageable_window * pre_overload_access
    opposition_increment = (
        0.95
        * accessibility
        * accessibility
        * opposition_window
        * code_opposition
        * count_coalition
        * weighted_selectivity
    )
    global_gate += opposition_increment
    global_gate = float(np.clip(global_gate, 0.0, 1.0))

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
- recruitment_threshold: [3.1, 4.0]
- arbitration_temperature: [0.35, 0.75]
- local_retention: [0.0, 0.04]
- local_gain: [1.35, 2.15]
- global_gain: [2.0, 4.2]
- validity_sensitivity: [0.65, 1.35]
- coalition_focus: [0.75, 1.65]
- overload_attenuation: [0.8, 1.6]
- lapse_rate: [0.03, 0.14]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4944 (var=0.0204) vs this=0.5681 (var=0.0095)
- Experiment 2: real=0.1938 (var=0.0049) vs this=0.1940 (var=0.0026)
- Experiment 3: real=0.5750 (var=0.0026) vs this=0.7098 (var=0.0021)
- Experiment 4: real=0.6650 (var=0.0621) vs this=0.6104 (var=0.0134)
- Experiment 5: real=0.0033 (var=0.0049) vs this=0.0079 (var=0.0079)
- Experiment 6: real=0.3067 (var=0.0051) vs this=0.3404 (var=0.0071)
- Experiment 7: real=0.0000 (var=0.2016) vs this=0.0000 (var=0.2356)
- Experiment 8: real=0.1721 (var=0.0112) vs this=0.3131 (var=0.0029)
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.0638 (var=0.0071)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.8717 (var=0.0018)
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.0642 (var=0.0023)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.7800 (var=0.0127)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0537 -> ACCEPTED
- iter 2: loss=0.0546 -> REJECTED
- iter 3: loss=0.0549 -> REJECTED
- iter 4: loss=0.0683 -> REJECTED
- iter 5: loss=0.0558 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0537 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.5218 (var=0.0132) (Δ vs real +0.0274)
  - iter 2: 0.4750 (var=0.0108) (Δ vs real -0.0194)
  - iter 3: 0.5325 (var=0.0084) (Δ vs real +0.0381)
  - iter 4: 0.5814 (var=0.0140) (Δ vs real +0.0869)
  - iter 5 (most recent): 0.4760 (var=0.0116) (Δ vs real -0.0185)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0000 (var=0.0049)
- pi_2: 0.7368 (var=0.0343)
- pi_3: 0.4743 (var=0.0168)
- pi_4: 0.5279 (var=0.0111)
- pi_5: 0.5949 (var=0.0094)
- pi_6: 0.6093 (var=0.0085)
- pi_7: 0.5681 (var=0.0095)

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
  - iter 1: 0.1669 (var=0.0014) (Δ vs real -0.0269)
  - iter 2: 0.1856 (var=0.0025) (Δ vs real -0.0081)
  - iter 3: 0.1752 (var=0.0018) (Δ vs real -0.0185)
  - iter 4: 0.1804 (var=0.0024) (Δ vs real -0.0133)
  - iter 5 (most recent): 0.2246 (var=0.0027) (Δ vs real +0.0308)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4942 (var=0.0026)
- pi_1: 0.8694 (var=0.0103)
- pi_3: 0.1931 (var=0.0026)
- pi_4: 0.2060 (var=0.0019)
- pi_5: 0.1969 (var=0.0043)
- pi_6: 0.2027 (var=0.0056)
- pi_7: 0.1940 (var=0.0026)

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
  - iter 1: 0.6719 (var=0.0030) (Δ vs real +0.0969)
  - iter 2: 0.6356 (var=0.0026) (Δ vs real +0.0606)
  - iter 3: 0.6835 (var=0.0038) (Δ vs real +0.1085)
  - iter 4: 0.6906 (var=0.0030) (Δ vs real +0.1156)
  - iter 5 (most recent): 0.6550 (var=0.0032) (Δ vs real +0.0800)
**Other theories' values on this metric (for reference):**
- pi_3: 0.7129 (var=0.0033)
- pi_2: 0.1462 (var=0.0111)
- pi_1: 0.1419 (var=0.0079)
- pi_4: 0.5346 (var=0.0080)
- pi_5: 0.5996 (var=0.0042)
- pi_6: 0.6577 (var=0.0054)
- pi_7: 0.7098 (var=0.0021)

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
  - iter 1: 0.6633 (var=0.0073) (Δ vs real -0.0017)
  - iter 2: 0.6550 (var=0.0113) (Δ vs real -0.0100)
  - iter 3: 0.6633 (var=0.0082) (Δ vs real -0.0017)
  - iter 4: 0.6496 (var=0.0072) (Δ vs real -0.0154)
  - iter 5 (most recent): 0.6492 (var=0.0092) (Δ vs real -0.0158)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0075 (var=0.0116)
- pi_3: 0.5262 (var=0.0122)
- pi_1: -0.2525 (var=0.0072)
- pi_4: 0.6104 (var=0.0504)
- pi_5: 0.6737 (var=0.0181)
- pi_6: 0.6704 (var=0.0140)
- pi_7: 0.6104 (var=0.0134)

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
  - iter 1: -0.0150 (var=0.0089) (Δ vs real -0.0183)
  - iter 2: 0.0079 (var=0.0092) (Δ vs real +0.0046)
  - iter 3: 0.0046 (var=0.0107) (Δ vs real +0.0013)
  - iter 4: -0.0158 (var=0.0113) (Δ vs real -0.0192)
  - iter 5 (most recent): -0.0112 (var=0.0068) (Δ vs real -0.0146)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0150 (var=0.0104)
- pi_4: 0.4617 (var=0.0125)
- pi_1: 0.6621 (var=0.0603)
- pi_2: -0.0008 (var=0.0123)
- pi_5: 0.0071 (var=0.0075)
- pi_6: -0.0142 (var=0.0089)
- pi_7: 0.0079 (var=0.0079)

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
  - iter 1: 0.3025 (var=0.0059) (Δ vs real -0.0042)
  - iter 2: 0.3225 (var=0.0076) (Δ vs real +0.0158)
  - iter 3: 0.2933 (var=0.0053) (Δ vs real -0.0133)
  - iter 4: 0.2796 (var=0.0037) (Δ vs real -0.0271)
  - iter 5 (most recent): 0.2929 (var=0.0040) (Δ vs real -0.0137)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7804 (var=0.0054)
- pi_3: 0.3379 (var=0.0051)
- pi_1: 0.8433 (var=0.0097)
- pi_2: 0.6654 (var=0.0060)
- pi_5: 0.3113 (var=0.0065)
- pi_6: 0.3067 (var=0.0061)
- pi_7: 0.3404 (var=0.0071)

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
  - iter 1: 0.0000 (var=0.2400) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2244) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2176) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.2244) (Δ vs real +0.0000)
  - iter 5 (most recent): 0.0000 (var=0.2176) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0000 (var=0.1824)
- pi_5: 1.0000 (var=0.1924)
- pi_1: 0.0000 (var=0.1476)
- pi_2: 0.0000 (var=0.2016)
- pi_4: 0.0000 (var=0.0384)
- pi_6: 0.0000 (var=0.2100)
- pi_7: 0.0000 (var=0.2356)

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
  - iter 1: 0.2942 (var=0.0024) (Δ vs real +0.1221)
  - iter 2: 0.3319 (var=0.0026) (Δ vs real +0.1598)
  - iter 3: 0.3021 (var=0.0039) (Δ vs real +0.1300)
  - iter 4: 0.3190 (var=0.0021) (Δ vs real +0.1469)
  - iter 5 (most recent): 0.2827 (var=0.0022) (Δ vs real +0.1106)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7169 (var=0.0030)
- pi_3: 0.2754 (var=0.0030)
- pi_1: 0.3269 (var=0.0033)
- pi_2: 0.4110 (var=0.0016)
- pi_4: 0.4802 (var=0.0042)
- pi_6: 0.2250 (var=0.0059)
- pi_7: 0.3131 (var=0.0029)

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
  - iter 1: -0.0208 (var=0.0052) (Δ vs real +0.0117)
  - iter 2: -0.1279 (var=0.0093) (Δ vs real -0.0954)
  - iter 3: -0.0063 (var=0.0046) (Δ vs real +0.0263)
  - iter 4: -0.0258 (var=0.0063) (Δ vs real +0.0067)
  - iter 5 (most recent): 0.0167 (var=0.0075) (Δ vs real +0.0492)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0083 (var=0.0103)
- pi_6: 0.2096 (var=0.0101)
- pi_1: 0.7021 (var=0.0307)
- pi_2: 0.7054 (var=0.0384)
- pi_4: 0.0375 (var=0.0075)
- pi_5: 0.0108 (var=0.0053)
- pi_7: 0.0638 (var=0.0071)

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
  - iter 1: 0.8918 (var=0.0014) (Δ vs real +0.0170)
  - iter 2: 0.8866 (var=0.0020) (Δ vs real +0.0119)
  - iter 3: 0.8815 (var=0.0020) (Δ vs real +0.0067)
  - iter 4: 0.8887 (var=0.0014) (Δ vs real +0.0140)
  - iter 5 (most recent): 0.8942 (var=0.0009) (Δ vs real +0.0195)
**Other theories' values on this metric (for reference):**
- pi_6: 0.9281 (var=0.0013)
- pi_3: 0.7843 (var=0.0034)
- pi_1: 0.5635 (var=0.0023)
- pi_2: 0.8101 (var=0.0072)
- pi_4: 0.8645 (var=0.0034)
- pi_5: 0.9256 (var=0.0034)
- pi_7: 0.8717 (var=0.0018)

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
  - iter 1: 0.8771 (var=0.0024) (Δ vs real +0.0346)
  - iter 2: 0.8546 (var=0.0039) (Δ vs real +0.0121)
  - iter 3: 0.8671 (var=0.0030) (Δ vs real +0.0246)
  - iter 4: 0.8621 (var=0.0027) (Δ vs real +0.0196)
  - iter 5 (most recent): 0.8717 (var=0.0044) (Δ vs real +0.0292)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6863 (var=0.0047)
- pi_7: 0.0642 (var=0.0023)
- pi_1: 0.1583 (var=0.0130)
- pi_2: 0.1342 (var=0.0084)
- pi_4: 0.3117 (var=0.0174)
- pi_5: 0.6737 (var=0.0053)
- pi_6: 0.6196 (var=0.0088)

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
  - iter 1: 0.1475 (var=0.0038) (Δ vs real -0.0238)
  - iter 2: 0.1656 (var=0.0061) (Δ vs real -0.0056)
  - iter 3: 0.1431 (var=0.0026) (Δ vs real -0.0281)
  - iter 4: 0.1387 (var=0.0047) (Δ vs real -0.0325)
  - iter 5 (most recent): 0.1450 (var=0.0048) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7800 (var=0.0127)
- pi_3: 0.2650 (var=0.0077)
- pi_1: 0.8450 (var=0.0106)
- pi_2: 0.8638 (var=0.0097)
- pi_4: 0.3144 (var=0.0590)
- pi_5: 0.2125 (var=0.0050)
- pi_6: 0.1788 (var=0.0107)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Context-Calibrated Ordinal Event Policies. Decision makers inspect experts in instructed-validity order but encode the resulting discriminating sequence as ordinal events: an initial commitment, coherent directional runs, reversals, and a terminal commitment. Choice is generated by a context-sensitive selection between two subject-stable policies. The default terminal policy expresses the most recently attended commitment, whereas a coherent-chunk policy preserves the initial commitment when a moderately long, densely populated display is naturally represented as two opposing blocks. The chunk policy is recruited by conjunctions of density, coherent run structure, manageable load, and boundary regularities learned across trials—not by raw switch count alone. Sparse late suffixes or isolated terminal opposition foreground the terminal event and increase its confidence, even at high nominal feature counts. Equal-weight tally agreement can strengthen or weaken confidence in an already selected commitment but cannot reverse that commitment. Communicated validities determine inspection order and contribute only a strongly compressed confidence adjustment. Stable individual differences govern attention, recency and primacy propensities, chunk-switch threshold, contextual calibration, response gain, and lapse.

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

    # Exact expected direction of the most recently attended commitment.
    # A later attended event replaces the previous one; unattended events do
    # not. The no-commitment state contributes zero directional evidence.
    recency_direction = 0.0
    for direction in directions:
        recency_direction = (
            (1.0 - attention) * recency_direction + attention * direction
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
- confidence_gain: [1.75, 2.65]
- tally_confidence: [0.05, 0.22]
- validity_compression: [0.0, 0.10]
- lapse_rate: [0.05, 0.17]

`rationale`: This model replaces pi_7's validity-weighted global coalition with policies defined over ordinal sequence events. Its default is terminal commitment, preserving the strong recency needed for short conflicts and stable late opposition. A distinct primacy/chunk policy is recruited only by the conjunction of dense evidence, exactly two coherent directional blocks, and a subject-specific moderate-load window. This should produce early dominance for six-reason block displays such as Experiment 8 while returning to terminal commitment for longer overload sequences, sparse suffix displays, and short four-cue displays. The gate cannot be activated by switch count alone, addressing the null arrangement contrasts in Experiments 7 and 9. History is used only to estimate structural context—terminal foregrounding and variability in coherent block boundaries—so policy calibration can develop across a randomized block without learning from nonexistent correctness feedback. Sparse gaps before a stable terminal suffix increase terminal confidence, supporting Experiments 11 and 12 despite large nominal feature counts. Tally agreement changes confidence but is structurally prohibited from reversing the selected commitment, avoiding pi_7's catastrophic majority-coalition prediction in Experiment 12. Validity magnitudes are compressed almost completely and mainly establish inspection order. The resulting preregisterable predictions are that dense two-block displays near the chunk threshold elicit primacy, sparse or isolated late suffixes elicit recency regardless of total feature count, longer dense displays weaken commitment without producing switch-count contrasts, and tally manipulations affect certainty more than choice direction.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed ordinal-event/context-policy family and is a substantial improvement over pi_7, especially on the decisive sparse-suffix and terminal-opposition tests. It closely fits Experiments 4, 5, 6, 7, 9, and 10, and it captures the reversal of pi_7's catastrophic prediction in Experiment 12. Experiments 1, 2, 11, and 12 have only modest point-estimate errors. However, two systematic misses remain. In Experiment 8, terminal agreement is much too high (0.294 versus 0.172), showing that the dense six-cue, exactly two-block display does not recruit the coherent-chunk/primacy policy strongly enough. In Experiment 3, terminal agreement is also too high (0.672 versus 0.575), indicating excessive confidence in recency for heterogeneous, often sparse and reversal-rich displays. These are separable calibration failures: Experiment 8 needs stronger policy switching, whereas Experiment 3 needs weaker expression of terminal commitment rather than indiscriminate primacy. The simulated between-subject variances are also generally too small, conspicuously in Experiments 4 and 11, suggesting that the supposedly stable trait distributions produce behavior that is too homogeneous.
Rationale: Keep the current mechanism and tune it locally. First, strengthen and sharpen the chunk gate specifically around m=6 for dense, exactly two-run displays—e.g. raise the baseline context factor or primacy propensity within that narrow load window—so Experiment 8 approaches its very low terminal-choice rate without changing the strong terminal effects at m=4 or m>=7. Second, add a confidence attenuation for non-foregrounded displays that combine low density with multiple reversals; this should lower terminal agreement in Experiment 3 and should also move Experiment 2's overly low TTB rate slightly upward toward the data. Do not apply this attenuation to sparse edge-separated suffixes, since Experiments 11 and 12 depend on terminal foregrounding. Finally, modestly widen the subject-level gain, lapse, and policy-threshold ranges around their current means to recover human heterogeneity while preserving pooled predictions. Avoid a global reduction in recency: the near-exact fits in Experiments 4, 6, and 10 show that the default terminal policy is already well calibrated outside the two problematic structural contexts.

**Outcome of this advice:** iter 1 candidate loss=0.0537 -> iter 2 candidate loss=0.0546 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current iteration is faithful to the prescribed context-calibrated ordinal-event family, but it was rejected by the loss gate and therefore cannot be shipped. The changes produced several local improvements: Experiments 2, 3, 5, 11, and 12 moved closer to their observed point estimates. However, these gains were outweighed by regressions. Most importantly, Experiment 8 moved in the wrong direction: terminal agreement increased from 0.294 to 0.332 even though humans are at 0.172. Experiment 9 also deteriorated sharply from an essentially exact fit (-0.021 versus -0.033) to excessive opposition to the initial cue (-0.128). Experiments 1 and 6 became modestly worse as well. Thus the newly added heterogeneous-reversal attenuation and widened parameter ranges introduced unwanted effects without solving the dense six-event block case. Experiment 3 remains too terminal despite its improvement (0.636 versus 0.575). Across-subject variance also remains substantially too low in key experiments, especially Experiments 4, 8, 10, and 11.
Rationale: Build on the unchanged accepted iteration-1 base. My previous bundled recommendation—adding broad reversal attenuation, widening trait ranges, and strengthening six-event recruitment—was rejected, so do not repeat that package. In particular, remove the new diffuse-reversal confidence divisor: the accepted base already fit Experiment 9, whereas this modification created a large anti-initial bias and risks making arrangement complexity function like a switch-count rule. Also retain the accepted base's narrower global gain/lapse ranges rather than widening them globally. For Experiment 8, try a different in-family adjustment: calibrate the expression of the selected chunk policy rather than adding another broad or hard-coded recruitment boost. The current probability-level mixture leaves a substantial residual recency response even when the chunk gate is high. Give the coherent-chunk policy its own response gain or temperature, activated only for dense two-run displays in a boundary-diverse task context, so that an engaged chunk policy expresses the initial commitment more deterministically while leaving the terminal policy unchanged. Make this dependence context-derived from the learned distribution of run boundaries rather than solely an exact m=6 constant. This is a surgical way to lower Experiment 8 without disturbing the accepted fits in Experiments 4, 6, 9, 10, and the foregrounded suffix cases 11–12. After isolating that change, reassess Experiment 3; its remaining excess terminal agreement should not be attacked again with the rejected generic reversal penalty.

**Outcome of this advice:** iter 2 candidate loss=0.0546 -> iter 3 candidate loss=0.0549 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed context-calibrated ordinal-event family, but the current surgical change failed the accept gate and did not solve its target. Experiment 8 remains the clearest failure: predicted terminal agreement is 0.302 versus 0.172 observed, slightly worse than the accepted base's 0.294. The added chunk-expression gain therefore has little leverage. Once the primacy branch is already fairly decisive, increasing its temperature cannot eliminate choices generated by the residual recency branch in the probability-level mixture. Experiment 3 also remains substantially too terminal (0.684 versus 0.575), while Experiment 1 is somewhat too tally-sensitive (0.533 versus 0.494). Smaller misses occur in Experiments 2, 6, 11, and 12. In contrast, Experiments 4, 5, 7, 9, and 10 remain close, confirming that the basic terminal policy, load attenuation, and protection against raw switch-count effects are sound. Between-subject variance is still too low in several important cases, especially Experiments 4, 8, 10, and 11, but the previously attempted global widening of trait ranges was rejected.
Rationale: Both prior remedies for Experiment 8 were rejected: the bundled broad gate/reversal/heterogeneity adjustment and the present chunk-expression-temperature adjustment. Do not repeat either. Build the next attempt on the unchanged accepted iteration-1 base and make one isolated in-family correction to policy selection itself. The current multiplicative primacy gate does not cleanly implement a subject-stable policy-switch threshold; it combines several factors into a graded probability mixture, leaving substantial recency mass even in the boundary-diverse dense two-block context. Replace that normalization with a parsimonious thresholded logistic or two-policy softmax over chunk evidence versus terminal evidence, with a moderately steep gate temperature and a subject-stable threshold. Boundary diversity should shift the relative policy log odds rather than merely multiply the gate, while density, the moderate-load window, and terminal foregrounding retain their present protective roles. This should allow Experiment 8's context to cross into chunk selection for most subjects without affecting sparse Experiment 11 displays, foregrounded Experiment 12 suffixes, m=4 Experiment 10 displays, or overloaded m>=7 cases. A narrow subject distribution over this policy threshold can also increase Experiment 8 heterogeneity without reopening the rejected global gain/lapse widening. Test this gate correction alone; do not simultaneously add another generic reversal penalty for Experiment 3, because that earlier intervention damaged Experiment 9.

**Outcome of this advice:** iter 3 candidate loss=0.0549 -> iter 4 candidate loss=0.0683 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed context-calibrated ordinal-event family, but it was decisively rejected by the accept gate (loss 0.0683 versus the accepted base's 0.0537). The thresholded policy gate failed at its intended target: Experiment 8 terminal agreement rose to 0.319 versus 0.172 observed, worse than the accepted base's 0.294. It also caused broad regressions rather than an isolated change. Experiment 1 is now excessively differentiated by tally context (0.581 versus 0.494), Experiment 3 remains far too terminal (0.691 versus 0.575), and terminal capture is too strong in Experiments 6 and 12, yielding early-choice and majority-choice rates below the data. Experiments 4, 5, 7, 9, 10, and 11 remain reasonably close, but they do not compensate for these failures. Subject-level variance also remains markedly too low in Experiments 4, 8, 10, and 11. Three successive attempts to solve Experiment 8 by increasing chunk recruitment, sharpening chunk expression, or replacing the gate with a steep threshold have all been rejected, so further pushing of the six-cue chunk gate is not supported.
Rationale: Build the next candidate on the unchanged accepted iteration-1 base. Do not repeat my rejected recommendation to sharpen or strengthen the dense two-block policy gate, and do not add another chunk-specific response temperature. Instead, leave Experiment 8's gate unchanged for this iteration and target the other shared residual—excessive commitment under structurally heterogeneous task contexts—with one isolated, in-family calibration. Add a context-level uncertainty or gain adjustment based on the across-trial dispersion of discriminating-cue load or density, not reversal count. High load heterogeneity should modestly reduce expression of whichever ordinal policy is selected, particularly for currently manageable, non-foregrounded displays. This is directly motivated by Experiment 3's varying 3/5/7/9-event displays and should reduce its excessive terminal agreement; it may also reduce Experiment 1's excessive contrast. Protect overloaded current displays and foregrounded edge-separated suffixes so that the accepted fits in Experiments 6, 11, and 12 are not weakened, and keep Experiment 9 protected by avoiding any switch-count or reversal-based term. Use a modest attenuation with strong shrinkage during early history and test it alone. Do not simultaneously widen global trait ranges, since that earlier package was rejected. Experiment 8 should remain acknowledged as unresolved, but the evidence now favors improving aggregate fit through a different context-calibration knob rather than a fourth attempt to force its chunk gate.

**Outcome of this advice:** iter 4 candidate loss=0.0683 -> iter 5 candidate loss=0.0558 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed context-calibrated ordinal-event family, but it was rejected by the accept gate (loss 0.0558 versus the accepted base's 0.0537) and should not be shipped. The new load-dispersion uncertainty term modestly improved its intended targets relative to the accepted base: Experiment 3 terminal agreement fell from 0.672 to 0.655, and Experiment 1 moved from 0.522 to 0.476 around the observed 0.494. Those gains were insufficient and came with regressions. Experiment 2 crossed from too little TTB agreement to too much (0.225 versus 0.194), Experiment 9 moved away from its near-exact accepted-base fit (-0.021) to 0.017 versus -0.033, and Experiments 4, 5, 6, 10, 11, and 12 generally deteriorated slightly. The main systematic failures remain excessive terminal agreement in Experiment 3 (0.655 versus 0.575) and especially Experiment 8 (0.283 versus 0.172). The latter improved only marginally and remains the largest point-estimate error. Between-subject variability is also much too low in Experiments 4, 8, 10, and 11, although previous global range widening was rejected. Overall, the ordinal-event account explains the qualitative cross-experiment pattern well—particularly Experiments 4, 6, 7, 10, 11, and 12—but its near-unit attention/replacement rate makes terminal commitment too automatic in dense or moderately loaded sequences.
Rationale: Build on the unchanged accepted iteration-1 base. My previous recommendation to add task-level load-dispersion uncertainty was rejected, so remove that adjustment rather than increasing it. Also do not revisit the three rejected attempts to strengthen, sharpen, or temperature-scale the chunk-policy gate, and do not globally widen trait ranges. Try a different isolated in-family knob: context-calibrate the event-replacement rate itself. The current attention range of 0.91–0.99 makes each attended direction almost completely overwrite the prior commitment, leaving little ordinal persistence before the terminal event. Introduce a modest, smooth reduction in effective replacement attention for non-foregrounded, moderate-load sequences (roughly the five-to-six-event region), while retaining the existing high rate for sparse foregrounded suffixes, four-event displays, and overload conditions. This would let preceding coherent commitments persist in Experiment 8 without directly increasing chunk selection, and could reduce excessive terminal control in the five-event portions of Experiment 3. It may also move the accepted base's low Experiment 2 TTB rate upward toward the observed value. Keep the adjustment dependent on load/density and terminal foregrounding, not reversal or switch count, preserving the null arrangement results in Experiments 7 and 9. Use a shallow modulation rather than a hard condition—for example, allow effective attention to fall only into approximately 0.82–0.94 at the center of the moderate-load, non-foregrounded window while retaining the accepted baseline elsewhere—and test this change alone.

**Outcome of this advice:** iter 5 candidate loss=0.0558 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Context-Calibrated Ordinal Event Policies. Decision makers inspect experts in instructed-validity order but encode the resulting discriminating sequence as ordinal events: an initial commitment, coherent directional runs, reversals, and a terminal commitment. Choice is generated by a context-sensitive selection between two subject-stable policies. The default terminal policy expresses the most recently attended commitment, whereas a coherent-chunk policy preserves the initial commitment when a moderately long, densely populated display is naturally represented as two opposing blocks. The chunk policy is recruited by conjunctions of density, coherent run structure, manageable load, and boundary regularities learned across trials—not by raw switch count alone. Sparse late suffixes or isolated terminal opposition foreground the terminal event and increase its confidence, even at high nominal feature counts. Equal-weight tally agreement can strengthen or weaken confidence in an already selected commitment but cannot reverse that commitment. Communicated validities determine inspection order and contribute only a strongly compressed confidence adjustment. Stable individual differences govern attention, recency and primacy propensities, chunk-switch threshold, contextual calibration, response gain, and lapse. In non-foregrounded moderate-load sequences, replacement of an existing commitment is slightly less complete, allowing preceding ordinal commitments to persist without changing policy selection.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Ordinal event policy expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    attention = float(parameters[\"attention_rate\"])\n    recency_propensity = float(parameters[\"recency_propensity\"])\n    primacy_propensity = float(parameters[\"primacy_propensity\"])\n    chunk_threshold = float(parameters[\"chunk_switch_threshold\"])\n    context_sensitivity = float(parameters[\"context_sensitivity\"])\n    load_tolerance = float(parameters[\"load_tolerance\"])\n    confidence_gain = float(parameters[\"confidence_gain\"])\n    tally_confidence = float(parameters[\"tally_confidence\"])\n    validity_compression = float(parameters[\"validity_compression\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    diff = stim[0] - stim[1]\n    discriminating = [int(j) for j in cue_order if diff[j] != 0]\n    m = len(discriminating)\n    if m == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    directions = np.asarray(\n        [float(np.sign(diff[j])) for j in discriminating], dtype=float\n    )\n\n    def sequence_summary(a, b):\n        a = np.asarray(a, dtype=float)\n        b = np.asarray(b, dtype=float)\n        if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size == 0:\n            return None\n        d = a - b\n        idx = np.flatnonzero(d != 0)\n        if idx.size == 0:\n            return None\n        ds = np.sign(d[idx]).astype(float)\n        run_lengths = []\n        current = 1\n        for k in range(1, ds.size):\n            if ds[k] == ds[k - 1]:\n                current += 1\n            else:\n                run_lengths.append(current)\n                current = 1\n        run_lengths.append(current)\n        gaps = np.diff(idx) if idx.size > 1 else np.asarray([], dtype=float)\n        max_gap = float(np.max(gaps) - 1.0) if gaps.size else 0.0\n        max_gap = max(0.0, max_gap)\n        return {\n            \"m\": float(idx.size),\n            \"density\": float(idx.size) / float(a.size),\n            \"n_runs\": float(len(run_lengths)),\n            \"boundary\": float(run_lengths[0]) / float(idx.size),\n            \"opposition\": float(ds[0] != ds[-1]),\n            \"foreground\": max_gap / float(max(1, a.size - 1)),\n            \"terminal_at_edge\": float(idx[-1] == a.size - 1)\n        }\n\n    summaries = []\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n    for i in range(n_past):\n        summary = sequence_summary(past_a[i], past_b[i])\n        if summary is not None:\n            summaries.append(summary)\n\n    current_summary = sequence_summary(stim[0], stim[1])\n    if current_summary is not None:\n        summaries.append(current_summary)\n\n    # Context statistics are based on structural distributions, not choices.\n    # Shrink early estimates toward neutral priors so the model behaves\n    # sensibly before it has sampled the task's display distribution.\n    if summaries:\n        sample_weight = min(1.0, len(summaries) / 12.0)\n        opposition_rate_raw = float(np.mean([s[\"opposition\"] for s in summaries]))\n        foreground_rate_raw = float(np.mean([\n            s[\"foreground\"] * s[\"terminal_at_edge\"] for s in summaries\n        ]))\n        coherent_boundaries = [\n            s[\"boundary\"] for s in summaries\n            if s[\"density\"] >= 0.85 and s[\"n_runs\"] == 2.0\n        ]\n        if len(coherent_boundaries) >= 2:\n            boundary_diversity_raw = min(\n                1.0, 4.0 * float(np.std(coherent_boundaries))\n            )\n        else:\n            boundary_diversity_raw = 0.0\n        opposition_rate = (\n            sample_weight * opposition_rate_raw + (1.0 - sample_weight) * 0.5\n        )\n        foreground_rate = sample_weight * foreground_rate_raw\n        boundary_diversity = sample_weight * boundary_diversity_raw\n    else:\n        opposition_rate = 0.5\n        foreground_rate = 0.0\n        boundary_diversity = 0.0\n\n    density = float(m) / float(max(1, n_features))\n\n    run_lengths = []\n    current_run = 1\n    for k in range(1, m):\n        if directions[k] == directions[k - 1]:\n            current_run += 1\n        else:\n            run_lengths.append(current_run)\n            current_run = 1\n    run_lengths.append(current_run)\n    n_runs = len(run_lengths)\n    coherent_two_block = 1.0 if n_runs == 2 else 0.0\n\n    physical_positions = np.sort(np.asarray(discriminating, dtype=int))\n    if physical_positions.size > 1:\n        empty_gaps = np.diff(physical_positions) - 1\n        largest_gap = max(0.0, float(np.max(empty_gaps)))\n    else:\n        largest_gap = 0.0\n    foreground = largest_gap / float(max(1, n_features - 1))\n    terminal_at_edge = float(np.max(physical_positions) == n_features - 1)\n    terminal_foreground = foreground * terminal_at_edge\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # A coherent-chunk policy is available only in a moderate-load window.\n    # Both sides of the window are smooth and subject-specific. Dense\n    # two-block structure is a necessary conjunction, so merely adding\n    # switches cannot recruit primacy. Across-trial boundary diversity makes\n    # block comparison somewhat more likely without making it mandatory.\n    chunk_onset = sigmoid((float(m) - chunk_threshold) / 0.35)\n    chunk_offset = sigmoid((chunk_threshold + 1.35 - float(m)) / 0.35)\n    density_gate = sigmoid((density - 0.88) / 0.045)\n    context_factor = 0.72 + context_sensitivity * boundary_diversity\n    primacy_gate = (\n        primacy_propensity\n        * coherent_two_block\n        * chunk_onset\n        * chunk_offset\n        * density_gate\n        * context_factor\n    )\n    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))\n\n    # Repeated sparse terminal opposition suppresses chunk recruitment and\n    # restores terminal capture. This is especially important when a late\n    # suffix is separated from the initial block by nondiscriminating cues.\n    terminal_context = np.clip(\n        terminal_foreground\n        + context_sensitivity * foreground_rate\n        + 0.20 * opposition_rate * terminal_foreground,\n        0.0,\n        1.0\n    )\n    primacy_gate *= (1.0 - 0.85 * terminal_context)\n    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))\n\n    # Replacement is modestly less complete in the subject's moderate-load\n    # window. Density scales the adjustment smoothly, while foregrounded\n    # terminal suffixes retain the baseline near-complete replacement rate.\n    replacement_onset = sigmoid(\n        (float(m) - (chunk_threshold - 0.50)) / 0.30\n    )\n    replacement_offset = sigmoid(\n        (chunk_threshold + 1.00 - float(m)) / 0.30\n    )\n    moderate_replacement_window = replacement_onset * replacement_offset\n    density_modulation = 0.70 + 0.30 * density\n    replacement_reduction = (\n        0.10\n        * moderate_replacement_window\n        * density_modulation\n        * (1.0 - terminal_context)\n    )\n    effective_replacement = float(np.clip(\n        attention - replacement_reduction, 0.0, 1.0\n    ))\n\n    # Exact expected direction of the most recently attended commitment.\n    # A later attended event replaces the previous one; unattended events do\n    # not. The no-commitment state contributes zero directional evidence.\n    recency_direction = 0.0\n    for direction in directions:\n        recency_direction = (\n            (1.0 - effective_replacement) * recency_direction\n            + effective_replacement * direction\n        )\n\n    # Exact expected direction of the first attended commitment.\n    primacy_direction = 0.0\n    probability_uncommitted = 1.0\n    for direction in directions:\n        encoded_here = probability_uncommitted * attention\n        primacy_direction += encoded_here * direction\n        probability_uncommitted *= (1.0 - attention)\n\n    recency_direction *= recency_propensity\n\n    # Nominal display load weakens expression, while coherent runs and a\n    # foregrounded late suffix restore confidence. Run coherence affects\n    # confidence, not policy recruitment, avoiding a raw-switch-count rule.\n    longest_run_fraction = float(max(run_lengths)) / float(m)\n    display_overload = max(0.0, float(n_features) - load_tolerance)\n    overload_divisor = 1.0 + 0.22 * display_overload\n    overload_divisor *= 1.0 + 0.10 * max(0.0, float(m) - load_tolerance)\n\n    coherence_support = max(0.0, longest_run_fraction - 0.35)\n    confidence_multiplier = 1.0 + 0.90 * coherence_support\n    confidence_multiplier += 3.0 * terminal_foreground\n    confidence_multiplier += 1.25 * context_sensitivity * foreground_rate\n\n    # Validity magnitudes have only compressed influence. Their main role was\n    # already to determine inspection order.\n    v = np.clip(validities[discriminating], 0.5001, 0.999)\n    log_odds = np.log(v / (1.0 - v))\n    mean_log_odds = max(float(np.mean(log_odds)), 1e-8)\n    terminal_reliability = max(float(log_odds[-1] / mean_log_odds), 1e-6)\n    validity_factor = float(\n        np.power(np.clip(terminal_reliability, 0.25, 4.0), validity_compression)\n    )\n\n    # Tally is a confidence cue only. It can reinforce or attenuate the\n    # terminal commitment but cannot change its sign or install a weighted\n    # global coalition as the selected response.\n    tally_direction = float(np.sign(np.sum(directions)))\n    terminal_direction = float(directions[-1])\n    if tally_direction == 0.0:\n        tally_factor = 1.0\n    elif tally_direction == terminal_direction:\n        tally_factor = 1.0 + tally_confidence\n    else:\n        tally_factor = 1.0 - 0.30 * tally_confidence\n\n    gain = (\n        confidence_gain\n        * confidence_multiplier\n        * validity_factor\n        * tally_factor\n        / max(overload_divisor, 1e-8)\n    )\n    gain = float(np.clip(gain, 0.05, 12.0))\n\n    p_a_recency = sigmoid(gain * recency_direction)\n    p_a_primacy = sigmoid(gain * primacy_direction)\n    p_a_core = (\n        (1.0 - primacy_gate) * p_a_recency\n        + primacy_gate * p_a_primacy\n    )\n\n    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / len(probs)\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "attention_rate": "[0.91, 0.99]",
    "recency_propensity": "[0.88, 1.0]",
    "primacy_propensity": "[0.76, 0.94]",
    "chunk_switch_threshold": "[5.05, 5.65]",
    "context_sensitivity": "[0.35, 0.75]",
    "load_tolerance": "[4.4, 5.8]",
    "confidence_gain": "[1.75, 2.65]",
    "tally_confidence": "[0.05, 0.22]",
    "validity_compression": "[0.0, 0.10]",
    "lapse_rate": "[0.05, 0.17]"
  },
  "rationale": "This is an isolated edit to the accepted iteration-1 model. The policy gate, confidence mechanism, parameter ranges, tally rule, and validity treatment are unchanged. Only the replacement rate used by the recency branch is context-calibrated. A smooth subject-relative window centered near five to six discriminating events reduces replacement by at most about 0.10, with a mild density scaling. Consequently, preceding commitments can persist in Experiment 8's dense six-event blocks and in moderate-load portions of Experiment 3 without another rejected attempt to force chunk-policy recruitment. The adjustment is negligible for four-event displays and overloaded sequences, while terminal foregrounding suppresses it for sparse edge-separated suffixes, protecting Experiments 10–12 and 6. Because the modulation uses load, density, and foregrounding rather than reversals or switch count, it also preserves the null arrangement predictions in Experiments 7 and 9."
}
```

## Usage

```json
{
  "input_tokens": 28494,
  "output_tokens": 5088
}
```
