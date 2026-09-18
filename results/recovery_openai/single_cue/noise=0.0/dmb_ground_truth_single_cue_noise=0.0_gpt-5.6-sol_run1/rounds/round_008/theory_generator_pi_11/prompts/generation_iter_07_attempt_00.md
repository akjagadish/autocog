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
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 2 (= `pi_10`).

Replace pi_10 with a theory of distributed, continuously weighted serial evidence integration. Each attended discriminating cue should contribute signed, compressed log-odds evidence to a leaky or interference-limited decision state, with a smooth serial-position accessibility gradient and divisive normalization by total conflicting support. This gives later cues a genuine but graded retrieval advantage without exact overwrite. Terminal influence should depend continuously on terminal reliability, opposing reliability-weighted coalition strength, and total interference—not on prefix coherence, switch count, singleton status, or a hard-coded 5–6 cue window. Load should primarily increase readout noise or interference, while unanimous evidence should show only a small vigilance decline. Stable subject heterogeneity can enter through recency-gradient strength, leak/interference, validity compression, attention, decision gain, and lapse rate, but the mechanism should remain stimulus-local and history-free unless future data establish learning. Such a theory should predict near-zero coherent-versus-fragmented and localized-window contrasts in Experiments 1, 2, and 7; preserve tally modulation in Experiment 3; permit very strong terminal control in Experiments 6, 12, 13, and 15; allow strong opposing coalitions to suppress terminal choices in Experiments 10 and 14; and avoid pi_3’s excessive unanimous-evidence fatigue in Experiment 17. It would therefore compete with pi_3 on its successful recency phenomena while explaining why recency is graded rather than an obligatory overwrite process.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.730)

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


---

### `pi_6` (overall score: 0.637)

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
- Experiment 13: real=0.8810 (var=0.0070) vs this=0.7470 (var=0.0062)
- Experiment 14: real=0.0192 (var=0.0062) vs this=-0.0137 (var=0.0073)
- Experiment 15: real=0.0471 (var=0.0135) vs this=0.3079 (var=0.0057)
- Experiment 16: real=0.0311 (var=0.0170) vs this=0.0001 (var=0.0112)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0043)
- Experiment 18: real=0.0262 (var=0.0075) vs this=0.0356 (var=0.0134)


---

### `pi_8` (overall score: 0.602)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1710 -> ACCEPTED
- iter 2: loss=0.1577 -> ACCEPTED
- iter 3: loss=0.1668 -> REJECTED
- iter 4: loss=0.2204 -> REJECTED
- iter 5: loss=0.1518 -> ACCEPTED
- iter 6: loss=0.1538 -> REJECTED
- iter 7: loss=0.2066 -> REJECTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.1518 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.8254 (var=0.0069) (Δ vs real +0.3310)
  - iter 2: 0.8953 (var=0.0027) (Δ vs real +0.4008)
  - iter 3: 0.8850 (var=0.0041) (Δ vs real +0.3906)
  - iter 4: 0.8618 (var=0.0033) (Δ vs real +0.3674)
  - iter 5: 0.9118 (var=0.0039) (Δ vs real +0.4174)
  - iter 6: 0.9044 (var=0.0037) (Δ vs real +0.4100)
  - iter 7 (most recent): 0.8940 (var=0.0049) (Δ vs real +0.3996)
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
  - iter 1: 0.4060 (var=0.0097) (Δ vs real +0.2123)
  - iter 2: 0.2065 (var=0.0039) (Δ vs real +0.0127)
  - iter 3: 0.2402 (var=0.0064) (Δ vs real +0.0465)
  - iter 4: 0.3331 (var=0.0054) (Δ vs real +0.1394)
  - iter 5: 0.2310 (var=0.0061) (Δ vs real +0.0373)
  - iter 6: 0.2042 (var=0.0051) (Δ vs real +0.0104)
  - iter 7 (most recent): 0.1850 (var=0.0048) (Δ vs real -0.0088)
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
  - iter 1: 0.7154 (var=0.0036) (Δ vs real +0.1404)
  - iter 2: 0.6898 (var=0.0056) (Δ vs real +0.1148)
  - iter 3: 0.6548 (var=0.0060) (Δ vs real +0.0798)
  - iter 4: 0.5952 (var=0.0039) (Δ vs real +0.0202)
  - iter 5: 0.6640 (var=0.0057) (Δ vs real +0.0890)
  - iter 6: 0.6771 (var=0.0054) (Δ vs real +0.1021)
  - iter 7 (most recent): 0.6665 (var=0.0049) (Δ vs real +0.0915)
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
  - iter 1: 0.4888 (var=0.0190) (Δ vs real -0.1763)
  - iter 2: 0.6392 (var=0.0105) (Δ vs real -0.0258)
  - iter 3: 0.5696 (var=0.0136) (Δ vs real -0.0954)
  - iter 4: 0.3729 (var=0.0157) (Δ vs real -0.2921)
  - iter 5: 0.6000 (var=0.0107) (Δ vs real -0.0650)
  - iter 6: 0.6433 (var=0.0131) (Δ vs real -0.0217)
  - iter 7 (most recent): 0.6242 (var=0.0142) (Δ vs real -0.0408)
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
  - iter 1: -0.0983 (var=0.0069) (Δ vs real -0.1017)
  - iter 2: -0.0650 (var=0.0046) (Δ vs real -0.0683)
  - iter 3: -0.0979 (var=0.0081) (Δ vs real -0.1012)
  - iter 4: -0.1479 (var=0.0079) (Δ vs real -0.1512)
  - iter 5: -0.0675 (var=0.0061) (Δ vs real -0.0708)
  - iter 6: -0.0629 (var=0.0050) (Δ vs real -0.0663)
  - iter 7 (most recent): -0.1225 (var=0.0077) (Δ vs real -0.1258)
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
  - iter 1: 0.1263 (var=0.0030) (Δ vs real -0.1804)
  - iter 2: 0.0650 (var=0.0015) (Δ vs real -0.2417)
  - iter 3: 0.0742 (var=0.0013) (Δ vs real -0.2325)
  - iter 4: 0.1817 (var=0.0067) (Δ vs real -0.1250)
  - iter 5: 0.0508 (var=0.0014) (Δ vs real -0.2558)
  - iter 6: 0.0625 (var=0.0021) (Δ vs real -0.2442)
  - iter 7 (most recent): 0.0617 (var=0.0011) (Δ vs real -0.2450)
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
  - iter 1: 0.0000 (var=0.2484) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2244) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.2400) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.1824) (Δ vs real +1.0000)
  - iter 5: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 7 (most recent): 1.0000 (var=0.2436) (Δ vs real +1.0000)
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
  - iter 1: 0.1615 (var=0.0045) (Δ vs real -0.0106)
  - iter 2: 0.0765 (var=0.0011) (Δ vs real -0.0956)
  - iter 3: 0.1119 (var=0.0027) (Δ vs real -0.0602)
  - iter 4: 0.1658 (var=0.0028) (Δ vs real -0.0063)
  - iter 5: 0.0902 (var=0.0016) (Δ vs real -0.0819)
  - iter 6: 0.0856 (var=0.0018) (Δ vs real -0.0865)
  - iter 7 (most recent): 0.1090 (var=0.0022) (Δ vs real -0.0631)
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
  - iter 1: 0.1250 (var=0.0120) (Δ vs real +0.1575)
  - iter 2: 0.1771 (var=0.0082) (Δ vs real +0.2096)
  - iter 3: 0.2125 (var=0.0143) (Δ vs real +0.2450)
  - iter 4: 0.1758 (var=0.0065) (Δ vs real +0.2083)
  - iter 5: 0.2071 (var=0.0101) (Δ vs real +0.2396)
  - iter 6: 0.1792 (var=0.0081) (Δ vs real +0.2117)
  - iter 7 (most recent): 0.2067 (var=0.0076) (Δ vs real +0.2392)
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
  - iter 1: 0.8881 (var=0.0026) (Δ vs real +0.0133)
  - iter 2: 0.9419 (var=0.0007) (Δ vs real +0.0671)
  - iter 3: 0.9203 (var=0.0013) (Δ vs real +0.0455)
  - iter 4: 0.9189 (var=0.0010) (Δ vs real +0.0441)
  - iter 5: 0.9404 (var=0.0009) (Δ vs real +0.0657)
  - iter 6: 0.9464 (var=0.0010) (Δ vs real +0.0716)
  - iter 7 (most recent): 0.9364 (var=0.0014) (Δ vs real +0.0616)
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
  - iter 1: 0.3846 (var=0.0213) (Δ vs real -0.4579)
  - iter 2: 0.5108 (var=0.0160) (Δ vs real -0.3317)
  - iter 3: 0.5604 (var=0.0139) (Δ vs real -0.2821)
  - iter 4: 0.5429 (var=0.0132) (Δ vs real -0.2996)
  - iter 5: 0.5446 (var=0.0060) (Δ vs real -0.2979)
  - iter 6: 0.5746 (var=0.0074) (Δ vs real -0.2679)
  - iter 7 (most recent): 0.4892 (var=0.0086) (Δ vs real -0.3533)
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
  - iter 1: 0.3144 (var=0.0136) (Δ vs real +0.1431)
  - iter 2: 0.1025 (var=0.0037) (Δ vs real -0.0688)
  - iter 3: 0.1519 (var=0.0079) (Δ vs real -0.0194)
  - iter 4: 0.1831 (var=0.0076) (Δ vs real +0.0119)
  - iter 5: 0.0931 (var=0.0034) (Δ vs real -0.0781)
  - iter 6: 0.1031 (var=0.0042) (Δ vs real -0.0681)
  - iter 7 (most recent): 0.1844 (var=0.0102) (Δ vs real +0.0131)
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
  - iter 1: 0.7553 (var=0.0035) (Δ vs real -0.1257)
  - iter 2: 0.8047 (var=0.0023) (Δ vs real -0.0763)
  - iter 3: 0.7940 (var=0.0034) (Δ vs real -0.0870)
  - iter 4: 0.7616 (var=0.0041) (Δ vs real -0.1194)
  - iter 5: 0.8098 (var=0.0018) (Δ vs real -0.0712)
  - iter 6: 0.8248 (var=0.0029) (Δ vs real -0.0562)
  - iter 7 (most recent): 0.7829 (var=0.0021) (Δ vs real -0.0981)
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
  - iter 1: 0.0125 (var=0.0080) (Δ vs real -0.0067)
  - iter 2: 0.0058 (var=0.0115) (Δ vs real -0.0133)
  - iter 3: -0.0283 (var=0.0083) (Δ vs real -0.0475)
  - iter 4: 0.0146 (var=0.0145) (Δ vs real -0.0046)
  - iter 5: 0.0229 (var=0.0101) (Δ vs real +0.0037)
  - iter 6: -0.0142 (var=0.0116) (Δ vs real -0.0333)
  - iter 7 (most recent): 0.0225 (var=0.0102) (Δ vs real +0.0033)
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
  - iter 1: 0.0027 (var=0.0024) (Δ vs real -0.0444)
  - iter 2: 0.0042 (var=0.0023) (Δ vs real -0.0429)
  - iter 3: 0.0090 (var=0.0029) (Δ vs real -0.0381)
  - iter 4: -0.0046 (var=0.0031) (Δ vs real -0.0517)
  - iter 5: -0.0006 (var=0.0024) (Δ vs real -0.0477)
  - iter 6: 0.0050 (var=0.0039) (Δ vs real -0.0421)
  - iter 7 (most recent): -0.0006 (var=0.0029) (Δ vs real -0.0477)
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
  - iter 1: 0.0165 (var=0.0056) (Δ vs real -0.0146)
  - iter 2: -0.0011 (var=0.0107) (Δ vs real -0.0322)
  - iter 3: 0.0040 (var=0.0092) (Δ vs real -0.0272)
  - iter 4: -0.0191 (var=0.0127) (Δ vs real -0.0502)
  - iter 5: 0.0129 (var=0.0107) (Δ vs real -0.0182)
  - iter 6: 0.0161 (var=0.0099) (Δ vs real -0.0150)
  - iter 7 (most recent): 0.0056 (var=0.0108) (Δ vs real -0.0255)
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
  - iter 1: 0.0000 (var=0.0022) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0010) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0112) (Δ vs real +0.0000)
  - iter 5: 0.0000 (var=0.0041) (Δ vs real +0.0000)
  - iter 6: 0.0000 (var=0.0060) (Δ vs real +0.0000)
  - iter 7 (most recent): 0.0000 (var=0.0022) (Δ vs real +0.0000)
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
  - iter 1: -0.0062 (var=0.0069) (Δ vs real -0.0325)
  - iter 2: -0.0031 (var=0.0179) (Δ vs real -0.0294)
  - iter 3: -0.0006 (var=0.0128) (Δ vs real -0.0269)
  - iter 4: -0.0325 (var=0.0117) (Δ vs real -0.0587)
  - iter 5: -0.0106 (var=0.0110) (Δ vs real -0.0369)
  - iter 6: 0.0013 (var=0.0159) (Δ vs real -0.0250)
  - iter 7 (most recent): -0.0000 (var=0.0104) (Δ vs real -0.0263)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Reliability-Compressed Recurrent Evidence Field. Decision makers inspect discriminating experts in descending instructed-validity order and encode each attended rating as signed, compressed log-odds evidence. Evidence remains distributed across cue traces, but accessibility declines smoothly with ordinal distance from response, producing graded recency rather than categorical replacement. Choice depends on the signed sum of accessible traces. Simultaneously active support for both products generates divisive interference, with an additional bounded increase when the accessible recurrent state is genuinely near balance. Increasing evidence load raises readout noise, especially under conflict. Cue-attention uncertainty contributes additional encoding noise, while a small vigilance decrement allows only a modest decline for long unanimous displays. Thus terminal control emerges continuously from terminal reliability, the accessibility gradient, opposing reliability-weighted coalition strength, and total interference. The mechanism has no sequence-class rules, physical-gap effects, fixed capacity boundary, or cross-trial learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recurrent evidence field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    compression = float(parameters["validity_compression"])
    retention = float(parameters["memory_retention"])
    attention = float(parameters["cue_attention"])
    conflict_normalization = float(parameters["conflict_normalization"])
    near_balance_normalization = float(parameters["near_balance_normalization"])
    load_noise = float(parameters["load_noise"])
    vigilance_decline = float(parameters["vigilance_decline"])
    decision_gain = float(parameters["decision_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is converted to normative log-odds and then
    # strongly compressed. Normalizing by the experiment's typical positive
    # log-odds preserves its reliability profile without allowing arbitrary
    # differences in scale to determine response gain.
    v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    positive = log_odds[log_odds > 0.0]
    reliability_scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0
    relative_reliability = np.clip(
        log_odds / reliability_scale, 1e-4, 20.0
    )
    cue_strength = np.power(relative_reliability, compression)
    cue_strength = np.clip(cue_strength, 0.05, 8.0)

    # Every reason retains a separate trace. Accessibility follows a smooth
    # exponential serial-position gradient: no reason is exactly overwritten,
    # and no capacity or sequence-length boundary is imposed.
    signed_traces = []
    unsigned_traces = []
    omission_variance = 0.0
    for serial_position, j in enumerate(discriminating):
        lag = (m - 1) - serial_position
        accessibility = retention ** float(lag)
        strength = float(cue_strength[j])
        direction = float(np.sign(diff[j]))

        expected_trace = attention * accessibility * strength
        signed_traces.append(direction * expected_trace)
        unsigned_traces.append(expected_trace)

        # Bernoulli attention uncertainty is represented at readout as the
        # variance of a potentially omitted trace.
        omission_variance += (
            attention * (1.0 - attention)
            * (accessibility * strength) ** 2
        )

    signed_traces = np.asarray(signed_traces, dtype=float)
    unsigned_traces = np.asarray(unsigned_traces, dtype=float)
    integrated_evidence = float(np.sum(signed_traces))

    positive_mass = float(np.sum(signed_traces[signed_traces > 0.0]))
    negative_mass = float(-np.sum(signed_traces[signed_traces < 0.0]))
    total_mass = positive_mass + negative_mass

    # Simultaneous opposing coalitions produce divisive interference. The
    # smooth balance index is zero for unanimous evidence and approaches one
    # when accessible support is evenly divided.
    if total_mass > 0.0 and np.isfinite(total_mass):
        conflict_balance = (
            4.0 * positive_mass * negative_mass
            / max(total_mass * total_mass, 1e-12)
        )
        conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))
        opposing_mass = 2.0 * min(positive_mass, negative_mass)
        interference_load = opposing_mass / (1.0 + total_mass)
    else:
        conflict_balance = 0.0
        interference_load = 0.0

    # Only an already accessible, genuinely near-balanced state receives an
    # extra bounded normalization. The sixth power leaves asymmetric states
    # essentially unchanged and introduces no sequence or stimulus class.
    near_balance_multiplier = (
        1.0 + near_balance_normalization * conflict_balance ** 6
    )
    conflict_divisor = 1.0 + (
        conflict_normalization
        * interference_load
        * np.sqrt(max(1.0, float(m - 1)))
        * near_balance_multiplier
    )

    # Load mainly increases readout uncertainty under conflict. A small
    # residual component remains for coherent displays, avoiding perfect
    # performance while not producing excessive unanimous-evidence fatigue.
    load_component = np.log1p(max(0.0, float(m - 1)))
    load_multiplier = 0.12 + 0.88 * conflict_balance
    readout_sd = np.sqrt(
        1.0
        + omission_variance
        + load_noise * load_component * load_multiplier
    )

    # A weak, continuous vigilance decline is the only load penalty that also
    # applies fully to unanimous evidence.
    vigilance = np.exp(-vigilance_decline * max(0.0, float(m - 2)))

    decision_variable = (
        decision_gain
        * vigilance
        * integrated_evidence
        / max(conflict_divisor * readout_sd, 1e-12)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])

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
- validity_compression: [0.0, 0.10]
- memory_retention: [0.38, 0.54]
- cue_attention: [0.86, 0.99]
- conflict_normalization: [0.15, 0.55]
- near_balance_normalization: [1.5, 2.5]
- load_noise: [0.03, 0.16]
- vigilance_decline: [0.008, 0.025]
- decision_gain: [3.5, 6.5]
- lapse_rate: [0.04, 0.16]

`rationale`: This is a minimal edit to the accepted iteration-2 model. The reliability transform, retention gradient, attention, gain, load noise, vigilance, and lapse ranges are unchanged. The only mechanistic addition is a narrow, bounded amplification of the existing divisive normalization based on conflict balance after serial accessibility has been applied. Raising accessible-state balance to the sixth power confines the adjustment to genuinely near-balanced recurrent states; asymmetric states receive almost no additional penalty. This should reduce excessive polarization in balanced conflicts without repeating the rejected retrieval-neutral interference term, injecting large extra readout noise, or weakening terminal control when recency has already made the accessible state strongly asymmetric. Because the adjustment depends only on signed accessible mass, it does not encode coherence, switch count, physical gaps, singleton status, fixed windows, or trial history.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed distributed, continuously weighted serial-integration family and successfully removes the prohibited structural effects. It fits the null coherence/foregrounding/history diagnostics in Experiments 7, 14, 16, 17, and 18, and it is also close in Experiments 8 and 10. However, its central reliability-by-recency calibration is not experiment-invariant. It overstates terminal/late control in Experiments 3 (0.715 versus 0.575) and 6 (early choice 0.126 versus 0.307), while simultaneously understating terminal control in Experiments 4 (0.489 versus 0.665), 11 (0.385 versus 0.843), 12 (majority choice 0.314 versus 0.171), and 13 (0.755 versus 0.881). It also produces excessive tally-conditioned separation in Experiment 1 (0.825 versus 0.494) and too much highest-validity-cue following in Experiments 2 (0.406 versus 0.194) and 9 (0.125 versus -0.033). This pattern indicates that the accessibility gradient is too steep while the compressed reliability profile still gives exceptionally high-validity early cues too much leverage relative to very low-validity terminal cues. The result is generic late-cue dominance when reliabilities are relatively similar, but inadequate terminal control when the terminal cue has extremely low instructed validity and faces an early coalition. Experiment 5's nonzero sequence contrast (-0.098 versus 0.003) is another consequence of overly position-sensitive weighting. The unanimous-load decline is also slightly too weak in Experiment 15 (0.003 versus 0.047), although avoiding a large fatigue effect is directionally correct.
Rationale: Keep the current mechanism family, but recalibrate its two main continuous gradients rather than adding sequence classes or capacity boundaries. Flatten validity weighting further by lowering or narrowing `validity_compression`, especially at its upper end, so extremely high-validity early cues do not dominate Experiments 2, 9, and 11 and low-validity terminal evidence remains behaviorally potent. At the same time, raise `memory_retention` moderately so earlier distributed traces survive enough to reduce excessive generic recency in Experiments 1, 3, 5, and 6. The combination—not either change alone—is important: flatter reliability should improve terminal control in Experiments 4, 11, 12, and 13, whereas greater retention should prevent that change from worsening the already excessive terminal effects in Experiments 3 and 6. A plausible next search should extend compression below 0.22 and shift retention above the current 0.22–0.44 range, with joint fitting rather than independent broad sampling. After fixing this tradeoff, modestly increase the unanimous component of vigilance/readout degradation to recover Experiment 15's small decline, but keep it weak enough to preserve the successful avoidance of pi_3-style fatigue. Preserve the history-free implementation and the absence of coherence, gap, singleton, and hard-window rules.

**Outcome of this advice:** iter 1 candidate loss=0.1710 -> iter 2 candidate loss=0.1577 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted recalibration improved aggregate loss and substantially repaired Experiments 2, 4, 11, 12, and 13, so the previous joint push toward flatter reliability weighting and greater retention was useful. The candidate also remains faithful to the prescribed distributed, history-free integration family and correctly yields null structural effects in Experiments 7, 14, 16, 17, and 18. Nevertheless, the fit is not yet experiment-invariant. It is now too choice-deterministic under many conflicts: Experiment 1's tally contrast is 0.895 versus 0.494, Experiment 3's terminal agreement is 0.690 versus 0.575, Experiment 6's early choice is only 0.065 versus 0.307, Experiment 9's early-cue association is 0.177 versus -0.033, and Experiment 10's terminal agreement is 0.942 versus 0.875. Experiment 12 likewise overshoots terminal control, driving majority choice below the target (0.103 versus 0.171). At the same time, terminal evidence remains too weak in Experiments 8, 11, and 13. Thus a single nearly flat power transform has traded one problem for another: exponents near zero erase useful moderate reliability differences, producing almost pure recency in Experiments 1, 3, 6, and 9, yet the coalition structure still prevents sufficiently strong singleton-terminal control in Experiments 8 and 11. The very low simulated between-subject variances in several of these experiments reinforce the diagnosis of overly categorical population predictions. The unanimous-load decline also remains too small in Experiment 15 (0.004 versus 0.047).
Rationale: Continue tuning within the prescribed recurrent evidence-integration family, but do not simply flatten validity further or reverse wholesale to the old compression range. The accepted change shows that extreme high-versus-low reliability differences need compression, while the new failures show that near-zero power exponents flatten the entire reliability profile too aggressively. Replace the current pure-power normalization with a bounded saturating transform of log odds that caps exceptionally reliable cues and floors very weak cues while retaining distinctions among moderate validities. This should preserve the gains for low-validity terminal cues in Experiments 4, 11, and 13 without turning Experiments 1, 3, 6, and 9 into near-pure recency tasks. In addition, modestly lower or cap `decision_gain` (or equivalently increase smooth conflict readout noise) to reduce the excessive polarization in Experiments 1, 3, 6, 9, 10, and 12; avoid a large reduction that would erase the strong terminal effects in Experiments 4 and 13. Broader continuous heterogeneity in retention or gain may also recover the observed variances without introducing stimulus classes. Finally, the previous vigilance increase barely changed Experiment 15, so do not repeat that adjustment alone: slightly raise the unanimous/coherent floor of the load-dependent readout-noise multiplier instead, keeping it far below the conflict multiplier. Preserve the successful history independence and the absence of coherence, gap, singleton, and hard-window rules.

**Outcome of this advice:** iter 2 candidate loss=0.1577 -> iter 3 candidate loss=0.1668 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed distributed, continuously weighted, history-free integration family, and it retains the important null results in Experiments 7, 17, and 18. However, this iteration was rejected by the accept gate: aggregate loss increased from 0.1577 to 0.1668. The bounded-tanh reliability transform and broader gain/retention ranges produced some local improvements—notably Experiments 3, 11, and 12—but regressed Experiments 2, 4, 5, 8, 9, 13, and 14. The central mismatch therefore remains. Balanced multi-cue conflicts are much too polarized: the tally contrast in Experiment 1 is 0.885 versus 0.494, early-option choice in Experiment 6 is only 0.074 versus 0.307, early-cue association in Experiment 9 is 0.213 versus -0.033, and terminal agreement in Experiment 10 is 0.920 versus 0.875. Experiment 3 also retains excessive terminal agreement. Conversely, comparatively unbalanced conflicts requiring strong terminal control remain too weak in Experiments 11 and 13. Experiment 5's nonzero ordering contrast also remains larger than the near-zero target. This pattern suggests that the accepted base needs conflict-sensitive uncertainty calibrated to coalition balance, rather than another global change to reliability compression, retention, or gain.
Rationale: Because the current candidate was rejected, build from the unchanged iteration-2 accepted base. Do not repeat the rejected wholesale replacement with the tanh reliability transform, the broad gain reduction, or another increase in the coherent load-noise floor. Instead, adjust the existing divisive-interference/readout component so that uncertainty rises much more sharply and continuously for nearly balanced opposing reliability-weighted coalitions, while remaining weak for strongly unbalanced conflicts and unanimous evidence. For example, use a steeper smooth function of `conflict_balance` and allow greater subject heterogeneity in its magnitude, rather than increasing global noise. This should move the highly polarized balanced-conflict predictions toward chance in the required directions: lower Experiment 1's contrast, raise early choice in Experiment 6, reduce the positive early-cue association in Experiment 9, lower terminal agreement in Experiment 10, and raise Experiment 12's majority-choice rate. It should also attenuate the spurious ordering contrast in Experiment 5. Keeping the penalty small for unbalanced coalitions should protect the accepted-base gains in Experiments 4, 11, and 13 better than a global gain or retention change. Leave the small Experiment 15 unanimous-load discrepancy secondary for now, since the prior coherent-noise-floor adjustment was rejected. Preserve the smooth serial gradient, stimulus-local computation, and absence of sequence-class, gap, singleton, hard-window, or learning rules.

**Outcome of this advice:** iter 3 candidate loss=0.1668 -> iter 4 candidate loss=0.2204 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful in broad terms to the prescribed distributed, continuous, history-free evidence-integration family, but it was decisively rejected by the accept gate: loss rose from the accepted 0.1577 to 0.2204. The new retrieval-neutral balanced-interference term produced useful local movement in Experiments 3, 6, 8, and 12, but its overall selectivity is poor. It badly regressed Experiment 2 (0.333 versus 0.194), destroyed the previously good terminal effect in Experiment 4 (0.373 versus 0.665), enlarged the ordering contrast in Experiment 5 (-0.148 versus 0.003), induced the prohibited practical coherence contrast in Experiment 7 (1 rather than 0), and weakened terminal control in Experiments 11 and 13. Experiment 1 remains far too tally-polarized (0.862 versus 0.494), while Experiment 9 remains too positively associated with the earliest cue (0.176 versus -0.033). The central problem is that retrieval-neutral coalition balance labels many objectively equal-count or equal-weight coalitions as highly conflicting even when the serial accessibility gradient has already made the decision state strongly asymmetric. The very large 25–65 interference range then injects excessive uncertainty into cases such as Experiment 4 that should exhibit strong terminal control, and differences in coalition balance across stimulus arrangements create an unintended structural contrast in Experiment 7. Several simulated between-subject variances also remain too low, indicating overly uniform predictions. Null effects in Experiments 14 and 17 and the close fits in Experiments 8 and 12 are not enough to offset these failures.
Rationale: Build again from the unchanged iteration-2 accepted base. My previous recommendation to add sharply amplified, retrieval-neutral coalition-balance noise was rejected by the gate, so do not repeat it or merely retune the 25–65 range. Remove the new `balanced_interference` term and its eighth-power coalition statistic. Try a smaller, qualitatively different in-family adjustment using the conflict balance of the accessible recurrent state—that is, positive and negative masses after serial accessibility has been applied—rather than balance before the recency gradient. Modestly strengthen the existing divisive normalization only when this accessible state is genuinely near balanced, using a bounded smooth multiplier or exponent and a narrow parameter range; avoid an additional large readout-noise term. This should depolarize Experiments 1, 6, 9, 10, and 12 when their final integrated states are close, while sparing Experiments 4, 11, and 13 when recency has made terminal support dominant despite retrieval-neutral coalition equality. It should also avoid generating arrangement effects such as the Experiment 7 failure. Keep the accepted reliability transform, retention, gain, vigilance, and coherent-load settings fixed for this test, since broader changes to those components were already rejected in iteration 3. Treat the small unanimous-load miss in Experiment 15 as secondary until the conflict calibration is repaired, and preserve the absence of sequence labels, switch-count rules, gaps, hard windows, and learning.

**Outcome of this advice:** iter 4 candidate loss=0.2204 -> iter 5 candidate loss=0.1518 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed distributed, continuously weighted, history-free evidence-integration family. It correctly preserves the key null structural results in Experiments 7, 14, 16, 17, and 18, and the accessible-state normalization proposed in the previous critique improved the accepted aggregate loss from 0.1577 to 0.1518. That direction was therefore productive. The fit is nevertheless not experiment-invariant. The largest failure is Experiment 1, where the tally-conditioned contrast is 0.912 rather than 0.494 and simulated heterogeneity is much too small. Recency is also too strong in longer conflicts: Experiment 3 overpredicts terminal agreement (0.664 versus 0.575), Experiment 6 nearly eliminates early-coalition choices (0.051 versus 0.307), and Experiment 12 underpredicts majority choices (0.093 versus 0.171). Experiment 9 retains an incorrectly positive earliest-cue association (0.207 versus -0.033). Conversely, terminal control is still too weak in Experiment 11 (0.545 versus 0.843), and moderately weak in Experiments 4, 8, and 13. This load-dependent reversal—insufficient terminal control at five to six cues but excessive terminal control at seven or more cues—indicates that a fixed exponential retention gradient combined with a sixth-power near-balance gate is too inflexible. The gate is so narrow that it barely regularizes many long, substantially conflicting states, while the same fixed leak does not give the terminal trace enough leverage in manageable-load conflicts. Experiment 15 also lacks the small observed unanimous-load decline, but that remains secondary.
Rationale: Build on this accepted candidate and retain its accessible-state, rather than retrieval-neutral, definition of conflict. Do not repeat the rejected large balanced-noise term or global reliability-transform replacement. Extend the successful adjustment by making accessible-state normalization broader but gentler: replace the sixth-power balance gate with a lower smooth power such as 3–4, reduce its maximum multiplier accordingly, and let the existing interference term grow somewhat more than square-root-wise with discriminating-cue load. This should move long, conflicting displays toward chance in Experiments 3, 6, 8, and 12 without imposing sequence classes or substantially affecting accessible states already dominated by the terminal coalition. To address Experiment 11 and the remaining short/manageable-load terminal deficits, test only a small intermediate reduction in retention—not a wholesale reversal of the previously successful retention increase—while using the stronger continuous load scaling to prevent renewed excessive recency at seven or more cues. Jointly fit these two knobs rather than changing global gain or reliability compression. Broaden subject heterogeneity in the accessible-conflict magnitude or load exponent to address the severely underestimated variances, especially in Experiments 1 and 4. Preserve the null coherence, gap, singleton, hard-window, and learning effects. Leave the modest Experiment 15 vigilance discrepancy for later, since previous direct attempts to increase coherent-load degradation were rejected.

**Outcome of this advice:** iter 5 candidate loss=0.1518 -> iter 6 candidate loss=0.1538 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed distributed, continuously weighted, history-free evidence-integration family and preserves the important null structural results in Experiments 7, 17, and 18. It is also close on Experiments 2, 4, 14, and 16. However, this iteration was rejected by the accept gate: loss increased from 0.1518 to 0.1538. Broadening the accessible-balance gate and increasing the load exponent yielded useful local movement in Experiments 2, 4, 6, 9, 11, and 13, but these gains were insufficient and were offset by regressions in Experiments 3, 10, and 12. The central load-dependent mismatch remains severe. Terminal influence is much too strong in long conflicts: Experiment 1's tally-conditioned separation is 0.904 versus 0.494, Experiment 3 terminal agreement is 0.677 versus 0.575, Experiment 6 early-coalition choice is only 0.063 versus 0.307, and Experiment 12 majority choice is 0.103 versus 0.171. Yet terminal influence is too weak in Experiment 11 (0.575 versus 0.843), as well as Experiments 8 and 13. Experiment 9 also retains the wrong positive association with the earliest cue (0.179 versus -0.033). Thus the broader normalization did not provide the required separation between manageable and long conflicts. The fixed exponential accessibility gradient appears to be the limiting component: a single retention rate cannot simultaneously produce steep local recency at roughly five cues and preserve enough distant evidence at seven or more cues. Between-subject variance also remains markedly too low in major failures such as Experiments 1, 4, 11, and 13.
Rationale: Build from the unchanged iteration-5 accepted base. My previous recommendation to broaden the balance gate and strengthen interference-load scaling was rejected, so do not repeat it or merely adjust the fourth-power gate and 0.60–0.78 exponent ranges. Restore the accepted base's accessible-state normalization and instead change the shape of the smooth serial-accessibility gradient. Replace the fixed exponential `retention ** lag` with a narrowly parameterized subexponential or fast-plus-persistent gradient, such as `exp(-lambda * lag**q)` with continuous subject-level heterogeneity in `lambda` and `q<1`. Calibrate it to have lower accessibility at the first few lags than the accepted exponential but a modestly heavier tail at long lags. This provides stronger terminal leverage in manageable conflicts—potentially improving Experiments 8, 11, and 13 and reducing the incorrect early-cue association in Experiment 9—while allowing distant early coalitions to survive in seven-plus-cue displays, raising early choice in Experiment 6 and reducing excessive terminal polarization in Experiments 1, 3, and 12. It remains a smooth, distributed recency mechanism with no hard boundary or sequence label. Fit the curvature and decay jointly, keeping the accepted reliability compression, gain, accessible-state conflict definition, and other noise parameters fixed for this test. Allow modest heterogeneity primarily in gradient curvature rather than globally broadening gain, which was previously rejected. The small unanimous-load miss in Experiment 15 should remain secondary. Preserve the successful absence of coherence, physical-gap, singleton, hard-window, and learning effects.

**Outcome of this advice:** iter 6 candidate loss=0.1538 -> iter 7 candidate loss=0.2066 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed distributed, continuously weighted, history-free evidence-integration family, but it was decisively rejected by the accept gate: loss rose from the iteration-5 best of 0.1518 to 0.2066. The curved subexponential accessibility gradient did not resolve the load-dependent tradeoff. Relative to the accepted base, it produced a prohibited practical sequence contrast in Experiment 7 (1 rather than 0), enlarged the ordering effect in Experiment 5 (-0.123 versus 0.003), and weakened terminal control in Experiments 11 (0.489 versus 0.843) and 13 (0.783 versus 0.881). It also left the principal long-conflict failures essentially intact: Experiment 1 remains far too tally-polarized (0.894 versus 0.494), Experiment 3 overpredicts terminal following (0.667 versus 0.575), Experiment 6 almost eliminates early-coalition choices (0.062 versus 0.307), and Experiment 9 retains an incorrectly positive earliest-cue association (0.207 versus -0.033). Experiment 12 improved substantially, but that isolated gain cannot offset the regressions. Simulated between-subject variance also remains too small in several central experiments, especially Experiments 1, 4, 11, and 13. The candidate still succeeds on the null effects in Experiments 14, 16, and 17 and is close on Experiments 2 and 12, but the empirical pattern is not experiment-invariant.
Rationale: Build from the unchanged iteration-5 accepted base and restore its exponential accessibility gradient. My previous recommendation to replace that gradient with a subexponential form was rejected strongly, so do not try another curvature range, fast-plus-slow gradient, or further local-versus-tail adjustment. Also do not repeat the rejected broadening of the accessible-balance gate. Instead, make one isolated adjustment to the accepted model's readout uncertainty: replace the weak logarithmic conflict-load term with a smoothly increasing, saturating function of discriminating-cue load multiplied by a quickly saturating function of total opposing accessible support. Keep the accepted near-balance divisive normalization unchanged. This readout term should be negligible for unanimous evidence, modest at four to six cues, and appreciably stronger in long conflicting displays, without a hard 5–6 cue boundary and without depending on switch count or sequence class. Saturating the dependence on opposing support, rather than making it sharply balance-sensitive, should reduce polarization in Experiments 1, 3, 6, 9, 10, and 12 while minimizing new coherent-versus-fragmented differences such as the Experiment 7 failure. Use a narrow parameter range and vary only its magnitude modestly across subjects to increase heterogeneity; keep reliability compression, retention, gain, attention, vigilance, and the accepted accessible-state normalization fixed so that this test is not confounded by previously rejected bundled changes. The persistent Experiment 11 deficit should remain a secondary target for this iteration, because both the broader normalization attempt and the curved-gradient attempt intended to repair it were rejected. Preserve the history-free mechanism and the absence of coherence, gap, singleton, hard-window, and learning rules.

**Outcome of this advice:** iter 7 candidate loss=0.2066 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Reliability-Compressed Recurrent Evidence Field. Decision makers inspect discriminating experts in descending instructed-validity order and encode each attended rating as signed, compressed log-odds evidence. Evidence remains distributed across cue traces, but accessibility declines smoothly with ordinal distance from response, producing graded recency rather than categorical replacement. Choice depends on the signed sum of accessible traces. Simultaneously active support for both products generates divisive interference, with an additional bounded increase when the accessible recurrent state is genuinely near balance. Readout uncertainty increases smoothly with discriminating-cue load when opposing accessible support is present, but saturates at high loads and remains negligible for unanimous evidence. Cue-attention uncertainty contributes additional encoding noise, while a small vigilance decrement allows only a modest decline for long unanimous displays. Thus terminal control emerges continuously from terminal reliability, the accessibility gradient, opposing reliability-weighted coalition strength, and total interference. The mechanism has no sequence-class rules, physical-gap effects, fixed capacity boundary, or cross-trial learning.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Recurrent evidence field expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    compression = float(parameters[\"validity_compression\"])\n    retention = float(parameters[\"memory_retention\"])\n    attention = float(parameters[\"cue_attention\"])\n    conflict_normalization = float(parameters[\"conflict_normalization\"])\n    near_balance_normalization = float(parameters[\"near_balance_normalization\"])\n    load_noise = float(parameters[\"load_noise\"])\n    vigilance_decline = float(parameters[\"vigilance_decline\"])\n    decision_gain = float(parameters[\"decision_gain\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    diff = stim[0] - stim[1]\n    cue_order = np.argsort(-validities, kind=\"stable\")\n    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]\n    m = len(discriminating)\n    if m == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Communicated validity is converted to normative log-odds and then\n    # strongly compressed. Normalizing by the experiment's typical positive\n    # log-odds preserves its reliability profile without allowing arbitrary\n    # differences in scale to determine response gain.\n    v = np.clip(validities, 0.5001, 0.999)\n    log_odds = np.log(v / (1.0 - v))\n    positive = log_odds[log_odds > 0.0]\n    reliability_scale = float(np.mean(positive)) if positive.size else 1.0\n    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:\n        reliability_scale = 1.0\n    relative_reliability = np.clip(\n        log_odds / reliability_scale, 1e-4, 20.0\n    )\n    cue_strength = np.power(relative_reliability, compression)\n    cue_strength = np.clip(cue_strength, 0.05, 8.0)\n\n    # Every reason retains a separate trace. Accessibility follows a smooth\n    # exponential serial-position gradient: no reason is exactly overwritten,\n    # and no capacity or sequence-length boundary is imposed.\n    signed_traces = []\n    unsigned_traces = []\n    omission_variance = 0.0\n    for serial_position, j in enumerate(discriminating):\n        lag = (m - 1) - serial_position\n        accessibility = retention ** float(lag)\n        strength = float(cue_strength[j])\n        direction = float(np.sign(diff[j]))\n\n        expected_trace = attention * accessibility * strength\n        signed_traces.append(direction * expected_trace)\n        unsigned_traces.append(expected_trace)\n\n        # Bernoulli attention uncertainty is represented at readout as the\n        # variance of a potentially omitted trace.\n        omission_variance += (\n            attention * (1.0 - attention)\n            * (accessibility * strength) ** 2\n        )\n\n    signed_traces = np.asarray(signed_traces, dtype=float)\n    unsigned_traces = np.asarray(unsigned_traces, dtype=float)\n    integrated_evidence = float(np.sum(signed_traces))\n\n    positive_mass = float(np.sum(signed_traces[signed_traces > 0.0]))\n    negative_mass = float(-np.sum(signed_traces[signed_traces < 0.0]))\n    total_mass = positive_mass + negative_mass\n\n    # Simultaneous opposing coalitions produce divisive interference. The\n    # smooth balance index is zero for unanimous evidence and approaches one\n    # when accessible support is evenly divided.\n    if total_mass > 0.0 and np.isfinite(total_mass):\n        conflict_balance = (\n            4.0 * positive_mass * negative_mass\n            / max(total_mass * total_mass, 1e-12)\n        )\n        conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))\n        opposing_mass = 2.0 * min(positive_mass, negative_mass)\n        interference_load = opposing_mass / (1.0 + total_mass)\n    else:\n        conflict_balance = 0.0\n        opposing_mass = 0.0\n        interference_load = 0.0\n\n    # Only an already accessible, genuinely near-balanced state receives an\n    # extra bounded normalization. The sixth power leaves asymmetric states\n    # essentially unchanged and introduces no sequence or stimulus class.\n    near_balance_multiplier = (\n        1.0 + near_balance_normalization * conflict_balance ** 6\n    )\n    conflict_divisor = 1.0 + (\n        conflict_normalization\n        * interference_load\n        * np.sqrt(max(1.0, float(m - 1)))\n        * near_balance_multiplier\n    )\n\n    # Readout uncertainty increases smoothly with discriminating-cue load,\n    # but only when accessible evidence supports both products. The bounded\n    # high-order load function is negligible at short loads, modest around\n    # five to six cues, and saturates for long displays without a hard window.\n    # Opposing support also saturates quickly, preventing extreme noise from\n    # growing without bound. Unanimous evidence receives no conflict noise.\n    load_power = float(m) ** 8\n    load_component = load_power / (load_power + 7.0 ** 8)\n    opposing_support = 1.0 - np.exp(-1.5 * max(0.0, opposing_mass))\n    readout_sd = np.sqrt(\n        1.0\n        + omission_variance\n        + load_noise * load_component * opposing_support\n    )\n\n    # A weak, continuous vigilance decline is the only load penalty that also\n    # applies fully to unanimous evidence.\n    vigilance = np.exp(-vigilance_decline * max(0.0, float(m - 2)))\n\n    decision_variable = (\n        decision_gain\n        * vigilance\n        * integrated_evidence\n        / max(conflict_divisor * readout_sd, 1e-12)\n    )\n    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-decision_variable))\n    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])\n\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "validity_compression": "[0.0, 0.10]",
    "memory_retention": "[0.38, 0.54]",
    "cue_attention": "[0.86, 0.99]",
    "conflict_normalization": "[0.15, 0.55]",
    "near_balance_normalization": "[1.5, 2.5]",
    "load_noise": "[7.0, 11.0]",
    "vigilance_decline": "[0.008, 0.025]",
    "decision_gain": "[3.5, 6.5]",
    "lapse_rate": "[0.04, 0.16]"
  },
  "rationale": "This is an isolated readout-uncertainty edit to the accepted iteration-5 model. The exponential accessibility gradient, reliability transform, attention process, gain, vigilance, lapse, and accepted accessible-state near-balance normalization are unchanged. The previous weak logarithmic load term and balance-weighted floor are replaced by the product of two bounded smooth functions: an eighth-order saturating function of discriminating-cue load and a quickly saturating function of total opposing accessible support. Consequently, conflict noise is negligible for unanimous evidence and short displays, modest at five to six cues, and appreciably stronger in long conflicts. This should reduce excessive polarization in Experiments 1, 3, 6, 9, 10, and 12 while limiting damage to manageable-load terminal effects and avoiding arrangement-dependent contrasts in Experiments 5 and 7. The relatively narrow load-noise range introduces modest stable heterogeneity without globally broadening gain. No switch count, coherence label, physical gap, singleton detector, hard capacity boundary, or trial history is used."
}
```

## Usage

```json
{
  "input_tokens": 40478,
  "output_tokens": 3942
}
```
