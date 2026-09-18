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
- THEORY 2 = `pi_8`
- The recommendation below acts on THEORY 2 (= `pi_8`).

Replace pi_8 with a new theory of context-calibrated sequential competition rather than regenerating its current model. The new theory should retain validity-ordered inspection but reject both the hard coherent-chunk/primacy policy and direct physical-edge foregrounding. Each attended cue should contribute signed evidence, with later evidence competing against and potentially inhibiting the current commitment. The degree of overwrite versus integration should be a continuous latent task-set learned from the experiment-wide distribution of scan-order conflicts, reversal locations, validity gradients, and repeated sequence families. Nondiscriminating physical gaps should have little or no direct effect, consistent with Experiment 2.

Unlike pi_8, the new theory should permit tally and weighted coalitions to alter the selected direction—not merely confidence—because Experiment 3 shows a large tally-dependent change. Unlike pi_3, reliability should not decline mechanically with discriminating-cue count: coherent or salient terminal reversals can remain highly reliable at four to six reasons, as in Experiments 1, 6, 12, and 13. Interference should instead depend on conflict geometry, relative run strength, validity profile, and learned task context. Stable subject-level variation can govern attention, leak/inhibition, context-learning rate, response gain, and lapse, but contextual effects should be continuous rather than a mixture of terminal and primacy policies.

The replacement should be constrained to reproduce the major cross-task signatures simultaneously: approximately 0.88 terminal following in Experiment 1; an approximately zero physical-foregrounding contrast in Experiment 2; the roughly 0.49 tally interaction in Experiment 3; weak terminal following in Experiments 5 and 10; strong terminal control in Experiments 6, 12, and 13; and rejection of the early 3-to-2 majority in Experiment 14. Explicit out-of-sample tests should cross the same ordered cue sequence with different validity profiles and experiment-wide sequence distributions, which would distinguish learned sequential competition from pi_3's task-invariant overwrite/fatigue account.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.810)

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


---

### `pi_3` (overall score: 0.783)

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


---

### `pi_8` (overall score: 0.622)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2560 -> ACCEPTED
- iter 2: loss=0.2844 -> REJECTED
- iter 3: loss=0.2185 -> ACCEPTED
- iter 4: loss=0.2882 -> REJECTED
- iter 5: loss=0.2816 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.2185 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.8978 (var=0.0036) (Δ vs real +0.4033)
  - iter 2: 0.8865 (var=0.0047) (Δ vs real +0.3921)
  - iter 3: 0.8503 (var=0.0038) (Δ vs real +0.3558)
  - iter 4: 0.7601 (var=0.0069) (Δ vs real +0.2657)
  - iter 5 (most recent): 0.8676 (var=0.0040) (Δ vs real +0.3732)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1504 (var=0.0030) (Δ vs real -0.0433)
  - iter 2: 0.2394 (var=0.0036) (Δ vs real +0.0456)
  - iter 3: 0.2919 (var=0.0051) (Δ vs real +0.0981)
  - iter 4: 0.2360 (var=0.0029) (Δ vs real +0.0423)
  - iter 5 (most recent): 0.2979 (var=0.0073) (Δ vs real +0.1042)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3783 (var=0.0074) (Δ vs real -0.1967)
  - iter 2: 0.3979 (var=0.0054) (Δ vs real -0.1771)
  - iter 3: 0.3585 (var=0.0076) (Δ vs real -0.2165)
  - iter 4: 0.2931 (var=0.0059) (Δ vs real -0.2819)
  - iter 5 (most recent): 0.4188 (var=0.0055) (Δ vs real -0.1562)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3742 (var=0.0174) (Δ vs real -0.2908)
  - iter 2: 0.2604 (var=0.0136) (Δ vs real -0.4046)
  - iter 3: 0.1767 (var=0.0315) (Δ vs real -0.4883)
  - iter 4: -0.0896 (var=0.0331) (Δ vs real -0.7546)
  - iter 5 (most recent): 0.2421 (var=0.0184) (Δ vs real -0.4229)
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
**Previous candidate values (this loop):**
  - iter 1: -0.3354 (var=0.0097) (Δ vs real -0.3387)
  - iter 2: -0.2242 (var=0.0079) (Δ vs real -0.2275)
  - iter 3: -0.0938 (var=0.0338) (Δ vs real -0.0971)
  - iter 4: 0.5758 (var=0.0322) (Δ vs real +0.5725)
  - iter 5 (most recent): -0.2413 (var=0.0250) (Δ vs real -0.2446)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1058 (var=0.0039) (Δ vs real -0.2008)
  - iter 2: 0.1225 (var=0.0036) (Δ vs real -0.1842)
  - iter 3: 0.1754 (var=0.0085) (Δ vs real -0.1312)
  - iter 4: 0.1433 (var=0.0049) (Δ vs real -0.1633)
  - iter 5 (most recent): 0.1642 (var=0.0054) (Δ vs real -0.1425)
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
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0900) (Δ vs real +1.0000)
  - iter 2: 1.0000 (var=0.2016) (Δ vs real +1.0000)
  - iter 3: 0.0000 (var=0.2304) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0196) (Δ vs real +0.0000)
  - iter 5 (most recent): 1.0000 (var=0.1924) (Δ vs real +1.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3221 (var=0.0021) (Δ vs real +0.1500)
  - iter 2: 0.3427 (var=0.0023) (Δ vs real +0.1706)
  - iter 3: 0.3848 (var=0.0018) (Δ vs real +0.2127)
  - iter 4: 0.4248 (var=0.0008) (Δ vs real +0.2527)
  - iter 5 (most recent): 0.3902 (var=0.0016) (Δ vs real +0.2181)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0392 (var=0.0052) (Δ vs real +0.0717)
  - iter 2: 0.1167 (var=0.0081) (Δ vs real +0.1492)
  - iter 3: 0.0638 (var=0.0093) (Δ vs real +0.0963)
  - iter 4: 0.0392 (var=0.0058) (Δ vs real +0.0717)
  - iter 5 (most recent): 0.0254 (var=0.0064) (Δ vs real +0.0579)
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
**Previous candidate values (this loop):**
  - iter 1: 0.9299 (var=0.0010) (Δ vs real +0.0551)
  - iter 2: 0.9095 (var=0.0012) (Δ vs real +0.0347)
  - iter 3: 0.8784 (var=0.0015) (Δ vs real +0.0036)
  - iter 4: 0.8099 (var=0.0025) (Δ vs real -0.0648)
  - iter 5 (most recent): 0.8923 (var=0.0020) (Δ vs real +0.0176)
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
**Previous candidate values (this loop):**
  - iter 1: 0.7779 (var=0.0051) (Δ vs real -0.0646)
  - iter 2: 0.6492 (var=0.0066) (Δ vs real -0.1933)
  - iter 3: 0.6488 (var=0.0106) (Δ vs real -0.1937)
  - iter 4: 0.7183 (var=0.0082) (Δ vs real -0.1242)
  - iter 5 (most recent): 0.6533 (var=0.0090) (Δ vs real -0.1892)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1156 (var=0.0046) (Δ vs real -0.0556)
  - iter 2: 0.2013 (var=0.0038) (Δ vs real +0.0300)
  - iter 3: 0.2181 (var=0.0053) (Δ vs real +0.0469)
  - iter 4: 0.1638 (var=0.0074) (Δ vs real -0.0075)
  - iter 5 (most recent): 0.2119 (var=0.0088) (Δ vs real +0.0406)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8049 (var=0.0032) (Δ vs real -0.0760)
  - iter 2: 0.7080 (var=0.0045) (Δ vs real -0.1730)
  - iter 3: 0.7349 (var=0.0057) (Δ vs real -0.1461)
  - iter 4: 0.7161 (var=0.0043) (Δ vs real -0.1649)
  - iter 5 (most recent): 0.7165 (var=0.0031) (Δ vs real -0.1645)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0213 (var=0.0068) (Δ vs real +0.0021)
  - iter 2: 0.0108 (var=0.0073) (Δ vs real -0.0083)
  - iter 3: -0.0050 (var=0.0093) (Δ vs real -0.0242)
  - iter 4: 0.0083 (var=0.0116) (Δ vs real -0.0108)
  - iter 5 (most recent): 0.0083 (var=0.0095) (Δ vs real -0.0108)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0863 (var=0.0106)
- pi_3: -0.0217 (var=0.0090)
- pi_1: -0.0004 (var=0.0063)
- pi_2: -0.0008 (var=0.0037)
- pi_4: -0.0146 (var=0.0066)
- pi_5: 0.0054 (var=0.0094)
- pi_6: -0.0137 (var=0.0073)
- pi_7: -0.0042 (var=0.0041)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Distribution-Calibrated Competitive Accumulation with Margin-Gated Coalition Access. People inspect discriminating cues in descending instructed-validity order and encode each attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the active commitment, while an experiment-level task set learned from ordinal conflict families continuously controls retention and overwrite. Current-sequence statistics receive only a bounded calibration weight once prior trials are available, reducing stimulus-specific policy shifts without eliminating early contextual adaptation. Reliability-weighted coalitions can reverse the sequential state, but their access to choice is proportional to their normalized evidence margin: substantial coalitions protect early majorities, whereas nearly balanced coalitions yield control to sequential competition and terminal opposition. Physical gaps are not represented, and reliability depends on conflict geometry and evidence margin rather than discriminating-cue count.

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

        return {
            "m": float(m),
            "directions": directions,
            "reversal_fractions": reversal_fractions,
            "terminal_singleton": float(run_lengths[-1] == 1 and len(run_lengths) > 1),
            "two_run": float(len(run_lengths) == 2),
            "switch_density": float(switches.size) / float(max(1, m - 1))
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
- context_learning_rate: [0.10, 0.24]
- response_gain: [1.8, 3.8]
- lapse_rate: [0.04, 0.15]

`rationale`: This is a minimal edit of the accepted candidate. First, validity sensitivity is raised and widened from [0.02, 0.12] to [0.15, 0.45], reducing the model's near-equal-weight tally behavior and allowing high-validity early evidence to resist large low-validity tallies in Experiment 1. Second, coalition access is now a smooth function of the reliability-weighted coalition margin. Balanced conflicts receive little global-coalition influence, allowing sequential opposition to increase terminal control in Experiments 3 and 4; substantial early coalitions retain influence, moderating excessive terminal following in Experiments 6 and 8. Third, the current sequence remains part of contextual calibration but receives 35% of its former recency weight whenever history exists. This intermediate regularization avoids repeating the rejected history-only edit while reducing stimulus-specific context shifts responsible for the spurious Experiment 5 and thresholded Experiment 7 contrasts. All other accumulation, inhibition, confidence, lapse, and physical-gap-independent mechanisms are unchanged, preserving strong terminal behavior in Experiments 10–13 and the near-zero foregrounding contrast in Experiment 14.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed sequential-competition family and successfully eliminates direct physical foregrounding: Experiment 14 is essentially exact (0.021 versus 0.019), while Experiments 2 and 9 are also reasonably close. It additionally captures the broad direction of strong terminal control in Experiments 10, 11, 12, and 13, although it is somewhat too deterministic in several of them. However, the cross-experiment calibration is not yet adequate. The largest failures are the excessive tally interaction in Experiment 1 (0.898 versus 0.494), insufficient terminal influence in Experiments 3 and 4 (0.378 versus 0.575 and 0.374 versus 0.665), a large spurious block-versus-mixed contrast in Experiment 5 (-0.335 versus approximately zero), excessive terminal replacement in Experiment 6 (early choice 0.106 versus 0.307), the incorrect thresholded contrast in Experiment 7 (1 rather than 0), and excessive terminal following in Experiment 8 (0.322 versus 0.172). These discrepancies indicate that the model is generating overly sharp, stimulus-dependent shifts between integration and overwrite rather than a sufficiently stable experiment-level continuous task set.
Rationale: Retain the distribution-calibrated competitive accumulator, but revise its calibration. The most likely implementation problem is that the current trial is appended to the history before computing the task set. Consequently, each stimulus's own terminal-singleton, two-run, and switch-density statistics directly change the supposedly experiment-wide task set. This can create the large within-experiment geometry effects seen in Experiments 1, 5, and 7. Compute contextual calibration primarily from preceding trials, with a neutral prior and at most a small explicitly bounded contribution from the current sequence. Also soften the large two-run, concentration, and switch-density coefficients or reduce context sensitivity so the continuous task set does not approach opposite extremes across individual stimuli. Do not simply weaken terminal competition globally: that would worsen Experiments 3, 4, and 13. Instead, add a modest baseline opposition-inhibition/recency component that remains active even in integration-calibrated contexts, while reserving the task-set-dependent increment for strong overwrite. This should allow a late cue to tip balanced or near-balanced coalitions in Experiments 3 and 4 without overturning the substantial early majorities in Experiment 8. Refit retention and coalition ranges after removing current-trial leakage, aiming specifically to reduce the Experiment 1 tally contrast, eliminate the Experiment 5 and 7 geometry artifacts, and moderate Experiment 6 while preserving the good near-zero physical-gap effect and strong late-block performance in Experiments 10–14.

**Outcome of this advice:** iter 1 candidate loss=0.2560 -> iter 2 candidate loss=0.2844 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to context-calibrated sequential competition and preserves the important absence of physical-gap foregrounding in Experiment 14 (0.011 versus 0.019). It also fits Experiments 2, 10, and 12 reasonably well. Nevertheless, it was rejected by the accept gate and is worse overall than the accepted iteration-1 base. The history-only calibration change modestly improved Experiments 3, 5, 6, 10, and 12, but substantially damaged terminal control in Experiments 4, 11, and 13, worsened Experiments 8 and 9, and did not fix the decisive failures in Experiments 1 or 7. The model still produces an excessive tally-dependent contrast in Experiment 1 (0.887 versus 0.494), too little terminal influence in balanced conflicts in Experiments 3 and 4, a spurious geometry contrast in Experiment 5, excessive terminal following against early majorities in Experiments 6 and 8, and the wrong thresholded result in Experiment 7. Its low between-subject variances in many experiments also indicate overly homogeneous predictions.
Rationale: Because the previous recommendation to remove current-trial calibration and add baseline inhibition was rejected, do not repeat that full history-only move. Build from the unchanged accepted iteration-1 base and try a smaller, different in-family correction. First, raise and widen validity sensitivity: the current 0.02–0.12 range makes evidence almost an equal-weight tally, plausibly causing the enormous Experiment 1 tally contrast. Stronger reliability weighting can reduce that contrast without requiring primacy or abandoning sequential competition. Second, make the coalition contribution depend smoothly on the validity-weighted coalition margin: retain more coalition influence when there is a substantial early majority, as in Experiments 6 and 8, but reduce it near balance so sequential opposition can let terminal evidence tip Experiments 3 and 4. This is a continuous normalization within the prescribed accumulator, not a policy mixture. Third, regularize rather than eliminate current-sequence calibration—for example, cap its contribution or blend the accepted base's task set with a mostly history-derived estimate. An intermediate calibration is warranted because complete exclusion improved Experiment 5 but sharply weakened the required terminal control in Experiments 11 and 13. Tune against the joint constraints that Experiment 5 approach zero and Experiment 7 fall below its 0.08 threshold while preserving strong terminal performance in Experiments 10–13 and the near-zero Experiment 14 contrast. Avoid another global reduction in response gain or terminal competition; the current iteration shows that this sacrifices the strong-terminal experiments without resolving Experiment 1.

**Outcome of this advice:** iter 2 candidate loss=0.2844 -> iter 3 candidate loss=0.2185 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted iteration is a genuine aggregate improvement and remains faithful to context-calibrated sequential competition. The intermediate current-trial regularization successfully removes the thresholded error in Experiment 7, substantially reduces the spurious Experiment 5 contrast, and gives an excellent fit in Experiment 10. Physical foregrounding remains appropriately absent in Experiment 14. Experiment 6 and the Experiment 1 tally interaction also move in the correct direction. However, the model is not yet experiment-invariant enough to ship. The tally interaction in Experiment 1 remains far too large (0.850 versus 0.494). Terminal evidence remains much too weak in Experiments 3 and 4 (0.359 versus 0.575; 0.177 versus 0.665), and terminal control is also underestimated in Experiments 11 and 13 (0.649 versus 0.843; 0.735 versus 0.881). Conversely, terminal following is excessive in Experiment 8 (0.385 versus 0.172), while highest-validity-cue following is too high in Experiment 2 (0.292 versus 0.194). Thus the static increase in validity sensitivity helped protect early coalitions, but it also broadly strengthened early evidence in contexts where humans exhibit overwrite. The margin gate alone cannot resolve this because reliability weighting is currently fixed across task sets. Several predicted between-subject variances, especially in Experiments 1, 8, and 10, are also too small, indicating overly uniform response regimes.
Rationale: Retain the accepted intermediate history calibration and margin-gated competitive accumulator; the previous advice leading to these changes was accepted and should not be undone. The next adjustment should make validity sensitivity itself continuously task-set dependent rather than increasing it globally again. Preserve relatively strong validity weighting in integration-calibrated or substantial-early-margin contexts, which should let the first cue resist opposing tallies in Experiment 1 and protect early runs in Experiment 8. Attenuate validity weighting as the learned overwrite/terminal-singleton context strengthens, allowing late opposition to regain control in Experiments 2, 3, 4, 11, and 13. A bounded interpolation with a nonzero sensitivity floor is preferable to a discrete rule. In addition, add only a modest coalition-margin-dependent floor to opposition inhibition for nearly balanced conflicts; the current inhibition is multiplied entirely by `overwrite_task_set`, leaving too little late competition in Experiments 3 and 4. Do not make this floor global or strong at large margins, because Experiment 6 already shows excessive overwrite and Experiment 12 is close. Tune these two couplings jointly, targeting a large reduction of Experiment 1's interaction and increased terminal choice in Experiments 3/4/11/13 without sacrificing the newly correct Experiments 5, 7, 10, and 14. Subject variation in the new coupling or response gain can then be widened slightly if necessary to address the remaining variance compression.

**Outcome of this advice:** iter 3 candidate loss=0.2185 -> iter 4 candidate loss=0.2882 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed context-calibrated sequential-competition family, but it was rejected by the accept gate and substantially regresses from the accepted iteration-3 base. The task-dependent validity exponent and balanced-inhibition floor move Experiment 1 somewhat toward the data and improve Experiments 2 and 12, but they produce severe collateral failures. Terminal influence becomes even weaker in Experiments 3 and 4 (0.293 versus 0.575 and -0.090 versus 0.665), the previously improved Experiment 5 contrast explodes from -0.094 to 0.576 rather than approximately zero, Experiment 8 moves farther above its low observed terminal-following rate, and Experiment 10 loses its nearly exact fit. Strong terminal control also remains underestimated in Experiments 11 and 13, while Experiment 6 still shows too much overwrite. Only Experiments 7, 9, 12, and 14 are acceptably close. The especially large reversal in Experiment 5 indicates that task-set-dependent validity compression is coupling too strongly to stimulus geometry and recreating within-experiment regime differences. The balanced-inhibition addition also fails its intended purpose: rather than increasing terminal control in Experiments 3 and 4, it reduces it, likely because preview-weighted balance and validity compression alter coalition signs or task calibration before the local competition unfolds.
Rationale: Because the previous recommendation to add task-dependent validity sensitivity and a balanced-conflict inhibition floor was rejected, do not repeat or enlarge either adjustment. The next candidate should build on the unchanged accepted iteration-3 base and keep its intermediate history calibration, static validity weighting, and successful fits in Experiments 5, 7, 10, and 14. Use a different in-family knob: revise the existing coalition-margin access function and local opposition update rather than changing experiment-level validity expression. In particular, remove or sharply reduce the coalition gate's nonzero access floor near a zero weighted margin and make access rise more convexly with margin. That allows the sequential state—not a residual global coalition—to select the direction in balanced conflicts such as Experiments 3 and 4, while retaining coalition protection for clear margins in Experiments 6, 8, and 12. If additional terminal competition is needed, implement it as a modest cumulative opposing-run attack on the incumbent state, normalized by incumbent strength and total attended evidence, rather than as another preview-balance floor. This should strengthen coherent late runs in Experiments 11 and 13 without globally lowering retention. Separately, use conflict-dependent response gain or temperature: lower gain for unresolved mixed sequences to reduce Experiment 1's excessive tally interaction, but preserve gain for coherent terminal runs so Experiments 10 and 13 are not weakened. Parameter ranges for these changes should be narrow around the accepted base, since the current broad couplings caused large sign-changing regressions and excessive cross-stimulus instability.

**Outcome of this advice:** iter 4 candidate loss=0.2882 -> iter 5 candidate loss=0.2816 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate remains faithful to context-calibrated sequential competition and continues to avoid direct physical-gap foregrounding: Experiment 14 is close to exact, and Experiments 9, 10, and 12 are reasonably fitted. Nevertheless, it was rejected by the accept gate and is substantially worse than the accepted iteration-3 base. The convex coalition gate and revised cumulative-run attack do not solve the central cross-task problem. Experiment 1's tally interaction rebounds to 0.868 versus 0.494; terminal control remains much too weak in Experiments 3, 4, 11, and 13; Experiment 5 again develops a sizable spurious geometry contrast; and Experiment 7 returns to the wrong thresholded outcome. The model also overwrites too strongly in Experiments 6 and 8, as shown by too little early following in Experiment 6 and too much terminal following in Experiment 8. Thus the same local competition rule cannot distinguish tasks requiring strong terminal control from tasks requiring protection of early coalitions. The likely bottleneck is the coarse context representation: it summarizes singleton frequency, switch density, two-run frequency, and reversal-bin concentration, but omits experiment-wide distributions of relative run strength and weighted prefix-versus-terminal strength. Its validity-profile representation is only a weak coefficient on the coefficient of variation of cue reliabilities. Consequently, distinct experiments can receive similar task sets despite sharply different validity profiles and conflict-strength distributions. Between-subject variance also remains compressed in several conditions, especially Experiment 1.
Rationale: Build the next attempt on the unchanged accepted iteration-3 base. The previous recommendation to convexify coalition access and strengthen cumulative opposing-run attack was rejected, so do not repeat or enlarge either change; likewise retain iteration 3's static validity sensitivity, intermediate current-trial regularization, and successful local update. Instead, improve the context calibration within the prescribed mechanism family. Add continuous experiment-level summaries of validity-weighted conflict geometry: terminal-run strength relative to incumbent prefix strength, weighted coalition margin at each reversal, reversal-location-by-run-balance interactions, and profile-specific quantities such as the log-odds validity slope and terminal-to-prefix reliability ratio. Aggregate these over preceding sequence families with the existing bounded current-trial contribution. Map them smoothly into the single overwrite task set with narrow coefficients and regularization, rather than adding a new policy or trial-specific gate. This should let identical ordinal patterns induce stronger overwrite under the profiles and sequence distributions of Experiments 3, 4, 11, and 13, but stronger integration when experiment-wide families contain protected early coalitions as in Experiments 6 and 8. Keep physical indices and nondiscriminating gaps excluded. Tune first against the joint constraints of Experiments 1, 3, 4, 5, 6, 8, 11, and 13 while requiring that the accepted fits in Experiments 7, 10, 12, and 14 do not regress. Only after directional fit improves should subject-level variation in context coefficients or response gain be widened modestly to address variance compression.

**Outcome of this advice:** iter 5 candidate loss=0.2816 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Distribution-Calibrated Competitive Accumulation with Validity-Weighted Conflict-Family Calibration. People inspect discriminating cues in descending instructed-validity order and encode every attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the current commitment, while a single continuous task set learned across the experiment regulates retention versus overwrite. In addition to ordinal reversal frequency and location, task-set learning represents the validity-weighted geometry of recurring conflicts: terminal-run strength relative to the preceding coalition, coalition margins at reversals, and interactions between reversal location and run balance. The instructed validity profile further calibrates overwrite through its log-odds slope and terminal-to-prefix reliability ratio. Thus, identical ordinal sequences can induce different degrees of integration across experiments without invoking discrete terminal and primacy policies. Reliability-weighted coalitions can reverse the sequential state, but their access to choice remains proportional to their normalized evidence margin. Physical cue gaps are not represented, and reliability does not mechanically decline with cue count.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Competitive accumulation expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    attention = float(parameters[\"attention_rate\"])\n    integration_retention = float(parameters[\"integration_retention\"])\n    overwrite_retention = float(parameters[\"overwrite_retention\"])\n    conflict_inhibition = float(parameters[\"conflict_inhibition\"])\n    coalition_weight = float(parameters[\"coalition_weight\"])\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    context_sensitivity = float(parameters[\"context_sensitivity\"])\n    context_threshold = float(parameters[\"context_threshold\"])\n    learning_rate = float(parameters[\"context_learning_rate\"])\n    geometry_context_weight = float(parameters[\"geometry_context_weight\"])\n    profile_context_weight = float(parameters[\"profile_context_weight\"])\n    response_gain = float(parameters[\"response_gain\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    clipped_v = np.clip(validities, 0.5001, 0.999)\n    reliability = np.log(clipped_v / (1.0 - clipped_v))\n    mean_positive_rel = float(np.mean(reliability[reliability > 0.0]))\n    if not np.isfinite(mean_positive_rel) or mean_positive_rel <= 0.0:\n        mean_positive_rel = 1.0\n    normalized_rel = np.clip(reliability / mean_positive_rel, 0.15, 6.0)\n    contextual_strength = np.power(normalized_rel, validity_sensitivity)\n\n    def ordinal_summary(a, b):\n        a = np.asarray(a, dtype=float)\n        b = np.asarray(b, dtype=float)\n        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:\n            return None\n        d = a - b\n        ordered_idx = [int(j) for j in cue_order if j < d.size and d[j] != 0]\n        if not ordered_idx:\n            return None\n        directions = np.asarray([np.sign(d[j]) for j in ordered_idx], dtype=float)\n        strengths = np.asarray(\n            [contextual_strength[j] for j in ordered_idx], dtype=float\n        )\n        m = directions.size\n        switches = np.flatnonzero(directions[1:] != directions[:-1]) + 1\n\n        run_lengths = []\n        run_starts = [0]\n        run = 1\n        for k in range(1, m):\n            if directions[k] == directions[k - 1]:\n                run += 1\n            else:\n                run_lengths.append(run)\n                run_starts.append(k)\n                run = 1\n        run_lengths.append(run)\n\n        if switches.size:\n            reversal_fractions = switches.astype(float) / float(m)\n            terminal_start = int(run_starts[-1])\n            terminal_strength = float(np.sum(strengths[terminal_start:]))\n            prefix_strength = float(np.sum(strengths[:terminal_start]))\n            weighted_terminal_share = terminal_strength / max(\n                terminal_strength + prefix_strength, 1e-9\n            )\n\n            reversal_interactions = []\n            for run_number in range(1, len(run_starts)):\n                start = int(run_starts[run_number])\n                stop = (\n                    int(run_starts[run_number + 1])\n                    if run_number + 1 < len(run_starts)\n                    else m\n                )\n                incumbent_strength = float(np.sum(strengths[:start]))\n                attacking_run_strength = float(np.sum(strengths[start:stop]))\n                balance = (\n                    2.0 * min(incumbent_strength, attacking_run_strength)\n                    / max(incumbent_strength + attacking_run_strength, 1e-9)\n                )\n                reversal_location = float(start) / float(m)\n                reversal_interactions.append(balance * reversal_location)\n            reversal_balance_location = float(np.mean(reversal_interactions))\n        else:\n            reversal_fractions = np.asarray([], dtype=float)\n            weighted_terminal_share = 0.0\n            reversal_balance_location = 0.0\n\n        weighted_coalition = float(np.sum(directions * strengths))\n        total_strength = float(np.sum(strengths))\n        weighted_final_margin = abs(weighted_coalition) / max(total_strength, 1e-9)\n\n        return {\n            \"m\": float(m),\n            \"directions\": directions,\n            \"reversal_fractions\": reversal_fractions,\n            \"terminal_singleton\": float(run_lengths[-1] == 1 and len(run_lengths) > 1),\n            \"two_run\": float(len(run_lengths) == 2),\n            \"switch_density\": float(switches.size) / float(max(1, m - 1)),\n            \"weighted_terminal_share\": float(weighted_terminal_share),\n            \"weighted_final_margin\": float(weighted_final_margin),\n            \"reversal_balance_location\": float(reversal_balance_location)\n        }\n\n    # Learn the task set from ordinal scan sequences only. Physical indices,\n    # distances between cues, and nondiscriminating gaps never enter these\n    # statistics. Exponential weighting implements continuous contextual\n    # learning while retaining a weak neutral prior early in the block.\n    summaries = []\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n    for i in range(n_past):\n        s = ordinal_summary(past_a[i], past_b[i])\n        if s is not None:\n            summaries.append(s)\n\n    current_summary = ordinal_summary(stim[0], stim[1])\n    if current_summary is None:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    summaries.append(current_summary)\n\n    ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)\n    weights = np.power(np.clip(1.0 - learning_rate, 1e-6, 1.0), ages)\n    # The current stimulus can update calibration, but once context exists it\n    # receives a bounded weight and cannot sharply change the experiment-level\n    # task set solely because of its own conflict geometry.\n    if len(summaries) > 1:\n        weights[-1] *= 0.35\n    weights /= max(float(weights.sum()), 1e-12)\n\n    terminal_singleton_rate = float(np.sum([\n        w * s[\"terminal_singleton\"] for w, s in zip(weights, summaries)\n    ]))\n    two_run_rate = float(np.sum([\n        w * s[\"two_run\"] for w, s in zip(weights, summaries)\n    ]))\n    switch_density = float(np.sum([\n        w * s[\"switch_density\"] for w, s in zip(weights, summaries)\n    ]))\n    weighted_terminal_share = float(np.sum([\n        w * s[\"weighted_terminal_share\"] for w, s in zip(weights, summaries)\n    ]))\n    weighted_final_margin = float(np.sum([\n        w * s[\"weighted_final_margin\"] for w, s in zip(weights, summaries)\n    ]))\n    reversal_balance_location = float(np.sum([\n        w * s[\"reversal_balance_location\"] for w, s in zip(weights, summaries)\n    ]))\n\n    # A five-bin distribution represents reversal locations in ordinal scan\n    # time, not their physical screen positions. Concentrated sequence\n    # families promote overwrite; high reversal entropy promotes integration.\n    reversal_hist = np.zeros(5, dtype=float)\n    reversal_mass = 0.0\n    for w, s in zip(weights, summaries):\n        fractions = s[\"reversal_fractions\"]\n        if fractions.size == 0:\n            continue\n        contribution = w / float(fractions.size)\n        for fraction in fractions:\n            bin_index = min(4, max(0, int(np.floor(5.0 * float(fraction)))))\n            reversal_hist[bin_index] += contribution\n            reversal_mass += contribution\n\n    if reversal_mass > 0.0:\n        reversal_distribution = reversal_hist / reversal_mass\n        concentration = float(np.sum(reversal_distribution ** 2))\n        positive = reversal_distribution[reversal_distribution > 0.0]\n        reversal_entropy = float(\n            -np.sum(positive * np.log(positive)) / np.log(5.0)\n        )\n    else:\n        concentration = 0.2\n        reversal_entropy = 1.0\n\n    # Validity-profile steepness can alter task calibration. In addition to\n    # dispersion, compare the least-reliable portion of the profile with its\n    # most-reliable portion and estimate the ordered log-odds slope.\n    mean_rel = max(float(np.mean(reliability)), 1e-8)\n    validity_gradient = float(np.std(reliability) / mean_rel)\n    validity_gradient = float(np.clip(validity_gradient, 0.0, 2.0))\n\n    ordered_reliability = reliability[cue_order]\n    profile_band = max(1, int(np.ceil(n_features / 3.0)))\n    prefix_reliability = float(np.mean(ordered_reliability[:profile_band]))\n    terminal_reliability = float(np.mean(ordered_reliability[-profile_band:]))\n    terminal_to_prefix_reliability = terminal_reliability / max(\n        prefix_reliability, 1e-9\n    )\n    terminal_to_prefix_reliability = float(np.clip(\n        terminal_to_prefix_reliability, 0.0, 1.0\n    ))\n    if n_features > 1:\n        profile_positions = np.linspace(0.0, 1.0, n_features)\n        log_odds_slope = -float(np.polyfit(\n            profile_positions, ordered_reliability, 1\n        )[0]) / mean_rel\n    else:\n        log_odds_slope = 0.0\n    log_odds_slope = float(np.clip(log_odds_slope, 0.0, 2.0))\n\n    # These bounded additions distinguish experiments with similar ordinal\n    # switch counts but different recurring strength geometry. Strong,\n    # late, balanced attacks favor overwrite; clear final coalitions favor\n    # integration. Flat validity profiles permit more overwrite than steep\n    # profiles in which early evidence is substantially more diagnostic.\n    weighted_geometry_score = (\n        1.20 * weighted_terminal_share\n        + 1.00 * reversal_balance_location\n        - 0.80 * weighted_final_margin\n    )\n    validity_profile_score = (\n        2.0 * terminal_to_prefix_reliability\n        - 1.0\n        - 0.25 * log_odds_slope\n    )\n\n    context_score = (\n        3.0 * terminal_singleton_rate\n        + 3.0 * concentration\n        + 2.0 * two_run_rate\n        - 2.5 * reversal_entropy\n        - 1.2 * switch_density\n        - 0.20 * validity_gradient\n        + geometry_context_weight * weighted_geometry_score\n        + profile_context_weight * validity_profile_score\n    )\n    context_logit = context_sensitivity * (context_score - context_threshold)\n    context_logit = float(np.clip(context_logit, -60.0, 60.0))\n    overwrite_task_set = 1.0 / (1.0 + np.exp(-context_logit))\n\n    # The latent task set continuously interpolates between integration and\n    # overwrite. It never selects a discrete primacy or terminal policy.\n    base_retention = (\n        (1.0 - overwrite_task_set) * integration_retention\n        + overwrite_task_set * overwrite_retention\n    )\n    base_retention = float(np.clip(base_retention, 0.01, 0.999))\n\n    diff = stim[0] - stim[1]\n    discriminating = [int(j) for j in cue_order if diff[j] != 0]\n    directions = np.asarray([np.sign(diff[j]) for j in discriminating], dtype=float)\n\n    cue_strength = np.power(normalized_rel, validity_sensitivity)\n\n    accumulator = 0.0\n    coalition = 0.0\n    current_run_strength = 0.0\n    previous_direction = 0.0\n\n    for j, direction in zip(discriminating, directions):\n        evidence = attention * float(cue_strength[j]) * float(direction)\n        coalition += evidence\n\n        if previous_direction == 0.0 or direction == previous_direction:\n            current_run_strength += abs(evidence)\n        else:\n            current_run_strength = abs(evidence)\n\n        retention = base_retention\n        if accumulator != 0.0 and np.sign(accumulator) != direction:\n            # Opposition inhibits the existing commitment in proportion to\n            # the new reason's strength relative to the active commitment.\n            relative_attack = abs(evidence) / (abs(accumulator) + abs(evidence) + 1e-9)\n            run_attack = current_run_strength / (abs(accumulator) + current_run_strength + 1e-9)\n            inhibition = conflict_inhibition * overwrite_task_set\n            retention *= np.exp(-inhibition * (0.65 * relative_attack + 0.35 * run_attack))\n\n        accumulator = retention * accumulator + evidence\n        previous_direction = float(direction)\n\n    # A reliability-weighted coalition has strong access only when it has a\n    # substantial normalized margin. Near balance, sequential opposition can\n    # determine the sign; clear early majorities remain protected by global\n    # coalition evidence. This is continuous evidence gating, not policy mixing.\n    total_input = float(np.sum([\n        attention * float(cue_strength[j]) for j in discriminating\n    ]))\n    coalition_margin = abs(coalition) / max(total_input, 1e-9)\n    coalition_margin = float(np.clip(coalition_margin, 0.0, 1.0))\n    margin_access = 0.10 + 0.90 * coalition_margin / (coalition_margin + 0.20)\n    effective_coalition_weight = (\n        coalition_weight * (1.0 - overwrite_task_set) * margin_access\n    )\n    decision_evidence = (\n        (1.0 - effective_coalition_weight) * accumulator\n        + effective_coalition_weight * coalition\n    )\n\n    # Confidence follows the final evidence margin. There is deliberately no\n    # fatigue or count penalty, allowing coherent long reversals to remain\n    # reliable while balanced conflict naturally remains uncertain.\n    margin = abs(decision_evidence) / max(total_input, 1e-9)\n    geometry_gain = 0.75 + 0.50 * np.sqrt(np.clip(margin, 0.0, 1.0))\n    decision_variable = response_gain * geometry_gain * decision_evidence\n    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))\n\n    p_a = 1.0 / (1.0 + np.exp(-decision_variable))\n    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5], dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "attention_rate": "[0.88, 1.0]",
    "integration_retention": "[0.84, 0.97]",
    "overwrite_retention": "[0.14, 0.36]",
    "conflict_inhibition": "[0.35, 1.20]",
    "coalition_weight": "[0.30, 0.62]",
    "validity_sensitivity": "[0.15, 0.45]",
    "context_sensitivity": "[1.4, 2.4]",
    "context_threshold": "[1.0, 1.6]",
    "context_learning_rate": "[0.10, 0.24]",
    "geometry_context_weight": "[0.35, 0.65]",
    "profile_context_weight": "[0.80, 1.30]",
    "response_gain": "[1.8, 3.8]",
    "lapse_rate": "[0.04, 0.15]"
  },
  "rationale": "This is a targeted edit of the accepted iteration-3 model. Its local accumulator, static validity exponent, coalition-margin gate, bounded current-trial calibration, response rule, and physical-gap exclusion are unchanged. The edit only enriches the learned overwrite task set with the conflict-strength information identified as missing by the critic. Each historical sequence now contributes three validity-weighted summaries: terminal-run share, final coalition margin, and reversal-location-by-run-balance. Two experiment-level validity-profile summaries—the terminal-to-prefix log-odds ratio and ordered log-odds slope—are also added. Narrow subject-level weights map these summaries into the same single continuous overwrite task set. Recurrent late balanced attacks therefore promote overwrite, whereas clear weighted coalitions and steep validity profiles promote integration. This can separate experiments with similar raw run counts but different validity profiles and conflict-strength distributions, while avoiding the rejected changes to local inhibition, coalition access, or task-dependent validity sensitivity. Because all summaries use validity scan order rather than physical indices, the near-zero physical-foregrounding prediction is preserved."
}
```

## Usage

```json
{
  "input_tokens": 31795,
  "output_tokens": 6646
}
```
