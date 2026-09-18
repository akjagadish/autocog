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
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a task-invariant, configuration-sensitive dual-route theory rather than another uniformly smooth evidence field. The new theory should posit that cues form perceptual coalitions or runs during sequential inspection. A bounded decision state then alternates between two operations: integrating support within a coherent coalition and switching commitment when an opposing cue or coalition produces sufficient relative surprise. The probability and strength of a switch should depend continuously on incumbent coalition strength, opposing-run coherence, discriminating-cue load, instructed reliability, and encoding accessibility. It should not assume either categorical cue-by-cue overwrite, as pi_3 does, or obligatory linear summation of independently retained traces, as pi_11 does.

Instructed validities should primarily affect attention and switch thresholds and should be strongly underweighted, rather than entering as reliably signed compressed log-odds. The theory should also distinguish serial processing time or absolute display position from ordinal position among discriminating cues. That distinction is necessary because Experiment 2 confounds terminal validity with terminal column: the negative contrast could reflect enhanced accessibility of later-displayed weak cues rather than inverse validity use. This spatial/temporal contribution must remain modest and interact with coalition state, given the near-null foregrounding result in Experiment 16.

The model should include a confidence-gated readout: near-balanced, highly conflicting configurations can collapse toward chance, accounting for Experiment 1, while a coherent incumbent followed by a salient singleton reversal can still produce strong terminal control, as in Experiments 12, 13, and 15. Load should reduce precision mainly under unresolved conflict, with only weak degradation for unanimous evidence, matching Experiment 17 better than pi_3's general fatigue. Stable subject-level differences in validity use, switch threshold, and readout noise should be represented to match the observed variances. Future predictions should jointly target levels and contrasts across terminal conformity, tally-versus-first-cue conflict, terminal-validity gradients, coherent versus fragmented prefixes, and unanimous load; parameter ranges should be rejected unless one shared range reproduces the sign of Experiment 2 and avoids pi_11's large errors in Experiments 1, 3, 8, 11, and 13.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.710)

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


---

### `pi_8` (overall score: 0.597)

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


---

### `pi_6` (overall score: 0.583)

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
- Experiment 19: real=0.4978 (var=0.0077) vs this=0.7239 (var=0.0096)
- Experiment 20: real=-0.1813 (var=0.0102) vs this=0.1275 (var=0.0180)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3096 -> ACCEPTED
- iter 2: loss=0.3657 -> REJECTED
- iter 3: loss=0.3588 -> REJECTED
- iter 4: loss=0.3406 -> REJECTED
- iter 5: loss=1.0000 -> REJECTED
- iter 6: loss=0.3177 -> REJECTED
- iter 7: loss=0.3130 -> REJECTED
- iter 8: loss=0.2795 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 8 at loss=0.2795 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.8876 (var=0.0025) (Δ vs real +0.3932)
  - iter 2: 0.9107 (var=0.0023) (Δ vs real +0.4163)
  - iter 3: 0.8528 (var=0.0046) (Δ vs real +0.3583)
  - iter 4: 0.8628 (var=0.0046) (Δ vs real +0.3683)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.8762 (var=0.0031) (Δ vs real +0.3818)
  - iter 7: 0.8894 (var=0.0031) (Δ vs real +0.3950)
  - iter 8 (most recent): 0.8781 (var=0.0030) (Δ vs real +0.3836)
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
  - iter 1: 0.1454 (var=0.0028) (Δ vs real -0.0483)
  - iter 2: 0.1537 (var=0.0019) (Δ vs real -0.0400)
  - iter 3: 0.2146 (var=0.0044) (Δ vs real +0.0208)
  - iter 4: 0.1462 (var=0.0021) (Δ vs real -0.0475)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.1504 (var=0.0028) (Δ vs real -0.0433)
  - iter 7: 0.1204 (var=0.0020) (Δ vs real -0.0733)
  - iter 8 (most recent): 0.1335 (var=0.0026) (Δ vs real -0.0602)
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
  - iter 1: 0.4288 (var=0.0201) (Δ vs real -0.1462)
  - iter 2: 0.4529 (var=0.0073) (Δ vs real -0.1221)
  - iter 3: 0.3550 (var=0.0094) (Δ vs real -0.2200)
  - iter 4: 0.4444 (var=0.0155) (Δ vs real -0.1306)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.4296 (var=0.0158) (Δ vs real -0.1454)
  - iter 7: 0.5233 (var=0.0151) (Δ vs real -0.0517)
  - iter 8 (most recent): 0.4981 (var=0.0166) (Δ vs real -0.0769)
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
  - iter 1: 0.3971 (var=0.0278) (Δ vs real -0.2679)
  - iter 2: 0.2442 (var=0.0164) (Δ vs real -0.4208)
  - iter 3: -0.0312 (var=0.0422) (Δ vs real -0.6963)
  - iter 4: 0.3592 (var=0.0387) (Δ vs real -0.3058)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.2692 (var=0.0381) (Δ vs real -0.3958)
  - iter 7: 0.5975 (var=0.0144) (Δ vs real -0.0675)
  - iter 8 (most recent): 0.4871 (var=0.0345) (Δ vs real -0.1779)
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
  - iter 1: -0.4071 (var=0.0499) (Δ vs real -0.4104)
  - iter 2: -0.4950 (var=0.0395) (Δ vs real -0.4983)
  - iter 3: -0.2817 (var=0.0865) (Δ vs real -0.2850)
  - iter 4: -0.4933 (var=0.0181) (Δ vs real -0.4967)
  - iter 5: n/a (var=n/a)
  - iter 6: -0.3742 (var=0.0422) (Δ vs real -0.3775)
  - iter 7: -0.5333 (var=0.0239) (Δ vs real -0.5367)
  - iter 8 (most recent): -0.4833 (var=0.0305) (Δ vs real -0.4867)
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
  - iter 1: 0.2950 (var=0.0155) (Δ vs real -0.0117)
  - iter 2: 0.1487 (var=0.0056) (Δ vs real -0.1579)
  - iter 3: 0.5763 (var=0.0308) (Δ vs real +0.2696)
  - iter 4: 0.3262 (var=0.0208) (Δ vs real +0.0196)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.3212 (var=0.0205) (Δ vs real +0.0146)
  - iter 7: 0.2121 (var=0.0090) (Δ vs real -0.0946)
  - iter 8 (most recent): 0.3125 (var=0.0249) (Δ vs real +0.0058)
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
  - iter 1: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0900) (Δ vs real +1.0000)
  - iter 3: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.2464) (Δ vs real +1.0000)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 7: 1.0000 (var=0.2100) (Δ vs real +1.0000)
  - iter 8 (most recent): 0.0000 (var=0.1924) (Δ vs real +0.0000)
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
  - iter 1: 0.6196 (var=0.0062) (Δ vs real +0.4475)
  - iter 2: 0.6927 (var=0.0028) (Δ vs real +0.5206)
  - iter 3: 0.5685 (var=0.0029) (Δ vs real +0.3965)
  - iter 4: 0.6181 (var=0.0052) (Δ vs real +0.4460)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.5988 (var=0.0070) (Δ vs real +0.4267)
  - iter 7: 0.6656 (var=0.0057) (Δ vs real +0.4935)
  - iter 8 (most recent): 0.6408 (var=0.0058) (Δ vs real +0.4688)
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
  - iter 1: 0.3571 (var=0.0168) (Δ vs real +0.3896)
  - iter 2: 0.2454 (var=0.0186) (Δ vs real +0.2779)
  - iter 3: 0.4804 (var=0.0135) (Δ vs real +0.5129)
  - iter 4: 0.3162 (var=0.0228) (Δ vs real +0.3488)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.3192 (var=0.0234) (Δ vs real +0.3517)
  - iter 7: 0.1750 (var=0.0180) (Δ vs real +0.2075)
  - iter 8 (most recent): 0.2188 (var=0.0191) (Δ vs real +0.2512)
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
  - iter 1: 0.9253 (var=0.0012) (Δ vs real +0.0506)
  - iter 2: 0.9303 (var=0.0010) (Δ vs real +0.0555)
  - iter 3: 0.8747 (var=0.0015) (Δ vs real -0.0001)
  - iter 4: 0.9095 (var=0.0013) (Δ vs real +0.0347)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.9190 (var=0.0009) (Δ vs real +0.0442)
  - iter 7: 0.9390 (var=0.0007) (Δ vs real +0.0643)
  - iter 8 (most recent): 0.9297 (var=0.0009) (Δ vs real +0.0549)
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
  - iter 1: 0.1008 (var=0.0055) (Δ vs real -0.7417)
  - iter 2: 0.2313 (var=0.0322) (Δ vs real -0.6113)
  - iter 3: 0.0512 (var=0.0018) (Δ vs real -0.7913)
  - iter 4: 0.1200 (var=0.0113) (Δ vs real -0.7225)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.0867 (var=0.0043) (Δ vs real -0.7558)
  - iter 7: 0.3400 (var=0.0281) (Δ vs real -0.5025)
  - iter 8 (most recent): 0.1454 (var=0.0123) (Δ vs real -0.6971)
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
  - iter 1: 0.6875 (var=0.0261) (Δ vs real +0.5162)
  - iter 2: 0.0988 (var=0.0052) (Δ vs real -0.0725)
  - iter 3: 0.9069 (var=0.0031) (Δ vs real +0.7356)
  - iter 4: 0.6500 (var=0.0330) (Δ vs real +0.4788)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.6600 (var=0.0249) (Δ vs real +0.4887)
  - iter 7: 0.2462 (var=0.0172) (Δ vs real +0.0750)
  - iter 8 (most recent): 0.5144 (var=0.0291) (Δ vs real +0.3431)
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
  - iter 1: 0.5891 (var=0.0039) (Δ vs real -0.2918)
  - iter 2: 0.6402 (var=0.0014) (Δ vs real -0.2407)
  - iter 3: 0.5112 (var=0.0036) (Δ vs real -0.3698)
  - iter 4: 0.6012 (var=0.0040) (Δ vs real -0.2798)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.5858 (var=0.0043) (Δ vs real -0.2952)
  - iter 7: 0.7002 (var=0.0043) (Δ vs real -0.1807)
  - iter 8 (most recent): 0.6370 (var=0.0043) (Δ vs real -0.2439)
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
  - iter 1: 0.0400 (var=0.0045) (Δ vs real +0.0208)
  - iter 2: 0.1000 (var=0.0100) (Δ vs real +0.0808)
  - iter 3: 0.0108 (var=0.0019) (Δ vs real -0.0083)
  - iter 4: 0.0467 (var=0.0042) (Δ vs real +0.0275)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.0583 (var=0.0070) (Δ vs real +0.0392)
  - iter 7: 0.1054 (var=0.0107) (Δ vs real +0.0862)
  - iter 8 (most recent): 0.0683 (var=0.0092) (Δ vs real +0.0492)
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
  - iter 1: -0.0106 (var=0.0021) (Δ vs real -0.0577)
  - iter 2: -0.0192 (var=0.0016) (Δ vs real -0.0663)
  - iter 3: -0.0158 (var=0.0022) (Δ vs real -0.0629)
  - iter 4: -0.0000 (var=0.0019) (Δ vs real -0.0471)
  - iter 5: n/a (var=n/a)
  - iter 6: -0.0129 (var=0.0019) (Δ vs real -0.0600)
  - iter 7: -0.0208 (var=0.0030) (Δ vs real -0.0679)
  - iter 8 (most recent): -0.0060 (var=0.0027) (Δ vs real -0.0531)
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
  - iter 1: 0.0014 (var=0.0047) (Δ vs real -0.0297)
  - iter 2: 0.0257 (var=0.0070) (Δ vs real -0.0055)
  - iter 3: -0.0157 (var=0.0056) (Δ vs real -0.0468)
  - iter 4: -0.0061 (var=0.0058) (Δ vs real -0.0372)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.0066 (var=0.0069) (Δ vs real -0.0245)
  - iter 7: -0.0016 (var=0.0086) (Δ vs real -0.0327)
  - iter 8 (most recent): 0.0148 (var=0.0060) (Δ vs real -0.0164)
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
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: n/a (var=n/a)
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
  - iter 1: -0.0381 (var=0.0087) (Δ vs real -0.0644)
  - iter 2: -0.1544 (var=0.0110) (Δ vs real -0.1806)
  - iter 3: -0.0163 (var=0.0037) (Δ vs real -0.0425)
  - iter 4: -0.0281 (var=0.0079) (Δ vs real -0.0544)
  - iter 5: n/a (var=n/a)
  - iter 6: -0.0556 (var=0.0078) (Δ vs real -0.0819)
  - iter 7: -0.0956 (var=0.0180) (Δ vs real -0.1219)
  - iter 8 (most recent): -0.0625 (var=0.0090) (Δ vs real -0.0887)
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
  - iter 1: 0.7256 (var=0.0066) (Δ vs real +0.2278)
  - iter 2: 0.7811 (var=0.0067) (Δ vs real +0.2833)
  - iter 3: 0.4122 (var=0.0274) (Δ vs real -0.0856)
  - iter 4: 0.5044 (var=0.0245) (Δ vs real +0.0067)
  - iter 5: n/a (var=n/a)
  - iter 6: 0.7172 (var=0.0121) (Δ vs real +0.2194)
  - iter 7: 0.7211 (var=0.0082) (Δ vs real +0.2233)
  - iter 8 (most recent): 0.7339 (var=0.0093) (Δ vs real +0.2361)
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
  - iter 1: -0.0194 (var=0.0098) (Δ vs real +0.1619)
  - iter 2: -0.3856 (var=0.0251) (Δ vs real -0.2044)
  - iter 3: 0.0275 (var=0.0042) (Δ vs real +0.2087)
  - iter 4: -0.0225 (var=0.0096) (Δ vs real +0.1587)
  - iter 5: n/a (var=n/a)
  - iter 6: -0.0169 (var=0.0060) (Δ vs real +0.1644)
  - iter 7: -0.2494 (var=0.0157) (Δ vs real -0.0681)
  - iter 8 (most recent): -0.0906 (var=0.0149) (Δ vs real +0.0906)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Surprise-Bound Coalition Relay. During left-to-right inspection, discriminating cues are compressed into directional coalitions: consecutive cues favoring the same option strengthen one bounded commitment rather than leaving independently summable traces. An opposing coalition initiates a graded contest with the incumbent. Switching is probabilistic and depends jointly on incumbent strength, challenger coherence, relative instructed reliability, terminal singleton salience, and the challenger's absolute display accessibility. Instructed validity therefore modulates encoding and switching thresholds but is not converted into signed additive evidence. Absolute display position is represented separately from ordinal position in the discriminating sequence. The final response averages over unresolved commitment trajectories and is confidence-gated: balanced evidence, ambiguous switches, and fragmented sequences approach chance, whereas a decisive coalition switch can be expressed confidently. Load impairs precision primarily when conflict remains unresolved; unanimous evidence receives only a small load-dependent decline. At a coalition boundary, a modest, regularized drop from the incumbent's exit reliability to the challenger's entry reliability contributes relative surprise, preserving local boundary information without allowing validity differences to dominate the contest.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Surprise-Bound Coalition Relay expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    base_attention = float(parameters["base_attention"])
    validity_use = float(parameters["validity_use"])
    position_accessibility = float(parameters["position_accessibility"])
    coalition_capacity = float(parameters["coalition_capacity"])
    coherence_gain = float(parameters["coherence_gain"])
    switch_threshold = float(parameters["switch_threshold"])
    switch_precision = float(parameters["switch_precision"])
    relative_surprise = float(parameters["relative_surprise"])
    terminal_salience = float(parameters["terminal_salience"])
    opposition_erosion = float(parameters["opposition_erosion"])
    response_gain = float(parameters["response_gain"])
    conflict_collapse = float(parameters["conflict_collapse"])
    load_noise = float(parameters["load_noise"])
    unanimous_decay = float(parameters["unanimous_decay"])
    lapse = float(parameters["lapse_rate"])

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    m = int(discriminating.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Reliability affects attentional accessibility and switch thresholds,
    # never the sign of an additive evidence trace.
    reliability = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)
    base_logit = np.log(
        np.clip(base_attention, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - base_attention, 1e-5, 1.0)
    )

    ordinal_directions = np.sign(diff[discriminating]).astype(float)
    cue_access = []
    for j in discriminating:
        absolute_position = float(j) / float(max(1, n_features - 1))
        attention = sigmoid(
            base_logit
            + validity_use * (float(reliability[j]) - 0.5)
            + position_accessibility * (absolute_position - 0.5)
        )
        # A modest independent accessibility advantage for later display
        # locations is retained after attentional gating.
        access = attention * (
            1.0 + position_accessibility * 0.45 * absolute_position
        )
        cue_access.append(float(access))
    cue_access = np.asarray(cue_access, dtype=float)

    # Form ordinal coalitions. Physical locations remain attached to each
    # coalition, so discriminating ordinal position and display position are
    # not conflated.
    runs = []
    start = 0
    while start < m:
        direction = float(ordinal_directions[start])
        end = start + 1
        while end < m and ordinal_directions[end] == direction:
            end += 1

        loc = slice(start, end)
        run_indices = discriminating[loc]
        raw_access = float(np.sum(cue_access[loc]))
        run_length = int(end - start)

        # Bounded within-coalition integration. Coherence increases strength
        # sublinearly and cannot create an unbounded linear sum.
        bounded = coalition_capacity * (
            1.0 - np.exp(-raw_access / max(coalition_capacity, 1e-8))
        )
        coherence = 1.0 + coherence_gain * (
            1.0 - 1.0 / float(run_length)
        )
        strength = min(coalition_capacity, bounded * coherence)

        weights = cue_access[loc]
        if float(np.sum(weights)) > 0.0:
            mean_reliability = float(
                np.average(reliability[run_indices], weights=weights)
            )
        else:
            mean_reliability = float(np.mean(reliability[run_indices]))

        final_physical_position = float(run_indices[-1]) / float(
            max(1, n_features - 1)
        )
        runs.append({
            "direction": direction,
            "strength": float(strength),
            "length": run_length,
            "reliability": mean_reliability,
            "entry_reliability": float(reliability[run_indices[0]]),
            "exit_reliability": float(reliability[run_indices[-1]]),
            "final_position": final_physical_position,
            "ends_sequence": bool(end == m)
        })
        start = end

    # Each entry is a possible commitment trajectory. Branching implements a
    # genuinely graded switch rather than deterministic overwrite.
    states = [{
        "weight": 1.0,
        "direction": runs[0]["direction"],
        "strength": runs[0]["strength"],
        "reliability": runs[0]["reliability"],
        "exit_reliability": runs[0]["exit_reliability"]
    }]
    switch_ambiguities = []

    for run in runs[1:]:
        next_states = []
        weighted_ambiguity = 0.0
        total_state_weight = 0.0

        for incumbent in states:
            w = float(incumbent["weight"])
            incumbent_strength = float(incumbent["strength"])
            challenger_strength = float(run["strength"])

            reliability_difference = (
                float(incumbent["reliability"])
                - float(run["reliability"])
            )
            threshold_multiplier = 1.0 + validity_use * reliability_difference
            threshold_multiplier = float(np.clip(
                threshold_multiplier, 0.55, 1.55
            ))

            relative_advantage = (
                challenger_strength - incumbent_strength
            ) / max(challenger_strength + incumbent_strength, 1e-8)

            is_terminal_singleton = float(
                run["ends_sequence"] and run["length"] == 1
            )
            singleton_access = (
                is_terminal_singleton
                * terminal_salience
                * (0.55 + 0.45 * float(run["final_position"]))
            )

            # Switching is initiated at the actual coalition boundary. A
            # downward reliability discontinuity can be surprising, but its
            # influence is bounded and strongly regularized. The coefficient
            # is deliberately intermediate rather than the rejected 0.75
            # boundary weight.
            boundary_drop = max(
                0.0,
                float(incumbent["exit_reliability"])
                - float(run["entry_reliability"])
            )
            boundary_surprise = (
                0.42 * boundary_drop / (0.50 + boundary_drop)
            )

            contest = (
                challenger_strength
                - switch_threshold
                * threshold_multiplier
                * incumbent_strength
                + relative_surprise * relative_advantage
                + singleton_access
                + boundary_surprise
            )
            q_switch = sigmoid(switch_precision * contest)

            weighted_ambiguity += w * 4.0 * q_switch * (1.0 - q_switch)
            total_state_weight += w

            # If the challenger is resisted, conflict still erodes incumbent
            # commitment. If it wins, its bounded coalition becomes incumbent.
            retained_strength = max(
                0.08,
                incumbent_strength
                - opposition_erosion * challenger_strength
            )
            switched_strength = min(
                coalition_capacity,
                challenger_strength
                + 0.10 * q_switch * incumbent_strength
            )

            if q_switch < 1.0:
                next_states.append({
                    "weight": w * (1.0 - q_switch),
                    "direction": float(incumbent["direction"]),
                    "strength": float(retained_strength),
                    "reliability": float(incumbent["reliability"]),
                    "exit_reliability": float(incumbent["exit_reliability"])
                })
            if q_switch > 0.0:
                next_states.append({
                    "weight": w * q_switch,
                    "direction": float(run["direction"]),
                    "strength": float(switched_strength),
                    "reliability": float(run["reliability"]),
                    "exit_reliability": float(run["exit_reliability"])
                })

        states = next_states
        if total_state_weight > 0.0:
            switch_ambiguities.append(
                weighted_ambiguity / total_state_weight
            )

    # Accessible support is used only to diagnose unresolved conflict. It is
    # not read out as a linear evidence total.
    positive_mass = float(np.sum(
        cue_access[ordinal_directions > 0.0]
    ))
    negative_mass = float(np.sum(
        cue_access[ordinal_directions < 0.0]
    ))
    total_mass = positive_mass + negative_mass
    if positive_mass <= 0.0 or negative_mass <= 0.0 or total_mass <= 0.0:
        balance = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))

    ambiguity = (
        float(np.mean(switch_ambiguities))
        if switch_ambiguities else 0.0
    )
    fragmentation = max(0.0, float(len(runs) - 2)) / float(max(1, m - 1))
    unresolved = balance * (
        0.20 + 0.55 * ambiguity + 0.45 * fragmentation
    )
    unresolved = float(np.clip(unresolved, 0.0, 1.0))

    confidence = max(0.08, 1.0 - conflict_collapse * unresolved)
    overload = max(0.0, float(m) - 4.0)
    conflict_divisor = 1.0 + load_noise * overload * unresolved * unresolved

    # Unanimous evidence avoids conflict noise but shows a slight empirical
    # decline at high loads rather than ever-increasing certainty.
    if balance == 0.0:
        unanimous_divisor = 1.0 + unanimous_decay * max(0.0, float(m) - 2.0)
    else:
        unanimous_divisor = 1.0

    trajectory_weight = float(np.sum([s["weight"] for s in states]))
    if trajectory_weight <= 0.0 or not np.isfinite(trajectory_weight):
        return np.array([0.5, 0.5], dtype=np.float64)

    p_a_core = 0.0
    for commitment in states:
        decision_variable = (
            response_gain
            * confidence
            * float(commitment["direction"])
            * float(commitment["strength"])
            / (conflict_divisor * unanimous_divisor)
        )
        p_a_state = sigmoid(decision_variable)
        p_a_core += float(commitment["weight"]) * p_a_state
    p_a_core /= trajectory_weight
    p_a_core = float(np.clip(p_a_core, 0.0, 1.0))

    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
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
- base_attention: [0.82, 0.95]
- validity_use: [0.22, 0.52]
- position_accessibility: [0.28, 0.58]
- coalition_capacity: [1.75, 2.45]
- coherence_gain: [0.10, 0.30]
- switch_threshold: [0.62, 0.88]
- switch_precision: [3.8, 6.2]
- relative_surprise: [0.45, 0.90]
- terminal_salience: [0.42, 0.78]
- opposition_erosion: [0.20, 0.42]
- response_gain: [2.5, 4.4]
- conflict_collapse: [0.48, 0.76]
- load_noise: [0.025, 0.085]
- unanimous_decay: [0.010, 0.032]
- lapse_rate: [0.04, 0.14]

`rationale`: This is a narrowly isolated interpolation from the accepted candidate. All attention, coalition integration, terminal salience, erosion, confidence, load, and subject-level parameter ranges are unchanged. The only mechanistic edit is to retain each coalition's entry and exit reliability and add a modest boundary-surprise term based on the actual incumbent-exit/challenger-entry discontinuity. The term uses weight 0.42 and a 0.50 regularizer, substantially weaker than the rejected full-strength boundary implementation. It should preserve part of that implementation's gains in Experiments 3, 4, 11–13, and 20 while reducing its collateral switching in Experiments 2, 5–8 and avoiding an excessive Experiment 20 gradient. Because only downward boundary discontinuities add surprise, ordinary validity differences continue to operate mainly through the accepted run-average threshold and attention mechanisms rather than becoming signed additive evidence.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed coalition-and-switch family and improves several important targets: Experiment 2 has the required low first-cue conformity and approximately correct level; Experiments 6, 7, 10, 14, 16, and 17 are also close. However, the fit is not experiment-invariant. Large level errors remain in both directions. Terminal control is far too weak in Experiments 3 (0.429 vs 0.575), 4 (0.397 vs 0.665), 11 (0.101 vs 0.843), 12 (majority choice 0.688 vs 0.171, implying insufficient late-minority control), and 13 (0.589 vs 0.881). At the same time, terminal conformity is much too strong in Experiments 8 (0.620 vs 0.172) and 19 (0.726 vs 0.498). The model also exaggerates tally/configuration separation in Experiment 1 (0.888 vs 0.494), produces an artifactual block-versus-mixed contrast in Experiment 5 (-0.407 vs 0.003), retains too much first-cue influence in Experiment 9 (0.357 vs -0.033), and nearly misses the inverse terminal-strength gradient in Experiment 20 (-0.019 vs -0.181). Thus, terminal salience is not merely globally too high or too low; it is insufficiently gated by configuration. Between-subject consistency is also miscalibrated: variance is much too low in Experiments 1 and 10 but too high in Experiments 3, 5, and 12.
Rationale: Keep the Surprise-Bound Coalition Relay family, but revise the contest and confidence gates. The current additive `terminal_salience` bonus and strongly saturated run strengths generate generic recency rather than the required configuration-sensitive reversals. Make terminal accessibility interact more sharply with incumbent/challenger coherence, physical delay or absolute position, discriminating load, and relative surprise; a late cue or suffix should not receive nearly the same effective boost in the contiguous two-run displays of Experiment 8 as in the decisive reversal configurations of Experiments 11–13. In particular, distinguish terminal suffixes from terminal singletons and allow a coherent late suffix to trigger a decisive switch without relying on the present singleton-only bonus. Conversely, increase confidence collapse specifically for balanced, fragmented, multi-switch trajectories to pull Experiments 1, 9, and 19 toward chance without attenuating a resolved two-coalition switch. The very large Experiment 5 contrast indicates that resisted challenges and `opposition_erosion` leave an excessive memory of run geometry; reduce that asymmetry or permit graded reinstatement when a later run returns to a previously supported direction. Finally, strengthen the interaction—not the main effect—between physical accessibility and underweighted validity so that late weak terminals can outperform earlier stronger terminals in Experiment 20 while preserving the near-null foregrounding contrast in Experiment 14. Retune subject-level ranges after correcting these structural gates: widen heterogeneity in validity use/readout noise where simulated variance is compressed, but narrow switch/erosion heterogeneity responsible for the excessive variance in Experiments 3, 5, and 12.

**Outcome of this advice:** iter 1 candidate loss=0.3096 -> iter 2 candidate loss=0.3657 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed dual-route coalition-and-switch family, but it is not experiment-invariant and was correctly rejected by the loss gate. The revision improved some intended targets—Experiment 11 rose from 0.101 to 0.231, Experiment 13 from 0.589 to 0.640, Experiment 9 moved toward zero, and Experiment 20 acquired the required negative contrast—but these gains were outweighed by substantial collateral failures. Experiment 4 deteriorated from 0.397 to 0.244 versus 0.665 observed; Experiment 5 moved farther from its null contrast (-0.495 versus 0.003); Experiment 6 fell from a near-exact 0.295 to 0.149 versus 0.307; Experiment 7 crossed the diagnostic threshold incorrectly; Experiment 8 increased to 0.693 versus 0.172; and Experiments 18 and 19 also worsened. The new gate additionally overshot Experiments 12 and 20: their directions are now useful, but their magnitudes are too extreme. Major accepted-base failures also remain in Experiments 1, 5, 8, 11, and 13. The low variances in Experiments 1, 10, and 13 continue to indicate overly homogeneous behavior.
Rationale: The previous broad push toward reinstatement, sharper conflict collapse, and a compound terminal-configuration bonus was rejected by the gate, so do not repeat that package. Work from the unchanged accepted base and make a more localized in-family revision. First, separate validity-dependent attention from validity-dependent incumbent stabilization rather than controlling both with one `validity_use` parameter. This would allow steep-validity contiguous displays such as Experiment 8 to resist a weak terminal coalition while absolute position and delay can still produce the negative gradient in Experiment 20 and the sparse terminal reversals in Experiment 11. Use one bounded position/delay term in the switch contest; avoid the current multiple counting through attention, multiplicative accessibility, terminal position, delay, and `delayed_weak_surprise`. The current compound term produces excessive switching in Experiments 6, 8, 18, and 20.

Second, replace the unsuccessful explicit reinstatement rule with symmetric bounded updating of incumbent strength after any resisted or supporting run. The reinstatement edit increased rather than removed the Experiment 5 block-versus-mixed artifact, so returning support should not receive a special strengthening operation. Calibrate this update so permutation geometry changes confidence modestly but does not create a roughly 50-point contrast.

Third, implement confidence gating from the final directional mixture of commitment trajectories—for example, entropy or absolute directional-weight imbalance—rather than further increasing the current mass-balance/mean-switch-ambiguity collapse. The prior sharpening failed to reduce Experiment 1 and worsened Experiment 7. A final-state mixture gate should selectively move genuinely split outcomes in Experiments 1, 9, and 19 toward chance while leaving a resolved terminal switch in Experiments 11–13 expressive. Keep terminal and gradient changes smaller than in this rejected attempt: Experiment 12 should lie between the accepted and current predictions, and Experiment 20 likewise needs an intermediate interaction strength. Only after level fit improves should subject ranges be retuned, chiefly by adding heterogeneity to the separated validity-threshold and readout-noise parameters rather than widening switch or erosion ranges.

**Outcome of this advice:** iter 2 candidate loss=0.3657 -> iter 3 candidate loss=0.3588 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed coalition-and-switch family, but it is not experiment-invariant and was rejected by the accept gate. It fits Experiments 2, 7, 10, 14, and 17, and improves Experiment 19 relative to the accepted base. Those gains are outweighed by severe regressions. Terminal control is now drastically too weak in Experiments 3 (0.355 vs 0.575), 4 (-0.031 vs 0.665), 11 (0.051 vs 0.843), and 13 (0.511 vs 0.881). Experiment 12 reverses sharply in the wrong direction, predicting 0.907 majority choice where humans show 0.171, and Experiment 6 changes from a near match in the accepted model to excessive early-coalition choice (0.576 vs 0.307). First-cue influence is also much too strong in Experiment 9 (0.480 vs -0.033). Meanwhile, terminal conformity remains excessive in Experiment 8 (0.569 vs 0.172), the Experiment 5 permutation contrast remains far from zero (-0.282 vs 0.003), and Experiment 20 loses the required negative terminal-position/validity gradient. Experiment 1 remains much too categorical and much less heterogeneous than the human data. Thus the new validity stabilization and readout package has produced excessive incumbent lock-in without solving the accepted model's configuration artifact.
Rationale: Both previous compound revisions were rejected, so do not repeat the broad terminal-gating package or the latest combination of separated validity stabilization, final-mixture collapse, and symmetric updating. Work from the unchanged iteration-1 base and make a narrowly isolated in-family change. The best next target is the persistent Experiment 5 artifact, using an order-neutral normalization of bounded coalition strength rather than another terminal bonus. The current subtractive opposition erosion and capacity-clipped support update are non-associative: partitioning the same directional support into different runs changes retained strength substantially, which makes block and mixed permutations diverge. Represent bounded strength in a latent occupancy coordinate, such as z = -log(1-s/C), combine same-direction support additively in z, and implement opposition as proportional or ratio-normalized erosion before mapping back to bounded strength. This preserves coalition compression and graded switching while making equal directional mass less sensitive to arbitrary run partitioning; it may also soften the exaggerated contrasts in Experiments 1 and 9. Keep the accepted base's validity, terminal-accessibility, confidence, and subject-range settings fixed during this test, because the current attempt shows that changing those jointly destroys the previously accurate Experiment 6 and worsens Experiments 3, 4, 11, 12, and 13. Only if the isolated normalization improves loss should terminal failures be tuned subsequently with small parameter steps. Variance tuning should also wait: current level errors dominate, although the model still underproduces heterogeneity in Experiments 1, 10, and 13.

**Outcome of this advice:** iter 3 candidate loss=0.3588 -> iter 4 candidate loss=0.3406 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed coalition-and-switch family, but it was rejected by the accept gate and is not experiment-invariant. The latent-occupancy revision did not solve its intended target: Experiment 5 worsened from the accepted base's -0.407 to -0.493, versus 0.003 observed. It also changed Experiment 7 from the correct diagnostic category to the wrong one and increased aggregate loss from 0.3096 to 0.3406. Several severe accepted-base failures remain: Experiment 1 is too categorical (0.863 vs 0.494), terminal control is far too weak in Experiments 3, 4, 11, and 13, majority-coalition choice is far too high in Experiment 12, and terminal conformity is far too high in Experiment 8. Experiment 9 retains excessive first-cue control, while Experiment 20 still lacks the required negative terminal-validity gradient. The close fits in Experiments 2, 6, 10, 14, 17, and 19 do not compensate for these large, opposing errors. The central diagnostic is that the model cannot differentiate configurations that require terminal resistance from those requiring terminal switching. In particular, Experiments 8 and 13 contain closely related contiguous two-run geometries but show radically different human terminal conformity; task-specific instructed-validity configurations therefore need to influence the switch threshold more effectively than the current weak run-mean reliability difference. Subject-level variance is also generally too compressed in Experiments 1, 10, and 13, but level fit remains the priority.
Rationale: The previous order-neutral occupancy recommendation was rejected and actually enlarged the Experiment 5 error, so do not repeat or extend that normalization. Build again from the unchanged iteration-1 accepted base. Try a different, narrowly isolated in-family revision: change only how instructed reliability enters the coalition switch threshold. The current weighted mean reliability discards the shape of the validity profile within a coalition, and multiplying its small difference by `validity_use` yields nearly task-insensitive contests. Replace it with a bounded, within-task-normalized coalition reliability contrast that retains local rank or validity-gap information, then let that contrast modulate the threshold—not additive evidence or the final response. Keep the effect strongly underweighted and test it without simultaneously changing attention, terminal salience, erosion, confidence collapse, or load noise. This should allow the shared mechanism to distinguish the terminal-resistant validity configuration of Experiment 8 from the terminal-switching configurations of Experiments 11–13, while providing a controlled route to the negative contrast in Experiment 20. Avoid another compound terminal-gating package, separated stabilization system, or final-mixture confidence rewrite, since those previous directions were rejected. Retain the accepted base's update geometry initially and use a small parameter search around the new threshold statistic; reject settings unless they preserve Experiments 2, 6, 7, 10, 14, and 17 while improving both sides of the Experiment 8 versus 11–13 dissociation. Only after that level dissociation improves should subject heterogeneity be widened.

**Outcome of this advice:** iter 4 candidate loss=0.3406 -> iter 5 candidate loss=1.0000 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is unscorable: every experiment returns n/a and aggregate loss is 1.0000, so there is no empirical evidence that the profile-normalized reliability revision improves the accepted iteration-1 model. The most likely cause is the new call to `scipy.stats.rankdata` without an accompanying import or guarantee that SciPy is available in the execution environment. Because this is the only newly introduced external dependency and all experiments fail simultaneously, a runtime exception is more plausible than a substantive metric failure. Conceptually, the candidate remains faithful to the prescribed coalition-and-switch family, but fit quality cannot be evaluated. The unchanged accepted base is itself not adequate, with major errors in Experiments 1, 3–5, 8, 9, and 11–13, so it cannot simply be shipped.
Rationale: The gate rejected this candidate, and a completely n/a result cannot be continued. Build from the unchanged iteration-1 accepted base and first restore a fully executable prediction function. Remove the undeclared SciPy dependency: compute fractional ranks with NumPy or omit the small rank component and use a purely bounded, regularized validity-gap statistic. Add smoke tests that call `predict` on every feature count in Experiments 1–20 and verify finite, normalized two-choice probabilities before simulation. Keep all other accepted-base mechanisms and parameter ranges fixed for this diagnostic run. The previous recommendation was formally rejected, but because the candidate never generated usable predictions, its threshold-profile idea was not actually tested; do not add further structural changes until a dependency-free version establishes whether the isolated switch-threshold modification helps. If that version is scorable, retain it only if it improves the Experiment 8 versus Experiments 11–13 dissociation while preserving the accepted fits in Experiments 2, 6, 7, 10, 14, and 17. Do not revisit the previously rejected occupancy normalization, broad terminal-gating package, or final-mixture confidence rewrite in this repair iteration.

**Outcome of this advice:** iter 5 candidate loss=1.0000 -> iter 6 candidate loss=0.3177 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The dependency repair succeeded, but the substantive profile-normalized threshold revision did not: aggregate loss is 0.3177, worse than the accepted iteration-1 loss of 0.3096, so the gate correctly rejected it. Predictions remain close to the accepted base and preserve good fits in Experiments 2, 6, 7, 10, 14, 16, and 17, but the central cross-experiment dissociation remains unresolved. Terminal control is far too weak in Experiments 3 (0.430 vs 0.575), 4 (0.269 vs 0.665), 11 (0.087 vs 0.843), and 13 (0.586 vs 0.881), while it is far too strong in Experiment 8 (0.599 vs 0.172) and Experiment 19 (0.717 vs 0.498). Experiment 12 likewise predicts excessive majority-coalition choice (0.660 vs 0.171). Experiment 1 remains much too categorical (0.876 vs 0.494), Experiment 5 retains a large spurious permutation effect (-0.374 vs 0.003), Experiment 9 retains excessive first-cue influence (0.319 vs -0.033), and Experiment 20 still barely expresses the required negative gradient (-0.017 vs -0.181). The reliability-profile change therefore neither improves the Experiment 8 versus 11–13 dissociation nor the overall fit. It also continues to underproduce between-subject variance in Experiments 1, 10, and 13. The theory remains faithful to the prescribed coalition-and-switch family, but this instantiation is not experiment-invariant.
Rationale: Do not repeat the current average-rank/profile implementation: unlike the previous unscorable iteration, it has now been tested and rejected. Return to the unchanged accepted iteration-1 base and make one corrected, narrowly isolated threshold-boundary revision. There is a concrete mismatch between the stated mechanism and the code: a challenge is initiated at the boundary between coalitions, but `run["boundary_profile"]` is taken from the challenger's last cue. For a multi-cue challenger, the relevant incoming boundary cue is its first cue; the incumbent boundary is its last cue. Store separate entry and exit profiles and compute the local switch contrast from incumbent exit versus challenger entry. Give this true boundary contrast more weight than the run average rather than diluting it to 30% while using the wrong endpoint.

Also avoid raw within-task min–max normalization, which maps arbitrarily tiny validity spans to the full 0–1 range and is inconsistent with strongly underweighted validity use. Use a regularized gap scale or a bounded blend with absolute reliability so that near-tied validities remain near-tied. Keep attention, terminal salience, erosion, confidence, load effects, and subject ranges fixed during this diagnostic test; prior compound revisions to those components were rejected. Retain the change only if it lowers aggregate loss while preserving Experiments 2, 6, 7, 10, 14, and 17 and jointly moves Experiment 8 downward and Experiments 11–13 upward. If the corrected boundary statistic still cannot produce that dissociation, discard it rather than adding further rank terms. Variance tuning should wait until these large level errors improve.

**Outcome of this advice:** iter 6 candidate loss=0.3177 -> iter 7 candidate loss=0.3130 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed coalition-and-switch family, and the corrected boundary statistic produced meaningful gains over the accepted base on several terminal-reversal targets: Experiment 3 improved from 0.429 to 0.523, Experiment 4 from 0.397 to 0.598, Experiment 11 from 0.101 to 0.340, Experiment 12 from 0.688 to 0.246, Experiment 13 from 0.589 to 0.700, and Experiment 20 from -0.019 to -0.249. These changes support using the actual incumbent-exit/challenger-entry boundary. However, the edit was still rejected by the gate because aggregate loss remained above the accepted base, 0.3130 versus 0.3096. The boundary-surprise effect is too strong and too broadly applied: Experiment 2 falls from 0.145 to 0.120 versus 0.194, Experiment 6 from a near-match of 0.295 to 0.212 versus 0.307, Experiment 7 crosses to the wrong binary category, Experiment 8 rises further to 0.666 versus 0.172, and Experiment 5 worsens to -0.533 versus a null observed contrast. It also overshoots the negative gradient in Experiment 20. Major invariant failures remain in Experiment 1 (0.889 versus 0.494), Experiment 5, Experiment 8, Experiment 9 (0.175 versus -0.033), Experiment 11 (still 0.503 too low), Experiment 13, and Experiment 19 (0.721 versus 0.498). Variances remain too compressed in Experiments 1 and 10, while terminal-switch variability is excessive in Experiments 3, 11, 12, and 20.
Rationale: Because the candidate was rejected, build from the unchanged iteration-1 accepted base. The previous boundary recommendation was directionally useful but its present magnitude was not accepted; do not repeat the full-strength implementation. Test an intermediate, narrowly isolated version of the same in-family boundary correction. Retain separate incumbent-exit and challenger-entry reliability, but reduce the boundary-surprise weight from 0.75 to roughly 0.35–0.50 and/or increase its regularizing denominator above 0.35. Keep the run-average term, attention, terminal salience, erosion, confidence, load terms, and subject ranges fixed. This interpolation is justified by the pattern: the current edit has useful room to weaken in Experiments 3, 4, 12, and 20, while attenuation should recover Experiments 2 and 6, avoid the Experiment 7 threshold crossing, and reduce the additional Experiment 5 distortion. It should also preserve part of the gains in Experiments 11 and 13 rather than reverting completely to the accepted base. Do not add another broad terminal gate, occupancy normalization, profile/rank statistic, or final-mixture confidence rewrite; all were previously rejected. Select the boundary coefficient by joint loss rather than optimizing Experiment 20 alone, and require preservation of the correct Experiment 7 category. If no intermediate coefficient beats 0.3096, discard this boundary-surprise edit rather than elaborating it. Subject-variance tuning should remain deferred until a level-improving boundary setting is accepted.

**Outcome of this advice:** iter 7 candidate loss=0.3130 -> iter 8 candidate loss=0.2795 -> the gate marked it ACCEPTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted intermediate boundary-surprise revision is a genuine improvement: aggregate loss fell from 0.3096 to 0.2795, validating the prior advice to retain the incumbent-exit/challenger-entry discontinuity at a moderated strength. It preserves accurate results in Experiments 6, 7, 16, and 17 and improves the required negative gradient in Experiment 20. Nevertheless, the model is not yet experiment-invariant. The largest failures remain severe: Experiment 11 predicts only 0.145 terminal choice versus 0.843 observed; Experiment 5 predicts a large spurious mixed-minus-block contrast of -0.483 versus 0.003; Experiment 8 predicts excessive terminal conformity, 0.641 versus 0.172; and Experiment 1 remains far too categorical, 0.878 versus 0.494. Terminal influence is also too weak in Experiments 3, 4, and 13, while majority-coalition choice is too high in Experiment 12. Experiments 9 and 19 retain excessive early/recency-field control. Thus the accepted boundary term helps but remains embedded in a switch contest that treats contiguous long incumbents and physically delayed challenges too similarly. Variance is also too compressed in Experiments 1, 10, and 13, although the large level errors remain the priority.
Rationale: Build on the accepted candidate and preserve the moderated boundary-surprise term; that interpolation was accepted, so do not remove it or return to the rejected full-strength version. The next change should be a localized, in-family persistence gate rather than another global increase in terminal salience or boundary weight. The bounded coalition strength saturates quickly, discarding how extensively and recently an incumbent coalition occupied the display. Add a modest incumbent persistence reserve to the switch threshold that increases with incumbent run coverage/coherence but decays with the physical gap or elapsed display distance between the incumbent exit and challenger entry. Consequently, an immediately adjacent challenger after a long coherent run should face more resistance, whereas a physically delayed terminal challenge should encounter a decayed incumbent. This should suppress excessive terminal switching in contiguous displays such as Experiment 8 and reduce the block advantage driving Experiment 5, while preserving boundary-triggered switching for the widely separated terminal cues in Experiments 11 and 12. Keep this reserve weak for short or fragmented incumbents so as not to damage Experiment 4, and allow the accepted reliability discontinuity and singleton salience to overcome it in Experiment 13 and the weak-terminal conditions of Experiment 20. Implement this as a threshold modulation, not additive signed evidence, and initially keep attention, erosion, confidence, load, and subject ranges fixed. Do not merely increase the current boundary coefficient: the iter-7 full-strength version was rejected and worsened Experiments 2, 5–8, and 20. Retain the new gate only if it jointly lowers Experiments 5 and 8 without sacrificing the current fits in Experiments 6, 7, 16, and 17 or eliminating the negative Experiment 20 contrast. Subject-level heterogeneity can be retuned afterward, chiefly by widening validity-use/readout variation rather than switch-threshold variation, which is already producing excessive variance on several reversal tasks.

**Outcome of this advice:** iter 8 candidate loss=0.2795 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Surprise-Bound Coalition Relay with Distance-Decaying Incumbent Persistence. During left-to-right inspection, consecutive discriminating cues supporting the same option form a bounded perceptual coalition. Opposing coalitions produce graded switching contests governed by coalition strength, strongly underweighted instructed reliability, local boundary surprise, terminal accessibility, and an incumbent persistence reserve. Persistence reflects the recent coherent occupation of the display by the incumbent coalition, but it decays when nondiscriminating display distance separates the incumbent from the challenger. Thus, contiguous challenges to long coherent coalitions encounter resistance, whereas physically delayed terminal challenges can trigger reversal. Choice is read out from the resulting mixture of commitment trajectories, with confidence selectively reduced by unresolved conflict and overload rather than by unanimous evidence.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Surprise-Bound Coalition Relay expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    base_attention = float(parameters[\"base_attention\"])\n    validity_use = float(parameters[\"validity_use\"])\n    position_accessibility = float(parameters[\"position_accessibility\"])\n    coalition_capacity = float(parameters[\"coalition_capacity\"])\n    coherence_gain = float(parameters[\"coherence_gain\"])\n    switch_threshold = float(parameters[\"switch_threshold\"])\n    switch_precision = float(parameters[\"switch_precision\"])\n    relative_surprise = float(parameters[\"relative_surprise\"])\n    terminal_salience = float(parameters[\"terminal_salience\"])\n    persistence_reserve = float(parameters[\"persistence_reserve\"])\n    opposition_erosion = float(parameters[\"opposition_erosion\"])\n    response_gain = float(parameters[\"response_gain\"])\n    conflict_collapse = float(parameters[\"conflict_collapse\"])\n    load_noise = float(parameters[\"load_noise\"])\n    unanimous_decay = float(parameters[\"unanimous_decay\"])\n    lapse = float(parameters[\"lapse_rate\"])\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    diff = stim[0] - stim[1]\n    discriminating = np.flatnonzero(diff != 0.0)\n    m = int(discriminating.size)\n    if m == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Reliability affects attentional accessibility and switch thresholds,\n    # never the sign of an additive evidence trace.\n    reliability = np.clip((validities - 0.5) / 0.5, 0.0, 1.0)\n    base_logit = np.log(\n        np.clip(base_attention, 1e-5, 1.0 - 1e-5)\n        / np.clip(1.0 - base_attention, 1e-5, 1.0)\n    )\n\n    ordinal_directions = np.sign(diff[discriminating]).astype(float)\n    cue_access = []\n    for j in discriminating:\n        absolute_position = float(j) / float(max(1, n_features - 1))\n        attention = sigmoid(\n            base_logit\n            + validity_use * (float(reliability[j]) - 0.5)\n            + position_accessibility * (absolute_position - 0.5)\n        )\n        # A modest independent accessibility advantage for later display\n        # locations is retained after attentional gating.\n        access = attention * (\n            1.0 + position_accessibility * 0.45 * absolute_position\n        )\n        cue_access.append(float(access))\n    cue_access = np.asarray(cue_access, dtype=float)\n\n    # Form ordinal coalitions. Physical locations remain attached to each\n    # coalition, so discriminating ordinal position and display position are\n    # not conflated.\n    runs = []\n    start = 0\n    while start < m:\n        direction = float(ordinal_directions[start])\n        end = start + 1\n        while end < m and ordinal_directions[end] == direction:\n            end += 1\n\n        loc = slice(start, end)\n        run_indices = discriminating[loc]\n        raw_access = float(np.sum(cue_access[loc]))\n        run_length = int(end - start)\n\n        # Bounded within-coalition integration. Coherence increases strength\n        # sublinearly and cannot create an unbounded linear sum.\n        bounded = coalition_capacity * (\n            1.0 - np.exp(-raw_access / max(coalition_capacity, 1e-8))\n        )\n        coherence = 1.0 + coherence_gain * (\n            1.0 - 1.0 / float(run_length)\n        )\n        strength = min(coalition_capacity, bounded * coherence)\n\n        weights = cue_access[loc]\n        if float(np.sum(weights)) > 0.0:\n            mean_reliability = float(\n                np.average(reliability[run_indices], weights=weights)\n            )\n        else:\n            mean_reliability = float(np.mean(reliability[run_indices]))\n\n        entry_physical_position = float(run_indices[0]) / float(\n            max(1, n_features - 1)\n        )\n        final_physical_position = float(run_indices[-1]) / float(\n            max(1, n_features - 1)\n        )\n        runs.append({\n            \"direction\": direction,\n            \"strength\": float(strength),\n            \"length\": run_length,\n            \"reliability\": mean_reliability,\n            \"entry_reliability\": float(reliability[run_indices[0]]),\n            \"exit_reliability\": float(reliability[run_indices[-1]]),\n            \"entry_position\": entry_physical_position,\n            \"final_position\": final_physical_position,\n            \"ends_sequence\": bool(end == m)\n        })\n        start = end\n\n    # Each entry is a possible commitment trajectory. Branching implements a\n    # genuinely graded switch rather than deterministic overwrite.\n    states = [{\n        \"weight\": 1.0,\n        \"direction\": runs[0][\"direction\"],\n        \"strength\": runs[0][\"strength\"],\n        \"reliability\": runs[0][\"reliability\"],\n        \"exit_reliability\": runs[0][\"exit_reliability\"],\n        \"run_length\": runs[0][\"length\"],\n        \"exit_position\": runs[0][\"final_position\"]\n    }]\n    switch_ambiguities = []\n\n    for run in runs[1:]:\n        next_states = []\n        weighted_ambiguity = 0.0\n        total_state_weight = 0.0\n\n        for incumbent in states:\n            w = float(incumbent[\"weight\"])\n            incumbent_strength = float(incumbent[\"strength\"])\n            challenger_strength = float(run[\"strength\"])\n\n            reliability_difference = (\n                float(incumbent[\"reliability\"])\n                - float(run[\"reliability\"])\n            )\n            threshold_multiplier = 1.0 + validity_use * reliability_difference\n            threshold_multiplier = float(np.clip(\n                threshold_multiplier, 0.55, 1.55\n            ))\n\n            relative_advantage = (\n                challenger_strength - incumbent_strength\n            ) / max(challenger_strength + incumbent_strength, 1e-8)\n\n            is_terminal_singleton = float(\n                run[\"ends_sequence\"] and run[\"length\"] == 1\n            )\n            singleton_access = (\n                is_terminal_singleton\n                * terminal_salience\n                * (0.55 + 0.45 * float(run[\"final_position\"]))\n            )\n\n            # Switching is initiated at the actual coalition boundary. A\n            # downward reliability discontinuity can be surprising, but its\n            # influence is bounded and strongly regularized. The coefficient\n            # is deliberately intermediate rather than the rejected 0.75\n            # boundary weight.\n            boundary_drop = max(\n                0.0,\n                float(incumbent[\"exit_reliability\"])\n                - float(run[\"entry_reliability\"])\n            )\n            boundary_surprise = (\n                0.42 * boundary_drop / (0.50 + boundary_drop)\n            )\n\n            # A coherent incumbent retains a modest persistence reserve when\n            # the challenger is physically adjacent. The reserve scales with\n            # incumbent coverage but decays rapidly across nondiscriminating\n            # display distance, keeping delayed terminal reversals accessible.\n            one_column = 1.0 / float(max(1, n_features - 1))\n            physical_gap = max(\n                0.0,\n                float(run[\"entry_position\"])\n                - float(incumbent[\"exit_position\"])\n                - one_column\n            )\n            incumbent_coverage = float(incumbent[\"run_length\"]) / float(m)\n            incumbent_coherence = max(\n                0.0,\n                1.0 - 1.0 / float(incumbent[\"run_length\"])\n            )\n            persistence_cost = (\n                persistence_reserve\n                * incumbent_strength\n                * incumbent_coverage\n                * incumbent_coherence\n                * np.exp(-6.0 * physical_gap)\n            )\n\n            contest = (\n                challenger_strength\n                - switch_threshold\n                * threshold_multiplier\n                * incumbent_strength\n                - persistence_cost\n                + relative_surprise * relative_advantage\n                + singleton_access\n                + boundary_surprise\n            )\n            q_switch = sigmoid(switch_precision * contest)\n\n            weighted_ambiguity += w * 4.0 * q_switch * (1.0 - q_switch)\n            total_state_weight += w\n\n            # If the challenger is resisted, conflict still erodes incumbent\n            # commitment. If it wins, its bounded coalition becomes incumbent.\n            retained_strength = max(\n                0.08,\n                incumbent_strength\n                - opposition_erosion * challenger_strength\n            )\n            switched_strength = min(\n                coalition_capacity,\n                challenger_strength\n                + 0.10 * q_switch * incumbent_strength\n            )\n\n            if q_switch < 1.0:\n                next_states.append({\n                    \"weight\": w * (1.0 - q_switch),\n                    \"direction\": float(incumbent[\"direction\"]),\n                    \"strength\": float(retained_strength),\n                    \"reliability\": float(incumbent[\"reliability\"]),\n                    \"exit_reliability\": float(incumbent[\"exit_reliability\"]),\n                    \"run_length\": int(incumbent[\"run_length\"]),\n                    \"exit_position\": float(incumbent[\"exit_position\"])\n                })\n            if q_switch > 0.0:\n                next_states.append({\n                    \"weight\": w * q_switch,\n                    \"direction\": float(run[\"direction\"]),\n                    \"strength\": float(switched_strength),\n                    \"reliability\": float(run[\"reliability\"]),\n                    \"exit_reliability\": float(run[\"exit_reliability\"]),\n                    \"run_length\": int(run[\"length\"]),\n                    \"exit_position\": float(run[\"final_position\"])\n                })\n\n        states = next_states\n        if total_state_weight > 0.0:\n            switch_ambiguities.append(\n                weighted_ambiguity / total_state_weight\n            )\n\n    # Accessible support is used only to diagnose unresolved conflict. It is\n    # not read out as a linear evidence total.\n    positive_mass = float(np.sum(\n        cue_access[ordinal_directions > 0.0]\n    ))\n    negative_mass = float(np.sum(\n        cue_access[ordinal_directions < 0.0]\n    ))\n    total_mass = positive_mass + negative_mass\n    if positive_mass <= 0.0 or negative_mass <= 0.0 or total_mass <= 0.0:\n        balance = 0.0\n    else:\n        balance = 2.0 * min(positive_mass, negative_mass) / total_mass\n        balance = float(np.clip(balance, 0.0, 1.0))\n\n    ambiguity = (\n        float(np.mean(switch_ambiguities))\n        if switch_ambiguities else 0.0\n    )\n    fragmentation = max(0.0, float(len(runs) - 2)) / float(max(1, m - 1))\n    unresolved = balance * (\n        0.20 + 0.55 * ambiguity + 0.45 * fragmentation\n    )\n    unresolved = float(np.clip(unresolved, 0.0, 1.0))\n\n    confidence = max(0.08, 1.0 - conflict_collapse * unresolved)\n    overload = max(0.0, float(m) - 4.0)\n    conflict_divisor = 1.0 + load_noise * overload * unresolved * unresolved\n\n    # Unanimous evidence avoids conflict noise but shows a slight empirical\n    # decline at high loads rather than ever-increasing certainty.\n    if balance == 0.0:\n        unanimous_divisor = 1.0 + unanimous_decay * max(0.0, float(m) - 2.0)\n    else:\n        unanimous_divisor = 1.0\n\n    trajectory_weight = float(np.sum([s[\"weight\"] for s in states]))\n    if trajectory_weight <= 0.0 or not np.isfinite(trajectory_weight):\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    p_a_core = 0.0\n    for commitment in states:\n        decision_variable = (\n            response_gain\n            * confidence\n            * float(commitment[\"direction\"])\n            * float(commitment[\"strength\"])\n            / (conflict_divisor * unanimous_divisor)\n        )\n        p_a_state = sigmoid(decision_variable)\n        p_a_core += float(commitment[\"weight\"]) * p_a_state\n    p_a_core /= trajectory_weight\n    p_a_core = float(np.clip(p_a_core, 0.0, 1.0))\n\n    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)\n    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(np.sum(probs))\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "base_attention": "[0.82, 0.95]",
    "validity_use": "[0.22, 0.52]",
    "position_accessibility": "[0.28, 0.58]",
    "coalition_capacity": "[1.75, 2.45]",
    "coherence_gain": "[0.10, 0.30]",
    "switch_threshold": "[0.62, 0.88]",
    "switch_precision": "[3.8, 6.2]",
    "relative_surprise": "[0.45, 0.90]",
    "terminal_salience": "[0.42, 0.78]",
    "persistence_reserve": "[0.10, 0.24]",
    "opposition_erosion": "[0.20, 0.42]",
    "response_gain": "[2.5, 4.4]",
    "conflict_collapse": "[0.48, 0.76]",
    "load_noise": "[0.025, 0.085]",
    "unanimous_decay": "[0.010, 0.032]",
    "lapse_rate": "[0.04, 0.14]"
  },
  "rationale": "This is a minimal extension of the accepted iteration-8 candidate. The moderated incumbent-exit/challenger-entry boundary-surprise term and every previously accepted attention, erosion, confidence, load, and readout equation are retained unchanged. The only substantive addition is a weak persistence cost in the switch contest, plus the run-length and physical-position metadata needed to compute it. Persistence increases with incumbent strength, ordinal coverage, and within-run coherence, but decays exponentially across empty physical columns. It should therefore resist terminal capture in contiguous long-prefix displays such as Experiment 8 and lower the block-specific terminal advantage responsible for Experiment 5, while remaining negligible for the widely separated terminal challenges in Experiments 11 and 12. Singleton incumbents receive no reserve, short incumbents receive little reserve, and the accepted boundary surprise and terminal salience can still overcome it in Experiments 4, 13, and 20. The new subject-level range is deliberately modest to avoid recreating the rejected full-strength threshold manipulations or damaging Experiments 6, 7, 16, and 17."
}
```

## Usage

```json
{
  "input_tokens": 46728,
  "output_tokens": 4801
}
```
