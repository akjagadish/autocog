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
- THEORY 2 = `pi_12`
- The recommendation below acts on THEORY 2 (= `pi_12`).

Replace pi_12 with a Spatially Indexed Reliability-Gated Revision theory. Subjects should normally inspect cues in physical left-to-right order, while instructed validity controls cue accessibility and revision strength rather than imposing a separate validity-sorted scan. Each attended discriminating cue updates a bounded directional commitment through leaky, recency-weighted revision. Repeated same-direction cues increase commitment with saturation, but they should not form the rigid coalitions responsible for pi_12's failures in Experiments 7, 10, 13, and 14. A local drop from a reliable incumbent cue to a weak opposing cue should trigger an orienting or surprise response that transiently increases attention to the challenger, accounting for Experiment 2. Absolute right-edge position and terminal isolation should also increase accessibility, accounting for Experiment 1, but these bonuses must be conditional and modest rather than guaranteeing a coalition switch. Commitment stability should depend jointly on recent direction, accumulated coherent support, cue validity, and load: recent cues can dominate short or manageable conflicts, as in Experiments 6, 12, 13, and 15, while strong coherent prefixes can resist a terminal singleton in other configurations, as required by Experiments 10 and 14. Additional reasons should introduce gradual precision loss rather than pi_3's fixed post-two-cue fatigue or pi_12's conflict-specific collapse; unanimous sequences should show only the small decline observed in Experiment 17. Use a single continuous state-update mechanism, not a trial-level mixture of unrelated strategies. Include stable subject-level parameters for spatial attention, validity gain, leak/overwrite strength, surprise orienting, commitment saturation, load noise, and lapse, with regularized ranges shared across experiments. The regenerated theory should be calibrated jointly against all continuous metrics, especially the positive effects in Experiments 1 and 2 and the near-zero contrasts in Experiments 7 and 16, while treating thresholded Experiments 9 and 19 as weak constraints.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_3` (overall score: 0.581)

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


---

### `pi_8` (overall score: 0.540)

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


---

### `pi_5` (overall score: 0.514)

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
- Experiment 9: real=-0.0325 (var=0.0099) vs this=0.0108 (var=0.0053)
- Experiment 10: real=0.8748 (var=0.0069) vs this=0.9256 (var=0.0034)
- Experiment 11: real=0.8425 (var=0.0123) vs this=0.6737 (var=0.0053)
- Experiment 12: real=0.1713 (var=0.0095) vs this=0.2125 (var=0.0050)
- Experiment 13: real=0.8810 (var=0.0070) vs this=0.6957 (var=0.0044)
- Experiment 14: real=0.0192 (var=0.0062) vs this=0.0054 (var=0.0094)
- Experiment 15: real=0.0471 (var=0.0135) vs this=0.1579 (var=0.0074)
- Experiment 16: real=0.0311 (var=0.0170) vs this=0.0102 (var=0.0090)
- Experiment 17: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0773)
- Experiment 18: real=0.0262 (var=0.0075) vs this=-0.0750 (var=0.0168)
- Experiment 19: real=0.4978 (var=0.0077) vs this=0.5828 (var=0.0078)
- Experiment 20: real=-0.1813 (var=0.0102) vs this=-0.0019 (var=0.0095)
- Experiment 21: real=0.3900 (var=0.0214) vs this=0.5100 (var=0.0281)
- Experiment 22: real=0.1750 (var=0.0208) vs this=-0.0050 (var=0.0066)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1951 -> ACCEPTED
- iter 2: loss=0.2022 -> REJECTED
- iter 3: loss=0.2160 -> REJECTED
- iter 4: loss=0.2120 -> REJECTED
- iter 5: loss=0.1880 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 5 at loss=0.1880 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.8106 (var=0.0045) (Δ vs real +0.3161)
  - iter 2: 0.7037 (var=0.0071) (Δ vs real +0.2093)
  - iter 3: 0.7974 (var=0.0045) (Δ vs real +0.3029)
  - iter 4: 0.8154 (var=0.0044) (Δ vs real +0.3210)
  - iter 5 (most recent): 0.8089 (var=0.0051) (Δ vs real +0.3144)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1508 (var=0.0020) (Δ vs real -0.0429)
  - iter 2: 0.2990 (var=0.0078) (Δ vs real +0.1052)
  - iter 3: 0.1285 (var=0.0017) (Δ vs real -0.0652)
  - iter 4: 0.1581 (var=0.0018) (Δ vs real -0.0356)
  - iter 5 (most recent): 0.1383 (var=0.0026) (Δ vs real -0.0554)
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
**Previous candidate values (this loop):**
  - iter 1: 0.7490 (var=0.0049) (Δ vs real +0.1740)
  - iter 2: 0.4958 (var=0.0104) (Δ vs real -0.0792)
  - iter 3: 0.7194 (var=0.0065) (Δ vs real +0.1444)
  - iter 4: 0.7202 (var=0.0071) (Δ vs real +0.1452)
  - iter 5 (most recent): 0.7638 (var=0.0064) (Δ vs real +0.1888)
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
**Previous candidate values (this loop):**
  - iter 1: 0.7121 (var=0.0066) (Δ vs real +0.0471)
  - iter 2: 0.3558 (var=0.0186) (Δ vs real -0.3092)
  - iter 3: 0.7250 (var=0.0111) (Δ vs real +0.0600)
  - iter 4: 0.6533 (var=0.0134) (Δ vs real -0.0117)
  - iter 5 (most recent): 0.7542 (var=0.0066) (Δ vs real +0.0892)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0033 (var=0.0044) (Δ vs real +0.0000)
  - iter 2: -0.0796 (var=0.0086) (Δ vs real -0.0829)
  - iter 3: -0.0063 (var=0.0047) (Δ vs real -0.0096)
  - iter 4: -0.0104 (var=0.0055) (Δ vs real -0.0137)
  - iter 5 (most recent): 0.0004 (var=0.0043) (Δ vs real -0.0029)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1054 (var=0.0016) (Δ vs real -0.2012)
  - iter 2: 0.1588 (var=0.0029) (Δ vs real -0.1479)
  - iter 3: 0.1158 (var=0.0018) (Δ vs real -0.1908)
  - iter 4: 0.1037 (var=0.0025) (Δ vs real -0.2029)
  - iter 5 (most recent): 0.1104 (var=0.0015) (Δ vs real -0.1962)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.1056) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.2100) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.1476) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.1824) (Δ vs real +0.0000)
  - iter 5 (most recent): 0.0000 (var=0.1924) (Δ vs real +0.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8633 (var=0.0018) (Δ vs real +0.6912)
  - iter 2: 0.7444 (var=0.0034) (Δ vs real +0.5723)
  - iter 3: 0.8450 (var=0.0019) (Δ vs real +0.6729)
  - iter 4: 0.8408 (var=0.0023) (Δ vs real +0.6687)
  - iter 5 (most recent): 0.8454 (var=0.0025) (Δ vs real +0.6733)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0892 (var=0.0075) (Δ vs real +0.1217)
  - iter 2: 0.3525 (var=0.0263) (Δ vs real +0.3850)
  - iter 3: 0.0862 (var=0.0095) (Δ vs real +0.1187)
  - iter 4: 0.1483 (var=0.0107) (Δ vs real +0.1808)
  - iter 5 (most recent): 0.0867 (var=0.0071) (Δ vs real +0.1192)
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
**Previous candidate values (this loop):**
  - iter 1: 0.9164 (var=0.0008) (Δ vs real +0.0417)
  - iter 2: 0.8843 (var=0.0014) (Δ vs real +0.0095)
  - iter 3: 0.9045 (var=0.0015) (Δ vs real +0.0298)
  - iter 4: 0.9093 (var=0.0010) (Δ vs real +0.0345)
  - iter 5 (most recent): 0.9070 (var=0.0015) (Δ vs real +0.0322)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8308 (var=0.0059) (Δ vs real -0.0117)
  - iter 2: 0.5529 (var=0.0144) (Δ vs real -0.2896)
  - iter 3: 0.7779 (var=0.0067) (Δ vs real -0.0646)
  - iter 4: 0.7746 (var=0.0074) (Δ vs real -0.0679)
  - iter 5 (most recent): 0.7858 (var=0.0086) (Δ vs real -0.0567)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1419 (var=0.0047) (Δ vs real -0.0294)
  - iter 2: 0.1919 (var=0.0070) (Δ vs real +0.0206)
  - iter 3: 0.1288 (var=0.0032) (Δ vs real -0.0425)
  - iter 4: 0.1419 (var=0.0045) (Δ vs real -0.0294)
  - iter 5 (most recent): 0.1556 (var=0.0037) (Δ vs real -0.0156)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8396 (var=0.0025) (Δ vs real -0.0414)
  - iter 2: 0.7048 (var=0.0032) (Δ vs real -0.1761)
  - iter 3: 0.8489 (var=0.0024) (Δ vs real -0.0321)
  - iter 4: 0.8082 (var=0.0030) (Δ vs real -0.0727)
  - iter 5 (most recent): 0.8218 (var=0.0025) (Δ vs real -0.0592)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0796 (var=0.0108) (Δ vs real +0.0604)
  - iter 2: 0.0458 (var=0.0081) (Δ vs real +0.0267)
  - iter 3: 0.0871 (var=0.0094) (Δ vs real +0.0679)
  - iter 4: 0.0908 (var=0.0093) (Δ vs real +0.0717)
  - iter 5 (most recent): 0.0533 (var=0.0097) (Δ vs real +0.0342)
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
**Previous candidate values (this loop):**
  - iter 1: -0.0052 (var=0.0051) (Δ vs real -0.0523)
  - iter 2: 0.0292 (var=0.0089) (Δ vs real -0.0179)
  - iter 3: 0.0158 (var=0.0045) (Δ vs real -0.0312)
  - iter 4: -0.0054 (var=0.0033) (Δ vs real -0.0525)
  - iter 5 (most recent): -0.0031 (var=0.0044) (Δ vs real -0.0502)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0155 (var=0.0112) (Δ vs real -0.0156)
  - iter 2: -0.0113 (var=0.0118) (Δ vs real -0.0424)
  - iter 3: -0.0177 (var=0.0057) (Δ vs real -0.0488)
  - iter 4: 0.0153 (var=0.0119) (Δ vs real -0.0158)
  - iter 5 (most recent): 0.0026 (var=0.0137) (Δ vs real -0.0285)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0006) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0067) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0016) (Δ vs real +0.0000)
  - iter 4: 0.0000 (var=0.0045) (Δ vs real +0.0000)
  - iter 5 (most recent): 0.0000 (var=0.0022) (Δ vs real +0.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0344 (var=0.0063) (Δ vs real +0.0081)
  - iter 2: 0.0137 (var=0.0170) (Δ vs real -0.0125)
  - iter 3: -0.0088 (var=0.0136) (Δ vs real -0.0350)
  - iter 4: 0.0150 (var=0.0129) (Δ vs real -0.0113)
  - iter 5 (most recent): 0.0350 (var=0.0120) (Δ vs real +0.0087)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8972 (var=0.0023) (Δ vs real +0.3994)
  - iter 2: 0.7944 (var=0.0055) (Δ vs real +0.2967)
  - iter 3: 0.8956 (var=0.0028) (Δ vs real +0.3978)
  - iter 4: 0.9000 (var=0.0027) (Δ vs real +0.4022)
  - iter 5 (most recent): 0.8761 (var=0.0031) (Δ vs real +0.3783)
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
**Previous candidate values (this loop):**
  - iter 1: -0.1062 (var=0.0087) (Δ vs real +0.0750)
  - iter 2: -0.1188 (var=0.0211) (Δ vs real +0.0625)
  - iter 3: -0.1106 (var=0.0157) (Δ vs real +0.0706)
  - iter 4: -0.1050 (var=0.0147) (Δ vs real +0.0763)
  - iter 5 (most recent): -0.1156 (var=0.0141) (Δ vs real +0.0656)
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
**Previous candidate values (this loop):**
  - iter 1: 0.6633 (var=0.0134) (Δ vs real +0.2733)
  - iter 2: 0.3242 (var=0.0320) (Δ vs real -0.0658)
  - iter 3: 0.6908 (var=0.0119) (Δ vs real +0.3008)
  - iter 4: 0.6458 (var=0.0186) (Δ vs real +0.2558)
  - iter 5 (most recent): 0.6367 (var=0.0113) (Δ vs real +0.2467)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1512 (var=0.0075) (Δ vs real -0.0238)
  - iter 2: 0.1542 (var=0.0127) (Δ vs real -0.0208)
  - iter 3: 0.1062 (var=0.0073) (Δ vs real -0.0688)
  - iter 4: 0.0975 (var=0.0054) (Δ vs real -0.0775)
  - iter 5 (most recent): 0.1525 (var=0.0081) (Δ vs real -0.0225)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Spatially Indexed Reliability-Gated Revision. People inspect product features in their displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment. Revision is leaky, so recent cues ordinarily have greater influence, while repeated same-direction cues build a saturating, reliability-sensitive stability trace rather than an all-or-none coalition. This trace remains associated with an established incumbent during a reversal and is eroded gradually by cumulative challenger evidence; consequently, every cue in an opposing run remains partially gated until the challenger has continuously displaced the incumbent. Instructed validity jointly affects cue accessibility, revision magnitude, and the rate at which evidence builds or erodes stability. A weak cue that locally opposes a more reliable preceding cue evokes surprise-driven orienting, temporarily increasing its accessibility and capacity to revise the incumbent commitment. Accessibility also rises gradually toward the right side of the display, although the positional gradient is compressed on very long displays, with a small additional benefit for an isolated discriminating cue at the physical right edge. Increasing display load gradually reduces response precision, including a small decline for long unanimous sequences. Choices are generated from this single evolving commitment state, with no trial-level mixture of strategies.

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
        # opposes the persistent stability trace, rather than only while it
        # opposes the immediately preceding cue or current run label.
        retained = float(np.clip(leak_rate * commitment, -0.999999, 0.999999))
        opposing_stability = bool(
            stability_trace != 0.0 and np.sign(stability_trace) != direction
        )
        if opposing_stability:
            stability_gate = 1.0 + (
                commitment_saturation
                * abs(stability_trace)
                * abs(retained)
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
        trace_retention = float(np.sqrt(leak_rate))
        stability_trace = (
            trace_retention * stability_trace
            + direction
            * accessibility
            * reliability_strength
            * assimilation
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
- load_noise: [0.012, 0.042]
- lapse_rate: [0.04, 0.13]

`rationale`: This is a minimal recurrence-level edit to the accepted candidate. The former nonnegative coherent-support variable was replaced by a signed persistent stability trace. Previously, the first reversal immediately changed the remembered direction, allowing every subsequent challenger cue to bypass the stability gate; now all challenger cues remain gated until their cumulative reliability-weighted evidence has gradually eroded the incumbent trace. Weak incumbents can still be revised rapidly, preserving short-display terminal effects, while established prefixes should better resist contiguous late runs in Experiments 1, 3, 6, and 8. The accepted relative-reliability transform, overwrite, leak, surprise mechanism, load noise, and parameter ranges are otherwise unchanged. The only additional edit compresses positional accessibility and the isolation bonus as display length grows, targeting the excessive endpoint effects in Experiments 14 and 21 without uniformly reducing spatial attention or removing the modest right-edge advantage.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed continuous spatial revision family and successfully captures several important signatures: the null contrasts in Experiments 5, 7, 16, and 17; terminal influence in Experiments 10, 11, and 13; the recency-majority result in Experiment 12; and the reliability-drop effect in Experiment 22. However, it is not yet experiment-invariant. Its dominant error is excessive, overly general terminal overwrite. Terminal-cue agreement is far too high in Experiment 8 (0.863 versus 0.172), Experiment 19 (0.897 versus 0.498), and Experiment 3 (0.749 versus 0.575), while early-run choice is consequently too low in Experiment 6 (0.105 versus 0.307). Experiment 1's tally-support contrast is also inflated (0.811 versus 0.494). Separately, spatial foregrounding is too strong: Experiment 21's endpoint effect is 0.663 versus 0.390, and Experiment 14's foregrounding contrast is 0.080 versus 0.019. The model also lacks the small unanimous-load decline in Experiment 15 (-0.005 versus 0.047). Many simulated between-subject variances are substantially below the observed variances, suggesting overly narrow effective heterogeneity.
Rationale: Retain the Spatially Indexed Reliability-Gated Revision mechanism, but recalibrate three in-family components. First, reduce generic overwrite by moderately increasing commitment retention and/or strengthening continuous coherent-support gating, particularly under long displays and established same-direction prefixes. The adjustment must be graded rather than categorical so that the strong terminal effects in Experiments 10, 11, and 13 remain. Most importantly, the surprise multiplier is currently applied to every opposing reliability drop, including adjacent block boundaries; make surprise orienting conditional on challenger isolation or a meaningful spatial gap. That should sharply reduce the spurious terminal dominance in Experiment 8 while preserving the separated-singleton effects in Experiments 20 and 22. Second, lower the spatial-attention range and especially the right-edge isolation coefficient to bring Experiments 21 and 14 down without eliminating their positive direction. Third, strengthen gradual load-dependent precision loss enough to yield Experiment 15's small positive decline, avoiding any fixed fatigue threshold. Jointly tune these knobs rather than merely raising lapse, since a global lapse would damage the already good directional contrasts. After correcting mean fit, modestly widen subject-level parameter ranges to better reproduce observed heterogeneity.

**Outcome of this advice:** iter 1 candidate loss=0.1951 -> iter 2 candidate loss=0.2022 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family, but it is worse than the accepted base and therefore should not be retained. The previous recommendation to broadly reduce overwrite and spatial attention was over-applied. It improved Experiment 1, the foregrounding effects in Experiments 14 and 21, the load decline in Experiment 15, and somewhat reduced the excessive terminal choice in Experiments 8 and 19. However, it simultaneously destroyed several previously good terminal-revision results: Experiment 4 fell from 0.712 to 0.356 versus 0.665 observed, Experiment 11 from 0.831 to 0.553 versus 0.843, and Experiment 13 from 0.840 to 0.705 versus 0.881. Experiment 2 also crossed from a small underestimate to a substantial overestimate, and Experiment 9 deteriorated sharply. The central failure nevertheless remains Experiment 8, where terminal agreement is still 0.744 rather than 0.172; Experiment 19 also remains too terminal-driven at 0.794 rather than 0.498, while early-run choice in Experiment 6 remains too low. Thus the model still cannot selectively protect an incumbent in the configurations that require resistance without suppressing revision in Experiments 4, 10, 11, and 13. A likely source is the within-experiment z-standardization of reliability: it discards absolute reliability level and profile information, making superficially similar spatial sequences across experiments receive similar update strengths even when their communicated validities differ materially.
Rationale: Because this iteration was rejected by the gate, build from the unchanged accepted iteration-1 base rather than repeating the rejected broad increase in retention, decrease in overwrite, and widening of heterogeneity. Use a different in-family knob: revise reliability calibration. Replace per-experiment z-scoring of log-odds reliability with a fixed, cross-experiment normalization or a compressive transform anchored at validity 0.5. Reliability should retain absolute magnitude and the absolute drop between incumbent and challenger, while remaining bounded. Then let the existing continuous coherence gate depend on accumulated reliability-weighted support, not merely accessibility summed across cues. This should permit strong reliable prefixes to resist weak terminal challengers in Experiment 8 and reduce the inflated terminal conformity in Experiments 6 and 19, without globally weakening terminal revision in Experiments 4, 11, and 13. Keep surprise orienting continuous and separation-sensitive, but do not remove adjacent revision wholesale; Experiment 2 and the separated-drop effect in Experiment 22 require reliability-drop sensitivity. Preserve the accepted base's generic overwrite, leak, spatial attention, and load settings initially, and tune the new fixed reliability scale jointly against Experiments 2, 4, 6, 8, 11, 13, 20, and 22. Treat Experiments 9 and 19 as weaker constraints as instructed. Do not further widen parameter ranges until the pooled means improve, since the rejected iteration's increased heterogeneity did not compensate for its mean-fit regressions.

**Outcome of this advice:** iter 2 candidate loss=0.2022 -> iter 3 candidate loss=0.2160 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed single-process Spatially Indexed Reliability-Gated Revision family, but its aggregate loss is worse than both the accepted base and the previous rejected attempt. The fixed absolute-reliability calibration did not solve the key selectivity problem. Relative to the accepted base, Experiment 8 barely improved (terminal agreement 0.845 versus 0.863, still far above 0.172), Experiment 19 was effectively unchanged (0.896 versus 0.897, observed 0.498), and Experiment 6 remained much too recency-driven (early choice 0.116 versus 0.307). It also worsened Experiment 2, Experiment 12, Experiment 16, Experiment 18, and the reliability-drop effect in Experiment 22. The persistent signature is excessive revision by contiguous late challengers: this inflates terminal influence in Experiments 3, 8, and 19, suppresses early-run choice in Experiment 6, and contributes to the excessive tally contrast in Experiment 1. Spatial sensitivity is also still too large in Experiments 14 and 21. At the same time, generic terminal revision is useful and should be preserved: Experiments 4, 10, 11, and 13 remain reasonably close, and the model correctly captures the null/near-null patterns in Experiments 5, 7, and 17. Simulated variances are often below observed variances, but mean-pattern errors remain the priority.
Rationale: The current candidate was rejected, and the prior fixed-reliability recommendation was also rejected, so do not continue tuning the absolute reliability anchor or reliability-weighted coherent-support accumulation. Make a narrower in-family intervention on the unchanged accepted base. Specifically, change the surprise term rather than globally changing leak or overwrite: a reliability drop should receive substantial orienting amplification only when the opposing challenger is spatially separated from the incumbent run, isolated at the right edge, or otherwise has a meaningful gap from the preceding discriminating cue. Adjacent opposing cues should retain ordinary continuous revision but receive little or no surprise multiplier. This directly distinguishes the contiguous reversals driving excessive terminal choice in Experiments 6, 8, and 19 from the separated challengers central to Experiments 2, 11, 20, and 22, while leaving generic recency available for the adjacent terminal effects in Experiments 4, 10, and 13. Implement the modulation smoothly, for example by multiplying surprise by a bounded function of physical gap plus a modest right-edge-isolation term, rather than imposing a categorical strategy or threshold. Separately, test a small attenuation or saturation of the smooth position-to-accessibility gradient while keeping leak and overwrite at the accepted settings; this targets the oversized endpoint and foregrounding effects in Experiments 21 and 14 without repeating the rejected broad suppression of revision. Jointly calibrate these two surgical changes against Experiments 1, 2, 4, 6, 8, 11, 13, 14, 21, and 22. Defer widening subject-level ranges until pooled means improve.

**Outcome of this advice:** iter 3 candidate loss=0.2160 -> iter 4 candidate loss=0.2120 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed Spatially Indexed Reliability-Gated Revision family, but it was rejected by the accept gate and remains substantially worse than the accepted iteration-1 base. The latest separation-gated surprise intervention did not address the dominant errors: terminal agreement remains far too high in Experiment 8 (0.841 versus 0.172) and Experiment 19 (0.900 versus 0.498), early-run choice remains far too low in Experiment 6 (0.104 versus 0.307), and the tally-support contrast remains inflated in Experiment 1 (0.815 versus 0.494). Endpoint sensitivity also remains excessive in Experiment 21 (0.646 versus 0.390), while Experiment 14's foregrounding contrast moved farther from zero. Moreover, attenuating separated surprise harmed results that the accepted base handled better: Experiment 11 fell from 0.831 to 0.775, Experiment 13 from 0.840 to 0.808, and Experiment 22 from 0.151 to 0.098. The candidate still succeeds on Experiments 4, 5, 7, 10, 12, 16, 17, and 18, but that is insufficiently experiment-invariant. The central implementation problem is now clearer: surprise is not the main source of excessive recency. After the first opposing cue, coherent_support is rapidly reassigned to the challenger, and subsequent same-direction challenger cues are classified as non-opposing and therefore bypass the stability gate. A contiguous late run can consequently erase a coherent incumbent through ordinary updates even when surprise is suppressed. Simulated between-subject variances also remain generally too small, but the large pooled-mean errors should be fixed first.
Rationale: Build from the unchanged accepted iteration-1 base. The previous pushes toward broad retention/overwrite reduction, fixed absolute-reliability normalization, and spatially gated surprise were all rejected, so do not repeat them. Instead, make a distinct in-family correction to the recurrence: maintain a persistent, continuous incumbent-stability trace that is not immediately relabeled or reset after the first reversal. Opposing evidence should erode this trace gradually according to cumulative challenger strength, reliability, leak, and load; every cue in an opposing run should remain partially gated until the challenger has continuously displaced the incumbent. This can be implemented with two smooth state variables—commitment and a decaying commitment-congruent support or stability trace—without introducing runs, thresholds, coalitions, or strategy mixtures. Preserve ordinary rapid revision when the incumbent is weak, thereby protecting Experiments 4, 10, 11, and 13, but let established coherent support resist multi-cue late runs enough to improve Experiments 6 and 8. Use the accepted relative reliability transform initially rather than reopening the rejected absolute-normalization change, and tune how reliability scales erosion of the persistent trace. Experiment 19 is a weak thresholded constraint, but its large error should still improve as a consequence rather than driving calibration. For Experiment 21, avoid another uniform reduction of spatial attention. Instead, compress the effective recency generated by scan position on very long displays—for example, make leak over successive discriminating cues mildly load-dependent or saturate positional accessibility with display length—while retaining a modest explicit right-edge accessibility bonus required by Experiment 1. Do not further weaken surprise: the latest change degraded Experiments 11, 13, and 22 without solving Experiment 8. After improving pooled means, subject-level parameter ranges can be widened modestly to address the systematically low variance.

**Outcome of this advice:** iter 4 candidate loss=0.2120 -> iter 5 candidate loss=0.1880 -> the gate marked it ACCEPTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted persistent-stability revision is a genuine improvement within the prescribed mechanism family: aggregate loss fell from 0.1951 to 0.1880, while the model retains excellent fits for the null contrasts in Experiments 5, 7, and 17, the near-null history result in Experiment 16, the boundary pattern in Experiment 18, the recency-majority result in Experiment 12, and the reliability-drop effect in Experiment 22. Experiment 14 and, more modestly, Experiment 21 also improved in the intended direction. Nevertheless, the model is not yet experiment-invariant. Its dominant empirical failure remains excessive terminal/late-cue control: Experiment 8 is 0.845 versus 0.172, Experiment 3 is 0.764 versus 0.575, and the weakly weighted Experiment 19 is 0.876 versus 0.498. Correspondingly, early-run choice in Experiment 6 is only 0.110 versus 0.307, and Experiment 1's tally-support contrast remains 0.809 versus 0.494. The new trace therefore improved aggregate loss without fixing the central recurrence error it was designed to address; Experiment 8 is essentially unchanged and Experiment 3 worsened. Spatial order effects also remain excessive in Experiment 21 (0.637 versus 0.390), although less so than before. The model additionally misses the small unanimous-load decline in Experiment 15 (-0.003 versus 0.047). Fits in Experiments 10, 12, 18, 20, and 22 show that reliability-gated revision itself is viable, but the simulated behavior is generally too extreme and too homogeneous: many between-subject variances are far below the observed values.
Rationale: Retain the accepted signed persistent-stability trace—the previous advice introducing it was accepted—but correct how it gates reversal. In the current code, the opposing gate is proportional to abs(retained). Precisely when challenger evidence drives commitment toward zero, this factor collapses and later challenger cues bypass the still-persistent incumbent trace. Thus the implementation does not fully realize the stated idea that every cue remains gated until cumulative displacement. Make the opposing gate depend smoothly on incumbent trace magnitude and accumulated challenger erosion, not on the instantaneous magnitude of commitment; allow it to weaken only as the signed trace itself is eroded. Calibrate this as a modest nonlinear or saturating change rather than globally raising leak or lowering overwrite, since those broad changes were previously rejected and harmed Experiments 4, 11, and 13. This targeted correction should protect coherent prefixes in Experiments 1, 6, and especially 8 while preserving rapid revision from weak incumbents in Experiments 4, 10, 11, and 13. Separately, extend the accepted display-length compression with a small long-display-specific reduction in effective recency—such as smoothly moving leak toward greater retention as total display length becomes unusually large—rather than another global reduction in spatial attention. This targets Experiments 14 and 21 without altering ordinary six- to ten-feature displays. Finally, modestly strengthen gradual evidence-load precision loss so unanimous sequences produce the small positive decline in Experiment 15. Treat Experiments 9 and 19 as weak calibration constraints, and defer broad range widening until these pooled means improve; afterward, slightly wider subject-level saturation, load-noise, and lapse ranges may address the systematically underestimated heterogeneity.

**Outcome of this advice:** iter 5 candidate loss=0.1880 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Spatially Indexed Reliability-Gated Revision. People inspect product features in their displayed left-to-right order rather than rearranging them by instructed validity. Each discriminating cue revises a bounded directional commitment. Revision is leaky, so recent cues ordinarily have greater influence, while repeated same-direction cues build a saturating, reliability-sensitive stability trace rather than an all-or-none coalition. This trace remains associated with an established incumbent during a reversal and is eroded gradually by cumulative challenger evidence; consequently, every cue in an opposing run remains partially gated until the challenger has continuously displaced the incumbent. Instructed validity jointly affects cue accessibility, revision magnitude, and the rate at which evidence builds or erodes stability. A weak cue that locally opposes a more reliable preceding cue evokes surprise-driven orienting, temporarily increasing its accessibility and capacity to revise the incumbent commitment. Accessibility also rises gradually toward the right side of the display, although the positional gradient is compressed on very long displays, with a small additional benefit for an isolated discriminating cue at the physical right edge. Increasing display load gradually reduces response precision, including a small decline for long unanimous sequences. Choices are generated from this single evolving commitment state, with no trial-level mixture of strategies.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Spatially Indexed Reliability-Gated Revision expects shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    spatial_attention = float(parameters[\"spatial_attention\"])\n    validity_gain = float(parameters[\"validity_gain\"])\n    leak_rate = float(parameters[\"leak_rate\"])\n    overwrite_strength = float(parameters[\"overwrite_strength\"])\n    surprise_orienting = float(parameters[\"surprise_orienting\"])\n    commitment_saturation = float(parameters[\"commitment_saturation\"])\n    load_noise = float(parameters[\"load_noise\"])\n    lapse_rate = float(parameters[\"lapse_rate\"])\n\n    diff = stim[0] - stim[1]\n    discriminating = np.flatnonzero(diff != 0.0)\n    if discriminating.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Communicated validity is represented as reliability, but it does not\n    # determine scan order. Standardization makes the same gain parameter\n    # usable across experiments with different validity ranges.\n    clipped_v = np.clip(validities, 0.5001, 0.999)\n    reliability = np.log(clipped_v / (1.0 - clipped_v))\n    rel_mean = float(np.mean(reliability))\n    rel_sd = float(np.std(reliability))\n    if rel_sd < 1e-8:\n        standardized_reliability = np.zeros(n_features, dtype=float)\n    else:\n        standardized_reliability = np.clip(\n            (reliability - rel_mean) / rel_sd, -2.5, 2.5\n        )\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # commitment is the bounded current decision state. stability_trace is a\n    # continuous signed record of commitment-congruent support. Unlike a run\n    # label, it is not reassigned after the first opposing update: cumulative\n    # challenger evidence must gradually erode and reverse it.\n    commitment = 0.0\n    stability_trace = 0.0\n    previous_direction = 0.0\n    previous_reliability = None\n    previous_index = None\n\n    # Compress only the positional component of accessibility on unusually\n    # long displays. This prevents normalized scan position from creating an\n    # unrealistically large endpoint effect while retaining a rightward bias.\n    position_scale = float(min(\n        1.0, np.sqrt(8.0 / float(max(8, n_features)))\n    ))\n\n    # On unusually long displays, intervening locations increase retention\n    # slightly. Ordinary displays retain the original recurrence unchanged.\n    long_display_fraction = float(np.clip(\n        (float(n_features) - 16.0) / 4.0, 0.0, 1.0\n    ))\n    effective_leak_rate = float(np.clip(\n        leak_rate + 0.10 * long_display_fraction, 0.0, 0.90\n    ))\n\n    for j in discriminating:\n        j = int(j)\n        direction = float(np.sign(diff[j]))\n        position = float(j) / float(max(1, n_features - 1))\n\n        # Ordinary accessibility follows a smooth left-to-right gradient.\n        # Validity changes accessibility without reordering the cues.\n        access_logit = (\n            1.15\n            + spatial_attention * position_scale * (2.0 * position - 1.0)\n            + validity_gain * standardized_reliability[j]\n        )\n\n        # A physically isolated discriminating item at the right boundary is\n        # modestly foregrounded. The conjunction prevents a generic terminal\n        # bonus from guaranteeing reversal on every display.\n        if previous_index is None:\n            gap_fraction = 0.0\n        else:\n            gap_fraction = float(max(0, j - previous_index - 1)) / float(\n                max(1, n_features - 1)\n            )\n        right_edge = float(j == n_features - 1)\n        isolation_bonus = (\n            0.32\n            * spatial_attention\n            * position_scale\n            * right_edge\n            * np.sqrt(gap_fraction)\n        )\n        access_logit += isolation_bonus\n        accessibility = sigmoid(access_logit)\n\n        # Reliability also scales encoded revision strength, but only\n        # compressively, so weak experts remain psychologically available.\n        reliability_strength = float(np.exp(\n            validity_gain * 0.55 * standardized_reliability[j]\n        ))\n        reliability_strength = float(np.clip(reliability_strength, 0.45, 2.20))\n\n        # Surprise is specifically a local reliability drop at an opposing\n        # boundary. It boosts attention to a weak challenger rather than\n        # installing a separate strategy or reversing the validity ordering.\n        orienting = 1.0\n        if previous_reliability is not None and previous_direction != direction:\n            raw_drop = max(0.0, float(previous_reliability - validities[j]))\n            incumbent_alignment = abs(commitment)\n            orienting += (\n                surprise_orienting\n                * raw_drop\n                * (0.45 + 0.55 * incumbent_alignment)\n            )\n\n        # Leakage creates recency. Every cue remains gated while its direction\n        # opposes the persistent stability trace. The gate depends on the trace\n        # itself, not instantaneous commitment, so it does not disappear merely\n        # because an initial challenger has moved commitment toward zero.\n        retained = float(np.clip(\n            effective_leak_rate * commitment, -0.999999, 0.999999\n        ))\n        opposing_stability = bool(\n            stability_trace != 0.0 and np.sign(stability_trace) != direction\n        )\n        if opposing_stability:\n            trace_gate_mass = min(2.5, abs(stability_trace) ** 0.75)\n            stability_gate = 1.0 + (\n                commitment_saturation * trace_gate_mass\n            )\n        else:\n            stability_gate = 1.0\n\n        revision = (\n            overwrite_strength\n            * accessibility\n            * reliability_strength\n            * orienting\n            / stability_gate\n        )\n\n        # Updating in bounded-logit coordinates gives gradual saturation for\n        # repeated agreement while preserving smooth directional revision.\n        bounded_coordinate = np.arctanh(retained)\n        commitment = float(np.tanh(\n            bounded_coordinate + direction * revision\n        ))\n\n        # The signed stability trace persists across a reversal. Congruent\n        # evidence consolidates it efficiently, whereas opposing evidence\n        # erodes it continuously and reliability-dependently before eventually\n        # installing a new incumbent. No switch threshold or run state is used.\n        trace_alignment = direction * float(np.tanh(stability_trace))\n        assimilation = (\n            0.50 + 0.50 * sigmoid(2.5 * trace_alignment)\n        )\n        trace_retention = float(np.sqrt(effective_leak_rate))\n        stability_trace = (\n            trace_retention * stability_trace\n            + direction\n            * accessibility\n            * reliability_strength\n            * assimilation\n        )\n        stability_trace = float(np.clip(stability_trace, -6.0, 6.0))\n\n        previous_direction = direction\n        previous_reliability = float(validities[j])\n        previous_index = j\n\n    m = int(discriminating.size)\n\n    # More reasons and more displayed locations introduce gradual precision\n    # loss. There is no fixed fatigue boundary and no conflict-only collapse.\n    display_load = float(max(0, n_features - 1))\n    evidence_load = float(max(0, m - 2))\n    precision = 3.35 / (\n        1.0\n        + load_noise * display_load\n        + 0.55 * load_noise * evidence_load\n    )\n    precision = float(np.clip(precision, 0.25, 6.0))\n\n    p_a_core = sigmoid(precision * commitment)\n    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)\n    probs = (\n        (1.0 - lapse_rate) * core\n        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)\n    )\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "spatial_attention": "[0.55, 1.15]",
    "validity_gain": "[0.12, 0.38]",
    "leak_rate": "[0.48, 0.67]",
    "overwrite_strength": "[1.75, 2.45]",
    "surprise_orienting": "[1.25, 2.30]",
    "commitment_saturation": "[0.48, 0.88]",
    "load_noise": "[0.020, 0.055]",
    "lapse_rate": "[0.04, 0.13]"
  },
  "rationale": "This is a minimal recurrence-level edit to the accepted candidate. The opposing stability gate no longer contains abs(retained), which previously caused protection to collapse exactly when an initial challenger moved commitment toward zero. It now depends directly and sublinearly on the still-persistent signed stability trace, so later cues remain gated until challenger evidence has actually eroded that trace. The saturating power transform avoids an all-or-none coalition and limits the impact on weak incumbents and short displays. A small retention increase is added only for displays longer than 16 features, targeting the excessive endpoint effects in Experiments 14 and 21 without changing the ordinary six- to ten-feature conditions. Finally, the load-noise range is modestly increased to produce a gradual decline in precision for long unanimous sequences, as required by Experiment 15. All other accessibility, validity, surprise, overwrite, lapse, and state-update equations are unchanged."
}
```

## Usage

```json
{
  "input_tokens": 46382,
  "output_tokens": 4198
}
```
