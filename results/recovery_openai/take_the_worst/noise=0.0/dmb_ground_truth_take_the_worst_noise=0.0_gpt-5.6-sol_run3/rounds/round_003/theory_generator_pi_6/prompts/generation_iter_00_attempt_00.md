# generation_iter_00_attempt_00

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_3`
- The recommendation below acts on THEORY 2 (= `pi_3`).

Replace pi_3 with a Structure-Sensitive Anchoring-and-Comparison theory. The new theory should propose that subjects first construct a display representation whose organization depends jointly on discriminating-cue saturation, repeated cue coalitions, spatial grouping, and the presence of a stable focal or anchor cue—not solely on absolute conflict count or validity-position rank coherence. Sparse or readily chunked displays should induce strong anchoring on an early cue or early cue coalition, permitting earliest-cue choice rates near 0.85 as in Experiment 2. Highly saturated displays with salient opposing coalitions should recruit comparative or tally-like processing, accounting for Experiment 8. The shift must be broad and structure-sensitive rather than a fixed transition near k=7.3, so k=7 and k=9 displays can yield nearly identical behavior as in Experiment 1. Communicated validity should usually have little direct effect, consistent with Experiments 4-7, but may reinforce a cue already made focal by position, repetition, or task structure; it should not be controlled by a global rank-correlation gate. Include subject variation in anchor strength and comparison recruitment to accommodate Experiment 2's larger observed between-subject variance. The new theory should make separable predictions from pi_5 by holding k constant while independently manipulating saturation k/n, spatial grouping, cue-coalition stability across trials, and whether one cue repeatedly occupies an invariant focal role. Its model should be constrained to reproduce four key facts simultaneously: no k=7-to-k=9 discontinuity, very strong k=5 primacy, negligible late-validity and validity-distance promotion, and majority-sensitive responding in fully saturated displays.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.550)

**Description**
Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Mixed support
    # among the residual cues creates interference with stopping and modestly
    # increases continuation; unanimous residual support leaves it unchanged.
    # Attention then continues through the residual hierarchy. Conditional on
    # a miss, residual accessibility decays shallowly by ordinal rank. A
    # separate terminal-gap penalty limits access to cues lying very far from
    # the leading cue without reinstating a steep gradient over early/middle
    # cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    if len(discriminating_order) > 2:
        residual_signs = np.sign(diff[np.asarray(discriminating_order[1:], dtype=int)])
        residual_mixedness = 1.0 - abs(float(np.mean(residual_signs)))
        mixed_residual_boost = float(parameters["mixed_residual_boost"])
        serial_second_look += mixed_residual_boost * residual_mixedness
        serial_second_look = float(np.clip(serial_second_look, 0.0, 1.0))

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
- transition_count: [7.25, 7.35]
- transition_steepness: [7.0, 8.5]
- position_bias: [0.90, 1.05]
- validity_modifier: [0.01, 0.04]
- serial_second_look: [0.36, 0.40]
- mixed_residual_boost: [0.10, 0.14]
- residual_rank_decay: [0.08, 0.12]
- terminal_gap_penalty: [4.0, 5.0]
- serial_precision: [1.65, 1.90]
- positional_discount: [0.06, 0.14]
- tally_precision: [1.05, 1.35]
- epsilon: [0.01, 0.04]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=-0.0008 (var=0.0242)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=-0.0925 (var=0.0100)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0196 (var=0.0013)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0012 (var=0.0143)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0185 (var=0.0006)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 7: real=0.0067 (var=0.0029) vs this=0.3450 (var=0.0092)
- Experiment 8: real=0.8446 (var=0.0066) vs this=0.6575 (var=0.0018)


---

### `pi_3` (overall score: 0.504)

**Description**
Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- accessibility_weights: [(0.48, 0.52)] * n_features
- position_strength: [0.02, 0.15]
- validity_strength: [3.0, 5.0]
- mapping_fidelity: [0.8, 1.0]
- complexity_power: [3.0, 8.0]
- attention_temperature: [0.30, 0.40]
- epsilon: [0.02, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=-0.0025 (var=0.0152)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=-0.0179 (var=0.0142)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0564 (var=0.0042)
- Experiment 4: real=0.0042 (var=0.0061) vs this=0.2100 (var=0.0120)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0571 (var=0.0041)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0112 (var=0.0004)
- Experiment 7: real=0.0067 (var=0.0029) vs this=-0.0229 (var=0.0102)
- Experiment 8: real=0.8446 (var=0.0066) vs this=0.4517 (var=0.0026)


---

### `pi_2` (overall score: 0.390)

**Description**
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=0.3208 (var=0.0247)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=0.0079 (var=0.0152)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0031 (var=0.0005)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0125 (var=0.0195)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0680 (var=0.0025)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 7: real=0.0067 (var=0.0029) vs this=-0.0104 (var=0.0034)
- Experiment 8: real=0.8446 (var=0.0066) vs this=0.1527 (var=0.0078)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code each response as adherence (1) versus non-adherence (0) to the
    # highest-validity cue, and recover the number of lower cues opposing it.
    adherence = []
    conflict_q = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        if a[0] > b[0]:
            top_cue_choice = 0
            q = int(np.sum(b[1:] > a[1:]))
        elif b[0] > a[0]:
            top_cue_choice = 1
            q = int(np.sum(a[1:] > b[1:]))
        else:
            continue

        adherence.append(float(response == top_cue_choice))
        conflict_q.append(q)

    adherence = np.asarray(adherence, dtype=float)
    conflict_q = np.asarray(conflict_q, dtype=int)

    low_conflict = conflict_q == 1
    high_conflict = conflict_q >= 4
    if not np.any(low_conflict) or not np.any(high_conflict):
        return float("nan")

    return float(np.mean(adherence[low_conflict]) -
                 np.mean(adherence[high_conflict]))

```

**Observed (real) value:** 0.0283 (var=0.0097)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0088 (var=0.0057)
- pi_2: 0.3208 (var=0.0247)
- pi_3: -0.0025 (var=0.0152)
- pi_4: -0.0108 (var=0.0114)
- pi_5: -0.0008 (var=0.0242)

### Experiment 2
**Design**
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    validities = np.asarray([0.67, 0.95, 0.58, 0.82, 0.53, 0.74, 0.61, 0.88], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    signed_agreement = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break

        if winner is not None and pd.notna(row["response"]):
            response = int(row["response"])
            signed_agreement.append(1.0 if response == winner else -1.0)

    return float(np.mean(signed_agreement)) if signed_agreement else float("nan")
```

**Observed (real) value:** -0.0050 (var=0.0051)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0079 (var=0.0152)
- pi_1: 0.7113 (var=0.0406)
- pi_3: -0.0179 (var=0.0142)
- pi_4: -0.0963 (var=0.0198)
- pi_5: -0.0925 (var=0.0100)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)

        # Feature 8 (index 7, equivalently the final feature here) is the
        # globally most valid cue in this experimental design.
        j = 7
        if a.size <= j or b.size <= j or a[j] == b[j]:
            continue

        cue8_winner = 0 if a[j] > b[j] else 1
        agreements.append(float(int(row['response']) == cue8_winner))

    if len(agreements) == 0:
        return float('nan')

    cue8_choice_rate = float(np.mean(agreements))

    # Soft classification around a preregistered separating rate. This is
    # smooth (unlike a hard threshold), remains defined per subject, and
    # compresses rates well inside either theory's predicted region.
    z = float(np.clip(20.0 * (cue8_choice_rate - 0.40), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0089 (var=0.0080)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9999 (var=0.0064)
- pi_3: 0.0564 (var=0.0042)
- pi_2: 0.0031 (var=0.0005)
- pi_4: 0.0084 (var=0.0002)
- pi_5: 0.0196 (var=0.0013)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast choices supported by the higher-validity cue on adjacent
    # conflicts versus conflicts separated by four or seven validity ranks.
    high_choice = {1: [], 4: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = np.flatnonzero(a != b)
        if diff.size != 2:
            continue

        high_idx = int(np.min(diff))  # Validities decrease with feature index.
        low_idx = int(np.max(diff))
        distance = low_idx - high_idx
        if distance not in high_choice:
            continue

        high_option = 0 if a[high_idx] > b[high_idx] else 1
        high_choice[distance].append(
            1.0 if int(row["response"]) == high_option else 0.0
        )

    if any(len(high_choice[d]) == 0 for d in (1, 4, 7)):
        return float("nan")

    adjacent = float(np.mean(high_choice[1]))
    well_separated = 0.5 * (
        float(np.mean(high_choice[4])) + float(np.mean(high_choice[7]))
    )
    return float(well_separated - adjacent)

```

**Observed (real) value:** 0.0042 (var=0.0061)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2100 (var=0.0120)
- pi_1: -0.0075 (var=0.0062)
- pi_2: -0.0125 (var=0.0195)
- pi_4: -0.0004 (var=0.0094)
- pi_5: -0.0012 (var=0.0143)

### Experiment 5
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Soft prevalence of subjects reliably choosing the later conflicting cue."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        later_cue_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            differing = np.flatnonzero(a != b)
            if differing.size == 0:
                continue

            # In this design, the later of the two discriminating cues is the
            # locally high-validity cue.
            later = int(np.max(differing))
            later_winner = 0 if a[later] > b[later] else 1
            later_cue_choices.append(float(int(row["response"]) == later_winner))

        if len(later_cue_choices) == 0:
            continue

        rate = float(np.mean(later_cue_choices))
        # Smoothly score whether the subject shows a reliable (>65%) rather
        # than near-chance preference for the validity-promoted later cue.
        z = float(np.clip(20.0 * (rate - 0.65), -60.0, 60.0))
        scores.append(1.0 / (1.0 + np.exp(-z)))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0033 (var=0.0001)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9652 (var=0.0010)
- pi_3: 0.0571 (var=0.0041)
- pi_1: 0.9496 (var=0.0122)
- pi_2: 0.0680 (var=0.0025)
- pi_5: 0.0185 (var=0.0006)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    target_choices = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 2 or b.size < 2:
            continue

        # Features 1 and 2 are aligned in this design. Define the target as
        # the option jointly recommended by these two high-validity cues.
        direction = float(np.sum(a[:2] - b[:2]))
        if direction > 0:
            target_response = 0
        elif direction < 0:
            target_response = 1
        else:
            continue

        response = int(row["response"])
        target_choices.append(float(response == target_response))

    if len(target_choices) == 0:
        return float("nan")

    target_rate = float(np.mean(target_choices))

    # Smoothly classify rates around the otherwise-empty region separating
    # the advocated prediction (roughly .38-.48) from the competitor's
    # prediction (roughly .80-.86). Saturation also limits binomial and
    # parameter-driven between-subject variation away from the boundary.
    z = float(np.clip(20.0 * (target_rate - 0.64), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0005 (var=0.0000)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0112 (var=0.0004)
- pi_4: 0.9796 (var=0.0006)
- pi_1: 0.9857 (var=0.0302)
- pi_2: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Difference in first-discriminating-cue choice rates at k=7 versus k=9."""
    sparse_hits = []
    dense_hits = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        k = int(discriminating.size)
        if k not in (7, 9):
            continue

        first = int(discriminating[0])
        first_cue_choice = 0 if diff[first] > 0 else 1
        hit = float(int(row["response"]) == first_cue_choice)

        if k == 7:
            sparse_hits.append(hit)
        else:
            dense_hits.append(hit)

    if len(sparse_hits) == 0 or len(dense_hits) == 0:
        return float("nan")

    return float(np.mean(sparse_hits) - np.mean(dense_hits))

```

**Observed (real) value:** 0.0067 (var=0.0029)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3450 (var=0.0092)
- pi_3: -0.0229 (var=0.0102)
- pi_1: -0.1917 (var=0.0066)
- pi_2: -0.0104 (var=0.0034)
- pi_4: 0.0150 (var=0.0076)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the earliest discriminating cue."""
    agreements = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0 or pd.isna(response):
            continue

        first = int(discriminating[0])
        leading_choice = 0 if diff[first] > 0 else 1
        agreements.append(float(int(response) == leading_choice))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.8446 (var=0.0066)
**Other theories' values on this metric (for reference):**
- pi_3: 0.4517 (var=0.0026)
- pi_5: 0.6575 (var=0.0018)
- pi_1: 0.1412 (var=0.0121)
- pi_2: 0.1527 (var=0.0078)
- pi_4: 0.8225 (var=0.0020)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

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
  "theory": "Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    # Relational Chunk-Anchor Arbitration.\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.shape != (n_features,):\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    diff = a - b\n    signs = np.sign(diff)\n    discriminating = np.flatnonzero(signs != 0.0)\n    k = int(discriminating.size)\n    if k == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def first_coalition(sign_vector):\n        idx = np.flatnonzero(sign_vector != 0.0)\n        if idx.size == 0:\n            return np.asarray([], dtype=int)\n        first = int(idx[0])\n        direction = float(sign_vector[first])\n        members = [first]\n        j = first + 1\n        # Spatial contiguity matters: a tie or reversal terminates the chunk.\n        while j < sign_vector.size and sign_vector[j] == direction:\n            members.append(j)\n            j += 1\n        return np.asarray(members, dtype=int)\n\n    anchor = first_coalition(signs)\n    anchor_direction = float(signs[int(anchor[0])])\n    anchor_len = int(anchor.size)\n\n    # Count contiguous, same-direction discriminating runs. Ties separate runs.\n    run_count = 0\n    previous_sign = 0.0\n    previous_index = -2\n    for j in discriminating:\n        j = int(j)\n        s = float(signs[j])\n        if j != previous_index + 1 or s != previous_sign:\n            run_count += 1\n        previous_index = j\n        previous_sign = s\n\n    if k > 1:\n        fragmentation = float(run_count - 1) / float(k - 1)\n    else:\n        fragmentation = 0.0\n    saturation = float(k) / float(max(n_features, 1))\n\n    # A good anchor schema has a compact initial chunk, little same-direction\n    # evidence outside that chunk, and a spatially coherent opposing coalition.\n    same_total = int(np.sum(signs == anchor_direction))\n    opposite_positions = np.flatnonzero(signs == -anchor_direction)\n    if opposite_positions.size == 0:\n        opposite_concentration = 0.0\n    else:\n        longest = 1\n        current = 1\n        for m in range(1, opposite_positions.size):\n            if int(opposite_positions[m]) == int(opposite_positions[m - 1]) + 1:\n                current += 1\n                longest = max(longest, current)\n            else:\n                current = 1\n        opposite_concentration = float(longest) / float(opposite_positions.size)\n\n    focal_exclusivity = float(anchor_len) / float(max(same_total, 1))\n    compactness = float(np.exp(-0.70 * float(anchor_len - 1)))\n    current_schema = compactness * focal_exclusivity * opposite_concentration\n\n    # Repeated displays stabilize a relational anchor without learning cue\n    # correctness. We measure whether the same earliest location and coalition\n    # have recurred in the subject's preceding displays.\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    focal_matches = []\n    coalition_matches = []\n    for old_a, old_b in zip(past_a, past_b):\n        old_a = np.asarray(old_a, dtype=np.float64)\n        old_b = np.asarray(old_b, dtype=np.float64)\n        if old_a.shape != (n_features,) or old_b.shape != (n_features,):\n            continue\n        old_signs = np.sign(old_a - old_b)\n        old_anchor = first_coalition(old_signs)\n        if old_anchor.size == 0:\n            continue\n        focal_matches.append(float(int(old_anchor[0]) == int(anchor[0])))\n        coalition_matches.append(float(np.array_equal(old_anchor, anchor)))\n\n    if focal_matches:\n        recurrence = 0.5 * float(np.mean(focal_matches)) + 0.5 * float(np.mean(coalition_matches))\n    else:\n        recurrence = 0.0\n\n    stability_weight = float(parameters[\"stability_weight\"])\n    schema_strength = current_schema * (0.65 + stability_weight * recurrence)\n    schema_strength = float(np.clip(schema_strength, 0.0, 1.0))\n\n    # Anchor route. Validity cannot select a nonfocal cue; it only produces a\n    # small reliability increment if the already focal coalition is relatively\n    # valid. This precludes global validity-rank search and late-cue promotion.\n    if np.ptp(validities) > 0.0:\n        v_scaled = (validities - np.mean(validities)) / np.ptp(validities)\n    else:\n        v_scaled = np.zeros(n_features, dtype=np.float64)\n\n    focal_validity = max(0.0, float(np.mean(v_scaled[anchor])))\n    anchor_accuracy = float(parameters[\"anchor_accuracy\"])\n    validity_reinforcement = float(parameters[\"validity_reinforcement\"])\n    anchor_accuracy += validity_reinforcement * focal_validity\n    anchor_accuracy = float(np.clip(anchor_accuracy, 0.5, 0.97))\n\n    anchor_winner = 0 if anchor_direction > 0.0 else 1\n    p_anchor = np.empty(2, dtype=np.float64)\n    p_anchor[anchor_winner] = anchor_accuracy\n    p_anchor[1 - anchor_winner] = 1.0 - anchor_accuracy\n\n    # Comparison route: a majority-sensitive tally with only shallow position\n    # and validity modulation. Neither modulation can create lexicographic use\n    # of a late high-validity cue.\n    if n_features > 1:\n        normalized_position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)\n    else:\n        normalized_position = np.zeros(1, dtype=np.float64)\n\n    comparison_position_discount = float(parameters[\"comparison_position_discount\"])\n    comparison_validity_weight = float(parameters[\"comparison_validity_weight\"])\n    comparison_weights = np.exp(-comparison_position_discount * normalized_position)\n    comparison_weights *= 1.0 + comparison_validity_weight * v_scaled\n    comparison_weights = np.clip(comparison_weights, 1e-8, None)\n\n    signed_comparison = float(np.dot(comparison_weights, signs))\n    comparison_precision = float(parameters[\"comparison_precision\"])\n    comparison_logits = np.array(\n        [comparison_precision * signed_comparison, 0.0], dtype=np.float64\n    )\n    comparison_logits -= np.max(comparison_logits)\n    p_compare = np.exp(comparison_logits)\n    p_compare /= p_compare.sum()\n\n    # Arbitration depends jointly on saturation, coalition fragmentation,\n    # relational anchor quality, and its recurrence. There is no fixed k\n    # threshold, so similarly organized k=7 and k=9 displays remain alike.\n    comparison_bias = float(parameters[\"comparison_bias\"])\n    saturation_weight = float(parameters[\"saturation_weight\"])\n    fragmentation_weight = float(parameters[\"fragmentation_weight\"])\n    anchor_schema_weight = float(parameters[\"anchor_schema_weight\"])\n\n    gate_argument = (\n        comparison_bias\n        + saturation_weight * saturation\n        + fragmentation_weight * fragmentation\n        - anchor_schema_weight * schema_strength\n    )\n    gate_argument = float(np.clip(gate_argument, -60.0, 60.0))\n    comparison_gate = 1.0 / (1.0 + np.exp(-gate_argument))\n\n    p_core = (1.0 - comparison_gate) * p_anchor + comparison_gate * p_compare\n\n    epsilon = float(np.clip(parameters[\"epsilon\"], 0.0, 1.0))\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "anchor_accuracy": "[0.84, 0.94]",
    "validity_reinforcement": "[0.00, 0.03]",
    "stability_weight": "[0.25, 0.35]",
    "comparison_bias": "[-4.5, -3.5]",
    "saturation_weight": "[6.5, 7.5]",
    "fragmentation_weight": "[3.5, 4.5]",
    "anchor_schema_weight": "[7.0, 8.0]",
    "comparison_position_discount": "[0.00, 0.04]",
    "comparison_validity_weight": "[0.00, 0.04]",
    "comparison_precision": "[1.5, 2.2]",
    "epsilon": "[0.01, 0.04]"
  },
  "rationale": "The model replaces pi_3's global validity-position coherence gate with a representation-based arbitration mechanism. Conflict count alone never selects a strategy. Compact early coalitions and recurring focal roles support anchoring, while saturation and fragmentation jointly recruit comparison. This preserves nearly constant top-cue adherence across Experiment 1's conflict levels, gives the readily chunked five-cue displays in Experiment 8 strong early-coalition choices near the observed 0.845, and sends the fully saturated displays of Experiment 6 toward the opposing majority. The similarly fragmented k=7 and k=9 displays in Experiment 7 both recruit comparison, avoiding pi_5's artificial discontinuity. Experiment 2's scattered coalitions also receive substantial comparative processing, making responses approximately orthogonal to validity-ranked search. Validity only adjusts the reliability of an already selected anchor and has very small weight in comparison, so it cannot promote the late cue in Experiments 3 and 5 or create a validity-distance effect in Experiment 4. Broad subject variation in anchor accuracy and the comparison gate supplies the requested heterogeneity. Unlike a fixed conflict threshold, the theory predicts separable effects of k/n, spatial contiguity, coalition recurrence, and invariant focal placement when k is held constant."
}
```

## Usage

```json
{
  "input_tokens": 17357,
  "output_tokens": 8064
}
```
