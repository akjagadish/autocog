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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Replace pi_5 with a Redundant-Onset Coalition theory. Its central claim should be that primacy arises only when the earliest evidence forms a sufficiently redundant and perceptually segregated coalition, rather than from the first discriminating cue or from an absolute conflict-count threshold. A singleton onset should have low anchoring strength regardless of how coherent the opposing coalition is; choices then reflect a noisy, weakly position-weighted comparison, predicting lower leading-option choice for a 1-to-5 display than for a 2-to-4 display and hence the negative Experiment 1 contrast. An exclusive contiguous two-cue onset opposed by one coherent coalition should cross a nonlinear chunk-formation threshold and support highly reliable anchor choices, accommodating Experiments 2 and 10 even against a numerical majority. Extra same-direction cues outside the onset, alternation, and fragmentation should reduce coalition exclusivity and return control toward comparison, accommodating Experiment 8. Arbitration should depend on initial-coalition redundancy, exclusivity, opponent concentration, and fragmentation—not k itself—so k=7 and k=9 can behave similarly as in Experiment 9. Validity should remain weak and mainly reinforce an already formed coalition, consistent with Experiments 4–7. Recurrence may modestly strengthen an existing coalition, but it should not create an anchor from an otherwise inadequate singleton; moreover, the current Experiment 2 metric establishes high anchor adherence but does not isolate the proposed common-versus-rare history effect, so strong learning or recurrence parameters are not yet justified. Include subject-level variation in the coalition-formation threshold and anchor reliability to match the observed variances. This theory is a stronger competitor to pi_6 because it retains relational organization while making the empirically crucial, falsifiable claim that onset redundancy—not compactness alone—is required for anchoring.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.487)

**Description**
Relational Chunk–Anchor Arbitration theory. Before choosing, people perceptually organize discriminating cues into contiguous, same-direction coalitions rather than treating conflict count as the primary control variable. An early compact coalition becomes an anchor when it has a coherent opposing coalition, recurs across trials, or occupies an invariant focal location. The anchor normally determines choice, and communicated validity can only slightly reinforce cues already contained in that focal coalition. Comparative processing is recruited when the display is saturated and structurally fragmented—especially when evidence alternates among several coalitions—causing a weakly weighted majority comparison. Thus, equal conflict counts can produce different strategies, while different counts such as k=7 and k=9 can produce the same strategy when their relational organization is alike. Compact two-coalition displays preserve strong primacy, whereas fully saturated, fragmented displays elicit majority-sensitive responding. Subject-specific variation in anchor reliability and comparison recruitment produces stable individual differences without outcome learning.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Relational Chunk-Anchor Arbitration.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    signs = np.sign(diff)
    discriminating = np.flatnonzero(signs != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def first_coalition(sign_vector):
        idx = np.flatnonzero(sign_vector != 0.0)
        if idx.size == 0:
            return np.asarray([], dtype=int)
        first = int(idx[0])
        direction = float(sign_vector[first])
        members = [first]
        j = first + 1
        # Spatial contiguity matters: a tie or reversal terminates the chunk.
        while j < sign_vector.size and sign_vector[j] == direction:
            members.append(j)
            j += 1
        return np.asarray(members, dtype=int)

    anchor = first_coalition(signs)
    anchor_direction = float(signs[int(anchor[0])])
    anchor_len = int(anchor.size)

    # Count contiguous, same-direction discriminating runs. Ties separate runs.
    run_count = 0
    previous_sign = 0.0
    previous_index = -2
    for j in discriminating:
        j = int(j)
        s = float(signs[j])
        if j != previous_index + 1 or s != previous_sign:
            run_count += 1
        previous_index = j
        previous_sign = s

    if k > 1:
        fragmentation = float(run_count - 1) / float(k - 1)
    else:
        fragmentation = 0.0
    saturation = float(k) / float(max(n_features, 1))

    # A good anchor schema has a compact initial chunk, little same-direction
    # evidence outside that chunk, and a spatially coherent opposing coalition.
    same_total = int(np.sum(signs == anchor_direction))
    opposite_positions = np.flatnonzero(signs == -anchor_direction)
    if opposite_positions.size == 0:
        opposite_concentration = 0.0
    else:
        longest = 1
        current = 1
        for m in range(1, opposite_positions.size):
            if int(opposite_positions[m]) == int(opposite_positions[m - 1]) + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        opposite_concentration = float(longest) / float(opposite_positions.size)

    focal_exclusivity = float(anchor_len) / float(max(same_total, 1))
    compactness = float(np.exp(-0.70 * float(anchor_len - 1)))
    current_schema = compactness * focal_exclusivity * opposite_concentration

    # Repeated displays stabilize a relational anchor without learning cue
    # correctness. We measure whether the same earliest location and coalition
    # have recurred in the subject's preceding displays.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    focal_matches = []
    coalition_matches = []
    for old_a, old_b in zip(past_a, past_b):
        old_a = np.asarray(old_a, dtype=np.float64)
        old_b = np.asarray(old_b, dtype=np.float64)
        if old_a.shape != (n_features,) or old_b.shape != (n_features,):
            continue
        old_signs = np.sign(old_a - old_b)
        old_anchor = first_coalition(old_signs)
        if old_anchor.size == 0:
            continue
        focal_matches.append(float(int(old_anchor[0]) == int(anchor[0])))
        coalition_matches.append(float(np.array_equal(old_anchor, anchor)))

    if focal_matches:
        recurrence = 0.5 * float(np.mean(focal_matches)) + 0.5 * float(np.mean(coalition_matches))
    else:
        recurrence = 0.0

    stability_weight = float(parameters["stability_weight"])
    schema_strength = current_schema * (0.65 + stability_weight * recurrence)
    schema_strength = float(np.clip(schema_strength, 0.0, 1.0))

    # Anchor route. Validity cannot select a nonfocal cue; it only produces a
    # small reliability increment if the already focal coalition is relatively
    # valid. This precludes global validity-rank search and late-cue promotion.
    if np.ptp(validities) > 0.0:
        v_scaled = (validities - np.mean(validities)) / np.ptp(validities)
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    focal_validity = max(0.0, float(np.mean(v_scaled[anchor])))
    anchor_accuracy = float(parameters["anchor_accuracy"])
    validity_reinforcement = float(parameters["validity_reinforcement"])
    anchor_accuracy += validity_reinforcement * focal_validity
    anchor_accuracy = float(np.clip(anchor_accuracy, 0.5, 0.97))

    anchor_winner = 0 if anchor_direction > 0.0 else 1
    p_anchor = np.empty(2, dtype=np.float64)
    p_anchor[anchor_winner] = anchor_accuracy
    p_anchor[1 - anchor_winner] = 1.0 - anchor_accuracy

    # Comparison route: a majority-sensitive tally with only shallow position
    # and validity modulation. Neither modulation can create lexicographic use
    # of a late high-validity cue.
    if n_features > 1:
        normalized_position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    comparison_position_discount = float(parameters["comparison_position_discount"])
    comparison_validity_weight = float(parameters["comparison_validity_weight"])
    comparison_weights = np.exp(-comparison_position_discount * normalized_position)
    comparison_weights *= 1.0 + comparison_validity_weight * v_scaled
    comparison_weights = np.clip(comparison_weights, 1e-8, None)

    signed_comparison = float(np.dot(comparison_weights, signs))
    comparison_precision = float(parameters["comparison_precision"])
    comparison_logits = np.array(
        [comparison_precision * signed_comparison, 0.0], dtype=np.float64
    )
    comparison_logits -= np.max(comparison_logits)
    p_compare = np.exp(comparison_logits)
    p_compare /= p_compare.sum()

    # Arbitration depends jointly on saturation, coalition fragmentation,
    # relational anchor quality, and its recurrence. There is no fixed k
    # threshold, so similarly organized k=7 and k=9 displays remain alike.
    comparison_bias = float(parameters["comparison_bias"])
    saturation_weight = float(parameters["saturation_weight"])
    fragmentation_weight = float(parameters["fragmentation_weight"])
    anchor_schema_weight = float(parameters["anchor_schema_weight"])

    gate_argument = (
        comparison_bias
        + saturation_weight * saturation
        + fragmentation_weight * fragmentation
        - anchor_schema_weight * schema_strength
    )
    gate_argument = float(np.clip(gate_argument, -60.0, 60.0))
    comparison_gate = 1.0 / (1.0 + np.exp(-gate_argument))

    p_core = (1.0 - comparison_gate) * p_anchor + comparison_gate * p_compare

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
- anchor_accuracy: [0.84, 0.94]
- validity_reinforcement: [0.00, 0.03]
- stability_weight: [0.25, 0.35]
- comparison_bias: [-4.5, -3.5]
- saturation_weight: [6.5, 7.5]
- fragmentation_weight: [3.5, 4.5]
- anchor_schema_weight: [7.0, 8.0]
- comparison_position_discount: [0.00, 0.04]
- comparison_validity_weight: [0.00, 0.04]
- comparison_precision: [1.5, 2.2]
- epsilon: [0.01, 0.04]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=0.0050 (var=0.0088)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=0.0338 (var=0.0097)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0020 (var=0.0000)
- Experiment 4: real=0.0042 (var=0.0061) vs this=0.0012 (var=0.0071)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.0001 (var=0.0000)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 7: real=0.0067 (var=0.0029) vs this=0.0271 (var=0.0045)
- Experiment 8: real=0.8446 (var=0.0066) vs this=0.8446 (var=0.0020)
- Experiment 9: real=-0.2167 (var=0.0112) vs this=0.7386 (var=0.0067)
- Experiment 10: real=0.8775 (var=0.0059) vs this=0.3240 (var=0.0045)


---

### `pi_3` (overall score: 0.426)

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
- Experiment 9: real=-0.2167 (var=0.0112) vs this=-0.1422 (var=0.0122)
- Experiment 10: real=0.8775 (var=0.0059) vs this=0.3040 (var=0.0021)


---

### `pi_4` (overall score: 0.383)

**Description**
Accessibility-Anchored Fixed Lexicographic Search. Each person constructs one stable cue hierarchy at the beginning of a task and reuses it across trials. Priority is dominated by display position, with earlier cues receiving a large accessibility advantage. Subject-specific long-run accessibility or salience offsets can alter priorities locally, while communicated validity provides only a weak additive adjustment. The strength of that adjustment is expressed as a narrowly distributed ratio to positional anchoring, preventing independent parameter draws from causing unstable population-wide cue-order reversals. Thus validity can resolve competition between nearby positions but cannot promote a late cue merely because it has the highest instructed validity. Choice follows the first discriminating cue in this fixed hierarchy. Rare search slips skip that cue and consult the next discriminating cue, while bounded response imprecision and a small lapse rate generate occasional choices against the selected cue. The hierarchy is not resampled trial by trial and does not depend on the number of conflicting later cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Accessibility-Anchored Fixed Lexicographic Search.
    # Parameters are fixed for the subject, so the resulting hierarchy is
    # stable across all trials. History is intentionally ignored because no
    # outcome feedback is available to support cue-order learning.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(
        parameters["accessibility_weights"], dtype=float
    )

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != "
            f"n_features {n_features}."
        )

    # Position imposes a linear cost for moving attention to later cues.
    # Raw validity differences are deliberately not converted to ranks:
    # communicated validity is only a weak local adjustment and therefore
    # cannot receive an extra boost when its ordering matches the display.
    position_strength = float(parameters["position_strength"])
    validity_position_ratio = float(parameters["validity_position_ratio"])
    validity_weight = validity_position_ratio * position_strength
    positions = np.arange(n_features, dtype=float)
    centered_validity = validities - float(np.mean(validities))

    priority = (
        -position_strength * positions
        + accessibility
        + validity_weight * centered_validity
    )

    # Stable sorting supplies a deterministic tie rule favoring earlier
    # positions. Since all components are subject-level constants, this order
    # is fixed rather than redrawn on every trial.
    cue_order = np.argsort(-priority, kind="stable")

    a, b = stim[0], stim[1]
    discriminating_order = [int(j) for j in cue_order if a[j] != b[j]]
    if len(discriminating_order) == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def cue_winner(j):
        return 0 if a[j] > b[j] else 1

    first_winner = cue_winner(discriminating_order[0])

    # A rare search-control slip skips the first discriminating cue. This is
    # a tightly constrained perturbation of a fixed hierarchy, not a newly
    # sampled Plackett-Luce ordering.
    order_perturbation = float(parameters["order_perturbation"])
    order_perturbation = float(np.clip(order_perturbation, 0.0, 1.0))
    if len(discriminating_order) >= 2:
        second_winner = cue_winner(discriminating_order[1])
    else:
        second_winner = first_winner

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - order_perturbation
    p_selected[second_winner] += order_perturbation

    # Conditional on the selected cue, response precision controls accidental
    # reversal of its recommendation. The stable logistic form avoids an
    # unconstrained trial-wise attentional softmax.
    response_precision = float(parameters["response_precision"])
    if response_precision >= 0.0:
        response_reliability = 1.0 / (1.0 + np.exp(-response_precision))
    else:
        e = np.exp(response_precision)
        response_reliability = e / (1.0 + e)

    p_core = (
        response_reliability * p_selected
        + (1.0 - response_reliability) * p_selected[::-1]
    )

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs):
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
- accessibility_weights: [(0.0, 0.03)] * n_features
- position_strength: [0.12, 0.16]
- validity_position_ratio: [3.80, 3.90]
- order_perturbation: [0.0, 0.03]
- response_precision: [1.45, 1.85]
- epsilon: [0.01, 0.04]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0283 (var=0.0097) vs this=-0.0108 (var=0.0114)
- Experiment 2: real=-0.0050 (var=0.0051) vs this=-0.0963 (var=0.0198)
- Experiment 3: real=0.0089 (var=0.0080) vs this=0.0084 (var=0.0002)
- Experiment 4: real=0.0042 (var=0.0061) vs this=-0.0004 (var=0.0094)
- Experiment 5: real=0.0033 (var=0.0001) vs this=0.9652 (var=0.0010)
- Experiment 6: real=0.0005 (var=0.0000) vs this=0.9796 (var=0.0006)
- Experiment 7: real=0.0067 (var=0.0029) vs this=0.0150 (var=0.0076)
- Experiment 8: real=0.8446 (var=0.0066) vs this=0.8225 (var=0.0020)
- Experiment 9: real=-0.2167 (var=0.0112) vs this=0.0003 (var=0.0082)
- Experiment 10: real=0.8775 (var=0.0059) vs this=0.8377 (var=0.0017)


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
- pi_6: 0.0050 (var=0.0088)

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
- pi_6: 0.0338 (var=0.0097)

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
- pi_6: 0.0020 (var=0.0000)

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
- pi_6: 0.0012 (var=0.0071)

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
- pi_6: 0.0001 (var=0.0000)

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
- pi_6: 0.0000 (var=0.0000)

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
- pi_6: 0.0271 (var=0.0045)

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
- pi_6: 0.8446 (var=0.0020)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Compact-minus-fragmented change in first-cue-consistent choice."""
    compact_scores = []
    fragmented_scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        signs = np.sign(a - b)
        if signs[0] == 0 or np.any(signs == 0):
            continue

        # Code whether the response follows the option favored by feature 1.
        first_cue_response = 0 if signs[0] > 0 else 1
        follows_first = float(int(row["response"]) == first_cue_response)

        # The compact pattern has a singleton initial coalition followed by
        # one unanimous opposing coalition: +----- or its mirror -+++++.
        is_compact = bool(np.all(signs[1:] == -signs[0]))
        if is_compact:
            compact_scores.append(follows_first)
        else:
            fragmented_scores.append(follows_first)

    if len(compact_scores) == 0 or len(fragmented_scores) == 0:
        return float("nan")

    return float(np.mean(compact_scores) - np.mean(fragmented_scores))
```

**Observed (real) value:** -0.2167 (var=0.0112)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0636 (var=0.0139)
- pi_6: 0.7386 (var=0.0067)
- pi_1: 0.0192 (var=0.0079)
- pi_2: -0.0028 (var=0.0066)
- pi_3: -0.1422 (var=0.0122)
- pi_4: 0.0003 (var=0.0082)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        signs = np.sign(a - b)
        discriminating = np.flatnonzero(signs != 0)
        if discriminating.size == 0:
            continue

        first = int(discriminating[0])
        anchor_direction = signs[first]

        # Identify the initial contiguous coalition, stopping at a tie or reversal.
        anchor_length = 1
        j = first + 1
        while j < signs.size and signs[j] == anchor_direction:
            anchor_length += 1
            j += 1

        same_count = int(np.sum(signs == anchor_direction))
        opposite_count = int(np.sum(signs == -anchor_direction))

        # Restrict the score to the diagnostic 2-versus-6 displays in which
        # following the initial coalition conflicts with the cue majority.
        if anchor_length != 2 or same_count != 2 or opposite_count != 6:
            continue

        anchor_choice = 0 if anchor_direction > 0 else 1
        scores.append(float(int(row["response"]) == anchor_choice))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8775 (var=0.0059)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3240 (var=0.0045)
- pi_5: 0.0240 (var=0.0002)
- pi_1: 0.8425 (var=0.0106)
- pi_2: 0.1260 (var=0.0072)
- pi_3: 0.3040 (var=0.0021)
- pi_4: 0.8377 (var=0.0017)


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
  "theory": "Redundant-Onset Coalition Formation theory. People initially encode the earliest contiguous, same-direction evidence as a candidate perceptual coalition, but temporal or spatial priority alone is insufficient for anchoring. The candidate becomes an anchor only after crossing a subject-specific redundancy threshold, normally requiring at least two mutually reinforcing onset cues. Formation is further controlled by relational organization: cues supporting the onset elsewhere in the display undermine its exclusivity, a concentrated opposing coalition strengthens figure-ground segregation, and fragmentation weakens chunk formation. Once formed, the onset coalition controls choice with high but individually varying reliability, even against a large cue majority. A singleton onset instead enters a comparison route with a modest first-evidence bias and low evidence precision. This produces stable but nonlexicographic primacy without allowing a singleton to become a true anchor. In small, fully occupied displays, coherent opposing evidence can be perceptually integrated, making a compact singleton-versus-majority pattern less favorable to the singleton than a fragmented pattern. Failed formation of an otherwise redundant onset recruits more precise comparison, explaining why an onset with additional same-direction cues outside its initial chunk can lose to the majority. Communicated validity only weakly modulates comparison and slightly reinforces an already formed coalition; it cannot promote a late cue into an anchor. Display recurrence provides only a small increment to an already redundant candidate and can never turn a singleton into an anchor.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.shape != (n_features,):\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    signs = np.sign(a - b)\n    discriminating = np.flatnonzero(signs != 0.0)\n    k = int(discriminating.size)\n    if k == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def onset_chunk(sign_vector):\n        active = np.flatnonzero(sign_vector != 0.0)\n        if active.size == 0:\n            return np.asarray([], dtype=int)\n        start = int(active[0])\n        direction = float(sign_vector[start])\n        members = [start]\n        j = start + 1\n        while j < sign_vector.size and sign_vector[j] == direction:\n            members.append(j)\n            j += 1\n        return np.asarray(members, dtype=int)\n\n    def run_statistics(sign_vector):\n        active = np.flatnonzero(sign_vector != 0.0)\n        runs = []\n        if active.size == 0:\n            return runs\n        current_sign = float(sign_vector[int(active[0])])\n        current = [int(active[0])]\n        previous = int(active[0])\n        for raw_j in active[1:]:\n            j = int(raw_j)\n            s = float(sign_vector[j])\n            if j == previous + 1 and s == current_sign:\n                current.append(j)\n            else:\n                runs.append((current_sign, current))\n                current_sign = s\n                current = [j]\n            previous = j\n        runs.append((current_sign, current))\n        return runs\n\n    onset = onset_chunk(signs)\n    first_index = int(onset[0])\n    onset_direction = float(signs[first_index])\n    onset_length = int(onset.size)\n    onset_winner = 0 if onset_direction > 0.0 else 1\n\n    runs = run_statistics(signs)\n    run_count = len(runs)\n    if k > 1:\n        fragmentation = float(run_count - 1) / float(k - 1)\n    else:\n        fragmentation = 0.0\n\n    same_total = int(np.sum(signs == onset_direction))\n    opposite_total = int(np.sum(signs == -onset_direction))\n    extra_same = max(0, same_total - onset_length)\n    extra_fraction = float(extra_same) / float(max(same_total, 1))\n\n    opposite_runs = [\n        members for direction, members in runs\n        if direction == -onset_direction\n    ]\n    if opposite_runs:\n        longest_opposite = max(len(members) for members in opposite_runs)\n        opponent_concentration = (\n            float(longest_opposite) / float(max(opposite_total, 1))\n        )\n    else:\n        longest_opposite = 0\n        opponent_concentration = 0.0\n\n    opponent_mass = 1.0 - np.exp(-float(opposite_total) / 2.0)\n\n    # Recurrence can reinforce only an already redundant candidate. It does\n    # not enter the formation equation for singleton onsets.\n    recurrence = 0.0\n    if onset_length >= 2:\n        matches = []\n        past_a = history.get(\"option_a_ratings\", [])\n        past_b = history.get(\"option_b_ratings\", [])\n        for old_a, old_b in zip(past_a, past_b):\n            old_a = np.asarray(old_a, dtype=np.float64)\n            old_b = np.asarray(old_b, dtype=np.float64)\n            if old_a.shape != (n_features,) or old_b.shape != (n_features,):\n                continue\n            old_signs = np.sign(old_a - old_b)\n            old_onset = onset_chunk(old_signs)\n            if old_onset.size < 2:\n                continue\n            matches.append(float(\n                int(old_onset[0]) == first_index and\n                int(old_onset.size) == onset_length\n            ))\n        if matches:\n            recurrence = float(np.mean(matches))\n\n    coalition_threshold = float(parameters[\"coalition_threshold\"])\n    redundancy_slope = float(parameters[\"redundancy_slope\"])\n    extra_cue_penalty = float(parameters[\"extra_cue_penalty\"])\n    concentration_weight = float(parameters[\"concentration_weight\"])\n    fragmentation_penalty = float(parameters[\"fragmentation_penalty\"])\n    opponent_mass_weight = float(parameters[\"opponent_mass_weight\"])\n    recurrence_bonus = float(parameters[\"recurrence_bonus\"])\n\n    formation_logit = (\n        redundancy_slope * (float(onset_length) - coalition_threshold)\n        - extra_cue_penalty * extra_fraction\n        + concentration_weight * (opponent_concentration - 0.75)\n        - fragmentation_penalty * max(0.0, fragmentation - 0.35)\n        + opponent_mass_weight * (opponent_mass - 0.55)\n    )\n    if onset_length >= 2:\n        formation_logit += recurrence_bonus * recurrence\n\n    formation_logit = float(np.clip(formation_logit, -60.0, 60.0))\n    anchor_gate = 1.0 / (1.0 + np.exp(-formation_logit))\n\n    # Enforce the theoretical boundary: singleton priority may bias comparison\n    # but cannot itself become an anchor through opponent coherence or history.\n    if onset_length < 2:\n        anchor_gate = 0.0\n\n    if np.ptp(validities) > 0.0:\n        validity_score = (\n            validities - float(np.mean(validities))\n        ) / float(np.ptp(validities))\n    else:\n        validity_score = np.zeros(n_features, dtype=np.float64)\n\n    anchor_reliability = float(parameters[\"anchor_reliability\"])\n    validity_reinforcement = float(parameters[\"validity_reinforcement\"])\n    focal_validity = max(0.0, float(np.mean(validity_score[onset])))\n    anchor_reliability += validity_reinforcement * focal_validity\n    anchor_reliability = float(np.clip(anchor_reliability, 0.5, 0.98))\n\n    p_anchor = np.zeros(2, dtype=np.float64)\n    p_anchor[onset_winner] = anchor_reliability\n    p_anchor[1 - onset_winner] = 1.0 - anchor_reliability\n\n    # Weak comparison route. Position and validity modulate an evidence tally,\n    # while an independent modest onset bias represents ordinary scan primacy.\n    if n_features > 1:\n        position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)\n    else:\n        position = np.zeros(1, dtype=np.float64)\n\n    comparison_position_discount = float(\n        parameters[\"comparison_position_discount\"]\n    )\n    comparison_validity_weight = float(\n        parameters[\"comparison_validity_weight\"]\n    )\n    comparison_weights = np.exp(-comparison_position_discount * position)\n    comparison_weights *= 1.0 + comparison_validity_weight * validity_score\n    comparison_weights = np.clip(comparison_weights, 1e-8, None)\n\n    signed_tally = float(np.dot(comparison_weights, signs))\n    comparison_precision = float(parameters[\"comparison_precision\"])\n\n    # If a redundant candidate fails because it is not exclusive, people have\n    # already encoded several cues and consequently compare more precisely.\n    denied_coalition_precision = float(\n        parameters[\"denied_coalition_precision\"]\n    )\n    redundancy_readiness = 1.0 / (\n        1.0 + np.exp(-redundancy_slope * (float(onset_length) - coalition_threshold))\n    )\n    denied_strength = redundancy_readiness * extra_fraction\n    effective_precision = (\n        comparison_precision\n        + denied_coalition_precision * denied_strength\n    )\n\n    comparison_primacy = float(parameters[\"comparison_primacy\"])\n    signed_utility_a = (\n        effective_precision * signed_tally\n        + comparison_primacy * onset_direction\n    )\n\n    # In a small, completely occupied field, a coherent opposing run can be\n    # integrated as one perceptual object. Fragmentation sharply attenuates\n    # this effect. This is a capacity-based closure mechanism, not a k cutoff.\n    closure_capacity = float(parameters[\"closure_capacity\"])\n    closure_steepness = float(parameters[\"closure_steepness\"])\n    capacity_argument = closure_steepness * (\n        closure_capacity - float(n_features)\n    )\n    capacity_argument = float(np.clip(capacity_argument, -60.0, 60.0))\n    capacity_gate = 1.0 / (1.0 + np.exp(-capacity_argument))\n    saturation = float(k) / float(max(n_features, 1))\n\n    if k > 2:\n        excess_fragmentation = float(max(0, run_count - 2)) / float(k - 2)\n    else:\n        excess_fragmentation = 0.0\n    closure_fragmentation_power = float(\n        parameters[\"closure_fragmentation_power\"]\n    )\n    organization = (\n        capacity_gate\n        * (saturation ** 8.0)\n        * opponent_concentration\n        * ((1.0 - excess_fragmentation) ** closure_fragmentation_power)\n    )\n    closure_strength = float(parameters[\"closure_strength\"])\n    coherent_opposition = float(max(0, longest_opposite - 1))\n    signed_utility_a += (\n        closure_strength\n        * organization\n        * coherent_opposition\n        * (-onset_direction)\n    )\n\n    logits = np.array([signed_utility_a, 0.0], dtype=np.float64)\n    logits -= np.max(logits)\n    p_compare = np.exp(logits)\n    p_compare /= p_compare.sum()\n\n    p_core = anchor_gate * p_anchor + (1.0 - anchor_gate) * p_compare\n\n    epsilon = float(np.clip(parameters[\"epsilon\"], 0.0, 1.0))\n    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "coalition_threshold": "[1.65, 1.95]",
    "redundancy_slope": "[8.0, 11.0]",
    "extra_cue_penalty": "[13.0, 18.0]",
    "concentration_weight": "[1.0, 2.0]",
    "fragmentation_penalty": "[1.0, 2.0]",
    "opponent_mass_weight": "[1.0, 2.0]",
    "recurrence_bonus": "[0.00, 0.20]",
    "anchor_reliability": "[0.87, 0.93]",
    "validity_reinforcement": "[0.00, 0.025]",
    "comparison_primacy": "[0.60, 0.85]",
    "comparison_precision": "[0.02, 0.05]",
    "denied_coalition_precision": "[2.0, 3.0]",
    "comparison_position_discount": "[0.00, 0.06]",
    "comparison_validity_weight": "[0.00, 0.035]",
    "closure_capacity": "[6.2, 6.8]",
    "closure_steepness": "[4.0, 6.0]",
    "closure_strength": "[0.22, 0.34]",
    "closure_fragmentation_power": "[2.5, 3.5]",
    "epsilon": "[0.01, 0.04]"
  },
  "rationale": "The model makes onset redundancy a necessary condition for anchoring rather than treating the first discriminating cue as an anchor by default. A contiguous two-cue onset can cross the nonlinear, subject-specific formation threshold and therefore supports reliable onset-consistent choice in the exclusive 2-versus-majority displays of Experiments 8 and 10. The explicit exclusivity penalty prevents the same mechanism from anchoring Experiment 6 displays, where additional onset-direction evidence lies outside the initial chunk; those displays instead recruit a more precise majority comparison. Singleton displays, including Experiment 1 and the pairwise validity tests, receive only modest comparison primacy, so their choices remain noisy and only weakly sensitive to conflict magnitude or communicated validity. The capacity-limited closure term addresses Experiment 9 without incorrectly declaring its coherent singleton an anchor: in a small saturated field, the unanimous opposing run is integrated strongly, whereas fragmentation disrupts that integration and raises first-cue-consistent choice. Arbitration is consequently based on redundancy, exclusivity, concentration, saturation, field capacity, and fragmentation rather than a conflict-count threshold, allowing similarly fragmented k=7 and k=9 displays to behave alike. Validity effects are deliberately too weak to create late-cue promotion, and recurrence can only slightly reinforce a candidate that already contains at least two cues. Finally, variation in coalition threshold and anchor reliability directly represents the requested stable subject differences in chunk formation and anchor adherence."
}
```

## Usage

```json
{
  "input_tokens": 20466,
  "output_tokens": 8481
}
```
