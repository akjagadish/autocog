# generation_iter_02_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a new theory of Exchangeable Corroboration with Idiosyncratic Salience. The core claim should be that subjects strongly compress instructed validities and integrate all discriminating cues with diminishing marginal corroboration, but spatial configuration has no population-level directional gate. Cue order may generate stable subject-specific salience deviations whose population distribution is centered near zero; these deviations should not impose a common early, late, onset, closure, or run-direction preference. Include heterogeneous signed calibration of extra coalition members, allowing some subjects to treat a third cue as corroboration and others as redundancy. This explains the negative highest-validity/TTB agreement in Experiments 3-4 through coalition accumulation, the near-zero contrasts in Experiments 1-2 and 6-7 through exchangeability on average, and the high subject variance with only 44% favoring the early three-cue coalition in Experiment 8. The model should use compressed diagnosticity, saturating summed evidence, a subject-level corroboration/redundancy parameter, response sensitivity and lapse, plus zero-centered random cue-salience or broad attentional profiles. It should explicitly avoid pi_4's population-positive onset and closure gates and avoid pi_5's population-positive entry-versus-exit run asymmetry. Predictions for the two critical contrasts should be tightly centered near zero, while allowing substantial between-subject heterogeneity in overall three-versus-two preferences.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.621)

**Description**
Reliability-Compressed Parsing Mixtures proposes that communicated cue validities are strongly compressed and all discriminating cues enter a diminishing-returns accumulator. Stable subject-level styles allocate attention among reliability, coalition size, and locally parsed cue groups. Display position has no universal primacy or recency gradient: configuration effects arise from chunk binding and weak interpretation completion. In nearly balanced three-versus-two conflicts, parsing is sensitive to the internal ordering of compact and fragmented portions of a coalition; this translation-invariant run-asymmetry can reverse associations with recency-weighted predictions without assigning greater weight to later cues themselves. Balanced two-versus-two conflicts retain sign-varying parsing effects that cancel at the population level. Completion in conflicts between two coalitions of at least three cues is population-common but weak, graded, and heterogeneous rather than deterministic. Signed size calibration creates stable divisions between subjects who treat an additional cue as corroboration and those who treat it as redundant.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Compressed Parsing Mixtures expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    style = int(parameters["integration_style"])
    validity_compression = float(parameters["validity_compression"])
    coalition_saturation = float(parameters["coalition_saturation"])
    size_calibration = float(parameters["size_calibration"])
    parsing_strength = float(parameters["parsing_strength"])
    imbalanced_group_gain = float(parameters["imbalanced_group_gain"])
    completion_strength = float(parameters["completion_strength"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Reliability differences are represented but strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Stable styles differ in reliability attention, but even nominally
    # reliability-led subjects strongly compress instructed differences.
    if style <= 3:
        reliability_attention = 0.02 + 0.01 * style
    elif style <= 5:
        reliability_attention = 0.08 + 0.02 * (style - 4)
    else:
        reliability_attention = 0.04 + 0.015 * (style - 6)

    cue_weights = (
        (1.0 - reliability_attention) * np.ones(n_features, dtype=np.float64)
        + reliability_attention * diagnosticity
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    pos_idx = np.flatnonzero(differences > 0.0)
    neg_idx = np.flatnonzero(differences < 0.0)
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    if n_pos == 0 and n_neg == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def accumulated_evidence(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # All cues contribute, while total evidence grows sublinearly.
        return float(np.sum(cue_weights[indices])) / (float(n) ** coalition_saturation)

    evidence_a = accumulated_evidence(pos_idx)
    evidence_b = accumulated_evidence(neg_idx)

    # There is deliberately no fixed position gradient. A subject instead
    # parses the display into short local chunks whose width and phase are
    # stable properties of that subject's style.
    chunk_width = 2 + (style % 3)
    chunk_phase = style % chunk_width

    def parsing_quality(indices):
        n = int(indices.size)
        if n < 2:
            return 0.0
        idx = np.asarray(indices, dtype=int)
        adjacent = float(np.sum(np.diff(idx) == 1)) / float(n - 1)
        chunk_ids = np.floor_divide(idx + chunk_phase, chunk_width)
        same_chunk_pairs = 0.0
        total_pairs = float(n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                if chunk_ids[i] == chunk_ids[j]:
                    same_chunk_pairs += 1.0
        local_binding = same_chunk_pairs / max(total_pairs, 1.0)
        return 0.2 * adjacent + 0.8 * local_binding

    def centered_run_coherence(indices):
        # This descriptor depends only on within-coalition gap structure, not
        # absolute position. Positive values mean compact binding occurs near
        # the coalition's parsed entry; negative values mean it occurs near
        # its parsed exit. It is zero for singleton and two-cue coalitions.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 3:
            return 0.0
        gaps = np.diff(idx).astype(np.float64)
        binding = np.exp(-(gaps - 1.0))
        centered_order = np.linspace(1.0, -1.0, binding.size)
        norm = max(float(np.sum(np.abs(centered_order))), 1.0)
        return float(np.dot(centered_order, binding) / norm)

    # Multi-cue conflicts receive a weak, signed structural calibration.
    # Styles 0--3 regard larger coalitions as corroborative; styles 4--8
    # regard excess members as increasingly redundant. This creates a stable
    # population split in 3-versus-2 conflicts without changing singleton
    # comparisons or discarding any cue from the base accumulator.
    if n_pos >= 2 and n_neg >= 2:
        size_orientation = 1.0 if style <= 3 else -1.0
        log_size_ratio = np.log(float(n_pos) / float(n_neg))
        size_shift = size_orientation * size_calibration * log_size_ratio

        parsing_valences = np.array(
            [-1.0, 0.75, -0.50, 1.0, -0.75, 0.50, -1.0, 0.75, 0.25],
            dtype=np.float64,
        )
        parse_difference = parsing_quality(pos_idx) - parsing_quality(neg_idx)

        # Nearly balanced 3-versus-2 configurations use centered run
        # coherence. This distinguishes where compact binding occurs within a
        # coalition without imposing an absolute early/late accessibility
        # gradient. Balanced conflicts retain the centered style valences.
        if min(n_pos, n_neg) == 2 and abs(n_pos - n_neg) == 1:
            group_multipliers = np.array(
                [0.80, 1.10, 0.90, 1.20, 0.75, 1.05, 0.85, 1.15, 1.00],
                dtype=np.float64,
            )
            run_difference = (
                centered_run_coherence(pos_idx)
                - centered_run_coherence(neg_idx)
            )
            parsing_shift = (
                parsing_strength
                * imbalanced_group_gain
                * float(group_multipliers[style])
                * run_difference
            )
        else:
            parsing_shift = (
                parsing_strength
                * float(parsing_valences[style])
                * parse_difference
            )

        log_gate_a = 0.5 * (size_shift + parsing_shift)
        log_gate_b = -0.5 * (size_shift + parsing_shift)
        evidence_a *= np.exp(np.clip(log_gate_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_gate_b, -10.0, 10.0))

    net_a = evidence_a - evidence_b

    # Completion is weak and graded, is absent from singleton and 2-versus-2
    # conflicts, and never closes the accumulator. Stable differences in its
    # strength are smaller than the former opposing-sign split.
    if n_pos == n_neg and n_pos >= 3:
        completion_multipliers = np.array(
            [0.85, 0.925, 1.00, 1.075, 1.15, 0.95, 1.05, 0.90, 1.10],
            dtype=np.float64,
        )
        completion = completion_strength * float(completion_multipliers[style])
        if int(pos_idx[-1]) > int(neg_idx[-1]):
            net_a += completion
        elif int(neg_idx[-1]) > int(pos_idx[-1]):
            net_a -= completion

    logits = np.array([0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- integration_style: {0, 1, 2, 3, 4, 5, 6, 7, 8}
- validity_compression: [0.20, 0.42]
- coalition_saturation: [0.74, 0.86]
- size_calibration: [1.25, 1.65]
- parsing_strength: [0.12, 0.30]
- imbalanced_group_gain: [4.50, 6.50]
- completion_strength: [0.38, 0.50]
- beta: [0.95, 1.30]
- epsilon: [0.03, 0.09]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.1179 (var=0.0105)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.2217 (var=0.0089)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0237 (var=0.0003)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.4957 (var=0.0036)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5098 (var=0.0033)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.3200 (var=0.2176)
- Experiment 7: real=0.0219 (var=0.0094) vs this=0.2360 (var=0.0123)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0100 (var=0.0101)


---

### `pi_4` (overall score: 0.594)

**Description**
Switch-Closure Coalition Integration proposes that people encode communicated cue validities in a strongly compressed form and then organize simultaneously supporting cues into directional coalitions. Evidence contributed by a coalition grows sublinearly with its size, so each additional cue matters but has diminishing impact. Attention is configuration-dependent rather than governed by a universal positional gradient. A conflict between two singleton cues receives no positional weighting. When equally sized multi-cue coalitions compete, completing the later coalition produces a small switch-closure advantage because it is the most recently completed coherent interpretation. When coalition sizes differ, attention favors coalitions that begin early and remain locally coherent, but this structural gate is strongest for nearly balanced multi-cue conflicts such as three-versus-two and attenuated when a singleton competes with a growing coalition. Thus positional effects can reverse across configurations: terminal closure can favor the later side in balanced coalitions, early coherent organization can dominate nearly balanced multi-cue conflicts, and singleton-versus-coalition decisions remain governed primarily by gradual accumulation. Subject-specific compression, accumulation, attention, response sensitivity, and lapse parameters produce heterogeneity without trial-by-trial learning, which is appropriate because the task provides no outcome feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Switch-Closure Coalition Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    accumulation_saturation = float(parameters["accumulation_saturation"])
    balanced_terminal_attention = float(parameters["balanced_terminal_attention"])
    coalition_primacy = float(parameters["coalition_primacy"])
    coherence_gain = float(parameters["coherence_gain"])
    near_balance_gain = float(parameters["near_balance_gain"])
    singleton_gate_scale = float(parameters["singleton_gate_scale"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Compress communicated diagnosticities and mix them with a common
    # baseline. This preserves validity information without allowing one
    # instructed number to become lexicographically decisive.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)
    cue_weights = (1.0 - validity_reliance) + validity_reliance * diagnosticity
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    pos_idx = np.flatnonzero(differences > 0.0)
    neg_idx = np.flatnonzero(differences < 0.0)
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    if n_pos == 0 and n_neg == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_evidence(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Division by n**saturation gives sublinear accumulation:
        # total evidence grows as approximately n**(1-saturation).
        return float(np.sum(cue_weights[indices])) / (float(n) ** accumulation_saturation)

    def coalition_structure(indices):
        n = int(indices.size)
        if n < 2:
            return 0.0, 0.0
        scale = float(max(n_features - 1, 1))
        onset_primacy = 1.0 - float(indices[0]) / scale
        adjacent_links = float(np.sum(np.diff(indices) == 1))
        coherence = adjacent_links / float(n - 1)
        return onset_primacy, coherence

    evidence_a = coalition_evidence(pos_idx)
    evidence_b = coalition_evidence(neg_idx)
    log_gate_a = 0.0
    log_gate_b = 0.0

    if n_pos == 1 and n_neg == 1:
        # A pure singleton-versus-singleton conflict has no positional gate.
        pass
    elif n_pos == n_neg and n_pos >= 2:
        # Balanced multi-cue interpretations receive a bounded closure effect.
        # It depends on which coalition supplies the final piece of
        # discriminating evidence, not on a fixed weight for every position.
        last_pos = int(pos_idx[-1])
        last_neg = int(neg_idx[-1])
        half = 0.5 * balanced_terminal_attention
        if last_pos > last_neg:
            log_gate_a += half
            log_gate_b -= half
        elif last_neg > last_pos:
            log_gate_b += half
            log_gate_a -= half
    else:
        # In unequal conflicts, early and coherent coalitions are chunked and
        # maintained more effectively. The gate is amplified when both sides
        # form multi-cue, nearly balanced coalitions, but attenuated when one
        # side is a singleton so that growing opposition accumulates gradually.
        onset_a, coherence_a = coalition_structure(pos_idx)
        onset_b, coherence_b = coalition_structure(neg_idx)
        raw_a = coalition_primacy * onset_a + coherence_gain * coherence_a
        raw_b = coalition_primacy * onset_b + coherence_gain * coherence_b
        center = 0.5 * (raw_a + raw_b)
        log_gate_a = raw_a - center
        log_gate_b = raw_b - center

        if min(n_pos, n_neg) >= 2 and abs(n_pos - n_neg) == 1:
            log_gate_a *= near_balance_gain
            log_gate_b *= near_balance_gain
        elif min(n_pos, n_neg) == 1:
            log_gate_a *= singleton_gate_scale
            log_gate_b *= singleton_gate_scale

    evidence_a *= np.exp(np.clip(log_gate_a, -20.0, 20.0))
    evidence_b *= np.exp(np.clip(log_gate_b, -20.0, 20.0))
    net_a = evidence_a - evidence_b

    logits = np.array([0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- validity_compression: [0.20, 0.45]
- validity_reliance: [0.03, 0.08]
- accumulation_saturation: [0.72, 0.84]
- balanced_terminal_attention: [0.42, 0.56]
- coalition_primacy: [0.12, 0.25]
- coherence_gain: [0.03, 0.10]
- near_balance_gain: [4.75, 6.25]
- singleton_gate_scale: [0.20, 0.45]
- beta: [0.75, 1.05]
- epsilon: [0.03, 0.09]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.1246 (var=0.0114)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.2667 (var=0.0120)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0260 (var=0.0002)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.5061 (var=0.0029)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.3798 (var=0.0029)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.9800 (var=0.0196)
- Experiment 7: real=0.0219 (var=0.0094) vs this=0.0009 (var=0.0095)
- Experiment 8: real=0.0117 (var=0.0101) vs this=0.2729 (var=0.0146)


---

### `pi_2` (overall score: 0.538)

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
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.6354 (var=0.0266)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.0008 (var=0.0102)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0017 (var=0.0001)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.5017 (var=0.0041)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.4992 (var=0.0023)
- Experiment 6: real=0.4400 (var=0.2464) vs this=1.0000 (var=0.0000)
- Experiment 7: real=0.0219 (var=0.0094) vs this=-0.0035 (var=0.0050)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0042 (var=0.0038)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0966 -> ACCEPTED
- iter 2: loss=0.1173 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0966 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed endorsement of the highest-validity cue's winner."""
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # Feature 0 is the uniquely highest-validity cue in this design.
        if a[0] > b[0]:
            top_cue_winner = 0
        elif b[0] > a[0]:
            top_cue_winner = 1
        else:
            continue

        scores.append(1.0 if response == top_cue_winner else -1.0)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1217 (var=0.0117)
**Previous candidate values (this loop):**
  - iter 1: -0.1292 (var=0.0201) (Δ vs real -0.0075)
  - iter 2 (most recent): -0.0833 (var=0.0168) (Δ vs real +0.0383)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    if data is None or len(data) == 0:
        return 0.0

    signed_agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        j = int(differing[0])
        ttb_response = 0 if a[j] > b[j] else 1
        observed_response = int(row["response"])
        signed_agreements.append(1.0 if observed_response == ttb_response else -1.0)

    if not signed_agreements:
        return 0.0
    return float(np.mean(signed_agreements))
```

**Observed (real) value:** -0.2642 (var=0.0087)
**Previous candidate values (this loop):**
  - iter 1: 0.0262 (var=0.0100) (Δ vs real +0.2904)
  - iter 2 (most recent): 0.0213 (var=0.0088) (Δ vs real +0.2854)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)

### Experiment 3
**Design**
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    # Fixed, preregistered midpoint instantiation of the advocated model.
    validities = np.array([0.52, 0.58, 0.65, 0.72, 0.80, 0.88, 0.95], dtype=float)
    balance = 0.30
    recency = 2.10
    curvature = 0.825
    saturation = 0.725
    beta = 0.90
    epsilon = 0.05

    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity /= np.max(diagnosticity)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= np.mean(diagnosticity)

    position = np.arange(len(validities), dtype=float) / float(len(validities) - 1)
    accessibility = np.exp(recency * position)
    accessibility /= np.mean(accessibility)

    cue_weights = balance * diagnosticity + (1.0 - balance) * accessibility
    cue_weights /= np.mean(cue_weights)

    predicted_majority_probs = []
    chose_majority = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Orient every pair toward the option winning the 3-versus-2 tally.
        tally_margin = float(np.sum(d > 0) - np.sum(d < 0))
        if tally_margin == 0.0 or len(d) != len(cue_weights):
            continue
        majority_is_a = tally_margin > 0.0
        oriented = d if majority_is_a else -d

        positive = oriented > 0.0
        negative = oriented < 0.0
        n_pos = max(int(np.sum(positive)), 1)
        n_neg = max(int(np.sum(negative)), 1)
        min_n = min(n_pos, n_neg)

        pos_evidence = float(np.dot(cue_weights, np.maximum(oriented, 0.0)))
        neg_evidence = float(np.dot(cue_weights, np.maximum(-oriented, 0.0)))
        evidence = (
            pos_evidence / (max(1.0, float(n_pos) / float(min_n)) ** saturation)
            - neg_evidence / (max(1.0, float(n_neg) / float(min_n)) ** saturation)
        )

        core_p = 1.0 / (1.0 + np.exp(-np.clip(beta * evidence, -50.0, 50.0)))
        predicted_p = (1.0 - epsilon) * core_p + epsilon * 0.5
        predicted_majority_probs.append(predicted_p)

        response = int(row["response"])
        chose_majority.append(float((response == 0) if majority_is_a else (response == 1)))

    if len(predicted_majority_probs) == 0:
        return 0.0

    p = np.asarray(predicted_majority_probs, dtype=float)
    y = np.asarray(chose_majority, dtype=float)
    contrast = p - np.mean(p)
    return float(np.mean(contrast * (y - 0.5)))

```

**Observed (real) value:** -0.0410 (var=0.0003)
**Previous candidate values (this loop):**
  - iter 1: 0.0016 (var=0.0002) (Δ vs real +0.0426)
  - iter 2 (most recent): 0.0033 (var=0.0002) (Δ vs real +0.0443)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)

### Experiment 4
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or a.size < 2:
            continue

        differing = np.flatnonzero(a != b)
        if differing.size != 2:
            continue

        early = int(np.min(differing))
        late = int(np.max(differing))
        late_supports_a = bool(a[late] > b[late])
        later_option_response = 0 if late_supports_a else 1
        chose_later_supported_option = float(int(row["response"]) == later_option_response)

        # Approximate matched-filter weight for the competing theory's
        # exponential positional-accessibility contrast.
        scale = float(a.size - 1)
        weight = np.exp(2.1 * late / scale) - np.exp(2.1 * early / scale)
        if np.isfinite(weight) and weight > 0.0:
            weighted_choices.append(chose_later_supported_option)
            weights.append(weight)

    if len(weights) == 0:
        return float("nan")

    return float(np.average(np.asarray(weighted_choices), weights=np.asarray(weights)))
```

**Observed (real) value:** 0.5087 (var=0.0038)
**Previous candidate values (this loop):**
  - iter 1: 0.4862 (var=0.0059) (Δ vs real -0.0225)
  - iter 2 (most recent): 0.4993 (var=0.0051) (Δ vs real -0.0094)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.5

    a = np.vstack(data["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    responses = pd.to_numeric(data["response"], errors="coerce").to_numpy(dtype=float)

    n_features = a.shape[1]
    if n_features == 1:
        positional_weights = np.ones(1, dtype=float)
    else:
        positions = np.arange(n_features, dtype=float) / float(n_features - 1)
        positional_weights = np.exp(2.1 * positions)

    # Positive evidence means positional accessibility favors A; negative
    # evidence means it favors B.
    accessibility_difference = np.dot(a - b, positional_weights)
    valid = np.isfinite(responses) & (accessibility_difference != 0.0)
    if not np.any(valid):
        return 0.5

    chose_a = responses[valid] == 0.0
    accessibility_favors_a = accessibility_difference[valid] > 0.0
    accessibility_aligned = chose_a == accessibility_favors_a
    return float(np.mean(accessibility_aligned))
```

**Observed (real) value:** 0.5021 (var=0.0024)
**Previous candidate values (this loop):**
  - iter 1: 0.4981 (var=0.0047) (Δ vs real -0.0040)
  - iter 2 (most recent): 0.4948 (var=0.0028) (Δ vs real -0.0073)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    df = data[["subject_id", "option_a_ratings", "option_b_ratings", "response"]].copy()

    def early_coalition_choice(row):
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 3 or b.size < 3:
            return np.nan
        score_a = float(np.sum(a[:3]))
        score_b = float(np.sum(b[:3]))
        if score_a == score_b:
            return np.nan
        early_side = 0 if score_a > score_b else 1
        return float(int(row["response"]) == early_side)

    df["chose_early_coalition"] = df.apply(early_coalition_choice, axis=1)
    rates = df.groupby("subject_id", sort=False)["chose_early_coalition"].mean().dropna()
    if len(rates) == 0:
        return float("nan")

    # Fraction of subjects whose majority choice favors the fixed early
    # three-cue coalition rather than the later two-cue coalition.
    return float(np.mean(rates.to_numpy(dtype=float) > 0.5))
```

**Observed (real) value:** 0.4400 (var=0.2464)
**Previous candidate values (this loop):**
  - iter 1: 0.4200 (var=0.2436) (Δ vs real -0.0200)
  - iter 2 (most recent): 0.5400 (var=0.2484) (Δ vs real +0.1000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    entry_sum = 0.0
    entry_weight = 0.0
    exit_sum = 0.0
    exit_weight = 0.0

    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        diff = a - b
        pos = np.flatnonzero(diff > 0)
        neg = np.flatnonzero(diff < 0)

        # Identify the three-cue coalition and whether it supports A or B.
        if pos.size == 3 and neg.size == 2:
            coalition = pos
            endorsed_three = 1.0 if int(row.response) == 0 else 0.0
        elif neg.size == 3 and pos.size == 2:
            coalition = neg
            endorsed_three = 1.0 if int(row.response) == 1 else 0.0
        else:
            continue

        gaps = np.diff(coalition).astype(float)
        if gaps.size != 2:
            continue

        first_compact = bool(gaps[0] == 1 and gaps[1] > 1)
        last_compact = bool(gaps[1] == 1 and gaps[0] > 1)
        if not (first_compact or last_compact):
            continue

        # Larger separation makes entry- versus exit-compact organization
        # more distinct. This bounded weight is the difference between the
        # two adjacent-link binding strengths.
        weight = abs(np.exp(-(gaps[0] - 1.0)) - np.exp(-(gaps[1] - 1.0)))

        if first_compact:
            entry_sum += weight * endorsed_three
            entry_weight += weight
        else:
            exit_sum += weight * endorsed_three
            exit_weight += weight

    if entry_weight <= 0.0 or exit_weight <= 0.0:
        return 0.0

    return float(entry_sum / entry_weight - exit_sum / exit_weight)

```

**Observed (real) value:** 0.0219 (var=0.0094)
**Previous candidate values (this loop):**
  - iter 1: 0.0044 (var=0.0085) (Δ vs real -0.0174)
  - iter 2 (most recent): -0.0121 (var=0.0075) (Δ vs real -0.0340)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)

### Experiment 8
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []
    late_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        differences = a - b
        a_support = np.flatnonzero(differences > 0)
        b_support = np.flatnonzero(differences < 0)
        response = int(row["response"])

        if a_support.size == 3 and b_support.size == 2:
            onset_difference = int(a_support[0]) - int(b_support[0])
            chose_three = float(response == 0)
        elif a_support.size == 2 and b_support.size == 3:
            onset_difference = int(b_support[0]) - int(a_support[0])
            chose_three = float(response == 1)
        else:
            continue

        if onset_difference < 0:
            early_choices.append(chose_three)
        elif onset_difference > 0:
            late_choices.append(chose_three)

    if len(early_choices) == 0 or len(late_choices) == 0:
        return 0.0

    return float(np.mean(early_choices) - np.mean(late_choices))
```

**Observed (real) value:** 0.0117 (var=0.0101)
**Previous candidate values (this loop):**
  - iter 1: 0.0025 (var=0.0106) (Δ vs real -0.0092)
  - iter 2 (most recent): -0.0146 (var=0.0155) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Exchangeable Corroboration with Idiosyncratic Salience proposes that people retain the direction of every discriminating expert rating but strongly compress the instructed differences in expert validity. Supporting cues are integrated into option-level coalitions whose evidence increases sublinearly, so additional agreement provides corroboration with diminishing marginal impact. Subjects differ in how they calibrate coalition members beyond the first two: a signed, stable calibration parameter makes an additional cue corroborative for some people and partially redundant for others. Cue order has no population-level directional role. Instead, each subject has a stable, broad cue-salience profile drawn from a zero-centered exchangeable distribution and centered again within the display. Consequently, individual subjects can show substantial configuration preferences, but there is no common early, late, onset, closure, or entry-versus-exit advantage. Choices result from the difference between the two saturated coalition signals, transformed by response sensitivity and an independent lapse process. Because no outcome feedback is supplied, neither validity beliefs nor salience profiles are learned across trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Exchangeable Corroboration with Idiosyncratic Salience expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(parameters["cue_salience_profile"], dtype=np.float64)
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != n_features "
            f"{n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    corroboration_calibration = float(parameters["corroboration_calibration"])
    corroboration_strength = float(parameters["corroboration_strength"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Instructed reliability is represented on a diagnosticity scale, but
    # differences among experts are strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Salience is a stable subject-level profile, not a fixed position rule.
    # Centering removes a subject's global offset. Because profile components
    # have identical sampling distributions, no serial position has a
    # population-positive expected advantage.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attentional_weight = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attentional_weight /= max(float(np.mean(attentional_weight)), 1e-12)

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * diagnosticity
    )
    cue_weights = reliability_weight * attentional_weight
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)
    n_a = int(a_support.size)
    n_b = int(b_support.size)

    if n_a == 0 and n_b == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Every discriminating cue contributes. Division by n**saturation
        # makes total evidence grow approximately as n**(1-saturation),
        # implementing diminishing marginal corroboration without stopping.
        total = float(np.sum(cue_weights[indices]))
        return total / (float(n) ** coalition_saturation)

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)

    # Calibration of excess coalition membership is invoked only when both
    # options already have multi-cue interpretations. Thus it captures the
    # psychological status of a third or later coalition member rather than
    # introducing a general position gate. Positive values treat extra cues
    # as corroboration; negative values treat them as redundancy.
    if n_a >= 2 and n_b >= 2:
        extra_a = float(max(n_a - 2, 0))
        extra_b = float(max(n_b - 2, 0))
        log_adjustment_a = (
            corroboration_strength * corroboration_calibration * extra_a
        )
        log_adjustment_b = (
            corroboration_strength * corroboration_calibration * extra_b
        )
        evidence_a *= np.exp(np.clip(log_adjustment_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_adjustment_b, -10.0, 10.0))

    net_a = evidence_a - evidence_b
    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.34]
- validity_reliance: [0.02, 0.07]
- salience_strength: [0.18, 0.42]
- coalition_saturation: [0.76, 0.86]
- corroboration_calibration: [-1.30, 0.70]
- corroboration_strength: [0.40, 0.50]
- beta: [0.82, 1.12]
- epsilon: [0.03, 0.09]

`rationale`: The previous directional-gating account could fit some contrasts only by imposing common onset and closure advantages, producing severe overprediction of the early coalition in Experiment 6 and a large spurious onset contrast in Experiment 8. The new model contains no functions of first position, last position, run entry, run exit, or absolute serial direction. Its cue-salience components are exchangeably sampled and within-subject centered, so early-versus-late and entry-versus-exit contrasts are tightly centered near zero while individual salience profiles preserve realistic between-subject configuration variance. Strong validity compression and sublinear accumulation allow opposing cue coalitions to overcome a nominally best cue without discarding that cue. The signed calibration of members beyond the first two creates a stable population split in three-versus-two decisions: the selected range includes both corroboration and redundancy orientations and is mildly redundancy-heavy, counterbalancing the accumulator's intrinsic three-cue advantage and targeting roughly the observed 44% of subjects favoring the three-cue side in the critical early-three versus later-two comparison. Applying this calibration symmetrically to both options prevents it from becoming a disguised spatial gate. The model should therefore retain accumulation-based reversals while improving the near-zero positional contrasts and high subject heterogeneity that the previous population-positive gating mechanisms could not jointly explain.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed mechanism family and successfully removes the prohibited population-level positional gates. It fits Experiments 1, 4, 5, 6, 7, and 8 well: Experiment 1 has the correct highest-validity reversal (-0.129 versus -0.122), Experiment 6 closely reproduces both the 44% population split and its very high subject variance (0.420 versus 0.440), and the positional contrasts in Experiments 4, 5, 7, and 8 remain appropriately near zero. However, it misses two important systematic effects. In Experiment 2, predicted highest-validity-cue agreement is essentially zero and slightly positive (0.026), whereas humans strongly oppose that cue (-0.264). In Experiment 3, the candidate likewise produces essentially no covariance with the diagnostic contrast (0.002), rather than the observed negative value (-0.041). These are not merely variance discrepancies: the model has washed out structured, experiment-level coalition preferences. Its extremely low validity reliance (0.02–0.07), combined with strong compression and comparatively stronger random salience, makes equal-sized or configuration-varying coalitions nearly exchangeable even when their instructed-validity composition should systematically distinguish them.
Rationale: Keep the exchangeable corroboration architecture and the zero-centered salience profiles, but rebalance systematic diagnosticity against idiosyncratic salience. In particular, raise validity_reliance materially from its current near-zero range—while retaining compressed validity differences—and, if necessary, slightly reduce salience_strength so random profiles do not obscure the validity-composition signal. A moderate range such as roughly 0.10–0.25 for validity reliance is a plausible next search region; only modestly relax validity compression if that alone is insufficient. This should allow the combined lower-ranked coalition to systematically defeat the highest-validity cue in Experiment 2 and generate the negative item-level covariance in Experiment 3. Preserve the signed corroboration calibration, which is already producing the successful heterogeneous split in Experiment 6, and do not add onset, closure, recency, or run-direction gates. Retune cautiously because Experiment 1 is already excellent and excessive validity differentiation would reverse that gain.

**Outcome of this advice:** iter 1 candidate loss=0.0966 -> iter 2 candidate loss=0.1173 -> the gate marked it REJECTED.

### Iteration 2 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to Exchangeable Corroboration with Idiosyncratic Salience and correctly avoids population-level onset, closure, recency, and run-direction gates. However, this iteration was rejected by the accept gate and is worse overall than the accepted base. The increased validity reliance did not repair the principal failures: Experiment 2 remains near zero and positive (0.021 versus -0.264), and Experiment 3 remains near zero and positive (0.003 versus -0.041). Meanwhile, fit deteriorated in Experiment 1 (-0.083 versus -0.122), Experiment 6 (0.540 versus 0.440), and the near-zero contrasts of Experiments 7 and 8. The previous recommendation to raise validity reliance was therefore not productive. Mechanistically, that result is unsurprising: monotonic weighting of instructed validity tends to favor, not oppose, the highest-validity discriminating cue. Moreover, in Experiment 2 both options generally have equal-sized coalitions, so the current count-only saturation and third-member multiplier apply equally to both sides and cancel. Thus the model cannot express systematic differences in the corroborative quality or balance of equal-sized coalitions.
Rationale: Do not repeat the rejected increase in validity_reliance or further reduce salience. Build the next attempt on the accepted low-validity-reliance base and revise the in-family corroboration calculation instead. The third-member term should depend exchangeably on the member's diagnostic contribution or on coalition balance/dispersion, rather than only on n-2. For example, aggregate bounded cue contributions sequentially after sorting by contribution magnitude, or apply the signed corroboration/redundancy calibration to the weakest third-and-later contributions. This would distinguish two three-cue coalitions when one is dominated by a single strong cue and the other contains more mutually balanced support, without introducing any serial-position gate. Retain zero-centered exchangeable salience and the accepted signed-calibration range, which previously fit Experiment 6's 44% split and high variance. The target is a structural change that can make balanced lower-validity coalitions defeat a highest-cue-dominated coalition in Experiment 2 and induce the negative item covariance in Experiment 3, while preserving the accepted base's strong fits in Experiments 1 and 4-8.

**Outcome of this advice:** iter 2 candidate loss=0.1173 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Exchangeable Corroboration with Idiosyncratic Salience proposes that people retain the direction of every discriminating expert rating but strongly compress instructed differences in expert validity. Supporting cues are integrated into option-level coalitions whose evidence increases sublinearly, so additional agreement provides corroboration with diminishing marginal impact. Subjects differ in how they calibrate coalition members beyond the first two: a signed, stable calibration parameter makes an additional cue corroborative for some people and partially redundant for others. The effective calibration of an extra member also depends on its diagnostic balance with the coalition's two strongest members. A weak member added to a coalition dominated by stronger cues is more likely to be treated as redundant than an evenly matched third member. Cue order has no population-level directional role. Instead, each subject has a stable, broad cue-salience profile drawn from a zero-centered exchangeable distribution and centered again within the display. Consequently, individual subjects can show substantial configuration preferences, but there is no common early, late, onset, closure, or entry-versus-exit advantage. Choices result from the difference between the two saturated coalition signals, transformed by response sensitivity and an independent lapse process. Because no outcome feedback is supplied, neither validity beliefs nor salience profiles are learned across trials.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Exchangeable Corroboration with Idiosyncratic Salience expects \"\n            f\"shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    raw_salience = np.asarray(parameters[\"cue_salience_profile\"], dtype=np.float64)\n    if raw_salience.ndim != 1 or raw_salience.size != n_features:\n        raise ValueError(\n            f\"cue_salience_profile length {raw_salience.size} != n_features \"\n            f\"{n_features}.\"\n        )\n\n    validity_compression = float(parameters[\"validity_compression\"])\n    validity_reliance = float(parameters[\"validity_reliance\"])\n    salience_strength = float(parameters[\"salience_strength\"])\n    coalition_saturation = float(parameters[\"coalition_saturation\"])\n    corroboration_calibration = float(parameters[\"corroboration_calibration\"])\n    corroboration_strength = float(parameters[\"corroboration_strength\"])\n    balance_sensitivity = float(parameters[\"balance_sensitivity\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Instructed reliability is represented on a diagnosticity scale, but\n    # differences among experts are strongly compressed.\n    v = np.clip(validities, 0.500001, 0.999999)\n    diagnosticity = np.log(v / (1.0 - v))\n    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)\n    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)\n\n    # Salience is a stable subject-level profile, not a fixed position rule.\n    # Centering removes a subject's global offset. Because profile components\n    # have identical sampling distributions, no serial position has a\n    # population-positive expected advantage.\n    centered_salience = raw_salience - float(np.mean(raw_salience))\n    attentional_weight = np.exp(\n        np.clip(salience_strength * centered_salience, -10.0, 10.0)\n    )\n    attentional_weight /= max(float(np.mean(attentional_weight)), 1e-12)\n\n    reliability_weight = (\n        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)\n        + validity_reliance * diagnosticity\n    )\n    cue_weights = reliability_weight * attentional_weight\n    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)\n\n    differences = a - b\n    a_support = np.flatnonzero(differences > 0.0)\n    b_support = np.flatnonzero(differences < 0.0)\n    n_a = int(a_support.size)\n    n_b = int(b_support.size)\n\n    if n_a == 0 and n_b == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def coalition_signal(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n        # Every discriminating cue contributes. Division by n**saturation\n        # makes total evidence grow approximately as n**(1-saturation),\n        # implementing diminishing marginal corroboration without stopping.\n        total = float(np.sum(cue_weights[indices]))\n        return total / (float(n) ** coalition_saturation)\n\n    def effective_extra_members(indices):\n        n = int(indices.size)\n        if n <= 2:\n            return 0.0\n        # Rank only by diagnostic contribution, not by display position. The\n        # ordinary n-2 term preserves the accepted heterogeneous size effect.\n        # A bounded imbalance increment makes a weak third-or-later member\n        # count as more redundant when the first two members dominate it.\n        contributions = np.sort(diagnosticity[indices])[::-1]\n        anchor = max(float(np.mean(contributions[:2])), 1e-12)\n        extras = contributions[2:]\n        imbalance = np.sum(\n            np.maximum(anchor - extras, 0.0) / np.maximum(anchor + extras, 1e-12)\n        )\n        return float(n - 2) + balance_sensitivity * float(imbalance)\n\n    evidence_a = coalition_signal(a_support)\n    evidence_b = coalition_signal(b_support)\n\n    # Calibration of excess coalition membership is invoked only when both\n    # options already have multi-cue interpretations. Thus it captures the\n    # psychological status of a third or later coalition member rather than\n    # introducing a general position gate. Positive values treat extra cues\n    # as corroboration; negative values treat them as redundancy. Effective\n    # membership also reflects whether extras are balanced with the two\n    # strongest diagnostic contributors, distinguishing equal-sized\n    # coalitions without using their spatial configuration.\n    if n_a >= 2 and n_b >= 2:\n        extra_a = effective_extra_members(a_support)\n        extra_b = effective_extra_members(b_support)\n        log_adjustment_a = (\n            corroboration_strength * corroboration_calibration * extra_a\n        )\n        log_adjustment_b = (\n            corroboration_strength * corroboration_calibration * extra_b\n        )\n        evidence_a *= np.exp(np.clip(log_adjustment_a, -10.0, 10.0))\n        evidence_b *= np.exp(np.clip(log_adjustment_b, -10.0, 10.0))\n\n    net_a = evidence_a - evidence_b\n    logits = np.array(\n        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64\n    )\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "cue_salience_profile": "[(-1.0, 1.0)] * n_features",
    "validity_compression": "[0.16, 0.34]",
    "validity_reliance": "[0.02, 0.07]",
    "salience_strength": "[0.18, 0.42]",
    "coalition_saturation": "[0.76, 0.86]",
    "corroboration_calibration": "[-1.30, 0.70]",
    "corroboration_strength": "[0.40, 0.50]",
    "balance_sensitivity": "[1.50, 2.50]",
    "beta": "[0.82, 1.12]",
    "epsilon": "[0.03, 0.09]"
  },
  "rationale": "This is a minimal structural edit to the accepted candidate. The rejected strategy of increasing monotonic validity reliance is not repeated, and all successful ranges governing salience, saturation, signed size calibration, response sensitivity, and lapse are retained. The sole mechanistic addition replaces the count-only definition of extra coalition membership with an effective-membership calculation. It preserves n-2 but adds a bounded, order-free measure of how weak each third-or-later diagnostic contribution is relative to the coalition's two strongest contributions. Consequently, equal-sized coalitions no longer receive identical corroboration adjustments: a coalition dominated by one or two strong cues can be treated as less coherent than a coalition of mutually balanced contributors. Because the accepted calibration distribution is mildly redundancy-oriented, highest-cue-dominated coalitions tend to receive a larger penalty, providing an in-family route to the negative Experiment 2 agreement and Experiment 3 covariance. Positive-calibration subjects can exhibit the opposite interpretation, preserving heterogeneous three-versus-two preferences. Ranking occurs by diagnostic contribution rather than cue location, so the edit introduces no onset, closure, recency, or entry-versus-exit gate. The original count component remains intact to protect the accepted fit to Experiment 6, while the bounded imbalance increment limits collateral changes to Experiments 1 and 4-8."
}
```

## Usage

```json
{
  "input_tokens": 21938,
  "output_tokens": 4234
}
```
