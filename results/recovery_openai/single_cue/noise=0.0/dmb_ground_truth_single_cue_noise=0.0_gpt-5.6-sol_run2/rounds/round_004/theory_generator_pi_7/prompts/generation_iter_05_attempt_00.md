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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_5`).

Replace pi_5 with a new order-invariant theory of contextual consensus and inferred expert dependence. The theory should propose that subjects strongly compress advertised validity, integrate all discriminating cues with diminishing returns, and treat agreement among experts with similar or extreme advertised reliabilities as evidence that may arise from a shared or redundant informational source. Coalition value should therefore depend on effective informational diversity rather than raw cue count or spatial arrangement: additional heterogeneous cues provide corroboration, while clusters of similarly high-validity cues can be sharply discounted. Stable subject-level priors over expert dependence and over the count-versus-reliability tradeoff should generate the extra heterogeneity seen in Experiment 2 and the subject split in Experiment 8. Unlike pi_5, the new theory must exclude population-common closure, onset, recency, and entry-versus-exit effects, thereby predicting the null results in Experiments 1, 6, 7, 9, and 10. Unlike pi_6, it should derive composition discounting from latent-source or similarity-based dependence inference rather than an unconstrained pair-product penalty, and it should allow nonlinear skepticism toward coalitions of equally extreme advertised validity. This should strengthen the negative effect in Experiment 2 while preserving majority-over-singleton behavior in Experiment 3 and negative agreement with highest-validity-cue heuristics in Experiment 4. The resulting competitor should make new predictions for coalitions that independently manipulate reliability similarity, reliability extremity, and coalition size while holding order fixed.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.978)

**Description**
Exchangeable Corroboration with Idiosyncratic Salience proposes that people retain the direction of every discriminating expert rating but strongly compress instructed differences in expert validity. Supporting cues are integrated into option-level coalitions whose evidence increases sublinearly, so additional agreement provides corroboration with diminishing marginal impact. Subjects differ in how they calibrate coalition members beyond the first two: a signed, stable calibration parameter makes an additional cue corroborative for some people and partially redundant for others. Separately, jointly diagnostic cues can be perceived as partially redundant: an order-free composition term discounts coalitions containing several simultaneously strong cues, with its effect centered between the competing coalitions. This composition discount is strongest for equal-sized coalitions but remains moderately active when coalition sizes differ. Cue order has no population-level directional role. Instead, each subject has a stable, broad cue-salience profile drawn from a zero-centered exchangeable distribution and centered again within the display. Consequently, individual subjects can show substantial configuration preferences, but there is no common early, late, onset, closure, or entry-versus-exit advantage. Choices result from the difference between the two saturated coalition signals, transformed by response sensitivity and an independent lapse process. Because no outcome feedback is supplied, neither validity beliefs nor salience profiles are learned across trials.

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
    composition_redundancy = float(parameters["composition_redundancy"])
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

    def composition_load(indices):
        # Mean pairwise joint diagnostic strength is a bounded, order-free
        # measure of potential redundancy. It is high when several members
        # are all strongly diagnostic, not when a coalition merely appears
        # early or late in the display.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 2:
            return 0.0
        bounded_strength = np.clip(2.0 * v[idx] - 1.0, 0.0, 1.0)
        pair_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_sum += float(bounded_strength[i] * bounded_strength[j])
                pair_count += 1
        return pair_sum / float(max(pair_count, 1))

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

        # Decouple composition from the heterogeneous size calibration.
        # Coalitions containing several jointly strong cues receive a mild
        # redundancy discount. Centering the adjustment between options
        # prevents a global response bias. Composition is most diagnostic
        # when coalition sizes are equal and remains moderately active in
        # size conflicts without becoming a spatial or directional gate.
        load_difference = composition_load(a_support) - composition_load(b_support)
        composition_scale = 1.0 if n_a == n_b else 0.60
        composition_shift = (
            composition_redundancy * composition_scale * load_difference
        )
        evidence_a *= np.exp(np.clip(-0.5 * composition_shift, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(0.5 * composition_shift, -10.0, 10.0))

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
- composition_redundancy: [2.00, 2.80]
- beta: [0.82, 1.12]
- epsilon: [0.03, 0.09]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.0779 (var=0.0176)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.2321 (var=0.0159)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0366 (var=0.0003)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.5046 (var=0.0065)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5123 (var=0.0025)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.3600 (var=0.2304)
- Experiment 7: real=0.0219 (var=0.0094) vs this=0.0257 (var=0.0065)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0154 (var=0.0165)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4996 (var=0.0028)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.2057 (var=0.0035)


---

### `pi_2` (overall score: 0.548)

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
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4985 (var=0.0025)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.4978 (var=0.0039)


---

### `pi_5` (overall score: 0.515)

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
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.6108 (var=0.0033)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.5023 (var=0.0031)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1222 -> ACCEPTED
- iter 2: loss=0.1134 -> ACCEPTED
- iter 3: loss=0.0930 -> ACCEPTED
- iter 4: loss=0.0958 -> REJECTED
- iter 5: loss=0.1045 -> REJECTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.0930 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.0308 (var=0.0347) (Δ vs real +0.0908)
  - iter 2: -0.2712 (var=0.0158) (Δ vs real -0.1496)
  - iter 3: -0.0204 (var=0.0248) (Δ vs real +0.1013)
  - iter 4: -0.0104 (var=0.0232) (Δ vs real +0.1113)
  - iter 5 (most recent): -0.0408 (var=0.0188) (Δ vs real +0.0808)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)
- pi_6: -0.0779 (var=0.0176)

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
  - iter 1: -0.0175 (var=0.0093) (Δ vs real +0.2467)
  - iter 2: -0.0046 (var=0.0125) (Δ vs real +0.2596)
  - iter 3: -0.0379 (var=0.0090) (Δ vs real +0.2263)
  - iter 4: -0.0525 (var=0.0098) (Δ vs real +0.2117)
  - iter 5 (most recent): -0.0392 (var=0.0094) (Δ vs real +0.2250)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)
- pi_6: -0.2321 (var=0.0159)

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
  - iter 1: -0.0104 (var=0.0002) (Δ vs real +0.0306)
  - iter 2: -0.0007 (var=0.0003) (Δ vs real +0.0403)
  - iter 3: -0.0119 (var=0.0002) (Δ vs real +0.0291)
  - iter 4: -0.0164 (var=0.0002) (Δ vs real +0.0246)
  - iter 5 (most recent): -0.0115 (var=0.0002) (Δ vs real +0.0295)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)
- pi_6: -0.0366 (var=0.0003)

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
  - iter 1: 0.4805 (var=0.0034) (Δ vs real -0.0281)
  - iter 2: 0.4940 (var=0.0035) (Δ vs real -0.0147)
  - iter 3: 0.5021 (var=0.0046) (Δ vs real -0.0065)
  - iter 4: 0.4773 (var=0.0048) (Δ vs real -0.0314)
  - iter 5 (most recent): 0.5038 (var=0.0033) (Δ vs real -0.0049)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)
- pi_6: 0.5046 (var=0.0065)

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
  - iter 1: 0.4963 (var=0.0022) (Δ vs real -0.0058)
  - iter 2: 0.5073 (var=0.0024) (Δ vs real +0.0052)
  - iter 3: 0.4908 (var=0.0024) (Δ vs real -0.0112)
  - iter 4: 0.4988 (var=0.0025) (Δ vs real -0.0033)
  - iter 5 (most recent): 0.4992 (var=0.0029) (Δ vs real -0.0029)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)
- pi_6: 0.5123 (var=0.0025)

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
  - iter 1: 0.2400 (var=0.1824) (Δ vs real -0.2000)
  - iter 2: 0.2800 (var=0.2016) (Δ vs real -0.1600)
  - iter 3: 0.3400 (var=0.2244) (Δ vs real -0.1000)
  - iter 4: 0.2200 (var=0.1716) (Δ vs real -0.2200)
  - iter 5 (most recent): 0.2400 (var=0.1824) (Δ vs real -0.2000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)
- pi_6: 0.3600 (var=0.2304)

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
  - iter 1: 0.0261 (var=0.0098) (Δ vs real +0.0042)
  - iter 2: -0.0025 (var=0.0088) (Δ vs real -0.0244)
  - iter 3: 0.0257 (var=0.0136) (Δ vs real +0.0038)
  - iter 4: 0.0119 (var=0.0086) (Δ vs real -0.0099)
  - iter 5 (most recent): -0.0240 (var=0.0082) (Δ vs real -0.0459)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)
- pi_6: 0.0257 (var=0.0065)

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
  - iter 1: -0.0179 (var=0.0103) (Δ vs real -0.0296)
  - iter 2: 0.0000 (var=0.0125) (Δ vs real -0.0117)
  - iter 3: -0.0017 (var=0.0075) (Δ vs real -0.0133)
  - iter 4: 0.0100 (var=0.0092) (Δ vs real -0.0017)
  - iter 5 (most recent): 0.0025 (var=0.0088) (Δ vs real -0.0092)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)
- pi_6: -0.0154 (var=0.0165)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    closure_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        difference = a - b
        discriminating = np.flatnonzero(difference != 0.0)

        if discriminating.size == 0:
            continue

        # The closure coalition is the option supported by the final
        # discriminating cue, irrespective of whether it is displayed as A or B.
        last_index = int(discriminating[-1])
        closure_response = 0 if difference[last_index] > 0.0 else 1
        closure_choices.append(float(int(row["response"]) == closure_response))

    if len(closure_choices) == 0:
        return float("nan")

    return float(np.mean(np.asarray(closure_choices, dtype=np.float64)))
```

**Observed (real) value:** 0.4958 (var=0.0041)
**Previous candidate values (this loop):**
  - iter 1: 0.4994 (var=0.0024) (Δ vs real +0.0035)
  - iter 2: 0.4992 (var=0.0021) (Δ vs real +0.0033)
  - iter 3: 0.5125 (var=0.0033) (Δ vs real +0.0167)
  - iter 4: 0.5054 (var=0.0032) (Δ vs real +0.0096)
  - iter 5 (most recent): 0.5002 (var=0.0012) (Δ vs real +0.0044)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6108 (var=0.0033)
- pi_6: 0.4996 (var=0.0028)
- pi_1: 0.4104 (var=0.0015)
- pi_2: 0.4985 (var=0.0025)
- pi_3: 0.6115 (var=0.0020)
- pi_4: 0.6271 (var=0.0029)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.55, 0.70,
        0.68, 0.74, 0.80, 0.86, 0.92, 0.97, 0.60, 0.70
    ], dtype=np.float64)
    bounded = np.clip(2.0 * validities - 1.0, 0.0, 1.0)

    weighted_choices = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        if a.size != validities.size or b.size != validities.size:
            continue

        differences = a - b
        a_support = np.flatnonzero(differences > 0.0)
        b_support = np.flatnonzero(differences < 0.0)
        if a_support.size != 2 or b_support.size != 2:
            continue

        a_equal = bool(np.isclose(validities[a_support[0]], validities[a_support[1]]))
        b_equal = bool(np.isclose(validities[b_support[0]], validities[b_support[1]]))
        if a_equal == b_equal:
            continue

        load_a = float(bounded[a_support[0]] * bounded[a_support[1]])
        load_b = float(bounded[b_support[0]] * bounded[b_support[1]])
        weight = abs(load_a - load_b)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        response = int(row["response"])
        chose_equal = float((response == 0) if a_equal else (response == 1))
        weighted_choices.append(weight * chose_equal)
        weights.append(weight)

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return 0.5
    return float(np.sum(weighted_choices) / total_weight)

```

**Observed (real) value:** 0.1723 (var=0.0081)
**Previous candidate values (this loop):**
  - iter 1: 0.4150 (var=0.0024) (Δ vs real +0.2427)
  - iter 2: 0.2683 (var=0.0034) (Δ vs real +0.0960)
  - iter 3: 0.2865 (var=0.0031) (Δ vs real +0.1142)
  - iter 4: 0.2419 (var=0.0030) (Δ vs real +0.0696)
  - iter 5 (most recent): 0.2571 (var=0.0020) (Δ vs real +0.0848)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2057 (var=0.0035)
- pi_5: 0.5023 (var=0.0031)
- pi_1: 0.8796 (var=0.0067)
- pi_2: 0.4978 (var=0.0039)
- pi_3: 0.5414 (var=0.0029)
- pi_4: 0.3753 (var=0.0039)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Reliability-Schema Source Inference proposes that decision makers treat expert agreement as evidence generated by latent informational sources rather than as conditionally independent votes. Advertised validities identify experts' reliability schemas. Dependence inference operates at two scales: an extremely narrow component assigns nearly identical, jointly extreme experts a high probability of sharing a source, while a weaker broad component captures partial dependence among merely similar reliability schemas. Thus equally extreme coalitions can be discounted sharply without treating moderately heterogeneous coalitions as fully independent. Every discriminating cue contributes, but its contribution is divided by inferred latent-source overlap. Coalition evidence has diminishing returns, and nominal count corroboration is capped after three members. The transition from two to three members receives an additional corroboration gain only to the extent that the coalition represents informationally diverse sources. Stable subject-level priors over source dependence and count-versus-independence tradeoffs produce heterogeneous three-versus-two judgments without rewarding arbitrarily large coalitions. Communicated reliability differences are strongly compressed, preventing a single advertised-best expert from dominating a diverse majority. The computation is exchangeable over expert identities and contains no onset, closure, recency, adjacency, entry, exit, or other spatial parsing mechanism.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Schema Source Inference expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_prior = float(parameters["dependence_prior"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    shoulder_strength = float(parameters["shoulder_strength"])
    shoulder_bandwidth = float(parameters["shoulder_bandwidth"])
    extremity_threshold = float(parameters["extremity_threshold"])
    extremity_slope = float(parameters["extremity_slope"])
    overlap_curvature = float(parameters["overlap_curvature"])
    count_independence_tradeoff = float(
        parameters["count_independence_tradeoff"]
    )
    diverse_third_gain = float(parameters["diverse_third_gain"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Advertised reliability is represented, but ordinal differences are
    # strongly compressed before use. The mean normalization keeps overall
    # evidence scale comparable across experiments with different validity
    # schedules.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    compressed = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    compressed /= max(float(np.mean(compressed)), 1e-12)
    cue_weights = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * compressed
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coalition_v = v[idx]
        overlap = np.zeros(n, dtype=np.float64)
        pair_posteriors = []

        # The narrow component detects almost identical reliability schemas.
        # A weaker, moderate-width shoulder represents partial common-source
        # inference for nonidentical but similar experts.
        for i in range(n):
            for j in range(i + 1, n):
                reliability_gap = float(coalition_v[i] - coalition_v[j])
                narrow_similarity = np.exp(
                    -0.5 * (reliability_gap / similarity_bandwidth) ** 2
                )
                shoulder_similarity = np.exp(
                    -0.5 * (reliability_gap / shoulder_bandwidth) ** 2
                )
                pair_extremity = 0.5 * float(coalition_v[i] + coalition_v[j])
                extreme_gate = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            extremity_slope
                            * (pair_extremity - extremity_threshold),
                            -50.0,
                            50.0,
                        )
                    )
                )

                narrow_affinity = (
                    dependence_prior
                    * narrow_similarity
                    * (0.01 + 0.99 * extreme_gate)
                )
                shoulder_affinity = (
                    shoulder_strength
                    * shoulder_similarity
                    * (0.25 + 0.75 * extreme_gate)
                )
                source_affinity = narrow_affinity + shoulder_affinity
                overlap[i] += source_affinity
                overlap[j] += source_affinity
                pair_posteriors.append(
                    source_affinity / (1.0 + source_affinity)
                )

        # Shared-source overlap is converted superlinearly into redundancy.
        # This permits sharp skepticism toward exactly equal extreme pairs
        # while preserving useful evidence from heterogeneous additions.
        independent_contributions = cue_weights[idx] / np.power(
            1.0 + overlap, overlap_curvature
        )
        independent_total = float(np.sum(independent_contributions))

        signal = independent_total / (float(n) ** coalition_saturation)

        # Nominal count corroboration is capped after the third member rather
        # than growing exponentially throughout a large coalition.
        capped_count_steps = float(min(n - 1, 2))
        signal *= np.exp(
            np.clip(
                count_independence_tradeoff * capped_count_steps,
                -10.0,
                10.0,
            )
        )

        # Crossing from two to at least three cues provides corroboration only
        # when those cues are inferred to represent informationally diverse
        # sources. The bonus is then capped, so fourth and later cues do not
        # receive repeated generic count rewards.
        if n >= 3:
            mean_shared_source = float(np.mean(pair_posteriors))
            effective_diversity = np.clip(
                1.0 - mean_shared_source, 0.0, 1.0
            )
            signal *= np.exp(
                np.clip(diverse_third_gain * effective_diversity, -10.0, 10.0)
            )

        return float(signal)

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
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
def policy(probs) -> int:
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
- validity_compression: [0.14, 0.30]
- validity_reliance: [0.01, 0.06]
- coalition_saturation: [0.62, 0.72]
- dependence_prior: [1.80, 4.20]
- similarity_bandwidth: [0.010, 0.022]
- shoulder_strength: [0.20, 0.65]
- shoulder_bandwidth: [0.055, 0.120]
- extremity_threshold: [0.70, 0.78]
- extremity_slope: [14.0, 24.0]
- overlap_curvature: [1.80, 3.00]
- count_independence_tradeoff: [-0.08, 0.16]
- diverse_third_gain: [0.08, 0.28]
- beta: [0.88, 1.18]
- epsilon: [0.03, 0.09]

`rationale`: This is a minimal extension of the accepted latent-source model. The original narrow similarity kernel is retained and modestly strengthened, with wider subject-level dependence and curvature ranges to sharpen and diversify the equal-extreme discount needed in Experiment 10. A weaker moderate-width shoulder is added to infer partial dependence among the nonidentical reliability schemas driving Experiments 2 and 3, instead of broadening the narrow kernel and thereby losing the exact-equality contrast. The former linear count reward is capped after two count transitions, reducing excessive support for four- through six-cue coalitions in Experiment 1. A single diversity-gated corroboration increment is added when a coalition reaches three members, targeting the heterogeneous three-versus-two advantage in Experiments 3 and 6 without repeatedly rewarding larger raw cue counts. All new composition effects remain derived from inferred shared-source probabilities, and the model remains completely order-invariant.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed order-invariant latent-source family and successfully removes the unwanted spatial signatures. Experiments 5, 7, and 9 are fitted closely, and Experiments 4 and 8 remain reasonably near their intended nulls. However, the central contextual-consensus effects are much too weak. Agreement with the highest-validity cue is insufficiently negative in Experiment 1 (-0.0308 versus -0.1217) and especially Experiment 2 (-0.0175 versus -0.2642). The composition contrast in Experiment 3 is likewise attenuated (-0.0104 versus -0.0410). Most decisively, Experiment 10 shows that equal-reliability coalitions require a far sharper discount: the model predicts 0.4150 endorsement versus 0.1723 observed. Experiment 6 also underpredicts the fraction favoring the three-cue coalition (0.24 versus 0.44), indicating that generic count corroboration is too weak even while similarity-specific redundancy is too weak. Thus the current implementation compresses both desired forces toward indifference rather than separating heterogeneous count benefits from homogeneous-source discounting.
Rationale: Retain the latent-source, exchangeable mechanism, but recalibrate how inferred dependence affects coalition value. In Experiment 10, the unequal comparison pair differs by only about 0.05 in advertised validity; the current similarity bandwidth of 0.035-0.09 often assigns that pair substantial source affinity, so it is not distinguished sharply enough from an exactly equal pair. Try a narrower bandwidth range and a stronger, nonlinear conversion from near-unit affinity to redundancy—such as a superlinear overlap denominator or coalition-level posterior probability of a shared source—rather than merely increasing a generic pair penalty. Increase or widen dependence strength and extremity sensitivity so exactly equal, high-reliability coalitions can be discounted much more strongly and with greater subject heterogeneity; the candidate's Experiment 10 variance is also too low (0.0024 versus 0.0081). At the same time, shift the count-versus-independence prior modestly toward corroboration, or reduce generic saturation, so heterogeneous three-cue coalitions recover the majority advantage needed in Experiments 1, 3, and 6. These two adjustments should be selective: stronger count support for reliability-diverse additions, but sharply stronger redundancy only for highly similar or equally extreme members. Avoid increasing validity reliance or introducing positional terms, since that would threaten the successful null fits and does not address the decisive equal-versus-unequal composition failure.

**Outcome of this advice:** iter 1 candidate loss=0.1222 -> iter 2 candidate loss=0.1134 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed exchangeable latent-source family and the accepted recalibration improved aggregate loss, especially by moving Experiment 10 from 0.4150 to 0.2683 while preserving the positional nulls. Experiments 4, 5, 8, and 9 are now close, and Experiment 7 remains broadly consistent with a null despite simulation noise. Nevertheless, the model still misses the central cross-experiment consensus pattern. Experiment 2 is the clearest failure: predicted agreement with the highest-validity discriminating cue is essentially zero (-0.0046) rather than strongly negative (-0.2642), and it worsened slightly despite the overall accepted improvement. Experiment 3's composition-sensitive contrast has also collapsed toward zero (-0.0007 versus -0.0410). Meanwhile, the count recalibration overshot Experiment 1 in the opposite direction: endorsement of the top cue changed from too weakly negative to far too negative (-0.2712 versus -0.1217). Experiment 6 still has too few subjects favoring the three-cue coalition (0.28 versus 0.44). Experiment 10 remains materially underfit (0.2683 versus 0.1723), with insufficient between-subject variance. Thus the successful exact/equal-reliability discount is still too weak, while the ultra-narrow similarity kernel makes dependence inference nearly absent for the nonidentical reliability compositions driving Experiments 2 and 3. Generic count corroboration is also poorly shaped: it overvalues larger coalitions in Experiment 1 without providing enough of the localized three-versus-two advantage in Experiment 6.
Rationale: Build on the accepted narrow-kernel latent-source model, but do not further increase the existing global count_independence_tradeoff: that accepted change helped Experiment 10 but contributed to the Experiment 1 overshoot and did not solve Experiment 2. Instead, use a two-scale or spike-plus-shoulder source-affinity function within the same inferred-dependence mechanism: retain a very narrow, strong component for exactly equal and jointly extreme reliabilities, while adding a weaker moderate-width similarity component for merely similar schemas. The narrow component should be strengthened cautiously to move Experiment 10 from 0.268 toward 0.172 and its subject-level strength should vary more widely to raise variance. The weak shoulder is needed to distinguish the mixed coalition compositions in Experiments 2 and 3, which the current 0.012-0.025 bandwidth effectively treats as independent. Re-shape generic diminishing returns rather than applying a linear exponential reward per additional cue: allow a specific corroboration gain from two to three effectively diverse sources, then sharply taper gains beyond three. A capped or concave count term can raise Experiment 6's three-versus-two split while reducing the excessive support for four- through six-cue coalitions that drives Experiment 1 too negative. Keep validity reliance low and retain complete order invariance; the successful null results provide no justification for positional terms.

**Outcome of this advice:** iter 2 candidate loss=0.1134 -> iter 3 candidate loss=0.0930 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed order-invariant latent-source family, and the accepted two-scale/capped-count revision improved aggregate loss substantially. It preserves the important spatial nulls: Experiments 4, 5, 7, 8, and 9 are all close to observed values. It also improved Experiment 6 from 0.28 to 0.34 and corrected the previous Experiment 1 overshoot. However, the central composition effects remain materially underfit. Experiment 2 is the largest failure: agreement with the highest-validity cue is only -0.0379 versus -0.2642 observed, so inferred source dependence still does not rank the competing three-cue compositions strongly enough. Experiment 3 is similarly attenuated (-0.0119 versus -0.0410). Experiment 10 remains too close to indifference (0.2865 versus 0.1723), and its between-subject variance is much too low (0.0031 versus 0.0081), indicating insufficiently strong and heterogeneous skepticism toward exactly equal extreme pairs. Experiment 1 has now moved from the prior overshoot to insufficient majority support (-0.0204 versus -0.1217), while Experiment 6 still underpredicts the three-over-two subject fraction (0.34 versus 0.44). Thus the accepted structural direction is productive, but the model still compresses the heterogeneous-coalition benefit and exact-equality discount too strongly toward zero.
Rationale: Retain the accepted spike-plus-shoulder latent-source mechanism and capped count term; the previous advice in that direction was accepted and should not be reversed. The next revision should make the two dependence scales more selective. In particular, the broad shoulder currently operates even for two-member coalitions, likely discounting the unequal comparison pair in Experiment 10 and eroding the contrast created by the narrow exact-equality spike. Reduce or gate the shoulder for n=2 while retaining it for coalitions of three or more, where multiple mutually similar schemas can support a coalition-level shared-source inference. For n>=3, convert the affinity matrix into a stronger coalition-level effective-source or clustered-diversity estimate rather than relying only on mean pairwise overlap; this should amplify the composition ranking in Experiments 2 and 3 without introducing positional effects. Preserve or cautiously strengthen the narrow jointly-extreme component for Experiment 10, and widen heterogeneity specifically in its dependence/extremity parameters to increase the currently deficient subject variance. For count support, do not return to the iter-2 global count increase, which overshot Experiment 1. Instead use an intermediate setting: modestly increase the existing diversity-gated third-cue gain while keeping the post-three cap. That targeted adjustment can move Experiment 6 toward 0.44 and Experiment 1 toward -0.12 without restoring excessive rewards for four-to-six raw cues. Keep validity reliance low, beta broadly unchanged, and maintain complete exchangeability.

**Outcome of this advice:** iter 3 candidate loss=0.0930 -> iter 4 candidate loss=0.0958 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed order-invariant latent-source family, but it was rejected by the accept gate (loss 0.0958 versus the accepted 0.0930), so it should not replace the iteration-3 base. The new revision produced useful local movement in the central composition effects: Experiment 2 became more negative (-0.0525 versus -0.0379), Experiment 3 improved (-0.0164 versus -0.0119), and Experiment 10 moved toward the strong equal-pair discount (0.2419 versus 0.2865). Nevertheless, these remain underfit relative to -0.2642, -0.0410, and 0.1723, respectively. More importantly, the coalition-level amplification caused a large regression in Experiment 6: the fraction favoring the three-cue coalition fell from 0.34 to 0.22, away from the observed 0.44. Experiment 1 also remains too close to indifference (-0.0104 versus -0.1217). The Experiment 4 deviation (0.4773 versus 0.5087) is probably mostly simulation variability because the mechanism is exchangeable and the design targets a positional contrast; Experiments 5, 7, 8, and 9 continue to support the intended spatial nulls. Overall, the blanket amplification of dependence for every coherent three-member coalition discounts some coalitions that should receive count corroboration, rather than selectively improving composition sensitivity.
Rationale: The previous recommendation to combine an n=2 shoulder gate, stronger n>=3 cluster amplification, and a larger diversity-gated third-cue gain was rejected by the gate, so do not repeat that bundled change. Starting from the unchanged accepted iteration-3 base, make a smaller ablation-focused revision. First test the n=2 shoulder gate and a cautious strengthening of the narrow exact-equality spike by themselves, while leaving the accepted n>=3 overlap calculation and third-cue gain unchanged. This isolates the part that plausibly improved Experiment 10 without reproducing the large Experiment 6 regression. If an n>=3 adjustment is retained, replace the unconditional coherent-density multiplier with a sharply extremity-gated or thresholded latent-source posterior: amplification should occur only when several pair affinities are simultaneously strong and jointly extreme, not whenever their mean is moderately positive. This preserves corroboration for heterogeneous three-cue coalitions while discounting genuinely redundant reliability clusters. Avoid another global increase in count reward, shoulder strength, or cluster amplification. Also do not tune positional machinery to the Experiment 4 fluctuation, since that would violate the recommendation and threaten the well-fitted nulls. The main target is a selective improvement in Experiment 10—and, secondarily, Experiments 2 and 3—without sacrificing the accepted Experiment 6 value.

**Outcome of this advice:** iter 4 candidate loss=0.0958 -> iter 5 candidate loss=0.1045 -> the gate marked it REJECTED.

### Iteration 5 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed exchangeable latent-source family, but it was rejected by the accept gate: loss rose to 0.1045 from the iteration-3 running best of 0.0930. The isolated two-member shoulder attenuation did not deliver the intended improvement. Relative to iteration 3, Experiment 10 improved only modestly (0.2571 versus 0.2865, with 0.1723 observed), while Experiment 6 regressed sharply from 0.34 to 0.24 against 0.44 observed. Experiment 7 also fluctuated away from its null target. The major substantive effects remain compressed toward indifference: Experiment 1 is -0.0408 versus -0.1217, Experiment 2 is -0.0392 versus -0.2642, and Experiment 3 is -0.0115 versus -0.0410. By contrast, the exchangeability predictions remain successful in Experiments 4, 5, 8, and 9. Thus the model has the correct qualitative mechanism and null structure, but its choice probabilities express the latent coalition differences much too weakly. The current global strengthening of dependence also appears to spill into three-member coalitions, damaging Experiment 6 instead of selectively improving equal-pair skepticism.
Rationale: Both the previous n>=3 amplification bundle and the latest n=2 shoulder-gating/dependence-strengthening ablation were rejected, so do not repeat either structural direction. Build from the unchanged accepted iteration-3 model and, for the next test, leave its spike, shoulder, overlap, count cap, and diversity-gated third-cue term intact. Instead test a response-scale recalibration: raise the center of beta moderately and, if necessary, reduce epsilon slightly. The accepted model produces the correct signs in Experiments 1, 2, 3, and 10 but attenuates all four toward chance, while its beta range of 0.88-1.18 and lapse range of 0.03-0.09 can strongly compress those effects. A cautious range such as beta around 1.2-1.5 with epsilon around 0.02-0.06 is a distinct, lower-risk in-family knob. It should strengthen negative top-cue agreement and equal-pair rejection without creating positional effects, and may raise Experiment 6 if subjects already have weakly positive three-cue evidence. Preserve some subject-level variation in beta, but do not strengthen the global dependence prior, since that was associated with the Experiment 6 regression. If Experiment 10 later needs additional selectivity, introduce an equality-specific narrow-spike strength separate from the n>=3 dependence prior rather than modifying the shared parameter.

**Outcome of this advice:** iter 5 candidate loss=0.1045 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Reliability-Schema Source Inference proposes that decision makers treat expert agreement as evidence generated by latent informational sources rather than as conditionally independent votes. Advertised validities identify experts' reliability schemas. Dependence inference operates at two scales: an extremely narrow component assigns nearly identical, jointly extreme experts a high probability of sharing a source, while a weaker broad component captures partial dependence among merely similar reliability schemas. Thus equally extreme coalitions can be discounted sharply without treating moderately heterogeneous coalitions as fully independent. Every discriminating cue contributes, but its contribution is divided by inferred latent-source overlap. Coalition evidence has diminishing returns, and nominal count corroboration is capped after three members. The transition from two to three members receives an additional corroboration gain only to the extent that the coalition represents informationally diverse sources. Stable subject-level priors over source dependence and count-versus-independence tradeoffs produce heterogeneous three-versus-two judgments without rewarding arbitrarily large coalitions. Communicated reliability differences are strongly compressed, preventing a single advertised-best expert from dominating a diverse majority. The computation is exchangeable over expert identities and contains no onset, closure, recency, adjacency, entry, exit, or other spatial parsing mechanism.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Reliability-Schema Source Inference expects shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    validity_compression = float(parameters[\"validity_compression\"])\n    validity_reliance = float(parameters[\"validity_reliance\"])\n    coalition_saturation = float(parameters[\"coalition_saturation\"])\n    dependence_prior = float(parameters[\"dependence_prior\"])\n    similarity_bandwidth = float(parameters[\"similarity_bandwidth\"])\n    shoulder_strength = float(parameters[\"shoulder_strength\"])\n    shoulder_bandwidth = float(parameters[\"shoulder_bandwidth\"])\n    extremity_threshold = float(parameters[\"extremity_threshold\"])\n    extremity_slope = float(parameters[\"extremity_slope\"])\n    overlap_curvature = float(parameters[\"overlap_curvature\"])\n    count_independence_tradeoff = float(\n        parameters[\"count_independence_tradeoff\"]\n    )\n    diverse_third_gain = float(parameters[\"diverse_third_gain\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Advertised reliability is represented, but ordinal differences are\n    # strongly compressed before use. The mean normalization keeps overall\n    # evidence scale comparable across experiments with different validity\n    # schedules.\n    v = np.clip(validities, 0.500001, 0.999999)\n    diagnosticity = np.log(v / (1.0 - v))\n    compressed = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)\n    compressed /= max(float(np.mean(compressed)), 1e-12)\n    cue_weights = (\n        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)\n        + validity_reliance * compressed\n    )\n    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)\n\n    differences = a - b\n    a_support = np.flatnonzero(differences > 0.0)\n    b_support = np.flatnonzero(differences < 0.0)\n\n    if a_support.size == 0 and b_support.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def coalition_signal(indices):\n        idx = np.asarray(indices, dtype=int)\n        n = int(idx.size)\n        if n == 0:\n            return 0.0\n        if n == 1:\n            return float(cue_weights[idx[0]])\n\n        coalition_v = v[idx]\n        overlap = np.zeros(n, dtype=np.float64)\n        pair_posteriors = []\n\n        # The narrow component detects almost identical reliability schemas.\n        # A weaker, moderate-width shoulder represents partial common-source\n        # inference for nonidentical but similar experts.\n        for i in range(n):\n            for j in range(i + 1, n):\n                reliability_gap = float(coalition_v[i] - coalition_v[j])\n                narrow_similarity = np.exp(\n                    -0.5 * (reliability_gap / similarity_bandwidth) ** 2\n                )\n                shoulder_similarity = np.exp(\n                    -0.5 * (reliability_gap / shoulder_bandwidth) ** 2\n                )\n                pair_extremity = 0.5 * float(coalition_v[i] + coalition_v[j])\n                extreme_gate = 1.0 / (\n                    1.0\n                    + np.exp(\n                        -np.clip(\n                            extremity_slope\n                            * (pair_extremity - extremity_threshold),\n                            -50.0,\n                            50.0,\n                        )\n                    )\n                )\n\n                narrow_affinity = (\n                    dependence_prior\n                    * narrow_similarity\n                    * (0.01 + 0.99 * extreme_gate)\n                )\n                shoulder_affinity = (\n                    shoulder_strength\n                    * shoulder_similarity\n                    * (0.25 + 0.75 * extreme_gate)\n                )\n                source_affinity = narrow_affinity + shoulder_affinity\n                overlap[i] += source_affinity\n                overlap[j] += source_affinity\n                pair_posteriors.append(\n                    source_affinity / (1.0 + source_affinity)\n                )\n\n        # Shared-source overlap is converted superlinearly into redundancy.\n        # This permits sharp skepticism toward exactly equal extreme pairs\n        # while preserving useful evidence from heterogeneous additions.\n        independent_contributions = cue_weights[idx] / np.power(\n            1.0 + overlap, overlap_curvature\n        )\n        independent_total = float(np.sum(independent_contributions))\n\n        signal = independent_total / (float(n) ** coalition_saturation)\n\n        # Nominal count corroboration is capped after the third member rather\n        # than growing exponentially throughout a large coalition.\n        capped_count_steps = float(min(n - 1, 2))\n        signal *= np.exp(\n            np.clip(\n                count_independence_tradeoff * capped_count_steps,\n                -10.0,\n                10.0,\n            )\n        )\n\n        # Crossing from two to at least three cues provides corroboration only\n        # when those cues are inferred to represent informationally diverse\n        # sources. The bonus is then capped, so fourth and later cues do not\n        # receive repeated generic count rewards.\n        if n >= 3:\n            mean_shared_source = float(np.mean(pair_posteriors))\n            effective_diversity = np.clip(\n                1.0 - mean_shared_source, 0.0, 1.0\n            )\n            signal *= np.exp(\n                np.clip(diverse_third_gain * effective_diversity, -10.0, 10.0)\n            )\n\n        return float(signal)\n\n    evidence_a = coalition_signal(a_support)\n    evidence_b = coalition_signal(b_support)\n    net_a = evidence_a - evidence_b\n\n    logits = np.array(\n        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64\n    )\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "validity_compression": "[0.14, 0.30]",
    "validity_reliance": "[0.01, 0.06]",
    "coalition_saturation": "[0.62, 0.72]",
    "dependence_prior": "[1.80, 4.20]",
    "similarity_bandwidth": "[0.010, 0.022]",
    "shoulder_strength": "[0.20, 0.65]",
    "shoulder_bandwidth": "[0.055, 0.120]",
    "extremity_threshold": "[0.70, 0.78]",
    "extremity_slope": "[14.0, 24.0]",
    "overlap_curvature": "[1.80, 3.00]",
    "count_independence_tradeoff": "[-0.08, 0.16]",
    "diverse_third_gain": "[0.08, 0.28]",
    "beta": "[1.20, 1.48]",
    "epsilon": "[0.02, 0.06]"
  },
  "rationale": "This is a minimal response-scale recalibration of the accepted iteration-3 model. The spike, shoulder, overlap conversion, capped count term, and diversity-gated third-cue mechanism are unchanged because both recent structural revisions were rejected. Only beta is shifted moderately upward, from [0.88, 1.18] to [1.20, 1.48], and the lapse range is reduced from [0.03, 0.09] to [0.02, 0.06]. This should express the model's already-correct directional coalition rankings more strongly in Experiments 1, 2, 3, and 10 and can increase stable three-cue preferences in Experiment 6. Because the recalibration acts only after order-invariant coalition evidence has been computed, it cannot introduce onset, closure, recency, or entry-versus-exit effects and therefore preserves the successful spatial null structure."
}
```

## Usage

```json
{
  "input_tokens": 29104,
  "output_tokens": 2583
}
```
