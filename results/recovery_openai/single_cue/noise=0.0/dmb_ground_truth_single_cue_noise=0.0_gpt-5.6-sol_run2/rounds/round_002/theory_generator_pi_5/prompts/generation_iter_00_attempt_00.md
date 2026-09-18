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
- THEORY 1 = `pi_3`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_3`).

Replace pi_3 with a subject-level strategy-allocation theory of reliability-compressed coalition accumulation. The new theory should reject any universal recency or primacy gradient. Subjects should compress communicated validities and integrate all discriminating cues with diminishing returns, but differ stably in how attention is allocated among cue reliability, unweighted coalition size, and coherent cue groups. Positional effects should arise only indirectly from grouping or display parsing and should average to approximately zero in isolated singleton conflicts and balanced two-versus-two conflicts, matching Experiments 1 and 6. It should not include pi_4's deterministic switch-closure and amplified early-coalition gates; instead, closure or coherence should be weak, optional subject-level tendencies whose signs and strengths can vary. A population mixture or hierarchical distribution over integration styles is essential to reproduce Experiment 2's near-split majority outcome and very high between-subject variance without treating behavior as mere trial-level noise. The accumulation component should allow multiple lower-validity cues to defeat the highest-validity cue, accounting for Experiments 3 and 4, and should permit configuration-sensitive coalition grouping sufficient to produce Experiment 5's negative association with pi_3's recency-weighted predictions. The resulting theory would preserve pi_4's successful compressed, nonlexicographic accumulation insight while offering a cleaner and more falsifiable explanation of why strong positional signatures disappear at the population level and why subjects divide sharply in nearly balanced coalition conflicts.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_4` (overall score: 0.631)

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


---

### `pi_3` (overall score: 0.576)

**Description**
Accessibility-Balanced Evidence Integration proposes that people encode each communicated cue validity as a subjective diagnosticity signal and combine it with a separate positional-accessibility signal. Diagnosticity favors objectively valid cues, with nonlinear compression or sharpening of validity differences, whereas accessibility can favor cues appearing later or lower in the display. A subject-specific balance determines how strongly each code influences cue weight. On each choice, all discriminating cues contribute their signed evidence to a common accumulator; therefore, several opposing cues can gradually overcome an initially favored cue. Because the accumulator retains evidence magnitude and configuration, the theory predicts graded coalition effects rather than selecting either the first or last discriminating cue. Evidence supporting each option exhibits mild, imbalance-dependent within-coalition diminishing returns before the two directional totals are compared, followed by ordinary response noise and occasional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Accessibility-Balanced Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    balance = float(parameters["validity_position_balance"])
    recency = float(parameters["recency_gradient"])
    curvature = float(parameters["validity_curvature"])
    coalition_saturation = float(parameters["coalition_saturation"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Communicated validities are represented as log-odds diagnosticities.
    # Curvature captures sharpening or compression of perceived differences.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    max_diagnosticity = float(np.max(diagnosticity))
    if max_diagnosticity > 0.0:
        diagnosticity = diagnosticity / max_diagnosticity
    else:
        diagnosticity = np.ones(n_features, dtype=np.float64)
    diagnosticity = np.power(np.maximum(diagnosticity, 0.0), curvature)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Later/lower display positions have greater accessibility when recency > 0.
    if n_features == 1:
        position = np.zeros(1, dtype=np.float64)
    else:
        position = np.arange(n_features, dtype=np.float64) / float(n_features - 1)
    accessibility = np.exp(recency * position)
    accessibility /= max(float(np.mean(accessibility)), 1e-12)

    # A convex mixture makes the validity-versus-position tradeoff explicit.
    weights = balance * diagnosticity + (1.0 - balance) * accessibility
    weights /= max(float(np.mean(weights)), 1e-12)

    differences = a - b
    discriminating = differences != 0.0
    n_discriminating = int(np.sum(discriminating))
    if n_discriminating == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Evidence is accumulated separately for the two options. Diminishing
    # returns depend on directional coalition imbalance rather than absolute
    # coalition size, preserving full configuration evidence for equal-sized
    # coalitions while tempering a numerical majority against a minority.
    positive = differences > 0.0
    negative = differences < 0.0
    n_positive = max(int(np.sum(positive)), 1)
    n_negative = max(int(np.sum(negative)), 1)
    min_coalition = min(n_positive, n_negative)
    positive_evidence = float(np.dot(weights, np.maximum(differences, 0.0)))
    negative_evidence = float(np.dot(weights, np.maximum(-differences, 0.0)))
    evidence_a = (
        positive_evidence
        / (max(1.0, float(n_positive) / float(min_coalition)) ** coalition_saturation)
        - negative_evidence
        / (max(1.0, float(n_negative) / float(min_coalition)) ** coalition_saturation)
    )

    logits = np.array(
        [0.5 * beta * evidence_a, -0.5 * beta * evidence_a],
        dtype=np.float64,
    )
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
- validity_position_balance: [0.25, 0.35]
- recency_gradient: [1.8, 2.4]
- validity_curvature: [0.65, 1.0]
- coalition_saturation: [0.65, 0.8]
- beta: [0.65, 1.15]
- epsilon: [0.02, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.1033 (var=0.0092)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.1892 (var=0.0123)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=0.0860 (var=0.0002)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.6281 (var=0.0049)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5692 (var=0.0021)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.0000 (var=0.0000)


---

### `pi_2` (overall score: 0.501)

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
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)

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
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)

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
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)

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
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)

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
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)


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
  "theory": "Reliability-Compressed Parsing Mixtures proposes that people encode communicated cue validities but strongly compress their differences, preventing the most reliable cue from becoming lexicographically decisive. Every discriminating cue enters a directional accumulator, with coalition evidence exhibiting diminishing returns. Stable subject-level integration styles allocate attention differently among compressed reliability, numerical coalition size, and locally parsed cue groups. Some subjects treat a larger coalition as corroboration, whereas others treat the extra members as partially redundant and favor a compact minority coalition; this signed calibration is applied only when both options have genuine multi-cue coalitions. Display order has no universal primacy or recency gradient. Instead, weak positional effects can emerge indirectly because subjects parse adjacent cues into different local chunks. Completion of an interpretation is an optional, sign-varying influence restricted to conflicts between two coalitions of at least three cues. Thus singleton conflicts and balanced two-versus-two conflicts have approximately zero population positional effect. The stable mixture produces sharply heterogeneous subjects in nearly balanced conflicts while retaining graded, nonlexicographic accumulation within every style.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Reliability-Compressed Parsing Mixtures expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    style = int(parameters[\"integration_style\"])\n    validity_compression = float(parameters[\"validity_compression\"])\n    coalition_saturation = float(parameters[\"coalition_saturation\"])\n    size_calibration = float(parameters[\"size_calibration\"])\n    parsing_strength = float(parameters[\"parsing_strength\"])\n    completion_strength = float(parameters[\"completion_strength\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Reliability differences are represented but strongly compressed.\n    v = np.clip(validities, 0.500001, 0.999999)\n    diagnosticity = np.log(v / (1.0 - v))\n    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)\n    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)\n\n    # The nine stable styles form three broad allocation families. Count-led\n    # subjects use little reliability differentiation, reliability-led\n    # subjects use more, and parsing-led subjects occupy an intermediate level.\n    if style <= 3:\n        reliability_attention = 0.04 + 0.015 * style\n    elif style <= 5:\n        reliability_attention = 0.28 + 0.05 * (style - 4)\n    else:\n        reliability_attention = 0.12 + 0.03 * (style - 6)\n\n    cue_weights = (\n        (1.0 - reliability_attention) * np.ones(n_features, dtype=np.float64)\n        + reliability_attention * diagnosticity\n    )\n    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)\n\n    differences = a - b\n    pos_idx = np.flatnonzero(differences > 0.0)\n    neg_idx = np.flatnonzero(differences < 0.0)\n    n_pos = int(pos_idx.size)\n    n_neg = int(neg_idx.size)\n\n    if n_pos == 0 and n_neg == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    def accumulated_evidence(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n        # All cues contribute, while total evidence grows sublinearly.\n        return float(np.sum(cue_weights[indices])) / (float(n) ** coalition_saturation)\n\n    evidence_a = accumulated_evidence(pos_idx)\n    evidence_b = accumulated_evidence(neg_idx)\n\n    # There is deliberately no fixed position gradient. A subject instead\n    # parses the display into short local chunks whose width and phase are\n    # stable properties of that subject's style.\n    chunk_width = 2 + (style % 3)\n    chunk_phase = style % chunk_width\n\n    def parsing_quality(indices):\n        n = int(indices.size)\n        if n < 2:\n            return 0.0\n        idx = np.asarray(indices, dtype=int)\n        adjacent = float(np.sum(np.diff(idx) == 1)) / float(n - 1)\n        chunk_ids = np.floor_divide(idx + chunk_phase, chunk_width)\n        same_chunk_pairs = 0.0\n        total_pairs = float(n * (n - 1) // 2)\n        for i in range(n):\n            for j in range(i + 1, n):\n                if chunk_ids[i] == chunk_ids[j]:\n                    same_chunk_pairs += 1.0\n        local_binding = same_chunk_pairs / max(total_pairs, 1.0)\n        return 0.5 * adjacent + 0.5 * local_binding\n\n    # Multi-cue conflicts receive a weak, signed structural calibration.\n    # Styles 0--3 regard larger coalitions as corroborative; styles 4--8\n    # regard excess members as increasingly redundant. This creates a stable\n    # population split in 3-versus-2 conflicts without changing singleton\n    # comparisons or discarding any cue from the base accumulator.\n    if n_pos >= 2 and n_neg >= 2:\n        size_orientation = 1.0 if style <= 3 else -1.0\n        log_size_ratio = np.log(float(n_pos) / float(n_neg))\n        size_shift = size_orientation * size_calibration * log_size_ratio\n\n        # Parsing valence varies across subjects, so no display location has a\n        # population-wide advantage. Centering the style-specific valence also\n        # keeps balanced 2-versus-2 effects close to zero in the population.\n        parsing_valences = np.array(\n            [-1.0, 0.75, -0.50, 1.0, -0.75, 0.50, -1.0, 0.75, 0.25],\n            dtype=np.float64,\n        )\n        parse_difference = parsing_quality(pos_idx) - parsing_quality(neg_idx)\n        parsing_shift = parsing_strength * float(parsing_valences[style]) * parse_difference\n\n        log_gate_a = 0.5 * (size_shift + parsing_shift)\n        log_gate_b = -0.5 * (size_shift + parsing_shift)\n        evidence_a *= np.exp(np.clip(log_gate_a, -10.0, 10.0))\n        evidence_b *= np.exp(np.clip(log_gate_b, -10.0, 10.0))\n\n    net_a = evidence_a - evidence_b\n\n    # Completion is optional, weak, and sign-varying. It is absent from\n    # singleton and 2-versus-2 conflicts and never closes the accumulator.\n    if n_pos == n_neg and n_pos >= 3:\n        completion_sign = 1.0 if style <= 5 else -1.0\n        if int(pos_idx[-1]) > int(neg_idx[-1]):\n            net_a += completion_sign * completion_strength\n        elif int(neg_idx[-1]) > int(pos_idx[-1]):\n            net_a -= completion_sign * completion_strength\n\n    logits = np.array([0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64)\n    logits -= np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    probs /= probs.sum()\n    return probs",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "integration_style": "{0, 1, 2, 3, 4, 5, 6, 7, 8}",
    "validity_compression": "[0.20, 0.42]",
    "coalition_saturation": "[0.74, 0.86]",
    "size_calibration": "[0.90, 1.35]",
    "parsing_strength": "[0.12, 0.30]",
    "completion_strength": "[0.38, 0.58]",
    "beta": "[0.85, 1.20]",
    "epsilon": "[0.03, 0.09]"
  },
  "rationale": "The model preserves the successful compressed, nonlexicographic accumulation mechanism: all discriminating cues contribute, multiple weaker cues can defeat a stronger cue, and directional totals grow with diminishing returns. It removes pi_3's universal recency gradient and pi_4's universal early-coalition gate. Singleton conflicts receive no structural or positional adjustment, and two-versus-two conflicts receive only sign-varying local parsing effects whose population mean is near zero, addressing Experiments 4 and 5. The nine stable styles instantiate a subject-level rather than trial-level mixture. Four styles treat an additional coalition member as corroboration and five treat it as partly redundant, yielding an approximately 4/9 split in sustained three-versus-two choices and therefore targeting Experiment 6's 0.44 subject fraction and high between-subject variance. Equal three-versus-three conflicts can exhibit a modest completion effect, but its sign differs across styles and accumulation remains active; this can reproduce Experiment 2's aggregate opposition to the first discriminating cue without asserting a universal recency rule. Finally, phase-shifted local chunk parsing makes predictions configuration-sensitive, which can reverse associations with a smooth recency model in Experiment 3 while averaging away positional signatures across isolated and balanced conflicts."
}
```

## Usage

```json
{
  "input_tokens": 13461,
  "output_tokens": 6357
}
```
