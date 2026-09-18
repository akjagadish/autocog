# generation_iter_09_attempt_00

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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 with a new Contextual Coalition-Topology theory. Preserve the empirically successful exchangeable foundation: every discriminating cue contributes; advertised validity differences are strongly compressed; coalition evidence grows sublinearly; subjects have stable count calibration and zero-centered idiosyncratic cue salience; and there are no onset, closure, adjacency, entry, exit, or recency mechanisms. Depart from both theories by making redundancy genuinely higher-order. For two-member coalitions, use only a smooth jointly-diagnostic composition discount, with no narrow equality-specific source prior and no sharp validity-gap transition. This should keep Experiment 1's low-versus-high gap contrast near 0.03 and permit Experiment 12's aversion to strong equal pairs through overall diagnostic composition rather than equality itself. For coalitions of three or more, introduce a triadic topology operation: when two members form a near-duplicate reliability cluster while another member occupies a distinct reliability region, the clustered members are compressed toward one effective informational unit. Distributed triples with comparable mean validity and pairwise composition remain more effective because they span more reliability-schema regions. This higher-order cluster-with-context mechanism should raise distributed choice in Experiment 2 toward 0.67 without predicting a large effect for isolated equal pairs in Experiment 1. Give subjects a stable topology-sensitivity parameter with appreciable dispersion to reproduce Experiment 2's excess between-subject variance, while keeping pair-composition sensitivity narrower to respect Experiment 1's low variance. Retain flexible but bounded third-cue corroboration to improve Experiment 8, and calibrate validity reliance and count saturation jointly against Experiments 3–5. The replacement should be tested with parameter ranges constrained to produce: a small Experiment-1 gap contrast, a strong Experiment-2 distributed-triple advantage, near-zero spatial contrasts in Experiments 6–7 and 9–11, and the observed smooth composition aversion in Experiment 12.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.884)

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
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.0217 (var=0.0086)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.4883 (var=0.0028)


---

### `pi_7` (overall score: 0.648)

**Description**
Reliability-Schema Source Inference proposes that decision makers treat expert agreement as evidence generated by latent informational sources rather than as conditionally independent votes. Advertised validities identify experts' reliability schemas. Dependence inference operates at two scales: an extremely narrow component assigns nearly identical, jointly extreme experts a high probability of sharing a source, while a narrower but stronger shoulder captures partial dependence among merely similar reliability schemas. For two-member coalitions, an additional equality-specific jointly extreme source prior produces nonlinear skepticism toward exactly matched reliable experts without increasing dependence for larger coalitions. Thus equally extreme coalitions can be discounted sharply without treating moderately heterogeneous coalitions as fully independent. Every discriminating cue contributes, but its contribution is divided by inferred latent-source overlap. Coalition evidence has diminishing returns, and nominal count corroboration is capped after three members. The transition from two to three members receives an additional corroboration gain only to the extent that the coalition represents informationally diverse sources. Stable subject-level priors over source dependence and count-versus-independence tradeoffs produce heterogeneous three-versus-two judgments without rewarding arbitrarily large coalitions. Communicated reliability differences are strongly compressed, preventing a single advertised-best expert from dominating a diverse majority. The computation is exchangeable over expert identities and contains no onset, closure, recency, adjacency, entry, exit, or other spatial parsing mechanism.

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
    equality_pair_strength = float(parameters["equality_pair_strength"])
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
        # A moderate-width shoulder represents partial common-source
        # inference for nonidentical but similar experts. An extra narrow
        # prior applies only to jointly extreme two-member coalitions, keeping
        # exact-pair skepticism separate from dependence among larger groups.
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
                if n == 2:
                    narrow_affinity += (
                        equality_pair_strength
                        * narrow_similarity
                        * extreme_gate
                    )
                shoulder_affinity = (
                    shoulder_strength
                    * shoulder_similarity
                    * (0.10 + 0.90 * extreme_gate)
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
- dependence_prior: [2.10, 3.90]
- equality_pair_strength: [0.60, 1.80]
- similarity_bandwidth: [0.010, 0.022]
- shoulder_strength: [0.62, 0.92]
- shoulder_bandwidth: [0.032, 0.065]
- extremity_threshold: [0.72, 0.78]
- extremity_slope: [20.0, 30.0]
- overlap_curvature: [2.10, 3.10]
- count_independence_tradeoff: [-0.02, 0.16]
- diverse_third_gain: [0.20, 0.40]
- beta: [1.20, 1.48]
- epsilon: [0.02, 0.06]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1217 (var=0.0117) vs this=-0.3083 (var=0.0280)
- Experiment 2: real=-0.2642 (var=0.0087) vs this=-0.0892 (var=0.0075)
- Experiment 3: real=-0.0410 (var=0.0003) vs this=-0.0224 (var=0.0002)
- Experiment 4: real=0.5087 (var=0.0038) vs this=0.4932 (var=0.0039)
- Experiment 5: real=0.5021 (var=0.0024) vs this=0.5052 (var=0.0023)
- Experiment 6: real=0.4400 (var=0.2464) vs this=0.3200 (var=0.2176)
- Experiment 7: real=0.0219 (var=0.0094) vs this=-0.0108 (var=0.0126)
- Experiment 8: real=0.0117 (var=0.0101) vs this=-0.0021 (var=0.0101)
- Experiment 9: real=0.4958 (var=0.0041) vs this=0.4946 (var=0.0027)
- Experiment 10: real=0.1723 (var=0.0081) vs this=0.1814 (var=0.0023)
- Experiment 11: real=0.0317 (var=0.0030) vs this=0.2729 (var=0.0090)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.5575 (var=0.0028)


---

### `pi_2` (overall score: 0.553)

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
- Experiment 11: real=0.0317 (var=0.0030) vs this=-0.0292 (var=0.0133)
- Experiment 12: real=0.6692 (var=0.0068) vs this=0.5073 (var=0.0027)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0651 -> ACCEPTED
- iter 2: loss=0.0539 -> ACCEPTED
- iter 3: loss=0.0404 -> ACCEPTED
- iter 4: loss=0.0530 -> REJECTED
- iter 5: loss=0.0447 -> REJECTED
- iter 6: loss=0.0846 -> REJECTED
- iter 7: loss=0.0391 -> ACCEPTED
- iter 8: loss=0.0408 -> REJECTED
- iter 9: loss=0.0363 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 9 at loss=0.0363 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: -0.1263 (var=0.0272) (Δ vs real -0.0046)
  - iter 2: -0.1154 (var=0.0132) (Δ vs real +0.0063)
  - iter 3: -0.1217 (var=0.0210) (Δ vs real +0.0000)
  - iter 4: -0.1046 (var=0.0175) (Δ vs real +0.0171)
  - iter 5: -0.1329 (var=0.0158) (Δ vs real -0.0112)
  - iter 6: -0.0096 (var=0.0197) (Δ vs real +0.1121)
  - iter 7: -0.0979 (var=0.0159) (Δ vs real +0.0238)
  - iter 8: -0.1375 (var=0.0102) (Δ vs real -0.0158)
  - iter 9 (most recent): -0.1138 (var=0.0159) (Δ vs real +0.0079)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7129 (var=0.0389)
- pi_2: -0.6354 (var=0.0266)
- pi_3: -0.1033 (var=0.0092)
- pi_4: -0.1246 (var=0.0114)
- pi_5: -0.1179 (var=0.0105)
- pi_6: -0.0779 (var=0.0176)
- pi_7: -0.3083 (var=0.0280)

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
  - iter 1: -0.2933 (var=0.0117) (Δ vs real -0.0292)
  - iter 2: -0.2571 (var=0.0138) (Δ vs real +0.0071)
  - iter 3: -0.2225 (var=0.0109) (Δ vs real +0.0417)
  - iter 4: -0.1996 (var=0.0074) (Δ vs real +0.0646)
  - iter 5: -0.2308 (var=0.0090) (Δ vs real +0.0333)
  - iter 6: -0.1938 (var=0.0088) (Δ vs real +0.0704)
  - iter 7: -0.2346 (var=0.0103) (Δ vs real +0.0296)
  - iter 8: -0.2379 (var=0.0092) (Δ vs real +0.0262)
  - iter 9 (most recent): -0.2492 (var=0.0117) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0008 (var=0.0102)
- pi_1: 0.7221 (var=0.0226)
- pi_3: -0.1892 (var=0.0123)
- pi_4: -0.2667 (var=0.0120)
- pi_5: -0.2217 (var=0.0089)
- pi_6: -0.2321 (var=0.0159)
- pi_7: -0.0892 (var=0.0075)

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
  - iter 1: -0.0479 (var=0.0003) (Δ vs real -0.0069)
  - iter 2: -0.0370 (var=0.0002) (Δ vs real +0.0040)
  - iter 3: -0.0378 (var=0.0002) (Δ vs real +0.0032)
  - iter 4: -0.0390 (var=0.0002) (Δ vs real +0.0020)
  - iter 5: -0.0389 (var=0.0002) (Δ vs real +0.0021)
  - iter 6: -0.0334 (var=0.0002) (Δ vs real +0.0076)
  - iter 7: -0.0391 (var=0.0002) (Δ vs real +0.0019)
  - iter 8: -0.0355 (var=0.0003) (Δ vs real +0.0055)
  - iter 9 (most recent): -0.0369 (var=0.0002) (Δ vs real +0.0041)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0860 (var=0.0002)
- pi_2: -0.0017 (var=0.0001)
- pi_1: 0.0899 (var=0.0009)
- pi_4: -0.0260 (var=0.0002)
- pi_5: -0.0237 (var=0.0003)
- pi_6: -0.0366 (var=0.0003)
- pi_7: -0.0224 (var=0.0002)

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
  - iter 1: 0.5013 (var=0.0059) (Δ vs real -0.0073)
  - iter 2: 0.4948 (var=0.0064) (Δ vs real -0.0139)
  - iter 3: 0.5121 (var=0.0052) (Δ vs real +0.0034)
  - iter 4: 0.5042 (var=0.0044) (Δ vs real -0.0045)
  - iter 5: 0.5020 (var=0.0041) (Δ vs real -0.0067)
  - iter 6: 0.4947 (var=0.0043) (Δ vs real -0.0139)
  - iter 7: 0.4986 (var=0.0047) (Δ vs real -0.0101)
  - iter 8: 0.5067 (var=0.0029) (Δ vs real -0.0020)
  - iter 9 (most recent): 0.5153 (var=0.0056) (Δ vs real +0.0067)
**Other theories' values on this metric (for reference):**
- pi_2: 0.5017 (var=0.0041)
- pi_3: 0.6281 (var=0.0049)
- pi_1: 0.1476 (var=0.0116)
- pi_4: 0.5061 (var=0.0029)
- pi_5: 0.4957 (var=0.0036)
- pi_6: 0.5046 (var=0.0065)
- pi_7: 0.4932 (var=0.0039)

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
  - iter 1: 0.5073 (var=0.0043) (Δ vs real +0.0052)
  - iter 2: 0.4988 (var=0.0031) (Δ vs real -0.0033)
  - iter 3: 0.5075 (var=0.0027) (Δ vs real +0.0054)
  - iter 4: 0.4944 (var=0.0016) (Δ vs real -0.0077)
  - iter 5: 0.4902 (var=0.0016) (Δ vs real -0.0119)
  - iter 6: 0.5023 (var=0.0023) (Δ vs real +0.0002)
  - iter 7: 0.5071 (var=0.0026) (Δ vs real +0.0050)
  - iter 8: 0.4906 (var=0.0023) (Δ vs real -0.0115)
  - iter 9 (most recent): 0.5004 (var=0.0026) (Δ vs real -0.0017)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5692 (var=0.0021)
- pi_4: 0.3798 (var=0.0029)
- pi_1: 0.1575 (var=0.0119)
- pi_2: 0.4992 (var=0.0023)
- pi_5: 0.5098 (var=0.0033)
- pi_6: 0.5123 (var=0.0025)
- pi_7: 0.5052 (var=0.0023)

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
  - iter 1: 0.6000 (var=0.2400) (Δ vs real +0.1600)
  - iter 2: 0.6200 (var=0.2356) (Δ vs real +0.1800)
  - iter 3: 0.4200 (var=0.2436) (Δ vs real -0.0200)
  - iter 4: 0.5600 (var=0.2464) (Δ vs real +0.1200)
  - iter 5: 0.3600 (var=0.2304) (Δ vs real -0.0800)
  - iter 6: 0.0800 (var=0.0736) (Δ vs real -0.3600)
  - iter 7: 0.4000 (var=0.2400) (Δ vs real -0.0400)
  - iter 8: 0.5200 (var=0.2496) (Δ vs real +0.0800)
  - iter 9 (most recent): 0.4400 (var=0.2464) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_4: 0.9800 (var=0.0196)
- pi_3: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0000)
- pi_2: 1.0000 (var=0.0000)
- pi_5: 0.3200 (var=0.2176)
- pi_6: 0.3600 (var=0.2304)
- pi_7: 0.3200 (var=0.2176)

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
  - iter 1: -0.0019 (var=0.0117) (Δ vs real -0.0238)
  - iter 2: -0.0106 (var=0.0134) (Δ vs real -0.0325)
  - iter 3: 0.0200 (var=0.0154) (Δ vs real -0.0019)
  - iter 4: -0.0018 (var=0.0112) (Δ vs real -0.0236)
  - iter 5: 0.0192 (var=0.0091) (Δ vs real -0.0026)
  - iter 6: 0.0263 (var=0.0130) (Δ vs real +0.0044)
  - iter 7: 0.0111 (var=0.0145) (Δ vs real -0.0108)
  - iter 8: -0.0140 (var=0.0123) (Δ vs real -0.0359)
  - iter 9 (most recent): -0.0071 (var=0.0103) (Δ vs real -0.0289)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2360 (var=0.0123)
- pi_4: 0.0009 (var=0.0095)
- pi_1: 0.0069 (var=0.0048)
- pi_2: -0.0035 (var=0.0050)
- pi_3: -0.0376 (var=0.0081)
- pi_6: 0.0257 (var=0.0065)
- pi_7: -0.0108 (var=0.0126)

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
  - iter 1: 0.0254 (var=0.0112) (Δ vs real +0.0138)
  - iter 2: -0.0029 (var=0.0121) (Δ vs real -0.0146)
  - iter 3: -0.0200 (var=0.0093) (Δ vs real -0.0317)
  - iter 4: -0.0121 (var=0.0151) (Δ vs real -0.0237)
  - iter 5: -0.0113 (var=0.0165) (Δ vs real -0.0229)
  - iter 6: -0.0158 (var=0.0100) (Δ vs real -0.0275)
  - iter 7: -0.0067 (var=0.0185) (Δ vs real -0.0183)
  - iter 8: -0.0292 (var=0.0125) (Δ vs real -0.0408)
  - iter 9 (most recent): 0.0204 (var=0.0144) (Δ vs real +0.0088)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2729 (var=0.0146)
- pi_5: -0.0100 (var=0.0101)
- pi_1: 0.7325 (var=0.0409)
- pi_2: -0.0042 (var=0.0038)
- pi_3: -0.4775 (var=0.0097)
- pi_6: -0.0154 (var=0.0165)
- pi_7: -0.0021 (var=0.0101)

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
  - iter 1: 0.4846 (var=0.0033) (Δ vs real -0.0113)
  - iter 2: 0.4900 (var=0.0036) (Δ vs real -0.0058)
  - iter 3: 0.5131 (var=0.0028) (Δ vs real +0.0173)
  - iter 4: 0.4900 (var=0.0029) (Δ vs real -0.0058)
  - iter 5: 0.5012 (var=0.0027) (Δ vs real +0.0054)
  - iter 6: 0.4892 (var=0.0021) (Δ vs real -0.0067)
  - iter 7: 0.5002 (var=0.0022) (Δ vs real +0.0044)
  - iter 8: 0.5029 (var=0.0023) (Δ vs real +0.0071)
  - iter 9 (most recent): 0.5031 (var=0.0038) (Δ vs real +0.0073)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6108 (var=0.0033)
- pi_6: 0.4996 (var=0.0028)
- pi_1: 0.4104 (var=0.0015)
- pi_2: 0.4985 (var=0.0025)
- pi_3: 0.6115 (var=0.0020)
- pi_4: 0.6271 (var=0.0029)
- pi_7: 0.4946 (var=0.0027)

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
  - iter 1: 0.1300 (var=0.0017) (Δ vs real -0.0424)
  - iter 2: 0.1563 (var=0.0016) (Δ vs real -0.0160)
  - iter 3: 0.1615 (var=0.0022) (Δ vs real -0.0108)
  - iter 4: 0.1586 (var=0.0024) (Δ vs real -0.0137)
  - iter 5: 0.1564 (var=0.0036) (Δ vs real -0.0159)
  - iter 6: 0.1509 (var=0.0017) (Δ vs real -0.0214)
  - iter 7: 0.1584 (var=0.0025) (Δ vs real -0.0140)
  - iter 8: 0.1725 (var=0.0027) (Δ vs real +0.0002)
  - iter 9 (most recent): 0.1680 (var=0.0016) (Δ vs real -0.0043)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2057 (var=0.0035)
- pi_5: 0.5023 (var=0.0031)
- pi_1: 0.8796 (var=0.0067)
- pi_2: 0.4978 (var=0.0039)
- pi_3: 0.5414 (var=0.0029)
- pi_4: 0.3753 (var=0.0039)
- pi_7: 0.1814 (var=0.0023)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        high_gap = []
        low_gap = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = a - b

            # Target coalitions occupy cues 0--15; cues 16--17 are the
            # fixed reference coalition. Each target coalition is one of
            # the adjacent cue pairs (0,1), (2,3), ..., (14,15).
            target_idx = np.flatnonzero(d[:16] != 0.0)
            if target_idx.size != 2:
                continue

            pair_id = int(np.min(target_idx) // 2)
            direction = float(np.mean(d[target_idx]))
            if direction == 0.0:
                continue

            target_response = 0 if direction > 0.0 else 1
            chose_target = float(int(row["response"]) == target_response)

            # Low gaps comprise 0.00 and 0.04; high gaps comprise 0.10
            # and 0.18. This uses all 96 observations per subject.
            if pair_id <= 3:
                low_gap.append(chose_target)
            else:
                high_gap.append(chose_target)

        if len(high_gap) == 0 or len(low_gap) == 0:
            return np.nan
        return float(np.mean(high_gap) - np.mean(low_gap))

    if "subject_id" in data.columns:
        values = [subject_contrast(df) for _, df in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_contrast(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))
```

**Observed (real) value:** 0.0317 (var=0.0030)
**Previous candidate values (this loop):**
  - iter 1: 0.0133 (var=0.0080) (Δ vs real -0.0183)
  - iter 2: 0.0237 (var=0.0098) (Δ vs real -0.0079)
  - iter 3: 0.0229 (var=0.0080) (Δ vs real -0.0087)
  - iter 4: 0.0138 (var=0.0074) (Δ vs real -0.0179)
  - iter 5: 0.0108 (var=0.0062) (Δ vs real -0.0208)
  - iter 6: 0.0333 (var=0.0078) (Δ vs real +0.0017)
  - iter 7: -0.0075 (var=0.0057) (Δ vs real -0.0392)
  - iter 8: 0.0138 (var=0.0082) (Δ vs real -0.0179)
  - iter 9 (most recent): -0.0071 (var=0.0070) (Δ vs real -0.0387)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2729 (var=0.0090)
- pi_6: 0.0217 (var=0.0086)
- pi_1: 0.0167 (var=0.0064)
- pi_2: -0.0292 (var=0.0133)
- pi_3: 0.1604 (var=0.0061)
- pi_4: 0.0054 (var=0.0079)
- pi_5: 0.0146 (var=0.0117)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    distributed = {(3, 4, 5), (9, 10, 11)}

    def subject_score(df):
        records = []
        for _, row in df.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = a - b
            a_support = tuple(np.flatnonzero(diff > 0).tolist())
            b_support = tuple(np.flatnonzero(diff < 0).tolist())
            a_is_distributed = a_support in distributed
            b_is_distributed = b_support in distributed
            if a_is_distributed == b_is_distributed:
                continue
            chose_a = int(row['response']) == 0
            chose_distributed = (
                (chose_a and a_is_distributed) or
                ((not chose_a) and b_is_distributed)
            )
            records.append((a_is_distributed, float(chose_distributed)))

        if not records:
            return np.nan

        records = pd.DataFrame(records, columns=['distributed_on_a', 'choice'])
        role_means = records.groupby('distributed_on_a')['choice'].mean()
        if len(role_means) == 2:
            return float(0.5 * (role_means.loc[False] + role_means.loc[True]))
        return float(records['choice'].mean())

    if len(data) == 0:
        return float('nan')

    if 'subject_id' in data.columns:
        scores = [subject_score(df) for _, df in data.groupby('subject_id', sort=False)]
    else:
        scores = [subject_score(data)]
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    return float(np.mean(scores)) if scores.size else float('nan')
```

**Observed (real) value:** 0.6692 (var=0.0068)
**Previous candidate values (this loop):**
  - iter 1: 0.5435 (var=0.0032) (Δ vs real -0.1256)
  - iter 2: 0.5758 (var=0.0044) (Δ vs real -0.0933)
  - iter 3: 0.5554 (var=0.0037) (Δ vs real -0.1138)
  - iter 4: 0.5752 (var=0.0027) (Δ vs real -0.0940)
  - iter 5: 0.5606 (var=0.0043) (Δ vs real -0.1085)
  - iter 6: 0.5102 (var=0.0026) (Δ vs real -0.1590)
  - iter 7: 0.7329 (var=0.0059) (Δ vs real +0.0638)
  - iter 8: 0.6488 (var=0.0045) (Δ vs real -0.0204)
  - iter 9 (most recent): 0.7160 (var=0.0064) (Δ vs real +0.0469)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4883 (var=0.0028)
- pi_7: 0.5575 (var=0.0028)
- pi_1: 0.1550 (var=0.0096)
- pi_2: 0.5073 (var=0.0027)
- pi_3: 0.6663 (var=0.0014)
- pi_4: 0.5715 (var=0.0020)
- pi_5: 0.5602 (var=0.0027)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Contextual Reliability-Region Compression proposes that people integrate every discriminating expert rating into an exchangeable option-level coalition while representing advertised validities on a strongly compressed reliability-region scale. Coalition evidence grows sublinearly, and stable count calibration can make a third cue either corroborative or partially redundant. Redundancy is order-specific in cardinality but not spatial position: two-cue coalitions receive only a smooth joint-diagnostic composition discount, whereas coalitions of three or more undergo a contextual topology operation. In the latter operation, two cues occupying nearly the same represented reliability region are compressed toward one informational unit when another cue occupies a distinct region. In equal-cardinality contests between multi-cue coalitions, people additionally compare the coalitions' topology relationally. This relational comparison is mildly concave: modest clustered-versus-distributed differences are transmitted approximately linearly, whereas larger differences receive diminishing amplification. The comparison fades smoothly when coalition cardinalities differ, preserving independently calibrated count judgments. Topology sensitivity is stable and heterogeneous across subjects, while pair-composition sensitivity is comparatively narrow. Stable, zero-centered idiosyncratic cue salience permits individual configuration preferences without population-level onset, closure, recency, adjacency, entry, or exit effects. Because subjects receive no outcome feedback, these representations remain fixed across trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contextual Reliability-Region Compression expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(
        parameters["cue_salience_profile"], dtype=np.float64
    )
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != "
            f"n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    count_calibration = float(parameters["count_calibration"])
    count_calibration_strength = float(parameters["count_calibration_strength"])
    pair_composition_sensitivity = float(
        parameters["pair_composition_sensitivity"]
    )
    topology_sensitivity = float(parameters["topology_sensitivity"])
    topology_contrast_gain = float(parameters["topology_contrast_gain"])
    cluster_bandwidth = float(parameters["cluster_bandwidth"])
    context_scale = float(parameters["context_scale"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Reliability is encoded on a diagnosticity scale, but differences are
    # strongly compressed. Mean normalization prevents an experiment's
    # overall validity level from changing the generic evidence scale.
    v = np.clip(validities, 0.500001, 0.999999)
    log_odds = np.log(v / (1.0 - v))
    represented_diagnosticity = np.power(
        np.maximum(log_odds, 1e-12), validity_compression
    )
    represented_diagnosticity /= max(
        float(np.mean(represented_diagnosticity)), 1e-12
    )

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * represented_diagnosticity
    )

    # Components are exchangeably sampled and centered within each subject.
    # Thus salience is stable and can generate individual differences, but
    # no display position has a positive population-level privilege.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attention = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attention /= max(float(np.mean(attention)), 1e-12)

    cue_weights = reliability_weight * attention
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    difference = a - b
    a_support = np.flatnonzero(difference > 0.0)
    b_support = np.flatnonzero(difference < 0.0)
    n_a = int(a_support.size)
    n_b = int(b_support.size)

    if n_a == 0 and n_b == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    bounded_diagnosticity = np.clip(2.0 * v - 1.0, 0.0, 1.0)

    def triadic_topology(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 3:
            return 0.0

        scores = []
        # Topological distance is computed in the same compressed,
        # mean-normalized reliability-region coordinate used by cognition,
        # rather than in raw advertised-validity units.
        for x in range(n - 2):
            for y in range(x + 1, n - 1):
                for z in range(y + 1, n):
                    triad = np.sort(
                        represented_diagnosticity[idx[[x, y, z]]]
                    )
                    lower_gap = float(triad[1] - triad[0])
                    upper_gap = float(triad[2] - triad[1])

                    if lower_gap <= upper_gap:
                        cluster_gap = lower_gap
                        cluster_center = 0.5 * float(triad[0] + triad[1])
                        outsider = float(triad[2])
                    else:
                        cluster_gap = upper_gap
                        cluster_center = 0.5 * float(triad[1] + triad[2])
                        outsider = float(triad[0])

                    separation = abs(outsider - cluster_center)
                    duplicate_affinity = np.exp(
                        -0.5 * (cluster_gap / cluster_bandwidth) ** 2
                    )
                    contextual_distinctness = separation / (
                        separation + context_scale
                    )
                    scores.append(
                        float(duplicate_affinity * contextual_distinctness)
                    )

        if not scores:
            return 0.0

        # A smooth probabilistic union detects whether the coalition contains
        # at least one duplicate-pair-in-context relation without privileging
        # a cue identity, ordering, or spatial adjacency.
        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 0.999999)
        return float(1.0 - np.prod(1.0 - scores))

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0

        total = float(np.sum(cue_weights[idx]))

        if n >= 3:
            topology = triadic_topology(idx)
            # Sensitivity maps smoothly to the fraction of one informational
            # unit removed. At its upper limit a near-duplicate pair is worth
            # approximately one unit rather than two; all cues nevertheless
            # retain a positive contribution.
            compression_fraction = (
                1.0 - np.exp(-topology_sensitivity * topology)
            )
            coalition_unit = float(np.mean(cue_weights[idx]))
            total -= compression_fraction * coalition_unit
            total = max(total, 1e-12)

        signal = total / (float(n) ** coalition_saturation)
        return float(signal)

    def pair_composition_load(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 2:
            return 0.0

        products = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                products.append(
                    float(
                        bounded_diagnosticity[idx[i]]
                        * bounded_diagnosticity[idx[j]]
                    )
                )
        return float(np.mean(products)) if products else 0.0

    topology_a = triadic_topology(a_support)
    topology_b = triadic_topology(b_support)
    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)

    # The status of third and later members is flexibly but boundedly
    # calibrated. This is a count-level corroboration judgment, not a cue
    # position mechanism, and is stable across the subject's trials.
    if n_a >= 2 and n_b >= 2:
        extra_a = float(max(n_a - 2, 0))
        extra_b = float(max(n_b - 2, 0))
        evidence_a *= np.exp(
            np.clip(
                count_calibration_strength * count_calibration * extra_a,
                -4.0,
                4.0,
            )
        )
        evidence_b *= np.exp(
            np.clip(
                count_calibration_strength * count_calibration * extra_b,
                -4.0,
                4.0,
            )
        )

        # Dyadic redundancy is deliberately gap-free and equality-free: it
        # depends only on smooth joint diagnostic composition. Centering the
        # adjustment between options avoids introducing an overall bias.
        load_difference = (
            pair_composition_load(a_support)
            - pair_composition_load(b_support)
        )
        size_scale = 1.0 if n_a == n_b else 0.60
        shift = (
            pair_composition_sensitivity * size_scale * load_difference
        )
        evidence_a *= np.exp(np.clip(-0.5 * shift, -4.0, 4.0))
        evidence_b *= np.exp(np.clip(0.5 * shift, -4.0, 4.0))

    # A relational topology comparison amplifies a clustered-versus-
    # distributed distinction when both alternatives support genuinely
    # multi-cue interpretations. Logistic cardinality gates and a smooth
    # balance factor make this additional contrast negligible in 3-versus-2
    # and one-versus-many judgments, leaving count calibration in control.
    # Mild concavity preserves modest contrasts approximately linearly while
    # attenuating only the larger topology differences.
    multi_a = 1.0 / (1.0 + np.exp(-6.0 * (float(n_a) - 2.5)))
    multi_b = 1.0 / (1.0 + np.exp(-6.0 * (float(n_b) - 2.5)))
    if n_a > 0 and n_b > 0:
        log_size_ratio = np.log(float(n_a) / float(n_b))
        cardinality_balance = np.exp(-6.0 * log_size_ratio * log_size_ratio)
    else:
        cardinality_balance = 0.0
    raw_topology_contrast = topology_a - topology_b
    mild_topology_contrast = np.tanh(1.5 * raw_topology_contrast) / 1.5
    topology_shift = (
        topology_contrast_gain
        * multi_a
        * multi_b
        * cardinality_balance
        * topology_sensitivity
        / (1.0 + topology_sensitivity)
        * mild_topology_contrast
    )
    evidence_a *= np.exp(np.clip(-0.5 * topology_shift, -4.0, 4.0))
    evidence_b *= np.exp(np.clip(0.5 * topology_shift, -4.0, 4.0))

    net_a = evidence_a - evidence_b
    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total_probability = float(probs.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total_probability


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
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.32]
- validity_reliance: [0.025, 0.070]
- salience_strength: [0.12, 0.27]
- coalition_saturation: [0.74, 0.84]
- count_calibration: [-0.75, 0.25]
- count_calibration_strength: [0.36, 0.48]
- pair_composition_sensitivity: [2.10, 2.65]
- topology_sensitivity: [0.40, 8.00]
- topology_contrast_gain: [1.50, 2.75]
- cluster_bandwidth: [0.018, 0.040]
- context_scale: [0.025, 0.060]
- beta: [1.02, 1.30]
- epsilon: [0.03, 0.08]

`rationale`: This is an isolated edit to the accepted iteration-7 relational topology term. The accepted detector, absolute one-unit compression, count calibration, pair-composition settings, salience profile, and response rule are unchanged. Only the linear topology difference is replaced by tanh(1.5*difference)/1.5, with a small approximately 6% increase in the relational-gain midpoint. Around zero this transform is nearly the identity, so the gain increase can modestly strengthen the smaller contrast implicated in Experiment 2. For larger contrasts it becomes gently concave, reducing the Experiment-12 overshoot without the strong saturation and variance suppression produced by the rejected iteration-8 bundle. Keeping absolute compression and count calibration fixed avoids the Experiment-1 and Experiment-6 reversals from that rejected attempt, while restoring the accepted dyadic settings prevents topology tuning from being confounded with changes to Experiments 10–11.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Contextual Coalition-Topology family and preserves the exchangeable, non-spatial foundation successfully. Point estimates are strong in Experiments 1, 3–5, and 7–9, with especially good fits for Experiment 1 (-0.1263 versus -0.1217) and the near-zero spatial effects. Experiment 2 is also reasonably close (-0.2933 versus -0.2642). However, the central new topology mechanism is much too weak where it is most directly tested: Experiment 12 shows only 0.5435 distributed choice against 0.6692 observed. Its between-subject variance is also too low (0.0032 versus 0.0068), contrary to the recommendation for appreciable stable topology heterogeneity. Experiment 6 is displaced in a compatible direction: the model predicts too many subjects favoring the three-cue coalition (0.60 versus 0.44), suggesting insufficient compression of clustered triples. The dyadic composition effect is somewhat excessive in Experiment 10 (0.1300 versus 0.1723), while Experiment 11's smooth gap contrast is slightly too small (0.0133 versus 0.0317). Several otherwise accurate experiments also have excess variance, especially Experiment 1 (0.0272 versus 0.0117).
Rationale: Keep the current mechanism family, but strengthen and sharpen the cluster-with-context operation. The topology transform does not currently remove enough effective evidence from a near-duplicate pair to generate the observed distributed-triple advantage. Increase the effective topology-sensitivity range and reduce context_scale so a clearly separated outsider drives compression closer to its one-unit ceiling. At the same time, use a narrower or sharper cluster bandwidth so this increase applies selectively to genuinely near-duplicate pairs rather than also compressing distributed triples. Broaden topology sensitivity across subjects enough to raise Experiment 12 variance, while checking that Experiment 2 remains near its current good fit. This same adjustment should lower Experiment 6's excessive three-cue endorsement toward 0.44. Separately, slightly weaken the pair-composition discount to move Experiment 10 upward from 0.13 toward 0.17; do not restore equality-specific or sharp gap mechanisms. Finally, narrow count-calibration and/or salience dispersion rather than changing their means, since Experiment 1's point estimate is excellent but its variance is more than twice the observed value. The accepted candidate is a useful base, but its defining higher-order mechanism has not yet produced the required Experiment-12 effect.

**Outcome of this advice:** iter 1 candidate loss=0.0651 -> iter 2 candidate loss=0.0539 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to the prescribed Contextual Coalition-Topology family. The prior tuning successfully improved Experiment 2 (-0.2571 versus -0.2642), Experiment 10 (0.1563 versus 0.1723), Experiment 11 (0.0237 versus 0.0317), and Experiment 12 (0.5758 versus 0.6692), while bringing Experiment 1's variance close to the observed value. Experiments 1, 3–5, and 9 are now strong, and the spatial metrics generally remain near zero as required. Nevertheless, the two coalition-size tests remain materially wrong. Experiment 12 still substantially underpredicts the distributed-triple advantage by 0.0933, and its variance remains too low (0.0044 versus 0.0068). Experiment 6 is worse, not better: three-cue endorsement increased to 0.62 against 0.44 observed. Experiment 2 is now accurate in mean but has excessive variance (0.0138 versus 0.0087), so another indiscriminate increase in topology-sensitivity dispersion would likely damage an already good result. Smaller residual discrepancies occur in Experiments 7, 8, and 10, but they are secondary to Experiments 6 and 12.
Rationale: Build on the accepted strengthening of higher-order topology, since that direction improved aggregate loss, but do not simply increase topology sensitivity again. The remaining pattern calls for two in-family adjustments. First, reduce generic three-versus-two endorsement through a modestly more negative center for count_calibration, or an equivalent bounded third-cue corroboration adjustment. This directly targets Experiment 6 while leaving equal-size three-versus-three comparisons in Experiment 12 largely unchanged; check Experiments 3 and 8 when making this shift. Second, improve the selectivity of the topology detector rather than its global gain. The current detector computes cluster gaps on raw advertised validities even though the theory says reliability is coarsely represented. Compute topology distances in the compressed, normalized reliability-region coordinate, then retune bandwidth and context scaling so canonical duplicate-pair-plus-outsider triples receive stronger compression while other triads—especially those driving Experiment 2—receive less heterogeneous activation. This should raise Experiment 12's mean and variance without inflating Experiment 2's already excessive variance. Preserve the smooth dyadic composition mechanism: Experiment 10 is now reasonably close, so only very small pair-discount changes are warranted. Do not add equality, positional, adjacency, onset, closure, or recency mechanisms.

**Outcome of this advice:** iter 2 candidate loss=0.0539 -> iter 3 candidate loss=0.0404 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Contextual Coalition-Topology family, and the current accepted revision is a substantial aggregate improvement. The more negative count calibration successfully repaired Experiment 6, moving three-cue endorsement from 0.62 to 0.42 against 0.44 observed, while Experiments 1, 3–5, 7, and 10–11 have good point estimates. The exchangeable non-spatial foundation also remains broadly successful. However, the defining higher-order topology effect is still not captured. Experiment 12 predicts only 0.5554 distributed choice versus 0.6692 observed, with insufficient between-subject variance (0.0037 versus 0.0068). Moreover, the latest topology-coordinate revision moved Experiment 12 backward from 0.5758 and worsened Experiment 2 from a near-exact -0.2571 to -0.2225 versus -0.2642. Thus, the aggregate gain came mainly from repairing Experiment 6 and several secondary metrics, not from solving the central topology tests. Experiment 1’s mean is exact, although its variance remains elevated, while residual deviations in Experiments 8 and 9 concern small spatial contrasts and should not motivate adding prohibited spatial mechanisms.
Rationale: Keep the successful count-calibration change and the same mechanism family, but revise how triadic topology is detected rather than increasing topology sensitivity globally. The previous move to absolute gaps in the compressed, mean-normalized coordinate was accepted overall, but it worsened both principal topology point estimates; simply sharpening that same absolute-bandwidth detector should therefore not be repeated. Use a smooth, scale-free cluster-with-context statistic, such as the ratio of the nearest-pair gap to the outsider separation or total triad span, with a tunable soft threshold and temperature. Every triad necessarily has a nearest pair, so the current product of absolute duplicate affinity and contextual distinctness can activate too broadly and fails to distinguish a genuine two-in-one-region-plus-outsider topology from a relatively evenly distributed triple. A relative-gap detector should selectively compress clustered triples across experiments without relying on experiment-specific normalization scale. Also recalibrate the bounded topology-sensitivity transform to avoid rapid saturation: the present 1-exp(-sensitivity*topology) mapping can make much of the broad sensitivity range behave nearly identically, suppressing Experiment 12’s required heterogeneity. A slower Hill-type bounded mapping or a narrower operating range centered on its nonsaturated region could preserve stable subject dispersion while keeping Experiment 2 variance near its observed level. Hold pair composition and count calibration approximately fixed, since Experiments 6 and 10 are now close, and do not introduce equality, onset, closure, adjacency, or recency terms.

**Outcome of this advice:** iter 3 candidate loss=0.0404 -> iter 4 candidate loss=0.0530 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Contextual Coalition-Topology family, but it was rejected by the accept gate and is materially worse than the accepted iteration-3 base (loss 0.0530 versus 0.0404). The scale-free nearest-gap detector and Hill transform did not solve the central topology problem. Experiment 12 remains strongly underfit (0.5752 versus 0.6692), with too little between-subject variance (0.0027 versus 0.0068). It also degrades Experiment 2 further, producing -0.1996 rather than -0.2642, and undoes the successful Experiment-6 calibration: three-cue endorsement rises from the accepted base's 0.42 to 0.56 versus 0.44 observed. Secondary regressions appear in Experiment 1 (-0.1046 versus -0.1217), Experiment 7 (-0.0018 versus 0.0219), and Experiment 11 (0.0138 versus 0.0317). Experiments 3–5 and 9 remain good, and the absence of systematic spatial effects is preserved. Thus, the failure is not the exchangeable foundation but the new topology implementation: normalizing every triad by its own span makes the detector insufficiently discriminative across the actual reliability configurations, while subtracting the mean weight of the whole coalition ignores which two cues constitute the duplicate cluster.
Rationale: Because the scale-free detector/Hill-transform recommendation was rejected by the gate, do not repeat or intensify it. Build from the unchanged accepted iteration-3 base, preserving its successful count calibration, pair-composition mechanism, and non-spatial exchangeable structure. Try a different in-family adjustment at the topology-operation stage: make compression pair-resolved rather than subtracting one generic whole-coalition mean unit. Let the accepted detector assign smooth responsibility to each candidate duplicate pair, then compress that pair toward one effective representative by subtracting a bounded function of the two clustered cues' weights (for example, their soft minimum or approximately half their sum), with the outsider left intact. This implements the prescribed claim more literally and retains information about whether the redundant pair is weak, strong, or asymmetric. Tune the compression amplitude in a nonsaturated range while keeping topology-sensitivity dispersion appreciable; do not globally broaden sensitivity, since Experiment 2 variance is already near or above target in the accepted base. Jointly verify that stronger pair-resolved compression raises Experiment 12 and makes Experiment 2 more negative without disturbing Experiment 6; if needed, make only a small compensating adjustment to the already successful count-calibration center. Hold the dyadic composition settings approximately fixed, as Experiment 10 is close, and add no equality-specific, positional, adjacency, onset, closure, or recency terms.

**Outcome of this advice:** iter 4 candidate loss=0.0530 -> iter 5 candidate loss=0.0447 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Contextual Coalition-Topology family, but the accept gate rejected it: loss 0.0447 is worse than the iteration-3 base at 0.0404. Pair-resolved compression produced only negligible gains on the defining topology tests relative to that base: Experiment 12 moved from 0.5554 to 0.5606 but remains far below 0.6692, and Experiment 2 moved from -0.2225 to -0.2308 but remains short of -0.2642. Experiment 12 variance is also still too low (0.0043 versus 0.0068). Those small gains came with regressions in Experiment 6 (0.36 versus 0.44, compared with 0.42 in the accepted base), Experiment 11 (0.0108 versus 0.0317), and several otherwise well-fit metrics. Experiments 3, 4, 7, and 9 remain good, and the model correctly avoids systematic spatial effects. Thus, the exchangeable foundation and count calibration are viable; the topology detector still fails to distinguish the experimentally relevant clustered and distributed reliability schemas strongly and selectively enough.
Rationale: Because the pair-resolved subtraction proposed in the previous critique was rejected, do not repeat or intensify that change. Build again from the unchanged iteration-3 base and preserve its count-calibration center, dyadic composition settings, and generic compression rule. Try a different in-family representation of reliability regions: replace nearest-pair responsibility with smooth occupancy over a small set of shared, anchored reliability-schema regions. A triad should receive high topology activation only when two cues have strong overlapping membership in one region and the third has strong membership in another; three cues spread across three regions should receive little activation. Use broad differentiable kernels or soft bins rather than equality tests, validity-gap thresholds, or within-triad span normalization. Anchoring the regions across experiments should avoid the distortions produced by experiment-wise mean normalization while implementing the arbiter's region-topology claim directly. Tune region width and topology gain jointly, but keep topology-sensitivity dispersion near the accepted base rather than globally broadening it, since Experiment 2 variance is already close to or above target. The immediate acceptance targets should be a substantial rise in Experiment 12 and a more negative Experiment 2 without moving Experiment 6 away from 0.44. Leave the pair mechanism nearly fixed because Experiments 10–11 were reasonably close in the accepted base, and do not add spatial, equality-specific, adjacency, onset, closure, entry/exit, or recency terms.

**Outcome of this advice:** iter 5 candidate loss=0.0447 -> iter 6 candidate loss=0.0846 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The current candidate is decisively worse than the accepted iteration-3 base (loss 0.0846 versus 0.0404) and was rejected by the gate. The anchored-region revision fails on all major coalition tests. Experiment 12 falls to near indifference (0.5102 versus 0.6692), so it does not distinguish distributed from clustered triples; its variance is also too low (0.0026 versus 0.0068). Experiment 2 likewise underexpresses the distributed-triple advantage (-0.1938 versus -0.2642). More seriously, the operation spills into unequal-cardinality comparisons: Experiment 6 collapses to 0.08 versus 0.44, with severely deficient variance, and Experiment 1 shifts from an exact fit in iteration 3 to -0.0096 versus -0.1217. Experiments 3–5, 7, 9, and 11 remain broadly compatible with the exchangeable, non-spatial foundation, while Experiment 10 is only modestly low. Thus, the failure is localized to the new topology implementation, not the overall mechanism family. The fixed soft-bin detector likely activates broadly or similarly for both coalitions: summing three pair assignments and clipping their union can erase the topology contrast, while independent one-unit subtraction can strongly perturb a three-versus-two contest even when the relative topology evidence is uninformative.
Rationale: The previous recommendation to use anchored reliability regions was strongly rejected, so do not repeat it or further tune its region centers and widths. Build from the unchanged accepted iteration-3 base, retaining its successful count calibration, pair-composition settings, generic one-unit compression, and exchangeable non-spatial structure. After several rejected detector replacements, make a smaller in-family change to how the accepted topology signal enters choice. Compute the topology operation as a centered contrast between the two coalitions rather than allowing two independent absolute penalties to dominate. For equal-cardinality contests with both coalitions containing at least three cues, apply a smooth contrast gain or temperature to topology_A minus topology_B; this can magnify the small but correctly directed distinction needed in Experiments 2 and 12 without globally increasing topology sensitivity. Attenuate this contrast smoothly when cardinalities differ, so topology does not overwhelm the established count calibration in Experiment 6 or contaminate the one-versus-many conditions in Experiment 1. Avoid a hard experimental-condition switch: use a bounded cardinality-balance factor based only on coalition sizes. Keep topology-sensitivity dispersion near the accepted range and tune the contrast temperature in a nonsaturated regime rather than broadening subject heterogeneity globally. Immediate acceptance checks should require recovery of Experiment 1 near -0.12 and Experiment 6 near 0.44, alongside meaningful improvements over iteration 3 in Experiments 2 and 12. Do not alter the dyadic mechanism substantially or introduce equality, position, adjacency, onset, closure, entry/exit, or recency effects.

**Outcome of this advice:** iter 6 candidate loss=0.0846 -> iter 7 candidate loss=0.0391 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Contextual Coalition-Topology family, and the centered relational topology adjustment was successful: it produced a new best loss (0.0391), preserved the exchangeable non-spatial pattern, and raised Experiment 12 from severe underprediction to a clear distributed-triple advantage. Experiments 3–5 and 9 are close, Experiment 6 remains reasonably calibrated (0.40 versus 0.44), and the topology-related variances are now near the data in Experiments 2 and 12. However, the relational correction overshoots its principal target: Experiment 12 is 0.7329 versus 0.6692. At the same time, Experiment 2 remains too weak in the relevant direction (-0.2346 versus -0.2642). Thus, a single linear topology-contrast gain is amplifying the large Experiment-12 contrast too much while not amplifying smaller topology contrasts enough. There are also meaningful residual misses in Experiment 1 (-0.0979 versus -0.1217) and Experiment 11 (-0.0075 versus 0.0317). Experiment 10 remains slightly too strongly averse to the target pair (0.1584 versus 0.1723), although its direction is correct. Spatial contrasts remain suitably small; Experiments 7 and 8 should not motivate prohibited positional mechanisms.
Rationale: Build on the accepted centered-contrast change, since that advice improved the running best; do not replace the accepted topology detector or return to the rejected scale-free, pair-resolved, or anchored-region variants. The next adjustment should make the relational topology contrast smoothly concave rather than linear—for example, apply a signed tanh or another bounded transform to topology_a-topology_b, with a separately tuned scale and gain. Calibrate it to give relatively more gain to modest contrasts like those driving Experiment 2 while saturating the larger contrast driving Experiment 12. This should move Experiment 2 more negative without pushing Experiment 12 further above 0.67. Keep topology-sensitivity dispersion close to its current range because the resulting variances are already near target. In addition, slightly reduce the independent absolute triadic compression while retaining the centered relational term. That should restore some multi-cue evidence in Experiment 1 and move Experiment 6 modestly upward toward 0.44, without removing the equal-cardinality topology advantage. Finally, make only a small weakening of pair-composition sensitivity: Experiment 10 is somewhat too extreme, and weakening it should move the estimate upward toward 0.172 while pulling Experiment 11's erroneous negative gap contrast toward zero. Avoid a large dyadic change, since earlier accepted iterations already established that this mechanism is broadly calibrated. Preserve the absence of equality-specific and spatial mechanisms.

**Outcome of this advice:** iter 7 candidate loss=0.0391 -> iter 8 candidate loss=0.0408 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Contextual Coalition-Topology family, and its central topology predictions are individually promising: Experiment 12 improves substantially from the accepted base's overshoot of 0.7329 to 0.6488 versus 0.6692, while Experiment 2 moves slightly toward the observed value (-0.2379 versus -0.2642). Experiment 10 also becomes essentially exact, and Experiment 11 moves in the correct direction. Nevertheless, the candidate was rejected because aggregate loss rose from 0.0391 to 0.0408. The bundled changes produced important collateral regressions: Experiment 6 crosses from 0.40 to 0.52 versus 0.44, Experiment 1 crosses from underexpressing to overexpressing the effect (-0.1375 versus -0.1217), and Experiments 5, 7, and 8 move farther from their near-zero targets. Experiment 12 variance also falls from 0.0059 to 0.0045 against 0.0068 observed, indicating that the saturating transform suppresses some of the desired topology heterogeneity. Thus, the accepted centered relational mechanism is viable, but the simultaneous reduction in absolute compression, stronger concavity, and dyadic retuning was too large and makes it unclear which component caused the aggregate regression.
Rationale: Build from the unchanged accepted iteration-7 base. The previous recommendation's full bundle was rejected, so do not repeat the reduced absolute triadic compression: it pushed Experiment 6 above target and drove Experiment 1 past its target in the opposite direction. Also restore the accepted pair-composition settings for this next test rather than combining another dyadic change with topology tuning; although the weaker pair discount improved Experiments 10–11, its net contribution cannot be identified from the rejected bundle. Make one smaller, isolated adjustment to the accepted relational term: use a mild interpolation between its original linear contrast and the rejected strongly saturating transform, or equivalently use a substantially larger saturation scale with a modestly adjusted gain. The operating point should lie between iterations 7 and 8, aiming to lower Experiment 12 from 0.733 toward 0.669 without reducing its variance below roughly 0.006, while preserving or slightly strengthening Experiment 2. This is preferable to repeating the current strong tanh change, which undershot Experiment 12 and barely improved Experiment 2. Keep count calibration, absolute compression, topology-sensitivity dispersion, and the exchangeable non-spatial structure at the accepted settings. Treat the Experiment 7–8 deviations as noise or spillover rather than adding prohibited positional mechanisms. If the isolated mild concavity is accepted, pair-composition sensitivity can subsequently be adjusted in a separate small step to address Experiments 10–11.

**Outcome of this advice:** iter 8 candidate loss=0.0408 -> iter 9 candidate loss=0.0363 -> the gate marked it ACCEPTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed Contextual Coalition-Topology family and is the best iteration so far (loss 0.0363). The isolated mild-concavity revision was successful: Experiment 2 improved to -0.2492 versus -0.2642, Experiment 12 retained the strong distributed-triple advantage with nearly correct variance (0.7160, var 0.0064, versus 0.6692, var 0.0068), and Experiment 6 became exact at 0.44. Experiments 1, 3–5, 8–10 are also close, while the exchangeable model continues to avoid systematic spatial effects. Nonetheless, the defining topology contrast still overshoots Experiment 12 by 0.0469. Experiment 11 is the largest additional point-estimate failure: the predicted gap contrast is slightly negative (-0.0071) rather than the observed positive 0.0317, with excessive variance. Experiment 7 also differs by -0.0289, but its small positional contrast is not consistent across iterations and should not motivate a prohibited spatial mechanism.
Rationale: Build on the accepted iteration-9 candidate. The prior advice to make the relational topology transform only mildly concave was accepted and improved aggregate fit, so extend that successful direction by one small, isolated step: modestly increase the curvature of the signed topology-contrast transform while holding its near-zero slope approximately fixed. This should selectively attenuate the larger contrast driving Experiment 12 toward 0.669 while leaving the smaller Experiment-2 contrast nearly linear and preserving its improved value and variance. Do not again reduce absolute triadic compression or alter count calibration, since their current combination gives an exact Experiment-6 fit and a close Experiment-1 fit. Keep topology-sensitivity dispersion unchanged because Experiment 12 variance is now essentially correct. Also avoid bundling this with pair-composition changes: Experiment 10 is already close, and the rejected iteration-8 bundle did not identify whether dyadic retuning caused its collateral regressions. If the topology-only step is accepted, Experiment 11 can subsequently be addressed with a separate, very small adjustment to the smooth pair-composition normalization or sensitivity, without adding equality-specific or sharp gap mechanisms. Treat Experiment 7 as residual sampling/spillover and preserve the exchangeable, non-spatial architecture.

**Outcome of this advice:** iter 9 candidate loss=0.0363 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Contextual Reliability-Region Compression proposes that people integrate every discriminating expert rating into an exchangeable option-level coalition while representing advertised validities on a strongly compressed reliability-region scale. Coalition evidence grows sublinearly, and stable count calibration can make a third cue either corroborative or partially redundant. Redundancy is order-specific in cardinality but not spatial position: two-cue coalitions receive only a smooth joint-diagnostic composition discount, whereas coalitions of three or more undergo a contextual topology operation. In the latter operation, two cues occupying nearly the same represented reliability region are compressed toward one informational unit when another cue occupies a distinct region. In equal-cardinality contests between multi-cue coalitions, people additionally compare the coalitions' topology relationally. This relational comparison is moderately concave: modest clustered-versus-distributed differences are transmitted approximately linearly, whereas larger differences receive stronger diminishing amplification. The comparison fades smoothly when coalition cardinalities differ, preserving independently calibrated count judgments. Topology sensitivity is stable and heterogeneous across subjects, while pair-composition sensitivity is comparatively narrow. Stable, zero-centered idiosyncratic cue salience permits individual configuration preferences without population-level onset, closure, recency, adjacency, entry, or exit effects. Because subjects receive no outcome feedback, these representations remain fixed across trials.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=np.float64)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Contextual Reliability-Region Compression expects shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    raw_salience = np.asarray(\n        parameters[\"cue_salience_profile\"], dtype=np.float64\n    )\n    if raw_salience.ndim != 1 or raw_salience.size != n_features:\n        raise ValueError(\n            f\"cue_salience_profile length {raw_salience.size} != \"\n            f\"n_features {n_features}.\"\n        )\n\n    validity_compression = float(parameters[\"validity_compression\"])\n    validity_reliance = float(parameters[\"validity_reliance\"])\n    salience_strength = float(parameters[\"salience_strength\"])\n    coalition_saturation = float(parameters[\"coalition_saturation\"])\n    count_calibration = float(parameters[\"count_calibration\"])\n    count_calibration_strength = float(parameters[\"count_calibration_strength\"])\n    pair_composition_sensitivity = float(\n        parameters[\"pair_composition_sensitivity\"]\n    )\n    topology_sensitivity = float(parameters[\"topology_sensitivity\"])\n    topology_contrast_gain = float(parameters[\"topology_contrast_gain\"])\n    cluster_bandwidth = float(parameters[\"cluster_bandwidth\"])\n    context_scale = float(parameters[\"context_scale\"])\n    beta = float(parameters[\"beta\"])\n    epsilon = float(parameters[\"epsilon\"])\n\n    # Reliability is encoded on a diagnosticity scale, but differences are\n    # strongly compressed. Mean normalization prevents an experiment's\n    # overall validity level from changing the generic evidence scale.\n    v = np.clip(validities, 0.500001, 0.999999)\n    log_odds = np.log(v / (1.0 - v))\n    represented_diagnosticity = np.power(\n        np.maximum(log_odds, 1e-12), validity_compression\n    )\n    represented_diagnosticity /= max(\n        float(np.mean(represented_diagnosticity)), 1e-12\n    )\n\n    reliability_weight = (\n        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)\n        + validity_reliance * represented_diagnosticity\n    )\n\n    # Components are exchangeably sampled and centered within each subject.\n    # Thus salience is stable and can generate individual differences, but\n    # no display position has a positive population-level privilege.\n    centered_salience = raw_salience - float(np.mean(raw_salience))\n    attention = np.exp(\n        np.clip(salience_strength * centered_salience, -10.0, 10.0)\n    )\n    attention /= max(float(np.mean(attention)), 1e-12)\n\n    cue_weights = reliability_weight * attention\n    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)\n\n    difference = a - b\n    a_support = np.flatnonzero(difference > 0.0)\n    b_support = np.flatnonzero(difference < 0.0)\n    n_a = int(a_support.size)\n    n_b = int(b_support.size)\n\n    if n_a == 0 and n_b == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    bounded_diagnosticity = np.clip(2.0 * v - 1.0, 0.0, 1.0)\n\n    def triadic_topology(indices):\n        idx = np.asarray(indices, dtype=int)\n        n = int(idx.size)\n        if n < 3:\n            return 0.0\n\n        scores = []\n        # Topological distance is computed in the same compressed,\n        # mean-normalized reliability-region coordinate used by cognition,\n        # rather than in raw advertised-validity units.\n        for x in range(n - 2):\n            for y in range(x + 1, n - 1):\n                for z in range(y + 1, n):\n                    triad = np.sort(\n                        represented_diagnosticity[idx[[x, y, z]]]\n                    )\n                    lower_gap = float(triad[1] - triad[0])\n                    upper_gap = float(triad[2] - triad[1])\n\n                    if lower_gap <= upper_gap:\n                        cluster_gap = lower_gap\n                        cluster_center = 0.5 * float(triad[0] + triad[1])\n                        outsider = float(triad[2])\n                    else:\n                        cluster_gap = upper_gap\n                        cluster_center = 0.5 * float(triad[1] + triad[2])\n                        outsider = float(triad[0])\n\n                    separation = abs(outsider - cluster_center)\n                    duplicate_affinity = np.exp(\n                        -0.5 * (cluster_gap / cluster_bandwidth) ** 2\n                    )\n                    contextual_distinctness = separation / (\n                        separation + context_scale\n                    )\n                    scores.append(\n                        float(duplicate_affinity * contextual_distinctness)\n                    )\n\n        if not scores:\n            return 0.0\n\n        # A smooth probabilistic union detects whether the coalition contains\n        # at least one duplicate-pair-in-context relation without privileging\n        # a cue identity, ordering, or spatial adjacency.\n        scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 0.999999)\n        return float(1.0 - np.prod(1.0 - scores))\n\n    def coalition_signal(indices):\n        idx = np.asarray(indices, dtype=int)\n        n = int(idx.size)\n        if n == 0:\n            return 0.0\n\n        total = float(np.sum(cue_weights[idx]))\n\n        if n >= 3:\n            topology = triadic_topology(idx)\n            # Sensitivity maps smoothly to the fraction of one informational\n            # unit removed. At its upper limit a near-duplicate pair is worth\n            # approximately one unit rather than two; all cues nevertheless\n            # retain a positive contribution.\n            compression_fraction = (\n                1.0 - np.exp(-topology_sensitivity * topology)\n            )\n            coalition_unit = float(np.mean(cue_weights[idx]))\n            total -= compression_fraction * coalition_unit\n            total = max(total, 1e-12)\n\n        signal = total / (float(n) ** coalition_saturation)\n        return float(signal)\n\n    def pair_composition_load(indices):\n        idx = np.asarray(indices, dtype=int)\n        n = int(idx.size)\n        if n < 2:\n            return 0.0\n\n        products = []\n        for i in range(n - 1):\n            for j in range(i + 1, n):\n                products.append(\n                    float(\n                        bounded_diagnosticity[idx[i]]\n                        * bounded_diagnosticity[idx[j]]\n                    )\n                )\n        return float(np.mean(products)) if products else 0.0\n\n    topology_a = triadic_topology(a_support)\n    topology_b = triadic_topology(b_support)\n    evidence_a = coalition_signal(a_support)\n    evidence_b = coalition_signal(b_support)\n\n    # The status of third and later members is flexibly but boundedly\n    # calibrated. This is a count-level corroboration judgment, not a cue\n    # position mechanism, and is stable across the subject's trials.\n    if n_a >= 2 and n_b >= 2:\n        extra_a = float(max(n_a - 2, 0))\n        extra_b = float(max(n_b - 2, 0))\n        evidence_a *= np.exp(\n            np.clip(\n                count_calibration_strength * count_calibration * extra_a,\n                -4.0,\n                4.0,\n            )\n        )\n        evidence_b *= np.exp(\n            np.clip(\n                count_calibration_strength * count_calibration * extra_b,\n                -4.0,\n                4.0,\n            )\n        )\n\n        # Dyadic redundancy is deliberately gap-free and equality-free: it\n        # depends only on smooth joint diagnostic composition. Centering the\n        # adjustment between options avoids introducing an overall bias.\n        load_difference = (\n            pair_composition_load(a_support)\n            - pair_composition_load(b_support)\n        )\n        size_scale = 1.0 if n_a == n_b else 0.60\n        shift = (\n            pair_composition_sensitivity * size_scale * load_difference\n        )\n        evidence_a *= np.exp(np.clip(-0.5 * shift, -4.0, 4.0))\n        evidence_b *= np.exp(np.clip(0.5 * shift, -4.0, 4.0))\n\n    # A relational topology comparison amplifies a clustered-versus-\n    # distributed distinction when both alternatives support genuinely\n    # multi-cue interpretations. Logistic cardinality gates and a smooth\n    # balance factor make this additional contrast negligible in 3-versus-2\n    # and one-versus-many judgments, leaving count calibration in control.\n    # Moderate concavity preserves small contrasts approximately linearly\n    # while selectively attenuating larger topology differences.\n    multi_a = 1.0 / (1.0 + np.exp(-6.0 * (float(n_a) - 2.5)))\n    multi_b = 1.0 / (1.0 + np.exp(-6.0 * (float(n_b) - 2.5)))\n    if n_a > 0 and n_b > 0:\n        log_size_ratio = np.log(float(n_a) / float(n_b))\n        cardinality_balance = np.exp(-6.0 * log_size_ratio * log_size_ratio)\n    else:\n        cardinality_balance = 0.0\n    raw_topology_contrast = topology_a - topology_b\n    mild_topology_contrast = np.tanh(1.8 * raw_topology_contrast) / 1.8\n    topology_shift = (\n        topology_contrast_gain\n        * multi_a\n        * multi_b\n        * cardinality_balance\n        * topology_sensitivity\n        / (1.0 + topology_sensitivity)\n        * mild_topology_contrast\n    )\n    evidence_a *= np.exp(np.clip(-0.5 * topology_shift, -4.0, 4.0))\n    evidence_b *= np.exp(np.clip(0.5 * topology_shift, -4.0, 4.0))\n\n    net_a = evidence_a - evidence_b\n    logits = np.array(\n        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - epsilon) * probs + epsilon * 0.5\n    probs = np.maximum(probs, 0.0)\n    total_probability = float(probs.sum())\n    if not np.isfinite(total_probability) or total_probability <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total_probability\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.maximum(probs, 0.0)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "cue_salience_profile": "[(-1.0, 1.0)] * n_features",
    "validity_compression": "[0.16, 0.32]",
    "validity_reliance": "[0.025, 0.070]",
    "salience_strength": "[0.12, 0.27]",
    "coalition_saturation": "[0.74, 0.84]",
    "count_calibration": "[-0.75, 0.25]",
    "count_calibration_strength": "[0.36, 0.48]",
    "pair_composition_sensitivity": "[2.10, 2.65]",
    "topology_sensitivity": "[0.40, 8.00]",
    "topology_contrast_gain": "[1.50, 2.75]",
    "cluster_bandwidth": "[0.018, 0.040]",
    "context_scale": "[0.025, 0.060]",
    "beta": "[1.02, 1.30]",
    "epsilon": "[0.03, 0.08]"
  },
  "rationale": "This is an isolated one-line functional edit to the accepted iteration-9 model. The curvature coefficient in the slope-preserving transform tanh(k*d)/k is increased modestly from 1.5 to 1.8. Because the derivative at zero remains exactly one, small topology contrasts such as those implicated in Experiment 2 remain approximately linear. Larger contrasts, especially the one driving Experiment 12's overshoot, are attenuated somewhat more strongly. Absolute triadic compression, count calibration, topology-sensitivity dispersion, pair composition, salience, and response parameters are unchanged, preserving the exact Experiment-6 result, the close Experiment-1 fit, and the nearly correct Experiment-12 variance. Experiment 11 is deliberately left for a later isolated dyadic adjustment rather than confounding this topology test."
}
```

## Usage

```json
{
  "input_tokens": 37962,
  "output_tokens": 3664
}
```
