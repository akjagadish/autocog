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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_7`).

Replace pi_7 with a new task-invariant competitor to pi_5, provisionally an Active-Set Conditional Competition theory. It should posit a stable, instruction-based representation with no feedback-free learning of cue dependence. On each trial, subjects first identify the currently discriminating cues; communicated validity, validity gaps, coalition geometry, and active top-validity ties then determine which cues are accessible and how they compete. Coalition accumulation should exhibit smooth, context-sensitive redundancy rather than a universal restoration threshold. Sparse opposition should receive a compact-challenge enhancement that can peak at one or two opponents, while additional opponents become increasingly redundant and can produce only gradual, context-dependent anchor recovery. This mechanism should predict an Experiment-1 contrast near the observed modest positive value rather than a jump near 0.67. A bounded presentation-order gate should be activated by a tie among currently active top-validity cues, not by inactive co-top cues and not as a generic primacy weight; this is required to reproduce the large positive Experiment-2 activation effect while preserving the null primacy crossover in Experiment 6 and the order effects in Experiments 3 and 7. Absolute anchor credibility and opponent-to-anchor validity gaps should modulate compact challenge continuously, allowing strong anchor adherence in Experiment 11 but pronounced singleton challenge in Experiment 8. Stable subject differences should be concentrated in active-tie gating and compact-challenge sensitivity, with correlated mixture types or random effects broad enough to match the observed variances in Experiments 1, 2, 11, and 12, but constrained so they do not generate spurious primacy crossovers. The new theory should explicitly exclude trial-history anti-concurrence updates unless future feedback-free exposure experiments establish them.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.667)

**Description**
Static Configuration-Contingent Integration (SCCI) claims that people form a fixed, instruction-based representation of cue reliability and apply it to the directional evidence profile on each trial. Each discriminating cue contributes reliability-weighted evidence, so independently supporting cues accumulate, while concordant cues are compressed as potentially redundant. This compression increases moderately convexly with coalition size, reflecting the declining informational value of each additional concordant cue. A bounded configuration gate supplements integration: strong protection of a uniquely most-valid cue is restricted to profiles in which that cue is isolated, has a sufficiently large communicated-validity advantage, and faces multiple concordant opponents. Cue uniqueness functions primarily as a tie detector, while a separate raw validity-gap criterion determines whether restoration is warranted. The transition to restoration is sharp, allowing an isolated anchor to be weak against one credible challenger but strongly protected against a redundant multi-cue bloc. Compact challenge amplification is strictly limited to isolated anchors and therefore cannot create general primacy effects when the anchor has allies. When the most-valid discriminating cues have equal communicated validity, they remain exchangeable in the integrator, but stable display order supplies a bounded tie-breaking anchor. Choices are history-invariant because concurrence, anti-concurrence, and discrimination frequency are not learned without correctness feedback. Stable profile heterogeneity arises from subject-level variation in validity sensitivity, coalition integration, redundancy, isolated-anchor challenge and restoration, tie-order bias, temperature, and lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCCI expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    coalition_accumulation = float(parameters["coalition_accumulation"])
    redundancy_discount = float(parameters["redundancy_discount"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    profile_balance_gain = float(parameters["profile_balance_gain"])
    anchor_gate_strength = float(parameters["anchor_gate_strength"])
    challenge_strength = float(parameters["challenge_strength"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap_threshold = float(parameters["restoration_gap_threshold"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    tie_order_bias = float(parameters["tie_order_bias"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    # Communicated validity is transformed into reliability evidence. The
    # subtraction only fixes scale and does not alter validity ordering.
    v = np.clip(validities, 0.500001, 0.999999)
    log_odds = np.log(v / (1.0 - v))
    active_reference = float(np.max(log_odds[active]))
    cue_weight = np.exp(
        validity_sensitivity * (log_odds - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Accumulation represents multiple nominally independent reasons.
        # A moderately convex denominator captures increasing redundancy
        # among additional cues in the same directional coalition.
        mean_reliability = float(np.mean(cue_weight[indices]))
        accumulated = mean_reliability * (float(n) ** coalition_accumulation)
        redundancy = 1.0 + redundancy_discount * (
            float(n - 1) ** redundancy_curvature
        )
        return accumulated / redundancy

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # Directional minorities receive only a very small configuration-based
    # salience adjustment. Most coalition effects arise from accumulation,
    # redundancy, and the isolated-anchor mechanism below.
    if n_a > 0 and n_b > 0 and n_a != n_b:
        if n_a < n_b:
            ratio = float(n_b) / float(n_a)
            support_a *= ratio ** profile_balance_gain
        else:
            ratio = float(n_a) / float(n_b)
            support_b *= ratio ** profile_balance_gain

    total_support = support_a + support_b
    core_evidence = (support_b - support_a) / max(total_support, 1e-12)

    # The validity anchor is the most-valid currently discriminating cue.
    # Equal-validity candidates have identical integrative weights; order is
    # consulted only to select a bounded tie anchor.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]
    anchor_cue = int(np.min(best_candidates))
    anchor_direction = float(directions[anchor_cue])

    # Uniqueness is a tie detector rather than a second graded validity-gap
    # penalty. The separate raw-gap sigmoid below selects strong anchors.
    if active.size > 1:
        sorted_active = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active[0] - sorted_active[1])
    else:
        validity_gap = 0.5
    unique_anchor = float(best_candidates.size == 1)

    if anchor_direction > 0:
        anchor_side_count = n_b
        opposition_count = n_a
        anchor_side_support = support_b
        opposition_support = support_a
    else:
        anchor_side_count = n_a
        opposition_count = n_b
        anchor_side_support = support_a
        opposition_support = support_b

    isolated_anchor = float(anchor_side_count == 1)

    if opposition_count == 0:
        weak_opposition = 1.0
    else:
        relative_opposition = opposition_support / max(
            anchor_side_support + opposition_support, 1e-12
        )
        weak_opposition = np.exp(-4.0 * relative_opposition)

    # A single independent opponent receives extra challenge amplification
    # only when the validity anchor is genuinely isolated. Non-isolated
    # profiles are governed by ordinary static coalition integration.
    if opposition_count > 0:
        compact_challenge = np.exp(
            -4.0 * float(opposition_count - 1) ** 2
        )
        challenge_gain = isolated_anchor * (
            coalition_accumulation + challenge_strength
        )
        challenged_opposition = opposition_support * (
            1.0
            + unique_anchor * challenge_gain * compact_challenge
        )
        anchor_margin = (
            anchor_side_support - challenged_opposition
        ) / max(anchor_side_support + challenged_opposition, 1e-12)
        core_evidence = anchor_direction * anchor_margin

    # Strong restoration is specific to an isolated and clearly superior
    # anchor facing multiple concordant opponents. Count and raw validity-gap
    # thresholds vary across subjects, with a sharp gap transition separating
    # modest-gap profiles from genuinely strong-anchor configurations.
    opposition_restoration = 1.0 / (
        1.0 + np.exp(
            -restoration_slope
            * (float(opposition_count) - restoration_threshold)
        )
    )
    gap_restoration = 1.0 / (
        1.0 + np.exp(
            -restoration_gap_slope
            * (validity_gap - restoration_gap_threshold)
        )
    )
    unique_gate_profile = (
        0.01 * weak_opposition
        + isolated_anchor
        * restoration_strength
        * opposition_restoration
        * gap_restoration
    )
    unique_gate = anchor_gate_strength * unique_anchor * float(
        np.clip(unique_gate_profile, 0.0, 1.0)
    )

    # Stable order matters only when multiple currently discriminating cues
    # share the maximum communicated validity.
    tied_active_top = float(best_candidates.size > 1)
    tie_gate = tie_order_bias * tied_active_top

    gate = 1.0 - (1.0 - unique_gate) * (1.0 - tie_gate)
    gate = float(np.clip(gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * anchor_direction
    )

    logits = beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence], dtype=np.float64
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs.astype(np.float64)


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
- validity_sensitivity: [0.03, 0.35]
- coalition_accumulation: [0.98, 1.16]
- redundancy_discount: [0.30, 0.42]
- redundancy_curvature: [1.15, 1.45]
- profile_balance_gain: [0.0, 0.0]
- anchor_gate_strength: [0.90, 0.995]
- challenge_strength: [0.8, 2.8]
- restoration_strength: [0.95, 1.0]
- restoration_threshold: [1.15, 1.55]
- restoration_slope: [16.0, 30.0]
- restoration_gap_threshold: [0.055, 0.075]
- restoration_gap_slope: [160.0, 240.0]
- tie_order_bias: [0.58, 0.85]
- beta: [0.9, 3.4]
- lapse: [0.0, 0.12]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.9600 (var=0.0284)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.4562 (var=0.0022)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.0462 (var=0.0019)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.2400 (var=0.1824)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.7727 (var=0.0071)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.1356 (var=0.0077)
- Experiment 7: real=0.0052 (var=0.0075) vs this=0.0092 (var=0.0215)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0018 (var=0.0001)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.8094 (var=0.0121)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.7306 (var=0.0084)
- Experiment 11: real=0.1093 (var=0.0652) vs this=-0.0085 (var=0.0107)
- Experiment 12: real=0.3533 (var=0.0427) vs this=0.2700 (var=0.0168)


---

### `pi_5` (overall score: 0.522)

**Description**
Adaptive Distinctive Coalition Gating (ADCG) proposes that binary-feature decisions arise from task-adaptive cue weights combining communicated validity, presentation prominence, distributional rarity, and learned cue dependence. Directionally concordant cues are compressed as potentially redundant, while sparse dissent can dominate when its cues have reliably and repeatedly opposed the anchor's cues. Learned anti-concurrence is reliability-sensitive: disagreement near chance is strongly suppressed, whereas well-observed disagreement above a subject-specific threshold receives a bounded nonlinear gain. A profile-driven gate interpolates between coalition comparison and a lexicographic anchor, preserving anchoring against large redundant coalitions while permitting diagnostic opposition to suppress it. For compact opposition, gate suppression is controlled by a selective summary of the strongest reliably learned opposing relationship rather than an indiscriminate average across all cue pairs; larger coalitions retain mean aggregation and strong size decay. In addition, a unique, early-presented validity anchor loses accessibility within the coalition core when it faces compact, reliably anti-concurrent opposition. This attenuation targets residual anchoring without increasing dissent support and vanishes for tied-validity anchors, late anchors, and large opposing coalitions. When top validities are tied, presentation order supplies a bounded anchor but the distinctive-coalition core is attenuated by a stable subject-specific amount. Stable variation in validity attention, order prominence, anti-concurrence threshold and gain, diagnostic selectivity, anchor attenuation, tied-top attenuation, redundancy, gating, and temperature produces structured individual differences without experiment-specific strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"ADCG expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_attention = float(parameters["validity_attention"])
    order_prominence = float(parameters["order_prominence"])
    distribution_attention = float(parameters["distribution_attention"])
    distinctiveness = float(parameters["distinctiveness"])
    dissent_bonus = float(parameters["dissent_bonus"])
    anti_concurrence_attention = float(parameters["anti_concurrence_attention"])
    anti_concurrence_threshold = float(parameters["anti_concurrence_threshold"])
    diagnostic_selectivity = float(parameters["diagnostic_selectivity"])
    anchor_attenuation = float(parameters["anchor_attenuation"])
    redundancy = float(parameters["redundancy"])
    lex_strength = float(parameters["lex_strength"])
    gate_sharpness = float(parameters["gate_sharpness"])
    gate_rarity = float(parameters["gate_rarity"])
    tie_gate_ceiling = float(parameters["tie_gate_ceiling"])
    tie_core_attenuation = float(parameters["tie_core_attenuation"])
    beta = float(parameters["beta"])

    # Communicated validity is represented on a reliability-evidence scale.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    reliability = (reliability - np.mean(reliability)) / (
        np.std(reliability) + 0.35
    )

    # Estimate how often each cue discriminates and how often cue pairs
    # concur. Symmetric pseudocounts make the model well-defined early in
    # the block while allowing randomized task experience to update weights.
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    discr_count = np.ones(n_features, dtype=float) * 2.0
    pair_both = np.ones((n_features, n_features), dtype=float) * 2.0
    pair_same = np.ones((n_features, n_features), dtype=float)

    for i in range(n_past):
        pa = np.asarray(past_a[i], dtype=float)
        pb = np.asarray(past_b[i], dtype=float)
        if pa.shape[0] != n_features or pb.shape[0] != n_features:
            continue
        s = np.sign(pb - pa)
        on = np.flatnonzero(s != 0)
        discr_count[on] += 1.0
        for q, j in enumerate(on):
            for k in on[q + 1:]:
                pair_both[j, k] += 1.0
                pair_both[k, j] += 1.0
                if s[j] == s[k]:
                    pair_same[j, k] += 1.0
                    pair_same[k, j] += 1.0

    discr_rate = discr_count / float(n_past + 4.0)
    reference_rate = float(np.mean(discr_rate))
    rarity = np.clip(reference_rate / discr_rate, 0.35, 3.5)

    # Learned anti-concurrence is high only for cue pairs that repeatedly
    # discriminate in opposite directions. Pair-specific observation counts
    # determine reliability, and a steep subject-specific threshold removes
    # weak or chance-level disagreement while retaining systematic dissent.
    anti_concurrence = np.clip(1.0 - pair_same / pair_both, 0.0, 1.0)
    pair_observations = np.maximum(pair_both - 2.0, 0.0)
    pair_confidence = pair_observations / (pair_observations + 2.0)
    pair_confidence *= np.clip((pair_observations - 1.0) / 3.0, 0.0, 1.0)
    threshold_baseline = 1.0 / (
        1.0 + np.exp(-18.0 * (0.5 - anti_concurrence_threshold))
    )
    thresholded_anti = 1.0 / (
        1.0 + np.exp(-18.0 * (anti_concurrence - anti_concurrence_threshold))
    )
    thresholded_anti = np.clip(
        (thresholded_anti - threshold_baseline)
        / max(1.0 - threshold_baseline, 1e-12),
        0.0,
        1.0,
    )
    learned_anti = pair_confidence * thresholded_anti

    # Baseline cue accessibility combines instructed validity, serial
    # prominence, and learned distributional rarity.
    if n_features > 1:
        position = 1.0 - np.arange(n_features, dtype=float) / (n_features - 1.0)
    else:
        position = np.ones(1, dtype=float)
    cue_weight = np.exp(
        validity_attention * reliability
        + order_prominence * position
        + distribution_attention * np.log(rarity)
    )

    # Cues that historically concur with their current coalition partners
    # are treated as less independent. A cue with no partner is not given
    # this historical redundancy penalty.
    for j in active:
        partners = [
            int(k) for k in active
            if k != j and directions[k] == directions[j]
        ]
        if partners:
            concurrence = [pair_same[j, k] / pair_both[j, k] for k in partners]
            corr_load = float(np.mean(concurrence))
            cue_weight[j] /= 1.0 + redundancy * corr_load

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    # Lexicographic anchor: highest communicated validity among currently
    # discriminating cues, with stable presentation order breaking ties.
    active_validity = validities[active]
    best_validity = float(np.max(active_validity))
    best_candidates = active[np.isclose(active_validity, best_validity, atol=1e-10)]
    lex_cue = int(np.min(best_candidates))
    lex_direction = float(directions[lex_cue])

    # A global top-validity tie attenuates profile-based singleton
    # amplification, preventing the first cue from receiving both a strong
    # order anchor and a full distinctiveness bonus.
    sorted_v = np.sort(validities)[::-1]
    if sorted_v.size >= 2:
        top_gap = float(sorted_v[0] - sorted_v[1])
        global_top_tie = float(np.exp(-80.0 * top_gap))
    else:
        global_top_tie = 1.0
    tie_core_multiplier = 1.0 - tie_core_attenuation * global_top_tie

    # Current-profile distinctiveness: members of the smaller directional
    # coalition stand out, while each coalition's accumulated contribution
    # is divided by a redundancy term.
    if n_a > 0 and n_b > 0:
        if n_a < n_b:
            cue_weight[a_cues] *= (float(n_b) / float(n_a)) ** (
                distinctiveness * tie_core_multiplier
            )
        elif n_b < n_a:
            cue_weight[b_cues] *= (float(n_a) / float(n_b)) ** (
                distinctiveness * tie_core_multiplier
            )

        # Sparse evidence opposing the validity-defined anchor receives its
        # ordinary bounded bonus. Additional influence is restricted to
        # dissenting cues with reliably learned above-threshold opposition.
        dissent_cues = a_cues if lex_direction > 0 else b_cues
        anchor_cues = b_cues if lex_direction > 0 else a_cues
        n_dissent = int(dissent_cues.size)
        if n_dissent > 0:
            for j in dissent_cues:
                learned_diagnosticity = 0.0
                if anchor_cues.size > 0:
                    learned_diagnosticity = float(
                        np.mean(learned_anti[j, anchor_cues])
                    )
                bonus = 1.0 + (dissent_bonus / float(n_dissent)) * (
                    1.0
                    + anti_concurrence_attention * learned_diagnosticity
                )
                cue_weight[j] *= bonus

            # A unique and presentation-prominent validity anchor becomes
            # less accessible in the coalition core when compact opposition
            # has a reliably learned anti-concurrent relationship with it.
            # Active-cue validity uniqueness leaves tied-top profiles intact;
            # squared prominence and size decay protect late anchors and
            # anchors facing large, plausibly redundant coalitions.
            if active_validity.size >= 2:
                sorted_active_v = np.sort(active_validity)[::-1]
                active_gap = float(sorted_active_v[0] - sorted_active_v[1])
                anchor_uniqueness = 1.0 - np.exp(-80.0 * active_gap)
            else:
                anchor_uniqueness = 1.0
            anchor_diagnosticity = float(
                np.max(learned_anti[dissent_cues, lex_cue])
            )
            compact_decay = np.exp(-1.5 * float(n_dissent - 1))
            anchor_signal = (
                anchor_uniqueness
                * float(position[lex_cue] ** 2)
                * anchor_diagnosticity
                * compact_decay
            )
            cue_weight[lex_cue] *= np.exp(
                -anchor_attenuation * anchor_signal
            )

    support_a = 0.0
    support_b = 0.0
    if n_a > 0:
        support_a = float(np.sum(cue_weight[a_cues])) / (float(n_a) ** redundancy)
    if n_b > 0:
        support_b = float(np.sum(cue_weight[b_cues])) / (float(n_b) ** redundancy)

    scale = support_a + support_b
    core_evidence = (support_b - support_a) / max(scale, 1e-12)

    # A near tie among the two globally strongest instructed validities
    # raises the stable-order gate, but anchoring has a subject-specific
    # ceiling rather than becoming obligatorily lexicographic.
    tied_top_gate = tie_gate_ceiling * global_top_tie

    # Profile-driven anchoring occurs when the earliest discriminating cue
    # is on an isolated minority side. It is attenuated when a sparse cue in
    # the opposing coalition is historically rare or reliably
    # anti-concurrent, while a large opposition remains interpretable as
    # redundant evidence.
    first_cue = int(active[0])
    first_side_count = n_b if directions[first_cue] > 0 else n_a
    other_side_count = n_a if directions[first_cue] > 0 else n_b
    if first_side_count > 0 and other_side_count > 0:
        isolation = 1.0 - float(first_side_count) / float(other_side_count)
        minority_first = float(first_side_count < other_side_count)
        opposing_cues = a_cues if directions[first_cue] > 0 else b_cues
        first_side_cues = b_cues if directions[first_cue] > 0 else a_cues
        opposing_rarity = float(np.max(np.maximum(np.log(rarity[opposing_cues]), 0.0)))
        opposing_distinctiveness = opposing_rarity / (
            np.log(3.5) * np.sqrt(float(other_side_count))
        )

        # For compact opposition, a reliability-weighted soft maximum lets
        # one strongly diagnostic relation disengage the anchor without
        # being diluted by weak pairings. Larger coalitions retain mean
        # aggregation, and all configurations retain exponential size decay.
        opposition_relations = learned_anti[
            np.ix_(opposing_cues, first_side_cues)
        ].reshape(-1)
        if other_side_count <= 3 and opposition_relations.size > 1:
            select_logits = diagnostic_selectivity * (
                opposition_relations - np.max(opposition_relations)
            )
            select_weights = np.exp(select_logits)
            select_weights /= np.sum(select_weights)
            learned_opposition = float(
                np.sum(select_weights * opposition_relations)
            )
        else:
            learned_opposition = float(np.mean(opposition_relations))
        learned_opposition *= np.exp(-1.5 * float(other_side_count - 1))
        opposing_distinctiveness += (
            anti_concurrence_attention * learned_opposition
        )
    else:
        isolation = 0.0
        minority_first = 0.0
        opposing_distinctiveness = 0.0

    profile_gate = minority_first / (
        1.0 + np.exp(-gate_sharpness * (isolation - 0.30))
    )
    profile_gate *= np.exp(-gate_rarity * opposing_distinctiveness)
    gate = 1.0 - (1.0 - tied_top_gate) * (1.0 - profile_gate)
    gate = float(np.clip(gate, 0.0, 1.0))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * lex_strength * lex_direction
    )

    logits = beta * np.array([-0.5 * choice_evidence, 0.5 * choice_evidence])
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs.astype(np.float64)

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
- validity_attention: [0.2, 0.9]
- order_prominence: [0.15, 0.7]
- distribution_attention: [0.8, 1.8]
- distinctiveness: [0.35, 0.9]
- dissent_bonus: [1.0, 2.2]
- anti_concurrence_attention: [1.0, 4.8]
- anti_concurrence_threshold: [0.62, 0.76]
- diagnostic_selectivity: [2.0, 8.0]
- anchor_attenuation: [0.8, 3.2]
- redundancy: [0.68, 1.15]
- lex_strength: [0.7, 1.4]
- gate_sharpness: [4.0, 11.0]
- gate_rarity: [1.0, 4.0]
- tie_gate_ceiling: [0.55, 0.85]
- tie_core_attenuation: [0.7, 0.95]
- beta: [0.6, 3.4]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.3463 (var=0.0061)
- Experiment 3: real=0.0721 (var=0.0052) vs this=0.1125 (var=0.0053)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.8223 (var=0.0088)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.5416 (var=0.0261)
- Experiment 7: real=0.0052 (var=0.0075) vs this=-0.1060 (var=0.0121)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0058 (var=0.0001)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.5687 (var=0.0166)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.4656 (var=0.0070)
- Experiment 11: real=0.1093 (var=0.0652) vs this=0.0092 (var=0.0107)
- Experiment 12: real=0.3533 (var=0.0427) vs this=-0.0208 (var=0.0079)


---

### `pi_7` (overall score: 0.471)

**Description**
Partitioned Dissent and Redundancy Restoration (PDRR) proposes that people construct a stable, instruction-based evidence representation and classify each choice configuration before integrating its cues. Communicated validities determine both relative cue strength and absolute anchor credibility, while cues supporting the same option accumulate sublinearly because concordant reports are treated as partly redundant. In ordinary embedded coalitions, individual cue weights are strongly compressed before accumulation, limiting the influence of any single non-anchor cue while preserving communicated-validity ordering. Opposition to a validity anchor is statically distinctive even when the anchor has allies, but balanced opposition receives a counter-anchor bonus only when the opposing cues are individually close in communicated validity to the anchor and the coalition is broad enough to support a coalition-level interpretation. Isolated-anchor opposition is processed separately. A single opponent receives especially strong challenge status, with stable subject types differing substantially in their sensitivity to this challenge, while exactly two opponents retain a joint-credibility mechanism with greater type-specific heterogeneity. These challenge and singleton-dissent mechanisms are additionally amplified when the locally strongest active cue has low absolute communicated validity, because being best among weak active cues does not create a compelling anchor. Coalitions of three or more trigger a sharp, nearly type-invariant redundancy-restoration branch. Embedded singleton dissent is strongly amplified only against very broad coalitions or weak absolute anchors; its generic effect against smaller coalitions is attenuated to avoid excessive profile variance. Presentation order affects choice primarily as a bounded tie breaker among currently discriminating cues with equal maximal communicated validity, avoiding a general primacy crossover. All mechanisms are history-invariant in this feedback-free task. Stable subject types and random effects over validity sensitivity, balanced dissent, compact challenge, redundancy, temperature, lapse, and absolute-anchor sensitivity generate persistent individual differences.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PDRR expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    redundancy_exponent = float(parameters["redundancy_exponent"])
    redundancy_saturation = float(parameters["redundancy_saturation"])
    ordinary_validity_compression = float(parameters["ordinary_validity_compression"])
    minority_amplification = float(parameters["minority_amplification"])
    balanced_opposition_bonus = float(parameters["balanced_opposition_bonus"])
    singleton_bonus = float(parameters["singleton_bonus"])
    compact_challenge = float(parameters["compact_challenge"])
    compact_width = float(parameters["compact_width"])
    credibility_gap_scale = float(parameters["credibility_gap_scale"])
    absolute_anchor_threshold = float(parameters["absolute_anchor_threshold"])
    absolute_anchor_slope = float(parameters["absolute_anchor_slope"])
    weak_anchor_gain = float(parameters["weak_anchor_gain"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap_threshold = float(parameters["restoration_gap_threshold"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    restoration_credibility_threshold = float(parameters["restoration_credibility_threshold"])
    restoration_credibility_slope = float(parameters["restoration_credibility_slope"])
    restoration_ceiling = float(parameters["restoration_ceiling"])
    tie_order_gate = float(parameters["tie_order_gate"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated, stable subject types alter configuration sensitivity without
    # introducing trial-wise strategy switching or exposure-based learning.
    if configuration_type == 0:       # compact-challenge-sensitive
        challenge_type_gain = 2.40
        one_challenge_type_gain = 4.20
        singleton_type_gain = 1.50
        restoration_type_gain = 0.95
    elif configuration_type == 1:     # count-restoring
        challenge_type_gain = 0.50
        one_challenge_type_gain = 0.75
        singleton_type_gain = 0.90
        restoration_type_gain = 1.25
    else:                              # weakly configuration-sensitive
        challenge_type_gain = 0.15
        one_challenge_type_gain = 0.15
        singleton_type_gain = 0.55
        restoration_type_gain = 1.00

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Translate communicated accuracy into reliability evidence, retaining
    # absolute ordering but using the strongest active cue as a scale origin.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(
        validity_sensitivity * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices, compress=False):
        n = int(indices.size)
        if n == 0:
            return 0.0
        weights = cue_weight[indices]
        if compress:
            # Bound cue-composition effects in ordinary embedded coalitions.
            # The power transform preserves validity ordering but prevents a
            # single non-anchor cue from creating a broad primacy crossover.
            weights = np.power(
                np.clip(weights, 1e-12, None), ordinary_validity_compression
            )
        raw = float(np.sum(weights))
        # The power term gives ordinary sublinear accumulation. The bounded
        # saturation term makes additional concordant reports increasingly
        # interpretable as redundant without erasing coalition size entirely.
        discount = (float(n) ** redundancy_exponent) * (
            1.0 + redundancy_saturation * float(max(n - 1, 0))
            / float(n + 1)
        )
        return raw / max(discount, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # Highest communicated validity among discriminating cues defines the
    # anchor. Stable display order is used only to resolve a top-validity tie.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]
    anchor_cue = int(np.min(best_candidates))
    anchor_direction = float(directions[anchor_cue])
    unique_anchor = float(best_candidates.size == 1)

    # Relative weights alone make every locally strongest cue equivalent.
    # This bounded gate preserves that representation while marking anchors
    # that are weak in absolute communicated-validity terms as challengeable.
    weak_absolute_anchor = sigmoid(
        absolute_anchor_slope * (absolute_anchor_threshold - best_v)
    )
    weak_anchor_multiplier = 1.0 + weak_anchor_gain * weak_absolute_anchor

    if active.size > 1:
        sorted_active_v = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active_v[0] - sorted_active_v[1])
    else:
        validity_gap = 0.5

    if anchor_direction > 0:
        anchor_side_count = n_b
        opposition_count = n_a
        anchor_support = support_b
        opposition_support = support_a
        opposition_cues = a_cues
    else:
        anchor_side_count = n_a
        opposition_count = n_b
        anchor_support = support_a
        opposition_support = support_b
        opposition_cues = b_cues

    isolated_anchor = anchor_side_count == 1 and opposition_count > 0

    # Credibility is based on the strongest one or two individual opponents.
    # For two-opponent challenge, the weaker member must also be credible, so
    # one strong report cannot make a heterogeneous pair diagnostic.
    if opposition_count > 0:
        opponent_validities = np.sort(validities[opposition_cues])[::-1]
        credibility_set = opponent_validities[:min(2, opposition_count)]
        opponent_credibility = float(np.mean(credibility_set))
        credibility_gap = max(best_v - opponent_credibility, 0.0)
        joint_credibility_gap = max(
            best_v - float(np.min(credibility_set)), 0.0
        )
    else:
        credibility_gap = 0.5
        joint_credibility_gap = 0.5

    if isolated_anchor:
        # Challenge is sharply asymmetric across counts. The one-opponent
        # branch has a less suppressive credibility gate and substantially
        # separated stable subject types. Weak absolute anchors amplify the
        # existing challenge rather than creating a general coalition bonus.
        if opposition_count == 1:
            compact_profile = 3.00
            challenge_gap = 0.35 * credibility_gap
            active_challenge_type_gain = one_challenge_type_gain
        elif opposition_count == 2:
            compact_profile = 1.35
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        else:
            distance = (float(opposition_count) - 1.5) / max(compact_width, 1e-6)
            compact_profile = np.exp(-(distance ** 4))
            challenge_gap = joint_credibility_gap
            active_challenge_type_gain = challenge_type_gain
        credibility_gate = np.exp(
            -challenge_gap / max(credibility_gap_scale, 1e-6)
        )
        challenged_opposition = opposition_support * (
            1.0
            + active_challenge_type_gain
            * compact_challenge
            * compact_profile
            * credibility_gate
            * weak_anchor_multiplier
        )
        margin = (anchor_support - challenged_opposition) / max(
            anchor_support + challenged_opposition, 1e-12
        )
        core_evidence = anchor_direction * margin
    else:
        # Compress cue-composition effects only for unique-anchor ordinary
        # coalitions. Tied-top profiles retain the accepted integrator so the
        # calibrated bounded order effect is not disturbed.
        if unique_anchor > 0.0:
            support_a = coalition_support(a_cues, compress=True)
            support_b = coalition_support(b_cues, compress=True)
            if anchor_direction > 0:
                anchor_support = support_b
                opposition_support = support_a
            else:
                anchor_support = support_a
                opposition_support = support_b

        # Balanced opposition is distinctive only when the strongest opposing
        # cue is close to the anchor in communicated validity. A bounded breadth
        # factor further prevents a compact 2-vs-2 profile from receiving the
        # same categorical boost as a broader coalition-level conflict.
        if (
            unique_anchor > 0.0
            and anchor_side_count == opposition_count
            and anchor_side_count > 1
        ):
            strongest_opponent_v = float(np.max(validities[opposition_cues]))
            balanced_validity_gap = max(best_v - strongest_opponent_v, 0.0)
            geometry_gate = np.exp(
                -balanced_validity_gap / max(credibility_gap_scale, 1e-6)
            )
            breadth_gate = min(
                1.0, float(anchor_side_count - 1) / 2.0
            )
            balanced_gain = (
                balanced_opposition_bonus * geometry_gate * breadth_gate
            )
            opposition_support *= 1.0 + balanced_gain
            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        # Embedded distinctiveness is sharply singleton-localized. Its broad-
        # coalition component remains strong, while the generic component is
        # attenuated for smaller coalitions to reduce excessive profile variance.
        # Absolute weakness amplifies this existing channel without altering
        # balanced or other ordinary embedded configurations.
        if opposition_count == 1 and anchor_side_count > 1:
            imbalance = (
                float(anchor_side_count - opposition_count)
                / float(anchor_side_count + opposition_count)
            )
            dissent_multiplier = 1.0 + (
                minority_amplification * imbalance * weak_anchor_multiplier
            )
            breadth_weight = 1.0 if anchor_side_count >= 5 else 0.20
            dissent_multiplier += (
                singleton_type_gain
                * singleton_bonus
                * breadth_weight
                * np.log1p(float(anchor_side_count - 1))
                * weak_anchor_multiplier
            )
            # A very broad concordant coalition makes its sole dissenter
            # especially distinctive, while leaving three-versus-one control
            # profiles close to the accepted static accumulator.
            if anchor_side_count >= 5:
                dissent_multiplier += (
                    singleton_type_gain
                    * singleton_bonus
                    * 0.45
                    * float(anchor_side_count - 4)
                    * weak_anchor_multiplier
                )
            opposition_support *= dissent_multiplier

            if anchor_direction > 0:
                support_a = opposition_support
            else:
                support_b = opposition_support

        total_support = support_a + support_b
        core_evidence = (support_b - support_a) / max(total_support, 1e-12)

    # Gap-sensitive restoration remains available, but exactly two opponents
    # receive little automatic restoration. A separate steep count branch is
    # almost off at two and effectively reaches the common gate ceiling at
    # three or more, without requiring the gap gates or subject type to agree.
    restoration_gate = 0.0
    if isolated_anchor and unique_anchor > 0.0:
        count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - restoration_threshold)
        )
        gap_gate = sigmoid(
            restoration_gap_slope
            * (validity_gap - restoration_gap_threshold)
        )
        credibility_restoration = sigmoid(
            restoration_credibility_slope
            * (credibility_gap - restoration_credibility_threshold)
        )
        selective_restoration = (
            restoration_ceiling
            * count_gate
            * gap_gate
            * credibility_restoration
        )

        if opposition_count == 2:
            selective_restoration *= 0.30

        large_count_gate = sigmoid(
            restoration_slope
            * (float(opposition_count) - 2.50)
        )
        count_dominant_restoration = 0.99 * large_count_gate
        restoration_gate = max(
            selective_restoration,
            count_dominant_restoration,
        )

    # Equal top validities remain exchangeable in integration. Order only
    # supplies a bounded tie anchor rather than a general first-cue bonus.
    tied_top = float(best_candidates.size > 1)
    order_gate = tie_order_gate * tied_top
    gate = 1.0 - (1.0 - restoration_gate) * (1.0 - order_gate)
    gate = float(np.clip(gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - gate) * core_evidence
        + gate * anchor_direction
    )

    logits = beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs.astype(np.float64)


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
- validity_sensitivity: [0.15, 0.85]
- redundancy_exponent: [0.28, 0.68]
- redundancy_saturation: [0.15, 0.65]
- ordinary_validity_compression: [0.03, 0.18]
- minority_amplification: [0.20, 0.70]
- balanced_opposition_bonus: [0.55, 1.25]
- singleton_bonus: [1.8, 4.5]
- compact_challenge: [2.4, 6.0]
- compact_width: [0.75, 1.15]
- credibility_gap_scale: [0.018, 0.055]
- absolute_anchor_threshold: [0.68, 0.78]
- absolute_anchor_slope: [20.0, 45.0]
- weak_anchor_gain: [1.2, 3.0]
- restoration_threshold: [1.65, 2.15]
- restoration_slope: [9.0, 18.0]
- restoration_gap_threshold: [0.035, 0.085]
- restoration_gap_slope: [70.0, 150.0]
- restoration_credibility_threshold: [0.025, 0.065]
- restoration_credibility_slope: [70.0, 150.0]
- restoration_ceiling: [0.82, 0.98]
- tie_order_gate: [0.48, 0.78]
- beta: [1.2, 4.2]
- lapse: [0.0, 0.14]
- configuration_type: {0, 1, 2}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.9400 (var=0.0464)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.5233 (var=0.0053)
- Experiment 3: real=0.0721 (var=0.0052) vs this=0.0358 (var=0.0025)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.2200 (var=0.1716)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.8063 (var=0.0071)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.3969 (var=0.0200)
- Experiment 7: real=0.0052 (var=0.0075) vs this=-0.0589 (var=0.0212)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0062 (var=0.0001)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.7512 (var=0.0113)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.4644 (var=0.0059)
- Experiment 11: real=0.1093 (var=0.0652) vs this=0.6688 (var=0.0308)
- Experiment 12: real=0.3533 (var=0.0427) vs this=0.2908 (var=0.0147)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3377 -> ACCEPTED
- iter 2: loss=0.2380 -> ACCEPTED
- iter 3: loss=0.1923 -> ACCEPTED
- iter 4: loss=0.2060 -> REJECTED
- iter 5: loss=0.1979 -> REJECTED
- iter 6: loss=0.1810 -> ACCEPTED
- iter 7: loss=0.1667 -> ACCEPTED
- iter 8: loss=0.1455 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 8 at loss=0.1455 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Prevalence of subjects choosing the first-discriminating-cue winner above chance."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(subj):
        agreements = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            diff = a - b
            discriminating = np.flatnonzero(diff != 0)
            if discriminating.size == 0:
                continue
            first = int(discriminating[0])
            ttb_response = 0 if diff[first] > 0 else 1
            agreements.append(float(int(row["response"]) == ttb_response))

        if len(agreements) == 0:
            return float("nan")
        rate = float(np.mean(agreements))
        if rate > 0.5:
            return 1.0
        if rate < 0.5:
            return 0.0
        return 0.5

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        score = subject_score(subj)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if len(scores) else float("nan")
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.5500 (var=0.2325) (Δ vs real -0.4500)
  - iter 3: 0.7200 (var=0.1916) (Δ vs real -0.2800)
  - iter 4: 0.9400 (var=0.0464) (Δ vs real -0.0600)
  - iter 5: 0.7100 (var=0.2009) (Δ vs real -0.2900)
  - iter 6: 0.7900 (var=0.1609) (Δ vs real -0.2100)
  - iter 7: 0.7700 (var=0.1621) (Δ vs real -0.2300)
  - iter 8 (most recent): 0.9200 (var=0.0736) (Δ vs real -0.0800)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)
- pi_7: 0.9400 (var=0.0464)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices consistent with the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    aligned = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        favored = None

        # Features are already ordered from highest to lowest validity.
        for j in range(min(len(a), len(b))):
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break

        # Omit completely nondiscriminating pairs, although none occur in
        # the specified design.
        if favored is not None:
            aligned.append(float(int(response) == favored))

    if not aligned:
        return float("nan")
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.3242 (var=0.0174)
**Previous candidate values (this loop):**
  - iter 1: 0.6956 (var=0.0028) (Δ vs real +0.3715)
  - iter 2: 0.5471 (var=0.0042) (Δ vs real +0.2229)
  - iter 3: 0.4665 (var=0.0042) (Δ vs real +0.1423)
  - iter 4: 0.5806 (var=0.0051) (Δ vs real +0.2565)
  - iter 5: 0.4717 (var=0.0046) (Δ vs real +0.1475)
  - iter 6: 0.4444 (var=0.0044) (Δ vs real +0.1202)
  - iter 7: 0.4129 (var=0.0041) (Δ vs real +0.0887)
  - iter 8 (most recent): 0.4369 (var=0.0043) (Δ vs real +0.1127)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)
- pi_7: 0.5233 (var=0.0053)

### Experiment 3
**Design**
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Balanced-minus-imbalanced change in choosing Expert 5's winner."""
    if data is None or len(data) == 0:
        return float("nan")

    interaction_scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = b - a
        signs = np.sign(diff)
        n_b = int(np.sum(signs > 0))
        n_a = int(np.sum(signs < 0))
        imbalance = abs(n_b - n_a)

        # The design's critical conditions are exact 2-vs-2 balance and
        # 3-vs-1 imbalance. Ignore any unexpected rows.
        if imbalance == 0:
            condition_sign = 1.0
        elif imbalance == 2:
            condition_sign = -1.0
        else:
            continue

        # Expert 5 is feature index 4 and always identifies the TTB winner.
        if diff[4] > 0:
            expert5_winner = 1
        elif diff[4] < 0:
            expert5_winner = 0
        else:
            continue

        chose_expert5_winner = float(int(row["response"]) == expert5_winner)
        # Averaging this score gives P(E5 winner | balanced) minus
        # P(E5 winner | imbalanced), since the schedule is balanced.
        interaction_scores.append(condition_sign * chose_expert5_winner)

    if len(interaction_scores) == 0:
        return float("nan")
    return float(np.mean(interaction_scores))
```

**Observed (real) value:** 0.0721 (var=0.0052)
**Previous candidate values (this loop):**
  - iter 1: -0.0235 (var=0.0014) (Δ vs real -0.0956)
  - iter 2: -0.0367 (var=0.0029) (Δ vs real -0.1087)
  - iter 3: -0.1100 (var=0.0033) (Δ vs real -0.1821)
  - iter 4: -0.0952 (var=0.0023) (Δ vs real -0.1673)
  - iter 5: -0.1050 (var=0.0024) (Δ vs real -0.1771)
  - iter 6: -0.1150 (var=0.0041) (Δ vs real -0.1871)
  - iter 7: -0.1135 (var=0.0023) (Δ vs real -0.1856)
  - iter 8 (most recent): -0.1002 (var=0.0021) (Δ vs real -0.1723)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)
- pi_7: 0.0358 (var=0.0025)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        if len(df) == 0:
            return np.nan

        a = np.stack(df["option_a_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        b = np.stack(df["option_b_ratings"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
        response = df["response"].to_numpy(dtype=int)

        # Expert 5 is uniquely most valid, so its favored option is the
        # Take-The-Best winner in this design.
        ttb_winner = (b[:, 4] > a[:, 4]).astype(int)
        chose_ttb_winner = (response == ttb_winner).astype(float)

        # Whether Expert 1 favors the same option as Expert 5.
        expert1_winner = (b[:, 0] > a[:, 0]).astype(int)
        expert1_supports = expert1_winner == ttb_winner

        if not np.any(expert1_supports) or not np.any(~expert1_supports):
            return np.nan

        crossover = (chose_ttb_winner[expert1_supports].mean() -
                     chose_ttb_winner[~expert1_supports].mean())

        # Classify a subject as showing a substantively sized primacy
        # crossover. The 0.12 margin suppresses chance sampling contrasts
        # around TTB's population prediction of exactly zero.
        return float(crossover > 0.12)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(scores.mean()) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.8600 (var=0.1204) (Δ vs real +0.8600)
  - iter 2: 0.2400 (var=0.1824) (Δ vs real +0.2400)
  - iter 3: 0.0800 (var=0.0736) (Δ vs real +0.0800)
  - iter 4: 0.3800 (var=0.2356) (Δ vs real +0.3800)
  - iter 5: 0.1400 (var=0.1204) (Δ vs real +0.1400)
  - iter 6: 0.2600 (var=0.1924) (Δ vs real +0.2600)
  - iter 7: 0.2000 (var=0.1600) (Δ vs real +0.2000)
  - iter 8 (most recent): 0.1400 (var=0.1204) (Δ vs real +0.1400)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.2400 (var=0.1824)
- pi_7: 0.2200 (var=0.1716)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices that follow the stable-order winner among the tied top cues."""
    if len(data) == 0:
        return float("nan")

    follows_top = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is consulted first under the communicated validity order.
        if a[0] > b[0]:
            top_choice = 0
        elif b[0] > a[0]:
            top_choice = 1
        else:
            continue

        follows_top.append(float(int(row["response"]) == top_choice))

    if len(follows_top) == 0:
        return float("nan")
    return float(np.mean(follows_top))
```

**Observed (real) value:** 0.7900 (var=0.0184)
**Previous candidate values (this loop):**
  - iter 1: 0.7002 (var=0.0106) (Δ vs real -0.0898)
  - iter 2: 0.6721 (var=0.0290) (Δ vs real -0.1179)
  - iter 3: 0.7442 (var=0.0305) (Δ vs real -0.0458)
  - iter 4: 0.7929 (var=0.0077) (Δ vs real +0.0029)
  - iter 5: 0.7494 (var=0.0252) (Δ vs real -0.0406)
  - iter 6: 0.7417 (var=0.0199) (Δ vs real -0.0483)
  - iter 7: 0.7533 (var=0.0208) (Δ vs real -0.0367)
  - iter 8 (most recent): 0.7544 (var=0.0233) (Δ vs real -0.0356)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)
- pi_7: 0.8063 (var=0.0071)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Difference in adherence to the highest-validity cue between
    # one-opponent and multi-opponent coalitions.
    lone_top_choices = []
    multi_top_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Feature 0 is the uniquely highest-validity expert in this design.
        top_direction = np.sign(b[0] - a[0])  # +1 means B is top-cue winner
        if top_direction == 0:
            continue

        lower_directions = np.sign(b[1:] - a[1:])
        n_opponents = int(np.sum(lower_directions * top_direction < 0))
        chose_top_winner = float(
            int(row["response"]) == (1 if top_direction > 0 else 0)
        )

        if n_opponents == 1:
            lone_top_choices.append(chose_top_winner)
        elif n_opponents >= 2:
            multi_top_choices.append(chose_top_winner)

    # The fixed schedule supplies both classes for every subject. These
    # fallbacks keep the function scalar-valued on unexpected partial data.
    if len(lone_top_choices) == 0 or len(multi_top_choices) == 0:
        return 0.0

    return float(np.mean(lone_top_choices) - np.mean(multi_top_choices))
```

**Observed (real) value:** -0.5450 (var=0.0462)
**Previous candidate values (this loop):**
  - iter 1: 0.1172 (var=0.0125) (Δ vs real +0.6622)
  - iter 2: -0.0900 (var=0.0092) (Δ vs real +0.4550)
  - iter 3: -0.1916 (var=0.0348) (Δ vs real +0.3534)
  - iter 4: -0.1578 (var=0.0329) (Δ vs real +0.3872)
  - iter 5: -0.1831 (var=0.0229) (Δ vs real +0.3619)
  - iter 6: -0.2844 (var=0.0153) (Δ vs real +0.2606)
  - iter 7: -0.3219 (var=0.0103) (Δ vs real +0.2231)
  - iter 8 (most recent): -0.2691 (var=0.0198) (Δ vs real +0.2759)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)
- pi_7: -0.3969 (var=0.0200)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Post-exposure diagnostic minus control adherence to the top-validity cue."""
    if data is None or len(data) == 0:
        return float("nan")

    def subject_score(df):
        diagnostic = {False: [], True: []}
        control = {False: [], True: []}
        occurrence = {}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            direction = np.sign(b - a)
            if direction[0] == 0:
                continue

            anchor_b = bool(direction[0] > 0)
            followed_anchor = float(int(row["response"]) == int(anchor_b))

            # Diagnostic trials have Expert 5 opposing the validity anchor.
            is_diagnostic = direction[4] == -direction[0]
            # Control trials instead have Expert 2 opposing the anchor.
            is_control = direction[4] == 0 and direction[1] == -direction[0]

            if is_diagnostic:
                # Use repetitions 3--8 of each exact ordered stimulus. This
                # removes acquisition trials while preserving equal numbers
                # of A- and B-anchor trials through the reversal pairs.
                key = (tuple(a.tolist()), tuple(b.tolist()))
                prior = occurrence.get(key, 0)
                occurrence[key] = prior + 1
                if prior >= 2:
                    diagnostic[anchor_b].append(followed_anchor)
            elif is_control:
                control[anchor_b].append(followed_anchor)

        if any(len(diagnostic[s]) == 0 or len(control[s]) == 0
               for s in (False, True)):
            return float("nan")

        diag_balanced = 0.5 * (
            float(np.mean(diagnostic[False])) +
            float(np.mean(diagnostic[True]))
        )
        control_balanced = 0.5 * (
            float(np.mean(control[False])) +
            float(np.mean(control[True]))
        )
        return diag_balanced - control_balanced

    if "subject_id" in data.columns:
        scores = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            score = subject_score(subj_df)
            if np.isfinite(score):
                scores.append(score)
        return float(np.mean(scores)) if scores else float("nan")

    return subject_score(data)

```

**Observed (real) value:** 0.0052 (var=0.0075)
**Previous candidate values (this loop):**
  - iter 1: -0.0050 (var=0.0057) (Δ vs real -0.0102)
  - iter 2: 0.0518 (var=0.0170) (Δ vs real +0.0467)
  - iter 3: 0.0687 (var=0.0183) (Δ vs real +0.0635)
  - iter 4: -0.0092 (var=0.0105) (Δ vs real -0.0143)
  - iter 5: 0.0632 (var=0.0158) (Δ vs real +0.0581)
  - iter 6: 0.0222 (var=0.0153) (Δ vs real +0.0170)
  - iter 7: 0.0388 (var=0.0120) (Δ vs real +0.0337)
  - iter 8 (most recent): 0.0168 (var=0.0154) (Δ vs real +0.0117)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)
- pi_7: -0.0589 (var=0.0212)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df: pd.DataFrame) -> float:
        # Encode every response as adherence to Expert 1, which is the TTB
        # winner on all trials, and group A/B reversals into invariant cue
        # profiles by orienting each profile toward Expert 1's direction.
        groups = {}
        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = np.sign(b - a).astype(int)
            if d.size == 0 or d[0] == 0:
                continue

            orientation = int(d[0])
            profile = tuple((d * orientation).tolist())
            ttb_response = 1 if d[0] > 0 else 0
            adherence = float(int(row["response"]) == ttb_response)
            groups.setdefault(profile, []).append(adherence)

        # The intended design supplies six profiles with many observations
        # each. Requiring at least two observations permits an unbiased
        # estimate of each profile mean's sampling variance.
        usable = [np.asarray(v, dtype=float) for v in groups.values() if len(v) >= 2]
        k = len(usable)
        if k < 2:
            return 0.0

        means = np.asarray([np.mean(v) for v in usable], dtype=float)

        # Observed population variance of profile-specific TTB-adherence
        # rates. Under TTB all profiles have the same latent rate, but its
        # finite-sample value is positive because of response noise.
        observed_profile_variance = float(np.var(means, ddof=0))

        # For Bernoulli observations, p_hat*(1-p_hat)/(n-1) is an unbiased
        # estimator of Var(p_hat). Subtract the exact contribution of these
        # estimation errors to the population variance across k profiles.
        mean_sampling_variances = []
        for v, p_hat in zip(usable, means):
            n = len(v)
            mean_sampling_variances.append(
                float(p_hat * (1.0 - p_hat) / float(n - 1))
            )
        noise_bias = (1.0 - 1.0 / float(k)) * float(
            np.mean(mean_sampling_variances)
        )

        return observed_profile_variance - noise_bias

    if len(data) == 0:
        return 0.0
    if "subject_id" in data.columns:
        scores = [
            subject_score(g)
            for _, g in data.groupby("subject_id", sort=False)
        ]
        return float(np.mean(scores)) if scores else 0.0
    return subject_score(data)

```

**Observed (real) value:** 0.0070 (var=0.0002)
**Previous candidate values (this loop):**
  - iter 1: 0.0024 (var=0.0000) (Δ vs real -0.0046)
  - iter 2: 0.0090 (var=0.0002) (Δ vs real +0.0020)
  - iter 3: 0.0062 (var=0.0001) (Δ vs real -0.0008)
  - iter 4: 0.0049 (var=0.0001) (Δ vs real -0.0021)
  - iter 5: 0.0096 (var=0.0001) (Δ vs real +0.0026)
  - iter 6: 0.0084 (var=0.0002) (Δ vs real +0.0014)
  - iter 7: 0.0070 (var=0.0001) (Δ vs real -0.0000)
  - iter 8 (most recent): 0.0059 (var=0.0002) (Δ vs real -0.0011)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)
- pi_7: 0.0062 (var=0.0001)

### Experiment 9
**Design**
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.5

    def subject_score(subj):
        anchor_choices = []

        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 5 or b.size < 5:
                continue

            directions = np.sign(b - a)
            active = set(np.flatnonzero(directions != 0).tolist())

            # The two isolated-anchor target classes and their A/B reversals.
            is_outer_target = active == {0, 3, 4}
            is_middle_target = active == {1, 2, 4}
            if not (is_outer_target or is_middle_target):
                continue

            lower = [0, 3] if is_outer_target else [1, 2]
            coalition_is_concordant = directions[lower[0]] == directions[lower[1]]
            coalition_opposes_anchor = directions[lower[0]] == -directions[4]
            if not (coalition_is_concordant and coalition_opposes_anchor):
                continue

            anchor_response = 1 if directions[4] > 0 else 0
            anchor_choices.append(float(int(row["response"]) == anchor_response))

        if len(anchor_choices) == 0:
            return np.nan
        return float(np.mean(anchor_choices))

    if "subject_id" in data.columns:
        scores = [subject_score(subj) for _, subj in data.groupby("subject_id", sort=False)]
        scores = [x for x in scores if np.isfinite(x)]
        return float(np.mean(scores)) if len(scores) > 0 else 0.5

    score = subject_score(data)
    return float(score) if np.isfinite(score) else 0.5

```

**Observed (real) value:** 0.7250 (var=0.0723)
**Previous candidate values (this loop):**
  - iter 1: 0.8237 (var=0.0067) (Δ vs real +0.0988)
  - iter 2: 0.4931 (var=0.0135) (Δ vs real -0.2319)
  - iter 3: 0.5787 (var=0.0079) (Δ vs real -0.1462)
  - iter 4: 0.6837 (var=0.0060) (Δ vs real -0.0413)
  - iter 5: 0.5731 (var=0.0084) (Δ vs real -0.1519)
  - iter 6: 0.5737 (var=0.0075) (Δ vs real -0.1512)
  - iter 7: 0.5806 (var=0.0089) (Δ vs real -0.1444)
  - iter 8 (most recent): 0.5681 (var=0.0089) (Δ vs real -0.1569)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)
- pi_7: 0.7512 (var=0.0113)

### Experiment 10
**Design**
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(subj):
        followed = []

        for _, row in subj.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=int)
            b = np.asarray(row['option_b_ratings'], dtype=int)
            if a.shape[0] != 6 or b.shape[0] != 6:
                continue

            directions = np.sign(b - a)

            # Diagnostic compact opposition: Experts 3 and 5 oppose the
            # unique validity anchor (trial-pair 1 and its reversal).
            compact_diagnostic = (
                np.array_equal(np.abs(directions),
                               np.array([1, 0, 1, 0, 1, 0]))
                and directions[2] == directions[4]
                and directions[0] == -directions[2]
            )

            # Singleton diagnostic dissent: Expert 3 alone opposes all five
            # other discriminating experts (trial-pair 3 and its reversal).
            singleton_diagnostic = (
                np.all(np.abs(directions) == 1)
                and directions[2] == -directions[0]
                and directions[1] == directions[0]
                and directions[3] == directions[0]
                and directions[4] == directions[0]
                and directions[5] == directions[0]
            )

            if not (compact_diagnostic or singleton_diagnostic):
                continue

            anchor_response = 1 if directions[0] > 0 else 0
            followed.append(float(int(row['response']) == anchor_response))

        if len(followed) == 0:
            return np.nan
        return float(np.mean(followed))

    scores = []
    for _, subj in data.groupby('subject_id', sort=False):
        value = subject_score(subj)
        if np.isfinite(value):
            scores.append(value)

    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4462 (var=0.0224)
**Previous candidate values (this loop):**
  - iter 1: 0.8263 (var=0.0086) (Δ vs real +0.3800)
  - iter 2: 0.5194 (var=0.0178) (Δ vs real +0.0731)
  - iter 3: 0.6469 (var=0.0086) (Δ vs real +0.2006)
  - iter 4: 0.7137 (var=0.0088) (Δ vs real +0.2675)
  - iter 5: 0.6050 (var=0.0122) (Δ vs real +0.1588)
  - iter 6: 0.6438 (var=0.0098) (Δ vs real +0.1975)
  - iter 7: 0.6338 (var=0.0067) (Δ vs real +0.1875)
  - iter 8 (most recent): 0.6194 (var=0.0079) (Δ vs real +0.1731)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)
- pi_7: 0.4644 (var=0.0059)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    high_count_anchor_choices = []
    two_opponent_anchor_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 8 or b.size < 8:
            continue

        directions = np.sign(b - a)
        anchor_direction = int(directions[7])
        if anchor_direction == 0:
            continue

        # Retain isolated-expert-8 profiles and count cues opposing it.
        other_directions = directions[:7]
        if np.any(other_directions == anchor_direction):
            continue
        opponent_count = int(np.sum(other_directions == -anchor_direction))

        anchor_response = 1 if anchor_direction > 0 else 0
        chose_anchor = float(int(row["response"]) == anchor_response)

        if opponent_count == 2:
            two_opponent_anchor_choices.append(chose_anchor)
        elif opponent_count >= 3:
            high_count_anchor_choices.append(chose_anchor)

    if not high_count_anchor_choices or not two_opponent_anchor_choices:
        return float("nan")

    return float(
        np.mean(high_count_anchor_choices)
        - np.mean(two_opponent_anchor_choices)
    )
```

**Observed (real) value:** 0.1093 (var=0.0652)
**Previous candidate values (this loop):**
  - iter 1: 0.2480 (var=0.0099) (Δ vs real +0.1387)
  - iter 2: 0.1413 (var=0.0133) (Δ vs real +0.0320)
  - iter 3: 0.1035 (var=0.0149) (Δ vs real -0.0058)
  - iter 4: 0.1663 (var=0.0145) (Δ vs real +0.0570)
  - iter 5: 0.0813 (var=0.0158) (Δ vs real -0.0280)
  - iter 6: 0.0795 (var=0.0155) (Δ vs real -0.0298)
  - iter 7: 0.0832 (var=0.0152) (Δ vs real -0.0262)
  - iter 8 (most recent): 0.0975 (var=0.0155) (Δ vs real -0.0118)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6688 (var=0.0308)
- pi_5: 0.0092 (var=0.0107)
- pi_1: 0.0137 (var=0.0129)
- pi_2: -0.0240 (var=0.0079)
- pi_3: -0.0033 (var=0.0095)
- pi_4: -0.0368 (var=0.0015)
- pi_6: -0.0085 (var=0.0107)

### Experiment 12
**Design**
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Expert-6 activation effect on choosing the Expert-4 anchor option."""
    if len(data) == 0:
        return float("nan")

    anchor_choices = []
    expert6_on = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)

        # Expert 4 (zero-based index 3) defines the designated anchor option.
        if a[3] > b[3]:
            anchor_response = 0
        elif b[3] > a[3]:
            anchor_response = 1
        else:
            continue

        anchor_choices.append(float(int(response) == anchor_response))
        expert6_on.append(bool(a[5] != b[5]))

    anchor_choices = np.asarray(anchor_choices, dtype=float)
    expert6_on = np.asarray(expert6_on, dtype=bool)

    if not np.any(expert6_on) or not np.any(~expert6_on):
        return float("nan")

    return float(
        np.mean(anchor_choices[expert6_on])
        - np.mean(anchor_choices[~expert6_on])
    )
```

**Observed (real) value:** 0.3533 (var=0.0427)
**Previous candidate values (this loop):**
  - iter 1: -0.0637 (var=0.0123) (Δ vs real -0.4171)
  - iter 2: 0.1625 (var=0.0098) (Δ vs real -0.1908)
  - iter 3: 0.3313 (var=0.0141) (Δ vs real -0.0221)
  - iter 4: 0.2358 (var=0.0073) (Δ vs real -0.1175)
  - iter 5: 0.3308 (var=0.0126) (Δ vs real -0.0225)
  - iter 6: 0.3021 (var=0.0141) (Δ vs real -0.0513)
  - iter 7: 0.3517 (var=0.0187) (Δ vs real -0.0017)
  - iter 8 (most recent): 0.3142 (var=0.0136) (Δ vs real -0.0392)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0208 (var=0.0079)
- pi_7: 0.2908 (var=0.0147)
- pi_1: 0.0083 (var=0.0081)
- pi_2: -0.3271 (var=0.0169)
- pi_3: 0.0450 (var=0.0087)
- pi_4: -0.5246 (var=0.0081)
- pi_6: 0.2700 (var=0.0168)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Active-Cue Accessibility Competition (ACAC) proposes that people use a stable representation of communicated cue validity and evaluate only cues that discriminate on the current trial. Active cues first form option-specific coalitions. Coalition support accumulates smoothly but is compressed according to coalition size, validity similarity, and effective cue multiplicity, so additional concordant cues become increasingly redundant without triggering a categorical restoration rule. In unique-anchor configurations where both options have multiple active supporters, coalition evidence is additionally normalized toward effective coalition counts. This prevents the allocation of one relatively valid lower cue from producing an implicit presentation-order crossover. Decisions in this symmetrically normalized geometry are made with moderately greater response precision, reflecting reduced conflict after cue identities have been compressed; this suppresses sampling-induced primacy classifications and strengthens balanced-profile anchor evidence without introducing an order weight. The strongest currently active cue serves as a validity anchor. Opposition receives a sharply count-selective challenge enhancement centered on exactly one opponent. This enhancement is strongest for an isolated anchor and attenuated for embedded-anchor singleton dissent; larger opposing coalitions retain only a small, smooth inverse-count challenge tail. In the exact unique-anchor, isolated, one-opponent configuration, stable competition types form a heterogeneous mixture containing a small genuinely resistant group and a larger set of strongly challenge-accessible integrators. Challenge strength depends continuously on the anchor's absolute validity and validity advantage over the strongest opponent. Presentation order is consulted only when multiple currently active cues share maximal communicated validity. A bounded active-tie gate favors the earliest member of that tied set, with an additional conflict-geometry increment when lower-validity active cues precede tied maxima supporting opposing options. Inactive co-top cues cannot activate either component. Tie accessibility retains the same marginal distribution across subject types as in the previous model. Because the task provides no correctness feedback, ACAC explicitly excludes trial-history learning of cue concurrence, anti-concurrence, or discrimination frequency.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"ACAC expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_gain = float(parameters["validity_gain"])
    redundancy_strength = float(parameters["redundancy_strength"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    similarity_sensitivity = float(parameters["similarity_sensitivity"])
    compact_challenge = float(parameters["compact_challenge"])
    compact_peak = float(parameters["compact_peak"])
    compact_width = float(parameters["compact_width"])
    isolated_singleton_access = float(parameters["isolated_singleton_access"])
    challenge_tail = float(parameters["challenge_tail"])
    embedded_compact_scale = float(parameters["embedded_compact_scale"])
    challenge_gap_scale = float(parameters["challenge_gap_scale"])
    absolute_threshold = float(parameters["absolute_threshold"])
    absolute_slope = float(parameters["absolute_slope"])
    weak_anchor_gain = float(parameters["weak_anchor_gain"])
    isolation_gain = float(parameters["isolation_gain"])
    credibility_protection = float(parameters["credibility_protection"])
    active_tie_ceiling = float(parameters["active_tie_ceiling"])
    tie_activation_rate = float(parameters["tie_activation_rate"])
    conflict_tie_boost = float(parameters["conflict_tie_boost"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    competition_type = int(parameters["competition_type"])

    # Six correlated mixture cells preserve the previous marginal distribution
    # of active-tie accessibility while assigning only one-sixth of subjects to
    # a genuinely singleton-resistant group. The remaining subjects express a
    # strong but heterogeneous exact-singleton challenge response.
    if competition_type == 0:       # genuinely singleton-resistant
        type_challenge = 0.05
        type_singleton_access = 0.00
    elif competition_type in (1, 2):  # strongly challenge-sensitive
        type_challenge = 4.50
        type_singleton_access = 0.00
    else:                            # strongly accessible integrators
        type_challenge = 1.00
        type_singleton_access = 1.50

    # Tie accessibility is crossed with singleton sensitivity so that changing
    # the singleton mixture does not change the marginal tie-type frequencies.
    if competition_type in (0, 1):
        type_tie = 1.65
    elif competition_type in (2, 3):
        type_tie = 0.30
    else:
        type_tie = 1.00

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Communicated accuracy is represented on a log-evidence scale. Centering
    # on the active maximum improves numerical stability without discarding
    # absolute validity, which is retained separately below.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    cue_weight = np.exp(validity_gain * (reliability - active_reference))

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        weights = np.asarray(cue_weight[indices], dtype=float)
        raw_support = float(np.sum(weights))

        # Effective multiplicity changes smoothly with coalition composition.
        # A coalition containing several similarly accessible cues has a high
        # effective count and is compressed more strongly than a heterogeneous
        # coalition dominated by one cue.
        effective_n = raw_support / max(float(np.max(weights)), 1e-12)
        effective_excess = max(effective_n - 1.0, 0.0)

        coalition_v = validities[indices]
        validity_spread = float(np.std(coalition_v)) if n > 1 else 0.0
        similarity = np.exp(-similarity_sensitivity * validity_spread)
        discount = 1.0 + redundancy_strength * similarity * (
            effective_excess ** redundancy_curvature
        )
        return raw_support / max(discount, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # The validity anchor is defined only over currently discriminating cues.
    active_v = validities[active]
    best_v = float(np.max(active_v))
    top_candidates = active[np.isclose(active_v, best_v, atol=1e-8)]
    anchor_cue = int(np.min(top_candidates))
    anchor_direction = float(directions[anchor_cue])
    unique_anchor = float(top_candidates.size == 1)

    # In unique-anchor multi-cue-versus-multi-cue configurations, normalize
    # coalition composition toward effective count support. This saturation
    # prevents reallocating one relatively valid lower cue from inducing an
    # implicit primacy crossover, while leaving isolated-anchor diagnostics
    # and all tied-top configurations unchanged.
    normalized_multicue_geometry = False
    if unique_anchor > 0.0 and n_a > 1 and n_b > 1:
        pooled_weight = float(np.mean(cue_weight[active]))

        def normalized_multicue_support(n):
            excess = float(n - 1)
            discount = 1.0 + redundancy_strength * (
                excess ** redundancy_curvature
            )
            return pooled_weight * float(n) / max(discount, 1e-12)

        support_a = normalized_multicue_support(n_a)
        support_b = normalized_multicue_support(n_b)
        normalized_multicue_geometry = True

    if anchor_direction > 0:
        anchor_indices = b_cues
        opponent_indices = a_cues
        anchor_support = support_b
        opponent_support = support_a
    else:
        anchor_indices = a_cues
        opponent_indices = b_cues
        anchor_support = support_a
        opponent_support = support_b

    anchor_count = int(anchor_indices.size)
    opponent_count = int(opponent_indices.size)

    # Absolute credibility and relative advantage are distinct. A cue can be
    # locally strongest yet remain challengeable when all active cues have low
    # communicated validity.
    weak_anchor = sigmoid(absolute_slope * (absolute_threshold - best_v))
    strong_anchor = 1.0 - weak_anchor

    if opponent_count > 0:
        strongest_opponent_v = float(np.max(validities[opponent_indices]))
        opponent_gap = max(best_v - strongest_opponent_v, 0.0)
    else:
        opponent_gap = 0.5

    gap_accessibility = np.exp(
        -opponent_gap / max(challenge_gap_scale, 1e-8)
    )
    gap_protection = 1.0 - gap_accessibility

    # Credibility protection is deliberately bounded so that merely moving a
    # relatively valid lower cue between coalitions cannot create an implicit
    # unique-anchor presentation-order effect.
    anchor_support *= 1.0 + (
        unique_anchor
        * credibility_protection
        * strong_anchor
        * gap_protection
    )

    if opponent_count > 0:
        # The high-amplitude component is extremely localized at one opponent.
        # Embedded singleton dissent receives only a bounded fraction of it,
        # whereas isolated one-versus-one conflict expresses the full stable
        # subject-type difference. In that exact geometry, a bounded access
        # increment selectively raises strongly accessible integrators while a
        # small mixture remains genuinely resistant.
        count_distance = (
            float(opponent_count) - compact_peak
        ) / max(compact_width, 1e-8)
        singleton_profile = np.exp(-0.5 * count_distance ** 2)

        if anchor_count == 1:
            singleton_geometry = 1.0
        else:
            singleton_geometry = embedded_compact_scale

        active_type_challenge = type_challenge
        if (
            unique_anchor > 0.0
            and anchor_count == 1
            and opponent_count == 1
            and top_candidates.size == 1
        ):
            active_type_challenge += (
                isolated_singleton_access * type_singleton_access
            )

        singleton_intensity = (
            active_type_challenge
            * compact_challenge
            * singleton_profile
            * singleton_geometry
        )
        smooth_tail = challenge_tail / float(opponent_count + 1)

        isolation = 1.0 / float(max(anchor_count, 1))
        geometry_gain = 1.0 + isolation_gain * isolation
        absolute_gain = 1.0 + weak_anchor_gain * weak_anchor

        challenge_multiplier = 1.0 + (
            singleton_intensity + smooth_tail
        ) * gap_accessibility * geometry_gain * absolute_gain
        opponent_support *= challenge_multiplier

    total_support = anchor_support + opponent_support
    if total_support <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total_support
        core_evidence = anchor_direction * anchor_margin

    # A bounded order gate is activated exclusively by a tie among currently
    # active maximum-validity cues. Conflict among tied maxima receives an
    # additional activation increment only when lower-validity active cues
    # precede the tied set. This captures activation-sensitive accessibility
    # without increasing first-cue adherence in unique-maximum profiles.
    active_top_count = int(top_candidates.size)
    if active_top_count >= 2:
        tie_gate = (
            type_tie
            * active_tie_ceiling
            * (1.0 - np.exp(
                -tie_activation_rate * float(active_top_count - 1)
            ))
        )
        tied_directions = directions[top_candidates]
        tied_conflict = float(
            np.any(tied_directions > 0) and np.any(tied_directions < 0)
        )
        preceded_tied_set = float(int(active[0]) < anchor_cue)
        tie_gate += (
            type_tie
            * conflict_tie_boost
            * tied_conflict
            * preceded_tied_set
        )
    else:
        tie_gate = 0.0
    tie_gate = float(np.clip(tie_gate, 0.0, 0.95))

    choice_evidence = (
        (1.0 - tie_gate) * core_evidence
        + tie_gate * anchor_direction
    )

    # Symmetric multi-cue normalization removes much of the identity-based
    # conflict, so choices in that geometry are moderately more precise. This
    # affects neither isolated singleton competition nor active-top-tie trials.
    effective_beta = beta * (1.60 if normalized_multicue_geometry else 1.0)
    logits = effective_beta * np.array(
        [-0.5 * choice_evidence, 0.5 * choice_evidence],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - lapse) * probs + lapse * np.array(
        [0.5, 0.5], dtype=np.float64
    )
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= total
    return probs.astype(np.float64)


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
- validity_gain: [0.05, 0.22]
- redundancy_strength: [0.62, 1.08]
- redundancy_curvature: [1.18, 1.58]
- similarity_sensitivity: [5.0, 14.0]
- compact_challenge: [5.0, 10.0]
- compact_peak: [0.98, 1.02]
- compact_width: [0.12, 0.25]
- isolated_singleton_access: [1.8, 2.8]
- challenge_tail: [0.08, 0.28]
- embedded_compact_scale: [0.08, 0.22]
- challenge_gap_scale: [0.08, 0.18]
- absolute_threshold: [0.68, 0.77]
- absolute_slope: [14.0, 32.0]
- weak_anchor_gain: [0.65, 1.75]
- isolation_gain: [0.35, 1.10]
- credibility_protection: [0.0, 0.18]
- active_tie_ceiling: [0.70, 0.90]
- tie_activation_rate: [1.25, 2.60]
- conflict_tie_boost: [0.60, 1.00]
- beta: [1.45, 3.80]
- lapse: [0.0, 0.10]
- competition_type: {0, 1, 2, 3, 4, 5}

`rationale`: This is a narrow extension of the accepted iteration-7 model. First, it replaces the increasingly homogeneous three-type singleton loading with six equally sampled correlated mixture cells. Only one-sixth of subjects remains genuinely resistant in the exact unique-anchor, isolated one-opponent branch, while the other five-sixths receive strong challenge through either a larger baseline sensitivity or integrative accessibility. This should further lower Experiment-2 anchor adherence and deepen Experiment 6 while restoring the missing between-subject variance. The tie loading is assigned independently across the six cells in two-cell groups, exactly preserving the previous one-third frequencies of high, low, and intermediate tie accessibility; therefore the already accurate Experiment-12 mean and the Experiment-5 tie effect are protected. Embedded challenge, two-opponent competition, credibility modulation, the high-count tail, and the tie equations are unchanged. Second, the existing hard symmetric normalization is left intact, but response precision is multiplied by 1.6 only in unique-anchor multi-cue-versus-multi-cue configurations. Experiment 4 has identical normalized coalition counts across its Expert-1 support conditions, so this reduces threshold-classification false positives without creating a conditional order effect. The same local precision increase should raise anchor adherence in normalized balanced 2-v-2 profiles relative to unnormalized 3-v-1 profiles, directly targeting Experiment 3's wrong-signed interaction without reviving the rejected convex validity-composition blend. Isolated profiles in Experiments 6, 9, 10, and 11 and all active-tie profiles remain outside this precision edit.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: ACAC is faithful to the prescribed family in its stable instruction-based representation, active-cue restriction, smooth redundancy, compact challenge, active-top-tie gate, and exclusion of history learning. It also succeeds on Experiment 1 and preserves the null exposure effect in Experiment 7; Experiment 8 is reasonably close. Nevertheless, the current implementation is not experiment-invariant. It badly overpredicts highest-validity-cue adherence in Experiment 2 (0.696 versus 0.324) and Experiment 10 (0.826 versus 0.446). Its coalition-count effect has the wrong sign in Experiment 6 (0.117 versus -0.545), indicating that the nominal compact challenge is not actually overcoming coalition accumulation for one opponent relative to larger coalitions. It also generates a very large forbidden unique-anchor primacy crossover in Experiment 4 (0.86 versus 0), despite the explicit tie-only order gate. Thus the crossover is being induced indirectly by coalition geometry and challenge, not by the formal order gate. The active-tie mechanism also fails its crucial activation diagnostic in Experiment 12, producing a negative effect (-0.064) instead of the large positive human effect (0.353). Anchor recovery remains too large in Experiment 11 (0.248 versus 0.109), and Experiment 3 has the wrong sign. Finally, simulated between-subject variance is substantially too small in Experiments 2, 6, 9, 10, 11, and 12.
Rationale: Retain the ACAC mechanism family but recalibrate and repair its interactions. First, reshape compact challenge so it peaks more sharply at one opponent: move compact_peak closer to 1, narrow compact_width, and increase challenge strength for one-opponent configurations. Retain a modest nonzero tail for three or more opponents so Experiment 11 shows only gradual, modest recovery rather than 0.248. This combination should move Experiments 2, 6, and 10 toward stronger sparse opposition without reinstating a categorical threshold. Second, prevent coalition geometry from creating an implicit generic primacy effect under a unique active maximum. For sufficiently credible unique anchors, compress or saturate the context-dependent protection/challenge contribution so changing whether an early lower-validity cue joins the anchor coalition cannot produce the large Experiment-4 crossover. This is an in-family normalization fix, not a generic order term. Third, audit the active-tie gate in Experiment 12: verify that the communicated tie is detected with an appropriate validity tolerance, that the gate favors the earliest member of the currently active tied-top set, and that its positive contribution is not overwhelmed by the accompanying coalition-support change. Tie strength may need to depend on active-tie conflict geometry while remaining exactly zero for unique maxima. Do not simply raise the global tie ceiling, since Experiment 2 already has excessive first/highest-cue adherence. Finally, after correcting the means, broaden correlated challenge and tie-gating subject effects to approach the observed variances, while explicitly constraining unique-anchor subjects so this heterogeneity cannot recreate Experiment 4's crossover.

**Outcome of this advice:** iter 1 candidate loss=0.3377 -> iter 2 candidate loss=0.2380 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to the prescribed ACAC family. The prior recommendations successfully reduced loss, reversed Experiment 6 in the correct direction, sharply reduced the forbidden Experiment-4 crossover, brought Experiments 10 and 11 close, and changed Experiment 12 from a negative to a positive activation effect. Experiments 7 and 8 are also satisfactory. However, the model is not yet experiment-invariant. The stronger compact challenge now produces major regressions in Experiment 1 (0.55 versus 1.00, with spurious variance 0.233) and Experiment 9 (0.493 versus 0.725), while Experiment 6 remains far too weakly negative (-0.090 versus -0.545). This combination indicates that challenge is still too diffuse across contexts and opponent counts: it suppresses anchors too much in some two-opponent profiles, yet does not create the required exceptionally strong one-opponent challenge. Experiment 4 also retains a nontrivial crossover prevalence of 0.24 rather than zero, so unique-anchor coalition normalization still depends too strongly on which relatively valid lower cue joins each side. The active-tie repair improved Experiment 12, but its effect remains only about half the observed size (0.163 versus 0.353), while tied-top adherence in Experiment 5 is too low (0.672 versus 0.790). Experiment 2 remains overly anchor-consistent (0.547 versus 0.324), and Experiment 3 still has the wrong sign. Variance is also misplaced: it is much too large in Experiment 1 but too small in Experiments 2, 6, 9, 11, and 12.
Rationale: Continue within ACAC, building on the accepted direction rather than reversing it. Make compact challenge more count-selective, not globally stronger: use a narrower or separately parameterized singleton-opponent component that is very strong at exactly one opponent and falls sharply by two opponents, while retaining only the small smooth tail needed for Experiment 11. This should deepen Experiment 6's negative contrast while restoring anchor adherence in Experiment 9. Modulate that singleton component more strongly by absolute anchor credibility and validity gap so it does not destabilize Experiment 1; challenge-sensitive subject types should vary chiefly when the compact challenge is genuinely activated, rather than multiplying a broad range of ordinary profiles. For Experiment 4, add an in-family saturation or normalization for unique-anchor, multi-cue-versus-multi-cue configurations so reallocating one relatively valid lower cue between coalitions cannot by itself push subjects across the crossover threshold. Keep this adjustment inactive for isolated-anchor diagnostics. Strengthen the active-tie gate selectively: raise the ordinary bounded tie contribution enough to improve Experiment 5, and raise the conflict-geometry increment for the exact lower-cue-precedes-opposing-tied-maxima configuration needed by Experiment 12. Do not turn this into generic primacy, and keep it exactly zero under unique active maxima. Because Experiment 2 is already too anchor-consistent, its correction should come from stronger competition in its non-tie or non-conflict geometries, not from weakening the targeted tie repair. Finally, redistribute heterogeneity rather than simply broadening it: narrow challenge variation in Experiment-1-like profiles, but allow larger stable variation in singleton challenge and active-conflict tie gating to improve the variances of Experiments 6, 11, and 12 without recreating Experiment 4's crossover.

**Outcome of this advice:** iter 2 candidate loss=0.2380 -> iter 3 candidate loss=0.1923 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The third iteration is another accepted improvement and remains faithful to the prescribed ACAC family. The targeted revisions substantially reduced the Experiment-4 forbidden crossover (0.08 versus 0), brought Experiment 5 close (0.744 versus 0.790), matched Experiment 8, matched the mean recovery effect in Experiment 11 (0.104 versus 0.109), and nearly captured the crucial Experiment-12 activation effect (0.331 versus 0.353). The singleton-focused change also moved Experiment 6 farther in the correct negative direction and increased its variance toward the data. Nevertheless, the model is not yet experiment-invariant. Experiment 6 remains the largest mechanistic failure: its contrast is only -0.192 versus -0.545, so isolated one-opponent challenge is still much too weak relative to multi-opponent competition. Highest-validity-cue adherence also remains excessive in Experiment 2 (0.467 versus 0.324). Conversely, Experiment 1 still has insufficient first-cue adherence (0.72 versus 1.00) and very large spurious heterogeneity. Experiment 9 remains too challenging to its isolated anchor (0.579 versus 0.725), while Experiment 10 regressed sharply from the prior iteration and is now insufficiently challenging (0.647 versus 0.446). This opposite error across Experiments 9 and 10 shows that opponent count alone is too coarse: the model needs stronger conditioning by anchor-coalition redundancy, absolute credibility, and validity gap. Experiment 3 also worsened to the wrong-signed -0.110 versus +0.072, suggesting that the hard pooled-weight replacement in unique-anchor multi-cue configurations suppresses behaviorally relevant validity composition along with the unwanted Experiment-4 crossover. Finally, heterogeneity remains misallocated: Experiment 5 is somewhat too variable, whereas Experiments 2, 9, 10, 11, and 12 are much too homogeneous.
Rationale: Continue tuning ACAC, building on the two consecutive accepted directions rather than undoing them. Keep the sharply localized isolated-singleton component, but strengthen it selectively for the exact isolated-anchor/one-opponent geometry that drives Experiment 6. Do not broaden it indiscriminately. Replace the fixed, very small embedded_compact_scale with a smooth redundancy-conditioned scale: singleton dissent should become more accessible when the anchor's many supporters have low effective multiplicity because they are highly redundant. That can lower excessive adherence in Experiments 2 and 10 without applying the full isolated-anchor boost everywhere. Add at most a modest two-opponent component, gated continuously by low absolute credibility and a small validity gap; retain strong protection for credible, well-separated isolated anchors so Experiment 9 rises toward 0.725 and Experiment 1 is not destabilized. The current absolute/gap modulation should therefore be recalibrated for greater discrimination between contexts, rather than globally increasing compact_challenge.

Revise the Experiment-4 normalization without removing it. The current hard replacement of both coalition supports by pooled-weight count support likely causes the Experiment-3 regression because it discards within-coalition validity composition. Use a convex blend or symmetric effective-count normalization that removes cue-position identity and limits lower-cue reallocation effects, but preserves validity-weight differences and balanced-versus-imbalanced coalition geometry. This should retain the near-null Experiment-4 crossover while allowing Experiment 3 to return positive.

Finally, separate heterogeneity in the ordinary active-tie gate from heterogeneity in the preceded-conflicting-tie increment. Narrow ordinary tie variation to reduce the excessive Experiment-5 variance, while broadening the conflict-specific increment to approach Experiment 12's observed variance without changing its already accurate mean. Mean-preserving variation in compact sensitivity or the smooth high-count tail can address the low variances in Experiments 6 and 11, but it should remain gated away from Experiment-1-like profiles. The active-only tie rule and exclusion of history learning should remain unchanged.

**Outcome of this advice:** iter 3 candidate loss=0.1923 -> iter 4 candidate loss=0.2060 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed ACAC family, but it was rejected by the accept gate and is worse overall than the iteration-3 base. Its gains are real but localized: Experiment 1 improves from 0.72 to 0.94, Experiment 5 reaches the observed mean, Experiment 9 improves to 0.684, and Experiments 7–8 remain close. These gains are outweighed by major regressions in the central diagnostics. Experiment 2 rises away from the data (0.581 versus 0.324), Experiment 4 recreates a forbidden unique-anchor crossover (0.38 versus 0), Experiment 6 becomes less negative (-0.158 versus -0.545), Experiment 10 rises to 0.714 versus 0.446, Experiment 11 moves away from its previously accurate value, and the crucial Experiment-12 activation effect falls from 0.331 to 0.236. Experiment 3 also remains wrong-signed. Thus the latest broad package simultaneously strengthened ordinary anchor adherence and destabilized the normalization and tie effects that iteration 3 had already fit. There is also a likely implementation-level problem in the new redundancy-conditioned embedded challenge: `compression_load` is computed from `anchor_support` only after credibility protection has multiplied that support. Protection can therefore erase or reverse the measured compression, making embedded challenge weakest precisely for protected, redundancy-compressed anchor coalitions. Between-subject variances remain too small in Experiments 2, 9, 10, 11, and 12, while Experiment 4 now has large entirely spurious variance.
Rationale: Build on the unchanged accepted iteration-3 base, not on this rejected package. My previous recommendation to combine stronger singleton challenge, redundancy-conditioned embedded challenge, a two-opponent term, blended normalization, and revised heterogeneity was rejected; do not repeat that multi-knob move. Preserve iteration 3's successful Experiment-4 normalization, Experiment-11 recovery profile, and Experiment-12 tie-gate calibration.

Try a narrower implementation correction first. If redundancy-conditioned embedded challenge is retained, calculate coalition compression from the raw coalition support and the redundancy-discounted support before applying credibility protection or any challenge multiplier. Keep credibility protection as a separate downstream factor. Use a bounded embedded gain so this channel can selectively reduce excessive anchor adherence in Experiments 2 and 10 without changing tied-top gating or multi-cue normalization. Do not add the rejected two-opponent term in this pass; Experiment 9 already indicates that broadly enhancing two-opponent challenge is hazardous.

For Experiment 6, the larger challenge-type amplitude in the rejected candidate did not help, so do not increase it again. Use an intermediate singleton loading between iterations 3 and 4, or adjust the prevalence of the challenge-sensitive type only within the exact unique-anchor/one-opponent/no-active-tie geometry. This is preferable to another global increase and should avoid sacrificing Experiment 1. Freeze the iteration-3 active-tie parameters, because their Experiment-12 mean was already nearly correct, and freeze the hard symmetric unique-anchor normalization that held Experiment 4 near zero. Address Experiment 3 only after the principal challenge correction is isolated; the attempted convex blend produced little improvement in Experiment 3 while severely damaging Experiment 4. Once means improve, heterogeneity can be widened through geometry-specific mixture prevalence rather than larger global type multipliers.

**Outcome of this advice:** iter 4 candidate loss=0.2060 -> iter 5 candidate loss=0.1979 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed ACAC family: it uses only active cues, smooth redundancy, compact challenge, an active-top-tie order gate, and no history learning. However, it was rejected by the accept gate (0.1979 versus the accepted iteration-3 loss of 0.1923), so it should not be shipped. The pre-protection compression correction produced only a modest improvement in Experiment 10 relative to iteration 3 (0.605 versus 0.647, still well above 0.446) and did not improve Experiment 2 (0.472 versus 0.467, observed 0.324). It also coincided with worse fits in Experiment 4 (0.14 versus 0.08, observed 0), Experiment 9 (0.573 versus 0.579, observed 0.725), and Experiment 11 (0.081 versus 0.104, observed 0.109). Thus the embedded-singleton adjustment did not provide the intended selective correction. The principal unresolved mechanistic failure remains Experiment 6: the one-versus-many contrast is only -0.183 rather than -0.545. Experiment 3 is also persistently wrong-signed (-0.105 versus +0.072), Experiment 1 continues to show insufficient and highly heterogeneous first-cue adherence (0.71 with variance 0.201 versus 1.00 with zero variance), and Experiment 2 remains too anchor-consistent. In contrast, Experiments 8, 11, and 12 are acceptably close in their means, so their mechanisms should be protected.
Rationale: Build on the unchanged accepted iteration-3 base. The latest advice to condition embedded challenge on pre-protection compression was tested in isolation and rejected, so do not repeat or further amplify that channel. Instead, make one narrowly targeted in-family change to the compact-challenge mechanism: introduce or recalibrate a bounded accessibility term that operates only when there is a unique active anchor, exactly one opposing cue, and no active-top tie. The current type multipliers leave the challenge-resistant type almost immune in this exact geometry (`type_challenge=0.05`), which likely prevents the sharp human singleton effect in Experiment 6. Raise the accessibility floor for that geometry or alter the geometry-specific prevalence/loading of the existing challenge-sensitive type, while leaving embedded singleton challenge, two-opponent challenge, and the high-count tail unchanged. This exact condition should substantially deepen Experiment 6 and may reduce excessive highest-cue adherence in Experiment 2, while it is absent from Experiment-1 profiles and therefore should avoid another Experiment-1 regression. Use a bounded, type-dependent adjustment rather than another global increase in `compact_challenge`; preserve some resistant subjects so Experiment 6's variance can rise toward 0.046 rather than collapse. Freeze iteration 3's hard Experiment-4 normalization, active-tie equations and calibration, credibility protection, and high-count recovery parameters. Do not revisit the rejected convex normalization, add a two-opponent boost, or modify the tie gate in this pass. Experiment 3 should be addressed only after this isolated-singleton test, because the previous attempt to recover its sign damaged the more diagnostic Experiment-4 null. This single-knob intervention offers a cleaner test of whether ACAC can capture the missing compact-challenge amplitude without sacrificing its successful Experiment-11 and Experiment-12 fits.

**Outcome of this advice:** iter 5 candidate loss=0.1979 -> iter 6 candidate loss=0.1810 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: ACAC remains faithful to the prescribed mechanism family and the current iteration is a genuine accepted improvement. The narrowly targeted isolated-singleton accessibility term worked in its intended direction: Experiment 6 improved from -0.192 to -0.284, and Experiment 2 improved from 0.467 to 0.444, while the good fits in Experiments 7, 8, and 11 were largely preserved. Experiment 12 also retains a substantial positive activation effect. This validates the most recent advice and supports further tuning of the same exact compact-challenge channel. Nevertheless, the model is not yet experiment-invariant. The central singleton contrast in Experiment 6 remains only about half as negative as observed (-0.284 versus -0.545), and highest-validity-cue adherence in Experiment 2 remains too high (0.444 versus 0.324). Experiment 3 is robustly wrong-signed (-0.115 versus +0.072), Experiment 4 again shows a sizable forbidden crossover prevalence (0.260 versus 0), Experiment 9 underpredicts isolated-anchor adherence (0.574 versus 0.725), and Experiment 10 overpredicts it (0.644 versus 0.446). Experiment 1 also remains too low and too heterogeneous (0.790, variance 0.161, versus 1.000 with zero variance). The opposite errors in Experiments 9 and 10 continue to show insufficient contextual discrimination, but previous attempts to repair that with embedded or two-opponent challenge were rejected, so those changes should not be repeated now. Variance is generally too small in Experiments 2, 6, 9, 10, 11, and 12, while Experiment 4 has entirely spurious heterogeneity.
Rationale: Build on this accepted candidate and extend the one intervention that the gate has now validated. Increase the exact unique-anchor, isolated one-opponent accessibility effect by a modest step, but tune its type-specific loadings rather than globally increasing compact_challenge. In particular, raise challenge accessibility primarily for the integrative type and somewhat for the resistant type while leaving the already challenge-sensitive type near its current setting. This should deepen Experiment 6 and further reduce Experiment 2 adherence without touching embedded singleton, two-opponent, high-count, or active-tie configurations. Avoid making the accessibility floor nearly identical across types: Experiment 6 needs both a more negative mean and greater between-subject variance, so retain a meaningful resistant-versus-sensitive separation or adjust the geometry-specific type mixture rather than homogenizing all subjects. Freeze the high-count tail and credibility settings that keep Experiment 11 close, and freeze the active-tie equations that preserve the positive Experiment-12 effect and the near-null Experiment-7 diagnostic. Do not repeat the rejected redundancy-conditioned embedded challenge, two-opponent boost, or convex normalization package. Also audit why Experiment 4 changed from 0.08 in the accepted iteration-3 base to 0.26 even though the new accessibility condition should be inactive in its multi-cue-versus-multi-cue trials. If this is Monte Carlo or parameter-range instability, tighten the normalization-related ranges around values that produce a stable null rather than changing the normalization mechanism. If the exact singleton branch is unexpectedly firing there, correct that gating bug. Defer a new Experiment-3 normalization redesign until the singleton extension and Experiment-4 audit are isolated, since the previous attempt to recover Experiment 3 damaged the more diagnostic crossover null.

**Outcome of this advice:** iter 6 candidate loss=0.1810 -> iter 7 candidate loss=0.1667 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed ACAC family and is another accepted improvement. The exact isolated-singleton intervention again worked in the intended direction: Experiment 6 moved from -0.284 to -0.322 and Experiment 2 from 0.444 to 0.413, while Experiments 8 and 12 are essentially exact, Experiment 11 is close, and Experiments 5 and 7 remain reasonable. Nevertheless, the fit is not yet experiment-invariant. Experiment 6 is still substantially too weak (-0.322 versus -0.545), and its variance has fallen to 0.010 versus 0.046, indicating that the latest type-specific accessibility increase made subjects more uniformly challengeable rather than reproducing heterogeneous singleton sensitivity. Experiment 2 remains too anchor-consistent and too homogeneous. Large failures also persist in Experiment 3, where the interaction has the wrong sign (-0.114 versus +0.072), Experiment 4, where a forbidden crossover remains prevalent (0.20 versus 0), Experiment 9, where anchor adherence is too low (0.581 versus 0.725), and Experiment 10, where it is too high (0.634 versus 0.446). Experiment 1 likewise remains too low and highly heterogeneous relative to the unanimous human classification. Thus the accepted singleton adjustment improved aggregate loss but has not resolved the broader coalition-geometry and variance problems.
Rationale: Build on this accepted candidate, but do not simply increase isolated_singleton_access again. That direction was accepted twice, but the latest step delivered a smaller mean improvement in Experiment 6 while reducing rather than increasing its variance. Instead, re-separate the stable competition types within the same exact unique-anchor, one-opponent branch: preserve a genuinely resistant type with little or no added accessibility, make the integrative and challenge-sensitive types sufficiently strong, and use explicit nonuniform mixture probabilities if necessary to lower the population mean without homogenizing subjects. This should continue improving Experiments 2 and 6 while restoring their missing between-subject variance.

Independently audit Experiment 4 at the probability level rather than attributing repeated values of 0.08–0.26 solely to Monte Carlo fluctuation. If its expected conditional crossover is already zero under the hard symmetric normalization, add higher response precision or lower lapse only in the unique-anchor, multi-cue-versus-multi-cue normalized geometry. That should suppress threshold-classification false positives in Experiment 4 and may also raise the overly variable Experiment-1 prevalence, without introducing a generic order weight. If the expected crossover is nonzero, correct the normalization symmetry instead; the singleton branch cannot be responsible because it is inactive there.

Finally, inspect the discontinuous normalization gate requiring both coalitions to contain more than one cue. It normalizes Experiment-3-like balanced 2-v-2 profiles but switches off for imbalanced 3-v-1 profiles, which is a plausible source of the persistent wrong-signed interaction. Do not revive the previously rejected convex validity-composition blend. A safer in-family test is a smooth activation of the existing symmetric effective-count normalization around coalition balance, while explicitly exempting the exact isolated 1-v-1 branch and freezing the successful tie equations and high-count tail. Preserve the current calibrations for Experiments 8, 11, and 12.

**Outcome of this advice:** iter 7 candidate loss=0.1667 -> iter 8 candidate loss=0.1455 -> the gate marked it ACCEPTED.

### Iteration 8 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed Active-Set Conditional Competition family and is an accepted aggregate improvement. The normalized-geometry precision adjustment appears useful: Experiment 1 improved substantially from 0.77 to 0.92, Experiment 4 declined from 0.20 to 0.14, and Experiment 3 moved slightly toward zero. Experiments 7, 8, and 11 remain close, while Experiments 5 and 12 retain the required positive active-tie effects. However, the model is not yet experiment-invariant. Its central compact-challenge diagnostics remain materially wrong. Experiment 6 regressed from -0.322 to -0.269 against -0.545 observed, and Experiment 2 regressed from 0.413 to 0.437 against 0.324. Thus the new six-cell mixture increased variance somewhat but did not deliver the claimed stronger population-level isolated-singleton challenge. Experiment 3 remains wrong-signed (-0.100 versus +0.072), and Experiment 4 still produces a forbidden crossover in 14% of subjects rather than zero. The opposing errors in Experiments 9 and 10 are also large: the model challenges the Experiment-9 anchor too much (0.568 versus 0.725) but the Experiment-10 anchor too little (0.619 versus 0.446). This indicates inadequate discrimination by absolute credibility, validity gap, and embedded-versus-isolated coalition geometry. Finally, variance remains too low in Experiments 2, 9, 10, 11, and 12, while Experiment 1 retains heterogeneity absent from the human classification.
Rationale: Build on this accepted candidate, preserving the normalized-geometry precision change that improved Experiments 1 and 4, but do not further increase the same six-cell singleton loadings blindly. Instead, inspect the Experiment-6 lone- and multi-opponent choice probabilities separately for each competition type. The nominally strong types are evidently not producing the intended aggregate contrast. Recalibrate only the exact unique-anchor, isolated one-opponent branch so that most types have lower lone-opponent anchor adherence, while a smaller genuinely resistant type remains sharply separated. If necessary, use nonuniform type probabilities rather than six equal cells. This should deepen Experiment 6 and reduce Experiment 2 while retaining rather than homogenizing between-subject variation.

The accepted geometry-specific precision adjustment can be increased only modestly or its beta/lapse range tightened, conditional on verifying that Experiment 4's expected conditional crossover is zero. That is a promising way to move Experiment 4 from 0.14 toward zero and Experiment 1 from 0.92 toward unanimity without introducing an order weight. It should remain confined to unique-anchor multi-cue-versus-multi-cue trials.

For Experiment 3, address the still-discontinuous normalization gate rather than reviving the rejected convex validity-composition blend. Smoothly extend the existing symmetric effective-count normalization from balanced 2-v-2 profiles toward the relevant 3-v-1 profiles, while exempting isolated 1-v-1 competition. The target is to reduce anchor adherence selectively in the imbalanced condition so the balanced-minus-imbalanced interaction becomes positive, without disturbing Experiment 4's already-normalized profiles.

Finally, use the existing credibility and gap channels to separate Experiments 9 and 10. Strengthen protection specifically for isolated, high-credibility, well-separated anchors to raise Experiment 9, while modestly increasing the existing embedded-singleton challenge scale only for weak or small-gap anchors to lower the singleton component of Experiment 10. This differs from the previously rejected redundancy-conditioned embedded-challenge proposal and should not require a broad two-opponent boost. Freeze the active-tie equations and high-count tail, since Experiments 11 and 12 have good means. After these mean corrections, broaden geometry-specific random effects for active-conflict tie gating and high-count recovery to improve the low variances without reintroducing Experiment-1 or Experiment-4 heterogeneity.

**Outcome of this advice:** iter 8 candidate loss=0.1455 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Active-Cue Accessibility Competition (ACAC) proposes that people use a stable representation of communicated cue validity and evaluate only cues that discriminate on the current trial. Active cues first form option-specific coalitions. Coalition support accumulates smoothly but is compressed according to coalition size, validity similarity, and effective cue multiplicity, so additional concordant cues become increasingly redundant without triggering a categorical restoration rule. In unique-anchor configurations where both options have multiple active supporters, coalition evidence is additionally normalized toward effective coalition counts. This prevents the allocation of one relatively valid lower cue from producing an implicit presentation-order crossover. Decisions in this symmetrically normalized geometry are made with greater response precision, reflecting reduced conflict after cue identities have been compressed; this suppresses sampling-induced primacy classifications without introducing an order weight. The strongest currently active cue serves as a validity anchor. Opposition receives a sharply count-selective challenge enhancement centered on exactly one opponent. This enhancement is strongest for an isolated anchor, while embedded singleton dissent receives a smaller challenge that remains sensitive to anchor weakness and opponent-to-anchor validity gaps. Larger opposing coalitions retain only a small, smooth inverse-count challenge tail. In the exact unique-anchor, isolated, one-opponent configuration, stable competition types form a heterogeneous mixture containing a small genuinely resistant group and a larger set of strongly challenge-accessible integrators. Absolute anchor credibility and validity advantage continuously protect credible isolated anchors while leaving weak or closely matched anchors challengeable. Presentation order is consulted only when multiple currently active cues share maximal communicated validity. A bounded active-tie gate favors the earliest member of that tied set, with an additional conflict-geometry increment when lower-validity active cues precede tied maxima supporting opposing options. Inactive co-top cues cannot activate either component. Because the task provides no correctness feedback, ACAC explicitly excludes trial-history learning of cue concurrence, anti-concurrence, or discrimination frequency.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"ACAC expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    # Positive directions favor B and negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    validity_gain = float(parameters[\"validity_gain\"])\n    redundancy_strength = float(parameters[\"redundancy_strength\"])\n    redundancy_curvature = float(parameters[\"redundancy_curvature\"])\n    similarity_sensitivity = float(parameters[\"similarity_sensitivity\"])\n    compact_challenge = float(parameters[\"compact_challenge\"])\n    compact_peak = float(parameters[\"compact_peak\"])\n    compact_width = float(parameters[\"compact_width\"])\n    isolated_singleton_access = float(parameters[\"isolated_singleton_access\"])\n    challenge_tail = float(parameters[\"challenge_tail\"])\n    embedded_compact_scale = float(parameters[\"embedded_compact_scale\"])\n    challenge_gap_scale = float(parameters[\"challenge_gap_scale\"])\n    absolute_threshold = float(parameters[\"absolute_threshold\"])\n    absolute_slope = float(parameters[\"absolute_slope\"])\n    weak_anchor_gain = float(parameters[\"weak_anchor_gain\"])\n    isolation_gain = float(parameters[\"isolation_gain\"])\n    credibility_protection = float(parameters[\"credibility_protection\"])\n    active_tie_ceiling = float(parameters[\"active_tie_ceiling\"])\n    tie_activation_rate = float(parameters[\"tie_activation_rate\"])\n    conflict_tie_boost = float(parameters[\"conflict_tie_boost\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n    competition_type = int(parameters[\"competition_type\"])\n\n    # Nine correlated mixture cells reduce the prevalence of genuinely\n    # singleton-resistant subjects from one-sixth to one-ninth. The remaining\n    # cells retain two distinct routes to strong singleton accessibility.\n    if competition_type == 0:       # genuinely singleton-resistant\n        type_challenge = 0.05\n        type_singleton_access = 0.00\n    elif competition_type in (1, 2, 3, 4):  # strongly challenge-sensitive\n        type_challenge = 4.50\n        type_singleton_access = 0.00\n    else:                            # strongly accessible integrators\n        type_challenge = 1.00\n        type_singleton_access = 1.50\n\n    # Three cells occupy each tie-accessibility level, preserving the previous\n    # one-third marginal frequencies while changing singleton-type prevalence.\n    if competition_type in (0, 1, 2):\n        type_tie = 1.65\n    elif competition_type in (3, 4, 5):\n        type_tie = 0.30\n    else:\n        type_tie = 1.00\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # Communicated accuracy is represented on a log-evidence scale. Centering\n    # on the active maximum improves numerical stability without discarding\n    # absolute validity, which is retained separately below.\n    v = np.clip(validities, 0.500001, 0.999999)\n    reliability = np.log(v / (1.0 - v))\n    active_reference = float(np.max(reliability[active]))\n    cue_weight = np.exp(validity_gain * (reliability - active_reference))\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n\n    def coalition_support(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n\n        weights = np.asarray(cue_weight[indices], dtype=float)\n        raw_support = float(np.sum(weights))\n\n        # Effective multiplicity changes smoothly with coalition composition.\n        # A coalition containing several similarly accessible cues has a high\n        # effective count and is compressed more strongly than a heterogeneous\n        # coalition dominated by one cue.\n        effective_n = raw_support / max(float(np.max(weights)), 1e-12)\n        effective_excess = max(effective_n - 1.0, 0.0)\n\n        coalition_v = validities[indices]\n        validity_spread = float(np.std(coalition_v)) if n > 1 else 0.0\n        similarity = np.exp(-similarity_sensitivity * validity_spread)\n        discount = 1.0 + redundancy_strength * similarity * (\n            effective_excess ** redundancy_curvature\n        )\n        return raw_support / max(discount, 1e-12)\n\n    support_a = coalition_support(a_cues)\n    support_b = coalition_support(b_cues)\n\n    # The validity anchor is defined only over currently discriminating cues.\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n    top_candidates = active[np.isclose(active_v, best_v, atol=1e-8)]\n    anchor_cue = int(np.min(top_candidates))\n    anchor_direction = float(directions[anchor_cue])\n    unique_anchor = float(top_candidates.size == 1)\n\n    # In unique-anchor multi-cue-versus-multi-cue configurations, normalize\n    # coalition composition toward effective count support. This saturation\n    # prevents reallocating one relatively valid lower cue from inducing an\n    # implicit primacy crossover, while leaving isolated-anchor diagnostics\n    # and all tied-top configurations unchanged.\n    normalized_multicue_geometry = False\n    if unique_anchor > 0.0 and n_a > 1 and n_b > 1:\n        pooled_weight = float(np.mean(cue_weight[active]))\n\n        def normalized_multicue_support(n):\n            excess = float(n - 1)\n            discount = 1.0 + redundancy_strength * (\n                excess ** redundancy_curvature\n            )\n            return pooled_weight * float(n) / max(discount, 1e-12)\n\n        support_a = normalized_multicue_support(n_a)\n        support_b = normalized_multicue_support(n_b)\n        normalized_multicue_geometry = True\n\n    if anchor_direction > 0:\n        anchor_indices = b_cues\n        opponent_indices = a_cues\n        anchor_support = support_b\n        opponent_support = support_a\n    else:\n        anchor_indices = a_cues\n        opponent_indices = b_cues\n        anchor_support = support_a\n        opponent_support = support_b\n\n    anchor_count = int(anchor_indices.size)\n    opponent_count = int(opponent_indices.size)\n\n    # Absolute credibility and relative advantage are distinct. A cue can be\n    # locally strongest yet remain challengeable when all active cues have low\n    # communicated validity.\n    weak_anchor = sigmoid(absolute_slope * (absolute_threshold - best_v))\n    strong_anchor = 1.0 - weak_anchor\n\n    if opponent_count > 0:\n        strongest_opponent_v = float(np.max(validities[opponent_indices]))\n        opponent_gap = max(best_v - strongest_opponent_v, 0.0)\n    else:\n        opponent_gap = 0.5\n\n    gap_accessibility = np.exp(\n        -opponent_gap / max(challenge_gap_scale, 1e-8)\n    )\n    gap_protection = 1.0 - gap_accessibility\n\n    # Credibility protection is bounded and continuous. Its narrower gap scale\n    # separates genuinely well-spaced isolated anchors from close competitions\n    # without introducing a restoration threshold or a cue-order effect.\n    anchor_support *= 1.0 + (\n        unique_anchor\n        * credibility_protection\n        * strong_anchor\n        * gap_protection\n    )\n\n    if opponent_count > 0:\n        # The high-amplitude component is extremely localized at one opponent.\n        # Embedded singleton dissent receives only a bounded fraction of it,\n        # whereas isolated one-versus-one conflict expresses the full stable\n        # subject-type difference. In that exact geometry, a bounded access\n        # increment selectively raises strongly accessible integrators while a\n        # small mixture remains genuinely resistant.\n        count_distance = (\n            float(opponent_count) - compact_peak\n        ) / max(compact_width, 1e-8)\n        singleton_profile = np.exp(-0.5 * count_distance ** 2)\n\n        if anchor_count == 1:\n            singleton_geometry = 1.0\n        else:\n            singleton_geometry = embedded_compact_scale\n\n        active_type_challenge = type_challenge\n        if (\n            unique_anchor > 0.0\n            and anchor_count == 1\n            and opponent_count == 1\n            and top_candidates.size == 1\n        ):\n            active_type_challenge += (\n                isolated_singleton_access * type_singleton_access\n            )\n\n        singleton_intensity = (\n            active_type_challenge\n            * compact_challenge\n            * singleton_profile\n            * singleton_geometry\n        )\n        smooth_tail = challenge_tail / float(opponent_count + 1)\n\n        isolation = 1.0 / float(max(anchor_count, 1))\n        geometry_gain = 1.0 + isolation_gain * isolation\n        absolute_gain = 1.0 + weak_anchor_gain * weak_anchor\n\n        challenge_multiplier = 1.0 + (\n            singleton_intensity + smooth_tail\n        ) * gap_accessibility * geometry_gain * absolute_gain\n        opponent_support *= challenge_multiplier\n\n    total_support = anchor_support + opponent_support\n    if total_support <= 1e-12:\n        core_evidence = 0.0\n    else:\n        anchor_margin = (anchor_support - opponent_support) / total_support\n        core_evidence = anchor_direction * anchor_margin\n\n    # A bounded order gate is activated exclusively by a tie among currently\n    # active maximum-validity cues. Conflict among tied maxima receives an\n    # additional activation increment only when lower-validity active cues\n    # precede the tied set. This captures activation-sensitive accessibility\n    # without increasing first-cue adherence in unique-maximum profiles.\n    active_top_count = int(top_candidates.size)\n    if active_top_count >= 2:\n        tie_gate = (\n            type_tie\n            * active_tie_ceiling\n            * (1.0 - np.exp(\n                -tie_activation_rate * float(active_top_count - 1)\n            ))\n        )\n        tied_directions = directions[top_candidates]\n        tied_conflict = float(\n            np.any(tied_directions > 0) and np.any(tied_directions < 0)\n        )\n        preceded_tied_set = float(int(active[0]) < anchor_cue)\n        tie_gate += (\n            type_tie\n            * conflict_tie_boost\n            * tied_conflict\n            * preceded_tied_set\n        )\n    else:\n        tie_gate = 0.0\n    tie_gate = float(np.clip(tie_gate, 0.0, 0.95))\n\n    choice_evidence = (\n        (1.0 - tie_gate) * core_evidence\n        + tie_gate * anchor_direction\n    )\n\n    # Symmetric multi-cue normalization removes identity-based conflict, so\n    # choices in that geometry are more precise. The adjustment affects neither\n    # isolated singleton competition nor active-top-tie trials.\n    effective_beta = beta * (1.90 if normalized_multicue_geometry else 1.0)\n    logits = effective_beta * np.array(\n        [-0.5 * choice_evidence, 0.5 * choice_evidence],\n        dtype=np.float64,\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array(\n        [0.5, 0.5], dtype=np.float64\n    )\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    probs /= total\n    return probs.astype(np.float64)\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "validity_gain": "[0.05, 0.22]",
    "redundancy_strength": "[0.62, 1.08]",
    "redundancy_curvature": "[1.18, 1.58]",
    "similarity_sensitivity": "[5.0, 14.0]",
    "compact_challenge": "[5.0, 10.0]",
    "compact_peak": "[0.98, 1.02]",
    "compact_width": "[0.12, 0.25]",
    "isolated_singleton_access": "[1.8, 2.8]",
    "challenge_tail": "[0.08, 0.28]",
    "embedded_compact_scale": "[0.16, 0.32]",
    "challenge_gap_scale": "[0.055, 0.12]",
    "absolute_threshold": "[0.68, 0.77]",
    "absolute_slope": "[14.0, 32.0]",
    "weak_anchor_gain": "[0.65, 1.75]",
    "isolation_gain": "[0.35, 1.10]",
    "credibility_protection": "[0.12, 0.42]",
    "active_tie_ceiling": "[0.70, 0.90]",
    "tie_activation_rate": "[1.25, 2.60]",
    "conflict_tie_boost": "[0.60, 1.00]",
    "beta": "[1.45, 3.80]",
    "lapse": "[0.0, 0.10]",
    "competition_type": "{0, 1, 2, 3, 4, 5, 6, 7, 8}"
  },
  "rationale": "This is a minimal extension of the accepted iteration-8 candidate. First, the exact isolated-singleton mixture is expanded from six to nine equally likely correlated cells while retaining only one genuinely resistant cell. This lowers resistant prevalence from one-sixth to one-ninth and should deepen the Experiment-6 contrast and reduce Experiment-2 anchor adherence. Three cells remain assigned to each tie-accessibility level, preserving the one-third marginal distribution that supported Experiments 5 and 12. Second, the embedded singleton scale is increased modestly rather than adding the previously rejected broad two-opponent challenge. Because this channel is still multiplied by gap accessibility and weak-anchor gain, it should selectively lower imbalanced-condition anchor adherence in Experiment 3 and the embedded-singleton component of Experiment 10. Third, the validity-gap scale is narrowed and bounded credibility protection is strengthened. This makes high-credibility, well-separated isolated anchors more resistant, targeting Experiment 9, while close-gap or weak anchors remain challengeable. No categorical restoration threshold is introduced. Finally, normalized multi-cue precision increases only from 1.60 to 1.90, preserving the accepted mechanism while further suppressing Experiment-4 threshold false positives and moving Experiment 1 toward unanimous first-cue classification. Active-tie equations, the high-count challenge tail, history invariance, and all other model equations are unchanged."
}
```

## Usage

```json
{
  "input_tokens": 41059,
  "output_tokens": 5947
}
```
