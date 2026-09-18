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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with a new task-invariant theory of configuration-dependent selective integration that competes with ADCG without treating feedback-free exposure as sufficient evidence for learned cue dependence. The theory should combine: (1) communicated-validity weighting; (2) redundancy-discounted coalition accumulation; (3) static minority or dissent distinctiveness that can operate even when the highest-validity cue has allies, allowing singleton dissent to matter in Experiment 2; (4) a sharply nonmonotonic isolated-anchor mechanism in which one or two credible opponents can challenge the anchor but a larger concordant coalition can trigger redundancy-based anchor restoration, as required by Experiment 8; (5) bounded presentation-order influence, including tied-top anchoring, without a general primacy crossover; and (6) stable subject-level mixtures or random effects over validity sensitivity, order use, minority amplification, restoration thresholds, temperature, and lapse to reproduce the observed heterogeneity. History-dependent anti-concurrence should be absent without correctness feedback, or confined to a small optional component that activates only with strong independent evidence and cannot generate the spurious Experiment 9 contrast. The new model should separately predict compact isolated-anchor trials and singleton-dissent trials rather than fitting their pooled Experiment 2 metric through one mechanism. Quantitatively, it should target intermediate anchor following in Experiment 1, low pooled anchor following in Experiment 2, a near-zero diagnostic-learning contrast in Experiment 9, and the large negative one-versus-multiple-opponent contrast in Experiment 8. This preserves SCCI's useful challenge/restoration insight while removing its degenerative restrictions and provides a stronger, more falsifiable competitor to ADCG.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.635)

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


---

### `pi_5` (overall score: 0.506)

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


---

### `pi_1` (overall score: 0.417)

**Description**
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.9900 (var=0.0049)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.8492 (var=0.0121)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.0008 (var=0.0015)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0400 (var=0.0384)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.8244 (var=0.0081)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=0.0103 (var=0.0049)
- Experiment 7: real=0.0052 (var=0.0075) vs this=-0.0058 (var=0.0071)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0002 (var=0.0000)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.8600 (var=0.0122)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.8250 (var=0.0150)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2803 -> ACCEPTED
- iter 2: loss=0.2859 -> REJECTED
- iter 3: loss=0.1994 -> ACCEPTED
- iter 4: loss=0.1886 -> ACCEPTED
- iter 5: loss=0.1677 -> ACCEPTED
- iter 6: loss=0.1123 -> ACCEPTED
- iter 7: loss=0.1073 -> ACCEPTED
- iter 8: loss=0.1238 -> REJECTED
- iter 9: loss=0.1204 -> REJECTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.1073 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.9500 (var=0.0425) (Δ vs real -0.0500)
  - iter 2: 0.9800 (var=0.0196) (Δ vs real -0.0200)
  - iter 3: 0.9000 (var=0.0800) (Δ vs real -0.1000)
  - iter 4: 0.9800 (var=0.0196) (Δ vs real -0.0200)
  - iter 5: 0.9500 (var=0.0425) (Δ vs real -0.0500)
  - iter 6: 0.9200 (var=0.0736) (Δ vs real -0.0800)
  - iter 7: 0.9900 (var=0.0049) (Δ vs real -0.0100)
  - iter 8: 0.9300 (var=0.0601) (Δ vs real -0.0700)
  - iter 9 (most recent): 0.9700 (var=0.0241) (Δ vs real -0.0300)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)

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
  - iter 1: 0.5373 (var=0.0075) (Δ vs real +0.2131)
  - iter 2: 0.5833 (var=0.0093) (Δ vs real +0.2592)
  - iter 3: 0.5813 (var=0.0030) (Δ vs real +0.2571)
  - iter 4: 0.5787 (var=0.0041) (Δ vs real +0.2546)
  - iter 5: 0.5867 (var=0.0036) (Δ vs real +0.2625)
  - iter 6: 0.4267 (var=0.0045) (Δ vs real +0.1025)
  - iter 7: 0.5281 (var=0.0037) (Δ vs real +0.2040)
  - iter 8: 0.5290 (var=0.0039) (Δ vs real +0.2048)
  - iter 9 (most recent): 0.5429 (var=0.0032) (Δ vs real +0.2188)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)

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
  - iter 1: 0.1102 (var=0.0068) (Δ vs real +0.0381)
  - iter 2: 0.0063 (var=0.0035) (Δ vs real -0.0658)
  - iter 3: 0.0779 (var=0.0034) (Δ vs real +0.0058)
  - iter 4: 0.1079 (var=0.0051) (Δ vs real +0.0358)
  - iter 5: 0.1367 (var=0.0033) (Δ vs real +0.0646)
  - iter 6: -0.0617 (var=0.0043) (Δ vs real -0.1338)
  - iter 7: 0.0525 (var=0.0037) (Δ vs real -0.0196)
  - iter 8: 0.0427 (var=0.0038) (Δ vs real -0.0294)
  - iter 9 (most recent): 0.0356 (var=0.0043) (Δ vs real -0.0365)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)

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
  - iter 1: 0.6000 (var=0.2400) (Δ vs real +0.6000)
  - iter 2: 0.5000 (var=0.2500) (Δ vs real +0.5000)
  - iter 3: 0.1200 (var=0.1056) (Δ vs real +0.1200)
  - iter 4: 0.1800 (var=0.1476) (Δ vs real +0.1800)
  - iter 5: 0.0800 (var=0.0736) (Δ vs real +0.0800)
  - iter 6: 0.2000 (var=0.1600) (Δ vs real +0.2000)
  - iter 7: 0.1000 (var=0.0900) (Δ vs real +0.1000)
  - iter 8: 0.1000 (var=0.0900) (Δ vs real +0.1000)
  - iter 9 (most recent): 0.1400 (var=0.1204) (Δ vs real +0.1400)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.2400 (var=0.1824)

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
  - iter 1: 0.7925 (var=0.0066) (Δ vs real +0.0025)
  - iter 2: 0.7473 (var=0.0092) (Δ vs real -0.0427)
  - iter 3: 0.7977 (var=0.0071) (Δ vs real +0.0077)
  - iter 4: 0.7690 (var=0.0092) (Δ vs real -0.0210)
  - iter 5: 0.7910 (var=0.0051) (Δ vs real +0.0010)
  - iter 6: 0.8106 (var=0.0083) (Δ vs real +0.0206)
  - iter 7: 0.7898 (var=0.0055) (Δ vs real -0.0002)
  - iter 8: 0.8048 (var=0.0077) (Δ vs real +0.0148)
  - iter 9 (most recent): 0.7971 (var=0.0069) (Δ vs real +0.0071)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)

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
  - iter 1: -0.0944 (var=0.0101) (Δ vs real +0.4506)
  - iter 2: -0.1503 (var=0.0158) (Δ vs real +0.3947)
  - iter 3: 0.0741 (var=0.0161) (Δ vs real +0.6191)
  - iter 4: -0.2309 (var=0.0132) (Δ vs real +0.3141)
  - iter 5: -0.2650 (var=0.0127) (Δ vs real +0.2800)
  - iter 6: -0.3928 (var=0.0156) (Δ vs real +0.1522)
  - iter 7: -0.4072 (var=0.0215) (Δ vs real +0.1378)
  - iter 8: -0.3181 (var=0.0371) (Δ vs real +0.2269)
  - iter 9 (most recent): -0.4503 (var=0.0180) (Δ vs real +0.0947)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)

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
  - iter 1: -0.0899 (var=0.0173) (Δ vs real -0.0951)
  - iter 2: -0.0942 (var=0.0201) (Δ vs real -0.0993)
  - iter 3: -0.0333 (var=0.0166) (Δ vs real -0.0384)
  - iter 4: -0.0398 (var=0.0126) (Δ vs real -0.0450)
  - iter 5: -0.0208 (var=0.0249) (Δ vs real -0.0260)
  - iter 6: -0.0049 (var=0.0187) (Δ vs real -0.0101)
  - iter 7: -0.0294 (var=0.0188) (Δ vs real -0.0346)
  - iter 8: -0.0292 (var=0.0204) (Δ vs real -0.0343)
  - iter 9 (most recent): -0.0272 (var=0.0156) (Δ vs real -0.0323)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)

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
  - iter 1: 0.0059 (var=0.0002) (Δ vs real -0.0011)
  - iter 2: 0.0017 (var=0.0001) (Δ vs real -0.0053)
  - iter 3: 0.0124 (var=0.0002) (Δ vs real +0.0054)
  - iter 4: 0.0164 (var=0.0003) (Δ vs real +0.0094)
  - iter 5: 0.0238 (var=0.0004) (Δ vs real +0.0168)
  - iter 6: 0.0090 (var=0.0002) (Δ vs real +0.0020)
  - iter 7: 0.0025 (var=0.0001) (Δ vs real -0.0045)
  - iter 8: 0.0041 (var=0.0001) (Δ vs real -0.0029)
  - iter 9 (most recent): 0.0014 (var=0.0001) (Δ vs real -0.0056)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)

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
  - iter 1: 0.3887 (var=0.0206) (Δ vs real -0.3362)
  - iter 2: 0.7444 (var=0.0417) (Δ vs real +0.0194)
  - iter 3: 0.8075 (var=0.0116) (Δ vs real +0.0825)
  - iter 4: 0.8219 (var=0.0096) (Δ vs real +0.0969)
  - iter 5: 0.7325 (var=0.0148) (Δ vs real +0.0075)
  - iter 6: 0.7412 (var=0.0155) (Δ vs real +0.0162)
  - iter 7: 0.7425 (var=0.0146) (Δ vs real +0.0175)
  - iter 8: 0.7675 (var=0.0097) (Δ vs real +0.0425)
  - iter 9 (most recent): 0.7575 (var=0.0171) (Δ vs real +0.0325)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)

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
  - iter 1: 0.3488 (var=0.0112) (Δ vs real -0.0975)
  - iter 2: 0.6813 (var=0.0230) (Δ vs real +0.2350)
  - iter 3: 0.5725 (var=0.0070) (Δ vs real +0.1263)
  - iter 4: 0.5288 (var=0.0041) (Δ vs real +0.0825)
  - iter 5: 0.4612 (var=0.0062) (Δ vs real +0.0150)
  - iter 6: 0.4906 (var=0.0068) (Δ vs real +0.0444)
  - iter 7: 0.4775 (var=0.0084) (Δ vs real +0.0312)
  - iter 8: 0.4813 (var=0.0068) (Δ vs real +0.0350)
  - iter 9 (most recent): 0.4825 (var=0.0055) (Δ vs real +0.0363)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Partitioned Dissent and Redundancy Restoration (PDRR) proposes that people construct a stable, instruction-based evidence representation and classify each choice configuration before integrating its cues. Communicated validities determine cue strength, while cues supporting the same option accumulate sublinearly because concordant reports are treated as partly redundant. In ordinary embedded coalitions, individual cue weights are strongly compressed before accumulation, limiting the influence of any single non-anchor cue while preserving communicated-validity ordering. Opposition to a validity anchor is statically distinctive even when the anchor has allies, but balanced opposition receives a counter-anchor bonus only when the opposing cues are individually close in communicated validity to the anchor and the coalition is broad enough to support a coalition-level interpretation. Isolated-anchor opposition is processed separately. A single opponent receives especially strong challenge status, with stable subject types differing substantially in their sensitivity to this challenge, while exactly two opponents retain a joint-credibility mechanism with greater type-specific heterogeneity. Coalitions of three or more trigger a sharp count-dominant redundancy-restoration branch. Embedded singleton dissent is strongly amplified only against very broad coalitions; its generic effect against smaller coalitions is attenuated to avoid excessive profile variance. Presentation order affects choice primarily as a bounded tie breaker among currently discriminating cues with equal maximal communicated validity, avoiding a general primacy crossover. All mechanisms are history-invariant in this feedback-free task. Stable subject types and random effects over validity sensitivity, balanced dissent, compact challenge, redundancy, temperature, and lapse generate persistent individual differences.

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
        # separated stable subject types. The exactly-two branch is unchanged.
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
        if opposition_count == 1 and anchor_side_count > 1:
            imbalance = (
                float(anchor_side_count - opposition_count)
                / float(anchor_side_count + opposition_count)
            )
            dissent_multiplier = 1.0 + minority_amplification * imbalance
            breadth_weight = 1.0 if anchor_side_count >= 5 else 0.20
            dissent_multiplier += (
                singleton_type_gain
                * singleton_bonus
                * breadth_weight
                * np.log1p(float(anchor_side_count - 1))
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
    # almost off at two and near its ceiling at three or more opponents.
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
        count_dominant_restoration = (
            restoration_ceiling
            * restoration_type_gain
            * large_count_gate
        )
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

`rationale`: This is a minimal, localized edit of the accepted PDRR model. The unconditional balanced-opposition bonus is replaced by the requested validity-geometry gate: opposition is amplified in a balanced embedded configuration only to the extent that its strongest member is close to the anchor in communicated validity. A bounded breadth factor gives 2-versus-2 conflicts only half of the maximum coalition bonus while leaving broader credible conflicts intact. This should preserve much of the Experiment 2 reduction in anchor following while removing excessive anchor challenge in Experiment 3 and reducing the cue-alignment sensitivity responsible for Experiment 4's crossover. No order, ordinary-compression, history, singleton, or restoration equation is changed. The only isolated-anchor adjustment is a modest further separation of stable subject types. The challenge-sensitive type receives stronger one-opponent challenge, targeting the residual Experiment 6 error, while the two-opponent gains are spread around approximately the same mixture mean. That should increase heterogeneity and slightly reduce excessive anchor following in Experiments 9 and 10 without changing the large-count restoration branch that protects Experiment 1.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: PDRR is faithful to the prescribed configuration-dependent, history-invariant mechanism family and is an accepted improvement, but it is not yet experiment-invariant. It fits Experiment 5 almost exactly (0.7925 vs. 0.7900), Experiment 8 well (0.0059 vs. 0.0070), and is reasonably close on Experiments 1 and 3. The central challenge/restoration predictions, however, remain substantially miscalibrated. Anchor following is too high in Experiment 2 (0.5373 vs. 0.3242), while the one-versus-multiple contrast in Experiment 6 is far too weak (-0.0944 vs. -0.5450). Conversely, two-opponent isolated-anchor adherence is much too low in Experiment 9 (0.3887 vs. 0.7250). Thus, opponent count alone is not separating challenge from restoration correctly across configurations. Experiment 4 also shows a large spurious crossover prevalence (0.6000 vs. 0), despite the explicit order gate being inactive there; this crossover must therefore be induced indirectly by the broad count-based minority amplification and validity-weighted coalition composition. Experiment 7 is history-invariant as required, but still produces a non-null static diagnostic-control contrast (-0.0899 vs. 0.0052), so omitting learning is necessary but not sufficient. Experiment 10 is moderately too anti-anchor (0.3488 vs. 0.4462). Between-subject variance is also too low in several critical experiments, especially Experiments 2, 6, 9, and 10.
Rationale: Keep PDRR's mechanism family, but recalibrate how configuration gates challenge, dissent, and restoration. First, the compact challenge should not be a function of opponent count alone. Gate its amplitude by the opponents' combined communicated reliability relative to the anchor. This can distinguish the credible compact opposition requiring low adherence in Experiments 2 and 10 from the weaker two-cue coalition in Experiment 9, where adherence must rise toward 0.725. Second, make the challenge peak more concentrated at one opponent and lower the restoration threshold so that multi-opponent trials restore the anchor much more strongly. That joint change should both lower one-opponent adherence in Experiment 6 and raise multi-opponent adherence, producing the required large negative contrast rather than merely shifting all isolated-anchor trials together. Third, replace the broad embedded-dissent multiplier applied whenever the anchor side is numerically larger with a sharply singleton-localized, bounded distinctiveness gate based on support ratio as well as count. The current general imbalance amplification creates the severe Experiment 4 crossover even without an explicit primacy term. Preserve strong singleton effects where needed for Experiment 2, but suppress amplification for ordinary two-versus-four and similar coalitions. Fourth, tune this singleton gate so that the matched diagnostic and control configurations in Experiment 7 have nearly equal static anchor adherence; the current -0.09 contrast is a configuration artifact, not learning. Finally, widen stable subject-level random effects for challenge strength, restoration threshold, singleton amplification, temperature, and lapse, since the critical simulated variances are systematically below the observed heterogeneity. Do not add history dependence or broaden the order mechanism: the tied-top order fit in Experiment 5 is already excellent, and the Experiment 4 failure originates elsewhere.

**Outcome of this advice:** iter 1 candidate loss=0.2803 -> iter 2 candidate loss=0.2859 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family, but the current revision was rejected by the loss gate and should not replace the accepted iteration-1 base. It successfully repaired Experiment 9 (0.7444 vs. 0.7250) and modestly strengthened the Experiment 6 contrast, but these gains came from a global challenge/restoration recalibration that substantially worsened Experiment 2 (0.5833 vs. 0.3242), Experiment 10 (0.6813 vs. 0.4462), Experiment 3 (0.0063 vs. 0.0721), and Experiment 5 (0.7473 vs. 0.7900). The key dissociation is now especially clear: the two-opponent coalition in Experiment 9 should preserve anchor following, whereas compact opposition in Experiment 10 should challenge it. Redundancy-discounted total support does not distinguish these cases adequately. Experiment 6 still has far too little challenge/restoration separation (-0.1503 vs. -0.5450), and Experiment 4 retains a severe spurious crossover prevalence (0.5000 vs. 0), even though its configurations do not activate the new singleton gate. That implicates ordinary validity-weighted coalition composition, rather than minority amplification alone. Experiment 7 also remains spuriously negative (-0.0942 vs. 0.0052). Fits to Experiments 1, 8, and 9 are good, but the pattern is not experiment-invariant.
Rationale: Because the previous recommendation's global lowering/sharpening of restoration and support-ratio gating was rejected, do not repeat or intensify those shifts. Build from the unchanged accepted iteration-1 base and use a more selective in-family factorization. For isolated-anchor trials, distinguish coalitions using the reliability of the strongest and second-strongest individual opponents, or an opponent-reliability dispersion statistic, rather than only their redundancy-discounted summed support. Couple restoration to a two-dimensional count-by-validity-gap gate: multi-opponent coalitions with a clearly superior anchor should restore strongly, as in Experiments 6 and 9, while two individually credible opponents with a small anchor gap should retain challenge, as in Experiment 10. Keep compact challenge and embedded singleton dissent on separately parameterized channels so lowering pooled anchor following in Experiment 2 does not globally suppress anchor following in Experiment 9 or 10.

Separately address Experiment 4 at the ordinary-coalition accumulator. Since the singleton gate is inactive there, bound the marginal influence of any one non-anchor cue or normalize non-anchor weights within each coalition before count discounting. This should make swapping Expert 1's alignment less capable of producing a subject-level crossover while preserving communicated-validity weighting and the tied-top order mechanism. Apply this compression only to embedded ordinary coalitions, not isolated-anchor trials. Re-establish the accepted base's tied-top parameters, which already fit Experiment 5 almost exactly, rather than changing order-related ranges. Finally, tune stable mixture weights rather than simply widening all parameter ranges: include challenge-sensitive, gap-restoring, and weakly configuration-sensitive subject types so the variances in Experiments 6, 9, and 10 can increase without moving all means together. Keep history dependence absent.

**Outcome of this advice:** iter 2 candidate loss=0.2859 -> iter 3 candidate loss=0.1994 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to the prescribed configuration-dependent, history-invariant PDRR family. The previously recommended selective factorization helped: Experiment 3 is now nearly exact (0.0779 vs. 0.0721), the spurious Experiment 4 crossover fell sharply from 0.6000 to 0.1200, Experiment 7 moved toward zero (-0.0333 vs. 0.0052), and the tied-top order effect remains excellent in Experiment 5 (0.7977 vs. 0.7900). Experiments 8 and 9 are also reasonably close in mean. Nevertheless, the central challenge/restoration signature still fails. Experiment 6 has the wrong sign and a very large error: the candidate predicts greater anchor adherence with one opponent than with multiple opponents (0.0741), whereas humans show a large negative contrast (-0.5450). Experiment 2 also remains much too anchor-following (0.5813 vs. 0.3242), and Experiment 10 is too anchor-following (0.5725 vs. 0.4462). Thus, credible compact opposition and singleton dissent remain too weak, while count-based restoration is not reliably dominating for larger coalitions. Experiment 9 is somewhat over-restored (0.8075 vs. 0.7250). The critical between-subject variances are also compressed, especially in Experiments 2, 6, 9, and 10. Experiment 1 has deteriorated to 0.9000 versus unanimous observed prevalence, likely because some large-coalition profiles are not entering restoration consistently.
Rationale: Continue from this accepted candidate and preserve the successful ordinary-coalition compression and tied-top order settings; the previous advice in that direction was accepted and substantially repaired Experiments 3, 4, and 7. The next revision should target the isolated-anchor gates rather than globally changing integration. First, inspect and recalibrate the conjunction of the three restoration gates. Multiplying count, validity-gap, and credibility-gap sigmoids appears to suppress restoration too often in Experiment 6, even for large coalitions. Add a count-dominant restoration branch for clearly large coalitions, such as count at least three, that is bounded but does not require all gap gates to be simultaneously near one. This should raise multi-opponent anchor adherence and recover the negative Experiment 6 contrast while also protecting Experiment 1. Do not globally lower the restoration threshold, since that earlier broad strategy was rejected and would further overfit Experiment 9. Second, make compact challenge asymmetric across counts rather than using the current profile centered symmetrically at 1.5: allow especially strong one-opponent challenge, with a separately calibrated two-opponent challenge controlled tightly by individual opponent credibility. Stronger challenge for genuinely near-anchor opponents should lower adherence in Experiment 10 and relevant compact trials in Experiment 2 without undoing weak-opponent restoration in Experiment 9. Third, strengthen or broaden the embedded singleton-dissent channel selectively for the Experiment 2 configurations; its current bounded multiplier is insufficient to produce the observed pooled rate. Use a support-ratio or dissent-validity gate so this adjustment does not revive the Experiment 4 crossover. Finally, implement genuinely stable mixture components or correlated random effects—challenge-sensitive, count-restoring, and weakly selective types—rather than only independent narrow parameter draws. This is needed to increase variance in Experiments 6 and 9 while allowing some subjects to challenge strongly enough in Experiments 2 and 10. Keep history dependence absent and retain the current bounded tied-top order mechanism.

**Outcome of this advice:** iter 3 candidate loss=0.1994 -> iter 4 candidate loss=0.1886 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The accepted revision is a further aggregate improvement and remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The count-dominant restoration branch moved Experiment 6 in the correct direction, from +0.0741 to -0.2309, and restored Experiment 1 from 0.9000 to 0.9800. The asymmetric challenge also improved Experiment 10 from 0.5725 to 0.5288. These gains confirm that the most recent, accepted advice identified useful in-family controls. Fits are good for Experiments 1, 3, 5, 7, and 8. Nevertheless, the model is not yet experiment-invariant. Experiment 2 remains the largest qualitative failure: anchor following is 0.5787 versus 0.3242, barely changed despite stronger compact and singleton mechanisms. Experiment 6 still expresses less than half of the observed challenge/restoration separation (-0.2309 versus -0.5450). Experiment 9 is now over-restored (0.8219 versus 0.7250), while Experiment 10 remains too anchor-following (0.5288 versus 0.4462). Experiment 4 also retains a nonzero spurious crossover prevalence (0.1800 versus 0), although it is much better than early iterations. Finally, stable subject heterogeneity is still much too weak in the critical configurations: variances are 0.0041 versus 0.0174 in Experiment 2, 0.0132 versus 0.0462 in Experiment 6, 0.0096 versus 0.0723 in Experiment 9, and 0.0041 versus 0.0224 in Experiment 10.
Rationale: Continue from this accepted candidate and preserve the successful ordinary-coalition compression, history invariance, and tied-top order mechanism. Extend the accepted count-restoration adjustment rather than reversing it: make the count-dominant gate closer to zero for exactly two opponents but steeper and nearer its ceiling for three or more. Also attenuate the selective restoration branch more strongly at exactly two opponents. This should lower the over-restored Experiment 9 value while raising multi-opponent adherence in Experiment 6, increasing the required negative contrast without globally lowering the restoration threshold—a broad move that was previously rejected.

The persistent Experiment 2 failure now requires condition-level calibration rather than another pooled global shift. Report or internally target its compact isolated-anchor and embedded-dissent trial classes separately. Increase challenge selectively for a single credible opponent, and increase two-opponent challenge only when both opponents are individually near the anchor in communicated validity. A modestly sharper credibility-gap gate paired with greater challenge amplitude can lower the credible compact conditions in Experiments 2 and 10 while sparing the weaker two-opponent coalition in Experiment 9. Strengthen embedded singleton dissent only where the dissenter is sufficiently credible relative to the anchor coalition; do not broaden it to ordinary two-versus-four profiles, given the remaining Experiment 4 crossover.

For Experiment 4, tighten the existing ordinary-coalition compression slightly or cap the marginal contribution of the strongest non-anchor cue, rather than changing order use or the isolated-anchor branches. Finally, increase separation among the stable configuration types specifically in one- and two-opponent challenge strength and restoration thresholds, and adjust their mixture weights. The current type gains do not produce the observed heterogeneity. Avoid merely widening all independent ranges, which could damage the near-unanimous Experiment 1 pattern. These are localized calibrations within PDRR, not a change of mechanism family.

**Outcome of this advice:** iter 4 candidate loss=0.1886 -> iter 5 candidate loss=0.1677 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is an accepted aggregate improvement and remains faithful to the prescribed configuration-dependent, history-invariant PDRR family. The latest localized changes were productive: Experiment 9 is now essentially correct in mean (0.7325 vs. 0.7250), Experiment 10 is close (0.4612 vs. 0.4462), Experiment 4 improved to 0.0800, and the tied-top order prediction in Experiment 5 remains nearly exact. Experiment 7 is also acceptably near zero, and no spurious exposure-learning mechanism has appeared. However, the core cross-experiment signature is still not captured. Experiment 2 remains dramatically too anchor-following (0.5867 vs. 0.3242), and the large negative one-versus-multiple contrast in Experiment 6 is still only about half its observed magnitude (-0.2650 vs. -0.5450). Experiment 3 has also moved away from its earlier near-exact fit (0.1367 vs. 0.0721), while Experiment 8 now has excessive configuration-dependent profile variance (0.0238 vs. 0.0070). The model also continues to underproduce subject heterogeneity in the central conditions: variances are 0.0036 vs. 0.0174 in Experiment 2, 0.0127 vs. 0.0462 in Experiment 6, 0.0148 vs. 0.0723 in Experiment 9, and 0.0062 vs. 0.0224 in Experiment 10. Thus the mean fit has improved substantially, but the candidate is not yet experiment-invariant.
Rationale: Continue from this accepted candidate, preserving history invariance, the tied-top order gate, the current exactly-two-opponent calibration, and the strong restoration branch for three or more opponents. The previous advice in this direction was accepted, and Experiment 9 and Experiment 10 should not be disturbed by another global challenge/restoration shift.

The next adjustment should isolate the one-opponent branch. Experiment 9 shows that exactly-two opposition is now correctly calibrated, whereas Experiment 6 still requires much lower anchor adherence with one opponent. Increase the one-opponent compact challenge amplitude—or make its credibility gate somewhat less suppressive—without changing the two-opponent branch. Keep restoration nearly deterministic for large coalitions across subject types. This should enlarge the negative Experiment 6 contrast and lower at least the compact one-opponent portion of Experiment 2 while protecting Experiments 1, 9, and the two-opponent compact component of Experiment 10.

Do not continue strengthening ordinary validity compression globally: although it helped Experiment 4, Experiment 3 has now overshot and Experiment 4 is already close enough. Likewise, reduce the generic embedded `minority_amplification` if condition-level checks show it drives Experiment 8's excessive profile variance, while retaining the separately localized broad-coalition `singleton_bonus` needed for the five-versus-one diagnostic in Experiment 10. This separates ordinary singleton effects from broad singleton dissent rather than moving both together.

Most importantly, calculate or target Experiment 2's constituent classes separately—isolated one-opponent, isolated two-opponent, restored three-opponent, broad singleton dissent, and ordinary balanced coalitions. Its pooled value has barely responded despite several accepted mechanism changes, so another pooled parameter shift is unlikely to work. Tune the existing class-specific gains based on which class remains excessively anchor-following; do not introduce a new mechanism family.

Finally, increase heterogeneity through mean-preserving stable mixtures rather than wider independent ranges. Use strongly separated one-opponent challenge types to raise variance in Experiments 2 and 6. For Experiment 9, introduce greater type-specific variation in the balance between exactly-two challenge and restoration while holding their mixture-weighted mean near 0.725. Keep large-count restoration relatively invariant across types to protect Experiment 1, and avoid type effects on the matched Experiment 7 configurations. This is a narrower extension of the accepted direction, not a reversal or a return to the previously rejected global threshold changes.

**Outcome of this advice:** iter 5 candidate loss=0.1677 -> iter 6 candidate loss=0.1123 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: Iteration 6 is the strongest accepted candidate so far and is faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The previous recommendation was productive: isolating the one-opponent challenge reduced Experiment 6's error substantially (-0.3928 versus -0.5450), and the new balanced-opposition channel finally moved Experiment 2 toward its low observed anchor adherence (0.4267 versus 0.3242). Experiment 8 has returned to a close fit, while Experiments 5, 7, and 9 remain good. However, the balanced-opposition adjustment appears too indiscriminate. Experiment 3 has flipped from the previously correct positive interaction to a negative one (-0.0617 versus 0.0721), indicating excessive challenge to the unique validity anchor specifically in balanced 2-versus-2 profiles. Experiment 4 also worsened from 0.08 to 0.20 spurious crossover prevalence, plausibly because the unconditional bonus makes balanced profiles overly sensitive to which coalition contains an early cue. Thus, the same change that repaired Experiment 2 damaged other balanced-coalition experiments. Experiment 2 also remains too anchor-following by 0.1025, and Experiment 6 still captures only about 72% of the observed negative contrast. Experiment 1 is somewhat weak at 0.92 versus unanimous observed prevalence, and Experiment 10 is modestly too anchor-following. Subject-level heterogeneity remains substantially compressed in Experiments 2, 6, 9, and 10 despite the stronger type separation.
Rationale: Build on this accepted candidate and retain the successful one-opponent branch, exactly-two calibration, large-count restoration, tied-top order gate, history invariance, and localized singleton treatment. The latest accepted direction should be refined rather than reversed.

The priority is to replace the unconditional `balanced_opposition_bonus` with a bounded validity-geometry gate. At present every non-isolated, unique-anchor balanced profile receives the same categorical opposition boost. Make that boost strong only when the opposing coalition contains individually credible cues and its reliability gap from the anchor is small, or when its redundancy-discounted support is already near the anchor coalition's support. Suppress it when the unique anchor is clearly stronger than every opponent. This should preserve the Experiment 2 gain while preventing the overchallenge that reversed Experiment 3. A bounded coalition-breadth interaction may also help distinguish Experiment 2's broader balanced profiles, but it should not be a raw count-only bonus. Do not tighten ordinary validity compression globally; prior results show that this risks disturbing Experiment 3 and is not necessary for the current diagnosis.

The same gated balanced channel should reduce Experiment 4's renewed crossover. In particular, cap how much alignment of one early non-anchor cue can change the gate, rather than changing the order mechanism. Experiment 5 demonstrates that tied-top order use is already calibrated correctly.

For Experiment 6, take a smaller further step in the accepted one-opponent direction: modestly increase the challenge-sensitive type's one-opponent gain or further relax its credibility suppression, while leaving exactly-two and three-or-more branches fixed. This should move -0.3928 toward -0.5450 and should also lower the isolated one-opponent component of Experiment 2 without disrupting the good Experiment 9 fit. Avoid another global restoration-threshold change, since that strategy was previously rejected.

Finally, increase heterogeneity through more separated but mean-preserving stable type mixtures. The simulated variances remain far below observed values in Experiment 6 (0.0156 versus 0.0462) and Experiment 9 (0.0155 versus 0.0723), with similar compression in Experiments 2 and 10. Give the stable types more divergent one- and two-opponent challenge/restoration balances and tune their mixture probabilities to preserve current means; keep large-count restoration comparatively invariant to protect Experiment 1. These are localized calibrations within PDRR, not a change of mechanism family.

**Outcome of this advice:** iter 6 candidate loss=0.1123 -> iter 7 candidate loss=0.1073 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: Iteration 7 is accepted and is the best aggregate fit so far, and it remains faithful to the prescribed history-invariant, configuration-dependent selective-integration family. The validity-geometry refinement successfully repaired damage from the unconditional balanced-opposition bonus: Experiment 3 returned near the observed positive interaction (0.0525 vs. 0.0721), Experiment 4 improved from 0.20 to 0.10 crossover prevalence, and Experiment 1 rose to 0.99. Experiments 5, 8, and 9 are also close in mean. However, this repair largely erased the previous gain on the most important unresolved condition: Experiment 2 regressed from 0.4267 to 0.5281 against an observed 0.3242. The strongest-opponent validity gate plus the reduced 2-vs-2 breadth factor is evidently suppressing distinctive opposition across too many of Experiment 2's balanced configurations. Experiment 6 remains materially too weak as well (-0.4072 vs. -0.5450), although the accepted one-opponent adjustment continues to move it in the correct direction. Experiment 10 is modestly too anchor-following (0.4775 vs. 0.4462), and Experiment 7 retains a small unwanted static contrast (-0.0294 vs. 0.0052). Most importantly, heterogeneity remains compressed in the central tests: variances are 0.0037 vs. 0.0174 in Experiment 2, 0.0215 vs. 0.0462 in Experiment 6, 0.0146 vs. 0.0723 in Experiment 9, and 0.0084 vs. 0.0224 in Experiment 10. Thus the model has a strong overall fit but has not yet achieved the prescribed cross-experiment dissociation between balanced opposition, compact challenge, and restoration.
Rationale: Build on this accepted candidate. The latest advice was accepted, so retain the validity-geometry gate rather than returning to the unconditional balanced bonus that damaged Experiments 3 and 4. The next adjustment should refine what counts as credible balanced opposition. The current gate uses only the strongest opponent and therefore misses distributed coalition credibility. Add a bounded interaction based on the number or proportion of opposing cues that are near the anchor in validity, or on redundancy-discounted opposition-to-anchor support parity. Let this distributed-credibility term become appreciable mainly in broader balanced coalitions, while keeping the present suppression for compact 2-vs-2 profiles with only one credible opponent. This should recover Experiment 2's lost reduction without recreating Experiment 3's sign reversal or Experiment 4's crossover. Do not simply enlarge `balanced_opposition_bonus` globally or remove its geometry gate.

Take one further small step along the accepted one-opponent direction: modestly increase the challenge-sensitive type's one-opponent gain or relax only that branch's credibility attenuation. Leave exactly-two and large-count restoration fixed. This should move Experiment 6 toward -0.545 and may improve the isolated one-opponent subset of Experiment 2 while preserving the good Experiment 1 and 9 means. Avoid a global restoration-threshold change, which was previously rejected.

Finally, make the stable types more behaviorally separated in a mean-preserving way. In particular, increase divergence in one-opponent challenge for Experiment 6 and in the exactly-two challenge/restoration balance for Experiment 9, compensating with type weights or opposing gain shifts so the current means remain near target. Correlating type-specific temperature or lapse with these configuration sensitivities may produce persistent subject differences more effectively than widening independent ranges. Keep large-count restoration relatively invariant across types and leave the tied-top order mechanism unchanged, since Experiments 1 and 5 are already essentially exact.

**Outcome of this advice:** iter 7 candidate loss=0.1073 -> iter 8 candidate loss=0.1238 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The current candidate remains faithful to the prescribed history-invariant PDRR mechanism family, but it was rejected by the accept gate and is worse than the iteration-7 base on several central tests. The new distributed-credibility term did not improve Experiment 2 at all: anchor following remains 0.5290 versus 0.3242. Meanwhile, the stronger type separation degraded Experiment 6 from -0.4072 to -0.3181, moved Experiment 9 farther above target from 0.7425 to 0.7675, and reduced Experiment 1 from 0.9900 to 0.9300. It also failed to improve the relevant heterogeneity consistently: Experiment 9 variance fell from 0.0146 to 0.0097 despite an observed 0.0723. The candidate still fits tied-top order in Experiment 5, profile variance in Experiment 8, and Experiment 10 reasonably well, while Experiment 3 remains acceptably positive and Experiment 7 remains only mildly non-null. The unresolved failures are therefore concentrated in Experiment 2's balanced subclasses and Experiment 6's one-opponent challenge, not in the overall mechanism family.
Rationale: Build on the unchanged iteration-7 accepted base. The most recent recommendation to add broad distributed credibility and increase type separation was rejected, so do not strengthen either adjustment. In particular, restore the iteration-7 degree of type separation and leave the exactly-two isolated restoration and large-count restoration branches fixed; those settings gave better means in Experiments 1, 6, and 9.

Use a different, more diagnostic calibration of Experiment 2. Its schedule contains distinct 1-vs-1, compact 2-vs-2, and broader balanced profiles, and the failed broad-coalition term could affect only part of that pooled metric. First obtain or internally target predictions for those subclasses. For isolated 1-vs-1 trials, add a modest type-independent challenge floor or baseline one-opponent gain, rather than increasing only the challenge-sensitive type. The last type-divergence edit perversely weakened the aggregate Experiment 6 contrast; a small challenge shared across types should lower one-opponent anchor adherence more reliably while leaving Experiment 9's exactly-two branch untouched.

For compact balanced 2-vs-2 trials, replace the strongest-opponent-only bonus with a bounded joint-credibility gate that requires both opposing cues to be near the anchor, or uses the weaker opponent's validity gap. This is distinct from the rejected broad distributed-mean term and may separate Experiment 2's credible pairs from Experiment 3 profiles where only one opponent is competitive. Keep the bonus suppressed when the second opponent is weak, protecting Experiment 3's positive interaction and limiting Experiment 4 crossover. Do not enlarge the balanced bonus globally.

Defer further heterogeneity expansion until these subclass means improve. The latest hard-coded gain separation increased Experiment 6 variance but damaged its mean and reduced Experiment 9 variance. Once the means are calibrated, mixture probabilities or temperature correlations can be adjusted in a mean-preserving manner. Preserve history invariance and the tied-top order gate, since Experiments 5 and 7 provide no reason to change them.

**Outcome of this advice:** iter 8 candidate loss=0.1238 -> iter 9 candidate loss=0.1204 -> the gate marked it REJECTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed PDRR family—communicated-validity weighting, redundancy discounting, configuration-specific challenge/restoration, bounded tied-top order, stable heterogeneity, and no exposure learning—but the current revision was rejected and does not improve on iteration 7. The added one-opponent floor moved Experiment 6 in the desired direction (-0.4503 versus -0.5450), but it did not reduce pooled anchor following in Experiment 2; that value instead rose to 0.5429 versus 0.3242. The joint-credibility edit likewise failed its intended dissociation: Experiment 3 moved farther below its positive target (0.0356 versus 0.0721), Experiment 4 crossover prevalence rose to 0.14 versus 0, and Experiment 2 did not improve. Other means remain broadly serviceable: Experiments 1, 5, 7, 8, 9, and 10 are reasonably close, although Experiment 8 slightly underproduces profile variance and Experiment 9 is somewhat over-restored. The dominant failure is still Experiment 2, followed by an insufficiently negative Experiment 6 contrast. Subject heterogeneity also remains markedly compressed in Experiments 2, 6, 9, and 10. Thus the theory family remains plausible, but this parameterization is not experiment-invariant.
Rationale: Build the next revision on the unchanged iteration-7 accepted base. The two most recent proposals—broad distributed credibility and then a shared one-opponent floor plus compact joint-credibility bonus—were both rejected, so do not repeat or intensify them. In particular, remove the new joint-credibility modification and do not globally increase one-opponent challenge.

Use a different in-family discriminator that the current model discards: absolute anchor credibility. Cue weights are normalized to the strongest active cue, making every locally strongest cue have weight 1 regardless of whether it is a genuinely high-validity expert or merely the best among low-validity active experts. Experiment 2 contains several profiles whose first discriminating cue occurs later in the validity ordering; these lower-absolute-validity anchors should be more challengeable than the strong anchors in Experiments 6 and 9. Add a bounded absolute-validity gate—based on the anchor's communicated log-odds or validity, not just its gap from opponents—to amplify existing isolated challenge and embedded singleton distinctiveness when the active anchor is weak. Keep this gate near neutral for highly credible anchors and do not apply it as a general balanced-coalition bonus, which would risk Experiments 3 and 4. Internally score Experiment 2 by active-anchor rank and by isolated versus embedded configuration so this adjustment is calibrated to the subclasses actually responsible for its high pooled adherence.

Address the remaining Experiment 6 error from the other side of the contrast: sharpen only the count-dominant restoration branch for three or more opponents, making it more nearly type-invariant and closer to its ceiling. Leave one- and exactly-two-opponent branches unchanged. This should increase multi-opponent anchor adherence and make the contrast more negative without disturbing Experiment 9's two-opponent mean or reviving the failed global challenge shifts; it should also preserve Experiment 1's near-unanimous anchor-following pattern.

Preserve iteration 7's ordinary-coalition compression, validity-geometry gate, tied-top order settings, and history invariance. Defer further expansion of hard-coded type separation: the previous attempt was rejected and did not reliably increase variance. Once the Experiment 2 subclass means and Experiment 6 contrast improve, heterogeneity can be tuned through mean-preserving mixture probabilities or correlated temperature/lapse effects rather than larger configuration gains.

**Outcome of this advice:** iter 9 candidate loss=0.1204 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Partitioned Dissent and Redundancy Restoration (PDRR) proposes that people construct a stable, instruction-based evidence representation and classify each choice configuration before integrating its cues. Communicated validities determine both relative cue strength and absolute anchor credibility, while cues supporting the same option accumulate sublinearly because concordant reports are treated as partly redundant. In ordinary embedded coalitions, individual cue weights are strongly compressed before accumulation, limiting the influence of any single non-anchor cue while preserving communicated-validity ordering. Opposition to a validity anchor is statically distinctive even when the anchor has allies, but balanced opposition receives a counter-anchor bonus only when the opposing cues are individually close in communicated validity to the anchor and the coalition is broad enough to support a coalition-level interpretation. Isolated-anchor opposition is processed separately. A single opponent receives especially strong challenge status, with stable subject types differing substantially in their sensitivity to this challenge, while exactly two opponents retain a joint-credibility mechanism with greater type-specific heterogeneity. These challenge and singleton-dissent mechanisms are additionally amplified when the locally strongest active cue has low absolute communicated validity, because being best among weak active cues does not create a compelling anchor. Coalitions of three or more trigger a sharp, nearly type-invariant redundancy-restoration branch. Embedded singleton dissent is strongly amplified only against very broad coalitions or weak absolute anchors; its generic effect against smaller coalitions is attenuated to avoid excessive profile variance. Presentation order affects choice primarily as a bounded tie breaker among currently discriminating cues with equal maximal communicated validity, avoiding a general primacy crossover. All mechanisms are history-invariant in this feedback-free task. Stable subject types and random effects over validity sensitivity, balanced dissent, compact challenge, redundancy, temperature, lapse, and absolute-anchor sensitivity generate persistent individual differences.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"PDRR expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    # Positive directions favor B and negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    redundancy_exponent = float(parameters[\"redundancy_exponent\"])\n    redundancy_saturation = float(parameters[\"redundancy_saturation\"])\n    ordinary_validity_compression = float(parameters[\"ordinary_validity_compression\"])\n    minority_amplification = float(parameters[\"minority_amplification\"])\n    balanced_opposition_bonus = float(parameters[\"balanced_opposition_bonus\"])\n    singleton_bonus = float(parameters[\"singleton_bonus\"])\n    compact_challenge = float(parameters[\"compact_challenge\"])\n    compact_width = float(parameters[\"compact_width\"])\n    credibility_gap_scale = float(parameters[\"credibility_gap_scale\"])\n    absolute_anchor_threshold = float(parameters[\"absolute_anchor_threshold\"])\n    absolute_anchor_slope = float(parameters[\"absolute_anchor_slope\"])\n    weak_anchor_gain = float(parameters[\"weak_anchor_gain\"])\n    restoration_threshold = float(parameters[\"restoration_threshold\"])\n    restoration_slope = float(parameters[\"restoration_slope\"])\n    restoration_gap_threshold = float(parameters[\"restoration_gap_threshold\"])\n    restoration_gap_slope = float(parameters[\"restoration_gap_slope\"])\n    restoration_credibility_threshold = float(parameters[\"restoration_credibility_threshold\"])\n    restoration_credibility_slope = float(parameters[\"restoration_credibility_slope\"])\n    restoration_ceiling = float(parameters[\"restoration_ceiling\"])\n    tie_order_gate = float(parameters[\"tie_order_gate\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n    configuration_type = int(parameters[\"configuration_type\"])\n\n    # Correlated, stable subject types alter configuration sensitivity without\n    # introducing trial-wise strategy switching or exposure-based learning.\n    if configuration_type == 0:       # compact-challenge-sensitive\n        challenge_type_gain = 2.40\n        one_challenge_type_gain = 4.20\n        singleton_type_gain = 1.50\n        restoration_type_gain = 0.95\n    elif configuration_type == 1:     # count-restoring\n        challenge_type_gain = 0.50\n        one_challenge_type_gain = 0.75\n        singleton_type_gain = 0.90\n        restoration_type_gain = 1.25\n    else:                              # weakly configuration-sensitive\n        challenge_type_gain = 0.15\n        one_challenge_type_gain = 0.15\n        singleton_type_gain = 0.55\n        restoration_type_gain = 1.00\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # Translate communicated accuracy into reliability evidence, retaining\n    # absolute ordering but using the strongest active cue as a scale origin.\n    v = np.clip(validities, 0.500001, 0.999999)\n    reliability = np.log(v / (1.0 - v))\n    active_reference = float(np.max(reliability[active]))\n    cue_weight = np.exp(\n        validity_sensitivity * (reliability - active_reference)\n    )\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n\n    def coalition_support(indices, compress=False):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n        weights = cue_weight[indices]\n        if compress:\n            # Bound cue-composition effects in ordinary embedded coalitions.\n            # The power transform preserves validity ordering but prevents a\n            # single non-anchor cue from creating a broad primacy crossover.\n            weights = np.power(\n                np.clip(weights, 1e-12, None), ordinary_validity_compression\n            )\n        raw = float(np.sum(weights))\n        # The power term gives ordinary sublinear accumulation. The bounded\n        # saturation term makes additional concordant reports increasingly\n        # interpretable as redundant without erasing coalition size entirely.\n        discount = (float(n) ** redundancy_exponent) * (\n            1.0 + redundancy_saturation * float(max(n - 1, 0))\n            / float(n + 1)\n        )\n        return raw / max(discount, 1e-12)\n\n    support_a = coalition_support(a_cues)\n    support_b = coalition_support(b_cues)\n\n    # Highest communicated validity among discriminating cues defines the\n    # anchor. Stable display order is used only to resolve a top-validity tie.\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]\n    anchor_cue = int(np.min(best_candidates))\n    anchor_direction = float(directions[anchor_cue])\n    unique_anchor = float(best_candidates.size == 1)\n\n    # Relative weights alone make every locally strongest cue equivalent.\n    # This bounded gate preserves that representation while marking anchors\n    # that are weak in absolute communicated-validity terms as challengeable.\n    weak_absolute_anchor = sigmoid(\n        absolute_anchor_slope * (absolute_anchor_threshold - best_v)\n    )\n    weak_anchor_multiplier = 1.0 + weak_anchor_gain * weak_absolute_anchor\n\n    if active.size > 1:\n        sorted_active_v = np.sort(active_v)[::-1]\n        validity_gap = float(sorted_active_v[0] - sorted_active_v[1])\n    else:\n        validity_gap = 0.5\n\n    if anchor_direction > 0:\n        anchor_side_count = n_b\n        opposition_count = n_a\n        anchor_support = support_b\n        opposition_support = support_a\n        opposition_cues = a_cues\n    else:\n        anchor_side_count = n_a\n        opposition_count = n_b\n        anchor_support = support_a\n        opposition_support = support_b\n        opposition_cues = b_cues\n\n    isolated_anchor = anchor_side_count == 1 and opposition_count > 0\n\n    # Credibility is based on the strongest one or two individual opponents.\n    # For two-opponent challenge, the weaker member must also be credible, so\n    # one strong report cannot make a heterogeneous pair diagnostic.\n    if opposition_count > 0:\n        opponent_validities = np.sort(validities[opposition_cues])[::-1]\n        credibility_set = opponent_validities[:min(2, opposition_count)]\n        opponent_credibility = float(np.mean(credibility_set))\n        credibility_gap = max(best_v - opponent_credibility, 0.0)\n        joint_credibility_gap = max(\n            best_v - float(np.min(credibility_set)), 0.0\n        )\n    else:\n        credibility_gap = 0.5\n        joint_credibility_gap = 0.5\n\n    if isolated_anchor:\n        # Challenge is sharply asymmetric across counts. The one-opponent\n        # branch has a less suppressive credibility gate and substantially\n        # separated stable subject types. Weak absolute anchors amplify the\n        # existing challenge rather than creating a general coalition bonus.\n        if opposition_count == 1:\n            compact_profile = 3.00\n            challenge_gap = 0.35 * credibility_gap\n            active_challenge_type_gain = one_challenge_type_gain\n        elif opposition_count == 2:\n            compact_profile = 1.35\n            challenge_gap = joint_credibility_gap\n            active_challenge_type_gain = challenge_type_gain\n        else:\n            distance = (float(opposition_count) - 1.5) / max(compact_width, 1e-6)\n            compact_profile = np.exp(-(distance ** 4))\n            challenge_gap = joint_credibility_gap\n            active_challenge_type_gain = challenge_type_gain\n        credibility_gate = np.exp(\n            -challenge_gap / max(credibility_gap_scale, 1e-6)\n        )\n        challenged_opposition = opposition_support * (\n            1.0\n            + active_challenge_type_gain\n            * compact_challenge\n            * compact_profile\n            * credibility_gate\n            * weak_anchor_multiplier\n        )\n        margin = (anchor_support - challenged_opposition) / max(\n            anchor_support + challenged_opposition, 1e-12\n        )\n        core_evidence = anchor_direction * margin\n    else:\n        # Compress cue-composition effects only for unique-anchor ordinary\n        # coalitions. Tied-top profiles retain the accepted integrator so the\n        # calibrated bounded order effect is not disturbed.\n        if unique_anchor > 0.0:\n            support_a = coalition_support(a_cues, compress=True)\n            support_b = coalition_support(b_cues, compress=True)\n            if anchor_direction > 0:\n                anchor_support = support_b\n                opposition_support = support_a\n            else:\n                anchor_support = support_a\n                opposition_support = support_b\n\n        # Balanced opposition is distinctive only when the strongest opposing\n        # cue is close to the anchor in communicated validity. A bounded breadth\n        # factor further prevents a compact 2-vs-2 profile from receiving the\n        # same categorical boost as a broader coalition-level conflict.\n        if (\n            unique_anchor > 0.0\n            and anchor_side_count == opposition_count\n            and anchor_side_count > 1\n        ):\n            strongest_opponent_v = float(np.max(validities[opposition_cues]))\n            balanced_validity_gap = max(best_v - strongest_opponent_v, 0.0)\n            geometry_gate = np.exp(\n                -balanced_validity_gap / max(credibility_gap_scale, 1e-6)\n            )\n            breadth_gate = min(\n                1.0, float(anchor_side_count - 1) / 2.0\n            )\n            balanced_gain = (\n                balanced_opposition_bonus * geometry_gate * breadth_gate\n            )\n            opposition_support *= 1.0 + balanced_gain\n            if anchor_direction > 0:\n                support_a = opposition_support\n            else:\n                support_b = opposition_support\n\n        # Embedded distinctiveness is sharply singleton-localized. Its broad-\n        # coalition component remains strong, while the generic component is\n        # attenuated for smaller coalitions to reduce excessive profile variance.\n        # Absolute weakness amplifies this existing channel without altering\n        # balanced or other ordinary embedded configurations.\n        if opposition_count == 1 and anchor_side_count > 1:\n            imbalance = (\n                float(anchor_side_count - opposition_count)\n                / float(anchor_side_count + opposition_count)\n            )\n            dissent_multiplier = 1.0 + (\n                minority_amplification * imbalance * weak_anchor_multiplier\n            )\n            breadth_weight = 1.0 if anchor_side_count >= 5 else 0.20\n            dissent_multiplier += (\n                singleton_type_gain\n                * singleton_bonus\n                * breadth_weight\n                * np.log1p(float(anchor_side_count - 1))\n                * weak_anchor_multiplier\n            )\n            # A very broad concordant coalition makes its sole dissenter\n            # especially distinctive, while leaving three-versus-one control\n            # profiles close to the accepted static accumulator.\n            if anchor_side_count >= 5:\n                dissent_multiplier += (\n                    singleton_type_gain\n                    * singleton_bonus\n                    * 0.45\n                    * float(anchor_side_count - 4)\n                    * weak_anchor_multiplier\n                )\n            opposition_support *= dissent_multiplier\n\n            if anchor_direction > 0:\n                support_a = opposition_support\n            else:\n                support_b = opposition_support\n\n        total_support = support_a + support_b\n        core_evidence = (support_b - support_a) / max(total_support, 1e-12)\n\n    # Gap-sensitive restoration remains available, but exactly two opponents\n    # receive little automatic restoration. A separate steep count branch is\n    # almost off at two and effectively reaches the common gate ceiling at\n    # three or more, without requiring the gap gates or subject type to agree.\n    restoration_gate = 0.0\n    if isolated_anchor and unique_anchor > 0.0:\n        count_gate = sigmoid(\n            restoration_slope\n            * (float(opposition_count) - restoration_threshold)\n        )\n        gap_gate = sigmoid(\n            restoration_gap_slope\n            * (validity_gap - restoration_gap_threshold)\n        )\n        credibility_restoration = sigmoid(\n            restoration_credibility_slope\n            * (credibility_gap - restoration_credibility_threshold)\n        )\n        selective_restoration = (\n            restoration_ceiling\n            * count_gate\n            * gap_gate\n            * credibility_restoration\n        )\n\n        if opposition_count == 2:\n            selective_restoration *= 0.30\n\n        large_count_gate = sigmoid(\n            restoration_slope\n            * (float(opposition_count) - 2.50)\n        )\n        count_dominant_restoration = 0.99 * large_count_gate\n        restoration_gate = max(\n            selective_restoration,\n            count_dominant_restoration,\n        )\n\n    # Equal top validities remain exchangeable in integration. Order only\n    # supplies a bounded tie anchor rather than a general first-cue bonus.\n    tied_top = float(best_candidates.size > 1)\n    order_gate = tie_order_gate * tied_top\n    gate = 1.0 - (1.0 - restoration_gate) * (1.0 - order_gate)\n    gate = float(np.clip(gate, 0.0, 0.95))\n\n    choice_evidence = (\n        (1.0 - gate) * core_evidence\n        + gate * anchor_direction\n    )\n\n    logits = beta * np.array(\n        [-0.5 * choice_evidence, 0.5 * choice_evidence],\n        dtype=np.float64,\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array(\n        [0.5, 0.5], dtype=np.float64\n    )\n    probs = np.clip(probs, 0.0, None)\n    probs /= probs.sum()\n    return probs.astype(np.float64)\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "validity_sensitivity": "[0.15, 0.85]",
    "redundancy_exponent": "[0.28, 0.68]",
    "redundancy_saturation": "[0.15, 0.65]",
    "ordinary_validity_compression": "[0.03, 0.18]",
    "minority_amplification": "[0.20, 0.70]",
    "balanced_opposition_bonus": "[0.55, 1.25]",
    "singleton_bonus": "[1.8, 4.5]",
    "compact_challenge": "[2.4, 6.0]",
    "compact_width": "[0.75, 1.15]",
    "credibility_gap_scale": "[0.018, 0.055]",
    "absolute_anchor_threshold": "[0.68, 0.78]",
    "absolute_anchor_slope": "[20.0, 45.0]",
    "weak_anchor_gain": "[1.2, 3.0]",
    "restoration_threshold": "[1.65, 2.15]",
    "restoration_slope": "[9.0, 18.0]",
    "restoration_gap_threshold": "[0.035, 0.085]",
    "restoration_gap_slope": "[70.0, 150.0]",
    "restoration_credibility_threshold": "[0.025, 0.065]",
    "restoration_credibility_slope": "[70.0, 150.0]",
    "restoration_ceiling": "[0.82, 0.98]",
    "tie_order_gate": "[0.48, 0.78]",
    "beta": "[1.2, 4.2]",
    "lapse": "[0.0, 0.14]",
    "configuration_type": "{0, 1, 2}"
  },
  "rationale": "This is a localized edit of the accepted iteration-7 model. It retains the accepted validity-geometry gate and does not repeat either rejected balanced-coalition modification. The principal addition is a bounded absolute-anchor-credibility gate. Because the existing relative normalization assigns weight 1 to whichever active cue is locally strongest, it previously treated a late, low-validity anchor like a genuinely high-validity anchor. The new sigmoid uses the anchor's communicated validity directly and amplifies only the already existing isolated-challenge and embedded-singleton channels. It therefore targets low-rank active anchors in Experiment 2 without creating another general balanced-opposition bonus that could reverse Experiment 3 or revive Experiment 4's crossover. High-validity anchors receive a multiplier close to one, protecting Experiments 6 and 9. The only other substantive change makes the accepted count-dominant restoration branch nearly type-invariant and effectively ceiling-level for three or more opponents while remaining almost inactive at two. This addresses the residual Experiment 6 contrast from its multi-opponent side and leaves Experiment 9's exactly-two calibration essentially unchanged. History invariance, ordinary-coalition compression, tied-top order use, exactly-two selective restoration, temperature, lapse, and all other equations and parameter ranges are preserved."
}
```

## Usage

```json
{
  "input_tokens": 36080,
  "output_tokens": 6210
}
```
