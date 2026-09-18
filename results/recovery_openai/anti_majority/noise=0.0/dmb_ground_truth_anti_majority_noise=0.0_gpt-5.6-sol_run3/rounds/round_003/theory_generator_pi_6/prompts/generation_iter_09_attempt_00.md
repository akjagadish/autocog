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
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 with a Static Profile-Gated Evidence Integration theory. Choices should combine communicated cue validity with the current directional configuration of evidence: independently supporting cues accumulate support, concordant cues receive a redundancy discount, and coalition size or balance can allow several lower-validity cues to overcome a unique high-validity cue. A bounded anchor gate should preserve strong reliance on the highest-validity cue when opposition is weak; stable presentation order should matter mainly for ties among top-validity cues, accounting for Experiment 7 without generating the primacy crossover rejected in Experiment 6. Crucially, cue weights should not be updated from mere exposure to concurrence, anti-concurrence, or discrimination frequency when there is no correctness feedback. Cues with equal communicated validity should therefore be exchangeable apart from narrowly specified tie-breaking, making the diagnostic and control profiles in Experiment 1 predict the same anchor adherence despite their different histories. The theory should predict static profile heterogeneity in Experiment 2, balanced-versus-imbalanced coalition effects in Experiment 5, substantial coalition-driven reversals in Experiments 4 and 8, and above-chance top-cue use in Experiment 3. Include stable subject-level variation in validity sensitivity, coalition accumulation, redundancy discount, anchor-gate strength, tie-order bias, and response temperature/lapse. This creates a substantially simpler and more falsifiable competitor to pi_5: it retains the coalition mechanisms supported across tasks while explicitly rejecting pi_5's unsupported history-dependent rarity and anti-concurrence learning.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.449)

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


---

### `pi_5` (overall score: 0.431)

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


---

### `pi_2` (overall score: 0.013)

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
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.4848 (var=0.0022)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.1762 (var=0.0033)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0600 (var=0.0564)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.1456 (var=0.0074)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=0.3781 (var=0.0114)
- Experiment 7: real=0.0052 (var=0.0075) vs this=-0.0056 (var=0.0083)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0206 (var=0.0003)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2510 -> ACCEPTED
- iter 2: loss=0.1940 -> ACCEPTED
- iter 3: loss=0.1797 -> ACCEPTED
- iter 4: loss=0.1682 -> ACCEPTED
- iter 5: loss=0.2266 -> REJECTED
- iter 6: loss=0.2569 -> REJECTED
- iter 7: loss=0.1857 -> REJECTED
- iter 8: loss=0.1607 -> ACCEPTED
- iter 9: loss=0.1599 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 9 at loss=0.1599 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.9200 (var=0.0636) (Δ vs real -0.0800)
  - iter 4: 0.8900 (var=0.0829) (Δ vs real -0.1100)
  - iter 5: 0.8900 (var=0.0929) (Δ vs real -0.1100)
  - iter 6: 0.3200 (var=0.2076) (Δ vs real -0.6800)
  - iter 7: 0.9400 (var=0.0564) (Δ vs real -0.0600)
  - iter 8: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 9 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)

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
  - iter 1: 0.6525 (var=0.0054) (Δ vs real +0.3283)
  - iter 2: 0.7000 (var=0.0051) (Δ vs real +0.3758)
  - iter 3: 0.4485 (var=0.0024) (Δ vs real +0.1244)
  - iter 4: 0.4625 (var=0.0036) (Δ vs real +0.1383)
  - iter 5: 0.4529 (var=0.0022) (Δ vs real +0.1288)
  - iter 6: 0.4442 (var=0.0033) (Δ vs real +0.1200)
  - iter 7: 0.4498 (var=0.0026) (Δ vs real +0.1256)
  - iter 8: 0.4462 (var=0.0035) (Δ vs real +0.1221)
  - iter 9 (most recent): 0.4665 (var=0.0022) (Δ vs real +0.1423)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)

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
  - iter 1: -0.0219 (var=0.0024) (Δ vs real -0.0940)
  - iter 2: 0.1098 (var=0.0034) (Δ vs real +0.0377)
  - iter 3: 0.0073 (var=0.0033) (Δ vs real -0.0648)
  - iter 4: -0.0623 (var=0.0027) (Δ vs real -0.1344)
  - iter 5: -0.0858 (var=0.0028) (Δ vs real -0.1579)
  - iter 6: -0.0690 (var=0.0036) (Δ vs real -0.1410)
  - iter 7: -0.0777 (var=0.0015) (Δ vs real -0.1498)
  - iter 8: -0.0610 (var=0.0032) (Δ vs real -0.1331)
  - iter 9 (most recent): -0.0587 (var=0.0032) (Δ vs real -0.1308)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)

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
  - iter 1: 0.4400 (var=0.2464) (Δ vs real +0.4400)
  - iter 2: 0.1000 (var=0.0900) (Δ vs real +0.1000)
  - iter 3: 0.2600 (var=0.1924) (Δ vs real +0.2600)
  - iter 4: 0.1200 (var=0.1056) (Δ vs real +0.1200)
  - iter 5: 0.3000 (var=0.2100) (Δ vs real +0.3000)
  - iter 6: 0.2000 (var=0.1600) (Δ vs real +0.2000)
  - iter 7: 0.2200 (var=0.1716) (Δ vs real +0.2200)
  - iter 8: 0.2600 (var=0.1924) (Δ vs real +0.2600)
  - iter 9 (most recent): 0.2200 (var=0.1716) (Δ vs real +0.2200)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)

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
  - iter 1: 0.7652 (var=0.0062) (Δ vs real -0.0248)
  - iter 2: 0.7227 (var=0.0069) (Δ vs real -0.0673)
  - iter 3: 0.7623 (var=0.0067) (Δ vs real -0.0277)
  - iter 4: 0.7758 (var=0.0074) (Δ vs real -0.0142)
  - iter 5: 0.7406 (var=0.0075) (Δ vs real -0.0494)
  - iter 6: 0.7521 (var=0.0089) (Δ vs real -0.0379)
  - iter 7: 0.7765 (var=0.0085) (Δ vs real -0.0135)
  - iter 8: 0.7785 (var=0.0073) (Δ vs real -0.0115)
  - iter 9 (most recent): 0.7546 (var=0.0090) (Δ vs real -0.0354)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)

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
  - iter 1: 0.0216 (var=0.0116) (Δ vs real +0.5666)
  - iter 2: -0.2597 (var=0.0125) (Δ vs real +0.2853)
  - iter 3: -0.2306 (var=0.0255) (Δ vs real +0.3144)
  - iter 4: -0.1891 (var=0.0301) (Δ vs real +0.3559)
  - iter 5: -0.0666 (var=0.0093) (Δ vs real +0.4784)
  - iter 6: -0.0703 (var=0.0105) (Δ vs real +0.4747)
  - iter 7: -0.2353 (var=0.0277) (Δ vs real +0.3097)
  - iter 8: -0.2372 (var=0.0344) (Δ vs real +0.3078)
  - iter 9 (most recent): -0.1462 (var=0.0161) (Δ vs real +0.3988)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)

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
  - iter 1: -0.0124 (var=0.0163) (Δ vs real -0.0176)
  - iter 2: 0.0043 (var=0.0211) (Δ vs real -0.0008)
  - iter 3: 0.0501 (var=0.0150) (Δ vs real +0.0449)
  - iter 4: 0.0240 (var=0.0134) (Δ vs real +0.0188)
  - iter 5: 0.0425 (var=0.0119) (Δ vs real +0.0373)
  - iter 6: 0.0165 (var=0.0157) (Δ vs real +0.0113)
  - iter 7: 0.0515 (var=0.0129) (Δ vs real +0.0463)
  - iter 8: -0.0093 (var=0.0150) (Δ vs real -0.0144)
  - iter 9 (most recent): -0.0016 (var=0.0127) (Δ vs real -0.0068)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)

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
  - iter 1: 0.0007 (var=0.0001) (Δ vs real -0.0063)
  - iter 2: 0.0216 (var=0.0002) (Δ vs real +0.0146)
  - iter 3: 0.0017 (var=0.0001) (Δ vs real -0.0053)
  - iter 4: 0.0025 (var=0.0001) (Δ vs real -0.0045)
  - iter 5: 0.0028 (var=0.0001) (Δ vs real -0.0042)
  - iter 6: 0.0028 (var=0.0001) (Δ vs real -0.0042)
  - iter 7: 0.0035 (var=0.0001) (Δ vs real -0.0035)
  - iter 8: 0.0020 (var=0.0001) (Δ vs real -0.0050)
  - iter 9 (most recent): 0.0034 (var=0.0001) (Δ vs real -0.0036)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Static Configuration-Contingent Integration (SCCI) claims that people form a fixed, instruction-based representation of cue reliability and apply it to the directional evidence profile on each trial. Each discriminating cue contributes reliability-weighted evidence, so independently supporting cues accumulate, while concordant cues are compressed as potentially redundant. This compression increases moderately convexly with coalition size, reflecting the declining informational value of each additional concordant cue. A bounded configuration gate supplements integration: strong protection of a uniquely most-valid cue is restricted to profiles in which that cue is isolated, has a sufficiently large communicated-validity advantage, and faces multiple concordant opponents. The transition to restoration is sharp, allowing an isolated anchor to be weak against one credible challenger but strongly protected against a redundant multi-cue bloc. Compact challenge amplification is strictly limited to isolated anchors and therefore cannot create general primacy effects when the anchor has allies. When the most-valid discriminating cues have equal communicated validity, they remain exchangeable in the integrator, but stable display order supplies a bounded tie-breaking anchor. Choices are history-invariant because concurrence, anti-concurrence, and discrimination frequency are not learned without correctness feedback. Stable profile heterogeneity is concentrated in a correlated configuration propensity: people with more readily restored isolated anchors also challenge those anchors more strongly in lone conflicts and rely less on validity in ordinary integration. Ordinary coalition parameters vary only narrowly, limiting unsupported ally-driven crossovers.

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

    # A low raw-gap threshold identifies a stable configuration-sensitive
    # subject. Correlating existing parameters concentrates heterogeneity in
    # the isolated-anchor contrast rather than ordinary coalition behavior.
    restoration_propensity = np.clip(
        (0.060 - restoration_gap_threshold) / 0.025, 0.0, 1.0
    )
    challenge_strength = float(np.clip(
        challenge_strength * (0.8 + 0.4 * restoration_propensity),
        0.8,
        2.8,
    ))
    validity_sensitivity = float(np.clip(
        validity_sensitivity * (1.2 - 0.4 * restoration_propensity),
        0.03,
        0.35,
    ))

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

    # Determine whether the active validity maximum is genuinely unique.
    if active.size > 1:
        sorted_active = np.sort(active_v)[::-1]
        validity_gap = float(sorted_active[0] - sorted_active[1])
    else:
        validity_gap = 0.5
    unique_anchor = 1.0 - np.exp(-40.0 * max(validity_gap, 0.0))

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
- redundancy_discount: [0.22, 0.42]
- redundancy_curvature: [1.15, 1.45]
- profile_balance_gain: [0.0, 0.0]
- anchor_gate_strength: [0.90, 0.995]
- challenge_strength: [0.8, 2.8]
- restoration_strength: [0.95, 1.0]
- restoration_threshold: [1.15, 1.55]
- restoration_slope: [16.0, 30.0]
- restoration_gap_threshold: [0.035, 0.060]
- restoration_gap_slope: [110.0, 190.0]
- tie_order_bias: [0.58, 0.85]
- beta: [0.9, 3.4]
- lapse: [0.0, 0.12]

`rationale`: This is a narrow calibration edit of the accepted iteration-8 SCCI model. The convex redundancy equation, challenge and restoration equations, tie-only order gate, response rule, and complete history invariance are retained. Ordinary coalition accumulation, redundancy discount, and curvature ranges are narrowed so subject variation in generic two-coalition integration is less likely to generate the unsupported Expert-1 ally crossover in Experiment 4. Because (n-1)^rho equals one for a two-cue coalition, curvature remains chiefly an incremental discount beyond the first two concordant cues. The existing raw validity-gap criterion is narrowed and sharpened, rather than replaced by the rejected transformed-gap gate, so modest-gap profiles should receive little restoration while stronger isolated anchors can still be restored reliably. Finally, the raw gap threshold supplies a stable correlated propensity using only existing parameters: subjects with easier isolated-anchor restoration receive stronger bounded lone-opponent challenge and weaker ordinary validity sensitivity. This should enlarge and diversify the Experiment 6 lone-versus-multiple contrast while reducing broad top-cue adherence in Experiment 2, without widening the marginal challenge range or adding history learning, response-temperature modulation, or another special-purpose gate.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: SCCI is faithful to the prescribed static, history-invariant mechanism family and succeeds on several important diagnostics: Experiment 1 is exact, Experiment 5 closely captures tied-top order adherence, Experiment 7 correctly predicts essentially no history-induced diagnostic/control difference, and Experiment 8 has the correct small profile-heterogeneity scale. Nevertheless, the aggregate fit is not adequate. Experiment 2 substantially overpredicts highest-validity-cue adherence (0.653 versus 0.324), indicating excessive general anchor protection or validity sensitivity. Experiment 3 gets the coalition-balance interaction in the wrong direction (-0.022 versus +0.072). Experiment 4 generates a large primacy crossover prevalence that is absent in humans (0.44 versus 0.00), despite order supposedly mattering only for tied top cues. Most seriously, Experiment 6 misses the strong increase in top-cue adherence under multi-opponent coalitions: the observed lone-minus-multi contrast is -0.545, whereas the model predicts approximately zero (+0.022). Thus the proposed U-shaped anchor gate is not producing its intended configuration dependence.
Rationale: Retain the static profile-gated family, but recalibrate its configuration terms. The symmetric minority-distinctiveness multiplier is likely a central error: in a 3-versus-1 profile it selectively boosts the minority, which can explain the reversed Experiment 3 interaction and may also create the spurious Experiment 4 crossover when the identity of the minority changes. Remove it or shrink profile_balance_gain close to zero. At the same time, lower broad anchor protection—especially the baseline and weak-opposition components, and possibly validity_sensitivity—to reduce the excessive top-cue adherence in Experiment 2. Do not simply lower the whole gate, however: Experiment 6 requires a much sharper count/configuration-contingent redundancy shield. Make anchor restoration rise steeply once opposition contains multiple concordant cues, rather than through the current weak product of isolation and redundancy_shield; keep this term bounded and independent of cue order. This separation should permit low anchor adherence against one credible opponent but high adherence against a multi-cue bloc treated as redundant. Finally, eliminate any effective dependence in Experiment 4 on whether early Expert 1 happens to join Expert 5, except where justified by ordinary validity-weighted support; reducing minority gain and validity-weight dispersion should help suppress that crossover. Wider subject-level variation can then be considered to match Experiment 2's underestimated variance, but only after correcting the mean patterns.

**Outcome of this advice:** iter 1 candidate loss=0.2510 -> iter 2 candidate loss=0.1940 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision is a meaningful improvement and remains faithful to the prescribed static, history-invariant profile-gated family. The prior recommendation helped: Experiment 3 now has the correct positive interaction (0.110 versus 0.072), Experiment 4’s false crossover prevalence fell from 0.44 to 0.10, Experiment 6 moved substantially in the required negative direction, and Experiment 7 is essentially exact. Experiment 1 also remains exact. However, the model is not yet experiment-invariant. Experiment 2 is the largest failure: highest-validity-cue adherence increased to 0.700 despite an observed value of 0.324, and its between-subject variance remains much too low (0.005 versus 0.017). Experiment 6 still captures only about half of the strong lone-minus-multi contrast (-0.260 versus -0.545), with too little heterogeneity. Experiment 4 retains a nonzero primacy crossover absent from the data, while tied-top order adherence in Experiment 5 has fallen to 0.723 versus 0.790. Experiment 8 also now overstates profile heterogeneity, though this discrepancy is comparatively small.
Rationale: Keep the static profile-gated mechanism, but make the restoration gate more configuration-specific rather than increasing its general strength. The current opposition_restoration depends almost entirely on opposition count, so it strongly protects a unique top cue whenever two or more cues oppose it. That likely explains why Experiment 2 became worse even after validity sensitivity was reduced. The previous reduction in validity sensitivity therefore should not simply be repeated; it did not fix Experiment 2. Instead, condition strong multi-opponent restoration on an isolated anchor—e.g., anchor_side_count == 1, a sufficiently large communicated-validity gap, and a compact concordant opposing bloc. Suppress restoration when the anchor has same-side allies or the validity maximum is only weakly unique. This should lower broad top-cue adherence in Experiment 2 and further eliminate Experiment 4’s crossover while preserving the special Experiment 6 pattern. Within that isolated-anchor regime, sharpen the one-versus-multiple separation: permit a stronger lone-challenge term and near-complete but bounded restoration for multi-cue opposition, which is needed to move Experiment 6 from -0.260 toward -0.545. Give lone-challenge strength and restoration threshold/slope separate subject-level parameters rather than tying the challenge boost to coalition_accumulation; widen their ranges to address the underestimated variances in Experiments 2 and 6. Finally, modestly raise the tie-specific order gate or its lower parameter bound to recover Experiment 5, without changing order effects for uniquely valid cues. Preserve the no-history property and the successful near-zero Experiment 7 contrast.

**Outcome of this advice:** iter 2 candidate loss=0.1940 -> iter 3 candidate loss=0.1797 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate is an accepted aggregate improvement and remains faithful to the prescribed static, history-invariant profile-gated family. The previous configuration-specific revision clearly helped Experiment 2, reducing excessive highest-validity-cue adherence from 0.700 to 0.449, and Experiment 5 returned close to its observed tied-order effect (0.762 versus 0.790). Experiment 7 remains near zero, and Experiment 8 is close in both mean and variance. Nevertheless, the model is not yet experiment-invariant. Experiment 6 remains the largest failure: the predicted lone-minus-multiple contrast is -0.231 rather than -0.545, and its variance is still too low (0.026 versus 0.046). Experiment 4 also regressed from 0.10 to 0.26 crossover prevalence despite an observed value of zero. Experiment 3 regressed from a good positive balance effect of 0.110 to nearly zero (0.007 versus 0.072). Experiment 2, although substantially improved, still overpredicts top-cue adherence by 0.124 and produces far too little subject heterogeneity. Experiment 1 also deteriorated from an exact fit to 0.920 with nonzero between-subject variance, suggesting that the new challenge mechanism is affecting profiles more broadly than intended.
Rationale: Retain SCCI and its no-learning commitment, but revise the scope and calibration of the challenge/restoration components. The prior advice was accepted overall, so configuration-specific gating is the right direction; however, simply strengthening the same isolated-anchor challenge again is not warranted because Experiment 6 became slightly less negative while Experiments 1, 3, and 4 worsened. A likely culprit is that `challenge_gain` always includes `coalition_accumulation`, even for non-isolated unique anchors. Make the compact lone-challenge boost strictly conditional on an isolated anchor, rather than applying a general challenge to every unique maximum. This should recover Experiment 1 and suppress the spurious Experiment 4 crossover without abandoning static integration. Then obtain the larger Experiment 6 contrast primarily by sharpening the multi-opponent restoration transition within genuinely isolated-anchor profiles, rather than by further increasing lone-challenge strength. Tune its count and validity-gap thresholds jointly and allow a steeper transition or somewhat higher effective response sensitivity once the gate is engaged; this should increase both the mean contrast and its subject variance. To protect Experiment 3, restoration should not indiscriminately rescue an isolated top cue in ordinary 3-versus-1 profiles with only a modest validity advantage. Use a sharper validity-gap requirement or a redundancy/coalition-compactness normalization so that the special restoration applies to the strong-anchor configurations in Experiment 6 but not broadly to every imbalanced coalition. Keep `profile_balance_gain` at or very near zero: its current minority boost works against the observed positive balanced-minus-imbalanced effect. Finally, widen heterogeneity in the restoration threshold/gap parameters rather than in broad validity sensitivity or challenge strength, since Experiment 2 needs more between-subject variance without restoring its excessive mean adherence. Preserve the tie-only order gate and history invariance, which continue to support Experiments 5 and 7.

**Outcome of this advice:** iter 3 candidate loss=0.1797 -> iter 4 candidate loss=0.1682 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed static, history-invariant profile-gated family, and the latest revision again improved aggregate loss. It preserves strong fits in Experiment 5 (0.776 versus 0.790), Experiment 7 (near-zero diagnostic/control difference), and Experiment 8 (small profile heterogeneity). Restricting challenge amplification to isolated anchors also reduced the false Experiment 4 crossover from 0.26 to 0.12. Nevertheless, the model is not experiment-invariant. Experiment 6 remains the largest failure: the human lone-minus-multiple contrast is -0.545, but the candidate produces only -0.189, worse than both preceding iterations despite the sharper restoration gate. Experiment 3 has also reversed into the wrong direction (-0.062 versus +0.072), indicating that restoration is protecting the top cue in imbalanced coalitions where humans instead allow coalition structure to reduce top-cue adherence. Experiment 2 still overpredicts top-cue adherence (0.463 versus 0.324) and severely underpredicts its between-subject variance. Experiment 1 has continued to deteriorate to 0.890 despite universal above-chance first-cue adherence in the data. Experiment 4's remaining 0.12 crossover prevalence is also non-negligible relative to the observed zero.
Rationale: Keep SCCI and its no-learning commitment, but do not repeat the latest strategy of merely lowering and steepening the count threshold: although that revision was accepted overall, it moved Experiment 6 in the wrong direction and made Experiment 3 negative. The restoration gate needs better profile selectivity plus a different way to increase its behavioral impact. First, make the validity-gap criterion scale-sensitive—such as a log-odds or relative gap—and place its threshold between the modest-gap profiles driving Experiment 3 and the genuinely strong-anchor profiles in Experiment 6. The current very low absolute gap thresholds allow restoration to engage too broadly. Set profile_balance_gain exactly to zero, since minority amplification works against the positive Experiment 3 interaction. Second, retain the isolated-anchor and multiple-opponent restrictions, but add a bounded restoration-specific response gain or temperature once that gate is active, rather than further changing the count threshold or increasing lone-challenge strength. This can raise multi-opponent anchor adherence in Experiment 6 without altering lone-opponent choices, tied-top order effects, or ordinary profiles. Keep challenge strength from increasing: the continuing Experiment 1 decline shows that stronger lone challenge would be unsafe. Outside the selectively restored regime, modestly strengthen ordinary coalition accumulation or weaken redundancy discount so that lower-validity coalitions can reduce the excessive top-cue adherence in Experiment 2 and recover the positive Experiment 3 balance effect. Finally, widen subject variation in the normalized gap threshold and coalition/redundancy parameters, rather than broad validity sensitivity, to address the underestimated variances in Experiments 2 and 6. Preserve the tie-only order gate and history invariance that support Experiments 5 and 7.

**Outcome of this advice:** iter 4 candidate loss=0.1682 -> iter 5 candidate loss=0.2266 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed static, history-invariant profile-gated family, but it was rejected by the accept gate and is substantially worse than the accepted iteration-4 base. The latest normalized-gap plus restoration-temperature package did not have its intended effect. Experiment 6 deteriorated sharply from -0.189 to -0.067 against the observed -0.545, so multi-opponent profiles are still not receiving effective anchor restoration. Experiment 3 moved further in the wrong direction (-0.086 versus +0.072), and the false crossover prevalence in Experiment 4 rose from 0.12 to 0.30. Experiment 5 also weakened from 0.776 to 0.741. Experiment 2 remains too anchor-consistent (0.453 versus 0.324) and much too homogeneous across subjects, while Experiment 1 remains below the universal above-chance prevalence in the data. Only Experiments 7 and 8 remain convincingly close, supporting the no-history commitment and generally low profile heterogeneity.
Rationale: Because the last recommendation produced a rejected candidate, do not repeat the package of normalized log-odds thresholds, stronger ordinary coalition accumulation, and restoration-specific response gain. In particular, a temperature gain is unsafe: when restoration activation is partial and the resulting evidence still opposes the anchor, increasing beta makes the wrong profile response more consistent rather than restoring the anchor. Build the next attempt on the unchanged accepted iteration-4 base and make a narrower in-family edit. Replace the broad count-plus-gap restoration trigger with a bounded redundancy-profile gate: require an isolated unique anchor, at least two concordant opponents, and evidence that the opposing cues form a redundant lower-validity bloc, such as low dispersion among opponent weights together with no individually near-anchor opponent. Then raise the anchor mixture itself, or impose a small positive anchor-margin floor, only after that gate activates; do not alter response temperature. This directly targets Experiment 6 while avoiding restoration in the ordinary imbalanced profiles responsible for the negative Experiment 3 interaction. Keep profile_balance_gain at zero, but otherwise retain the accepted base's coalition and redundancy ranges rather than again strengthening coalition accumulation globally, since the global change coincided with regressions in Experiments 3, 4, and 5. Also leave the tie-order gate unchanged because iteration 4 already fit Experiment 5 well. Finally, expose subject variation mainly through the new redundancy-profile threshold and gate strength; this may increase the deficient variances in Experiments 2 and 6 without broadening validity sensitivity or disturbing the successful null history effect in Experiment 7.

**Outcome of this advice:** iter 5 candidate loss=0.2266 -> iter 6 candidate loss=0.2569 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed static, history-invariant profile-gated family, but it was rejected by the accept gate and is markedly worse than the accepted iteration-4 base. The newly introduced redundancy-profile restoration gate failed in both selectivity and efficacy. Experiment 6 remains far from the defining human reversal (-0.070 versus -0.545), so the gate is not substantially increasing multi-opponent anchor adherence. At the same time, Experiment 1 collapses from 0.890 in the accepted base to 0.320 versus an observed 1.000, with greatly inflated between-subject variance, indicating severe unintended effects outside the target profile. Experiment 4 also retains a false primacy crossover (0.200 versus 0.000), and Experiment 3 still has the coalition-balance interaction in the wrong direction (-0.069 versus +0.072). Experiment 2 remains too anchor-consistent and much too homogeneous across subjects. Experiments 5, 7, and 8 are reasonably close, continuing to support the tie-specific order mechanism, absence of history learning, and generally small profile heterogeneity.
Rationale: The previous recommendation to replace the broad restoration trigger with a homogeneous-lower-validity-bloc gate was rejected, so do not repeat or further tune that gate. Its thresholds in transformed-weight space appear brittle: it underactivates on the Experiment 6 profiles while producing large collateral changes elsewhere. Use the unchanged accepted iteration-4 base and shift attention to the other side of the Experiment 6 contrast: lower adherence specifically in the lone-opponent condition rather than again trying to raise multi-opponent adherence through a new restoration rule. Make challenge amplification an exact conjunction of a strictly unique anchor, an isolated anchor side, and opposition_count == 1; replace the current Gaussian count kernel, which leaks into multi-opponent profiles, with a hard or very sharp lone-opponent indicator. Also separate challenge magnitude from coalition_accumulation rather than using their sum, then moderately widen or raise the independent challenge-strength range. This narrowly scoped adjustment should make Experiment 6 more negative by reducing lone-condition adherence and should also reduce excessive top-cue adherence in Experiment 2's pairwise conflicts, while leaving the multi-cue profiles in Experiments 1, 3, and 4 and the tied-top profiles in Experiment 5 largely untouched. Subject variation in this exact lone-challenge parameter may also increase the underestimated variance in Experiment 2. Preserve iteration 4's ordinary coalition, redundancy, tie-order, temperature, lapse, and no-history components; do not reintroduce restoration-specific temperature, normalized-gap gating, or the rejected homogeneity/deficit restoration package.

**Outcome of this advice:** iter 6 candidate loss=0.2569 -> iter 7 candidate loss=0.1857 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The candidate remains within the prescribed static, history-invariant profile-gated integration family, but the current iteration was rejected and is worse than the accepted iteration-4 base. The exact lone-opponent challenge produced only a modest improvement in Experiment 6 relative to that base (-0.235 versus -0.189), still far from the observed -0.545, while worsening other diagnostics. Experiment 3 remains reversed (-0.078 versus +0.072), Experiment 4 again shows an impermissible primacy crossover (0.22 versus 0.00), and Experiment 2 still overpredicts top-cue adherence (0.450 versus 0.324) with much too little between-subject variance. Experiment 7 also drifts away from its required null contrast. Experiments 1, 5, and 8 are reasonably close, supporting static cue representations, tie-specific order bias, and low profile heterogeneity, but they do not offset the systematic coalition-profile failures.
Rationale: Because the exact lone-opponent challenge recommendation was rejected by the gate, do not further increase or widen challenge_strength; build from the unchanged iteration-4 base instead. The next narrow in-family change should target the core redundancy normalization rather than another special-purpose challenge or restoration gate. The current coalition formula uses a linear denominator, so support can continue growing substantially with coalition size. Introduce a moderately convex, subject-varying concordance discount—for example, replace 1 + d(n-1) with 1 + d(n-1)^rho, with rho modestly above 1—while retaining validity-weighted accumulation. This supplies one common static mechanism for the key patterns: it discounts the multi-cue opposing bloc in Experiment 6, discounts Expert 5's three-cue majority more than its balanced coalition in Experiment 3 and should therefore restore a positive balanced-minus-imbalanced effect, protects the first cue from very large blocs in Experiment 1, and may reduce count-driven crossover heterogeneity in Experiment 4. Narrow profile_balance_gain to zero or nearly zero because its minority boost directly counteracts the needed Experiment 3 effect. Do not reintroduce the rejected transformed-gap, restoration-temperature, or homogeneous-bloc gates, and retain iteration 4's tie-order and no-history components. Let heterogeneity in the new redundancy-curvature parameter increase configuration-specific variance without broadening validity sensitivity. After those coalition patterns improve, Experiment 2's residual excessive top-cue adherence can be addressed separately; a broad increase in challenge or restoration is not supported by the present result.

**Outcome of this advice:** iter 7 candidate loss=0.1857 -> iter 8 candidate loss=0.1607 -> the gate marked it ACCEPTED.

### Iteration 8
Verdict: regenerate
Interpretation: The accepted convex-redundancy revision improves aggregate loss and remains faithful to the prescribed static, history-invariant profile-gated family. It restores an exact fit in Experiment 1, closely matches tied-top order adherence in Experiment 5, preserves the null history contrast in Experiment 7, and keeps Experiment 8 profile heterogeneity small. The previous advice therefore helped overall. However, the central cross-experiment coalition pattern is still not captured. Experiment 6 remains far too weakly negative (-0.237 versus -0.545), Experiment 3 remains in the wrong direction (-0.061 versus +0.072), and Experiment 4 has worsened to 0.26 false crossover prevalence despite an observed value of zero. Experiment 2 also continues to overpredict top-cue adherence (0.446 versus 0.324) while producing only about one fifth of the observed between-subject variance. Thus the new curvature mainly rescued Experiment 1; it did not deliver the intended balanced-versus-imbalanced effect and introduced substantial subject-level crossover behavior.
Rationale: Retain the accepted convex concordance discount, but do not simply increase redundancy_curvature: its introduction was accepted, yet it barely changed Experiment 3 and coincided with a marked Experiment 4 regression. The next revision should focus on calibration and static parameter dependence rather than another new gate. First, narrow the ordinary coalition_accumulation/redundancy ranges so that fewer subjects develop a large Expert-1-ally crossover in Experiment 4; preserve curvature chiefly as an incremental discount for cues beyond the first two, rather than strongly changing two-cue coalitions. Second, use the existing raw validity-gap restoration criterion more selectively—without returning to the rejected transformed-gap or restoration-temperature schemes—so restoration is largely inactive for the modest-gap imbalanced profiles in Experiment 3 but reliably active for the stronger isolated-anchor profiles in Experiment 6. Third, exploit the stable profile heterogeneity already prescribed by correlating existing subject parameters rather than widening challenge_strength again: subjects with stronger isolated-anchor restoration should also have stronger lone-opponent challenge and weaker ordinary validity sensitivity, while keeping the current marginal challenge range. This can amplify the Experiment 6 contrast and increase its variance without broadly raising top-cue adherence in Experiment 2. Variation should be shifted from ordinary coalition parameters, which generate the false Experiment 4 crossover, into the raw gap threshold and this correlated gate/challenge propensity, which may also raise Experiment 2's deficient variance. Preserve the successful tie-only order bias, convex redundancy mechanism, and complete history invariance.

**Outcome of this advice:** iter 8 candidate loss=0.1607 -> iter 9 candidate loss=0.1599 -> the gate marked it ACCEPTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: The current iteration is a marginal aggregate improvement and remains faithful to the prescribed static, history-invariant profile-gated family. It exactly fits Experiment 1, preserves the near-zero history contrast in Experiment 7, and remains close on Experiments 5 and 8. Narrowing ordinary coalition parameters also reduced the false Experiment 4 crossover slightly, from 0.26 to 0.22. However, the candidate is not experiment-invariant. Experiment 2 still substantially overpredicts highest-validity-cue adherence (0.467 versus 0.324) and severely underpredicts between-subject variance. Experiment 3 remains directionally wrong (-0.059 versus +0.072). Experiment 4 still predicts crossover behavior for 22% of subjects when none is observed. Most importantly, Experiment 6 deteriorated from -0.237 to -0.146, far from the observed -0.545. Thus the newly imposed restoration–challenge–validity correlation did not produce its intended profile contrast, despite the tiny aggregate-loss improvement.
Rationale: Retain SCCI, convex redundancy, tie-only order bias, and complete history invariance, but do not further strengthen the new correlated propensity: it was accepted only marginally and moved Experiment 6 sharply in the wrong direction while worsening Experiment 2 and Experiment 5. Remove or greatly attenuate that deterministic correlation and address a more specific attenuation in the existing anchor gate. The code multiplies restoration by both `gap_restoration` and the graded `unique_anchor = 1-exp(-40*gap)`. Consequently, even a profile that passes the raw-gap restoration threshold cannot receive the intended near-complete gate unless its raw gap is very large. Use uniqueness chiefly as a tie detector—zero or near zero for tied maxima and approximately one for a genuinely unique maximum—while leaving the existing raw-gap sigmoid as the sole graded selector. Calibrate that sigmoid narrowly between the modest-gap profiles implicated in Experiment 3 and the stronger isolated-anchor profiles in Experiment 6. This is an in-family correction to the current gate, not a return to the rejected transformed-gap or temperature schemes. It should let qualifying multi-opponent profiles approach full bounded restoration without broadly increasing anchor adherence in Experiment 2. Separately, continue the accepted reduction of ordinary coalition-generated heterogeneity, but target the two-cue ally effect directly: modestly raise and narrow the redundancy-discount floor rather than increasing curvature, since curvature does not alter a two-cue coalition and therefore cannot eliminate Experiment 4's Expert-1 ally crossover. This adjustment may also move Experiment 3 positive by discounting the larger concordant coalition and should raise Experiment 5 toward its observed adherence. Preserve the exact lone-challenge scope and avoid widening challenge strength, adding response-temperature gates, or introducing history learning, all of which were unsupported in prior rejected attempts.

**Outcome of this advice:** iter 9 candidate loss=0.1599 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Static Configuration-Contingent Integration (SCCI) claims that people form a fixed, instruction-based representation of cue reliability and apply it to the directional evidence profile on each trial. Each discriminating cue contributes reliability-weighted evidence, so independently supporting cues accumulate, while concordant cues are compressed as potentially redundant. This compression increases moderately convexly with coalition size, reflecting the declining informational value of each additional concordant cue. A bounded configuration gate supplements integration: strong protection of a uniquely most-valid cue is restricted to profiles in which that cue is isolated, has a sufficiently large communicated-validity advantage, and faces multiple concordant opponents. Cue uniqueness functions primarily as a tie detector, while a separate raw validity-gap criterion determines whether restoration is warranted. The transition to restoration is sharp, allowing an isolated anchor to be weak against one credible challenger but strongly protected against a redundant multi-cue bloc. Compact challenge amplification is strictly limited to isolated anchors and therefore cannot create general primacy effects when the anchor has allies. When the most-valid discriminating cues have equal communicated validity, they remain exchangeable in the integrator, but stable display order supplies a bounded tie-breaking anchor. Choices are history-invariant because concurrence, anti-concurrence, and discrimination frequency are not learned without correctness feedback. Stable profile heterogeneity arises from subject-level variation in validity sensitivity, coalition integration, redundancy, isolated-anchor challenge and restoration, tie-order bias, temperature, and lapse.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"SCCI expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.shape[0] != n_features:\n        raise ValueError(\n            f\"validities length {validities.shape[0]} != n_features {n_features}.\"\n        )\n\n    # Positive directions favor B and negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    coalition_accumulation = float(parameters[\"coalition_accumulation\"])\n    redundancy_discount = float(parameters[\"redundancy_discount\"])\n    redundancy_curvature = float(parameters[\"redundancy_curvature\"])\n    profile_balance_gain = float(parameters[\"profile_balance_gain\"])\n    anchor_gate_strength = float(parameters[\"anchor_gate_strength\"])\n    challenge_strength = float(parameters[\"challenge_strength\"])\n    restoration_strength = float(parameters[\"restoration_strength\"])\n    restoration_threshold = float(parameters[\"restoration_threshold\"])\n    restoration_slope = float(parameters[\"restoration_slope\"])\n    restoration_gap_threshold = float(parameters[\"restoration_gap_threshold\"])\n    restoration_gap_slope = float(parameters[\"restoration_gap_slope\"])\n    tie_order_bias = float(parameters[\"tie_order_bias\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n\n    # Communicated validity is transformed into reliability evidence. The\n    # subtraction only fixes scale and does not alter validity ordering.\n    v = np.clip(validities, 0.500001, 0.999999)\n    log_odds = np.log(v / (1.0 - v))\n    active_reference = float(np.max(log_odds[active]))\n    cue_weight = np.exp(\n        validity_sensitivity * (log_odds - active_reference)\n    )\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n\n    def coalition_support(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n        # Accumulation represents multiple nominally independent reasons.\n        # A moderately convex denominator captures increasing redundancy\n        # among additional cues in the same directional coalition.\n        mean_reliability = float(np.mean(cue_weight[indices]))\n        accumulated = mean_reliability * (float(n) ** coalition_accumulation)\n        redundancy = 1.0 + redundancy_discount * (\n            float(n - 1) ** redundancy_curvature\n        )\n        return accumulated / redundancy\n\n    support_a = coalition_support(a_cues)\n    support_b = coalition_support(b_cues)\n\n    # Directional minorities receive only a very small configuration-based\n    # salience adjustment. Most coalition effects arise from accumulation,\n    # redundancy, and the isolated-anchor mechanism below.\n    if n_a > 0 and n_b > 0 and n_a != n_b:\n        if n_a < n_b:\n            ratio = float(n_b) / float(n_a)\n            support_a *= ratio ** profile_balance_gain\n        else:\n            ratio = float(n_a) / float(n_b)\n            support_b *= ratio ** profile_balance_gain\n\n    total_support = support_a + support_b\n    core_evidence = (support_b - support_a) / max(total_support, 1e-12)\n\n    # The validity anchor is the most-valid currently discriminating cue.\n    # Equal-validity candidates have identical integrative weights; order is\n    # consulted only to select a bounded tie anchor.\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]\n    anchor_cue = int(np.min(best_candidates))\n    anchor_direction = float(directions[anchor_cue])\n\n    # Uniqueness is a tie detector rather than a second graded validity-gap\n    # penalty. The separate raw-gap sigmoid below selects strong anchors.\n    if active.size > 1:\n        sorted_active = np.sort(active_v)[::-1]\n        validity_gap = float(sorted_active[0] - sorted_active[1])\n    else:\n        validity_gap = 0.5\n    unique_anchor = float(best_candidates.size == 1)\n\n    if anchor_direction > 0:\n        anchor_side_count = n_b\n        opposition_count = n_a\n        anchor_side_support = support_b\n        opposition_support = support_a\n    else:\n        anchor_side_count = n_a\n        opposition_count = n_b\n        anchor_side_support = support_a\n        opposition_support = support_b\n\n    isolated_anchor = float(anchor_side_count == 1)\n\n    if opposition_count == 0:\n        weak_opposition = 1.0\n    else:\n        relative_opposition = opposition_support / max(\n            anchor_side_support + opposition_support, 1e-12\n        )\n        weak_opposition = np.exp(-4.0 * relative_opposition)\n\n    # A single independent opponent receives extra challenge amplification\n    # only when the validity anchor is genuinely isolated. Non-isolated\n    # profiles are governed by ordinary static coalition integration.\n    if opposition_count > 0:\n        compact_challenge = np.exp(\n            -4.0 * float(opposition_count - 1) ** 2\n        )\n        challenge_gain = isolated_anchor * (\n            coalition_accumulation + challenge_strength\n        )\n        challenged_opposition = opposition_support * (\n            1.0\n            + unique_anchor * challenge_gain * compact_challenge\n        )\n        anchor_margin = (\n            anchor_side_support - challenged_opposition\n        ) / max(anchor_side_support + challenged_opposition, 1e-12)\n        core_evidence = anchor_direction * anchor_margin\n\n    # Strong restoration is specific to an isolated and clearly superior\n    # anchor facing multiple concordant opponents. Count and raw validity-gap\n    # thresholds vary across subjects, with a sharp gap transition separating\n    # modest-gap profiles from genuinely strong-anchor configurations.\n    opposition_restoration = 1.0 / (\n        1.0 + np.exp(\n            -restoration_slope\n            * (float(opposition_count) - restoration_threshold)\n        )\n    )\n    gap_restoration = 1.0 / (\n        1.0 + np.exp(\n            -restoration_gap_slope\n            * (validity_gap - restoration_gap_threshold)\n        )\n    )\n    unique_gate_profile = (\n        0.01 * weak_opposition\n        + isolated_anchor\n        * restoration_strength\n        * opposition_restoration\n        * gap_restoration\n    )\n    unique_gate = anchor_gate_strength * unique_anchor * float(\n        np.clip(unique_gate_profile, 0.0, 1.0)\n    )\n\n    # Stable order matters only when multiple currently discriminating cues\n    # share the maximum communicated validity.\n    tied_active_top = float(best_candidates.size > 1)\n    tie_gate = tie_order_bias * tied_active_top\n\n    gate = 1.0 - (1.0 - unique_gate) * (1.0 - tie_gate)\n    gate = float(np.clip(gate, 0.0, 0.95))\n\n    choice_evidence = (\n        (1.0 - gate) * core_evidence\n        + gate * anchor_direction\n    )\n\n    logits = beta * np.array(\n        [-0.5 * choice_evidence, 0.5 * choice_evidence], dtype=np.float64\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array([0.5, 0.5])\n    probs = np.clip(probs, 0.0, None)\n    probs /= probs.sum()\n    return probs.astype(np.float64)\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "validity_sensitivity": "[0.03, 0.35]",
    "coalition_accumulation": "[0.98, 1.16]",
    "redundancy_discount": "[0.30, 0.42]",
    "redundancy_curvature": "[1.15, 1.45]",
    "profile_balance_gain": "[0.0, 0.0]",
    "anchor_gate_strength": "[0.90, 0.995]",
    "challenge_strength": "[0.8, 2.8]",
    "restoration_strength": "[0.95, 1.0]",
    "restoration_threshold": "[1.15, 1.55]",
    "restoration_slope": "[16.0, 30.0]",
    "restoration_gap_threshold": "[0.055, 0.075]",
    "restoration_gap_slope": "[160.0, 240.0]",
    "tie_order_bias": "[0.58, 0.85]",
    "beta": "[0.9, 3.4]",
    "lapse": "[0.0, 0.12]"
  },
  "rationale": "This is a minimal edit of the accepted iteration-9 SCCI model. First, it removes the deterministic restoration–challenge–validity correlation, which had moved Experiment 6 sharply in the wrong direction and increased ordinary top-cue adherence. Second, `unique_anchor` is changed from the graded expression `1-exp(-40*gap)` to a strict communicated-validity tie detector. Thus a genuinely unique anchor is no longer attenuated twice by both uniqueness and the raw-gap sigmoid; the sharpened raw-gap threshold is now the sole graded selector separating modest-gap profiles from strong isolated anchors. Qualifying multi-opponent profiles can consequently receive near-complete bounded restoration without adding a temperature manipulation or history learning. Third, the redundancy-discount floor is modestly raised and narrowed while retaining the accepted convex curvature. This directly discounts two-cue allied coalitions, which curvature alone cannot do, and more strongly compresses larger concordant coalitions. It should reduce the unsupported Experiment 4 ally crossover, reduce Expert 5 support in Experiment 3's imbalanced majority profiles, and preserve protection against very large blocs in Experiment 1. All other integration, challenge, tie-order, response-noise, and history-invariance mechanisms are unchanged."
}
```

## Usage

```json
{
  "input_tokens": 28523,
  "output_tokens": 3869
}
```
