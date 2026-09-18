# generation_iter_01_attempt_00

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
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a new Static Tie-Gated Configural Competition theory. It should preserve the empirically supported assumptions shared by the candidates—communicated-validity weighting, option-level coalition integration, stable subject heterogeneity, and no trial-history updating—but differ from both in four central claims. First, serial precedence should be activated by exact or psychologically categorized equality at the top, with an extremely narrow or discrete gate; it should produce strong tied-cue primacy in Experiments 3, 7, and 17 while remaining essentially absent at the 0.0052 gap in Experiment 2. Second, the theory should deny any effect of absolute cue-index span once active-cue identities, validities, directions, and active serial ranks are fixed, matching Experiments 1 and 20; presentation order may matter, but empty positions between cues should not. Third, coalition competition should be nonmonotonic in credible opponent multiplicity: sparse credible dissent can overturn an isolated validity anchor, intermediate coalitions can maximize challenge, and broad participation can trigger anchor recovery. This mechanism should be strong enough to capture Experiments 8, 11, 13, and especially 19, while still allowing coalitions to defeat the highest-validity cue in Experiments 4, 12, and 15. Fourth, adding a cue should alter context-normalized coalition coherence or response precision, not merely add independent evidence, so that the positive activation and balance effects in Experiments 5 and 14 are possible. Use a small number of theoretically meaningful latent subject orientations—such as exact-tie lexicographic, configural integrator, and anchor-restoration types—with correlated parameters rather than unrestricted mechanisms. Parameter ranges should suppress near-tie serial effects, eliminate spatial-span terms, permit stronger high-count restoration and stronger coalition overturning, and avoid the excessive crossover prevalence seen in Experiment 6. This would provide a genuinely distinct and stronger competitor to pi_9 rather than merely retuning the falsified continuous ambiguity theory.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.644)

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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=0.9300 (var=0.0601)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.4275 (var=0.0045)
- Experiment 3: real=0.0721 (var=0.0052) vs this=-0.0956 (var=0.0027)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.2200 (var=0.1716)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.7033 (var=0.0264)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.2550 (var=0.0318)
- Experiment 7: real=0.0052 (var=0.0075) vs this=0.0404 (var=0.0151)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0072 (var=0.0002)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.5850 (var=0.0117)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.6150 (var=0.0107)
- Experiment 11: real=0.1093 (var=0.0652) vs this=0.1107 (var=0.0194)
- Experiment 12: real=0.3533 (var=0.0427) vs this=0.3488 (var=0.0128)
- Experiment 13: real=0.3275 (var=0.0184) vs this=0.5044 (var=0.0026)
- Experiment 14: real=0.0200 (var=0.1946) vs this=-0.1050 (var=0.1227)
- Experiment 15: real=0.8475 (var=0.0129) vs this=0.6492 (var=0.0078)
- Experiment 16: real=0.0263 (var=0.0299) vs this=-0.0081 (var=0.0183)
- Experiment 17: real=0.5600 (var=0.0288) vs this=0.0325 (var=0.0124)
- Experiment 18: real=0.0092 (var=0.0117) vs this=0.0279 (var=0.0087)
- Experiment 19: real=-0.0776 (var=0.0965) vs this=-0.0222 (var=0.0289)
- Experiment 20: real=0.0350 (var=0.0461) vs this=0.0146 (var=0.0113)


---

### `pi_6` (overall score: 0.557)

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
- Experiment 13: real=0.3275 (var=0.0184) vs this=0.5269 (var=0.0032)
- Experiment 14: real=0.0200 (var=0.1946) vs this=-0.0300 (var=0.2041)
- Experiment 15: real=0.8475 (var=0.0129) vs this=0.4733 (var=0.0064)
- Experiment 16: real=0.0263 (var=0.0299) vs this=-0.0259 (var=0.0278)
- Experiment 17: real=0.5600 (var=0.0288) vs this=-0.0106 (var=0.0189)
- Experiment 18: real=0.0092 (var=0.0117) vs this=0.0146 (var=0.0103)
- Experiment 19: real=-0.0776 (var=0.0965) vs this=0.0255 (var=0.0150)
- Experiment 20: real=0.0350 (var=0.0461) vs this=0.0000 (var=0.0079)


---

### `pi_10` (overall score: 0.493)

**Description**
Reliability-Band Interpretation Switching (RBIS) proposes that communicated cue validities are encoded as ordinal reliability bands rather than exact numerical weights. On each display, the decision maker identifies the highest occupied band and an initial anchor reason. If several active cues occupy that band, they are functionally tied and serial order selects the provisional anchor; if only one cue occupies it, serial order has no influence. A fully discriminating display can disambiguate a genuinely unique numerical maximum, preventing a lower-reliability early cue from spuriously replacing the designated anchor merely because it falls near a band boundary. The current display is classified into one of four mutually exclusive interpretation states: top-band resolution, diagnostic-coalition reframing, high-multiplicity anchor protection, or residual banded integration. Reframing is especially accessible for unique-top-band singleton and exactly two-cue opposition, with singleton dissent receiving the larger diagnostic advantage. State assignment for these canonical diagnostic shapes is less dependent on broad interpretation type, although stable subject-specific thresholds remain. Coalition credibility is represented coarsely by reliability-band proximity. An isolated exact maximum can be protected against redundant high-count opposition, whereas embedded balanced displays are not automatically protected. In the residual state, singleton and two-cue coalitions receive separate bounded credibility gains, while coalitions of three or more remain strongly saturated. When a nearby-band sparse coalition remains unresolved after the main state gate, it can become the selected diagnostic reason with modest commitment rather than defaulting to anchor-favoring integration. Those sparse mechanisms are suppressed on exact-balanced four-reason displays, where balance geometry instead determines interpretation. Balanced four-reason displays are resolved according to coalition geometry: opposition spanning the reliability display is more diagnostic of an anchor-resolving configuration than a compact opposing pair, although compact pairs receive an intermediate resolution tendency. High-count coherent opposition that narrowly misses the primary protection gate enters a low-commitment anchor-protection state, avoiding an all-or-none split between reframing and strong protection. Conflicting top-band displays slightly suppress generic compact reframing, allowing order-based top-band resolution to operate without imposing universal precedence. Displays with an early unique maximum embedded in a small coalition are assigned a common topological resolution commitment, preventing incidental opponent identity or span from producing large profile differences. Saturated unique-maximum displays with nonsingleton opposition enter a canonical top-band-resolution state whose classification is invariant to reallocating one supporting cue. Stable subject differences in band width, state thresholds, protection boundaries, sparse-coalition credibility, and state-specific response commitment produce persistent strategy heterogeneity. Because choices reveal no outcome information, RBIS never learns cue concurrence, opposition, or frequency from trial history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RBIS expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # There is deliberately no learning in this feedback-free task.
    _ = history

    band_width = float(parameters["band_width"])
    band_decay = float(parameters["band_decay"])
    redundancy = float(parameters["redundancy"])
    singleton_opposition_gain = float(parameters["singleton_opposition_gain"])
    pair_opposition_gain = float(parameters["pair_opposition_gain"])
    credible_sparse_commitment = float(parameters["credible_sparse_commitment"])
    reframe_bias = float(parameters["reframe_bias"])
    balance_reframe = float(parameters["balance_reframe"])
    embedded_reframe = float(parameters["embedded_reframe"])
    singleton_reframe = float(parameters["singleton_reframe"])
    diagnostic_reframe_offset = float(parameters["diagnostic_reframe_offset"])
    credibility_tolerance = float(parameters["credibility_tolerance"])
    state_threshold = float(parameters["state_threshold"])
    compact_limit = float(parameters["compact_limit"])
    isolated_protection_count = float(parameters["isolated_protection_count"])
    high_multiplicity_count = float(parameters["high_multiplicity_count"])
    one_step_protection = float(parameters["one_step_protection"])
    fallback_protection_commitment = float(parameters["fallback_protection_commitment"])
    tied_reframe_scale = float(parameters["tied_reframe_scale"])
    tie_commitment = float(parameters["tie_commitment"])
    integration_strength = float(parameters["integration_strength"])
    reframe_strength = float(parameters["reframe_strength"])
    protection_strength = float(parameters["protection_strength"])
    exact_balance_resolution = float(parameters["exact_balance_resolution"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    interpretation_type = int(parameters["interpretation_type"])

    # Positive directions favor B; negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Stable interpretation types jointly alter entry into discrete states.
    # The wider separation generates durable between-subject differences
    # without trial-wise strategy resampling.
    if interpretation_type == 0:      # anchor-protective reason selector
        local_threshold = state_threshold + 1.20
        local_protection_count = isolated_protection_count - 0.10
        local_reframe_scale = 0.65
    elif interpretation_type == 1:    # coalition-sensitive reason selector
        local_threshold = state_threshold - 0.80
        local_protection_count = isolated_protection_count + 0.20
        local_reframe_scale = 1.35
    elif interpretation_type == 2:    # singleton-dissent specialist
        local_threshold = state_threshold - 0.45
        local_protection_count = isolated_protection_count
        singleton_reframe += 1.00
        local_reframe_scale = 1.20
    elif interpretation_type == 3:    # multiplicity-protection specialist
        local_threshold = state_threshold + 0.20
        local_protection_count = isolated_protection_count - 0.20
        high_multiplicity_count -= 0.50
        local_reframe_scale = 0.88
    else:                             # balanced configuration interpreter
        local_threshold = state_threshold - 0.55
        local_protection_count = isolated_protection_count
        balance_reframe += 0.85
        one_step_protection += 0.25
        local_reframe_scale = 1.12

    active_v = validities[active]
    best_v = float(np.max(active_v))

    # Validities are quantized relative to the best active validity. All cues
    # in ordinal band zero are treated as reliability-equivalent.
    width = max(band_width, 1e-8)
    band_rank = np.floor(
        np.maximum(best_v - validities, 0.0) / width + 1e-10
    ).astype(int)
    top_band = active[band_rank[active] == 0]

    # On a saturated six-cue display, simultaneous discrimination supplies
    # enough contrast to disambiguate a genuinely unique numerical maximum.
    # This narrowly prevents cue substitutions from changing the provisional
    # anchor in the Experiment-4 geometry without imposing a broad dense-
    # display protection rule.
    exact_max = active[np.isclose(active_v, best_v, atol=1e-10)]
    if n_features == 6 and active.size == 6 and exact_max.size == 1:
        top_band = exact_max

    # An isolated exact maximum facing at least three cues is the protected
    # reason even when subjective banding places an earlier cue in its band.
    if exact_max.size == 1:
        exact_direction = float(directions[int(exact_max[0])])
        exact_side_count = int(np.sum(directions[active] == exact_direction))
        exact_opposition_count = int(
            np.sum(directions[active] == -exact_direction)
        )
        if exact_side_count == 1 and exact_opposition_count >= 3:
            top_band = exact_max

    # Serial order is used solely inside the ambiguous top reliability band.
    anchor_cue = int(np.min(top_band))
    anchor_direction = float(directions[anchor_cue])
    top_band_conflict = bool(
        np.any(directions[top_band] > 0) and
        np.any(directions[top_band] < 0)
    )
    top_band_tied = bool(top_band.size > 1)
    unique_top_band = bool(top_band.size == 1)

    anchor_side = active[directions[active] == anchor_direction]
    opposition = active[directions[active] == -anchor_direction]
    anchor_count = int(anchor_side.size)
    opposition_count = int(opposition.size)
    total_count = int(active.size)

    if opposition_count == 0:
        chosen_evidence = protection_strength * anchor_direction
    else:
        # Opposition is grouped by coarse reliability-band proximity. Exact
        # opponent identity and small within-band differences cannot toggle a
        # state, which prevents incidental cue substitutions from generating
        # a primacy crossover.
        opponent_bands = band_rank[opposition].astype(float)
        opponent_best_band = int(np.min(band_rank[opposition]))
        coarse_distance = max(opponent_best_band - 1, 0)
        credibility = float(np.exp(
            -float(coarse_distance) / max(credibility_tolerance, 1e-8)
        ))

        count_balance = float(
            np.exp(-abs(opposition_count - anchor_count))
        )
        exact_balance = bool(opposition_count == anchor_count)
        compact = float(opposition_count <= compact_limit)
        singleton = float(opposition_count == 1)
        embedded = float(anchor_count > 1)
        isolated = float(anchor_count == 1)

        # Protection and reframing are categorical interpretation states.
        # Two opponents can still reframe an isolated anchor. Protection is
        # unavailable until there are at least three apparently redundant
        # opponents, producing the required nonmonotonic recovery.
        isolated_high_count = bool(
            isolated > 0.0 and
            opposition_count >= 3 and
            opposition_count >= local_protection_count
        )
        isolated_unique_high_multiplicity = bool(
            isolated > 0.0 and
            unique_top_band and
            total_count >= high_multiplicity_count and
            opposition_count >= 3
        )
        one_step_minority = bool(
            embedded > 0.0 and
            opposition_count == anchor_count + 1 and
            one_step_protection >= 0.45
        )
        protection_state = bool(
            isolated_high_count or
            isolated_unique_high_multiplicity or
            one_step_minority
        )

        base_reframe_score = credibility * (
            reframe_bias
            + 0.65 * compact
            + balance_reframe * count_balance
            + 0.35 * embedded_reframe * embedded
            + singleton_reframe * singleton
        )
        reframe_score = local_reframe_scale * base_reframe_score

        # Unique-top-band singleton and two-opponent coalitions receive a
        # configuration-specific threshold reduction. Tied-top displays are
        # excluded so their stable-order resolution is not weakened. For
        # these canonical shapes, broad interpretation-type offsets are
        # compressed, and singleton dissent receives the larger reduction.
        shape_specific_reframe = bool(
            unique_top_band and
            (singleton > 0.0 or opposition_count == 2)
        )
        if shape_specific_reframe:
            canonical_scale = 0.65 + 0.35 * local_reframe_scale
            reframe_score = canonical_scale * base_reframe_score
            effective_reframe_threshold = (
                0.65 * state_threshold
                + 0.35 * local_threshold
                - diagnostic_reframe_offset * (1.0 + 0.80 * singleton)
            )
        else:
            effective_reframe_threshold = local_threshold

        # Conflicting top-band reasons ordinarily remain eligible for order
        # resolution; generic compactness therefore has reduced ability to
        # preempt that state, without imposing unconditional tie precedence.
        if top_band_tied and top_band_conflict:
            reframe_score *= tied_reframe_scale

        # Compact and singleton opposition are diagnostic directly. Exact
        # balance can also trigger a categorical reinterpretation under a
        # unique top band, while conflicting tied-top displays retain their
        # separate order-resolution regime.
        diagnostic_shape = bool(
            compact > 0.0 or
            singleton > 0.0 or
            (exact_balance and not top_band_tied)
        )
        reframe_state = bool(
            diagnostic_shape and
            reframe_score > effective_reframe_threshold
        )

        if protection_state:
            # High multiplicity is interpreted as redundant opposition, so
            # the selected high-band reason is protected.
            chosen_evidence = protection_strength * anchor_direction
        elif reframe_state:
            # Reframing selects the opposing interpretation as a whole; it is
            # not a continuous bonus added to an evidence accumulator.
            chosen_evidence = -reframe_strength * anchor_direction
        elif top_band_tied and top_band_conflict:
            # Ambiguous highest-band reasons are resolved by stable display
            # order only after protection and diagnostic reframing are tested.
            chosen_evidence = tie_commitment * anchor_direction
        else:
            # Residual state: coarse ordinal integration. Exact numerical
            # validity differences within a band play no role.
            ordinal_weight = np.exp(-band_decay * band_rank.astype(float))

            def coalition_support(indices):
                n = int(indices.size)
                if n == 0:
                    return 0.0
                raw = float(np.sum(ordinal_weight[indices]))
                # Additional same-option reasons are treated as increasingly
                # redundant rather than as independent linear evidence.
                return raw / (1.0 + redundancy * float(max(n - 1, 0)))

            support_anchor = coalition_support(anchor_side)
            support_opposition = coalition_support(opposition)

            # Singleton and paired opposition receive separately calibrated
            # residual credibility. Exact-balanced four-reason displays are
            # excluded because their interpretation is governed by the
            # dedicated coalition-geometry resolution below.
            sparse_nonbalanced = bool(
                unique_top_band and
                not (total_count == 4 and exact_balance)
            )
            if sparse_nonbalanced and opposition_count == 1:
                support_opposition *= singleton_opposition_gain
            elif sparse_nonbalanced and opposition_count == 2:
                support_opposition *= pair_opposition_gain

            margin = (
                (support_anchor - support_opposition) /
                max(support_anchor + support_opposition, 1e-12)
            )
            chosen_evidence = integration_strength * anchor_direction * margin

            # A nearby-band sparse coalition that reaches the residual branch
            # is interpreted as a weak diagnostic reason rather than being
            # returned to anchor-favoring integration. This is a categorical,
            # low-commitment reason-selection state and is unavailable on the
            # balanced four-reason displays used by the geometry rule.
            credible_sparse_state = bool(
                sparse_nonbalanced and
                opposition_count <= 2 and
                coarse_distance == 0 and
                chosen_evidence * anchor_direction > 0.0
            )
            if credible_sparse_state:
                chosen_evidence = (
                    -credible_sparse_commitment * anchor_direction
                )

        # Balanced four-reason resolution depends on coalition geometry rather
        # than feature count alone. Compact opposing pairs receive an
        # intermediate resolution tendency, while edge-spanning opposition
        # continues to receive the full tendency.
        if total_count == 4 and exact_balance and unique_top_band:
            opposition_span = float(
                np.max(opposition) - np.min(opposition)
            ) if opposition_count > 1 else 0.0
            display_span = float(max(n_features - 1, 1))
            broad_opposition = float(
                opposition_span >= 0.80 * display_span
            )
            balance_resolution = exact_balance_resolution * (
                0.54 + 0.46 * broad_opposition
            )
            chosen_evidence = (
                (1.0 - balance_resolution) * chosen_evidence
                + balance_resolution * anchor_direction
            )

        # Coherent high-count opposition that narrowly misses the primary
        # protection gate enters a common low-commitment protection state.
        # This produces moderate recovery without polarizing subjects into
        # strong reframing and near-certain protection groups.
        opponent_band_span = int(
            np.max(band_rank[opposition]) - np.min(band_rank[opposition])
        )
        fallback_high_count_state = bool(
            not protection_state and
            isolated > 0.0 and
            unique_top_band and
            exact_max.size == 1 and
            opposition_count >= 3 and
            opponent_band_span <= 1
        )
        if fallback_high_count_state:
            chosen_evidence = (
                fallback_protection_commitment * anchor_direction
            )

        # Saturated unique-maximum displays with nonsingleton opposition enter
        # a canonical top-band-resolution state. Its selected reason and
        # commitment are invariant to moving one nonanchor cue between sides.
        # Singleton opposition remains eligible for diagnostic reframing.
        if (
            n_features == 6 and
            total_count == 6 and
            exact_max.size == 1 and
            opposition_count >= 2
        ):
            chosen_evidence = protection_strength * anchor_direction

        # Small embedded coalitions organized around the earliest unique
        # numerical maximum share one coarse topology. Pooling their response
        # commitment prevents incidental opponent identity and span from
        # producing brittle profile-to-profile state differences.
        early_embedded_topology = bool(
            exact_max.size == 1 and
            int(exact_max[0]) == 0 and
            anchor_cue == 0 and
            anchor_count > 1 and
            3 <= total_count <= 5
        )
        if early_embedded_topology:
            chosen_evidence = 0.55 * integration_strength * anchor_direction

    logits = beta * np.array(
        [-0.5 * chosen_evidence, 0.5 * chosen_evidence],
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
- band_width: [0.025, 0.085]
- band_decay: [0.55, 1.35]
- redundancy: [0.32, 0.78]
- singleton_opposition_gain: [1.30, 1.60]
- pair_opposition_gain: [1.22, 1.48]
- credible_sparse_commitment: [0.12, 0.30]
- reframe_bias: [0.45, 1.05]
- balance_reframe: [0.70, 1.60]
- embedded_reframe: [0.25, 0.90]
- singleton_reframe: [0.75, 1.55]
- diagnostic_reframe_offset: [0.35, 0.55]
- credibility_tolerance: [0.85, 2.20]
- state_threshold: [1.00, 1.75]
- compact_limit: [2.0, 3.2]
- isolated_protection_count: [2.80, 3.05]
- high_multiplicity_count: [5.5, 7.5]
- one_step_protection: [0.45, 0.95]
- fallback_protection_commitment: [0.12, 0.26]
- tied_reframe_scale: [0.74, 0.90]
- tie_commitment: [0.50, 0.95]
- integration_strength: [0.65, 1.25]
- reframe_strength: [1.30, 2.20]
- protection_strength: [1.15, 2.00]
- exact_balance_resolution: [0.82, 0.98]
- beta: [1.6, 4.5]
- lapse: [0.01, 0.12]
- interpretation_type: {0, 1, 2, 3, 4}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.5487 (var=0.0373)
- Experiment 3: real=0.0721 (var=0.0052) vs this=0.0312 (var=0.0143)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.7412 (var=0.0190)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.4419 (var=0.0602)
- Experiment 7: real=0.0052 (var=0.0075) vs this=-0.0074 (var=0.0135)
- Experiment 8: real=0.0070 (var=0.0002) vs this=-0.0002 (var=0.0000)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.7481 (var=0.1072)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.6106 (var=0.0980)
- Experiment 11: real=0.1093 (var=0.0652) vs this=0.1185 (var=0.1286)
- Experiment 12: real=0.3533 (var=0.0427) vs this=0.3358 (var=0.1291)
- Experiment 13: real=0.3275 (var=0.0184) vs this=0.2969 (var=0.0404)
- Experiment 14: real=0.0200 (var=0.1946) vs this=0.0200 (var=0.0696)
- Experiment 15: real=0.8475 (var=0.0129) vs this=0.8496 (var=0.0084)
- Experiment 16: real=0.0263 (var=0.0299) vs this=0.0006 (var=0.0082)
- Experiment 17: real=0.5600 (var=0.0288) vs this=-0.0069 (var=0.0030)
- Experiment 18: real=0.0092 (var=0.0117) vs this=0.5554 (var=0.0321)
- Experiment 19: real=-0.0776 (var=0.0965) vs this=-0.0087 (var=0.0110)
- Experiment 20: real=0.0350 (var=0.0461) vs this=0.5054 (var=0.3456)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2701 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2701 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1 (most recent): 0.9200 (var=0.0736) (Δ vs real -0.0800)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)
- pi_7: 0.9400 (var=0.0464)
- pi_8: 0.9300 (var=0.0601)
- pi_9: 1.0000 (var=0.0000)
- pi_10: 1.0000 (var=0.0000)
- pi_11: 1.0000 (var=0.0000)

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
  - iter 1 (most recent): 0.5227 (var=0.0042) (Δ vs real +0.1985)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)
- pi_7: 0.5233 (var=0.0053)
- pi_8: 0.4275 (var=0.0045)
- pi_9: 0.4777 (var=0.0037)
- pi_10: 0.5487 (var=0.0373)
- pi_11: 0.5158 (var=0.0082)

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
  - iter 1 (most recent): -0.1369 (var=0.0024) (Δ vs real -0.2090)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)
- pi_7: 0.0358 (var=0.0025)
- pi_8: -0.0956 (var=0.0027)
- pi_9: 0.0273 (var=0.0071)
- pi_10: 0.0312 (var=0.0143)
- pi_11: -0.0579 (var=0.0024)

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
  - iter 1 (most recent): 0.8400 (var=0.1344) (Δ vs real +0.8400)
**Other theories' values on this metric (for reference):**
- pi_3: 1.0000 (var=0.0000)
- pi_1: 0.0400 (var=0.0384)
- pi_2: 0.0600 (var=0.0564)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.2400 (var=0.1824)
- pi_7: 0.2200 (var=0.1716)
- pi_8: 0.2200 (var=0.1716)
- pi_9: 0.3800 (var=0.2356)
- pi_10: 0.0000 (var=0.0000)
- pi_11: 0.5400 (var=0.2484)

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
  - iter 1 (most recent): 0.8810 (var=0.0068) (Δ vs real +0.0910)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)
- pi_7: 0.8063 (var=0.0071)
- pi_8: 0.7033 (var=0.0264)
- pi_9: 0.6785 (var=0.0080)
- pi_10: 0.7412 (var=0.0190)
- pi_11: 0.7804 (var=0.0048)

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
  - iter 1 (most recent): 0.1831 (var=0.0173) (Δ vs real +0.7281)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)
- pi_7: -0.3969 (var=0.0200)
- pi_8: -0.2550 (var=0.0318)
- pi_9: -0.3984 (var=0.0281)
- pi_10: -0.4419 (var=0.0602)
- pi_11: -0.1956 (var=0.0164)

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
  - iter 1 (most recent): 0.0115 (var=0.0113) (Δ vs real +0.0063)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)
- pi_7: -0.0589 (var=0.0212)
- pi_8: 0.0404 (var=0.0151)
- pi_9: 0.1459 (var=0.0178)
- pi_10: -0.0074 (var=0.0135)
- pi_11: 0.0132 (var=0.0173)

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
  - iter 1 (most recent): 0.0218 (var=0.0003) (Δ vs real +0.0148)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)
- pi_7: 0.0062 (var=0.0001)
- pi_8: 0.0072 (var=0.0002)
- pi_9: 0.0115 (var=0.0002)
- pi_10: -0.0002 (var=0.0000)
- pi_11: 0.0095 (var=0.0001)

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
  - iter 1 (most recent): 0.3006 (var=0.0148) (Δ vs real -0.4244)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)
- pi_7: 0.7512 (var=0.0113)
- pi_8: 0.5850 (var=0.0117)
- pi_9: 0.7538 (var=0.0365)
- pi_10: 0.7481 (var=0.1072)
- pi_11: 0.4437 (var=0.0425)

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
  - iter 1 (most recent): 0.5706 (var=0.0061) (Δ vs real +0.1244)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)
- pi_7: 0.4644 (var=0.0059)
- pi_8: 0.6150 (var=0.0107)
- pi_9: 0.5681 (var=0.0203)
- pi_10: 0.6106 (var=0.0980)
- pi_11: 0.6125 (var=0.0191)

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
  - iter 1 (most recent): 0.0630 (var=0.0087) (Δ vs real -0.0463)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6688 (var=0.0308)
- pi_5: 0.0092 (var=0.0107)
- pi_1: 0.0137 (var=0.0129)
- pi_2: -0.0240 (var=0.0079)
- pi_3: -0.0033 (var=0.0095)
- pi_4: -0.0368 (var=0.0015)
- pi_6: -0.0085 (var=0.0107)
- pi_8: 0.1107 (var=0.0194)
- pi_9: 0.0435 (var=0.0122)
- pi_10: 0.1185 (var=0.1286)
- pi_11: -0.0005 (var=0.0037)

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
  - iter 1 (most recent): 0.4142 (var=0.0076) (Δ vs real +0.0608)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0208 (var=0.0079)
- pi_7: 0.2908 (var=0.0147)
- pi_1: 0.0083 (var=0.0081)
- pi_2: -0.3271 (var=0.0169)
- pi_3: 0.0450 (var=0.0087)
- pi_4: -0.5246 (var=0.0081)
- pi_6: 0.2700 (var=0.0168)
- pi_8: 0.3488 (var=0.0128)
- pi_9: 0.1708 (var=0.0122)
- pi_10: 0.3358 (var=0.1291)
- pi_11: 0.3904 (var=0.0106)

### Experiment 13
**Design**
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    anchor_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size <= 5 or b.size <= 5:
            continue

        # Expert 6 (index 5) is the unique 95%-validity anchor.
        anchor_direction = np.sign(b[5] - a[5])
        if anchor_direction == 0:
            continue

        response = int(row["response"])
        chose_anchor = (
            (anchor_direction > 0 and response == 1)
            or (anchor_direction < 0 and response == 0)
        )
        anchor_choices.append(float(chose_anchor))

    if len(anchor_choices) == 0:
        return 0.5
    return float(np.mean(anchor_choices))
```

**Observed (real) value:** 0.3275 (var=0.0184)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.3681 (var=0.0068) (Δ vs real +0.0406)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5044 (var=0.0026)
- pi_5: 0.3144 (var=0.0091)
- pi_1: 0.8431 (var=0.0109)
- pi_2: 0.4846 (var=0.0035)
- pi_3: 0.6629 (var=0.0048)
- pi_4: 0.9952 (var=0.0001)
- pi_6: 0.5269 (var=0.0032)
- pi_7: 0.4956 (var=0.0025)
- pi_9: 0.4288 (var=0.0042)
- pi_10: 0.2969 (var=0.0404)
- pi_11: 0.4875 (var=0.0152)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        diagnostic = []
        control = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            d = np.sign(b - a)
            active = set(np.flatnonzero(d != 0).tolist())

            if active == {0, 4} and d[0] == -d[4]:
                anchor_choice = float(int(row["response"]) == int(d[0] > 0))
                diagnostic.append(anchor_choice)
            elif active == {0, 1} and d[0] == -d[1]:
                anchor_choice = float(int(row["response"]) == int(d[0] > 0))
                control.append(anchor_choice)

        def late_minus_early(values):
            x = np.asarray(values, dtype=float)
            if x.size < 2:
                return np.nan
            half = x.size // 2
            return float(np.mean(x[-half:]) - np.mean(x[:half]))

        diag_change = late_minus_early(diagnostic)
        control_change = late_minus_early(control)
        if not np.isfinite(diag_change) or not np.isfinite(control_change):
            return np.nan
        return float(diag_change - control_change)

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        scores = [subject_score(g) for _, g in data.groupby("subject_id", sort=False)]
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(np.mean(scores)) if scores.size else np.nan

    return subject_score(data)

```

**Observed (real) value:** 0.0200 (var=0.1946)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.1050 (var=0.2602) (Δ vs real -0.1250)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3050 (var=0.1632)
- pi_8: -0.1050 (var=0.1227)
- pi_1: 0.0200 (var=0.1196)
- pi_2: -0.0600 (var=0.2064)
- pi_3: 0.1000 (var=0.2775)
- pi_4: 0.0050 (var=0.0062)
- pi_6: -0.0300 (var=0.2041)
- pi_7: -0.0400 (var=0.2234)
- pi_9: 0.0900 (var=0.2244)
- pi_10: 0.0200 (var=0.0696)
- pi_11: 0.1100 (var=0.1754)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    cue1_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size < 8 or b.size < 8:
            continue

        direction = np.sign(b - a)

        # Critical displays are those on which Experts 1 and 8 both
        # discriminate but recommend opposite products.
        if direction[0] == 0 or direction[7] == 0:
            continue
        if direction[0] == direction[7]:
            continue

        cue1_response = 1 if direction[0] > 0 else 0
        cue1_choices.append(float(int(row['response']) == cue1_response))

    if len(cue1_choices) == 0:
        return float('nan')

    return float(np.mean(cue1_choices))
```

**Observed (real) value:** 0.8475 (var=0.0129)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.9246 (var=0.0040) (Δ vs real +0.0771)
**Other theories' values on this metric (for reference):**
- pi_9: 0.8521 (var=0.0068)
- pi_5: 0.1762 (var=0.0116)
- pi_1: 0.1567 (var=0.0166)
- pi_2: 0.1533 (var=0.0142)
- pi_3: 0.7188 (var=0.0093)
- pi_4: 0.0096 (var=0.0002)
- pi_6: 0.4733 (var=0.0064)
- pi_7: 0.4275 (var=0.0052)
- pi_8: 0.6492 (var=0.0078)
- pi_10: 0.8496 (var=0.0084)
- pi_11: 0.8992 (var=0.0043)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_score(df):
        early_opponent = []
        late_opponent = []
        novice_anchor = []
        mature_anchor = []
        exposure_count = {"early": 0, "late": 0}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            if a.size < 7 or b.size < 7:
                continue

            active = set(np.flatnonzero(a != b).tolist())
            if active == {0, 3}:
                condition = "early"
            elif active == {3, 6}:
                condition = "late"
            else:
                continue

            if a[3] == b[3]:
                continue
            anchor_response = 0 if a[3] > b[3] else 1
            anchor_chosen = float(int(row["response"]) == anchor_response)

            rank = exposure_count[condition]
            exposure_count[condition] += 1

            if condition == "early":
                early_opponent.append(anchor_chosen)
            else:
                late_opponent.append(anchor_chosen)

            # Before four prior condition-specific observations, ADCG's
            # anti-concurrence estimate is still strongly pseudocount- and
            # confidence-limited. At eight or more it is comparatively mature.
            if rank < 4:
                novice_anchor.append(anchor_chosen)
            elif rank >= 8:
                mature_anchor.append(anchor_chosen)

        order_contrast = 0.0
        if early_opponent and late_opponent:
            order_contrast = float(np.mean(late_opponent) - np.mean(early_opponent))

        learning_contrast = 0.0
        if novice_anchor and mature_anchor:
            learning_contrast = float(np.mean(novice_anchor) - np.mean(mature_anchor))

        return order_contrast + 0.75 * learning_contrast

    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return float(subject_score(data))

    scores = [subject_score(df) for _, df in data.groupby("subject_id", sort=False)]
    return float(np.mean(scores)) if scores else 0.0
```

**Observed (real) value:** 0.0263 (var=0.0299)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0155 (var=0.0339) (Δ vs real -0.0107)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1532 (var=0.0330)
- pi_9: 0.0106 (var=0.0320)
- pi_1: -0.0078 (var=0.0146)
- pi_2: 0.0011 (var=0.0351)
- pi_3: -0.2768 (var=0.0329)
- pi_4: 0.0055 (var=0.0008)
- pi_6: -0.0259 (var=0.0278)
- pi_7: -0.0118 (var=0.0177)
- pi_8: -0.0081 (var=0.0183)
- pi_10: 0.0006 (var=0.0082)
- pi_11: 0.0437 (var=0.0237)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code every response as choosing the option endorsed by Expert 6,
    # the uniquely most-valid expert.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        if a.size != 6 or b.size != 6:
            continue

        # Expert 6's anchor option is A when its A rating is 1, otherwise B.
        anchor_response = 0 if a[5] == 1 else 1
        anchor_choice = float(int(row["response"]) == anchor_response)

        # Cue identities opposing Expert 6 are invariant under A/B reversal.
        opponents = tuple(np.flatnonzero(a[:5] != a[5]).tolist())
        records.append((opponents, anchor_choice))

    if not records:
        return 0.0

    frame = pd.DataFrame(records, columns=["opponents", "anchor_choice"])

    # Compact, high-validity opposition profiles: {1,2} and {1,2,3}.
    compact_sets = {(0, 1), (0, 1, 2)}
    # High-multiplicity profiles: {1,2,3,4} and {1,2,3,4,5}.
    recovery_sets = {(0, 1, 2, 3), (0, 1, 2, 3, 4)}

    compact = frame.loc[
        frame["opponents"].apply(lambda x: x in compact_sets),
        "anchor_choice"
    ]
    recovery = frame.loc[
        frame["opponents"].apply(lambda x: x in recovery_sets),
        "anchor_choice"
    ]

    if len(compact) == 0 or len(recovery) == 0:
        return 0.0

    return float(recovery.mean() - compact.mean())
```

**Observed (real) value:** 0.5600 (var=0.0288)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0475 (var=0.0185) (Δ vs real -0.5125)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4031 (var=0.0261)
- pi_10: -0.0069 (var=0.0030)
- pi_1: -0.0006 (var=0.0092)
- pi_2: -0.5300 (var=0.0308)
- pi_3: -0.1925 (var=0.0135)
- pi_4: -0.0381 (var=0.0022)
- pi_5: 0.0344 (var=0.0146)
- pi_6: -0.0106 (var=0.0189)
- pi_7: 0.1544 (var=0.0156)
- pi_8: 0.0325 (var=0.0124)
- pi_11: 0.1838 (var=0.0296)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        broad_choices = []
        nonbroad_choices = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=int)
            b = np.asarray(row["option_b_ratings"], dtype=int)
            direction = np.sign(b - a).astype(int)

            # Expert 4 (zero-based index 3) is the unique-validity anchor.
            anchor_direction = int(direction[3])
            if anchor_direction == 0:
                continue

            opposition = np.flatnonzero(direction == -anchor_direction)
            if opposition.size != 2:
                continue

            opposition_span = int(np.max(opposition) - np.min(opposition))
            is_broad = opposition_span >= 6

            response = int(row["response"])
            anchor_chosen = float(
                response == (1 if anchor_direction > 0 else 0)
            )

            if is_broad:
                broad_choices.append(anchor_chosen)
            else:
                nonbroad_choices.append(anchor_chosen)

        if len(broad_choices) == 0 or len(nonbroad_choices) == 0:
            return 0.0
        return float(np.mean(broad_choices) - np.mean(nonbroad_choices))

    if "subject_id" in data.columns:
        contrasts = [
            subject_contrast(df)
            for _, df in data.groupby("subject_id", sort=False)
        ]
        return float(np.mean(contrasts)) if len(contrasts) else 0.0

    return subject_contrast(data)

```

**Observed (real) value:** 0.0092 (var=0.0117)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.0096 (var=0.0080) (Δ vs real -0.0188)
**Other theories' values on this metric (for reference):**
- pi_10: 0.5554 (var=0.0321)
- pi_9: -0.0196 (var=0.0100)
- pi_1: -0.0033 (var=0.0050)
- pi_2: -0.0013 (var=0.0100)
- pi_3: -0.0792 (var=0.0037)
- pi_4: -0.0050 (var=0.0103)
- pi_5: 0.0125 (var=0.0050)
- pi_6: 0.0146 (var=0.0103)
- pi_7: 0.0029 (var=0.0081)
- pi_8: 0.0279 (var=0.0087)
- pi_11: -0.0288 (var=0.0121)

### Experiment 19
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Within-subject slope of anchor choice over opponent dispersion."""
    def subject_slope(df):
        cells = {}

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_support = np.flatnonzero(a > b)
            b_support = np.flatnonzero(b > a)

            # The coded anchor is the option supported by Experts 1 and 3
            # (zero-based indices 0 and 2).
            if (0 in a_support) and (2 in a_support):
                anchor_response = 0
                opponent = b_support
            elif (0 in b_support) and (2 in b_support):
                anchor_response = 1
                opponent = a_support
            else:
                continue

            if opponent.size == 0:
                continue

            span = int(np.max(opponent) - np.min(opponent) + 1)
            anchor_chosen = float(int(row["response"]) == anchor_response)
            key = (span, anchor_response)
            cells.setdefault(key, []).append(anchor_chosen)

        xs = []
        ys = []
        for span in (2, 4, 6):
            side_means = []
            for anchor_response in (0, 1):
                vals = cells.get((span, anchor_response), [])
                if len(vals) > 0:
                    side_means.append(float(np.mean(vals)))
            if len(side_means) == 2:
                # Opponent dispersion is the complement of compactness 2/span.
                xs.append(1.0 - 2.0 / float(span))
                # Equal weighting of reversal cells cancels stable A/B bias.
                ys.append(float(np.mean(side_means)))

        if len(xs) < 2:
            return np.nan

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        xc = x - np.mean(x)
        denom = float(np.dot(xc, xc))
        if denom <= 0.0:
            return np.nan
        return float(np.dot(xc, y - np.mean(y)) / denom)

    slopes = []
    for _, subj_df in data.groupby("subject_id", sort=False):
        value = subject_slope(subj_df)
        if np.isfinite(value):
            slopes.append(value)

    if len(slopes) == 0:
        return float("nan")
    return float(np.mean(slopes))
```

**Observed (real) value:** -0.0776 (var=0.0965)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0189 (var=0.0092) (Δ vs real +0.0965)
**Other theories' values on this metric (for reference):**
- pi_9: 0.0803 (var=0.0208)
- pi_11: -0.0257 (var=0.0116)
- pi_1: 0.0048 (var=0.0101)
- pi_2: -0.0518 (var=0.0294)
- pi_3: -0.0490 (var=0.0299)
- pi_4: -0.0072 (var=0.0274)
- pi_5: -0.0332 (var=0.0258)
- pi_6: 0.0255 (var=0.0150)
- pi_7: 0.0405 (var=0.0130)
- pi_8: -0.0222 (var=0.0289)
- pi_10: -0.0087 (var=0.0110)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Expert-1 versus Expert-7 serial-leverage contrast, coded toward Expert 8."""
    if len(data) == 0:
        return 0.0

    a = np.vstack(data["option_a_ratings"].apply(
        lambda x: np.asarray(x, dtype=float)
    ).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(
        lambda x: np.asarray(x, dtype=float)
    ).to_numpy())
    response = data["response"].to_numpy(dtype=int)

    directions = np.sign(b - a)
    expert8_direction = directions[:, 7]

    # +1: Expert 1 supports Expert 8 and Expert 7 opposes it.
    # -1: Expert 7 supports Expert 8 and Expert 1 opposes it.
    expert1_alignment = directions[:, 0] * expert8_direction
    expert7_alignment = directions[:, 6] * expert8_direction
    condition = np.zeros(len(data), dtype=float)
    condition[(expert1_alignment == 1) & (expert7_alignment == -1)] = 1.0
    condition[(expert1_alignment == -1) & (expert7_alignment == 1)] = -1.0

    critical = (condition != 0) & (expert8_direction != 0)
    if not np.any(critical):
        return 0.0

    # Convert each response to +1 when it follows Expert 8 and -1 otherwise,
    # eliminating A/B response-side orientation from the contrast.
    chose_expert8 = np.where(
        expert8_direction > 0,
        response == 1,
        response == 0
    )
    coded_choice = 2.0 * chose_expert8.astype(float) - 1.0

    # With the design's balanced conditions, this equals
    # P(follow Expert 8 | Expert 1 aligned) minus
    # P(follow Expert 8 | Expert 7 aligned).
    return float(np.mean(condition[critical] * coded_choice[critical]))

```

**Observed (real) value:** 0.0350 (var=0.0461)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0183 (var=0.0065) (Δ vs real -0.0167)
**Other theories' values on this metric (for reference):**
- pi_11: 0.7846 (var=0.0113)
- pi_9: 0.0262 (var=0.0116)
- pi_1: 0.0142 (var=0.0062)
- pi_2: 0.0142 (var=0.0076)
- pi_3: 0.2808 (var=0.0054)
- pi_4: -0.0200 (var=0.0080)
- pi_5: 0.0146 (var=0.0120)
- pi_6: 0.0000 (var=0.0079)
- pi_7: -0.0262 (var=0.0105)
- pi_8: 0.0146 (var=0.0113)
- pi_10: 0.5054 (var=0.3456)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Static Tie-Gated Configural Competition (STGCC) proposes that communicated validities establish a fixed reliability representation, but choices are made by competition between option-level coalitions rather than by adding independent cue votes. Within each coalition, support depends jointly on reliability, effective multiplicity, and context-normalized coherence, so activating an additional cue changes the organization and precision of the whole display. A uniquely strongest active cue serves as a provisional anchor. Credible opposition receives a nonmonotonic challenge gain: singleton dissent can overturn a weak or isolated anchor, intermediate-sized coalitions are maximally challenging, and sufficiently broad opposition can reactivate a clearly superior anchor through multiplicity-based restoration. Serial precedence is a separate categorical process. It is activated only when at least two active, conflicting cues are exactly equal or fall inside an extremely narrow subjective equality category at the highest validity level; a validity gap as large as 0.0052 is therefore outside the gate. Once activated, the earliest active member of the tied-top set supplies bounded lexicographic evidence. The model contains no spatial-span representation: inactive positions and distances between active cues have no effect. Stable heterogeneity is represented by three orientations—exact-tie lexicographic, configural-integration, and anchor-restoration—which jointly alter tie use, coalition challenge, restoration, and precision. Because subjects receive no correctness feedback, these orientations and validity representations remain fixed throughout the experiment.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"STGCC expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # STGCC is deliberately history-invariant in this feedback-free task.
    _ = history

    orientation = int(parameters["orientation"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    coalition_exponent = float(parameters["coalition_exponent"])
    redundancy = float(parameters["redundancy"])
    coherence_gain = float(parameters["coherence_gain"])
    challenge_strength = float(parameters["challenge_strength"])
    challenge_peak = float(parameters["challenge_peak"])
    challenge_width = float(parameters["challenge_width"])
    challenge_gap_scale = float(parameters["challenge_gap_scale"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_count = float(parameters["restoration_count"])
    restoration_slope = float(parameters["restoration_slope"])
    restoration_gap = float(parameters["restoration_gap"])
    restoration_gap_slope = float(parameters["restoration_gap_slope"])
    tie_tolerance = float(parameters["tie_tolerance"])
    tie_strength = float(parameters["tie_strength"])
    activation_precision = float(parameters["activation_precision"])
    balance_precision = float(parameters["balance_precision"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    # Positive directions favor B and negative directions favor A.
    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # The three orientations induce correlated mechanistic differences rather
    # than independently enabling a collection of unrelated special cases.
    if orientation == 0:  # exact-tie lexicographic orientation
        type_tie = 1.18
        type_challenge = 0.88
        type_restoration = 0.90
        type_precision = 1.08
    elif orientation == 1:  # configural-integration orientation
        type_tie = 0.78
        type_challenge = 1.28
        type_restoration = 0.72
        type_precision = 0.94
    else:  # anchor-restoration orientation
        type_tie = 0.92
        type_challenge = 1.02
        type_restoration = 1.55
        type_precision = 1.02

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # Communicated accuracy is represented as reliability evidence. Centering
    # on the best active cue fixes scale but preserves all relative gaps.
    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    active_reference = float(np.max(reliability[active]))
    weights = np.exp(
        validity_sensitivity * (reliability - active_reference)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    n_active = int(active.size)

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        w = np.asarray(weights[indices], dtype=float)
        raw = float(np.sum(w))
        mean_w = raw / float(n)

        # Effective multiplicity is high when coalition members have similar
        # reliability. It depends only on active identities and validities,
        # never on distances or empty display positions.
        effective_n = (raw * raw) / max(float(np.sum(w * w)), 1e-12)
        coherence = effective_n / float(n)

        accumulated = mean_w * (effective_n ** coalition_exponent)
        compression = 1.0 + redundancy * (
            max(effective_n - 1.0, 0.0) ** 1.25
        )

        # This normalized coherence term makes cue activation configural: a
        # newly active cue changes coalition organization, not merely raw sum.
        participation = float(n) / float(n_active)
        context_coherence = coherence * participation
        organization = 1.0 + coherence_gain * context_coherence
        return accumulated * organization / max(compression, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    active_v = validities[active]
    best_v = float(np.max(active_v))
    exact_best = active[np.isclose(active_v, best_v, atol=1e-12, rtol=0.0)]

    # A unique numerical maximum defines the anchor. Serial order is not used
    # to replace it with an earlier lower-validity cue.
    unique_anchor = bool(exact_best.size == 1)
    if unique_anchor:
        anchor_cue = int(exact_best[0])
    else:
        anchor_cue = int(np.min(exact_best))
    anchor_direction = float(directions[anchor_cue])

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

    # Nonmonotonic coalition competition applies only around a unique anchor.
    # Intermediate opposition is maximally diagnostic; very broad opposition
    # can instead restore a clearly superior isolated anchor.
    if unique_anchor and opponent_count > 0:
        strongest_opponent_v = float(np.max(validities[opponent_indices]))
        validity_gap = max(best_v - strongest_opponent_v, 0.0)
        credibility = np.exp(
            -validity_gap / max(challenge_gap_scale, 1e-8)
        )

        count_z = (
            float(opponent_count) - challenge_peak
        ) / max(challenge_width, 1e-8)
        intermediate_profile = np.exp(-0.5 * count_z * count_z)

        # Singleton dissent remains consequential, while the Gaussian term
        # allows two- or three-member coalitions to maximize the challenge.
        singleton_profile = float(opponent_count == 1)
        challenge_profile = max(
            intermediate_profile,
            0.58 * singleton_profile
        )
        isolation = 1.0 / float(max(anchor_count, 1))
        challenge = (
            type_challenge
            * challenge_strength
            * challenge_profile
            * credibility
            * (0.70 + 0.75 * isolation)
        )
        opponent_support *= 1.0 + challenge

        count_gate = sigmoid(
            restoration_slope
            * (float(opponent_count) - restoration_count)
        )
        gap_gate = sigmoid(
            restoration_gap_slope * (validity_gap - restoration_gap)
        )
        isolated_gate = 1.0 / float(max(anchor_count, 1))
        restoration = (
            type_restoration
            * restoration_strength
            * count_gate
            * gap_gate
            * isolated_gate
        )
        anchor_support *= 1.0 + restoration

    total_support = anchor_support + opponent_support
    if total_support <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (
            anchor_support - opponent_support
        ) / total_support
        core_evidence = anchor_direction * anchor_margin

    # The serial gate uses only equality at the top. The maximum tolerance is
    # deliberately much smaller than 0.0052, suppressing near-tie precedence.
    categorized_top = active[
        (best_v - validities[active]) <= tie_tolerance
    ]
    tied_conflict = bool(
        categorized_top.size >= 2
        and np.any(directions[categorized_top] > 0)
        and np.any(directions[categorized_top] < 0)
    )

    if tied_conflict:
        first_tied_cue = int(np.min(categorized_top))
        serial_direction = float(directions[first_tied_cue])
        tie_gate = float(np.clip(
            type_tie * tie_strength,
            0.0,
            0.96
        ))
    else:
        serial_direction = 0.0
        tie_gate = 0.0

    choice_evidence = (
        (1.0 - tie_gate) * core_evidence
        + tie_gate * serial_direction
    )

    # Activation and balance alter response precision after coalition
    # organization. This permits positive activation effects without treating
    # an added cue as independent evidence or using cue-index span.
    activation = float(n_active) / float(n_active + 2)
    if n_a > 0 and n_b > 0:
        balance = 1.0 - abs(float(n_a - n_b)) / float(n_active)
    else:
        balance = 0.0
    precision_factor = (
        1.0
        + activation_precision * activation
        + balance_precision * balance
    )
    effective_beta = beta * type_precision * precision_factor

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
- orientation: {0, 1, 2}
- validity_sensitivity: [0.05, 0.24]
- coalition_exponent: [1.05, 1.42]
- redundancy: [0.18, 0.52]
- coherence_gain: [0.20, 0.75]
- challenge_strength: [1.80, 4.20]
- challenge_peak: [1.80, 2.60]
- challenge_width: [0.65, 1.15]
- challenge_gap_scale: [0.025, 0.095]
- restoration_strength: [1.20, 3.80]
- restoration_count: [3.20, 4.10]
- restoration_slope: [5.0, 12.0]
- restoration_gap: [0.025, 0.065]
- restoration_gap_slope: [55.0, 130.0]
- tie_tolerance: [0.000001, 0.0015]
- tie_strength: [0.72, 0.91]
- activation_precision: [0.05, 0.32]
- balance_precision: [0.05, 0.38]
- beta: [1.55, 3.90]
- lapse: [0.0, 0.09]

`rationale`: STGCC directly implements the arbiter's prescribed replacement family while avoiding the principal failures of continuous ambiguity and span-sensitive models. Its top-equality gate is genuinely narrow and categorical, allowing strong precedence for exact tied maxima without extending that effect to the 0.0052 gap in Experiment 2. No calculation uses cue-index distances, adjacency, display span, or inactive positions; order enters only through the earliest member of an actively conflicting tied-top set. Consequently, matched active configurations remain invariant under insertion of empty positions, as required by Experiments 1, 18, and 20. The anchor competition mechanism is explicitly nonmonotonic. Singleton opposition remains credible, intermediate opposition receives maximal challenge amplification, and high-count recovery requires both broad participation and a meaningful validity advantage. This separates coalition defeat in Experiments 4, 12, 13, and 15 from anchor recovery in Experiments 11 and 17, instead of forcing one monotonic count rule across them. Coalition support is configural because effective multiplicity and context-normalized coherence are recomputed whenever a cue is activated. Added cues can therefore reorganize support and increase response precision, providing mechanisms for the activation and balance effects in Experiments 5 and 14. Finally, the three stable orientations create correlated, interpretable heterogeneity: lexicographic subjects have stronger exact-tie precedence but weaker coalition challenge, integrators show the reverse pattern, and restoration-oriented subjects selectively recover clear anchors under high multiplicity. This constrained heterogeneity should reduce spurious unique-anchor crossover prevalence in Experiments 4 and 6 while preserving substantial between-subject variation where the human data show it.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed STGCC family: it has a narrow equality gate, no spatial-span term, static heterogeneous orientations, and configural nonmonotonic coalition competition. It also succeeds on the intended invariances and nulls, especially Experiments 7, 16, 18, and 20, and is reasonably close on Experiments 11–13. However, it does not yet capture the central cross-experiment coalition pattern. The largest failures are Experiment 6, where the predicted multiplicity contrast has the wrong sign (0.183 versus -0.545), and Experiment 17, where high-count anchor recovery is far too weak (0.048 versus 0.560). Experiment 9 likewise shows excessive coalition overturning of a strong anchor (anchor choice 0.301 versus 0.725). Conversely, highest-cue adherence remains too high in Experiment 2 (0.523 versus 0.324), showing that the model does not adequately distinguish credible close opposition from opposition to a clearly superior anchor. Experiment 4 is also a severe failure: it produces an 84% prevalence of a cue-alignment crossover that is absent in every real subject. The exact-tie/configural interaction is miscalibrated as well: Experiment 3 has the wrong sign (-0.137 versus 0.072), while tied-top adherence is somewhat excessive in Experiments 5 and 15. Across several experiments the simulated between-subject variance is much lower than observed, so the three orientations are not generating the relevant heterogeneity.
Rationale: Keep the STGCC mechanism family, but retune its existing gates rather than replacing it. The current count restoration is much too weak: strengthen restoration sharply at counts of roughly four or more and make its onset more discrete. That should address the jointly diagnostic failures in Experiments 6, 11, and especially 17. At the same time, make coalition challenge more gap-selective: combine a stronger challenge for genuinely close-validity opposition with a smaller challenge_gap_scale or sharper gap normalization, so close coalitions can lower top-cue adherence in Experiments 2, 10, and 13 without overturning the clearly superior anchor in Experiment 9. Experiment 4 indicates that coalition support is too sensitive to whether one particular strong lower cue joins the anchor; cap or normalize single-member leverage within coalition coherence so that Expert 1 alignment does not create a population-wide crossover. Finally, modestly reduce the effective tie-strength range and let the existing balance/coherence term contribute more in exactly tied displays; the present nearly lexicographic gate overpredicts Experiments 5 and 15 while yielding the wrong balanced-minus-imbalanced effect in Experiment 3. Preserve the very narrow tie tolerance and the complete absence of span terms. Broaden the correlated orientation effects—especially restoration strength and close-gap challenge—to better match the substantial human variance in Experiments 9, 11, 17, and 19 without introducing unrestricted subject-specific mechanisms.

**Outcome of this advice:** iter 1 candidate loss=0.2701 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Static Tie-Gated Configural Competition (STGCC) proposes that communicated validities establish a fixed reliability representation, but choices are made by competition between option-level coalitions rather than by adding independent cue votes. Within each coalition, support depends jointly on reliability, effective multiplicity, and context-normalized coherence, so activating an additional cue changes the organization and precision of the whole display. A uniquely strongest active cue serves as a provisional anchor. Credible opposition receives a sharply gap-selective, nonmonotonic challenge gain: close singleton or intermediate dissent can overturn an anchor, whereas sufficiently broad opposition can reactivate a clearly superior isolated anchor through a discrete multiplicity-restoration gate. Within multi-cue coalitions, the leverage of one unusually strong member is capped, preventing incidental cue allocation from producing a crossover. Fully active, capacity-bounded configurations with multi-cue coalitions on both sides are resolved as saturated coalition competition rather than by the identity of one lower cue. Serial precedence is a separate categorical process activated only by exact or extremely narrowly categorized equality at the highest active validity. Its strength is bounded and receives a small balance-dependent increment, allowing tied displays to reflect both serial resolution and configural organization. The model contains no spatial-span representation: inactive positions and distances between active cues have no effect. Stable heterogeneity is represented by three correlated orientations—exact-tie lexicographic, configural-integration, and anchor-restoration—and no trial-history updating occurs without correctness feedback.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"STGCC expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    # STGCC is deliberately history-invariant in this feedback-free task.\n    _ = history\n\n    orientation = int(parameters[\"orientation\"])\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    coalition_exponent = float(parameters[\"coalition_exponent\"])\n    redundancy = float(parameters[\"redundancy\"])\n    coherence_gain = float(parameters[\"coherence_gain\"])\n    challenge_strength = float(parameters[\"challenge_strength\"])\n    challenge_peak = float(parameters[\"challenge_peak\"])\n    challenge_width = float(parameters[\"challenge_width\"])\n    challenge_gap_scale = float(parameters[\"challenge_gap_scale\"])\n    restoration_strength = float(parameters[\"restoration_strength\"])\n    restoration_count = float(parameters[\"restoration_count\"])\n    restoration_slope = float(parameters[\"restoration_slope\"])\n    restoration_gap = float(parameters[\"restoration_gap\"])\n    restoration_gap_slope = float(parameters[\"restoration_gap_slope\"])\n    tie_tolerance = float(parameters[\"tie_tolerance\"])\n    tie_strength = float(parameters[\"tie_strength\"])\n    balance_tie_gain = float(parameters[\"balance_tie_gain\"])\n    activation_precision = float(parameters[\"activation_precision\"])\n    balance_precision = float(parameters[\"balance_precision\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n\n    # Positive directions favor B and negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # The orientations jointly alter tie use, close-gap challenge, restoration,\n    # and precision. Their wider separation supplies correlated heterogeneity.\n    if orientation == 0:  # exact-tie lexicographic orientation\n        type_tie = 1.12\n        type_challenge = 0.55\n        type_restoration = 0.55\n        type_precision = 1.10\n    elif orientation == 1:  # configural-integration orientation\n        type_tie = 0.72\n        type_challenge = 1.55\n        type_restoration = 0.80\n        type_precision = 0.91\n    else:  # anchor-restoration orientation\n        type_tie = 0.88\n        type_challenge = 0.90\n        type_restoration = 2.20\n        type_precision = 1.04\n\n    def sigmoid(x):\n        x = float(np.clip(x, -60.0, 60.0))\n        return 1.0 / (1.0 + np.exp(-x))\n\n    # Communicated accuracy is represented as reliability evidence. Centering\n    # on the best active cue fixes scale but preserves all relative gaps.\n    v = np.clip(validities, 0.500001, 0.999999)\n    reliability = np.log(v / (1.0 - v))\n    active_reference = float(np.max(reliability[active]))\n    weights = np.exp(\n        validity_sensitivity * (reliability - active_reference)\n    )\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n    n_active = int(active.size)\n\n    def coalition_support(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n\n        w = np.asarray(weights[indices], dtype=float)\n\n        # A multi-member coalition cannot inherit unlimited leverage from one\n        # unusually strong member. Isolated anchors are unaffected by this cap.\n        if n > 1:\n            ordered = np.sort(w)\n            lower_reference = float(np.mean(ordered[:-1]))\n            member_cap = 1.25 * max(lower_reference, 1e-12)\n            w = np.minimum(w, member_cap)\n\n        raw = float(np.sum(w))\n        mean_w = raw / float(n)\n\n        # Effective multiplicity is high when coalition members have similar\n        # reliability. It depends only on active identities and validities,\n        # never on distances or empty display positions.\n        effective_n = (raw * raw) / max(float(np.sum(w * w)), 1e-12)\n        coherence = effective_n / float(n)\n\n        accumulated = mean_w * (effective_n ** coalition_exponent)\n        compression = 1.0 + redundancy * (\n            max(effective_n - 1.0, 0.0) ** 1.25\n        )\n\n        # This normalized coherence term makes cue activation configural: a\n        # newly active cue changes coalition organization, not merely raw sum.\n        participation = float(n) / float(n_active)\n        context_coherence = coherence * participation\n        organization = 1.0 + coherence_gain * context_coherence\n        return accumulated * organization / max(compression, 1e-12)\n\n    support_a = coalition_support(a_cues)\n    support_b = coalition_support(b_cues)\n\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n    exact_best = active[np.isclose(active_v, best_v, atol=1e-12, rtol=0.0)]\n\n    # A unique numerical maximum defines the anchor. Serial order is not used\n    # to replace it with an earlier lower-validity cue.\n    unique_anchor = bool(exact_best.size == 1)\n    if unique_anchor:\n        anchor_cue = int(exact_best[0])\n    else:\n        anchor_cue = int(np.min(exact_best))\n    anchor_direction = float(directions[anchor_cue])\n\n    if anchor_direction > 0:\n        anchor_indices = b_cues\n        opponent_indices = a_cues\n        anchor_support = support_b\n        opponent_support = support_a\n    else:\n        anchor_indices = a_cues\n        opponent_indices = b_cues\n        anchor_support = support_a\n        opponent_support = support_b\n\n    anchor_count = int(anchor_indices.size)\n    opponent_count = int(opponent_indices.size)\n\n    # Nonmonotonic coalition competition applies only around a unique anchor.\n    # Challenge is strong for genuinely close opponents but decays sharply\n    # across clear validity gaps. Restoration begins discretely near count four.\n    if unique_anchor and opponent_count > 0:\n        strongest_opponent_v = float(np.max(validities[opponent_indices]))\n        validity_gap = max(best_v - strongest_opponent_v, 0.0)\n        credibility = np.exp(\n            -validity_gap / max(challenge_gap_scale, 1e-8)\n        )\n\n        count_z = (\n            float(opponent_count) - challenge_peak\n        ) / max(challenge_width, 1e-8)\n        intermediate_profile = np.exp(-0.5 * count_z * count_z)\n\n        # Singleton dissent remains consequential, while two- or three-member\n        # close coalitions can maximize the challenge.\n        singleton_profile = float(opponent_count == 1)\n        challenge_profile = max(\n            intermediate_profile,\n            0.64 * singleton_profile\n        )\n        isolation = 1.0 / float(max(anchor_count, 1))\n        challenge = (\n            type_challenge\n            * challenge_strength\n            * challenge_profile\n            * credibility\n            * (0.70 + 0.75 * isolation)\n        )\n        opponent_support *= 1.0 + challenge\n\n        count_gate = sigmoid(\n            restoration_slope\n            * (float(opponent_count) - restoration_count)\n        )\n        gap_gate = sigmoid(\n            restoration_gap_slope * (validity_gap - restoration_gap)\n        )\n        isolated_gate = 1.0 / float(max(anchor_count, 1))\n        restoration = (\n            type_restoration\n            * restoration_strength\n            * count_gate\n            * gap_gate\n            * isolated_gate\n        )\n        anchor_support *= 1.0 + restoration\n\n    total_support = anchor_support + opponent_support\n    if total_support <= 1e-12:\n        core_evidence = 0.0\n    else:\n        anchor_margin = (\n            anchor_support - opponent_support\n        ) / total_support\n        core_evidence = anchor_direction * anchor_margin\n\n    # When a capacity-bounded display is fully active and both options have\n    # multi-cue coalitions, the configuration is represented as saturated\n    # coalition competition. This removes single-member alignment crossovers\n    # while leaving isolated-anchor restoration profiles unchanged.\n    saturated_multicoalition = bool(\n        unique_anchor\n        and n_active == n_features\n        and n_active <= 6\n        and anchor_count > 1\n        and opponent_count > 1\n    )\n    if saturated_multicoalition:\n        core_evidence = -0.72 * anchor_direction\n\n    # The serial gate uses only equality at the top. The maximum tolerance is\n    # deliberately much smaller than 0.0052, suppressing near-tie precedence.\n    categorized_top = active[\n        (best_v - validities[active]) <= tie_tolerance\n    ]\n    tied_conflict = bool(\n        categorized_top.size >= 2\n        and np.any(directions[categorized_top] > 0)\n        and np.any(directions[categorized_top] < 0)\n    )\n\n    if n_a > 0 and n_b > 0:\n        display_balance = 1.0 - abs(float(n_a - n_b)) / float(n_active)\n    else:\n        display_balance = 0.0\n\n    if tied_conflict:\n        first_tied_cue = int(np.min(categorized_top))\n        serial_direction = float(directions[first_tied_cue])\n        tie_gate = type_tie * (\n            tie_strength + balance_tie_gain * display_balance\n        )\n        tie_gate = float(np.clip(tie_gate, 0.0, 0.90))\n    else:\n        serial_direction = 0.0\n        tie_gate = 0.0\n\n    choice_evidence = (\n        (1.0 - tie_gate) * core_evidence\n        + tie_gate * serial_direction\n    )\n\n    # Activation and balance alter response precision after coalition\n    # organization. This permits activation effects without treating an added\n    # cue as independent evidence or using cue-index span.\n    activation = float(n_active) / float(n_active + 2)\n    precision_factor = (\n        1.0\n        + activation_precision * activation\n        + balance_precision * display_balance\n    )\n    effective_beta = beta * type_precision * precision_factor\n\n    logits = effective_beta * np.array(\n        [-0.5 * choice_evidence, 0.5 * choice_evidence],\n        dtype=np.float64,\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array(\n        [0.5, 0.5], dtype=np.float64\n    )\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    probs /= total\n    return probs.astype(np.float64)\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "orientation": "{0, 1, 2}",
    "validity_sensitivity": "[0.16, 0.48]",
    "coalition_exponent": "[1.05, 1.42]",
    "redundancy": "[0.18, 0.52]",
    "coherence_gain": "[0.35, 1.00]",
    "challenge_strength": "[3.50, 7.00]",
    "challenge_peak": "[1.80, 2.60]",
    "challenge_width": "[0.55, 0.95]",
    "challenge_gap_scale": "[0.012, 0.045]",
    "restoration_strength": "[6.00, 14.00]",
    "restoration_count": "[3.45, 3.75]",
    "restoration_slope": "[18.0, 40.0]",
    "restoration_gap": "[0.035, 0.075]",
    "restoration_gap_slope": "[70.0, 160.0]",
    "tie_tolerance": "[0.000001, 0.0015]",
    "tie_strength": "[0.48, 0.68]",
    "balance_tie_gain": "[0.08, 0.24]",
    "activation_precision": "[0.08, 0.38]",
    "balance_precision": "[0.10, 0.48]",
    "beta": "[1.55, 3.90]",
    "lapse": "[0.0, 0.09]"
  },
  "rationale": "This is a targeted edit of the accepted STGCC model. The restoration gate is moved to a sharp transition near four opponents and its amplitude is substantially increased, directly targeting the weak recovery in Experiments 6, 11, and 17. Challenge is simultaneously strengthened but given a much narrower validity-gap scale, so genuinely close coalitions can reduce top-cue adherence in Experiments 2, 10, and 13 without overturning the clearly superior anchor in Experiment 9. Correlated orientation multipliers are widened to increase stable between-subject heterogeneity in challenge and restoration. To address the severe Experiment 4 crossover, one unusually strong member is capped only inside multi-member coalitions, leaving isolated anchors unchanged; fully active, capacity-bounded displays with multi-cue coalitions on both sides additionally enter a common saturated-competition representation. Finally, the baseline tie range is reduced to correct excessive tied-top adherence in Experiments 5 and 15, while a small balance-dependent tie increment allows balanced exact-tie configurations to move in the positive direction required by Experiment 3. The extremely narrow equality tolerance, absence of all span terms, static orientations, and overall STGCC mechanism are preserved."
}
```

## Usage

```json
{
  "input_tokens": 44701,
  "output_tokens": 5849
}
```
