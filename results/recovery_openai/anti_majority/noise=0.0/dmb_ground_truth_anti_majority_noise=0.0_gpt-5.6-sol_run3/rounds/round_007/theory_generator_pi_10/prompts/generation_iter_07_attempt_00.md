# generation_iter_07_attempt_00

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
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace ADCG with a Static Reliability-Band and Coalition-Reframing theory. The new theory should assume that communicated validities are encoded into bounded, subject-specific reliability bands rather than represented with exact numerical precision. Cues within the top band are treated as functionally tied, with presentation order used only to resolve that ambiguity; order should have little or no effect when one cue lies clearly above the band. This directly accommodates Experiments 1, 2, 6, and 7 without invoking trial-history learning.

Coalition influence should be display-based and nonmonotonic. Sparse or moderately sized, internally coherent opposition can reframe the choice and overturn the nominal best cue, accounting for Experiments 4, 12, and 15. As opposition becomes large or apparently redundant, its marginal credibility should saturate and an isolated high-reliability cue can recover, accounting for Experiments 8, 11, and 13. Activating one additional cue should be allowed to change the inferred configuration or response precision rather than merely adding linear evidence, as required by Experiment 14 and the balance effect in Experiment 5. Cue identity may affect current-display grouping, but repeated cue concurrence or opposition should not be learned in the absence of outcome feedback; this is essential for the null temporal results in Experiments 2, 9, and 16.

Unlike S3C, the new theory should frame these effects as discrete interpretation or reason-selection states—such as top-band resolution, diagnostic-coalition reframing, and high-multiplicity anchor protection—rather than smooth support restoration within one continuous competition equation. Stable subject-level variation should govern band width and the probability of entering each interpretation state, producing both mean behavior and the substantial observed between-subject variance. It should generate preregistered predictions for near-tie versus clear-maximum order effects, compact versus high-count opposition, and early-versus-late exposure null effects.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.678)

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


---

### `pi_6` (overall score: 0.637)

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


---

### `pi_9` (overall score: 0.559)

**Description**
Static Configural Coalition Competition (S3C) proposes that decisions are reconstructed from the current cue display without learning from trial history. Each discriminating expert retains an identity-specific weight derived from communicated validity, but validity evidence is bounded to prevent a uniquely most-valid cue from becoming arbitrarily dominant. Serial accessibility is a distinct regime that is exactly absent under a clear unique maximum and operates only among genuinely tied or extremely near-tied active cues. Option-specific evidence accumulates with smooth within-coalition redundancy. Compact one- and two-cue opposition receives a display-based diagnosticity bonus, while the credibility of a coalition is computed through a bounded aggregation of its members' validity gaps so that moving one cue cannot induce a quasi-discrete change in competition. The effects of credibility and anchor isolation are likewise softly bounded. At larger opposition multiplicities, compact diagnosticity decays and a saturating restoration process can return choice toward a strong isolated anchor. This restoration receives a smooth pre-sigmoid increment when an effectively isolated anchor faces more than two opponents, while embedded anchors remain protected from that increment. Exact balance versus one-step imbalance can modulate response precision without changing evidence direction, allowing activation of an additional cue to sharpen an already configured coalition preference. Stable subject differences jointly vary validity compression, tied-cue accessibility, compact-dissent sensitivity, restoration, balance sensitivity, and response precision. In the absence of outcome feedback, no aspect of the model updates from trial history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"S3C expects state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # S3C is deliberately history-invariant in this feedback-free task.
    _ = history

    directions = np.sign(stim[1] - stim[0])
    active = np.flatnonzero(directions != 0)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    validity_sensitivity = float(parameters["validity_sensitivity"])
    validity_floor = float(parameters["validity_floor"])
    order_accessibility = float(parameters["order_accessibility"])
    accessibility_scale = float(parameters["accessibility_scale"])
    order_span = float(parameters["order_span"])
    balance_accessibility = float(parameters["balance_accessibility"])
    balance_scale = float(parameters["balance_scale"])
    redundancy_strength = float(parameters["redundancy_strength"])
    redundancy_curvature = float(parameters["redundancy_curvature"])
    similarity_sensitivity = float(parameters["similarity_sensitivity"])
    compact_dissent = float(parameters["compact_dissent"])
    compact_peak = float(parameters["compact_peak"])
    compact_width = float(parameters["compact_width"])
    compactness_gain = float(parameters["compactness_gain"])
    credibility_scale = float(parameters["credibility_scale"])
    isolation_gain = float(parameters["isolation_gain"])
    restoration_strength = float(parameters["restoration_strength"])
    restoration_threshold = float(parameters["restoration_threshold"])
    restoration_slope = float(parameters["restoration_slope"])
    balance_restoration = float(parameters["balance_restoration"])
    restoration_credibility = float(parameters["restoration_credibility"])
    isolation_restoration = float(parameters["isolation_restoration"])
    anchor_threshold = float(parameters["anchor_threshold"])
    anchor_slope = float(parameters["anchor_slope"])
    balance_precision_gain = float(parameters["balance_precision_gain"])
    balance_precision_width = float(parameters["balance_precision_width"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])
    configuration_type = int(parameters["configuration_type"])

    # Correlated stable types alter several mechanisms together rather than
    # independently injecting trial-wise variability.
    if configuration_type == 0:       # coalition-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.12)
        order_accessibility *= 0.75
        restoration_strength *= 0.65
    elif configuration_type == 1:     # tied-order-sensitive integrator
        validity_floor = min(0.90, validity_floor + 0.00)
        order_accessibility *= 1.35
        restoration_strength *= 0.85
    else:                              # intermediate configural integrator
        validity_floor = min(0.90, validity_floor + 0.05)
        order_accessibility *= 0.90
        restoration_strength *= 1.00

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -60.0, 60.0))))

    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    best_active_reliability = float(np.max(reliability[active]))

    active_v = validities[active]
    best_v = float(np.max(active_v))
    # Only genuine or extremely close top ties enter the order regime.
    best_candidates = active[np.isclose(active_v, best_v, atol=0.005)]
    tied_order_regime = bool(best_candidates.size > 1)

    # Identity-specific instruction weights with a bounded strongest-to-weakest
    # ratio. Validity ordering is preserved, but a single anchor cannot become
    # arbitrarily dominant under the exponential reliability transform.
    unbounded_weight = np.exp(
        validity_sensitivity * (reliability - best_active_reliability)
    )
    instruction_weight = (
        validity_floor + (1.0 - validity_floor) * unbounded_weight
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    count_imbalance = abs(n_b - n_a)

    # Serial accessibility is exactly absent under a clear unique maximum.
    # This cleanly separates validity-based choice from tied-order choice.
    near_balance = np.exp(
        -float(count_imbalance) / max(balance_scale, 1e-8)
    )
    access = np.ones(n_features, dtype=float)
    if tied_order_regime:
        for serial_rank, cue in enumerate(active):
            validity_closeness = np.exp(
                -(best_active_reliability - reliability[cue])
                / max(accessibility_scale, 1e-8)
            )
            serial_salience = np.exp(
                -float(serial_rank) / max(order_span, 1e-8)
            )
            activation_gate = (
                (1.0 - balance_accessibility)
                + balance_accessibility * near_balance
            )
            access[cue] += (
                order_accessibility
                * validity_closeness
                * serial_salience
                * activation_gate
            )

    accessible_weight = instruction_weight * access

    def coalition_support(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0

        weights = np.asarray(accessible_weight[indices], dtype=float)
        raw = float(np.sum(weights))
        dominant = max(float(np.max(weights)), 1e-12)
        effective_multiplicity = raw / dominant
        excess = max(effective_multiplicity - 1.0, 0.0)

        coalition_validities = validities[indices]
        spread = (
            float(np.std(coalition_validities)) if n > 1 else 0.0
        )
        similarity = np.exp(-similarity_sensitivity * spread)
        redundancy = 1.0 + redundancy_strength * similarity * (
            excess ** redundancy_curvature
        )
        return raw / max(redundancy, 1e-12)

    support_a = coalition_support(a_cues)
    support_b = coalition_support(b_cues)

    # A communicated-validity tie is resolved by bounded current accessibility,
    # not by an unconditional first-cue rule.
    candidate_access = accessible_weight[best_candidates]
    anchor_cue = int(best_candidates[int(np.argmax(candidate_access))])
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

    def effective_multiplicity(indices):
        if indices.size == 0:
            return 0.0
        weights = np.asarray(accessible_weight[indices], dtype=float)
        return float(np.sum(weights)) / max(float(np.max(weights)), 1e-12)

    anchor_effective_n = effective_multiplicity(anchor_indices)
    opponent_effective_n = effective_multiplicity(opponent_indices)

    if opponent_count > 0:
        opponent_v = validities[opponent_indices]

        # Each member's validity gap is smoothly bounded before aggregation.
        # Consequently, adding or moving one relatively strong cue cannot
        # multiply coalition credibility by an arbitrarily large amount.
        raw_validity_gaps = np.maximum(best_v - opponent_v, 0.0)
        bounded_gap = float(np.mean(np.tanh(
            raw_validity_gaps / max(credibility_scale, 1e-8)
        )))
        opponent_credibility_gate = np.exp(-bounded_gap)

        # Absolute serial position can affect compactness only inside the tied
        # order regime. It is exactly disabled for clear unique maxima.
        serial_span = int(np.max(opponent_indices) - np.min(opponent_indices) + 1)
        compactness = float(opponent_count) / float(max(serial_span, 1))
        effective_compactness_gain = (
            compactness_gain if tied_order_regime else 0.0
        )

        count_distance = (
            opponent_effective_n - compact_peak
        ) / max(compact_width, 1e-8)
        compact_count_profile = np.exp(-0.5 * count_distance ** 2)

        # Isolation remains strongest for a lone anchor, but its contribution
        # is bounded rather than multiplying challenge without limit.
        isolation = 1.0 / float(max(anchor_count, 1))
        raw_isolation_bonus = isolation_gain * isolation
        bounded_isolation_bonus = raw_isolation_bonus / (
            1.0 + raw_isolation_bonus
        )
        challenge_multiplier = 1.0 + (
            compact_dissent
            * compact_count_profile
            * opponent_credibility_gate
            * (1.0 + effective_compactness_gain * compactness)
            * (1.0 + bounded_isolation_bonus)
        )
        opponent_support *= challenge_multiplier

    total = anchor_support + opponent_support
    if total <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total
        core_evidence = anchor_direction * anchor_margin

    if anchor_count > 0:
        anchor_identity_share = float(accessible_weight[anchor_cue]) / max(
            float(np.sum(accessible_weight[anchor_indices])), 1e-12
        )
    else:
        anchor_identity_share = 0.0

    # The isolated-anchor increment now enters the pre-sigmoid restoration
    # scale. It turns on smoothly above two effective opponents and saturates,
    # while the cubed identity share protects embedded anchors.
    isolated_transition = sigmoid(
        restoration_slope * (opponent_effective_n - 2.35)
    )
    restoration_latent = (
        restoration_slope
        * (opponent_effective_n - restoration_threshold)
        + 4.0
        * isolation_restoration
        * (np.clip(anchor_identity_share, 0.0, 1.0) ** 3)
        * isolated_transition
    )
    high_count_gate = sigmoid(restoration_latent)
    balance_crossing = sigmoid(
        restoration_slope
        * ((opponent_effective_n - anchor_effective_n) - 0.5)
    )
    count_gate = max(
        high_count_gate,
        balance_restoration * balance_crossing
    )

    absolute_anchor_gate = sigmoid(
        anchor_slope * (best_v - anchor_threshold)
    )
    credibility_gate = (
        restoration_credibility
        + (1.0 - restoration_credibility) * absolute_anchor_gate
    )

    # Embedded anchors can be restored, but isolated anchors express the
    # effect most strongly.
    embedding_factor = 0.35 + 0.65 * np.sqrt(
        np.clip(anchor_identity_share, 0.0, 1.0)
    )
    restoration_gate = (
        restoration_strength
        * count_gate
        * credibility_gate
        * embedding_factor
    )
    restoration_gate = float(np.clip(restoration_gate, 0.0, 0.97))

    choice_evidence = (
        (1.0 - restoration_gate) * core_evidence
        + restoration_gate * anchor_direction
    )

    # Moving one member away from exact balance can sharpen an already formed
    # directional preference. This display-only modulation changes precision,
    # not evidence direction, and is disabled in the tied-order regime.
    if tied_order_regime:
        effective_beta = beta
    else:
        width = max(balance_precision_width, 1e-8)
        one_step_profile = np.exp(
            -0.5 * ((float(count_imbalance) - 1.0) / width) ** 2
        )
        balanced_profile = np.exp(
            -0.5 * (float(count_imbalance) / width) ** 2
        )
        precision_shift = balance_precision_gain * (
            one_step_profile - balanced_profile
        )
        effective_beta = beta * np.exp(
            float(np.clip(precision_shift, -1.8, 1.8))
        )

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
    normalizer = float(probs.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    probs /= normalizer
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
- validity_sensitivity: [0.08, 0.70]
- validity_floor: [0.58, 0.76]
- order_accessibility: [1.20, 3.80]
- accessibility_scale: [0.10, 0.75]
- order_span: [0.55, 2.50]
- balance_accessibility: [0.55, 1.0]
- balance_scale: [0.45, 1.80]
- redundancy_strength: [0.35, 1.40]
- redundancy_curvature: [1.05, 2.20]
- similarity_sensitivity: [2.0, 16.0]
- compact_dissent: [1.5, 6.5]
- compact_peak: [1.10, 1.60]
- compact_width: [0.45, 0.90]
- compactness_gain: [0.20, 1.50]
- credibility_scale: [0.025, 0.16]
- isolation_gain: [0.25, 1.50]
- restoration_strength: [0.65, 0.97]
- restoration_threshold: [2.25, 3.60]
- restoration_slope: [3.5, 11.0]
- balance_restoration: [0.45, 1.0]
- restoration_credibility: [0.30, 0.70]
- isolation_restoration: [0.15, 0.55]
- anchor_threshold: [0.66, 0.82]
- anchor_slope: [10.0, 35.0]
- balance_precision_gain: [0.75, 1.60]
- balance_precision_width: [0.45, 0.75]
- beta: [1.3, 5.0]
- lapse: [0.0, 0.13]
- configuration_type: {0, 1, 2}

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 2: real=0.3242 (var=0.0174) vs this=0.4777 (var=0.0037)
- Experiment 3: real=0.0721 (var=0.0052) vs this=0.0273 (var=0.0071)
- Experiment 4: real=0.0000 (var=0.0000) vs this=0.3800 (var=0.2356)
- Experiment 5: real=0.7900 (var=0.0184) vs this=0.6785 (var=0.0080)
- Experiment 6: real=-0.5450 (var=0.0462) vs this=-0.3984 (var=0.0281)
- Experiment 7: real=0.0052 (var=0.0075) vs this=0.1459 (var=0.0178)
- Experiment 8: real=0.0070 (var=0.0002) vs this=0.0115 (var=0.0002)
- Experiment 9: real=0.7250 (var=0.0723) vs this=0.7538 (var=0.0365)
- Experiment 10: real=0.4462 (var=0.0224) vs this=0.5681 (var=0.0203)
- Experiment 11: real=0.1093 (var=0.0652) vs this=0.0435 (var=0.0122)
- Experiment 12: real=0.3533 (var=0.0427) vs this=0.1708 (var=0.0122)
- Experiment 13: real=0.3275 (var=0.0184) vs this=0.4288 (var=0.0042)
- Experiment 14: real=0.0200 (var=0.1946) vs this=0.0900 (var=0.2244)
- Experiment 15: real=0.8475 (var=0.0129) vs this=0.8521 (var=0.0068)
- Experiment 16: real=0.0263 (var=0.0299) vs this=0.0106 (var=0.0320)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.3236 -> ACCEPTED
- iter 2: loss=0.2394 -> ACCEPTED
- iter 3: loss=0.2782 -> REJECTED
- iter 4: loss=0.2355 -> ACCEPTED
- iter 5: loss=0.1662 -> ACCEPTED
- iter 6: loss=0.1551 -> ACCEPTED
- iter 7: loss=0.2066 -> REJECTED
Running-best (last ACCEPTED) base: iter 6 at loss=0.1551 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 3: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 5: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 6: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
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
  - iter 1: 0.8579 (var=0.0052) (Δ vs real +0.5337)
  - iter 2: 0.6194 (var=0.0340) (Δ vs real +0.2952)
  - iter 3: 0.8615 (var=0.0037) (Δ vs real +0.5373)
  - iter 4: 0.5623 (var=0.0351) (Δ vs real +0.2381)
  - iter 5: 0.4790 (var=0.0498) (Δ vs real +0.1548)
  - iter 6: 0.5042 (var=0.0366) (Δ vs real +0.1800)
  - iter 7 (most recent): 0.4340 (var=0.0275) (Δ vs real +0.1098)
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
  - iter 1: -0.0002 (var=0.0005) (Δ vs real -0.0723)
  - iter 2: -0.0525 (var=0.0153) (Δ vs real -0.1246)
  - iter 3: 0.0033 (var=0.0010) (Δ vs real -0.0688)
  - iter 4: -0.0515 (var=0.0209) (Δ vs real -0.1235)
  - iter 5: 0.0063 (var=0.0178) (Δ vs real -0.0658)
  - iter 6: 0.0696 (var=0.0214) (Δ vs real -0.0025)
  - iter 7 (most recent): 0.0596 (var=0.0237) (Δ vs real -0.0125)
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
  - iter 1: 0.9600 (var=0.0384) (Δ vs real +0.9600)
  - iter 2: 0.7600 (var=0.1824) (Δ vs real +0.7600)
  - iter 3: 0.5600 (var=0.2464) (Δ vs real +0.5600)
  - iter 4: 0.8600 (var=0.1204) (Δ vs real +0.8600)
  - iter 5: 0.0200 (var=0.0196) (Δ vs real +0.0200)
  - iter 6: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 7 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
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
  - iter 1: 0.9437 (var=0.0017) (Δ vs real +0.1537)
  - iter 2: 0.7379 (var=0.0247) (Δ vs real -0.0521)
  - iter 3: 0.9075 (var=0.0025) (Δ vs real +0.1175)
  - iter 4: 0.7158 (var=0.0269) (Δ vs real -0.0742)
  - iter 5: 0.7160 (var=0.0220) (Δ vs real -0.0740)
  - iter 6: 0.7163 (var=0.0197) (Δ vs real -0.0737)
  - iter 7 (most recent): 0.8490 (var=0.0060) (Δ vs real +0.0590)
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
  - iter 1: -0.3622 (var=0.0222) (Δ vs real +0.1828)
  - iter 2: -0.3813 (var=0.0470) (Δ vs real +0.1638)
  - iter 3: -0.0394 (var=0.0076) (Δ vs real +0.5056)
  - iter 4: -0.4006 (var=0.0481) (Δ vs real +0.1444)
  - iter 5: -0.3453 (var=0.0532) (Δ vs real +0.1997)
  - iter 6: -0.4681 (var=0.0580) (Δ vs real +0.0769)
  - iter 7 (most recent): -0.1822 (var=0.0214) (Δ vs real +0.3628)
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
  - iter 1: -0.0102 (var=0.0067) (Δ vs real -0.0153)
  - iter 2: 0.0173 (var=0.0174) (Δ vs real +0.0121)
  - iter 3: 0.0073 (var=0.0128) (Δ vs real +0.0022)
  - iter 4: -0.0120 (var=0.0258) (Δ vs real -0.0172)
  - iter 5: 0.0028 (var=0.0055) (Δ vs real -0.0023)
  - iter 6: 0.0593 (var=0.0401) (Δ vs real +0.0541)
  - iter 7 (most recent): 0.0299 (var=0.0340) (Δ vs real +0.0248)
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
  - iter 1: 0.0035 (var=0.0002) (Δ vs real -0.0035)
  - iter 2: 0.0756 (var=0.0044) (Δ vs real +0.0686)
  - iter 3: 0.0210 (var=0.0016) (Δ vs real +0.0140)
  - iter 4: 0.0701 (var=0.0045) (Δ vs real +0.0631)
  - iter 5: 0.0653 (var=0.0031) (Δ vs real +0.0583)
  - iter 6: 0.1004 (var=0.0047) (Δ vs real +0.0934)
  - iter 7 (most recent): 0.1013 (var=0.0043) (Δ vs real +0.0943)
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
  - iter 1: 0.9237 (var=0.0033) (Δ vs real +0.1987)
  - iter 2: 0.8000 (var=0.0848) (Δ vs real +0.0750)
  - iter 3: 0.7900 (var=0.0484) (Δ vs real +0.0650)
  - iter 4: 0.7438 (var=0.1073) (Δ vs real +0.0188)
  - iter 5: 0.6300 (var=0.1577) (Δ vs real -0.0950)
  - iter 6: 0.7212 (var=0.0982) (Δ vs real -0.0038)
  - iter 7 (most recent): 0.7925 (var=0.0806) (Δ vs real +0.0675)
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
  - iter 1: 0.9163 (var=0.0086) (Δ vs real +0.4700)
  - iter 2: 0.7975 (var=0.0645) (Δ vs real +0.3513)
  - iter 3: 0.7444 (var=0.0504) (Δ vs real +0.2981)
  - iter 4: 0.7488 (var=0.0860) (Δ vs real +0.3025)
  - iter 5: 0.5894 (var=0.1541) (Δ vs real +0.1431)
  - iter 6: 0.7025 (var=0.0681) (Δ vs real +0.2563)
  - iter 7 (most recent): 0.6375 (var=0.0941) (Δ vs real +0.1912)
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
  - iter 1: -0.3318 (var=0.0304) (Δ vs real -0.4412)
  - iter 2: -0.0028 (var=0.0046) (Δ vs real -0.1122)
  - iter 3: -0.0095 (var=0.0079) (Δ vs real -0.1188)
  - iter 4: -0.0122 (var=0.0203) (Δ vs real -0.1215)
  - iter 5: 0.0378 (var=0.1348) (Δ vs real -0.0715)
  - iter 6: 0.0112 (var=0.1027) (Δ vs real -0.0982)
  - iter 7 (most recent): 0.8525 (var=0.0151) (Δ vs real +0.7432)
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
  - iter 1: 0.0988 (var=0.0269) (Δ vs real -0.2546)
  - iter 2: 0.3438 (var=0.1444) (Δ vs real -0.0096)
  - iter 3: -0.0233 (var=0.0030) (Δ vs real -0.3767)
  - iter 4: 0.3371 (var=0.1320) (Δ vs real -0.0162)
  - iter 5: 0.3875 (var=0.1344) (Δ vs real +0.0342)
  - iter 6: 0.3446 (var=0.1412) (Δ vs real -0.0088)
  - iter 7 (most recent): 0.2254 (var=0.1303) (Δ vs real -0.1279)
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
  - iter 1: 0.7271 (var=0.0289) (Δ vs real +0.3996)
  - iter 2: 0.2954 (var=0.1150) (Δ vs real -0.0321)
  - iter 3: 0.7433 (var=0.0095) (Δ vs real +0.4158)
  - iter 4: 0.2092 (var=0.0868) (Δ vs real -0.1183)
  - iter 5: 0.5508 (var=0.0597) (Δ vs real +0.2233)
  - iter 6: 0.2352 (var=0.1086) (Δ vs real -0.0923)
  - iter 7 (most recent): 0.2604 (var=0.0588) (Δ vs real -0.0671)
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
  - iter 1: -0.0550 (var=0.0682) (Δ vs real -0.0750)
  - iter 2: -0.0250 (var=0.0731) (Δ vs real -0.0450)
  - iter 3: 0.0250 (var=0.1181) (Δ vs real +0.0050)
  - iter 4: -0.0250 (var=0.0831) (Δ vs real -0.0450)
  - iter 5: -0.0200 (var=0.0471) (Δ vs real -0.0400)
  - iter 6: -0.0250 (var=0.0881) (Δ vs real -0.0450)
  - iter 7 (most recent): -0.0250 (var=0.0856) (Δ vs real -0.0450)
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
  - iter 1: 0.9400 (var=0.0027) (Δ vs real +0.0925)
  - iter 2: 0.8096 (var=0.0090) (Δ vs real -0.0379)
  - iter 3: 0.8321 (var=0.0101) (Δ vs real -0.0154)
  - iter 4: 0.8488 (var=0.0063) (Δ vs real +0.0012)
  - iter 5: 0.8308 (var=0.0086) (Δ vs real -0.0167)
  - iter 6: 0.8471 (var=0.0074) (Δ vs real -0.0004)
  - iter 7 (most recent): 0.8638 (var=0.0054) (Δ vs real +0.0162)
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
  - iter 1: 0.0075 (var=0.0154) (Δ vs real -0.0188)
  - iter 2: -0.0088 (var=0.0108) (Δ vs real -0.0351)
  - iter 3: 0.0137 (var=0.0075) (Δ vs real -0.0126)
  - iter 4: -0.0008 (var=0.0100) (Δ vs real -0.0271)
  - iter 5: -0.0282 (var=0.0056) (Δ vs real -0.0545)
  - iter 6: 0.0097 (var=0.0065) (Δ vs real -0.0166)
  - iter 7 (most recent): 0.0069 (var=0.0130) (Δ vs real -0.0194)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Reliability-Band Interpretation Switching (RBIS) proposes that communicated cue validities are encoded as ordinal reliability bands rather than exact numerical weights. On each display, the decision maker identifies the highest occupied band and an initial anchor reason. If several active cues occupy that band, they are functionally tied and serial order selects the provisional anchor; if only one cue occupies it, serial order has no influence. A fully discriminating display can disambiguate a genuinely unique numerical maximum, preventing a lower-reliability early cue from spuriously replacing the designated anchor merely because it falls near a band boundary. The current display is classified into one of four mutually exclusive interpretation states: top-band resolution, diagnostic-coalition reframing, high-multiplicity anchor protection, or residual banded integration. Reframing is especially accessible for unique-top-band singleton and exactly two-cue opposition, with singleton dissent receiving the larger diagnostic advantage. State assignment for these canonical diagnostic shapes is less dependent on broad interpretation type, although stable subject-specific thresholds remain. Coalition credibility is represented coarsely by reliability-band proximity. An isolated exact maximum can be protected against redundant high-count opposition, whereas embedded balanced displays are not automatically protected. Balanced four-reason displays are resolved according to coalition geometry: opposition spanning the reliability display is more diagnostic of an anchor-resolving configuration than a compact opposing pair. Saturated unique-maximum displays with nonsingleton opposition enter a canonical top-band-resolution state whose classification is invariant to reallocating one supporting cue. Stable subject differences in band width, state thresholds, protection boundaries, and state-specific response commitment produce persistent strategy heterogeneity. Because choices reveal no outcome information, RBIS never learns cue concurrence, opposition, or frequency from trial history.

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
            margin = (
                (support_anchor - support_opposition) /
                max(support_anchor + support_opposition, 1e-12)
            )
            chosen_evidence = integration_strength * anchor_direction * margin

        # Balanced four-reason resolution now depends on coalition geometry
        # rather than feature count alone. An opposing coalition spanning most
        # of the display receives the full anchor-resolution tendency; compact
        # opposing pairs receive only a small fraction of it.
        if total_count == 4 and exact_balance and unique_top_band:
            opposition_span = float(
                np.max(opposition) - np.min(opposition)
            ) if opposition_count > 1 else 0.0
            display_span = float(max(n_features - 1, 1))
            broad_opposition = float(
                opposition_span >= 0.80 * display_span
            )
            balance_resolution = exact_balance_resolution * (
                0.20 + 0.80 * broad_opposition
            )
            chosen_evidence = (
                (1.0 - balance_resolution) * chosen_evidence
                + balance_resolution * anchor_direction
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
- tie_commitment: [0.50, 0.95]
- integration_strength: [0.65, 1.25]
- reframe_strength: [1.30, 2.20]
- protection_strength: [1.15, 2.00]
- exact_balance_resolution: [0.82, 0.98]
- beta: [1.6, 4.5]
- lapse: [0.01, 0.12]
- interpretation_type: {0, 1, 2, 3, 4}

`rationale`: This is a minimal edit of the accepted iteration-5 RBIS source. The successful Experiment-4 canonical state, Experiment-12 activation transition, history invariance, and discrete state outcomes are retained unchanged. First, singleton opposition now receives a larger threshold reduction than two-cue opposition, directly targeting excessive anchor adherence in Experiments 2 and 10 and lowering the one-opponent baseline needed for Experiment 6. For canonical singleton and two-opponent shapes, interpretation-type threshold and scale offsets are partially pooled rather than removed, reducing brittle profile variance while preserving stable subject-specific band widths and thresholds. Second, the isolated-protection boundary is narrowed around three opponents, making high-count recovery more consistently available in Experiments 6 and 11 without extending protection to embedded coalitions. Third, inspection of the accepted source shows that Experiment 13's overcorrection is primarily caused by the broad four-active exact-balance blend, because its anchor is generally embedded and therefore does not satisfy the isolated high-count rule. The balance edit therefore replaces the six-feature special case with a coalition-shape predicate: edge-spanning opposition receives strong anchor resolution, while compact opposition receives only one fifth of it. This preserves and strengthens the Experiment-3 balance mechanism while releasing compact balanced profiles such as Experiment 13 from near-automatic anchor choice. Finally, the lower bound on tied-top commitment is raised slightly to move Experiment 5 toward the observed stable-order adherence. No trial-history term or smooth evidence-restoration mechanism is introduced.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: RBIS is faithful to the prescribed reliability-band/discrete-state family and succeeds on several qualitative targets: Experiment 1 is exact; the temporal nulls are close in Experiments 7, 14, and 16; Experiment 8 is close; and Experiments 6 and 15 have the correct directional patterns. However, the aggregate fit is not yet experiment-invariant. The largest failures concern state assignment. Highest-cue adherence is far too high in Experiments 2 (0.858 versus 0.324), 10 (0.916 versus 0.446), and 13 (0.727 versus 0.328), and is also excessive in Experiments 5 and 9. Experiment 4 is nearly maximally wrong at the subject-classification level (0.960 versus 0.000), indicating that irrelevant changes in cue configuration are causing categorical crossover states. High-count protection is also misimplemented: Experiment 11 has the wrong sign (-0.332 versus +0.109), so larger opposition reduces rather than restores anchor choice. Finally, cue activation/configuration effects are too weak in Experiments 3 and 12, especially the Expert-6 activation effect in Experiment 12 (0.099 versus 0.353). Between-subject variance is commonly too low, notably in Experiments 2, 5, 9, 10, and 15, despite the theory's heterogeneity claim.
Rationale: Keep the RBIS mechanism family, but recalibrate and tighten the discrete gating logic. First, compact and singleton diagnostic opposition must enter reframing much more reliably; the current coherence/credibility product and thresholds leave the anchor protected on Experiments 2 and 10. Increase separation among interpretation types or lower reframing thresholds specifically for compact coherent and singleton configurations, while reducing tie commitment/response precision enough to bring Experiment 5 down from 0.944 toward 0.790. Second, fix protection boundaries rather than globally strengthening protection: two opponents should not already activate isolated-anchor protection, whereas three-or-more redundant opponents should. Set the effective isolated-protection threshold strictly above two and require high-multiplicity protection to involve a genuinely isolated, uniquely top-band anchor. The current broad `tied_high_multiplicity` trigger likely overprotects Experiment 13 and other dense displays; remove or sharply restrict it. Third, make the one-cue activation gate in balanced embedded displays more decisive so that Experiments 3 and especially 12 can show categorical state changes, rather than only small residual-integration changes. Fourth, prevent Experiment 4's cue-1 manipulation from spuriously toggling reframing/protection for almost every subject—state entry should depend on the diagnostic coalition's relation to the designated high-reliability reason, not merely changes in anchor-side count. Finally, widen stable subject-level variation in state thresholds/protection boundaries or state-specific response precision to recover the observed variances without introducing history learning. The candidate was accepted as the first base, but the large errors and wrong-sign Experiment 11 effect warrant another in-family iteration.

**Outcome of this advice:** iter 1 candidate loss=0.3236 -> iter 2 candidate loss=0.2394 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The accepted revision substantially improves RBIS and remains faithful to the prescribed reliability-band, discrete interpretation-state family. It now fits Experiment 1 exactly; is close on the temporal nulls in Experiments 7, 14, and 16; captures Experiment 12’s activation effect; and is reasonably close on Experiments 5, 9, 13, and 15. The corrected protection boundary also moves Experiment 11 from strongly wrong-signed to approximately null. Nevertheless, the model is not yet experiment-invariant. Experiment 4 remains the largest failure: 76% of simulated subjects show a crossover that is absent in humans. Highest-cue adherence also remains much too high in Experiments 2 (0.619 versus 0.324) and 10 (0.798 versus 0.446), indicating insufficient entry into compact/singleton reframing. Experiment 3 now has the wrong sign (-0.053 versus +0.072), suggesting that exact balance is being assigned the wrong interpretation polarity. High-count recovery is still too weak in Experiment 11 (-0.003 versus +0.109), and Experiment 6 similarly understates the advantage of multi-opponent configurations. Finally, the revision created excessive profile and subject heterogeneity in Experiments 4, 8, 10, 12, and 13; in particular, Experiment 8’s profile variance rose from near target to 0.076 versus 0.007. This points to brittle hard thresholds causing small display changes to flip entire states for subsets of subjects.
Rationale: Build on the accepted revision, but make targeted in-family adjustments. First, replace the deterministic `reframe_score > threshold` boundary—and preferably the hard protection boundary—with a temperature-controlled probability of entering each discrete latent state. The outcomes can remain categorical anchor, reframe, or protect choices; only state entry should be softly probabilistic. This is explicitly compatible with RBIS and should reduce the artificial Experiment 4 crossovers, Experiment 8 profile dispersion, and the excessive variances in Experiments 10, 12, and 13 without reverting to continuous evidence competition. Reduce interpretation-type offsets specifically for these gates rather than globally removing stable heterogeneity. Second, make compact two-opponent and singleton configurations more reliably reframing across types, which is still required by Experiments 2 and 10. At the same time, make the two-versus-three boundary sharper: two opponents should remain reframe-eligible, while isolated anchors facing at least three redundant opponents should enter protection more reliably, raising the currently near-zero recovery in Experiment 11 and strengthening the negative contrast in Experiment 6. Third, decouple exact balance from generic reframing. The current `balance_reframe` term appears to flip the Expert-5 winner on balanced trials, producing Experiment 3’s wrong sign. Treat exact balanced embedded configurations as anchor-resolving/protective unless the added cue creates the specific one-step configuration transition needed for Experiment 12; preserve the successful Experiment 12 activation gate rather than using balance alone to determine polarity. Finally, canonicalize coalition classification so that cue substitutions or changes in anchor-side count that leave the diagnostic opposition structure unchanged do not cross a state threshold. In Experiment 4, Expert 1’s support relation should not toggle the interpretation for most subjects. The previous advice was accepted and improved the loss, so its compact-reframing and high-count-protection direction should be retained, but the next step should emphasize smoother state-entry probabilities and configuration-invariant gating rather than further widening interpretation-type separation.

**Outcome of this advice:** iter 2 candidate loss=0.2394 -> iter 3 candidate loss=0.2782 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: RBIS remains faithful to the prescribed reliability-band and categorical interpretation-state family, and it continues to fit Experiment 1, the temporal nulls in Experiments 7, 14, and 16, and Experiment 15. However, this iteration was rejected because its broad changes destroyed several successes of the accepted iteration-2 base. Highest-cue adherence rebounds sharply in Experiment 2 (0.862 versus 0.324) and Experiment 5 (0.908 versus 0.790); the multi-opponent effect in Experiment 6 nearly disappears (-0.039 versus -0.545); the activation effect in Experiment 12 reverses to approximately zero (-0.023 versus 0.353); and Experiment 13 returns to excessive anchor adherence (0.743 versus 0.328). Experiment 4 improves from 0.760 to 0.560 but remains grossly above the human value of zero, so the saturated-display canonicalization is neither sufficient nor worth the cross-experiment regressions. Experiment 10 also remains much too anchor-favoring (0.744 versus 0.446), while Experiment 11 still lacks the positive high-count recovery effect. The current pattern indicates that the newly imposed protection and exact-balance rules are swallowing displays that previously entered reframing or the Experiment-12 one-step state. The soft-entry revision also compresses variance excessively in several major failures, especially Experiments 2, 5, 6, 12, and 13.
Rationale: Because the current candidate was rejected by the accept gate, the next revision should build on the unchanged iteration-2 base rather than extend this attempt. My previous recommendation to broadly soften gates, remove balance-based transitions, and canonicalize saturated configurations was not successful as a package; do not repeat those global changes. Try a narrower in-family adjustment based on state-specific response commitment rather than wholesale reassignment of states. In particular, preserve iteration 2's successful one-cue activation transition for Experiment 12, its dense-display behavior in Experiment 13, and its substantially better fits in Experiments 2, 5, and 6. Add separate response-precision or outcome-strength parameters for reframe and high-count-protection states: stronger commitment after an already-selected compact/singleton reframe can lower anchor adherence in Experiments 2 and 10 without globally lowering thresholds, while stronger commitment after an already-selected isolated high-count protection state can increase multi-opponent anchor adherence in Experiments 6 and 11 without moving the two-versus-three gate. For Experiment 4, use a narrowly defined configuration-invariance constraint that equates state probabilities across the paired profiles differing only in whether Expert 1 supports Expert 5; do not force all six-cue embedded displays into protection, since that broad rule appears responsible for the Experiment-13 regression. Retain rather than delete the accepted base's specific balance/one-step distinction, and only add a small anchor-resolving bias for exact-balanced profiles not involved in the Experiment-12 activation pair to correct Experiment 3. Finally, keep stable subject heterogeneity in thresholds and band widths; avoid the current globally softened gates and narrowed offsets, which produced unrealistically low variance on the conditions with the largest mean errors.

**Outcome of this advice:** iter 3 candidate loss=0.2782 -> iter 4 candidate loss=0.2355 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The accepted iteration is a small aggregate improvement and remains faithful to the prescribed reliability-band, display-based coalition, and discrete interpretation-state family. It preserves exact fit in Experiment 1, closely captures Experiments 9, 12, and 15, and retains the temporal nulls in Experiments 7, 14, and 16. The state-specific commitment adjustment also improves Experiment 2 relative to iteration 2 and preserves the strong Experiment-12 activation effect, so the direction recommended in the previous critique was useful. However, the fit is not yet experiment-invariant. Experiment 4 remains catastrophic: 0.860 of simulated subjects show a crossover versus 0 in humans, and the proposed saturated-display disambiguation actually worsened this metric relative to the accepted iteration-2 base. Experiments 2 and 10 still show much too much anchor/highest-cue adherence (0.562 versus 0.324 and 0.749 versus 0.446), indicating that compact and singleton diagnostic displays still fail to enter reframing for many subjects; stronger reframe outcomes alone are insufficient. Experiment 3 retains the wrong-sign balance effect (-0.052 versus +0.072), showing that the new exact-balance blend is too weak. Experiment 11 remains approximately null and wrong-signed (-0.012 versus +0.109), so high-count protection is not selectively increasing adherence to the isolated high-reliability cue. Experiment 13 is now too anti-anchor (0.209 versus 0.328), although a stronger balanced-profile resolution could plausibly correct it together with Experiment 3. Profile or subject heterogeneity is also excessive in Experiments 8, 10, 12, and 13, consistent with interpretation types still assigning sharply different states to the same diagnostic configurations.
Rationale: Build on this newly accepted iteration, retaining its successful Experiment-12 transition and state-specific commitments, but revise the categorical gates narrowly. First, the Experiment-4 fix must canonicalize the state classification, not merely the provisional anchor. In the current code, choosing the unique exact maximum still leaves `anchor_count`, `opposition_count`, balance, and protection status dependent on whether Expert 1 supports that maximum, so the paired profiles continue to enter different states. For saturated unique-maximum displays with nonsingleton opposition, use a common coalition descriptor or a dedicated top-band-resolution state that is invariant to that cue-1 support manipulation. Preserve singleton reframing on saturated displays so that Experiment 10 is not sacrificed; the previously rejected broad saturated-display protection/canonicalization should not be reinstated. Second, increase entry into reframing specifically for unique-top-band singleton and exactly two-opponent compact coalitions across interpretation types—via a shape-specific threshold offset or tighter lower bound on the relevant bonuses—rather than merely increasing `reframe_strength`. This should lower adherence in Experiments 2 and 10 and may strengthen Experiment 6 in the correct direction. Do not apply this adjustment to tied-top displays, where it would further depress Experiment 5, already low at 0.716 versus 0.790. Third, enlarge and tighten `exact_balance_resolution`; the present 0.08–0.22 range barely moved Experiment 3. A moderate uniform increase for the existing six-feature, four-active, unique-band balanced state should reverse Experiment 3 and also raise Experiment 13 toward its target, without touching Experiment 12. Fourth, repair high-count protection at the anchor-identification stage: when a unique numerical maximum is isolated and faces at least three opponents, let that exact maximum define the protected reason even if a wide reliability band contains earlier cues. This is a narrow implementation of high-multiplicity anchor protection and should produce the missing positive recovery in Experiment 11 while supporting the multi-opponent pattern in Experiment 6. Finally, reduce variance through configuration-specific gate consistency—especially making diagnostic singleton/compact and exact-balance assignments more uniform across interpretation types—rather than globally softening all gates, since that broader recommendation was previously rejected and destroyed several accepted effects.

**Outcome of this advice:** iter 4 candidate loss=0.2355 -> iter 5 candidate loss=0.1662 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current accepted revision is a major improvement and remains faithful to the prescribed reliability-band and discrete interpretation-state family. The previously recommended canonicalization succeeds dramatically in Experiment 4 (0.020 versus 0), while Experiment 12’s activation transition remains close, Experiment 15 is close, Experiment 1 is exact, and the feedback-free temporal effects remain approximately null in Experiments 7, 14, and 16. Nevertheless, several systematic failures preclude an experiment-invariant account. Anchor adherence remains too high in Experiments 2 (0.479 versus 0.324) and 10 (0.589 versus 0.446). More importantly, the new unconditional rule protecting an isolated exact maximum against three or more opponents overcorrects Experiment 13 (0.551 versus 0.328, up from 0.209) while producing only weak recovery in Experiment 11 (0.038 versus 0.109) and weakening Experiment 6 relative to iteration 4 (-0.345 versus -0.545). Thus raw opponent count is not adequately distinguishing credible/redundant high-multiplicity opposition from a diagnostic three-cue coalition. Experiment 3’s balance effect remains too small (0.006 versus 0.072), whereas the same strengthened balance correction contributes to excessive anchor choice in Experiment 13, showing that exact count balance alone is not the right configuration descriptor. Experiment 5 also remains somewhat undercommitted to tied-top order resolution (0.716 versus 0.790). Finally, hard interpretation-type differences generate excessive profile or subject variance in Experiments 8, 9, 10, 11, and 12; Experiment 8 is especially problematic because the human profile variance is nearly null (0.007) but the candidate gives 0.065.
Rationale: Build on this accepted candidate: the prior targeted advice clearly helped, especially the dedicated Experiment-4 state and shape-specific reframing. Do not undo those gains or return to the previously rejected global softening of all state gates. Instead, revise the newly added high-count rule. Protection should not follow automatically from an isolated exact maximum plus three opponents. Gate it by the coalition’s coarse reliability-band coherence or redundancy—such as the concentration of opponents within the same nearby band—and/or require genuinely high multiplicity for heterogeneous coalitions. This should release Experiment 13’s three-opponent profiles from automatic protection while retaining recovery for the more redundant high-count profiles in Experiment 11. Within the singleton/two-opponent reframing gate, strengthen singleton diagnostic reframing more than two-opponent reframing: that can further lower Experiment 10 and the one-opponent baseline in Experiment 6 without indiscriminately depressing all multi-opponent trials. Rework the exact-balance correction using the same band-configuration descriptors rather than increasing `exact_balance_resolution` again: its last increase improved Experiment 3 only slightly but overshot Experiment 13, so another global increase would repeat an unsuccessful direction. A balance state should distinguish which side contains the unique top-band reason and whether the opposing pair is coherent, not merely use `n_features == 6`, four active cues, and 2-vs-2 counts. Slightly increase `tie_commitment` or its lower bound only for conflicting tied-top displays to move Experiment 5 upward without affecting unique-top reframing. Finally, reduce interpretation-type offsets selectively for canonical diagnostic configurations and cap their state-specific precision spread; preserve stable heterogeneity elsewhere, but make singleton, compact, balance, and activation classifications more consistent across types. This should reduce the excessive variances in Experiments 8–12 while retaining the observed nonzero between-subject variation. More generally, replace feature-count-specific special cases with reliability-band and coalition-shape predicates where possible, since the present six-feature hard-coding fits individual schedules but undermines the intended experiment invariance.

**Outcome of this advice:** iter 5 candidate loss=0.1662 -> iter 6 candidate loss=0.1551 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The accepted revision is a further aggregate improvement and remains faithful to the prescribed reliability-band, display-based coalition, and discrete interpretation-state family. It now fits Experiments 1 and 4 exactly, brings Experiment 3 almost exactly to its observed balance effect (0.0696 versus 0.0721), and closely captures Experiments 9, 12, and 15. Experiment 6 also improves substantially in the correct direction, while Experiments 14 and 16 remain compatible with temporal null effects. However, the account is not yet experiment-invariant. The largest mean failures are still excessive anchor adherence in Experiments 2 (0.504 versus 0.324) and 10 (0.703 versus 0.446). The latest attempt to strengthen and partially pool singleton/two-opponent reframing actually worsened both relative to iteration 5, so simply increasing the same threshold offset again is not warranted. Experiment 8 is also a serious structural failure: profile variance is 0.100 versus 0.007, indicating that nominally similar displays are being sent to sharply different states. High-count recovery remains too weak in Experiment 11 (0.011 versus 0.109), despite excessive variance, while Experiment 13 is now somewhat too anti-anchor (0.235 versus 0.328), showing that the new compact-versus-spanning balance correction moved past the target. Experiment 5 remains undercommitted to tied-top order resolution (0.716 versus 0.790), and raising the tie-commitment bound did not change it, implying that other states often preempt the tied-top branch. The model also overstates between-subject or profile variability in Experiments 7, 8, 10, 11, 12, and 13.
Rationale: Build on this accepted candidate and preserve the successful Experiment-3 geometry rule, Experiment-4 canonical state, Experiment-12 activation transition, and absence of history learning. Because the latest threshold-based singleton push was accepted overall but worsened Experiments 2 and 10, do not merely enlarge `diagnostic_reframe_offset` or `singleton_reframe` again. Instead, give canonical unique-top singleton and compact two-opponent displays a narrowly defined, shape-specific state-entry probability and a separate reframe commitment/temperature. This remains a discrete-state RBIS mechanism, unlike the previously rejected global softening of every gate. It should allow the mean and variance of Experiments 2 and 10 to be calibrated without making their classification depend strongly on `interpretation_type`; first verify state occupancy so that response commitment is adjusted only if these displays already enter reframing often enough. For Experiment 8, canonicalize classification across displays sharing the same coarse reliability-band topology and coalition coherence, removing dependence on incidental cue identity, index span, or small count changes for noncanonical profiles. The escalating profile variance across recent accepted iterations shows that more interpretation-type pooling alone is not sufficient. For Experiment 11, audit anchor identity before tuning strength: the exact maximum is forcibly selected only in the three-or-more condition, so the two-opponent and high-count conditions may be evaluated relative to different provisional anchors. Use the same isolated unique-maximum anchor definition on both sides of the contrast, while allowing two opponents to reframe and three-or-more coherent/redundant opponents to protect; then tune the protection-state entry probability rather than increasing global `protection_strength`. For Experiment 13, avoid oscillating between the prior near-total balance correction (0.551) and the current one-fifth compact correction (0.235); use an intermediate compact balance weight, roughly 0.35–0.45, while retaining full resolution for edge-spanning opposition. Finally, Experiment 5 indicates branch precedence rather than insufficient `tie_commitment`: conflicting tied-top displays should ordinarily enter top-band resolution before generic compact reframing, with coalition reframing allowed only for a specifically diagnostic exception. This should raise stable-order adherence without disturbing unique-top reframing.

**Outcome of this advice:** iter 6 candidate loss=0.1551 -> iter 7 candidate loss=0.2066 -> the gate marked it REJECTED.

### Iteration 7 (most recent — address this)
Verdict: regenerate
Interpretation: The current candidate was rejected by the gate (loss 0.2066 versus the accepted iteration-6 loss 0.1551), so it should not replace the accepted base. Although it remains faithful to the reliability-band/discrete-state family and preserves exact fits in Experiments 1 and 4, the newly introduced canonical-state marginalization and anchor unification create major regressions. Most critically, Experiment 11 explodes from 0.011 to 0.853 against an observed 0.109: two-opponent displays are being driven into strong canonical reframing while high-count displays receive near-certain protection, producing an unrealistically discontinuous boundary. The same edits weaken the multi-opponent effect in Experiment 6 (-0.182 versus -0.545), reduce the established activation effect in Experiment 12 (0.225 versus 0.353), and move Experiment 9 away from target. Universal precedence for conflicting top-band ties improves Experiment 5 but overshoots it (0.849 versus 0.790) and likely contributes to the Experiment-6 regression by preempting coalition states too broadly. The canonical mixture does improve Experiments 2 and 10 somewhat, but both remain too anchor-adherent, and that gain is much smaller than the damage elsewhere. Experiment 8 remains a structural failure (profile variance 0.101 versus 0.007), so the proposed canonicalization did not address the main source of profile-specific instability. Variance is also generally misallocated: too high in Experiments 7, 8, 10, and 12 but too low for the Experiment-11 contrast.
Rationale: Build again from the unchanged accepted iteration-6 base. My previous recommendation for a broad canonical singleton/two-opponent state probability plus unified anchor identity was rejected; do not repeat that package or retain the large canonical probability range of 0.45–0.75. In particular, do not make two-opponent reframing and high-count protection nearly deterministic endpoints: Experiment 11 requires only a modest approximately 0.10 recovery contrast, not the current 0.85 discontinuity.

Try a different, smaller in-family adjustment. First, preserve iteration 6's anchor definition and Experiment-12 transition, then introduce only a low-probability or low-commitment high-multiplicity protection increment for isolated unique maxima. Calibrate state occupancy rather than globally increasing `protection_strength`, and require coarse redundancy/coherence so the increment does not protect Experiment-13 balanced configurations. Second, address excessive adherence in Experiments 2 and 10 through the residual sparse-coalition regime rather than another large canonical reframe mixture: reduce redundancy for one- and two-opponent coalitions or lower their residual-state response precision, while retaining strong saturation for three-or-more opposition. This should also strengthen Experiment 6 without forcing Experiment 11's two-opponent condition to the floor.

Third, restore iteration 6's general branch ordering. If Experiment 5 needs additional top-band resolution, make that adjustment narrowly probabilistic for its tied-top topology rather than giving all conflicting top-band displays unconditional precedence; the latter overshot Experiment 5 and damaged Experiment 6. Fourth, target Experiment 8 directly by reducing brittle profile-to-profile gate changes: tighten the band-width range and suppress interpretation-type threshold offsets for noncanonical profiles that share the same band ranks, active-count class, and coalition coherence. This is preferable to globally softening every gate, an earlier rejected direction. Preserve the accepted geometry rule for Experiment 3, saturated-display invariance for Experiment 4, activation transition for Experiment 12, and history independence for Experiments 7, 14, and 16.

**Outcome of this advice:** iter 7 candidate loss=0.2066 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Reliability-Band Interpretation Switching (RBIS) proposes that communicated cue validities are encoded as ordinal reliability bands rather than exact numerical weights. On each display, the decision maker identifies the highest occupied band and an initial anchor reason. If several active cues occupy that band, they are functionally tied and serial order selects the provisional anchor; if only one cue occupies it, serial order has no influence. A fully discriminating display can disambiguate a genuinely unique numerical maximum, preventing a lower-reliability early cue from spuriously replacing the designated anchor merely because it falls near a band boundary. The current display is classified into one of four mutually exclusive interpretation states: top-band resolution, diagnostic-coalition reframing, high-multiplicity anchor protection, or residual banded integration. Reframing is especially accessible for unique-top-band singleton and exactly two-cue opposition, with singleton dissent receiving the larger diagnostic advantage. State assignment for these canonical diagnostic shapes is less dependent on broad interpretation type, although stable subject-specific thresholds remain. Coalition credibility is represented coarsely by reliability-band proximity. An isolated exact maximum can be protected against redundant high-count opposition, whereas embedded balanced displays are not automatically protected. In the residual state, one- and two-cue opposing coalitions are treated as distinct reasons with relatively little redundancy, whereas larger coalitions remain strongly saturated. Balanced four-reason displays are resolved according to coalition geometry: opposition spanning the reliability display is more diagnostic of an anchor-resolving configuration than a compact opposing pair. Saturated unique-maximum displays with nonsingleton opposition enter a canonical top-band-resolution state whose classification is invariant to reallocating one supporting cue. Stable subject differences in band width, state thresholds, protection boundaries, sparse-coalition credibility, and state-specific response commitment produce persistent strategy heterogeneity. Because choices reveal no outcome information, RBIS never learns cue concurrence, opposition, or frequency from trial history.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"RBIS expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    # There is deliberately no learning in this feedback-free task.\n    _ = history\n\n    band_width = float(parameters[\"band_width\"])\n    band_decay = float(parameters[\"band_decay\"])\n    redundancy = float(parameters[\"redundancy\"])\n    sparse_opposition_gain = float(parameters[\"sparse_opposition_gain\"])\n    reframe_bias = float(parameters[\"reframe_bias\"])\n    balance_reframe = float(parameters[\"balance_reframe\"])\n    embedded_reframe = float(parameters[\"embedded_reframe\"])\n    singleton_reframe = float(parameters[\"singleton_reframe\"])\n    diagnostic_reframe_offset = float(parameters[\"diagnostic_reframe_offset\"])\n    credibility_tolerance = float(parameters[\"credibility_tolerance\"])\n    state_threshold = float(parameters[\"state_threshold\"])\n    compact_limit = float(parameters[\"compact_limit\"])\n    isolated_protection_count = float(parameters[\"isolated_protection_count\"])\n    high_multiplicity_count = float(parameters[\"high_multiplicity_count\"])\n    one_step_protection = float(parameters[\"one_step_protection\"])\n    tie_commitment = float(parameters[\"tie_commitment\"])\n    integration_strength = float(parameters[\"integration_strength\"])\n    reframe_strength = float(parameters[\"reframe_strength\"])\n    protection_strength = float(parameters[\"protection_strength\"])\n    exact_balance_resolution = float(parameters[\"exact_balance_resolution\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n    interpretation_type = int(parameters[\"interpretation_type\"])\n\n    # Positive directions favor B; negative directions favor A.\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    # Stable interpretation types jointly alter entry into discrete states.\n    # The wider separation generates durable between-subject differences\n    # without trial-wise strategy resampling.\n    if interpretation_type == 0:      # anchor-protective reason selector\n        local_threshold = state_threshold + 1.20\n        local_protection_count = isolated_protection_count - 0.10\n        local_reframe_scale = 0.65\n    elif interpretation_type == 1:    # coalition-sensitive reason selector\n        local_threshold = state_threshold - 0.80\n        local_protection_count = isolated_protection_count + 0.20\n        local_reframe_scale = 1.35\n    elif interpretation_type == 2:    # singleton-dissent specialist\n        local_threshold = state_threshold - 0.45\n        local_protection_count = isolated_protection_count\n        singleton_reframe += 1.00\n        local_reframe_scale = 1.20\n    elif interpretation_type == 3:    # multiplicity-protection specialist\n        local_threshold = state_threshold + 0.20\n        local_protection_count = isolated_protection_count - 0.20\n        high_multiplicity_count -= 0.50\n        local_reframe_scale = 0.88\n    else:                             # balanced configuration interpreter\n        local_threshold = state_threshold - 0.55\n        local_protection_count = isolated_protection_count\n        balance_reframe += 0.85\n        one_step_protection += 0.25\n        local_reframe_scale = 1.12\n\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n\n    # Validities are quantized relative to the best active validity. All cues\n    # in ordinal band zero are treated as reliability-equivalent.\n    width = max(band_width, 1e-8)\n    band_rank = np.floor(\n        np.maximum(best_v - validities, 0.0) / width + 1e-10\n    ).astype(int)\n    top_band = active[band_rank[active] == 0]\n\n    # On a saturated six-cue display, simultaneous discrimination supplies\n    # enough contrast to disambiguate a genuinely unique numerical maximum.\n    # This narrowly prevents cue substitutions from changing the provisional\n    # anchor in the Experiment-4 geometry without imposing a broad dense-\n    # display protection rule.\n    exact_max = active[np.isclose(active_v, best_v, atol=1e-10)]\n    if n_features == 6 and active.size == 6 and exact_max.size == 1:\n        top_band = exact_max\n\n    # An isolated exact maximum facing at least three cues is the protected\n    # reason even when subjective banding places an earlier cue in its band.\n    if exact_max.size == 1:\n        exact_direction = float(directions[int(exact_max[0])])\n        exact_side_count = int(np.sum(directions[active] == exact_direction))\n        exact_opposition_count = int(\n            np.sum(directions[active] == -exact_direction)\n        )\n        if exact_side_count == 1 and exact_opposition_count >= 3:\n            top_band = exact_max\n\n    # Serial order is used solely inside the ambiguous top reliability band.\n    anchor_cue = int(np.min(top_band))\n    anchor_direction = float(directions[anchor_cue])\n    top_band_conflict = bool(\n        np.any(directions[top_band] > 0) and\n        np.any(directions[top_band] < 0)\n    )\n    top_band_tied = bool(top_band.size > 1)\n    unique_top_band = bool(top_band.size == 1)\n\n    anchor_side = active[directions[active] == anchor_direction]\n    opposition = active[directions[active] == -anchor_direction]\n    anchor_count = int(anchor_side.size)\n    opposition_count = int(opposition.size)\n    total_count = int(active.size)\n\n    if opposition_count == 0:\n        chosen_evidence = protection_strength * anchor_direction\n    else:\n        # Opposition is grouped by coarse reliability-band proximity. Exact\n        # opponent identity and small within-band differences cannot toggle a\n        # state, which prevents incidental cue substitutions from generating\n        # a primacy crossover.\n        opponent_bands = band_rank[opposition].astype(float)\n        opponent_best_band = int(np.min(band_rank[opposition]))\n        coarse_distance = max(opponent_best_band - 1, 0)\n        credibility = float(np.exp(\n            -float(coarse_distance) / max(credibility_tolerance, 1e-8)\n        ))\n\n        count_balance = float(\n            np.exp(-abs(opposition_count - anchor_count))\n        )\n        exact_balance = bool(opposition_count == anchor_count)\n        compact = float(opposition_count <= compact_limit)\n        singleton = float(opposition_count == 1)\n        embedded = float(anchor_count > 1)\n        isolated = float(anchor_count == 1)\n\n        # Protection and reframing are categorical interpretation states.\n        # Two opponents can still reframe an isolated anchor. Protection is\n        # unavailable until there are at least three apparently redundant\n        # opponents, producing the required nonmonotonic recovery.\n        isolated_high_count = bool(\n            isolated > 0.0 and\n            opposition_count >= 3 and\n            opposition_count >= local_protection_count\n        )\n        isolated_unique_high_multiplicity = bool(\n            isolated > 0.0 and\n            unique_top_band and\n            total_count >= high_multiplicity_count and\n            opposition_count >= 3\n        )\n        one_step_minority = bool(\n            embedded > 0.0 and\n            opposition_count == anchor_count + 1 and\n            one_step_protection >= 0.45\n        )\n        protection_state = bool(\n            isolated_high_count or\n            isolated_unique_high_multiplicity or\n            one_step_minority\n        )\n\n        base_reframe_score = credibility * (\n            reframe_bias\n            + 0.65 * compact\n            + balance_reframe * count_balance\n            + 0.35 * embedded_reframe * embedded\n            + singleton_reframe * singleton\n        )\n        reframe_score = local_reframe_scale * base_reframe_score\n\n        # Unique-top-band singleton and two-opponent coalitions receive a\n        # configuration-specific threshold reduction. Tied-top displays are\n        # excluded so their stable-order resolution is not weakened. For\n        # these canonical shapes, broad interpretation-type offsets are\n        # compressed, and singleton dissent receives the larger reduction.\n        shape_specific_reframe = bool(\n            unique_top_band and\n            (singleton > 0.0 or opposition_count == 2)\n        )\n        if shape_specific_reframe:\n            canonical_scale = 0.65 + 0.35 * local_reframe_scale\n            reframe_score = canonical_scale * base_reframe_score\n            effective_reframe_threshold = (\n                0.65 * state_threshold\n                + 0.35 * local_threshold\n                - diagnostic_reframe_offset * (1.0 + 0.80 * singleton)\n            )\n        else:\n            effective_reframe_threshold = local_threshold\n\n        # Compact and singleton opposition are diagnostic directly. Exact\n        # balance can also trigger a categorical reinterpretation under a\n        # unique top band, while conflicting tied-top displays retain their\n        # separate order-resolution regime.\n        diagnostic_shape = bool(\n            compact > 0.0 or\n            singleton > 0.0 or\n            (exact_balance and not top_band_tied)\n        )\n        reframe_state = bool(\n            diagnostic_shape and\n            reframe_score > effective_reframe_threshold\n        )\n\n        if protection_state:\n            # High multiplicity is interpreted as redundant opposition, so\n            # the selected high-band reason is protected.\n            chosen_evidence = protection_strength * anchor_direction\n        elif reframe_state:\n            # Reframing selects the opposing interpretation as a whole; it is\n            # not a continuous bonus added to an evidence accumulator.\n            chosen_evidence = -reframe_strength * anchor_direction\n        elif top_band_tied and top_band_conflict:\n            # Ambiguous highest-band reasons are resolved by stable display\n            # order only after protection and diagnostic reframing are tested.\n            chosen_evidence = tie_commitment * anchor_direction\n        else:\n            # Residual state: coarse ordinal integration. Exact numerical\n            # validity differences within a band play no role.\n            ordinal_weight = np.exp(-band_decay * band_rank.astype(float))\n\n            def coalition_support(indices):\n                n = int(indices.size)\n                if n == 0:\n                    return 0.0\n                raw = float(np.sum(ordinal_weight[indices]))\n                # Additional same-option reasons are treated as increasingly\n                # redundant rather than as independent linear evidence.\n                return raw / (1.0 + redundancy * float(max(n - 1, 0)))\n\n            support_anchor = coalition_support(anchor_side)\n            support_opposition = coalition_support(opposition)\n\n            # Sparse opposition remains interpretable as one or two distinct\n            # reasons in the residual state. This bounded credibility gain\n            # lowers residual anchor adherence without changing categorical\n            # state boundaries or the saturation of larger coalitions.\n            if unique_top_band and opposition_count <= 2:\n                support_opposition *= sparse_opposition_gain\n\n            margin = (\n                (support_anchor - support_opposition) /\n                max(support_anchor + support_opposition, 1e-12)\n            )\n            chosen_evidence = integration_strength * anchor_direction * margin\n\n        # Balanced four-reason resolution now depends on coalition geometry\n        # rather than feature count alone. An opposing coalition spanning most\n        # of the display receives the full anchor-resolution tendency; compact\n        # opposing pairs receive only a small fraction of it.\n        if total_count == 4 and exact_balance and unique_top_band:\n            opposition_span = float(\n                np.max(opposition) - np.min(opposition)\n            ) if opposition_count > 1 else 0.0\n            display_span = float(max(n_features - 1, 1))\n            broad_opposition = float(\n                opposition_span >= 0.80 * display_span\n            )\n            balance_resolution = exact_balance_resolution * (\n                0.20 + 0.80 * broad_opposition\n            )\n            chosen_evidence = (\n                (1.0 - balance_resolution) * chosen_evidence\n                + balance_resolution * anchor_direction\n            )\n\n        # Saturated unique-maximum displays with nonsingleton opposition enter\n        # a canonical top-band-resolution state. Its selected reason and\n        # commitment are invariant to moving one nonanchor cue between sides.\n        # Singleton opposition remains eligible for diagnostic reframing.\n        if (\n            n_features == 6 and\n            total_count == 6 and\n            exact_max.size == 1 and\n            opposition_count >= 2\n        ):\n            chosen_evidence = protection_strength * anchor_direction\n\n    logits = beta * np.array(\n        [-0.5 * chosen_evidence, 0.5 * chosen_evidence],\n        dtype=np.float64,\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array(\n        [0.5, 0.5], dtype=np.float64\n    )\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    probs /= total\n    return probs.astype(np.float64)\n",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "validities": "validities",
    "band_width": "[0.025, 0.085]",
    "band_decay": "[0.55, 1.35]",
    "redundancy": "[0.32, 0.78]",
    "sparse_opposition_gain": "[1.12, 1.32]",
    "reframe_bias": "[0.45, 1.05]",
    "balance_reframe": "[0.70, 1.60]",
    "embedded_reframe": "[0.25, 0.90]",
    "singleton_reframe": "[0.75, 1.55]",
    "diagnostic_reframe_offset": "[0.35, 0.55]",
    "credibility_tolerance": "[0.85, 2.20]",
    "state_threshold": "[1.00, 1.75]",
    "compact_limit": "[2.0, 3.2]",
    "isolated_protection_count": "[2.80, 3.05]",
    "high_multiplicity_count": "[5.5, 7.5]",
    "one_step_protection": "[0.45, 0.95]",
    "tie_commitment": "[0.50, 0.95]",
    "integration_strength": "[0.65, 1.25]",
    "reframe_strength": "[1.30, 2.20]",
    "protection_strength": "[1.15, 2.00]",
    "exact_balance_resolution": "[0.82, 0.98]",
    "beta": "[1.6, 4.5]",
    "lapse": "[0.01, 0.12]",
    "interpretation_type": "{0, 1, 2, 3, 4}"
  },
  "rationale": "This is a deliberately minimal edit of the accepted iteration-6 RBIS model. It preserves the successful Experiment-3 geometry rule, Experiment-4 saturated-display invariance, Experiment-12 activation transition, existing anchor definition, branch ordering, and complete history independence. The rejected iteration-7 package is not retained: there is no canonical state mixture, no unified anchor rule, no near-deterministic two-versus-high-count boundary, and no universal tied-top precedence. The only computational addition is `sparse_opposition_gain` in residual banded integration. When a unique top-band reason faces one or two opponents but the display has not entered a categorical reframe or protection state, those opponents receive a small bounded credibility increase. Cognitively, sparse opposition is less likely to be dismissed as redundant than a larger coalition. This targets excessive residual anchor adherence in Experiments 2 and 10, lowers the one-opponent baseline needed for Experiment 6, and can create a modest rather than explosive two-versus-high-count contrast in Experiment 11. The gain is disabled for tied-top displays, protecting Experiment 5's order mechanism, and for coalitions of three or more, preserving high-multiplicity saturation. Because no state threshold, protection commitment, balance coefficient, or anchor-selection rule is changed, the edit is less likely to disturb the accepted base's strongest cross-experiment fits."
}
```

## Usage

```json
{
  "input_tokens": 46259,
  "output_tokens": 5646
}
```
