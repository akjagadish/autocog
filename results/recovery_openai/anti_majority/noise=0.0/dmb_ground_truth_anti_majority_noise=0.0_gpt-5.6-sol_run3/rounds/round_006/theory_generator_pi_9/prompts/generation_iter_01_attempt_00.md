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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Replace pi_8 with a new theory of Static Configural Coalition Competition. The theory should posit that communicated validity, serial accessibility, and the current directional cue configuration jointly determine evidence on each trial, with no learning from trial history in the absence of feedback. Preserve individual cue identities and validity weights in every profile rather than replacing multi-cue coalitions with equal count-normalized support. Apply smooth within-coalition redundancy so that additional concordant cues have diminishing returns, but add a nonmonotonic opposition function: one- and two-cue opposition can be highly diagnostic, whereas sufficiently large concordant coalitions are increasingly interpreted as redundant and can restore reliance on a strong isolated anchor. This mechanism should accommodate the strong multi-opponent restoration effects in Experiments 8 and 13 while still allowing low anchor adherence in Experiments 1, 4, and 12.

Presentation order should operate through bounded accessibility, strongest for tied or near-tied validities and for configurations in which activation of an extra cue changes coalition balance. It should not be a universal first-cue rule. This can preserve the strong order-related behavior in Experiments 3 and 7 and the Expert-6 activation effect in Experiment 14 without producing the spurious crossover prevalence seen for pi_8 in Experiment 6. The model should also permit a coalition-level diagnosticity bonus based solely on the current display—validity spacing, coalition size, and whether dissent is compact—not on learned pairwise anti-concurrence. Consequently it should predict approximately zero experience interactions in Experiments 2 and 9.

Use stable subject mixtures or continuous random effects for validity sensitivity, order accessibility, compact-dissent sensitivity, redundancy curvature, high-count restoration, and response precision. Parameter ranges should be broad enough to reproduce the observed between-subject variance, but the count-response curve and absence of history dependence should be shared theoretical commitments rather than experiment-specific switches. Critical quantitative targets are: anchor choice near 0.33 in Experiment 1; near-zero interactions in Experiments 2 and 9; strong first/tied-cue adherence in Experiments 3 and 7; no primacy-crossover classifications in Experiment 6; approximately -0.55 in Experiment 8; positive high-count restoration near 0.11 in Experiment 13; and a large positive activation effect near 0.35 in Experiment 14.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.659)

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


---

### `pi_6` (overall score: 0.655)

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


---

### `pi_7` (overall score: 0.501)

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
- Experiment 13: real=0.3275 (var=0.0184) vs this=0.4956 (var=0.0025)
- Experiment 14: real=0.0200 (var=0.1946) vs this=-0.0400 (var=0.2234)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2484 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.2484 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.9900 (var=0.0049)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 1.0000 (var=0.0000)
- pi_4: 1.0000 (var=0.0000)
- pi_5: 1.0000 (var=0.0000)
- pi_6: 0.9600 (var=0.0284)
- pi_7: 0.9400 (var=0.0464)
- pi_8: 0.9300 (var=0.0601)

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
  - iter 1 (most recent): 0.7323 (var=0.0070) (Δ vs real +0.4081)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4848 (var=0.0022)
- pi_1: 0.8492 (var=0.0121)
- pi_3: 0.3298 (var=0.0054)
- pi_4: 0.9925 (var=0.0001)
- pi_5: 0.3463 (var=0.0061)
- pi_6: 0.4562 (var=0.0022)
- pi_7: 0.5233 (var=0.0053)
- pi_8: 0.4275 (var=0.0045)

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
  - iter 1 (most recent): -0.0456 (var=0.0023) (Δ vs real -0.1177)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0008 (var=0.0015)
- pi_3: -0.3629 (var=0.0033)
- pi_2: -0.1762 (var=0.0033)
- pi_4: 0.0050 (var=0.0001)
- pi_5: 0.1125 (var=0.0053)
- pi_6: -0.0462 (var=0.0019)
- pi_7: 0.0358 (var=0.0025)
- pi_8: -0.0956 (var=0.0027)

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
  - iter 1 (most recent): 0.7885 (var=0.0093) (Δ vs real -0.0015)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8244 (var=0.0081)
- pi_4: 0.0179 (var=0.0002)
- pi_2: 0.1456 (var=0.0074)
- pi_3: 0.6381 (var=0.0031)
- pi_5: 0.8223 (var=0.0088)
- pi_6: 0.7727 (var=0.0071)
- pi_7: 0.8063 (var=0.0071)
- pi_8: 0.7033 (var=0.0264)

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
  - iter 1 (most recent): -0.3206 (var=0.0164) (Δ vs real +0.2244)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5559 (var=0.0104)
- pi_1: 0.0103 (var=0.0049)
- pi_2: 0.3781 (var=0.0114)
- pi_3: -0.3150 (var=0.0123)
- pi_5: -0.5416 (var=0.0261)
- pi_6: -0.1356 (var=0.0077)
- pi_7: -0.3969 (var=0.0200)
- pi_8: -0.2550 (var=0.0318)

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
  - iter 1 (most recent): 0.0037 (var=0.0145) (Δ vs real -0.0014)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0058 (var=0.0071)
- pi_5: -0.1060 (var=0.0121)
- pi_2: -0.0056 (var=0.0083)
- pi_3: 0.0227 (var=0.0111)
- pi_4: 0.0025 (var=0.0020)
- pi_6: 0.0092 (var=0.0215)
- pi_7: -0.0589 (var=0.0212)
- pi_8: 0.0404 (var=0.0151)

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
  - iter 1 (most recent): 0.0101 (var=0.0002) (Δ vs real +0.0031)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0058 (var=0.0001)
- pi_1: 0.0002 (var=0.0000)
- pi_2: 0.0206 (var=0.0003)
- pi_3: 0.0317 (var=0.0005)
- pi_4: 0.0000 (var=0.0000)
- pi_6: 0.0018 (var=0.0001)
- pi_7: 0.0062 (var=0.0001)
- pi_8: 0.0072 (var=0.0002)

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
  - iter 1 (most recent): 0.8144 (var=0.0110) (Δ vs real +0.0894)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8094 (var=0.0121)
- pi_5: 0.5687 (var=0.0166)
- pi_1: 0.8600 (var=0.0122)
- pi_2: 0.1737 (var=0.0171)
- pi_3: 0.2675 (var=0.0088)
- pi_4: 0.9844 (var=0.0007)
- pi_7: 0.7512 (var=0.0113)
- pi_8: 0.5850 (var=0.0117)

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
  - iter 1 (most recent): 0.7963 (var=0.0155) (Δ vs real +0.3500)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4656 (var=0.0070)
- pi_6: 0.7306 (var=0.0084)
- pi_1: 0.8250 (var=0.0150)
- pi_2: 0.5275 (var=0.0076)
- pi_3: 0.7612 (var=0.0061)
- pi_4: 0.9844 (var=0.0006)
- pi_7: 0.4644 (var=0.0059)
- pi_8: 0.6150 (var=0.0107)

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
  - iter 1 (most recent): 0.2558 (var=0.0373) (Δ vs real +0.1465)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6688 (var=0.0308)
- pi_5: 0.0092 (var=0.0107)
- pi_1: 0.0137 (var=0.0129)
- pi_2: -0.0240 (var=0.0079)
- pi_3: -0.0033 (var=0.0095)
- pi_4: -0.0368 (var=0.0015)
- pi_6: -0.0085 (var=0.0107)
- pi_8: 0.1107 (var=0.0194)

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
  - iter 1 (most recent): 0.0050 (var=0.0096) (Δ vs real -0.3483)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0208 (var=0.0079)
- pi_7: 0.2908 (var=0.0147)
- pi_1: 0.0083 (var=0.0081)
- pi_2: -0.3271 (var=0.0169)
- pi_3: 0.0450 (var=0.0087)
- pi_4: -0.5246 (var=0.0081)
- pi_6: 0.2700 (var=0.0168)
- pi_8: 0.3488 (var=0.0128)

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
  - iter 1 (most recent): 0.3887 (var=0.0257) (Δ vs real +0.0612)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5044 (var=0.0026)
- pi_5: 0.3144 (var=0.0091)
- pi_1: 0.8431 (var=0.0109)
- pi_2: 0.4846 (var=0.0035)
- pi_3: 0.6629 (var=0.0048)
- pi_4: 0.9952 (var=0.0001)
- pi_6: 0.5269 (var=0.0032)
- pi_7: 0.4956 (var=0.0025)

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
  - iter 1 (most recent): 0.0250 (var=0.1431) (Δ vs real +0.0050)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3050 (var=0.1632)
- pi_8: -0.1050 (var=0.1227)
- pi_1: 0.0200 (var=0.1196)
- pi_2: -0.0600 (var=0.2064)
- pi_3: 0.1000 (var=0.2775)
- pi_4: 0.0050 (var=0.0062)
- pi_6: -0.0300 (var=0.2041)
- pi_7: -0.0400 (var=0.2234)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Static Configural Coalition Competition (S3C) proposes that choices are reconstructed from the current display rather than learned from previous trials. Every discriminating expert retains an identity-specific weight derived from communicated validity. These weights are modulated by bounded serial accessibility: early experts become more accessible primarily when their validities are tied or close to the best active validity and when the displayed coalitions are nearly balanced. Option-specific evidence then accumulates with smooth redundancy, so each additional concordant expert contributes less without being replaced by a simple cue count. Opposition has a nonmonotonic interpretation. Compact coalitions of one or two credible dissenters receive a diagnosticity bonus, but larger concordant blocs are increasingly interpreted as redundant. When such a bloc changes coalition balance or becomes sufficiently large, reliance is restored toward the strongest currently accessible anchor. Thus the same shared count-response curve can produce weak anchor adherence against compact opposition and renewed adherence against larger opposition. Stable between-subject variation in validity sensitivity, accessibility, compact-dissent sensitivity, redundancy curvature, restoration, and response precision produces persistent individual differences. Because the task supplies no outcome feedback, the theory prohibits updating from trial history.

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
    anchor_threshold = float(parameters["anchor_threshold"])
    anchor_slope = float(parameters["anchor_slope"])
    beta = float(parameters["beta"])
    lapse = float(parameters["lapse"])

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -60.0, 60.0))))

    v = np.clip(validities, 0.500001, 0.999999)
    reliability = np.log(v / (1.0 - v))
    best_active_reliability = float(np.max(reliability[active]))

    # Identity-specific instruction weights. Centering changes only scale.
    instruction_weight = np.exp(
        validity_sensitivity * (reliability - best_active_reliability)
    )

    a_cues = np.flatnonzero(directions < 0)
    b_cues = np.flatnonzero(directions > 0)
    n_a = int(a_cues.size)
    n_b = int(b_cues.size)
    count_imbalance = abs(n_b - n_a)

    # Order becomes accessible chiefly under close validity and near balance.
    near_balance = np.exp(
        -float(count_imbalance) / max(balance_scale, 1e-8)
    )
    access = np.ones(n_features, dtype=float)
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

    active_v = validities[active]
    best_v = float(np.max(active_v))
    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]

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

    if opponent_count > 0:
        opponent_v = validities[opponent_indices]
        strongest_opponent_v = float(np.max(opponent_v))
        validity_gap = max(best_v - strongest_opponent_v, 0.0)
        opponent_credibility_gate = np.exp(
            -validity_gap / max(credibility_scale, 1e-8)
        )

        # Compactness is a property of the visible cue identities: a dense
        # serial cluster is treated as more coherent than dispersed dissent.
        serial_span = int(np.max(opponent_indices) - np.min(opponent_indices) + 1)
        compactness = float(opponent_count) / float(max(serial_span, 1))

        count_distance = (
            float(opponent_count) - compact_peak
        ) / max(compact_width, 1e-8)
        compact_count_profile = np.exp(-0.5 * count_distance ** 2)

        isolation = 1.0 / float(max(anchor_count, 1))
        challenge_multiplier = 1.0 + (
            compact_dissent
            * compact_count_profile
            * opponent_credibility_gate
            * (1.0 + compactness_gain * compactness)
            * (1.0 + isolation_gain * isolation)
        )
        opponent_support *= challenge_multiplier

    total = anchor_support + opponent_support
    if total <= 1e-12:
        core_evidence = 0.0
    else:
        anchor_margin = (anchor_support - opponent_support) / total
        core_evidence = anchor_direction * anchor_margin

    # Large blocs can restore an anchor because their extra members are read as
    # redundant. Restoration is especially activated when one added opponent
    # moves the display away from balance, but it also grows smoothly with the
    # absolute opposition count.
    high_count_gate = sigmoid(
        restoration_slope
        * (float(opponent_count) - restoration_threshold)
    )
    balance_crossing = sigmoid(
        restoration_slope
        * (float(opponent_count - anchor_count) - 0.5)
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

    if anchor_count > 0:
        anchor_identity_share = float(accessible_weight[anchor_cue]) / max(
            float(np.sum(accessible_weight[anchor_indices])), 1e-12
        )
    else:
        anchor_identity_share = 0.0

    # Embedded anchors can be restored, but isolated anchors express the
    # effect most strongly. This permits activation-induced restoration when
    # an extra displayed cue changes coalition balance.
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
- validity_sensitivity: [0.08, 0.75]
- order_accessibility: [0.55, 2.20]
- accessibility_scale: [0.10, 0.75]
- order_span: [0.55, 2.50]
- balance_accessibility: [0.55, 1.0]
- balance_scale: [0.45, 1.80]
- redundancy_strength: [0.35, 1.40]
- redundancy_curvature: [1.05, 2.20]
- similarity_sensitivity: [2.0, 16.0]
- compact_dissent: [1.5, 6.5]
- compact_peak: [1.25, 1.85]
- compact_width: [0.45, 0.95]
- compactness_gain: [0.20, 1.50]
- credibility_scale: [0.025, 0.16]
- isolation_gain: [0.25, 1.50]
- restoration_strength: [0.65, 0.97]
- restoration_threshold: [2.25, 3.60]
- restoration_slope: [3.5, 11.0]
- balance_restoration: [0.45, 1.0]
- restoration_credibility: [0.30, 0.70]
- anchor_threshold: [0.66, 0.82]
- anchor_slope: [10.0, 35.0]
- beta: [1.3, 5.0]
- lapse: [0.0, 0.13]

`rationale`: S3C directly addresses the previous model's main failures without introducing experiment-specific branches. First, it never replaces a coalition by an equal count-normalized surrogate: every cue's communicated validity, identity, serial location, and current direction remain present in accessible_weight and in the effective-multiplicity redundancy calculation. Second, opposition follows one shared nonmonotonic curve. Its diagnostic challenge peaks between one and two opponents, whereas high-count and balance-crossing gates restore the strongest accessible anchor. This can generate the large negative one-versus-multiple contrast, modest high-count restoration, low adherence to an embedded strong anchor in balanced profiles, and the positive effect of activating an extra cue that moves a display away from balance. Third, order accessibility is conditional rather than universal. It is amplified by validity proximity and near balance, making it strong for tied-top designs while sharply limiting unique-anchor primacy crossovers. Fourth, compact dissent depends only on current validity spacing, cue count, and serial density; no concurrence or anti-concurrence statistics are learned. Consequently exact repeated stimuli have stationary choice probabilities, yielding approximately zero exposure interactions in the feedback-free experiments. Finally, broad subject-level ranges over all psychologically central components provide stable heterogeneity in adherence, restoration, and response precision while retaining common qualitative commitments across experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed static configural family: it preserves cue identities and validity weights, uses display-dependent redundancy and opposition, and is genuinely history-invariant. The latter commitment works well: the nominal experience-sensitive metrics are approximately zero in Experiments 7 and 14, and several other targets are captured closely (Experiments 1, 5, and 8). Experiment 13 is also reasonably close. Nevertheless, the aggregate fit is not yet experiment-invariant. The largest failures are Experiment 2 (top-cue adherence 0.732 versus 0.324), Experiment 4 (primacy-crossover prevalence 0.84 versus 0), Experiment 10 (anchor adherence 0.796 versus 0.446), and Experiment 12 (activation effect 0.005 versus 0.353). The count-response function is also miscalibrated: restoration is too weak in Experiment 6 (-0.321 versus -0.545) but too strong in Experiment 11 (0.256 versus 0.109). Experiment 3 has the wrong sign (-0.046 versus 0.072). Between-subject variability is generally compressed, especially in Experiments 2, 9, 10, and 12, although the Experiment 4 discrepancy is a systematic mean failure rather than merely insufficient heterogeneity.
Rationale: Retain Static Configural Coalition Competition and history invariance, but revise its gating and curve calibration. First, the current accessibility gate is not selective enough: broad validity-closeness scales, weak validity sensitivity, and coalition-level propagation of early-cue accessibility generate a nearly universal first-cue crossover in Experiment 4 despite a unique validity anchor. Make order accessibility fall much more sharply when the early cue is not tied or extremely close to the best active cue—e.g., tighten accessibility_scale, strengthen validity-gap suppression, or gate order modulation by an explicit smooth near-tie function. Preserve strong accessibility in tied-top displays so the excellent Experiment 5 fit is not lost. Second, compact opposition remains too weak relative to anchor evidence in Experiments 2 and 10. Increase the effective diagnosticity of credible one- and two-cue dissent, particularly when dissent is compact or the nominal anchor is embedded rather than isolated; this should lower excessive anchor adherence without abandoning identity-specific validity weights. Avoid solving this through a global reduction in validity sensitivity, which would threaten the tied/order and restoration successes. Third, reshape rather than uniformly strengthen high-count restoration. The model needs a steeper transition from compact opposition to moderate multi-opponent restoration to reach Experiment 6, followed by earlier saturation or weaker growth at the largest counts to avoid the excessive Experiment 11 contrast. A peaked or saturating restoration increment, instead of the current monotone max of two sigmoids, remains fully within the prescribed nonmonotonic count-response mechanism. Fourth, the balance-crossing implementation is behaviorally inert in Experiment 12. Add or strengthen a smooth current-display activation/balance bonus that is directed by the coalition whose status changes when an additional cue is active, rather than allowing restoration only toward the globally best active cue. This must be configuration-based and history-free. Finally, broaden stable subject variation in compact-dissent sensitivity, restoration threshold/curvature, and response precision after correcting the mean structure; do not broaden non-tied order accessibility, since Experiment 4 requires crossover prevalence to remain essentially zero.

**Outcome of this advice:** iter 1 candidate loss=0.2484 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Static Configural Coalition Competition (S3C) proposes that choices are reconstructed from the current display rather than learned from previous trials. Every discriminating expert retains an identity-specific weight derived from communicated validity. These weights are modulated by bounded serial accessibility, but early experts become substantially more accessible only when their communicated validity is tied or extremely close to the best active validity. Option-specific evidence accumulates with smooth redundancy, so additional concordant experts contribute progressively less without being replaced by cue counts. Opposition has a nonmonotonic interpretation: compact coalitions of one or two credible dissenters receive a diagnosticity bonus, especially when dissent challenges an embedded anchor, whereas moderate and large coalitions trigger a rapidly saturating restoration response. A separate display-based balance-change mechanism boosts the side made locally distinctive when activation of one additional cue produces a near-balanced majority-minority configuration. Stable between-subject variation governs validity sensitivity, accessibility, compact-dissent sensitivity, redundancy, restoration, balance-change sensitivity, and response precision. Because no outcome feedback is supplied, trial history never changes choice probabilities.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"S3C expects state with shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = stim.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    # S3C is deliberately history-invariant in this feedback-free task.\n    _ = history\n\n    directions = np.sign(stim[1] - stim[0])\n    active = np.flatnonzero(directions != 0)\n    if active.size == 0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n\n    validity_sensitivity = float(parameters[\"validity_sensitivity\"])\n    order_accessibility = float(parameters[\"order_accessibility\"])\n    accessibility_scale = float(parameters[\"accessibility_scale\"])\n    order_span = float(parameters[\"order_span\"])\n    balance_accessibility = float(parameters[\"balance_accessibility\"])\n    balance_scale = float(parameters[\"balance_scale\"])\n    redundancy_strength = float(parameters[\"redundancy_strength\"])\n    redundancy_curvature = float(parameters[\"redundancy_curvature\"])\n    similarity_sensitivity = float(parameters[\"similarity_sensitivity\"])\n    compact_dissent = float(parameters[\"compact_dissent\"])\n    compact_peak = float(parameters[\"compact_peak\"])\n    compact_width = float(parameters[\"compact_width\"])\n    compactness_gain = float(parameters[\"compactness_gain\"])\n    credibility_scale = float(parameters[\"credibility_scale\"])\n    isolation_gain = float(parameters[\"isolation_gain\"])\n    embedded_challenge = float(parameters[\"embedded_challenge\"])\n    activation_balance_bonus = float(parameters[\"activation_balance_bonus\"])\n    restoration_strength = float(parameters[\"restoration_strength\"])\n    restoration_threshold = float(parameters[\"restoration_threshold\"])\n    restoration_slope = float(parameters[\"restoration_slope\"])\n    balance_restoration = float(parameters[\"balance_restoration\"])\n    restoration_credibility = float(parameters[\"restoration_credibility\"])\n    anchor_threshold = float(parameters[\"anchor_threshold\"])\n    anchor_slope = float(parameters[\"anchor_slope\"])\n    beta = float(parameters[\"beta\"])\n    lapse = float(parameters[\"lapse\"])\n\n    def sigmoid(x):\n        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -60.0, 60.0))))\n\n    v = np.clip(validities, 0.500001, 0.999999)\n    reliability = np.log(v / (1.0 - v))\n    best_active_reliability = float(np.max(reliability[active]))\n\n    # Identity-specific instruction weights. Centering changes only scale.\n    instruction_weight = np.exp(\n        validity_sensitivity * (reliability - best_active_reliability)\n    )\n\n    a_cues = np.flatnonzero(directions < 0)\n    b_cues = np.flatnonzero(directions > 0)\n    n_a = int(a_cues.size)\n    n_b = int(b_cues.size)\n    count_imbalance = abs(n_b - n_a)\n\n    # Order is effectively restricted to exact and very near validity ties.\n    near_balance = np.exp(\n        -float(count_imbalance) / max(balance_scale, 1e-8)\n    )\n    access = np.ones(n_features, dtype=float)\n    for serial_rank, cue in enumerate(active):\n        validity_closeness = np.exp(\n            -(best_active_reliability - reliability[cue])\n            / max(accessibility_scale, 1e-8)\n        )\n        serial_salience = np.exp(\n            -float(serial_rank) / max(order_span, 1e-8)\n        )\n        activation_gate = (\n            (1.0 - balance_accessibility)\n            + balance_accessibility * near_balance\n        )\n        access[cue] += (\n            order_accessibility\n            * validity_closeness\n            * serial_salience\n            * activation_gate\n        )\n\n    accessible_weight = instruction_weight * access\n\n    def coalition_support(indices):\n        n = int(indices.size)\n        if n == 0:\n            return 0.0\n\n        weights = np.asarray(accessible_weight[indices], dtype=float)\n        raw = float(np.sum(weights))\n        dominant = max(float(np.max(weights)), 1e-12)\n        effective_multiplicity = raw / dominant\n        excess = max(effective_multiplicity - 1.0, 0.0)\n\n        coalition_validities = validities[indices]\n        spread = float(np.std(coalition_validities)) if n > 1 else 0.0\n        similarity = np.exp(-similarity_sensitivity * spread)\n        redundancy = 1.0 + redundancy_strength * similarity * (\n            excess ** redundancy_curvature\n        )\n        return raw / max(redundancy, 1e-12)\n\n    support_a = coalition_support(a_cues)\n    support_b = coalition_support(b_cues)\n\n    active_v = validities[active]\n    best_v = float(np.max(active_v))\n    best_candidates = active[np.isclose(active_v, best_v, atol=1e-10)]\n\n    # A communicated-validity tie is resolved by bounded current accessibility.\n    candidate_access = accessible_weight[best_candidates]\n    anchor_cue = int(best_candidates[int(np.argmax(candidate_access))])\n    anchor_direction = float(directions[anchor_cue])\n\n    if anchor_direction > 0:\n        anchor_indices = b_cues\n        opponent_indices = a_cues\n        anchor_support = support_b\n        opponent_support = support_a\n    else:\n        anchor_indices = a_cues\n        opponent_indices = b_cues\n        anchor_support = support_a\n        opponent_support = support_b\n\n    anchor_count = int(anchor_indices.size)\n    opponent_count = int(opponent_indices.size)\n\n    if opponent_count > 0:\n        opponent_v = validities[opponent_indices]\n        strongest_opponent_v = float(np.max(opponent_v))\n        validity_gap = max(best_v - strongest_opponent_v, 0.0)\n        opponent_credibility_gate = np.exp(\n            -validity_gap / max(credibility_scale, 1e-8)\n        )\n\n        # Compactness is computed solely from the current display.\n        serial_span = int(np.max(opponent_indices) - np.min(opponent_indices) + 1)\n        compactness = float(opponent_count) / float(max(serial_span, 1))\n\n        count_distance = (\n            float(opponent_count) - compact_peak\n        ) / max(compact_width, 1e-8)\n        compact_count_profile = np.exp(-0.5 * count_distance ** 2)\n\n        isolation = 1.0 / float(max(anchor_count, 1))\n        embedding = 1.0 - isolation\n        challenge_multiplier = 1.0 + (\n            compact_dissent\n            * compact_count_profile\n            * opponent_credibility_gate\n            * (1.0 + compactness_gain * compactness)\n            * (1.0 + isolation_gain * isolation)\n            * (1.0 + embedded_challenge * embedding)\n        )\n        opponent_support *= challenge_multiplier\n\n    # Activation of one extra cue can make the newly outnumbered coalition\n    # distinctive. This is a smooth current-display effect, not learning and\n    # not necessarily restoration toward the globally strongest cue.\n    if n_a > 0 and n_b > 0:\n        near_one_imbalance = np.exp(\n            -0.5 * ((float(count_imbalance) - 1.0) / 0.35) ** 2\n        )\n        broad_coalition_gate = sigmoid(\n            4.0 * (float(min(n_a, n_b)) - 2.5)\n        )\n        activation_gate = near_one_imbalance * broad_coalition_gate\n        activation_multiplier = 1.0 + (\n            activation_balance_bonus * activation_gate\n        )\n\n        if n_a < n_b:\n            if anchor_direction < 0:\n                anchor_support *= activation_multiplier\n            else:\n                opponent_support *= activation_multiplier\n        elif n_b < n_a:\n            if anchor_direction > 0:\n                anchor_support *= activation_multiplier\n            else:\n                opponent_support *= activation_multiplier\n\n    total = anchor_support + opponent_support\n    if total <= 1e-12:\n        core_evidence = 0.0\n    else:\n        anchor_margin = (anchor_support - opponent_support) / total\n        core_evidence = anchor_direction * anchor_margin\n\n    # Restoration turns on around moderate opposition and then saturates,\n    # limiting additional separation between medium and very large counts.\n    high_count_gate = sigmoid(\n        restoration_slope\n        * (float(opponent_count) - restoration_threshold)\n    )\n    balance_crossing = sigmoid(\n        restoration_slope\n        * (float(opponent_count - anchor_count) - 0.5)\n    )\n    count_gate = max(\n        high_count_gate,\n        balance_restoration * balance_crossing\n    )\n\n    absolute_anchor_gate = sigmoid(\n        anchor_slope * (best_v - anchor_threshold)\n    )\n    credibility_gate = (\n        restoration_credibility\n        + (1.0 - restoration_credibility) * absolute_anchor_gate\n    )\n\n    if anchor_count > 0:\n        anchor_identity_share = float(accessible_weight[anchor_cue]) / max(\n            float(np.sum(accessible_weight[anchor_indices])), 1e-12\n        )\n    else:\n        anchor_identity_share = 0.0\n\n    embedding_factor = 0.35 + 0.65 * np.sqrt(\n        np.clip(anchor_identity_share, 0.0, 1.0)\n    )\n    restoration_gate = (\n        restoration_strength\n        * count_gate\n        * credibility_gate\n        * embedding_factor\n    )\n    restoration_gate = float(np.clip(restoration_gate, 0.0, 0.97))\n\n    choice_evidence = (\n        (1.0 - restoration_gate) * core_evidence\n        + restoration_gate * anchor_direction\n    )\n\n    logits = beta * np.array(\n        [-0.5 * choice_evidence, 0.5 * choice_evidence],\n        dtype=np.float64,\n    )\n    logits = logits - np.max(logits)\n    probs = np.exp(logits)\n    probs /= probs.sum()\n\n    probs = (1.0 - lapse) * probs + lapse * np.array(\n        [0.5, 0.5], dtype=np.float64\n    )\n    probs = np.clip(probs, 0.0, None)\n    normalizer = float(probs.sum())\n    if not np.isfinite(normalizer) or normalizer <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    probs /= normalizer\n    return probs.astype(np.float64)",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "validity_sensitivity": "[0.08, 0.75]",
    "order_accessibility": "[0.55, 2.20]",
    "accessibility_scale": "[0.015, 0.060]",
    "order_span": "[0.55, 2.50]",
    "balance_accessibility": "[0.55, 1.0]",
    "balance_scale": "[0.45, 1.80]",
    "redundancy_strength": "[0.35, 1.40]",
    "redundancy_curvature": "[1.05, 2.20]",
    "similarity_sensitivity": "[2.0, 16.0]",
    "compact_dissent": "[3.0, 10.0]",
    "compact_peak": "[1.25, 1.85]",
    "compact_width": "[0.45, 0.95]",
    "compactness_gain": "[0.20, 1.50]",
    "credibility_scale": "[0.025, 0.16]",
    "isolation_gain": "[0.25, 1.50]",
    "embedded_challenge": "[1.0, 3.5]",
    "activation_balance_bonus": "[1.2, 3.8]",
    "restoration_strength": "[0.65, 0.97]",
    "restoration_threshold": "[1.65, 2.05]",
    "restoration_slope": "[5.0, 10.0]",
    "balance_restoration": "[0.35, 0.80]",
    "restoration_credibility": "[0.30, 0.70]",
    "anchor_threshold": "[0.66, 0.82]",
    "anchor_slope": "[10.0, 35.0]",
    "beta": "[1.1, 5.5]",
    "lapse": "[0.0, 0.15]"
  },
  "rationale": "This is a targeted edit of the accepted S3C model. First, accessibility_scale is sharply narrowed, making serial accessibility effectively an exact- or near-tie mechanism. Tied-top behavior should therefore remain strong while the unique-anchor Expert-1 crossover in Experiment 4 is suppressed. Second, compact opposition is strengthened and receives a new smooth embedded-anchor multiplier. This directly lowers the excessive anchor adherence in Experiments 2 and 10 and modestly lowers Experiment 13 without discarding cue identities or globally weakening validity sensitivity. Third, the restoration threshold is moved earlier and its slope increased. Restoration is consequently already substantial at moderate counts but saturates quickly, strengthening the negative contrast in Experiment 6 while reducing the excessive high-count-versus-two-count separation in Experiment 11. Fourth, the added activation-balance term boosts whichever broad coalition becomes outnumbered by exactly one cue. In Experiment 12, activating Expert 6 changes a balanced display into a 4-versus-3 configuration, making the Expert-4 coalition the distinctive minority; the bonus is therefore directed toward that coalition rather than toward the globally strongest cue. Its broad-coalition gate prevents it from becoming a generic first-cue or one-versus-one rule. All additions depend only on the current display, and history remains explicitly unused."
}
```

## Usage

```json
{
  "input_tokens": 35189,
  "output_tokens": 5564
}
```
