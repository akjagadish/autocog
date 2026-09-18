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
- THEORY 1 = `pi_13`
- THEORY 2 = `pi_14`
- The recommendation below acts on THEORY 2 (= `pi_14`).

Replace pi_14 with a Diversity-Legitimacy Attribution theory. The new theory should propose that people infer not only endorsement strength but also the likely informational independence or rhetorical coordination of each coalition. Homogeneous, high-validity coalitions are compressed into coordinated option-level claims and can become targets of skepticism in unresolved bilateral conflicts. In contrast, a coalition combining weak or neutral endorsers with an exceptional expert can be construed as diverse corroboration: heterogeneity makes the exceptional endorsement look independently diagnostic rather than like a stronger advocacy claim.

The theory should use a continuous, option-symmetric arbitration among: (1) a default nonlinear raw tally that dominates large absolute and proportional margins; (2) coordinated-claim discounting or pragmatic reversal for coherent coalitions in unresolved bilateral conflicts; and (3) a diversity-legitimacy signal combining breadth, dispersion, and upper-tail surprise. The third signal must be an interaction rather than a generic soft maximum: an extreme expert should gain positive influence when embedded in a sufficiently broad heterogeneous coalition, while an equally extreme but isolated or homogeneous endorsement may still be discounted as an overclaim. Coalition heterogeneity should therefore be capable of increasing evidential credibility, not only reducing advocacy salience.

This account should predict a negative trend in Experiment 1 because replacing neutral fillers with moderately valid fillers removes diversity and upper-tail distinctiveness; high capture in Experiment 2 because the perfect expert is legitimized by a broad heterogeneous coalition; the negative dispersion gradient in Experiment 19; strong heterogeneous-coalition choices in Experiment 21; modest following of unilateral coalitions in Experiment 26; and majority dominance in Experiments 6, 8, and 22. All gates should vary smoothly with breadth, dispersion, validity contrast, and tally decisiveness, without endpoint-specific rules. Stable subject differences should primarily govern sensitivity to coordination versus diversity, with constrained choice precision and lapse parameters. Calibration should be joint across all experiments and should reproduce both pooled metrics and the generally modest subject-level variances.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_11` (overall score: 0.628)

**Description**
Coherence-Gated Relative Coalition Comparison theory proposes that people compress endorsements into option-level coalition representations before making a choice. Each representation combines smoothly encoded mean assertiveness, sublinear consensus, and a coherence gate. The coherence gate is threshold-shaped: ordinary variation among coalition members has little effect, whereas sufficiently extreme dispersion sharply weakens the coalition. Consequently, a maximally dispersed opposing coalition makes a homogeneous coalition relatively prominent and thus a stronger target of skeptical interpretation, without making routine coalition expansion intrinsically reactance-producing. Sublinear consensus is offset by breadth and active-evidence normalization, yielding approximately breadth-invariant comparison of equal-sized homogeneous coalitions. A narrow-conflict focus rule further distinguishes maximal relative contrasts from ordinary singleton disagreements: a smooth 100%-versus-50% contrast remains strongly skepticism-provoking, while less extreme singleton contrasts are attenuated without an endpoint-specific bonus. A separate validity-blind tally channel grows concavely with vote margin. Its gain is reduced while its curvature is increased, pivoting tally evidence away from one-vote margins and toward moderate and large margins. Continuous correlation between tally orientation and response noise prevents high-tally subjects from becoming deterministically majority-following. Skeptical orientation, tally sensitivity, and decision noise vary continuously across people, with composition-dependent heterogeneity concentrated in genuinely dispersed conflicts. With no correctness feedback, processing remains history independent and exactly option-swap symmetric.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Coherence-Gated Relative Coalition Comparison. Choice history is ignored
    # because previous choices are not accompanied by correctness feedback.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive signs endorse A and negative signs endorse B. Features on which
    # the options agree carry neither a categorical vote nor a directional claim.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    total_breadth = int(np.count_nonzero(active))

    # A single smooth, monotone, saturating validity code. The normalization
    # maps 50% to zero and 100% to one without assigning either 95% or 100% a
    # qualitatively distinct representation.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["validity_curvature"])
    endpoint = np.tanh(0.5 * curvature)
    assertiveness = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)

    skeptic_trait = float(parameters["skeptical_orientation"])

    def summarize(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_strength = float(np.mean(members))
        dispersion = float(np.mean((members - mean_strength) ** 2))

        # The same continuous skeptical trait changes sensitivity to internal
        # inconsistency. The superlinear dispersion ratio leaves ordinary
        # coalition composition largely intact but sharply gates sufficiently
        # dispersed coalitions.
        scale = float(parameters["coherence_scale"]) * np.exp(
            float(parameters["coherence_trait_coupling"])
            * (skeptic_trait - 0.5)
        )
        floor = float(parameters["coherence_floor"])
        dispersion_ratio = dispersion / max(scale, 1e-12)
        coherence = floor + (1.0 - floor) * np.exp(
            -(dispersion_ratio ** float(parameters["coherence_power"]))
        )

        consensus = float(breadth) ** float(parameters["consensus_exponent"])
        representation = mean_strength * consensus * coherence
        return float(representation), dispersion, breadth

    rep_a, dispersion_a, breadth_a = summarize(1.0)
    rep_b, dispersion_b, breadth_b = summarize(-1.0)

    if total_breadth == 0:
        contextual_contrast = 0.0
        pooled_dispersion = 0.0
    else:
        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)

        # For equal breadth n on both sides, homogeneous representations grow
        # as n**alpha while this breadth unit grows by the same factor. Thus
        # repetition alone produces almost no change in validity contrast.
        alpha = float(parameters["consensus_exponent"])
        breadth_unit = (float(total_breadth) / 2.0) ** alpha

        # Total active assertiveness consumes limited comparison capacity. This
        # is a relative, composition-sensitive normalization and therefore
        # preserves a maximal 100%-versus-50% singleton contrast.
        active_mean = float(np.mean(assertiveness[active]))
        denominator = breadth_unit * (
            float(parameters["attention_floor"])
            + float(parameters["evidence_normalization"]) * active_mean
        )
        raw_relative_contrast = (rep_a - rep_b) / max(denominator, 1e-12)
        contextual_contrast = float(np.tanh(
            float(parameters["contrast_accumulation"])
            * raw_relative_contrast
        ))

    # Stronger relative coalition claims provoke skepticism. Trait effects are
    # selectively coupled to composition and margin, so individual differences
    # grow where construal is genuinely ambiguous rather than on every trial.
    dispersion_index = pooled_dispersion / (pooled_dispersion + 0.04)
    margin_index = abs(tally_margin) / (abs(tally_margin) + 2.0)
    skeptical_multiplier = 0.70 + 0.60 * skeptic_trait
    skeptical_multiplier *= (
        1.0
        + float(parameters["composition_trait_gain"])
        * (skeptic_trait - 0.5)
        * dispersion_index
    )
    skeptical_multiplier *= (
        1.0
        + float(parameters["skeptic_margin_coupling"])
        * (skeptic_trait - 0.5)
        * margin_index
    )
    skeptical_multiplier = max(float(skeptical_multiplier), 0.05)

    # In a narrow bilateral conflict, weak relative contrasts receive less
    # pragmatic focus than maximal contrasts. The adjustment vanishes smoothly
    # with breadth and therefore does not create a general repetition benefit.
    narrow_weight = np.exp(-max(float(total_breadth) - 2.0, 0.0) / 2.0)
    contrast_focus = 1.0 - narrow_weight * (
        1.0 - float(parameters["singleton_focus_floor"])
    ) * (1.0 - abs(contextual_contrast))
    skeptical_evidence = -float(parameters["skeptical_strength"]) * (
        skeptical_multiplier * contrast_focus * contextual_contrast
    )

    # The independent tally channel is validity blind. Its concave nonlinear
    # growth differentiates margins and protects large majorities, but is less
    # steep and less confident than a superlinear margin rule.
    tally_trait = float(parameters["tally_trait"])
    tally_gain = float(parameters["tally_sensitivity"]) * (
        0.20 + 1.60 * tally_trait
    )
    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        magnitude = abs(tally_margin)
        tally_curve = (1.0 + magnitude) ** float(parameters["tally_exponent"]) - 1.0
        tally_evidence = tally_gain * np.sign(tally_margin) * tally_curve

    total_evidence = tally_evidence + skeptical_evidence

    # Noise varies continuously, but conflict-dependent noise is concentrated
    # in dispersed, low-margin comparisons instead of being a blanket variance
    # adjustment. High tally orientation is also coupled to lower response
    # precision, preventing high-gain subjects from becoming deterministic.
    noise_trait = float(parameters["noise_trait"])
    base_gain = float(parameters["choice_sensitivity"]) * (
        0.72 + 0.56 * noise_trait
    )
    base_gain /= (
        1.0
        + float(parameters["tally_noise_correlation"])
        * tally_trait
    )
    low_margin_weight = 1.0 / (1.0 + abs(tally_margin) / 2.0)
    conflict_load = dispersion_index * low_margin_weight
    effective_gain = base_gain / (
        1.0
        + float(parameters["conflict_noise"])
        * (1.20 - 0.40 * noise_trait)
        * conflict_load
    )
    decision_variable = float(effective_gain * total_evidence)

    # Positive evidence favors A. Symmetric logits guarantee exact invariance
    # under swapping the two displayed options.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))


`parameters`:
- validity_curvature: [1.6, 3.0]
- consensus_exponent: [0.58, 0.76]
- coherence_scale: [0.035, 0.075]
- coherence_floor: [0.05, 0.25]
- coherence_power: [1.8, 3.0]
- coherence_trait_coupling: [-0.70, 0.70]
- attention_floor: [0.28, 0.46]
- evidence_normalization: [0.48, 0.76]
- contrast_accumulation: [1.15, 1.75]
- singleton_focus_floor: [0.45, 0.70]
- skeptical_strength: [1.85, 2.85]
- skeptical_orientation: [0, 1]
- composition_trait_gain: [0.30, 0.90]
- skeptic_margin_coupling: [-0.30, 0.45]
- tally_sensitivity: [0.25, 0.43]
- tally_trait: [0, 1]
- tally_exponent: [0.78, 0.96]
- choice_sensitivity: [0.82, 1.18]
- noise_trait: [0, 1]
- tally_noise_correlation: [0.15, 0.35]
- conflict_noise: [0.10, 0.55]
- lapse_rate: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.4133 (var=0.0100)
- Experiment 2: real=89.1422 (var=7.5896) vs this=51.5068 (var=3.0531)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.3771 (var=0.0040)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8800 (var=0.0067)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.3760 (var=0.0522)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8948 (var=0.0037)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.1813 (var=0.0070)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-4.5864 (var=0.0269)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.2755 (var=0.0039)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.8533 (var=0.0093)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.1917 (var=0.0067)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.1503 (var=0.0048)
- Experiment 13: real=0.0094 (var=0.0050) vs this=0.0403 (var=0.0095)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.1763 (var=0.0121)
- Experiment 15: real=0.0000 (var=0.2400) vs this=0.0000 (var=0.1476)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.6425 (var=0.0062)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=-0.1862 (var=0.0175)
- Experiment 18: real=0.0058 (var=0.0053) vs this=-0.0058 (var=0.0109)
- Experiment 19: real=0.8617 (var=0.0116) vs this=0.6850 (var=0.0125)
- Experiment 20: real=0.8567 (var=0.0052) vs this=0.7683 (var=0.0169)
- Experiment 21: real=0.8667 (var=0.0085) vs this=0.6383 (var=0.0061)
- Experiment 22: real=-0.0058 (var=0.0033) vs this=-0.1446 (var=0.0057)
- Experiment 23: real=0.5000 (var=0.0088) vs this=0.4625 (var=0.0112)
- Experiment 24: real=0.5763 (var=0.0023) vs this=0.2412 (var=0.0072)
- Experiment 25: real=-0.3065 (var=0.0176) vs this=-0.0285 (var=0.0096)
- Experiment 26: real=0.8683 (var=0.0070) vs this=0.4842 (var=0.0171)


---

### `pi_12` (overall score: 0.571)

**Description**
Uncertainty-Arbitrated Overclaim Construal theory proposes that people form two competing, stimulus-dependent interpretations of binary expert advice. The reliability construal compresses each option's endorsers into a coalition claim defined by mean communicated reliability and an inferred trustworthiness estimate. Coalition breadth does not directly amplify this claim, so repeating an otherwise homogeneous coalition is approximately representationally redundant. Trustworthiness remains high under ordinary heterogeneity but collapses after dispersion crosses a subject-specific uncertainty threshold. This makes an extremely split coalition a poor bearer of a unified claim and can render the opposing homogeneous coalition the uniquely salient overclaim. A skeptical pragmatic response then favors the option opposite that salient claim. In parallel, a social-count construal ignores validity and represents only the raw endorsement margin. Its evidence follows a sigmoid power curve with convex initial growth and a confidence-limited asymptote. A localized confidence-control window attenuates moderate tally margins while leaving decisive margins nearly unchanged. A metacognitive gate arbitrates between the two construals according to claim certainty, coalition dispersion, tally decisiveness, and representational disagreement. Moderate tied claims receive additional scrutiny only inside a narrow reliability-contrast band, preventing the intervention from spreading to endpoint-like equal-coalition conflicts. Precision is selectively reduced for one-vote conflicts and for endpoint-sized bilateral claims supported by two trustworthy coalitions. Consequently, ambiguous conflicts move toward chance, moderate overclaims can evoke skepticism, and extremely dispersed claims remain discounted. With no correctness feedback, the process is history independent. Every computation uses signed option contrasts, guaranteeing exact option-swap symmetry.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Uncertainty-Arbitrated Overclaim Construal model. History is intentionally
    # ignored because choices are never followed by correctness feedback.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive signs endorse A and negative signs endorse B. Agreements are
    # non-discriminating and enter neither representation.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)
    active_count = int(np.count_nonzero(active))

    # A smooth bounded reliability code avoids endpoint categories while
    # retaining sensitivity throughout the communicated validity scale.
    validity = np.clip(validities, 0.5, 1.0)
    reliability_curvature = float(parameters["reliability_curvature"])
    endpoint = np.tanh(0.5 * reliability_curvature)
    reliability = np.tanh(
        reliability_curvature * (validity - 0.5)
    ) / max(float(endpoint), 1e-12)

    # The arbitration threshold is a stable continuous individual difference.
    arbitration_threshold = float(parameters["arbitration_threshold"])
    dispersion_threshold = float(parameters["dispersion_threshold"]) * (
        0.82 + 0.36 * arbitration_threshold
    )

    def coalition(side):
        members = reliability[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0

        # Mean reliability is the coalition's location estimate. Breadth is not
        # an unconditional multiplier: homogeneous repetitions communicate the
        # same option-level claim rather than successively stronger claims.
        location = float(np.mean(members))
        dispersion = float(np.mean((members - location) ** 2))

        # Inferred coalition trustworthiness has a threshold-shaped collapse.
        # Ordinary heterogeneity is tolerated, whereas a maximally split
        # coalition is treated as unable to sustain a unified assertion.
        steepness = float(parameters["dispersion_steepness"])
        z = np.clip(steepness * (dispersion - dispersion_threshold), -60.0, 60.0)
        core_trust = 1.0 / (1.0 + np.exp(z))
        tail = float(parameters["trust_tail"])
        trust = tail + (1.0 - tail) * core_trust

        # Very small coalitions carry somewhat less metacognitive certainty,
        # but this saturates rapidly and cannot generate repeated-consensus
        # reactance in broad homogeneous coalitions.
        sampling_confidence = 1.0 - np.exp(
            -float(breadth) / float(parameters["sampling_scale"])
        )
        metacognitive_trust = trust * (
            float(parameters["sampling_floor"])
            + (1.0 - float(parameters["sampling_floor"])) * sampling_confidence
        )
        claim = location * trust
        return float(claim), float(metacognitive_trust), dispersion, breadth

    claim_a, trust_a, dispersion_a, breadth_a = coalition(1.0)
    claim_b, trust_b, dispersion_b, breadth_b = coalition(-1.0)

    if active_count == 0:
        reliability_contrast = 0.0
        pooled_dispersion = 0.0
        claim_certainty = 0.0
    else:
        # Relative normalization makes claim comparison breadth invariant for
        # equal homogeneous coalitions and bounds it under arbitrary designs.
        claim_mass = abs(claim_a) + abs(claim_b)
        reliability_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + claim_mass
        )
        reliability_contrast = float(np.clip(reliability_contrast, -1.0, 1.0))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(active_count)

        # A contrast can be scrutinized when at least one side supports a
        # trustworthy claim. Thus discounting a split coalition can expose the
        # opposing homogeneous coalition as the salient overclaim.
        strongest_trust = max(trust_a, trust_b)
        claim_certainty = strongest_trust * abs(reliability_contrast)

    # Skeptical interpretation reverses a salient reliability-weighted claim.
    # Individual differences are amplified in low-margin, representation-level
    # conflicts rather than uniformly across all trials.
    skeptical_trait = float(parameters["skeptical_orientation"])
    ambiguity_index = 1.0 / (
        1.0 + (margin_magnitude / float(parameters["ambiguity_margin_scale"])) ** 2
    )
    skeptical_multiplier = 1.0 + float(parameters["skeptic_trait_spread"]) * (
        skeptical_trait - 0.5
    ) * (0.35 + 0.65 * ambiguity_index)
    skeptical_gain = float(parameters["skeptical_gain"]) * max(
        skeptical_multiplier, 0.10
    )
    bounded_claim = np.tanh(
        float(parameters["claim_accumulation"]) * reliability_contrast
    )
    skeptical_evidence = -skeptical_gain * float(bounded_claim)

    # The independent social-count representation has convex initial growth,
    # an intermediate steep region, and a confidence-limited asymptote.
    if margin_magnitude < 1e-12:
        tally_shape = 0.0
    else:
        tally_power = float(parameters["tally_power"])
        half_margin = float(parameters["tally_half_margin"])
        numerator = margin_magnitude ** tally_power
        tally_shape = numerator / (half_margin ** tally_power + numerator)

    # Tally-gain heterogeneity is attenuated as count evidence becomes decisive,
    # helping preserve the low between-subject variance of clear margin series.
    tally_trait = float(parameters["tally_gain_trait"])
    tally_uncertainty = 1.0 - tally_shape
    tally_multiplier = 1.0 + float(parameters["tally_trait_spread"]) * (
        tally_trait - 0.5
    ) * (0.25 + 0.75 * tally_uncertainty)
    tally_gain = float(parameters["tally_gain"]) * max(tally_multiplier, 0.10)
    tally_evidence = (
        tally_gain * np.sign(tally_margin) * tally_shape
        if margin_magnitude > 0.0 else 0.0
    )

    # Confidence control selectively attenuates moderate margins, where raw
    # counting is informative but not yet decisive. The Gaussian window has
    # little effect on one-vote trials or margins seven and above.
    mid_margin_window = np.exp(
        -0.5 * (
            (margin_magnitude - float(parameters["mid_margin_center"]))
            / float(parameters["mid_margin_width"])
        ) ** 2
    )
    tally_evidence *= 1.0 - float(
        parameters["mid_margin_tally_damping"]
    ) * mid_margin_window

    # Arbitration depends jointly on claim reliability, coalition dispersion,
    # tally decisiveness, and disagreement between the two construals.
    normalized_claim_direction = float(np.tanh(skeptical_evidence))
    normalized_tally_direction = float(np.tanh(tally_evidence))
    disagreement = abs(
        normalized_claim_direction - normalized_tally_direction
    ) / 2.0
    dispersion_index = pooled_dispersion / (
        pooled_dispersion + float(parameters["dispersion_reference"])
    ) if active_count > 0 else 0.0

    # A localized boost admits skeptical scrutiny of moderate reliability
    # contrasts only when the tally is unresolved and both coalitions remain
    # trustworthy. It therefore does not revive a maximally split coalition.
    both_trustworthy = min(trust_a, trust_b)
    contrast_magnitude = abs(reliability_contrast)
    moderate_center = float(parameters["moderate_contrast_center"])
    moderate_width = float(parameters["moderate_contrast_width"])
    moderate_band = np.exp(
        -0.5 * ((contrast_magnitude - moderate_center) / moderate_width) ** 2
    )
    tie_index = np.exp(-(margin_magnitude / 0.35) ** 2)
    moderate_tie_boost = (
        float(parameters["moderate_tie_gate_boost"])
        * tie_index
        * both_trustworthy
        * moderate_band
    )

    gate_drive = (
        float(parameters["certainty_weight"]) * claim_certainty
        + float(parameters["disagreement_weight"]) * disagreement
        - float(parameters["dispersion_uncertainty_weight"]) * dispersion_index
        - float(parameters["margin_resolution_weight"]) * tally_shape
        - arbitration_threshold
        + moderate_tie_boost
    )
    gate_temperature = float(parameters["gate_temperature"])
    gate_logit = np.clip(gate_drive / gate_temperature, -60.0, 60.0)
    claim_weight = 1.0 / (1.0 + np.exp(-gate_logit))

    # Arbitration is a reliability-weighted competition between construals,
    # not an additive fixed coalition-plus-tally score.
    total_evidence = (
        claim_weight * skeptical_evidence
        + (1.0 - claim_weight) * tally_evidence
    )

    # Precision differences matter chiefly when the stimulus remains ambiguous;
    # their expression contracts on decisive tallies.
    precision_trait = float(parameters["precision_trait"])
    unresolved = 1.0 - tally_shape * (1.0 - claim_weight)
    precision_multiplier = 1.0 + float(parameters["precision_trait_spread"]) * (
        precision_trait - 0.5
    ) * (0.20 + 0.80 * unresolved)
    precision = float(parameters["base_precision"]) * max(
        precision_multiplier, 0.10
    )

    # Selective confidence limiting pulls one-vote conflicts toward chance
    # without changing the moderate-to-large tally asymptote. A second limiter
    # applies to endpoint-sized bilateral claims only when both coalitions are
    # trustworthy, preserving the extreme-dispersion crossover.
    nonzero_margin_index = 1.0 - np.exp(-(margin_magnitude / 0.35) ** 2)
    small_margin_load = nonzero_margin_index * np.exp(
        -max(margin_magnitude - 1.0, 0.0)
        / float(parameters["small_margin_decay"])
    )
    precision *= 1.0 - float(
        parameters["small_margin_precision_damping"]
    ) * small_margin_load

    extreme_center = float(parameters["extreme_contrast_center"])
    extreme_width = float(parameters["extreme_contrast_width"])
    extreme_z = np.clip(
        (contrast_magnitude - extreme_center) / extreme_width, -60.0, 60.0
    )
    extreme_contrast_index = 1.0 / (1.0 + np.exp(-extreme_z))
    trusted_extreme_load = tie_index * both_trustworthy * extreme_contrast_index
    precision *= 1.0 - float(
        parameters["trusted_extreme_precision_damping"]
    ) * trusted_extreme_load

    decision_variable = float(precision * total_evidence)

    # Symmetric logits ensure exact invariance when A and B are exchanged.
    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))


`parameters`:
- reliability_curvature: [2.0, 3.2]
- dispersion_threshold: [0.050, 0.082]
- dispersion_steepness: [32.0, 52.0]
- trust_tail: [0.0, 0.07]
- sampling_scale: [0.70, 1.30]
- sampling_floor: [0.72, 0.90]
- claim_floor: [0.16, 0.30]
- skeptical_orientation: [0, 1]
- skeptic_trait_spread: [0.50, 0.90]
- skeptical_gain: [2.6, 3.6]
- claim_accumulation: [1.25, 1.75]
- ambiguity_margin_scale: [2.2, 3.8]
- tally_power: [1.75, 2.35]
- tally_half_margin: [2.5, 3.4]
- tally_gain_trait: [0, 1]
- tally_trait_spread: [0.22, 0.48]
- tally_gain: [1.85, 2.35]
- mid_margin_center: [3.6, 4.4]
- mid_margin_width: [0.9, 1.3]
- mid_margin_tally_damping: [0.14, 0.24]
- arbitration_threshold: [0.28, 0.62]
- certainty_weight: [1.45, 2.10]
- disagreement_weight: [0.55, 1.05]
- dispersion_uncertainty_weight: [0.12, 0.38]
- margin_resolution_weight: [1.10, 1.65]
- dispersion_reference: [0.035, 0.065]
- gate_temperature: [0.16, 0.28]
- moderate_tie_gate_boost: [1.8, 2.8]
- moderate_contrast_center: [0.20, 0.27]
- moderate_contrast_width: [0.035, 0.070]
- precision_trait: [0, 1]
- precision_trait_spread: [0.28, 0.55]
- base_precision: [1.05, 1.35]
- small_margin_precision_damping: [0.50, 0.66]
- small_margin_decay: [0.75, 1.15]
- trusted_extreme_precision_damping: [0.35, 0.52]
- extreme_contrast_center: [0.62, 0.72]
- extreme_contrast_width: [0.045, 0.080]
- lapse_rate: [0.0, 0.055]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3004 (var=0.0051)
- Experiment 2: real=89.1422 (var=7.5896) vs this=156.3801 (var=2.2245)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.3973 (var=0.0029)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8381 (var=0.0062)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.4507 (var=0.0378)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8283 (var=0.0029)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.2380 (var=0.0051)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-5.1299 (var=0.0137)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.3307 (var=0.0044)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.8767 (var=0.0047)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.2394 (var=0.0066)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.1225 (var=0.0047)
- Experiment 13: real=0.0094 (var=0.0050) vs this=0.0016 (var=0.0084)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.1631 (var=0.0099)
- Experiment 15: real=0.0000 (var=0.2400) vs this=0.0000 (var=0.2100)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.5591 (var=0.0047)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=-0.3481 (var=0.0127)
- Experiment 18: real=0.0058 (var=0.0053) vs this=0.0112 (var=0.0045)
- Experiment 19: real=0.8617 (var=0.0116) vs this=0.8750 (var=0.0092)
- Experiment 20: real=0.8567 (var=0.0052) vs this=0.8512 (var=0.0024)
- Experiment 21: real=0.8667 (var=0.0085) vs this=0.7713 (var=0.0051)
- Experiment 22: real=-0.0058 (var=0.0033) vs this=-0.0321 (var=0.0055)
- Experiment 23: real=0.5000 (var=0.0088) vs this=0.4281 (var=0.0107)
- Experiment 24: real=0.5763 (var=0.0023) vs this=0.1106 (var=0.0028)
- Experiment 25: real=-0.3065 (var=0.0176) vs this=-0.0690 (var=0.0064)
- Experiment 26: real=0.8683 (var=0.0070) vs this=0.5133 (var=0.0104)


---

### `pi_13` (overall score: 0.547)

**Description**
Conflict-Geometry Recruitment theory proposes that people probabilistically recruit one of three representations according to current conflict geometry. Raw counting dominates increasingly decisive vote imbalances. Coalition-claim construal compresses homogeneous repetition into redundant option-level claims and produces a rapidly saturating skeptical response to communicated-authority contrasts, but it is recruited only after isolated endorsements have smoothly coalesced into an active coalition. Moreover, when coalition construal is selected before both opposing sides have formed stable multi-expert coalitions, its pragmatic reversal is proportionally weak rather than fully expressed. Exemplar construal preserves diagnostic experts within heterogeneous coalitions through sidewise upper-tail concentration, but is recruited chiefly when tally conflict remains unresolved. Its suppression grows convexly with tally decisiveness, allowing diagnostic cues to influence weak conflicts without overriding decisive majorities. Stable correlated subject propensities influence recruitment, while trial-level stochastic construal is marginalized into predicted choice probabilities. The process is history independent and exactly invariant under exchanging options A and B.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Conflict-Geometry Recruitment model. History is intentionally ignored:
    # choices are not followed by information that could update expert quality.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match n_features {n_features}."
        )

    # Positive votes favor A and negative votes favor B. Agreements are not
    # evidence for either displayed option.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smooth authority coding on the full communicated-validity continuum.
    validity = np.clip(validities, 0.5, 1.0)
    raw_authority = 2.0 * (validity - 0.5)
    curvature = float(parameters["validity_curvature"])
    authority = np.tanh(curvature * raw_authority) / max(
        float(np.tanh(curvature)), 1e-12
    )

    exemplar_temperature = float(parameters["exemplar_temperature"])

    def represent(side):
        members = authority[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        # Coalition claims encode mean authority. Consequently, homogeneous
        # repetition is representationally redundant rather than cumulative.
        claim = float(np.mean(members))
        variance = float(np.mean((members - claim) ** 2))
        heterogeneity = float(np.clip(4.0 * variance, 0.0, 1.0))

        # A normalized soft maximum retains diagnostic members without using a
        # hard winner or an endpoint category. It equals the common authority
        # in a homogeneous coalition and approaches its upper tail when mixed.
        centered = exemplar_temperature * (members - np.max(members))
        weights = np.exp(centered)
        weights /= np.sum(weights)
        exemplar = float(np.dot(weights, members))

        # Upper-tail concentration is computed separately within each side, so
        # diagnostic substructure cannot cancel across opposing coalitions.
        upper_tail = max(exemplar - claim, 0.0)
        return claim, exemplar, heterogeneity, upper_tail, breadth

    claim_a, exemplar_a, hetero_a, tail_a, breadth_a = represent(1.0)
    claim_b, exemplar_b, hetero_b, tail_b, breadth_b = represent(-1.0)

    if active_count == 0:
        claim_contrast = 0.0
        exemplar_contrast = 0.0
        pooled_heterogeneity = 0.0
        pooled_upper_tail = 0.0
        sparsity = 1.0
    else:
        claim_contrast = (claim_a - claim_b) / (
            float(parameters["claim_floor"]) + abs(claim_a) + abs(claim_b)
        )
        claim_contrast = float(np.clip(claim_contrast, -1.0, 1.0))

        exemplar_contrast = (exemplar_a - exemplar_b) / (
            float(parameters["exemplar_floor"])
            + abs(exemplar_a)
            + abs(exemplar_b)
        )
        exemplar_contrast = float(np.clip(exemplar_contrast, -1.0, 1.0))

        pooled_heterogeneity = (
            breadth_a * hetero_a + breadth_b * hetero_b
        ) / float(active_count)
        pooled_upper_tail = (
            breadth_a * tail_a + breadth_b * tail_b
        ) / float(active_count)
        sparsity = 1.0 - float(active_count) / float(max(n_features, 1))

    # Decisiveness requires both a large proportional imbalance and enough
    # absolute votes. This distinguishes a sparse one-vote conflict from a
    # substantial majority without introducing windows for particular margins.
    if active_count == 0:
        normalized_margin = 0.0
    else:
        normalized_margin = margin_magnitude / float(active_count)
    size_certainty = 1.0 - np.exp(
        -margin_magnitude / float(parameters["margin_scale"])
    )
    decisiveness = float(normalized_margin * size_certainty)

    # Signed validity geometry relative to the tally is invariant to swapping
    # A and B: both factors reverse under a swap. Positive values mean that
    # authority and the raw majority point in the same direction.
    if margin_magnitude < 1e-12:
        validity_tally_alignment = 0.0
    else:
        validity_tally_alignment = float(
            np.sign(tally_margin) * claim_contrast
        )

    diagnostic_concentration = abs(exemplar_contrast - claim_contrast)
    diagnostic_scale = float(parameters["diagnostic_scale"])
    diagnostic_extremity = (
        diagnostic_concentration ** 2
        / (diagnostic_scale ** 2 + diagnostic_concentration ** 2 + 1e-12)
    )

    # Sidewise upper-tail concentration only exists when a coalition contains
    # diagnostic internal structure. Gating by heterogeneity prevents a merely
    # homogeneous authority contrast from duplicating coalition-claim evidence.
    upper_tail_salience = (
        pooled_upper_tail ** 2
        / (diagnostic_scale ** 2 + pooled_upper_tail ** 2 + 1e-12)
    ) * np.sqrt(max(pooled_heterogeneity, 0.0))

    # Correlated hierarchical propensities. One shared style variable loads in
    # opposite directions on counting versus interpretive construals, while
    # representation-specific deviations retain individual flexibility.
    shared_style = float(parameters["shared_style"]) - 0.5
    loading = float(parameters["trait_loading"])
    count_trait = (
        -loading * shared_style + float(parameters["count_offset"])
    )
    skeptic_trait = (
        loading * shared_style + float(parameters["skeptic_offset"])
    )
    exemplar_trait = (
        0.55 * loading * shared_style + float(parameters["exemplar_offset"])
    )

    # Claim recruitment responds rapidly to authority contrast independently
    # of coalition breadth. Only an additional, smaller smooth term depends on
    # sparsity, and it is strongest for highly authoritative claims.
    absolute_claim = abs(claim_contrast)
    saturated_claim = 1.0 - np.exp(
        -float(parameters["claim_saturation"]) * absolute_claim
    )
    authority_extremity = max(abs(claim_a), abs(claim_b)) ** 4

    # Isolated endorsements do not automatically form a coalition-level claim.
    # The gate rises smoothly with the absolute number of discriminating
    # experts and is already close to saturation for two multi-expert sides.
    formation_scale = float(parameters["coalition_formation_scale"])
    formation_power = float(parameters["coalition_formation_power"])
    coalition_formation = 1.0 - np.exp(
        -(float(active_count) / formation_scale) ** formation_power
    )
    coalition_formation = float(np.clip(coalition_formation, 1e-8, 1.0))

    # Smooth recruitment utilities. Convex decisiveness suppression leaves
    # weak conflicts accessible to exemplars while strongly protecting large
    # majorities. Upper-tail recruitment is restricted to unresolved conflict.
    count_utility = (
        float(parameters["count_bias"])
        + count_trait
        + float(parameters["margin_recruitment"]) * decisiveness
        + float(parameters["alignment_recruitment"])
        * validity_tally_alignment
    )
    claim_utility = (
        float(parameters["claim_bias"])
        + skeptic_trait
        + float(parameters["claim_salience"]) * saturated_claim
        + float(parameters["sparsity_claim_gain"])
        * sparsity * authority_extremity
        + float(parameters["margin_recruitment"]) * (1.0 - decisiveness)
        - float(parameters["heterogeneity_escape"])
        * pooled_heterogeneity
        + float(parameters["coalition_formation_gain"])
        * np.log(coalition_formation)
    )
    exemplar_utility = (
        float(parameters["exemplar_bias"])
        + exemplar_trait
        + float(parameters["exemplar_heterogeneity"])
        * pooled_heterogeneity
        + (1.0 - decisiveness)
        * (
            float(parameters["exemplar_diagnostic"])
            * diagnostic_extremity
            + float(parameters["exemplar_extremity_gain"])
            * upper_tail_salience
        )
        - float(parameters["exemplar_margin_suppression"])
        * decisiveness ** 2
    )

    utilities = np.asarray(
        [count_utility, claim_utility, exemplar_utility], dtype=np.float64
    )
    utilities /= float(parameters["recruitment_temperature"])
    utilities -= np.max(utilities)
    recruitment = np.exp(utilities)
    recruitment /= np.sum(recruitment)

    # Each representation supplies a complete conditional choice distribution.
    # The count channel grows with margin, the coalition channel reverses a
    # rapidly saturating authority contrast, and the exemplar channel follows
    # the diagnostic upper-tail contrast.
    if margin_magnitude < 1e-12:
        count_evidence = 0.0
    else:
        count_evidence = (
            float(parameters["count_strength"])
            * np.sign(tally_margin)
            * margin_magnitude ** float(parameters["count_power"])
        )

    # Full pragmatic reversal requires both sides to be construed as formed
    # coalitions. Separate smooth side-formation terms make singleton conflicts
    # weaker while rapidly approaching one for established multi-expert sides.
    bilateral_scale = float(parameters["bilateral_formation_scale"])
    bilateral_power = float(parameters["bilateral_formation_power"])
    side_formation_a = 1.0 - np.exp(
        -(float(breadth_a) / bilateral_scale) ** bilateral_power
    )
    side_formation_b = 1.0 - np.exp(
        -(float(breadth_b) / bilateral_scale) ** bilateral_power
    )
    bilateral_core = np.sqrt(max(side_formation_a * side_formation_b, 0.0))
    bilateral_floor = float(parameters["bilateral_formation_floor"])
    bilateral_formation = bilateral_floor + (
        1.0 - bilateral_floor
    ) * bilateral_core

    skeptical_multiplier = max(0.20, 1.0 + 0.55 * skeptic_trait)
    claim_evidence = -float(parameters["skeptic_strength"]) * (
        bilateral_formation
        * skeptical_multiplier
        * np.tanh(float(parameters["claim_slope"]) * claim_contrast)
    )

    exemplar_multiplier = max(0.20, 1.0 + 0.45 * exemplar_trait)
    exemplar_evidence = float(parameters["exemplar_strength"]) * (
        exemplar_multiplier
        * np.tanh(
            float(parameters["exemplar_slope"]) * exemplar_contrast
        )
    )

    precision = float(parameters["choice_precision"])
    evidences = precision * np.asarray(
        [count_evidence, claim_evidence, exemplar_evidence],
        dtype=np.float64,
    )

    # Marginalize stochastic representation recruitment. Computing a mixture
    # of channel-specific choices, rather than averaging evidence, preserves
    # psychologically distinct construals and their trial-level competition.
    channel_p_a = np.empty(3, dtype=np.float64)
    for i, evidence in enumerate(evidences):
        channel_logits = np.asarray(
            [0.5 * evidence, -0.5 * evidence], dtype=np.float64
        )
        channel_logits -= np.max(channel_logits)
        channel_probs = np.exp(channel_logits)
        channel_probs /= np.sum(channel_probs)
        channel_p_a[i] = channel_probs[0]

    p_a = float(np.dot(recruitment, channel_p_a))
    probabilities = np.asarray([p_a, 1.0 - p_a], dtype=np.float64)

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.sum(probabilities)
    return probabilities


`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(
            probabilities.size,
            1.0 / probabilities.size,
            dtype=np.float64,
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))


`parameters`:
- validity_curvature: [1.2, 2.4]
- claim_floor: [0.055, 0.14]
- claim_slope: [2.2, 4.0]
- claim_saturation: [3.5, 6.0]
- exemplar_temperature: [7.0, 12.0]
- exemplar_floor: [0.08, 0.18]
- exemplar_slope: [2.0, 3.6]
- diagnostic_scale: [0.12, 0.22]
- margin_scale: [1.8, 3.2]
- count_power: [0.42, 0.60]
- count_strength: [0.52, 0.72]
- skeptic_strength: [2.5, 3.8]
- exemplar_strength: [5.0, 8.0]
- shared_style: [0, 1]
- count_offset: [-0.55, 0.55]
- skeptic_offset: [-0.55, 0.55]
- exemplar_offset: [-0.25, 0.25]
- trait_loading: [0.75, 1.30]
- count_bias: [-0.15, 0.30]
- claim_bias: [0.05, 0.45]
- exemplar_bias: [-1.25, -0.65]
- margin_recruitment: [2.2, 3.5]
- alignment_recruitment: [-0.35, 0.35]
- claim_salience: [0.30, 0.65]
- sparsity_claim_gain: [0.02, 0.15]
- heterogeneity_escape: [1.4, 2.8]
- coalition_formation_scale: [2.3, 2.8]
- coalition_formation_power: [2.8, 3.8]
- coalition_formation_gain: [0.75, 1.10]
- bilateral_formation_floor: [0.35, 0.55]
- bilateral_formation_scale: [1.15, 1.50]
- bilateral_formation_power: [1.8, 2.8]
- exemplar_heterogeneity: [1.2, 2.5]
- exemplar_diagnostic: [3.5, 5.5]
- exemplar_extremity_gain: [1.4, 2.8]
- exemplar_margin_suppression: [10.0, 15.0]
- recruitment_temperature: [0.65, 1.15]
- choice_precision: [0.90, 1.20]
- lapse_rate: [0.0, 0.055]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.2571 (var=0.0056)
- Experiment 2: real=89.1422 (var=7.5896) vs this=101.4886 (var=2.8840)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.2502 (var=0.0050)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8313 (var=0.0084)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.3333 (var=0.0486)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8612 (var=0.0028)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.0835 (var=0.0026)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-4.4524 (var=0.0353)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.2713 (var=0.0034)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.8358 (var=0.0117)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.0964 (var=0.0043)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.1078 (var=0.0040)
- Experiment 13: real=0.0094 (var=0.0050) vs this=0.0481 (var=0.0062)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.3596 (var=0.0395)
- Experiment 15: real=0.0000 (var=0.2400) vs this=0.0000 (var=0.2176)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.6663 (var=0.0061)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=-0.1138 (var=0.0119)
- Experiment 18: real=0.0058 (var=0.0053) vs this=0.0700 (var=0.0088)
- Experiment 19: real=0.8617 (var=0.0116) vs this=0.8958 (var=0.0071)
- Experiment 20: real=0.8567 (var=0.0052) vs this=0.8075 (var=0.0051)
- Experiment 21: real=0.8667 (var=0.0085) vs this=0.8703 (var=0.0098)
- Experiment 22: real=-0.0058 (var=0.0033) vs this=-0.0554 (var=0.0066)
- Experiment 23: real=0.5000 (var=0.0088) vs this=0.5337 (var=0.0069)
- Experiment 24: real=0.5763 (var=0.0023) vs this=0.5363 (var=0.0187)
- Experiment 25: real=-0.3065 (var=0.0176) vs this=0.4533 (var=0.0337)
- Experiment 26: real=0.8683 (var=0.0070) vs this=0.7600 (var=0.0287)


## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Choice rate for the Take-The-Best winner on TTB-vs-tally conflict trials."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # Validities are strictly descending in feature-index order, so the
        # first nonzero difference identifies the Take-The-Best winner.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        ttb_winner = 0 if diff[discriminating[0]] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Retain only directional conflicts between the two heuristics.
        if ttb_winner != tally_winner:
            aligned.append(float(int(row["response"]) == ttb_winner))

    # The selected design always contains conflict trials; the fallback keeps
    # the function defined for any unexpected slice.
    return float(np.mean(aligned)) if aligned else 0.5
```

**Observed (real) value:** 0.4033 (var=0.0042)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)
- pi_8: 0.3383 (var=0.0039)
- pi_9: 0.3208 (var=0.0107)
- pi_10: 0.3250 (var=0.0112)
- pi_11: 0.4133 (var=0.0100)
- pi_12: 0.3004 (var=0.0051)
- pi_13: 0.2571 (var=0.0056)
- pi_14: 0.3292 (var=0.0051)

### Experiment 2
**Design**
  A=[1, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Total log marginal-likelihood ratio favoring Tallying over TTB.
    # The parameter grids approximate uniform priors over the stated ranges.
    beta_grid = np.linspace(0.1, 20.0, 120)
    epsilon_grid = np.linspace(0.0, 0.5, 61)
    beta = beta_grid[:, None]
    epsilon = epsilon_grid[None, :]

    def logmeanexp(x):
        x = np.asarray(x, dtype=float)
        m = float(np.max(x))
        return float(m + np.log(np.mean(np.exp(x - m))))

    def subject_log_bf(df):
        margins = []
        correct = []

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            a_wins = int(np.sum(a > b))
            b_wins = int(np.sum(b > a))
            margin = abs(a_wins - b_wins)

            if margin == 0:
                continue

            winner = 0 if a_wins > b_wins else 1
            margins.append(float(margin))
            correct.append(float(int(row["response"]) == winner))

        if len(margins) == 0:
            return np.nan

        margins = np.asarray(margins, dtype=float)
        correct = np.asarray(correct, dtype=float)
        tiny = 1e-12

        # TTB assigns the same winner probability to every non-tied pair.
        core_ttb = 1.0 / (1.0 + np.exp(-beta))
        p_ttb = (1.0 - epsilon) * core_ttb + epsilon * 0.5
        p_ttb = np.clip(p_ttb, tiny, 1.0 - tiny)
        n_correct = float(np.sum(correct))
        n_error = float(correct.size - n_correct)
        ll_ttb = n_correct * np.log(p_ttb) + n_error * np.log1p(-p_ttb)

        # Tallying's winner probability depends on the observed tally margin.
        ll_tally = np.zeros_like(p_ttb, dtype=float)
        for d, y in zip(margins, correct):
            x = beta * d
            core = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
            p = (1.0 - epsilon) * core + epsilon * 0.5
            p = np.clip(p, tiny, 1.0 - tiny)
            ll_tally += y * np.log(p) + (1.0 - y) * np.log1p(-p)

        log_ml_tally = logmeanexp(ll_tally)
        log_ml_ttb = logmeanexp(ll_ttb)
        return float(log_ml_tally - log_ml_ttb)

    if "subject_id" in data.columns:
        values = [subject_log_bf(g) for _, g in data.groupby("subject_id", sort=False)]
    else:
        values = [subject_log_bf(data)]

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    # Independent participants contribute additive evidence. On a one-subject
    # slice this is simply that participant's log Bayes factor.
    return float(np.sum(values))

```

**Observed (real) value:** 89.1422 (var=7.5896)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)
- pi_8: 56.1425 (var=1.2063)
- pi_9: 59.9467 (var=3.7118)
- pi_10: 89.8046 (var=9.1438)
- pi_11: 51.5068 (var=3.0531)
- pi_12: 156.3801 (var=2.2245)
- pi_13: 101.4886 (var=2.8840)
- pi_14: 187.4611 (var=2.5841)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with fewer positive ratings."""
    if data is None or len(data) == 0:
        return float("nan")

    minority_choices = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        tally_margin_a = float(np.sum(a > b) - np.sum(b > a))
        r = int(response)

        if tally_margin_a < 0:       # A has fewer cue wins
            minority_choices.append(float(r == 0))
        elif tally_margin_a > 0:     # B has fewer cue wins
            minority_choices.append(float(r == 1))
        # Tied tallies have no minority option and are omitted.

    if len(minority_choices) == 0:
        return float("nan")
    return float(np.mean(minority_choices))
```

**Observed (real) value:** 0.4508 (var=0.0031)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)
- pi_8: 0.4106 (var=0.0035)
- pi_9: 0.3440 (var=0.0028)
- pi_10: 0.3654 (var=0.0036)
- pi_11: 0.3771 (var=0.0040)
- pi_12: 0.3973 (var=0.0029)
- pi_13: 0.2502 (var=0.0050)
- pi_14: 0.3169 (var=0.0017)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    a = np.vstack([np.asarray(x, dtype=float) for x in data["option_a_ratings"]])
    b = np.vstack([np.asarray(x, dtype=float) for x in data["option_b_ratings"]])
    responses = np.asarray(data["response"], dtype=int)

    # Positive margin means that A wins more binary feature comparisons;
    # negative margin means that B does.
    delta = a - b
    tally_margin = np.sum(delta > 0, axis=1) - np.sum(delta < 0, axis=1)

    # Select the two orientations of the diagnostic 6-versus-1 conflict:
    # the tally winner has a five-cue margin but is opposed by expert 1.
    strongest_cue = delta[:, 0]
    critical = (np.abs(tally_margin) == 5) & (strongest_cue * tally_margin < 0)
    if not np.any(critical):
        return float("nan")

    chose_a = responses == 0
    tally_winner_is_a = tally_margin > 0

    # Rate at which the numerous weak experts override the strongest expert.
    return float(np.mean(chose_a[critical] == tally_winner_is_a[critical]))
```

**Observed (real) value:** 0.8213 (var=0.0169)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)
- pi_8: 0.8544 (var=0.0058)
- pi_9: 0.9294 (var=0.0023)
- pi_10: 0.9350 (var=0.0031)
- pi_11: 0.8800 (var=0.0067)
- pi_12: 0.8381 (var=0.0062)
- pi_13: 0.8313 (var=0.0084)
- pi_14: 0.8056 (var=0.0058)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as choosing the option endorsed by the 95%-valid
    # expert (feature 0), irrespective of whether that option is A or B.
    by_size = {1: [], 2: [], 4: []}

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        difference = a - b

        # In this design, coalition size is the number of experts endorsing A
        # over B (equal to the number endorsing B over A).
        coalition_size = int(np.sum(difference > 0.0))
        if coalition_size not in by_size:
            continue

        high_expert_option = 0 if difference[0] > 0.0 else 1
        chose_high_expert = float(int(row['response']) == high_expert_option)
        by_size[coalition_size].append(chose_high_expert)

    if any(len(by_size[k]) == 0 for k in (1, 2, 4)):
        return float('nan')

    p1 = float(np.mean(by_size[1]))
    p2 = float(np.mean(by_size[2]))
    p4 = float(np.mean(by_size[4]))

    # Approximate inverse-noise weighting of the conditions with the largest
    # predicted between-theory probability differences. The size-3 condition
    # is omitted because both theories predict it close to chance.
    return float(1.8 * (p1 - 0.5) + 0.9 * (p2 - 0.5) - (p4 - 0.5))
```

**Observed (real) value:** -0.0382 (var=0.0352)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)
- pi_8: 0.0104 (var=0.0390)
- pi_9: -0.4666 (var=0.0908)
- pi_10: -0.4282 (var=0.0521)
- pi_11: -0.3760 (var=0.0522)
- pi_12: -0.4507 (var=0.0378)
- pi_13: -0.3333 (var=0.0486)
- pi_14: -0.1537 (var=0.0452)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the option with the larger raw vote tally."""
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=np.float64)
        b = np.asarray(b_cell, dtype=np.float64)
        margin = float(np.sum(a - b))
        r = int(response)

        if margin > 0.0:
            scores.append(1.0 if r == 0 else 0.0)
        elif margin < 0.0:
            scores.append(1.0 if r == 1 else 0.0)
        else:
            scores.append(0.5)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8462 (var=0.0122)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)
- pi_8: 0.8592 (var=0.0031)
- pi_9: 0.9035 (var=0.0023)
- pi_10: 0.9150 (var=0.0039)
- pi_11: 0.8948 (var=0.0037)
- pi_12: 0.8283 (var=0.0029)
- pi_13: 0.8612 (var=0.0028)
- pi_14: 0.8171 (var=0.0013)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response selected the option endorsed by the
    # high-validity experts (features 0 through 6), irrespective of screen side.
    weighted_success = 0.0
    total_weight = 0.0

    # Increasing weights emphasize the larger-coalition conditions, where the
    # theories' predicted choice probabilities diverge most, while retaining
    # lower-m conditions to keep the subject-level estimate stable.
    weights = np.asarray([0.30, 0.55, 0.72, 0.85, 0.94, 1.00], dtype=float)

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"],
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        high_margin = float(np.sum(a[:7] - b[:7]))
        m = int(round(abs(high_margin)))
        if high_margin == 0.0 or m < 2 or m > 7:
            continue

        chose_a = int(response) == 0
        high_option_is_a = high_margin > 0.0
        chose_high_coalition = float(chose_a == high_option_is_a)

        weight = float(weights[m - 2])
        weighted_success += weight * chose_high_coalition
        total_weight += weight

    if total_weight == 0.0:
        return float("nan")
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.1161 (var=0.0079)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)
- pi_8: 0.2347 (var=0.0051)
- pi_9: 0.3269 (var=0.0321)
- pi_10: 0.0811 (var=0.0040)
- pi_11: 0.1813 (var=0.0070)
- pi_12: 0.2380 (var=0.0051)
- pi_13: 0.0835 (var=0.0026)
- pi_14: 0.0942 (var=0.0009)

### Experiment 8
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    # Bounded reliability code specified by Compressed-Reliability Consensus.
    validities = np.array([0.5, 0.6, 0.8, 1.0], dtype=np.float64)
    reliability_code = np.tanh(3.0 * (validities - 0.5)) / np.tanh(1.5)

    numerators = []
    weights = []
    subjects = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.float64)
        b = np.asarray(row["option_b_ratings"], dtype=np.float64)
        contrast = float(np.dot(a - b, reliability_code))
        weight = abs(contrast)
        if not np.isfinite(weight) or weight <= 0.0:
            continue

        # +1 means the response followed the more reliable endorser, and -1
        # means it followed the less reliable endorser. This is invariant to
        # whether that endorser appeared on option A or option B.
        choice_sign = 1.0 - 2.0 * float(row["response"])
        aligned_choice = np.sign(contrast) * choice_sign
        numerators.append(weight * aligned_choice)
        weights.append(weight)
        subjects.append(row["subject_id"])

    if len(weights) == 0:
        return 0.0

    work = pd.DataFrame({
        "subject_id": subjects,
        "numerator": numerators,
        "weight": weights,
    })
    grouped = work.groupby("subject_id", sort=False)[["numerator", "weight"]].sum()
    valid = grouped["weight"] > 0.0
    if not bool(valid.any()):
        return 0.0

    subject_scores = (
        grouped.loc[valid, "numerator"] / grouped.loc[valid, "weight"]
    ).to_numpy(dtype=np.float64)

    # Normalized cumulative directional evidence: for one subject this is the
    # subject's weighted alignment score; pooled evidence grows with sqrt(N).
    return float(np.sqrt(subject_scores.size) * np.mean(subject_scores))

```

**Observed (real) value:** -3.5372 (var=0.0438)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)
- pi_8: -4.4717 (var=0.0102)
- pi_9: -2.4502 (var=0.0542)
- pi_10: -3.8304 (var=0.0211)
- pi_11: -4.5864 (var=0.0269)
- pi_12: -5.1299 (var=0.0137)
- pi_13: -4.4524 (var=0.0353)
- pi_14: -4.6563 (var=0.0108)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    weighted_success = 0.0
    total_weight = 0.0

    # Conditions are represented in a canonical orientation using the sign of
    # the overall tally. This makes exact A/B-reversed pairs contribute to the
    # same planned condition.
    condition_weights = {
        7: 2.00,
        3: 1.45,
        -3: 0.80,
        -6: 1.10,
    }

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=np.float64)
        b = np.asarray(row['option_b_ratings'], dtype=np.float64)
        if a.size < 7 or b.size != a.size:
            continue

        signed_votes = np.sign(a - b)
        high_validity_margin = float(np.sum(signed_votes[:7]))
        tally_margin = float(np.sum(signed_votes))
        if high_validity_margin == 0.0 or tally_margin == 0.0:
            continue

        canonical_margin = int(round(high_validity_margin * np.sign(tally_margin)))
        weight = condition_weights.get(canonical_margin, 1.0)

        response = int(row['response'])
        chose_high_validity_side = (
            (high_validity_margin > 0.0 and response == 0) or
            (high_validity_margin < 0.0 and response == 1)
        )
        weighted_success += weight * float(chose_high_validity_side)
        total_weight += weight

    if total_weight == 0.0:
        return float('nan')
    return float(weighted_success / total_weight)
```

**Observed (real) value:** 0.3248 (var=0.0018)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)
- pi_8: 0.2814 (var=0.0029)
- pi_9: 0.3376 (var=0.0157)
- pi_10: 0.1905 (var=0.0047)
- pi_11: 0.2755 (var=0.0039)
- pi_12: 0.3307 (var=0.0044)
- pi_13: 0.2713 (var=0.0034)
- pi_14: 0.1516 (var=0.0017)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # The concentrated endpoint condition has exactly two discriminating
        # experts: one 100%-valid expert and the final 50%-valid expert.
        if np.count_nonzero(diff) != 2 or diff.size < 8 or diff[7] == 0:
            continue

        response = int(row["response"])
        low_validity_side = 0 if diff[7] > 0 else 1
        scores.append(float(response == low_validity_side))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0104)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)
- pi_8: 0.8408 (var=0.0059)
- pi_9: 0.7475 (var=0.0177)
- pi_10: 0.8658 (var=0.0100)
- pi_11: 0.8533 (var=0.0093)
- pi_12: 0.8767 (var=0.0047)
- pi_13: 0.8358 (var=0.0117)
- pi_14: 0.8267 (var=0.0060)

### Experiment 11
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b

        # The first five experts have 70% validity. Restrict attention to the
        # diffuse conflicts (3-vs-2, 4-vs-3, and 5-vs-4), where at least five
        # experts discriminate between the products.
        discriminating_count = int(np.count_nonzero(diff))
        if discriminating_count < 5:
            continue

        high_validity_margin = float(np.sum(diff[:5]))
        if high_validity_margin > 0:
            chose_high_validity_coalition = int(row['response']) == 0
        elif high_validity_margin < 0:
            chose_high_validity_coalition = int(row['response']) == 1
        else:
            continue

        scores.append(float(chose_high_validity_coalition))

    if not scores:
        return float('nan')
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1572 (var=0.0077)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)
- pi_8: 0.2039 (var=0.0054)
- pi_9: 0.4628 (var=0.0171)
- pi_10: 0.1292 (var=0.0058)
- pi_11: 0.1917 (var=0.0067)
- pi_12: 0.2394 (var=0.0066)
- pi_13: 0.0964 (var=0.0043)
- pi_14: 0.0842 (var=0.0012)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    conventional_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # Select the 12 mirrored conditions comparing equal-sized coalitions
        # of 70%-valid experts (indices 0:6) and 50%-valid experts (6:12),
        # excluding all conditions involving 95%-valid experts.
        d70 = d[:6]
        d50 = d[6:12]
        d95 = d[12:15]
        n70 = int(np.count_nonzero(d70))
        n50 = int(np.count_nonzero(d50))
        s70 = float(np.sum(d70))
        s50 = float(np.sum(d50))

        is_target = (
            n70 > 0
            and n70 == n50
            and abs(s70) == n70
            and abs(s50) == n50
            and np.sign(s70) == -np.sign(s50)
            and np.count_nonzero(d95) == 0
        )
        if not is_target:
            continue

        response = int(row["response"])
        # Score 1 when the chosen product is endorsed by the 70%-valid
        # coalition, regardless of whether that product appears as A or B.
        chose_70_option = ((s70 > 0 and response == 0) or
                           (s70 < 0 and response == 1))
        conventional_choices.append(float(chose_70_option))

    if len(conventional_choices) == 0:
        return float("nan")
    return float(np.mean(conventional_choices))
```

**Observed (real) value:** 0.1261 (var=0.0089)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)
- pi_8: 0.1583 (var=0.0037)
- pi_9: 0.3469 (var=0.0105)
- pi_10: 0.1528 (var=0.0051)
- pi_11: 0.1503 (var=0.0048)
- pi_12: 0.1225 (var=0.0047)
- pi_13: 0.1078 (var=0.0040)
- pi_14: 0.0944 (var=0.0011)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subject_contrast(df):
        successes = np.zeros(3, dtype=np.float64)
        totals = np.zeros(3, dtype=np.float64)

        for _, row in df.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=np.float64)
            b = np.asarray(row["option_b_ratings"], dtype=np.float64)
            signed = np.sign(a - b)

            # Exactly one of the first three (90%, 95%, 100%) experts is
            # discriminating in each designed trial.
            focal = np.flatnonzero(signed[:3] != 0.0)
            if focal.size != 1:
                continue
            condition = int(focal[0])
            focal_sign = float(signed[condition])
            response = int(row["response"])

            # Score choosing the option endorsed by the focal high-validity
            # expert, making exact A/B reversals contribute on the same scale.
            chose_focal_option = (
                (focal_sign > 0.0 and response == 0)
                or (focal_sign < 0.0 and response == 1)
            )
            successes[condition] += float(chose_focal_option)
            totals[condition] += 1.0

        if np.any(totals == 0.0):
            return np.nan
        rates = successes / totals
        return float(rates[1] - 0.5 * (rates[0] + rates[2]))

    if "subject_id" in data.columns and data["subject_id"].nunique() > 1:
        values = []
        for _, subj_df in data.groupby("subject_id", sort=False):
            value = subject_contrast(subj_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return subject_contrast(data)

```

**Observed (real) value:** 0.0094 (var=0.0050)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0569 (var=0.0085)
- pi_6: -0.0147 (var=0.0080)
- pi_1: -0.0147 (var=0.0061)
- pi_2: 0.0028 (var=0.0045)
- pi_3: -0.0878 (var=0.0068)
- pi_4: -0.0187 (var=0.0085)
- pi_5: 0.0069 (var=0.0102)
- pi_7: -0.0338 (var=0.0098)
- pi_9: -0.0147 (var=0.0077)
- pi_10: 0.0109 (var=0.0179)
- pi_11: 0.0403 (var=0.0095)
- pi_12: 0.0016 (var=0.0084)
- pi_13: 0.0481 (var=0.0062)
- pi_14: -0.0197 (var=0.0084)

### Experiment 14
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Preference for the option not endorsed by the perfectly valid expert,
    # scored separately at each coalition breadth and display orientation.
    cell_values = {(m, side): [] for m in range(1, 7) for side in (-1, 1)}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size == 0 or b.size != a.size:
            continue

        endpoint_side = int(np.sign(a[0] - b[0]))
        if endpoint_side == 0:
            continue

        active_count = int(np.count_nonzero(a - b))
        m = active_count // 2
        if m < 1 or m > 6 or active_count != 2 * m:
            continue

        response = int(row["response"])
        # endpoint_side=+1 means the certainty expert endorses A, so choosing
        # B is the anti-endpoint choice. For endpoint_side=-1, choosing A is.
        anti_endpoint_choice = float(
            (endpoint_side == 1 and response == 1)
            or (endpoint_side == -1 and response == 0)
        )
        cell_values[(m, endpoint_side)].append(anti_endpoint_choice)

    breadth_means = []
    for m in range(1, 7):
        orientation_means = []
        for side in (-1, 1):
            values = cell_values[(m, side)]
            if len(values) == 0:
                return float("nan")
            orientation_means.append(float(np.mean(values)))
        breadth_means.append(float(np.mean(orientation_means)))

    # A normalized linear low-to-high coalition-breadth contrast. Positive
    # values indicate that anti-endpoint preference attenuates as m increases.
    weights = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5], dtype=float)
    return float(np.dot(weights, np.asarray(breadth_means)) / 4.5)

```

**Observed (real) value:** 0.1781 (var=0.0129)
**Other theories' values on this metric (for reference):**
- pi_6: -0.0096 (var=0.0069)
- pi_8: 0.1467 (var=0.0083)
- pi_1: 0.0031 (var=0.0052)
- pi_2: 0.0064 (var=0.0127)
- pi_3: 0.0007 (var=0.0000)
- pi_4: -0.0000 (var=0.0117)
- pi_5: -0.0346 (var=0.0125)
- pi_7: -0.0029 (var=0.0065)
- pi_9: 0.1044 (var=0.0094)
- pi_10: 0.1275 (var=0.0095)
- pi_11: 0.1763 (var=0.0121)
- pi_12: 0.1631 (var=0.0099)
- pi_13: 0.3596 (var=0.0395)
- pi_14: -0.0294 (var=0.0060)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_endpoint_choices = []
    high_endpoint_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 5 or b.size != a.size:
            continue

        diff = a - b
        active = np.flatnonzero(diff != 0)
        if active.size != 2 or 0 not in active:
            continue

        opponents = active[active != 0]
        if opponents.size != 1:
            continue
        opponent = int(opponents[0])
        if diff[0] * diff[opponent] >= 0:
            continue

        response = int(row["response"])
        endpoint_chosen = (
            (diff[0] > 0 and response == 0)
            or (diff[0] < 0 and response == 1)
        )

        if opponent in (1, 2):
            low_endpoint_choices.append(float(endpoint_chosen))
        elif opponent == 4:
            high_endpoint_choices.append(float(endpoint_chosen))

    if len(low_endpoint_choices) == 0 or len(high_endpoint_choices) == 0:
        return float("nan")

    low_rate = float(np.mean(low_endpoint_choices))
    high_rate = float(np.mean(high_endpoint_choices))

    return float((low_rate < 0.5) and (high_rate > 0.5))

```

**Observed (real) value:** 0.0000 (var=0.2400)
**Other theories' values on this metric (for reference):**
- pi_8: 1.0000 (var=0.1600)
- pi_9: 0.0000 (var=0.1204)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.1600)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.1204)
- pi_5: 0.0000 (var=0.0900)
- pi_6: 1.0000 (var=0.2484)
- pi_7: 0.0000 (var=0.0000)
- pi_10: 0.0000 (var=0.1716)
- pi_11: 0.0000 (var=0.1476)
- pi_12: 0.0000 (var=0.2100)
- pi_13: 0.0000 (var=0.2176)
- pi_14: 1.0000 (var=0.2400)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    selected_choices = []

    # Coalitions opposed to the four 75%-valid experts. The maximally
    # dispersed coalition is intentionally excluded because the competing
    # theory predicts that its anti-reliability signal is strongly diluted.
    homogeneous = (4, 5, 6, 7)
    moderate_dispersion = (8, 9, 10, 11)
    selected = {homogeneous, moderate_dispersion}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        response = int(row["response"])

        # Locate the option endorsed by the four 75%-valid experts.
        a_has_low = int(np.sum(a[:4])) > int(np.sum(b[:4]))
        low_option = a if a_has_low else b
        opposing_option = b if a_has_low else a

        # Require the expected four-expert low-validity coalition and select
        # the two conditions where the theories' absolute predictions diverge.
        low_set = tuple(np.flatnonzero(low_option).tolist())
        opposing_set = tuple(np.flatnonzero(opposing_option).tolist())
        if low_set != (0, 1, 2, 3) or opposing_set not in selected:
            continue

        chose_low = (response == 0) if a_has_low else (response == 1)
        selected_choices.append(float(chose_low))

    if len(selected_choices) == 0:
        return 0.0
    return float(np.mean(selected_choices))

```

**Observed (real) value:** 0.8819 (var=0.0059)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5850 (var=0.0057)
- pi_8: 0.7128 (var=0.0076)
- pi_1: 0.1353 (var=0.0104)
- pi_2: 0.5084 (var=0.0042)
- pi_3: 0.2400 (var=0.0050)
- pi_4: 0.5028 (var=0.0039)
- pi_5: 0.5072 (var=0.0053)
- pi_6: 0.4859 (var=0.0035)
- pi_7: 0.7762 (var=0.0258)
- pi_10: 0.7981 (var=0.0069)
- pi_11: 0.6425 (var=0.0062)
- pi_12: 0.5591 (var=0.0047)
- pi_13: 0.6663 (var=0.0061)
- pi_14: 0.5331 (var=0.0030)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the dispersion gradient after orienting every trial toward the
    # homogeneous coalition of four 75%-valid experts (features 0--3).
    cells = {("low", "A"): [], ("low", "B"): [],
             ("high", "A"): [], ("high", "B"): []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=np.int64)
        b = np.asarray(row["option_b_ratings"], dtype=np.int64)
        if a.size < 16 or b.size < 16:
            continue

        a_is_reference = bool(np.all(a[:4] == 1) and np.all(a[4:] == 0))
        b_is_reference = bool(np.all(b[:4] == 1) and np.all(b[4:] == 0))

        if a_is_reference == b_is_reference:
            continue

        if a_is_reference:
            reference_side = "A"
            competitor = b
            chose_reference = float(int(row["response"]) == 0)
        else:
            reference_side = "B"
            competitor = a
            chose_reference = float(int(row["response"]) == 1)

        if np.all(competitor[4:8] == 1) and int(np.sum(competitor)) == 4:
            level = "low"
        elif np.all(competitor[12:16] == 1) and int(np.sum(competitor)) == 4:
            level = "high"
        else:
            # The moderately dispersed coalition is not needed for the
            # endpoint contrast.
            continue

        cells[(level, reference_side)].append(chose_reference)

    means = {}
    for key, values in cells.items():
        if len(values) == 0:
            return float("nan")
        means[key] = float(np.mean(np.asarray(values, dtype=np.float64)))

    # Average mirrored, side-specific contrasts so fixed A/B response bias
    # cannot masquerade as a dispersion effect.
    gradient_a = means[("high", "A")] - means[("low", "A")]
    gradient_b = means[("high", "B")] - means[("low", "B")]
    return float(0.5 * (gradient_a + gradient_b))
```

**Observed (real) value:** -0.3250 (var=0.0394)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0656 (var=0.0171)
- pi_10: -0.0869 (var=0.0165)
- pi_1: -0.6887 (var=0.0481)
- pi_2: -0.0075 (var=0.0180)
- pi_3: -0.0756 (var=0.0152)
- pi_4: -0.0106 (var=0.0148)
- pi_5: 0.0225 (var=0.0162)
- pi_6: -0.0081 (var=0.0173)
- pi_7: 0.3625 (var=0.0685)
- pi_9: 0.0256 (var=0.0200)
- pi_11: -0.1862 (var=0.0175)
- pi_12: -0.3481 (var=0.0127)
- pi_13: -0.1138 (var=0.0119)
- pi_14: -0.3506 (var=0.0116)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    singleton_choices = []
    broad_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # In this design, the first six experts have validity .80 and the
        # final six have validity .55. Determine which displayed option is
        # endorsed by the higher-validity coalition.
        high_direction = float(np.sum(a[:6] - b[:6]))
        m = int(np.sum(np.abs(a[:6] - b[:6])))
        if high_direction == 0.0:
            continue

        # Code 1 when the lower-validity coalition's option was selected,
        # invariant to whether that option was displayed as A or B.
        if high_direction > 0.0:
            lower_validity_choice = float(response == 1)
        else:
            lower_validity_choice = float(response == 0)

        if m == 1:
            singleton_choices.append(lower_validity_choice)
        elif m in (4, 6):
            broad_choices.append(lower_validity_choice)

    if len(singleton_choices) == 0 or len(broad_choices) == 0:
        return float("nan")

    return float(np.mean(broad_choices) - np.mean(singleton_choices))
```

**Observed (real) value:** 0.0058 (var=0.0053)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1996 (var=0.0174)
- pi_8: 0.0329 (var=0.0084)
- pi_1: 0.0217 (var=0.0047)
- pi_2: -0.0258 (var=0.0192)
- pi_3: -0.2842 (var=0.0112)
- pi_4: 0.0117 (var=0.0128)
- pi_5: -0.0238 (var=0.0143)
- pi_6: 0.0196 (var=0.0149)
- pi_7: 0.0146 (var=0.0242)
- pi_9: 0.0392 (var=0.0136)
- pi_11: -0.0058 (var=0.0109)
- pi_12: 0.0112 (var=0.0045)
- pi_13: 0.0700 (var=0.0088)
- pi_14: 0.0896 (var=0.0063)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 20 or b.size < 20:
            continue

        active = np.abs(a - b) > 0.5
        # Select the maximally dispersed 50/50/100/100 condition and its
        # option-reversed counterpart: four conflicts at indices 0:4 and four
        # at indices 16:20, with no other active features.
        is_extreme = (
            int(np.sum(active[:4])) == 4
            and int(np.sum(active[16:20])) == 4
            and int(np.sum(active)) == 8
        )
        if not is_extreme:
            continue

        # The heterogeneous coalition is the option endorsed at indices 16:20.
        hetero_is_a = float(np.sum(a[16:20])) > float(np.sum(b[16:20]))
        response = int(row["response"])
        chose_heterogeneous = (response == 0) if hetero_is_a else (response == 1)
        scores.append(float(chose_heterogeneous))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8617 (var=0.0116)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6850 (var=0.0125)
- pi_10: 0.4383 (var=0.0097)
- pi_1: 0.8400 (var=0.0148)
- pi_2: 0.4925 (var=0.0099)
- pi_3: 0.9967 (var=0.0001)
- pi_4: 0.5492 (var=0.0114)
- pi_5: 0.4933 (var=0.0092)
- pi_6: 0.5200 (var=0.0113)
- pi_7: 0.1133 (var=0.0090)
- pi_8: 0.3575 (var=0.0132)
- pi_9: 0.3425 (var=0.0165)
- pi_12: 0.8750 (var=0.0092)
- pi_13: 0.8958 (var=0.0071)
- pi_14: 0.8000 (var=0.0069)

### Experiment 20
**Design**
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Majority-consistent choice rate on the three largest tally margins."""
    if len(data) == 0:
        return float("nan")

    margins = np.asarray([
        float(np.sum(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
        for a, b in zip(data["option_a_ratings"], data["option_b_ratings"])
    ])
    responses = data["response"].to_numpy(dtype=int)

    # In this design, this selects margins 5, 7, and 9 and their reversals.
    selected = np.abs(margins) >= 5.0
    if not np.any(selected):
        return float("nan")

    # A is majority-supported for positive margins; B for negative margins.
    majority_choice = np.where(margins > 0.0, responses == 0, responses == 1)
    return float(np.mean(majority_choice[selected]))
```

**Observed (real) value:** 0.8567 (var=0.0052)
**Other theories' values on this metric (for reference):**
- pi_10: 0.9071 (var=0.0048)
- pi_11: 0.7683 (var=0.0169)
- pi_1: 0.8667 (var=0.0078)
- pi_2: 0.8796 (var=0.0088)
- pi_3: 0.5863 (var=0.0044)
- pi_4: 0.8721 (var=0.0046)
- pi_5: 0.8692 (var=0.0022)
- pi_6: 0.8175 (var=0.0031)
- pi_7: 0.8971 (var=0.0021)
- pi_8: 0.8579 (var=0.0046)
- pi_9: 0.9187 (var=0.0024)
- pi_12: 0.8512 (var=0.0024)
- pi_13: 0.8075 (var=0.0051)
- pi_14: 0.8633 (var=0.0030)

### Experiment 21
**Design**
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the probability of choosing the lower-validity (50%) coalition
    # in the five intermediate conditions: 51.25%, 52%, 52.75%, 53.5%, 55%.
    lower_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        difference = a - b

        # The nonbaseline coalition occupies one adjacent feature pair. The
        # selected range corresponds to feature indices 4 through 13.
        active_later = np.flatnonzero(np.abs(difference[2:]) > 0.5) + 2
        if active_later.size == 0:
            continue
        pair_start = int(np.min(active_later))
        if pair_start < 4 or pair_start > 12:
            continue

        # Positive means the more-valid coalition endorses A; negative means
        # it endorses B. Align responses so 1 always denotes selection of the
        # opposing, 50%-valid coalition.
        high_side = float(np.sum(difference[2:]))
        response = int(row["response"])
        if high_side > 0:
            lower_choices.append(float(response == 1))
        elif high_side < 0:
            lower_choices.append(float(response == 0))

    if len(lower_choices) == 0:
        return float("nan")
    return float(np.mean(lower_choices))
```

**Observed (real) value:** 0.8667 (var=0.0085)
**Other theories' values on this metric (for reference):**
- pi_11: 0.6383 (var=0.0061)
- pi_12: 0.7713 (var=0.0051)
- pi_1: 0.1560 (var=0.0099)
- pi_2: 0.5060 (var=0.0040)
- pi_3: 0.4937 (var=0.0044)
- pi_4: 0.4840 (var=0.0040)
- pi_5: 0.4897 (var=0.0034)
- pi_6: 0.6213 (var=0.0050)
- pi_7: 0.3313 (var=0.0056)
- pi_8: 0.6583 (var=0.0050)
- pi_9: 0.5213 (var=0.0048)
- pi_10: 0.5533 (var=0.0037)
- pi_13: 0.8703 (var=0.0098)
- pi_14: 0.8800 (var=0.0026)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response is skeptical: choosing the option opposite
    # the coalition containing the informative (> .50-valid) experts.
    ordinary_scores = []
    endpoint_scores = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])
        signed = a - b

        ordinary_direction = float(np.sum(signed[:6]))   # 55%-valid experts
        endpoint_direction = float(np.sum(signed[6:12])) # 100%-valid experts

        if abs(ordinary_direction) > 1e-12:
            # Positive means the stronger coalition supports A, so choosing B
            # is the skeptical response. Negative reverses that coding.
            skeptical = response if ordinary_direction > 0 else 1 - response
            ordinary_scores.append(float(skeptical))
        elif abs(endpoint_direction) > 1e-12:
            skeptical = response if endpoint_direction > 0 else 1 - response
            endpoint_scores.append(float(skeptical))

    if len(ordinary_scores) == 0 or len(endpoint_scores) == 0:
        return float("nan")

    return float(np.mean(ordinary_scores) - np.mean(endpoint_scores))
```

**Observed (real) value:** -0.0058 (var=0.0033)
**Other theories' values on this metric (for reference):**
- pi_12: -0.0321 (var=0.0055)
- pi_11: -0.1446 (var=0.0057)
- pi_1: -0.0183 (var=0.0028)
- pi_2: 0.0004 (var=0.0094)
- pi_3: 0.4383 (var=0.0047)
- pi_4: 0.0229 (var=0.0117)
- pi_5: 0.0138 (var=0.0150)
- pi_6: -0.0417 (var=0.0067)
- pi_7: -0.6117 (var=0.0473)
- pi_8: -0.0883 (var=0.0095)
- pi_9: -0.1883 (var=0.0185)
- pi_10: -0.2504 (var=0.0061)
- pi_13: -0.0554 (var=0.0066)
- pi_14: -0.0254 (var=0.0050)

### Experiment 23
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Reversal-balanced choice rate for the high-spread skewed coalition."""
    y_choice_by_orientation = {True: [], False: []}
    target_active = np.arange(18, 26, dtype=int)

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=int)
        b = np.asarray(row["option_b_ratings"], dtype=int)
        if a.size < 26 or b.size < 26:
            continue

        active = np.flatnonzero(a != b)
        if not np.array_equal(active, target_active):
            continue

        # In the largest-spread pair, expert 25 is the unique upper-tail
        # member of the positively skewed coalition Y.
        y_is_a = bool(a[25] > b[25])
        response = int(row["response"])
        chose_y = float(response == (0 if y_is_a else 1))
        y_choice_by_orientation[y_is_a].append(chose_y)

    orientation_means = []
    for y_is_a in (True, False):
        values = y_choice_by_orientation[y_is_a]
        if len(values) > 0:
            orientation_means.append(float(np.mean(values)))

    if len(orientation_means) == 0:
        return 0.5
    return float(np.mean(orientation_means))
```

**Observed (real) value:** 0.5000 (var=0.0088)
**Other theories' values on this metric (for reference):**
- pi_13: 0.5337 (var=0.0069)
- pi_12: 0.4281 (var=0.0107)
- pi_1: 0.8306 (var=0.0151)
- pi_2: 0.4850 (var=0.0078)
- pi_3: 0.5300 (var=0.0079)
- pi_4: 0.4838 (var=0.0101)
- pi_5: 0.5100 (var=0.0067)
- pi_6: 0.5144 (var=0.0109)
- pi_7: 0.4450 (var=0.0091)
- pi_8: 0.4925 (var=0.0088)
- pi_9: 0.4850 (var=0.0116)
- pi_10: 0.4881 (var=0.0077)
- pi_11: 0.4625 (var=0.0112)
- pi_14: 0.4625 (var=0.0068)

### Experiment 24
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate of choosing the option endorsed by the unilateral coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        margin = float(np.sum(a - b))
        if not np.isfinite(margin) or abs(margin) < 1e-12:
            continue

        response = int(row["response"])
        chose_endorsed = ((margin > 0.0 and response == 0) or
                          (margin < 0.0 and response == 1))
        scores.append(float(chose_endorsed))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.5763 (var=0.0023)
**Other theories' values on this metric (for reference):**
- pi_12: 0.1106 (var=0.0028)
- pi_13: 0.5363 (var=0.0187)
- pi_1: 0.8290 (var=0.0100)
- pi_2: 0.8840 (var=0.0070)
- pi_3: 0.8248 (var=0.0030)
- pi_4: 0.6258 (var=0.0040)
- pi_5: 0.7140 (var=0.0028)
- pi_6: 0.4058 (var=0.0090)
- pi_7: 0.6960 (var=0.0175)
- pi_8: 0.4792 (var=0.0113)
- pi_9: 0.4990 (var=0.0291)
- pi_10: 0.2062 (var=0.0081)
- pi_11: 0.2412 (var=0.0072)
- pi_14: 0.5962 (var=0.0031)

### Experiment 25
**Design**
  A=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Estimate the linear change in preference for the 70%-expert coalition as
    # its three fillers are progressively upgraded from 50% to 60% validity.
    records = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 7 or b.size < 7:
            continue

        # Feature 6 is the fixed 70%-valid expert and uniquely identifies the
        # focal coalition. Its display side is reversed in half the trials.
        if a[6] > b[6]:
            focal_is_a = True
            focal = a
        elif b[6] > a[6]:
            focal_is_a = False
            focal = b
        else:
            continue

        # Features 3--5 are the three possible 60%-valid fillers. Their sum is
        # the designed progression level k in {0, 1, 2, 3}.
        k = int(np.rint(np.sum(focal[3:6])))
        if k < 0 or k > 3:
            continue

        response = int(row["response"])
        chose_focal = float((response == 0) if focal_is_a else (response == 1))
        records.append((focal_is_a, k, chose_focal))

    if len(records) == 0:
        return 0.0

    # Compute the trend separately for the two display orientations, then
    # average. This cancels stable A/B response biases while giving all four
    # experimental conditions equal weight.
    orientation_slopes = []
    for focal_is_a in (False, True):
        condition_means = []
        levels = []
        for k in range(4):
            vals = [y for side, level, y in records
                    if side == focal_is_a and level == k]
            if len(vals) > 0:
                levels.append(float(k))
                condition_means.append(float(np.mean(vals)))

        if len(levels) >= 2:
            x = np.asarray(levels, dtype=float)
            y = np.asarray(condition_means, dtype=float)
            xc = x - np.mean(x)
            denom = float(np.sum(xc ** 2))
            if denom > 0.0:
                orientation_slopes.append(float(np.sum(xc * y) / denom))

    if len(orientation_slopes) == 0:
        return 0.0

    # Multiplication by three expresses the fitted trend as the predicted
    # preference change from k=0 to k=3.
    return float(3.0 * np.mean(orientation_slopes))
```

**Observed (real) value:** -0.3065 (var=0.0176)
**Other theories' values on this metric (for reference):**
- pi_13: 0.4533 (var=0.0337)
- pi_14: -0.0902 (var=0.0067)
- pi_1: 0.0125 (var=0.0113)
- pi_2: -0.0430 (var=0.0145)
- pi_3: 0.0315 (var=0.0064)
- pi_4: -0.0068 (var=0.0180)
- pi_5: 0.0112 (var=0.0203)
- pi_6: -0.2332 (var=0.0153)
- pi_7: 0.0670 (var=0.0139)
- pi_8: -0.0320 (var=0.0077)
- pi_9: -0.0730 (var=0.0151)
- pi_10: 0.0152 (var=0.0045)
- pi_11: -0.0285 (var=0.0096)
- pi_12: -0.0690 (var=0.0064)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    captures = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features 0:8 define the heterogeneous coalition; features 8:12
        # define the homogeneous 51%-valid coalition. Determine which option
        # is supported by the former without relying on display side.
        mixed_margin = float(np.sum(a[:8]) - np.sum(b[:8]))
        if mixed_margin == 0.0:
            continue

        mixed_is_a = mixed_margin > 0.0
        mixed_ratings = a if mixed_is_a else b
        n_perfect_endorsers = int(np.sum(mixed_ratings[:4]))

        # The maximally heterogeneous condition has one 100%-valid endorser
        # and three 50%-valid endorsers.
        if n_perfect_endorsers != 1:
            continue

        response = int(row["response"])
        chose_mixed = (mixed_is_a and response == 0) or (
            (not mixed_is_a) and response == 1
        )
        captures.append(float(chose_mixed))

    if len(captures) == 0:
        return float("nan")
    return float(np.mean(captures))
```

**Observed (real) value:** 0.8683 (var=0.0070)
**Other theories' values on this metric (for reference):**
- pi_14: 0.2225 (var=0.0071)
- pi_13: 0.7600 (var=0.0287)
- pi_1: 0.8550 (var=0.0183)
- pi_2: 0.4992 (var=0.0105)
- pi_3: 0.9992 (var=0.0000)
- pi_4: 0.5467 (var=0.0118)
- pi_5: 0.4808 (var=0.0090)
- pi_6: 0.2417 (var=0.0074)
- pi_7: 0.1750 (var=0.0274)
- pi_8: 0.3258 (var=0.0111)
- pi_9: 0.3617 (var=0.0152)
- pi_10: 0.3475 (var=0.0110)
- pi_11: 0.4842 (var=0.0171)
- pi_12: 0.5133 (var=0.0104)


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
  "theory": "Diversity-Legitimacy Attribution theory proposes that decision makers interpret a set of endorsers as evidence about both product quality and the social origin of the endorsement. Three option-symmetric construals compete continuously. A default tally construal counts discriminating endorsements and becomes dominant when both the absolute and proportional vote margin are decisive. A coordinated-claim construal compresses internally homogeneous endorsements into an option-level advocacy claim; during unresolved bilateral conflicts, the stronger coherent claim is pragmatically discounted or reversed. A diversity-legitimacy construal instead treats a coalition as credible corroboration when breadth, internal dispersion, upper-tail surprise, and the presence of weak background endorsers jointly support an independence attribution. Consequently, an exceptional expert receives positive influence when embedded among several heterogeneous, relatively weak endorsers, but not merely because it is the strongest expert. Homogeneous repetition creates coordination without diversity, whereas replacing neutral fillers with moderately valid fillers reduces both the exceptional expert's distinctiveness and the coalition's weak-background anchor. Stable individual differences chiefly shift arbitration between coordinated-claim skepticism and diversity-based legitimacy. Choice precision and lapses are constrained, and history is ignored because choices receive no correctness feedback.",
  "predict": "def predict(parameters, state, history):\n    # Diversity-Legitimacy Attribution model. With no outcome feedback, past\n    # choices provide no basis for learning expert validity and are ignored.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    n_features = stimulus.shape[1]\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match n_features {n_features}.\"\n        )\n\n    # Positive votes favor A and negative votes favor B. Equal ratings are\n    # nondiagnostic and do not enter either option's inferred coalition.\n    signed_votes = np.sign(stimulus[0] - stimulus[1])\n    active = signed_votes != 0.0\n    active_count = int(np.count_nonzero(active))\n    tally_margin = float(np.sum(signed_votes))\n    margin_magnitude = abs(tally_margin)\n\n    # Smooth coding of communicated validity. No validity receives a special\n    # categorical status, including the endpoints of the scale.\n    validity = np.clip(validities, 0.5, 1.0)\n    curvature = float(parameters[\"validity_curvature\"])\n    endpoint = float(np.tanh(0.5 * curvature))\n    authority = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)\n\n    def summarize(side):\n        members = authority[active & (signed_votes == side)]\n        breadth = int(members.size)\n        if breadth == 0:\n            return {\n                \"breadth\": 0,\n                \"formation\": 0.0,\n                \"coordination\": 0.0,\n                \"legitimacy\": 0.0,\n                \"dispersion\": 0.0,\n            }\n\n        mean_strength = float(np.mean(members))\n        dispersion = float(np.sqrt(np.mean((members - mean_strength) ** 2)))\n        maximum = float(np.max(members))\n        upper_surprise = max(maximum - mean_strength, 0.0)\n\n        # Coalition formation saturates smoothly: repetition establishes that\n        # there is a coalition but does not add independent evidence linearly.\n        breadth_scale = float(parameters[\"breadth_scale\"])\n        breadth_power = float(parameters[\"breadth_power\"])\n        formation = 1.0 - np.exp(\n            -(float(breadth) / breadth_scale) ** breadth_power\n        )\n\n        # Homogeneous, authoritative endorsers are compressed into a coherent\n        # option-level claim. Dispersion weakens coordination attribution.\n        coordination_homogeneity = np.exp(\n            -(dispersion / float(parameters[\"coordination_dispersion_scale\"])) ** 2\n        )\n        coordination = (\n            formation\n            * coordination_homogeneity\n            * mean_strength ** float(parameters[\"coordination_extremity\"])\n        )\n\n        # Diversity legitimacy is a genuine interaction. It is exactly weak\n        # when any constituent is absent: coalition breadth, dispersion,\n        # upper-tail surprise, exceptional authority, or weak-background\n        # corroboration. Thus this is not a generic soft maximum.\n        dispersion_gate = dispersion / (\n            dispersion + float(parameters[\"diversity_dispersion_scale\"])\n        )\n        surprise_gate = upper_surprise / (\n            upper_surprise + float(parameters[\"surprise_scale\"])\n        )\n        weak_anchor = float(np.mean(np.exp(\n            -members / float(parameters[\"weak_anchor_scale\"])\n        )))\n        breadth_gate = formation ** float(parameters[\"diversity_breadth_power\"])\n        legitimacy = (\n            breadth_gate\n            * dispersion_gate\n            * surprise_gate\n            * maximum\n            * weak_anchor\n        )\n\n        return {\n            \"breadth\": breadth,\n            \"formation\": float(formation),\n            \"coordination\": float(coordination),\n            \"legitimacy\": float(legitimacy),\n            \"dispersion\": dispersion,\n        }\n\n    side_a = summarize(1.0)\n    side_b = summarize(-1.0)\n\n    coordination_contrast = (\n        side_a[\"coordination\"] - side_b[\"coordination\"]\n    )\n    legitimacy_contrast = side_a[\"legitimacy\"] - side_b[\"legitimacy\"]\n\n    # Decisiveness combines absolute margin and proportional margin. A sparse\n    # one-vote advantage is therefore not equated with a broad majority.\n    if active_count == 0:\n        proportional_margin = 0.0\n        decisiveness = 0.0\n    else:\n        proportional_margin = margin_magnitude / float(active_count)\n        absolute_certainty = 1.0 - np.exp(\n            -(margin_magnitude / float(parameters[\"margin_scale\"]))\n            ** float(parameters[\"margin_power\"])\n        )\n        decisiveness = absolute_certainty * (\n            proportional_margin ** float(parameters[\"proportion_power\"])\n        )\n    decisiveness = float(np.clip(decisiveness, 0.0, 1.0))\n    unresolved = 1.0 - decisiveness\n\n    # Stable attribution style reallocates interpretive attention between\n    # coordination skepticism and diversity legitimacy without altering the\n    # communicated validities or creating endpoint-specific rules.\n    attribution_style = float(parameters[\"attribution_style\"]) - 0.5\n    style_shift = float(parameters[\"style_span\"]) * attribution_style\n\n    strongest_coordination = max(\n        side_a[\"coordination\"], side_b[\"coordination\"]\n    )\n    strongest_legitimacy = max(side_a[\"legitimacy\"], side_b[\"legitimacy\"])\n\n    # Smooth arbitration utilities. The raw tally receives increasing priority\n    # with decisiveness; attributional construals receive priority in unresolved\n    # conflicts according to their stimulus-specific support.\n    count_utility = (\n        float(parameters[\"count_bias\"])\n        + float(parameters[\"count_gate_gain\"]) * decisiveness\n    )\n    coordination_utility = (\n        float(parameters[\"coordination_bias\"])\n        + float(parameters[\"unresolved_gain\"]) * unresolved\n        + float(parameters[\"coordination_salience\"]) * strongest_coordination\n        + style_shift\n    )\n    diversity_utility = (\n        float(parameters[\"diversity_bias\"])\n        + float(parameters[\"unresolved_gain\"]) * unresolved\n        + float(parameters[\"diversity_salience\"]) * strongest_legitimacy\n        - style_shift\n    )\n\n    utilities = np.asarray(\n        [count_utility, coordination_utility, diversity_utility],\n        dtype=np.float64,\n    )\n    utilities /= float(parameters[\"arbitration_temperature\"])\n    utilities -= np.max(utilities)\n    arbitration = np.exp(utilities)\n    arbitration /= np.sum(arbitration)\n\n    # Default tally evidence grows nonlinearly with the absolute vote margin.\n    # A proportional factor further distinguishes broad majorities from thin\n    # advantages while retaining a modest unilateral-coalition preference.\n    if margin_magnitude < 1e-12:\n        count_evidence = 0.0\n    else:\n        count_curve = (\n            (1.0 + margin_magnitude) ** float(parameters[\"count_exponent\"])\n            - 1.0\n        )\n        count_evidence = (\n            float(parameters[\"count_strength\"])\n            * np.sign(tally_margin)\n            * count_curve\n            * (0.5 + 0.5 * proportional_margin)\n        )\n\n    # Pragmatic reversal requires two represented sides. Its strength falls\n    # continuously as the tally resolves the conflict, preventing skepticism\n    # from defeating clear majorities or reversing unilateral endorsement.\n    bilateral_scale = float(parameters[\"bilateral_scale\"])\n    bilateral = np.sqrt(\n        max(side_a[\"formation\"] * side_b[\"formation\"], 0.0)\n    )\n    bilateral = bilateral / (bilateral + bilateral_scale)\n    conflict_focus = np.exp(\n        -(margin_magnitude / float(parameters[\"conflict_margin_scale\"])) ** 2\n    )\n    coordination_evidence = -float(parameters[\"coordination_strength\"]) * (\n        bilateral\n        * conflict_focus\n        * np.tanh(\n            float(parameters[\"coordination_slope\"])\n            * coordination_contrast\n        )\n    )\n\n    # Legitimate diversity supports, rather than reverses, the coalition with\n    # broad heterogeneous corroboration and a surprising upper-tail expert.\n    diversity_evidence = float(parameters[\"diversity_strength\"]) * (\n        unresolved\n        * np.tanh(\n            float(parameters[\"diversity_slope\"])\n            * legitimacy_contrast\n        )\n    )\n\n    precision_trait = float(parameters[\"precision_trait\"]) - 0.5\n    precision = float(parameters[\"choice_precision\"]) * (\n        1.0 + float(parameters[\"precision_span\"]) * precision_trait\n    )\n    precision = max(precision, 0.10)\n\n    # Each construal supplies a complete conditional choice distribution. We\n    # marginalize over construal recruitment instead of averaging incompatible\n    # evidence representations into a single latent score.\n    evidences = precision * np.asarray(\n        [count_evidence, coordination_evidence, diversity_evidence],\n        dtype=np.float64,\n    )\n    channel_p_a = np.empty(3, dtype=np.float64)\n    for i, evidence in enumerate(evidences):\n        logits = np.asarray([0.5 * evidence, -0.5 * evidence], dtype=np.float64)\n        logits -= np.max(logits)\n        channel_probs = np.exp(logits)\n        channel_probs /= np.sum(channel_probs)\n        channel_p_a[i] = channel_probs[0]\n\n    p_a = float(np.dot(arbitration, channel_p_a))\n    probabilities = np.asarray([p_a, 1.0 - p_a], dtype=np.float64)\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= np.sum(probabilities)\n    return probabilities\n",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = float(np.sum(probabilities))\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.full(\n            probabilities.size,\n            1.0 / probabilities.size,\n            dtype=np.float64,\n        )\n    else:\n        probabilities /= total\n    return int(np.random.choice(probabilities.size, p=probabilities))\n",
  "parameters": {
    "validity_curvature": "[1.6, 2.6]",
    "breadth_scale": "[0.85, 1.25]",
    "breadth_power": "[0.80, 1.20]",
    "coordination_dispersion_scale": "[0.16, 0.25]",
    "coordination_extremity": "[1.15, 1.70]",
    "diversity_dispersion_scale": "[0.15, 0.24]",
    "surprise_scale": "[0.14, 0.24]",
    "weak_anchor_scale": "[0.28, 0.45]",
    "diversity_breadth_power": "[1.15, 1.75]",
    "margin_scale": "[2.2, 3.2]",
    "margin_power": "[1.25, 1.75]",
    "proportion_power": "[0.65, 1.05]",
    "count_bias": "[-0.10, 0.15]",
    "count_gate_gain": "[3.2, 4.6]",
    "coordination_bias": "[-0.15, 0.20]",
    "diversity_bias": "[-1.00, -0.55]",
    "unresolved_gain": "[0.75, 1.20]",
    "coordination_salience": "[1.35, 2.10]",
    "diversity_salience": "[5.0, 7.5]",
    "attribution_style": "[0, 1]",
    "style_span": "[0.35, 0.65]",
    "arbitration_temperature": "[0.72, 1.00]",
    "count_strength": "[0.58, 0.78]",
    "count_exponent": "[0.68, 0.86]",
    "coordination_strength": "[4.6, 6.2]",
    "coordination_slope": "[2.4, 3.6]",
    "bilateral_scale": "[0.20, 0.34]",
    "conflict_margin_scale": "[2.4, 3.4]",
    "diversity_strength": "[5.2, 7.4]",
    "diversity_slope": "[3.0, 4.8]",
    "choice_precision": "[0.92, 1.12]",
    "precision_trait": "[0, 1]",
    "precision_span": "[0.08, 0.20]",
    "lapse_rate": "[0.01, 0.055]",
    "validities": "validities"
  },
  "rationale": "The model directly implements the arbiter's missing attributional mechanism. Previous upper-tail or exemplar accounts made an extreme expert influential largely because it was extreme. Here, positive diversity evidence is multiplicative: breadth, dispersion, upper-tail surprise, exceptional authority, and weak-background support must all be present. This makes a perfect expert among several neutral or weak endorsers credible as independently corroborated, supporting high capture in Experiments 2, 19, 21, and 26. Upgrading neutral fillers reduces dispersion, surprise, and the weak-background anchor, producing the required negative trend in Experiments 1 and 25. In contrast, homogeneous authoritative groups produce coordinated claims rather than independent evidence and can be pragmatically reversed in unresolved bilateral conflicts, preserving anti-validity effects such as Experiments 7, 10, 11, 12, and 16. The reversal is suppressed for unilateral coalitions and decisive vote margins, yielding modest unilateral following and majority dominance. All transitions depend smoothly on general geometric quantities rather than feature indices, validity endpoints, or experiment identities. Finally, representing arbitration as a mixture of complete construal-specific choice distributions prevents strong but rarely recruited signals from unrealistically cancelling the default tally on clear-majority trials."
}
```

## Usage

```json
{
  "input_tokens": 51604,
  "output_tokens": 6066
}
```
