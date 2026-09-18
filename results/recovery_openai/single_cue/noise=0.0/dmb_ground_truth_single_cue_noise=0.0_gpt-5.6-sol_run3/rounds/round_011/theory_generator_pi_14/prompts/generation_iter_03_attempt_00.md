# generation_iter_03_attempt_00

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
- THEORY 2 = `pi_12`
- The recommendation below acts on THEORY 2 (= `pi_12`).

Replace pi_12 with a Bilateral Advocacy-Attribution theory. Its central claim should be that communicated validities have two context-dependent meanings. They are ordinary evidence when endorsements are unilateral or the raw tally is decisive, but they can signal advocacy intensity or overstatement when two sufficiently formed coalitions oppose one another and the tally remains unresolved. Skeptical reversal must therefore require smooth bilateral formation; it must not be triggered by a trustworthy unilateral coalition.

The new theory should contain: (1) a default nonlinear tally process whose influence increases smoothly with absolute and proportional margin and saturates for decisive majorities; (2) a bilateral claim process based on bounded mean communicated validity, with homogeneous repetition largely redundant; (3) a smooth coalition-coherence term based on dispersion, without pi_12's hard trust collapse; and (4) arbitration governed by bilateral formation, tally decisiveness, reliability contrast, and coherence asymmetry. In balanced conflicts, the more reliable and coherent coalition becomes the more salient advocacy claim, so skepticism favors its opponent. If one coalition is highly dispersed, it is a weak unified claim, making the opposing homogeneous coalition the salient overclaim and thereby favoring the dispersed side. This can produce the Experiment 19 gradient and Experiment 21 preference without invoking an upper-tail exemplar channel. Large tally margins should continuously suppress this reversal.

Avoid narrow margin windows, endpoint-specific limiters, and a generic rule that stronger validity always elicits stronger skepticism. Include stable subject variation in bilateral-skepticism propensity and choice precision, but constrain it enough to reproduce the low variances in Experiments 2, 11, and 22. The resulting model should predict approximately chance in Experiment 1, above-chance endorsement consistency in Experiment 2, strong tally following at large margins, and anti-validity choices chiefly in tied or weak-margin bilateral coalition conflicts.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_11` (overall score: 0.664)

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


---

### `pi_13` (overall score: 0.636)

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


---

### `pi_12` (overall score: 0.586)

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


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1312 -> ACCEPTED
- iter 2: loss=0.1576 -> REJECTED
- iter 3: loss=0.1842 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1312 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
**Previous candidate values (this loop):**
  - iter 1: 0.3133 (var=0.0037) (Δ vs real -0.0900)
  - iter 2: 0.3450 (var=0.0061) (Δ vs real -0.0583)
  - iter 3 (most recent): 0.3079 (var=0.0044) (Δ vs real -0.0954)
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
**Previous candidate values (this loop):**
  - iter 1: 192.8541 (var=3.8651) (Δ vs real +103.7119)
  - iter 2: 213.8409 (var=3.0110) (Δ vs real +124.6986)
  - iter 3 (most recent): 198.6054 (var=3.0123) (Δ vs real +109.4632)
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
**Previous candidate values (this loop):**
  - iter 1: 0.3233 (var=0.0021) (Δ vs real -0.1275)
  - iter 2: 0.3740 (var=0.0022) (Δ vs real -0.0769)
  - iter 3 (most recent): 0.2994 (var=0.0026) (Δ vs real -0.1515)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8087 (var=0.0048) (Δ vs real -0.0125)
  - iter 2: 0.8063 (var=0.0058) (Δ vs real -0.0150)
  - iter 3 (most recent): 0.8131 (var=0.0048) (Δ vs real -0.0081)
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
**Previous candidate values (this loop):**
  - iter 1: -0.1249 (var=0.0365) (Δ vs real -0.0867)
  - iter 2: -0.0596 (var=0.0406) (Δ vs real -0.0214)
  - iter 3 (most recent): -0.1952 (var=0.0347) (Δ vs real -0.1571)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8179 (var=0.0016) (Δ vs real -0.0283)
  - iter 2: 0.7985 (var=0.0016) (Δ vs real -0.0477)
  - iter 3 (most recent): 0.8042 (var=0.0014) (Δ vs real -0.0421)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0941 (var=0.0009) (Δ vs real -0.0221)
  - iter 2: 0.2314 (var=0.0025) (Δ vs real +0.1152)
  - iter 3 (most recent): 0.1042 (var=0.0012) (Δ vs real -0.0120)
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
**Previous candidate values (this loop):**
  - iter 1: -4.8115 (var=0.0117) (Δ vs real -1.2743)
  - iter 2: -5.0322 (var=0.0056) (Δ vs real -1.4950)
  - iter 3 (most recent): -4.7277 (var=0.0077) (Δ vs real -1.1905)
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
**Previous candidate values (this loop):**
  - iter 1: 0.1353 (var=0.0025) (Δ vs real -0.1896)
  - iter 2: 0.3168 (var=0.0028) (Δ vs real -0.0080)
  - iter 3 (most recent): 0.1520 (var=0.0017) (Δ vs real -0.1728)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8442 (var=0.0048) (Δ vs real -0.0175)
  - iter 2: 0.8475 (var=0.0078) (Δ vs real -0.0142)
  - iter 3 (most recent): 0.8400 (var=0.0050) (Δ vs real -0.0217)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0892 (var=0.0010) (Δ vs real -0.0681)
  - iter 2: 0.2017 (var=0.0024) (Δ vs real +0.0444)
  - iter 3 (most recent): 0.1036 (var=0.0020) (Δ vs real -0.0536)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0936 (var=0.0013) (Δ vs real -0.0325)
  - iter 2: 0.1564 (var=0.0024) (Δ vs real +0.0303)
  - iter 3 (most recent): 0.1056 (var=0.0011) (Δ vs real -0.0206)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0428 (var=0.0093) (Δ vs real +0.0334)
  - iter 2: -0.0028 (var=0.0075) (Δ vs real -0.0122)
  - iter 3 (most recent): -0.0319 (var=0.0090) (Δ vs real -0.0413)
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
**Previous candidate values (this loop):**
  - iter 1: -0.0174 (var=0.0052) (Δ vs real -0.1954)
  - iter 2: 0.0981 (var=0.0102) (Δ vs real -0.0800)
  - iter 3 (most recent): -0.0126 (var=0.0059) (Δ vs real -0.1907)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.2356) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.2464) (Δ vs real +1.0000)
  - iter 3 (most recent): 1.0000 (var=0.2400) (Δ vs real +1.0000)
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
**Previous candidate values (this loop):**
  - iter 1: 0.5219 (var=0.0041) (Δ vs real -0.3600)
  - iter 2: 0.5316 (var=0.0028) (Δ vs real -0.3503)
  - iter 3 (most recent): 0.5500 (var=0.0050) (Δ vs real -0.3319)
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
**Previous candidate values (this loop):**
  - iter 1: -0.3269 (var=0.0142) (Δ vs real -0.0019)
  - iter 2: -0.2450 (var=0.0141) (Δ vs real +0.0800)
  - iter 3 (most recent): -0.3006 (var=0.0117) (Δ vs real +0.0244)
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
**Previous candidate values (this loop):**
  - iter 1: 0.0583 (var=0.0103) (Δ vs real +0.0525)
  - iter 2: -0.0421 (var=0.0107) (Δ vs real -0.0479)
  - iter 3 (most recent): 0.1038 (var=0.0102) (Δ vs real +0.0979)
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
**Previous candidate values (this loop):**
  - iter 1: 0.7792 (var=0.0079) (Δ vs real -0.0825)
  - iter 2: 0.7658 (var=0.0100) (Δ vs real -0.0958)
  - iter 3 (most recent): 0.7700 (var=0.0076) (Δ vs real -0.0917)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8525 (var=0.0031) (Δ vs real -0.0042)
  - iter 2: 0.8425 (var=0.0023) (Δ vs real -0.0142)
  - iter 3 (most recent): 0.8483 (var=0.0028) (Δ vs real -0.0083)
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
**Previous candidate values (this loop):**
  - iter 1: 0.8867 (var=0.0021) (Δ vs real +0.0200)
  - iter 2: 0.8597 (var=0.0035) (Δ vs real -0.0070)
  - iter 3 (most recent): 0.8697 (var=0.0024) (Δ vs real +0.0030)
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
**Previous candidate values (this loop):**
  - iter 1: -0.0142 (var=0.0046) (Δ vs real -0.0083)
  - iter 2: -0.0096 (var=0.0056) (Δ vs real -0.0038)
  - iter 3 (most recent): -0.0063 (var=0.0041) (Δ vs real -0.0004)
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
**Previous candidate values (this loop):**
  - iter 1: 0.4531 (var=0.0050) (Δ vs real -0.0469)
  - iter 2: 0.3725 (var=0.0054) (Δ vs real -0.1275)
  - iter 3 (most recent): 0.4319 (var=0.0070) (Δ vs real -0.0681)
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
**Previous candidate values (this loop):**
  - iter 1: 0.5863 (var=0.0028) (Δ vs real +0.0100)
  - iter 2: 0.6127 (var=0.0031) (Δ vs real +0.0365)
  - iter 3 (most recent): 0.6000 (var=0.0026) (Δ vs real +0.0237)
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

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Bilateral Advocacy-Attribution theory proposes that communicated expert validity has two context-sensitive meanings. By default, endorsements are treated as ordinary evidence and choices follow a nonlinear raw tally whose influence rises smoothly with both absolute and proportional vote margin before saturating. This default governs unilateral endorsement and decisive majorities. A second interpretation becomes available only when endorsers have formed coalitions on both sides and the tally remains unresolved: coalition validity then conveys the intensity of an option-level advocacy claim rather than literal evidential weight. Each claim is represented by bounded mean communicated validity, so homogeneous repetition is largely redundant. Coalition coherence varies smoothly with within-side dispersion and modulates claim salience without any threshold or trust collapse. In balanced conflict, a more reliable and coherent coalition constitutes the more salient advocacy claim and is therefore the target of pragmatic skepticism. When one coalition is highly dispersed, it is a weak unified claim; the opposing homogeneous coalition becomes the salient overclaim, causing skepticism to favor the dispersed side. Arbitration into this skeptical interpretation increases continuously with bilateral formation, reliability contrast, and coherence asymmetry, while absolute and proportional tally decisiveness continuously suppress it. Thus trustworthy unilateral coalitions are followed rather than reversed, large majorities dominate, and anti-validity choice is concentrated in tied or weak-margin bilateral conflicts. Stable but constrained individual differences affect bilateral-skepticism propensity and choice precision.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Bilateral Advocacy-Attribution model. History is intentionally ignored:
    # without correctness feedback, past choices provide no basis for learning
    # the experts' diagnosticities.
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

    # Positive votes favor A and negative votes favor B. Features on which the
    # options agree are nondiscriminating and belong to neither coalition.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    active_count = int(np.count_nonzero(active))
    tally_margin = float(np.sum(signed_votes))
    margin_magnitude = abs(tally_margin)

    # Smoothly bounded coding of communicated validity. This has no special
    # endpoint category and retains sensitivity over the complete range.
    validity = np.clip(validities, 0.5, 1.0)
    curvature = float(parameters["reliability_curvature"])
    raw_reliability = 2.0 * (validity - 0.5)
    reliability = np.tanh(curvature * raw_reliability) / max(
        float(np.tanh(curvature)), 1e-12
    )

    def coalition(side):
        members = reliability[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0.0, 0

        # Advocacy intensity is an option-level bounded mean. Consequently,
        # repeating identical endorsers does not multiply the claim.
        mean_reliability = float(np.mean(members))
        dispersion = float(np.mean((members - mean_reliability) ** 2))

        # Dispersion weakens unified advocacy smoothly. The positive floor
        # prevents the hard trust collapse of threshold-based accounts.
        coherence_floor = float(parameters["coherence_floor"])
        coherence_scale = float(parameters["coherence_scale"])
        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(
            -dispersion / max(coherence_scale, 1e-12)
        )

        # A coherent reliable coalition is a salient advocacy claim. A highly
        # dispersed coalition remains represented, but as a weaker unified act.
        salience = mean_reliability * coherence
        return mean_reliability, float(coherence), float(salience), breadth

    mean_a, coherence_a, salience_a, breadth_a = coalition(1.0)
    mean_b, coherence_b, salience_b, breadth_b = coalition(-1.0)

    if active_count == 0:
        proportional_margin = 0.0
    else:
        proportional_margin = margin_magnitude / float(active_count)

    # Default nonlinear tally: absolute margin supplies sample strength, while
    # proportional margin supplies agreement. Both effects are continuous and
    # the resulting evidence saturates for decisive majorities.
    if margin_magnitude < 1e-12:
        absolute_resolution = 0.0
        tally_evidence = 0.0
    else:
        absolute_resolution = 1.0 - np.exp(
            -(margin_magnitude / float(parameters["margin_scale"]))
            ** float(parameters["margin_power"])
        )
        proportional_resolution = (
            float(parameters["proportion_floor"])
            + (1.0 - float(parameters["proportion_floor"]))
            * proportional_margin ** float(parameters["proportion_power"])
        )
        tally_shape = absolute_resolution * proportional_resolution
        tally_evidence = (
            float(parameters["tally_strength"])
            * np.sign(tally_margin)
            * tally_shape
        )

    # Reliability and coherence contrasts are relative option-level contrasts.
    # Relative normalization allows a small informative claim opposing a 50%
    # coalition to be salient without making validity magnitude generically
    # produce ever-stronger skepticism.
    salience_contrast = (salience_a - salience_b) / (
        float(parameters["claim_floor"])
        + abs(salience_a)
        + abs(salience_b)
    )
    salience_contrast = float(np.clip(salience_contrast, -1.0, 1.0))

    mean_contrast = (mean_a - mean_b) / (
        float(parameters["ordinary_claim_floor"])
        + abs(mean_a)
        + abs(mean_b)
    )
    mean_contrast = float(np.clip(mean_contrast, -1.0, 1.0))

    # Each side must form before validity can be attributed to opposed
    # advocacy. The geometric combination is exactly zero for unilateral
    # endorsement and rises smoothly from singleton to broad coalitions.
    formation_scale = float(parameters["formation_scale"])
    formation_power = float(parameters["formation_power"])
    formation_a = 1.0 - np.exp(
        -(float(breadth_a) / formation_scale) ** formation_power
    ) if breadth_a > 0 else 0.0
    formation_b = 1.0 - np.exp(
        -(float(breadth_b) / formation_scale) ** formation_power
    ) if breadth_b > 0 else 0.0
    bilateral_formation = float(np.sqrt(max(formation_a * formation_b, 0.0)))

    # Reliability contrast and coherence asymmetry make advocacy attribution
    # more diagnostic. Coherence asymmetry does not select an upper-tail member:
    # it only identifies which option-level coalition is the clearer claim.
    contrast_readiness = float(np.tanh(
        float(parameters["salience_slope"]) * abs(salience_contrast)
    ))
    coherence_asymmetry = abs(coherence_a - coherence_b)
    coherence_readiness = 1.0 + float(
        parameters["coherence_asymmetry_gain"]
    ) * coherence_asymmetry
    claim_readiness = float(np.clip(
        contrast_readiness * coherence_readiness, 0.0, 1.0
    ))

    # Decisive counting continuously resolves the ambiguity that licenses an
    # advocacy interpretation. There are no margin windows or endpoint rules.
    decisiveness = absolute_resolution * (
        float(parameters["decisiveness_floor"])
        + (1.0 - float(parameters["decisiveness_floor"]))
        * proportional_margin
    )
    unresolved = (1.0 - float(np.clip(decisiveness, 0.0, 1.0))) ** float(
        parameters["resolution_power"]
    )

    arbitration_weight = float(np.clip(
        bilateral_formation * claim_readiness * unresolved,
        0.0,
        1.0,
    ))

    # Outside bilateral unresolved conflict, mean validity is ordinary evidence.
    # This contribution is deliberately modest because the raw tally remains
    # the default, and it vanishes as advocacy attribution takes control.
    ordinary_evidence = (
        float(parameters["ordinary_strength"])
        * (1.0 - arbitration_weight)
        * mean_contrast
    )

    # In bilateral conflict, the more salient claim is pragmatically reversed.
    # If one side is dispersed, its salience falls and the coherent opponent is
    # treated as the overclaim, thereby favoring the dispersed coalition.
    skeptic_trait = float(parameters["bilateral_skepticism"])
    skeptic_multiplier = 1.0 + float(parameters["skeptic_trait_spread"]) * (
        skeptic_trait - 0.5
    )
    skeptic_multiplier = max(float(skeptic_multiplier), 0.25)
    advocacy_direction = float(np.tanh(
        float(parameters["advocacy_accumulation"]) * salience_contrast
    ))
    skeptical_evidence = -float(parameters["reversal_strength"]) * (
        skeptic_multiplier * arbitration_weight * advocacy_direction
    )

    total_evidence = tally_evidence + ordinary_evidence + skeptical_evidence

    # Choice precision is a stable, narrowly constrained individual property.
    # Ambiguous bilateral conflict modestly reduces effective precision without
    # generating large variance on clear tally trials.
    precision = float(parameters["choice_precision"])
    conflict_load = arbitration_weight * (1.0 - abs(advocacy_direction))
    precision /= 1.0 + float(parameters["conflict_noise"]) * conflict_load
    decision_variable = float(precision * total_evidence)

    # Positive evidence favors A. Symmetric stable softmax guarantees exact
    # invariance under exchanging the two displayed options.
    logits = np.asarray(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)

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
- reliability_curvature: [1.45, 1.85]
- coherence_floor: [0.24, 0.34]
- coherence_scale: [0.055, 0.075]
- claim_floor: [0.025, 0.045]
- ordinary_claim_floor: [0.20, 0.30]
- formation_scale: [0.82, 1.02]
- formation_power: [1.45, 1.80]
- salience_slope: [2.8, 3.5]
- coherence_asymmetry_gain: [0.55, 0.85]
- margin_scale: [3.55, 3.90]
- margin_power: [3.0, 3.4]
- proportion_floor: [0.42, 0.52]
- proportion_power: [0.70, 0.95]
- tally_strength: [1.85, 2.05]
- decisiveness_floor: [0.60, 0.72]
- resolution_power: [1.65, 2.10]
- ordinary_strength: [0.12, 0.22]
- advocacy_accumulation: [1.45, 1.85]
- reversal_strength: [2.70, 3.15]
- bilateral_skepticism: [0, 1]
- skeptic_trait_spread: [0.12, 0.22]
- choice_precision: [0.94, 1.08]
- conflict_noise: [0.08, 0.22]
- lapse_rate: [0.015, 0.055]
- validities: validities

`rationale`: The model directly repairs the previous theory's central failure: a reliable unilateral coalition can no longer trigger skeptical reversal because bilateral formation is exactly zero when either side lacks endorsers. Such trials are handled by ordinary mean-validity evidence plus the default tally, which should move Experiment 24 from strong anti-endorsement toward the observed mild endorsement preference. The tally function uses continuous absolute and proportional margin information and saturates naturally, replacing narrow margin windows and confidence limiters. Its high curvature keeps one- and two-vote evidence modest while preserving strong following of margins five and above, supporting Experiments 2, 4, 6, and 20 without special cases.

In bilateral unresolved conflicts, the model compares option-level advocacy claims rather than individual exemplars. Mean bounded reliability makes homogeneous repetition redundant, while smooth coherence weighting explains the dispersion gradient. A reliable homogeneous coalition is normally the salient overclaim, producing anti-validity behavior in Experiments 7, 8, 10, 11, 12, and 21. If its opponent becomes highly dispersed, the dispersed side's unified salience falls, making the homogeneous side the skeptical target and favoring the dispersed coalition in Experiments 17 and 19. Because this depends on coalition means and dispersion rather than upper-tail extraction, skewed coalitions with matched option-level representations remain near chance in Experiment 23.

The arbitration weight is the continuous product of bilateral formation, claim readiness, and unresolved tally conflict. Consequently, reversal is strongest for ties and weak margins and disappears continuously as the tally becomes decisive. The relative normalization of claim salience also avoids a generic rule that greater absolute validity must always cause greater skepticism: a slight informative coalition opposing 50% endorsers can be highly salient, while contrasts between two already reliable coalitions depend on their relative advocacy geometry. Finally, skepticism and precision vary stably across subjects but only over constrained ranges; this allows individual differences in ambiguous conflicts while protecting the low between-subject variances of clear or symmetric experiments.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Bilateral Advocacy-Attribution family and achieves several important qualitative successes. It repairs unilateral endorsement especially well in Experiment 24 (0.586 versus 0.576), preserves strong large-margin tally following in Experiments 4, 6, and 20, captures the Experiment 17 dispersion gradient almost exactly (-0.327 versus -0.325), and performs well on the anti-validity effects in Experiments 7, 10, 12, and 21. Experiment 22 is also close, indicating that the model does not impose a generic endpoint-specific skepticism rule.

Nevertheless, the fit is not yet experiment-invariant. The largest failure is Experiment 16: the model predicts only 0.522 preference for the 75%-valid coalition, compared with 0.882 observed. At the same time, it reproduces the extreme-dispersion gradient in Experiment 17 and comes reasonably close in Experiment 19. This combination suggests that the exponential coherence function loses coherence too quickly at moderate dispersion: moderate heterogeneity is treated almost like extreme heterogeneity. Human behavior instead appears to preserve a unified advocacy attribution through moderate dispersion and then decline sharply but smoothly at the extreme.

The model also entirely misses the coalition-breadth attenuation in Experiment 14 (-0.017 versus 0.178). Its bilateral-formation term grows monotonically with breadth while tied tallies always have zero decisiveness, so homogeneous repetition can only maintain or strengthen reversal. The same structural issue is visible in Experiment 18, where the predicted breadth effect is too positive (0.058 versus 0.006). A tied but increasingly broad contest needs some smooth reduction in advocacy ambiguity beyond raw signed tally margin.

More generally, reversal is too strong across several ordinary bilateral conflicts: Experiments 8 (-4.812 versus -3.537), 9 (0.135 versus 0.325), 11 (0.089 versus 0.157), and 12 (0.094 versus 0.126). Weak-margin tally/validity conflicts are also too tally-consistent in Experiments 1 and 3, while Experiment 2 produces much too much evidence for margin-sensitive tallying (192.9 versus 89.1). Thus the current arbitration and tally curves are overly polarized: weak conflicts are resolved too deterministically, either by tally or by skeptical reversal, rather than retaining modest conventional validity influence.
Rationale: Keep the Bilateral Advocacy-Attribution mechanism, but recalibrate its smooth components. First, replace the simple exponential coherence decay with a smooth higher-order dispersion transform, such as an estimated dispersion exponent or sigmoid-shaped decay. It should leave homogeneous and moderately dispersed coalitions relatively coherent while sharply—but continuously—weakening maximally dispersed coalitions. This is the most direct way to raise Experiment 16 substantially without sacrificing the successful Experiment 17 gradient or the heterogeneous-coalition preference in Experiment 19.

Second, add a generic, smooth contest-breadth or active-evidence saturation factor to unresolved arbitration. Even when the signed tally is tied, broader opposing coalitions should gradually reduce the ambiguity that licenses skeptical attribution. This must depend on total active breadth rather than validity endpoints. It should reverse the erroneous Experiment 14 breadth slope and reduce the excess breadth effect in Experiment 18 while respecting the recommendation that homogeneous repetition is largely redundant.

Third, soften skeptical replacement in ordinary bilateral conflicts without changing unilateral trials. A small residual conventional mean-validity contribution inside the arbitration state, or a modest reduction of readiness/reversal for low-coherence-contrast conflicts, would move Experiments 8, 9, 11, 12, and 23 toward the data while preserving the excellent unilateral fit in Experiment 24. Do not simply weaken reversal globally, because Experiments 16, 17, 19, and 21 require strong context-specific effects.

Finally, flatten the tally margin-response curve while preserving its fitted large-margin asymptote—for example, lower the margin exponent and jointly retune margin scale and tally strength so margins 5+ remain near their current probabilities. This should reduce the excessive Experiment 2 Bayes factor. Counterbalance any resulting increase in weak-margin tally influence with the residual conventional-validity term, which is also needed to raise TTB/minority choices in Experiments 1 and 3. After correcting the means, slightly widen stable skepticism or precision variation only if needed: variances are often too small, especially in Experiments 11 and 21, but the ranges should remain constrained enough to protect Experiments 2 and 22.

**Outcome of this advice:** iter 1 candidate loss=0.1312 -> iter 2 candidate loss=0.1576 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to Bilateral Advocacy-Attribution, but the current recalibration is not an improvement: aggregate loss rose from 0.1312 to 0.1576 and the gate rejected it. The added breadth attenuation and residual conventional channel improved Experiments 9, 11, 12, 14, and 21, but produced larger regressions elsewhere. In particular, Experiment 7 moved from an excellent 0.094 versus 0.116 to 0.231; Experiment 15 flipped from the correct categorical result to the wrong one; Experiment 23 moved farther from chance (0.373 versus 0.500); Experiment 17 lost its nearly exact dispersion gradient; and large-margin tally following weakened in Experiment 6. The already excessive margin sensitivity in Experiment 2 became worse (213.8 versus 89.1). Most importantly, the intended coherence modification barely affected Experiment 16, which remains near chance at 0.532 despite an observed preference of 0.882. Thus the rejected package trades several previously successful effects for modest gains without repairing the central Experiment 16 failure. The model still succeeds on decisive tally behavior in Experiments 4 and 20, endpoint reversal in Experiment 10, the heterogeneous-coalition preference in Experiment 19, anti-validity choice in Experiment 21, and the absence of generic endpoint skepticism in Experiment 22, so the prescribed mechanism family remains viable.
Rationale: Build from the unchanged iteration-1 accepted base, and do not repeat the rejected combination of higher-order coherence decay, explicit contest-breadth attenuation, and a broad residual conventional-validity term. The gate shows that this package over-attenuates broad advocacy conflicts and does not solve Experiment 16.

Instead, inspect and recalibrate the arbitration gating itself. The accepted model appears to attenuate modest salience contrasts twice: contrast magnitude first controls claim readiness and then again controls advocacy direction. Near zero, skeptical evidence therefore grows approximately quadratically with coalition contrast. That can leave the moderate, equal-breadth contrasts in Experiment 16 near chance even though bilateral formation is strong. Use the bounded salience contrast once for signed direction, while giving a smooth nonzero readiness floor to well-formed bilateral coalitions and letting coherence asymmetry modulate that readiness. Retune the corresponding gain and reversal strength downward together so already-large contrasts do not become more extreme. This remains squarely within bilateral advocacy attribution and targets moderate contrasts without adding a margin window or endpoint rule.

A less saturating bounded-mean reliability transform is another compatible, small adjustment: modestly lower reliability curvature so upper-middle communicated validities remain distinguishable. This may increase the advocacy contrast needed in Experiment 16 without invoking an exemplar channel. Check that Experiments 10, 13, 17, 19, and 21 remain stable.

Do not again flatten the tally curve in the same manner; that rejected change increased, rather than reduced, Experiment 2's Bayes factor. If Experiment 2 is addressed now, use constrained stable precision heterogeneity or a small common lapse/temperature recalibration to reduce overly deterministic margin differentiation while preserving the means at margins 5+ in Experiments 4, 6, and 20. Keep the adjustment narrow because Experiment 22's variance is already slightly high. The next candidate should specifically preserve the accepted base's strong Experiments 7, 15, 17, 20, 23, and 24 fits while raising Experiment 16 and avoiding further growth of the Experiment 2 discrepancy.

**Outcome of this advice:** iter 2 candidate loss=0.1576 -> iter 3 candidate loss=0.1842 -> the gate marked it REJECTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Bilateral Advocacy-Attribution family, but it is not an experiment-invariant improvement. Its aggregate loss increased from the accepted iteration-1 value of 0.1312 to 0.1842, so the gate correctly rejected it. The new readiness floor barely repairs its intended target: Experiment 16 rises only from 0.522 to 0.550, still far below the observed 0.882. Meanwhile, it damages several fits that the accepted base handled better. Experiment 15 flips from the correct categorical value 0 to 1; Experiments 1, 3, 5, 9, 13, 14, 18, and 23 move farther from their observed values; and Experiment 2 remains drastically too margin-sensitive at 198.6 versus 89.1. The model still captures decisive tally following in Experiments 4 and 20, the dispersion gradient in Experiment 17, anti-validity choice in Experiments 10 and 21, the absence of endpoint-specific skepticism in Experiment 22, and unilateral endorsement in Experiment 24. However, these successes largely reproduce the accepted base rather than justify the rejected modification. My previous recommendation to add a bilateral readiness floor was therefore not supported by the gate and should not be repeated.
Rationale: Build again from the unchanged iteration-1 accepted base. Remove the added readiness floor rather than enlarging it: the floor broadly increases eligibility for skepticism but does not distinguish the Experiment 16 configuration, producing regressions without enough gain.

Try a different in-family calibration of the signed advocacy response. Replace the product of a contrast-readiness term and a shallow signed direction with a single, steeply saturating signed transform, for example skeptical evidence proportional to bilateral_formation * unresolved * coherence_readiness * tanh(k * salience_contrast). Use a substantially steeper k together with a lower reversal-strength ceiling. This makes moderate contrasts approximately linear and strongly discriminating instead of quadratically attenuated, while the lower ceiling can preserve behavior for already-large contrasts in Experiments 7, 10, 17, 19, and 21. Unlike the rejected readiness floor, it remains exactly zero when the salience contrast is zero.

To isolate broad, well-formed coalitions such as those in Experiment 16 from singleton conflicts, modestly sharpen the existing smooth bilateral-formation curve rather than adding total-breadth attenuation. Increase formation scale and/or power enough to reduce singleton formation while leaving four-per-side coalitions near saturation. Jointly tune this with the steeper signed transform: broad moderate contrasts should gain skeptical force, whereas singleton and weakly formed contests should not. This stays within the arbiter's bilateral-formation mechanism and avoids repeating the explicitly rejected generic breadth-suppression package.

A narrowly lower reliability-curvature range is also worth testing because the current tanh coding compresses upper-middle validities; the current candidate did not implement that part of the prior suggestion. Keep this change small and verify Experiments 13, 17, 19, and 21. Do not alter the coherence curve, add a residual conventional-validity channel, or flatten the tally curve in the same way as iteration 2, since that package was rejected. Preserve the accepted base's parameter variation and large-margin tally settings for now. The next simulation should be accepted only if it materially raises Experiment 16 while retaining the iteration-1 results for Experiments 7, 15, 17, 20, 23, and 24; otherwise the targeted transformation has not succeeded.

**Outcome of this advice:** iter 3 candidate loss=0.1842 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Bilateral Advocacy-Attribution theory proposes that communicated expert validity has two context-sensitive meanings. By default, endorsements are treated as ordinary evidence and choices follow a nonlinear raw tally whose influence rises smoothly with both absolute and proportional vote margin before saturating. This default governs unilateral endorsement and decisive majorities. A second interpretation becomes available only when endorsers have formed coalitions on both sides and the tally remains unresolved: coalition validity then conveys the intensity of an option-level advocacy claim rather than literal evidential weight. Each claim is represented by bounded mean communicated validity, so homogeneous repetition is largely redundant. Coalition coherence varies smoothly with within-side dispersion and modulates claim salience without any threshold or trust collapse. In balanced conflict, a more reliable and coherent coalition constitutes the more salient advocacy claim and is therefore the target of pragmatic skepticism. When one coalition is highly dispersed, it is a weak unified claim; the opposing homogeneous coalition becomes the salient overclaim, causing skepticism to favor the dispersed side. The signed skeptical response is a single saturating transformation of the advocacy contrast, gated by bilateral formation, unresolved tally conflict, and coherence asymmetry. Thus moderate contrasts in well-formed bilateral coalitions are not attenuated twice, whereas unilateral and weakly formed conflicts remain protected. Stable but constrained individual differences affect bilateral-skepticism propensity and choice precision.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    # Bilateral Advocacy-Attribution model. History is intentionally ignored:\n    # without correctness feedback, past choices provide no basis for learning\n    # the experts' diagnosticities.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    n_features = stimulus.shape[1]\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match n_features {n_features}.\"\n        )\n\n    # Positive votes favor A and negative votes favor B. Features on which the\n    # options agree are nondiscriminating and belong to neither coalition.\n    signed_votes = np.sign(stimulus[0] - stimulus[1])\n    active = signed_votes != 0.0\n    active_count = int(np.count_nonzero(active))\n    tally_margin = float(np.sum(signed_votes))\n    margin_magnitude = abs(tally_margin)\n\n    # Smoothly bounded coding of communicated validity. This has no special\n    # endpoint category and retains sensitivity over the complete range.\n    validity = np.clip(validities, 0.5, 1.0)\n    curvature = float(parameters[\"reliability_curvature\"])\n    raw_reliability = 2.0 * (validity - 0.5)\n    reliability = np.tanh(curvature * raw_reliability) / max(\n        float(np.tanh(curvature)), 1e-12\n    )\n\n    def coalition(side):\n        members = reliability[active & (signed_votes == side)]\n        breadth = int(members.size)\n        if breadth == 0:\n            return 0.0, 0.0, 0.0, 0\n\n        # Advocacy intensity is an option-level bounded mean. Consequently,\n        # repeating identical endorsers does not multiply the claim.\n        mean_reliability = float(np.mean(members))\n        dispersion = float(np.mean((members - mean_reliability) ** 2))\n\n        # Dispersion weakens unified advocacy smoothly. The positive floor\n        # prevents the hard trust collapse of threshold-based accounts.\n        coherence_floor = float(parameters[\"coherence_floor\"])\n        coherence_scale = float(parameters[\"coherence_scale\"])\n        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(\n            -dispersion / max(coherence_scale, 1e-12)\n        )\n\n        # A coherent reliable coalition is a salient advocacy claim. A highly\n        # dispersed coalition remains represented, but as a weaker unified act.\n        salience = mean_reliability * coherence\n        return mean_reliability, float(coherence), float(salience), breadth\n\n    mean_a, coherence_a, salience_a, breadth_a = coalition(1.0)\n    mean_b, coherence_b, salience_b, breadth_b = coalition(-1.0)\n\n    if active_count == 0:\n        proportional_margin = 0.0\n    else:\n        proportional_margin = margin_magnitude / float(active_count)\n\n    # Default nonlinear tally: absolute margin supplies sample strength, while\n    # proportional margin supplies agreement. Both effects are continuous and\n    # the resulting evidence saturates for decisive majorities.\n    if margin_magnitude < 1e-12:\n        absolute_resolution = 0.0\n        tally_evidence = 0.0\n    else:\n        absolute_resolution = 1.0 - np.exp(\n            -(margin_magnitude / float(parameters[\"margin_scale\"]))\n            ** float(parameters[\"margin_power\"])\n        )\n        proportional_resolution = (\n            float(parameters[\"proportion_floor\"])\n            + (1.0 - float(parameters[\"proportion_floor\"]))\n            * proportional_margin ** float(parameters[\"proportion_power\"])\n        )\n        tally_shape = absolute_resolution * proportional_resolution\n        tally_evidence = (\n            float(parameters[\"tally_strength\"])\n            * np.sign(tally_margin)\n            * tally_shape\n        )\n\n    # Reliability and coherence contrasts are relative option-level contrasts.\n    # Relative normalization allows a small informative claim opposing a 50%\n    # coalition to be salient without making validity magnitude generically\n    # produce ever-stronger skepticism.\n    salience_contrast = (salience_a - salience_b) / (\n        float(parameters[\"claim_floor\"])\n        + abs(salience_a)\n        + abs(salience_b)\n    )\n    salience_contrast = float(np.clip(salience_contrast, -1.0, 1.0))\n\n    mean_contrast = (mean_a - mean_b) / (\n        float(parameters[\"ordinary_claim_floor\"])\n        + abs(mean_a)\n        + abs(mean_b)\n    )\n    mean_contrast = float(np.clip(mean_contrast, -1.0, 1.0))\n\n    # Each side must form before validity can be attributed to opposed\n    # advocacy. The geometric combination is exactly zero for unilateral\n    # endorsement and rises smoothly from singleton to broad coalitions.\n    formation_scale = float(parameters[\"formation_scale\"])\n    formation_power = float(parameters[\"formation_power\"])\n    formation_a = 1.0 - np.exp(\n        -(float(breadth_a) / formation_scale) ** formation_power\n    ) if breadth_a > 0 else 0.0\n    formation_b = 1.0 - np.exp(\n        -(float(breadth_b) / formation_scale) ** formation_power\n    ) if breadth_b > 0 else 0.0\n    bilateral_formation = float(np.sqrt(max(formation_a * formation_b, 0.0)))\n\n    # Reliability contrast and coherence asymmetry make advocacy attribution\n    # more diagnostic. Coherence asymmetry does not select an upper-tail member:\n    # it only identifies which option-level coalition is the clearer claim.\n    contrast_readiness = float(np.tanh(\n        float(parameters[\"salience_slope\"]) * abs(salience_contrast)\n    ))\n    coherence_asymmetry = abs(coherence_a - coherence_b)\n    coherence_readiness = 1.0 + float(\n        parameters[\"coherence_asymmetry_gain\"]\n    ) * coherence_asymmetry\n    claim_readiness = float(np.clip(\n        contrast_readiness * coherence_readiness, 0.0, 1.0\n    ))\n\n    # Decisive counting continuously resolves the ambiguity that licenses an\n    # advocacy interpretation. There are no margin windows or endpoint rules.\n    decisiveness = absolute_resolution * (\n        float(parameters[\"decisiveness_floor\"])\n        + (1.0 - float(parameters[\"decisiveness_floor\"]))\n        * proportional_margin\n    )\n    unresolved = (1.0 - float(np.clip(decisiveness, 0.0, 1.0))) ** float(\n        parameters[\"resolution_power\"]\n    )\n\n    arbitration_weight = float(np.clip(\n        bilateral_formation * claim_readiness * unresolved,\n        0.0,\n        1.0,\n    ))\n\n    # Outside bilateral unresolved conflict, mean validity is ordinary evidence.\n    # This contribution is deliberately modest because the raw tally remains\n    # the default, and it vanishes as advocacy attribution takes control.\n    ordinary_evidence = (\n        float(parameters[\"ordinary_strength\"])\n        * (1.0 - arbitration_weight)\n        * mean_contrast\n    )\n\n    # Skepticism uses the signed advocacy contrast only once. Bilateral\n    # formation and unresolved conflict determine eligibility, while coherence\n    # asymmetry modulates the clarity of the option-level advocacy attribution.\n    # This avoids the previous quadratic attenuation of moderate contrasts.\n    skeptic_trait = float(parameters[\"bilateral_skepticism\"])\n    skeptic_multiplier = 1.0 + float(parameters[\"skeptic_trait_spread\"]) * (\n        skeptic_trait - 0.5\n    )\n    skeptic_multiplier = max(float(skeptic_multiplier), 0.25)\n    advocacy_direction = float(np.tanh(\n        float(parameters[\"advocacy_accumulation\"]) * salience_contrast\n    ))\n    skeptical_weight = float(np.clip(\n        bilateral_formation * unresolved,\n        0.0,\n        1.0,\n    ))\n    skeptical_evidence = -float(parameters[\"reversal_strength\"]) * (\n        skeptic_multiplier\n        * skeptical_weight\n        * coherence_readiness\n        * advocacy_direction\n    )\n\n    total_evidence = tally_evidence + ordinary_evidence + skeptical_evidence\n\n    # Choice precision is a stable, narrowly constrained individual property.\n    # Ambiguous bilateral conflict modestly reduces effective precision without\n    # generating large variance on clear tally trials.\n    precision = float(parameters[\"choice_precision\"])\n    conflict_load = arbitration_weight * (1.0 - abs(advocacy_direction))\n    precision /= 1.0 + float(parameters[\"conflict_noise\"]) * conflict_load\n    decision_variable = float(precision * total_evidence)\n\n    # Positive evidence favors A. Symmetric stable softmax guarantees exact\n    # invariance under exchanging the two displayed options.\n    logits = np.asarray(\n        [0.5 * decision_variable, -0.5 * decision_variable],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    probabilities = np.exp(logits)\n    probabilities /= np.sum(probabilities)\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= np.sum(probabilities)\n    return probabilities\n",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = float(np.sum(probabilities))\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.full(\n            probabilities.size,\n            1.0 / probabilities.size,\n            dtype=np.float64,\n        )\n    else:\n        probabilities /= total\n    return int(np.random.choice(probabilities.size, p=probabilities))\n",
  "parameters": {
    "reliability_curvature": "[1.15, 1.45]",
    "coherence_floor": "[0.24, 0.34]",
    "coherence_scale": "[0.055, 0.075]",
    "claim_floor": "[0.025, 0.045]",
    "ordinary_claim_floor": "[0.20, 0.30]",
    "formation_scale": "[1.25, 1.45]",
    "formation_power": "[2.0, 2.4]",
    "salience_slope": "[2.8, 3.5]",
    "coherence_asymmetry_gain": "[0.55, 0.85]",
    "margin_scale": "[3.55, 3.90]",
    "margin_power": "[3.0, 3.4]",
    "proportion_floor": "[0.42, 0.52]",
    "proportion_power": "[0.70, 0.95]",
    "tally_strength": "[1.85, 2.05]",
    "decisiveness_floor": "[0.60, 0.72]",
    "resolution_power": "[1.65, 2.10]",
    "ordinary_strength": "[0.12, 0.22]",
    "advocacy_accumulation": "[3.2, 4.2]",
    "reversal_strength": "[2.0, 2.3]",
    "bilateral_skepticism": "[0, 1]",
    "skeptic_trait_spread": "[0.12, 0.22]",
    "choice_precision": "[0.94, 1.08]",
    "conflict_noise": "[0.08, 0.22]",
    "lapse_rate": "[0.015, 0.055]",
    "validities": "validities"
  },
  "rationale": "This is a targeted edit of the accepted iteration-1 model. The tally, coherence function, ordinary-evidence channel, precision variation, and unilateral mechanism are unchanged. The principal change removes the double attenuation of moderate advocacy contrasts from skeptical evidence. Previously, absolute contrast entered claim readiness and the same contrast entered a shallow signed transform, making skeptical evidence approximately quadratic near zero. Skepticism now uses bilateral formation times unresolved conflict times coherence readiness times one steeper signed tanh transform. It remains exactly zero for a zero contrast or a unilateral coalition, but moderate contrasts among well-formed coalitions can now generate substantial reversal. The reversal-strength ceiling is lowered simultaneously so already-large contrasts in Experiments 7, 10, 17, 19, and 21 do not become indiscriminately more extreme.\n\nThe bilateral-formation curve is modestly sharpened: singleton sides form less readily, while four-member sides remain near saturation. This concentrates the newly strengthened response on broad configurations such as Experiment 16 and protects singleton conflicts and the categorical result in Experiment 15. Reliability curvature is lowered slightly so upper-middle communicated validities are less compressed relative to one another, improving sensitivity to the moderate coalition contrast in Experiment 16 without adding an endpoint category or exemplar channel. The original readiness calculation is retained only for the ordinary-versus-advocacy arbitration term, minimizing collateral changes to the accepted model's conventional evidence contribution. No rejected breadth-suppression factor, readiness floor, residual channel, coherence replacement, or tally recalibration is introduced."
}
```

## Usage

```json
{
  "input_tokens": 56919,
  "output_tokens": 4816
}
```
