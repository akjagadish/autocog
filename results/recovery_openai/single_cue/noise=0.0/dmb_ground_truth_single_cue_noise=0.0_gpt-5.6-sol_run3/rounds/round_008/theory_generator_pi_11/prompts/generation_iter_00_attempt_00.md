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
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_10`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Replace pi_8 with a new theory of coherence-gated, breadth-normalized skeptical comparison. The new theory should retain pi_10's empirically supported idea that internal heterogeneity weakens a coalition representation: when the opposing coalition becomes dispersed, its representation falls, increasing the relative prominence of the homogeneous coalition and therefore increasing skeptical reaction against that homogeneous coalition, producing the negative gradient in Experiment 1. Unlike pi_10, however, coalition repetition should not intrinsically produce a large increase in reactance. Add two approximately offsetting breadth processes: sublinear consensus formation and divisive normalization by total active evidence or limited attention. For equal-sized homogeneous coalitions, these should yield an almost breadth-invariant validity contrast, matching Experiment 2. Use a separate nonlinear tally channel that protects large margins but is less steep and less confident than pi_10's current channel, thereby matching Experiments 5-8 and 11 while retaining the strong tally evidence in Experiments 4 and 6. Validity should be encoded on a smooth saturating continuum, without pi_8's 95% compression notch or a literal-certainty bonus. Strong 100%-versus-50% reversals should emerge from maximal relative contrast and contextual normalization, not from an endpoint-specific mechanism. Include continuous subject variation in skeptical orientation, tally sensitivity, and noise, with heterogeneity allowed to depend on conflict composition and margin; avoid both discrete skeptic classes and blanket variance inflation. The resulting theory should jointly target the strong negative dispersion effect, negligible homogeneous breadth effect, substantial but non-ceiling majority reliance, and smooth endpoint behavior.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_10` (overall score: 0.700)

**Description**
Coalition-Level Skeptical Consensus theory proposes that people do not compare communicated validities expert by expert. They first construct a representation of each option's endorsing coalition from three attributes: its mean assertiveness, its breadth, and its internal coherence. Mean assertiveness increases smoothly with communicated validity from 50% through 100%, without endpoint bonuses, notches, or categorical boundaries. Repeated similarly assertive endorsements strengthen the coalition representation according to a sublinear consensus function, whereas heterogeneous coalitions are represented less coherently. The contrast between the two coalition representations is additionally normalized by total conflict breadth and compositional mixture. In narrow conflicts, reactance is further normalized when both coalitions make substantive claims or when neither coalition approaches the maximum of the same smooth assertiveness continuum. This leaves a maximal 100%-versus-50% contrast intact while attenuating moderate and bilateral narrow conflicts without creating an endpoint bonus or threshold. People pragmatically react against the coalition making the stronger aggregate claim, so the coalition contrast enters choice with a skeptical sign. This reactance is continuous across individuals and can be weak rather than defining discrete skeptic classes. A smooth breadth-dependent gain makes isolated conflicts less reactance-provoking than repeated consensus, while preserving compression in very broad conflicts. A separate categorical tally channel ignores validity and increases nonlinearly with the absolute vote margin. It is weak enough at ties and one-vote margins to permit anti-validity reversals, but gradually protects decisive majorities without forcing ceiling-level tally choices. A shared continuous reactance trait modulates margin curvature. This modulation is centered and analytically mean-corrected, so it preserves the population-average tally function while producing increasingly large individual differences at large margins. Independent continuous variation in tally sensitivity and decision noise supplies further subject heterogeneity. Because choices receive no correctness feedback, coalition construction is history independent; because all evidence is expressed as signed option contrasts, the model is exactly invariant to swapping A and B.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Coalition-Level Skeptical Consensus model. Choice history is deliberately
    # ignored because no outcome feedback is available for learning validity.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    n_features = stimulus.shape[1]
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a discriminating endorsement of A and -1 an endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Assertiveness is completely smooth and monotone on the communicated
    # validity scale. In particular, 90%, 95%, and 100% receive neighboring
    # values on the same linear continuum.
    validity = np.clip(validities, 0.5, 1.0)
    assertiveness = 2.0 * (validity - 0.5)

    def coalition_summary(side):
        members = assertiveness[active & (signed_votes == side)]
        breadth = int(members.size)
        if breadth == 0:
            return 0.0, 0.0, 0

        mean_assertiveness = float(np.mean(members))
        dispersion = float(np.mean((members - mean_assertiveness) ** 2))

        # Similar members form a coherent aggregate claim. Mixed validity
        # compositions reduce coherence continuously rather than triggering an
        # exception for any particular validity or coalition size.
        coherence_scale = float(parameters["coherence_scale"])
        coherence_floor = float(parameters["coherence_floor"])
        coherence = coherence_floor + (1.0 - coherence_floor) * np.exp(
            -dispersion / coherence_scale
        )

        # Repetition reinforces consensus, but with diminishing returns. This
        # differs from allocating a fixed attentional pool among individual
        # claims: adding a similar endorser always increases the summary.
        consensus_exponent = float(parameters["consensus_exponent"])
        consensus = float(breadth) ** consensus_exponent
        summary = mean_assertiveness * consensus * coherence
        return float(summary), dispersion, breadth

    summary_a, dispersion_a, breadth_a = coalition_summary(1.0)
    summary_b, dispersion_b, breadth_b = coalition_summary(-1.0)
    raw_coalition_contrast = summary_a - summary_b
    total_breadth = breadth_a + breadth_b

    if total_breadth == 0:
        contextual_contrast = 0.0
    else:
        # Broad conflicts compress pragmatic contrast. Internal mixture adds a
        # second coalition-level normalizer, weighted by coalition breadth.
        excess_breadth = max(total_breadth - 2, 0)
        breadth_normalizer = 1.0 + float(
            parameters["breadth_normalization"]
        ) * np.log1p(float(excess_breadth))

        pooled_dispersion = (
            breadth_a * dispersion_a + breadth_b * dispersion_b
        ) / float(total_breadth)
        composition_normalizer = 1.0 + float(
            parameters["composition_normalization"]
        ) * float(total_breadth) * pooled_dispersion

        contextual_contrast = raw_coalition_contrast / (
            breadth_normalizer * composition_normalizer + 1e-12
        )

        # Narrow conflicts are additionally normalized when both coalition
        # summaries are substantive or when the strongest summary remains
        # below the top of the smooth assertiveness scale. Both effects decay
        # continuously with conflict breadth. A maximal singleton contrast of
        # 100% against 50% has zero added normalization.
        narrow_weight = np.exp(-float(excess_breadth) / 2.0)
        bilateral_mass = min(abs(summary_a), abs(summary_b))
        strongest_summary = min(max(abs(summary_a), abs(summary_b)), 1.0)
        submaximal_gap = 1.0 - strongest_summary
        narrow_normalizer = 1.0 + narrow_weight * (
            float(parameters["bilateral_narrow_normalization"])
            * bilateral_mass
            + float(parameters["submaximal_narrow_normalization"])
            * submaximal_gap ** 2
        )
        contextual_contrast /= float(narrow_normalizer)

        # A singleton conflict evokes less coalition-level reactance than a
        # repeated consensus. The gain rises smoothly with breadth and remains
        # distinct from the broad-conflict normalization above.
        reinforcement_floor = float(parameters["breadth_reinforcement_floor"])
        reinforcement_gain = reinforcement_floor + (
            1.0 - reinforcement_floor
        ) * (1.0 - np.exp(-float(excess_breadth) / 2.0))
        contextual_contrast *= float(reinforcement_gain)

    # A continuous trait controls skeptical construal. It also contributes to
    # margin curvature below, permitting correlated heterogeneity without a
    # categorical skeptic/conventional mixture.
    reactance_trait = float(parameters["reactance_trait"])
    reactance_strength = float(parameters["reactance_strength"]) * (
        0.65 + 0.70 * reactance_trait
    )
    accumulated_contrast = np.tanh(
        float(parameters["coalition_accumulation"]) * contextual_contrast
    )
    skeptical_evidence = -reactance_strength * float(accumulated_contrast)

    # The raw tally is represented independently of communicated validity. Its
    # mildly superlinear margin function leaves a one-vote lead vulnerable but
    # lets decisive margins gradually dominate bounded reactance without
    # producing near-deterministic tally choices.
    tally_trait = float(parameters["tally_trait"])
    tally_sensitivity = float(parameters["tally_sensitivity"]) * (
        0.70 + 0.60 * tally_trait
    )
    base_margin_exponent = float(parameters["margin_exponent"])
    margin_coupling = float(parameters["reactance_margin_coupling"])
    exponent_deviation = margin_coupling * (reactance_trait - 0.5)
    margin_exponent = base_margin_exponent + exponent_deviation

    if abs(tally_margin) < 1e-12:
        tally_evidence = 0.0
    else:
        margin_magnitude = abs(tally_margin)

        # Centering the exponent deviation preserves one-vote evidence exactly.
        # For larger margins, divide by its analytic population mean under the
        # continuous uniform reactance trait. Thus widening the coupling adds
        # heterogeneity that grows with margin without changing the accepted
        # population-average tally curve.
        mean_span = 0.5 * margin_coupling * np.log(margin_magnitude)
        if abs(mean_span) < 1e-8:
            heterogeneity_correction = 1.0
        else:
            heterogeneity_correction = float(np.sinh(mean_span) / mean_span)

        tally_evidence = (
            tally_sensitivity
            * np.sign(tally_margin)
            * margin_magnitude ** margin_exponent
            / heterogeneity_correction
        )

    total_evidence = tally_evidence + skeptical_evidence

    # Decision noise varies continuously and independently across subjects.
    noise_trait = float(parameters["noise_trait"])
    choice_sensitivity = float(parameters["choice_sensitivity"]) * (
        0.65 + 0.70 * noise_trait
    )
    decision_variable = choice_sensitivity * total_evidence

    # Positive evidence favors A. Symmetric logits guarantee exact option-swap
    # symmetry, including on tied and mirrored trials.
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
- consensus_exponent: [0.68, 0.84]
- coherence_scale: [0.025, 0.090]
- coherence_floor: [0.42, 0.68]
- breadth_normalization: [0.05, 0.14]
- composition_normalization: [0.8, 2.2]
- bilateral_narrow_normalization: [2.0, 4.0]
- submaximal_narrow_normalization: [1.0, 2.0]
- breadth_reinforcement_floor: [0.62, 0.76]
- coalition_accumulation: [1.15, 1.75]
- reactance_strength: [2.4, 3.7]
- reactance_trait: [0, 1]
- tally_sensitivity: [0.20, 0.38]
- tally_trait: [0, 1]
- margin_exponent: [1.15, 1.40]
- reactance_margin_coupling: [-0.50, 0.60]
- choice_sensitivity: [0.78, 1.24]
- noise_trait: [0, 1]
- lapse_rate: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3250 (var=0.0112)
- Experiment 2: real=89.1422 (var=7.5896) vs this=89.8046 (var=9.1438)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.3654 (var=0.0036)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.9350 (var=0.0031)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.4282 (var=0.0521)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.9150 (var=0.0039)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.0811 (var=0.0040)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-3.8304 (var=0.0211)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.1905 (var=0.0047)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.8658 (var=0.0100)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.1292 (var=0.0058)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.1528 (var=0.0051)
- Experiment 13: real=0.0094 (var=0.0050) vs this=0.0109 (var=0.0179)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.1275 (var=0.0095)
- Experiment 15: real=0.0000 (var=0.2400) vs this=0.0000 (var=0.1716)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.7981 (var=0.0069)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=-0.0869 (var=0.0165)
- Experiment 18: real=0.0058 (var=0.0053) vs this=0.1996 (var=0.0174)


---

### `pi_9` (overall score: 0.658)

**Description**
Finite-Pool Pragmatic Claim Competition theory proposes that binary expert endorsements generate two separable signals. A categorical vote channel counts every discriminating expert equally and therefore represents coalition consensus independently of communicated validity. A pragmatic claim channel interprets an expert's stated validity as the assertiveness of the expert's claim. Claim assertiveness is encoded by a smooth linear function of the communicated scale, including from 90% through 100%. Most people respond skeptically to assertive claims, although the polarity and magnitude of this interpretation vary continuously across individuals and can include weakly conventional interpretations. Claim processing draws on a finite attentional pool. Every discriminating expert, including a nominally 50%-valid expert whose own claim carries no directional strength, competes for that pool. Competition increases both with active-set size and with similarity among simultaneously active claims. Consequently, an isolated high-validity claim can receive substantial attention, whereas its influence declines when embedded in a broad or compositionally redundant coalition. This single allocation rule produces breadth and composition effects without endpoint bonuses, validity notches, or condition-specific exemptions. Pragmatic claim evidence is bounded, while categorical tally evidence continues to increase with vote margin. Skepticism can thus reverse tied or weak tallies but normally yields to a decisive raw-vote coalition. In the absence of outcome feedback, neither signal is recalibrated from choice history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Finite-Pool Pragmatic Claim Competition model. History is deliberately
    # ignored because previous choices provide no feedback about cue accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # +1 denotes a unique endorsement of A, -1 a unique endorsement of B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))

    # Categorical votes and interpreted numerical claims are represented in
    # separate channels. This exactly linear assertiveness code has no special
    # status for 95% or 100% and therefore has zero intrinsic 90/95/100
    # curvature.
    v = np.clip(validities, 0.5, 1.0)
    claim_assertiveness = 2.0 * (v - 0.5)

    if not np.any(active):
        allocated_claim_contrast = 0.0
    else:
        active_signs = signed_votes[active]
        active_claims = claim_assertiveness[active]
        active_count = active_claims.size

        # All discriminating experts enter the competition matrix. Thus a 50%
        # expert has zero directional claim strength but still consumes claim-
        # interpretation capacity. Competition has one general form: a global
        # component plus a similarity-dependent component.
        width = float(parameters["similarity_width"])
        competition_floor = float(parameters["competition_floor"])
        distances = np.abs(active_claims[:, None] - active_claims[None, :])
        similarity = np.exp(-distances / width)
        pairwise_competition = (
            competition_floor + (1.0 - competition_floor) * similarity
        )
        np.fill_diagonal(pairwise_competition, 0.0)
        competition_load = np.sum(pairwise_competition, axis=1)

        # Finite capacity determines the attention retained by each claim.
        # Similar claims in broad coalitions strongly compete, while isolated
        # claims receive most of the available processing weight.
        capacity = float(parameters["attentional_capacity"])
        attention = capacity / (capacity + competition_load)
        allocated_claim_contrast = float(
            np.dot(active_signs, attention * active_claims)
        )

    # Pragmatic interpretation is continuous across subjects. Positive values
    # encode skepticism/reactance and reverse the communicated claim contrast;
    # zero ignores it; negative values produce a conventional interpretation.
    # The declared range makes skepticism predominant without introducing a
    # categorical latent class.
    accumulation = float(parameters["claim_accumulation"])
    bounded_claim = float(np.tanh(accumulation * allocated_claim_contrast))
    claim_evidence = -float(parameters["claim_construal"]) * bounded_claim

    # The categorical vote channel grows smoothly with margin. Its saturation
    # level exceeds the maximum typical claim contribution, so large coalitions
    # reliably dominate even though claim evidence can reverse weak margins.
    tally_scale = float(parameters["tally_saturation"])
    tally_evidence = (
        float(parameters["tally_sensitivity"])
        * tally_scale
        * np.tanh(tally_margin / tally_scale)
    )

    total_evidence = tally_evidence + claim_evidence
    decision_variable = float(parameters["choice_sensitivity"]) * total_evidence

    # Positive evidence favors A. Symmetric logits ensure exact option-swap
    # invariance.
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
- claim_construal: [-0.1, 3.6]
- claim_accumulation: [1.3, 2.0]
- attentional_capacity: [0.9, 1.6]
- competition_floor: [0.30, 0.52]
- similarity_width: [0.22, 0.38]
- tally_sensitivity: [0.55, 0.95]
- tally_saturation: [4.0, 5.8]
- choice_sensitivity: [0.68, 1.18]
- lapse_rate: [0.0, 0.10]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3208 (var=0.0107)
- Experiment 2: real=89.1422 (var=7.5896) vs this=59.9467 (var=3.7118)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.3440 (var=0.0028)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.9294 (var=0.0023)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.4666 (var=0.0908)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.9035 (var=0.0023)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.3269 (var=0.0321)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-2.4502 (var=0.0542)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.3376 (var=0.0157)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.7475 (var=0.0177)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.4628 (var=0.0171)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.3469 (var=0.0105)
- Experiment 13: real=0.0094 (var=0.0050) vs this=-0.0147 (var=0.0077)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.1044 (var=0.0094)
- Experiment 15: real=0.0000 (var=0.2400) vs this=0.0000 (var=0.1204)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.5850 (var=0.0057)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=0.0256 (var=0.0200)
- Experiment 18: real=0.0058 (var=0.0053) vs this=0.0392 (var=0.0136)


---

### `pi_8` (overall score: 0.629)

**Description**
Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Contextual Anti-Reliability Arbitration. History is intentionally ignored:
    # without outcome feedback, prior choices cannot reveal expert accuracy.
    stimulus = np.asarray(state, dtype=np.float64)
    if stimulus.ndim != 2 or stimulus.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stimulus.shape}."
        )

    n_features = stimulus.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} does not match "
            f"n_features {n_features}."
        )

    # Positive endorsements favor A and negative endorsements favor B.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    active = signed_votes != 0.0
    tally_margin = float(np.sum(signed_votes))
    active_count = int(np.count_nonzero(active))

    # Encode the assertiveness of every communicated above-chance claim.
    # Moderate claims already carry pragmatic suspicion. The ordinary channel
    # is strongly compressed near 95%, preventing near-endpoint claims from
    # behaving like literal-certainty claims.
    v = np.clip(validities, 0.5, 1.0)
    above_chance = np.clip(2.0 * (v - 0.5), 0.0, 1.0)
    ordinary_base = np.power(above_chance, 0.82)

    def logistic(x):
        if x >= 0.0:
            return 1.0 / (1.0 + np.exp(-x))
        ex = np.exp(x)
        return ex / (1.0 + ex)

    upper_compression = np.array(
        [logistic(50.0 * (float(x) - 0.86)) for x in above_chance],
        dtype=np.float64,
    )
    ordinary_claim = ordinary_base * (1.0 - 0.55 * upper_compression)

    # Endpoint distrust is reserved for claims extremely close to certainty.
    endpoint_low = logistic(100.0 * (0.0 - 0.98))
    endpoint_high = logistic(100.0 * (1.0 - 0.98))
    endpoint_curve = np.array(
        [
            (logistic(100.0 * (float(x) - 0.98)) - endpoint_low)
            / (endpoint_high - endpoint_low)
            for x in above_chance
        ],
        dtype=np.float64,
    )
    endpoint_curve = np.clip(endpoint_curve, 0.0, 1.0)

    if active_count == 0:
        anti_reliability = 0.0
    else:
        active_signs = signed_votes[active]
        active_ordinary = ordinary_claim[active]
        active_endpoint = endpoint_curve[active]
        active_above_chance = above_chance[active]

        # Endpoint distrust is preserved against a neutral 50% opponent, but
        # attenuated by the smooth maximum ordinary strength on the opposing
        # side. The attenuation applies to the endpoint cue's total distrust,
        # not merely to its endpoint bonus.
        nonendpoint_ordinary = active_ordinary * (1.0 - active_endpoint)
        endpoint_focus = np.ones(active_count, dtype=np.float64)
        for i in range(active_count):
            opposing = nonendpoint_ordinary[active_signs != active_signs[i]]
            if opposing.size > 0:
                attention_logits = 6.0 * opposing
                attention_logits -= np.max(attention_logits)
                attention = np.exp(attention_logits)
                attention /= attention.sum()
                opposing_peak = float(np.dot(attention, opposing))
                endpoint_focus[i] = 1.0 / (
                    1.0 + 3.0 * opposing_peak ** 1.4
                )

        focused_endpoint = active_endpoint * endpoint_focus
        active_claims = (
            active_ordinary * (1.0 - active_endpoint)
            + (
                active_ordinary
                + float(parameters["endpoint_distrust"])
            ) * focused_endpoint
        )
        signed_claim_contrast = float(np.dot(active_signs, active_claims))
        total_claim_mass = float(np.sum(active_claims))

        # Relative contrast measures whether distrust is directionally
        # concentrated rather than spread across two similarly assertive sides.
        concentration = abs(signed_claim_contrast) / (total_claim_mass + 1e-12)
        concentration = float(np.clip(concentration, 0.0, 1.0))
        focus_floor = float(parameters["concentration_floor"])
        focus = focus_floor + (1.0 - focus_floor) * concentration ** 1.35

        # Estimate heterogeneity within each endorsing coalition. Mixing claims
        # of substantially different strength on the same side makes their
        # pragmatic interpretation less coherent and hence more strongly diluted.
        positive_claims = active_claims[active_signs > 0.0]
        negative_claims = active_claims[active_signs < 0.0]

        def side_dispersion(values):
            if values.size <= 1:
                return 0.0
            mean_value = float(np.mean(values))
            return float(np.mean((values - mean_value) ** 2))

        positive_dispersion = side_dispersion(positive_claims)
        negative_dispersion = side_dispersion(negative_claims)
        within_dispersion = (
            positive_claims.size * positive_dispersion
            + negative_claims.size * negative_dispersion
        ) / float(active_count)

        breadth_exponent = float(parameters["coalition_dilution_exponent"])
        breadth_normalizer = float(active_count) ** breadth_exponent
        heterogeneity_normalizer = 1.0 + (
            float(parameters["composition_dilution"])
            * float(active_count)
            * within_dispersion
        )

        # Within-side dispersion is exactly zero when each option has only one
        # active endorser. Across-active dispersion therefore supplies a second,
        # smoothly sparse-weighted composition diagnostic. It is switched off
        # for a literal endpoint claim against a neutral claim, preserving the
        # isolated 100%-versus-50% distrust response.
        across_dispersion = float(np.mean((active_claims - np.mean(active_claims)) ** 2))
        sparse_weight = (2.0 / float(active_count)) ** 2
        endpoint_neutral_exemption = float(np.max(active_endpoint)) * (
            1.0 - float(np.max(nonendpoint_ordinary))
        )
        across_gate = 1.0 - np.clip(endpoint_neutral_exemption, 0.0, 1.0)
        across_normalizer = 1.0 + (
            float(parameters["across_composition_dilution"])
            * sparse_weight
            * across_gate
            * across_dispersion
        )

        contextual_contrast = (
            signed_claim_contrast * focus
            / (
                breadth_normalizer
                * heterogeneity_normalizer
                * across_normalizer
                + 1e-12
            )
        )

        # Moderate pragmatic claims receive a small accumulation boost in broad
        # but not maximally dense coalitions. The smooth validity and breadth
        # gates avoid strengthening sparse 95%-claim conflicts or dense diffuse
        # conflicts indiscriminately.
        strongest_assertiveness = float(np.max(active_above_chance))
        moderate_gate = logistic(12.0 * (0.72 - strongest_assertiveness))
        breadth_peak = (
            (1.0 - np.exp(-max(active_count - 2, 0) / 2.0))
            * np.exp(-((active_count - 7.0) / 4.0) ** 2)
        )
        contextual_contrast *= float(
            1.0 + 0.15 * moderate_gate * breadth_peak
        )

        # The anti-reliability signal is bounded and continuously heterogeneous
        # across subjects. Its sign opposes the communicated validity contrast.
        accumulated = np.tanh(
            float(parameters["anti_accumulation"]) * contextual_contrast
        )
        anti_reliability = -float(parameters["anti_reliability_strength"]) * float(accumulated)

        # A smooth arbitration function cedes control to the tally as its margin
        # becomes decisive; there are no exact-tie or one-margin special cases.
        margin_scale = float(parameters["margin_protection_scale"])
        margin_power = float(parameters["margin_protection_power"])
        margin_gate = 1.0 / (
            1.0 + (abs(tally_margin) / margin_scale) ** margin_power
        )
        anti_reliability *= float(margin_gate)

    # Raw votes retain their categorical status but saturate gradually, limiting
    # overconfidence while preserving dominance by large vote coalitions.
    tally_evidence = 4.5 * np.tanh(tally_margin / 4.5)
    total_evidence = tally_evidence + anti_reliability

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = float(parameters["choice_sensitivity"]) * normalized_evidence

    # Confidence separates most strongly at intermediate margins. The bounded
    # gain is zero through margin one and decreases again for very large margins,
    # increasing heterogeneous margin sensitivity without indiscriminately
    # sharpening every decisive coalition.
    margin_excess = max(abs(tally_margin) - 1.0, 0.0)
    intermediate_margin_shape = (
        0.5 * margin_excess * np.exp(1.0 - margin_excess / 2.0)
        if margin_excess > 0.0 else 0.0
    )
    margin_temperature = 1.0 + float(parameters["margin_temperature_gain"]) * intermediate_margin_shape
    decision_variable *= float(margin_temperature)

    # Positive evidence favors A. Symmetric logits guarantee that swapping the
    # displayed options swaps their probabilities exactly.
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
            probabilities.size, 1.0 / probabilities.size, dtype=np.float64
        )
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- anti_reliability_strength: [3.0, 5.6]
- anti_accumulation: [2.0, 3.5]
- endpoint_distrust: [1.10, 2.05]
- coalition_dilution_exponent: [0.46, 0.72]
- composition_dilution: [0.8, 2.4]
- across_composition_dilution: [1.0, 3.5]
- concentration_floor: [0.38, 0.62]
- margin_protection_scale: [1.55, 2.25]
- margin_protection_power: [2.2, 3.4]
- evidence_scale: [2.2, 3.0]
- choice_sensitivity: [1.45, 2.65]
- margin_temperature_gain: [0.0, 0.32]
- lapse_rate: [0.0, 0.10]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3383 (var=0.0039)
- Experiment 2: real=89.1422 (var=7.5896) vs this=56.1425 (var=1.2063)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.4106 (var=0.0035)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8544 (var=0.0058)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=0.0104 (var=0.0390)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8592 (var=0.0031)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.2347 (var=0.0051)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-4.4717 (var=0.0102)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.2814 (var=0.0029)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.8408 (var=0.0059)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.2039 (var=0.0054)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.1583 (var=0.0037)
- Experiment 13: real=0.0094 (var=0.0050) vs this=0.0569 (var=0.0085)
- Experiment 14: real=0.1781 (var=0.0129) vs this=0.1467 (var=0.0083)
- Experiment 15: real=0.0000 (var=0.2400) vs this=1.0000 (var=0.1600)
- Experiment 16: real=0.8819 (var=0.0059) vs this=0.7128 (var=0.0076)
- Experiment 17: real=-0.3250 (var=0.0394) vs this=0.0656 (var=0.0171)
- Experiment 18: real=0.0058 (var=0.0053) vs this=0.0329 (var=0.0084)


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
  "theory": "Coherence-Gated Relative Coalition Comparison theory proposes that people first compress the endorsements for each option into coalition-level representations rather than summing communicated validities expert by expert. A coalition's representation combines its mean smoothly encoded assertiveness, a sublinear consensus term, and a coherence gate that weakens internally heterogeneous coalitions. The two representations are compared only after divisive normalization by conflict breadth and total active assertiveness. Sublinear consensus and breadth normalization approximately cancel for equal-sized homogeneous coalitions, making their validity contrast nearly invariant to coalition repetition. Dispersion does not cancel: when one coalition becomes heterogeneous, its representation loses coherence, making the opposing homogeneous coalition relatively prominent and therefore a stronger target of skeptical interpretation. A separate validity-blind tally channel grows nonlinearly but concavely with vote margin, protecting decisive majorities without producing ceiling-level confidence. Validity is represented by one smooth saturating continuum with neither a 95% notch nor a certainty bonus; strong reversal of a 100% claim against a 50% claim follows from maximal normalized contrast. Skeptical orientation, tally sensitivity, and decision noise vary continuously across people. Skeptical heterogeneity is selectively amplified by coalition dispersion and margin, while noise is selectively amplified in compositionally conflicted, low-margin comparisons. With no outcome feedback, the representations remain history independent and option-swap symmetric.",
  "predict": "def predict(parameters, state, history):\n    # Coherence-Gated Relative Coalition Comparison. Choice history is ignored\n    # because previous choices are not accompanied by correctness feedback.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    n_features = stimulus.shape[1]\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match n_features {n_features}.\"\n        )\n\n    # Positive signs endorse A and negative signs endorse B. Features on which\n    # the options agree carry neither a categorical vote nor a directional claim.\n    signed_votes = np.sign(stimulus[0] - stimulus[1])\n    active = signed_votes != 0.0\n    tally_margin = float(np.sum(signed_votes))\n    total_breadth = int(np.count_nonzero(active))\n\n    # A single smooth, monotone, saturating validity code. The normalization\n    # maps 50% to zero and 100% to one without assigning either 95% or 100% a\n    # qualitatively distinct representation.\n    validity = np.clip(validities, 0.5, 1.0)\n    curvature = float(parameters[\"validity_curvature\"])\n    endpoint = np.tanh(0.5 * curvature)\n    assertiveness = np.tanh(curvature * (validity - 0.5)) / max(endpoint, 1e-12)\n\n    skeptic_trait = float(parameters[\"skeptical_orientation\"])\n\n    def summarize(side):\n        members = assertiveness[active & (signed_votes == side)]\n        breadth = int(members.size)\n        if breadth == 0:\n            return 0.0, 0.0, 0\n\n        mean_strength = float(np.mean(members))\n        dispersion = float(np.mean((members - mean_strength) ** 2))\n\n        # The same continuous skeptical trait changes sensitivity to internal\n        # inconsistency. This creates composition-specific heterogeneity rather\n        # than indiscriminately inflating response variance.\n        scale = float(parameters[\"coherence_scale\"]) * np.exp(\n            float(parameters[\"coherence_trait_coupling\"])\n            * (skeptic_trait - 0.5)\n        )\n        floor = float(parameters[\"coherence_floor\"])\n        coherence = floor + (1.0 - floor) * np.exp(\n            -dispersion / max(scale, 1e-12)\n        )\n\n        consensus = float(breadth) ** float(parameters[\"consensus_exponent\"])\n        representation = mean_strength * consensus * coherence\n        return float(representation), dispersion, breadth\n\n    rep_a, dispersion_a, breadth_a = summarize(1.0)\n    rep_b, dispersion_b, breadth_b = summarize(-1.0)\n\n    if total_breadth == 0:\n        contextual_contrast = 0.0\n        pooled_dispersion = 0.0\n    else:\n        pooled_dispersion = (\n            breadth_a * dispersion_a + breadth_b * dispersion_b\n        ) / float(total_breadth)\n\n        # For equal breadth n on both sides, homogeneous representations grow\n        # as n**alpha while this breadth unit grows by the same factor. Thus\n        # repetition alone produces almost no change in validity contrast.\n        alpha = float(parameters[\"consensus_exponent\"])\n        breadth_unit = (float(total_breadth) / 2.0) ** alpha\n\n        # Total active assertiveness consumes limited comparison capacity. This\n        # is a relative, composition-sensitive normalization and therefore\n        # preserves a maximal 100%-versus-50% singleton contrast.\n        active_mean = float(np.mean(assertiveness[active]))\n        denominator = breadth_unit * (\n            float(parameters[\"attention_floor\"])\n            + float(parameters[\"evidence_normalization\"]) * active_mean\n        )\n        raw_relative_contrast = (rep_a - rep_b) / max(denominator, 1e-12)\n        contextual_contrast = float(np.tanh(\n            float(parameters[\"contrast_accumulation\"])\n            * raw_relative_contrast\n        ))\n\n    # Stronger relative coalition claims provoke skepticism. Trait effects are\n    # selectively coupled to composition and margin, so individual differences\n    # grow where construal is genuinely ambiguous rather than on every trial.\n    dispersion_index = pooled_dispersion / (pooled_dispersion + 0.04)\n    margin_index = abs(tally_margin) / (abs(tally_margin) + 2.0)\n    skeptical_multiplier = 0.70 + 0.60 * skeptic_trait\n    skeptical_multiplier *= (\n        1.0\n        + float(parameters[\"composition_trait_gain\"])\n        * (skeptic_trait - 0.5)\n        * dispersion_index\n    )\n    skeptical_multiplier *= (\n        1.0\n        + float(parameters[\"skeptic_margin_coupling\"])\n        * (skeptic_trait - 0.5)\n        * margin_index\n    )\n    skeptical_multiplier = max(float(skeptical_multiplier), 0.05)\n    skeptical_evidence = -float(parameters[\"skeptical_strength\"]) * (\n        skeptical_multiplier * contextual_contrast\n    )\n\n    # The independent tally channel is validity blind. Its concave nonlinear\n    # growth differentiates margins and protects large majorities, but is less\n    # steep and less confident than a superlinear margin rule.\n    tally_trait = float(parameters[\"tally_trait\"])\n    tally_gain = float(parameters[\"tally_sensitivity\"]) * (\n        0.75 + 0.50 * tally_trait\n    )\n    if abs(tally_margin) < 1e-12:\n        tally_evidence = 0.0\n    else:\n        magnitude = abs(tally_margin)\n        tally_curve = (1.0 + magnitude) ** float(parameters[\"tally_exponent\"]) - 1.0\n        tally_evidence = tally_gain * np.sign(tally_margin) * tally_curve\n\n    total_evidence = tally_evidence + skeptical_evidence\n\n    # Noise varies continuously, but conflict-dependent noise is concentrated\n    # in dispersed, low-margin comparisons instead of being a blanket variance\n    # adjustment. The independent trait also supplies ordinary subject-level\n    # differences in response consistency.\n    noise_trait = float(parameters[\"noise_trait\"])\n    base_gain = float(parameters[\"choice_sensitivity\"]) * (\n        0.72 + 0.56 * noise_trait\n    )\n    low_margin_weight = 1.0 / (1.0 + abs(tally_margin) / 2.0)\n    conflict_load = dispersion_index * low_margin_weight\n    effective_gain = base_gain / (\n        1.0\n        + float(parameters[\"conflict_noise\"])\n        * (1.20 - 0.40 * noise_trait)\n        * conflict_load\n    )\n    decision_variable = float(effective_gain * total_evidence)\n\n    # Positive evidence favors A. Symmetric logits guarantee exact invariance\n    # under swapping the two displayed options.\n    logits = np.array(\n        [0.5 * decision_variable, -0.5 * decision_variable],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    probabilities = np.exp(logits)\n    probabilities /= probabilities.sum()\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= probabilities.sum()\n    return probabilities\n",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = float(np.sum(probabilities))\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.full(\n            probabilities.size,\n            1.0 / probabilities.size,\n            dtype=np.float64,\n        )\n    else:\n        probabilities /= total\n    return int(np.random.choice(probabilities.size, p=probabilities))\n",
  "parameters": {
    "validity_curvature": "[1.6, 3.0]",
    "consensus_exponent": "[0.58, 0.76]",
    "coherence_scale": "[0.010, 0.040]",
    "coherence_floor": "[0.12, 0.36]",
    "coherence_trait_coupling": "[-0.50, 0.50]",
    "attention_floor": "[0.28, 0.46]",
    "evidence_normalization": "[0.48, 0.76]",
    "contrast_accumulation": "[1.15, 1.75]",
    "skeptical_strength": "[1.85, 2.85]",
    "skeptical_orientation": "[0, 1]",
    "composition_trait_gain": "[0.20, 0.75]",
    "skeptic_margin_coupling": "[-0.30, 0.45]",
    "tally_sensitivity": "[0.62, 0.88]",
    "tally_trait": "[0, 1]",
    "tally_exponent": "[0.72, 0.94]",
    "choice_sensitivity": "[0.82, 1.18]",
    "noise_trait": "[0, 1]",
    "conflict_noise": "[0.10, 0.55]",
    "lapse_rate": "[0.0, 0.08]",
    "validities": "validities"
  },
  "rationale": "The model directly addresses the central failures of the earlier candidates. First, coherence is applied to each coalition before comparison, so dispersing an opposing coalition lowers that coalition's representation and increases the relative prominence—and skeptical cost—of the homogeneous coalition. The low coherence floor and narrow continuous scale make this mechanism strong enough to generate the pronounced negative dispersion gradient rather than the weak gradient produced by pi_10. Second, homogeneous repetition has no intrinsic reactance boost. Coalition consensus grows as n raised to a sublinear exponent, while the conflict denominator grows by the matching breadth factor. For equal-sized homogeneous coalitions these processes approximately cancel, targeting the negligible breadth effects in Experiments 5 and 18. Third, total active assertiveness supplies an additional limited-attention normalization: bilateral or diffuse evidence is compressed, whereas a smooth 100%-versus-50% singleton comparison remains maximally asymmetric. This preserves the endpoint reversal without any endpoint-specific bonus, exemption, or 95% compression notch. Fourth, the separate tally channel is concave rather than superlinear. It remains margin-sensitive enough to support the strong tally Bayes factor and decisive-majority choices, but should reduce the excessive majority confidence shown by pi_10 in Experiments 4 and 6 and its excessive breadth-dependent effects in Experiment 5. Finally, continuous skeptical, tally, and noise traits replace discrete classes. Their interactions with dispersion and margin create structured subject heterogeneity only where conflict composition makes it psychologically relevant, avoiding blanket variance inflation."
}
```

## Usage

```json
{
  "input_tokens": 33681,
  "output_tokens": 4812
}
```
