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

Replace pi_7 with a new Contextual Anti-Reliability Arbitration theory. It should propose that communicated numerical validities are commonly treated as socially or pragmatically suspect rather than as direct diagnostic weights: above-chance claims generally contribute evidence opposite to their stated direction, including moderate 60%-70% claims, while near-perfect claims can evoke an additional endpoint distrust response. Unlike pi_6, this should not posit an almost categorical 99% reversed-construal subgroup or exact-tie and one-margin gates. Instead, use continuous subject heterogeneity in anti-reliability strength and a continuous composition-sensitive arbitration rule. Negative claim evidence should be strongest when the validity contrast is concentrated and the raw tally is tied or weak, should be divisively normalized or diluted in dense heterogeneous coalitions, and should lose control smoothly as the absolute tally margin becomes decisive. This architecture should preserve option-swap symmetry and history independence. It should reproduce the very low conventional-validity choice rates in Experiments 1, 2, 9, and 10; permit near-indifference and minority choices in Experiments 3, 5, and 7; preserve large-coalition tally dominance in Experiments 4, 6, and 8; and include an endpoint-specific distrust component strong enough for Experiment 12. To address Experiment 11, anti-reliability influence should attenuate when a high-validity side is embedded in a broad coalition or when opposing evidence is compositionally diffuse, without relying on pi_6's discontinuous special cases. Parameter ranges should allow more anti-validity strength and heterogeneity than pi_6 currently does, because Experiments 1 and 2 are more extreme—and Experiment 2 more variable—than its simulations predict. This new theory would be a genuine competitor to pi_6: both predict broad anti-validity behavior, but they differ on whether it reflects categorical scale reversal with hand-built gates or continuous pragmatic distrust regulated by evidence concentration and coalition composition.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.894)

**Description**
Latent Validity-Construal Tally theory proposes that participants first encode each discriminating expert as one categorical vote and form an unweighted vote margin. Communicated validities are represented separately as monotonic diagnostic-strength codes whose signed sum is compressed into bounded reliability evidence. Crucially, participants differ in their interpretation of the communicated validity scale. The dominant latent construal treats lower stated values as safer or more diagnostic, reversing the direction of reliability evidence. Small subgroups either disregard the validity information or interpret it conventionally. Because reliability evidence is bounded, the dominant reversed construal can overturn a one-vote tally or resolve an exact tally, but it cannot overturn a decisive many-versus-few coalition. When reliability opposes a one-vote tally, its influence is selectively attenuated if that tally advantage arises from a compact, proportionally coherent coalition; reliability remains nearly unchanged in diffuse many-versus-many conflicts. Reliability-based commitment applies only when validity overturns a nonzero tally; exact-tally decisions remain graded rather than receiving the same categorical amplification. In exact tallies, reliability is further gated by its relative salience: the absolute signed validity imbalance is compared with the total encoded validity strength on both sides. Diffuse conflicts between two substantial coalitions receive little reliability influence, whereas endpoint contrasts involving a near-chance expert remain salient. The raw tally is also smoothly saturated, reducing excessive confidence differences between moderate and large margins while retaining its dominance in decisive coalitions. The model is reversal-symmetric and history-independent because no outcome feedback is available.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Latent Validity-Construal Tally model. History is intentionally ignored:
    # choices without correctness feedback cannot reveal expert accuracy.
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

    # +1 denotes an expert uniquely endorsing A, -1 an expert uniquely
    # endorsing B, and zero a nondiscriminating expert.
    signed_votes = np.sign(stimulus[0] - stimulus[1])
    tally_margin = float(np.sum(signed_votes))

    # The communicated scale is encoded monotonically before its psychological
    # direction is selected. Near-chance experts retain a categorical vote but
    # supply little reliability evidence. Extreme values are compressed.
    v = np.clip(validities, 0.5, 1.0)
    validity_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)
    signed_validity_sum = float(np.dot(signed_votes, validity_code))

    # Accumulation is cue-by-cue but saturates, so validity can settle a weak
    # conflict without growing enough to defeat a large tally coalition.
    accumulation = float(parameters["reliability_accumulation"])
    bounded_reliability = float(np.tanh(accumulation * signed_validity_sum))

    # In an exact tally, distinguish a concentrated diagnostic imbalance from
    # a diffuse conflict between two substantial endorsing coalitions. The
    # normalization is used only at exact ties, leaving every nonzero-tally
    # prediction and its successful margin profile unchanged.
    if abs(tally_margin) < 1e-12:
        total_validity_strength = float(
            np.dot(np.abs(signed_votes), validity_code)
        )
        if total_validity_strength > 1e-12:
            relative_salience = np.clip(
                abs(signed_validity_sum) / total_validity_strength, 0.0, 1.0
            )
        else:
            relative_salience = 0.0

        threshold = float(parameters["exact_tie_salience_threshold"])
        slope = float(parameters["exact_tie_salience_slope"])
        raw_gate = 1.0 / (
            1.0 + np.exp(-slope * (relative_salience - threshold))
        )
        endpoint_gate = 1.0 / (
            1.0 + np.exp(-slope * (1.0 - threshold))
        )
        bounded_reliability *= float(raw_gate / endpoint_gate)

    # The continuous latent index implements a task-invariant population
    # mixture. Nearly all participants reverse the instructed scale; very small
    # groups ignore it or use its conventional direction.
    construal_quantile = float(parameters["construal_quantile"])
    if construal_quantile < 0.99:
        validity_orientation = -1.0
    elif construal_quantile < 0.998:
        validity_orientation = 0.0
    else:
        validity_orientation = 1.0

    orientation_strength = float(parameters["orientation_strength"])
    reliability_evidence = (
        validity_orientation * orientation_strength * bounded_reliability
    )

    # In a nonzero one-vote conflict, coalition structure regulates only the
    # opposing reliability signal. A compact proportional majority protects
    # its tally, whereas a diffuse many-versus-many margin leaves validity
    # almost unchanged. Larger tally margins and aligned reliability are not
    # modified, preserving the established margin-confidence profile.
    discriminating_count = int(np.count_nonzero(signed_votes))
    if (
        discriminating_count > 0
        and 0.5 < abs(tally_margin) < 1.5
        and reliability_evidence * tally_margin < 0.0
    ):
        relative_consensus = abs(tally_margin) / float(discriminating_count)
        coherent_conflict = 1.0 / (
            1.0 + np.exp(-24.0 * (relative_consensus - 0.25))
        )
        conflict_floor = float(parameters["conflict_consensus_floor"])
        conflict_gate = 1.0 - (1.0 - conflict_floor) * coherent_conflict
        reliability_evidence *= float(conflict_gate)

    # Smoothly saturating the raw tally reduces excessive margin-dependent
    # confidence while preserving its sign and its dominance for large margins.
    tally_evidence = 3.8 * np.tanh(tally_margin / 3.8)
    total_evidence = tally_evidence + reliability_evidence

    # A reliability-based construal that reverses a weak nonzero tally is
    # treated as a categorical resolution. Exact-tally reliability choices are
    # left graded, avoiding excessive amplification in balanced coalitions.
    reliability_controls = False
    if abs(total_evidence) > 1e-12 and abs(reliability_evidence) > 1e-12:
        if abs(tally_margin) >= 1e-12:
            if np.sign(total_evidence) != np.sign(tally_margin):
                reliability_controls = True

    evidence_scale = float(parameters["evidence_scale"])
    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))
    decision_variable = (
        float(parameters["choice_sensitivity"]) * normalized_evidence
    )
    if reliability_controls:
        decision_variable *= float(parameters["construal_commitment"])

    # Positive evidence favors A. Construct symmetric logits to preserve exact
    # invariance under swapping the two displayed options.
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
- construal_quantile: [0, 1]
- orientation_strength: [1.9, 2.4]
- reliability_accumulation: [1.8, 3.2]
- exact_tie_salience_threshold: [0.45, 0.60]
- exact_tie_salience_slope: [12, 18]
- conflict_consensus_floor: [0.35, 0.60]
- evidence_scale: [2.3, 2.9]
- choice_sensitivity: [1.55, 2.15]
- construal_commitment: [2.1, 2.5]
- lapse_rate: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.4721 (var=0.0071)
- Experiment 2: real=89.1422 (var=7.5896) vs this=84.3674 (var=2.5041)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.3958 (var=0.0029)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.7837 (var=0.0060)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.0866 (var=0.0563)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8273 (var=0.0015)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.1972 (var=0.0033)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-3.4648 (var=0.0108)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.1697 (var=0.0029)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.7625 (var=0.0137)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.2006 (var=0.0082)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.2356 (var=0.0041)


---

### `pi_4` (overall score: 0.614)

**Description**
Compressed-Reliability Consensus theory holds that people represent every discriminating expert primarily as one categorical vote and encode communicated reliability through a separate, bounded comparison of the two endorsing groups. For each option, attention is biased moderately toward its more reliable endorsers, but their reliabilities are summarized as a softmax-weighted mean rather than accumulated expert by expert. Net evidence is therefore the full signed tally plus an identity-sensitive contrast between the two bounded group summaries. Agreement additionally amplifies confidence as the absolute tally margin grows, with amplification saturating after a decisive coalition has formed. This representation permits expert identity to determine equal-tally choices, preserves substantial influence for near-chance experts, and allows sufficiently numerous weak experts to override the strongest expert. Reliability sensitivity, consensus amplification, choice sensitivity, and lapse rate vary across subjects. Because the task provides no correctness feedback, choices are reversal-symmetric and independent of history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Compressed-Reliability Consensus model. History is intentionally ignored
    # because the task supplies no outcome feedback from which to recalibrate.
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

    # Every discriminating expert contributes a full baseline vote through the
    # signed tally. Reliability is encoded separately as a bounded summary of
    # each endorsing side, preventing identity increments from accumulating
    # without bound as a coalition grows.
    v = np.clip(validities, 0.5, 1.0)
    reliability_code = np.tanh(3.0 * (v - 0.5)) / np.tanh(1.5)

    difference = stimulus[0] - stimulus[1]
    a_endorsers = difference > 0.0
    b_endorsers = difference < 0.0
    tally_margin = float(np.sum(a_endorsers) - np.sum(b_endorsers))

    def side_reliability(mask):
        values = reliability_code[mask]
        if values.size == 0:
            return 0.0
        attention_logits = 4.0 * values
        attention_logits = attention_logits - np.max(attention_logits)
        attention = np.exp(attention_logits)
        attention /= attention.sum()
        return float(np.dot(attention, values))

    reliability_contrast = (
        side_reliability(a_endorsers) - side_reliability(b_endorsers)
    )
    alpha = float(parameters["reliability_sensitivity"])
    weighted_balance = tally_margin + alpha * reliability_contrast

    # Agreement is represented separately by the unweighted vote margin.
    # It changes confidence but does not erase which experts supplied the
    # votes. Leaving margin-one and tied tallies unamplified preserves
    # uncertainty in sparse conflicts and identity effects in equal tallies.
    # Amplification saturates at the level reached by a margin-five coalition.
    excess_consensus = max(abs(tally_margin) - 1.0, 0.0)
    consensus_gain = float(parameters["consensus_amplification"])
    confidence_multiplier = 1.0 + consensus_gain * min(excess_consensus ** 1.5, 8.0)
    evidence_for_a = weighted_balance * confidence_multiplier

    sensitivity = float(parameters["choice_sensitivity"])
    logits = np.array(
        [0.5 * sensitivity * evidence_for_a,
         -0.5 * sensitivity * evidence_for_a],
        dtype=np.float64,
    )
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * probabilities + lapse * 0.5
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= probabilities.sum()
    return probabilities

`policy(probs) -> int`:
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(probabilities.size, 1.0 / probabilities.size)
    else:
        probabilities /= total
    return int(np.random.choice(probabilities.size, p=probabilities))

`parameters`:
- reliability_sensitivity: [0.65, 1.05]
- consensus_amplification: [0.03, 0.07]
- choice_sensitivity: [0.13, 0.30]
- lapse_rate: [0.0, 0.08]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3392 (var=0.0040)
- Experiment 2: real=89.1422 (var=7.5896) vs this=100.3141 (var=2.8389)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.4698 (var=0.0021)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.7775 (var=0.0083)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=0.0071 (var=0.0478)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8037 (var=0.0033)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.5919 (var=0.0025)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=0.4221 (var=0.0108)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.5348 (var=0.0021)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.4383 (var=0.0129)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.5706 (var=0.0031)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.5364 (var=0.0028)


---

### `pi_5` (overall score: 0.570)

**Description**
Uncertain-Credibility Normalized Voting theory proposes that people do not treat instructed expert validities as precisely known reliabilities. In the absence of feedback, each communicated validity is shrunk toward a shared, moderately credible prototype. Consequently, every discriminating expert supplies one dominant categorical vote, while the incompletely trusted deviation of that expert's validity from the prototype supplies only a residual adjustment. Extreme communicated deviations are compressed smoothly because participants regard unusually high or low precision claims as especially uncertain, while still retaining limited distinctions among the most credible experts. These residual adjustments are accumulated cue by cue, preserving expert identity, but their total is divisively normalized and increasingly bounded once a clear tally coalition exists. Residual reliability is also contextually gated: it is strongly discounted in exact tally ties, where conflicting reliability claims provide no categorical direction, while otherwise retaining its ordinary residual weight without a special margin-one amplification. Before saturation, total evidence undergoes a moderately convex magnitude transformation, making small vote margins comparatively uncertain while preserving confidence for large coalitions. The resulting evidence is passed through a saturating response transformation, so additional agreement has diminishing effects once evidence is decisive. Between-person differences arise in reliability shrinkage, response sensitivity, and lapse rate. Because choices reveal no correctness information, the representation is not updated from choice history.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Uncertain-Credibility Normalized Voting. History is intentionally ignored:
    # without outcome feedback, previous choices cannot identify expert accuracy.
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

    # A difference of +1 is an endorsement of A and -1 an endorsement of B.
    # Nondiscriminating experts contribute neither a vote nor reliability evidence.
    signed_endorsements = np.sign(stimulus[0] - stimulus[1])
    discriminating = signed_endorsements != 0.0
    tally_evidence = float(np.sum(signed_endorsements))

    # Map validity onto above-chance diagnostic strength. The reference value
    # zero corresponds to a common validity of .75. Lower-tail deviations are
    # winsorized, while upper-tail deviations are smoothly compressed so that
    # very credible experts remain distinguishable without receiving full weight.
    v = np.clip(validities, 0.5, 1.0)
    communicated_strength = 2.0 * (v - 0.5)
    raw_deviation = communicated_strength - 0.5
    lower_bounded = np.maximum(raw_deviation, -0.25)
    upper_excess = np.maximum(lower_bounded - 0.25, 0.0)
    reliability_deviation = (
        np.minimum(lower_bounded, 0.25)
        + 0.13 * np.tanh(upper_excess / 0.10)
    )

    # Strong shrinkage leaves only a small fraction of communicated deviations.
    # Deviations are nevertheless summed expert by expert, unlike a summary of
    # the two endorsing groups. Divisive normalization bounds their joint impact.
    shrinkage = float(parameters["reliability_shrinkage"])
    retained_precision = 1.0 - shrinkage
    active_deviations = reliability_deviation[discriminating]
    active_signs = signed_endorsements[discriminating]

    if active_deviations.size == 0:
        reliability_adjustment = 0.0
    else:
        signed_sum = float(np.dot(active_signs, active_deviations))
        deviation_mass = float(np.sum(np.abs(active_deviations)))
        coalition_protection = 0.12 * max(abs(tally_evidence) - 1.0, 0.0) ** 2
        reliability_adjustment = (
            4.0 * retained_precision * signed_sum
            / (1.0 + retained_precision * deviation_mass + coalition_protection)
        )

        # Reliability is least trusted when the categorical tally is exactly
        # tied. Outside exact ties it receives its ordinary residual weight;
        # there is no separate amplification at a one-vote margin.
        absolute_tally = abs(tally_evidence)
        tie_gate = 1.0 - 0.94 * np.exp(-((absolute_tally / 0.35) ** 2))
        reliability_adjustment *= tie_gate

    total_evidence_for_a = tally_evidence + reliability_adjustment

    # Slightly stronger convexity compresses low evidence further. The revised
    # scale is calibrated to residual-shifted large-margin evidence rather than
    # to a raw tally of exactly five.
    response_curvature = 1.28
    saturation_scale = 4.50
    curved_evidence = np.sign(total_evidence_for_a) * (
        abs(total_evidence_for_a) / saturation_scale
    ) ** response_curvature
    normalized_evidence = np.tanh(curved_evidence)
    sensitivity = float(parameters["choice_sensitivity"])
    decision_variable = sensitivity * normalized_evidence

    logits = np.array(
        [0.5 * decision_variable, -0.5 * decision_variable],
        dtype=np.float64,
    )
    logits -= np.max(logits)
    p_core = np.exp(logits)
    p_core /= p_core.sum()

    lapse = float(parameters["lapse_rate"])
    probabilities = (1.0 - lapse) * p_core + lapse * 0.5
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
- reliability_shrinkage: [0.58, 0.84]
- choice_sensitivity: [2.20, 2.70]
- lapse_rate: [0.0, 0.12]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.4033 (var=0.0042) vs this=0.3108 (var=0.0052)
- Experiment 2: real=89.1422 (var=7.5896) vs this=59.0660 (var=1.4267)
- Experiment 3: real=0.4508 (var=0.0031) vs this=0.4573 (var=0.0018)
- Experiment 4: real=0.8213 (var=0.0169) vs this=0.8337 (var=0.0061)
- Experiment 5: real=-0.0382 (var=0.0352) vs this=-0.0126 (var=0.0452)
- Experiment 6: real=0.8462 (var=0.0122) vs this=0.8506 (var=0.0014)
- Experiment 7: real=0.1161 (var=0.0079) vs this=0.7494 (var=0.0030)
- Experiment 8: real=-3.5372 (var=0.0438) vs this=-0.0028 (var=0.0102)
- Experiment 9: real=0.3248 (var=0.0018) vs this=0.6632 (var=0.0032)
- Experiment 10: real=0.8617 (var=0.0104) vs this=0.5375 (var=0.0072)
- Experiment 11: real=0.1572 (var=0.0077) vs this=0.6192 (var=0.0027)
- Experiment 12: real=0.1261 (var=0.0089) vs this=0.4892 (var=0.0026)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1379 -> ACCEPTED
- iter 2: loss=0.1534 -> REJECTED
- iter 3: loss=0.0897 -> ACCEPTED
- iter 4: loss=0.0878 -> ACCEPTED
- iter 5: loss=0.0884 -> REJECTED
- iter 6: loss=0.0824 -> ACCEPTED
- iter 7: loss=0.0748 -> ACCEPTED
- iter 8: loss=0.0828 -> REJECTED
- iter 9: loss=0.0765 -> REJECTED
Running-best (last ACCEPTED) base: iter 7 at loss=0.0748 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
  - iter 1: 0.2450 (var=0.0029) (Δ vs real -0.1583)
  - iter 2: 0.2958 (var=0.0043) (Δ vs real -0.1075)
  - iter 3: 0.3371 (var=0.0045) (Δ vs real -0.0662)
  - iter 4: 0.3504 (var=0.0057) (Δ vs real -0.0529)
  - iter 5: 0.3454 (var=0.0055) (Δ vs real -0.0579)
  - iter 6: 0.3483 (var=0.0046) (Δ vs real -0.0550)
  - iter 7: 0.3400 (var=0.0058) (Δ vs real -0.0633)
  - iter 8: 0.3463 (var=0.0047) (Δ vs real -0.0571)
  - iter 9 (most recent): 0.3758 (var=0.0048) (Δ vs real -0.0275)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8450 (var=0.0143)
- pi_2: 0.1350 (var=0.0104)
- pi_3: 0.3775 (var=0.0056)
- pi_4: 0.3392 (var=0.0040)
- pi_5: 0.3108 (var=0.0052)
- pi_6: 0.4721 (var=0.0071)
- pi_7: 0.0896 (var=0.0023)

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
  - iter 1: 62.5625 (var=2.3906) (Δ vs real -26.5797)
  - iter 2: 134.7762 (var=4.5803) (Δ vs real +45.6340)
  - iter 3: 56.9131 (var=1.3751) (Δ vs real -32.2292)
  - iter 4: 58.9240 (var=0.8961) (Δ vs real -30.2182)
  - iter 5: 64.6543 (var=1.4107) (Δ vs real -24.4879)
  - iter 6: 74.6422 (var=2.0366) (Δ vs real -14.5000)
  - iter 7: 76.1205 (var=1.9792) (Δ vs real -13.0218)
  - iter 8: 57.7588 (var=1.9464) (Δ vs real -31.3834)
  - iter 9 (most recent): 83.4290 (var=2.0613) (Δ vs real -5.7132)
**Other theories' values on this metric (for reference):**
- pi_2: 1.9009 (var=0.0916)
- pi_1: 0.5503 (var=0.0488)
- pi_3: 49.6494 (var=1.7369)
- pi_4: 100.3141 (var=2.8389)
- pi_5: 59.0660 (var=1.4267)
- pi_6: 84.3674 (var=2.5041)
- pi_7: 83.3362 (var=1.6438)

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
  - iter 1: 0.3467 (var=0.0021) (Δ vs real -0.1042)
  - iter 2: 0.3623 (var=0.0025) (Δ vs real -0.0885)
  - iter 3: 0.3927 (var=0.0037) (Δ vs real -0.0581)
  - iter 4: 0.3825 (var=0.0022) (Δ vs real -0.0683)
  - iter 5: 0.3900 (var=0.0014) (Δ vs real -0.0608)
  - iter 6: 0.3879 (var=0.0031) (Δ vs real -0.0629)
  - iter 7: 0.4060 (var=0.0044) (Δ vs real -0.0448)
  - iter 8: 0.4075 (var=0.0025) (Δ vs real -0.0433)
  - iter 9 (most recent): 0.4892 (var=0.0034) (Δ vs real +0.0383)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5394 (var=0.0025)
- pi_2: 0.1190 (var=0.0098)
- pi_1: 0.7531 (var=0.0085)
- pi_4: 0.4698 (var=0.0021)
- pi_5: 0.4573 (var=0.0018)
- pi_6: 0.3958 (var=0.0029)
- pi_7: 0.1542 (var=0.0036)

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
  - iter 1: 0.8475 (var=0.0054) (Δ vs real +0.0262)
  - iter 2: 0.8213 (var=0.0063) (Δ vs real +0.0000)
  - iter 3: 0.8431 (var=0.0027) (Δ vs real +0.0219)
  - iter 4: 0.8369 (var=0.0038) (Δ vs real +0.0156)
  - iter 5: 0.8287 (var=0.0068) (Δ vs real +0.0075)
  - iter 6: 0.8462 (var=0.0064) (Δ vs real +0.0250)
  - iter 7: 0.8531 (var=0.0060) (Δ vs real +0.0319)
  - iter 8: 0.8512 (var=0.0049) (Δ vs real +0.0300)
  - iter 9 (most recent): 0.8594 (var=0.0049) (Δ vs real +0.0381)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8606 (var=0.0060)
- pi_3: 0.3869 (var=0.0089)
- pi_1: 0.1325 (var=0.0094)
- pi_4: 0.7775 (var=0.0083)
- pi_5: 0.8337 (var=0.0061)
- pi_6: 0.7837 (var=0.0060)
- pi_7: 0.9413 (var=0.0016)

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
  - iter 1: -0.6893 (var=0.0459) (Δ vs real -0.6512)
  - iter 2: -0.6038 (var=0.0941) (Δ vs real -0.5657)
  - iter 3: -0.0890 (var=0.0375) (Δ vs real -0.0508)
  - iter 4: -0.1201 (var=0.0238) (Δ vs real -0.0819)
  - iter 5: -0.1988 (var=0.0558) (Δ vs real -0.1606)
  - iter 6: -0.1558 (var=0.0484) (Δ vs real -0.1177)
  - iter 7: -0.0852 (var=0.0386) (Δ vs real -0.0470)
  - iter 8: -0.0203 (var=0.0549) (Δ vs real +0.0178)
  - iter 9 (most recent): 0.2146 (var=0.0488) (Δ vs real +0.2527)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5887 (var=0.0712)
- pi_4: 0.0071 (var=0.0478)
- pi_1: 0.5484 (var=0.0538)
- pi_2: -0.0256 (var=0.0579)
- pi_5: -0.0126 (var=0.0452)
- pi_6: -0.0866 (var=0.0563)
- pi_7: -0.5556 (var=0.0918)

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
  - iter 1: 0.8279 (var=0.0019) (Δ vs real -0.0183)
  - iter 2: 0.8140 (var=0.0021) (Δ vs real -0.0323)
  - iter 3: 0.8323 (var=0.0020) (Δ vs real -0.0140)
  - iter 4: 0.8417 (var=0.0022) (Δ vs real -0.0046)
  - iter 5: 0.8329 (var=0.0032) (Δ vs real -0.0133)
  - iter 6: 0.8638 (var=0.0018) (Δ vs real +0.0175)
  - iter 7: 0.8746 (var=0.0029) (Δ vs real +0.0283)
  - iter 8: 0.8377 (var=0.0028) (Δ vs real -0.0085)
  - iter 9 (most recent): 0.8629 (var=0.0018) (Δ vs real +0.0167)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8037 (var=0.0033)
- pi_3: 0.5665 (var=0.0017)
- pi_1: 0.8565 (var=0.0102)
- pi_2: 0.8612 (var=0.0065)
- pi_5: 0.8506 (var=0.0014)
- pi_6: 0.8273 (var=0.0015)
- pi_7: 0.8325 (var=0.0029)

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
  - iter 1: 0.1955 (var=0.0042) (Δ vs real +0.0794)
  - iter 2: 0.2509 (var=0.0098) (Δ vs real +0.1348)
  - iter 3: 0.2263 (var=0.0032) (Δ vs real +0.1102)
  - iter 4: 0.2258 (var=0.0055) (Δ vs real +0.1097)
  - iter 5: 0.2099 (var=0.0058) (Δ vs real +0.0938)
  - iter 6: 0.2208 (var=0.0057) (Δ vs real +0.1046)
  - iter 7: 0.2146 (var=0.0061) (Δ vs real +0.0985)
  - iter 8: 0.2150 (var=0.0042) (Δ vs real +0.0989)
  - iter 9 (most recent): 0.2498 (var=0.0069) (Δ vs real +0.1337)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7494 (var=0.0030)
- pi_4: 0.5919 (var=0.0025)
- pi_1: 0.8622 (var=0.0079)
- pi_2: 0.8381 (var=0.0106)
- pi_3: 0.9864 (var=0.0001)
- pi_6: 0.1972 (var=0.0033)
- pi_7: 0.2755 (var=0.0544)

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
  - iter 1: -4.5609 (var=0.0087) (Δ vs real -1.0237)
  - iter 2: -3.5294 (var=0.0204) (Δ vs real +0.0078)
  - iter 3: -4.4411 (var=0.0084) (Δ vs real -0.9039)
  - iter 4: -4.5082 (var=0.0162) (Δ vs real -0.9710)
  - iter 5: -4.4412 (var=0.0185) (Δ vs real -0.9040)
  - iter 6: -4.6047 (var=0.0171) (Δ vs real -1.0675)
  - iter 7: -4.6129 (var=0.0115) (Δ vs real -1.0757)
  - iter 8: -4.6111 (var=0.0138) (Δ vs real -1.0739)
  - iter 9 (most recent): -4.7130 (var=0.0110) (Δ vs real -1.1758)
**Other theories' values on this metric (for reference):**
- pi_4: 0.4221 (var=0.0108)
- pi_5: -0.0028 (var=0.0102)
- pi_1: 4.5627 (var=0.0649)
- pi_2: 0.0176 (var=0.0099)
- pi_3: 3.9747 (var=0.0054)
- pi_6: -3.4648 (var=0.0108)
- pi_7: -1.4013 (var=0.0925)

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
  - iter 1: 0.2965 (var=0.0029) (Δ vs real -0.0283)
  - iter 2: 0.2591 (var=0.0061) (Δ vs real -0.0657)
  - iter 3: 0.3005 (var=0.0038) (Δ vs real -0.0243)
  - iter 4: 0.2760 (var=0.0046) (Δ vs real -0.0489)
  - iter 5: 0.2741 (var=0.0034) (Δ vs real -0.0507)
  - iter 6: 0.2712 (var=0.0033) (Δ vs real -0.0536)
  - iter 7: 0.2702 (var=0.0030) (Δ vs real -0.0546)
  - iter 8: 0.2732 (var=0.0044) (Δ vs real -0.0517)
  - iter 9 (most recent): 0.2876 (var=0.0040) (Δ vs real -0.0373)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6632 (var=0.0032)
- pi_6: 0.1697 (var=0.0029)
- pi_1: 0.7324 (var=0.0054)
- pi_2: 0.5977 (var=0.0022)
- pi_3: 0.9801 (var=0.0004)
- pi_4: 0.5348 (var=0.0021)
- pi_7: 0.3215 (var=0.0354)

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
  - iter 1: 0.8467 (var=0.0067) (Δ vs real -0.0150)
  - iter 2: 0.8158 (var=0.0074) (Δ vs real -0.0458)
  - iter 3: 0.8283 (var=0.0071) (Δ vs real -0.0333)
  - iter 4: 0.8467 (var=0.0052) (Δ vs real -0.0150)
  - iter 5: 0.8458 (var=0.0076) (Δ vs real -0.0158)
  - iter 6: 0.8267 (var=0.0083) (Δ vs real -0.0350)
  - iter 7: 0.8250 (var=0.0114) (Δ vs real -0.0367)
  - iter 8: 0.8342 (var=0.0057) (Δ vs real -0.0275)
  - iter 9 (most recent): 0.8508 (var=0.0056) (Δ vs real -0.0108)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7625 (var=0.0137)
- pi_5: 0.5375 (var=0.0072)
- pi_1: 0.1683 (var=0.0142)
- pi_2: 0.4767 (var=0.0123)
- pi_3: 0.0008 (var=0.0000)
- pi_4: 0.4383 (var=0.0129)
- pi_7: 0.8467 (var=0.0208)

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
  - iter 1: 0.2303 (var=0.0044) (Δ vs real +0.0731)
  - iter 2: 0.2994 (var=0.0131) (Δ vs real +0.1422)
  - iter 3: 0.2242 (var=0.0050) (Δ vs real +0.0669)
  - iter 4: 0.2100 (var=0.0042) (Δ vs real +0.0528)
  - iter 5: 0.2347 (var=0.0048) (Δ vs real +0.0775)
  - iter 6: 0.2131 (var=0.0049) (Δ vs real +0.0558)
  - iter 7: 0.1967 (var=0.0045) (Δ vs real +0.0394)
  - iter 8: 0.2053 (var=0.0053) (Δ vs real +0.0481)
  - iter 9 (most recent): 0.2031 (var=0.0051) (Δ vs real +0.0458)
**Other theories' values on this metric (for reference):**
- pi_7: 0.9125 (var=0.0016)
- pi_6: 0.2006 (var=0.0082)
- pi_1: 0.8514 (var=0.0098)
- pi_2: 0.8394 (var=0.0083)
- pi_3: 0.7772 (var=0.0054)
- pi_4: 0.5706 (var=0.0031)
- pi_5: 0.6192 (var=0.0027)

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
  - iter 1: 0.1675 (var=0.0030) (Δ vs real +0.0414)
  - iter 2: 0.2369 (var=0.0047) (Δ vs real +0.1108)
  - iter 3: 0.1761 (var=0.0018) (Δ vs real +0.0500)
  - iter 4: 0.1872 (var=0.0035) (Δ vs real +0.0611)
  - iter 5: 0.1658 (var=0.0032) (Δ vs real +0.0397)
  - iter 6: 0.1697 (var=0.0047) (Δ vs real +0.0436)
  - iter 7: 0.1603 (var=0.0033) (Δ vs real +0.0342)
  - iter 8: 0.1708 (var=0.0028) (Δ vs real +0.0447)
  - iter 9 (most recent): 0.1547 (var=0.0025) (Δ vs real +0.0286)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2356 (var=0.0041)
- pi_7: 0.8364 (var=0.0032)
- pi_1: 0.8367 (var=0.0093)
- pi_2: 0.4964 (var=0.0036)
- pi_3: 0.7497 (var=0.0047)
- pi_4: 0.5364 (var=0.0028)
- pi_5: 0.4892 (var=0.0026)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth and within-side compositional heterogeneity. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.

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
        contextual_contrast = (
            signed_claim_contrast * focus
            / (breadth_normalizer * heterogeneity_normalizer + 1e-12)
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
- composition_dilution: [0.6, 1.8]
- concentration_floor: [0.30, 0.55]
- margin_protection_scale: [1.55, 2.25]
- margin_protection_power: [2.2, 3.4]
- evidence_scale: [2.2, 3.0]
- choice_sensitivity: [1.45, 2.65]
- margin_temperature_gain: [0.0, 0.32]
- lapse_rate: [0.0, 0.10]
- validities: validities

`rationale`: This is a targeted edit to the accepted iteration-6 model. First, the existing opposing-claim focus now attenuates the endpoint cue's entire contribution—ordinary distrust plus endpoint bonus—rather than only its bonus. A neutral 50% opponent still produces no attenuation, preserving Experiment 10, while substantive 60%-80% opponents suppress the endpoint cue more strongly in Experiment 8. Second, upper-range ordinary compression increases from 0.42 to 0.55. This selectively weakens 95% claims in Experiment 5 without reducing distrust of moderate 60%-70% claims; the endpoint coefficient is raised only slightly so literal 100% claims remain strong. Third, a fixed smooth boost is applied to moderate-claim accumulation only for intermediate coalition breadth. It leaves sparse near-endpoint conflicts and maximally dense diffuse conflicts largely unchanged while targeting Experiments 7, 11, and 12 without repeating the rejected global breadth-exponent edit. Finally, the accepted margin-temperature mechanism is retained but changed from monotonic growth to a smooth intermediate-margin profile, and its heterogeneous range is widened. This should increase Experiment 2's margin-dependent mean and variance while limiting additional sharpening of the fixed large-margin conditions in Experiments 4 and 6.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Contextual Anti-Reliability Arbitration family and captures several important qualitative patterns: endpoint reversal in Experiment 10, large-margin tally dominance in Experiments 4 and 6, anti-validity behavior in Experiments 7, 9, 11, and 12, and option-swap symmetry. However, the fit is not yet experiment-invariant. The decisive failure is Experiment 5, where the predicted contrast is far too negative (-0.689 versus -0.038), indicating excessive anti-reliability control in sparse, equal-tally coalitions. The same sparse-conflict overreaction appears in Experiment 1 (TTB choice 0.245 versus 0.403), Experiment 3 (minority choice 0.347 versus 0.451), and Experiment 8 (alignment -4.561 versus -3.537). Conversely, anti-validity is somewhat too weak in broader coalitions: conventional/high-validity choices remain too high in Experiments 7 (0.196 versus 0.116), 11 (0.230 versus 0.157), and 12 (0.168 versus 0.126). Thus the current breadth normalization has the wrong quantitative gradient: it suppresses anti-reliability too strongly as coalitions become broad while leaving it too strong in sparse non-endpoint conflicts. Experiment 2 also shows insufficient margin-dependent tally evidence (log Bayes factor 62.6 versus 89.1) and substantially too little between-subject variance. Variance is generally compressed across experiments, consistent with the nested tanh transformations saturating individual differences.
Rationale: Retain the prescribed continuous pragmatic-distrust architecture, but rebalance sparse versus dense arbitration. Lower ordinary non-endpoint anti-reliability strength or accumulation while substantially weakening the active-count breadth penalty (for example, lower the coalition-dilution exponent). This paired change can reduce excessive reversal in sparse conditions—especially Experiments 1, 3, 5, and 8—without weakening, and potentially strengthening, the broad-coalition reversals needed in Experiments 7, 11, and 12. Keep endpoint distrust separately strong, or raise its coefficient to compensate for the lower ordinary strength, so that the already accurate Experiment 10 result is preserved. Experiment 5 should be the primary calibration target: the present model effectively treats its sparse equal-tally contrasts as near-categorical reversals, contrary to the required continuous arbitration. In addition, make raw tally evidence somewhat less saturating across margins so Experiment 2 exhibits stronger margin-dependent tally support, while retuning overall sensitivity to avoid overshooting Experiments 4 and 6. Finally, reduce saturation of the anti-evidence transform or widen effective subject-level strength and sensitivity ranges; nominally broad parameter ranges currently yield much less variance than observed, particularly in Experiment 2. These are in-family normalization, scaling, and heterogeneity adjustments rather than a change of mechanism.

**Outcome of this advice:** iter 1 candidate loss=0.1379 -> iter 2 candidate loss=0.1534 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family, but it is not experiment-invariant and was rejected by the accept gate (loss 0.1534 versus the accepted base's 0.1379). The recalibration improved Experiments 1, 3, 4, 5, and especially 8, but most gains were too small or came at larger costs elsewhere. Experiment 5 remains the dominant failure: -0.604 versus -0.038 indicates near-categorical rejection of the 95%-endorsed option where humans are approximately balanced. The current endpoint implementation likely contributes to this because a 95% validity maps to above_chance=0.90 and still receives both nearly maximal ordinary distrust and a nontrivial endpoint bonus. Meanwhile, the broad-coalition fits deteriorated substantially in Experiments 7 (0.251 versus 0.116), 11 (0.299 versus 0.157), and 12 (0.237 versus 0.126), showing that the reduction in ordinary anti-reliability outweighed the weaker breadth dilution. Experiment 2 also overshot sharply from insufficient to excessive margin-dependent evidence (134.8 versus 89.1), while Experiments 6, 9, and 10 moved away from the data. Thus the previous recommendation to jointly lower ordinary distrust, weaken breadth normalization substantially, and increase tally-margin separation was not successful as implemented.
Rationale: Build from the unchanged accepted iteration-1 base and try a more selective in-family recalibration rather than repeating the rejected global strength/breadth changes. First, separate ordinary distrust more cleanly from endpoint distrust: prevent 95% claims from receiving almost maximal ordinary evidence plus appreciable endpoint amplification, while retaining a steep continuous endpoint response very close to 100%. For example, cap or compress the ordinary-claim component near its upper range and move the endpoint sigmoid threshold closer to above_chance 0.97-0.99 with a steeper slope. This directly targets Experiment 5 while preserving the genuinely endpoint-specific reversal in Experiment 10. Second, retain the accepted base's broad-coalition normalization or alter it only modestly; the large reduction in breadth dilution was rejected and worsened Experiments 7, 11, and 12 once paired with weaker ordinary strength. If sparse-versus-broad balance still needs adjustment, use a continuous side-composition normalization that discounts contrasts dominated by one near-endpoint expert, rather than lowering ordinary anti-reliability globally. Third, do not retain the rejected increase in tally dynamic range: use the accepted tally saturation or an intermediate value, because the current change drove Experiment 2 far past its target and slightly harmed Experiments 6 and 9. Preserve the continuous heterogeneity, symmetry, and history independence of the prescribed mechanism.

**Outcome of this advice:** iter 2 candidate loss=0.1534 -> iter 3 candidate loss=0.0897 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The accepted candidate is a substantial improvement and remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family. The selective separation of ordinary from endpoint distrust worked: Experiment 5 improved dramatically from -0.689 to -0.089 against -0.038, while Experiments 1 and 3 also moved closer, and tally dominance remains accurate in Experiments 4 and 6. Experiments 9 and 10 are reasonably captured, and the model preserves symmetry and history independence. Nevertheless, important experiment-invariance failures remain. Anti-validity is still too weak in broad moderate-validity conflicts: high-validity choices are too frequent in Experiments 7 (0.226 versus 0.116), 11 (0.224 versus 0.157), and 12 (0.176 versus 0.126). Conversely, Experiment 8 is too strongly anti-validity (-4.441 versus -3.537), even though the concentrated endpoint reversal in Experiment 10 is slightly too weak (0.828 versus 0.862). This indicates that endpoint distrust is not sufficiently regulated by contrast concentration: it overacts in the more compositionally distributed Experiment 8 while underacting in the sparse 100%-versus-50% condition. Experiment 2 also remains materially low (56.9 versus 89.1), with much too little between-subject variance (1.38 versus 7.59), indicating insufficient margin-dependent separation and compressed heterogeneity. The current nested tanh transformations likely contribute to that compression.
Rationale: Retain the successful ordinary/endpoint separation introduced in this accepted iteration; the prior selective endpoint advice clearly helped and should not be undone. Make the next changes small and targeted. First, apply a stronger concentration-sensitive multiplier specifically to the endpoint component, leaving concentrated 100%-versus-50% contrasts nearly intact or slightly stronger while diluting endpoint evidence when it is embedded among several heterogeneous claims. This should move Experiment 8 upward without sacrificing Experiment 10 or reintroducing the Experiment 5 failure. Second, modestly reduce breadth or composition dilution for moderate 60%-70% claim contrasts so broad coalitions exert somewhat more anti-reliability influence in Experiments 7, 11, and 12. The previous large global reduction in breadth dilution was rejected, so this should be a smaller adjustment, preferably validity- and composition-sensitive rather than another global strength change. Third, use an intermediate tally dynamic range between the accepted saturation and the rejected iteration-2 setting: the former produces 56.9 in Experiment 2 and the latter produced 134.8, bracketing the observed 89.1. Do not repeat the full rejected increase. Finally, mildly reduce saturation in the final evidence mapping or widen effective choice-sensitivity heterogeneity to raise Experiment 2's subject variance, while monitoring the already reasonable means in Experiments 4 and 6. These are calibration changes within the prescribed arbitration mechanism, not a change of theory family.

**Outcome of this advice:** iter 3 candidate loss=0.0897 -> iter 4 candidate loss=0.0878 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family, and the current iteration was accepted with a small loss improvement (0.0897 to 0.0878). It captures several central qualitative regularities: strong isolated endpoint distrust in Experiment 10, tally dominance in Experiments 4 and 6, and substantial anti-validity behavior in Experiments 7, 9, 11, and 12. The accepted endpoint/ordinary separation also continues to keep Experiment 5 far closer to humans than the early candidates. However, the fit is not yet experiment-invariant. Experiment 2 remains substantially too weak in margin-dependent tally evidence (58.9 versus 89.1), and its between-subject variance is extremely compressed (0.90 versus 7.59). Broad moderate-validity coalitions still exert too little anti-reliability influence in Experiments 7 (0.226 versus 0.116), 11 (0.210 versus 0.157), and 12 (0.187 versus 0.126). The new endpoint embedding adjustment did not solve Experiment 8: alignment became slightly more negative, reaching -4.508 versus -3.537, while Experiment 10 improved. This means the model now handles the isolated endpoint contrast well but still overgeneralizes endpoint distrust to contrasts between the endpoint expert and substantive 60%-80% opponents. Smaller residual errors also remain in Experiments 1 and 3, while Experiment 5 has slipped slightly away from its near-zero target. Across several experiments, simulated variances remain much lower than observed, indicating that widening choice sensitivity did not create the required heterogeneity because the nested saturation stages still map subjects to similar behavior.
Rationale: Continue from this accepted candidate, but use different targeted knobs rather than merely repeating the latest changes. First, revise the endpoint-focus calculation so endpoint distrust is preserved when the endpoint expert opposes a genuinely neutral 50% cue, as in Experiment 10, but is attenuated more strongly when the opposing active cue itself has substantive ordinary claim strength, as in Experiment 8. A smooth multiplier based on opposing non-endpoint claim mass—not just total surrounding mass—would make this distinction while preserving symmetry and continuity. The present global endpoint-focus ratio is evidently insufficient, since Experiment 8 worsened despite the intended dilution. Second, strengthen endpoint-free moderate anti-reliability in broad homogeneous coalitions using a validity-sensitive breadth exponent or normalization: apply weaker breadth dilution specifically when the active claims are predominantly moderate 60%-70% claims, while retaining stronger dilution for endpoint-containing or highly heterogeneous compositions. This targets Experiments 7, 11, and 12 without reviving the sparse 95% failure in Experiment 5; the earlier large global breadth change was rejected, so avoid another indiscriminate reduction. Third, make tally curvature itself continuously heterogeneous across subjects rather than further widening choice sensitivity. Introduce a narrow subject-level range around the fixed tally saturation value of 4.5, with a mean slightly above 4.5, to increase both the average margin separation and its variance in Experiment 2. The latest choice-sensitivity widening actually reduced Experiment 2 variance, so that recommendation should not be repeated. Monitor Experiments 4 and 6 to ensure that their already accurate large-margin tally dominance is preserved. These are minor normalization and parameterization changes within the prescribed arbitration mechanism.

**Outcome of this advice:** iter 4 candidate loss=0.0878 -> iter 5 candidate loss=0.0884 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Contextual Anti-Reliability Arbitration family, but the current revision was rejected by the accept gate: loss increased from 0.0878 to 0.0884. Its targeted changes produced only partial gains. Experiment 2 improved from 58.9 to 64.7 but remains far below the observed 89.1, and subject variance remains severely compressed at 1.41 versus 7.59. Experiments 7, 8, and 12 moved modestly toward the data, but Experiment 11 moved substantially away, from 0.210 to 0.235 versus 0.157. More importantly, Experiment 5 deteriorated from -0.120 to -0.199 versus -0.038, indicating renewed overreaction to the near-endpoint expert in balanced coalitions. Tally dominance also weakened in Experiment 6. The endpoint-focus modification therefore did not provide a clean distinction between isolated endpoint distrust and endpoint claims embedded in substantive coalitions, while the moderate-coalition breadth relief did not consistently strengthen the required broad anti-validity pattern. The narrow tally-saturation heterogeneity increased Experiment 2's mean and variance slightly, but not enough to justify its collateral losses. The accepted iteration-4 base remains qualitatively strong on isolated endpoint distrust in Experiment 10 and large-coalition tally dominance in Experiments 4 and 6, but still misses Experiments 2, 7, 8, and 11 and retains smaller errors in Experiments 1, 3, 5, 9, and 12.
Rationale: Build from the unchanged accepted iteration-4 base. Do not repeat the rejected opposing-claim-mass endpoint focus, moderate breadth-exponent relief, or variable tally-saturation change. For endpoint regulation, use a different continuous composition statistic: attenuate the endpoint bonus according to the strongest or smooth-maximum opposing non-endpoint claim, rather than summed opposing mass. This should leave a neutral 50% opponent effectively inert in Experiment 10, while strongly attenuating a 100% claim opposed by a 60% or 80% cue in Experiment 8. Keep the already successful steep separation between 95% ordinary distrust and genuinely endpoint-specific distrust so Experiment 5 does not regress. To strengthen broad moderate anti-validity without another breadth-normalization edit, modestly increase anti_accumulation while leaving anti_reliability_strength fixed. Because sparse concentrated contrasts are already closer to the tanh ceiling whereas diluted broad contrasts remain in its approximately linear region, this can selectively lower high-validity choice rates in Experiments 7, 11, and 12 with less impact on sparse Experiment 5. Use a small range change and verify that Experiment 1 does not fall further. Finally, restore fixed tally saturation and address Experiment 2 with a mild, continuous margin-dependent decision temperature or sensitivity multiplier, anchored at margin one and capped by moderate margins. Give its slope continuous subject heterogeneity. This targets both the mean margin dependence and between-subject variance in Experiment 2 more directly than altering tally saturation, while a low cap should preserve Experiments 4 and 6. These are in-family changes to endpoint composition, accumulation, and smooth arbitration rather than a new mechanism family.

**Outcome of this advice:** iter 5 candidate loss=0.0884 -> iter 6 candidate loss=0.0824 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The accepted revision improves aggregate loss from 0.0878 to 0.0824 and remains faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family. The most successful new component is the margin-dependent temperature: Experiment 2 rises from 58.9 to 74.6 and its variance from 0.90 to 2.04. The accumulation adjustment also improves Experiment 12 substantially, while Experiments 4 and 6 retain strong tally dominance. Nevertheless, the fit is not yet experiment-invariant. Broad moderate-validity anti-reliability remains too weak in Experiments 7 (0.221 versus 0.116), 11 (0.213 versus 0.157), and 12 (0.170 versus 0.126). At the same time, high and endpoint claims are over-reversed in Experiment 5 (-0.156 versus -0.038) and especially Experiment 8 (-4.605 versus -3.537), whereas isolated endpoint distrust in Experiment 10 is slightly too weak (0.827 versus 0.862). Thus the smooth-maximum endpoint edit did not achieve its intended contextual selectivity: it attenuates only the endpoint bonus, while the endpoint cue's ordinary anti-evidence remains active in substantive conflicts. Experiment 2 is still low by 14.5 units and has only about one quarter of the observed subject variance. Residual directional errors also remain in Experiments 1 and 3, where the model chooses the TTB/minority side too rarely, and Experiment 9, where high-validity choices are too rare.
Rationale: Continue from this accepted candidate and retain the changes that helped: the fixed tally saturation, continuous margin-temperature mechanism, steep ordinary/endpoint separation, and current arbitration architecture. For the principal endpoint problem, do not merely strengthen the current smooth-maximum focus. Instead, apply the opposing-substantive-claim attenuation to the endpoint cue's total anti-reliability contribution—ordinary plus endpoint bonus—while leaving it near one against a neutral 50% opponent. This should weaken excessive reversal in Experiment 8 while preserving or strengthening the isolated 100%-versus-50% reversal in Experiment 10. Separately increase upper-range compression around 95% claims, with the endpoint bonus still turning on steeply only near 100%, to move Experiment 5 toward zero without weakening moderate 60%-70% anti-validity. The accepted increase in anti_accumulation helped Experiment 12 and modestly helped Experiment 7, so a small further increase or slight range widening is defensible, but it should be confined to endpoint-free, moderate-claim compositions; a global increase risks worsening Experiments 1, 8, and 9. Do not repeat the previously rejected global breadth-exponent relief. Finally, retain the margin-temperature idea but widen its subject-level gain range modestly and use a smooth bounded shape concentrated at intermediate margins, rather than increasing its mean indiscriminately. This may raise Experiment 2's mean and variance while limiting further overprediction of tally dominance in Experiments 4 and 6. The next calibration should prioritize Experiments 7, 8, and 2 while treating Experiments 5 and 10 jointly as the diagnostic test of whether 95% compression and genuinely endpoint-specific distrust have been separated correctly.

**Outcome of this advice:** iter 6 candidate loss=0.0824 -> iter 7 candidate loss=0.0748 -> the gate marked it ACCEPTED.

### Iteration 7
Verdict: regenerate
Interpretation: The current candidate is faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family and is the best accepted iteration so far, reducing aggregate loss from 0.0824 to 0.0748. The latest selective changes clearly helped Experiment 5 (-0.085 versus -0.038), Experiment 3 (0.406 versus 0.451), Experiment 11 (0.197 versus 0.157), and Experiment 12 (0.160 versus 0.126). It also preserves the central successes: isolated endpoint distrust in Experiment 10 and large-coalition tally dominance in Experiments 4 and 6. However, it is not yet experiment-invariant. Experiment 8 remains the largest qualitative calibration failure: alignment is much too anti-validity (-4.613 versus -3.537), and the latest total-endpoint attenuation did not improve that metric. Broad moderate-validity reversal is still too weak in Experiment 7 (0.215 versus 0.116), while Experiments 1 and 9 err in the opposite direction, choosing the conventional high-validity side too rarely (0.340 versus 0.403 and 0.270 versus 0.325). This combination shows that a global increase in anti-reliability would be inappropriate; the missing selectivity concerns homogeneous weak-margin coalitions versus heterogeneous or endpoint-containing conflicts. Experiment 2 also remains low at 76.1 versus 89.1, with severely compressed subject variance of 1.98 versus 7.59. Finally, simulated variance is too low in Experiments 4, 6, 8, and 12, suggesting that the nested bounded mappings still suppress effective heterogeneity.
Rationale: Continue from this accepted candidate and retain the successful 95%-compression, symmetry, history independence, and intermediate-margin temperature architecture. Do not globally raise anti-reliability strength or broadly weaken coalition dilution, because that would aggravate Experiments 1, 8, and 9. Instead, strengthen the existing moderate-claim accumulation boost only when opposing coalitions are internally homogeneous, endpoint-free, and the tally margin is smoothly near zero or one. A continuous gate based on low within-side dispersion, low endpoint mass, and the existing margin-protection function should selectively lower conventional choices in Experiments 7, 11, and 12 while largely sparing the heterogeneous validity profiles in Experiments 1 and 9. For Experiments 8 and 10, the latest endpoint-total attenuation was accepted overall but failed locally: rather than repeating the same coefficient change indiscriminately, make endpoint attenuation steeper as a function of the smooth maximum substantive opposing claim, and jointly allow a modestly larger endpoint_distrust coefficient. That pairing should weaken the endpoint cue against 60%-80% opponents in Experiment 8 while preserving or restoring isolated 100%-versus-50% distrust in Experiment 10; keep the successful 95% compression unchanged. For Experiment 2, reshape rather than simply widen the current temperature gain: concentrate its peak more narrowly around tally margin three and let its peak height or width vary continuously across subjects. Experiment 2 contains useful margin-three contrasts, whereas Experiments 4 and 6 are dominated by margin-five conflicts and are already overpredicted, so faster decay beyond margin three can increase Bayes-factor separation and heterogeneity without further sharpening those large-margin experiments. Avoid another global choice-sensitivity expansion, which previously failed to generate sufficient variance through the saturated evidence map.

**Outcome of this advice:** iter 7 candidate loss=0.0748 -> iter 8 candidate loss=0.0828 -> the gate marked it REJECTED.

### Iteration 8
Verdict: regenerate
Interpretation: The candidate remains broadly faithful to the prescribed continuous Contextual Anti-Reliability Arbitration family, but it was rejected by the accept gate: aggregate loss worsened from the iteration-7 best of 0.0748 to 0.0828. The new margin-temperature shape is the clearest failure. Experiment 2 fell sharply from 76.12 to 57.76 versus 89.14, while its already deficient between-subject variance did not improve. The added broad-homogeneous moderate boost also failed its intended targets: Experiment 7 was unchanged at 0.215 versus 0.116, and Experiments 11 and 12 worsened to 0.205 and 0.171. Likewise, steeper endpoint attenuation did essentially nothing for Experiment 8, which remains much too anti-validity at -4.611 versus -3.537. Improvements in Experiments 5, 6, and 10 are real but insufficient to offset these losses. The remaining pattern is systematic: anti-reliability is too strong for heterogeneous substantive contrasts in Experiments 1, 8, and 9, yet too weak for coalitions composed of moderate claims in Experiments 7 and 11. This points to a miscalibrated validity-to-distrust curve rather than a need for another coalition gate or a stronger global anti-reliability coefficient. Variance also remains substantially compressed, especially in Experiments 2, 4, 6, 8, and 12.
Rationale: Build on the unchanged accepted iteration-7 base, but do not repeat the rejected broad-homogeneous multiplier, narrower Gaussian margin-temperature change, or further steepening of the current endpoint-focus coefficient. Instead, recalibrate the ordinary claim-strength mapping itself. Use a smooth response that reaches substantial anti-reliability strength already around 60%-70% validity but becomes flatter or more compressed across roughly 75%-95%, while preserving the separate steep endpoint component near 100%. Such a moderate plateau would strengthen accumulated distrust in the broad moderate coalitions of Experiments 7 and 11 (and preserve Experiment 12), while reducing the directional contrast between substantive 60%-80%-100% opponents that currently drives excessive reversal in Experiment 8. It should also weaken over-discrimination among heterogeneous validity profiles in Experiments 1 and 9 without globally lowering anti-reliability. Preserve the successful special compression around 95% so Experiment 5 does not regress, and retain endpoint strength against a genuinely neutral 50% opponent for Experiment 10. For Experiment 2, retain the accepted iteration-7 intermediate-margin temperature architecture rather than reshaping it again; the rejected Gaussian edit shows that narrowing or relocating its peak is counterproductive. If further calibration is attempted, vary the curvature or gain only mildly around the accepted form and avoid another large structural change. Finally, reduce one of the nested saturations—preferably the final evidence normalization—with compensating sensitivity rescaling, so continuous subject differences remain behaviorally expressed rather than collapsing to similar probabilities. This is an in-family recalibration of the distrust response and decision mapping, not a change of mechanism family.

**Outcome of this advice:** iter 8 candidate loss=0.0828 -> iter 9 candidate loss=0.0765 -> the gate marked it REJECTED.

### Iteration 9 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Contextual Anti-Reliability Arbitration family, but it was rejected by the accept gate: loss increased from the iteration-7 best of 0.0748 to 0.0765. The plateau mapping delivered meaningful gains in Experiments 1, 2, 10, and 12, but its cross-experiment tradeoff is unfavorable. Most importantly, Experiment 5 reverses from a close fit at iteration 7 (-0.085 versus -0.038) to a large error in the opposite direction (0.215), showing that the new validity mapping destroys the successful separation between 95% compression and endpoint distrust. Experiment 7 also worsens from 0.215 to 0.250 versus 0.116, so the plateau did not strengthen broad moderate-claim reversal as intended. Experiment 8 becomes still more anti-validity (-4.713 versus -3.537), and Experiment 3 overshoots from too little minority choice to too much. Thus the rejected plateau advice should not be repeated. The persistent pattern in the accepted iteration-7 base is more specifically compositional: anti-reliability is excessive for endpoint-containing or internally heterogeneous profiles in Experiments 5 and 8 and for heterogeneous profiles in Experiments 1 and 9, but insufficient for broad, comparatively coherent moderate-claim coalitions in Experiments 7 and 11. The current within-side dispersion term cannot detect sparse heterogeneous conflicts such as Experiment 8 because each side often contains only one active claim, making both within-side dispersions exactly zero. Simulated between-subject variance also remains markedly compressed, especially in Experiments 2, 4, 6, 8, and 12.
Rationale: Build from the unchanged accepted iteration-7 base and restore its ordinary validity mapping and endpoint range; do not retain the rejected rapid plateau or its endpoint compensation. Instead, recalibrate the existing composition-sensitive arbitration. Add a smooth endpoint-mixture or across-active compositional dilution term that can detect a near-endpoint claim opposed by a substantive nonendpoint claim even when each coalition is a singleton. Apply this dilution to the total contextual contrast rather than again steepening the endpoint-focus coefficient, which has repeatedly failed to move Experiment 8. It should be near zero for the isolated 100%-versus-50% contrast in Experiment 10, but larger for the 100%-versus-60%/80% conflicts in Experiment 8 and the near-endpoint balanced conflicts in Experiment 5. Separately, modestly increase the influence of the existing concentration floor while strengthening within-side composition dilution. That paired adjustment can give diffuse but internally coherent moderate coalitions more anti-reliability control in Experiments 7 and 11, while offsetting that increase in heterogeneous profiles such as Experiments 1 and 9. This uses existing continuous concentration and composition mechanisms rather than another rejected broad-coalition boost or a global strength increase. Preserve the accepted intermediate-margin temperature form; the rejected reshaping harmed Experiment 2. If heterogeneity is adjusted, widen subject-level ranges for concentration and composition sensitivity around their current means rather than widening global choice sensitivity, so contextual differences remain expressed without further sharpening the already overpredicted tally dominance in Experiments 4 and 6.

**Outcome of this advice:** iter 9 candidate loss=0.0765 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Contextual Anti-Reliability Arbitration theory proposes that communicated validity numbers are interpreted partly as pragmatic claims whose assertiveness can undermine their credibility. Every above-chance claim therefore generates continuous evidence against the option endorsed by that expert, rather than being categorically reversed by a special subgroup. The magnitude of this anti-reliability response varies continuously across people. Near-perfect claims additionally evoke endpoint distrust because unusually strong assurances are perceived as especially suspicious. When an endpoint cue faces a substantive opposing claim, both its ordinary and endpoint-specific distrust contributions are attenuated; against a genuinely neutral cue, both remain intact. Claims around 95% are more strongly compressed in the ordinary channel and do not receive the endpoint bonus, separating near-endpoint skepticism from distrust of literal certainty. Anti-reliability evidence is accumulated separately from the raw vote tally and divisively normalized by coalition breadth, within-side compositional heterogeneity, and sparse across-side claim mismatch. The latter detects heterogeneous singleton conflicts that within-side dispersion cannot detect, while exempting an isolated certainty claim opposed by a neutral cue. A small smooth boost preserves moderate-claim distrust in broad but not maximally dense coalitions. Its influence is strongest when the validity contrast is concentrated and the tally is tied or weak, while it decays smoothly as the absolute tally margin grows. Raw tally evidence additionally receives a continuously heterogeneous, intermediate-margin decision-temperature adjustment. The mechanism remains continuous, option-swap symmetric, and history independent because choices are followed by no accuracy feedback.",
  "predict": "def predict(parameters, state, history):\n    # Contextual Anti-Reliability Arbitration. History is intentionally ignored:\n    # without outcome feedback, prior choices cannot reveal expert accuracy.\n    stimulus = np.asarray(state, dtype=np.float64)\n    if stimulus.ndim != 2 or stimulus.shape[0] != 2:\n        raise ValueError(\n            f\"Expected state with shape (2, n_features); got {stimulus.shape}.\"\n        )\n\n    n_features = stimulus.shape[1]\n    validities = np.asarray(parameters[\"validities\"], dtype=np.float64)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} does not match \"\n            f\"n_features {n_features}.\"\n        )\n\n    # Positive endorsements favor A and negative endorsements favor B.\n    signed_votes = np.sign(stimulus[0] - stimulus[1])\n    active = signed_votes != 0.0\n    tally_margin = float(np.sum(signed_votes))\n    active_count = int(np.count_nonzero(active))\n\n    # Encode the assertiveness of every communicated above-chance claim.\n    # Moderate claims already carry pragmatic suspicion. The ordinary channel\n    # is strongly compressed near 95%, preventing near-endpoint claims from\n    # behaving like literal-certainty claims.\n    v = np.clip(validities, 0.5, 1.0)\n    above_chance = np.clip(2.0 * (v - 0.5), 0.0, 1.0)\n    ordinary_base = np.power(above_chance, 0.82)\n\n    def logistic(x):\n        if x >= 0.0:\n            return 1.0 / (1.0 + np.exp(-x))\n        ex = np.exp(x)\n        return ex / (1.0 + ex)\n\n    upper_compression = np.array(\n        [logistic(50.0 * (float(x) - 0.86)) for x in above_chance],\n        dtype=np.float64,\n    )\n    ordinary_claim = ordinary_base * (1.0 - 0.55 * upper_compression)\n\n    # Endpoint distrust is reserved for claims extremely close to certainty.\n    endpoint_low = logistic(100.0 * (0.0 - 0.98))\n    endpoint_high = logistic(100.0 * (1.0 - 0.98))\n    endpoint_curve = np.array(\n        [\n            (logistic(100.0 * (float(x) - 0.98)) - endpoint_low)\n            / (endpoint_high - endpoint_low)\n            for x in above_chance\n        ],\n        dtype=np.float64,\n    )\n    endpoint_curve = np.clip(endpoint_curve, 0.0, 1.0)\n\n    if active_count == 0:\n        anti_reliability = 0.0\n    else:\n        active_signs = signed_votes[active]\n        active_ordinary = ordinary_claim[active]\n        active_endpoint = endpoint_curve[active]\n        active_above_chance = above_chance[active]\n\n        # Endpoint distrust is preserved against a neutral 50% opponent, but\n        # attenuated by the smooth maximum ordinary strength on the opposing\n        # side. The attenuation applies to the endpoint cue's total distrust,\n        # not merely to its endpoint bonus.\n        nonendpoint_ordinary = active_ordinary * (1.0 - active_endpoint)\n        endpoint_focus = np.ones(active_count, dtype=np.float64)\n        for i in range(active_count):\n            opposing = nonendpoint_ordinary[active_signs != active_signs[i]]\n            if opposing.size > 0:\n                attention_logits = 6.0 * opposing\n                attention_logits -= np.max(attention_logits)\n                attention = np.exp(attention_logits)\n                attention /= attention.sum()\n                opposing_peak = float(np.dot(attention, opposing))\n                endpoint_focus[i] = 1.0 / (\n                    1.0 + 3.0 * opposing_peak ** 1.4\n                )\n\n        focused_endpoint = active_endpoint * endpoint_focus\n        active_claims = (\n            active_ordinary * (1.0 - active_endpoint)\n            + (\n                active_ordinary\n                + float(parameters[\"endpoint_distrust\"])\n            ) * focused_endpoint\n        )\n        signed_claim_contrast = float(np.dot(active_signs, active_claims))\n        total_claim_mass = float(np.sum(active_claims))\n\n        # Relative contrast measures whether distrust is directionally\n        # concentrated rather than spread across two similarly assertive sides.\n        concentration = abs(signed_claim_contrast) / (total_claim_mass + 1e-12)\n        concentration = float(np.clip(concentration, 0.0, 1.0))\n        focus_floor = float(parameters[\"concentration_floor\"])\n        focus = focus_floor + (1.0 - focus_floor) * concentration ** 1.35\n\n        # Estimate heterogeneity within each endorsing coalition. Mixing claims\n        # of substantially different strength on the same side makes their\n        # pragmatic interpretation less coherent and hence more strongly diluted.\n        positive_claims = active_claims[active_signs > 0.0]\n        negative_claims = active_claims[active_signs < 0.0]\n\n        def side_dispersion(values):\n            if values.size <= 1:\n                return 0.0\n            mean_value = float(np.mean(values))\n            return float(np.mean((values - mean_value) ** 2))\n\n        positive_dispersion = side_dispersion(positive_claims)\n        negative_dispersion = side_dispersion(negative_claims)\n        within_dispersion = (\n            positive_claims.size * positive_dispersion\n            + negative_claims.size * negative_dispersion\n        ) / float(active_count)\n\n        breadth_exponent = float(parameters[\"coalition_dilution_exponent\"])\n        breadth_normalizer = float(active_count) ** breadth_exponent\n        heterogeneity_normalizer = 1.0 + (\n            float(parameters[\"composition_dilution\"])\n            * float(active_count)\n            * within_dispersion\n        )\n\n        # Within-side dispersion is exactly zero when each option has only one\n        # active endorser. Across-active dispersion therefore supplies a second,\n        # smoothly sparse-weighted composition diagnostic. It is switched off\n        # for a literal endpoint claim against a neutral claim, preserving the\n        # isolated 100%-versus-50% distrust response.\n        across_dispersion = float(np.mean((active_claims - np.mean(active_claims)) ** 2))\n        sparse_weight = (2.0 / float(active_count)) ** 2\n        endpoint_neutral_exemption = float(np.max(active_endpoint)) * (\n            1.0 - float(np.max(nonendpoint_ordinary))\n        )\n        across_gate = 1.0 - np.clip(endpoint_neutral_exemption, 0.0, 1.0)\n        across_normalizer = 1.0 + (\n            float(parameters[\"across_composition_dilution\"])\n            * sparse_weight\n            * across_gate\n            * across_dispersion\n        )\n\n        contextual_contrast = (\n            signed_claim_contrast * focus\n            / (\n                breadth_normalizer\n                * heterogeneity_normalizer\n                * across_normalizer\n                + 1e-12\n            )\n        )\n\n        # Moderate pragmatic claims receive a small accumulation boost in broad\n        # but not maximally dense coalitions. The smooth validity and breadth\n        # gates avoid strengthening sparse 95%-claim conflicts or dense diffuse\n        # conflicts indiscriminately.\n        strongest_assertiveness = float(np.max(active_above_chance))\n        moderate_gate = logistic(12.0 * (0.72 - strongest_assertiveness))\n        breadth_peak = (\n            (1.0 - np.exp(-max(active_count - 2, 0) / 2.0))\n            * np.exp(-((active_count - 7.0) / 4.0) ** 2)\n        )\n        contextual_contrast *= float(\n            1.0 + 0.15 * moderate_gate * breadth_peak\n        )\n\n        # The anti-reliability signal is bounded and continuously heterogeneous\n        # across subjects. Its sign opposes the communicated validity contrast.\n        accumulated = np.tanh(\n            float(parameters[\"anti_accumulation\"]) * contextual_contrast\n        )\n        anti_reliability = -float(parameters[\"anti_reliability_strength\"]) * float(accumulated)\n\n        # A smooth arbitration function cedes control to the tally as its margin\n        # becomes decisive; there are no exact-tie or one-margin special cases.\n        margin_scale = float(parameters[\"margin_protection_scale\"])\n        margin_power = float(parameters[\"margin_protection_power\"])\n        margin_gate = 1.0 / (\n            1.0 + (abs(tally_margin) / margin_scale) ** margin_power\n        )\n        anti_reliability *= float(margin_gate)\n\n    # Raw votes retain their categorical status but saturate gradually, limiting\n    # overconfidence while preserving dominance by large vote coalitions.\n    tally_evidence = 4.5 * np.tanh(tally_margin / 4.5)\n    total_evidence = tally_evidence + anti_reliability\n\n    evidence_scale = float(parameters[\"evidence_scale\"])\n    normalized_evidence = float(np.tanh(total_evidence / evidence_scale))\n    decision_variable = float(parameters[\"choice_sensitivity\"]) * normalized_evidence\n\n    # Confidence separates most strongly at intermediate margins. The bounded\n    # gain is zero through margin one and decreases again for very large margins,\n    # increasing heterogeneous margin sensitivity without indiscriminately\n    # sharpening every decisive coalition.\n    margin_excess = max(abs(tally_margin) - 1.0, 0.0)\n    intermediate_margin_shape = (\n        0.5 * margin_excess * np.exp(1.0 - margin_excess / 2.0)\n        if margin_excess > 0.0 else 0.0\n    )\n    margin_temperature = 1.0 + float(parameters[\"margin_temperature_gain\"]) * intermediate_margin_shape\n    decision_variable *= float(margin_temperature)\n\n    # Positive evidence favors A. Symmetric logits guarantee that swapping the\n    # displayed options swaps their probabilities exactly.\n    logits = np.array(\n        [0.5 * decision_variable, -0.5 * decision_variable],\n        dtype=np.float64,\n    )\n    logits -= np.max(logits)\n    probabilities = np.exp(logits)\n    probabilities /= probabilities.sum()\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probabilities = (1.0 - lapse) * probabilities + 0.5 * lapse\n    probabilities = np.clip(probabilities, 0.0, 1.0)\n    probabilities /= probabilities.sum()\n    return probabilities",
  "policy": "def policy(probs):\n    probabilities = np.asarray(probs, dtype=np.float64)\n    probabilities = np.clip(probabilities, 0.0, None)\n    total = float(np.sum(probabilities))\n    if not np.isfinite(total) or total <= 0.0:\n        probabilities = np.full(\n            probabilities.size, 1.0 / probabilities.size, dtype=np.float64\n        )\n    else:\n        probabilities /= total\n    return int(np.random.choice(probabilities.size, p=probabilities))",
  "parameters": {
    "anti_reliability_strength": "[3.0, 5.6]",
    "anti_accumulation": "[2.0, 3.5]",
    "endpoint_distrust": "[1.10, 2.05]",
    "coalition_dilution_exponent": "[0.46, 0.72]",
    "composition_dilution": "[0.8, 2.4]",
    "across_composition_dilution": "[1.0, 3.5]",
    "concentration_floor": "[0.38, 0.62]",
    "margin_protection_scale": "[1.55, 2.25]",
    "margin_protection_power": "[2.2, 3.4]",
    "evidence_scale": "[2.2, 3.0]",
    "choice_sensitivity": "[1.45, 2.65]",
    "margin_temperature_gain": "[0.0, 0.32]",
    "lapse_rate": "[0.0, 0.10]",
    "validities": "validities"
  },
  "rationale": "This is a minimal edit to the accepted iteration-7 candidate. The ordinary validity mapping, endpoint range, intermediate-margin temperature, tally mapping, and response transformation are all retained. The main addition is an across-active compositional normalizer. Unlike within-side variance, it remains informative when each side has only one endorser, so it can dilute sparse mismatched claim contrasts in Experiment 8 and related heterogeneous profiles. Its effect decreases quadratically with coalition size, and a continuous endpoint-neutral exemption preserves the successful isolated 100%-versus-50% reversal in Experiment 10. The concentration-floor range is shifted modestly upward to strengthen anti-reliability in diffuse but coherent moderate coalitions such as Experiments 7 and 11. The within-side composition-dilution range is widened upward to offset that increase in heterogeneous broad profiles such as Experiments 1 and 9. These changes express heterogeneity through contextual composition parameters rather than global choice sensitivity, while preserving option-swap symmetry, smooth arbitration, and history independence."
}
```

## Usage

```json
{
  "input_tokens": 36295,
  "output_tokens": 5336
}
```
