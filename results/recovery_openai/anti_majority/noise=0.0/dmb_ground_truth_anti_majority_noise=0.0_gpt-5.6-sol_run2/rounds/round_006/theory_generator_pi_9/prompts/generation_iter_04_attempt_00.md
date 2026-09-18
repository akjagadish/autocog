# generation_iter_04_attempt_00

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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a new Channel-Specific Coherence Integration theory that competes directly with pi_8. Its central claim should be that count and reliability information are represented through functionally distinct summaries. The count channel rapidly computes an option-level mark difference and is robust to the number of additional mutually canceling cues; its response to count margin should be smooth but bounded, accounting for Experiments 5, 10, 12, and 14 without pi_4's excessive linear margin effect. The reliability channel requires binding cue identities to communicated validities and is therefore selectively weakened when the same net reliability residual is distributed across many internally opposing cue groups. Its gain should depend on signed-evidence coherence or concentration, and possibly communicated profile contrast, rather than on total feature count or a generic overload variable. This selective binding interference predicts Experiment 1's attenuation while preserving the count invariance observed in Experiments 2 and 11.

The new theory should retain stable semantic-polarity inference, a small lapse process, and stable subject differences in count-versus-reliability weighting and response precision. It should not retain pi_4's fixed linear weighted sum, and it should not inherit pi_8's generic divisive compression of count evidence based on the number of discriminating cues. The reliability modulation must be independently identified from count margin, diagnostic-cue count, and total display size. Parameter ranges should permit substantial but not obligatory reliability sensitivity, bounded count accumulation, and greater stable heterogeneity than either current simulation, while preserving A/B reversal symmetry. Critical validation predictions are: positive attenuation at fixed reliability residual when reliability evidence becomes internally canceling; approximately zero attenuation at fixed count margin when added cues cancel; no generic late-load knee; and intermediate, profile-dependent reliability slopes in the designs represented by Experiments 7–9 and 13.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_8` (overall score: 0.620)

**Description**
Resource-Adaptive Diagnostic Field Integration theory: people establish a stable interpretation of whether marked ratings are adverse or beneficial, then allocate attention to currently discriminating evidence rather than total displayed feature count. Count and reliability information form a joint diagnostic field. Tally evidence accumulates smoothly with count margin, while reliability attention increases when discriminating cues expose a salient and internally organized reliability contrast. Accessibility is additionally calibrated by the clustering structure of the full communicated validity profile: coherent high-versus-low reliability blocks preserve attention, whereas uniformly graded profiles receive less reliability amplification. Only an unusually large set of simultaneously discriminating cues creates overload; under such overload, count evidence is divisively compressed, especially at larger margins. Opposing count and reliability signals usually inhibit one another through smooth conflict normalization. However, this normalization is locally weaker when the first count advantage appears, because a single additional mark is treated as diagnostic evidence rather than as a reason to globally discount the pre-existing reliability field. Stable subject-level integration style follows a softly polarized continuum between predominantly tally-oriented and predominantly reliability-sensitive policies. Subjects also differ stably in how deterministically they express their integrated evidence. This precision heterogeneity is mean-preserving and becomes less behaviorally expressed as diagnostic demand and overload increase, producing persistent individual differences in ordinary tasks without destabilizing dense displays or changing semantic polarity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Resource-Adaptive Diagnostic Field Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    discriminating = np.abs(difference) > 0.5
    n_discriminating = float(np.sum(discriminating))
    count_difference = float(np.sum(difference))
    margin = abs(count_difference)

    # A stable integration style creates a softly polarized continuum between
    # tally-oriented and reliability-sensitive subjects. It changes relative
    # allocation but does not create trial-specific strategy switches.
    raw_style = float(parameters["reliability_style"])
    style = 0.5 + 0.5 * np.tanh(2.5 * (2.0 * raw_style - 1.0)) / np.tanh(2.5)
    count_allocation = 1.25 - 0.50 * style
    reliability_allocation = 0.45 + 1.10 * style

    # Ordinary count margins accumulate smoothly and approximately linearly
    # near zero, with gradual bounded compression rather than margin-one
    # saturation.
    count_scale = float(parameters["count_scale"])
    base_count = count_scale * np.tanh(count_difference / count_scale)

    # Resource demand is determined by the number of cues that actually need
    # comparison. Non-discriminating display elements impose no direct load.
    # Softplus supplies a differentiable transition with no feature-count knee.
    span = float(parameters["diagnostic_span"])
    softness = float(parameters["overload_softness"])
    z = (n_discriminating - span) / softness
    overload = softness * (np.log1p(np.exp(-abs(z))) + max(z, 0.0))

    # Dense diagnostic fields selectively compress larger count margins. The
    # powered activation is negligible at margin one but rises smoothly for
    # margins around five and above.
    margin_scale = float(parameters["compression_margin_scale"])
    margin_power = float(parameters["margin_activation_power"])
    if margin > 0.0:
        margin_activation = margin ** margin_power / (
            margin ** margin_power + margin_scale ** margin_power
        )
    else:
        margin_activation = 0.0
    count_divisor = 1.0 + float(parameters["overload_compression"]) * overload * margin_activation
    count_signal = (
        float(parameters["count_gain"])
        * count_allocation
        * base_count
        / count_divisor
    )

    # Reliability evidence is residualized against the common tally
    # component. Its accessibility depends on diagnostic contrast and on how
    # coherently the currently discriminating cues organize that contrast.
    centered_validities = validities - float(np.mean(validities))
    reliability_residual = float(np.dot(centered_validities, difference))

    # Clustering is computed from the full communicated validity profile, not
    # from the active subset. A large gap relative to ordinary profile spacing
    # identifies a genuinely organized high-versus-low reliability structure.
    sorted_validities = np.sort(validities)
    profile_gaps = np.diff(sorted_validities)
    positive_gaps = profile_gaps[profile_gaps > 1e-9]
    if positive_gaps.size > 0:
        largest_gap = float(np.max(positive_gaps))
        typical_gap = float(np.median(positive_gaps))
        profile_clustering = np.clip(
            (largest_gap - typical_gap) / (largest_gap + typical_gap + 1e-9),
            0.0,
            1.0,
        )
    else:
        profile_clustering = 0.0

    if n_discriminating > 0.0:
        active_centered = centered_validities[discriminating]
        contrast = float(np.std(active_centered))
        profile_range = float(np.max(validities) - np.min(validities))
        contrast_salience = contrast / (contrast + 0.25 * profile_range + 1e-9)

        absolute_reliability = float(
            np.sum(np.abs(centered_validities[discriminating] * difference[discriminating]))
        )
        organization = abs(reliability_residual) / (absolute_reliability + 1e-9)
    else:
        contrast_salience = 0.0
        organization = 0.0

    # Smooth validity gradients receive less amplification, while clustered
    # profiles retain nearly all of the accepted model's diagnostic attention.
    profile_attention = 0.70 + 0.30 * profile_clustering
    diagnostic_attention = (
        float(parameters["baseline_reliability_attention"])
        + float(parameters["contrast_attention_gain"])
        * contrast_salience
        * organization
        * profile_attention
    )

    # Unorganized reliability detail can suffer under heavy diagnostic load,
    # while a coherent high-versus-low contrast remains available even in a
    # large display.
    reliability_divisor = 1.0 + (
        float(parameters["diffuse_reliability_compression"])
        * overload
        * (1.0 - organization)
    )
    reliability_signal = (
        reliability_allocation
        * diagnostic_attention
        * reliability_residual
        / reliability_divisor
    )

    # Count and reliability are jointly represented. When they recommend
    # opposite markedness directions, divisive cross-inhibition usually
    # reduces total confidence smoothly. A narrow notch around margin one
    # prevents the first count cue from producing excessive normalization.
    scale_sum = abs(count_signal) + abs(reliability_signal)
    if scale_sum > 1e-12 and count_signal * reliability_signal < 0.0:
        conflict = (
            4.0 * abs(count_signal * reliability_signal)
            / (scale_sum * scale_sum + 1e-12)
        )
    else:
        conflict = 0.0

    margin_one_notch = np.exp(-((margin - 1.0) / 0.55) ** 2)
    conflict *= 1.0 - 0.40 * margin_one_notch

    joint_divisor = 1.0 + float(parameters["conflict_normalization"]) * conflict

    # Stable, symmetric precision deviations increase ordinary-task
    # heterogeneity without shifting mean precision. Their expression is
    # attenuated when many cues must be compared and further attenuated by the
    # same overload signal that regulates dense diagnostic fields.
    precision_shrinkage = (
        1.0 + overload + n_discriminating / max(span, 1e-9)
    )
    effective_response_precision = (
        float(parameters["response_precision"])
        + float(parameters["precision_deviation"]) / precision_shrinkage
    )

    markedness_evidence = effective_response_precision * (
        count_signal + reliability_signal
    ) / joint_divisor

    # Positive markedness means A carries more adverse evidence and therefore
    # favors choosing B under the dominant adverse-mark interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total

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
- reliability_style: [0.0, 1.0]
- count_gain: [0.80, 1.24]
- count_scale: [3.2, 5.8]
- diagnostic_span: [20.0, 23.0]
- overload_softness: [0.6, 1.2]
- compression_margin_scale: [2.2, 3.4]
- margin_activation_power: [2.6, 3.4]
- overload_compression: [0.45, 0.80]
- baseline_reliability_attention: [0.25, 0.61]
- contrast_attention_gain: [0.90, 1.50]
- diffuse_reliability_compression: [0.01, 0.08]
- conflict_normalization: [0.05, 0.35]
- response_precision: [0.78, 1.30]
- precision_deviation: [-0.28, 0.28]
- adverse_polarity_confidence: [0.82, 1.0]
- lapse_rate: [0.0, 0.12]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4355 (var=0.0764) vs this=-0.2923 (var=0.0208)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 3: real=0.7367 (var=0.0520) vs this=0.7500 (var=0.0328)
- Experiment 4: real=-0.4825 (var=0.0708) vs this=-0.4454 (var=0.0299)
- Experiment 5: real=0.2306 (var=0.0353) vs this=0.1401 (var=0.0073)
- Experiment 6: real=0.1062 (var=0.0136) vs this=0.1287 (var=0.0071)
- Experiment 7: real=0.0392 (var=0.0080) vs this=0.1037 (var=0.0153)
- Experiment 8: real=-28.0000 (var=21.1456) vs this=-16.0000 (var=35.5776)
- Experiment 9: real=-0.0407 (var=0.3436) vs this=0.8355 (var=0.6985)
- Experiment 10: real=0.8750 (var=0.0078) vs this=0.8628 (var=0.0039)
- Experiment 11: real=0.8442 (var=0.0078) vs this=0.8721 (var=0.0046)
- Experiment 12: real=0.0793 (var=0.0131) vs this=0.0680 (var=0.0018)
- Experiment 13: real=0.0315 (var=0.0055) vs this=0.0333 (var=0.0023)
- Experiment 14: real=-0.0264 (var=0.0053) vs this=0.2169 (var=0.0099)


---

### `pi_5` (overall score: 0.569)

**Description**
Discrepancy-Evoked Residual Audit with Capacity-Limited Tie Escalation: people first form an adverse-markedness preference from a nonlinear count difference, while count-residualized validity information remains continuously available. Count discrepancies evoke an additional validity audit, but exact count ties trigger a distinct escalation because tallying provides no provisional winner. This tie escalation is strongest when the cue set fits within a fixed attentional capacity and rapidly dilutes as the number of cues exceeds that capacity. Stable subject differences in tie escalation complement existing differences in count reliance, discrepancy auditing, semantic polarity, precision, and lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Discrepancy-Evoked Residual Audit expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))
    margin = abs(count_difference)

    # A bounded provisional tally prevents the strength of the count signal
    # from growing without limit as experiments add more features.
    count_scale = float(parameters["count_scale"])
    count_signal = float(parameters["count_strength"]) * np.tanh(
        count_difference / count_scale
    )

    # Small discrepancies leave the provisional tally relatively uncertain.
    # This redistributes count strength away from margins one and two without
    # weakening the separately reinforced moderate and extreme branches.
    if 0.0 < margin <= 2.0:
        count_signal *= float(parameters["small_margin_attenuation"])

    # Reinforce moderate discrepancies without changing margin-one or
    # margin-two behavior and without adding growth to the extreme branch.
    if 3.0 <= margin < 5.0:
        count_signal += (
            float(parameters["intermediate_count_reinforcement"])
            * float(np.sign(count_difference))
        )

    # Selectively reinforce only decisive margins, leaving small- and
    # moderate-margin behavior otherwise unchanged.
    if margin >= 5.0:
        extreme_activation = np.tanh(max(0.0, margin - 4.0))
        count_signal += (
            float(parameters["extreme_count_tail"])
            * float(np.sign(count_difference))
            * extreme_activation
        )

    # Centering cue validity removes the component of weighted evidence that
    # simply restates which option has more marks. The remaining signal asks
    # whether an option's marks are carried by unusually reliable experts.
    centered_validities = validities - float(np.mean(validities))
    residual_validity = float(np.dot(centered_validities, difference))

    # Validity is always available through baseline_validity_gain. A count
    # discrepancy evokes an extra audit, maximal at a one-mark margin and
    # progressively suppressed when the count margin is already decisive.
    baseline_gain = float(parameters["baseline_validity_gain"])
    audit_gain = float(parameters["audit_gain"])
    audit_decay = float(parameters["audit_decay"])
    if margin > 0.0:
        discrepancy_activation = margin * np.exp(
            -audit_decay * max(0.0, margin - 1.0)
        )
    else:
        discrepancy_activation = 0.0
    validity_gain = baseline_gain + audit_gain * discrepancy_activation

    # When tallying supplies no provisional winner, validity inspection is
    # escalated. The escalation is diluted beyond a seven-cue attentional
    # capacity, separating compact tie problems from large cue arrays while
    # retaining the nonzero validity floor in every task.
    if np.isclose(margin, 0.0):
        tie_capacity = np.exp(-max(0.0, float(n_features) - 7.0))
        validity_gain += float(parameters["tie_validity_bonus"]) * tie_capacity

    # A small, structurally localized audit may be elicited at margin three.
    # Its reduced range prevents it from counteracting moderate count evidence.
    if np.isclose(margin, 3.0):
        validity_gain += float(parameters["margin_three_audit_gain"])

    markedness_evidence = count_signal + validity_gain * residual_validity
    markedness_evidence *= float(parameters["response_precision"])

    # Positive evidence means A carries the greater adverse burden and hence
    # favors B. The alternative semantic interpretation reverses all evidence,
    # and stable polarity confidence mixes the two interpretations.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- validities: validities
- count_strength: [2.1, 3.5]
- count_scale: [1.2, 2.0]
- small_margin_attenuation: [0.72, 0.92]
- intermediate_count_reinforcement: [1.6, 3.4]
- extreme_count_tail: [1.0, 2.8]
- baseline_validity_gain: [0.38, 0.88]
- tie_validity_bonus: [0.05, 1.05]
- audit_gain: [0.9, 1.6]
- audit_decay: [0.7, 1.4]
- margin_three_audit_gain: [0.0, 0.1]
- response_precision: [0.75, 1.4]
- adverse_polarity_confidence: [0.74, 1.0]
- lapse_rate: [0.0, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4355 (var=0.0764) vs this=-0.2945 (var=0.0273)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 3: real=0.7367 (var=0.0520) vs this=0.6617 (var=0.0343)
- Experiment 4: real=-0.4825 (var=0.0708) vs this=-0.3983 (var=0.0255)
- Experiment 5: real=0.2306 (var=0.0353) vs this=0.2406 (var=0.0065)
- Experiment 6: real=0.1062 (var=0.0136) vs this=0.0940 (var=0.0057)
- Experiment 7: real=0.0392 (var=0.0080) vs this=-0.0929 (var=0.0062)
- Experiment 8: real=-28.0000 (var=21.1456) vs this=12.0000 (var=54.0224)
- Experiment 9: real=-0.0407 (var=0.3436) vs this=-1.6900 (var=0.8398)
- Experiment 10: real=0.8750 (var=0.0078) vs this=0.8325 (var=0.0068)
- Experiment 11: real=0.8442 (var=0.0078) vs this=0.7937 (var=0.0092)
- Experiment 12: real=0.0793 (var=0.0131) vs this=0.1059 (var=0.0024)
- Experiment 13: real=0.0315 (var=0.0055) vs this=-0.0049 (var=0.0022)
- Experiment 14: real=-0.0264 (var=0.0053) vs this=-0.0057 (var=0.0066)


---

### `pi_4` (overall score: 0.538)

**Description**
Semantic-Polarity Smooth Integration theory: people first interpret what the marked binary state means rather than assuming that 1 is intrinsically desirable. Each subject forms a stable posterior belief that a mark is adverse or suspicious. Conditional on either the adverse interpretation or its residual beneficial alternative, the subject smoothly integrates two continuous signals: the difference in total marks and a partially count-residualized validity-weighted difference. The count signal normally dominates, while the weighted signal supplies graded sensitivity to which experts carry the marks on every trial. The two semantic interpretations generate opposite choice distributions, which are averaged according to the subject's polarity confidence. A small lapse process captures stimulus-independent errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Semantic-Polarity Smooth Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    # Positive differences mean that A contains more marked states than B.
    count_difference = float(np.sum(a - b))
    partial_centering = float(parameters["partial_centering"])
    effective_validities = validities - partial_centering * float(np.mean(validities))
    weighted_difference = float(np.dot(effective_validities, a - b))

    # There is no count gate: validity contributes continuously on every
    # discriminating trial. Partial centering limits the count component
    # embedded in the weighted sum without removing it completely.
    integrated_markedness = (
        float(parameters["count_weight"]) * count_difference
        + float(parameters["validity_weight"]) * weighted_difference
    )

    adverse_logits = np.array(
        [-0.5 * integrated_markedness, 0.5 * integrated_markedness],
        dtype=float,
    )
    adverse_logits = adverse_logits - np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # The beneficial interpretation reverses the semantic polarity while
    # preserving the same smooth evidence-integration computation.
    beneficial_probs = adverse_probs[::-1]
    polarity_confidence = float(parameters["adverse_polarity_confidence"])
    core_probs = (
        polarity_confidence * adverse_probs
        + (1.0 - polarity_confidence) * beneficial_probs
    )

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- validities: validities
- count_weight: [0.38, 1.05]
- validity_weight: [0.7, 1.8]
- partial_centering: [0.1, 0.3]
- adverse_polarity_confidence: [0.78, 1.0]
- lapse_rate: [0.0, 0.12]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.4355 (var=0.0764) vs this=-0.3570 (var=0.0237)
- Experiment 2: real=0.0000 (var=0.0000) vs this=0.0000 (var=0.0000)
- Experiment 3: real=0.7367 (var=0.0520) vs this=0.7317 (var=0.0354)
- Experiment 4: real=-0.4825 (var=0.0708) vs this=-0.5396 (var=0.0205)
- Experiment 5: real=0.2306 (var=0.0353) vs this=0.2053 (var=0.0075)
- Experiment 6: real=0.1062 (var=0.0136) vs this=0.2086 (var=0.0061)
- Experiment 7: real=0.0392 (var=0.0080) vs this=0.0925 (var=0.0089)
- Experiment 8: real=-28.0000 (var=21.1456) vs this=-112.0000 (var=44.4224)
- Experiment 9: real=-0.0407 (var=0.3436) vs this=1.3194 (var=0.6212)
- Experiment 10: real=0.8750 (var=0.0078) vs this=0.8800 (var=0.0062)
- Experiment 11: real=0.8442 (var=0.0078) vs this=0.8542 (var=0.0061)
- Experiment 12: real=0.0793 (var=0.0131) vs this=0.0704 (var=0.0014)
- Experiment 13: real=0.0315 (var=0.0055) vs this=-0.0062 (var=0.0023)
- Experiment 14: real=-0.0264 (var=0.0053) vs this=-0.0049 (var=0.0057)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0983 -> ACCEPTED
- iter 2: loss=0.1296 -> REJECTED
- iter 3: loss=0.1044 -> REJECTED
- iter 4: loss=0.1270 -> REJECTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0983 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    aligned = []
    tally_margins = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Validities are strictly descending in feature order, so the first
        # feature that differs identifies the Take-The-Best winner.
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue
        j = int(differing[0])
        ttb_winner = 0 if a[j] > b[j] else 1

        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        tally_margins.append(abs(a_wins - b_wins))
        aligned.append(float(int(row["response"]) == ttb_winner))

    if len(aligned) == 0:
        return float("nan")

    aligned = np.asarray(aligned, dtype=float)
    tally_margins = np.asarray(tally_margins, dtype=int)
    tie = tally_margins == 0
    conflict = tally_margins > 0

    if not np.any(tie) or not np.any(conflict):
        return float("nan")

    # Tally-margin sensitivity: excess TTB agreement on tally ties relative
    # to trials where the tally favors the opposite option.
    return float(np.mean(aligned[tie]) - np.mean(aligned[conflict]))
```

**Observed (real) value:** -0.4355 (var=0.0764)
**Previous candidate values (this loop):**
  - iter 1: -0.2380 (var=0.0154) (Δ vs real +0.1975)
  - iter 2: -0.2267 (var=0.0261) (Δ vs real +0.2088)
  - iter 3: -0.2288 (var=0.0273) (Δ vs real +0.2068)
  - iter 4 (most recent): -0.2278 (var=0.0233) (Δ vs real +0.2078)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0052 (var=0.0117)
- pi_2: 0.3523 (var=0.0241)
- pi_3: -0.4570 (var=0.0555)
- pi_4: -0.3570 (var=0.0237)
- pi_5: -0.2945 (var=0.0273)
- pi_6: -0.3425 (var=0.0220)
- pi_7: -0.3760 (var=0.0198)
- pi_8: -0.2923 (var=0.0208)

### Experiment 2
**Design**
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score each choice against the winner designated by the highest-validity
    # discriminating cue. Validities are strictly decreasing with feature index.
    signed_scores = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        first = int(differing[0])
        ttb_winner = 0 if a[first] > b[first] else 1
        signed_scores.append(1.0 if int(response) == ttb_winner else -1.0)

    if not signed_scores:
        return float("nan")

    alignment = float(np.mean(signed_scores))

    # This corresponds to choosing the validity-ordered winner on more than
    # 62.5% of trials. It is well beyond ordinary finite-sample fluctuation
    # around Tallying's predicted 50%, while remaining below typical TTB rates.
    return float(alignment > 0.25)

```

**Observed (real) value:** 0.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3: 0.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 4 (most recent): 0.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0000 (var=0.0000)
- pi_1: 1.0000 (var=0.0900)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0000 (var=0.0000)
- pi_6: 0.0000 (var=0.0000)
- pi_7: 0.0000 (var=0.0000)
- pi_8: 0.0000 (var=0.0000)

### Experiment 3
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        count_delta = float(np.sum(a) - np.sum(b))

        # Restrict attention to the strongly diagnostic count-margin trials.
        if abs(count_delta) >= 5.0:
            response_sign = 2.0 * float(response) - 1.0  # B=+1, A=-1
            scores.append(float(np.sign(count_delta) * response_sign))

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** 0.7367 (var=0.0520)
**Previous candidate values (this loop):**
  - iter 1: 0.6133 (var=0.0363) (Δ vs real -0.1233)
  - iter 2: 0.6650 (var=0.0365) (Δ vs real -0.0717)
  - iter 3: 0.6833 (var=0.0331) (Δ vs real -0.0533)
  - iter 4 (most recent): 0.6100 (var=0.0310) (Δ vs real -0.1267)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6383 (var=0.0899)
- pi_2: -0.7000 (var=0.0639)
- pi_1: -0.6600 (var=0.0730)
- pi_4: 0.7317 (var=0.0354)
- pi_5: 0.6617 (var=0.0343)
- pi_6: 0.7217 (var=0.0254)
- pi_7: 0.7817 (var=0.0222)
- pi_8: 0.7500 (var=0.0328)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Weighted choice alignment with the feature-count majority."""
    if data is None or len(data) == 0:
        return 0.0

    weighted_scores = []
    weights = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        count_delta = float(np.sum(a) - np.sum(b))

        # Count ties do not identify a tallying preference and are omitted.
        if count_delta == 0.0:
            continue

        response = int(row["response"])
        # +1 when the option with more positive ratings was selected,
        # -1 when the option with fewer positive ratings was selected.
        choice_sign = 1.0 if response == 0 else -1.0
        alignment = choice_sign * np.sign(count_delta)

        # Margin-two trials receive more weight because both theories make
        # more precise, oppositely directed predictions on them.
        weight = abs(count_delta)
        weighted_scores.append(weight * alignment)
        weights.append(weight)

    if not weights:
        return 0.0
    return float(np.sum(weighted_scores) / np.sum(weights))

```

**Observed (real) value:** -0.4825 (var=0.0708)
**Previous candidate values (this loop):**
  - iter 1: -0.3079 (var=0.0509) (Δ vs real +0.1746)
  - iter 2: -0.3204 (var=0.0408) (Δ vs real +0.1621)
  - iter 3: -0.3237 (var=0.0321) (Δ vs real +0.1588)
  - iter 4 (most recent): -0.3183 (var=0.0194) (Δ vs real +0.1642)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7229 (var=0.0432)
- pi_3: -0.4696 (var=0.0469)
- pi_1: -0.6617 (var=0.0574)
- pi_4: -0.5396 (var=0.0205)
- pi_5: -0.3983 (var=0.0255)
- pi_6: -0.6242 (var=0.0183)
- pi_7: -0.6488 (var=0.0220)
- pi_8: -0.4454 (var=0.0299)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array(
        [1.0, 0.98, 0.96, 0.94, 0.92, 0.58, 0.56, 0.54, 0.52, 0.5, 0.5],
        dtype=float,
    )

    validity_advantage = []
    chose_fewer_marks = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        count_a = float(a.sum())
        count_b = float(b.sum())

        if count_a > count_b:
            # A is the option with more marks; B has fewer marks.
            rich_minus_sparse = a - b
            fewer_choice = float(response == 1)
        elif count_b > count_a:
            # B is the option with more marks; A has fewer marks.
            rich_minus_sparse = b - a
            fewer_choice = float(response == 0)
        else:
            # Not expected in this design, but omit count ties safely.
            continue

        validity_advantage.append(float(np.dot(validities, rich_minus_sparse)))
        chose_fewer_marks.append(fewer_choice)

    x = np.asarray(validity_advantage, dtype=float)
    y = np.asarray(chose_fewer_marks, dtype=float)
    if x.size < 2:
        return 0.0

    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0

    # OLS slope of choosing the fewer-mark option on the validity-weighted
    # advantage held by the more-marked option.
    return float(np.dot(x_centered, y - y.mean()) / denominator)

```

**Observed (real) value:** 0.2306 (var=0.0353)
**Previous candidate values (this loop):**
  - iter 1: 0.1179 (var=0.0070) (Δ vs real -0.1128)
  - iter 2: 0.1401 (var=0.0071) (Δ vs real -0.0906)
  - iter 3: 0.1349 (var=0.0074) (Δ vs real -0.0957)
  - iter 4 (most recent): 0.1352 (var=0.0081) (Δ vs real -0.0954)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0002 (var=0.0045)
- pi_4: 0.2053 (var=0.0075)
- pi_1: -0.2514 (var=0.0060)
- pi_2: -0.0002 (var=0.0032)
- pi_5: 0.2406 (var=0.0065)
- pi_6: 0.2402 (var=0.0072)
- pi_7: 0.1866 (var=0.0085)
- pi_8: 0.1401 (var=0.0073)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array(
        [0.99, 0.94, 0.89, 0.84, 0.79, 0.74, 0.69, 0.64, 0.59, 0.54, 0.50],
        dtype=float,
    )

    if len(data) == 0:
        return float("nan")

    a = np.vstack(data["option_a_ratings"].apply(lambda z: np.asarray(z, dtype=float)).to_numpy())
    b = np.vstack(data["option_b_ratings"].apply(lambda z: np.asarray(z, dtype=float)).to_numpy())
    response = data["response"].to_numpy(dtype=float)

    # Orient each complementary stimulus so that the option marked by the
    # highest-validity expert is the reference marked option.
    orientation = np.sign(a[:, 0] - b[:, 0])
    usable = np.isfinite(response) & (orientation != 0)
    if np.sum(usable) < 2:
        return float("nan")

    orientation = orientation[usable]
    response = response[usable]
    diff = (a[usable] - b[usable])

    # Weighted markedness of the highest-validity-marked option relative to
    # its alternative. Complementary presentations receive the same x value.
    x = orientation * np.dot(diff, validities)

    # One means choosing the option unmarked by the highest-validity expert.
    y = np.where(orientation > 0, response, 1.0 - response)

    # Within-data OLS slope. Centering removes the competitor's subject-level
    # but condition-invariant preference for the unmarked option.
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.dot(xc, y - np.mean(y)) / denom)

```

**Observed (real) value:** 0.1062 (var=0.0136)
**Previous candidate values (this loop):**
  - iter 1: 0.0991 (var=0.0079) (Δ vs real -0.0072)
  - iter 2: 0.1665 (var=0.0086) (Δ vs real +0.0603)
  - iter 3: 0.1021 (var=0.0071) (Δ vs real -0.0042)
  - iter 4 (most recent): 0.1279 (var=0.0046) (Δ vs real +0.0217)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2086 (var=0.0061)
- pi_3: 0.0054 (var=0.0029)
- pi_1: 0.0091 (var=0.0021)
- pi_2: 0.0039 (var=0.0034)
- pi_5: 0.0940 (var=0.0057)
- pi_6: 0.2569 (var=0.0066)
- pi_7: 0.2163 (var=0.0049)
- pi_8: 0.1287 (var=0.0071)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.75,
        1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94
    ], dtype=float)
    centered = validities - validities.mean()

    aligned_choices = []
    margins = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        difference = a - b
        margin = abs(float(difference.sum()))
        residual = float(np.dot(centered, difference))

        # Code 1 when the chosen option is the one whose marks are relatively
        # concentrated among above-mean-validity experts. This makes reversed
        # A/B versions contribute in the same psychological direction.
        if residual < 0.0:
            aligned = float(int(response) == 1)  # B is the high-validity side
        elif residual > 0.0:
            aligned = float(int(response) == 0)  # A is the high-validity side
        else:
            continue

        aligned_choices.append(aligned)
        margins.append(margin)

    aligned_choices = np.asarray(aligned_choices, dtype=float)
    margins = np.asarray(margins, dtype=float)
    tie = np.isclose(margins, 0.0)
    one = np.isclose(margins, 1.0)
    if not np.any(tie) or not np.any(one):
        return float("nan")

    return float(aligned_choices[one].mean() - aligned_choices[tie].mean())
```

**Observed (real) value:** 0.0392 (var=0.0080)
**Previous candidate values (this loop):**
  - iter 1: 0.0683 (var=0.0074) (Δ vs real +0.0292)
  - iter 2: 0.0879 (var=0.0224) (Δ vs real +0.0488)
  - iter 3: 0.0625 (var=0.0093) (Δ vs real +0.0233)
  - iter 4 (most recent): 0.0675 (var=0.0080) (Δ vs real +0.0283)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0929 (var=0.0062)
- pi_4: 0.0925 (var=0.0089)
- pi_1: 0.0079 (var=0.0048)
- pi_2: -0.3358 (var=0.0121)
- pi_3: 0.3675 (var=0.0270)
- pi_6: 0.0688 (var=0.0111)
- pi_7: 0.4404 (var=0.0130)
- pi_8: 0.1037 (var=0.0153)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code each response as choosing the option with fewer marked ratings.
    # A positive count difference means A has more marks, so response 1 (B)
    # is the fewer-mark choice; a negative difference makes response 0 (A)
    # the fewer-mark choice.
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = float(np.sum(a - b))
        margin = abs(d)
        if not (np.isclose(margin, 1.0) or np.isclose(margin, 5.0)):
            continue

        fewer_choice = 1 if d > 0.0 else 0
        chose_fewer = int(row["response"]) == fewer_choice
        choice_sign = 1.0 if chose_fewer else -1.0
        margin_sign = 1.0 if np.isclose(margin, 5.0) else -1.0
        scores.append(margin_sign * choice_sign)

    if len(scores) == 0:
        return 0.0
    return float(np.sum(scores))
```

**Observed (real) value:** -28.0000 (var=21.1456)
**Previous candidate values (this loop):**
  - iter 1: -64.0000 (var=87.4816) (Δ vs real -36.0000)
  - iter 2: 120.0000 (var=33.1200) (Δ vs real +148.0000)
  - iter 3: 28.0000 (var=42.4064) (Δ vs real +56.0000)
  - iter 4 (most recent): 60.0000 (var=58.5600) (Δ vs real +88.0000)
**Other theories' values on this metric (for reference):**
- pi_4: -112.0000 (var=44.4224)
- pi_5: 12.0000 (var=54.0224)
- pi_1: -8.0000 (var=59.4944)
- pi_2: -156.0000 (var=46.7456)
- pi_3: 360.0000 (var=66.7200)
- pi_6: -14.0000 (var=36.3216)
- pi_7: -18.0000 (var=45.2304)
- pi_8: -16.0000 (var=35.5776)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return 0.0

    def subject_score(frame: pd.DataFrame) -> float:
        x_values = []
        outcomes = []

        for _, row in frame.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            difference = a - b
            n_features = int(difference.size)
            n_discriminating = float(np.sum(np.abs(difference)))
            count_difference = float(np.sum(difference))

            if n_features <= 0 or n_discriminating <= 0.0 or count_difference == 0.0:
                continue

            # Choice coded toward the option with fewer marked ratings.
            response = int(row["response"])
            chose_fewer = float(response if count_difference > 0.0 else 1 - response)

            fraction = n_discriminating / float(n_features)
            x = np.tanh(
                1.0
                / (
                    0.9
                    * max(fraction, 1.0 / float(n_features)) ** 0.325
                )
            )
            x_values.append(float(x))
            outcomes.append(chose_fewer)

        if len(x_values) < 2:
            return 0.0

        x = np.asarray(x_values, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        weights = x - float(np.mean(x))
        weight_energy = float(np.dot(weights, weights))
        if weight_energy <= 0.0 or not np.isfinite(weight_energy):
            return 0.0

        # Directional trial-level evidence, standardized by its Bernoulli-null
        # standard deviation. Centering the weights removes overall response
        # propensity and polarity confidence.
        numerator = float(np.dot(weights, y - 0.5))
        denominator = float(np.sqrt(0.25 * weight_energy))
        value = numerator / denominator
        return float(value) if np.isfinite(value) else 0.0

    scores = []
    for _, subject_frame in data.groupby("subject_id", sort=False):
        scores.append(subject_score(subject_frame))

    if len(scores) == 0:
        return 0.0

    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0

    # Stouffer aggregation: independent subjects contribute cumulative
    # standardized evidence. For a one-subject slice this is simply that
    # subject's score, so the metric remains well-defined for variance
    # estimation.
    value = float(np.sum(scores) / np.sqrt(float(scores.size)))
    return value if np.isfinite(value) else 0.0

```

**Observed (real) value:** -0.0407 (var=0.3436)
**Previous candidate values (this loop):**
  - iter 1: 0.3635 (var=0.8864) (Δ vs real +0.4041)
  - iter 2: 0.7439 (var=1.4927) (Δ vs real +0.7846)
  - iter 3: 0.7636 (var=0.5960) (Δ vs real +0.8043)
  - iter 4 (most recent): 0.2658 (var=0.5632) (Δ vs real +0.3064)
**Other theories' values on this metric (for reference):**
- pi_6: 2.4236 (var=0.6234)
- pi_4: 1.3194 (var=0.6212)
- pi_1: 0.0576 (var=0.4622)
- pi_2: 1.3124 (var=0.3777)
- pi_3: 1.4891 (var=1.1584)
- pi_5: -1.6900 (var=0.8398)
- pi_7: -0.8319 (var=0.4823)
- pi_8: 0.8355 (var=0.6985)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    aligned_large_margin = []

    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        signed_margin = float(np.sum(a) - np.sum(b))

        # Restrict attention to the four large-margin conditions (5, 7, 9, 13).
        if abs(signed_margin) < 5.0:
            continue

        # One indicates selection of the option with fewer marked ratings.
        if signed_margin > 0.0:
            aligned_large_margin.append(float(response == 1))
        elif signed_margin < 0.0:
            aligned_large_margin.append(float(response == 0))

    if len(aligned_large_margin) == 0:
        return float('nan')

    return float(np.mean(np.asarray(aligned_large_margin, dtype=float)))
```

**Observed (real) value:** 0.8750 (var=0.0078)
**Previous candidate values (this loop):**
  - iter 1: 0.8116 (var=0.0063) (Δ vs real -0.0634)
  - iter 2: 0.8394 (var=0.0071) (Δ vs real -0.0356)
  - iter 3: 0.8275 (var=0.0053) (Δ vs real -0.0475)
  - iter 4 (most recent): 0.8003 (var=0.0055) (Δ vs real -0.0747)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8800 (var=0.0062)
- pi_6: 0.7953 (var=0.0044)
- pi_1: 0.3047 (var=0.0036)
- pi_2: 0.1163 (var=0.0072)
- pi_3: 0.7650 (var=0.0150)
- pi_5: 0.8325 (var=0.0068)
- pi_7: 0.8641 (var=0.0034)
- pi_8: 0.8628 (var=0.0039)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91,
        0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.75,
        0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
        0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68
    ], dtype=float)
    centered = validities - validities.mean()
    scores = []

    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.size != validities.size or b.size != validities.size:
            continue

        difference = a - b
        residual = float(np.dot(centered, difference))
        marks_per_option = int(round(float(np.sum(a))))

        # Retain the three strongest, most diagnostic conflict levels
        # (m = 7, 11, or 18), including both A/B orientations.
        if marks_per_option < 7 or abs(residual) < 2.5:
            continue

        response = int(row['response'])
        # Under the adverse-mark interpretation, the preferred option is the
        # one carrying the lower-validity marks: B if A has the high-validity
        # marks (positive residual), and A in the reversed presentation.
        chose_low_validity_marked_option = (
            (residual > 0.0 and response == 1) or
            (residual < 0.0 and response == 0)
        )
        scores.append(float(chose_low_validity_marked_option))

    if len(scores) == 0:
        return float('nan')
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8442 (var=0.0078)
**Previous candidate values (this loop):**
  - iter 1: 0.8029 (var=0.0098) (Δ vs real -0.0413)
  - iter 2: 0.7963 (var=0.0133) (Δ vs real -0.0479)
  - iter 3: 0.8192 (var=0.0132) (Δ vs real -0.0250)
  - iter 4 (most recent): 0.8325 (var=0.0082) (Δ vs real -0.0117)
**Other theories' values on this metric (for reference):**
- pi_7: 0.5038 (var=0.0050)
- pi_4: 0.8542 (var=0.0061)
- pi_1: 0.1546 (var=0.0096)
- pi_2: 0.4808 (var=0.0046)
- pi_3: 0.6883 (var=0.0142)
- pi_5: 0.7937 (var=0.0092)
- pi_6: 0.8725 (var=0.0061)
- pi_8: 0.8721 (var=0.0046)

### Experiment 12
**Design**
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    margins = []
    adverse_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        count_difference = float(np.sum(a) - np.sum(b))
        margin = abs(count_difference)

        # Exclude count ties; the diagnostic trajectory is over margins 1--5.
        if margin < 0.5:
            continue

        response = int(row["response"])
        # Under the adverse-mark interpretation, choose B when A has more
        # marks and choose A when B has more marks.
        adverse_choice = response if count_difference > 0 else 1 - response
        margins.append(margin)
        adverse_choices.append(float(adverse_choice))

    if len(margins) < 2:
        return float("nan")

    x = np.asarray(margins, dtype=float)
    y = np.asarray(adverse_choices, dtype=float)
    x_centered = x - np.mean(x)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return float("nan")

    # OLS increase in adverse-aligned choice probability per unit margin.
    return float(np.dot(x_centered, y - np.mean(y)) / denominator)
```

**Observed (real) value:** 0.0793 (var=0.0131)
**Previous candidate values (this loop):**
  - iter 1: 0.0555 (var=0.0016) (Δ vs real -0.0238)
  - iter 2: 0.0585 (var=0.0013) (Δ vs real -0.0207)
  - iter 3: 0.0600 (var=0.0016) (Δ vs real -0.0193)
  - iter 4 (most recent): 0.0511 (var=0.0014) (Δ vs real -0.0281)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0704 (var=0.0014)
- pi_7: -0.0020 (var=0.0013)
- pi_1: -0.0019 (var=0.0008)
- pi_2: -0.0053 (var=0.0004)
- pi_3: 0.0224 (var=0.0011)
- pi_5: 0.1059 (var=0.0024)
- pi_6: -0.0039 (var=0.0015)
- pi_8: 0.0680 (var=0.0018)

### Experiment 13
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.99, 0.51, 0.98, 0.52, 0.97, 0.53, 0.96, 0.54,
        0.95, 0.55, 0.94, 0.56, 0.93, 0.57, 0.92, 0.58,
        0.91, 0.59, 0.90, 0.60, 0.89, 0.61, 0.88, 0.62,
        0.87, 0.63, 0.86, 0.64, 0.85, 0.65, 0.84, 0.66
    ], dtype=float)
    centered = validities - np.mean(validities)

    loads = []
    aligned_choices = []
    for a_cell, b_cell, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        difference = a - b
        load = int(np.sum(np.abs(difference) > 0.5))
        residual = float(np.dot(centered, difference))

        # One means choosing the option favored under the adverse-mark
        # interpretation of the fixed reliability residual.
        if residual > 0.0:
            aligned = float(response == 1)
        elif residual < 0.0:
            aligned = float(response == 0)
        else:
            aligned = 0.5

        loads.append(load)
        aligned_choices.append(aligned)

    loads = np.asarray(loads, dtype=int)
    y = np.asarray(aligned_choices, dtype=float)
    if y.size == 0:
        return float('nan')

    # Matched contrast for the advocated theory's predicted attenuation
    # profile. The coefficients sum to zero over the eight equally frequent
    # diagnostic-load conditions, removing a subject's overall choice level.
    weight_map = {
        4: 16.0,
        8: 5.0,
        12: 1.0,
        16: -1.0,
        20: -2.0,
        24: -4.0,
        28: -7.0,
        32: -8.0
    }
    w = np.asarray([weight_map.get(int(k), 0.0) for k in loads], dtype=float)

    # Empirical centering retains intercept cancellation if a slice happens
    # to contain unequal numbers of repetitions.
    w = w - np.mean(w)
    rms = float(np.sqrt(np.mean(w * w)))
    if rms <= 1e-12:
        return 0.0
    w = w / rms

    return float(np.mean(w * y))
```

**Observed (real) value:** 0.0315 (var=0.0055)
**Previous candidate values (this loop):**
  - iter 1: 0.0311 (var=0.0021) (Δ vs real -0.0004)
  - iter 2: 0.0298 (var=0.0024) (Δ vs real -0.0017)
  - iter 3: 0.0239 (var=0.0033) (Δ vs real -0.0076)
  - iter 4 (most recent): 0.0205 (var=0.0031) (Δ vs real -0.0110)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0333 (var=0.0023)
- pi_4: -0.0062 (var=0.0023)
- pi_1: 0.0024 (var=0.0015)
- pi_2: -0.0067 (var=0.0018)
- pi_3: 0.0005 (var=0.0033)
- pi_5: -0.0049 (var=0.0022)
- pi_6: 0.0051 (var=0.0016)
- pi_7: 0.0047 (var=0.0033)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sparse = []
    dense = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        difference = a - b
        count_difference = float(np.sum(difference))
        n_discriminating = int(np.sum(np.abs(difference) > 0.5))

        if count_difference > 0:
            adverse_consistent = float(int(row["response"]) == 1)
        elif count_difference < 0:
            adverse_consistent = float(int(row["response"]) == 0)
        else:
            continue

        if n_discriminating <= 19:
            sparse.append(adverse_consistent)
        elif n_discriminating >= 27:
            dense.append(adverse_consistent)

    if len(sparse) == 0 or len(dense) == 0:
        return float("nan")

    return float(np.mean(sparse) - np.mean(dense))
```

**Observed (real) value:** -0.0264 (var=0.0053)
**Previous candidate values (this loop):**
  - iter 1: -0.0029 (var=0.0076) (Δ vs real +0.0235)
  - iter 2: -0.0056 (var=0.0064) (Δ vs real +0.0208)
  - iter 3: -0.0143 (var=0.0073) (Δ vs real +0.0121)
  - iter 4 (most recent): 0.0010 (var=0.0060) (Δ vs real +0.0274)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0049 (var=0.0057)
- pi_8: 0.2169 (var=0.0099)
- pi_1: -0.0187 (var=0.0066)
- pi_2: -0.0262 (var=0.0049)
- pi_3: 0.0015 (var=0.0071)
- pi_5: -0.0057 (var=0.0066)
- pi_6: 0.0365 (var=0.0102)
- pi_7: -0.0257 (var=0.0058)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Channel-Specific Coherence Integration theory: binary counts and cue reliabilities are encoded by distinct cognitive channels before being combined. The count channel rapidly computes the difference between option-level numbers of marked ratings. It depends only on that signed margin, so adding mutually canceling cues does not dilute count evidence. Its response is smooth and bounded, reflecting saturation of an approximate magnitude representation rather than unlimited linear accumulation. The reliability channel instead binds each discriminating cue's identity to its communicated validity. It first removes the validity component redundant with the count margin and then measures the signed coherence of the remaining cue-validity contributions. A reliability residual supported by mutually consistent contributions receives high gain, whereas the same residual embedded in opposing contributions receives lower gain because maintaining the required cue-identity bindings is difficult. This loss is determined by evidence coherence and communicated validity-profile contrast, not by feature count, diagnostic-cue count, or a generic load threshold. Stable subject differences govern relative allocation to the two channels and response precision. A stable semantic-polarity belief determines whether marks are interpreted as adverse or beneficial, and a small lapse process captures stimulus-independent choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Channel-Specific Coherence Integration expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    difference = a - b
    count_difference = float(np.sum(difference))

    # Stable integration style allocates expression strength between two
    # separately computed channels. The smooth transformation allows broad
    # subject heterogeneity without imposing discrete strategy classes.
    raw_style = float(parameters["reliability_style"])
    style = 0.5 + 0.5 * np.tanh(2.2 * (2.0 * raw_style - 1.0)) / np.tanh(2.2)
    count_allocation = 1.30 - 0.60 * style
    reliability_allocation = 0.35 + 1.35 * style

    # The count channel reads only the option-level mark difference. Hence
    # mutually canceling added cues have exactly no effect. Saturation is
    # gradual and smooth rather than a hard margin rule or load-dependent
    # divisive compression.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_gain"])
        * count_allocation
        * np.tanh(count_difference / count_scale)
    )

    # Reliability is represented as evidence orthogonal to ordinary tallying.
    # Centering ensures that a common shift of all communicated validities
    # cannot masquerade as additional count evidence.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    absolute_support = float(np.sum(np.abs(signed_components)))

    # Coherence compares the net residual with all reliability evidence that
    # had to be identity-bound. It is one for uniformly directed support and
    # approaches zero when a fixed residual is embedded in opposing groups.
    # It contains no explicit count-margin, display-size, or cue-count term.
    if absolute_support > 1e-12:
        coherence = abs(reliability_residual) / absolute_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    # Communicated profiles with meaningful reliability contrast make identity
    # bindings more distinguishable. This is a global property of the stated
    # profile, not a trial-level overload proxy.
    profile_mean = float(np.mean(validities))
    profile_spread = float(np.sqrt(np.mean((validities - profile_mean) ** 2)))
    profile_range = float(np.max(validities) - np.min(validities))
    if profile_range > 1e-12:
        profile_contrast = np.clip(2.0 * profile_spread / profile_range, 0.0, 1.0)
    else:
        profile_contrast = 0.0

    coherence_power = float(parameters["coherence_power"])
    coherent_access = coherence ** coherence_power
    profile_factor = (
        1.0
        + float(parameters["profile_contrast_gain"])
        * profile_contrast
    )

    # A nonzero floor permits reliability sensitivity even for diffuse fields,
    # while the binding term produces selective attenuation when opposing
    # reliability contributions proliferate.
    reliability_gain = (
        float(parameters["reliability_floor"])
        + float(parameters["coherent_binding_gain"])
        * coherent_access
        * profile_factor
    )
    reliability_signal = (
        reliability_allocation
        * reliability_gain
        * reliability_residual
    )

    # The independently summarized channels meet only after their respective
    # transformations. Precision is stable within subject and introduces
    # persistent heterogeneity in the expression of both summaries.
    markedness_evidence = float(parameters["response_precision"]) * (
        count_signal + reliability_signal
    )

    # Positive markedness indicates that A carries more adverse evidence and
    # therefore favors B under the dominant adverse-mark interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Semantic polarity is stable across trials. The alternative beneficial
    # interpretation reverses the complete markedness representation and thus
    # preserves exact A/B reversal symmetry.
    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs

    lapse = float(parameters["lapse_rate"])
    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
    return probs / total

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
- reliability_style: [0.0, 1.0]
- count_gain: [1.9, 3.0]
- count_scale: [2.8, 4.8]
- reliability_floor: [0.08, 0.30]
- coherent_binding_gain: [0.25, 0.78]
- coherence_power: [0.65, 1.65]
- profile_contrast_gain: [0.0, 0.45]
- response_precision: [0.68, 1.48]
- adverse_polarity_confidence: [0.80, 1.0]
- lapse_rate: [0.0, 0.12]

`rationale`: The model directly addresses the prior mechanistic failures by making count and reliability genuinely different summaries. Count evidence is a bounded function of signed margin alone. Consequently, adding any number of mutually canceling cues leaves it unchanged, predicting the count invariance required by Experiments 2, 11, and 14 while avoiding pi_8's diagnostic-load compression. Gradual tanh saturation preserves strong large-margin choices in Experiments 3 and 10 but limits the excessive linear margin separation seen for pi_4 in Experiment 8 and supports a moderate trajectory in Experiment 12. Reliability modulation is independently identified: coherence is the absolute residual divided by the absolute reliability support, so two trials can have identical residuals, margins, diagnostic counts, and display sizes while differing in internal cancellation. This predicts positive attenuation in Experiment 13 and weaker reliability influence in internally opposing conditions such as Experiment 1. Coherent high-versus-low profiles retain substantial reliability sensitivity, supporting Experiment 11, while the continuously varying coherence and profile terms produce intermediate rather than all-or-none reliability slopes in Experiments 6–9 and 13. There is no feature-count term, generic overload variable, capacity threshold, or late-load knee. Broad style and precision ranges create stable subject heterogeneity, while centered reliability evidence, global semantic reversal, and symmetric lapse mixing preserve A/B reversal symmetry.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is mechanistically faithful to the prescribed Channel-Specific Coherence Integration family and captures its most diagnostic dissociation: Experiment 13 is essentially exact (0.0311 versus 0.0315), while the absence of generic count-load compression gives the approximately invariant pattern in Experiment 14 (-0.0029 versus -0.0264) and the correct classification in Experiment 2. Intermediate reliability effects are also good in Experiment 6 (0.0991 versus 0.1062) and reasonably close in Experiment 7. However, the fit is not yet experiment-invariant. Count-driven adverse-mark responding is systematically too weak in Experiments 1 (-0.2380 versus -0.4355), 3 (0.6133 versus 0.7367), 4 (-0.3079 versus -0.4825), 10 (0.8116 versus 0.8750), and 12 (0.0555 versus 0.0793). Experiment 8 is also substantially too negative (-64 versus -28), indicating that the model produces too large a small-versus-large-margin contrast in that reliability-conflict design. Reliability sensitivity is underexpressed in Experiments 5 (0.1179 versus 0.2306) and 11 (0.8029 versus 0.8442), although a global reliability increase would risk spoiling the excellent Experiment 6 fit and the already somewhat high Experiment 7 value. Experiment 9 retains an unwanted diagnostic-density trend (0.3635 versus -0.0407), suggesting that coherence attenuation is leaking into a profile where behavior should be nearly flat. Finally, the model usually generates much less between-subject variance than observed, especially in Experiments 1, 5, 12, and 13, so the promised stable heterogeneity is not yet adequately expressed.
Rationale: Keep the current mechanism but recalibrate its channels. For the count channel, jointly raise count strength and weaken premature saturation—for example, allow somewhat larger count_gain while shifting count_scale upward—rather than applying a generic precision increase. This should strengthen large-margin behavior in Experiments 3 and 10, increase the margin trajectory in Experiment 12, improve tally effects in Experiments 1 and 4, and reduce the excessive negative contrast in Experiment 8 by preserving more differentiation between margins 1 and 5. For reliability, do not raise the floor globally: instead modestly increase gain for coherent, high-contrast profiles through the profile-dependent binding term, targeting Experiments 5 and 11 while preserving Experiment 6 and avoiding further inflation in Experiment 7. The residual Experiment 9 trend calls for tightening how coherence is computed or gated so that attenuation responds to genuinely opposing reliability groups, as in Experiment 13, rather than to diagnostic density incidentally correlated with the current absolute-support ratio. Preserve the exact Experiment 13 result and the count invariance in Experiments 2 and 14 as hard constraints. Finally, broaden independently stable subject variation in count scale/gain and reliability binding parameters rather than mainly relying on shared response_precision; the present shared scaling produces overly homogeneous subject-level metrics despite the theory's heterogeneity claim.

**Outcome of this advice:** iter 1 candidate loss=0.0983 -> iter 2 candidate loss=0.1296 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Channel-Specific Coherence Integration family, but this iteration is worse overall and was correctly rejected by the accept gate (loss 0.1296 versus the accepted base's 0.0983). The previous recommendation to jointly increase count gain and scale was too aggressive: it modestly improved Experiments 3, 4, 10, and 12, but catastrophically reversed Experiment 8 from -64 to +120 when the human value is -28. The accompanying reliability recalibration also damaged a formerly excellent constraint: Experiment 6 moved from 0.0991, near the observed 0.1062, to 0.1665. Experiment 9's unwanted density-correlated effect nearly doubled from 0.3635 to 0.7439, and Experiment 7 moved farther above its small observed effect. Meanwhile, the changes did not solve Experiment 1, which remains much too weak (-0.2267 versus -0.4355), or Experiment 11, which remains slightly underexpressed. The core successes survive—Experiment 13 remains essentially exact, Experiment 2 is correct, and Experiment 14 remains near count-load invariance—but they do not compensate for the cross-experiment regressions. Between-subject variance also remains generally too low in several key experiments.
Rationale: Because the last count-strength/scale push was rejected, do not repeat a broad joint increase from the accepted base. Use a substantially smaller or intermediate count-channel adjustment, preferably by adding one bounded shape parameter to the count transform rather than simultaneously widening both gain and scale. Calibrate that parameter against Experiment 8 as a hard constraint: the margin-1 versus margin-5 contrast must move only from the accepted base's -64 toward -28, not cross through zero, while retaining modest improvements in Experiments 3, 4, 10, and 12. Experiment 1 likely cannot be repaired by count amplification alone without recreating the Experiment 8 failure, so avoid using global precision as a substitute. For reliability, retain the accepted base's floor and global profile-contrast gain; the attempted global strengthening spoiled Experiment 6. Instead, use a robust bimodality or band-concentration statistic to modulate only coherent binding gain for genuinely clustered validity profiles, targeting Experiments 5 and 11 while leaving the smooth-gradient Experiment 6 largely unchanged. Replace the rejected largest-gap/median-gap gate with a smooth concentration measure over validity bands or signed residual support; the current gate still lets diagnostic density leak into Experiment 9. Preserve Experiment 13, Experiment 2, and Experiment 14 as hard constraints. Any added heterogeneity should be channel-specific and modest, since widening shared effects risks further inflating Experiment 7 and the already excessive variance in Experiment 9.

**Outcome of this advice:** iter 2 candidate loss=0.1296 -> iter 3 candidate loss=0.1044 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate remains faithful to the prescribed Channel-Specific Coherence Integration family, retaining separate count and reliability summaries, bounded count accumulation, coherence-sensitive reliability binding, polarity stability, and A/B symmetry. However, it was rejected by the accept gate because its loss (0.1044) exceeds the accepted base's 0.0983. The new cubic count tail improved large-margin alignment in Experiment 3 (0.6833 versus 0.7367) and slightly improved Experiment 12, but it overshot the key Experiment 8 contrast: the accepted base was too negative at -64, whereas the current model crossed the target of -28 and reached +28. Thus the added tail produces too much margin-5 amplification relative to margin-1 evidence. It also did not materially repair Experiment 1 (-0.2288 versus -0.4355) or Experiment 4 (-0.3237 versus -0.4825), showing that their deficits are not simply insufficient large-margin count strength. Experiment 9 remains a major failure and worsened from the accepted base's 0.3635 to 0.7636 against an observed value near zero, indicating that reliability coherence or its interaction with fixed margin-one count evidence is still varying with diagnostic density where it should be approximately flat. The reliability channel otherwise retains important successes: Experiment 6 is nearly exact, Experiment 13 still shows the required attenuation, Experiment 14 remains near count-load invariance, and Experiment 2 is correct. Reliability effects remain too weak in Experiment 5, while between-subject variability is still generally too low in Experiments 1, 5, and 12.
Rationale: Build again from the unchanged accepted iter-1 base. The current cubic-tail range was rejected, so do not repeat it. If a count-shape adjustment is retained, use a much smaller intermediate deformation—roughly a narrow tail-shape range around 0.05–0.10 rather than 0.14–0.28—and require Experiment 8 to remain negative and near -28. Do not try to solve Experiments 1 and 4 through further global count gain, scale, precision, or polarity increases; the two rejected iterations show that broad count strengthening creates disproportionate Experiment 8 changes without fixing Experiment 1. The next substantive adjustment should instead address the reliability channel's Experiment 9 leakage while preserving the accepted base's Experiment 6 and 13 fits. The present coherence ratio, |residual|/absolute_support, is mechanically correlated with the number and arrangement of discriminating contributions. Gate its attenuation with a smooth profile- or signed-group-concentration statistic so that cancellation reduces access only when a fixed reliability residual is genuinely embedded in opposing validity bands, as in Experiment 13, rather than whenever additional diagnostic cues appear, as in Experiment 9. This should be a modulation of coherent binding gain, not a generic feature-count or load term. A profile-selective binding boost may then target the underexpressed Experiment 5 effect without raising the global reliability floor and spoiling Experiment 6 or inflating Experiment 7. Preserve Experiments 2, 6, 13, and 14 as hard constraints, and add only modest channel-specific heterogeneity rather than widening shared response precision.

**Outcome of this advice:** iter 3 candidate loss=0.1044 -> iter 4 candidate loss=0.1270 -> the gate marked it REJECTED.

### Iteration 4 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is faithful to the prescribed Channel-Specific Coherence Integration family, but it was rejected by the accept gate and substantially regresses from the accepted iter-1 base (loss 0.1270 versus 0.0983). The small cubic-tail intervention failed its intended test: Experiment 8 crossed from the accepted base's -64 past the observed -28 to +60. It also weakened rather than strengthened large-margin behavior in Experiments 3 (0.6100 versus 0.7367), 10 (0.8003 versus 0.8750), and 12 (0.0511 versus 0.0793). Thus even the reduced cubic range is too unstable and does not provide a reliable bounded-margin correction. Experiment 9 remains qualitatively wrong, with a positive diagnostic-density trend of 0.2658 when the observed effect is approximately zero (-0.0407). The candidate also continues to understate Experiments 1, 4, and 5, while producing substantially less between-subject variance than the data in several experiments. Its principal successes are the correct Experiment 2 classification, near-target Experiment 11, approximate count-load invariance in Experiment 14, and retention of positive reliability attenuation in Experiment 13, although Experiment 13 has now drifted below its formerly near-exact value. Experiment 6 is still acceptable but no longer as exact as in the accepted base.
Rationale: Build from the unchanged accepted iter-1 base and abandon the cubic tail. Both the larger cubic range and the supposedly conservative 0.05–0.10 range were rejected, so another tail-strength adjustment would repeat a failed direction. Keep the original bounded count transform for now rather than trying to repair Experiments 1 and 4 through global count gain, precision, or polarity changes; those deficits have persisted while count amplification has repeatedly destabilized Experiment 8.

The next iteration should actually implement the still-untested substantive correction to the reliability coherence calculation. Replace cue-level coherence, abs(sum(c_i))/sum(abs(c_i)), with coherence computed after smoothly pooling signed cue-validity contributions into a small fixed number of validity bands. For example, use soft assignments to three or four bins over the communicated validity range, sum signed contributions within each band, and calculate net residual divided by the absolute support of those band-level sums. Local positive and negative contributions among similarly reliable cues would then cancel before binding interference is assessed, reducing the spurious diagnostic-density trend in Experiment 9. Contributions that oppose one another across distinguishable reliability bands would remain incoherent, preserving the required attenuation in Experiments 1 and 13. The number of bands and bandwidth should be fixed or narrowly ranged and defined by validity position, not feature count or diagnostic load, thereby remaining inside the prescribed mechanism family.

Calibrate this band-level coherence with Experiments 6 and 13 as hard constraints: Experiment 6 should remain near 0.106 and Experiment 13 near 0.0315. It may also restore reliability expression in Experiment 5 without globally raising the reliability floor. Do not combine this change with another count-shape intervention, so its effect can be identified. Preserve exact count-margin invariance under added canceling cues and A/B reversal symmetry. Only after the Experiment 9 leakage is reduced should modest channel-specific heterogeneity be widened to address the consistently underestimated variances.

**Outcome of this advice:** iter 4 candidate loss=0.1270 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Channel-Specific Coherence Integration theory: binary counts and cue reliabilities are encoded by distinct cognitive channels before being combined. The count channel rapidly computes the difference between option-level numbers of marked ratings. It depends only on that signed margin, so adding mutually canceling cues does not dilute count evidence. Its response is smooth and bounded, reflecting saturation of an approximate magnitude representation rather than unlimited linear accumulation. The reliability channel instead binds discriminating cue identities to communicated validities after removing the validity component redundant with count. Before coherence is assessed, signed reliability contributions are smoothly pooled within a small, fixed set of validity bands. Opposing contributions from similarly reliable cues cancel within a common binding unit, whereas opposition across distinguishable reliability bands remains costly. Reliability gain therefore declines when a residual is distributed across internally opposing validity groups, but not merely because more diagnostic cues are displayed. Communicated profile contrast further affects the accessibility of these bindings. Stable subject differences govern relative allocation to the two channels, validity-band resolution, and response precision. A stable semantic-polarity belief determines whether marks are interpreted as adverse or beneficial, and a small lapse process captures stimulus-independent choices.",
  "predict": "def predict(parameters, state, history):\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Channel-Specific Coherence Integration expects shape \"\n            f\"(2, n_features); got {stim.shape}.\"\n        )\n\n    a, b = stim[0], stim[1]\n    n_features = int(stim.shape[1])\n    validities = np.asarray(parameters[\"validities\"], dtype=float)\n    if validities.ndim != 1 or validities.size != n_features:\n        raise ValueError(\n            f\"validities length {validities.size} != n_features {n_features}.\"\n        )\n\n    difference = a - b\n    count_difference = float(np.sum(difference))\n\n    # Stable integration style allocates expression strength between two\n    # separately computed channels. The smooth transformation allows broad\n    # subject heterogeneity without imposing discrete strategy classes.\n    raw_style = float(parameters[\"reliability_style\"])\n    style = 0.5 + 0.5 * np.tanh(2.2 * (2.0 * raw_style - 1.0)) / np.tanh(2.2)\n    count_allocation = 1.30 - 0.60 * style\n    reliability_allocation = 0.35 + 1.35 * style\n\n    # The count channel reads only the option-level mark difference. Hence\n    # mutually canceling added cues have exactly no effect. Saturation is\n    # gradual and smooth rather than a hard margin rule or load-dependent\n    # divisive compression.\n    count_scale = float(parameters[\"count_scale\"])\n    count_signal = (\n        float(parameters[\"count_gain\"])\n        * count_allocation\n        * np.tanh(count_difference / count_scale)\n    )\n\n    # Reliability is represented as evidence orthogonal to ordinary tallying.\n    # Centering ensures that a common shift of all communicated validities\n    # cannot masquerade as additional count evidence.\n    centered_validities = validities - float(np.mean(validities))\n    signed_components = centered_validities * difference\n    reliability_residual = float(np.sum(signed_components))\n\n    # Coherence is evaluated after smoothly pooling contributions into four\n    # fixed validity bands. Nearby cue identities therefore share a binding\n    # unit and can cancel locally before interference is assessed. Opposition\n    # across distinguishable validity bands remains represented. The bands\n    # are defined over validity position, never by cue count or display size.\n    validity_min = float(np.min(validities))\n    validity_range = float(np.max(validities) - validity_min)\n    if validity_range > 1e-12:\n        validity_position = (validities - validity_min) / validity_range\n        band_centers = np.linspace(0.0, 1.0, 4)\n        bandwidth = float(parameters[\"validity_bandwidth\"])\n        band_weights = np.exp(\n            -0.5\n            * ((validity_position[:, None] - band_centers[None, :]) / bandwidth) ** 2\n        )\n        band_weights /= np.sum(band_weights, axis=1, keepdims=True)\n        band_components = np.sum(\n            signed_components[:, None] * band_weights,\n            axis=0,\n        )\n        absolute_support = float(np.sum(np.abs(band_components)))\n    else:\n        absolute_support = 0.0\n\n    # Coherence compares the net residual with reliability support remaining\n    # after within-band cancellation. It is one for uniformly directed support\n    # and approaches zero when distinguishable validity groups oppose one\n    # another. It contains no count-margin, display-size, or cue-count term.\n    if absolute_support > 1e-12:\n        coherence = abs(reliability_residual) / absolute_support\n    else:\n        coherence = 0.0\n    coherence = float(np.clip(coherence, 0.0, 1.0))\n\n    # Communicated profiles with meaningful reliability contrast make identity\n    # bindings more distinguishable. This is a global property of the stated\n    # profile, not a trial-level overload proxy.\n    profile_mean = float(np.mean(validities))\n    profile_spread = float(np.sqrt(np.mean((validities - profile_mean) ** 2)))\n    profile_range = float(np.max(validities) - np.min(validities))\n    if profile_range > 1e-12:\n        profile_contrast = np.clip(2.0 * profile_spread / profile_range, 0.0, 1.0)\n    else:\n        profile_contrast = 0.0\n\n    coherence_power = float(parameters[\"coherence_power\"])\n    coherent_access = coherence ** coherence_power\n    profile_factor = (\n        1.0\n        + float(parameters[\"profile_contrast_gain\"])\n        * profile_contrast\n    )\n\n    # A nonzero floor permits reliability sensitivity even for diffuse fields,\n    # while the binding term produces selective attenuation when opposing\n    # reliability contributions proliferate across distinct validity bands.\n    reliability_gain = (\n        float(parameters[\"reliability_floor\"])\n        + float(parameters[\"coherent_binding_gain\"])\n        * coherent_access\n        * profile_factor\n    )\n    reliability_signal = (\n        reliability_allocation\n        * reliability_gain\n        * reliability_residual\n    )\n\n    # The independently summarized channels meet only after their respective\n    # transformations. Precision is stable within subject and introduces\n    # persistent heterogeneity in the expression of both summaries.\n    markedness_evidence = float(parameters[\"response_precision\"]) * (\n        count_signal + reliability_signal\n    )\n\n    # Positive markedness indicates that A carries more adverse evidence and\n    # therefore favors B under the dominant adverse-mark interpretation.\n    adverse_logits = np.array(\n        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float\n    )\n    adverse_logits -= np.max(adverse_logits)\n    adverse_probs = np.exp(adverse_logits)\n    adverse_probs /= adverse_probs.sum()\n\n    # Semantic polarity is stable across trials. The alternative beneficial\n    # interpretation reverses the complete markedness representation and thus\n    # preserves exact A/B reversal symmetry.\n    beneficial_probs = adverse_probs[::-1]\n    polarity = float(parameters[\"adverse_polarity_confidence\"])\n    core_probs = polarity * adverse_probs + (1.0 - polarity) * beneficial_probs\n\n    lapse = float(parameters[\"lapse_rate\"])\n    probs = (1.0 - lapse) * core_probs + lapse * np.array([0.5, 0.5])\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=float)\n    return probs / total",
  "policy": "def policy(probs):\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))",
  "parameters": {
    "validities": "validities",
    "reliability_style": "[0.0, 1.0]",
    "count_gain": "[1.9, 3.0]",
    "count_scale": "[2.8, 4.8]",
    "reliability_floor": "[0.08, 0.30]",
    "coherent_binding_gain": "[0.25, 0.78]",
    "coherence_power": "[0.65, 1.65]",
    "validity_bandwidth": "[0.18, 0.26]",
    "profile_contrast_gain": "[0.0, 0.45]",
    "response_precision": "[0.68, 1.48]",
    "adverse_polarity_confidence": "[0.80, 1.0]",
    "lapse_rate": "[0.0, 0.12]"
  },
  "rationale": "This is a minimal edit to the accepted iter-1 model. The count transform, channel allocations, reliability residual, profile modulation, polarity process, precision, and lapse process are unchanged. Only the coherence denominator is changed: signed cue contributions are first softly pooled into four validity-position bands, with a narrowly varying bandwidth. This directly implements the critic's still-untested diagnosis. In profiles like Experiment 9, positive and negative contributions from similarly reliable neighboring cues cancel within bands, preventing diagnostic density from mechanically lowering coherence. In Experiments 1 and 13, opposition between distinguishable high- and low-validity groups remains in separate band totals and therefore continues to attenuate reliability binding. The normalized soft assignments preserve the original residual exactly, and the calculation contains no feature-count, diagnostic-count, count-margin, or generic overload term. Leaving the bounded count channel untouched avoids repeating the rejected cubic-tail and global count-strength interventions, while preserving exact invariance to added count-canceling cues and A/B reversal symmetry."
}
```

## Usage

```json
{
  "input_tokens": 48956,
  "output_tokens": 2806
}
```
