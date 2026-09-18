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
- THEORY 1 = `pi_4`
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a new theory of reliability-segregated consensus and conflict-triggered hierarchy protection. The theory should posit a stable highest-validity lexical anchor plus a separate, graded representation of numerical consensus. Consensus strength should depend continuously on normalized tally margin, allowing the strong margin contrast in Experiment 2, rather than being sign-normalized as in pi_4. Crucially, advertised validity should not monotonically facilitate the majority route as in pi_6. When an anti-leader coalition contains several individually diagnostic experts, the recognized high-reliability conflict should trigger scrutiny and renewed deference to the top-ranked expert; this produces the negative coalition-validity slope in Experiment 1. By contrast, approximately equivalent or low-diagnosticity lower cues should be encoded as fungible votes, so a conspicuous numerical majority can override the leader, especially in dense or structurally simple blocks, accounting for Experiments 2 and 8. Block density and recent display statistics may alter the accessibility of consensus encoding, but they should modulate rather than determine arbitration. The new account should therefore predict lexical behavior in sparse or reliability-heterogeneous settings, graded majority effects when many cues are readily compressed into votes, and hierarchy protection when strong lower experts create explicit reliability conflict. A quantitative implementation could use a persistent latent task set with lexical and graded-consensus evidence, plus an inhibitory diagnostic-conflict term based on the number and strength of anti-leader cues. It should avoid cue-identity learning without feedback, constrain stable subject heterogeneity so it does not create pi_6's excessive variance in Experiment 5, and be jointly calibrated against all ten metrics rather than tuned primarily to the dense-margin experiment.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_6` (overall score: 0.526)

**Description**
Contextual Margin Arbitration theory proposes that people maintain two competing representations of binary-feature comparisons. The advertised-rank route retrieves the most valid discriminating expert and favors its option. The conflict-compression route represents all discriminating experts as a directional tally while retaining the tally's graded normalized magnitude. Accessibility of compression is jointly controlled by a slowly persistent estimate of block density, the current normalized majority margin, a recency-weighted trace of recent margins, and the aggregate advertised-validity support for the current majority. Dense blocks make compression chronically accessible, whereas sparse blocks require a clearly suprathreshold majority to overcome rank processing. Density and margin provide compensatory routes to compression, so their overlap does not produce a superadditive accessibility boost. Aggregate validity shifts route accessibility and evidence continuously. A smoothly saturating diagnosticity representation preserves unusually extreme advertised-validity dominance without turning it into a categorical veto: dense context strongly reduces validity opposition, whereas margin-driven release is weaker and leaves a residual that grows continuously with the extremity of anti-majority support. Compression uses a mildly convex transform of normalized tally magnitude, making conspicuously large majorities more persuasive than small anti-leader coalitions while retaining continuous evidence. Repetition of particular cue identities has no independent learning effect because choices receive no correctness feedback. Stable individual differences in contextual thresholds, margin thresholds, gating sensitivities, evidence precision, and lapses generate heterogeneous intermediate behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contextual Margin Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Diagnosticity weights express advertised reliability. Smooth saturation
    # preserves the special extremity of a near-perfect expert while ensuring
    # that no finite advertised validity becomes a logical veto.
    v = np.clip(validities, 0.500001, 0.999999)
    raw_diagnosticity = np.log(v / (1.0 - v))
    validity_saturation = float(parameters["validity_saturation"])
    diagnosticity = validity_saturation * np.tanh(
        raw_diagnosticity / validity_saturation
    )

    memory_span = float(parameters["context_memory_span"])
    alpha = 1.0 / max(memory_span, 1.0)
    density_trace = float(parameters["density_prior"])
    margin_trace = float(parameters["margin_prior"])

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    # Density and margin traces are learned from displays alone. Cue identities
    # and previous responses do not enter because there is no outcome feedback.
    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        d_i = np.sign(a_i - b_i)
        k_i = int(np.count_nonzero(d_i))
        density_i = float(k_i) / float(max(n_features, 1))
        margin_i = (
            abs(float(np.sum(d_i))) / float(k_i) if k_i > 0 else 0.0
        )
        density_trace += alpha * (density_i - density_trace)
        margin_trace += alpha * (margin_i - margin_trace)

    directions = np.sign(stim[0] - stim[1])
    discriminating = np.flatnonzero(directions != 0.0)
    n_discriminating = int(discriminating.size)
    current_density = float(n_discriminating) / float(max(n_features, 1))
    raw_tally = float(np.sum(directions))
    current_margin = (
        abs(raw_tally) / float(n_discriminating)
        if n_discriminating > 0 else 0.0
    )

    # The current display is incorporated weakly into the persistent context.
    context_density = density_trace + alpha * (current_density - density_trace)
    recent_margin = margin_trace + alpha * (current_margin - margin_trace)

    # Advertised-rank evidence comes from the most valid discriminating cue.
    lexical_delta = 0.0
    if n_discriminating > 0:
        cue_order = np.argsort(-validities, kind="stable")
        for cue in cue_order:
            if directions[cue] != 0.0:
                lexical_delta = float(directions[cue])
                break

    # Signed aggregate validity support lies in [-1, 1]. Positive values favor
    # A and negative values favor B. Smoothly saturated weights retain graded
    # information about exceptionally dominant advertised experts.
    if n_discriminating > 0:
        active_weights = diagnosticity[discriminating]
        denom = float(np.sum(active_weights))
        if denom > 1e-12:
            validity_direction = float(
                np.dot(active_weights, directions[discriminating]) / denom
            )
        else:
            validity_direction = 0.0
    else:
        validity_direction = 0.0

    tally_sign = float(np.sign(raw_tally))
    validity_for_majority = tally_sign * validity_direction

    # Density remains an asymmetric contextual bonus. A sharper, individually
    # varying soft hinge makes margins below threshold contribute negligibly.
    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )
    margin_offset = current_margin - float(parameters["margin_threshold"])
    recent_offset = recent_margin - float(parameters["margin_threshold"])
    smoothness = float(parameters["margin_smoothness"])
    margin_excess = float(np.logaddexp(0.0, smoothness * margin_offset) / smoothness)
    recent_excess = float(np.logaddexp(0.0, smoothness * recent_offset) / smoothness)

    # Opposing advertised-validity support is released strongly by a genuinely
    # dense context but only weakly by momentary margin salience. A nonzero
    # attenuation floor preserves graded residual validity opposition without
    # allowing it to become a categorical veto.
    positive_validity = max(validity_for_majority, 0.0)
    negative_validity = min(validity_for_majority, 0.0)
    release_floor = float(parameters["validity_release_floor"])
    release_exponent = (
        float(parameters["density_validity_release"]) * density_excess
        + float(parameters["margin_validity_release"]) * margin_excess
    )
    negative_attenuation = release_floor + (1.0 - release_floor) * np.exp(
        -release_exponent
    )
    effective_validity_access = (
        positive_validity + negative_validity * negative_attenuation
    )

    # Density and current margin are compensatory routes to compression. The
    # overlap correction makes their combination approximately max/noisy-OR
    # like instead of granting a superadditive bonus when both are salient.
    density_drive = (
        float(parameters["density_sensitivity"]) * density_excess
    )
    margin_drive = (
        float(parameters["margin_sensitivity"]) * margin_excess
    )
    overlap_correction = (
        float(parameters["density_margin_interaction"])
        * min(density_drive, margin_drive)
    )

    gate_logit = (
        float(parameters["base_compression_access"])
        + density_drive
        + margin_drive
        - overlap_correction
        + float(parameters["recent_margin_sensitivity"]) * recent_excess
        + float(parameters["validity_access_bias"]) * effective_validity_access
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    validity_evidence_gain = float(parameters["validity_evidence_gain"])
    rank_delta = lexical_delta + validity_evidence_gain * validity_direction

    # Carry the same salience-dependent release into compressed evidence. Once
    # a majority representation is active, opposing validity remains a graded
    # residual bias instead of re-entering as a full anti-majority safeguard.
    if tally_sign != 0.0:
        compressed_validity_direction = (
            tally_sign * effective_validity_access
        )
    else:
        compressed_validity_direction = validity_direction

    # Preserve continuous normalized tally magnitude. Mild convexity separates
    # conspicuously large majorities from small anti-leader margins without
    # introducing dependence on the experiment's number of active experts.
    transformed_margin = current_margin ** float(
        parameters["tally_margin_exponent"]
    )
    compressed_delta = (
        tally_sign
        * transformed_margin
        * float(parameters["tally_evidence_scale"])
        + validity_evidence_gain * compressed_validity_direction
    )

    rank_sensitivity = float(parameters["rank_sensitivity"])
    tally_sensitivity = float(parameters["tally_sensitivity"])

    def binary_probs(delta, sensitivity):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_rank = binary_probs(rank_delta, rank_sensitivity)
    p_compressed = binary_probs(compressed_delta, tally_sensitivity)
    p_core = (
        (1.0 - compression_probability) * p_rank
        + compression_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- validity_saturation: [5.0, 7.0]
- context_memory_span: [14.0, 26.0]
- density_prior: [0.48, 0.58]
- margin_prior: [0.10, 0.22]
- density_threshold: [0.68, 0.71]
- margin_threshold: [0.24, 0.28]
- margin_smoothness: [45.0, 70.0]
- base_compression_access: [-2.8, -2.0]
- density_sensitivity: [30.0, 40.0]
- margin_sensitivity: [30.0, 42.0]
- recent_margin_sensitivity: [0.5, 1.6]
- density_margin_interaction: [0.75, 1.0]
- validity_access_bias: [2.0, 4.0]
- density_validity_release: [10.0, 16.0]
- margin_validity_release: [1.0, 3.0]
- validity_release_floor: [0.20, 0.40]
- validity_evidence_gain: [0.15, 0.65]
- tally_margin_exponent: [1.05, 1.15]
- tally_evidence_scale: [4.8, 5.4]
- rank_sensitivity: [1.8, 3.2]
- tally_sensitivity: [0.70, 0.90]
- lapse: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.3328 (var=0.0017)
- Experiment 2: real=0.3433 (var=0.0123) vs this=-0.1700 (var=0.0303)
- Experiment 3: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.1536)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.6739 (var=0.0057)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.5434 (var=0.0026)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.5779 (var=0.0042)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.5474 (var=0.0043)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.1853 (var=0.0081)
- Experiment 9: real=-0.1996 (var=0.0064) vs this=0.0516 (var=0.0017)
- Experiment 10: real=0.3168 (var=0.0102) vs this=0.3067 (var=0.0079)


---

### `pi_4` (overall score: 0.441)

**Description**
Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ecological Encoding Hysteresis expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Reconstruct the slowly accumulated block ecology. Each observation is
    # the fraction of cues that discriminate in a comparison; its complement
    # is tie prevalence. A prior prevents one unusual early trial from
    # immediately replacing the initial ordered-search task set.
    density_sum = 0.0
    n_observed = 0
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        density_sum += float(np.mean(a_i != b_i))
        n_observed += 1

    # Viewing the current display supplies another ecological observation,
    # but its effect shrinks as evidence about the block accumulates.
    current_density = float(np.mean(stim[0] != stim[1]))
    density_sum += current_density
    n_observed += 1

    prior_density = float(parameters["context_prior"])
    prior_strength = float(parameters["prior_strength"])
    context_density = (
        prior_strength * prior_density + density_sum
    ) / (prior_strength + float(n_observed))

    # Saturated ecologies trigger a compressed relational encoding. The
    # ceiling allows some ordered processing to survive even in dense blocks.
    threshold = float(parameters["context_threshold"])
    steepness = float(parameters["context_steepness"])
    gate_logit = steepness * (context_density - threshold)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    compression_activation = 1.0 / (1.0 + np.exp(-gate_logit))
    pooling_probability = (
        float(parameters["pooling_ceiling"]) * compression_activation
    )

    directions = np.sign(stim[0] - stim[1])

    # Ordered encoding: consult cues by advertised validity and stop at the
    # first discrimination.
    lexical_delta = 0.0
    cue_order = np.argsort(-validities, kind="stable")
    for cue in cue_order:
        if directions[cue] != 0.0:
            lexical_delta = float(directions[cue])
            break

    # Compressed encoding: retain all cue directions but discard fine-grained
    # rank and magnitude information. Sign normalization puts its response
    # scale on the same footing as the lexical representation.
    pooled_delta = float(np.sign(np.sum(directions)))

    sensitivity = float(parameters["evidence_sensitivity"])

    def choice_probs(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_ordered = choice_probs(lexical_delta)
    p_compressed = choice_probs(pooled_delta)
    p_core = (
        (1.0 - pooling_probability) * p_ordered
        + pooling_probability * p_compressed
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- context_prior: [0.50, 0.60]
- prior_strength: [1.0, 3.0]
- context_threshold: [0.68, 0.71]
- context_steepness: [35.0, 55.0]
- pooling_ceiling: [0.68, 0.76]
- evidence_sensitivity: [1.6, 3.0]
- lapse: [0.0, 0.08]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.0091 (var=0.0018)
- Experiment 2: real=0.3433 (var=0.0123) vs this=0.2442 (var=0.0221)
- Experiment 3: real=1.0000 (var=0.0000) vs this=1.0000 (var=0.0000)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.5411 (var=0.0106)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.6725 (var=0.0026)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.1183 (var=0.0028)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.6617 (var=0.0029)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.1520 (var=0.0100)
- Experiment 9: real=-0.1996 (var=0.0064) vs this=-0.0002 (var=0.0023)
- Experiment 10: real=0.3168 (var=0.0102) vs this=0.0268 (var=0.0100)


---

### `pi_3` (overall score: 0.334)

**Description**
Rank-Budget Arbitration theory proposes that people initially represent advertised cue validities as an ordered hierarchy and try to make a lexicographic choice using the highest-ranked discriminating cue. Reliable access to that hierarchy is capacity-limited, however. The probability that rank retrieval fails increases smoothly with the observable representational load imposed by the number of available cues and the number of cues that must be compared on the current trial. When rank retrieval succeeds, choice approximates Take The Best and lower-ranked tally direction has little influence. When retrieval fails, cue ranks are compressed into a coarse common category, and the person pools the directions of all discriminating cues with approximately equal weight. Strategy arbitration occurs anew on every trial rather than assigning a person permanently to one heuristic. Individual capacity determines the transition location, gate steepness determines how abrupt it is, evidence sensitivity controls response reliability, and lapses produce occasional uninformed choices. Thus, lexicographic and tally-like rules are limiting regimes of a single adaptive representational mechanism.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Rank-Budget Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    diff = stim[0] - stim[1]
    directions = np.sign(diff)
    discriminating = np.flatnonzero(directions != 0)

    # Successful rank retrieval: use only the most valid discriminating cue.
    if discriminating.size == 0:
        lexical_delta = 0.0
    else:
        cue_order = np.argsort(-validities, kind="stable")
        lexical_delta = 0.0
        for cue in cue_order:
            if directions[cue] != 0:
                lexical_delta = float(directions[cue])
                break

    # Failed rank retrieval: cue ranks are compressed and directions are pooled.
    # The sign transform prevents larger tallies from receiving mechanically
    # greater response precision than lexicographic evidence.
    pooled_delta = float(np.sign(np.sum(directions)))

    # Besides cue-set size, simultaneous discriminations add comparison load.
    comparison_fraction = float(discriminating.size) / float(max(n_features, 1))
    observable_load = float(n_features) + 0.25 * comparison_fraction

    capacity = float(parameters["capacity"])
    steepness = float(parameters["gate_steepness"])
    gate_logit = steepness * (observable_load - capacity)
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    pooling_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    sensitivity = float(parameters["evidence_sensitivity"])

    def binary_softmax(delta):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    p_lexical = binary_softmax(lexical_delta)
    p_pooled = binary_softmax(pooled_delta)
    p_core = ((1.0 - pooling_probability) * p_lexical
              + pooling_probability * p_pooled)

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- capacity: [9.7, 9.9]
- gate_steepness: [2.0, 2.4]
- evidence_sensitivity: [1.0, 5.0]
- lapse: [0.0, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.0011 (var=0.0014) vs this=0.0019 (var=0.0011)
- Experiment 2: real=0.3433 (var=0.0123) vs this=0.3675 (var=0.0173)
- Experiment 3: real=1.0000 (var=0.0000) vs this=-1.0000 (var=0.0196)
- Experiment 4: real=0.5200 (var=0.0305) vs this=0.5706 (var=0.0125)
- Experiment 5: real=0.5773 (var=0.0012) vs this=0.2905 (var=0.0033)
- Experiment 6: real=0.8300 (var=0.0061) vs this=0.6185 (var=0.0035)
- Experiment 7: real=0.6404 (var=0.0014) vs this=0.8793 (var=0.0044)
- Experiment 8: real=0.1631 (var=0.0139) vs this=0.1195 (var=0.0100)
- Experiment 9: real=-0.1996 (var=0.0064) vs this=0.0020 (var=0.0015)
- Experiment 10: real=0.3168 (var=0.0102) vs this=-0.0022 (var=0.0102)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.2756 -> ACCEPTED
- iter 2: loss=0.3578 -> REJECTED
- iter 3: loss=0.2587 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 3 at loss=0.2587 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
### Experiment 1
**Design**
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Recode each response as agreement with Take-The-Best (TTB), then
    # estimate the within-data slope of that agreement on the direction
    # of the equal-weight tally relative to the TTB winner.
    x = []
    y = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b

        # Features are ordered from highest to lowest advertised validity.
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue
        first = int(discriminating[0])
        ttb_winner = 0 if diff[first] > 0 else 1

        a_wins = int(np.sum(diff > 0))
        b_wins = int(np.sum(diff < 0))
        tally_margin_for_ttb = (
            a_wins - b_wins if ttb_winner == 0 else b_wins - a_wins
        )

        # -1: tally opposes TTB; 0: tally ties; +1: tally supports TTB.
        x.append(float(np.sign(tally_margin_for_ttb)))
        y.append(float(int(response) == ttb_winner))

    if len(x) < 2:
        return float("nan")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return float("nan")

    return float(np.dot(xc, y - np.mean(y)) / denom)
```

**Observed (real) value:** 0.0011 (var=0.0014)
**Previous candidate values (this loop):**
  - iter 1: 0.1571 (var=0.0018) (Δ vs real +0.1560)
  - iter 2: 0.2130 (var=0.0017) (Δ vs real +0.2120)
  - iter 3 (most recent): 0.1974 (var=0.0018) (Δ vs real +0.1963)
**Other theories' values on this metric (for reference):**
- pi_1: -0.0095 (var=0.0016)
- pi_2: 0.3604 (var=0.0118)
- pi_3: 0.0019 (var=0.0011)
- pi_4: 0.0091 (var=0.0018)
- pi_5: 0.3640 (var=0.0028)
- pi_6: 0.3328 (var=0.0017)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    contrasts = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        cue_directions = np.sign(a - b)

        # +1 denotes an A winner and -1 a B winner.
        tally_winner = float(np.sign(cue_directions.sum()))

        discriminating = np.flatnonzero(cue_directions != 0)
        if tally_winner == 0 or discriminating.size == 0:
            continue
        ttb_winner = float(cue_directions[discriminating[0]])

        # Retain only trials on which the theories make opposite predictions.
        if tally_winner != ttb_winner:
            chosen_side = 1.0 if int(row["response"]) == 0 else -1.0
            contrasts.append(chosen_side * tally_winner)

    if len(contrasts) == 0:
        return float("nan")
    return float(np.mean(contrasts))
```

**Observed (real) value:** 0.3433 (var=0.0123)
**Previous candidate values (this loop):**
  - iter 1: -0.4758 (var=0.0233) (Δ vs real -0.8192)
  - iter 2: -0.4517 (var=0.0240) (Δ vs real -0.7950)
  - iter 3 (most recent): -0.0575 (var=0.0212) (Δ vs real -0.4008)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7633 (var=0.0349)
- pi_1: -0.6767 (var=0.0457)
- pi_3: 0.3675 (var=0.0173)
- pi_4: 0.2442 (var=0.0221)
- pi_5: 0.3233 (var=0.0234)
- pi_6: -0.1700 (var=0.0303)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Polarity of preference for the highest-validity discriminating cue."""
    aligned = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are supplied in descending validity order.
        first = int(discriminating[0])
        lexical_winner = 0 if diff[first] > 0 else 1
        aligned.append(float(int(row["response"]) == lexical_winner))

    if len(aligned) == 0:
        return 0.0

    rate = float(np.mean(aligned))
    if rate > 0.5:
        return 1.0
    if rate < 0.5:
        return -1.0
    return 0.0
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Previous candidate values (this loop):**
  - iter 1: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 2: 1.0000 (var=0.0000) (Δ vs real +0.0000)
  - iter 3 (most recent): 1.0000 (var=0.0000) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 1.0000 (var=0.0000)
- pi_3: -1.0000 (var=0.0196)
- pi_2: -1.0000 (var=0.0784)
- pi_4: 1.0000 (var=0.0000)
- pi_5: -1.0000 (var=0.0000)
- pi_6: 1.0000 (var=0.1536)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for a, b, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        cue_directions = np.sign(a_arr - b_arr)
        pooled_direction = float(np.sign(cue_directions.sum()))

        # Tie-tally trials carry no pooled directional prediction.
        if pooled_direction == 0.0:
            continue

        # Positive choice direction denotes choosing A; negative denotes B.
        choice_direction = 1.0 if int(response) == 0 else -1.0
        scores.append(choice_direction * pooled_direction)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.5200 (var=0.0305)
**Previous candidate values (this loop):**
  - iter 1: 0.5522 (var=0.0113) (Δ vs real +0.0322)
  - iter 2: 0.1250 (var=0.0086) (Δ vs real -0.3950)
  - iter 3 (most recent): 0.3022 (var=0.0237) (Δ vs real -0.2178)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5706 (var=0.0125)
- pi_1: 0.0139 (var=0.0047)
- pi_2: 0.7294 (var=0.0327)
- pi_4: 0.5411 (var=0.0106)
- pi_5: 0.4717 (var=0.0088)
- pi_6: 0.6739 (var=0.0057)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    d = data.copy()
    # Trial position is reconstructed separately for each subject because rows
    # are supplied in within-subject trial order.
    d["trial_position"] = d.groupby("subject_id", sort=False).cumcount()

    a_sum = d["option_a_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )
    b_sum = d["option_b_ratings"].apply(
        lambda x: float(np.sum(np.asarray(x, dtype=float)))
    )

    # All designed trials have an unambiguous cue-majority winner. Determine
    # it from the displayed ratings so the score remains invariant to A/B
    # reversal and does not depend on memorized trial labels.
    majority_response = np.where(b_sum > a_sum, 1, 0)
    d["majority_choice"] = (
        d["response"].to_numpy(dtype=int) == majority_response
    ).astype(float)

    # Omit the first eight observations, during which the advocated model is
    # acquiring the dense-block context. The competitor is stationary, while
    # the advocated model should predominantly use compressed pooling after
    # this burn-in.
    late = d[d["trial_position"] >= 8]
    if len(late) == 0:
        late = d

    # Averaging subject-level rates makes the pooled-data definition agree
    # with the quantity recomputed on each individual subject.
    subject_rates = late.groupby("subject_id", sort=False)["majority_choice"].mean()
    if len(subject_rates) == 0:
        return float("nan")
    return float(subject_rates.mean())
```

**Observed (real) value:** 0.5773 (var=0.0012)
**Previous candidate values (this loop):**
  - iter 1: 0.2995 (var=0.0041) (Δ vs real -0.2777)
  - iter 2: 0.3302 (var=0.0026) (Δ vs real -0.2470)
  - iter 3 (most recent): 0.2216 (var=0.0020) (Δ vs real -0.3557)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6725 (var=0.0026)
- pi_3: 0.2905 (var=0.0033)
- pi_1: 0.1370 (var=0.0071)
- pi_2: 0.8314 (var=0.0161)
- pi_5: 0.4523 (var=0.0026)
- pi_6: 0.5434 (var=0.0026)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices favoring the equal-weight cue-majority winner."""
    majority_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
        b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
        if a.size != b.size or a.size == 0:
            continue

        majority_direction = float(np.sum(np.sign(a - b)))
        if majority_direction == 0.0:
            continue

        majority_response = 0 if majority_direction > 0.0 else 1
        response = int(row["response"])
        if response in (0, 1):
            majority_choices.append(float(response == majority_response))

    if len(majority_choices) == 0:
        return float("nan")
    return float(np.mean(majority_choices))
```

**Observed (real) value:** 0.8300 (var=0.0061)
**Previous candidate values (this loop):**
  - iter 1: 0.2677 (var=0.0033) (Δ vs real -0.5623)
  - iter 2: 0.3835 (var=0.0028) (Δ vs real -0.4465)
  - iter 3 (most recent): 0.4542 (var=0.0039) (Δ vs real -0.3758)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0035)
- pi_4: 0.1183 (var=0.0028)
- pi_1: 0.1504 (var=0.0063)
- pi_2: 0.8581 (var=0.0081)
- pi_5: 0.8013 (var=0.0034)
- pi_6: 0.5779 (var=0.0042)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    # Omit only the brief initialization period in which Ecological Encoding
    # Hysteresis is still updating its estimate of the block ecology.
    if 'subject_id' in data.columns:
        serial_position = data.groupby('subject_id', sort=False).cumcount().to_numpy()
    else:
        serial_position = np.arange(len(data))
    late = serial_position >= 4

    scores = []
    for keep, a_cell, b_cell, response in zip(
        late,
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        if not keep or pd.isna(response):
            continue
        a = np.asarray(a_cell, dtype=float).reshape(-1)
        b = np.asarray(b_cell, dtype=float).reshape(-1)
        if a.size == 0 or a.size != b.size:
            continue
        difference = float(np.sum(a) - np.sum(b))
        if difference == 0.0 or not np.isfinite(difference):
            continue
        majority_response = 0 if difference > 0.0 else 1
        scores.append(float(int(response) == majority_response))

    if len(scores) == 0:
        return float('nan')
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** 0.6404 (var=0.0014)
**Previous candidate values (this loop):**
  - iter 1: 0.7220 (var=0.0039) (Δ vs real +0.0815)
  - iter 2: 0.7393 (var=0.0021) (Δ vs real +0.0989)
  - iter 3 (most recent): 0.7230 (var=0.0034) (Δ vs real +0.0826)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6617 (var=0.0029)
- pi_5: 0.0711 (var=0.0013)
- pi_1: 0.1522 (var=0.0140)
- pi_2: 0.8537 (var=0.0083)
- pi_3: 0.8793 (var=0.0044)
- pi_6: 0.5474 (var=0.0043)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Acquisition-weighted capture by the recurring anti-leader coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        coalition_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float).reshape(-1)
            b = np.asarray(row["option_b_ratings"], dtype=float).reshape(-1)
            if a.size == 0 or b.size == 0 or a[0] == b[0]:
                continue

            # Expert 1 is the highest-validity cue. The coalition winner is
            # the option opposed to the option favored by that expert.
            leader_response = 0 if a[0] > b[0] else 1
            response = int(row["response"])
            coalition_choices.append(float(response != leader_response))

        n = len(coalition_choices)
        if n == 0:
            continue

        choices = np.asarray(coalition_choices, dtype=float)
        early_n = min(8, n)
        late_n = min(32, n)
        early_rate = float(np.mean(choices[:early_n]))
        late_rate = float(np.mean(choices[-late_n:]))

        # Late coalition capture plus a modest acquisition bonus.
        scores.append(late_rate + 0.25 * (late_rate - early_rate))

    if not scores:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.1631 (var=0.0139)
**Previous candidate values (this loop):**
  - iter 1: 0.1336 (var=0.0104) (Δ vs real -0.0295)
  - iter 2: 0.3578 (var=0.0142) (Δ vs real +0.1947)
  - iter 3 (most recent): 0.1406 (var=0.0070) (Δ vs real -0.0225)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9923 (var=0.0035)
- pi_4: 0.1520 (var=0.0100)
- pi_1: 0.1422 (var=0.0186)
- pi_2: 0.8414 (var=0.0147)
- pi_3: 0.1195 (var=0.0100)
- pi_6: 0.1853 (var=0.0081)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.90, 0.84, 0.79, 0.74, 0.69, 0.64, 0.60, 0.57, 0.54, 0.52], dtype=float)
    diagnosticity = np.log(validities / (1.0 - validities))

    def subject_slope(df):
        df = df.reset_index(drop=True)
        n = len(df)
        if n < 4:
            return np.nan

        coalition_scores = np.empty(n, dtype=float)
        majority_b = np.empty(n, dtype=float)
        majority_choice = np.empty(n, dtype=float)

        for i in range(n):
            a = np.asarray(df.loc[i, "option_a_ratings"], dtype=float)
            b = np.asarray(df.loc[i, "option_b_ratings"], dtype=float)
            directions = np.sign(a - b)
            active = directions != 0.0
            tally = float(np.sum(directions))
            majority_sign = float(np.sign(tally))

            if majority_sign == 0.0 or not np.any(active):
                coalition_scores[i] = 0.0
                majority_b[i] = 0.5
                majority_choice[i] = 0.0
                continue

            denom = float(np.sum(diagnosticity[active]))
            coalition_scores[i] = majority_sign * float(
                np.dot(diagnosticity[active], directions[active])
            ) / denom

            maj_b = 1.0 if majority_sign < 0.0 else 0.0
            majority_b[i] = maj_b
            majority_choice[i] = 1.0 if float(df.loc[i, "response"]) == maj_b else 0.0

        rounded = np.round(coalition_scores, 10)
        levels = np.sort(np.unique(rounded))
        if levels.size < 2:
            return np.nan

        rank_map = {float(v): float(j) for j, v in enumerate(levels)}
        x = np.array([rank_map[float(v)] for v in rounded], dtype=float)
        x -= (levels.size - 1.0) / 2.0

        # Remove assignment to short serial-position strata, as well as any
        # incidental association with which screen side contains the majority.
        bins = np.arange(n, dtype=int) // 6
        unique_bins = np.unique(bins)
        design = np.zeros((n, unique_bins.size + 1), dtype=float)
        for j, value in enumerate(unique_bins):
            design[:, j] = (bins == value).astype(float)
        design[:, -1] = majority_b

        beta = np.linalg.lstsq(design, x, rcond=None)[0]
        x_resid = x - design.dot(beta)
        denom = float(np.dot(x_resid, x_resid))
        if denom <= 1e-12:
            return np.nan

        return float(np.dot(x_resid, majority_choice) / denom)

    if "subject_id" in data.columns:
        values = []
        for _, subject_df in data.groupby("subject_id", sort=False):
            value = subject_slope(subject_df)
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    return float(subject_slope(data))

```

**Observed (real) value:** -0.1996 (var=0.0064)
**Previous candidate values (this loop):**
  - iter 1: -0.1244 (var=0.0022) (Δ vs real +0.0752)
  - iter 2: -0.1343 (var=0.0015) (Δ vs real +0.0653)
  - iter 3 (most recent): -0.1543 (var=0.0018) (Δ vs real +0.0453)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0002 (var=0.0023)
- pi_6: 0.0516 (var=0.0017)
- pi_1: -0.0045 (var=0.0013)
- pi_2: -0.0036 (var=0.0011)
- pi_3: 0.0020 (var=0.0015)
- pi_5: 0.2398 (var=0.0019)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    # Code whether the response followed the highest-validity cue (cue 1).
    lexical = []
    small_margin = []
    for a_cell, b_cell, response in zip(
        data['option_a_ratings'], data['option_b_ratings'], data['response']
    ):
        a = np.asarray(a_cell, dtype=float).reshape(-1)
        b = np.asarray(b_cell, dtype=float).reshape(-1)
        directions = np.sign(a - b)

        cue1_direction = directions[0]
        r = int(response)
        followed_cue1 = ((cue1_direction > 0 and r == 0) or
                         (cue1_direction < 0 and r == 1))
        lexical.append(float(followed_cue1))

        n_active = int(np.count_nonzero(directions))
        margin = (abs(float(np.sum(directions))) / float(n_active)
                  if n_active > 0 else 0.0)
        # Separates the subthreshold 6-4 majority from all larger majorities.
        small_margin.append(float(margin < 0.3))

    work = pd.DataFrame({
        'subject_id': data['subject_id'].to_numpy(),
        'lexical': np.asarray(lexical, dtype=float),
        'small': np.asarray(small_margin, dtype=float)
    })

    # Rows are in trial order within subject. Twelve-trial strata remove the
    # large common acquisition trajectory while retaining six repetitions per
    # stimulus and adequate observations from both margin classes per stratum.
    work['position'] = work.groupby('subject_id', sort=False).cumcount()
    work['time_bin'] = (work['position'] // 12).astype(int)
    bin_mean = work.groupby(
        ['subject_id', 'time_bin'], sort=False
    )['small'].transform('mean')
    residual_small = work['small'].to_numpy(dtype=float) - bin_mean.to_numpy(dtype=float)
    y = work['lexical'].to_numpy(dtype=float)

    denominator = float(np.dot(residual_small, residual_small))
    if denominator <= 1e-12:
        x = work['small'].to_numpy(dtype=float)
        x = x - float(np.mean(x))
        denominator = float(np.dot(x, x))
        if denominator <= 1e-12:
            return 0.0
        return float(np.dot(x, y) / denominator)

    return float(np.dot(residual_small, y) / denominator)
```

**Observed (real) value:** 0.3168 (var=0.0102)
**Previous candidate values (this loop):**
  - iter 1: 0.2342 (var=0.0093) (Δ vs real -0.0826)
  - iter 2: 0.1723 (var=0.0065) (Δ vs real -0.1445)
  - iter 3 (most recent): 0.4105 (var=0.0123) (Δ vs real +0.0937)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3067 (var=0.0079)
- pi_4: 0.0268 (var=0.0100)
- pi_1: 0.0091 (var=0.0063)
- pi_2: 0.0179 (var=0.0059)
- pi_3: -0.0022 (var=0.0102)
- pi_5: 0.0194 (var=0.0036)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Reliability-Segregated Conflict Arbitration theory proposes that people preserve the advertised validity hierarchy as a stable lexical representation while separately constructing a graded representation of numerical consensus. The lexical representation retrieves the highest-validity discriminating expert. The consensus representation encodes the signed tally and its continuous normalized margin rather than merely its sign. A slowly updated task-set state, inferred from the density and margins of recently viewed displays, regulates how accessible consensus encoding is. Reliability structure determines whether lower experts are treated as fungible votes or as individually diagnostic sources. Similar-reliability experts and uniformly weak lower experts are readily compressed into votes, whereas a heterogeneous hierarchy preserves their reliability identities. When a numerical majority opposes the lexical leader, hierarchy protection depends specifically on whether diagnostic support is concentrated in a coherent high-validity prefix of the lower-cue ranking. Such a coalition is recognized as an explicit reliability conflict and renews lexical scrutiny; dispersed or low-ranked support is more readily treated as fungible consensus. Ordinary consensus accessibility is additionally nonlinear in normalized tally margin, allowing conspicuous majorities to become salient without globally increasing pooling. Consequently, increasing the reliability of a coherent anti-leader coalition can reduce majority following even though increasing its numerical margin ordinarily increases majority following. Because no outcomes are provided, the model learns only block-level display statistics and never learns cue identities or cue-response associations.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Segregated Conflict Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(parameters["n_features"])
    if stim.shape[1] != n_features:
        raise ValueError(
            f"Stimulus has {stim.shape[1]} features but n_features={n_features}."
        )

    validities = np.asarray(parameters["validities"], dtype=float).reshape(-1)
    if validities.size != n_features:
        raise ValueError("validities must contain one value per feature.")

    # Advertised reliabilities are represented as diagnostic log odds. No
    # feedback-dependent estimates or learned cue-identity terms are used.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))

    # Reliability segregation describes whether the cue set naturally divides
    # into distinguishable reliability levels. Nearly equivalent cues have low
    # segregation even if all of them have fairly high advertised validity.
    mean_diagnosticity = float(np.mean(diagnosticity))
    spread = float(np.std(diagnosticity))
    segregation_scale = float(parameters["segregation_scale"])
    segregation = float(np.tanh(spread / max(segregation_scale, 1e-9)))

    # Individually diagnostic cues retain their reliability identity. The
    # smooth criterion avoids imposing an arbitrary categorical validity tier.
    cutoff = float(parameters["diagnostic_cutoff"])
    sharpness = float(parameters["diagnostic_sharpness"])
    z = np.clip(sharpness * (diagnosticity - cutoff), -60.0, 60.0)
    individual_diagnosticity = 1.0 / (1.0 + np.exp(-z))

    # Lower cues are fungible if their reliability levels are approximately
    # equivalent or if they are mostly below the diagnosticity criterion.
    top_index = int(np.argmax(validities))
    if n_features > 1:
        lower_mask = np.ones(n_features, dtype=bool)
        lower_mask[top_index] = False
        lower_strength = float(np.mean(individual_diagnosticity[lower_mask]))
    else:
        lower_strength = 0.0
    fungibility = float(np.clip(1.0 - segregation * lower_strength, 0.0, 1.0))

    # Reconstruct a persistent task-set state from displays alone. Density and
    # normalized tally margin are exponentially integrated across the block.
    span = max(float(parameters["context_memory_span"]), 1.0)
    alpha = 1.0 / span
    density_trace = float(parameters["density_prior"])
    margin_trace = float(parameters["margin_prior"])

    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))

    for i in range(n_past):
        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)
        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)
        if a_i.size != n_features or b_i.size != n_features:
            continue
        d_i = np.sign(a_i - b_i)
        active_i = int(np.count_nonzero(d_i))
        density_i = float(active_i) / float(max(n_features, 1))
        margin_i = (
            abs(float(np.sum(d_i))) / float(active_i)
            if active_i > 0 else 0.0
        )
        density_trace += alpha * (density_i - density_trace)
        margin_trace += alpha * (margin_i - margin_trace)

    directions = np.sign(stim[0] - stim[1])
    discriminating = np.flatnonzero(directions != 0.0)
    n_active = int(discriminating.size)
    density = float(n_active) / float(max(n_features, 1))
    raw_tally = float(np.sum(directions))
    tally_sign = float(np.sign(raw_tally))
    margin = abs(raw_tally) / float(n_active) if n_active > 0 else 0.0

    # The current display transiently moves, but does not replace, the latent
    # task set. This lets ecology modulate arbitration without determining it.
    context_density = density_trace + 0.35 * (density - density_trace)
    recent_margin = margin_trace + 0.35 * (margin - margin_trace)

    # Stable lexical representation: retrieve the most valid currently
    # discriminating expert. Stable sorting respects instructed rank on ties.
    lexical_direction = 0.0
    if n_active > 0:
        order = np.argsort(-validities, kind="stable")
        for cue in order:
            if directions[cue] != 0.0:
                lexical_direction = float(directions[cue])
                break

    # Diagnostic conflict is protected when anti-leader support forms a
    # coherent high-validity prefix of the lower-cue hierarchy. Equally large
    # coalitions dispersed among low-ranked cues remain compressible votes.
    conflict = 0.0
    if tally_sign != 0.0 and lexical_direction != 0.0 and tally_sign != lexical_direction:
        coalition = np.flatnonzero(directions == tally_sign)
        coalition_count = int(coalition.size)
        if coalition_count > 0:
            lower_order = [
                int(cue) for cue in np.argsort(-validities, kind="stable")
                if int(cue) != top_index
            ]
            lower_rank = np.full(n_features, float(n_features), dtype=float)
            for rank, cue in enumerate(lower_order):
                lower_rank[cue] = float(rank)

            prefix_span = max(float(parameters["ranked_prefix_span"]), 1e-9)
            coalition_weights = np.exp(-lower_rank[coalition] / prefix_span)
            prefix_strength = float(np.dot(
                individual_diagnosticity[coalition], coalition_weights
            ))
            ideal_weights = np.exp(
                -np.arange(coalition_count, dtype=float) / prefix_span
            )
            ideal_mass = float(np.sum(ideal_weights))
            prefix_coherence = (
                prefix_strength / ideal_mass if ideal_mass > 1e-12 else 0.0
            )
            prefix_coherence = float(np.clip(prefix_coherence, 0.0, 1.0))

            count_scale = max(float(parameters["conflict_count_scale"]), 1e-9)
            count_salience = 1.0 - np.exp(-float(coalition_count) / count_scale)
            conflict = float(segregation * prefix_coherence * count_salience)

    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )

    # Consensus salience changes nonlinearly around a psychologically visible
    # normalized margin. Fungibility modulates, but does not veto, this signal.
    margin_transition = float(parameters["margin_transition"])
    margin_steepness = float(parameters["margin_steepness"])
    margin_logit = np.clip(
        margin_steepness * (margin - margin_transition), -60.0, 60.0
    )
    visible_margin = 1.0 / (1.0 + np.exp(-margin_logit))
    effective_margin_access = visible_margin * (0.65 + 0.35 * fungibility)

    # Consensus access receives independent contributions from the persistent
    # dense-task set, visible graded margin, recent margin statistics, and cue
    # fungibility. Coherent ranked conflict selectively inhibits access.
    gate_logit = (
        float(parameters["base_consensus_access"])
        + float(parameters["density_access_gain"]) * density_excess
        + float(parameters["margin_access_gain"]) * effective_margin_access
        + float(parameters["recent_margin_gain"]) * recent_margin
        + float(parameters["fungibility_gain"]) * fungibility
        - float(parameters["conflict_inhibition"]) * conflict
    )
    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))
    consensus_probability = 1.0 / (1.0 + np.exp(-gate_logit))

    # Conflict also renews attention to the instructed hierarchy. This is a
    # protection response, not negative weighting of reliable coalition cues.
    lexical_evidence = lexical_direction * (
        1.0 + float(parameters["hierarchy_protection_gain"]) * conflict
    )

    # Numerical consensus retains continuous normalized margin. Thus a 9-1
    # split is represented more strongly than a 6-4 split even though both
    # have the same tally sign.
    if tally_sign == 0.0:
        consensus_evidence = 0.0
    else:
        transformed_margin = margin ** float(parameters["consensus_margin_exponent"])
        consensus_evidence = (
            tally_sign
            * float(parameters["consensus_evidence_scale"])
            * transformed_margin
        )

    lexical_sensitivity = float(parameters["lexical_sensitivity"])
    consensus_sensitivity = float(parameters["consensus_sensitivity"])

    def binary_probs(delta, sensitivity):
        logits = np.array([
            0.5 * sensitivity * delta,
            -0.5 * sensitivity * delta
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    p_lexical = binary_probs(lexical_evidence, lexical_sensitivity)
    p_consensus = binary_probs(consensus_evidence, consensus_sensitivity)
    p_core = (
        (1.0 - consensus_probability) * p_lexical
        + consensus_probability * p_consensus
    )

    lapse = float(parameters["lapse"])
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- n_features: n_features
- validities: validities
- segregation_scale: [0.45, 0.65]
- diagnostic_cutoff: [0.75, 0.95]
- diagnostic_sharpness: [6.0, 9.0]
- context_memory_span: [16.0, 22.0]
- density_prior: [0.48, 0.54]
- margin_prior: [0.15, 0.21]
- density_threshold: [0.56, 0.62]
- base_consensus_access: [-2.0, -1.7]
- density_access_gain: [5.5, 6.5]
- margin_access_gain: [3.5, 4.5]
- recent_margin_gain: [0.5, 1.0]
- fungibility_gain: [1.3, 1.8]
- ranked_prefix_span: [0.9, 1.3]
- conflict_count_scale: [1.5, 2.2]
- conflict_inhibition: [5.0, 7.0]
- hierarchy_protection_gain: [0.5, 1.0]
- margin_transition: [0.23, 0.26]
- margin_steepness: [15.0, 19.0]
- consensus_margin_exponent: [0.90, 1.10]
- consensus_evidence_scale: [4.5, 5.5]
- lexical_sensitivity: [2.9, 3.3]
- consensus_sensitivity: [1.4, 1.8]
- lapse: [0.02, 0.06]

`rationale`: This is a minimal edit of the accepted candidate. The global reliability representation, task-set traces, lexical and consensus routes, conflict inhibition, and graded consensus evidence are retained. Only two selectivity mechanisms change. First, conflict strength now depends on diagnostic anti-leader support occupying a short high-validity lower-cue prefix. This preserves hierarchy protection for recurring structured coalitions such as Experiment 8 and for reliability conflicts in Experiments 1 and 9, while releasing consensus when support is dispersed across lower ranks, as required in Experiments 2, 5, and 6. Second, the linear margin contribution to consensus access is replaced by a steep but continuous sigmoid around a normalized margin of approximately 0.25, weakly modulated by fungibility. This selectively boosts conspicuous 5-3 and larger majorities, while changing small dense-block margins and Experiment 7 much less. Consensus evidence itself remains continuously graded, preserving the Experiment 10 contrast. A small narrowing and increase of lexical sensitivity makes ordinary lexical responses more stable, helping flatten Experiment 1 and protect Experiment 8 without introducing new subject-level variance. No cue-identity or response learning was added.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate captures several important signatures of the prescribed family: lexical dominance is correct in Experiment 3; anti-leader capture is close in Experiment 8; the coalition-validity effect has the correct negative sign in Experiment 9; and the graded-margin effect is reasonably close in Experiment 10. Experiment 4 is also well fit. However, it fails the central experiment-invariance requirement. It remains much too lexical in Experiments 2, 5, and 6: Experiment 2 reverses the observed majority preference (candidate -0.4758 versus observed +0.3433), while majority choice is severely underestimated in Experiment 5 (0.2995 versus 0.5773) and Experiment 6 (0.2677 versus 0.8300). At the same time, tally influence is too strong in Experiment 1 (slope 0.1571 versus approximately zero), and majority use is somewhat excessive in Experiment 7 (0.7220 versus 0.6404). Thus, a uniform increase in consensus access would not solve the pattern; consensus must become more selectively available for fungible lower cues while hierarchy protection becomes more selective and effective for genuinely diagnostic coalitions.
Rationale: The accepted candidate is a useful in-family base, but its conflict detector appears miscalibrated. In particular, global `segregation` is computed across all cues, including the top lexical anchor. A large leader-to-lower-cue gap can therefore make the entire set look segregated, lower `fungibility`, and activate conflict inhibition even when the lower cues themselves are approximately equivalent or weak—the exact situations that should support pooling in Experiments 2, 5, and especially 6. Revise segregation/fungibility to characterize the lower-cue set separately from the stable leader. Conflict should depend on the count or summed strength of anti-leader lower cues that individually exceed the diagnostic threshold, together with coalition-specific reliability heterogeneity, rather than global spread times the coalition's mean diagnosticity. This should release consensus for weak or mutually similar coalitions without abandoning hierarchy protection. Once that selectivity is fixed, modestly increase fungibility-driven consensus access or lower the density threshold so dense and structurally simple blocks reach the observed majority rates. Do not simply raise baseline or margin access globally, because that would worsen Experiments 1 and 7. For Experiment 1, make the qualified diagnostic-conflict response more sharply count-sensitive or let qualified conflict inhibit the margin contribution, so increasing an opposing diagnostic coalition does not leave the positive tally slope of 0.157. Preserve the current negative effect in Experiment 9 and the good lexical fits in Experiments 3 and 8. Finally, after correcting mean fit, narrow the context/access ranges if needed: between-subject variance remains too high in Experiments 2 and 5 relative to the data.

**Outcome of this advice:** iter 1 candidate loss=0.2756 -> iter 2 candidate loss=0.3578 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The current candidate is not an improvement over the accepted base: aggregate loss rose from 0.2756 to 0.3578, so the gate correctly rejected it. The revised lower-cue segregation rule produced only modest gains in the intended experiments—Experiment 2 remains directionally wrong (-0.4517 versus +0.3433), and majority choice remains far too low in Experiments 5 (0.3302 versus 0.5773) and 6 (0.3835 versus 0.8300). Those small gains came at substantial cost elsewhere. Experiment 4 deteriorated from a close 0.5522 to 0.1250 against 0.5200, Experiment 8's coalition capture rose from a close 0.1336 to 0.3578 against 0.1631, Experiment 1's excessive tally slope worsened from 0.1571 to 0.2130 against approximately zero, and Experiment 10's graded-margin signature weakened from 0.2342 to 0.1723 against 0.3168. Experiment 7 also remains somewhat too majority-oriented. The candidate still preserves lexical polarity in Experiment 3 and the negative coalition-validity slope in Experiment 9, but it does not achieve experiment-invariant arbitration.
Rationale: Because this iteration was rejected, build on the unchanged iteration-1 base rather than extending the rejected lower-cue-spread revision. My previous recommendation to define fungibility through lower-set or coalition-specific reliability heterogeneity did not help: it released consensus on the wrong displays, especially Experiment 8, while barely addressing Experiments 2, 5, and 6. Try a different in-family selectivity variable. Preserve the accepted base's conflict machinery and characterize protected conflict by reliability-ranked coalition coherence—for example, the concentration of anti-leader support within a short high-validity lower-cue prefix—rather than by the standard deviation of all lower cues or the within-coalition spread. This can protect the recurring high-ranked anti-leader configuration in Experiment 8 and structured diagnostic conflicts in Experiments 1 and 9 while treating dispersed or predominantly low-ranked support as fungible votes. Separately, make ordinary consensus access more nonlinear in normalized margin: use a steeper thresholded/saturating margin transformation or a margin-by-fungibility interaction so conspicuous majorities can dominate in Experiments 2, 5, and 6 without uniformly increasing baseline consensus. Keep diagnostic conflict as a direct lexical-protection term, so stronger reliable conflict can still suppress the tally effect in Experiment 1. Avoid the current combination in which conflict simultaneously suppresses the gate, suppresses margin, and amplifies lexical evidence; that triple action makes calibration brittle. A cleaner division is graded margin/fungibility controlling consensus access and one conflict-dependent hierarchy-protection term controlling lexical resistance. Preserve the accepted base's Experiment 4 and 8 settings, retain the successful negative sign in Experiment 9, and restore rather than weaken Experiment 10's margin contrast. Parameter ranges should remain narrow because the present variances are mostly acceptable; the principal problem is mean-pattern selectivity, not heterogeneity.

**Outcome of this advice:** iter 2 candidate loss=0.3578 -> iter 3 candidate loss=0.2587 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent — address this)
Verdict: regenerate
Interpretation: The accepted revision is a genuine improvement and validates the prior recommendation: ranked-prefix conflict preserved the close Experiment 8 fit and strengthened the negative coalition-validity effect in Experiment 9, while nonlinear margin access moved Experiment 2 substantially toward the observed majority preference and improved Experiment 6. Experiment 3 remains exact. Nevertheless, the model is not yet experiment-invariant. It remains too lexical in Experiment 2 (-0.0575 versus 0.3433), Experiment 5 (0.2216 versus 0.5773), Experiment 6 (0.4542 versus 0.8300), and Experiment 4 (0.3022 versus 0.5200). Conversely, tally influence in Experiment 1 remains much too positive (0.1974 versus 0.0011), Experiment 7 is somewhat too majority-oriented (0.7230 versus 0.6404), and the graded-margin contrast now overshoots in Experiment 10 (0.4105 versus 0.3168). The largest improvement therefore came from the correct in-family selectivity idea, but its threshold and conflict implementation are still miscalibrated.
Rationale: Continue from this accepted ranked-coherence/nonlinear-margin solution rather than revisiting the rejected lower-set heterogeneity proposal. First fix a likely structural defect in the conflict calculation: `prefix_coherence` divides by an `ideal_mass` that grows with total coalition size. Adding low-ranked anti-leader cues can therefore dilute an already coherent diagnostic prefix, causing hierarchy protection to weaken precisely as the opposing tally grows. That is a plausible source of Experiment 1's large positive tally slope. Normalize prefix strength against a fixed prefix capacity, or compute saturating occupancy of the first few diagnostic lower ranks independently of additional low-ranked supporters. Qualified conflict should then remain stable or increase when extra votes are added. This should flatten Experiment 1 while preserving Experiment 8 and the successful negative sign in Experiment 9; avoid a global increase in conflict inhibition, which would suppress the needed pooling elsewhere. Second, broaden and modestly lower the currently narrow margin-access transition rather than increasing baseline consensus. The steep threshold at roughly 0.25 strongly discounts Experiment 5's recurring 5-4 margins and many moderate-majority trials in Experiments 4 and 6. A somewhat lower transition with reduced steepness should raise consensus for these moderate margins, further improve Experiment 2, and reduce the excessive categorical separation now seen in Experiment 10. Keep the change modest because Experiment 7 is already too majority-oriented. If additional selectivity is needed, let fungibility scale the broadened subthreshold part of margin access, while conspicuous margins retain access across cue structures. Finally, after improving the means, narrow the margin/access parameter ranges: Experiment 2's between-subject variance remains appreciably above the observed value. The prescribed reliability-segregated lexical-plus-graded-consensus family should be retained; the remaining failures concern conflict normalization and access calibration, not the mechanism family.

**Outcome of this advice:** iter 3 candidate loss=0.2587 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


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
  "theory": "Reliability-Segregated Conflict Arbitration theory proposes that people preserve the advertised validity hierarchy as a stable lexical representation while separately constructing a graded representation of numerical consensus. The lexical representation retrieves the highest-validity discriminating expert. The consensus representation encodes the signed tally and its continuous normalized margin rather than merely its sign. A slowly updated task-set state, inferred from the density and margins of recently viewed displays, regulates how accessible consensus encoding is. Reliability structure determines whether lower experts are treated as fungible votes or as individually diagnostic sources. Similar-reliability experts and uniformly weak lower experts are readily compressed into votes, whereas a heterogeneous hierarchy preserves their reliability identities. When a numerical majority opposes the lexical leader, hierarchy protection depends specifically on whether diagnostic support occupies a fixed-capacity high-validity prefix of the lower-cue ranking. Occupancy of that protected prefix cannot be diluted by adding low-ranked supporters, so a recognized reliability conflict sustains lexical scrutiny as the opposing tally grows. Dispersed or low-ranked support is more readily treated as fungible consensus. Ordinary consensus accessibility is nonlinear but broadly graded in normalized tally margin, allowing moderate as well as conspicuous majorities to become salient without globally increasing pooling. Consequently, increasing the reliability of a coherent anti-leader coalition can reduce majority following even though increasing its numerical margin ordinarily increases majority following. Because no outcomes are provided, the model learns only block-level display statistics and never learns cue identities or cue-response associations.",
  "predict": "def predict(parameters, state, history) -> np.ndarray:\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"Reliability-Segregated Conflict Arbitration expects shape (2, n_features); got {stim.shape}.\"\n        )\n\n    n_features = int(parameters[\"n_features\"])\n    if stim.shape[1] != n_features:\n        raise ValueError(\n            f\"Stimulus has {stim.shape[1]} features but n_features={n_features}.\"\n        )\n\n    validities = np.asarray(parameters[\"validities\"], dtype=float).reshape(-1)\n    if validities.size != n_features:\n        raise ValueError(\"validities must contain one value per feature.\")\n\n    # Advertised reliabilities are represented as diagnostic log odds. No\n    # feedback-dependent estimates or learned cue-identity terms are used.\n    v = np.clip(validities, 0.500001, 0.999999)\n    diagnosticity = np.log(v / (1.0 - v))\n\n    # Reliability segregation describes whether the cue set naturally divides\n    # into distinguishable reliability levels. Nearly equivalent cues have low\n    # segregation even if all of them have fairly high advertised validity.\n    mean_diagnosticity = float(np.mean(diagnosticity))\n    spread = float(np.std(diagnosticity))\n    segregation_scale = float(parameters[\"segregation_scale\"])\n    segregation = float(np.tanh(spread / max(segregation_scale, 1e-9)))\n\n    # Individually diagnostic cues retain their reliability identity. The\n    # smooth criterion avoids imposing an arbitrary categorical validity tier.\n    cutoff = float(parameters[\"diagnostic_cutoff\"])\n    sharpness = float(parameters[\"diagnostic_sharpness\"])\n    z = np.clip(sharpness * (diagnosticity - cutoff), -60.0, 60.0)\n    individual_diagnosticity = 1.0 / (1.0 + np.exp(-z))\n\n    # Lower cues are fungible if their reliability levels are approximately\n    # equivalent or if they are mostly below the diagnosticity criterion.\n    top_index = int(np.argmax(validities))\n    if n_features > 1:\n        lower_mask = np.ones(n_features, dtype=bool)\n        lower_mask[top_index] = False\n        lower_strength = float(np.mean(individual_diagnosticity[lower_mask]))\n    else:\n        lower_strength = 0.0\n    fungibility = float(np.clip(1.0 - segregation * lower_strength, 0.0, 1.0))\n\n    # Reconstruct a persistent task-set state from displays alone. Density and\n    # normalized tally margin are exponentially integrated across the block.\n    span = max(float(parameters[\"context_memory_span\"]), 1.0)\n    alpha = 1.0 / span\n    density_trace = float(parameters[\"density_prior\"])\n    margin_trace = float(parameters[\"margin_prior\"])\n\n    past_a = history.get(\"option_a_ratings\", [])\n    past_b = history.get(\"option_b_ratings\", [])\n    n_past = min(len(past_a), len(past_b))\n\n    for i in range(n_past):\n        a_i = np.asarray(past_a[i], dtype=float).reshape(-1)\n        b_i = np.asarray(past_b[i], dtype=float).reshape(-1)\n        if a_i.size != n_features or b_i.size != n_features:\n            continue\n        d_i = np.sign(a_i - b_i)\n        active_i = int(np.count_nonzero(d_i))\n        density_i = float(active_i) / float(max(n_features, 1))\n        margin_i = (\n            abs(float(np.sum(d_i))) / float(active_i)\n            if active_i > 0 else 0.0\n        )\n        density_trace += alpha * (density_i - density_trace)\n        margin_trace += alpha * (margin_i - margin_trace)\n\n    directions = np.sign(stim[0] - stim[1])\n    discriminating = np.flatnonzero(directions != 0.0)\n    n_active = int(discriminating.size)\n    density = float(n_active) / float(max(n_features, 1))\n    raw_tally = float(np.sum(directions))\n    tally_sign = float(np.sign(raw_tally))\n    margin = abs(raw_tally) / float(n_active) if n_active > 0 else 0.0\n\n    # The current display transiently moves, but does not replace, the latent\n    # task set. This lets ecology modulate arbitration without determining it.\n    context_density = density_trace + 0.35 * (density - density_trace)\n    recent_margin = margin_trace + 0.35 * (margin - margin_trace)\n\n    # Stable lexical representation: retrieve the most valid currently\n    # discriminating expert. Stable sorting respects instructed rank on ties.\n    lexical_direction = 0.0\n    if n_active > 0:\n        order = np.argsort(-validities, kind=\"stable\")\n        for cue in order:\n            if directions[cue] != 0.0:\n                lexical_direction = float(directions[cue])\n                break\n\n    # Diagnostic conflict is protected when anti-leader support occupies a\n    # coherent high-validity prefix of the lower-cue hierarchy. Normalization\n    # uses the fixed capacity of that hierarchy, so extra low-ranked supporters\n    # cannot dilute an already occupied diagnostic prefix.\n    conflict = 0.0\n    if tally_sign != 0.0 and lexical_direction != 0.0 and tally_sign != lexical_direction:\n        coalition = np.flatnonzero(directions == tally_sign)\n        coalition_count = int(coalition.size)\n        if coalition_count > 0:\n            lower_order = [\n                int(cue) for cue in np.argsort(-validities, kind=\"stable\")\n                if int(cue) != top_index\n            ]\n            lower_rank = np.full(n_features, float(n_features), dtype=float)\n            for rank, cue in enumerate(lower_order):\n                lower_rank[cue] = float(rank)\n\n            prefix_span = max(float(parameters[\"ranked_prefix_span\"]), 1e-9)\n            coalition_weights = np.exp(-lower_rank[coalition] / prefix_span)\n            prefix_strength = float(np.dot(\n                individual_diagnosticity[coalition], coalition_weights\n            ))\n            fixed_prefix_capacity = float(np.sum(np.exp(\n                -np.arange(len(lower_order), dtype=float) / prefix_span\n            )))\n            prefix_coherence = (\n                prefix_strength / fixed_prefix_capacity\n                if fixed_prefix_capacity > 1e-12 else 0.0\n            )\n            prefix_coherence = float(np.clip(prefix_coherence, 0.0, 1.0))\n\n            count_scale = max(float(parameters[\"conflict_count_scale\"]), 1e-9)\n            count_salience = 1.0 - np.exp(-float(coalition_count) / count_scale)\n            conflict = float(segregation * prefix_coherence * count_salience)\n\n    density_excess = max(\n        context_density - float(parameters[\"density_threshold\"]), 0.0\n    )\n\n    # Consensus salience changes smoothly around a psychologically visible\n    # normalized margin. Fungibility modulates, but does not veto, this signal.\n    margin_transition = float(parameters[\"margin_transition\"])\n    margin_steepness = float(parameters[\"margin_steepness\"])\n    margin_logit = np.clip(\n        margin_steepness * (margin - margin_transition), -60.0, 60.0\n    )\n    visible_margin = 1.0 / (1.0 + np.exp(-margin_logit))\n    effective_margin_access = visible_margin * (0.65 + 0.35 * fungibility)\n\n    # Consensus access receives independent contributions from the persistent\n    # dense-task set, visible graded margin, recent margin statistics, and cue\n    # fungibility. Coherent ranked conflict selectively inhibits access.\n    gate_logit = (\n        float(parameters[\"base_consensus_access\"])\n        + float(parameters[\"density_access_gain\"]) * density_excess\n        + float(parameters[\"margin_access_gain\"]) * effective_margin_access\n        + float(parameters[\"recent_margin_gain\"]) * recent_margin\n        + float(parameters[\"fungibility_gain\"]) * fungibility\n        - float(parameters[\"conflict_inhibition\"]) * conflict\n    )\n    gate_logit = float(np.clip(gate_logit, -60.0, 60.0))\n    consensus_probability = 1.0 / (1.0 + np.exp(-gate_logit))\n\n    # Conflict also renews attention to the instructed hierarchy. This is a\n    # protection response, not negative weighting of reliable coalition cues.\n    lexical_evidence = lexical_direction * (\n        1.0 + float(parameters[\"hierarchy_protection_gain\"]) * conflict\n    )\n\n    # Numerical consensus retains continuous normalized margin. Thus a 9-1\n    # split is represented more strongly than a 6-4 split even though both\n    # have the same tally sign.\n    if tally_sign == 0.0:\n        consensus_evidence = 0.0\n    else:\n        transformed_margin = margin ** float(parameters[\"consensus_margin_exponent\"])\n        consensus_evidence = (\n            tally_sign\n            * float(parameters[\"consensus_evidence_scale\"])\n            * transformed_margin\n        )\n\n    lexical_sensitivity = float(parameters[\"lexical_sensitivity\"])\n    consensus_sensitivity = float(parameters[\"consensus_sensitivity\"])\n\n    def binary_probs(delta, sensitivity):\n        logits = np.array([\n            0.5 * sensitivity * delta,\n            -0.5 * sensitivity * delta\n        ], dtype=float)\n        logits -= np.max(logits)\n        p = np.exp(logits)\n        p /= p.sum()\n        return p\n\n    p_lexical = binary_probs(lexical_evidence, lexical_sensitivity)\n    p_consensus = binary_probs(consensus_evidence, consensus_sensitivity)\n    p_core = (\n        (1.0 - consensus_probability) * p_lexical\n        + consensus_probability * p_consensus\n    )\n\n    lapse = float(parameters[\"lapse\"])\n    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        return np.array([0.5, 0.5], dtype=np.float64)\n    return probs / total\n",
  "policy": "def policy(probs) -> int:\n    probs = np.asarray(probs, dtype=np.float64)\n    probs = np.clip(probs, 0.0, None)\n    total = float(probs.sum())\n    if not np.isfinite(total) or total <= 0.0:\n        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))\n    else:\n        probs /= total\n    return int(np.random.choice(len(probs), p=probs))\n",
  "parameters": {
    "n_features": "n_features",
    "validities": "validities",
    "segregation_scale": "[0.45, 0.65]",
    "diagnostic_cutoff": "[0.75, 0.95]",
    "diagnostic_sharpness": "[6.0, 9.0]",
    "context_memory_span": "[16.0, 22.0]",
    "density_prior": "[0.48, 0.54]",
    "margin_prior": "[0.15, 0.21]",
    "density_threshold": "[0.56, 0.62]",
    "base_consensus_access": "[-2.0, -1.7]",
    "density_access_gain": "[5.5, 6.5]",
    "margin_access_gain": "[3.5, 4.5]",
    "recent_margin_gain": "[0.5, 1.0]",
    "fungibility_gain": "[1.3, 1.8]",
    "ranked_prefix_span": "[0.9, 1.3]",
    "conflict_count_scale": "[1.5, 2.2]",
    "conflict_inhibition": "[5.0, 7.0]",
    "hierarchy_protection_gain": "[0.5, 1.0]",
    "margin_transition": "[0.18, 0.21]",
    "margin_steepness": "[9.0, 12.0]",
    "consensus_margin_exponent": "[0.90, 1.10]",
    "consensus_evidence_scale": "[4.5, 5.5]",
    "lexical_sensitivity": "[2.9, 3.3]",
    "consensus_sensitivity": "[1.4, 1.8]",
    "lapse": "[0.02, 0.06]"
  },
  "rationale": "This is a minimal edit of the accepted candidate. The first change repairs the diagnosed conflict-normalization defect: ranked-prefix strength is now divided by the fixed capacity of the complete lower-cue hierarchy rather than by a denominator that grows with coalition size. Adding low-ranked anti-leader votes therefore cannot dilute an already coherent diagnostic prefix. Qualified hierarchy protection should remain stable or strengthen as a coalition expands, reducing Experiment 1's excessive positive tally slope while preserving the protected coalition behavior in Experiments 8 and 9. The second change lowers and broadens the margin-access transition, from [0.23, 0.26] with steepness [15, 19] to [0.18, 0.21] with steepness [9, 12]. This gives moderate and 5-4 majorities appreciable consensus access in Experiments 2, 4, 5, and 6 without raising baseline access globally. The broader transition should also reduce the overly categorical small-versus-large margin contrast in Experiment 10. Fungibility continues to modulate margin access, limiting indiscriminate increases in majority use in Experiment 7. All other task-set, lexical, consensus-evidence, conflict, sensitivity, and lapse mechanisms are unchanged."
}
```

## Usage

```json
{
  "input_tokens": 31350,
  "output_tokens": 4010
}
```
