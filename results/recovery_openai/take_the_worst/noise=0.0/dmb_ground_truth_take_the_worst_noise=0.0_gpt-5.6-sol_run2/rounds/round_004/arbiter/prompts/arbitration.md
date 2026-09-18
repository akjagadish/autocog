# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
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


## THEORY 2 — pi_6
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=10):** [0.9, 0.84, 0.79, 0.74, 0.69, 0.64, 0.6, 0.57, 0.54, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  trial 13: A=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** Use 10 experts and hold all structural quantities used by Ecological Encoding Hysteresis constant across conditions. Every trial has exactly seven discriminating cues (density 0.70), a four-to-three majority, normalized majority margin 1/7, and Expert 1 as the highest-validity discriminator. In each base orientation Expert 1 favors A while four lower experts favor B, so ordered search favors A and equal-weight pooling favors B. The manipulation only permutes which lower-validity experts constitute the B-majority coalition. The four coalition conditions range from the four strongest lower experts supporting B to the four weakest experts supporting B; two common tie-value variants make distinct displays without changing any theoretical evidence. Ecological Encoding Hysteresis ignores lower-cue validity magnitudes after identifying the first discriminator and sign-compresses the tally. Because density, lexical direction, and pooled direction are identical on every trial, it predicts a common choice-probability trajectory across all coalition permutations, conditional on serial position, even though density 0.70 lies near its contextual transition. Contextual Margin Arbitration instead retains aggregate advertised-validity support. It therefore predicts a graded decline in B-majority choices as the B coalition moves from high- to low-validity experts: high-validity majority support increases compression accessibility and reinforces compressed B evidence, whereas low-validity majority support leaves strong residual opposition and strengthens the A-favoring rank route. The fixed, subthreshold margin of 1/7 prevents this gradient from being attributed to varying tally margins. Exact A/B reversals balance response side. Six repetitions of each of the 16 unique pairs yield 96 trials. The decisive analysis is the within-position coalition-validity slope: approximately zero under the advocated theory but substantial and monotonic under the competitor.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is the within-position slope of choosing the four-cue majority on the advertised-validity strength of that majority coalition. Coalition strength is computed from log-odds diagnosticity and converted to its four ordinal levels, avoiding excessive leverage from the strongest expert. Ecological Encoding Hysteresis discards lower-cue validity magnitudes once lexical and tally directions are fixed, so it predicts a slope near zero. Contextual Margin Arbitration retains aggregate validity support in both route accessibility and evidence, predicting a positive slope. Residualizing the coalition rank within six-trial serial strata removes the advocated theory's potentially steep context trajectory and substantially reduces randomization-induced between-subject variance. Residualization against majority side also removes incidental response-side imbalance. The pooled result is the mean subject slope, making the same estimand valid for pooled and single-subject inputs.

Source:
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


### RESULTS
- Predicted under pi_4 (simulated): -0.0002 (var=0.0023)
- Predicted under pi_6 (simulated): 0.0516 (var=0.0017)
- Observed on real data: -0.1996 (var=0.0064)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=10):** [0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  trial 11: A=[0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 12: A=[0, 1, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** Use 10 experts and make every comparison maximally dense: all 10 ratings discriminate, so both theories receive exactly the same density history (1.0) on every trial. Expert 1 (90%) always favors the lexical winner, Expert 2 (80%) always opposes it, and the remaining eight experts are advertised at chance validity (50%). Those chance-validity experts still contribute directions to Contextual Margin Arbitration's tally but contribute essentially no aggregate diagnosticity, allowing tally margin to vary while lexical evidence and validity-weighted evidence remain effectively fixed. In the eight base pairs, A is always the lexical winner and B is always the pooled-majority winner, but B's majority increases from 6-4 to 7-3, 8-2, and 9-1, corresponding to normalized margins of 0.20, 0.40, 0.60, and 0.80. Three, two, two, and one lower-cue permutations instantiate these levels; exact A/B reversals control side bias, giving 16 unique pairs and six repetitions each. Ecological Encoding Hysteresis sign-normalizes pooled evidence and uses only density to govern pooling. Conditional on serial position and orientation, it therefore predicts an exact plateau across all four majority margins: lexical direction, pooled direction, current density, and density history are identical. Contextual Margin Arbitration instead preserves graded normalized tally magnitude with a mildly convex transform. Once the uniformly dense context has made compression accessible, it predicts a steep monotonic decline in lexical-winner choices as the opposing majority grows from 6-4 to 9-1. It may also show earlier release for the larger margins, whereas the competitor's acquisition trajectory remains margin-invariant. The decisive analysis is the serial-position-adjusted slope of choosing the lexical winner against normalized majority margin, especially in later trials: strongly negative under the advocated theory and zero under the competing theory.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is the serial-position-stratified difference in probability of following the highest-validity expert between 6-4 displays (normalized margin 0.20) and all larger opposing majorities (margins 0.40–0.80). Contextual Margin Arbitration places its margin threshold between 0.20 and 0.40, so the 6-4 displays preferentially retain the lexical route while larger majorities strongly activate compression; it therefore predicts a positive contrast. Ecological Encoding Hysteresis uses sign-normalized pooled evidence and has identical density on every display, so it predicts no margin-class contrast. Residualizing the margin-class indicator within 12-trial subject-specific bins removes the pronounced common density-acquisition trajectory and reduces variance caused by random trial order. Pooling the three larger-margin classes also gives 36 versus 60 observations per subject, making the contrast substantially more stable than an endpoint-only slope.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.0268 (var=0.0100)
- Predicted under pi_6 (simulated): 0.3067 (var=0.0079)
- Observed on real data: 0.3168 (var=0.0102)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_4:** 0.0091 (var=0.0018)
**Predicted under pi_6:** 0.3328 (var=0.0017)

### Experiment 4
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
**Predicted under pi_4:** 0.2442 (var=0.0221)
**Predicted under pi_6:** -0.1700 (var=0.0303)

### Experiment 5
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
**Predicted under pi_4:** 1.0000 (var=0.0000)
**Predicted under pi_6:** 1.0000 (var=0.1536)

### Experiment 6
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
**Predicted under pi_4:** 0.5411 (var=0.0106)
**Predicted under pi_6:** 0.6739 (var=0.0057)

### Experiment 7
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
**Predicted under pi_4:** 0.6725 (var=0.0026)
**Predicted under pi_6:** 0.5434 (var=0.0026)

### Experiment 8
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
**Predicted under pi_4:** 0.1183 (var=0.0028)
**Predicted under pi_6:** 0.5779 (var=0.0042)

### Experiment 9
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
**Predicted under pi_4:** 0.6617 (var=0.0029)
**Predicted under pi_6:** 0.5474 (var=0.0043)

### Experiment 10
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
**Predicted under pi_4:** 0.1520 (var=0.0100)
**Predicted under pi_6:** 0.1853 (var=0.0081)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, pi_4 is the better provisional account, but neither theory is task-invariant. By absolute point-estimate error, pi_4 is closer in Experiments 1, 3, 4, 5, 6, 9, and 10, whereas pi_6 is closer in Experiments 2, 7, and 8. Pi_4 is especially successful in Experiment 3 (0.0091 versus 0.0011 observed), Experiment 6 (0.5411 versus 0.5200), Experiment 9 (0.6617 versus 0.6404), and Experiment 10 (0.1520 versus 0.1631). It also captures the perfectly consistent lexical polarity in Experiment 5, including the zero between-subject variance; pi_6 has the correct pooled polarity there but predicts implausibly large heterogeneity (var=0.1536). In contrast, pi_6 decisively explains the graded-margin effect in Experiment 2 (0.3067 versus 0.3168), which pi_4's sign-normalized pooling cannot generate. Pi_6 is also closer in Experiments 7 and 8, although both theories substantially underpredict the very strong majority preference in Experiment 8. The most theoretically important result is Experiment 1: the observed coalition-validity slope is strongly negative (-0.1996), not approximately zero as pi_4 predicts and not positive as pi_6 predicts. Thus pi_4 is only numerically closer there; its claimed invariance to lower-cue validity is qualitatively rejected. Pi_6 makes the more serious directional error because its central mechanism says stronger aggregate validity support for a majority should increase compression accessibility and majority evidence. Its broad failures in Experiments 3 and 4 reinforce the problem: it predicts a large tally effect where virtually none is observed in Experiment 3 and predicts lexical-over-majority behavior in Experiment 4 when the data favor the majority. These failures cannot be repaired merely by retuning pi_6: suppressing compression enough to fit Experiment 3 would worsen Experiments 2 and 8, while reversing the advertised-validity effect needed for Experiment 1 contradicts its theory description. Pi_4 therefore remains the stronger benchmark, but its categorical sign compression and validity blindness make it insufficient as a final explanation.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_6 with a new theory of reliability-segregated consensus and conflict-triggered hierarchy protection. The theory should posit a stable highest-validity lexical anchor plus a separate, graded representation of numerical consensus. Consensus strength should depend continuously on normalized tally margin, allowing the strong margin contrast in Experiment 2, rather than being sign-normalized as in pi_4. Crucially, advertised validity should not monotonically facilitate the majority route as in pi_6. When an anti-leader coalition contains several individually diagnostic experts, the recognized high-reliability conflict should trigger scrutiny and renewed deference to the top-ranked expert; this produces the negative coalition-validity slope in Experiment 1. By contrast, approximately equivalent or low-diagnosticity lower cues should be encoded as fungible votes, so a conspicuous numerical majority can override the leader, especially in dense or structurally simple blocks, accounting for Experiments 2 and 8. Block density and recent display statistics may alter the accessibility of consensus encoding, but they should modulate rather than determine arbitration. The new account should therefore predict lexical behavior in sparse or reliability-heterogeneous settings, graded majority effects when many cues are readily compressed into votes, and hierarchy protection when strong lower experts create explicit reliability conflict. A quantitative implementation could use a persistent latent task set with lexical and graded-consensus evidence, plus an inhibitory diagnostic-conflict term based on the number and strength of anti-leader cues. It should avoid cue-identity learning without feedback, constrain stable subject heterogeneity so it does not create pi_6's excessive variance in Experiment 5, and be jointly calibrated against all ten metrics rather than tuned primarily to the dense-margin experiment."
}
```

## Usage

```json
{
  "input_tokens": 21245,
  "output_tokens": 2266
}
```
