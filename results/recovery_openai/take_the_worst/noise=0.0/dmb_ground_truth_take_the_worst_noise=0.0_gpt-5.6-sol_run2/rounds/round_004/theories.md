# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Ecological Encoding Hysteresis theory claims that decision makers learn how to encode an entire block of comparisons. They begin with a sparse-ecology prior favoring maintenance of the advertised validity hierarchy. After each comparison, they update a slowly changing estimate of the proportion of cues that discriminate. When discrimination remains sparse and ties are common, cue ranks remain cognitively useful and an ordered-search task set persists: the highest-validity discriminating cue controls choice, even on an occasional dense trial. When comparisons repeatedly contain many simultaneous discriminations, maintaining separate cue ranks becomes inefficient. People then compress cue directions into a relational gist and approximately pool them with equal weights. A steep contextual transition and a ceiling on compression implement hysteresis-like persistence rather than trial-by-trial load arbitration. Individual differences in the contextual threshold, prior strength, compression ceiling, sensitivity, and lapses produce heterogeneous behavior near the transition. Thus, identical diagnostic comparisons can elicit lexical choices in sparse filler blocks but majority choices in saturated filler blocks.

**Rationale:** The model addresses the central failure of invariant Take-The-Best by permitting genuine lower-cue influence, but only after the block ecology induces a compressed encoding. Experiment 1 has a relatively tie-rich average ecology, so its contextual estimate remains below threshold and even dense embedded trials are usually processed lexicographically, yielding negligible tally modulation. Experiment 2 repeatedly presents high discrimination density, rapidly moving the contextual state above threshold; the resulting mixture is majority-favoring but retains enough ordered processing to avoid the excessive tally contrast of a pure equal-weight rule. Experiment 3 averages below the transition despite containing comparisons with several discriminating cues, preserving a reliably lexical preference. Experiment 4 is maximally saturated, so compression approaches its subject-specific ceiling and produces the observed majority preference. Unlike a current-load gate, the decision rule for a stimulus depends on preceding filler trials. The model therefore predicts that placing identical conflict trials in sparse versus saturated blocks will reverse their dominant influence from the validity-leading cue toward the cue majority.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `context_prior`: `[0.50, 0.60]`
  - `prior_strength`: `[1.0, 3.0]`
  - `context_threshold`: `[0.68, 0.71]`
  - `context_steepness`: `[35.0, 55.0]`
  - `pooling_ceiling`: `[0.68, 0.76]`
  - `evidence_sensitivity`: `[1.6, 3.0]`
  - `lapse`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Contextual Margin Arbitration theory proposes that people maintain two competing representations of binary-feature comparisons. The advertised-rank route retrieves the most valid discriminating expert and favors its option. The conflict-compression route represents all discriminating experts as a directional tally while retaining the tally's graded normalized magnitude. Accessibility of compression is jointly controlled by a slowly persistent estimate of block density, the current normalized majority margin, a recency-weighted trace of recent margins, and the aggregate advertised-validity support for the current majority. Dense blocks make compression chronically accessible, whereas sparse blocks require a clearly suprathreshold majority to overcome rank processing. Density and margin provide compensatory routes to compression, so their overlap does not produce a superadditive accessibility boost. Aggregate validity shifts route accessibility and evidence continuously. A smoothly saturating diagnosticity representation preserves unusually extreme advertised-validity dominance without turning it into a categorical veto: dense context strongly reduces validity opposition, whereas margin-driven release is weaker and leaves a residual that grows continuously with the extremity of anti-majority support. Compression uses a mildly convex transform of normalized tally magnitude, making conspicuously large majorities more persuasive than small anti-leader coalitions while retaining continuous evidence. Repetition of particular cue identities has no independent learning effect because choices receive no correctness feedback. Stable individual differences in contextual thresholds, margin thresholds, gating sensitivities, evidence precision, and lapses generate heterogeneous intermediate behavior.

**Rationale:** This is a minimal in-family revision of the accepted candidate. The hard diagnosticity cap at 3 is replaced by smooth tanh saturation with a higher subject-level scale. Consequently, a uniquely near-perfect anti-majority leader produces stronger but still bounded residual opposition than an ordinary validity imbalance. This specifically supplies the missing distinction between Experiment 1 and the less validity-extreme large-margin conflicts in Experiment 6 without introducing pi_5's categorical safeguard. The density threshold is moved only modestly downward to an intermediate range, partially undoing the latest change so dense Experiments 2, 5, and 7 can recover chronic compression while leaving the baseline gate unchanged. Finally, normalized tally magnitude receives a very mild convex transform with a recalibrated scale. This preserves small-margin evidence at approximately its previous level but selectively strengthens conspicuous majorities in Experiment 6. All contextual traces, compensatory gate geometry, validity release, and the absence of cue-identity learning are otherwise unchanged, protecting the improved fits in Experiments 4, 5, and 8.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `validity_saturation`: `[5.0, 7.0]`
  - `context_memory_span`: `[14.0, 26.0]`
  - `density_prior`: `[0.48, 0.58]`
  - `margin_prior`: `[0.10, 0.22]`
  - `density_threshold`: `[0.68, 0.71]`
  - `margin_threshold`: `[0.24, 0.28]`
  - `margin_smoothness`: `[45.0, 70.0]`
  - `base_compression_access`: `[-2.8, -2.0]`
  - `density_sensitivity`: `[30.0, 40.0]`
  - `margin_sensitivity`: `[30.0, 42.0]`
  - `recent_margin_sensitivity`: `[0.5, 1.6]`
  - `density_margin_interaction`: `[0.75, 1.0]`
  - `validity_access_bias`: `[2.0, 4.0]`
  - `density_validity_release`: `[10.0, 16.0]`
  - `margin_validity_release`: `[1.0, 3.0]`
  - `validity_release_floor`: `[0.20, 0.40]`
  - `validity_evidence_gain`: `[0.15, 0.65]`
  - `tally_margin_exponent`: `[1.05, 1.15]`
  - `tally_evidence_scale`: `[4.8, 5.4]`
  - `rank_sensitivity`: `[1.8, 3.2]`
  - `tally_sensitivity`: `[0.70, 0.90]`
  - `lapse`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Reliability-Segregated Conflict Arbitration theory proposes that people preserve the advertised validity hierarchy as a stable lexical representation while separately constructing a graded representation of numerical consensus. The lexical representation retrieves the highest-validity discriminating expert. The consensus representation encodes the signed tally and its continuous normalized margin rather than merely its sign. A slowly updated task-set state, inferred from the density and margins of recently viewed displays, regulates how accessible consensus encoding is. Reliability structure determines whether lower experts are treated as fungible votes or as individually diagnostic sources. Similar-reliability experts and uniformly weak lower experts are readily compressed into votes, whereas a heterogeneous hierarchy preserves their reliability identities. A sparse feature set combined with a segregated reliability hierarchy protects lexical processing across the whole task ecology. When a numerical majority opposes the lexical leader, hierarchy protection is triggered nonlinearly by joint occupancy of several high-validity positions in a fixed-capacity lower-cue prefix. Consensus release is therefore determined directly by the absence of a protected prefix, not indirectly by transforming global reliability segregation. Independently, sustained exposure to a densely discriminating block makes vote compression chronically accessible, but this learned accessibility is suppressed by protected-prefix conflict and attenuated under extreme simultaneous cue load. Consequently, increasing the reliability and ranked coherence of an anti-leader coalition can reduce majority following, whereas conspicuous majorities composed of fungible or dispersed cues can override the leader. Because no outcomes are provided, the model learns only block-level display statistics and never learns cue identities or cue-response associations.

**Rationale:** This is a targeted edit of accepted iteration 8. The global segregation mapping, diagnostic cutoff, prefix-synergy calculation, graded consensus evidence, and response scale are preserved. First, `protected_coherence` is exposed outside the conflict block and its inverse now gates only the additional conditional compression routes. This directly distinguishes a coalition occupying the protected lower-rank prefix from dispersed support, avoiding iteration 9's rejected supralinear transformation of global segregation and protecting Experiment 8. Second, a sustained-density term is added using the past-only density trace and a high threshold. It primarily becomes active after prolonged exposure to nearly fully discriminating blocks, addressing the late-trial majority deficit in Experiment 5 and dense Experiment 4 without globally raising the consensus intercept. The term is suppressed by protected-prefix coherence and exponentially attenuated above ten active cues. A small similarly gated fungibility increment targets the sparse but readily compressible coalitions in Experiment 6. Third, the direct above-ten-cue penalty is modestly strengthened to reduce Experiment 7 without reinstating the rejected inverted-U count gate. Seven-cue hierarchy protection is increased only slightly to continue flattening Experiment 1. Finally, the consensus evidence scale is restored unchanged; the Experiment 10 contrast is instead addressed by a modestly sharper access transition around the existing margin threshold, preserving weak-margin pooling through the separate sustained-density route.

**Parameters:**
  - `n_features`: `n_features`
  - `validities`: `validities`
  - `segregation_scale`: `[0.45, 0.65]`
  - `diagnostic_cutoff`: `[0.75, 0.95]`
  - `diagnostic_sharpness`: `[6.0, 9.0]`
  - `context_memory_span`: `[16.0, 22.0]`
  - `density_prior`: `[0.48, 0.54]`
  - `margin_prior`: `[0.15, 0.21]`
  - `density_threshold`: `[0.56, 0.62]`
  - `history_density_threshold`: `[0.78, 0.84]`
  - `history_density_gain`: `[7.0, 9.0]`
  - `base_consensus_access`: `[-2.0, -1.7]`
  - `density_access_gain`: `[5.5, 6.5]`
  - `margin_access_gain`: `[3.5, 4.5]`
  - `recent_margin_gain`: `[0.5, 1.0]`
  - `fungibility_gain`: `[1.3, 1.8]`
  - `coherence_fungibility_gain`: `[0.4, 0.8]`
  - `ranked_prefix_span`: `[0.9, 1.3]`
  - `prefix_synergy_center`: `[0.85, 0.95]`
  - `prefix_synergy_sharpness`: `[7.0, 9.0]`
  - `prefix_conflict_floor`: `[0.05, 0.12]`
  - `prefix_synergy_gain`: `[1.15, 1.35]`
  - `conflict_count_scale`: `[1.5, 2.2]`
  - `conflict_inhibition`: `[5.0, 7.0]`
  - `hierarchy_protection_gain`: `[0.5, 1.0]`
  - `conditional_density_gain`: `[4.5, 5.5]`
  - `conditional_fungibility_gain`: `[0.8, 1.2]`
  - `high_load_penalty`: `[1.45, 1.70]`
  - `set_size_transition`: `[7.4, 7.6]`
  - `set_size_sharpness`: `[5.0, 7.0]`
  - `small_set_protection_gain`: `[2.4, 3.1]`
  - `margin_transition`: `[0.18, 0.21]`
  - `margin_steepness`: `[10.5, 13.5]`
  - `consensus_margin_exponent`: `[0.90, 1.10]`
  - `consensus_evidence_scale`: `[4.5, 5.5]`
  - `lexical_sensitivity`: `[2.9, 3.3]`
  - `consensus_sensitivity`: `[1.4, 1.8]`
  - `lapse`: `[0.02, 0.06]`

**`predict(parameters, stimulus, history)`:**
```python
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

    # Diagnostic conflict is protected when anti-leader support occupies a
    # coherent high-validity prefix of the lower-cue hierarchy. Normalization
    # uses the fixed capacity of that hierarchy, so extra low-ranked supporters
    # cannot dilute an already occupied diagnostic prefix. A nonlinear joint-
    # occupancy term distinguishes several protected cues from one strong cue
    # accompanied by dispersed low-ranked votes.
    conflict = 0.0
    protected_coherence = 0.0
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
            weighted_occupancy = individual_diagnosticity[coalition] * coalition_weights
            prefix_strength = float(np.sum(weighted_occupancy))
            fixed_prefix_capacity = float(np.sum(np.exp(
                -np.arange(len(lower_order), dtype=float) / prefix_span
            )))
            prefix_coherence = (
                prefix_strength / fixed_prefix_capacity
                if fixed_prefix_capacity > 1e-12 else 0.0
            )
            prefix_coherence = float(np.clip(prefix_coherence, 0.0, 1.0))

            synergy_logit = np.clip(
                float(parameters["prefix_synergy_sharpness"])
                * (prefix_strength - float(parameters["prefix_synergy_center"])),
                -60.0,
                60.0
            )
            prefix_synergy = 1.0 / (1.0 + np.exp(-synergy_logit))
            coherence_multiplier = (
                float(parameters["prefix_conflict_floor"])
                + float(parameters["prefix_synergy_gain"]) * prefix_synergy
            )
            protected_coherence = float(np.clip(
                prefix_coherence * coherence_multiplier, 0.0, 1.0
            ))

            count_scale = max(float(parameters["conflict_count_scale"]), 1e-9)
            count_salience = 1.0 - np.exp(-float(coalition_count) / count_scale)
            conflict = float(segregation * protected_coherence * count_salience)

    density_excess = max(
        context_density - float(parameters["density_threshold"]), 0.0
    )

    # Consensus salience changes smoothly around a psychologically visible
    # normalized margin. Fungibility modulates, but does not veto, this signal.
    margin_transition = float(parameters["margin_transition"])
    margin_steepness = float(parameters["margin_steepness"])
    margin_logit = np.clip(
        margin_steepness * (margin - margin_transition), -60.0, 60.0
    )
    visible_margin = 1.0 / (1.0 + np.exp(-margin_logit))
    effective_margin_access = visible_margin * (0.65 + 0.35 * fungibility)

    # Compression release is tied directly to coalition configuration. A
    # coherent occupied prefix remains protected even if global segregation is
    # only moderate, while dispersed coalitions receive density/fungibility
    # access. Displays beyond ten active cues retain a shallow load penalty.
    conflict_release = (1.0 - conflict) ** 2
    configuration_release = (1.0 - protected_coherence) ** 2
    conditional_compression = conflict_release * configuration_release * (
        float(parameters["conditional_density_gain"]) * density_excess
        + float(parameters["conditional_fungibility_gain"])
        * fungibility * visible_margin
    )
    high_load = max(float(n_active) - 10.0, 0.0)

    # Sustained dense exposure supplies a distinct learned task-set benefit.
    # It depends on past displays rather than the current trial, is gated by
    # protected-prefix coherence, and fades under extreme simultaneous load.
    history_density_excess = max(
        density_trace - float(parameters["history_density_threshold"]), 0.0
    )
    sustained_density_access = (
        float(parameters["history_density_gain"])
        * history_density_excess
        * configuration_release
        * np.exp(-high_load)
    )
    low_conflict_fungibility = (
        float(parameters["coherence_fungibility_gain"])
        * fungibility
        * visible_margin
        * configuration_release
    )

    # A segregated hierarchy in a genuinely small feature set maintains a
    # lexical task set across the ecology, even when tally and leader agree.
    # The sharp size transition confines this protection primarily to seven-
    # cue sets instead of suppressing ordinary eight-to-ten-cue compression.
    size_logit = np.clip(
        float(parameters["set_size_sharpness"])
        * (float(parameters["set_size_transition"]) - float(n_features)),
        -60.0,
        60.0
    )
    small_set_activation = 1.0 / (1.0 + np.exp(-size_logit))
    small_set_protection = segregation * small_set_activation

    # Consensus access receives independent contributions from the persistent
    # dense-task set, visible graded margin, recent margin statistics, and cue
    # fungibility. Coherent ranked conflict and sparse segregated ecologies
    # selectively inhibit access.
    gate_logit = (
        float(parameters["base_consensus_access"])
        + float(parameters["density_access_gain"]) * density_excess
        + float(parameters["margin_access_gain"]) * effective_margin_access
        + float(parameters["recent_margin_gain"]) * recent_margin
        + float(parameters["fungibility_gain"]) * fungibility
        + conditional_compression
        + sustained_density_access
        + low_conflict_fungibility
        - float(parameters["high_load_penalty"]) * high_load
        - float(parameters["conflict_inhibition"]) * conflict
        - float(parameters["small_set_protection_gain"]) * small_set_protection
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
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
