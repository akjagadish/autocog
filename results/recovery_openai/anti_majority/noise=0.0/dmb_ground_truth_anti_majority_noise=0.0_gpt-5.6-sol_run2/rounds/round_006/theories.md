# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** Resource-Adaptive Diagnostic Field Integration theory: people establish a stable interpretation of whether marked ratings are adverse or beneficial, then allocate attention to currently discriminating evidence rather than total displayed feature count. Count and reliability information form a joint diagnostic field. Tally evidence accumulates smoothly with count margin, while reliability attention increases when discriminating cues expose a salient and internally organized reliability contrast. Accessibility is additionally calibrated by the clustering structure of the full communicated validity profile: coherent high-versus-low reliability blocks preserve attention, whereas uniformly graded profiles receive less reliability amplification. Only an unusually large set of simultaneously discriminating cues creates overload; under such overload, count evidence is divisively compressed, especially at larger margins. Opposing count and reliability signals usually inhibit one another through smooth conflict normalization. However, this normalization is locally weaker when the first count advantage appears, because a single additional mark is treated as diagnostic evidence rather than as a reason to globally discount the pre-existing reliability field. Stable subject-level integration style follows a softly polarized continuum between predominantly tally-oriented and predominantly reliability-sensitive policies. Subjects also differ stably in how deterministically they express their integrated evidence. This precision heterogeneity is mean-preserving and becomes less behaviorally expressed as diagnostic demand and overload increase, producing persistent individual differences in ordinary tasks without destabilizing dense displays or changing semantic polarity.

**Rationale:** This is a minimal edit of the accepted iteration-6 source. The count process, reliability-attention mapping, full-profile clustering measure, overload gate, conflict normalization, and successful margin-one notch are unchanged. The only mechanistic addition is a stable subject-level precision deviation centered symmetrically on zero. It is added around the accepted response-precision policy rather than widening style endpoints or reallocating count and reliability resources, both of which were rejected in prior iterations. The deviation is divided by one plus the existing overload signal and normalized diagnostic demand. Thus, it is expressed most strongly in ordinary sparse comparisons, where simulated between-subject variances are systematically too low, but is damped in Experiments 8, 9, and 11, where additional heterogeneity could destabilize the accepted dense-display fits. Because the deviation has a symmetric zero-centered range, it preserves average precision to first order and should change pooled point estimates much less than a global gain or allocation edit. It also makes a falsifiable cross-task prediction: a subject who is unusually deterministic in one ordinary task should remain unusually deterministic in other ordinary tasks, but this individual difference should contract when many cues simultaneously require comparison.

**Parameters:**
  - `validities`: `validities`
  - `reliability_style`: `[0.0, 1.0]`
  - `count_gain`: `[0.80, 1.24]`
  - `count_scale`: `[3.2, 5.8]`
  - `diagnostic_span`: `[20.0, 23.0]`
  - `overload_softness`: `[0.6, 1.2]`
  - `compression_margin_scale`: `[2.2, 3.4]`
  - `margin_activation_power`: `[2.6, 3.4]`
  - `overload_compression`: `[0.45, 0.80]`
  - `baseline_reliability_attention`: `[0.25, 0.61]`
  - `contrast_attention_gain`: `[0.90, 1.50]`
  - `diffuse_reliability_compression`: `[0.01, 0.08]`
  - `conflict_normalization`: `[0.05, 0.35]`
  - `response_precision`: `[0.78, 1.30]`
  - `precision_deviation`: `[-0.28, 0.28]`
  - `adverse_polarity_confidence`: `[0.82, 1.0]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — KILLED ✗

**Description:** Semantic-Polarity Smooth Integration theory: people first interpret what the marked binary state means rather than assuming that 1 is intrinsically desirable. Each subject forms a stable posterior belief that a mark is adverse or suspicious. Conditional on either the adverse interpretation or its residual beneficial alternative, the subject smoothly integrates two continuous signals: the difference in total marks and a partially count-residualized validity-weighted difference. The count signal normally dominates, while the weighted signal supplies graded sensitivity to which experts carry the marks on every trial. The two semantic interpretations generate opposite choice distributions, which are averaged according to the subject's polarity confidence. A small lapse process captures stimulus-independent errors.

**Rationale:** This is a minimal edit to the accepted iteration-2 model. The raw validity signal is only partially, rather than fully, residualized against total count by subtracting 10–30% of mean validity. This preserves the count component whose complete removal caused the rejected iteration-4 deterioration, while modestly reducing the excessive effective count strength responsible for Experiment 4's overly negative prediction. The count-weight range is raised only slightly to help retain the Experiment 1 and large-margin Experiment 3 effects. Sampling partial centering per subject also introduces targeted heterogeneity in the count-to-validity tradeoff without globally increasing lapse or weakening adverse polarity. Validity remains continuous on all trials, so the edit does not introduce a count gate.

**Parameters:**
  - `validities`: `validities`
  - `count_weight`: `[0.38, 1.05]`
  - `validity_weight`: `[0.7, 1.8]`
  - `partial_centering`: `[0.1, 0.3]`
  - `adverse_polarity_confidence`: `[0.78, 1.0]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_9` → slot 2 (via `new_theory`)

**Description:** Channel-Specific Coherence Integration theory: binary counts and cue reliabilities are encoded by distinct cognitive channels before being combined. The count channel rapidly computes the difference between option-level numbers of marked ratings. It depends only on that signed margin, so adding mutually canceling cues does not dilute count evidence. Its response is smooth and bounded, reflecting saturation of an approximate magnitude representation rather than unlimited linear accumulation. The reliability channel instead binds each discriminating cue's identity to its communicated validity. It first removes the validity component redundant with the count margin and then measures the signed coherence of the remaining cue-validity contributions. A reliability residual supported by mutually consistent contributions receives high gain, whereas the same residual embedded in opposing contributions receives lower gain because maintaining the required cue-identity bindings is difficult. This loss is determined by evidence coherence and communicated validity-profile contrast, not by feature count, diagnostic-cue count, or a generic load threshold. Stable subject differences govern relative allocation to the two channels and response precision. A stable semantic-polarity belief determines whether marks are interpreted as adverse or beneficial, and a small lapse process captures stimulus-independent choices.

**Rationale:** The model directly addresses the prior mechanistic failures by making count and reliability genuinely different summaries. Count evidence is a bounded function of signed margin alone. Consequently, adding any number of mutually canceling cues leaves it unchanged, predicting the count invariance required by Experiments 2, 11, and 14 while avoiding pi_8's diagnostic-load compression. Gradual tanh saturation preserves strong large-margin choices in Experiments 3 and 10 but limits the excessive linear margin separation seen for pi_4 in Experiment 8 and supports a moderate trajectory in Experiment 12. Reliability modulation is independently identified: coherence is the absolute residual divided by the absolute reliability support, so two trials can have identical residuals, margins, diagnostic counts, and display sizes while differing in internal cancellation. This predicts positive attenuation in Experiment 13 and weaker reliability influence in internally opposing conditions such as Experiment 1. Coherent high-versus-low profiles retain substantial reliability sensitivity, supporting Experiment 11, while the continuously varying coherence and profile terms produce intermediate rather than all-or-none reliability slopes in Experiments 6–9 and 13. There is no feature-count term, generic overload variable, capacity threshold, or late-load knee. Broad style and precision ranges create stable subject heterogeneity, while centered reliability evidence, global semantic reversal, and symmetric lapse mixing preserve A/B reversal symmetry.

**Parameters:**
  - `validities`: `validities`
  - `reliability_style`: `[0.0, 1.0]`
  - `count_gain`: `[1.9, 3.0]`
  - `count_scale`: `[2.8, 4.8]`
  - `reliability_floor`: `[0.08, 0.30]`
  - `coherent_binding_gain`: `[0.25, 0.78]`
  - `coherence_power`: `[0.65, 1.65]`
  - `profile_contrast_gain`: `[0.0, 0.45]`
  - `response_precision`: `[0.68, 1.48]`
  - `adverse_polarity_confidence`: `[0.80, 1.0]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
