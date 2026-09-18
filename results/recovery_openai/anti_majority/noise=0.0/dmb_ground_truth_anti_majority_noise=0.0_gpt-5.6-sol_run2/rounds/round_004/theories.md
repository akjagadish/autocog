# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Capacity-Calibrated Dual-Channel Integration theory: people continuously integrate a rapid representation of the relative number of marked ratings with a validity-sensitive representation of where those marks occur. Neither representation is encoded in raw additive units. The count channel expresses the tally margin relative to the proportion of displayed cues that discriminate and then passes it through a smooth saturating response, so large margins remain decisive without producing indefinitely increasing evidence. A stable discrimination-scaling exponent governs how strongly sparse cue sets dilute count evidence. The validity channel represents count-residualized reliability evidence using softly standardized units and divisively normalizes it according to display size, the number of discriminating cues, and cue conflict. Partial rather than full reliability-energy standardization preserves psychologically meaningful differences between sharply separated and shallow validity profiles without allowing low-dispersion profiles to become artificially amplified. Validity remains available on every trial, including count ties, but there are no special tie, audit, or selected-margin rules. A stable allocation parameter controls the balance between channels. Capacity also contracts the behavioral expression of individual differences in allocation and precision as displays become large, yielding appreciable heterogeneity in compact tasks but more homogeneous behavior in very high-dimensional tasks. Stable semantic-polarity confidence determines whether marked states are interpreted as adverse or beneficial, while stable response precision and lapse rate govern stochasticity.

**Rationale:** This is an isolated count-side edit to the accepted iteration-7 model. The fixed square root of discrimination fraction is replaced by discrimination_fraction raised to a stable subject-level exponent in [0.20, 0.45]. Because this exponent is below one-half, it produces a larger count denominator when only a sparse subset of displayed cues discriminates, selectively weakening the excessive adverse count effect in Experiment 4. When most or all cues discriminate, the fraction approaches one and the change becomes negligible, protecting Experiment 3's strong extreme-margin behavior and reducing the risk of another Experiment 8 sign reversal. The accepted count-saturation range, validity standardization, validity normalization, capacity contraction, and all other parameter ranges remain unchanged. No tie rule, selected-margin bonus, profile-dispersion adaptation, or new validity nonlinearity is introduced.

**Parameters:**
  - `validities`: `validities`
  - `count_allocation`: `[0.58, 0.86]`
  - `count_saturation_scale`: `[0.6, 1.2]`
  - `count_discrimination_exponent`: `[0.20, 0.45]`
  - `integration_capacity`: `[7.0, 14.0]`
  - `validity_sensitivity`: `[2.2, 4.5]`
  - `reliability_standardization_exponent`: `[0.05, 0.30]`
  - `display_compression`: `[0.35, 1.1]`
  - `discrimination_compression`: `[0.3, 1.0]`
  - `conflict_compression`: `[0.45, 1.35]`
  - `response_precision`: `[3.0, 6.0]`
  - `adverse_polarity_confidence`: `[0.84, 1.0]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Calibrated Dual-Channel Integration expects shape "
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
    discriminating = np.abs(difference)
    n_discriminating = float(np.sum(discriminating))
    count_difference = float(np.sum(difference))

    if n_discriminating <= 0.0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["integration_capacity"])

    # Large displays reduce the behavioral expression of stable parameter
    # differences. This is a continuous capacity effect rather than a switch.
    heterogeneity_expression = capacity / (capacity + float(n_features))
    raw_allocation = float(parameters["count_allocation"])
    allocation = 0.5 + heterogeneity_expression * (raw_allocation - 0.5)

    raw_precision = float(parameters["response_precision"])
    effective_precision = 1.0 + heterogeneity_expression * (raw_precision - 1.0)

    # The tally margin is represented relative to the fraction of the display
    # that actually discriminates. Smooth saturation makes margins near the
    # extremes robust while compressing the difference between moderate and
    # very large raw margins. The exponent below one-half weakens count
    # evidence selectively when only a sparse subset of cues discriminates.
    discrimination_fraction = n_discriminating / float(n_features)
    count_denominator = float(parameters["count_saturation_scale"]) * (
        max(discrimination_fraction, 1.0 / float(n_features))
        ** float(parameters["count_discrimination_exponent"])
    )
    relative_count = count_difference / max(count_denominator, 1e-12)

    count_load_factor = np.sqrt(capacity / (capacity + float(n_features)))
    count_signal = (
        2.0
        * allocation
        * count_load_factor
        * np.tanh(relative_count)
    )

    # Remove the validity component that merely repeats total mark count.
    centered_validities = validities - float(np.mean(validities))
    raw_validity = float(np.dot(centered_validities, difference))

    # Soft standardization preserves differences in validity-profile
    # dispersion. An exponent below one prevents shallow profiles with small
    # reliability energy from being disproportionately amplified.
    reliability_energy = float(
        np.sqrt(np.dot(centered_validities ** 2, discriminating))
    )
    reliability_exponent = float(parameters["reliability_standardization_exponent"])
    validity_denominator = max(reliability_energy, 0.025) ** reliability_exponent
    standardized_validity = raw_validity / validity_denominator

    # Conflict is maximal when discriminating cues split evenly and declines
    # smoothly as their count majority becomes coherent.
    conflict = 1.0 - min(1.0, abs(count_difference) / n_discriminating)
    display_load = float(n_features) / (float(n_features) + capacity)
    discrimination_load = n_discriminating / (n_discriminating + capacity)

    normalization = (
        1.0
        + float(parameters["display_compression"]) * display_load
        + float(parameters["discrimination_compression"]) * discrimination_load
        + float(parameters["conflict_compression"]) * conflict
    )
    validity_signal = (
        (1.0 - allocation)
        * float(parameters["validity_sensitivity"])
        * standardized_validity
        / normalization
    )

    markedness_evidence = effective_precision * (count_signal + validity_signal)

    # Under the adverse interpretation, positive markedness evidence means A
    # carries the greater burden and therefore favors response B.
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Polarity-Anchored Dual-Timescale Margin Integration theory: subjects maintain a stable semantic interpretation of whether binary marks are adverse or beneficial. Conditional on that interpretation, tally evidence depends only on the signed absolute count margin, never on display density or the number of discriminating cues. A rapidly compressing accumulator supplies strong evidence at the first nonzero margin, while a very small logarithmic tail preserves continued but sharply diminishing growth across moderate and large margins. Count-residualized validity evidence remains continuously available but is softly standardized and subject to a later, steeper capacity curve that slightly reduces average validity influence at 11 features, preserves relatively more influence at 15 features, and strongly attenuates it by 19 and 37 features. Stable subject differences in count strength, compression, validity attention, semantic polarity, precision, and lapses produce persistent behavioral heterogeneity.

**Rationale:** This is a parameter-localized refinement of the accepted architecture. The logarithmic-tail range is reduced and narrowed while remaining strictly positive. Thus the saturated fast component continues to support the already accurate large-margin choice rate in Experiment 10, but there is less incremental count evidence between margins 1 and 5, targeting Experiment 8's residual +44 contrast and limiting its excessive variance. The validity-capacity knee is moved later and made steeper while mean validity attention is reduced. Relative to the previous curve, this can slightly reduce average validity influence at 11 cues while retaining more at 15 cues, allowing residual validity to compete with the first count increment in Experiment 7 without restoring the Experiment 9 density association at 19 cues or validity effects at 37 cues. Reliability standardization is moved close to zero so smooth, low-energy profiles are no longer disproportionately strengthened, targeting the excessive Experiment 6 slope relative to Experiment 5. Finally, the validity-attention range is widened around its revised mean to increase subject heterogeneity in Experiments 1 and 3–6, while the narrowed tail avoids adding further heterogeneity to Experiment 8.

**Parameters:**
  - `validities`: `validities`
  - `count_gain`: `[1.8, 2.6]`
  - `fast_margin_scale`: `[0.25, 0.55]`
  - `log_tail_gain`: `[0.002, 0.02]`
  - `validity_attention`: `[0.5, 1.8]`
  - `reliability_standardization`: `[0.0, 0.06]`
  - `reliability_shrinkage`: `[0.15, 0.35]`
  - `validity_capacity`: `[14.0, 16.0]`
  - `display_load_exponent`: `[5.0, 7.0]`
  - `response_precision`: `[0.7, 1.45]`
  - `adverse_polarity_confidence`: `[0.92, 1.0]`
  - `lapse_rate`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Polarity-Anchored Dual-Timescale Margin Integration expects "
            f"shape (2, n_features); got {stim.shape}."
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
    margin = abs(count_difference)

    # The count representation depends exclusively on the raw tally margin.
    # A dominant fast component makes the first nonzero margin informative
    # and then compresses rapidly. A much smaller logarithmic tail preserves
    # continued but diminishing growth at moderate and large margins.
    if margin > 0.0:
        fast_component = float(parameters["count_gain"]) * np.tanh(
            margin / float(parameters["fast_margin_scale"])
        )
        slow_component = float(parameters["log_tail_gain"]) * np.log1p(margin)
        count_magnitude = fast_component + slow_component
        count_signal = float(np.sign(count_difference)) * count_magnitude
    else:
        count_signal = 0.0

    # Centering removes the portion of validity-weighted evidence that merely
    # restates which option has more marks. The resulting reliability residual
    # remains available at every margin, including exact count ties.
    centered_validities = validities - float(np.mean(validities))
    raw_validity_residual = float(np.dot(centered_validities, difference))

    # Soft shrinkage standardization avoids disproportionately amplifying
    # low-energy reliability profiles while retaining meaningful differences
    # in reliability dispersion. It uses no trial-specific density or conflict.
    profile_energy = float(np.sqrt(np.dot(
        centered_validities, centered_validities
    )))
    standardization_exponent = float(
        parameters["reliability_standardization"]
    )
    shrinkage_scale = np.sqrt(
        float(parameters["reliability_shrinkage"])
        + profile_energy ** 2
    )
    validity_units = raw_validity_residual / (
        shrinkage_scale ** standardization_exponent
    )

    # Reliability inspection decreases continuously with display complexity.
    # The later, steeper knee preserves relatively more access near 15 cues
    # than the previous curve while retaining strong attenuation by 19 and 37.
    capacity = float(parameters["validity_capacity"])
    load_exponent = float(parameters["display_load_exponent"])
    attention_factor = 1.0 / (
        1.0 + (float(n_features) / capacity) ** load_exponent
    )
    validity_signal = (
        float(parameters["validity_attention"])
        * attention_factor
        * validity_units
    )

    markedness_evidence = float(parameters["response_precision"]) * (
        count_signal + validity_signal
    )

    # Positive markedness means that A carries more adverse evidence and hence
    # favors B under the adverse semantic interpretation.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence],
        dtype=float,
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # A stable alternative interpretation treats marks as beneficial and
    # reverses the entire integrated evidence distribution.
    beneficial_probs = adverse_probs[::-1]
    polarity = float(parameters["adverse_polarity_confidence"])
    core_probs = (
        polarity * adverse_probs
        + (1.0 - polarity) * beneficial_probs
    )

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
