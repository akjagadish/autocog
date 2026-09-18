# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

**Description:** Relative-Evidence Conflict Arbitration theory: people construct two density-invariant summaries of a binary-feature display. A signed count accumulator represents only the difference in the numbers of marks carried by the options. It grows approximately linearly across ordinary margins and saturates only at relatively large margins, so adding equal numbers of opposing marks cannot dilute count evidence. In parallel, communicated validities are centered around their profile mean and bound to cue directions. Access to this reliability residual is governed by its absolute magnitude and signed coherence: consistently directed reliability contributions become accessible, whereas additions containing genuinely opposing contributions reduce access by increasing gross support without increasing the residual. Accessibility never depends directly on display size, the number of discriminating cues, or active-validity dispersion. A shallow, band-limited calibration discounts intermediate reliability magnitudes but recovers for genuinely strong residuals, representing temporary ambiguity in mapping moderately accumulated reliability evidence onto a categorical preference rather than generic overload. The two summaries are combined by relative-evidence arbitration. Congruent summaries receive modest mutual reinforcement. During conflict, balanced channels inhibit confidence, while sufficiently asymmetric conflicts recruit a winner-enhancement process favoring whichever signed summary is stronger. Thus pure count problems show robust margin growth, but structured count-reliability conflicts can be compressed or reversed without invoking density overload. After arbitration, a mild bounded-confidence transformation imposes diminishing decisional returns on repeatedly accumulated coherent evidence without altering the raw channel ratio used to resolve conflicts. Stable but moderate differences in channel balance and response precision coexist with stable semantic polarity and occasional lapses.

**Rationale:** This is an isolated parameter-range recalibration of the accepted iteration-6 model; no prediction equation, notch, arbitration branch, terminal bound, or sampling policy has been changed. The reliability floor is modestly lowered, while coherence gain and coherence curvature are increased. Consequently, weak or internally opposed signed reliability contributions receive less influence, targeting the insufficient tally dominance in Experiment 1 and the excessive reliability-related contrasts in Experiments 7 and 9. The magnitude half-saturation is lowered so moderate but highly coherent residuals remain accessible, protecting and potentially strengthening the structured reliability effects needed in Experiments 5 and 8. The midpoint of reliability_floor plus coherence_gain remains approximately unchanged from the accepted base, preserving asymptotic high-coherence reliability strength for Experiments 10, 11, and 13. The successful recovering magnitude notch is retained verbatim to protect the negative Experiment 16 result. Because only parameter domains changed, matched parameter dictionaries produce exactly the same probabilities as iteration 6 on every stimulus, including all count-neutral Experiment 16 trials; this avoids the hidden branch collateral effects seen in the two rejected edits.

**Parameters:**
  - `validities`: `validities`
  - `channel_balance`: `[0.0, 1.0]`
  - `count_gain`: `[0.92, 1.18]`
  - `count_scale`: `[7.0, 11.5]`
  - `reliability_floor`: `[0.14, 0.32]`
  - `coherence_gain`: `[0.66, 1.00]`
  - `coherence_power`: `[1.00, 1.70]`
  - `magnitude_half_saturation`: `[0.08, 0.28]`
  - `reliability_gain`: `[0.72, 1.16]`
  - `reliability_notch_center`: `[1.6, 2.2]`
  - `reliability_notch_width`: `[0.45, 0.80]`
  - `reliability_notch_depth`: `[0.12, 0.26]`
  - `congruent_synergy`: `[0.0, 0.14]`
  - `conflict_compression`: `[0.35, 0.95]`
  - `winner_enhancement`: `[0.18, 0.62]`
  - `arbitration_curvature`: `[0.65, 1.35]`
  - `decision_bound`: `[2.8, 4.8]`
  - `response_precision`: `[0.82, 1.22]`
  - `adverse_polarity_confidence`: `[0.84, 0.98]`
  - `lapse_rate`: `[0.0, 0.10]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Relative-Evidence Conflict Arbitration expects shape "
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

    # Stable channel balance produces moderate individual differences without
    # creating discrete strategy classes or trial-dependent attention shifts.
    raw_balance = float(parameters["channel_balance"])
    centered_balance = np.tanh(1.6 * (2.0 * raw_balance - 1.0)) / np.tanh(1.6)
    count_allocation = 1.0 - 0.14 * centered_balance
    reliability_allocation = 1.0 + 0.22 * centered_balance

    # The count accumulator reads only the signed mark margin. Its large scale
    # gives approximately linear growth through ordinary margins and delays
    # saturation. Equal and opposite added marks leave it exactly unchanged.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        count_allocation
        * float(parameters["count_gain"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # Reliability information is the component of validity-weighted evidence
    # not already represented by an undifferentiated count. Centering also
    # prevents the common validity level from acting as a second tally.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    gross_reliability_support = float(np.sum(np.abs(signed_components)))

    # Coherence is determined solely by the geometry of signed reliability
    # contributions. Consistent contributions have coherence one; opposing
    # additions increase gross support and reduce coherence even if the net
    # residual is unchanged. No cue-count or display-density term appears.
    if gross_reliability_support > 1e-12:
        coherence = abs(reliability_residual) / gross_reliability_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    residual_magnitude = abs(reliability_residual)
    half_saturation = float(parameters["magnitude_half_saturation"])
    magnitude_access = residual_magnitude / (
        residual_magnitude + half_saturation + 1e-12
    )

    accessibility = (
        float(parameters["reliability_floor"])
        + float(parameters["coherence_gain"])
        * (coherence ** float(parameters["coherence_power"]))
        * magnitude_access
    )

    # Intermediate reliability magnitudes receive a shallow ambiguity cost,
    # but accessibility recovers for genuinely strong residuals. This smooth
    # notch depends only on signed reliability magnitude, not display density.
    notch_center = float(parameters["reliability_notch_center"])
    notch_width = float(parameters["reliability_notch_width"])
    notch = np.exp(
        -0.5 * ((residual_magnitude - notch_center) / notch_width) ** 2
    )
    magnitude_calibration = (
        1.0 - float(parameters["reliability_notch_depth"]) * notch
    )

    reliability_signal = (
        reliability_allocation
        * float(parameters["reliability_gain"])
        * accessibility
        * magnitude_calibration
        * reliability_residual
    )

    c_abs = abs(count_signal)
    r_abs = abs(reliability_signal)
    total_abs = c_abs + r_abs

    if total_abs <= 1e-12:
        integrated_signal = 0.0
    elif count_signal * reliability_signal > 0.0:
        # Agreement is mildly superadditive but remains a function of relative
        # signed support, not the number of displayed or active features.
        agreement_balance = 4.0 * c_abs * r_abs / (total_abs * total_abs + 1e-12)
        integrated_signal = (count_signal + reliability_signal) * (
            1.0
            + float(parameters["congruent_synergy"]) * agreement_balance
        )
    elif count_signal * reliability_signal < 0.0:
        # Balanced conflict compresses confidence. Asymmetric conflict recruits
        # winner enhancement, permitting either robust count choices or
        # reliability-driven reversals according to relative evidence alone.
        conflict_balance = 4.0 * c_abs * r_abs / (
            total_abs * total_abs + 1e-12
        )
        relative_dominance = abs(c_abs - r_abs) / (total_abs + 1e-12)
        dominance_gate = relative_dominance ** float(
            parameters["arbitration_curvature"]
        )

        compressed_sum = (count_signal + reliability_signal) / (
            1.0
            + float(parameters["conflict_compression"])
            * conflict_balance
        )
        winner_signal = count_signal if c_abs >= r_abs else reliability_signal
        winner_bonus = (
            float(parameters["winner_enhancement"])
            * conflict_balance
            * dominance_gate
            * winner_signal
        )
        integrated_signal = compressed_sum + winner_bonus
    else:
        integrated_signal = count_signal + reliability_signal

    # Arbitration uses the unbounded channel summaries above, preserving the
    # raw count-to-reliability ratio in structured conflicts. Only the final
    # decisional representation has diminishing returns, so repeated coherent
    # contributions cannot increase confidence without limit.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )

    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive markedness means A carries the greater adverse burden and thus
    # favors B. Semantic polarity is fixed across the entire subject run.
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


### slot 2 — `pi_11` — KILLED ✗

**Description:** Contrast-Calibrated Reliability-Precision Competition theory: people encode a binary-feature display through two stable-polarity evidence channels. The count channel accumulates only the signed difference in marks and is therefore invariant to additions that cancel in the tally. It saturates smoothly after relatively small margins, while retaining a substantial precision floor, so ordinary and large margins differ less sharply in confidence. The reliability channel computes centered validity evidence but treats that residual as an uncertain estimate. Its precision is inferred jointly from residual magnitude, signed coherence, and active-field validity contrast relative to the characteristic dispersion of the communicated validity profile. Weak residuals retain partial availability through a confidence-weighted precision floor: coherent residuals in a distinct validity field remain available, whereas diffuse weak residuals receive little precision. Reliability is selectively diluted when active validity contrast falls relative to the full validity field, rather than merely because more cues are active. Count and reliability estimates enter a smooth precision-weighted competition: count controls choices when reliability is uncertain, whereas sufficiently precise coherent reliability can dominate count or resolve count ties. Broad continuous subject variation in count weighting, relative channel precision, and response precision produces persistent heterogeneity without discrete strategy classes.

**Rationale:** This is a minimal in-family edit of the accepted candidate. First, active contrast is divided by the full communicated profile's RMS dispersion before gating. The resulting dimensionless contrast pathway, combined with a smooth cubic mapping, strengthens detection of the deliberately declining active-field contrast in Experiment 16 without adding a raw feature-count term. Second, the coherence floor was raised and the magnitude and coherence powers were narrowed toward the intermediate setting requested by the critic. This preserves superlinear magnitude-by-coherence precision while reducing the severe Experiment 18 overshoot and should also weaken the unwanted Experiment 9 trajectory. Third, the reliability floor is no longer unconditional: it is multiplied by coherence and relative-contrast confidence. Its parameter range is modestly raised so weak but coherent, high-contrast residuals can counter excessive small-margin count control in Experiment 4, while diffuse weak evidence in Experiment 6 remains strongly downweighted. The count equations are unchanged to preserve the accepted fits in Experiments 1, 3, 8, 10, and 12. Only the count-precision, reliability-precision, and response-precision ranges were slightly broadened to increase continuous between-subject heterogeneity.

**Parameters:**
  - `validities`: `validities`
  - `count_weight`: `[1.25, 1.90]`
  - `count_scale`: `[2.8, 4.2]`
  - `count_precision`: `[0.35, 3.10]`
  - `count_precision_floor`: `[0.52, 0.82]`
  - `count_precision_half`: `[0.80, 2.00]`
  - `reliability_gain`: `[1.8, 4.4]`
  - `reliability_precision`: `[0.15, 3.10]`
  - `reliability_floor`: `[0.10, 0.28]`
  - `magnitude_half`: `[0.08, 0.30]`
  - `magnitude_power`: `[1.00, 1.70]`
  - `coherence_floor`: `[0.15, 0.35]`
  - `coherence_power`: `[0.85, 1.65]`
  - `contrast_half`: `[0.55, 1.25]`
  - `dilution_floor`: `[0.12, 0.42]`
  - `competition_scale`: `[1.45, 2.45]`
  - `decision_bound`: `[3.4, 6.8]`
  - `response_precision`: `[0.40, 2.10]`
  - `adverse_polarity_confidence`: `[0.82, 0.99]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Contrast-Calibrated Reliability-Precision Competition expects "
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
    count_magnitude = abs(count_difference)

    # This accumulator depends only on signed tally margin. Adding equally
    # many marks to opposing sides leaves both its signal and precision fixed.
    # Its relatively fast saturation flattens small-to-large margin growth.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_weight"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # A substantial baseline precision makes tallying available at ordinary
    # margins without making precision grow excessively with margin size.
    count_access = count_magnitude / (
        count_magnitude + float(parameters["count_precision_half"]) + 1e-12
    )
    count_precision = float(parameters["count_precision"]) * (
        float(parameters["count_precision_floor"])
        + (1.0 - float(parameters["count_precision_floor"])) * count_access
    )

    # Centering removes the validity component redundant with ordinary count.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    residual_magnitude = abs(reliability_residual)
    gross_support = float(np.sum(np.abs(signed_components)))

    if gross_support > 1e-12:
        coherence = residual_magnitude / gross_support
        coherence = float(np.clip(coherence, 0.0, 1.0))
    else:
        coherence = 0.0

    # Active contrast is interpreted relative to the characteristic dispersion
    # of the communicated profile. This dimensionless reference distinguishes
    # a genuinely weakening active field from a mere change in measurement
    # units, while introducing no direct cue-count or display-size penalty.
    active = np.abs(difference) > 0.5
    if np.any(active):
        active_deviations = centered_validities[active]
        active_contrast = float(np.sqrt(np.mean(active_deviations ** 2)))
    else:
        active_contrast = 0.0

    profile_contrast = float(
        np.sqrt(np.mean(centered_validities ** 2))
    )
    if profile_contrast > 1e-12:
        relative_contrast = active_contrast / profile_contrast
    else:
        relative_contrast = 0.0

    contrast_confidence = relative_contrast / (
        relative_contrast + float(parameters["contrast_half"]) + 1e-12
    )
    contrast_confidence = float(np.clip(contrast_confidence, 0.0, 1.0))

    # Selective dilution follows relative active-field signal quality, not raw
    # feature count. The smooth cubic mapping makes deliberately declining
    # contrast diagnostic while retaining the nonzero accessibility floor.
    dilution_floor = float(parameters["dilution_floor"])
    selective_dilution = (
        dilution_floor
        + (1.0 - dilution_floor) * contrast_confidence ** 3
    )

    # Reliability precision has smooth magnitude-by-coherence curvature. Weak
    # residuals remain partly available, while coherent large residuals become
    # disproportionately trustworthy. The softened coherence transformation
    # prevents an unrealistically extreme coherent/diffuse endpoint.
    magnitude_confidence = residual_magnitude / (
        residual_magnitude + float(parameters["magnitude_half"]) + 1e-12
    )
    magnitude_confidence = magnitude_confidence ** float(
        parameters["magnitude_power"]
    )
    coherence_confidence = coherence ** float(parameters["coherence_power"])
    coherence_term = (
        float(parameters["coherence_floor"])
        + (1.0 - float(parameters["coherence_floor"]))
        * coherence_confidence
    )

    diagnostic_confidence = (
        magnitude_confidence * coherence_term * selective_dilution
    )

    # The weak-residual floor is itself confidence weighted. Small coherent
    # residuals in a distinctive active field remain accessible, whereas weak
    # diffuse residuals no longer receive the same unconditional precision.
    reliability_floor = float(parameters["reliability_floor"])
    floor_confidence = coherence_term * selective_dilution
    reliability_quality = (
        reliability_floor * floor_confidence
        + (1.0 - reliability_floor) * diagnostic_confidence
    )

    if residual_magnitude <= 1e-12:
        reliability_precision = 0.0
        reliability_signal = 0.0
    else:
        reliability_precision = (
            float(parameters["reliability_precision"])
            * reliability_quality
        )
        reliability_signal = (
            float(parameters["reliability_gain"])
            * reliability_residual
        )

    # Precision-weighted averaging implements smooth competition without a
    # notch, hard strategy switch, or explicit winner-enhancement operation.
    # It applies identically to congruent and conflicting channel directions.
    total_precision = count_precision + reliability_precision
    if total_precision <= 1e-12:
        integrated_signal = 0.0
    else:
        integrated_signal = float(parameters["competition_scale"]) * (
            count_precision * count_signal
            + reliability_precision * reliability_signal
        ) / total_precision

    # A final bounded decision representation supplies diminishing returns but
    # does not alter the precision estimates used to arbitrate the channels.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )
    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive evidence means A carries greater adverse markedness and favors B.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Semantic polarity remains fixed over the subject run.
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


## Replacement

### `pi_12` → slot 2 (via `new_theory`)

**Description:** Bounded Diagnostic-Gating with Comparative Arbitration theory: people form a precise, density-invariant tally representation and a separate residual-reliability representation. The tally channel encodes only the signed difference in the number of marked ratings. It is already highly available at a margin of one, grows approximately linearly over ordinary margins, and has smoothly bounded returns only at large margins. The reliability channel removes the validity component redundant with tallying and represents the remaining centered-validity residual. Its diagnostic gate has an unconditional floor plus bounded, additive contributions from signed coherence and active-field validity contrast. Contrast modulation is restricted to weak-to-moderate residuals and vanishes smoothly toward an asymptote for large residuals. Consequently, coherence can alter the proportional slope of reliability evidence but cannot create a superproportional magnitude-by-coherence gain. Integration is determined by comparative channel strength rather than unrestricted summation or precision averaging. Weak reliability leaves the count estimate essentially intact. Balanced conflicts are smoothly averaged and therefore reduce confidence. Reliability controls a conflict only after its coherence-adjusted strength clearly exceeds a subject-specific threshold relative to count strength. Congruent reliability is admitted incrementally according to the same comparative test, so weak agreement neither automatically boosts confidence nor dilutes the precise tally estimate. Stable continuous individual differences in count precision, contrast sensitivity, reliability gain, arbitration threshold, response precision, and semantic-polarity confidence generate heterogeneous but internally consistent choice patterns.

**Rationale:** This is a deliberately staged one-line parameter edit to the accepted iteration-1 model. The lower and upper bounds of arbitration_coherence_floor are raised from [0.28, 0.60] to [0.40, 0.72]. Moderate but nonzero reliability evidence can therefore participate more strongly in genuine count-reliability conflicts without lowering the arbitration threshold, sharpening its transition, changing count amplitude, or globally increasing reliability gain. This directly targets the excessive tally protection in Experiments 8 and 17 while retaining the accepted architecture. The raw-residual contrast band, count trajectory, and diagnostic gate are unchanged, preserving the accepted candidate's near-target nonpositive curvature in Experiment 18 and its near-zero generic density effects. Because arbitration_coherence_floor matters only when both channels are available, this edit leaves pure count and count-tie cases substantially more stable than a global response-precision or reliability-gain change. Its somewhat wider range also modestly increases continuous between-subject variation in conflict resolution. The suggested active-dispersion addition is deferred: in repeated high/low cue-pair designs, active validity dispersion can remain constant as support expands, so adding that statistic now would not reliably repair Experiment 16 and would risk disturbing the already successful Experiment 18 constraint.

**Parameters:**
  - `validities`: `validities`
  - `count_gain`: `[1.45, 2.20]`
  - `count_scale`: `[7.0, 12.0]`
  - `count_precision`: `[0.70, 2.60]`
  - `count_precision_floor`: `[0.72, 0.94]`
  - `count_precision_half`: `[0.55, 1.60]`
  - `reliability_gain`: `[1.20, 4.20]`
  - `reliability_floor`: `[0.16, 0.34]`
  - `coherence_gain`: `[0.24, 0.62]`
  - `coherence_half`: `[0.20, 0.60]`
  - `contrast_sensitivity`: `[0.05, 0.58]`
  - `contrast_half`: `[0.55, 1.20]`
  - `contrast_band_low`: `[0.04, 0.20]`
  - `contrast_band_high`: `[0.65, 1.80]`
  - `gate_minimum`: `[0.08, 0.18]`
  - `gate_maximum`: `[0.68, 1.05]`
  - `arbitration_coherence_floor`: `[0.40, 0.72]`
  - `arbitration_threshold`: `[1.05, 1.85]`
  - `arbitration_temperature`: `[0.28, 0.72]`
  - `congruent_admission`: `[0.28, 0.72]`
  - `decision_bound`: `[3.2, 5.8]`
  - `response_precision`: `[0.55, 1.75]`
  - `adverse_polarity_confidence`: `[0.74, 0.995]`
  - `lapse_rate`: `[0.0, 0.12]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Bounded Diagnostic-Gating with Comparative Arbitration expects "
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
    count_magnitude = abs(count_difference)

    # A signed tally accumulator with an approximately linear ordinary range.
    # Its input contains neither active-cue count nor display density.
    count_scale = float(parameters["count_scale"])
    count_signal = (
        float(parameters["count_gain"])
        * count_scale
        * np.tanh(count_difference / count_scale)
    )

    # Count reliability has a high margin-one floor. Precision affects which
    # channel controls arbitration rather than multiplying the count estimate.
    count_access = count_magnitude / (
        count_magnitude + float(parameters["count_precision_half"]) + 1e-12
    )
    count_certainty = float(parameters["count_precision"]) * (
        float(parameters["count_precision_floor"])
        + (1.0 - float(parameters["count_precision_floor"])) * count_access
    )
    if count_magnitude <= 1e-12:
        count_certainty = 0.0

    # Centered validity evidence is the reliability information not already
    # represented by a simple count difference.
    centered_validities = validities - float(np.mean(validities))
    signed_components = centered_validities * difference
    reliability_residual = float(np.sum(signed_components))
    residual_magnitude = abs(reliability_residual)
    gross_support = float(np.sum(np.abs(signed_components)))

    if gross_support > 1e-12:
        coherence = residual_magnitude / gross_support
    else:
        coherence = 0.0
    coherence = float(np.clip(coherence, 0.0, 1.0))

    # Active-field contrast is measured relative to the communicated profile.
    # It is not a proxy for the number or fraction of active features.
    active = np.abs(difference) > 0.5
    profile_rms = float(np.sqrt(np.mean(centered_validities ** 2)))
    if np.any(active):
        active_rms = float(
            np.sqrt(np.mean(centered_validities[active] ** 2))
        )
    else:
        active_rms = 0.0

    if profile_rms > 1e-12:
        relative_contrast = active_rms / profile_rms
    else:
        relative_contrast = 0.0

    contrast_index = relative_contrast / (
        relative_contrast + float(parameters["contrast_half"]) + 1e-12
    )
    contrast_index = float(np.clip(contrast_index, 0.0, 1.0))

    # Coherence supplies only a bounded additive increment. It is never
    # multiplied by an increasing magnitude-confidence term, precluding the
    # superproportional magnitude-by-coherence interaction.
    coherence_access = coherence / (
        coherence + float(parameters["coherence_half"]) + 1e-12
    )

    # Contrast matters most for weak-to-moderate residuals. Its contribution
    # is bounded and tends back toward zero at large magnitudes rather than
    # compounding with residual size.
    low_half = float(parameters["contrast_band_low"])
    high_half = float(parameters["contrast_band_high"])
    contrast_band = (
        residual_magnitude / (residual_magnitude + low_half + 1e-12)
    ) * (
        high_half / (residual_magnitude + high_half + 1e-12)
    )
    centered_contrast = 2.0 * contrast_index - 1.0

    diagnostic_gate = (
        float(parameters["reliability_floor"])
        + float(parameters["coherence_gain"]) * coherence_access
        + float(parameters["contrast_sensitivity"])
        * centered_contrast
        * contrast_band
    )
    diagnostic_gate = float(
        np.clip(
            diagnostic_gate,
            float(parameters["gate_minimum"]),
            float(parameters["gate_maximum"]),
        )
    )

    if residual_magnitude <= 1e-12:
        reliability_signal = 0.0
    else:
        reliability_signal = (
            float(parameters["reliability_gain"])
            * diagnostic_gate
            * reliability_residual
        )

    # Comparative strengths determine access to the final decision. An
    # incoherent reliability estimate must be larger to defeat the count.
    count_strength = count_certainty * abs(count_signal)
    reliability_strength = abs(reliability_signal) * (
        float(parameters["arbitration_coherence_floor"])
        + (1.0 - float(parameters["arbitration_coherence_floor"]))
        * coherence_access
    )

    if count_strength <= 1e-12 and reliability_strength <= 1e-12:
        integrated_signal = 0.0
    elif count_strength <= 1e-12:
        integrated_signal = reliability_signal
    elif reliability_strength <= 1e-12:
        integrated_signal = count_signal
    else:
        ratio = reliability_strength / (count_strength + 1e-12)
        threshold = float(parameters["arbitration_threshold"])
        temperature = float(parameters["arbitration_temperature"])
        log_ratio = np.log(ratio + 1e-12) - np.log(threshold)
        reliability_control = 1.0 / (
            1.0 + np.exp(-np.clip(log_ratio / temperature, -40.0, 40.0))
        )

        if count_signal * reliability_signal < 0.0:
            # Balanced conflicts are averaged. Count controls weak-reliability
            # conflicts; reliability reverses choice only after clear relative
            # dominance.
            integrated_signal = (
                (1.0 - reliability_control) * count_signal
                + reliability_control * reliability_signal
            )
        elif count_signal * reliability_signal > 0.0:
            # Congruent reliability is admitted only in proportion to its
            # comparative diagnostic strength. Weak agreement leaves the
            # precise tally estimate nearly unchanged and never dilutes it.
            integrated_signal = count_signal + (
                float(parameters["congruent_admission"])
                * reliability_control
                * reliability_signal
            )
        else:
            integrated_signal = count_signal + reliability_signal

    # Bounded decisional returns preserve ordinary-margin growth while
    # preventing extreme displays from producing unlimited confidence.
    decision_bound = float(parameters["decision_bound"])
    integrated_signal = decision_bound * np.tanh(
        integrated_signal / decision_bound
    )
    markedness_evidence = (
        float(parameters["response_precision"]) * integrated_signal
    )

    # Positive markedness means A carries more adverse evidence and favors B.
    adverse_logits = np.array(
        [-0.5 * markedness_evidence, 0.5 * markedness_evidence], dtype=float
    )
    adverse_logits -= np.max(adverse_logits)
    adverse_probs = np.exp(adverse_logits)
    adverse_probs /= adverse_probs.sum()

    # Polarity is stable over the subject's run but varies continuously across
    # subjects. The alternative interpretation reverses the full decision.
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
