# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

**Description:** Discrepancy-Evoked Residual Audit with Capacity-Limited Tie Escalation: people first form an adverse-markedness preference from a nonlinear count difference, while count-residualized validity information remains continuously available. Count discrepancies evoke an additional validity audit, but exact count ties trigger a distinct escalation because tallying provides no provisional winner. This tie escalation is strongest when the cue set fits within a fixed attentional capacity and rapidly dilutes as the number of cues exceeds that capacity. Stable subject differences in tie escalation complement existing differences in count reliance, discrepancy auditing, semantic polarity, precision, and lapses.

**Rationale:** This is a minimal extension of the accepted candidate: it adds only a tie-specific validity bonus and its parameter, leaving every accepted count branch, the continuous validity floor, the discrepancy audit, and the response process unchanged. The bonus acts only when counts tie, so it can reduce Experiment 1's excessive tie alignment without altering its conflict trials or Experiment 5's already accurate one-mark sensitivity. A fixed seven-cue attentional-capacity rule makes the same cognitive mechanism strong in Experiment 1's compact seven-cue display but attenuates it by exp(-4) in Experiment 6's eleven-cue display, protecting the latter's current approximately 0.11 slope. The wide, mean-centered tie-bonus range adds stable subject heterogeneity specifically to the deficient tie-processing channel without repeating rejected broad changes to polarity, precision, or lapse. Experiments 3 and 4 should remain protected because their scored trials are non-ties and all previously accepted count-margin parameters are unchanged.

**Parameters:**
  - `validities`: `validities`
  - `count_strength`: `[2.1, 3.5]`
  - `count_scale`: `[1.2, 2.0]`
  - `small_margin_attenuation`: `[0.72, 0.92]`
  - `intermediate_count_reinforcement`: `[1.6, 3.4]`
  - `extreme_count_tail`: `[1.0, 2.8]`
  - `baseline_validity_gain`: `[0.38, 0.88]`
  - `tie_validity_bonus`: `[0.05, 1.05]`
  - `audit_gain`: `[0.9, 1.6]`
  - `audit_decay`: `[0.7, 1.4]`
  - `margin_three_audit_gain`: `[0.0, 0.1]`
  - `response_precision`: `[0.75, 1.4]`
  - `adverse_polarity_confidence`: `[0.74, 1.0]`
  - `lapse_rate`: `[0.0, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
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

### `pi_6` → slot 1 (via `new_theory`)

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
