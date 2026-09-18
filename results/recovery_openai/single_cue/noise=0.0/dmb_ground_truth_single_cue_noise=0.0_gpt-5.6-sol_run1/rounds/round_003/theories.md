# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Rationale:** This is a minimal extension of the accepted recency-overwrite model. The cue scan and overwrite equations are unchanged. The only mechanistic addition is a discriminating-cue-count fatigue gate that begins after the second reason, together with a broader and slightly lower lapse range. Two-reason trials therefore retain the strong overwrite needed for Experiment 2, whereas long reason sequences are pulled modestly toward uncertainty, reducing the excessive Experiment-1 contrast. Lower average lapse simultaneously strengthens anti-TTB behavior in Experiment 2. Broad subject-level ranges for fatigue and lapse should also increase the previously underestimated between-subject variability. The gate is task-invariant and depends only on processing load, not experiment identity or stimulus-specific switches.

**Parameters:**
  - `validities`: `validities`
  - `cue_attention`: `[0.95, 1.0]`
  - `overwrite_rate`: `[0.95, 1.0]`
  - `update_fatigue`: `[0.06, 0.17]`
  - `lapse_rate`: `[0.15, 0.41]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recency-Gated Reason Overwrite expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["cue_attention"])
    overwrite = float(parameters["overwrite_rate"])
    fatigue = float(parameters["update_fatigue"])
    lapse = float(parameters["lapse_rate"])

    # Stable sorting represents the displayed high-to-low validity scan;
    # ties retain their display order.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]

    # Exact marginal distribution over the provisional state:
    # A selected, B selected, or no discriminating cue encoded yet.
    p_a, p_b, p_uncommitted = 0.0, 0.0, 1.0
    discriminating_count = 0

    for j in cue_order:
        direction = diff[j]
        if direction == 0:
            continue

        discriminating_count += 1
        old_a, old_b, old_u = p_a, p_b, p_uncommitted

        if direction > 0:
            # An attended A cue initializes an uncommitted decision and
            # overwrites a conflicting B decision with the overwrite rate.
            p_a = old_a + attention * overwrite * old_b + attention * old_u
            p_b = old_b * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)
        else:
            p_b = old_b + attention * overwrite * old_a + attention * old_u
            p_a = old_a * (1.0 - attention * overwrite)
            p_uncommitted = old_u * (1.0 - attention)

    # If no discriminating cue was encoded, either response is an equal guess.
    p_core = np.array([
        p_a + 0.5 * p_uncommitted,
        p_b + 0.5 * p_uncommitted
    ], dtype=np.float64)

    # Processing the first reason and its immediate successor is reliable.
    # Additional discriminating updates progressively weaken expression of
    # the retained choice without disrupting the crucial second-cue reversal.
    excess_updates = max(0, discriminating_count - 2)
    retention = (1.0 - fatigue) ** excess_updates
    p_core = retention * p_core + (1.0 - retention) * np.array([0.5, 0.5])

    # Independent execution/inattention lapse.
    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Saturated Directional-Run Chunking with Boundary Interference. People scan discriminating cues in their instructed order and compress consecutive cues favoring the same option into a single directional chunk. Chunk confidence rises rapidly with run length but saturates, so two agreeing terminal cues can be nearly as effective as a much longer terminal run. At each directional boundary, the newly completed chunk suppresses the accumulated representation of earlier chunks. A coherent terminal run receives an additional replacement advantage, causing stimuli with the same strong suffix to converge despite differences in prefix segmentation. Short four-reason sequences preserve run structure at high temporal resolution and express terminal chunks with a heterogeneous gain, producing strong but individually variable terminal control. In longer sequences, switch-dense patterns suffer source interference, while long low-switch conflicts undergo a separate modest loss of response precision without recovering early high-validity cues. Instructed validity only modestly scales chunk encoding and cannot become the retrieval asymptote. Choices arise from recency-weighted competition among run summaries plus a small response lapse.

**Rationale:** This is a two-range calibration of the accepted model; all equations and successful mechanisms are unchanged. The long-low-switch loss is strengthened and widened from [0.05, 0.20] to [0.08, 0.34]. This selectively reduces the overly extreme choices produced by coherent seven-plus-cue conflicts in Experiment 1 and increases subject heterogeneity, while remaining moderate enough to move Experiment 6 only slightly toward chance. Switch interference is increased from [1.5, 2.5] to [1.8, 2.8], adding uncertainty only after more than two directional boundaries and thereby targeting Experiment 3 without weakening the coherent-suffix mechanism responsible for the strong Experiment 5 fit. The four-reason gain, suffix replacement, generic length interference, and validity encoding are preserved to protect the accepted fits in Experiments 2, 4, and 5.

**Parameters:**
  - `validities`: `validities`
  - `run_saturation`: `[1.8, 3.2]`
  - `boundary_suppression`: `[0.35, 0.55]`
  - `chunk_recency_decay`: `[0.78, 0.94]`
  - `short_resolution_boost`: `[0.12, 0.68]`
  - `suffix_replacement`: `[0.08, 0.25]`
  - `four_reason_expression_gain`: `[0.35, 2.60]`
  - `switch_interference`: `[1.8, 2.8]`
  - `length_interference`: `[0.06, 0.12]`
  - `long_low_switch_loss`: `[0.08, 0.34]`
  - `validity_encoding_scale`: `[0.02, 0.20]`
  - `choice_gain`: `[2.6, 3.8]`
  - `lapse_rate`: `[0.04, 0.14]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Directional-run chunking expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    saturation = float(parameters["run_saturation"])
    suppression = float(parameters["boundary_suppression"])
    recency_decay = float(parameters["chunk_recency_decay"])
    short_boost = float(parameters["short_resolution_boost"])
    suffix_replacement = float(parameters["suffix_replacement"])
    four_gain = float(parameters["four_reason_expression_gain"])
    switch_interference = float(parameters["switch_interference"])
    length_interference = float(parameters["length_interference"])
    long_low_switch_loss = float(parameters["long_low_switch_loss"])
    validity_scale = float(parameters["validity_encoding_scale"])
    choice_gain = float(parameters["choice_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_indices = np.flatnonzero(diff != 0)
    m = int(cue_indices.size)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[cue_indices]).astype(float)
    cue_validities = np.clip(validities[cue_indices], 0.5, 1.0)

    # Validity changes encoding strength only modestly. Every discriminating
    # cue has a substantial baseline representation, preventing overload from
    # turning retrieval into a validity-biased early-cue process.
    encoded_strengths = 1.0 + validity_scale * (2.0 * cue_validities - 1.0)

    # Compress contiguous cues with a common direction into run summaries.
    runs = []
    start = 0
    for i in range(1, m + 1):
        if i == m or directions[i] != directions[start]:
            direction = float(directions[start])
            encoding_mass = float(np.sum(encoded_strengths[start:i]))
            # Repetition increases chunk confidence, but confidence rapidly
            # saturates rather than growing as an unbounded cue tally.
            confidence = 1.0 - np.exp(-saturation * encoding_mass)
            runs.append((direction, confidence, i - start))
            start = i

    n_switches = max(0, len(runs) - 1)

    # Compete run summaries online. A new coherent chunk both enters memory
    # and suppresses prior chunks. Short sequences retain sharper boundary
    # resolution, yielding especially effective terminal replacement.
    directional_memory = 0.0
    represented_mass = 0.0
    for run_number, (direction, confidence, run_length) in enumerate(runs):
        if run_number == 0:
            directional_memory = direction * confidence
            represented_mass = confidence
            continue

        effective_suppression = suppression
        if m <= 4:
            effective_suppression += short_boost
        # A repeated-direction suffix is encoded as a completed terminal
        # decision chunk and therefore more fully replaces prefix summaries.
        if run_number == len(runs) - 1 and run_length >= 2:
            effective_suppression += suffix_replacement
        effective_suppression = float(np.clip(effective_suppression, 0.0, 0.98))

        retention = recency_decay * (1.0 - effective_suppression * confidence)
        retention = float(np.clip(retention, 0.0, 1.0))
        directional_memory = retention * directional_memory + direction * confidence
        represented_mass = retention * represented_mass + confidence

    if represented_mass <= 0.0 or not np.isfinite(represented_mass):
        normalized_direction = 0.0
    else:
        normalized_direction = float(
            np.clip(directional_memory / represented_mass, -1.0, 1.0)
        )

    # Switch-dense sequences lose directional source resolution. Generic
    # length uncertainty remains mild, while long low-switch conflicts have
    # a separate expression loss that attenuates extremity without restoring
    # privileged access to early cues.
    excess_switches = max(0, n_switches - 2)
    low_switch_excess = max(0, m - 5) if n_switches <= 2 else 0
    precision_denominator = (
        1.0
        + switch_interference * float(excess_switches ** 2)
        + length_interference * float(max(0, m - 4))
        + long_low_switch_loss * float(low_switch_excess)
    )
    effective_gain = choice_gain / precision_denominator

    # Four-reason displays preserve unusually high temporal resolution. The
    # broad subject-level gain range captures heterogeneity in expressing the
    # terminal chunk without changing longer conflicts.
    if m == 4:
        effective_gain *= four_gain

    logit_a = float(np.clip(effective_gain * normalized_direction, -60.0, 60.0))
    p_a_core = 1.0 / (1.0 + np.exp(-logit_a))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Validity-Ordered Competitive Evidence Buffer with Conflict-Normalized Readout. Decision makers inspect cues in descending order of instructed validity and encode every discriminating cue as signed evidence rather than as a provisional choice or directional chunk. Each newly encoded reason causes graded displacement of all earlier traces in a capacity-limited working-memory buffer. Displacement becomes stronger as the buffer exceeds its stable subject-specific capacity, but old evidence is never categorically erased. At response, the decision maker sums all surviving validity-scaled traces. Readout normalization depends on the balance of accessible evidence: it is strongest when substantial trace mass supports both options and weak when the accessible buffer contains a coherent majority. Readout uncertainty additionally grows quadratically with capacity overload and receives a count-based attenuation beginning explicitly at seven discriminating reasons, contracting long conflicts toward chance without imposing arrangement-specific effects. Stable individual differences in effective capacity, displacement rate, validity sensitivity, response gain, and conflict normalization generate heterogeneous recency and integration behavior.

**Rationale:** This is a one-line calibration edit to the accepted iteration-9 model: the response_gain range is broadened downward from [3.0, 6.5] to [2.0, 6.5]. All trace encoding, validity-ranked inspection, displacement, capacity overload, conflict-sensitive normalization, and seven-plus uncertainty equations remain unchanged. The lower mean gain provides the modest contraction toward chance jointly indicated by Experiments 1, 2, 3, 4, and 6: it should reduce excessive tally and terminal effects while raising the currently underpredicted below-chance rates in Experiments 2 and 6. The added low-gain subjects also provide a direct source of stable response-expression heterogeneity, especially around the four-reason conflict in Experiment 4, without reopening validity or displacement heterogeneity. Retaining the previous upper bound preserves strongly recency-controlled individuals, while the accepted conflict-sensitive normalization continues to protect coherent anti-terminal majorities in Experiment 8 better than a generic increase in normalization would. No run, suffix, switch-density, display-position, or categorical-overwrite mechanism is introduced.

**Parameters:**
  - `validities`: `validities`
  - `buffer_capacity`: `[3.0, 3.8]`
  - `displacement_rate`: `[2.1, 2.9]`
  - `validity_sensitivity`: `[0.0, 0.08]`
  - `response_gain`: `[2.0, 6.5]`
  - `integration_normalization`: `[0.02, 0.34]`
  - `overflow_noise`: `[0.75, 1.50]`
  - `lapse_rate`: `[0.03, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive evidence buffer expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["buffer_capacity"])
    displacement = float(parameters["displacement_rate"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    response_gain = float(parameters["response_gain"])
    integration_normalization = float(parameters["integration_normalization"])
    overflow_noise = float(parameters["overflow_noise"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]

    # Inspection follows instructed validity, independently of physical
    # feature position. Stable sorting only resolves exact validity ties.
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Convert instructed reliability into evidence strength. Normalization
    # removes arbitrary differences in the overall validity scale between
    # experiments while preserving all within-experiment validity ratios.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    positive_reliability = reliability[reliability > 0.0]
    reliability_scale = float(np.mean(positive_reliability))
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0

    normalized_reliability = np.clip(reliability / reliability_scale, 1e-4, 20.0)
    cue_strength = np.power(normalized_reliability, validity_sensitivity)
    cue_strength = np.clip(cue_strength, 1e-6, 30.0)

    # The buffer contains separate reason traces. A new discriminating reason
    # interferes continuously with every old trace. Displacement is present
    # even below capacity and increases smoothly under overload; there is no
    # categorical slot deletion and no compression into directional runs.
    traces = []
    for j in discriminating:
        current_load = float(len(traces))
        overload_before_entry = max(0.0, current_load + 1.0 - capacity)
        load_factor = 1.0 + overload_before_entry / max(capacity, 1e-6)
        retention = np.exp(-displacement * load_factor / max(capacity, 1e-6))
        retention = float(np.clip(retention, 0.0, 1.0))

        if traces:
            traces = [retention * trace for trace in traces]

        direction = float(np.sign(diff[j]))  # positive favors A
        traces.append(direction * float(cue_strength[j]))

    accessible_evidence = float(np.sum(traces))

    # Preserve the accepted quadratic capacity-overload uncertainty through
    # six reasons. Beginning explicitly at seven discriminating reasons, add
    # a modest count-based attenuation that is not smeared by capacity.
    overflow = max(0.0, float(m) - capacity)
    long_sequence_excess = max(0.0, float(m) - 6.0)
    convex_multiplier = 1.0 + 0.45 * long_sequence_excess
    readout_sd = np.sqrt(
        1.0 + overflow_noise * overflow * overflow * convex_multiplier
    )

    # Normalize integrated evidence only when accessible traces substantially
    # support both options. Squaring the balance ratio leaves coherent
    # majorities largely intact. The effect remains absent for two reasons.
    positive_mass = float(np.sum([trace for trace in traces if trace > 0.0]))
    negative_mass = float(-np.sum([trace for trace in traces if trace < 0.0]))
    total_mass = positive_mass + negative_mass
    if m <= 2 or total_mass <= 0.0 or not np.isfinite(total_mass):
        conflict_load = 0.0
    else:
        balance = 2.0 * min(positive_mass, negative_mass) / total_mass
        balance = float(np.clip(balance, 0.0, 1.0))
        conflict_load = np.sqrt(max(0.0, float(m) - 2.0)) * balance * balance
    integration_divisor = 1.0 + integration_normalization * conflict_load

    decision_variable = (
        response_gain * accessible_evidence
        / (readout_sd * integration_divisor)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a_core = 1.0 / (1.0 + np.exp(-decision_variable))
    p_core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)

    probs = (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
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
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
