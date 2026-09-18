# Round 6 — Theories

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


### slot 2 — `pi_8` — KILLED ✗

**Description:** Context-Calibrated Ordinal Event Policies. Decision makers inspect experts in instructed-validity order but encode the resulting discriminating sequence as ordinal events: an initial commitment, coherent directional runs, reversals, and a terminal commitment. Choice is generated by a context-sensitive selection between two subject-stable policies. The default terminal policy expresses the most recently attended commitment, whereas a coherent-chunk policy preserves the initial commitment when a moderately long, densely populated display is naturally represented as two opposing blocks. The chunk policy is recruited by conjunctions of density, coherent run structure, manageable load, and boundary regularities learned across trials—not by raw switch count alone. Sparse late suffixes or isolated terminal opposition foreground the terminal event and increase its confidence, even at high nominal feature counts. Equal-weight tally agreement can strengthen or weaken confidence in an already selected commitment but cannot reverse that commitment. Communicated validities determine inspection order and contribute only a strongly compressed confidence adjustment. Stable individual differences govern attention, recency and primacy propensities, chunk-switch threshold, contextual calibration, response gain, and lapse. In non-foregrounded moderate-load sequences, replacement of an existing commitment is slightly less complete, allowing preceding ordinal commitments to persist without changing policy selection.

**Rationale:** This is a single-parameter-range edit to the accepted iteration-6 model. All policy-selection, context-learning, event-replacement, tally, validity, and load equations are unchanged. The confidence_gain range shifts from [1.75, 2.65] to [1.60, 2.50], lowering its mean by about 7% without widening subject heterogeneity. This conservatively reduces excessive decisiveness: it should lower terminal agreement in Experiment 3 and the tally-support contrast in Experiment 1, while moving the below-chance initial, early, and majority-choice rates in Experiments 2, 6, and 12 toward their observed values. The edit does not alter which ordinal policy is selected, does not introduce sensitivity to raw switch count, and preserves the accepted moderate-load persistence mechanism. It may mildly weaken the close terminal fits in Experiments 10 and 11 and cannot by itself solve Experiment 8's policy-selection residual, so the shift is deliberately small.

**Parameters:**
  - `validities`: `validities`
  - `attention_rate`: `[0.91, 0.99]`
  - `recency_propensity`: `[0.88, 1.0]`
  - `primacy_propensity`: `[0.76, 0.94]`
  - `chunk_switch_threshold`: `[5.05, 5.65]`
  - `context_sensitivity`: `[0.35, 0.75]`
  - `load_tolerance`: `[4.4, 5.8]`
  - `confidence_gain`: `[1.60, 2.50]`
  - `tally_confidence`: `[0.05, 0.22]`
  - `validity_compression`: `[0.0, 0.10]`
  - `lapse_rate`: `[0.05, 0.17]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Ordinal event policy expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["attention_rate"])
    recency_propensity = float(parameters["recency_propensity"])
    primacy_propensity = float(parameters["primacy_propensity"])
    chunk_threshold = float(parameters["chunk_switch_threshold"])
    context_sensitivity = float(parameters["context_sensitivity"])
    load_tolerance = float(parameters["load_tolerance"])
    confidence_gain = float(parameters["confidence_gain"])
    tally_confidence = float(parameters["tally_confidence"])
    validity_compression = float(parameters["validity_compression"])
    lapse = float(parameters["lapse_rate"])

    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    def sequence_summary(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size or a.size == 0:
            return None
        d = a - b
        idx = np.flatnonzero(d != 0)
        if idx.size == 0:
            return None
        ds = np.sign(d[idx]).astype(float)
        run_lengths = []
        current = 1
        for k in range(1, ds.size):
            if ds[k] == ds[k - 1]:
                current += 1
            else:
                run_lengths.append(current)
                current = 1
        run_lengths.append(current)
        gaps = np.diff(idx) if idx.size > 1 else np.asarray([], dtype=float)
        max_gap = float(np.max(gaps) - 1.0) if gaps.size else 0.0
        max_gap = max(0.0, max_gap)
        return {
            "m": float(idx.size),
            "density": float(idx.size) / float(a.size),
            "n_runs": float(len(run_lengths)),
            "boundary": float(run_lengths[0]) / float(idx.size),
            "opposition": float(ds[0] != ds[-1]),
            "foreground": max_gap / float(max(1, a.size - 1)),
            "terminal_at_edge": float(idx[-1] == a.size - 1)
        }

    summaries = []
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        summary = sequence_summary(past_a[i], past_b[i])
        if summary is not None:
            summaries.append(summary)

    current_summary = sequence_summary(stim[0], stim[1])
    if current_summary is not None:
        summaries.append(current_summary)

    # Context statistics are based on structural distributions, not choices.
    # Shrink early estimates toward neutral priors so the model behaves
    # sensibly before it has sampled the task's display distribution.
    if summaries:
        sample_weight = min(1.0, len(summaries) / 12.0)
        opposition_rate_raw = float(np.mean([s["opposition"] for s in summaries]))
        foreground_rate_raw = float(np.mean([
            s["foreground"] * s["terminal_at_edge"] for s in summaries
        ]))
        coherent_boundaries = [
            s["boundary"] for s in summaries
            if s["density"] >= 0.85 and s["n_runs"] == 2.0
        ]
        if len(coherent_boundaries) >= 2:
            boundary_diversity_raw = min(
                1.0, 4.0 * float(np.std(coherent_boundaries))
            )
        else:
            boundary_diversity_raw = 0.0
        opposition_rate = (
            sample_weight * opposition_rate_raw + (1.0 - sample_weight) * 0.5
        )
        foreground_rate = sample_weight * foreground_rate_raw
        boundary_diversity = sample_weight * boundary_diversity_raw
    else:
        opposition_rate = 0.5
        foreground_rate = 0.0
        boundary_diversity = 0.0

    density = float(m) / float(max(1, n_features))

    run_lengths = []
    current_run = 1
    for k in range(1, m):
        if directions[k] == directions[k - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    n_runs = len(run_lengths)
    coherent_two_block = 1.0 if n_runs == 2 else 0.0

    physical_positions = np.sort(np.asarray(discriminating, dtype=int))
    if physical_positions.size > 1:
        empty_gaps = np.diff(physical_positions) - 1
        largest_gap = max(0.0, float(np.max(empty_gaps)))
    else:
        largest_gap = 0.0
    foreground = largest_gap / float(max(1, n_features - 1))
    terminal_at_edge = float(np.max(physical_positions) == n_features - 1)
    terminal_foreground = foreground * terminal_at_edge

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # A coherent-chunk policy is available only in a moderate-load window.
    # Both sides of the window are smooth and subject-specific. Dense
    # two-block structure is a necessary conjunction, so merely adding
    # switches cannot recruit primacy. Across-trial boundary diversity makes
    # block comparison somewhat more likely without making it mandatory.
    chunk_onset = sigmoid((float(m) - chunk_threshold) / 0.35)
    chunk_offset = sigmoid((chunk_threshold + 1.35 - float(m)) / 0.35)
    density_gate = sigmoid((density - 0.88) / 0.045)
    context_factor = 0.72 + context_sensitivity * boundary_diversity
    primacy_gate = (
        primacy_propensity
        * coherent_two_block
        * chunk_onset
        * chunk_offset
        * density_gate
        * context_factor
    )
    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))

    # Repeated sparse terminal opposition suppresses chunk recruitment and
    # restores terminal capture. This is especially important when a late
    # suffix is separated from the initial block by nondiscriminating cues.
    terminal_context = np.clip(
        terminal_foreground
        + context_sensitivity * foreground_rate
        + 0.20 * opposition_rate * terminal_foreground,
        0.0,
        1.0
    )
    primacy_gate *= (1.0 - 0.85 * terminal_context)
    primacy_gate = float(np.clip(primacy_gate, 0.0, 0.96))

    # Replacement is modestly less complete in the subject's moderate-load
    # window. Density scales the adjustment smoothly, while foregrounded
    # terminal suffixes retain the baseline near-complete replacement rate.
    replacement_onset = sigmoid(
        (float(m) - (chunk_threshold - 0.50)) / 0.30
    )
    replacement_offset = sigmoid(
        (chunk_threshold + 1.00 - float(m)) / 0.30
    )
    moderate_replacement_window = replacement_onset * replacement_offset
    density_modulation = 0.70 + 0.30 * density
    replacement_reduction = (
        0.10
        * moderate_replacement_window
        * density_modulation
        * (1.0 - terminal_context)
    )
    effective_replacement = float(np.clip(
        attention - replacement_reduction, 0.0, 1.0
    ))

    # Exact expected direction of the most recently attended commitment.
    # A later attended event replaces the previous one; unattended events do
    # not. The no-commitment state contributes zero directional evidence.
    recency_direction = 0.0
    for direction in directions:
        recency_direction = (
            (1.0 - effective_replacement) * recency_direction
            + effective_replacement * direction
        )

    # Exact expected direction of the first attended commitment.
    primacy_direction = 0.0
    probability_uncommitted = 1.0
    for direction in directions:
        encoded_here = probability_uncommitted * attention
        primacy_direction += encoded_here * direction
        probability_uncommitted *= (1.0 - attention)

    recency_direction *= recency_propensity

    # Nominal display load weakens expression, while coherent runs and a
    # foregrounded late suffix restore confidence. Run coherence affects
    # confidence, not policy recruitment, avoiding a raw-switch-count rule.
    longest_run_fraction = float(max(run_lengths)) / float(m)
    display_overload = max(0.0, float(n_features) - load_tolerance)
    overload_divisor = 1.0 + 0.22 * display_overload
    overload_divisor *= 1.0 + 0.10 * max(0.0, float(m) - load_tolerance)

    coherence_support = max(0.0, longest_run_fraction - 0.35)
    confidence_multiplier = 1.0 + 0.90 * coherence_support
    confidence_multiplier += 3.0 * terminal_foreground
    confidence_multiplier += 1.25 * context_sensitivity * foreground_rate

    # Validity magnitudes have only compressed influence. Their main role was
    # already to determine inspection order.
    v = np.clip(validities[discriminating], 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    mean_log_odds = max(float(np.mean(log_odds)), 1e-8)
    terminal_reliability = max(float(log_odds[-1] / mean_log_odds), 1e-6)
    validity_factor = float(
        np.power(np.clip(terminal_reliability, 0.25, 4.0), validity_compression)
    )

    # Tally is a confidence cue only. It can reinforce or attenuate the
    # terminal commitment but cannot change its sign or install a weighted
    # global coalition as the selected response.
    tally_direction = float(np.sign(np.sum(directions)))
    terminal_direction = float(directions[-1])
    if tally_direction == 0.0:
        tally_factor = 1.0
    elif tally_direction == terminal_direction:
        tally_factor = 1.0 + tally_confidence
    else:
        tally_factor = 1.0 - 0.30 * tally_confidence

    gain = (
        confidence_gain
        * confidence_multiplier
        * validity_factor
        * tally_factor
        / max(overload_divisor, 1e-8)
    )
    gain = float(np.clip(gain, 0.05, 12.0))

    p_a_recency = sigmoid(gain * recency_direction)
    p_a_primacy = sigmoid(gain * primacy_direction)
    p_a_core = (
        (1.0 - primacy_gate) * p_a_recency
        + primacy_gate * p_a_primacy
    )

    core = np.array([p_a_core, 1.0 - p_a_core], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])
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

### `pi_9` → slot 2 (via `new_theory`)

**Description:** Distribution-Calibrated Competitive Accumulation with Susceptibility-Dominant Context Learning. People inspect discriminating cues in descending instructed-validity order and encode each attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the active commitment, while a continuous experiment-level task set controls retention and overwrite. The task set is learned primarily from how readily opposing runs reverse a reference accumulator across the experiment's sequence families. Coarse ordinal properties such as terminal singletons, two-run structure, and reversal-location concentration provide only weak, centered calibration rather than directly producing overwrite. Reliability-weighted coalitions can reverse the sequential state, but their access to choice is proportional to normalized evidence margin. Physical gaps are not represented, and reliability depends on conflict geometry and evidence margin rather than discriminating-cue count.

**Rationale:** This is a minimal edit of the accepted iteration-7 candidate. The arithmetic model-implied reversal-susceptibility statistic, reference accumulator, learning rate, bounded current-trial contribution, static validity weighting, coalition-margin gate, local competition rule, response mapping, parameter ranges, and exclusion of physical gaps are unchanged. Only the context-score mapping is revised. The direct coefficients on terminal-singleton rate, reversal concentration, and two-run frequency are reduced from 3.0, 3.0, and 2.0 to 1.2, 1.2, and 0.8; entropy and switch-density effects are also conservatively reduced. These coarse statistics are centered around neutral reference values, with a 1.30 intercept preserving the accepted model's typical operating level rather than causing a global collapse in overwrite. Consequently, merely having a two-block or terminal-singleton display should no longer create excessive overwrite in Experiments 6 and 8 or a large within-task contrast in Experiment 5. Cross-experiment differentiation is instead carried more proportionally by the already accepted reversal-susceptibility term, which asks whether opposing runs are strong enough to overcome an incumbent in sequential evidence units. No upper-tail or terminal-weighted aggregation is introduced because that change was rejected. The edit therefore targets context miscalibration without globally changing terminal competition or risking the accepted fits in Experiments 7, 10, 12, and 14.

**Parameters:**
  - `validities`: `validities`
  - `attention_rate`: `[0.88, 1.0]`
  - `integration_retention`: `[0.84, 0.97]`
  - `overwrite_retention`: `[0.14, 0.36]`
  - `conflict_inhibition`: `[0.35, 1.20]`
  - `coalition_weight`: `[0.30, 0.62]`
  - `validity_sensitivity`: `[0.15, 0.45]`
  - `context_sensitivity`: `[1.4, 2.4]`
  - `context_threshold`: `[1.0, 1.6]`
  - `context_learning_rate`: `[0.05, 0.12]`
  - `susceptibility_coupling`: `[0.60, 1.10]`
  - `response_gain`: `[1.8, 3.8]`
  - `lapse_rate`: `[0.04, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Competitive accumulation expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    attention = float(parameters["attention_rate"])
    integration_retention = float(parameters["integration_retention"])
    overwrite_retention = float(parameters["overwrite_retention"])
    conflict_inhibition = float(parameters["conflict_inhibition"])
    coalition_weight = float(parameters["coalition_weight"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    context_sensitivity = float(parameters["context_sensitivity"])
    context_threshold = float(parameters["context_threshold"])
    learning_rate = float(parameters["context_learning_rate"])
    susceptibility_coupling = float(parameters["susceptibility_coupling"])
    response_gain = float(parameters["response_gain"])
    lapse = float(parameters["lapse_rate"])

    cue_order = np.argsort(-validities, kind="stable")

    def ordinal_summary(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
            return None
        d = a - b
        ordered_idx = [int(j) for j in cue_order if j < d.size and d[j] != 0]
        if not ordered_idx:
            return None
        directions = np.asarray([np.sign(d[j]) for j in ordered_idx], dtype=float)
        m = directions.size
        switches = np.flatnonzero(directions[1:] != directions[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if directions[k] == directions[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        if switches.size:
            reversal_fractions = switches.astype(float) / float(m)
        else:
            reversal_fractions = np.asarray([], dtype=float)

        # Estimate reversal susceptibility with the same signed evidence units
        # used by the decision process. At every run boundary, compare the
        # incoming opposing run with the reference-retained incumbent. The
        # smooth score records both actual reversals and near reversals.
        local_v = np.clip(validities, 0.5001, 0.999)
        local_reliability = np.log(local_v / (1.0 - local_v))
        local_scale = float(np.mean(local_reliability[local_reliability > 0.0]))
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = 1.0
        local_normalized = np.clip(local_reliability / local_scale, 0.15, 6.0)
        local_strength = np.power(local_normalized, validity_sensitivity)
        ordered_strength = np.asarray(
            [attention * float(local_strength[j]) for j in ordered_idx],
            dtype=float
        )

        reference_retention = 0.55
        reference_accumulator = 0.0
        susceptibility_scores = []
        k = 0
        while k < m:
            run_direction = float(directions[k])
            run_end = k + 1
            while run_end < m and directions[run_end] == run_direction:
                run_end += 1

            run_evidence = ordered_strength[k:run_end]
            run_length = int(run_end - k)
            if reference_accumulator != 0.0 and np.sign(reference_accumulator) != run_direction:
                retained_incumbent = abs(reference_accumulator) * (
                    reference_retention ** run_length
                )
                attack = 0.0
                for strength in run_evidence:
                    attack = reference_retention * attack + float(strength)
                normalized_advantage = (
                    attack - retained_incumbent
                ) / max(attack + retained_incumbent, 1e-9)
                smooth_reversal = 1.0 / (
                    1.0 + np.exp(-4.0 * np.clip(normalized_advantage, -15.0, 15.0))
                )
                susceptibility_scores.append(float(smooth_reversal))

            for strength in run_evidence:
                reference_accumulator = (
                    reference_retention * reference_accumulator
                    + run_direction * float(strength)
                )
            k = run_end

        if susceptibility_scores:
            reversal_susceptibility = float(np.mean(susceptibility_scores))
        else:
            reversal_susceptibility = 0.35

        return {
            "m": float(m),
            "directions": directions,
            "reversal_fractions": reversal_fractions,
            "terminal_singleton": float(run_lengths[-1] == 1 and len(run_lengths) > 1),
            "two_run": float(len(run_lengths) == 2),
            "switch_density": float(switches.size) / float(max(1, m - 1)),
            "reversal_susceptibility": reversal_susceptibility
        }

    # Learn the task set from ordinal scan sequences only. Physical indices,
    # distances between cues, and nondiscriminating gaps never enter these
    # statistics. Exponential weighting implements continuous contextual
    # learning while retaining a weak neutral prior early in the block.
    summaries = []
    past_a = history.get("option_a_ratings", [])
    past_b = history.get("option_b_ratings", [])
    n_past = min(len(past_a), len(past_b))
    for i in range(n_past):
        s = ordinal_summary(past_a[i], past_b[i])
        if s is not None:
            summaries.append(s)

    current_summary = ordinal_summary(stim[0], stim[1])
    if current_summary is None:
        return np.array([0.5, 0.5], dtype=np.float64)
    summaries.append(current_summary)

    ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
    weights = np.power(np.clip(1.0 - learning_rate, 1e-6, 1.0), ages)
    # The current stimulus can update calibration, but once context exists it
    # receives a bounded weight and cannot sharply change the experiment-level
    # task set solely because of its own conflict geometry.
    if len(summaries) > 1:
        weights[-1] *= 0.35
    weights /= max(float(weights.sum()), 1e-12)

    terminal_singleton_rate = float(np.sum([
        w * s["terminal_singleton"] for w, s in zip(weights, summaries)
    ]))
    two_run_rate = float(np.sum([
        w * s["two_run"] for w, s in zip(weights, summaries)
    ]))
    switch_density = float(np.sum([
        w * s["switch_density"] for w, s in zip(weights, summaries)
    ]))
    reversal_susceptibility = float(np.sum([
        w * s["reversal_susceptibility"] for w, s in zip(weights, summaries)
    ]))

    # A five-bin distribution represents reversal locations in ordinal scan
    # time, not their physical screen positions. Concentrated sequence
    # families provide only weak contextual calibration; whether opposing
    # runs can actually reverse accumulated evidence is the primary signal.
    reversal_hist = np.zeros(5, dtype=float)
    reversal_mass = 0.0
    for w, s in zip(weights, summaries):
        fractions = s["reversal_fractions"]
        if fractions.size == 0:
            continue
        contribution = w / float(fractions.size)
        for fraction in fractions:
            bin_index = min(4, max(0, int(np.floor(5.0 * float(fraction)))))
            reversal_hist[bin_index] += contribution
            reversal_mass += contribution

    if reversal_mass > 0.0:
        reversal_distribution = reversal_hist / reversal_mass
        concentration = float(np.sum(reversal_distribution ** 2))
        positive = reversal_distribution[reversal_distribution > 0.0]
        reversal_entropy = float(
            -np.sum(positive * np.log(positive)) / np.log(5.0)
        )
    else:
        concentration = 0.2
        reversal_entropy = 1.0

    # Validity-profile steepness can alter task calibration, but only weakly;
    # this allows the same ordinal sequence to generalize differently under a
    # changed reliability profile without turning validity into a hard rule.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    mean_rel = max(float(np.mean(reliability)), 1e-8)
    validity_gradient = float(np.std(reliability) / mean_rel)
    validity_gradient = float(np.clip(validity_gradient, 0.0, 2.0))

    # Coarse geometry terms are shrunken and centered around neutral task
    # values. They can adjust calibration but no longer directly force high
    # overwrite merely because a task contains two-run or singleton displays.
    # The accepted process-derived susceptibility statistic carries the main
    # cross-task signal while the centering preserves the prior operating
    # level of the continuous task set.
    context_score = (
        1.30
        + 1.20 * (terminal_singleton_rate - 0.30)
        + 1.20 * (concentration - 0.30)
        + 0.80 * (two_run_rate - 0.50)
        - 1.50 * (reversal_entropy - 0.70)
        - 0.70 * (switch_density - 0.25)
        - 0.20 * validity_gradient
        + susceptibility_coupling * (reversal_susceptibility - 0.35)
    )
    context_logit = context_sensitivity * (context_score - context_threshold)
    context_logit = float(np.clip(context_logit, -60.0, 60.0))
    overwrite_task_set = 1.0 / (1.0 + np.exp(-context_logit))

    # The latent task set continuously interpolates between integration and
    # overwrite. It never selects a discrete primacy or terminal policy.
    base_retention = (
        (1.0 - overwrite_task_set) * integration_retention
        + overwrite_task_set * overwrite_retention
    )
    base_retention = float(np.clip(base_retention, 0.01, 0.999))

    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    directions = np.asarray([np.sign(diff[j]) for j in discriminating], dtype=float)

    mean_positive_rel = float(np.mean(reliability[reliability > 0.0]))
    if not np.isfinite(mean_positive_rel) or mean_positive_rel <= 0.0:
        mean_positive_rel = 1.0
    normalized_rel = np.clip(reliability / mean_positive_rel, 0.15, 6.0)
    cue_strength = np.power(normalized_rel, validity_sensitivity)

    accumulator = 0.0
    coalition = 0.0
    current_run_strength = 0.0
    previous_direction = 0.0

    for j, direction in zip(discriminating, directions):
        evidence = attention * float(cue_strength[j]) * float(direction)
        coalition += evidence

        if previous_direction == 0.0 or direction == previous_direction:
            current_run_strength += abs(evidence)
        else:
            current_run_strength = abs(evidence)

        retention = base_retention
        if accumulator != 0.0 and np.sign(accumulator) != direction:
            # Opposition inhibits the existing commitment in proportion to
            # the new reason's strength relative to the active commitment.
            relative_attack = abs(evidence) / (abs(accumulator) + abs(evidence) + 1e-9)
            run_attack = current_run_strength / (abs(accumulator) + current_run_strength + 1e-9)
            inhibition = conflict_inhibition * overwrite_task_set
            retention *= np.exp(-inhibition * (0.65 * relative_attack + 0.35 * run_attack))

        accumulator = retention * accumulator + evidence
        previous_direction = float(direction)

    # A reliability-weighted coalition has strong access only when it has a
    # substantial normalized margin. Near balance, sequential opposition can
    # determine the sign; clear early majorities remain protected by global
    # coalition evidence. This is continuous evidence gating, not policy mixing.
    total_input = float(np.sum([
        attention * float(cue_strength[j]) for j in discriminating
    ]))
    coalition_margin = abs(coalition) / max(total_input, 1e-9)
    coalition_margin = float(np.clip(coalition_margin, 0.0, 1.0))
    margin_access = 0.10 + 0.90 * coalition_margin / (coalition_margin + 0.20)
    effective_coalition_weight = (
        coalition_weight * (1.0 - overwrite_task_set) * margin_access
    )
    decision_evidence = (
        (1.0 - effective_coalition_weight) * accumulator
        + effective_coalition_weight * coalition
    )

    # Confidence follows the final evidence margin. There is deliberately no
    # fatigue or count penalty, allowing coherent long reversals to remain
    # reliable while balanced conflict naturally remains uncertain.
    margin = abs(decision_evidence) / max(total_input, 1e-9)
    geometry_gain = 0.75 + 0.50 * np.sqrt(np.clip(margin, 0.0, 1.0))
    decision_variable = response_gain * geometry_gain * decision_evidence
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5], dtype=np.float64)
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
