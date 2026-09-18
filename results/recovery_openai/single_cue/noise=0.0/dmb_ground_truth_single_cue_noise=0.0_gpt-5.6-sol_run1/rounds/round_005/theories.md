# Round 5 — Theories

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


### slot 2 — `pi_7` — KILLED ✗

**Description:** Load-Contingent Dual-Code Arbitration. While inspecting discriminating experts in instructed-validity order, a decision maker constructs two qualitatively different codes. The local code is a rapidly revised categorical commitment: each new discriminating reason almost completely replaces the previous commitment, so reasons outside the recent suffix become functionally absent rather than merely weakened traces. In parallel, the person can form a validity-weighted global coalition summary that preserves cumulative directional evidence without preserving cue order. Choice is governed by smooth arbitration between these codes. Global comparison is recruited only after conflict is sufficiently elaborate to warrant comparison, is favored when validity-weighted evidence forms a diagnostic coalition supported by an unweighted directional majority, and loses accessibility when subject-specific effective capacity is exceeded. Coalition selectivity is itself load-contingent: a coalition must be especially strong near the recruitment boundary, whereas weaker cumulative coalitions become usable inside the subject's manageable comparison window before becoming inaccessible under overload. Within that manageable window, global control receives a selective increment when the global coalition opposes the current local commitment, because only such disagreement warrants effortful comparison. Thus short or count-balanced conflicts are dominated by local revision, manageable majority-coalition conflicts can be decided cumulatively, and overloaded conflicts revert toward an uncertainty-attenuated recent commitment. Capacity, recruitment threshold, coalition selectivity, and local and global gains are stable subject characteristics shared across trials.

**Rationale:** This is a minimal edit to the accepted candidate. The recruitment-threshold range is shifted modestly upward from [2.8, 3.8] to [3.1, 4.0], reducing validity-weighted global leakage at the three-reason boundary relevant to Experiment 9 while leaving five- and six-reason conflicts above recruitment. The additive opposition coefficient is increased from 0.45 to 0.95, but the increment now contains accessibility squared. This gives stronger disagreement-contingent global reversal inside the manageable window, targeting the excessive terminal agreement in Experiments 3 and 8, while causing the increment to decay more rapidly once load approaches or exceeds capacity, protecting Experiment 6. Count-balanced protection, nonsaturating count diagnosticity, categorical local revision, global gain, and all other equations and ranges are unchanged.

**Parameters:**
  - `validities`: `validities`
  - `effective_capacity`: `[5.8, 6.8]`
  - `recruitment_threshold`: `[3.1, 4.0]`
  - `arbitration_temperature`: `[0.35, 0.75]`
  - `local_retention`: `[0.0, 0.04]`
  - `local_gain`: `[1.35, 2.15]`
  - `global_gain`: `[2.0, 4.2]`
  - `validity_sensitivity`: `[0.65, 1.35]`
  - `coalition_focus`: `[0.75, 1.65]`
  - `overload_attenuation`: `[0.8, 1.6]`
  - `lapse_rate`: `[0.03, 0.14]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Dual-code arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    capacity = float(parameters["effective_capacity"])
    recruitment_threshold = float(parameters["recruitment_threshold"])
    arbitration_temperature = float(parameters["arbitration_temperature"])
    local_retention = float(parameters["local_retention"])
    local_gain = float(parameters["local_gain"])
    global_gain = float(parameters["global_gain"])
    validity_sensitivity = float(parameters["validity_sensitivity"])
    coalition_focus = float(parameters["coalition_focus"])
    overload_attenuation = float(parameters["overload_attenuation"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    # LOCAL CODE: a categorical commitment rather than a common evidence
    # buffer. Every newly encoded reason installs its own direction and only
    # a small residue of the immediately preceding commitment remains.
    # Consequently, sufficiently early reasons are functionally absent.
    local_commitment = 0.0
    for direction in directions:
        local_commitment = direction + local_retention * local_commitment
    local_commitment = float(np.clip(local_commitment, -1.5, 1.5))

    # GLOBAL CODE: cumulative validity-weighted coalition evidence. Log odds
    # represent communicated reliability, while normalization makes the code
    # portable across experiments with different absolute validity scales.
    v = np.clip(validities[discriminating], 0.5001, 0.999)
    reliability = np.log(v / (1.0 - v))
    positive = reliability[reliability > 0.0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    weights = np.power(
        np.clip(reliability / scale, 1e-4, 20.0), validity_sensitivity
    )
    weights = np.clip(weights, 1e-5, 30.0)

    signed_global = float(np.sum(weights * directions))
    global_norm = float(np.sqrt(np.sum(weights * weights)))
    if not np.isfinite(global_norm) or global_norm <= 0.0:
        global_norm = 1.0
    global_summary = signed_global / global_norm

    total_weight = float(np.sum(weights))
    coalition_strength = abs(signed_global) / max(total_weight, 1e-9)
    coalition_strength = float(np.clip(coalition_strength, 0.0, 1.0))

    # Conflict balance is high when both options have several reasons. It is
    # arrangement-free: only directional masses enter, never switch counts or
    # directional runs.
    positive_mass = float(np.sum(weights[directions > 0.0]))
    negative_mass = float(np.sum(weights[directions < 0.0]))
    if positive_mass + negative_mass <= 0.0:
        conflict_balance = 0.0
    else:
        conflict_balance = (
            2.0 * min(positive_mass, negative_mass)
            / (positive_mass + negative_mass)
        )
    conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))

    temp = max(arbitration_temperature, 1e-6)

    # Recruitment rises smoothly once a conflict warrants an explicit global
    # comparison, but global accessibility falls smoothly near the person's
    # effective capacity. No fixed cue-count boundary is imposed.
    onset_x = np.clip((float(m) - recruitment_threshold) / temp, -60.0, 60.0)
    onset = 1.0 / (1.0 + np.exp(-onset_x))
    capacity_x = np.clip((capacity - float(m)) / temp, -60.0, 60.0)
    accessibility = 1.0 / (1.0 + np.exp(-capacity_x))

    # Validity weighting determines the content of the global summary, but
    # recruitment also requires an unweighted directional coalition. A
    # smooth, nonsaturating transform gives a one-vote majority only partial
    # diagnosticity, while preserving exact protection for count balance and
    # approaching full support only for genuinely lopsided coalitions.
    raw_count_coalition = abs(float(np.sum(directions))) / float(m)
    raw_count_coalition = float(np.clip(raw_count_coalition, 0.0, 1.0))
    count_coalition = (
        (1.0 - np.exp(-2.0 * raw_count_coalition))
        / (1.0 - np.exp(-2.0))
    )
    count_coalition = float(np.clip(count_coalition, 0.0, 1.0))

    # Coalition selectivity varies smoothly with effective load. Near the
    # subject's recruitment boundary, only a strong weighted coalition can
    # engage global comparison. Once load is clearly above that boundary but
    # still below effective capacity, weaker cumulative coalitions become
    # usable. Selectivity tightens again as overload approaches. The window is
    # defined relative to stable subject-specific thresholds rather than by a
    # fixed number of cues.
    comparison_x = np.clip(
        (float(m) - recruitment_threshold - 0.75) / temp, -60.0, 60.0
    )
    comparison_ready = 1.0 / (1.0 + np.exp(-comparison_x))
    pre_overload_x = np.clip(
        (capacity + 0.50 - float(m)) / temp, -60.0, 60.0
    )
    pre_overload_access = 1.0 / (1.0 + np.exp(-pre_overload_x))
    manageable_window = comparison_ready * pre_overload_access
    selectivity_cutoff = 0.56 - 0.30 * manageable_window

    weighted_selectivity_x = np.clip(
        (coalition_strength - selectivity_cutoff) / 0.12, -60.0, 60.0
    )
    weighted_selectivity = 1.0 / (1.0 + np.exp(-weighted_selectivity_x))
    diagnosticity = np.power(
        np.clip(coalition_strength, 0.0, 1.0), coalition_focus
    )
    diagnosticity *= count_coalition * weighted_selectivity
    diagnosticity = float(np.clip(diagnosticity, 0.0, 1.0))

    global_gate = float(np.clip(onset * accessibility * diagnosticity, 0.0, 1.0))

    # Effortful global arbitration is selectively useful when its cumulative
    # conclusion opposes the current local commitment. A bounded additive
    # increment can rescue a weak baseline gate inside the manageable window,
    # while count support, weighted selectivity, and accessibility suppress it
    # near recruitment, in balanced displays, and under overload.
    code_opposition = 0.5 * (
        1.0 - np.tanh(local_commitment * global_summary / 0.20)
    )
    opposition_window = manageable_window * manageable_window * pre_overload_access
    opposition_increment = (
        0.95
        * accessibility
        * accessibility
        * opposition_window
        * code_opposition
        * count_coalition
        * weighted_selectivity
    )
    global_gate += opposition_increment
    global_gate = float(np.clip(global_gate, 0.0, 1.0))

    # Capacity pressure attenuates expression of the local commitment. Its
    # smooth onset slightly precedes nominal capacity, reflecting the cost of
    # maintaining both codes before capacity is completely exceeded.
    overload = temp * np.logaddexp(
        0.0, (float(m) - capacity + 0.75) / temp
    )
    local_divisor = np.sqrt(1.0 + overload_attenuation * overload * overload)
    local_dv = local_gain * local_commitment / local_divisor
    global_dv = global_gain * global_summary

    decision_variable = (
        (1.0 - global_gate) * local_dv + global_gate * global_dv
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
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

### `pi_8` → slot 2 (via `new_theory`)

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
