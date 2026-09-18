# Round 8 — Theories

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


### slot 2 — `pi_10` — KILLED ✗

**Description:** Reliability-Gated Decision-Object Arbitration with Prefix-Coherence-Conditioned Terminal Access. People construct at most three bounded objects from each display: an initial high-validity commitment, a reliability-weighted summary of the visible coalition, and a privileged terminal reason or terminal directional run. Arbitration is stimulus-local and history-free. Terminal evidence has a genuine recency advantage, but weak terminal evidence receives reliable access only when it constitutes the sole coherent reversal against a unified prefix at manageable load. A terminal reason following an already fragmented prefix receives no such accessibility floor and may be mildly discounted. Very short conflicts provide too little separation for a strongly privileged counterargument, whereas seven-or-more-reason conflicts reduce access through object competition rather than general count fatigue. The initial commitment remains a weak but non-negligible fallback, coalition evidence saturates, and unanimous evidence experiences only a modest vigilance decline. Stable variation in coherent-terminal accessibility, terminal susceptibility, coalition reliance, attention, and lapses produces heterogeneous conflict choices without learned context.

**Rationale:** This is a narrow edit to the accepted iteration-3 model. The early, coalition, confidence, vigilance, load-window, and history-free mechanisms are unchanged. The only substantive change is inside the existing terminal reliability gate. In five- or six-reason conflicts, a singleton or short terminal run now receives a bounded subject-specific accessibility floor when all preceding reasons form one coherent opposing coalition. This targets the weak low-validity reversal in Experiments 11 and 13 and should reduce erroneous majority following in Experiment 12 without globally increasing terminal susceptibility. A short terminal run following an already fragmented prefix receives a mild 12% accessibility discount, targeting excessive terminal agreement in Experiment 3 and the mixed-condition excess in Experiment 5. The discount does not apply to four-cue conflicts, protecting Experiment 4, and coherent seven-plus reversals receive neither the new floor nor the new penalty, protecting the accepted overload fit in Experiment 6. The new floor varies stably across subjects, expressing heterogeneity specifically where coherent terminal and coalition objects compete rather than broadening susceptibility in every condition.

**Parameters:**
  - `validities`: `validities`
  - `validity_compression`: `[0.18, 0.42]`
  - `early_commitment`: `[0.04, 0.22]`
  - `coalition_reliance`: `[0.20, 0.85]`
  - `terminal_susceptibility`: `[1.2, 8.3]`
  - `terminal_reliability_threshold`: `[0.28, 0.62]`
  - `terminal_reliability_sharpness`: `[0.06, 0.14]`
  - `coherent_access_floor`: `[0.58, 0.78]`
  - `terminal_run_bonus`: `[0.25, 0.95]`
  - `terminal_singleton_bonus`: `[0.40, 1.00]`
  - `conflict_moderation`: `[0.10, 0.65]`
  - `attentional_reliability`: `[0.82, 0.99]`
  - `choice_gain`: `[2.4, 4.8]`
  - `vigilance_decline`: `[0.12, 0.25]`
  - `lapse_rate`: `[0.03, 0.15]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Decision-object arbitration expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    early_commitment = float(parameters["early_commitment"])
    coalition_reliance = float(parameters["coalition_reliance"])
    terminal_susceptibility = float(parameters["terminal_susceptibility"])
    reliability_threshold = float(parameters["terminal_reliability_threshold"])
    reliability_sharpness = float(parameters["terminal_reliability_sharpness"])
    coherent_access_floor = float(parameters["coherent_access_floor"])
    terminal_run_bonus = float(parameters["terminal_run_bonus"])
    terminal_singleton_bonus = float(parameters["terminal_singleton_bonus"])
    conflict_moderation = float(parameters["conflict_moderation"])
    attentional_reliability = float(parameters["attentional_reliability"])
    choice_gain = float(parameters["choice_gain"])
    vigilance_decline = float(parameters["vigilance_decline"])
    lapse_rate = float(parameters["lapse_rate"])

    # Communicated validity determines psychological inspection order. Exact
    # ties retain display order, but physical gaps otherwise have no role.
    cue_order = np.argsort(-validities, kind="stable")
    diff = stim[0] - stim[1]
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.asarray(
        [float(np.sign(diff[j])) for j in discriminating], dtype=float
    )

    # Convert instructed validities to reliability strengths. Compression
    # preserves whether a validity profile is flat or steep without allowing
    # one extreme cue to create an unbounded accumulator.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(clipped_v / (1.0 - clipped_v))
    positive = log_odds[log_odds > 0.0]
    scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = np.clip(log_odds / scale, 1e-4, 30.0)
    strengths_all = np.power(normalized, validity_compression)
    strengths = np.asarray(
        [float(strengths_all[j]) for j in discriminating], dtype=float
    )

    def sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # OBJECT 1: an early high-validity commitment. Its availability receives
    # a small boost only when the first reason is unusually reliable.
    first_direction = float(directions[0])
    first_strength = float(strengths[0])
    maximum_strength = max(float(np.max(strengths)), 1e-8)
    first_relative = first_strength / maximum_strength
    early_availability = 1.0 - np.exp(-first_strength)
    w_early = early_commitment * early_availability * (
        1.0 + 0.5 * first_relative
    )

    # OBJECT 2: a bounded reliability-weighted summary of the visible
    # coalition. Directional mass saturates; no sequential accumulator or
    # learned contextual state is constructed. Balance sensitivity is
    # compressed so the coalition does not duplicate the initial object.
    signed_support = float(np.sum(directions * strengths))
    total_support = max(float(np.sum(strengths)), 1e-12)
    coalition_balance = abs(signed_support) / total_support
    coalition_balance = float(np.clip(coalition_balance, 0.0, 1.0))
    coalition_direction = float(np.sign(signed_support))
    coalition_saturation = 1.0 - np.exp(-total_support / 2.0)
    if coalition_direction == 0.0:
        w_coalition = 0.0
    else:
        w_coalition = (
            coalition_reliance
            * coalition_saturation
            * (0.22 + 0.58 * coalition_balance)
        )

    # OBJECT 3: the final directional run in validity order. Run support is
    # represented as one chunk, and its length benefit rapidly saturates.
    terminal_direction = float(directions[-1])
    run_start = m - 1
    while run_start > 0 and directions[run_start - 1] == terminal_direction:
        run_start -= 1
    terminal_run_length = m - run_start
    terminal_run_strength = float(np.sum(strengths[run_start:]))
    run_saturation = 1.0 - np.exp(-terminal_run_strength)
    n_runs = 1 + int(np.sum(directions[1:] != directions[:-1]))

    # Terminal access depends on reliability relative to the strongest
    # currently visible reason. This creates nonuniform recency across
    # validity environments without learning an experiment-level overwrite
    # state. Even weak terminal reasons retain a small intrinsic privilege.
    terminal_relative = float(strengths[-1] / maximum_strength)
    terminal_access = sigmoid(
        (terminal_relative - reliability_threshold)
        / max(reliability_sharpness, 1e-6)
    )

    # A weak terminal counterargument is distinctly retrievable when it is a
    # short, sole reversal against a coherent prefix at manageable load. No
    # such floor is granted after a fragmented prefix; short terminal runs in
    # those displays receive only a mild local discount. Four-cue conflicts
    # and seven-plus coherent reversals are left unchanged.
    prefix = directions[:run_start]
    prefix_coherent = bool(
        prefix.size > 0
        and np.all(prefix == prefix[0])
        and prefix[0] != terminal_direction
    )
    short_terminal_run = terminal_run_length <= max(2, m // 2)
    if m in (5, 6) and prefix_coherent and short_terminal_run:
        singleton_boost = 0.08 * float(terminal_run_length == 1)
        terminal_access = max(
            terminal_access,
            min(0.92, coherent_access_floor + singleton_boost)
        )
    elif m >= 5 and terminal_run_length <= 2 and prefix.size > 1:
        prefix_fragmented = bool(np.any(prefix[1:] != prefix[:-1]))
        if prefix_fragmented:
            terminal_access *= 0.88

    reliability_gate = 0.08 + 0.92 * terminal_access

    # A coherent reversal is maximally accessible at manageable loads. The
    # smooth window weakens this privilege for very short conflicts and for
    # seven-or-more-reason overload without imposing general count fatigue.
    manageable_access = (
        sigmoid((float(m) - 3.5) / 0.35)
        * sigmoid((6.5 - float(m)) / 0.35)
    )
    geometry_multiplier = 1.0 + 0.25 * terminal_run_bonus * run_saturation
    if n_runs == 2:
        singleton = float(terminal_run_length == 1)
        coherent_access = 2.0 + terminal_singleton_bonus * singleton
        geometry_multiplier *= (
            1.0
            + terminal_run_bonus * manageable_access * coherent_access
        )
    elif n_runs >= 3:
        # Fragmented conflicts provide several competing retrieval boundaries;
        # their terminal reason remains privileged but is less diagnostic.
        fragmentation = min(2.0, float(n_runs - 2))
        geometry_multiplier /= (
            1.0 + 0.50 * terminal_run_bonus * fragmentation
        )
    w_terminal = (
        terminal_susceptibility
        * reliability_gate
        * (0.35 + 0.65 * attentional_reliability)
        * geometry_multiplier
    )

    # A conflicting coalition moderates, but cannot categorically erase, the
    # terminal object. Switch count and physical gaps do not enter this gate.
    if coalition_direction != 0.0 and coalition_direction != terminal_direction:
        opposing_advantage = coalition_balance * min(
            2.0, np.sqrt(float(m) / 2.0)
        )
        w_terminal /= 1.0 + conflict_moderation * opposing_advantage

    # Each object has a bounded probability of expressing its favored option.
    # Attentional reliability affects readout globally while the object
    # strengths determine confidence locally.
    early_confidence = sigmoid(
        choice_gain * attentional_reliability * (0.65 + 0.35 * early_availability)
    )
    coalition_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.40 + 0.50 * coalition_balance)
    )
    terminal_confidence = sigmoid(
        choice_gain
        * attentional_reliability
        * (0.62 + 0.38 * run_saturation)
    )

    def p_a_from_object(direction, confidence):
        if direction > 0.0:
            return float(confidence)
        if direction < 0.0:
            return float(1.0 - confidence)
        return 0.5

    object_weights = np.asarray(
        [w_early, w_coalition, w_terminal], dtype=np.float64
    )
    object_predictions = np.asarray([
        p_a_from_object(first_direction, early_confidence),
        p_a_from_object(coalition_direction, coalition_confidence),
        p_a_from_object(terminal_direction, terminal_confidence)
    ], dtype=np.float64)

    weight_sum = float(np.sum(object_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        p_a_core = 0.5
    else:
        p_a_core = float(np.dot(object_weights, object_predictions) / weight_sum)

    # Only coherent added evidence incurs a mild vigilance decrement. Because
    # all objects agree on such trials, evidence remains strong and the load
    # effect is much smaller than obligatory count fatigue.
    unanimous = bool(np.all(directions == directions[0]))
    extra_lapse = 0.0
    if unanimous:
        extra_lapse = vigilance_decline * min(
            1.0, max(0.0, float(m) - 2.0) / 8.0
        )
    effective_lapse = float(np.clip(lapse_rate + extra_lapse, 0.0, 0.45))

    p_a = (1.0 - effective_lapse) * p_a_core + effective_lapse * 0.5
    probs = np.array([p_a, 1.0 - p_a], dtype=np.float64)
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

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Reliability-Compressed Recurrent Evidence Field. Decision makers inspect discriminating experts in descending instructed-validity order and encode each attended rating as signed, compressed log-odds evidence. Evidence remains distributed across cue traces, but accessibility declines smoothly with ordinal distance from response, producing graded recency rather than categorical replacement. Choice depends on the signed sum of accessible traces. Simultaneously active support for both products generates divisive interference, with an additional bounded increase when the accessible recurrent state is genuinely near balance. Increasing evidence load raises readout noise, especially under conflict. Cue-attention uncertainty contributes additional encoding noise, while a small vigilance decrement allows only a modest decline for long unanimous displays. Thus terminal control emerges continuously from terminal reliability, the accessibility gradient, opposing reliability-weighted coalition strength, and total interference. The mechanism has no sequence-class rules, physical-gap effects, fixed capacity boundary, or cross-trial learning.

**Rationale:** This is a minimal edit to the accepted iteration-2 model. The reliability transform, retention gradient, attention, gain, load noise, vigilance, and lapse ranges are unchanged. The only mechanistic addition is a narrow, bounded amplification of the existing divisive normalization based on conflict balance after serial accessibility has been applied. Raising accessible-state balance to the sixth power confines the adjustment to genuinely near-balanced recurrent states; asymmetric states receive almost no additional penalty. This should reduce excessive polarization in balanced conflicts without repeating the rejected retrieval-neutral interference term, injecting large extra readout noise, or weakening terminal control when recency has already made the accessible state strongly asymmetric. Because the adjustment depends only on signed accessible mass, it does not encode coherence, switch count, physical gaps, singleton status, fixed windows, or trial history.

**Parameters:**
  - `validities`: `validities`
  - `validity_compression`: `[0.0, 0.10]`
  - `memory_retention`: `[0.38, 0.54]`
  - `cue_attention`: `[0.86, 0.99]`
  - `conflict_normalization`: `[0.15, 0.55]`
  - `near_balance_normalization`: `[1.5, 2.5]`
  - `load_noise`: `[0.03, 0.16]`
  - `vigilance_decline`: `[0.008, 0.025]`
  - `decision_gain`: `[3.5, 6.5]`
  - `lapse_rate`: `[0.04, 0.16]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Recurrent evidence field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    compression = float(parameters["validity_compression"])
    retention = float(parameters["memory_retention"])
    attention = float(parameters["cue_attention"])
    conflict_normalization = float(parameters["conflict_normalization"])
    near_balance_normalization = float(parameters["near_balance_normalization"])
    load_noise = float(parameters["load_noise"])
    vigilance_decline = float(parameters["vigilance_decline"])
    decision_gain = float(parameters["decision_gain"])
    lapse = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    cue_order = np.argsort(-validities, kind="stable")
    discriminating = [int(j) for j in cue_order if diff[j] != 0.0]
    m = len(discriminating)
    if m == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Communicated validity is converted to normative log-odds and then
    # strongly compressed. Normalizing by the experiment's typical positive
    # log-odds preserves its reliability profile without allowing arbitrary
    # differences in scale to determine response gain.
    v = np.clip(validities, 0.5001, 0.999)
    log_odds = np.log(v / (1.0 - v))
    positive = log_odds[log_odds > 0.0]
    reliability_scale = float(np.mean(positive)) if positive.size else 1.0
    if not np.isfinite(reliability_scale) or reliability_scale <= 0.0:
        reliability_scale = 1.0
    relative_reliability = np.clip(
        log_odds / reliability_scale, 1e-4, 20.0
    )
    cue_strength = np.power(relative_reliability, compression)
    cue_strength = np.clip(cue_strength, 0.05, 8.0)

    # Every reason retains a separate trace. Accessibility follows a smooth
    # exponential serial-position gradient: no reason is exactly overwritten,
    # and no capacity or sequence-length boundary is imposed.
    signed_traces = []
    unsigned_traces = []
    omission_variance = 0.0
    for serial_position, j in enumerate(discriminating):
        lag = (m - 1) - serial_position
        accessibility = retention ** float(lag)
        strength = float(cue_strength[j])
        direction = float(np.sign(diff[j]))

        expected_trace = attention * accessibility * strength
        signed_traces.append(direction * expected_trace)
        unsigned_traces.append(expected_trace)

        # Bernoulli attention uncertainty is represented at readout as the
        # variance of a potentially omitted trace.
        omission_variance += (
            attention * (1.0 - attention)
            * (accessibility * strength) ** 2
        )

    signed_traces = np.asarray(signed_traces, dtype=float)
    unsigned_traces = np.asarray(unsigned_traces, dtype=float)
    integrated_evidence = float(np.sum(signed_traces))

    positive_mass = float(np.sum(signed_traces[signed_traces > 0.0]))
    negative_mass = float(-np.sum(signed_traces[signed_traces < 0.0]))
    total_mass = positive_mass + negative_mass

    # Simultaneous opposing coalitions produce divisive interference. The
    # smooth balance index is zero for unanimous evidence and approaches one
    # when accessible support is evenly divided.
    if total_mass > 0.0 and np.isfinite(total_mass):
        conflict_balance = (
            4.0 * positive_mass * negative_mass
            / max(total_mass * total_mass, 1e-12)
        )
        conflict_balance = float(np.clip(conflict_balance, 0.0, 1.0))
        opposing_mass = 2.0 * min(positive_mass, negative_mass)
        interference_load = opposing_mass / (1.0 + total_mass)
    else:
        conflict_balance = 0.0
        interference_load = 0.0

    # Only an already accessible, genuinely near-balanced state receives an
    # extra bounded normalization. The sixth power leaves asymmetric states
    # essentially unchanged and introduces no sequence or stimulus class.
    near_balance_multiplier = (
        1.0 + near_balance_normalization * conflict_balance ** 6
    )
    conflict_divisor = 1.0 + (
        conflict_normalization
        * interference_load
        * np.sqrt(max(1.0, float(m - 1)))
        * near_balance_multiplier
    )

    # Load mainly increases readout uncertainty under conflict. A small
    # residual component remains for coherent displays, avoiding perfect
    # performance while not producing excessive unanimous-evidence fatigue.
    load_component = np.log1p(max(0.0, float(m - 1)))
    load_multiplier = 0.12 + 0.88 * conflict_balance
    readout_sd = np.sqrt(
        1.0
        + omission_variance
        + load_noise * load_component * load_multiplier
    )

    # A weak, continuous vigilance decline is the only load penalty that also
    # applies fully to unanimous evidence.
    vigilance = np.exp(-vigilance_decline * max(0.0, float(m - 2)))

    decision_variable = (
        decision_gain
        * vigilance
        * integrated_evidence
        / max(conflict_divisor * readout_sd, 1e-12)
    )
    decision_variable = float(np.clip(decision_variable, -60.0, 60.0))

    p_a = 1.0 / (1.0 + np.exp(-decision_variable))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)
    probs = (1.0 - lapse) * core + lapse * np.array([0.5, 0.5])

    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy(probs)`:**
```python
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```
