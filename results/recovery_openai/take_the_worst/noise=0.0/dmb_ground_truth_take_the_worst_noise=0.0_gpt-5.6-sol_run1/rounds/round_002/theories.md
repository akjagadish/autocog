# Round 2 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

**Description:** Prospective Capacity-Gated Comparison theory: before choosing, people perceptually register how many cues discriminate between the options and compare that comparison load with a stable cognitive capacity. If the discriminating set is manageable, they integrate all discriminating cues with an equal-weight majority rule. If it exceeds capacity, they compress the comparison into a validity-ranked lexicographic search and rely on the first discriminating cue. Strategy recruitment changes smoothly near the capacity boundary, reflecting variability in momentary load, but is determined by an observable property of the option pair rather than by experiment identity. Ordinary response noise is applied only after a strategy has produced its intended choice.

**Rationale:** The prior fixed heuristics fail jointly because each assumes the same evidence-processing regime regardless of comparison load. The proposed gate makes the two regimes consequences of a common, experiment-invariant capacity constraint. Experiment 2 contains only three discriminating cues, placing it below the capacity range and producing predominantly tally-based choices; the resulting noisy majority adherence is expected to be near the observed positive value. Experiment 1 contains eight discriminating cues, placing it well above capacity and producing almost exclusively lexicographic choices. Because lexicographic adherence does not systematically change with tally congruence, its matched contrast remains near zero, as observed. The smooth gate avoids an experiment-specific switch and predicts a graded transition: with feature count, validity profile, and tally margin controlled, majority choice should dominate at roughly two to four diagnostic cues, mixed behavior should emerge near five cues, and top-validity-cue dominance should increase around six to eight cues. History is intentionally unused because the task provides no outcome feedback from which strategy utility or cue validity could be learned.

**Parameters:**
  - `validities`: `validities`
  - `capacity`: `[4.5, 5.5]`
  - `gate_slope`: `[2.0, 5.0]`
  - `beta`: `[1.55, 2.05]`
  - `epsilon`: `[0.0, 0.08]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Capacity-Gated Comparison expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    discriminating = np.flatnonzero(differences != 0)
    load = float(discriminating.size)
    if load == 0:
        return np.array([0.5, 0.5], dtype=float)

    capacity = float(parameters["capacity"])
    gate_slope = float(parameters["gate_slope"])

    # Probability that exhaustive equal-weight integration is recruited.
    # This stable logistic gate approaches one below capacity and zero above it.
    gate_logit = gate_slope * (capacity - load)
    if gate_logit >= 0.0:
        exp_neg = np.exp(-gate_logit)
        p_integrate = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(gate_logit)
        p_integrate = exp_pos / (1.0 + exp_pos)

    # Manageable-load strategy: all discriminating cues receive equal weight.
    tally_margin = float(np.sum(np.sign(differences)))
    if tally_margin > 0.0:
        tally_scores = np.array([1.0, 0.0])
    elif tally_margin < 0.0:
        tally_scores = np.array([0.0, 1.0])
    else:
        tally_scores = np.array([0.0, 0.0])

    # Overload strategy: validity-ranked compression to the first diagnostic cue.
    cue_order = np.argsort(-validities, kind="stable")
    lex_scores = np.array([0.0, 0.0])
    for cue in cue_order:
        if differences[cue] > 0.0:
            lex_scores = np.array([1.0, 0.0])
            break
        if differences[cue] < 0.0:
            lex_scores = np.array([0.0, 1.0])
            break

    beta = float(parameters["beta"])

    def noisy_strategy_choice(scores):
        logits = beta * np.asarray(scores, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    # Response noise follows, rather than determines, strategy recruitment.
    p_tally = noisy_strategy_choice(tally_scores)
    p_lex = noisy_strategy_choice(lex_scores)
    p_core = p_integrate * p_tally + (1.0 - p_integrate) * p_lex

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, 1.0)
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


### slot 2 — `pi_4` — KILLED ✗

**Description:** Precision-Annealed Rank Integration theory: people always combine the signed evidence supplied by every discriminating cue. Comparison load does not recruit a different heuristic; instead, it continuously changes the precision with which cue priorities can be represented. At low load, the attention distribution is diffuse and discriminating cues receive nearly equal weight, approximating tallying. As the number of discriminating cues exceeds a stable representational capacity, attention is progressively concentrated toward cues higher in the communicated-validity ordering. Attention is implemented as a Gibbs distribution over validity ranks, with its concentration increasing smoothly with load. The load curve is mildly powered while anchored at high load, reducing premature concentration without weakening overload attention. The highest-validity cue retains its original Gibbs attention mass, while the conditional distribution over remaining cues is only mildly flattened, preserving collective tail evidence without excessively reallocating attention toward the lowest-validity cues. The normalized weighted evidence is then translated into a choice through ordinary logistic response noise. Stable individual differences generate heterogeneous but gradual crossovers without categorical switching between tallying and Take The Best.

**Rationale:** This is the smallest edit to the accepted iteration-5 model: only the tail_flatten range changes, from strong flattening at [0.45, 0.70] to an intermediate [0.75, 0.95]. The rejected sharpening range improved Experiments 2 and 3 somewhat but harmed Experiment 4, whereas the accepted strong-flattening range preserved Experiment 4 but produced an excessive Experiment 3 crossover. The intermediate range should retain part of the former improvements without the large change in signed tail evidence that reduced leading-cue adherence. The successful powered, load-7-anchored schedule, preserved top-cue mass, response rule, and all other parameter ranges remain unchanged. The model therefore continues to implement one continuous, load-adaptive integration process based solely on pair-level load and communicated validity ordering.

**Parameters:**
  - `validities`: `validities`
  - `capacity`: `[5.3, 5.8]`
  - `transition_width`: `[0.6, 0.95]`
  - `concentration_gain`: `[1.15, 1.55]`
  - `schedule_power`: `[1.15, 1.35]`
  - `tail_flatten`: `[0.75, 0.95]`
  - `beta`: `[4.2, 5.4]`
  - `epsilon`: `[0.0, 0.06]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Precision-Annealed Rank Integration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    differences = a - b
    signs = np.sign(differences)
    discriminating = np.flatnonzero(signs != 0.0)
    load = float(discriminating.size)

    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    # A cue's rank is the number of cues having strictly greater communicated
    # validity. Thus equally valid cues receive exactly equal attention.
    ranks = np.asarray(
        [np.sum(validities > validities[j]) for j in range(n_features)],
        dtype=float,
    )

    capacity = float(parameters["capacity"])
    transition_width = float(parameters["transition_width"])
    concentration_gain = float(parameters["concentration_gain"])
    schedule_power = float(parameters["schedule_power"])

    # Smoothly increasing attention concentration. Powering the curve relative
    # to a load-7 anchor reduces premature low-load concentration while leaving
    # the overload concentration of the accepted model approximately intact.
    def stable_sigmoid(value):
        if value >= 0.0:
            return 1.0 / (1.0 + np.exp(-value))
        ex = np.exp(value)
        return ex / (1.0 + ex)

    x = (load - capacity) / transition_width
    concentration_fraction = stable_sigmoid(x)
    anchor_fraction = stable_sigmoid((7.0 - capacity) / transition_width)
    concentration = concentration_gain * anchor_fraction * (
        concentration_fraction / anchor_fraction
    ) ** schedule_power

    active_ranks = ranks[discriminating]
    log_attention = -concentration * active_ranks
    log_attention -= np.max(log_attention)
    attention = np.exp(log_attention)
    attention /= attention.sum()

    # Preserve the Gibbs mass assigned to the highest-validity active cue(s),
    # but flatten only the conditional distribution over the remaining cues.
    top_mask = active_ranks == np.min(active_ranks)
    tail_mask = ~top_mask
    if np.any(tail_mask):
        top_mass = float(np.sum(attention[top_mask]))
        tail_flatten = float(parameters["tail_flatten"])
        tail_logits = tail_flatten * log_attention[tail_mask]
        tail_logits -= np.max(tail_logits)
        tail_distribution = np.exp(tail_logits)
        tail_distribution /= tail_distribution.sum()

        adjusted_attention = np.zeros_like(attention)
        adjusted_attention[top_mask] = attention[top_mask]
        adjusted_attention[tail_mask] = (1.0 - top_mass) * tail_distribution
        attention = adjusted_attention

    # Positive evidence favors A and negative evidence favors B. Normalizing
    # attention keeps response precision comparable across different loads.
    evidence = float(np.dot(attention, signs[discriminating]))

    beta = float(parameters["beta"])
    logits = np.array([0.5 * beta * evidence, -0.5 * beta * evidence], dtype=float)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs /= probs.sum()
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / len(probs)
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 2 (via `new_theory`)

**Description:** Load-Indexed Format Arbitration theory: people first register the observable number of cues that discriminate between the options and use it to recruit among three comparison formats. At low discriminating-cue load, they diffusely pool all diagnostic cues with equal weight, but the confidence attached to this normalized majority evidence varies stably across individuals. When the display contains a partially tied leading-validity tier, confidence in diffuse pooling is selectively attenuated at loads 3–4 because the tied priority structure interferes with forming a unitary low-load comparison. At intermediate load, people retrieve diagnostic cues sequentially from the display; retrieval accessibility combines communicated-validity tier with a recency gradient that is attenuated at very low loads and reaches full strength by overload loads. At high load, they compress the display to the highest-validity active tier, with a format-specific and person-specific evidence gain. The high-load transition distinguishes loads 6–7 from load 8, and its location varies coherently with recency strength. Observable validity-tier structure changes compressibility: a distinctive partially tied leading tier facilitates compression, whereas an entirely undifferentiated validity display delays it. Equal-validity tiers invoke a stable person-specific convention: some individuals use the first displayed diagnostic member of the tier, whereas others pool graded signed evidence across all diagnostic members of that tier. Under high load, whole-tier poolers obtain a modest coherence benefit when compressing a partially tied leading tier, strengthening—but not discretizing—the tier's graded mean evidence. These representations and their format-specific confidence generate signed evidence before ordinary response noise is applied.

**Rationale:** This is a single localized edit to the accepted iteration-6 model. For whole-tier poolers only, high-format gain is multiplied by a modest subject-specific factor when the observable display has a partially tied leading tier and the discriminating load is at least six. The first-member subgroup, format-recruitment gates, recency profile, low-load attenuation, tie-rule prevalence, and response noise are unchanged. In Experiment 3, the pooled leading tier has graded two-to-one evidence in the broader majority direction; strengthening that evidence should increase high-load majority consistency and reduce the excessive crossover. The evidence remains the tier mean, so a weak majority is still less compelling than unanimity. The condition excludes all-equal Experiment 1 displays and strictly ordered Experiments 2, 4, and 6, while Experiment 5's stable classification split should remain intact because the edit reinforces poolers' existing direction rather than changing their tie rule.

**Parameters:**
  - `validities`: `validities`
  - `low_center`: `[4.3, 4.8]`
  - `high_center`: `[7.0, 7.35]`
  - `transition_width`: `[0.9, 1.3]`
  - `high_transition_width`: `[0.3, 0.5]`
  - `partial_tie_acceleration`: `[0.9, 1.4]`
  - `all_equal_delay`: `[0.9, 1.3]`
  - `recency_compression_coupling`: `[0.25, 0.45]`
  - `recency_strength`: `[4.8, 6.8]`
  - `validity_strength`: `[0.2, 0.65]`
  - `tie_rule`: `{0, 1}`
  - `partial_tie_low_attenuation`: `[0.35, 0.6]`
  - `low_evidence_gain`: `[0.6, 0.85]`
  - `high_evidence_gain`: `[0.4, 0.65]`
  - `partial_tier_pooling_gain`: `[1.45, 1.8]`
  - `beta`: `[4.8, 6.2]`
  - `epsilon`: `[0.0, 0.06]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Load-Indexed Format Arbitration expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    signs = np.sign(a - b)
    active = np.flatnonzero(signs != 0.0)
    load = float(active.size)
    if active.size == 0:
        return np.array([0.5, 0.5], dtype=float)

    def stable_sigmoid(x):
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)

    # Observable load gradually opens sequential retrieval and, later,
    # validity-tier compression. The low transition begins slightly earlier
    # so retrieval can contribute on three-cue conflicts.
    low_width = float(parameters["transition_width"])
    entered_retrieval = stable_sigmoid(
        (load - float(parameters["low_center"])) / low_width
    )

    global_best = float(np.max(validities))
    global_top_tier = np.flatnonzero(np.isclose(
        validities, global_best, rtol=1e-9, atol=1e-12
    ))
    partial_top_tie = 1.0 if 1 < global_top_tier.size < n_features else 0.0
    all_equal = 1.0 if global_top_tier.size == n_features else 0.0

    # Recency strength and compression timing are stably correlated within a
    # person. Strong-recency subjects retain retrieval longer; validity-led
    # subjects compress earlier. Distinct partial tiers facilitate chunking,
    # whereas an all-equal display supplies no validity boundary to compress.
    recency_strength = float(parameters["recency_strength"])
    recency_profile = (recency_strength - 5.8) / 1.0
    effective_high_center = (
        float(parameters["high_center"])
        + float(parameters["recency_compression_coupling"]) * recency_profile
        - partial_top_tie * float(parameters["partial_tie_acceleration"])
        + all_equal * float(parameters["all_equal_delay"])
    )
    compressed = stable_sigmoid(
        (load - effective_high_center)
        / float(parameters["high_transition_width"])
    )

    w_low = 1.0 - entered_retrieval
    w_high = entered_retrieval * compressed
    w_mid = entered_retrieval * (1.0 - compressed)

    # Low-load format: diffuse equal-weight comparison. Normalization makes
    # the evidence the diagnostic-cue majority margin rather than display size.
    low_evidence = float(np.mean(signs[active]))

    # A partially tied leading tier selectively lowers confidence in diffuse
    # pooling at loads 3-4, without changing strictly ordered low-load displays.
    low_gain_multiplier = 1.0
    if partial_top_tie > 0.0 and 3.0 <= load <= 4.0:
        low_gain_multiplier = float(parameters["partial_tie_low_attenuation"])

    # Construct communicated-validity tiers. Cues tied in communicated
    # validity have the same rank; display position remains separately visible.
    tier_rank = np.asarray(
        [np.sum(validities > validities[j] + 1e-12) for j in range(n_features)],
        dtype=float,
    )
    rank_scale = max(1.0, float(n_features - 1))
    normalized_rank = tier_rank / rank_scale
    if n_features > 1:
        display_position = np.arange(n_features, dtype=float) / float(n_features - 1)
    else:
        display_position = np.zeros(1, dtype=float)

    # Intermediate format: expected evidence from sequential retrieval.
    # Positional accessibility is mildly attenuated at three-cue load and
    # reaches its original strength by load six, preserving overload gradients.
    recency_load_scale = 0.85 + 0.15 * np.clip((load - 3.0) / 3.0, 0.0, 1.0)
    retrieval_log_access = (
        recency_load_scale * recency_strength * display_position[active]
        - float(parameters["validity_strength"]) * normalized_rank[active]
    )
    retrieval_log_access -= np.max(retrieval_log_access)
    retrieval_weights = np.exp(retrieval_log_access)
    retrieval_weights /= retrieval_weights.sum()
    mid_evidence = float(np.dot(retrieval_weights, signs[active]))

    # High-load format: compress to the best communicated-validity tier that
    # is diagnostic on this trial. Stable tie_rule=0 uses its first displayed
    # member; tie_rule=1 pools graded evidence over the complete active tier.
    best_validity = float(np.max(validities[active]))
    top_tier = active[np.isclose(
        validities[active], best_validity, rtol=1e-9, atol=1e-12
    )]

    tie_rule = int(parameters["tie_rule"])
    if tie_rule == 0:
        first_member = int(np.min(top_tier))
        high_evidence = float(signs[first_member])
    else:
        high_evidence = float(np.mean(signs[top_tier]))

    # At overload, coherent retrieval of an entire partially tied leading tier
    # gives whole-tier poolers a bounded confidence benefit. The underlying
    # evidence remains its graded mean rather than being collapsed to its sign.
    high_gain = float(parameters["high_evidence_gain"])
    if tie_rule == 1 and partial_top_tie > 0.0 and load >= 6.0:
        high_gain *= float(parameters["partial_tier_pooling_gain"])

    beta = float(parameters["beta"])

    def evidence_probabilities(evidence, gain=1.0):
        logits = np.array([
            0.5 * beta * float(gain) * float(evidence),
            -0.5 * beta * float(gain) * float(evidence),
        ], dtype=float)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= p.sum()
        return p

    # Format recruitment is part of representation, not response noise.
    # Stable format-specific gains calibrate confidence without weakening the
    # intermediate positional profile that is needed at overload loads.
    p_core = (
        w_low * evidence_probabilities(
            low_evidence,
            low_gain_multiplier * float(parameters["low_evidence_gain"])
        )
        + w_mid * evidence_probabilities(mid_evidence)
        + w_high * evidence_probabilities(high_evidence, high_gain)
    )

    # Ordinary lapses occur only after each format has generated evidence.
    epsilon = float(parameters["epsilon"])
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=float)
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
