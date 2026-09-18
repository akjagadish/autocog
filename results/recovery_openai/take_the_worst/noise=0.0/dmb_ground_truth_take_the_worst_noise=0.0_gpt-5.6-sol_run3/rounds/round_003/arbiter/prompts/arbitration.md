# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Conflict-Density Adaptive Serial–Parallel Control theory. People do not commit to one search architecture for an entire experiment. Instead, each display automatically recruits a control mode based on the absolute number of discriminating cues. Below a capacity-like transition point, attention follows a stable hierarchy dominated by display position and choice is normally based on the first discriminating cue. A constrained accessibility failure can occasionally make that cue unavailable, after which attention continues serially through the remaining discriminating cues using a shallow rank-normalized accessibility gradient. Residual cues supporting both options create modest interference with the initial stopping decision and therefore smoothly increase the probability of continuing beyond the leading cue; unanimous residual evidence does not. A terminal-gap penalty selectively reduces access to residual cues that are separated from the leading cue by a very large positional gap. Instructed validity only provides a weak tie-breaking adjustment and cannot independently promote a substantially later cue. When the number of discriminating cues exceeds the transition point, the display supports parallel comparison and evidence is integrated using a noisy tally with only weak positional and validity weighting. The transition between modes is smooth and stimulus-contingent, rather than determined by experiment identity, trial history, or feedback.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    # Conflict-Density Adaptive Serial-Parallel Control.
    # History is intentionally unused: without outcome feedback, the stable
    # accessibility hierarchy and stimulus-triggered control rule do not learn
    # across trials.
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    diff = a - b
    discriminating = np.flatnonzero(diff != 0.0)
    k = int(discriminating.size)
    if k == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Put validity on a bounded scale. Its influence remains weak in both
    # modes and therefore cannot by itself overcome a full position step.
    v_range = float(np.max(validities) - np.min(validities))
    if np.isfinite(v_range) and v_range > 0.0:
        v_scaled = (validities - np.mean(validities)) / v_range
    else:
        v_scaled = np.zeros(n_features, dtype=np.float64)

    validity_modifier = float(parameters["validity_modifier"])

    # Sparse mode: stable, position-dominated serial hierarchy. The validity
    # adjustment is much smaller than the priority difference between adjacent
    # display positions, so it serves only as a local tie-breaker.
    position_bias = float(parameters["position_bias"])
    positions = np.arange(n_features, dtype=np.float64)
    serial_priority = -position_bias * positions + validity_modifier * v_scaled
    serial_order = np.argsort(-serial_priority, kind="stable")

    discriminating_order = [int(j) for j in serial_order if diff[j] != 0.0]
    first_discriminating = discriminating_order[0]
    first_winner = 0 if diff[first_discriminating] > 0.0 else 1

    # The hierarchy itself remains stable, but a constrained accessibility
    # failure sometimes omits its leading discriminating member. Mixed support
    # among the residual cues creates interference with stopping and modestly
    # increases continuation; unanimous residual support leaves it unchanged.
    # Attention then continues through the residual hierarchy. Conditional on
    # a miss, residual accessibility decays shallowly by ordinal rank. A
    # separate terminal-gap penalty limits access to cues lying very far from
    # the leading cue without reinstating a steep gradient over early/middle
    # cues.
    serial_second_look = float(np.clip(
        parameters["serial_second_look"], 0.0, 1.0
    ))
    if len(discriminating_order) > 2:
        residual_signs = np.sign(diff[np.asarray(discriminating_order[1:], dtype=int)])
        residual_mixedness = 1.0 - abs(float(np.mean(residual_signs)))
        mixed_residual_boost = float(parameters["mixed_residual_boost"])
        serial_second_look += mixed_residual_boost * residual_mixedness
        serial_second_look = float(np.clip(serial_second_look, 0.0, 1.0))

    p_selected = np.zeros(2, dtype=np.float64)
    p_selected[first_winner] += 1.0 - serial_second_look

    if len(discriminating_order) == 1:
        p_selected[first_winner] += serial_second_look
    else:
        residual = np.asarray(discriminating_order[1:], dtype=int)
        residual_rank_decay = float(parameters["residual_rank_decay"])
        terminal_gap_penalty = float(parameters["terminal_gap_penalty"])

        if residual.size > 1:
            residual_rank = np.arange(
                residual.size, dtype=np.float64
            ) / float(residual.size - 1)
        else:
            residual_rank = np.zeros(1, dtype=np.float64)

        if n_features > 1:
            normalized_gap = (
                positions[residual] - positions[first_discriminating]
            ) / float(n_features - 1)
        else:
            normalized_gap = np.zeros(residual.size, dtype=np.float64)

        large_gap = np.maximum(normalized_gap - 0.55, 0.0)
        residual_logits = (
            -residual_rank_decay * residual_rank
            -terminal_gap_penalty * large_gap
        )
        residual_logits -= np.max(residual_logits)
        residual_probs = np.exp(residual_logits)
        residual_probs /= residual_probs.sum()

        for j, cue_prob in zip(residual, residual_probs):
            residual_winner = 0 if diff[j] > 0.0 else 1
            p_selected[residual_winner] += serial_second_look * cue_prob

    serial_precision = float(parameters["serial_precision"])
    if serial_precision >= 0.0:
        serial_reliability = 1.0 / (1.0 + np.exp(-serial_precision))
    else:
        ex = np.exp(serial_precision)
        serial_reliability = ex / (1.0 + ex)

    p_serial = (
        serial_reliability * p_selected
        + (1.0 - serial_reliability) * p_selected[::-1]
    )

    # Dense mode: all discriminating cues contribute in parallel. Position
    # produces only a shallow gradient here, and validity is a weak multiplier
    # rather than a lexicographic ordering mechanism.
    if n_features > 1:
        normalized_position = positions / float(n_features - 1)
    else:
        normalized_position = np.zeros(1, dtype=np.float64)

    positional_discount = float(parameters["positional_discount"])
    tally_weights = np.exp(-positional_discount * normalized_position)
    tally_weights *= 1.0 + validity_modifier * v_scaled
    tally_weights = np.clip(tally_weights, 1e-8, None)

    signed_tally = float(np.dot(tally_weights, diff))
    tally_precision = float(parameters["tally_precision"])
    tally_logits = np.array(
        [tally_precision * signed_tally, 0.0], dtype=np.float64
    )
    tally_logits -= np.max(tally_logits)
    p_tally = np.exp(tally_logits)
    p_tally /= p_tally.sum()

    # A smooth capacity transition makes the architecture contingent on the
    # current conflict count. The narrow parameter ranges encode constrained
    # subject heterogeneity while preserving a common control regime.
    transition_count = float(parameters["transition_count"])
    transition_steepness = float(parameters["transition_steepness"])
    transition_argument = float(np.clip(
        transition_steepness * (float(k) - transition_count), -60.0, 60.0
    ))
    dense_gate = 1.0 / (1.0 + np.exp(-transition_argument))

    p_core = (1.0 - dense_gate) * p_serial + dense_gate * p_tally

    epsilon = float(np.clip(parameters["epsilon"], 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_3
Accessibility-Gated Stochastic Lexicographic Search: People maintain a stable attentional priority over cues based primarily on visual position and learned accessibility. Communicated validities modify that priority only when the validity-to-position mapping is simple enough to encode and maintain. Mapping simplicity is assessed from how closely validity rank follows display order, while a subject-specific mapping-fidelity parameter limits the influence of this information. On each choice, attention follows a noisy priority ordering and search stops at the first cue that discriminates. Thus, stable priorities guide decisions without imposing an identical deterministic cue order on every trial. When validity and position are aligned, validity information gives the accessible first cue a sufficiently large advantage to produce nearly invariant one-cue choices. Under a nonmonotonic mapping, weak validity gating and modest attentional fluctuations distribute search across accessible cues, producing systematic choices that are approximately orthogonal to validity-ranked search without requiring cancellation between strongly polarized subjects. A small lapse probability captures response errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Accessibility-Gated Stochastic Lexicographic Search.
    # The subject has a stable, subject-specific cue priority. Validity
    # affects this priority strongly only when validity rank is easy to map
    # onto visual position. A Plackett-Luce attention process samples the
    # cue order, and search stops at the first discriminating cue.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expected state with shape (2, n_features); got {stim.shape}."
        )

    n_features = stim.shape[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    accessibility = np.asarray(parameters["accessibility_weights"], dtype=float)

    if validities.shape != (n_features,):
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )
    if accessibility.shape != (n_features,):
        raise ValueError(
            f"accessibility_weights length {accessibility.size} != n_features {n_features}."
        )

    if n_features == 1:
        position = np.zeros(1, dtype=float)
        validity_rank_score = np.ones(1, dtype=float)
        mapping_simplicity = 1.0
    else:
        # Earlier positions have greater default visual accessibility.
        position = np.arange(n_features, dtype=float) / float(n_features - 1)

        # Bounded validity-rank representation: 1 for the most valid cue and
        # 0 for the least valid. Stable sorting makes ties favor earlier cues.
        validity_order = np.argsort(-validities, kind="stable")
        ranks = np.empty(n_features, dtype=float)
        ranks[validity_order] = np.arange(n_features, dtype=float)
        validity_rank_score = 1.0 - ranks / float(n_features - 1)

        # A descending validity sequence is easy to map onto display position.
        # Compute rank-position coherence directly with NumPy so prediction
        # does not depend on an external scipy namespace being initialized.
        x = position - np.mean(position)
        y = validity_rank_score - np.mean(validity_rank_score)
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if not np.isfinite(denom) or denom <= 0.0:
            mapping_simplicity = 0.0
        else:
            rho = float(np.sum(x * y) / denom)
            mapping_simplicity = float(np.clip(-rho, 0.0, 1.0))

    complexity_power = float(parameters["complexity_power"])
    mapping_fidelity = float(parameters["mapping_fidelity"])
    mapping_gate = mapping_fidelity * (mapping_simplicity ** complexity_power)
    mapping_gate = float(np.clip(mapping_gate, 0.0, 1.0))

    position_strength = float(parameters["position_strength"])
    validity_strength = float(parameters["validity_strength"])

    # When the mapping is easy, communicated validity reshapes accessibility.
    # When it is difficult, stable learned accessibility dominates instead.
    learned_component = (1.0 - mapping_gate) * accessibility
    positional_component = position_strength * (1.0 - position)
    validity_component = validity_strength * mapping_gate * validity_rank_score
    priority = learned_component + positional_component + validity_component

    a, b = stim[0], stim[1]
    discriminating = np.flatnonzero(a != b)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    # Under a Plackett-Luce/Gumbel ordering, the probability that a cue is the
    # first discriminating cue is its softmax weight among discriminating cues.
    attention_temperature = float(parameters["attention_temperature"])
    logits = priority[discriminating] / attention_temperature
    logits = logits - np.max(logits)
    cue_probs = np.exp(logits)
    cue_probs /= cue_probs.sum()

    p_core = np.zeros(2, dtype=np.float64)
    for j, cue_prob in zip(discriminating, cue_probs):
        if a[j] > b[j]:
            p_core[0] += cue_prob
        else:
            p_core[1] += cue_prob

    epsilon = float(parameters["epsilon"])
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    probs = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    probs /= probs.sum()
    return probs


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=10):** [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]
  trial 9: A=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  trial 11: A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  trial 13: A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
  trial 15: A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]

**Rationale:** This design targets the advocated theory's stimulus-contingent transition at approximately 7.3 discriminating cues. Four matched families compare a sparse display with k=7 against a dense display with k=9. Every sparse display has three cues favoring the leading-cue option and four favoring its opponent; the corresponding dense display adds one cue for each option, yielding four versus five. Thus the raw tally margin remains exactly -1 and the proportion of cues supporting the leading-cue option changes only from 3/7=.429 to 4/9=.444. The first discriminating cue always supports the tally minority. At k=7, the advocated theory remains predominantly serial and should choose that first-cue option above chance despite its 3-to-4 tally disadvantage. At k=9, its dense gate is essentially fully active, so it should instead integrate the 4-to-5 tally and choose the same option only about .25-.35 of the time. It therefore predicts a large within-family downward discontinuity and a choice reversal across the transition. The validity sequence increases with display position, producing mapping simplicity zero in the competing theory; communicated validity is consequently gated out. That theory uses nearly uniform accessibility weights with only a shallow positional gradient, so its probability for the designated option is governed mainly by its cue share and should remain near .43-.47 across k=7 and k=9, with no sharp transition or reversal generated by conflict count. The four positional arrangements test generality and vary which late cues are added. Every stimulus has an A/B mirror, controlling product and response-key biases. There are 16 unique pairs, each repeated six times, for 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject conflict-density discontinuity score. It measures how much more often the option supported by the earliest discriminating cue is selected on sparse k=7 displays than on their dense k=9 counterparts. The advocated theory predicts a large positive value: serial control at k=7 favors the earliest cue, whereas parallel tally integration at k=9 favors its tally-majority opponent. The competing stochastic lexicographic theory uses essentially the same stable accessibility process at both densities, so it predicts similar earliest-cue choice rates and a score near zero. Using all 48 observations at each density and taking a within-subject difference limits binomial noise and cancels stable response, lapse, and accessibility tendencies.

Source:
def metric(data: pd.DataFrame) -> float:
    """Difference in first-discriminating-cue choice rates at k=7 versus k=9."""
    sparse_hits = []
    dense_hits = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        k = int(discriminating.size)
        if k not in (7, 9):
            continue

        first = int(discriminating[0])
        first_cue_choice = 0 if diff[first] > 0 else 1
        hit = float(int(row["response"]) == first_cue_choice)

        if k == 7:
            sparse_hits.append(hit)
        else:
            dense_hits.append(hit)

    if len(sparse_hits) == 0 or len(dense_hits) == 0:
        return float("nan")

    return float(np.mean(sparse_hits) - np.mean(dense_hits))


### RESULTS
- Predicted under pi_5 (simulated): 0.3450 (var=0.0092)
- Predicted under pi_3 (simulated): -0.0229 (var=0.0102)
- Observed on real data: 0.0067 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=10):** [0.51, 0.56, 0.61, 0.66, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This sparse-conflict design isolates the competing theory's terminal-gap mechanism without crossing its serial–parallel transition. Validities increase monotonically with display position, yielding rank-position correlation +1 and therefore mapping simplicity 0 in Accessibility-Gated Stochastic Lexicographic Search. Its validity gate is exactly closed: choice is governed by nearly uniform accessibility plus a shallow positional gradient. Each base trial has five discriminating cues. Features 1 and 2 favor the designated option, while three cues favor its opponent. Those three opposing cues move together from features 3–5 through features 8–10. Accessibility-gated search should therefore initially favor the 3-to-2 cue majority and show only a gradual increase in designated-option choice as the opposing cues move later (roughly from about .37–.45 toward .45–.55, depending on attentional parameters), with no special discontinuity at a particular gap. Conflict-Density Adaptive Serial–Parallel Control remains almost purely serial because k=5 is far below its 7.25–7.35 transition. Its leading discriminating cue always favors the designated option. More importantly, opposing residual cues begin receiving a strong terminal-gap penalty once they lie more than 55% of the display beyond feature 1. It consequently predicts substantially higher designated-option choice throughout (approximately .60–.65 for opposing cues at features 3–5) and a comparatively sharp rise to approximately .70–.76 when they reach features 8–10. The critical quantitative dissociation is thus a low, shallow cue-location gradient under accessibility-gated stochastic search versus a high curve with accelerated growth as residual cues cross the terminal-gap boundary under adaptive serial control. Every configuration has an A/B mirror, giving 12 unique pairs repeated eight times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the position-led stopping rate: the fraction of trials on which the subject chooses the option supported by the earliest discriminating feature. In every designed pair, the earliest cue supports the designated two-cue option, while three other cues support its opponent. Accessibility-gated stochastic search distributes first attention across all five discriminating cues and should therefore produce a rate near 0.4–0.5. The competing serial-control theory assigns substantial probability to stopping at the leading cue and predicts a markedly higher rate, approximately 0.6–0.7. Pooling all 96 trials gives a low-noise subject-level estimate, while the mirrored A/B trials make the measure insensitive to a general response-side bias.

Source:
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the earliest discriminating cue."""
    agreements = []

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0 or pd.isna(response):
            continue

        first = int(discriminating[0])
        leading_choice = 0 if diff[first] > 0 else 1
        agreements.append(float(int(response) == leading_choice))

    if len(agreements) == 0:
        return float("nan")
    return float(np.mean(agreements))

### RESULTS
- Predicted under pi_5 (simulated): 0.6575 (var=0.0018)
- Predicted under pi_3 (simulated): 0.4517 (var=0.0026)
- Observed on real data: 0.8446 (var=0.0066)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code each response as adherence (1) versus non-adherence (0) to the
    # highest-validity cue, and recover the number of lower cues opposing it.
    adherence = []
    conflict_q = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        if a[0] > b[0]:
            top_cue_choice = 0
            q = int(np.sum(b[1:] > a[1:]))
        elif b[0] > a[0]:
            top_cue_choice = 1
            q = int(np.sum(a[1:] > b[1:]))
        else:
            continue

        adherence.append(float(response == top_cue_choice))
        conflict_q.append(q)

    adherence = np.asarray(adherence, dtype=float)
    conflict_q = np.asarray(conflict_q, dtype=int)

    low_conflict = conflict_q == 1
    high_conflict = conflict_q >= 4
    if not np.any(low_conflict) or not np.any(high_conflict):
        return float("nan")

    return float(np.mean(adherence[low_conflict]) -
                 np.mean(adherence[high_conflict]))

```

**Observed (real) value:** 0.0283 (var=0.0097)
**Predicted under pi_5:** -0.0008 (var=0.0242)
**Predicted under pi_3:** -0.0025 (var=0.0152)

### Experiment 4
**Design**
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Mean signed agreement with the highest-validity discriminating cue."""
    validities = np.asarray([0.67, 0.95, 0.58, 0.82, 0.53, 0.74, 0.61, 0.88], dtype=float)
    cue_order = np.argsort(-validities, kind="stable")

    signed_agreement = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break

        if winner is not None and pd.notna(row["response"]):
            response = int(row["response"])
            signed_agreement.append(1.0 if response == winner else -1.0)

    return float(np.mean(signed_agreement)) if signed_agreement else float("nan")
```

**Observed (real) value:** -0.0050 (var=0.0051)
**Predicted under pi_5:** -0.0925 (var=0.0100)
**Predicted under pi_3:** -0.0179 (var=0.0142)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)

        # Feature 8 (index 7, equivalently the final feature here) is the
        # globally most valid cue in this experimental design.
        j = 7
        if a.size <= j or b.size <= j or a[j] == b[j]:
            continue

        cue8_winner = 0 if a[j] > b[j] else 1
        agreements.append(float(int(row['response']) == cue8_winner))

    if len(agreements) == 0:
        return float('nan')

    cue8_choice_rate = float(np.mean(agreements))

    # Soft classification around a preregistered separating rate. This is
    # smooth (unlike a hard threshold), remains defined per subject, and
    # compresses rates well inside either theory's predicted region.
    z = float(np.clip(20.0 * (cue8_choice_rate - 0.40), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0089 (var=0.0080)
**Predicted under pi_5:** 0.0196 (var=0.0013)
**Predicted under pi_3:** 0.0564 (var=0.0042)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast choices supported by the higher-validity cue on adjacent
    # conflicts versus conflicts separated by four or seven validity ranks.
    high_choice = {1: [], 4: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = np.flatnonzero(a != b)
        if diff.size != 2:
            continue

        high_idx = int(np.min(diff))  # Validities decrease with feature index.
        low_idx = int(np.max(diff))
        distance = low_idx - high_idx
        if distance not in high_choice:
            continue

        high_option = 0 if a[high_idx] > b[high_idx] else 1
        high_choice[distance].append(
            1.0 if int(row["response"]) == high_option else 0.0
        )

    if any(len(high_choice[d]) == 0 for d in (1, 4, 7)):
        return float("nan")

    adjacent = float(np.mean(high_choice[1]))
    well_separated = 0.5 * (
        float(np.mean(high_choice[4])) + float(np.mean(high_choice[7]))
    )
    return float(well_separated - adjacent)

```

**Observed (real) value:** 0.0042 (var=0.0061)
**Predicted under pi_5:** -0.0012 (var=0.0143)
**Predicted under pi_3:** 0.2100 (var=0.0120)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Soft prevalence of subjects reliably choosing the later conflicting cue."""
    if data is None or len(data) == 0:
        return float("nan")

    scores = []
    for _, subj in data.groupby("subject_id", sort=False):
        later_cue_choices = []
        for _, row in subj.iterrows():
            a = np.asarray(row["option_a_ratings"], dtype=float)
            b = np.asarray(row["option_b_ratings"], dtype=float)
            differing = np.flatnonzero(a != b)
            if differing.size == 0:
                continue

            # In this design, the later of the two discriminating cues is the
            # locally high-validity cue.
            later = int(np.max(differing))
            later_winner = 0 if a[later] > b[later] else 1
            later_cue_choices.append(float(int(row["response"]) == later_winner))

        if len(later_cue_choices) == 0:
            continue

        rate = float(np.mean(later_cue_choices))
        # Smoothly score whether the subject shows a reliable (>65%) rather
        # than near-chance preference for the validity-promoted later cue.
        z = float(np.clip(20.0 * (rate - 0.65), -60.0, 60.0))
        scores.append(1.0 / (1.0 + np.exp(-z)))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.0033 (var=0.0001)
**Predicted under pi_5:** 0.0185 (var=0.0006)
**Predicted under pi_3:** 0.0571 (var=0.0041)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data is None or len(data) == 0:
        return float("nan")

    target_choices = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.size < 2 or b.size < 2:
            continue

        # Features 1 and 2 are aligned in this design. Define the target as
        # the option jointly recommended by these two high-validity cues.
        direction = float(np.sum(a[:2] - b[:2]))
        if direction > 0:
            target_response = 0
        elif direction < 0:
            target_response = 1
        else:
            continue

        response = int(row["response"])
        target_choices.append(float(response == target_response))

    if len(target_choices) == 0:
        return float("nan")

    target_rate = float(np.mean(target_choices))

    # Smoothly classify rates around the otherwise-empty region separating
    # the advocated prediction (roughly .38-.48) from the competitor's
    # prediction (roughly .80-.86). Saturation also limits binomial and
    # parameter-driven between-subject variation away from the boundary.
    z = float(np.clip(20.0 * (target_rate - 0.64), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-z)))
```

**Observed (real) value:** 0.0005 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)
**Predicted under pi_3:** 0.0112 (var=0.0004)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, pi_5 is the better of the two theories, but neither is task-invariantly adequate. Pi_5 is closer in Experiments 2, 3, 5, 6, 7, and 8; pi_3 is closer in Experiments 1 and 4. Most importantly, Experiment 1 directly falsifies pi_5's signature prediction: the observed k=7 versus k=9 discontinuity is 0.0067, essentially zero and very close to pi_3's -0.0229, rather than pi_5's 0.3450. The low observed variance (0.0029) makes it unlikely that a large transition was hidden by heterogeneous subject effects. Experiment 2, however, strongly rejects pi_3: earliest-cue agreement is 0.8446, far above its 0.4517 prediction. Pi_5 is directionally better at 0.6575 but still substantially underestimates the remarkably strong and somewhat heterogeneous primacy effect. Experiments 3 and 4 are comparatively nondiagnostic because both theories are near the observed values, although pi_3 is especially close in Experiment 4. Experiments 5 and 7 show almost no reliable promotion of late high-validity cues and favor pi_5's weak-validity, position-dominated account. Experiment 6 decisively rejects pi_3's predicted validity-distance effect: observed 0.0042 versus 0.2100, while pi_5 predicts approximately zero. Experiment 8 strongly favors pi_5 and indicates majority-like integration on a fully conflicting display. The joint pattern is therefore not captured by either theory's proposed control variable. Conflict count alone cannot explain strong primacy at k=5, no k=7-to-k=9 transition, stable behavior across the conflict manipulation in Experiment 3, and majority-like responding in Experiment 8. Conversely, pi_3's mapping-simplicity gate and stochastic first-discriminating-cue mechanism cannot generate the very strong primacy in Experiment 2 while also retaining the near-zero validity-distance and late-validity effects in Experiments 5-7. These are structural contradictions, not merely parameter-range errors. Pi_5 remains the provisional winner, but its sharp density-transition claim has failed its own critical test; pi_3 is the weaker overall theory and should be replaced by a stronger competitor rather than merely retuned.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 with a Structure-Sensitive Anchoring-and-Comparison theory. The new theory should propose that subjects first construct a display representation whose organization depends jointly on discriminating-cue saturation, repeated cue coalitions, spatial grouping, and the presence of a stable focal or anchor cue\u2014not solely on absolute conflict count or validity-position rank coherence. Sparse or readily chunked displays should induce strong anchoring on an early cue or early cue coalition, permitting earliest-cue choice rates near 0.85 as in Experiment 2. Highly saturated displays with salient opposing coalitions should recruit comparative or tally-like processing, accounting for Experiment 8. The shift must be broad and structure-sensitive rather than a fixed transition near k=7.3, so k=7 and k=9 displays can yield nearly identical behavior as in Experiment 1. Communicated validity should usually have little direct effect, consistent with Experiments 4-7, but may reinforce a cue already made focal by position, repetition, or task structure; it should not be controlled by a global rank-correlation gate. Include subject variation in anchor strength and comparison recruitment to accommodate Experiment 2's larger observed between-subject variance. The new theory should make separable predictions from pi_5 by holding k constant while independently manipulating saturation k/n, spatial grouping, cue-coalition stability across trials, and whether one cue repeatedly occupies an invariant focal role. Its model should be constrained to reproduce four key facts simultaneously: no k=7-to-k=9 discontinuity, very strong k=5 primacy, negligible late-validity and validity-distance promotion, and majority-sensitive responding in fully saturated displays."
}
```

## Usage

```json
{
  "input_tokens": 15033,
  "output_tokens": 2066
}
```
