# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_14") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_14" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_14
Parallel Contrastive Coalition Field with Novelty-Protected Competition. All discriminating cues are encoded concurrently into a capacity-limited comparison field rather than inspected in a mandatory validity or spatial sequence. Cue activation combines instructed reliability, U-shaped boundary distinctiveness, an estimable spatial tilt, and local contrast. Same-option activations pool into saturating coalitions before divisive competition. A coalition containing a sharply contrastive challenger receives a small, bounded release from cross-coalition inhibition after saturation, allowing prediction-error signals to survive competition without letting raw cue activation bypass capacity limits. Choice follows the normalized coalition difference with load-dependent precision loss and a task-invariant lapse. The model has no trial-history learning, strategy mixture, fixed scan order, or sequential commitment state.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Parallel Contrastive Coalition Field expects shape (2, n_features); got {stim.shape}."
        )

    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    reliability_gain = float(parameters["reliability_gain"])
    boundary_gain = float(parameters["boundary_gain"])
    spatial_tilt = float(parameters["spatial_tilt"])
    coalition_saturation = float(parameters["coalition_saturation"])
    novelty_gain = float(parameters["novelty_gain"])
    novelty_release = float(parameters["novelty_release"])
    protection_threshold = float(parameters["protection_threshold"])
    capacity_loss = float(parameters["capacity_loss"])
    competition_temperature = float(parameters["competition_temperature"])
    choice_precision = float(parameters["choice_precision"])
    lapse_rate = float(parameters["lapse_rate"])

    diff = stim[0] - stim[1]
    discriminating = np.flatnonzero(diff != 0.0)
    if discriminating.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    directions = np.sign(diff[discriminating]).astype(float)
    m = int(discriminating.size)

    # Reliability is represented on the normative log-odds scale. Centering
    # preserves within-display reliability contrasts while making the shared
    # gain portable across experiments with different validity schedules.
    clipped_v = np.clip(validities, 0.5001, 0.999)
    reliability = np.log(clipped_v / (1.0 - clipped_v))
    rel_mean = float(np.mean(reliability))
    rel_sd = float(np.std(reliability))
    if rel_sd < 1e-8:
        rel_z = np.zeros(n_features, dtype=float)
    else:
        rel_z = np.clip((reliability - rel_mean) / rel_sd, -2.5, 2.5)

    raw_activation = np.zeros(m, dtype=float)
    novelty_signals = np.zeros(m, dtype=float)

    # These activations are computed from the complete display. References to
    # left and right neighbors describe simultaneous spatial relations, not an
    # update order or a serial scan.
    for k, j_cell in enumerate(discriminating):
        j = int(j_cell)
        direction = float(directions[k])
        position = float(j) / float(max(1, n_features - 1))
        signed_position = 2.0 * position - 1.0

        # Both endpoints can be distinctive. The separately estimated tilt
        # permits endpoint asymmetry without assuming universal right recency.
        boundary = signed_position * signed_position
        spatial_log_activation = (
            boundary_gain * boundary + spatial_tilt * signed_position
        )

        surprise = 0.0
        symmetric_contrast = 0.0

        if k > 0:
            left_j = int(discriminating[k - 1])
            left_opposes = float(directions[k - 1] != direction)
            if left_opposes > 0.0:
                # A reliability drop at an opposing spatial boundary is a
                # prediction-error signal, not evidence that cues were scanned.
                drop = max(0.0, float(validities[left_j] - validities[j]))
                surprise += min(1.25, drop / 0.40)
                symmetric_contrast += min(
                    1.0, abs(float(validities[left_j] - validities[j])) / 0.40
                )

        if k + 1 < m:
            right_j = int(discriminating[k + 1])
            right_opposes = float(directions[k + 1] != direction)
            if right_opposes > 0.0:
                symmetric_contrast += min(
                    1.0, abs(float(validities[right_j] - validities[j])) / 0.40
                )

        # Most contrast is a sharp-drop novelty response. A smaller symmetric
        # term captures ordinary local pop-out at either side of a boundary.
        novelty_signal = surprise + 0.18 * symmetric_contrast
        novelty_signals[k] = novelty_signal
        log_activation = (
            reliability_gain * rel_z[j]
            + spatial_log_activation
            + novelty_gain * novelty_signal
        )
        raw_activation[k] = float(np.exp(np.clip(log_activation, -8.0, 8.0)))

    # Concurrent same-option cues pool before competition. Thus they exhibit
    # diminishing returns but do not consume capacity by suppressing one
    # another at the cue level.
    support_a = float(np.sum(raw_activation[directions > 0.0]))
    support_b = float(np.sum(raw_activation[directions < 0.0]))

    sat = max(coalition_saturation, 1e-8)
    coalition_a = sat * (1.0 - np.exp(-support_a / sat))
    coalition_b = sat * (1.0 - np.exp(-support_b / sat))

    # Sharp contrast receives only a bounded post-saturation release from
    # cross-coalition inhibition. This preserves whole-coalition saturation:
    # novelty cannot add activation or pass an unsaturated cue maximum into
    # competition. Smooth, ordinary boundaries remain effectively unprotected.
    novelty_a = float(np.max(novelty_signals[directions > 0.0])) if np.any(directions > 0.0) else 0.0
    novelty_b = float(np.max(novelty_signals[directions < 0.0])) if np.any(directions < 0.0) else 0.0
    protection_a = 1.0 / (
        1.0 + np.exp(-np.clip((novelty_a - protection_threshold) / 0.12, -60.0, 60.0))
    )
    protection_b = 1.0 / (
        1.0 + np.exp(-np.clip((novelty_b - protection_threshold) / 0.12, -60.0, 60.0))
    )
    release_a = float(np.clip(1.0 - novelty_release * protection_a, 0.80, 1.0))
    release_b = float(np.clip(1.0 - novelty_release * protection_b, 0.80, 1.0))

    # Capacity competition occurs between option-level coalitions. The mild
    # temperature sharpens genuine coalition differences after repeated cues
    # have already been compressed by saturation.
    competitive_a = coalition_a / ((1.0 + capacity_loss * coalition_b) * release_a)
    competitive_b = coalition_b / ((1.0 + capacity_loss * coalition_a) * release_b)
    competitive_a = float(np.power(max(competitive_a, 0.0), competition_temperature))
    competitive_b = float(np.power(max(competitive_b, 0.0), competition_temperature))

    # Divisive comparison expresses evidence relative to the total activated
    # field, preventing large displays from creating unbounded decision values.
    coalition_total = competitive_a + competitive_b
    if coalition_total <= 1e-12 or not np.isfinite(coalition_total):
        normalized_difference = 0.0
    else:
        normalized_difference = (competitive_a - competitive_b) / coalition_total

    # Display and evidence load modestly lower readout precision. Saturating
    # pooling and this precision loss can produce a small decline even for long
    # unanimous displays, without a discrete fatigue threshold.
    display_load = float(max(0, n_features - 1))
    evidence_load = float(max(0, m - 2))
    precision = choice_precision / (
        1.0
        + 0.55 * capacity_loss * display_load
        + 0.35 * capacity_loss * evidence_load
    )
    precision = float(np.clip(precision, 0.20, 8.0))

    logit_a = float(np.clip(precision * normalized_difference, -60.0, 60.0))
    p_a = 1.0 / (1.0 + np.exp(-logit_a))
    core = np.array([p_a, 1.0 - p_a], dtype=np.float64)

    probs = (
        (1.0 - lapse_rate) * core
        + lapse_rate * np.array([0.5, 0.5], dtype=np.float64)
    )
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total


`policy(probs) -> int`:
def policy(probs) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=10):** [0.52, 0.88, 0.52, 0.8, 0.52, 0.7, 0.52, 0.6, 0.55, 0.99]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

**Rationale:** This is a fixed-evidence spatial-displacement test tailored to the competitor's concurrent activation field. Each base trial contains exactly two discriminating experts: the fixed 99%-valid Expert 10 favors B, and one of four equally valid 52% experts favors A. Thus, in descending-validity order, every base trial presents the identical direction sequence B,A. Recency-Gated Reason Overwrite consequently predicts the same strong A preference in all four conditions: the low-validity A reason is processed last, overwrites B, and two reasons incur no update-fatigue penalty. The competitor instead processes the complete physical display and predicts a monotonic decline in A choice as the 52% A cue moves from Expert 1 to Expert 7. Its signed positions are -1, -5/9, -1/9, and 1/3, over which boundary_gain*x^2 + spatial_tilt*x decreases throughout the allowed parameter ranges. The 99% B cue remains fixed at the right endpoint. Reliability, cue count, coalition membership, display load, and local contrast are controlled. In particular, the physical order is always low-validity A followed by high-validity B, so neither cue receives sharp-drop surprise; both receive the same small symmetric-contrast signal. The competitor should therefore shift from relatively strong A support when A occupies the privileged left boundary toward substantially greater B support when A is displaced rightward, potentially producing a modal crossover. The decisive quantitative signature is P(A at Expert 1) > P(A at Expert 3) > P(A at Expert 5) > P(A at Expert 7), against overwrite's exactly flat A-choice profile. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the response-side-balanced proportion of trials on which the subject follows the 52%-valid cue rather than the conflicting 99%-valid cue. It pools all 96 trials, yielding a stable per-subject estimate, while the exact A/B reversals remove generic response-side bias. Recency-Gated Reason Overwrite processes the 99% cue first and the 52% cue last, so it predicts a consistently high weak-cue choice rate across all four spatial positions. The competing concurrent field preserves a large reliability advantage for Expert 10; its boundary and spatial effects may help the weak cue at the left endpoint, but should produce a substantially lower weak-cue rate when averaged across the four placements. Thus the metric captures the consequential behavioral reversal—systematic selection of the objectively weaker reason—rather than relying only on the noisier spatial slope.

Source:
def metric(data: pd.DataFrame) -> float:
    weak_cue_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b

        # In this design, Expert 10 (index 9) is the 99%-valid cue.
        # The other discriminating cue is one of the 52%-valid experts.
        weak_indices = np.flatnonzero(diff[:9] != 0.0)
        if weak_indices.size != 1:
            continue

        weak_direction = diff[int(weak_indices[0])]
        weak_supported_response = 0 if weak_direction > 0.0 else 1
        weak_cue_choices.append(
            float(int(row["response"]) == weak_supported_response)
        )

    if len(weak_cue_choices) == 0:
        return float("nan")
    return float(np.mean(weak_cue_choices))

### RESULTS
- Predicted under pi_3 (simulated): 0.8175 (var=0.0031)
- Predicted under pi_14 (simulated): 0.4600 (var=0.0025)
- Observed on real data: 0.5908 (var=0.0015)

## EXPERIMENT 2 (proposed by pi_14)

### DESIGN
**Validities (n_features=15):** [0.9, 0.9, 0.9, 0.98, 0.98, 0.98, 0.5, 0.58, 0.68, 0.68, 0.68, 0.58, 0.58, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This is a novelty-threshold gradient with validity-ordered evidence held constant. Three base trials activate separate five-expert banks. In every bank, descending-validity processing yields exactly A,A,B,B,B: the 98% and 90% experts favor A, followed by the 68%, 58%, and 50% experts favoring B. Recency-Gated Reason Overwrite therefore predicts exactly the same terminal-B choice probability in all three conditions; cue count, ordered directions, overwrite dynamics, three-update fatigue, and lapse are identical. In physical order, however, each bank is A,A,B,B,B, and the validity of the B cue entering at the sole directional boundary decreases across conditions from 68% to 58% to 50%, always immediately after the 98% A cue. The advocated parallel field assigns these boundary entrants novelty signals of approximately 0.885, 1.18, and 1.38. These values move from below the 1.10–1.30 protection threshold, through its transition region, to above it. It therefore predicts a nonlinear increase in B choice, with the largest increment for the 50% entrant because its sharply contrastive B coalition receives bounded post-saturation protection. The bank locations also place the strongest-novelty entrant slightly farther left, reinforcing rather than opposing this ordering under the model's negative spatial tilt. Thus the diagnostic profile is P(B|68% entrant) < P(B|58% entrant) < P(B|50% entrant), potentially with a particularly large final step, versus overwrite's exactly flat and generally much stronger terminal-B profile. Exact A/B reversals balance response side. Six unique pairs repeated 16 times produce 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This is a low-variance indicator of strong dominance by the three-cue coalition. In every designed pair, the option with three positive ratings is the coalition supported by the final three reasons in descending-validity order; exact A/B reversals make the statistic response-side neutral. Overwrite predicts this option should be selected well above 62.5% because three successive B-directed updates usually control the retained decision even after fatigue and lapse. The advocated parallel field instead saturates both coalitions and grants only bounded novelty protection, leaving the three-cue option much closer to chance. Pooling all 96 trials makes crossing the 62.5% threshold unlikely under the parallel-field model and likely under overwrite, while the same computation remains well-defined and stable on each subject's data.

Source:
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    chose_three_cue_option = []
    for a_ratings, b_ratings, response in zip(
        data['option_a_ratings'],
        data['option_b_ratings'],
        data['response']
    ):
        a_count = float(np.sum(np.asarray(a_ratings, dtype=float)))
        b_count = float(np.sum(np.asarray(b_ratings, dtype=float)))
        r = int(response)
        chose_three = ((a_count > b_count) and (r == 0)) or ((b_count > a_count) and (r == 1))
        chose_three_cue_option.append(float(chose_three))

    dominance_rate = float(np.mean(chose_three_cue_option))
    return float(dominance_rate >= 0.625)


### RESULTS
- Predicted under pi_3 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_14 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 1.0000 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Contrast TTB-consistent choice rates between trials where the
    # equal-weight tally supports versus opposes the TTB winner.
    support_choices = []
    oppose_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        differing = np.flatnonzero(d != 0)
        if differing.size == 0:
            continue

        # Features are already ordered from highest to lowest validity.
        ttb_direction = float(np.sign(d[differing[0]]))  # +1: A, -1: B
        ttb_response = 0 if ttb_direction > 0 else 1
        chose_ttb = float(int(row["response"]) == ttb_response)

        # Positive iff the tally margin points toward the TTB winner.
        tally_a_margin = float(np.sign(d).sum())
        oriented_tally_margin = ttb_direction * tally_a_margin

        if oriented_tally_margin > 0:
            support_choices.append(chose_ttb)
        elif oriented_tally_margin < 0:
            oppose_choices.append(chose_ttb)

    if len(support_choices) == 0 or len(oppose_choices) == 0:
        return float("nan")

    return float(np.mean(support_choices) - np.mean(oppose_choices))
```

**Observed (real) value:** 0.4944 (var=0.0204)
**Predicted under pi_3:** 0.4743 (var=0.0168)
**Predicted under pi_14:** 0.3789 (var=0.0089)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Rate at which choices follow the highest-validity discriminating cue."""
    if len(data) == 0:
        return float("nan")

    matches = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Features are already ordered from highest to lowest validity.
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            # An all-tie pair provides no directional diagnosticity.
            continue

        first = int(discriminating[0])
        ttb_response = 0 if diff[first] > 0 else 1
        matches.append(float(int(row["response"]) == ttb_response))

    if not matches:
        return float("nan")
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1938 (var=0.0049)
**Predicted under pi_3:** 0.1931 (var=0.0026)
**Predicted under pi_14:** 0.4854 (var=0.0035)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Proportion of choices agreeing with the least-valid discriminating cue."""
    if len(data) == 0:
        return float("nan")

    agreements = []
    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        differing = np.flatnonzero(a != b)
        if differing.size == 0:
            continue

        # Features are displayed in descending validity order, so the
        # largest differing index is the final discriminating reason.
        final_idx = int(differing[-1])
        final_cue_choice = 0 if a[final_idx] > b[final_idx] else 1
        agreements.append(float(int(response) == final_cue_choice))

    if not agreements:
        return float("nan")
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.5750 (var=0.0026)
**Predicted under pi_3:** 0.7129 (var=0.0033)
**Predicted under pi_14:** 0.4919 (var=0.0025)

### Experiment 6
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signed association between choice and the lowest-validity (fourth) cue.
    # +1 means the response follows that cue, -1 means it opposes that cue.
    products = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_direction = np.sign(a[-1] - b[-1])  # +1 favors A, -1 favors B
        if terminal_direction == 0:
            continue
        response = int(row['response'])
        choice_direction = 1.0 if response == 0 else -1.0
        products.append(choice_direction * terminal_direction)
    if len(products) == 0:
        return float('nan')
    return float(np.mean(products))
```

**Observed (real) value:** 0.6650 (var=0.0621)
**Predicted under pi_3:** 0.5262 (var=0.0122)
**Predicted under pi_14:** -0.0242 (var=0.0099)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    block_terminal = []
    mixed_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        d = a - b

        # The block permutation has one option favored by cues 1--5 and
        # the other option favored by cues 6--10.
        is_block = (
            d.size == 10
            and np.all(d[:5] == d[0])
            and np.all(d[5:] == d[5])
            and d[0] == -d[5]
        )

        # In this design the final two cues agree. Identify whether the
        # response selected the option favored by those terminal cues.
        terminal_response = 0 if d[-1] > 0 else 1
        chose_terminal = float(int(row["response"]) == terminal_response)

        if is_block:
            block_terminal.append(chose_terminal)
        else:
            mixed_terminal.append(chose_terminal)

    if len(block_terminal) == 0 or len(mixed_terminal) == 0:
        return float("nan")

    return float(np.mean(mixed_terminal) - np.mean(block_terminal))
```

**Observed (real) value:** 0.0033 (var=0.0049)
**Predicted under pi_3:** 0.0150 (var=0.0104)
**Predicted under pi_14:** -0.0050 (var=0.0087)

### Experiment 8
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    early_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)

        # Restrict to the 7- and 8-reason overload conditions.
        if discriminating.size < 7:
            continue

        # The first discriminating cue identifies the option favored by the
        # high-validity, early run, regardless of physical A/B orientation.
        first_direction = diff[discriminating[0]]
        response = int(row["response"])
        chose_early_option = (
            (first_direction > 0 and response == 0) or
            (first_direction < 0 and response == 1)
        )
        early_choices.append(float(chose_early_option))

    if not early_choices:
        return float("nan")
    return float(np.mean(early_choices))
```

**Observed (real) value:** 0.3067 (var=0.0051)
**Predicted under pi_3:** 0.3379 (var=0.0051)
**Predicted under pi_14:** 0.5279 (var=0.0042)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    low_follow = []
    high_follow = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = diff[diff != 0]

        # The strongest predicted chunking contrast occurs for the 8- and
        # 10-reason displays.
        if discriminating.size < 8:
            continue

        directions = np.sign(discriminating)
        n_switches = int(np.sum(directions[1:] != directions[:-1]))

        # Response coding is 0=A and 1=B. Determine whether the response
        # follows the terminal discriminating cue, independent of side.
        terminal_response = 0 if directions[-1] > 0 else 1
        follows_terminal = float(int(row["response"]) == terminal_response)

        if n_switches <= 1:
            low_follow.append(follows_terminal)
        elif n_switches >= 4:
            high_follow.append(follows_terminal)

    if len(low_follow) == 0 or len(high_follow) == 0:
        return float("nan")

    contrast = float(np.mean(low_follow) - np.mean(high_follow))

    # A preregisterable practical-significance diagnostic: whether coherent
    # low-switch sequences have more than an 8-percentage-point terminal-cue
    # advantage over switch-dense sequences.
    return float(contrast > 0.08)

```

**Observed (real) value:** 0.0000 (var=0.2016)
**Predicted under pi_3:** 0.0000 (var=0.1824)
**Predicted under pi_14:** 0.0000 (var=0.1924)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float('nan')

    agreements = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Direction of the terminal discriminating cue in displayed order.
        terminal_direction = diff[discriminating[-1]]
        terminal_response = 0 if terminal_direction > 0 else 1
        agreements.append(float(int(row['response']) == terminal_response))

    if len(agreements) == 0:
        return float('nan')
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.1721 (var=0.0112)
**Predicted under pi_3:** 0.2754 (var=0.0030)
**Predicted under pi_14:** 0.4813 (var=0.0033)

### Experiment 11
**Design**
  A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # Features are already arranged in descending instructed validity.
        first_direction = float(np.sign(diff[int(discriminating[0])]))
        choice_direction = 1.0 if int(row["response"]) == 0 else -1.0
        scores.append(first_direction * choice_direction)

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(np.asarray(scores, dtype=float)))
```

**Observed (real) value:** -0.0325 (var=0.0099)
**Predicted under pi_3:** -0.0083 (var=0.0103)
**Predicted under pi_14:** 0.1450 (var=0.0175)

### Experiment 12
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Coherence-weighted agreement with the least-valid terminal cue."""
    if len(data) == 0:
        return float("nan")

    pattern_weights = {
        (1, 1, 1, 1): 1.00,   # fully coherent
        (-1, 1, 1, 1): 0.80,  # earliest cue conflicts
        (1, -1, 1, 1): 0.65,  # second cue conflicts
        (-1, -1, 1, 1): 0.35  # both prefix cues conflict
    }

    weighted_agreement = 0.0
    total_weight = 0.0

    for a_cell, b_cell, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        diff = a - b
        if diff.size != 4 or diff[3] == 0:
            continue

        # Orient every trial so that the terminal (least-valid) cue is +1.
        terminal_sign = float(np.sign(diff[3]))
        relative_pattern = tuple(
            int(np.sign(x) * terminal_sign) for x in diff
        )
        weight = pattern_weights.get(relative_pattern, 0.0)
        if weight <= 0.0:
            continue

        terminal_favoring_response = 0 if terminal_sign > 0 else 1
        agreement = float(int(response) == terminal_favoring_response)
        weighted_agreement += weight * agreement
        total_weight += weight

    if total_weight <= 0.0:
        return float("nan")
    return float(weighted_agreement / total_weight)
```

**Observed (real) value:** 0.8748 (var=0.0069)
**Predicted under pi_3:** 0.7843 (var=0.0034)
**Predicted under pi_14:** 0.6261 (var=0.0019)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    scores = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        m = int(np.count_nonzero(diff))

        # The diagnostic manageable-load conflicts contain 5 or 6
        # discriminating cues. The final (lowest-validity) cue is index 9.
        if m not in (5, 6) or diff[-1] == 0:
            continue

        terminal_response = 0 if diff[-1] > 0 else 1
        scores.append(float(int(row["response"]) == terminal_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.8425 (var=0.0123)
**Predicted under pi_3:** 0.6863 (var=0.0047)
**Predicted under pi_14:** 0.4788 (var=0.0041)

### Experiment 14
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Side-balanced rate of choosing the 3-to-2 majority in strong-coalition trials."""
    majority_a = []
    majority_b = []
    strong_set = {0, 1, 2, 11, 12}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = set(np.flatnonzero(diff != 0).tolist())
        if discriminating != strong_set:
            continue

        response = int(row["response"])
        majority_direction = float(np.sum(np.sign(diff)))
        if majority_direction > 0:
            majority_a.append(float(response == 0))
        elif majority_direction < 0:
            majority_b.append(float(response == 1))

    if majority_a and majority_b:
        return float(0.5 * (np.mean(majority_a) + np.mean(majority_b)))

    pooled = majority_a + majority_b
    if pooled:
        return float(np.mean(pooled))
    return 0.5

```

**Observed (real) value:** 0.1713 (var=0.0095)
**Predicted under pi_3:** 0.2650 (var=0.0077)
**Predicted under pi_14:** 0.5112 (var=0.0056)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    d = data.copy()
    if len(d) == 0:
        return float('nan')

    # The competing theory's contextual calibration has reached full weight
    # well before this point; the advocated theory has no history dependence.
    d['_trial_index'] = d.groupby('subject_id', sort=False).cumcount()
    d = d.loc[d['_trial_index'] >= 24].copy()
    if len(d) == 0:
        return float('nan')

    def terminal_match(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        terminal_diff = a[-1] - b[-1]
        if terminal_diff > 0:
            terminal_response = 0
        elif terminal_diff < 0:
            terminal_response = 1
        else:
            return np.nan
        return float(int(row['response']) == terminal_response)

    d['_terminal_match'] = d.apply(terminal_match, axis=1)
    d['_stimulus_key'] = d.apply(
        lambda r: (tuple(r['option_a_ratings']), tuple(r['option_b_ratings'])),
        axis=1
    )

    # Equal-weight subject-by-stimulus cells, eliminating variance caused by
    # random trial order and unequal late-block counts of the six pairs.
    cells = d.groupby(
        ['subject_id', '_stimulus_key'], sort=False
    )['_terminal_match'].mean()
    return float(cells.mean())
```

**Observed (real) value:** 0.8810 (var=0.0070)
**Predicted under pi_3:** 0.7201 (var=0.0040)
**Predicted under pi_14:** 0.4336 (var=0.0032)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Classify each trial structurally and code whether the response followed
    # the physically terminal discriminating reason. In this design that
    # reason is also the final reason in instructed-validity scan order.
    records = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        idx = np.flatnonzero(diff != 0)
        if idx.size == 0:
            continue

        terminal_idx = int(idx[-1])
        terminal_is_b = bool(diff[terminal_idx] < 0)
        chose_terminal = float(int(row["response"]) == int(terminal_is_b))

        if idx.size > 1:
            largest_gap = int(np.max(np.diff(idx) - 1))
        else:
            largest_gap = 0
        foregrounded = bool(terminal_idx == (len(diff) - 1) and largest_gap > 0)
        records.append((foregrounded, terminal_is_b, chose_terminal))

    if not records:
        return float("nan")

    frame = pd.DataFrame(
        records, columns=["foregrounded", "terminal_is_b", "chose_terminal"]
    )

    # Compute the foregrounding contrast separately for terminal-A and
    # terminal-B trials, then average. This exactly removes any stable A/B
    # response-side preference rather than allowing it to inflate variance.
    side_contrasts = []
    for terminal_is_b in (False, True):
        side = frame[frame["terminal_is_b"] == terminal_is_b]
        fg = side.loc[side["foregrounded"], "chose_terminal"]
        clustered = side.loc[~side["foregrounded"], "chose_terminal"]
        if len(fg) > 0 and len(clustered) > 0:
            side_contrasts.append(float(fg.mean() - clustered.mean()))

    if not side_contrasts:
        return float("nan")
    return float(np.mean(side_contrasts))
```

**Observed (real) value:** 0.0192 (var=0.0062)
**Predicted under pi_3:** -0.0217 (var=0.0090)
**Predicted under pi_14:** -0.0733 (var=0.0126)

### Experiment 17
**Design**
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Winner-choice rate at each unanimous evidence load.
    correct_by_load = {2: [], 4: [], 7: [], 10: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        load = int(np.count_nonzero(diff))
        net_direction = float(np.sum(diff))

        if load not in correct_by_load or net_direction == 0.0:
            continue

        response = int(row["response"])
        favored_response = 0 if net_direction > 0.0 else 1
        correct_by_load[load].append(float(response == favored_response))

    rates = {}
    for load in (2, 4, 7, 10):
        values = correct_by_load[load]
        if len(values) == 0:
            return float("nan")
        rates[load] = float(np.mean(values))

    # Endpoint-emphasized, zero-sum decline contrast. Positive values indicate
    # that unanimous evidence becomes less effective as more reasons are added.
    return float(
        0.75 * rates[2]
        + 0.25 * rates[4]
        - 0.25 * rates[7]
        - 0.75 * rates[10]
    )
```

**Observed (real) value:** 0.0471 (var=0.0135)
**Predicted under pi_3:** 0.1525 (var=0.0093)
**Predicted under pi_14:** 0.0292 (var=0.0079)

### Experiment 18
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.asarray([0.99, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.51], dtype=float)
    reliability = np.log(np.clip(validities, 0.5001, 0.999) / (1.0 - np.clip(validities, 0.5001, 0.999)))
    scale = float(np.mean(reliability[reliability > 0.0]))
    strengths = np.power(np.clip(reliability / scale, 0.15, 6.0), 0.30)

    def summarize(a, b):
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        idx = np.flatnonzero(d != 0.0)
        if idx.size == 0:
            return None
        dirs = np.sign(d[idx])
        m = int(dirs.size)
        switches = np.flatnonzero(dirs[1:] != dirs[:-1]) + 1

        run_lengths = []
        run = 1
        for k in range(1, m):
            if dirs[k] == dirs[k - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)

        accumulator = 0.0
        susceptibility = []
        k = 0
        while k < m:
            direction = float(dirs[k])
            end = k + 1
            while end < m and dirs[end] == direction:
                end += 1
            incoming = strengths[idx[k:end]]
            run_length = end - k
            if accumulator != 0.0 and np.sign(accumulator) != direction:
                incumbent = abs(accumulator) * (0.55 ** run_length)
                attack = 0.0
                for s in incoming:
                    attack = 0.55 * attack + float(s)
                advantage = (attack - incumbent) / max(attack + incumbent, 1e-12)
                susceptibility.append(1.0 / (1.0 + np.exp(-4.0 * np.clip(advantage, -15.0, 15.0))))
            for s in incoming:
                accumulator = 0.55 * accumulator + direction * float(s)
            k = end

        reversal_susceptibility = float(np.mean(susceptibility)) if susceptibility else 0.35
        fractions = switches.astype(float) / float(m) if switches.size else np.asarray([], dtype=float)
        canonical = tuple(int(x) for x in (dirs * dirs[0]))
        return {
            'terminal_singleton': float(len(run_lengths) > 1 and run_lengths[-1] == 1),
            'two_run': float(len(run_lengths) == 2),
            'switch_density': float(switches.size) / float(max(1, m - 1)),
            'susceptibility': reversal_susceptibility,
            'fractions': fractions,
            'canonical': canonical,
            'terminal_direction': float(dirs[-1])
        }

    def context_value(summaries, learning_rate):
        ages = np.arange(len(summaries) - 1, -1, -1, dtype=float)
        w = np.power(1.0 - learning_rate, ages)
        if len(summaries) > 1:
            w[-1] *= 0.35
        w = w / max(float(np.sum(w)), 1e-12)

        terminal = float(np.sum([wi * s['terminal_singleton'] for wi, s in zip(w, summaries)]))
        two_run = float(np.sum([wi * s['two_run'] for wi, s in zip(w, summaries)]))
        switch_density = float(np.sum([wi * s['switch_density'] for wi, s in zip(w, summaries)]))
        susceptibility = float(np.sum([wi * s['susceptibility'] for wi, s in zip(w, summaries)]))

        hist = np.zeros(5, dtype=float)
        mass = 0.0
        for wi, s in zip(w, summaries):
            fr = s['fractions']
            if fr.size == 0:
                continue
            contribution = float(wi) / float(fr.size)
            for f in fr:
                j = min(4, max(0, int(np.floor(5.0 * float(f)))))
                hist[j] += contribution
                mass += contribution
        if mass > 0.0:
            p = hist / mass
            concentration = float(np.sum(p ** 2))
            positive = p[p > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)) / np.log(5.0))
        else:
            concentration = 0.2
            entropy = 1.0

        validity_gradient = float(np.clip(np.std(reliability) / max(np.mean(reliability), 1e-8), 0.0, 2.0))
        score = (1.30
                 + 1.20 * (terminal - 0.30)
                 + 1.20 * (concentration - 0.30)
                 + 0.80 * (two_run - 0.50)
                 - 1.50 * (entropy - 0.70)
                 - 0.70 * (switch_density - 0.25)
                 - 0.20 * validity_gradient
                 + 0.85 * (susceptibility - 0.35))
        return 1.0 / (1.0 + np.exp(-1.90 * np.clip(score - 1.30, -30.0, 30.0)))

    def one_subject(frame):
        summaries = []
        records = []
        for _, row in frame.iterrows():
            s = summarize(row['option_a_ratings'], row['option_b_ratings'])
            if s is None:
                continue
            summaries.append(s)

            # Average across the advocated learning-rate range to reduce
            # sensitivity to unknown individual parameter values.
            x = float(np.mean([context_value(summaries, lr) for lr in (0.05, 0.085, 0.12)]))

            # Retain conflict sequences only. The three canonical geometries
            # are +-----, +++-, and +++++- (including their A/B reversals).
            if len(s['canonical']) > 1 and len(set(s['canonical'])) > 1:
                chosen_direction = 1.0 if int(row['response']) == 0 else -1.0
                y = 1.0 if chosen_direction == s['terminal_direction'] else -1.0
                records.append((s['canonical'], x, y))

        if len(records) < 4:
            return 0.0

        residual_x = np.zeros(len(records), dtype=float)
        keys = [r[0] for r in records]
        xvals = np.asarray([r[1] for r in records], dtype=float)
        yvals = np.asarray([r[2] for r in records], dtype=float)

        for key in set(keys):
            loc = np.asarray([k == key for k in keys], dtype=bool)
            residual_x[loc] = xvals[loc] - float(np.mean(xvals[loc]))

        rms = float(np.sqrt(np.mean(residual_x ** 2)))
        if not np.isfinite(rms) or rms < 1e-10:
            return 0.0
        z = residual_x / rms
        return float(np.mean(z * yvals))

    if 'subject_id' in data.columns:
        values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]
        return float(np.mean(values)) if values else 0.0
    return one_subject(data)

```

**Observed (real) value:** 0.0311 (var=0.0170)
**Predicted under pi_3:** -0.0111 (var=0.0098)
**Predicted under pi_14:** 0.0170 (var=0.0111)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    coherent_terminal = []
    fragmented_terminal = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        directions = np.sign(diff[diff != 0])
        m = int(directions.size)

        if m not in (5, 6):
            continue

        # The diagnostic displays have a singleton terminal reversal.
        terminal_singleton = bool(
            m >= 2 and directions[-1] != directions[-2]
        )
        if not terminal_singleton:
            continue

        terminal_direction = float(directions[-1])
        response = int(row["response"])
        chose_terminal_option = float(
            (terminal_direction > 0 and response == 0)
            or (terminal_direction < 0 and response == 1)
        )

        prefix = directions[:-1]
        coherent = bool(
            prefix.size > 0
            and np.all(prefix == prefix[0])
            and prefix[0] != terminal_direction
        )

        if coherent:
            coherent_terminal.append(chose_terminal_option)
        else:
            fragmented_terminal.append(chose_terminal_option)

    if len(coherent_terminal) == 0 or len(fragmented_terminal) == 0:
        return float("nan")

    raw_contrast = (
        float(np.mean(coherent_terminal))
        - float(np.mean(fragmented_terminal))
    )

    # A 10-percentage-point dead band treats smaller differences as negligible;
    # effects of 30 percentage points or more receive the ceiling score.
    return float(np.clip((raw_contrast - 0.10) / 0.20, 0.0, 1.0))
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_3:** 0.0000 (var=0.0227)
**Predicted under pi_14:** 0.0000 (var=0.0031)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Code whether the response selected the option favored by the final
    # discriminating expert in descending-validity order. In this experiment,
    # feature columns are already ordered from highest to lowest validity.
    by_load = {4: [], 5: [], 6: [], 7: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        load = int(discriminating.size)
        if load not in by_load:
            continue

        final_index = int(discriminating[-1])
        terminal_option = 0 if diff[final_index] > 0 else 1
        terminal_chosen = float(int(row["response"]) == terminal_option)
        by_load[load].append(terminal_chosen)

    if any(len(by_load[m]) == 0 for m in (4, 5, 6, 7)):
        return float("nan")

    p = {m: float(np.mean(by_load[m])) for m in (4, 5, 6, 7)}
    # Average of the upward boundary jump (4->5) and downward boundary
    # jump in terminal choice (6->7).
    return float(0.5 * ((p[5] - p[4]) + (p[6] - p[7])))
```

**Observed (real) value:** 0.0262 (var=0.0075)
**Predicted under pi_3:** -0.0175 (var=0.0140)
**Predicted under pi_14:** 0.0169 (var=0.0134)

### Experiment 21
**Design**
  A=[0, 1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Select sequences whose recency-weighted evidence strongly supports the
    # terminal reason. These are trials 1, 7, 8 and their A/B reversals in the
    # specified design, without relying on trial numbers or presentation order.
    retention_probe = 0.46
    conformity = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        directions = np.sign(diff[diff != 0.0])
        if directions.size == 0:
            continue

        terminal_direction = float(directions[-1])
        m = int(directions.size)
        aligned_field = 0.0
        for pos, direction in enumerate(directions):
            lag = (m - 1) - pos
            aligned_field += (
                terminal_direction
                * float(direction)
                * retention_probe ** float(lag)
            )

        if aligned_field >= 1.0:
            terminal_response = 0 if terminal_direction > 0.0 else 1
            conformity.append(float(int(row["response"]) == terminal_response))

    if not conformity:
        return float("nan")
    return float(np.mean(conformity))
```

**Observed (real) value:** 0.4978 (var=0.0077)
**Predicted under pi_3:** 0.6767 (var=0.0091)
**Predicted under pi_14:** 0.4928 (var=0.0087)

### Experiment 22
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Weak- versus strong-terminal increase in choosing the three-cue coalition."""
    if data is None or len(data) == 0:
        return float("nan")

    weak_choices = []
    strong_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size != 4:
            continue

        # In this design, the first three discriminating experts constitute
        # the repeated coalition and the last one is the opposing terminal cue.
        terminal_idx = int(discriminating[-1])
        coalition_supports_a = bool(diff[discriminating[0]] > 0)
        response = int(row["response"])
        chose_coalition = float(
            (response == 0 and coalition_supports_a)
            or (response == 1 and not coalition_supports_a)
        )

        # Strong terminals: validities .86 and .70 (indices 3 and 5).
        # Weak terminals: validities .505 and .50 (indices 10 and 11).
        if terminal_idx in (3, 5):
            strong_choices.append(chose_coalition)
        elif terminal_idx in (10, 11):
            weak_choices.append(chose_coalition)

    if len(weak_choices) == 0 or len(strong_choices) == 0:
        return float("nan")

    return float(np.mean(weak_choices) - np.mean(strong_choices))
```

**Observed (real) value:** -0.1813 (var=0.0102)
**Predicted under pi_3:** 0.0013 (var=0.0091)
**Predicted under pi_14:** -0.0481 (var=0.0156)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    endpoint_choices = {0: [], 19: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        response = int(row["response"])

        # In this design, one option carries the singleton 96%-valid cue,
        # whereas the other carries four lower-validity cues.
        if np.sum(a) == 1 and np.sum(b) == 4:
            singleton = a
            chose_singleton = float(response == 0)
        elif np.sum(b) == 1 and np.sum(a) == 4:
            singleton = b
            chose_singleton = float(response == 1)
        else:
            continue

        positions = np.flatnonzero(singleton != 0)
        if positions.size == 1:
            pos = int(positions[0])
            if pos in endpoint_choices:
                endpoint_choices[pos].append(chose_singleton)

    if len(endpoint_choices[0]) == 0 or len(endpoint_choices[19]) == 0:
        return float("nan")

    left_rate = float(np.mean(endpoint_choices[0]))
    right_rate = float(np.mean(endpoint_choices[19]))
    return float(abs(right_rate - left_rate))
```

**Observed (real) value:** 0.3900 (var=0.0214)
**Predicted under pi_3:** 0.0167 (var=0.0062)
**Predicted under pi_14:** 0.0342 (var=0.0077)

### Experiment 24
**Design**
  A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([
        0.86, 0.98, 0.98, 0.86,
        0.53, 0.86, 0.86, 0.53,
        0.98, 0.53, 0.53, 0.98,
        0.52, 0.52, 0.52, 0.52
    ], dtype=float)

    large_drop_choices = []
    small_drop_choices = []

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0.0)
        if discriminating.size < 2:
            continue

        # Coalitions are defined in displayed (left-to-right) order. In this
        # design the final discriminating cue is the terminal singleton and
        # the preceding discriminating cue is the incumbent coalition's exit.
        terminal_index = int(discriminating[-1])
        exit_index = int(discriminating[-2])
        boundary_drop = float(validities[exit_index] - validities[terminal_index])

        response = int(row["response"])
        terminal_favors_a = bool(diff[terminal_index] > 0.0)
        chose_terminal = float(
            (terminal_favors_a and response == 0)
            or ((not terminal_favors_a) and response == 1)
        )

        if boundary_drop > 0.20:
            large_drop_choices.append(chose_terminal)
        else:
            small_drop_choices.append(chose_terminal)

    if len(large_drop_choices) == 0 or len(small_drop_choices) == 0:
        return 0.0

    return float(
        np.mean(large_drop_choices) - np.mean(small_drop_choices)
    )
```

**Observed (real) value:** 0.1750 (var=0.0208)
**Predicted under pi_3:** 0.0117 (var=0.0066)
**Predicted under pi_14:** 0.1617 (var=0.0104)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Score whether each response follows the physically last discriminating
    # cue, then contrast crossover conditions 1-2 against conditions 3-4.
    condition_scores = {0: [], 1: [], 2: [], 3: []}

    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        diff = a - b
        discriminating = np.flatnonzero(diff != 0)
        if discriminating.size == 0:
            continue

        # In this design, the leftmost discriminating location (0, 1, 2, or 3)
        # uniquely identifies the reliability-crossover condition, including
        # the response-side-reversed versions of each pair.
        condition = int(discriminating[0])
        if condition not in condition_scores:
            continue

        last_direction = diff[int(discriminating[-1])]
        terminal_response = 0 if last_direction > 0 else 1
        follows_terminal = 1.0 if int(row["response"]) == terminal_response else -1.0
        condition_scores[condition].append(follows_terminal)

    means = {}
    for condition, values in condition_scores.items():
        if len(values) == 0:
            return float("nan")
        means[condition] = float(np.mean(values))

    early_reliability_regime = 0.5 * (means[0] + means[1])
    late_reliability_regime = 0.5 * (means[2] + means[3])
    return float(early_reliability_regime - late_reliability_regime)

```

**Observed (real) value:** 0.4183 (var=0.0370)
**Predicted under pi_3:** 0.8758 (var=0.0591)
**Predicted under pi_14:** 0.0075 (var=0.0288)

### Experiment 26
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return float("nan")

    scores = []
    for a_ratings, b_ratings, response in zip(
        data["option_a_ratings"],
        data["option_b_ratings"],
        data["response"]
    ):
        a = np.asarray(a_ratings)
        b = np.asarray(b_ratings)
        terminal_diff = float(a[-1] - b[-1])
        if terminal_diff == 0.0:
            continue
        terminal_supported_response = 0 if terminal_diff > 0.0 else 1
        scores.append(float(int(response) == terminal_supported_response))

    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4192 (var=0.0032)
**Predicted under pi_3:** 0.1706 (var=0.0019)
**Predicted under pi_14:** 0.4319 (var=0.0023)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the complete set of experiments, pi_3 is the stronger task-general account, although its current overwrite mechanism is too rigid. It is exact or substantially closer on many diagnostics of late-cue control: Experiments 2, 3, 4, 6, 8, 10\u201316, and several null contrasts. Especially important are Experiment 4 (observed highest-validity-cue choice 0.1938; pi_3 0.1931 versus pi_14 0.4854), Experiment 6 (terminal association 0.6650; pi_3 0.5262 versus pi_14 -0.0242), and Experiments 12, 13, and 15, where observed terminal-cue choice is very high and pi_3 is consistently much closer. These results are structurally difficult for pi_14 because its parallel coalition field has no sequential commitment, decay, or privileged late readout. Experiment 2 also favors pi_3 categorically, although its thresholded metric discards the underlying dominance rate and the observed between-subject variance of 0.2496 shows that pi_3's simulated zero variance is not a complete fit. Conversely, pi_3 systematically overstates invariant overwrite. In Experiment 1 the observed weak-cue rate of 0.5908 lies between pi_3's 0.8175 and pi_14's 0.4600. Pi_3 also overpredicts terminal control in Experiments 5, 21, and 26 and misses the validity/location effects in Experiment 22. Pi_14 performs well on several important results: the modest unanimous-load decline in Experiment 17, the near-null context association in Experiment 18, chance-level terminal conformity in Experiment 21, Experiment 24's local validity-drop effect, and Experiment 26. Nevertheless, its successes do not compensate for its broad failure on the strong recency results. Moreover, neither theory explains the large endpoint asymmetry in Experiment 23 (observed 0.3900 versus 0.0167 and 0.0342), and both miss much of the crossover magnitude in Experiment 25. The combined evidence therefore supports neither pure mandatory validity-ordered overwrite nor a purely parallel coalition field. It supports a sequentially expressed decision process whose effective order and update strength are jointly determined by instructed validity, physical location, local contrast, and existing coalition strength. Because adding sequential state to pi_14 would directly contradict its defining prohibition on scan order and sequential commitment, this is a theory-level failure rather than a parameterization problem.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_14 with a new Salience-Guided Sequential Integration theory. The display should first generate a parallel preattentive priority map from instructed validity, endpoint salience, stable directional scan bias, and local validity/directional contrast. Cues are then sampled or refreshed in the priority-dependent order, and their signed evidence enters a leaky bounded accumulator. Recent evidence receives greater readout weight because earlier evidence decays, but updates are graded rather than categorical: update strength should depend on cue reliability, contrast surprise, the coherence and strength of the incumbent coalition, and capacity load. Thus a late weak cue can reverse an earlier decision without doing so invariably; several agreeing cues can establish an anchor that sometimes resists a singleton; and conspicuous endpoints or sharp validity drops can alter which reason is processed or refreshed last. Use task-invariant subject-level parameters for validity sensitivity, spatial scan direction, endpoint salience, contrast gain, accumulator retention, anchoring, capacity, and lapse, with a hierarchical distribution to represent the observed heterogeneity. The regenerated theory should be jointly constrained to produce partial weak-cue reversal in Experiment 1, three-cue dominance in Experiment 2, strong but nonuniform recency in Experiments 4, 6, 12, 13, and 15, moderated terminal influence in Experiments 10, 21, and 26, the small unanimous-load decline in Experiment 17, the large endpoint asymmetry in Experiment 23, the local-drop effect in Experiment 24, and the intermediate crossover in Experiment 25. It should predict trial-level probabilities and graded condition profiles rather than being tuned primarily to thresholded metrics."
}
```

## Usage

```json
{
  "input_tokens": 30918,
  "output_tokens": 2716
}
```
