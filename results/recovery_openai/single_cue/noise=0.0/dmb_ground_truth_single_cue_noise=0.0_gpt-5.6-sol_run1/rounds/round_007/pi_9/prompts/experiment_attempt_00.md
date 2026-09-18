# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Distribution-Calibrated Competitive Accumulation with Susceptibility-Dominant Context Learning. People inspect discriminating cues in descending instructed-validity order and encode each attended cue as signed evidence in a continuously evolving competitive accumulator. Opposing evidence inhibits the active commitment, while a continuous experiment-level task set controls retention and overwrite. The task set is learned primarily from how readily opposing runs reverse a reference accumulator across the experiment's sequence families. Coarse ordinal properties such as terminal singletons, two-run structure, and reversal-location concentration provide only weak, centered calibration rather than directly producing overwrite. Reliability-weighted coalitions can reverse the sequential state, but their access to choice is proportional to normalized evidence margin. Physical gaps are not represented, and reliability depends on conflict geometry and evidence margin rather than discriminating-cue count.

**Parameters:**
- validities: validities
- attention_rate: [0.88, 1.0]
- integration_retention: [0.84, 0.97]
- overwrite_retention: [0.14, 0.36]
- conflict_inhibition: [0.35, 1.20]
- coalition_weight: [0.30, 0.62]
- validity_sensitivity: [0.15, 0.45]
- context_sensitivity: [1.4, 2.4]
- context_threshold: [1.0, 1.6]
- context_learning_rate: [0.05, 0.12]
- susceptibility_coupling: [0.60, 1.10]
- response_gain: [1.8, 3.8]
- lapse_rate: [0.04, 0.15]

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** Recency-Gated Reason Overwrite with Processing Fatigue. People inspect cues sequentially in descending order of instructed validity and continue after finding a discriminating cue. An attended discriminating cue establishes a provisional choice, and each subsequently attended discriminating cue can overwrite that choice. Consequently, the final attended reason has disproportionate control. Occasional failures to attend a cue preserve an earlier reason, while response lapses dilute strong recency effects. In addition, processing more than two discriminating reasons produces task-invariant update fatigue: the retained decision is expressed with progressively lower reliability as the discriminating-cue count increases. This preserves strong second-reason overwrite on short conflicts while moderating near-deterministic last-reason control on longer sequences.

**Parameters:**
- validities: validities
- cue_attention: [0.95, 1.0]
- overwrite_rate: [0.95, 1.0]
- update_fatigue: [0.06, 0.17]
- lapse_rate: [0.15, 0.41]

**`predict source code`:**
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

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design creates a within-subject quantitative dissociation between Take The Best (TTB) and Tallying. The validities impose an unambiguous cue order. In each base pair, TTB favors A because the highest-validity cue that discriminates favors A; therefore, apart from common beta/epsilon noise, TTB predicts the same probability of choosing its winner across all eight base conditions. By contrast, the signed Tallying margins for A are -5, -3, -1, 0, +2, +3, +7, and -4, respectively. Tallying consequently predicts a strong monotonic probability gradient, including choices opposite TTB in five conditions, exact 50/50 responding in the zero-margin condition, and varying confidence where the rules agree. Every base pair is accompanied by its A/B reversal, balancing response side while preserving the diagnostic evidence structure. With 16 unique pairs, each can be repeated six times for exactly 96 trials.

[1] This is a concentrated cancellation test. In every unique pair, A and B win on exactly the same number of discriminating features: seven base pairs contain one win per option, and the eighth contains three wins per option. Tallying therefore predicts exactly P(A)=P(B)=0.50 on every trial, independently of beta and epsilon. Nevertheless, each pair has a uniquely defined first discriminating cue in validity order. That cue favors A in the eight base pairs and B in their exact reversals. Take The Best therefore predicts a systematic preference for the designated winner, with probability (1-epsilon)*sigmoid(beta)+epsilon/2, rather than chance. The first discriminating cue ranges from validity rank 1 through rank 7, while one condition includes six conflicting discriminations, demonstrating that the tally cancellation prediction is invariant to both cue rank and the amount of opposing evidence. Exact A/B reversals balance response side. With 16 unique pairs repeated six times, the experiment has exactly 96 trials.

[2] This design holds the Tallying margin exactly constant while manipulating the number and recency order of discriminating reasons. In each of the four base pairs, A wins exactly one more feature than B: the tallies are 2–1, 3–2, 4–3, and 5–4. Tallying therefore predicts the identical base-trial probability P(A) = (1-epsilon)*sigmoid(beta) + epsilon/2 at every discriminating-cue count. In contrast, cues are encountered in descending validity order, and the final discriminating cue always favors B. Recency-Gated Reason Overwrite consequently predicts B rather than the tally winner A. Moreover, the base sequences all end in an A,A,B motif, keeping the terminal overwrite structure comparable, while the number of discriminating cues increases from 3 to 5 to 7 to 9. Its fatigue mechanism predicts that the deviation toward the final-cue winner B will progressively contract toward 0.50; each two-cue increase applies an additional retention factor of (1-update_fatigue)^2, approximately 0.69–0.88 over the allowed fatigue range. Exact A/B reversals balance response side and produce the mirror-image predictions. Thus, Tallying predicts a count-invariant preference for the feature-majority option, whereas the advocated theory predicts an opposing last-reason preference with a systematic fatigue gradient. There are eight unique pairs, each repeated 12 times for exactly 96 trials.

[3] This is a fixed-evidence permutation test. Every trial has four discriminating cues, with exactly two favoring A and two favoring B. Tallying therefore predicts P(A)=0.50 on all six trial types, independently of beta and epsilon. The trials exhaust all validity-ordered permutations of two A-favoring and two B-favoring reasons, so the total evidence and processing-fatigue factor are held constant while reason order changes. Recency-Gated Reason Overwrite instead predicts that the fourth (lowest-validity) cue controls the modal response: AABB, ABAB, and BAAB should favor B, whereas BBAA, BABA, and ABBA should favor A. It also predicts a secondary quantitative ordering: sequences ending with two concordant reasons (AABB and BBAA) should generally express the terminal preference more strongly than sequences requiring a single final overwrite, because a missed final update is less consequential when the preceding reason already supports the same option. Thus, systematic terminal-cue preferences—and especially terminal-run strength differences—contradict Tallying's exact chance prediction. The six conditions form three exact A/B-reversal pairs, balance response side, and are each repeated 16 times for 96 trials.

[4] This is a fixed-length, fixed-tally permutation test targeting the competing theory's overload-induced, validity-biased accessibility. All trials contain 10 discriminating cues, with five favoring each option, so discriminating-cue count, fatigue, and total directional tally are held constant. In the two base trials, the final two cues both favor B. The first base sequence is AAAAABBBBB: A owns the five most valid cues, whereas B owns the five least valid cues. The second is BBBAAAAABB: B owns the three most valid cues and the final two cues, while A owns the middle five. Recency-Gated Reason Overwrite predicts similar B preferences in both base conditions because two consecutive terminal B reasons almost completely overwrite any earlier state; with ten reasons, both receive the identical fatigue factor (1-update_fatigue)^8. Even at the lowest attention and overwrite rates, two terminal overwrites leave less than about 1% of a fully conflicting prior state, so the different prefixes should have little effect. In contrast, the retrospective-sampling theory predicts a large permutation effect. At m=10, cubic interference nearly eliminates its ordinal-recency gradient and retrieval approaches a validity-biased asymptote. It should therefore favor A in AAAAABBBBB, where A possesses all five high-validity reasons, but favor B in BBBAAAAABB, where B possesses the three strongest reasons as well as the terminal reasons. Thus, the critical result is not merely overall accuracy but the within-subject contrast: the advocated theory predicts nearly equal terminal-option choice rates across the two base permutations, whereas the competitor predicts a substantial shift, potentially crossing 0.50. Exact A/B reversals balance response side. Four unique pairs repeated 24 times yield 96 trials.

[5] This is a nested conflict-length crossover test. In every base trial, the discriminating cues occur in descending-validity order as an initial run favoring A followed by a terminal run favoring B. The 2-cue condition uses the nearly matched 69% and 68% experts (AB), and the 4-cue condition uses the nearly matched 70%–67% experts (AABB). The advocated retrospective-sampling theory should therefore favor terminal option B at both short lengths: its two-reason terminal-recency floor is especially strong, and recency remains fully expressed at four reasons. The 7- and 8-cue conditions add highly valid early A reasons and low-validity terminal B reasons. Once conflict exceeds four reasons, the advocated theory's cubic interference sharply flattens recency while accessibility becomes validity-biased. It should consequently cross over to A for AAAABBB and AAAABBBB, aided by A's ownership of the 99% and 95% experts (and by a 4–3 reason advantage at length 7). The competing overwrite theory predicts no such sign reversal: its final attended cue and redundant terminal B run should preserve a B preference at every length. Its update-fatigue mechanism predicts only a smooth contraction of that B preference toward 0.50 as length increases. Thus, the key quantitative signature is an approximately B, B, A, A profile across 2, 4, 7, and 8 discriminating reasons under retrospective sampling, versus B preferences throughout with gradually declining magnitude under overwrite. The shorter conditions deliberately use experts with similar validities, ensuring that their expected B preference is driven by recency rather than a large reliability difference. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield exactly 96 trials.

[6] This is a switch-density-by-length test with a protected two-reason suffix. At each discriminating-cue count (6, 8, or 10), the low-switch and high-switch base trials contain equal numbers of A- and B-favoring cues and end with two B-favoring cues. The low-switch sequences are AAABBB, AAAABBBB, and AAAAABBBBB; the matched high-switch sequences are AABABB, ABABAABB, and ABABABAABB. Recency-Gated Reason Overwrite predicts nearly equal B-choice rates for the two patterns at each length: two terminal B reasons reduce any effect of the prefix to at most roughly (1-cue_attention*overwrite_rate)^2 before common fatigue and lapse transformations. It predicts primarily a shared contraction toward 0.50 as length increases because both members of each pair have the same discriminating-cue count. Directional-Run Chunking instead predicts a growing segmentation contrast. The low-switch sequences retain a coherent terminal B chunk, whereas the high-switch sequences contain 3, 5, and 7 directional boundaries. Its quadratic excess-switch interference should make the high-switch conditions progressively approach chance, especially at lengths 8 and 10, while the corresponding low-switch conditions continue to favor B. Thus the critical signature is approximately zero low-versus-high contrast at each length under overwrite, versus an increasingly large low-switch B advantage under chunking. Exact A/B reversals balance response side. The 12 unique pairs are repeated eight times for exactly 96 trials.

[7] This fixed-length scan-order conflict test exploits the theories' different definitions of cue order. The experts are displayed in index order, but their validities imply the descending-validity order 3, 4, 5, 6, 2, 1. In every base trial, the displayed sequence consists of one A run followed by a terminal B run: AAAAAB, AAAABB, AAABBB, or AABBBB. Directional-Run Chunking therefore predicts choices favoring B. It additionally predicts a nonlinear suffix-length profile: the singleton B suffix should be weakest, adding a second B cue should produce a marked increase through run completion and suffix replacement, and terminal runs of two through four cues should be relatively similar because run confidence saturates rapidly. Length, discriminating-cue count, and switch count are fixed at six, six, and one, respectively. In contrast, the competing theory scans in descending validity, where Experts 2 and 1 are always the final two attended discriminating cues and both favor A on every base trial. Two near-certain terminal overwrites should thus yield a strong, approximately condition-invariant A preference; common six-cue fatigue only attenuates that preference. The decisive dissociation is therefore opposite modal choices, supplemented by a singleton-versus-repeated-suffix increase and a two-to-four-cue plateau under chunking versus a nearly flat profile under overwrite. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times produce 96 trials.

[8] This is a below-capacity trace-survival test using three discriminating reasons. The critical base sequences are ABA and BBA in descending-validity order; both share the same final B,A suffix and therefore end in favor of A, but their first reason has opposite direction. Under Recency-Gated Reason Overwrite, the shared B,A suffix almost completely eliminates the influence of the first reason. The exact core-probability difference between ABA and BBA is cue_attention*(1-cue_attention*overwrite_rate)^2. After the common three-reason fatigue and lapse transformations, the allowed parameter ranges bound the predicted A-choice contrast below approximately 0.008. Thus, the advocated theory predicts virtually identical A-choice rates for these two trials. The competitive buffer instead retains a graded trace of the first reason because three reasons do not exceed its minimum capacity. Before readout normalization, changing the first reason from B to A changes accessible evidence by 2*s1*exp(-2*displacement_rate/buffer_capacity), approximately 0.29–0.66 times the first cue's validity strength. It therefore predicts reliably more support for A in ABA than in BBA, potentially including a modal-choice crossover for some parameter values. BAB and AAB are exact option reversals, providing the corresponding B-choice contrast while balancing response side. Experts 1 and 5 are nondiscriminating but establish a broad advertised-validity range; the diagnostic reasons are Experts 2–4. Four unique pairs repeated 24 times produce exactly 96 trials.

[9] This is a fixed-length prefix-coherence test with a protected two-reason suffix. In descending-validity order, the four base sequences are AAAA, BAAA, ABAA, and BBAA; thus every base trial has four discriminating reasons and ends with two reasons favoring A. Recency-Gated Reason Overwrite predicts almost identical A-choice rates across these conditions. If q = 1 - cue_attention*overwrite_rate, two terminal A updates reduce any prefix-dependent difference by q^2; over the allowed ranges q^2 is at most about 0.0095, and common fatigue and lapse transformations reduce the observable contrast further. The Competitive Evidence Buffer instead preserves graded traces of both prefix reasons. It predicts P(A): AAAA > BAAA > ABAA > BBAA. Reversing the earliest reason reduces accessible A evidence by twice its surviving trace; reversing the second reason has a larger effect because that trace undergoes one fewer displacement step. Reversing both produces the largest reduction. Conflict normalization reinforces this gradient because the one- and two-B conditions contain increasingly substantial accessible evidence for both options, whereas AAAA is completely coherent. The critical quantitative signature is therefore a monotonic coherence gradient, including a larger AAAA-versus-ABAA contrast than AAAA-versus-BAAA, against the overwrite theory's essentially flat profile. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

[10] This is a load-window crossover test that holds the terminal reason fixed while varying the number of discriminating reasons. In descending-validity order, the four base sequences are AB, AAAAB, AAAAAB, and AAAAAAAAAB, with the same lowest-validity expert supplying the final B-favoring reason. Recency-Gated Reason Overwrite therefore predicts a B preference in every base condition: the final attended reason nearly replaces the preceding A commitment. Its distinctive quantitative prediction is a smooth contraction of that B preference toward 0.50 as discriminating-cue count rises from 2 to 5 to 6 to 10, because only updates beyond the second incur fatigue. Load-Contingent Dual-Code Arbitration instead predicts a nonmonotonic B–A–A–B profile. At two reasons, exact count balance prevents global recruitment, leaving the final-reason local code in control. At five and six reasons, the display lies in the manageable comparison window; four-of-five or five-of-six unweighted majorities favor A, nearly all validity-weighted evidence favors A, and this global conclusion opposes the terminal B commitment, activating the opposition increment. At ten reasons, load exceeds even the maximum effective capacity, sharply reducing global accessibility and attenuating the local code; the model should revert to a weaker, near-chance B preference. Thus the decisive conditions are m=5 and m=6, where overwrite predicts B but dual-code arbitration predicts A, supplemented by the competitor-specific load-dependent return toward B at m=10. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times produce exactly 96 trials.

[11] This is a fixed-load, fixed-direction-sequence coalition-strength test. Every base trial has exactly five discriminating experts, encountered in descending instructed-validity order as A,A,A,B,B. Thus the local commitment ends in favor of B, the unweighted count favors A by 3–2, and discriminating-cue count, update fatigue, terminal direction, and directional sequence are identical across conditions. Only the validities of the five discriminating experts change. The weak coalition uses nearly matched 72%–66% experts; the intermediate coalition gives A three substantially stronger experts than B; and the strong coalition gives A the 99%, 95%, and 90% experts while B receives the 56% and 51% experts. Recency-Gated Reason Overwrite does not use validity magnitudes after establishing scan order, so it predicts exactly the same B-choice probability in all three base conditions. Load-Contingent Dual-Code Arbitration instead predicts a pronounced validity-coalition gradient at the manageable load of five reasons. The weak weighted coalition should generally fail selectivity, leaving the terminal-B local code dominant; intermediate and especially strong A coalitions should increasingly recruit the global code and can reverse choice toward A because that code opposes the local commitment. The critical signature is therefore increasing choice of the 3–2 majority option from weak through intermediate to strong coalition strength, potentially crossing 0.50, against the overwrite theory's flat terminal-B profile. Exact A/B reversals balance response side. Six unique pairs repeated 16 times yield 96 trials.

[12] This is a dense two-block boundary-diversity test at the competing theory's six-reason chunking window. In descending-validity order, the three base sequences are ABBBBB, AAABBB, and AAAAAB; their reversals balance response side. Every trial has six discriminating cues, exactly two coherent opposing runs, and no gaps. Recency-Gated Reason Overwrite predicts the terminal option on every trial. Because discriminating-cue count is always six, all conditions receive the same four-update fatigue factor; terminal-run length may modulate preference strength, but it cannot reverse the predicted terminal choice. The competing ordinal-event theory instead learns that dense two-block boundaries vary widely (first-run fractions 1/6, 3/6, and 5/6). Repetition drives its boundary-diversity statistic toward its maximum, increasing recruitment of the coherent-chunk policy, which preserves the initial commitment. It therefore predicts an increasing initial-option preference as structural context accumulates, ultimately opposing overwrite on all six trial types. The critical quantitative tests are terminal-option choice versus initial-option choice and its interaction with trial history: overwrite predicts stable terminal control apart from random responses, whereas the competitor predicts a history-dependent shift toward primacy as boundary diversity is sampled. Six unique pairs are each repeated 16 times for exactly 96 trials.

[13] This is a physical-foregrounding test with scan-order evidence held exactly constant. In every base trial, the six discriminating experts are encountered in descending-validity order as A,A,A,A,A,B. The same six validity magnitudes are used in both conditions, the tally is 5–1, the terminal reason favors B, and n_features=18 and m=6 are fixed. The conditions differ only in where the discriminating experts occur in the displayed vector. In the clustered condition they occupy indices 0–5, with no internal gap and no discriminating cue at the physical right edge. In the foregrounded-terminal condition they occupy indices 6–10 and 17: the final B reason is isolated at the right edge by six nondiscriminating experts, yielding terminal_foreground=6/17≈0.353. Context-Calibrated Ordinal Event Policies predicts terminal-option choices in both conditions because density is only 1/3, effectively precluding coherent-chunk recruitment. Crucially, it predicts substantially stronger terminal-B choice in the foregrounded condition: the isolated edge cue directly adds about 3*(6/17)=1.06 to its confidence multiplier, increases terminal-context support, and counteracts the considerable 18-feature display overload. The clustered condition should show a weaker B preference. Recency-Gated Reason Overwrite is exactly invariant across these conditions because it skips nondiscriminating cues and receives the same ordered direction sequence, discriminating count, attention/overwrite process, fatigue factor, and lapse transformation. Thus its predicted B-choice probabilities are identical. Exact A/B reversals balance response side. Four unique pairs repeated 24 times produce 96 trials; the diagnostic contrast is greater choice of the final-reason option for the isolated-edge pairs than for the clustered pairs.

[14] This is a unanimous-evidence load test. Experts 1 and 12 provide the advertised validity spread but are nondiscriminating; the ten potentially discriminating experts all have 70% validity and therefore equal evidence strength. The four base conditions contain 2, 4, 7, or 10 discriminating cues, all favoring A; their exact reversals all favor B. Recency-Gated Reason Overwrite predicts a distinctive decline in choice reliability as unanimous reasons are added. Encoding is already virtually certain with two cues, while its fatigue factor changes from 1 at m=2 to (1-update_fatigue)^2, (1-update_fatigue)^5, and (1-update_fatigue)^8 at m=4, 7, and 10. It therefore predicts a monotonic contraction of winner choice toward 0.50 despite every added reason supporting the same option. Across the allowed parameter ranges, winner choice is approximately 0.80–0.93 at m=2 but only about 0.57–0.76 at m=10. The competing accumulator predicts the opposite quantitative profile. Every stimulus has one coherent run, no switches or reversals, no terminal singleton, and the same normalized evidence margin of 1, so its learned context statistics are invariant across conditions and histories. Because the diagnostic cues have equal validity, each added cue contributes equal signed evidence without conflict inhibition; accumulated and coalition evidence therefore increase or saturate rather than incurring a count penalty. It predicts stable-to-increasing winner choice from 2 to 10 cues, not the advocated theory's pronounced decline. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times produce exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This is a history-dependent context-calibration test. Experts 1 and 8 provide the advertised validity spread but are nondiscriminating; the six diagnostic experts have similar strengths and are scanned in order. Two inducer families create sharply different learned task sets. The coherent AAAAAA/B\u200bBBBBB inducer family contains no opposition or reversal and should lower the advocated accumulator's overwrite calibration. The ABBBBB/BAAAAA family has a concentrated two-run boundary followed by a long, highly reversal-susceptible opposing run and should raise overwrite calibration. The critical probes are AAAB and AAAAAB, plus their exact option reversals. Each probe has an early majority favoring one option but a terminal reason favoring the other. After histories dominated by coherent inducers, the advocated theory predicts greater integration and coalition access, hence more choices of the early-majority option. After histories rich in reversal-susceptible inducers, it predicts lower retention, stronger conflict inhibition, and substantially more choices of the terminal option; sufficiently strong calibration should produce a within-probe modal crossover. This effect should follow the advocated model's exponentially weighted pretrial history statistic, not merely the immediately preceding response. Recency-Gated Reason Overwrite is completely history-independent and therefore predicts no inducer-history effect for either probe. It predicts a stable terminal-option preference determined only by the current sequence, with the six-reason probe attenuated more toward chance than the four-reason probe because of its larger fatigue factor. The primary analysis should regress terminal-option choice on the advocated theory's pretrial context score, computed from stimulus history while excluding the current trial, with separate probe intercepts; the advocated theory predicts a positive context slope, whereas overwrite predicts exactly zero. Eight unique pairs are each repeated 12 times, producing 96 trials, and exact A/B reversals balance response side.",
  "validities": [
    0.99,
    0.75,
    0.73,
    0.71,
    0.69,
    0.67,
    0.65,
    0.51
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 10747,
  "output_tokens": 2717
}
```
