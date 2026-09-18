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


## COMPETING THEORY
**Description:** Parallel Contrastive Coalition Field with Novelty-Protected Competition. All discriminating cues are encoded concurrently into a capacity-limited comparison field rather than inspected in a mandatory validity or spatial sequence. Cue activation combines instructed reliability, U-shaped boundary distinctiveness, an estimable spatial tilt, and local contrast. Same-option activations pool into saturating coalitions before divisive competition. A coalition containing a sharply contrastive challenger receives a small, bounded release from cross-coalition inhibition after saturation, allowing prediction-error signals to survive competition without letting raw cue activation bypass capacity limits. Choice follows the normalized coalition difference with load-dependent precision loss and a task-invariant lapse. The model has no trial-history learning, strategy mixture, fixed scan order, or sequential commitment state.

**Parameters:**
- validities: validities
- reliability_gain: [0.25, 0.50]
- boundary_gain: [0.25, 0.55]
- spatial_tilt: [-0.90, -0.55]
- coalition_saturation: [0.55, 0.95]
- novelty_gain: [1.50, 2.40]
- novelty_release: [0.04, 0.10]
- protection_threshold: [1.10, 1.30]
- capacity_loss: [0.08, 0.18]
- competition_temperature: [1.05, 1.35]
- choice_precision: [3.00, 4.30]
- lapse_rate: [0.05, 0.12]

**`predict source code`:**
```python
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
```

**`policy source code`:**
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

[15] This is a history-dependent context-calibration test. Experts 1 and 8 provide the advertised validity spread but are nondiscriminating; the six diagnostic experts have similar strengths and are scanned in order. Two inducer families create sharply different learned task sets. The coherent AAAAAA/B​BBBBB inducer family contains no opposition or reversal and should lower the advocated accumulator's overwrite calibration. The ABBBBB/BAAAAA family has a concentrated two-run boundary followed by a long, highly reversal-susceptible opposing run and should raise overwrite calibration. The critical probes are AAAB and AAAAAB, plus their exact option reversals. Each probe has an early majority favoring one option but a terminal reason favoring the other. After histories dominated by coherent inducers, the advocated theory predicts greater integration and coalition access, hence more choices of the early-majority option. After histories rich in reversal-susceptible inducers, it predicts lower retention, stronger conflict inhibition, and substantially more choices of the terminal option; sufficiently strong calibration should produce a within-probe modal crossover. This effect should follow the advocated model's exponentially weighted pretrial history statistic, not merely the immediately preceding response. Recency-Gated Reason Overwrite is completely history-independent and therefore predicts no inducer-history effect for either probe. It predicts a stable terminal-option preference determined only by the current sequence, with the six-reason probe attenuated more toward chance than the four-reason probe because of its larger fatigue factor. The primary analysis should regress terminal-option choice on the advocated theory's pretrial context score, computed from stimulus history while excluding the current trial, with separate probe intercepts; the advocated theory predicts a positive context slope, whereas overwrite predicts exactly zero. Eight unique pairs are each repeated 12 times, producing 96 trials, and exact A/B reversals balance response side.

[16] This is a coherent-prefix terminal-access test at the competing theory's privileged five- and six-reason load window. In descending-validity order, the base sequences are AAAAB versus ABAAB at m=5 and AAAAAB versus ABAAAB at m=6. Within each load, the pair has the same weak terminal B reason and a shared multi-cue A,A,B suffix; only an earlier reason changes. Recency-Gated Reason Overwrite predicts nearly identical B-choice rates within each pair. The shared A,A,B suffix repeatedly overwrites the earlier difference, and both conditions at a given load receive exactly the same fatigue and lapse transformations. It predicts only a modest common contraction toward chance from five to six reasons. Decision-Object Arbitration instead predicts a large prefix-coherence contrast. AAAAB and AAAAAB are sole singleton reversals against unified prefixes at manageable load, so the 50%-valid terminal reason receives the coherent-access floor, singleton bonus, and two-run geometry advantage. ABAAB and ABAAAB have fragmented prefixes, so that floor is absent, weak-terminal access is discounted, and four-run fragmentation further suppresses the terminal object; their A-favoring coalitions should consequently exert much greater control. The critical quantitative signature is substantially more terminal-option choice for coherent than fragmented prefixes at both loads, potentially with fragmented trials favoring A, against overwrite's approximately zero within-load contrast. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

[17] This is an adjacent-load window test of coherent-singleton terminal access. In every base trial, cues are encountered in descending-validity order as a coherent run favoring A followed by the same extremely weak terminal cue favoring B. The six sequences are AAB, AAAB, AAAAB, AAAAAB, AAAAAAB, and AAAAAAAB, giving m=3 through m=8 while keeping the terminal expert, terminal direction, singleton-run geometry, and absence of prefix fragmentation fixed. Decision-Object Arbitration predicts a sharply nonmonotonic terminal-B profile. At m=3–4, the conflict is too short for the full coherent-reversal privilege, and the 50% terminal expert has little reliability-gated access. At m=5–6, the coherent-prefix access floor, singleton bonus, and manageable-load geometry jointly produce a large increase in terminal-B choices despite the growing A coalition. At m=7–8, the access floor is removed and object competition suppresses the terminal reversal, so choices should shift back toward the coherent A prefix. The diagnostic signature is therefore a localized peak in terminal-option choice at m=5–6, especially a positive jump from m=4 to m=5 followed by a drop from m=6 to m=7. Recency-Gated Reason Overwrite predicts B at every load because the same final attended cue overwrites the A state. Its core terminal control is nearly constant across these unanimous prefixes, while update fatigue produces only a smooth monotonic contraction of B choice toward 0.50 as m increases. It therefore cannot predict either the m=4-to-5 increase or the window-shaped profile. Exact A/B reversals balance response side. Twelve unique pairs repeated eight times yield 96 trials.

[18] This is an adjacent-reason displacement-gradient test. Experts 1 and 9 are always nondiscriminating and provide the advertised validity spread; Experts 2–8 have equal 70% validity, so they are scanned in display order and contribute equal reliability to the competing evidence field. Every base trial has seven discriminating reasons, a 4–3 tally favoring A, and a final reason favoring B. The eight base trials form four matched pairs that differ only by swapping an adjacent A,B pair to B,A. The swapped pair is followed by respectively 1, 2, 3, or 4 discriminating reasons. Discriminating count, tally, terminal direction, fatigue, and validity magnitudes are therefore fixed. Recency-Gated Reason Overwrite predicts B choices throughout and an extremely steep loss of sensitivity to the swap: after s subsequent reasons, its effect is approximately proportional to q^s, where q=1-cue_attention*overwrite_rate is at most 0.0975. Hence the four-reason-distant swap should have essentially no observable effect, and even the two- and three-reason-distant effects should be negligible after fatigue and lapse. The competing recurrent field retains separate traces, so the corresponding evidence contrast declines only approximately as memory_retention^s, with retention between 0.38 and 0.54. Its distant-swap contrast should consequently remain measurable, and the ratio of the four-reason-distant contrast to the one-reason-distant contrast should be roughly retention^3 (about 0.055–0.157), versus at most q^3 (below 0.001) under overwrite. Thus the diagnostic outcome is a graded, slowly decaying serial-displacement function rather than overwrite's near-step-function loss of prefix influence. Exact A/B reversals balance response side. Sixteen unique pairs repeated six times produce exactly 96 trials.

[19] This is a fixed-sequence terminal-reliability gradient. In every base trial, exactly four experts discriminate and their descending-validity direction sequence is A,A,A,B. Experts 1–3 always form the same early A coalition; only the identity—and hence advertised validity—of the terminal B expert changes across 86%, 70%, 58%, 52%, 50.5%, and 50%. Recency-Gated Reason Overwrite ignores validity magnitudes after determining scan order, so it predicts exactly the same B-choice probability in all six base conditions: the direction sequence, discriminating count, terminal overwrite, fatigue factor, and lapse transformation are identical. The Reliability-Compressed Recurrent Evidence Field instead has pre-normalization evidence proportional to r^3*s1 + r^2*s2 + r*s3 - s_terminal. It therefore predicts a graded increase in A choice as terminal validity declines. High-validity terminal cues should retain B control, whereas sufficiently weak terminal cues can be outweighed by the surviving three-cue A coalition, potentially producing a modal crossover. Near the crossover, its near-balance normalization additionally predicts especially shallow, near-chance responding rather than a condition-invariant terminal preference. The reliability gradient weakens toward zero only at the boundary case of zero validity compression, providing a direct quantitative estimate of the advocated compression mechanism. Exact A/B reversals balance response side. Twelve unique pairs repeated eight times yield exactly 96 trials.

[20] This is an interleaved-bank scan-order test. The 20 experts form four five-expert banks occupying comparable positions across the display. On each base trial, only one bank discriminates. Within every bank, the descending-validity direction sequence is exactly A,B,B,B,B: the 96% expert favors A, followed by the 82%, 70%, 60%, and 51% experts favoring B. Recency-Gated Reason Overwrite therefore receives the identical five-reason sequence on every base trial and predicts exactly the same B-choice probability, including identical attention, overwrite, three-update fatigue, and lapse transformations. In physical left-to-right order, however, the four sequences are ABBBB, BABBB, BBBAB, and BBBBA. Surprise-Bound Coalition Relay processes these as different coalition structures. It should strongly favor B for ABBBB because a coherent four-cue B coalition challenges a singleton A commitment. As the high-validity A cue moves rightward, coalition boundaries and switch contests change; most decisively, BBBBA ends with an accessible, high-validity terminal A singleton at the extreme right edge, receiving both terminal-singleton salience and maximal positional accessibility. The competitor should therefore predict a large increase in A choice, plausibly a modal crossover, from ABBBB to BBBBA, with the fragmented intermediate sequences showing distinct and generally less confident relay trajectories. The advocated theory predicts a perfectly flat profile across these same conditions. Exact A/B reversals balance response side. Eight unique pairs are repeated 12 times for 96 trials.

[21] This is a boundary-surprise test with scan-order, coalition size, reliability multiset, and aggregate physical accessibility controlled. On every base trial, exactly four experts discriminate. In descending-validity order their directions are always A,A,A,B, with respective validity magnitudes 98%, 86%, 53%, and 52%. Recency-Gated Reason Overwrite therefore predicts exactly the same terminal-B choice probability in all four base conditions: cue order, direction sequence, attention and overwrite process, discriminating count, fatigue, and lapse are identical. Surprise-Bound Coalition Relay instead inspects the displayed order, which is also A,A,A,B, but distinguishes the reliability discontinuity at the sole coalition boundary. In the two large-drop conditions, the A coalition exits at 98% and the terminal B singleton enters at 52%, producing boundary_surprise about 0.42*0.46/(0.50+0.46)=0.201. In the two small-drop conditions, the coalition exits at 53% before the same 52% singleton, producing only about 0.008. This approximately 0.193 contest difference corresponds to a 0.73–1.20 shift in the switching logit over the allowed switch-precision range, so the relay predicts appreciably more choices of the terminal option after the large reliability drop. Other reliability effects are controlled because every condition uses the same 98/86/53/52 multiset and assigns the first three values to one coalition. The four interleaved expert banks also balance display position: large-drop and small-drop conditions have the same mean terminal position and the same aggregate mean position for the three prefix cues. Exact A/B reversals balance response side. Thus the critical statistic is terminal-option choice averaged over the two large-drop banks minus that averaged over the two small-drop banks: the relay predicts a positive contrast, whereas overwrite predicts exactly zero. Eight unique pairs repeated 12 times yield 96 trials.

[22] This is a reliability-role crossover with the physical direction sequence held fixed. The 24 displayed experts contain four interleaved early banks and four interleaved late banks. On every base trial, one early bank favors A and one late bank favors B, so Spatially Indexed Reliability-Gated Revision always encounters the same left-to-right sequence A,A,A,B,B,B. Across the four conditions, however, reliability is transferred progressively from the early A bank to the late B bank: very-high versus low, moderately-high versus moderately-low, moderately-low versus moderately-high, and low versus very-high. The spatial theory should therefore show stable-to-increasing choice of terminal option B across these conditions. Its reliability-sensitive incumbent trace may protect A when the early bank is strongest, whereas increasingly reliable late challengers should revise that incumbent more effectively; positional means and the number, tally, run structure, and physical direction order of discriminating cues are approximately matched. Recency-Gated Reason Overwrite instead sorts by validity. In conditions 1 and 2 it processes A,A,A,B,B,B and predicts B, but in conditions 3 and 4 it processes B,B,B,A,A,A and predicts A. It thus predicts a sharp decrease in B choice between the middle conditions, opposite the spatial theory's increasing terminal-B profile. Within each side of that crossover, overwrite predicts nearly equal choice rates because validity magnitudes are ignored after sorting, and all trials have six discriminating cues and the same fatigue factor. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

[23] This rank-order-inverted dyad design creates both an opposite-choice dissociation and a quantitative gradient. Each base trial has exactly two discriminating experts: an earlier expert favors B and the fixed right-edge Expert 10 (82% validity) favors A. The earlier B expert varies jointly from very weak and far away (Expert 1: 51%) to nearly as reliable and relatively close (Expert 7: 80%). Spatially Indexed Reliability-Gated Revision processes the physical sequence B,A, so the reliable, right-edge A cue revises the leaked B commitment and should make A modal. It additionally predicts the ordered profile P(A; Expert 1) > P(A; Expert 3) > P(A; Expert 5) > P(A; Expert 7): the earlier B cue becomes more reliable and more spatially accessible across conditions, while its separation from the right-edge A cue—and hence A's isolation bonus—decreases. Recency-Gated Reason Overwrite instead sorts the two discriminating experts by validity. Expert 10 is always processed first and favors A; the lower-validity expert is always processed last and favors B. Because every condition has the identical validity-ordered direction sequence A,B and exactly two reasons, the overwrite theory predicts the same strong B preference in all four conditions, with no fatigue and no validity-gradient effect. Thus the theories predict opposite modal choices throughout, while only the spatial theory predicts a systematic initial-cue-strength/distance gradient. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This is a fixed-evidence spatial-displacement test tailored to the competitor's concurrent activation field. Each base trial contains exactly two discriminating experts: the fixed 99%-valid Expert 10 favors B, and one of four equally valid 52% experts favors A. Thus, in descending-validity order, every base trial presents the identical direction sequence B,A. Recency-Gated Reason Overwrite consequently predicts the same strong A preference in all four conditions: the low-validity A reason is processed last, overwrites B, and two reasons incur no update-fatigue penalty. The competitor instead processes the complete physical display and predicts a monotonic decline in A choice as the 52% A cue moves from Expert 1 to Expert 7. Its signed positions are -1, -5/9, -1/9, and 1/3, over which boundary_gain*x^2 + spatial_tilt*x decreases throughout the allowed parameter ranges. The 99% B cue remains fixed at the right endpoint. Reliability, cue count, coalition membership, display load, and local contrast are controlled. In particular, the physical order is always low-validity A followed by high-validity B, so neither cue receives sharp-drop surprise; both receive the same small symmetric-contrast signal. The competitor should therefore shift from relatively strong A support when A occupies the privileged left boundary toward substantially greater B support when A is displaced rightward, potentially producing a modal crossover. The decisive quantitative signature is P(A at Expert 1) > P(A at Expert 3) > P(A at Expert 5) > P(A at Expert 7), against overwrite's exactly flat A-choice profile. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.",
  "validities": [
    0.52,
    0.88,
    0.52,
    0.8,
    0.52,
    0.7,
    0.52,
    0.6,
    0.55,
    0.99
  ],
  "trial_a_ratings": [
    [
      1,
      0,
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
      0,
      1,
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
      0,
      0,
      0,
      1,
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
      1,
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
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
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
      0,
      1,
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
      0,
      0,
      0,
      1,
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
      1,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 12527,
  "output_tokens": 3236
}
```
