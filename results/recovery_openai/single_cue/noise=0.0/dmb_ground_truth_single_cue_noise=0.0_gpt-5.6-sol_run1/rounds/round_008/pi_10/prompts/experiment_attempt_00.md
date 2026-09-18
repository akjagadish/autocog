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
**Description:** Reliability-Gated Decision-Object Arbitration with Prefix-Coherence-Conditioned Terminal Access. People construct at most three bounded objects from each display: an initial high-validity commitment, a reliability-weighted summary of the visible coalition, and a privileged terminal reason or terminal directional run. Arbitration is stimulus-local and history-free. Terminal evidence has a genuine recency advantage, but weak terminal evidence receives reliable access only when it constitutes the sole coherent reversal against a unified prefix at manageable load. A terminal reason following an already fragmented prefix receives no such accessibility floor and may be mildly discounted. Very short conflicts provide too little separation for a strongly privileged counterargument, whereas seven-or-more-reason conflicts reduce access through object competition rather than general count fatigue. The initial commitment remains a weak but non-negligible fallback, coalition evidence saturates, and unanimous evidence experiences only a modest vigilance decline. Stable variation in coherent-terminal accessibility, terminal susceptibility, coalition reliance, attention, and lapses produces heterogeneous conflict choices without learned context.

**Parameters:**
- validities: validities
- validity_compression: [0.18, 0.42]
- early_commitment: [0.04, 0.22]
- coalition_reliance: [0.20, 0.85]
- terminal_susceptibility: [1.2, 8.3]
- terminal_reliability_threshold: [0.28, 0.62]
- terminal_reliability_sharpness: [0.06, 0.14]
- coherent_access_floor: [0.58, 0.78]
- terminal_run_bonus: [0.25, 0.95]
- terminal_singleton_bonus: [0.40, 1.00]
- conflict_moderation: [0.10, 0.65]
- attentional_reliability: [0.82, 0.99]
- choice_gain: [2.4, 4.8]
- vigilance_decline: [0.12, 0.25]
- lapse_rate: [0.03, 0.15]

**`predict source code`:**
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

[15] This is a history-dependent context-calibration test. Experts 1 and 8 provide the advertised validity spread but are nondiscriminating; the six diagnostic experts have similar strengths and are scanned in order. Two inducer families create sharply different learned task sets. The coherent AAAAAA/B​BBBBB inducer family contains no opposition or reversal and should lower the advocated accumulator's overwrite calibration. The ABBBBB/BAAAAA family has a concentrated two-run boundary followed by a long, highly reversal-susceptible opposing run and should raise overwrite calibration. The critical probes are AAAB and AAAAAB, plus their exact option reversals. Each probe has an early majority favoring one option but a terminal reason favoring the other. After histories dominated by coherent inducers, the advocated theory predicts greater integration and coalition access, hence more choices of the early-majority option. After histories rich in reversal-susceptible inducers, it predicts lower retention, stronger conflict inhibition, and substantially more choices of the terminal option; sufficiently strong calibration should produce a within-probe modal crossover. This effect should follow the advocated model's exponentially weighted pretrial history statistic, not merely the immediately preceding response. Recency-Gated Reason Overwrite is completely history-independent and therefore predicts no inducer-history effect for either probe. It predicts a stable terminal-option preference determined only by the current sequence, with the six-reason probe attenuated more toward chance than the four-reason probe because of its larger fatigue factor. The primary analysis should regress terminal-option choice on the advocated theory's pretrial context score, computed from stimulus history while excluding the current trial, with separate probe intercepts; the advocated theory predicts a positive context slope, whereas overwrite predicts exactly zero. Eight unique pairs are each repeated 12 times, producing 96 trials, and exact A/B reversals balance response side.

[16] This is a coherent-prefix terminal-access test at the competing theory's privileged five- and six-reason load window. In descending-validity order, the base sequences are AAAAB versus ABAAB at m=5 and AAAAAB versus ABAAAB at m=6. Within each load, the pair has the same weak terminal B reason and a shared multi-cue A,A,B suffix; only an earlier reason changes. Recency-Gated Reason Overwrite predicts nearly identical B-choice rates within each pair. The shared A,A,B suffix repeatedly overwrites the earlier difference, and both conditions at a given load receive exactly the same fatigue and lapse transformations. It predicts only a modest common contraction toward chance from five to six reasons. Decision-Object Arbitration instead predicts a large prefix-coherence contrast. AAAAB and AAAAAB are sole singleton reversals against unified prefixes at manageable load, so the 50%-valid terminal reason receives the coherent-access floor, singleton bonus, and two-run geometry advantage. ABAAB and ABAAAB have fragmented prefixes, so that floor is absent, weak-terminal access is discounted, and four-run fragmentation further suppresses the terminal object; their A-favoring coalitions should consequently exert much greater control. The critical quantitative signature is substantially more terminal-option choice for coherent than fragmented prefixes at both loads, potentially with fragmented trials favoring A, against overwrite's approximately zero within-load contrast. Exact A/B reversals balance response side. Eight unique pairs repeated 12 times yield 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This is an adjacent-load window test of coherent-singleton terminal access. In every base trial, cues are encountered in descending-validity order as a coherent run favoring A followed by the same extremely weak terminal cue favoring B. The six sequences are AAB, AAAB, AAAAB, AAAAAB, AAAAAAB, and AAAAAAAB, giving m=3 through m=8 while keeping the terminal expert, terminal direction, singleton-run geometry, and absence of prefix fragmentation fixed. Decision-Object Arbitration predicts a sharply nonmonotonic terminal-B profile. At m=3\u20134, the conflict is too short for the full coherent-reversal privilege, and the 50% terminal expert has little reliability-gated access. At m=5\u20136, the coherent-prefix access floor, singleton bonus, and manageable-load geometry jointly produce a large increase in terminal-B choices despite the growing A coalition. At m=7\u20138, the access floor is removed and object competition suppresses the terminal reversal, so choices should shift back toward the coherent A prefix. The diagnostic signature is therefore a localized peak in terminal-option choice at m=5\u20136, especially a positive jump from m=4 to m=5 followed by a drop from m=6 to m=7. Recency-Gated Reason Overwrite predicts B at every load because the same final attended cue overwrites the A state. Its core terminal control is nearly constant across these unanimous prefixes, while update fatigue produces only a smooth monotonic contraction of B choice toward 0.50 as m increases. It therefore cannot predict either the m=4-to-5 increase or the window-shaped profile. Exact A/B reversals balance response side. Twelve unique pairs repeated eight times yield 96 trials.",
  "validities": [
    0.99,
    0.95,
    0.9,
    0.84,
    0.78,
    0.71,
    0.63,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
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
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 10793,
  "output_tokens": 2258
}
```
