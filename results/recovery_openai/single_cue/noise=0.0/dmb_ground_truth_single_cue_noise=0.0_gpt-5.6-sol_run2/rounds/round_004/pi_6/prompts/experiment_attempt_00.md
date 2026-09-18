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
**Description:** Exchangeable Corroboration with Idiosyncratic Salience proposes that people retain the direction of every discriminating expert rating but strongly compress instructed differences in expert validity. Supporting cues are integrated into option-level coalitions whose evidence increases sublinearly, so additional agreement provides corroboration with diminishing marginal impact. Subjects differ in how they calibrate coalition members beyond the first two: a signed, stable calibration parameter makes an additional cue corroborative for some people and partially redundant for others. Separately, jointly diagnostic cues can be perceived as partially redundant: an order-free composition term discounts coalitions containing several simultaneously strong cues, with its effect centered between the competing coalitions. This composition discount is strongest for equal-sized coalitions but remains moderately active when coalition sizes differ. Cue order has no population-level directional role. Instead, each subject has a stable, broad cue-salience profile drawn from a zero-centered exchangeable distribution and centered again within the display. Consequently, individual subjects can show substantial configuration preferences, but there is no common early, late, onset, closure, or entry-versus-exit advantage. Choices result from the difference between the two saturated coalition signals, transformed by response sensitivity and an independent lapse process. Because no outcome feedback is supplied, neither validity beliefs nor salience profiles are learned across trials.

**Parameters:**
- validities: validities
- cue_salience_profile: [(-1.0, 1.0)] * n_features
- validity_compression: [0.16, 0.34]
- validity_reliance: [0.02, 0.07]
- salience_strength: [0.18, 0.42]
- coalition_saturation: [0.76, 0.86]
- corroboration_calibration: [-1.30, 0.70]
- corroboration_strength: [0.40, 0.50]
- composition_redundancy: [2.00, 2.80]
- beta: [0.82, 1.12]
- epsilon: [0.03, 0.09]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Exchangeable Corroboration with Idiosyncratic Salience expects "
            f"shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    raw_salience = np.asarray(parameters["cue_salience_profile"], dtype=np.float64)
    if raw_salience.ndim != 1 or raw_salience.size != n_features:
        raise ValueError(
            f"cue_salience_profile length {raw_salience.size} != n_features "
            f"{n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    salience_strength = float(parameters["salience_strength"])
    coalition_saturation = float(parameters["coalition_saturation"])
    corroboration_calibration = float(parameters["corroboration_calibration"])
    corroboration_strength = float(parameters["corroboration_strength"])
    composition_redundancy = float(parameters["composition_redundancy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Instructed reliability is represented on a diagnosticity scale, but
    # differences among experts are strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Salience is a stable subject-level profile, not a fixed position rule.
    # Centering removes a subject's global offset. Because profile components
    # have identical sampling distributions, no serial position has a
    # population-positive expected advantage.
    centered_salience = raw_salience - float(np.mean(raw_salience))
    attentional_weight = np.exp(
        np.clip(salience_strength * centered_salience, -10.0, 10.0)
    )
    attentional_weight /= max(float(np.mean(attentional_weight)), 1e-12)

    reliability_weight = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * diagnosticity
    )
    cue_weights = reliability_weight * attentional_weight
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)
    n_a = int(a_support.size)
    n_b = int(b_support.size)

    if n_a == 0 and n_b == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # Every discriminating cue contributes. Division by n**saturation
        # makes total evidence grow approximately as n**(1-saturation),
        # implementing diminishing marginal corroboration without stopping.
        total = float(np.sum(cue_weights[indices]))
        return total / (float(n) ** coalition_saturation)

    def composition_load(indices):
        # Mean pairwise joint diagnostic strength is a bounded, order-free
        # measure of potential redundancy. It is high when several members
        # are all strongly diagnostic, not when a coalition merely appears
        # early or late in the display.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 2:
            return 0.0
        bounded_strength = np.clip(2.0 * v[idx] - 1.0, 0.0, 1.0)
        pair_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_sum += float(bounded_strength[i] * bounded_strength[j])
                pair_count += 1
        return pair_sum / float(max(pair_count, 1))

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)

    # Calibration of excess coalition membership is invoked only when both
    # options already have multi-cue interpretations. Thus it captures the
    # psychological status of a third or later coalition member rather than
    # introducing a general position gate. Positive values treat extra cues
    # as corroboration; negative values treat them as redundancy.
    if n_a >= 2 and n_b >= 2:
        extra_a = float(max(n_a - 2, 0))
        extra_b = float(max(n_b - 2, 0))
        log_adjustment_a = (
            corroboration_strength * corroboration_calibration * extra_a
        )
        log_adjustment_b = (
            corroboration_strength * corroboration_calibration * extra_b
        )
        evidence_a *= np.exp(np.clip(log_adjustment_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_adjustment_b, -10.0, 10.0))

        # Decouple composition from the heterogeneous size calibration.
        # Coalitions containing several jointly strong cues receive a mild
        # redundancy discount. Centering the adjustment between options
        # prevents a global response bias. Composition is most diagnostic
        # when coalition sizes are equal and remains moderately active in
        # size conflicts without becoming a spatial or directional gate.
        load_difference = composition_load(a_support) - composition_load(b_support)
        composition_scale = 1.0 if n_a == n_b else 0.60
        composition_shift = (
            composition_redundancy * composition_scale * load_difference
        )
        evidence_a *= np.exp(np.clip(-0.5 * composition_shift, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(0.5 * composition_shift, -10.0, 10.0))

    net_a = evidence_a - evidence_b
    logits = np.array(
        [0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64
    )
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.5, 0.5], dtype=np.float64)
    return probs / total
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Reliability-Compressed Parsing Mixtures proposes that communicated cue validities are strongly compressed and all discriminating cues enter a diminishing-returns accumulator. Stable subject-level styles allocate attention among reliability, coalition size, and locally parsed cue groups. Display position has no universal primacy or recency gradient: configuration effects arise from chunk binding and weak interpretation completion. In nearly balanced three-versus-two conflicts, parsing is sensitive to the internal ordering of compact and fragmented portions of a coalition; this translation-invariant run-asymmetry can reverse associations with recency-weighted predictions without assigning greater weight to later cues themselves. Balanced two-versus-two conflicts retain sign-varying parsing effects that cancel at the population level. Completion in conflicts between two coalitions of at least three cues is population-common but weak, graded, and heterogeneous rather than deterministic. Signed size calibration creates stable divisions between subjects who treat an additional cue as corroboration and those who treat it as redundant.

**Parameters:**
- validities: validities
- integration_style: {0, 1, 2, 3, 4, 5, 6, 7, 8}
- validity_compression: [0.20, 0.42]
- coalition_saturation: [0.74, 0.86]
- size_calibration: [1.25, 1.65]
- parsing_strength: [0.12, 0.30]
- imbalanced_group_gain: [4.50, 6.50]
- completion_strength: [0.38, 0.50]
- beta: [0.95, 1.30]
- epsilon: [0.03, 0.09]

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Compressed Parsing Mixtures expects shape (2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])
    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    style = int(parameters["integration_style"])
    validity_compression = float(parameters["validity_compression"])
    coalition_saturation = float(parameters["coalition_saturation"])
    size_calibration = float(parameters["size_calibration"])
    parsing_strength = float(parameters["parsing_strength"])
    imbalanced_group_gain = float(parameters["imbalanced_group_gain"])
    completion_strength = float(parameters["completion_strength"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Reliability differences are represented but strongly compressed.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    diagnosticity = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    diagnosticity /= max(float(np.mean(diagnosticity)), 1e-12)

    # Stable styles differ in reliability attention, but even nominally
    # reliability-led subjects strongly compress instructed differences.
    if style <= 3:
        reliability_attention = 0.02 + 0.01 * style
    elif style <= 5:
        reliability_attention = 0.08 + 0.02 * (style - 4)
    else:
        reliability_attention = 0.04 + 0.015 * (style - 6)

    cue_weights = (
        (1.0 - reliability_attention) * np.ones(n_features, dtype=np.float64)
        + reliability_attention * diagnosticity
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    pos_idx = np.flatnonzero(differences > 0.0)
    neg_idx = np.flatnonzero(differences < 0.0)
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)

    if n_pos == 0 and n_neg == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def accumulated_evidence(indices):
        n = int(indices.size)
        if n == 0:
            return 0.0
        # All cues contribute, while total evidence grows sublinearly.
        return float(np.sum(cue_weights[indices])) / (float(n) ** coalition_saturation)

    evidence_a = accumulated_evidence(pos_idx)
    evidence_b = accumulated_evidence(neg_idx)

    # There is deliberately no fixed position gradient. A subject instead
    # parses the display into short local chunks whose width and phase are
    # stable properties of that subject's style.
    chunk_width = 2 + (style % 3)
    chunk_phase = style % chunk_width

    def parsing_quality(indices):
        n = int(indices.size)
        if n < 2:
            return 0.0
        idx = np.asarray(indices, dtype=int)
        adjacent = float(np.sum(np.diff(idx) == 1)) / float(n - 1)
        chunk_ids = np.floor_divide(idx + chunk_phase, chunk_width)
        same_chunk_pairs = 0.0
        total_pairs = float(n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                if chunk_ids[i] == chunk_ids[j]:
                    same_chunk_pairs += 1.0
        local_binding = same_chunk_pairs / max(total_pairs, 1.0)
        return 0.2 * adjacent + 0.8 * local_binding

    def centered_run_coherence(indices):
        # This descriptor depends only on within-coalition gap structure, not
        # absolute position. Positive values mean compact binding occurs near
        # the coalition's parsed entry; negative values mean it occurs near
        # its parsed exit. It is zero for singleton and two-cue coalitions.
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n < 3:
            return 0.0
        gaps = np.diff(idx).astype(np.float64)
        binding = np.exp(-(gaps - 1.0))
        centered_order = np.linspace(1.0, -1.0, binding.size)
        norm = max(float(np.sum(np.abs(centered_order))), 1.0)
        return float(np.dot(centered_order, binding) / norm)

    # Multi-cue conflicts receive a weak, signed structural calibration.
    # Styles 0--3 regard larger coalitions as corroborative; styles 4--8
    # regard excess members as increasingly redundant. This creates a stable
    # population split in 3-versus-2 conflicts without changing singleton
    # comparisons or discarding any cue from the base accumulator.
    if n_pos >= 2 and n_neg >= 2:
        size_orientation = 1.0 if style <= 3 else -1.0
        log_size_ratio = np.log(float(n_pos) / float(n_neg))
        size_shift = size_orientation * size_calibration * log_size_ratio

        parsing_valences = np.array(
            [-1.0, 0.75, -0.50, 1.0, -0.75, 0.50, -1.0, 0.75, 0.25],
            dtype=np.float64,
        )
        parse_difference = parsing_quality(pos_idx) - parsing_quality(neg_idx)

        # Nearly balanced 3-versus-2 configurations use centered run
        # coherence. This distinguishes where compact binding occurs within a
        # coalition without imposing an absolute early/late accessibility
        # gradient. Balanced conflicts retain the centered style valences.
        if min(n_pos, n_neg) == 2 and abs(n_pos - n_neg) == 1:
            group_multipliers = np.array(
                [0.80, 1.10, 0.90, 1.20, 0.75, 1.05, 0.85, 1.15, 1.00],
                dtype=np.float64,
            )
            run_difference = (
                centered_run_coherence(pos_idx)
                - centered_run_coherence(neg_idx)
            )
            parsing_shift = (
                parsing_strength
                * imbalanced_group_gain
                * float(group_multipliers[style])
                * run_difference
            )
        else:
            parsing_shift = (
                parsing_strength
                * float(parsing_valences[style])
                * parse_difference
            )

        log_gate_a = 0.5 * (size_shift + parsing_shift)
        log_gate_b = -0.5 * (size_shift + parsing_shift)
        evidence_a *= np.exp(np.clip(log_gate_a, -10.0, 10.0))
        evidence_b *= np.exp(np.clip(log_gate_b, -10.0, 10.0))

    net_a = evidence_a - evidence_b

    # Completion is weak and graded, is absent from singleton and 2-versus-2
    # conflicts, and never closes the accumulator. Stable differences in its
    # strength are smaller than the former opposing-sign split.
    if n_pos == n_neg and n_pos >= 3:
        completion_multipliers = np.array(
            [0.85, 0.925, 1.00, 1.075, 1.15, 0.95, 1.05, 0.90, 1.10],
            dtype=np.float64,
        )
        completion = completion_strength * float(completion_multipliers[style])
        if int(pos_idx[-1]) > int(neg_idx[-1]):
            net_a += completion
        elif int(neg_idx[-1]) > int(pos_idx[-1]):
            net_a -= completion

    logits = np.array([0.5 * beta * net_a, -0.5 * beta * net_a], dtype=np.float64)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()

    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        probs = np.ones(len(probs), dtype=np.float64) / float(len(probs))
    else:
        probs /= total
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This 7-cue design creates a parameter-robust quantitative dissociation. On every trial, the highest-validity cue discriminates and favors one option, so Take The Best stops immediately and predicts the same choice probability for that option at every evidence level: (1-epsilon)*sigmoid(beta)+epsilon/2. In contrast, the lower-validity cues increasingly favor the other option. The tally margin opposing the top-cue winner ranges from 0 through 5. Tallying therefore predicts exactly 0.5 choice probability at margin 0 and an increasingly strong preference for the opposite option at margins 1-5, with probability (1-epsilon)*sigmoid(-beta*d)+epsilon/2 for the top-cue winner. Thus the critical signature is TTB's flat, above-chance endorsement of the top-cue winner versus Tallying's graded reversal as the opposing cue count grows. Mirrored A/B versions control for response-side bias. There are 12 unique pairs, yielding 8 repetitions each and exactly 96 trials.

[1] This design tests a validity-order invariance unique to Tallying. Every pair contains exactly three A-winning cues, three B-winning cues, and two tied cues. Tallying therefore predicts P(A)=0.5 on every trial, exactly and independently of beta and epsilon. In contrast, Take The Best finds a discriminating cue at validity rank 1, 2, or 3 and predicts the option favored by that cue with probability q=(1-epsilon)*sigmoid(beta)+epsilon/2. Thus TTB predicts a stable, above-chance directional preference even though the total cue tally is perfectly balanced, whereas Tallying predicts uniform guessing. The first discriminating cue favors A in half the pairs and B in the corresponding option-swapped mirrors, controlling response-side bias. There are two distinct evidence configurations at each stopping rank and their mirrors, yielding four trials per rank and 12 unique pairs total; each is repeated eight times for 96 trials. The critical advocated-theory signature is chance-level choice across all configurations despite large differences in which high-validity cue favors which option; TTB instead predicts complementary probabilities q and 1-q for the mirrored pairs.

[2] This 7-cue design holds the tally margin constant while systematically relocating the same 3-versus-2 cue coalition from early, low-validity positions to late, high-validity positions. In each of the first six pairs, A wins exactly three cues, B wins exactly two, and two cues tie. Tallying therefore predicts exactly the same A-choice probability on all six: (1-epsilon)*sigmoid(beta)+epsilon/2. The final six pairs are A/B-swapped mirrors, for which Tallying predicts the complementary constant probability. Accessibility-Balanced Evidence Integration instead assigns progressively greater weight to later cues; the ascending validities reinforce this positional gradient throughout the advocated parameter ranges. Consequently, its predicted A-choice probability should increase monotonically across the first six configurations. At the low end, A's three early wins should be overcome by B's two late wins, reversing the tally preference; at the high end, A's late coalition should produce a strong A preference. Coalition saturation further penalizes A's three-cue majority relative to B's two-cue coalition but is constant in form across the progression. The mirrored pairs should exhibit the complementary graded pattern, controlling response-side bias. Thus the decisive signature is a large configuration-dependent slope, including a preference reversal despite an invariant tally margin; Tallying permits neither. There are 12 unique pairs, each repeated eight times, yielding exactly 96 trials.

[3] This design isolates a positional-accessibility gradient while holding both cue tally and diagnosticity exactly constant. In each of the first six pairs, A wins one cue and B wins one cue, so Tallying predicts P(A)=0.5 exactly, independently of beta and epsilon. B's win is always at Expert 2 (index 1), whereas A's win moves successively from Expert 3 through Expert 8 (indices 2-7). All of these discriminating experts have identical advertised validity (0.70), so their diagnosticity components are equal. Accessibility-Balanced Evidence Integration therefore predicts a strictly increasing A-choice probability as A's supporting cue moves later: its evidence difference is proportional to accessibility(target)-accessibility(index 1). Coalition saturation is inactive because the coalitions are always one cue versus one cue. Across the competing theory's parameter ranges, the nearest-position contrast should be only slightly above chance (approximately 0.51-0.52), while the most separated contrast should produce approximately 0.62-0.79 A choices. The final six pairs swap A and B, yielding complementary predictions and controlling response-side bias. The high- and low-validity experts provide the required communicated validity spread but are tied in these trials, ensuring that the graded competing-theory prediction is specifically attributable to positional accessibility rather than validity. There are 12 unique pairs, each repeated eight times for 96 trials.

[4] This 10-cue design pits universal positional accessibility against switch-closure in balanced two-versus-two conflicts. Experts 1 and 4-10 all have validity 0.70; only these equal-validity experts discriminate. Experts 2 and 3 provide the required validity spread (0.95 and 0.55) but are tied, so validity differences cannot explain choices. In each of the first six pairs, A is supported by two cues including the final discriminating cue, whereas B is supported by two cues whose combined positions are later on average. Switch-Closure Coalition Integration enters its balanced-coalition branch: because A supplies the final discriminating cue and all four discriminating cues have identical validity, it predicts the same above-chance A-choice probability for all six configurations (approximately 0.58-0.66 across its parameter ranges), irrespective of spacing. Accessibility-Balanced Evidence Integration instead sums the cues' exponentially increasing positional weights. In every pair, B's two adjacent late cues have greater total accessibility than A's early-plus-terminal pair, so it robustly predicts P(A)<0.5. It additionally predicts graded changes as A's early cue moves later and the accessibility deficit shrinks, whereas the competing theory remains flat because coalition size and closure direction are unchanged. The final six pairs are exact A/B mirrors, producing complementary predictions and controlling response-side bias. Thus the decisive signature is a sign reversal—advocated-theory preference for B versus competing-theory preference for A—together with a graded spacing effect predicted only by the advocated theory. There are 12 unique pairs, each repeated eight times, for 96 trials.

[5] This 10-cue design targets the advocated theory's amplified coalition-primacy gate in unequal but nearly balanced three-versus-two conflicts. In each of the first four pairs, A receives a fixed, adjacent three-cue coalition at Experts 1-3, while B receives an adjacent two-cue coalition that moves progressively later, from Experts 4-5 through Experts 7-8. All discriminating experts have identical validity (0.70); the high- and low-validity Experts 9-10 are always tied and provide the required validity spread without affecting directional evidence. Switch-Closure Coalition Integration predicts A above chance throughout: its sublinear accumulation gives the three-cue coalition a modest numerical advantage, and its near-balance gate strongly favors A's earlier-onset coherent coalition. As B's coalition moves later, its onset-primacy score declines, so the advocated theory predicts a monotonic increase in P(A). Accessibility-Balanced Evidence Integration makes the opposite predictions. Its strong universal recency gradient gives B's two later cues more total weight than A's saturation-penalized three early cues, producing P(A)<0.5 even in the earliest configuration. Moving B later further increases its accessibility, so it predicts a monotonic decrease in P(A). Thus the theories differ both categorically in preferred option and quantitatively in the sign of the positional slope. The final four pairs are exact A/B mirrors, controlling response-side bias and yielding complementary predictions. Eight unique pairs are repeated 12 times each for exactly 96 trials.

[6] This 10-cue design isolates the advocated theory's translation-invariant run-asymmetry mechanism in nearly balanced three-versus-two conflicts. In each of the first six trials, A is supported by three equal-validity cues and B by the fixed adjacent pair at Experts 7-8. A's coalition always begins at Expert 1 and always contains exactly one adjacent link, but its internal ordering alternates between an entry-compact form [1,1,g] and an exit-compact form [g,1], for gaps g=2, 3, and 4. Thus each matched pair has identical coalition sizes, onset positions, total validity, and conventional adjacency coherence. Switch-Closure Coalition Integration consequently predicts exactly the same A-choice probability across all six configurations: its unequal-coalition gate only represents onset primacy and the proportion of adjacent links, both of which are held constant. Reliability-Compressed Parsing Mixtures instead assigns positive centered run coherence to entry-compact coalitions and negative coherence to exit-compact coalitions. It therefore predicts higher P(A) for every entry-compact configuration than for its matched exit-compact configuration. Moreover, the entry-minus-exit contrast should increase from gap 2 to gap 4 because the compact-versus-fragmented distinction becomes more pronounced. This contrast has the same sign for every integration style, although the stable size-calibration split can shift subjects' overall preference for the three-cue coalition. The final six trials are exact A/B mirrors, so the corresponding prediction is higher endorsement of the three-cue coalition when it is entry-compact, while controlling response-side bias. Experts 9-10 provide the required validity spread but are tied; every discriminating expert has validity 0.70. Twelve unique pairs repeated eight times yield exactly 96 trials.

[7] This 16-cue design tests absolute coalition-onset sensitivity against the competing theory's translation-invariant parsing rule in nearly balanced three-versus-two conflicts. In the first six pairs, A always has a three-cue coalition with the identical internal gap pattern [1,2] (one adjacent link), while B always has an adjacent two-cue coalition. Only their locations change: A's coalition begins progressively earlier than B's in the first three configurations and later than B's in the final three. All discriminating experts have validity 0.70; Experts 15 and 16 provide the validity spread but are tied. Switch-Closure Coalition Integration holds coalition size, accumulation, coherence, and validity evidence constant while its amplified near-balance gate increasingly disfavors A as A's coalition onset moves from before to after B's. It therefore predicts a marked monotonic decline in endorsement of the three-cue coalition, with a robust early-versus-late preference reversal. Reliability-Compressed Parsing Mixtures predicts exactly invariant endorsement of the three-cue coalition across these translations for every integration style: the size ratio is fixed, the three-cue coalition's centered run coherence is unchanged, the two-cue coalition's centered run coherence is always zero, and absolute chunk position is not used in its three-versus-two branch. The last six pairs are exact A/B mirrors, permitting the critical analysis to code choice of the three-cue coalition rather than response side. Twelve unique pairs repeated eight times yield 96 trials.

[8] This 8-cue design isolates the advocated theory's population-common completion term in balanced three-versus-three conflicts. Experts 1-6 all have validity 0.70 and discriminate on every trial; Experts 7-8 have validities 0.95 and 0.55 but give tied ratings, providing the required validity spread without creating reliability or composition differences between coalitions. In each of the first eight pairs, A is supported by exactly three cues, including Expert 6, the final discriminating cue, while B is supported by the complementary three cues. The eight configurations vary which two earlier cues join Expert 6, sampling compact, fragmented, and interleaved coalitions. Reliability-Compressed Parsing Mixtures adds a completion signal favoring A in every configuration because A supplies the final discriminating cue. Its completion increment ranges from 0.323 to 0.575 before response sensitivity, implying a common above-chance closure-coalition tendency, with smaller style-dependent parsing deviations around it. Exchangeable Corroboration with Idiosyncratic Salience has no completion mechanism: coalition sizes, validities, and order-free composition loads are exactly equal, so its population prediction for the closure coalition is 0.5. It permits stable individual configuration preferences from random salience profiles, but these preferences are zero-centered and therefore should produce approximately equal numbers of positive and negative subject-level closure effects rather than the advocated theory's common positive displacement. The final eight pairs are exact A/B mirrors, controlling response-side bias while preserving which spatial coalition contains the final discriminating cue. The decisive quantitative statistics are the closure-coded grand mean and the distribution of subject-level closure effects: reliably above 0.5 and predominantly positive under the advocated theory, versus population-centered at 0.5 with sign-symmetric heterogeneity under the competing theory. Sixteen unique pairs repeated six times yield exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 16-cue design isolates the advocated theory's order-free composition-redundancy mechanism in balanced two-versus-two conflicts. In the first six pairs, A is supported by two experts having the same advertised validity, which increases across conditions from 0.68 to 0.97; B is always supported by the low-validity Experts 7 and 15 (0.55 and 0.60). Every supporting pair is separated by eight positions. Consequently, both coalitions have zero adjacency and cannot place both members in the same width-2, width-3, or width-4 parsing chunk under any competing-theory integration style. Coalition size is also exactly balanced, and the two-versus-two format prevents completion. Reliability-Compressed Parsing Mixtures is therefore reduced to additive compressed reliability evidence and predicts a modest A preference that increases monotonically as A's two validities rise. Exchangeable Corroboration with Idiosyncratic Salience instead computes the pairwise composition load. A's load rises from (2*0.68-1)^2 = 0.1296 to (2*0.97-1)^2 = 0.8836, whereas B's load remains (2*0.55-1)(2*0.60-1) = 0.02. Its strong composition discount overwhelms its very weak reliability weighting, predicting preference for the low-validity B coalition and a steep monotonic decrease in P(A) as A's validity increases. Exchangeable salience can create stable individual deviations but supplies no population-level slope tied to these translated configurations. The final six pairs are exact A/B mirrors. The primary statistic should code selection of the equal-validity coalition: the advocated theory predicts a pronounced negative validity slope, while the competing theory predicts a shallow positive slope. Twelve unique pairs repeated eight times yield exactly 96 trials.",
  "validities": [
    0.68,
    0.74,
    0.8,
    0.86,
    0.92,
    0.97,
    0.55,
    0.7,
    0.68,
    0.74,
    0.8,
    0.86,
    0.92,
    0.97,
    0.6,
    0.7
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      0,
      0,
      0,
      0,
      0,
      1,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
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
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
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
  "input_tokens": 8688,
  "output_tokens": 3649
}
```
