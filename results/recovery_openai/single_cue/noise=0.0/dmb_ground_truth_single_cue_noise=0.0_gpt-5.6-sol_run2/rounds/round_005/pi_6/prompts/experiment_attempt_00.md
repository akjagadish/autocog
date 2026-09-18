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
**Description:** Reliability-Schema Source Inference proposes that decision makers treat expert agreement as evidence generated by latent informational sources rather than as conditionally independent votes. Advertised validities identify experts' reliability schemas. Dependence inference operates at two scales: an extremely narrow component assigns nearly identical, jointly extreme experts a high probability of sharing a source, while a narrower but stronger shoulder captures partial dependence among merely similar reliability schemas. For two-member coalitions, an additional equality-specific jointly extreme source prior produces nonlinear skepticism toward exactly matched reliable experts without increasing dependence for larger coalitions. Thus equally extreme coalitions can be discounted sharply without treating moderately heterogeneous coalitions as fully independent. Every discriminating cue contributes, but its contribution is divided by inferred latent-source overlap. Coalition evidence has diminishing returns, and nominal count corroboration is capped after three members. The transition from two to three members receives an additional corroboration gain only to the extent that the coalition represents informationally diverse sources. Stable subject-level priors over source dependence and count-versus-independence tradeoffs produce heterogeneous three-versus-two judgments without rewarding arbitrarily large coalitions. Communicated reliability differences are strongly compressed, preventing a single advertised-best expert from dominating a diverse majority. The computation is exchangeable over expert identities and contains no onset, closure, recency, adjacency, entry, exit, or other spatial parsing mechanism.

**Parameters:**
- validities: validities
- validity_compression: [0.14, 0.30]
- validity_reliance: [0.01, 0.06]
- coalition_saturation: [0.62, 0.72]
- dependence_prior: [2.10, 3.90]
- equality_pair_strength: [0.60, 1.80]
- similarity_bandwidth: [0.010, 0.022]
- shoulder_strength: [0.62, 0.92]
- shoulder_bandwidth: [0.032, 0.065]
- extremity_threshold: [0.72, 0.78]
- extremity_slope: [20.0, 30.0]
- overlap_curvature: [2.10, 3.10]
- count_independence_tradeoff: [-0.02, 0.16]
- diverse_third_gain: [0.20, 0.40]
- beta: [1.20, 1.48]
- epsilon: [0.02, 0.06]

**`predict source code`:**
```python
def predict(parameters, state, history) -> np.ndarray:
    stim = np.asarray(state, dtype=np.float64)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reliability-Schema Source Inference expects shape "
            f"(2, n_features); got {stim.shape}."
        )

    a, b = stim[0], stim[1]
    n_features = int(stim.shape[1])

    validities = np.asarray(parameters["validities"], dtype=np.float64)
    if validities.ndim != 1 or validities.size != n_features:
        raise ValueError(
            f"validities length {validities.size} != n_features {n_features}."
        )

    validity_compression = float(parameters["validity_compression"])
    validity_reliance = float(parameters["validity_reliance"])
    coalition_saturation = float(parameters["coalition_saturation"])
    dependence_prior = float(parameters["dependence_prior"])
    equality_pair_strength = float(parameters["equality_pair_strength"])
    similarity_bandwidth = float(parameters["similarity_bandwidth"])
    shoulder_strength = float(parameters["shoulder_strength"])
    shoulder_bandwidth = float(parameters["shoulder_bandwidth"])
    extremity_threshold = float(parameters["extremity_threshold"])
    extremity_slope = float(parameters["extremity_slope"])
    overlap_curvature = float(parameters["overlap_curvature"])
    count_independence_tradeoff = float(
        parameters["count_independence_tradeoff"]
    )
    diverse_third_gain = float(parameters["diverse_third_gain"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Advertised reliability is represented, but ordinal differences are
    # strongly compressed before use. The mean normalization keeps overall
    # evidence scale comparable across experiments with different validity
    # schedules.
    v = np.clip(validities, 0.500001, 0.999999)
    diagnosticity = np.log(v / (1.0 - v))
    compressed = np.power(np.maximum(diagnosticity, 1e-12), validity_compression)
    compressed /= max(float(np.mean(compressed)), 1e-12)
    cue_weights = (
        (1.0 - validity_reliance) * np.ones(n_features, dtype=np.float64)
        + validity_reliance * compressed
    )
    cue_weights /= max(float(np.mean(cue_weights)), 1e-12)

    differences = a - b
    a_support = np.flatnonzero(differences > 0.0)
    b_support = np.flatnonzero(differences < 0.0)

    if a_support.size == 0 and b_support.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)

    def coalition_signal(indices):
        idx = np.asarray(indices, dtype=int)
        n = int(idx.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(cue_weights[idx[0]])

        coalition_v = v[idx]
        overlap = np.zeros(n, dtype=np.float64)
        pair_posteriors = []

        # The narrow component detects almost identical reliability schemas.
        # A moderate-width shoulder represents partial common-source
        # inference for nonidentical but similar experts. An extra narrow
        # prior applies only to jointly extreme two-member coalitions, keeping
        # exact-pair skepticism separate from dependence among larger groups.
        for i in range(n):
            for j in range(i + 1, n):
                reliability_gap = float(coalition_v[i] - coalition_v[j])
                narrow_similarity = np.exp(
                    -0.5 * (reliability_gap / similarity_bandwidth) ** 2
                )
                shoulder_similarity = np.exp(
                    -0.5 * (reliability_gap / shoulder_bandwidth) ** 2
                )
                pair_extremity = 0.5 * float(coalition_v[i] + coalition_v[j])
                extreme_gate = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            extremity_slope
                            * (pair_extremity - extremity_threshold),
                            -50.0,
                            50.0,
                        )
                    )
                )

                narrow_affinity = (
                    dependence_prior
                    * narrow_similarity
                    * (0.01 + 0.99 * extreme_gate)
                )
                if n == 2:
                    narrow_affinity += (
                        equality_pair_strength
                        * narrow_similarity
                        * extreme_gate
                    )
                shoulder_affinity = (
                    shoulder_strength
                    * shoulder_similarity
                    * (0.10 + 0.90 * extreme_gate)
                )
                source_affinity = narrow_affinity + shoulder_affinity
                overlap[i] += source_affinity
                overlap[j] += source_affinity
                pair_posteriors.append(
                    source_affinity / (1.0 + source_affinity)
                )

        # Shared-source overlap is converted superlinearly into redundancy.
        # This permits sharp skepticism toward exactly equal extreme pairs
        # while preserving useful evidence from heterogeneous additions.
        independent_contributions = cue_weights[idx] / np.power(
            1.0 + overlap, overlap_curvature
        )
        independent_total = float(np.sum(independent_contributions))

        signal = independent_total / (float(n) ** coalition_saturation)

        # Nominal count corroboration is capped after the third member rather
        # than growing exponentially throughout a large coalition.
        capped_count_steps = float(min(n - 1, 2))
        signal *= np.exp(
            np.clip(
                count_independence_tradeoff * capped_count_steps,
                -10.0,
                10.0,
            )
        )

        # Crossing from two to at least three cues provides corroboration only
        # when those cues are inferred to represent informationally diverse
        # sources. The bonus is then capped, so fourth and later cues do not
        # receive repeated generic count rewards.
        if n >= 3:
            mean_shared_source = float(np.mean(pair_posteriors))
            effective_diversity = np.clip(
                1.0 - mean_shared_source, 0.0, 1.0
            )
            signal *= np.exp(
                np.clip(diverse_third_gain * effective_diversity, -10.0, 10.0)
            )

        return float(signal)

    evidence_a = coalition_signal(a_support)
    evidence_b = coalition_signal(b_support)
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
def policy(probs) -> int:
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

[9] This 16-cue design isolates the advocated theory's order-free composition-redundancy mechanism in balanced two-versus-two conflicts. In the first six pairs, A is supported by two experts having the same advertised validity, which increases across conditions from 0.68 to 0.97; B is always supported by the low-validity Experts 7 and 15 (0.55 and 0.60). Every supporting pair is separated by eight positions. Consequently, both coalitions have zero adjacency and cannot place both members in the same width-2, width-3, or width-4 parsing chunk under any competing-theory integration style. Coalition size is also exactly balanced, and the two-versus-two format prevents completion. Reliability-Compressed Parsing Mixtures is therefore reduced to additive compressed reliability evidence and predicts a modest A preference that increases monotonically as A's two validities rise. Exchangeable Corroboration with Idiosyncratic Salience instead computes the pairwise composition load. A's load rises from (2*0.68-1)^2 = 0.1296 to (2*0.97-1)^2 = 0.8836, whereas B's load remains (2*0.55-1)(2*0.60-1) = 0.02. Its strong composition discount overwhelms its very weak reliability weighting, predicting preference for the low-validity B coalition and a steep monotonic decrease in P(A) as A's validity increases. Exchangeable salience can create stable individual deviations but supplies no population-level slope tied to these translated configurations. The final six pairs are exact A/B mirrors. The primary statistic should code selection of the equal-validity coalition: the advocated theory predicts a pronounced negative validity slope, while the competing theory predicts a shallow positive slope. Twelve unique pairs repeated eight times yield exactly 96 trials.

[10] This 18-cue design isolates the advocated theory's reliability-gap-dependent latent-source inference in balanced two-versus-two conflicts. In the first eight pairs, A is supported by a target pair whose mean advertised validity is always 0.90, while B is supported by the fixed heterogeneous reference pair (0.82, 0.68). The target pair's within-pair validity gap increases from 0.00 through 0.04 and 0.10 to 0.18; each level has two order-reversed or duplicate realizations to average over idiosyncratic cue salience. Reliability-Schema Source Inference predicts a large positive gap effect: the equal extreme pair (0.90, 0.90) receives both narrow shared-source discounting and the equality-specific extreme-pair penalty, whereas increasing the gap rapidly releases the target cues from inferred overlap. At the 0.18 gap, the (0.99, 0.81) pair is treated as nearly independent, so target-pair choice should approach approximately one half despite the reference coalition. Exchangeable Corroboration with Idiosyncratic Salience predicts a much flatter profile. Its target composition load changes only from 0.64 at equality to 0.6076 at the largest gap because the target mean remains fixed, and all target loads remain far above the reference load of 0.2304. It should therefore continue strongly discounting the target coalition, generally keeping target choice around the lower quarter of the response range with only a small gap-related increase. The decisive statistic is thus the slope or equality-to-0.18 contrast in target-coalition choice: large under the advocated theory but small under the competitor, particularly producing near-chance versus clearly below-chance target choice at the widest gap. The final eight pairs are exact A/B mirrors. Sixteen unique pairs repeated six times yield exactly 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "This 12-cue design tests whether the internal reliability topology of a three-expert coalition matters after coalition size, mean validity, and the advocated theory's order-free composition load have been matched. Each clustered coalition has validities (0.87, 0.87, 0.96), whereas each distributed coalition has validities approximately (0.84804, 0.90, 0.95196). On the bounded-strength scale x=2v-1, these are respectively (0.74, 0.74, 0.92) and (0.69608, 0.80, 0.90392). Both triples have the same sum, 2.40, and exactly the same mean pairwise product, 0.6364. Thus Exchangeable Corroboration with Idiosyncratic Salience gives the coalitions equal composition discounts, equal size saturation, and identical three-member corroboration calibration. Their summed compressed-reliability weights are also nearly identical throughout the advocated parameter ranges. Its population prediction is therefore approximately P(distributed)=0.50, although stable individual salience profiles can produce subject-specific deviations. Reliability-Schema Source Inference instead predicts substantially greater choice of the distributed coalition. The clustered coalition contains an exactly matched, jointly reliable 0.87 pair, activating strong narrow shared-source overlap for two of its three members. The distributed coalition has no equal pair; its adjacent validity gaps are about 0.052 and its largest gap is about 0.104, yielding markedly less overlap despite some shoulder dependence. The four crossed comparisons use two duplicate instances of each coalition type, balancing how often every expert participates and reducing contamination from cue-specific salience. Four exact A/B mirrors control response-side bias. The decisive statistic is distributed-coalition choice pooled across mirrors: approximately chance under the advocated theory but robustly above chance under the competing theory. Eight unique pairs are repeated 12 times each for exactly 96 trials.",
  "validities": [
    0.87,
    0.87,
    0.96,
    0.8480384758,
    0.9,
    0.9519615242,
    0.87,
    0.87,
    0.96,
    0.8480384758,
    0.9,
    0.9519615242
  ],
  "trial_a_ratings": [
    [
      1,
      1,
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
      1,
      1,
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
      0,
      0,
      0,
      0,
      1,
      1,
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
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
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
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
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
      1,
      1,
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
      0,
      0,
      0,
      0,
      1,
      1,
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
      1,
      1,
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
  "input_tokens": 9289,
  "output_tokens": 5955
}
```
